import torch
torch.set_default_dtype(torch.float64)
from StructuredKernelInterpolation import *
import rasterio
import firedrake as df
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import path
import pyproj
import fiona
import scipy
import pandas as pd
import verde as vd
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter
import scipy
import geojson
from plot_pits import *

def project_array(coordinates, from_epsg=4326, to_epsg=3338, always_xy=True):
    """
    Project a numpy (n,2) array from <from_epsg> to <to_epsg>
    Returns the projected numpy (n,2) array.
    """
    tform = pyproj.Transformer.from_crs(crs_from=from_epsg, crs_to=to_epsg, always_xy=always_xy)
    fx, fy = tform.transform(coordinates[:,0], coordinates[:,1])
    # Re-create (n,2) coordinates
    return np.dstack([fx, fy])[0]

def optim_id (A , k ):
    _ , R , P = scipy.linalg.qr (A,
        pivoting = True,
        mode = 'economic' ,
        check_finite = False)
    R_k = R [:k ,:k]
    cols = P [:k]
    C = A [: ,cols]
    Z = scipy.linalg.solve(R_k.T @ R_k + np.eye(k)*1e-4,
        C.T @ A,
        overwrite_a = True ,
        overwrite_b = True ,
        assume_a = 'pos')
    approx = C @ Z
    return approx, cols, Z

def build_interpolation_matrix(X,X_,x_,y_):
    rows = []
    cols = []
    vals = []
    
    #nx = len(x_)
    #ny = len(y_)
    delta_x = x_[1] - x_[0]
    delta_y = y_[1] - y_[0]

    nx = len(x_)
    ny = len(y_)
    m = nx*ny
    
    xmin = x_.min()
    #xmax = xmin + (nx + 1)*delta_x
    
    ymin = y_.min()
    #ymax = ymin + (ny + 1)*delta_y

    for ii,xx in enumerate(X):
        
        x_low = int(torch.floor((xx[0] - xmin)/delta_x))
        x_high = x_low + 1

        y_low = int(torch.floor((xx[1] - ymin)/delta_y))
        y_high = y_low + 1
        
        #print(xx,x[x_low],x[x_high],y[y_low],y[y_high])

        ll = x_low + y_low*nx
        ul = x_low + y_high*nx
        lr = x_high + y_low*nx
        ur = x_high + y_high*nx
        bbox = [ll,ul,lr,ur]

        dist = torch.sqrt(((xx - X_[bbox])**2).sum(axis=1))

        w = 1./dist**4
        w/=w.sum()

        rows.append(torch.ones((4))*ii)
        cols.append(torch.tensor(bbox))
        vals.append(w) 

    inds = torch.vstack((torch.hstack(rows),torch.hstack(cols)))
    tens = torch.sparse_coo_tensor(inds,torch.hstack(vals),(X.shape[0],m))
    return tens,torch.transpose(tens,1,0)

mesh_directory = '../meshes/mesh_2201/'

mesh = df.Mesh(f'{mesh_directory}/mesh.msh',name='mesh')

Z_loc=0.0
Z_scale = 5000.0
X_loc = np.array([ 743600.34864157, 1204809.11692685])
X_scale = 49038.977309852955

mesh.coordinates.dat.data[:] -= X_loc
mesh.coordinates.dat.data[:] /= X_scale


E = df.FiniteElement("DG",mesh.ufl_cell(),0)
v_x = df.VectorFunctionSpace(mesh,E)
X = np.array(df.interpolate(mesh.coordinates,v_x).dat.data_ro)

#X_loc = 0.5*(X.max(axis=0) + X.min(axis=0))
#X_scale = 0.5*max(X.max(axis=0) - X.min(axis=0))

#X -= X_loc
#X /= X_scale

V = df.FunctionSpace(mesh,E)
phi = df.TestFunction(V)
area = torch.from_numpy(df.assemble(phi*df.dx).dat.data[:])


tree = KDTree((mesh.coordinates.dat.data - X_loc)/X_scale)

X = torch.tensor(X)#.to(torch.float)

bounds = (707145.3017168527, 780055.3955662812, 1155770.1396169967, 1253848.0942367027)

nx = 401
x_ = torch.linspace(-1.1,1.1,nx)
y_ = torch.linspace(-1.1,1.1,nx)

X_,Y_ = torch.meshgrid([x_,y_])
X_ = torch.vstack((batch_vec(X_.unsqueeze(0)),batch_vec(Y_.unsqueeze(0)))).T
W,W_t = build_interpolation_matrix(X,X_,x_,y_)

def k_se(x1,x2,l,amplitude):
    D = torch.cdist(x1.unsqueeze(-1),x2.unsqueeze(-1))
    return amplitude*torch.exp(-D**2/(2*l**2))

def k_mat2(x1,x2,l,amplitude):
    D = torch.cdist(x1.unsqueeze(-1),x2.unsqueeze(-1))
    return amplitude*torch.exp(-np.sqrt(5)*D/l)*(1 + np.sqrt(5)*D/l + 5*D**2/(3*l**2))

def k_mat(x1,x2,l,amplitude):
    D = torch.cdist(x1.unsqueeze(-1),x2.unsqueeze(-1))
    return torch.exp(-np.sqrt(3)*D/l)*(1 + np.sqrt(3)*D/l)

l = 40000/X_scale

Kx = k_se(x_,x_,l,amplitude=1)

k = int(2.2*(x_.max() - x_.min())/l)
cols = [int(j) for j in (np.round(np.linspace(0,len(x_)-1,k)))]

Ks = Kx[cols,:][:,cols]
u,s,v = np.linalg.svd(Ks)
Lx = Kx[:,cols] @ u * 1./np.sqrt(s) @ u.T
f = 1/np.sqrt((Lx**2).sum(axis=1))
Lx = torch.diag(f) @ Lx
Ly = Lx

L = W @ torch.kron(Lx,Ly)

def reshape_fortran(x, shape):
  if len(x.shape) > 0:
    x = x.permute(*reversed(range(len(x.shape))))
  return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

X__,Y__ = torch.meshgrid(x_[cols],x_[cols])
X_center = torch.hstack((reshape_fortran(X__,(-1,1)),reshape_fortran(Y__,(-1,1))))
near = tree.query(X_center,k=1)[0] < 2*l


data_ref = pickle.load(open(f'{mesh_directory}/surface/time_series/map_cop30.p','rb'))
S_mean = data_ref['model_basis']['mean']
S_map = data_ref['model_basis']['coeff_map']
wmean_post = data_ref['coefficients']['post_mean']

S_init = S_mean + S_map @ wmean_post

thk_scale = 5000

z = np.array([0,900,1600,6000])/thk_scale

m_z = len(z)
m_x = L.shape[1]

def build_f(x,d):
    f0 = torch.maximum(1 - (x-d[0])/(d[1] - d[0]),torch.zeros_like(x))
    cols = [f0]
    m = len(d)
    for j in range(1,m-1):
        f = torch.maximum(torch.minimum( (x - d[j-1])/(d[j] - d[j-1]), 1-(x-d[j])/(d[j+1] - d[j])),torch.zeros_like(x))
        cols.append(f)
    fm = torch.maximum((x - d[m-2])/(d[m-1] - d[m-2]),torch.zeros_like(x))
    cols.append(fm)
    return torch.vstack(cols).T

f = build_f(S_init,z)*10/Z_scale
L0x = torch.hstack((f,L/Z_scale*0.2))
L1x = build_f(S_init,z)/Z_scale
L1x[:,-1] *= 0.01

L0t = torch.tensor([[1],[1]])
L1t = torch.tensor([[2.,0],
                    [2.,6.]])


scales = torch.tensor([1./3.,1./9.,1./9.,0.01])
#scales = torch.ones(4)*0.01
L2x = build_f(S_init,z)/Z_scale * scales
L2t = torch.eye(2)

L0 = np.kron(L0t,L0x)
L1 = np.kron(L1t,L1x)
L2 = np.kron(L2t,L2x)

L_all = torch.from_numpy(np.hstack((L0,L1,L2)))
L_ref = torch.from_numpy(np.hstack((L0,np.zeros_like(L1),np.zeros_like(L2))))
L_dif = torch.from_numpy(np.hstack((np.zeros_like(L0),L1,L2)))

n_x = f.shape[0]

with open('../data/boundary/MalaspinaMaximum.json') as f:
    gj = geojson.load(f)
features = gj['features'][0]['geometry']['coordinates']

X_m = np.array(features)[:,:2]
transformer = pyproj.Transformer.from_crs("EPSG:4326","EPSG:3338",always_xy=True)
X_m = np.array(transformer.transform(*X_m.T)).T
X_m -= X_loc
X_m /= X_scale
p = path.Path(X_m)
ins = p.contains_points(X)

inds_ela = torch.where(torch.logical_and((S_init*5000)>900,(S_init*5000)<1000))[0]
n_ela = len(inds_ela)

data = np.loadtxt('../data/snoe/Seward2021_Michael.csv',delimiter=',',usecols=[0,1,5,-1],skiprows=4,converters = lambda s: float(s.strip() or np.nan))

data = data[~np.isnan(data[:,3])]

X_radar = project_array(data[:,:2])
Z_radar = data[:,2]
H_radar = data[:,3]
reducer = vd.BlockReduce(reduction=np.median, spacing=1000)
coordinates, elev = reducer.filter(
        coordinates=(X_radar[:,0], X_radar[:,1]), data=Z_radar
        )
_, thk = reducer.filter(
        coordinates=(X_radar[:,0], X_radar[:,1]), data=H_radar
)

X_radar = (torch.tensor(coordinates).T - X_loc)/X_scale
Z_radar = torch.tensor(elev)/Z_scale
# !!!! H_radar = (torch.tensor(thk)/thk.mean()*1.4)/Z_scale
H_radar = (torch.tensor(thk)/thk.mean()*1.4)/Z_scale

inds_radar = torch.tensor([torch.argmin(((p - X)**2).sum(axis=1)) for p in X_radar])
n_radar = len(inds_radar)

O_radar = torch.zeros((n_radar,L_all.shape[0]),dtype=torch.float64)
t = 1
for i in range(n_radar):
    O_radar[i,t*n_x + inds_radar[i]] = 1.0

y_radar = H_radar
sigma_radar = 0.2/Z_scale*torch.ones_like(y_radar)

t = 1
O_global = torch.zeros((1,L_all.shape[0]))
O_global[0,n_x:] = area*ins

y_global = torch.zeros(1,dtype=torch.float64)
y_global[:] = -4e-5
sigma_global = torch.zeros(1,dtype=torch.float64)
sigma_global[:] = 1e-6

pts = torch.tensor([[-0.17524644,  0.97031763]])
inds_other = torch.tensor([torch.argmin(((p - X)**2).sum(axis=1)) for p in pts])
n_other = 1#2*len(inds_other)
O_other = torch.zeros((1,L_all.shape[0]),dtype=torch.float64)
O_other[0,n_x+inds_other[0]] = 1.0
#O_other[1,n_x + inds_other[0]] = 1.0

y_other = torch.zeros(1,dtype=torch.float64)
y_other[:] = 0.4/thk_scale
sigma_other = torch.zeros(1,dtype=torch.float64)
sigma_other[:] = 0.03/thk_scale

O_ela = torch.zeros((n_ela,L_all.shape[0]))

t = 1
for i in range(n_ela):
    O_ela[i,t*n_x + inds_ela[i]] = 1.0

y_ela = torch.zeros((n_ela))
sigma_ela = torch.zeros((n_ela))
sigma_ela[:] = 0.2/thk_scale*torch.ones_like(y_ela)

#y_ref = y_global
#sigma_ref = sigma_global
#O_ref = O_global @ L_ref

y_ref = torch.hstack((y_radar,y_other,y_ela,y_global))
sigma_ref = torch.hstack((sigma_radar,sigma_other, sigma_ela, sigma_global))
O_ref = torch.vstack((O_radar @ L_ref, O_other @ L_ref, O_ela @ L_ref, O_global @ L_ref))

y_dif = torch.zeros(y_radar.shape[0]+y_other.shape[0]+y_ela.shape[0])
sigma_dif = torch.ones_like(y_dif)*0.01/thk_scale
O_dif = torch.vstack((O_radar @ L_dif, O_other @ L_dif, O_ela @ L_dif))

y = torch.hstack((y_ref,y_dif))
sigma = torch.hstack((sigma_ref,sigma_dif))
O = torch.vstack((O_ref,O_dif))

Tau_post = O.T /sigma**2 @ O + torch.eye(L_all.shape[1])
s,u = torch.linalg.eigh(Tau_post)

L_post = u / torch.sqrt(s) @ u.T

w_post = L_post @ L_post.T @ (O.T/sigma**2) @ y


w_post = w_post[:-8]
Sigma_post = (L_post @ L_post.T)[:-8,:-8]
s_,u_ = torch.linalg.eigh(Sigma_post)
L_post = u_ * torch.sqrt(s_) @ u_.T 

z = w_post + L_post @ torch.randn(L_post.shape[1])

z_ref = z[:L0x.shape[1]]
z_dif = z[L0x.shape[1]:]

pred = L0x @ z_ref.reshape(-1,1) + L1x @ z_dif.T.reshape(2,4).T @ L1t.T
fig,axs=plt.subplots(ncols=3)
axs[0].scatter(*X.T,c=5000*pred[:,0],vmin=-4,vmax=2)
axs[0].set_aspect('equal')

axs[1].scatter(*X.T,c=5000*pred[:,1],vmin=-4,vmax=2)
axs[1].set_aspect('equal')

axs[2].scatter(*X.T,c=5000*L0x @ z_ref.reshape(-1,1),vmin=-4,vmax=2)
axs[2].set_aspect('equal')

#mesh.coordinates.dat.data[:] *= X_scale
#mesh.coordinates.dat.data[:] += X_loc

fig,axs = plt.subplots(nrows=2,gridspec_kw={'height_ratios': [1, 2]})
Q = df.FunctionSpace(mesh,'DG',0)
adot = df.Function(Q)
adot.dat.data[:] = 5000*L0x @ z_ref

ax = axs[1]
c = df.tripcolor(adot,axes=ax,cmap=plt.cm.seismic,vmin=-4,vmax=4,rasterized=True)

ax.scatter(*X_radar.T,c=H_radar*5000,vmin=-4,vmax=4,cmap=plt.cm.seismic)
ax.scatter(*((project_array(np.array([[p1_xy[1],p1_xy[0]]])) - X_loc)/X_scale).squeeze(),color='green',marker='*')
ax.scatter(*((project_array(np.array([[p2_xy[1],p1_xy[0]]])) - X_loc)/X_scale).squeeze(),color='green',marker='*')
ax.scatter(*((project_array(np.array([[p3_xy[1],p1_xy[0]]])) - X_loc)/X_scale).squeeze(),color='green',marker='*')

ax.scatter(*X[inds_ela].T,color='yellow',marker='^')
ax.scatter(*pts.T,color='orange',marker='s')

melt_meas = np.array([[59.871034,-140.336906],[59.826082,-140.671635],[59.834383,-140.775957],[60.003848,-141.061608]])
ax.scatter(*((project_array(melt_meas[:,::-1])-X_loc)/X_scale).T ,color='purple',marker='X')

import shapefile as shp  # Requires the pyshp package

sf = shp.Reader("../data/boundary/geology/ice_margin.shp")
vg = shp.Reader("../data/boundary/geology/veg_line.shp")

for shape in sf.shapeRecords():
    xx = np.array([i[0] for i in shape.shape.points[:]])
    yy = np.array([i[1] for i in shape.shape.points[:]])
    X_b = project_array(np.vstack((xx,yy)).T,from_epsg=32607)
    ax.plot(*((X_b - X_loc)/X_scale).T,'w-')

ax.set_aspect('equal')
cbar = plt.colorbar(c,orientation='horizontal',shrink=0.8,extend='both',pad=0.02,)

cbar.set_label('SMB (m a$^{-1}$)')

ax.set_xticks([])
ax.set_yticks([])

ax = axs[0]
ax.stairs(p1_d,p1_z,orientation='horizontal',color='red',baseline=None,label='Pit 1, Core 1')
ax.stairs(p1h1_d,p1h1_z,orientation='horizontal',color='red',baseline=None)
ax.stairs(p1h2_d,p1h2_z,orientation='horizontal',color='green',baseline=None,label='Pit 1, Core 2')
ax.stairs(p2_d,p2_z,orientation='horizontal',color='blue',baseline=None,label='Pit 2')
ax.stairs(p2h1_d,p2h1_z,orientation='horizontal',color='blue',baseline=None)
ax.stairs(p3_d,p3_z,orientation='horizontal',color='orange',baseline=None,label='Pit 3')
ax.stairs(p3h1_d,p3h1_z,orientation='horizontal',color='orange',baseline=None)
ax.axhline(454,color='k',linestyle='--',label='2022 Surface')
ax.axhline(740,color='k',linestyle=':',label='2021 Surface')
ax.invert_yaxis()
ax.set_xlabel('$\\rho$ (kg/m$^3$)')
ax.set_ylabel('Depth (cm)')
ax.legend(loc='upper left')

axs[0].text(0.99, 0.99, 'a',
     horizontalalignment='right',
     verticalalignment='top',
     transform = axs[0].transAxes,fontweight='bold')
axs[1].text(0.99, 0.99, 'b',
     horizontalalignment='right',
     verticalalignment='top',
     transform = axs[1].transAxes,fontweight='bold')

fig.set_size_inches(4.5,10.5)
fig.subplots_adjust(hspace=0.05)
fig.savefig(f'./smb.pdf',dpi=300,bbox_inches='tight')


pickle.dump([L0x,L1x,w_post,L_post],open(f'{mesh_directory}/adot/adot_basis.p','wb'))

