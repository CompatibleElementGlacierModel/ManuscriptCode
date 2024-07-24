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
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree

device = 'cuda'

# define a method to convert between coordinate systems - for most of our data, the input epsg will be 4326, and we'll want to project to 3338

from_epsg = 4326
to_epsg = 3338

def project_array(coordinates, from_epsg=4326, to_epsg=3338, always_xy=True):
    """
    Project a numpy (n,2) array from <from_epsg> to <to_epsg>
    Returns the projected numpy (n,2) array.
    """
    tform = pyproj.Transformer.from_crs(crs_from=from_epsg, crs_to=to_epsg, always_xy=always_xy)
    fx, fy = tform.transform(coordinates[:,0], coordinates[:,1])
    # Re-create (n,2) coordinates
    return np.dstack([fx, fy])[0]

def get_training_data_from_bbox(dem,bounds,X_loc,X_scale,Z_loc,Z_scale,target_resolution,nodatavalue):

    # Get DEM elevations
    Z_dem = dem.read().squeeze()[::-1].astype(float)

    # Get edge coordinates
    x_dem_ = np.linspace(dem.bounds.left,dem.bounds.right,dem.width+1)
    y_dem_ = np.linspace(dem.bounds.bottom,dem.bounds.top,dem.height+1)

    # Get cell center coordinates
    x_dem = 0.5*(x_dem_[:-1] + x_dem_[1:])
    y_dem = 0.5*(y_dem_[:-1] + y_dem_[1:])

    # Get extremal locations
    x_min,x_max,y_min,y_max = bounds

    # 
    x_in = (x_dem > (x_min-target_resolution)) & (x_dem < (x_max+target_resolution))
    y_in = (y_dem > (y_min-target_resolution)) & (y_dem < (y_max+target_resolution))

    # Keep valid locations
    x_dem = x_dem[x_in]
    y_dem = y_dem[y_in]
    Z_dem = Z_dem[y_in][:,x_in]

    # Downsample DEM
    dx = abs(x_dem[1] - x_dem[0])
    dy = abs(y_dem[1] - y_dem[0])

    skip_x = int(target_resolution // dx)
    skip_y = int(target_resolution // dy)

    #print(Z_dem.shape)
    #Z_dem = gaussian_filter(Z_dem,[skip_y,skip_x])
    #print(Z_dem.shape)

    x_ = (x_dem[::skip_x].reshape(-1,1) - X_loc[0])/X_scale
    y_ = (y_dem[::skip_y].reshape(-1,1) - X_loc[1])/X_scale

    X_dem,Y_dem = np.meshgrid(x_,y_)
    V = Z_dem.copy()
    V[Z_dem==nodatavalue] = 0
    VV = gaussian_filter(V,[skip_x,skip_y],mode='nearest')

    W = np.ones_like(V)
    W[Z_dem==nodatavalue] = 0
    WW = gaussian_filter(W,[skip_x,skip_y],mode='nearest')
    ZZ = VV/WW
    ZZ[Z_dem==nodatavalue] = nodatavalue
    Z_dem = ZZ[::skip_y,::skip_x]

    z = Z_dem.ravel(order='F')
    inds = (z!=nodatavalue)
    Z_train = z[inds]

    Z_train -= Z_loc
    Z_train /= Z_scale

    X_train = np.c_[X_dem.ravel(order='F'),Y_dem.ravel(order='F')][inds]
    return X_train,Z_train

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

        w = 1./dist
        w/=w.sum()

        rows.append(torch.ones((4))*ii)
        cols.append(torch.tensor(bbox))
        vals.append(w) 

    inds = torch.vstack((torch.hstack(rows),torch.hstack(cols)))
    tens = torch.sparse_coo_tensor(inds,torch.hstack(vals),(X.shape[0],m))
    return tens,torch.transpose(tens,1,0)

mesh_directory = '../meshes/mesh_2201/'

dem_path = '../data/dem/cop30_3338.tif'
nodatavalue = 3.4028234663852886e+38

mesh = df.Mesh(f'{mesh_directory}/mesh.msh',name='mesh')

dem = rasterio.open(dem_path)

E = df.FiniteElement("DG",mesh.ufl_cell(),0)
v_x = df.VectorFunctionSpace(mesh,E)
X = np.array(df.interpolate(mesh.coordinates,v_x).dat.data_ro)

E_cg2 = df.FiniteElement("CG",mesh.ufl_cell(),2)
v_x2 = df.VectorFunctionSpace(mesh,E_cg2)
X_cg2 = np.array(df.interpolate(mesh.coordinates,v_x2).dat.data_ro)

Z_loc=0.0
Z_scale = 5000.0
X_loc = np.array([ 743600.34864157, 1204809.11692685])
X_scale = 49038.977309852955
#X_loc = 0.5*(X.max(axis=0) + X.min(axis=0))
#X_scale = 0.5*max(X.max(axis=0) - X.min(axis=0))

X -= X_loc
X /= X_scale

X_cg2 -= X_loc
X_cg2 /= X_scale

tree = KDTree(X)
ice_mask = pickle.load(open('../data/boundary/boundary.p','rb'))

#x_max,y_max = X.max(axis=0)*X_scale + X_loc
#x_min,y_min = X.min(axis=0)*X_scale + X_loc

#bounds = (x_min,x_max,y_min,y_max)
bounds = (707145.3017168527-3000, 780055.3955662812+3000, 1155770.1396169967-3000, 1253848.0942367027+3000)

X_train,Z_train = get_training_data_from_bbox(dem,bounds,X_loc,X_scale,Z_loc,Z_scale,500,nodatavalue)

compute_mask=False
if compute_mask:
    mask = np.zeros(X_train.shape[0]).astype(bool)

    data = fiona.open('../data/boundary/glims_download_17210/glims_polygons.shp')
    q = [d for d in data if d['properties']['line_type']=='glac_bound']
    for glacier in q:
        print(glacier['properties']['glac_name'])
        l_mask = np.zeros(X_train.shape[0]).astype(bool)
        coords = glacier['geometry']['coordinates']
        try:
            outline = np.array(coords[0])[:,:2]
            outline = project_array(outline,from_epsg=4326, to_epsg=3338, always_xy=True)
            outer_ice = path.Path((outline-X_loc)/X_scale)
            within_outer = outer_ice.contains_points(X_train)
            l_mask[within_outer] = True
            holes = []
            for c in coords[1:]:
                hole = np.array(c)[:,:2]
                hole = project_array(hole,from_epsg=4326, to_epsg=3338, always_xy=True)
                inner_ice = path.Path((hole-X_loc)/X_scale)
                within_inner = inner_ice.contains_points(X_train)
                l_mask[within_inner] = False
                #plt.plot(*hole.T,'r-')
            #plt.plot(*outline.T,'k-')
            mask += l_mask
        except:
            pass

    mask = np.invert(mask)
    pickle.dump(mask,open('bed_mask.p','wb'))
else:
    mask = pickle.load(open('bed_mask.p','rb'))

X_train = X_train[mask]
Z_train = Z_train[mask]

X_train = torch.from_numpy(X_train)#.to(torch.float)
X = torch.from_numpy(X)#.to(torch.float)
X_cg2 = torch.from_numpy(X_cg2)

tree = KDTree((mesh.coordinates.dat.data - X_loc)/X_scale)
l = 2000.0/X_scale
near = tree.query(X_train,k=1)[0] < 2*l

X_train = X_train[near]

n_train = len(X_train)
n_test = X.shape[0]

Z_train = torch.from_numpy(Z_train)#.to(torch.float)
#Z_train -= 20./Z_scale
Z_train = Z_train[near]

dfr = pd.read_csv('../data/bed/IRARES2_malaspina.csv', delimiter=',')
# track IRARES1B_20160528-224052 has ~50 +/- 20 m average xover in ice thickness with other non-parallel tracks, so we'll omit this track from the interpolation
#dfr = dfr[(dfr.track!="IRARES1B_20160528-224052")]
# filter null points
dfr = dfr[dfr.bed_hgt.notnull()]
# parse out columns of interest and convert to numpy array
# we want lon,lat,bed_elev
picks = dfr.loc[:,['lon','lat',"bed_hgt"]]
# project xy coords to epsg:3338
pick_coords = project_array(picks[['lon','lat']].to_numpy(), from_epsg=4326, to_epsg=to_epsg, always_xy=True)
picks[['lon','lat']] = pick_coords
# rename lon, lat to x, y
picks = picks.rename(columns={"lon": "x", "lat": "y", "bed_hgt":"z"})

# use verde to reduce along track data
# we choose median over a mean because bed elevation can vary abruptly and the mean would smooth the data too much
reducer = vd.BlockReduce(reduction=np.median, spacing=1000)
coordinates, elev = reducer.filter(
    coordinates=(picks.x, picks.y), data=picks.z
)

picks_irares = np.vstack((coordinates[0], coordinates[1], elev)).T

dfr = pd.read_csv('../data/bed/IRUAFHF2_malaspina.csv', delimiter=',')
# track IRARES1B_20160528-224052 has ~50 +/- 20 m average xover in ice thickness with other non-parallel tracks, so we'll omit this track from the interpolation
#dfr = dfr[(dfr.track!="IRARES1B_20160528-224052")]
# filter null points
dfr = dfr[dfr.bed_hgt.notnull()]
# parse out columns of interest and convert to numpy array
# we want lon,lat,bed_elev
picks = dfr.loc[:,['lon','lat',"bed_hgt"]]
# project xy coords to epsg:3338
pick_coords = project_array(picks[['lon','lat']].to_numpy(), from_epsg=4326, to_epsg=to_epsg, always_xy=True)
picks[['lon','lat']] = pick_coords
# rename lon, lat to x, y
picks = picks.rename(columns={"lon": "x", "lat": "y", "bed_hgt":"z"})

# use verde to reduce along track data
# we choose median over a mean because bed elevation can vary abruptly and the mean would smooth the data too much
reducer = vd.BlockReduce(reduction=np.median, spacing=1000)
coordinates, elev = reducer.filter(
    coordinates=(picks.x, picks.y), data=picks.z
)

picks_iruafhf = np.vstack((coordinates[0], coordinates[1], elev)).T

picks = np.vstack((picks_irares,picks_iruafhf))

picks[:,:2] -= X_loc
picks[:,:2] /= X_scale
picks[:,2] -= Z_loc
picks[:,2] /= Z_scale

X_train = torch.vstack((X_train,torch.from_numpy(picks[:,:2])))#.to(torch.float)))
Z_train = torch.hstack((Z_train,torch.from_numpy(picks[:,2])))#.to(torch.float)))
#X_train = torch.from_numpy(picks[:,:2]).to(torch.float)
#Z_train = torch.from_numpy(picks[:,2]).to(torch.float)

X_all = torch.vstack((X_train,X_cg2))

nx = 301
x_ = torch.linspace(-1.1,1.1,nx)
y_ = torch.linspace(-1.1,1.1,nx)

X_,Y_ = torch.meshgrid([x_,y_])
X_ = torch.vstack((batch_vec(X_.unsqueeze(0)),batch_vec(Y_.unsqueeze(0)))).T
W,W_t = build_interpolation_matrix(X_all,X_,x_,y_)

def k(x1,x2,l,amplitude):
    D = torch.cdist(x1.unsqueeze(-1),x2.unsqueeze(-1))
    return amplitude*torch.exp(-D**2/(l**2))

def k_rq(x1,x2,l,amplitude,alpha=1.0):
    D = torch.cdist(x1.unsqueeze(-1),x2.unsqueeze(-1))
    return amplitude*(1 + D**2/(2*alpha*l**2))**(-alpha)

def k_mat2(x1,x2,l,amplitude,unsqueeze=True):
    if unsqueeze:
        D = torch.cdist(x1.unsqueeze(-1),x2.unsqueeze(-1))
    else:
        D = torch.cdist(x1,x2)
    return amplitude*torch.exp(-np.sqrt(5)*D/l)*(1 + np.sqrt(5)*D/l + 5*D**2/(3*l**2))

def k_mat(x1,x2,l,amplitude,unsqueeze=True):
    if unsqueeze:
        D = torch.cdist(x1.unsqueeze(-1),x2.unsqueeze(-1))
    else:
        D = torch.cdist(x1,x2)
    return amplitude*torch.exp(-np.sqrt(3)*D/l)*(1 + np.sqrt(3)*D/l)



length_scales = np.array([3000])/X_scale
amplitudes = np.array([1000])/Z_scale

Ls = []

for l,a in zip(length_scales,amplitudes):

    Kx = k_mat(x_,x_,l,amplitude=1)# + torch.eye(nx)*1

    n_cols = int(2.2*(x_.max() - x_.min())/l)
    cols = [int(j) for j in (np.round(np.linspace(0,len(x_)-1,n_cols)))]

    Ks = Kx[cols,:][:,cols]
    u,s,v = np.linalg.svd(Ks)
    Lx = Kx[:,cols] @ u * 1./np.sqrt(s) @ u.T

    Ly = Lx

    L = W @ torch.kron(Lx,Ly)

    L *= a

    def reshape_fortran(x, shape):
      if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
      return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

    X__,Y__ = torch.meshgrid(x_[cols],x_[cols])
    X_center = torch.hstack((reshape_fortran(X__,(-1,1)),reshape_fortran(Y__,(-1,1))))

    compute_indices_to_keep = False
    index_directory = '../meshes/mesh_1899/'

    if compute_indices_to_keep:
        near = tree.query(X_center,k=1)[0] < 2*l
        pickle.dump(near,open(f'{mesh_directory}/bed/low_res_valid_center_bed.p','wb'))
    else:
        near = pickle.load(open(f'{index_directory}/bed/low_res_valid_center_bed.p','rb'))

    L = L[:,near]
    Ls.append(L)

L = torch.from_numpy(np.hstack((Ls)))

Sigma_diag = (L**2).sum(axis=1)
sigma2_obs = (50.0/Z_scale)**2

n_train = len(X_train)
n_test = X.shape[0]
n = len(X_all)

Psi =  torch.hstack((L,torch.ones((n,1)),X_all,X_all**2,(X_all[:,0]*X_all[:,1]).reshape(-1,1)))

h_train = Psi[:n_train,-6:][-picks.shape[0]:]
h_train_f = Psi[:n_train,-6:]
h_test = Psi[n_train:,-6:]
h_all = Psi[:,-6:]
beta = torch.linalg.solve(h_train.T / sigma2_obs @ h_train, h_train.T/sigma2_obs @ Z_train[-picks.shape[0]:])
#beta[:] = 0

mu_train = h_train_f @ beta
mu_test = h_test @ beta

Z_resid = Z_train - mu_train

L_train = Psi[:n_train,:-6]
L_test = Psi[n_train:,:-6]

mesh.coordinates.dat.data[:] -= X_loc
mesh.coordinates.dat.data[:] /= X_scale
Q_dg = df.FunctionSpace(mesh,E)
Q_cg2 = df.FunctionSpace(mesh,E_cg2)
phi_dg = df.TestFunction(Q_dg)
w_dg = df.TrialFunction(Q_dg)
phi_cg2 = df.TrialFunction(Q_cg2)
M_dg = df.assemble(phi_dg*w_dg*df.dx)
M_m = df.assemble(phi_dg * phi_cg2 * df.dx)
mesh.coordinates.dat.data[:] *= X_scale
mesh.coordinates.dat.data[:] += X_loc

import scipy.sparse as sp
petsc_mat = M_m.M.handle
indptr,indics,data = petsc_mat.getValuesCSR()
M_m = sp.csr_matrix((data,indics,indptr), shape=petsc_mat.getSize())

petsc_mat = M_dg.M.handle
indptr,indics,data = petsc_mat.getValuesCSR()
M_dg = sp.csr_matrix((1./data,indics,indptr), shape=petsc_mat.getSize())

L_test = torch.from_numpy(M_dg @ M_m @ L_test)
h_test = torch.from_numpy(M_dg @ M_m @ h_test)
mu_test = h_test @ beta

I = torch.eye(L_train.shape[1])


sigma_sys = 1./5000.0
V = torch.ones(L_train.shape[0])*sigma_sys

Q = L_train.T/sigma2_obs @ V

R = 1 + V.T/sigma2_obs @ V

Tau_post = I + L_train.T /sigma2_obs @ L_train - torch.outer(Q /R, Q)


st,ut = torch.linalg.eigh(Tau_post)
G = ut*1./torch.sqrt(st) @ ut.T
wmean_post = G @ (G.T @ (((L_train.T/sigma2_obs)) @ Z_resid - Q / R * (V.T/sigma2_obs @ Z_resid )))

data_dict = {'observation_basis':{'coeff_map':L_train,
                                  'mean_map':h_train_f},

             'model_basis':{'coeff_map':L_test,
                            'mean_map':h_test},

             'coefficients':{'post_cov_root':G,
                             'post_mean':wmean_post,
                             'mean_coeff':beta},

             'inverse_maps':None,

             'data':{'x_obs':X_train,
                     'z_obs':Z_train,
                     'x_test':X}}

mean_test = mu_test + L_test @ wmean_post
mean_train = mu_train + L_train @ wmean_post

std_test = ((L_test @ G)**2).sum(axis=1)**0.5

v_dg = df.FunctionSpace(mesh,'DG',0)
f = df.Function(v_dg)
fig,axs = plt.subplots(ncols=2,nrows=2)
#fig.set_size_inches(4,4)
axs = axs.ravel()
for i,ax in enumerate(axs):
    if i==0:
        f.dat.data[:] = mean_test# + L_test @ G @ torch.randn(G.shape[1])
    else:
        f.dat.data[:] = mean_test + L_test @ G @ torch.randn(G.shape[1])
    df.tripcolor(f,axes=ax,cmap=plt.cm.bwr,vmin=-1000/5000,vmax=3000/5000)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Prior Mean')
fig.savefig('plots/bed/prior_mean.png',bbox_inches='tight')

fig,axs = plt.subplots(ncols=1)
fig.set_size_inches(4,4)
f.dat.data[:] = std_test
df.tripcolor(f,axes=axs,cmap=plt.cm.inferno,vmin=0,vmax=100/5000)
plt.scatter(*(X_train*X_scale + X_loc).T,c='r',s=2)

axs.set_aspect('equal')
axs.set_xticks([])
axs.set_yticks([])
axs.set_title('Prior Marginal StD')
fig.savefig('plots/bed/prior_std.png',bbox_inches='tight')

"""
fig,ax = plt.subplots()
ax.scatter(*X_train.T,c=Z_train*5000,cmap=plt.cm.gist_earth,vmin=-1000,vmax=4000)
ax.set_title('Bed')
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal')
fig.set_size_inches(4,4)
fig.savefig('./plots/bed/bed_obs.png')
"""

pickle.dump(data_dict,open(f'{mesh_directory}/bed/bed_basis.p','wb'))

