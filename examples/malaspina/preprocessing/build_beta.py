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

mesh_directory = '../meshes/mesh_1000/'

mesh = df.Mesh(f'{mesh_directory}/mesh.msh',name='mesh')

E = df.FiniteElement("CG",mesh.ufl_cell(),1)
v_x = df.VectorFunctionSpace(mesh,E)
X = np.array(df.interpolate(mesh.coordinates,v_x).dat.data_ro)

Z_loc=0.0
Z_scale = 5000.0
X_loc = np.array([ 743600.34864157, 1204809.11692685])
X_scale = 49038.977309852955

X -= X_loc
X /= X_scale

X = torch.tensor(X)

bounds = (707145.3017168527, 780055.3955662812, 1155770.1396169967, 1253848.0942367027)

nx = 301
x_ = torch.linspace(-1.1,1.1,nx)
y_ = torch.linspace(-1.1,1.1,nx)

X_,Y_ = torch.meshgrid([x_,y_])
X_ = torch.vstack((batch_vec(X_.unsqueeze(0)),batch_vec(Y_.unsqueeze(0)))).T
W,W_t = build_interpolation_matrix(X,X_,x_,y_)

def k(x1,x2,l,amplitude):
    D = torch.cdist(x1.unsqueeze(-1),x2.unsqueeze(-1))
    return amplitude*torch.exp(-D**2/(2*l**2))

def k_mat2(x1,x2,l,amplitude):
    D = torch.cdist(x1.unsqueeze(-1),x2.unsqueeze(-1))
    return amplitude*torch.exp(-np.sqrt(5)*D/l)*(1 + np.sqrt(5)*D/l + 5*D**2/(3*l**2))

def k_mat(x1,x2,l,amplitude):
    D = torch.cdist(x1.unsqueeze(-1),x2.unsqueeze(-1))
    return torch.exp(-np.sqrt(3)*D/l)*(1 + np.sqrt(3)*D/l)

l = 2000/X_scale

Kx = k_mat2(x_,x_,l,amplitude=1)

_ ,_ ,P = scipy.linalg.qr(Kx,
        pivoting = True,
        mode = 'economic' ,
        check_finite = False)
ki = int(2.2*(x_.max() - x_.min())/l)
cols = [int(j) for j in (np.round(np.linspace(0,len(x_)-1,ki)))]
#cols = np.random.choice(range(nx),k,replace=False)
Ks = Kx[cols,:][:,cols]
u,s,v = np.linalg.svd(Ks)
Lx = Kx[:,cols] @ u * 1./np.sqrt(s) @ u.T
#f = 1/np.sqrt((Lx**2).sum(axis=1))
#Lx = torch.diag(f) @ Lx
Ly = Lx

L = W @ torch.kron(Lx,Ly)
L_norm = torch.linalg.norm(L,ord=torch.inf,axis=0)

compute_indices_to_keep=False
index_directory = '../meshes/mesh_1899/'
if compute_indices_to_keep:
    keep = L_norm>1e-3
    pickle.dump(keep,open(f'{mesh_directory}/beta/low_res_valid_center_beta.p','wb'))
else:
    keep = pickle.load(open(f'{index_directory}/beta/low_res_valid_center_beta.p','rb'))

L = L[:,keep]
L = torch.hstack((L,torch.ones(L.shape[0],1)))

ref_index = 2013-1985
t = torch.linspace(0,35,36)
Kt = k(t,t,0.5,1.)

Ks = Kt # - np.outer(Kt[ref_index],Kt[ref_index])
u,s,v = np.linalg.svd(Ks)


Lt = torch.from_numpy(u * np.sqrt(s) @ u.T)
#Lt = torch.hstack((Lt,torch.ones(Lt.shape[1],1)))
#Lt_obs = Lt[15:25]
#f = 1/np.sqrt((Lt**2).sum(axis=1))
#Lt = torch.diag(f) @ Lt             
#l,u = torch.linalg.eigh(Kt)
#l = torch.flip(l,(0,))
#u = torch.flip(u,(1,))
#inds = (l/l[0])>1e-2
#Lt = u[:,inds] * torch.sqrt(l[inds])
mean_zero = False
if mean_zero:
    O = torch.ones((1,Lt.shape[0]))
    Z = O @ Lt
    sigma2_c = 1.0
    Tau_post = Z.T/sigma2_c @ Z + torch.eye(Z.shape[1])
    u,s,_ = torch.linalg.svd(Tau_post)
    L_post = u / (s**0.5) @ u.T
    Lt = Lt @ L_post


"""
Ot = torch.ones(1,Lt.shape[0])
Ft = Ot @ Lt
F = torch.kron(Ft,L)
I = (F @ F.T + 10000*torch.eye(F.shape[0])).to_dense()
C = torch.linalg.cholesky(I)
Cinv = torch.linalg.inv(C)
V = (Cinv @ F).T

Z = torch.randn(L.shape[1],Lt.shape[1])
z = Z.T.ravel()
z_ = z - V @ (V.T @ z)
Z_ = z_.reshape(Z.T.shape).T
beta0 = L @ Z_ @ Lt.T


V[abs(V)<1e-4] = 0.0
Vs = V.to_sparse_csr()
Vst = V.T.to_sparse_csr()

z_ = z - Vs @ (Vst @ z)
Z_ = z_.reshape(Z.T.shape).T
beta1 = L @ Z_ @ Lt.T

"""
"""


log_beta = L @ torch.randn(L.shape[1],Lt.shape[1]) @ Lt.T

v_dg = df.FunctionSpace(mesh,'CG',1)
f = df.Function(v_dg)
for i in range(3):
    fig,axs = plt.subplots(ncols=1)
    fig.set_size_inches(4,4)
    f.dat.data[:] = log_beta[:,i]
    df.tripcolor(f,axes=axs,cmap=plt.cm.viridis)
    axs.set_aspect('equal')
    axs.set_xticks([])
    axs.set_yticks([])
    axs.set_title(f't={i+1}')
    fig.savefig(f'plots/beta/prior_beta_{i}.png',bbox_inches='tight')
"""
#pickle.dump((L,Lt,Vs,Vst),open(f'{mesh_directory}/beta/beta_basis.p','wb'))
pickle.dump((L,Lt),open(f'{mesh_directory}/beta/beta_basis.p','wb'))
