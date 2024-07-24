import torch
torch.set_default_dtype(torch.float64)
from StructuredKernelInterpolation import *
import rasterio
import firedrake as df
import numpy as np
import pickle
import scipy
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import binary_erosion
import os

device = 'cuda'

def get_training_data_from_bbox(dem,bounds,X_loc,X_scale,Z_loc,Z_scale,target_resolution,nodatavalue,smooth=False,band=0,mask_inds=False):

    # Get DEM elevations
    Z_dem = dem.read()[band][::-1].astype(float)

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

    x_ = (x_dem[::skip_x].reshape(-1,1) - X_loc[0])/X_scale
    y_ = (y_dem[::skip_y].reshape(-1,1) - X_loc[1])/X_scale

    X_dem,Y_dem = np.meshgrid(x_,y_)

    if smooth:
        V = Z_dem.copy()
        V[Z_dem==nodatavalue + np.isnan(Z_dem)] = 0
        VV = gaussian_filter(V,[skip_x,skip_y],mode='nearest')

        W = np.ones_like(V)
        W[Z_dem==nodatavalue + np.isnan(Z_dem)] = 0
        WW = gaussian_filter(W,[skip_x,skip_y],mode='nearest')
        ZZ = VV/WW
        ZZ[Z_dem==nodatavalue] = nodatavalue
        Z_dem = ZZ[::skip_y,::skip_x]
    else:
        Z_dem = Z_dem[::skip_y,::skip_x]

    if mask_inds:

        gx,gy = np.gradient(Z_dem,dx*skip_x,dy*skip_y)
        gnorm = (gx**2 + gy**2)**0.5
        steepness_threshold=15
        steepness = np.rad2deg(np.arctan(gnorm))
        k = (steepness<steepness_threshold)#.ravel(order='F')
        k = binary_erosion(k,iterations=1).ravel(order='F')

    else:
        k = np.ones_like(Z_dem).ravel(order='F').astype(bool)

    z = Z_dem.ravel(order='F')
    inds = (z!=nodatavalue)*~np.isnan(z)
    
    Z_train = z[inds]
    k = k[inds]

    Z_train -= Z_loc
    Z_train /= Z_scale

    X_train = np.c_[X_dem.ravel(order='F'),Y_dem.ravel(order='F')][inds]
    print(X_train.shape,Z_train.shape,k.shape)
    return X_train,Z_train,k

def build_interpolation_matrix(X,X_,x_,y_,p=2):
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

        w = (1./dist)**p
        w/=w.sum()

        rows.append(torch.ones((4))*ii)
        cols.append(torch.tensor(bbox))
        vals.append(w) 

    inds = torch.vstack((torch.hstack(rows),torch.hstack(cols)))
    tens = torch.sparse_coo_tensor(inds,torch.hstack(vals),(X.shape[0],m))
    return tens,torch.transpose(tens,1,0)

def k(x1,x2,l,amplitude):
    D = torch.cdist(x1.unsqueeze(-1),x2.unsqueeze(-1))
    return amplitude*torch.exp(-D**2/(l**2))

def k_rq(x1,x2,l,amplitude,alpha=1.0):
    D = torch.cdist(x1.unsqueeze(-1),x2.unsqueeze(-1))
    return amplitude*(1 + D**2/(2*alpha*l**2))**(-alpha)

def k_se(x1,x2,l,amplitude):
    D = torch.cdist(x1.unsqueeze(-1),x2.unsqueeze(-1))
    return amplitude*torch.exp(-D**2/(l**2))

def k_mat(x1,x2,l,amplitude):
    D = torch.cdist(x1.unsqueeze(-1),x2.unsqueeze(-1))
    return torch.exp(-np.sqrt(3)*D/l)*(1 + np.sqrt(3)*D/l)

def k_mat2(x1,x2,l,amplitude):
    D = torch.cdist(x1.unsqueeze(-1),x2.unsqueeze(-1))
    return amplitude*torch.exp(-np.sqrt(5)*D/l)*(1 + np.sqrt(5)*D/l + 5*D**2/(3*l**2))

def reshape_fortran(x, shape):
  if len(x.shape) > 0:
    x = x.permute(*reversed(range(len(x.shape))))
  return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def build_surface_representation(mesh_directory,dem_paths,nodatavalue,known_beta=None,return_posterior=True,build_model_to_coeff_map=True,save_model_basis=True,plotting=True,offset=0.0,smooth=True,band=0,index_directory=None,sigma2_obs=(50/5000)**2,sigma2_model=(1/5000)**2,mask=None,mask_inds=False):
    mesh = df.Mesh(f'{mesh_directory}/mesh.msh',name='mesh')


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

    X -= X_loc
    X /= X_scale

    X_cg2 -= X_loc
    X_cg2 /= X_scale

    bounds = (707145.3017168527 - 3000, 780055.3955662812 + 3000, 1155770.1396169967 - 3000, 1253848.0942367027 + 3000)
    X_trains = []
    Z_trains = []
    k_trains = []
    for dem_path in dem_paths:
        with rasterio.open(dem_path) as dem:
            X_train,Z_train,inds = get_training_data_from_bbox(dem,bounds,X_loc,X_scale,Z_loc,Z_scale,500,nodatavalue,smooth=smooth,band=band,mask_inds=mask_inds)

        Z_train += offset/Z_scale
        X_train = torch.from_numpy(X_train)#.to(torch.float)

        X_trains.append(X_train)
        Z_trains.append(torch.from_numpy(Z_train))#.to(torch.float))
        k_trains.append(inds)

    X_train = torch.vstack(X_trains)
    Z_train = torch.hstack(Z_trains)
    inds = np.hstack(k_trains)

    X = torch.from_numpy(X)#.to(torch.float)
    X_cg2 = torch.from_numpy(X_cg2)

    tree = KDTree((mesh.coordinates.dat.data - X_loc)/X_scale)
    l = 3000.0/X_scale
    near = tree.query(X_train,k=1)[0] < 2*l
    X_train = X_train[near]
    #print(inds.shape,inds,near.shape,near)
    inds = inds[near]

    X_all = torch.vstack((X_train,X_cg2))#.to(torch.float)
    #Z_train = torch.from_numpy(Z_train).to(torch.float)#
    Z_train = Z_train[near]

    nx = 301

    x_ = torch.linspace(-1.1,1.1,nx)
    y_ = torch.linspace(-1.1,1.1,nx)

    X_,Y_ = torch.meshgrid([x_,y_])
    X_ = torch.vstack((batch_vec(X_.unsqueeze(0)),batch_vec(Y_.unsqueeze(0)))).T

    W,W_t = build_interpolation_matrix(X_all,X_,x_,y_)

    Kx = k_mat(x_,x_,l,amplitude=1)
    k = int(2.2*(x_.max() - x_.min())/l)
    cols = [int(j) for j in (np.round(np.linspace(0,len(x_)-1,k)))]
    Ks = Kx[cols,:][:,cols]
    u,s,v = np.linalg.svd(Ks)
    Lx = Kx[:,cols] @ u * 1./np.sqrt(s) @ u.T
    #f = 1/np.sqrt((Lx**2).sum(axis=1))
    #Lx = torch.diag(f) @ Lx

    Ly = Lx

    L = W @ torch.kron(Lx,Ly)

    L *= 1000/Z_scale

    X__,Y__ = torch.meshgrid(x_[cols],x_[cols])
    X_center = torch.hstack((reshape_fortran(X__,(-1,1)),reshape_fortran(Y__,(-1,1))))
    if index_directory:
        near = pickle.load(open(f'{index_directory}/bed/low_res_valid_center_bed.p','rb'))
    else:
        near = tree.query(X_center,k=1)[0] < 2*l

    L = L[:,near]

    cell_index = [mesh.locate_cell(xx) for xx in X_center[near]*X_scale+X_loc]
    inside = []
    for i in cell_index:
        if i is not None:
            inside.append(True)
        else:
            inside.append(False)

    Sigma_diag = (L**2).sum(axis=1)

    n_train = len(X_train)
    n_test = X.shape[0]
    n = len(X_all)

    Psi =  torch.hstack((L,torch.ones((n,1)),X_all,X_all**2,(X_all[:,0]*X_all[:,1]).reshape(-1,1)))

    h_train = Psi[:n_train,-6:]
    h_test = Psi[n_train:,-6:]
    h_all = Psi[:,-6:]
    if known_beta is None:
        beta = torch.linalg.solve(h_train.T / sigma2_obs @ h_train, h_train.T/sigma2_obs @ Z_train)
    else:
        beta = known_beta

    mu_train = h_train @ beta
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

    if return_posterior:
        Z_resid = Z_train - mu_train

        I = torch.eye(L_train.shape[1])
        Tau_post = I + (L_train.T / sigma2_obs) @ L_train

        st,ut = torch.linalg.eigh(Tau_post)
        G = ut * 1./torch.sqrt(st) @ ut.T
        wmean_post = G @ (G.T @ (((L_train.T/sigma2_obs)) @ Z_resid))
    else:
        G = None
        wmean_post = None

    if build_model_to_coeff_map:
        Tau_model = L_test.T / sigma2_model @ L_test + torch.eye(L_test.shape[1])
        F_model = torch.linalg.inv(Tau_model + L_train[inds].T/sigma2_obs @ L_train[inds])
        Sigma_model = torch.linalg.inv(Tau_model)
        #ll,vv = torch.linalg.eigh(Tau_model + L_train.T/sigma2_obs @ L_train)
        #print(min(1./ll))
        #inds = (1/ll)>0
        #F_model = (vv[:,inds] * 1./ll[inds]) @ vv[:,inds].T
    else:
        F_model = None
        Sigma_model = None

    #if build_
    Z_inv = None#torch.linalg.pinv(L_test @ G)

    if plotting==True:
        pl_vars = [X,mu_test + L_test @ wmean_post]
    else:
        pl_vars = None
    #mean_train = mu_train + L_train @ wmean_post

    observation_basis = [L_train,h_train]
    if save_model_basis:
        model_basis = [L_test,h_test]
    else:
        model_basis = [None,None]
    coefficients = [G,wmean_post,beta]
    inverse_operators = [F_model,Sigma_model,Z_inv]
    data = [X_train,Z_train]

    s = os.path.basename(dem_path)
    print(s)

    t = 2013

    for y in range(1985,2023):
        if str(y) in s:
            t = y

    data_dict = {'observation_basis':{'coeff_map':L_train,
                                      'mean_map':h_train,
                                      'mean':mu_train},
                 'model_basis':{'coeff_map':L_test,
                                'mean_map':h_test,
                                'mean':mu_test},
                 'coefficients':{'post_cov_root':G,
                                 'post_mean':wmean_post,
                                 'mean_coeff':beta},
                 'inverse_maps':{'mil_inner_inv':F_model,
                                 'model_to_basis_cov':Sigma_model,
                                 'model_to_basis_map':Z_inv},
                 'data':{'x_obs':X_train,
                         'z_obs':Z_train,
                         's_inds':inds},
                 'time':t,
                 'plotting': pl_vars

                 }             

    return data_dict

mesh_directory = '../meshes/mesh_1000/'
index_directory = '../meshes/mesh_1899/'

nodatavalue = 3.4028234663852886e+38
cop_path = '../data/dem/cop30_3338.tif'
cop_data = build_surface_representation(mesh_directory,[cop_path],nodatavalue,return_posterior=True,smooth=True,index_directory=index_directory,mask_inds=True)
t = cop_data['time']

pickle.dump(cop_data,open(f'{mesh_directory}/surface/time_series/map_cop30.p','wb'))


time_series_dir = '../data/surface/proj3338_400m'
time_series_paths = os.listdir(time_series_dir)
time_series_paths_m = [t for t in time_series_paths if 'Malaspina' in t]
time_series_paths_a = [t for t in time_series_paths if 'Agassiz' in t]
years = range(1985,2022)
all_years = []
for y in years:
    paths = [y]
    for t in time_series_paths_m:
        if str(y) in t:
            paths.append(t)
    for t in time_series_paths_a:
        if str(y) in t:
            paths.append(t)

    if len(paths)>1:
        all_years.append(paths)

for paths in all_years:
    dem_paths = [f'{time_series_dir}/{f}' for f in paths[1:]]
    data = build_surface_representation(mesh_directory,dem_paths,nodatavalue,known_beta=cop_data['coefficients']['mean_coeff'],return_posterior=False,save_model_basis=False,plotting=False,smooth=False,band=0,index_directory=index_directory)
    t = data['time']
    print(t)
    pickle.dump(data,open(f'{mesh_directory}/surface/time_series/map_{t}.p','wb'))

"""
for paths in all_years:
    print(paths)
    dem_paths = [f'{time_series_dir}/{f}' for f in paths[1:]]
    data = build_surface_representation(mesh_directory,dem_paths,nodatavalue,known_beta=torch.zeros_like(cop_data['coefficients']['mean_coeff']),return_posterior=False,save_model_basis=False,plotting=False,smooth=False,band=1,build_model_to_coeff_map=True,index_directory=index_directory)
    t = data['time']
    print(t)
    pickle.dump(data,open(f'{mesh_directory}/surface/time_series/map_rel_{t}.p','wb'))
"""

