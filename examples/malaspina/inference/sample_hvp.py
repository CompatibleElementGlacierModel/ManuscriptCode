
import os
import sys
import pickle
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.append('../../..')

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("once",category=DeprecationWarning)
    import firedrake as df
    from firedrake.petsc import PETSc
    from speceis_dg.hybrid_diff import CoupledModel, CoupledModelAdjoint, FenicsModel, SurfaceIntegral, SurfaceCost, VelocityIntegral, VelocityCost
import logging
logging.captureWarnings(True)
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

import pyevtk.hl
import time

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

def kl_identity(mu,sigma):
    return 0.5*((mu**2).sum() + (sigma**2).sum() - torch.log(sigma**2).sum() - len(mu))

def kl_identity_lr(mu,sigma,A):
    return 0.5*(mu.T @ mu + (sigma**2).sum() + (A**2).sum() - torch.log(sigma**2).sum() - torch.logdet(torch.eye(A.shape[1]) + (A.T / (sigma**2)) @ A) - len(mu))

len_scale = 50000
thk_scale = 5000
vel_scale = 100


#data_dir = '../meshes/mesh_1899/'
#data_dir = '../meshes/mesh_2200/'
data_dir = '../meshes/mesh_2201/'
prefix = 'v5'

initialize = False
hot_start = False

results_dir = f'{data_dir}/{prefix}/uncertainty/'

mesh = df.Mesh(f'{data_dir}/mesh.msh',name='mesh')
mesh.coordinates.dat.data[:] -= (mesh.coordinates.dat.data.max(axis=0) + mesh.coordinates.dat.data.min(axis=0))/2.
mesh.coordinates.dat.data[:] /= len_scale

config = {'solver_type': 'gmres',
          'sliding_law': 'Budd',
          'velocity_function_space':'MTW',
          'sia':False,
          'vel_scale': vel_scale,
          'thk_scale': thk_scale,
          'len_scale': len_scale,
          'beta_scale': 1000.,
          'theta': 1.0,
          'thklim': 1./thk_scale,
          'alpha': 1000.0,
          'z_sea': 0.0,
          'boundary_markers':[1000,1001],
          'calve': 'b'}
  
model = CoupledModel(mesh,**config)
adjoint = CoupledModelAdjoint(model)
ui = VelocityIntegral(model,mode='lin',p=2.0,gamma=1.0)
#ui = VelocityIntegral(model,mode='log',p=2.0,gamma=1.0)

load_relative=False
if initialize:
    files=['cop30']
    times=['2013_175']
    dates=[int(s[:4]) for s in times]
else:
    if load_relative:
        files=['rel_1995','rel_2000','rel_2003','rel_2007','rel_2009','rel_2010','cop30','rel_2014','rel_2015','rel_2016','rel_2017','rel_2018','rel_2019','rel_2020','rel_2021']
    else:
        files=['1995','2000','2003','2007','2009','2010','cop30','2014','2015','2016','2017','2018','2019','2020','2021']

    times=['1995','2000','2003','2007','2009','2010','2013','2014','2015','2016','2017','2018','2019','2020','2021']
    dates=[int(s[:4]) for s in times]

years = [y for y in range(1985,2021)]
inds = [y-1985 for y in years]
not_inds = [y-1985 for y in range(1985,2021) if y not in years]

surfaces = [None for i in range(len(years))]
velocities = [None for i in range(len(years))]
velocities_0 = [None for i in range(len(years))]
velocities_tau = [None for i in range(len(years))]

with open(f'{data_dir}/velocity/velocity.p','rb') as fi:
    [v_avg,v_mask] = pickle.load(fi)
    v_avg/=vel_scale
v_avg[np.isnan(v_avg)] = 0.0

for i,y in enumerate(years):
    try:
        with open(f'{data_dir}/velocity/itslive_annual/velocity_{y}.p','rb') as fi:
            v = pickle.load(fi)/vel_scale
        velocities_0[i] = v
    except FileNotFoundError:
        pass

v_avg = np.nanmean(velocities_0[:-2],axis=0)
v_avg[np.isnan(v_avg)] = 0.0

for i,y in enumerate(years):
    try:
        with open(f'{data_dir}/velocity/itslive_annual/velocity_{y}.p','rb') as fi:
            v = pickle.load(fi)/vel_scale
        v_tau = np.ones_like(v)
        v_tau[np.isnan(v)] = 0.0
        v_tau = v_tau[:,0]
        v[np.isnan(v)] = v_avg[np.isnan(v)]
        #v_tau[np.linalg.norm(v,axis=1)<0.2] = 0.  ###!!!
        #v_tau[np.linalg.norm(v,axis=1)<(10/100)] = 0.0
        velocities[i] = v
        #velocities[i] = v_avg
        velocities_tau[i] = v_tau
    except FileNotFoundError:
        v = v_avg
        v_tau = np.ones_like(v)
        #v_tau[np.linalg.norm(v,axis=1)<(10/100)] = 0.0
        velocities[i] = v
        velocities_tau[i] = v_tau[:,0]
#velocities[0] = v_avg
#velocities_tau[0] = np.ones_like(v_avg)[:,0]

for i,y in enumerate(years):
    try:
        f = dict(zip(dates,files))[y]
        with open(f'{data_dir}/surface/time_series/map_{f}.p','rb') as fi:
            surfaces[i] = pickle.load(fi)
    except KeyError:
        pass

data_ref = surfaces[years.index(2013)]

#S_obs = data_ref['data']['z_obs']

L_S_obs = data_ref['observation_basis']['coeff_map']
h_S_obs = data_ref['observation_basis']['mean_map']

L_S_mod = data_ref['model_basis']['coeff_map']
h_S_mod = data_ref['model_basis']['mean_map']

wmean_post = data_ref['coefficients']['post_mean']
beta = data_ref['coefficients']['mean_coeff']

F_model = data_ref['inverse_maps']['mil_inner_inv']
Sigma_model = data_ref['inverse_maps']['model_to_basis_cov']

mu_mod = h_S_mod @ beta
mu_obs = h_S_obs @ beta

S_mean = mu_mod
S_map = L_S_mod

with open(f'{data_dir}/bed/bed_basis_checkerboard.p','rb') as fi:
    data_bed = pickle.load(fi)
L_B_obs = data_bed['observation_basis']['coeff_map']
h_B_obs = data_bed['observation_basis']['mean_map']

L_B_mod = data_bed['model_basis']['coeff_map']
h_B_mod = data_bed['model_basis']['mean_map']

wmean_post_B = data_bed['coefficients']['post_mean']
beta_B = data_bed['coefficients']['mean_coeff']
G_B = data_bed['coefficients']['post_cov_root']
B_obs = data_bed['data']['z_obs']

mu_B_mod = h_B_mod @ beta_B
mu_B_obs = h_B_obs @ beta_B
B_mean = mu_B_mod + L_B_mod @ wmean_post_B 
B_map = L_B_mod @ G_B

with open(f'{data_dir}/beta/beta_basis.p','rb') as fi:
    beta_map_x,beta_map_t = pickle.load(fi)

model.beta2.interpolate(df.Constant(1.0))
thklim = torch.ones_like(B_mean)
thklim.data[:] = 1/thk_scale

fm = FenicsModel
sm = SurfaceCost
um = VelocityCost

S_init = S_mean + S_map @ wmean_post

with open(f'{data_dir}/adot/adot_basis.p','rb') as fi:
    L0x,L1x,adot_mean,adot_map = pickle.load(fi)

n_hist_steps = 10
hist_length = 75

L1t = torch.vander(torch.linspace(4,6,36),N=2,increasing=True)
L1t[:,0] = 2.0
L1t_hist = torch.vander(torch.linspace(0,4,n_hist_steps),N=2,increasing=True)
L1t_hist[:,0] = 2.0
if initialize:
    L1t[:] = 0.
    L1t_hist[:] = 0.

scales = torch.tensor([1,1./3.,1./3.,1])
#scales = torch.ones(4)*0.01
L2x = L1x*scales
L2t = torch.eye(L1t.shape[0])

z_B = torch.zeros(B_map.shape[1],requires_grad=True)
z_S = torch.zeros(S_map.shape[1],requires_grad=True)
z_S.data[:] = wmean_post

z_beta_ref = torch.randn(beta_map_x.shape[1],requires_grad=True)
z_beta_t = torch.randn(beta_map_x.shape[1],beta_map_t.shape[1],requires_grad=True)

z_beta_ref.data[:]*=0.001
z_beta_t.data[:]*=0.001

z_adot = torch.zeros(adot_map.shape[1],requires_grad=True)
z_nse = torch.zeros(L2x.shape[1],L2t.shape[1],requires_grad=True)

solver_args =         {'picard_tol':1e-3,
                       'momentum':0.5,
                       'max_iter':100,
                       'convergence_norm':'l2',
                       'update':True,
                       'enforce_positivity':True}

sigma2_obs = (50/thk_scale)**2
#sigma2_vel = (50/vel_scale)**2
sigma2_vel = (25/vel_scale)**2
rel_frac = 0.0
sigma2_mod = (1/thk_scale)**2

rho = 500.0
surface_loss_weight = 1
zero = torch.zeros_like(S_init)

phi = df.TestFunction(model.Q_thk)
area = torch.from_numpy(df.assemble(phi*df.dx).dat.data[:])

Ubar_prev = torch.tensor(model.Ubar0.dat.data[:])
Udef_prev = torch.tensor(model.Udef0.dat.data[:])
Ubar_steady = torch.tensor(model.Ubar0.dat.data[:])
Udef_steady = torch.tensor(model.Udef0.dat.data[:])
B_steady = torch.tensor(model.B.dat.data[:])
H_steady = torch.tensor(model.H0.dat.data[:])

i = 0
tt = 0
inits = []

def closure():
    global Ubar_prev
    global Udef_prev
    global Ubar_steady
    global Udef_steady
    global B_steady
    global H_steady
    global i
    global tt
    global inits

    lbfgs.zero_grad()


    B = B_mean + B_map @ z_B
    S0 = S_init# + 50/thk_scale
    S0 = torch.maximum(B+thklim,S0)
    
    H0 = (S0-B)

    z_ = adot_mean + adot_map @ z_adot
    z_ref = z_[:L0x.shape[1]]
    z_dif = z_[L0x.shape[1]:]

    adot_ref = L0x @ z_ref
    adot_dif = adot_ref.reshape(-1,1) + L1x @ z_dif.T.reshape(2,4).T @ L1t.T + L2x @ z_nse @ L2t.T
    adot_hist = adot_ref.reshape(-1,1) + L1x @ z_dif.T.reshape(2,4).T @ L1t_hist.T
    
    adot = adot_hist[:,0]

    log_beta_ref = beta_map_x @ z_beta_ref
    log_beta_t = beta_map_x @ (z_beta_ref.reshape(-1,1) + z_beta_t @ beta_map_t.T)
    beta2_ref = torch.exp(log_beta_ref)
    beta2 = torch.exp(log_beta_t)

    Ubar,Udef,H0 = fm.apply(H0,B,beta2_ref,adot,Ubar_prev,Udef_prev,model,adjoint,0.0,5,solver_args)
    Ubar,Udef,H0 = fm.apply(H0,B,beta2_ref,adot,Ubar,Udef,model,adjoint,0.0,10,solver_args)
    Ubar,Udef,H0 = fm.apply(H0,B,beta2_ref,adot,Ubar,Udef,model,adjoint,0.0,20,solver_args)
    Ubar,Udef,H0 = fm.apply(H0,B,beta2_ref,adot,Ubar,Udef,model,adjoint,0.0,40,solver_args)

    if i==0:
        n_steps = 25
    else:
        n_steps = 25
    
    for k in range(n_steps):
        Ubar,Udef,H0 = fm.apply(H0,B,beta2_ref,adot,Ubar,Udef,model,adjoint,0.0,50,solver_args)

    for k in range(n_hist_steps):
        Ubar,Udef,H0 = fm.apply(H0,B,beta2_ref,adot_hist[:,k],Ubar,Udef,model,adjoint,0.0,hist_length/n_hist_steps,solver_args)
    
    Ubar,Udef,H0 = fm.apply(H0,B,beta2_ref,adot_hist[:,k],Ubar,Udef,model,adjoint,0.0,0.01,solver_args)

    L_surface = 0.0
    L_velocity = 0.0

    S_count = 0
    U_count = torch.tensor(1e-10)

    for j,(U_obs,U_tau,data,y) in enumerate(zip(velocities,velocities_tau,surfaces,years)):
        #Ubar_init,Udef_init = inits[j]
        if static_traction:
            beta2_i = beta2_ref
            adot_i = adot_dif[:,y-1985]
        else:
            beta2_i = beta2[:,y-1985]
            adot_i = adot_dif[:,y-1985]
            #Ubar,Udef,H0 = fm.apply(H0,B,beta2_i,adot_i,Ubar_init,Udef_init,model,adjoint,0.0,1.0,solver_args)
            Ubar,Udef,H0 = fm.apply(H0,B,beta2_i,adot_i,Ubar,Udef,model,adjoint,0.0,1.0,solver_args)
        #inits[j] = (Ubar.detach(),Udef.detach())
        S = (B + H0)
        
        if data is not None:
            if not relative_surface_loss:
                z_1 = Sigma_model @ ((L_S_mod.T / sigma2_mod) @ (S - mu_mod))
            
                mu_obs = data['observation_basis']['mean_map'] @ beta
                L_S_obs = data['observation_basis']['coeff_map']
                F_model = data['inverse_maps']['mil_inner_inv']
                S_inds = data['data']['s_inds']
                S_obs = data['data']['z_obs'][S_inds]

                L_S_obs = L_S_obs[S_inds]
                mu_obs = mu_obs[S_inds]

                S_pred = (L_S_obs @ z_1 + mu_obs)

            else:
                z_1 = Sigma_model @ ((L_S_mod.T /sigma2_mod) @ (S - S_steady))
            
                L_S_obs = data['observation_basis']['coeff_map']
                F_model = data['inverse_maps']['mil_inner_inv']


                if y!=2013:
                    S_obs = data['data']['z_obs']
                else:
                    S_inds = data['data']['s_inds']
                    S_obs = torch.zeros_like(data['data']['z_obs'])[S_inds]
                    L_S_obs = L_S_obs[S_inds]

                S_pred = L_S_obs @ z_1

            L_surface_i = surface_loss_weight*torch.dot(S_pred-S_obs, (S_pred-S_obs)/sigma2_obs - L_S_obs@(F_model@(L_S_obs.T@((S_pred - S_obs)/sigma2_obs)))/sigma2_obs)
            #if y!=2013:
            #    L_surface_i*=9
            L_surface += L_surface_i


            S_count += 1
        
        if U_obs is not None:
            #tau_obs = 10*torch.ones(U_obs.shape[0])#1./np.maximum((np.linalg.norm(U_obs,axis=1)*rel_frac)**2,sigma2_vel)
            tau_obs = U_tau/np.maximum((np.linalg.norm(U_obs,axis=1)*rel_frac)**2,sigma2_vel)
            if not relative_velocity_loss:
                L_velocity_i = um.apply(Ubar,Udef,U_obs,tau_obs,v_mask,ui)*rho
            else:
                Ubar_rel = Ubar - Ubar_steady
                Udef_rel = Udef - Udef_steady
                U_obs_rel = U_obs - v_avg
                L_velocity_i = um.apply(Ubar_rel, Udef_rel,U_obs_rel,tau_obs,v_mask,ui)*rho
            L_velocity = L_velocity + L_velocity_i
            U_count += 1
        
    L_prior = (z_B**2).sum() + (z_beta_ref**2).sum() + (z_beta_t**2).sum() + ((z_adot)**2).sum() + (z_nse**2).sum()
    if initialize:
        L_velocity = L_velocity/np.sqrt(U_count)
    else:
        L_velocity = L_velocity/np.sqrt(U_count)

    L =  L_velocity + L_surface + L_prior
    dV_0 = ((area*adot).sum()/area.sum()).detach()
    dV_1 = ((area*adot_i).sum()/area.sum()).detach()
    print(i,L.item(),L_velocity.item(),L_surface.item(),L_prior.item(),dV_0.item(),dV_1.item(),adot.max().detach().item(),adot_i.max().detach().item())

    L.backward()
    
    i+=1
    tt+=1
    return L


lbfgs = torch.optim.LBFGS([z_B,z_beta_ref,z_beta_t,z_adot,z_nse],
                        history_size=50,
                        line_search_fn="strong_wolfe")#,max_iter=1,max_eval=20)

initdir = f'{data_dir}/v5/time/states/'
initfile = 'state_009.p'
with open(f'{initdir}/{initfile}','rb') as fi:
    data = pickle.load(fi)
    z_beta_ref.data[:],z_beta_t.data[:],z_B.data[:],z_adot.data[:],z_nse.data[:] = data[:5]
    Ubar_steady.data[:],Udef_steady.data[:],B_steady.data[:],H_steady.data[:] = [torch.from_numpy(x) for x in data[5]]

i = 0 

static_traction = False
relative_surface_loss = False
relative_velocity_loss = True

closure()

g1_B = torch.tensor(z_B.grad)
g1_beta = torch.tensor(z_beta_ref.grad)
g1_adot = torch.tensor(z_adot.grad)
g1_nse = torch.tensor(z_nse.grad)
g1 = torch.hstack((g1_B,g1_beta,g1_adot.ravel(),g1_nse.ravel()))

n_B = g1_B.shape[0]
n_beta = g1_beta.shape[0]
n_adot = g1_adot.ravel().shape[0]
n_nse = g1_nse.ravel().shape[0]

z_B0 = torch.tensor(z_B.data[:])
z_beta0 = torch.tensor(z_beta_ref.data[:])
z_adot0 = torch.tensor(z_adot.data[:])
z_nse0 = torch.tensor(z_nse.data[:])

r = 1e-3
n_samples = 100

for l in range(1,30):

    size = g1.shape[0]

    Z = torch.randn(size,n_samples)
    G = torch.zeros(size,n_samples)
    Z_B = Z[:n_B]
    Z_beta = Z[n_B:n_B + n_beta]
    Z_adot = Z[n_B + n_beta:n_B + n_beta + n_adot]
    Z_nse = Z[n_B + n_beta + n_adot:]
    for ii in range(n_samples):
        z_B.data[:] = z_B0 + r*Z_B[:,ii]
        z_beta_ref.data[:] = z_beta0 + r*Z_beta[:,ii]
        z_adot.data[:] = z_adot0 + r*Z_adot[:,ii].reshape((z_adot0.shape))
        z_nse.data[:] = z_nse0 + r*Z_nse[:,ii].reshape((z_nse0.shape))
        closure()
        G[:n_B,ii] = torch.tensor(z_B.grad)
        G[n_B:n_B+n_beta,ii] = torch.tensor(z_beta_ref.grad)
        G[n_B+n_beta:n_B + n_beta + n_adot,ii] = torch.tensor(z_adot.grad.ravel())
        G[n_B + n_beta + n_adot:,ii] = torch.tensor(z_nse.grad.ravel())

    HVP = (G - g1.reshape(-1,1))/r
    with open(f'{results_dir}/hvps/hvp_{l}.p','wb') as fi:
        pickle.dump((G,g1,Z),fi)


    


