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
    from speceis_dg.Bases import BedMap, SurfaceMap, BetaMap, AdotMap, LaplaceFromSamples
import logging
logging.captureWarnings(True)
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import pyevtk.hl

from pathlib import Path
import blosc


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

len_scale = 50000
thk_scale = 5000
vel_scale = 100

try:
    run_number = int(sys.argv[1])
except:
    run_number = 999

random_sample = False
linearize_beta = False
data_dir = '../meshes/mesh_1000/'
prefix = 'v1'
results_dir = f'{data_dir}/{prefix}/ensemble_linear/projected_climate_no_calve/'

Path(f'{results_dir}/run_{run_number}/').mkdir(parents=True, exist_ok=True)

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
fm = FenicsModel

with open(f'{data_dir}/surface/time_series/map_cop30.p','rb') as fi:
    data_ref = pickle.load(fi)

L_S_mod = data_ref['model_basis']['coeff_map']
h_S_mod = data_ref['model_basis']['mean_map']

wmean_post = data_ref['coefficients']['post_mean']
beta = data_ref['coefficients']['mean_coeff']

mu_mod = h_S_mod @ beta

S_mean = mu_mod
S_map = L_S_mod

with open(f'{data_dir}/bed/bed_basis.p','rb') as fi:
    data_bed = pickle.load(fi)
L_B_mod = data_bed['model_basis']['coeff_map']
h_B_mod = data_bed['model_basis']['mean_map']

wmean_post_B = data_bed['coefficients']['post_mean']
beta_B = data_bed['coefficients']['mean_coeff']
G_B = data_bed['coefficients']['post_cov_root']

mu_B_mod = h_B_mod @ beta_B

B_mean = mu_B_mod + L_B_mod @ wmean_post_B 
B_map = L_B_mod @ G_B

with open(f'{data_dir}/beta/beta_basis.p','rb') as fi:
    beta_map_x,beta_map_t = pickle.load(fi)

thklim = torch.ones_like(B_mean)
thklim.data[:] = 1/thk_scale

S_init = S_mean + S_map @ wmean_post

with open(f'{data_dir}/adot/adot_basis.p','rb') as fi:
    L0x,L1x,adot_mean,adot_map = pickle.load(fi)

n_hist_steps = 10
hist_length = 75

L1t = torch.vander(torch.linspace(4,4+2*10,36*10),N=2,increasing=True)
#L1t[:,1] = torch.minimum(L1t[:,1],torch.ones_like(L1t[:,1])*6)
L1t[:,0] = 2.0
L1t_hist = torch.vander(torch.linspace(0,4,n_hist_steps),N=2,increasing=True)
L1t_hist[:,0] = 2.0

scales = torch.tensor([1,1./3.,1./3.,1])
L2x = L1x*scales
L2t = torch.eye(L1t.shape[0])

z_B = torch.zeros(B_map.shape[1])
z_S = torch.zeros(S_map.shape[1])
z_S.data[:] = wmean_post

z_beta_ref = torch.randn(beta_map_x.shape[1])
z_beta_t = torch.randn(beta_map_x.shape[1],beta_map_t.shape[1])

z_adot = torch.zeros(adot_map.shape[1])
z_nse = torch.zeros(L2x.shape[1],L2t.shape[1])

solver_args =         {'picard_tol':1e-4,
                       'momentum':0.5,
                       'max_iter':100,
                       'convergence_norm':'l2',
                       'update':True,
                       'enforce_positivity':True}


output_H = df.Function(model.Q_thk,name='H')
output_f = df.Function(model.Q_thk,name='f')
output_S = df.Function(model.Q_thk,name='S')
output_log_beta = df.Function(model.Q_cg1,name='log_beta')
output_delta = df.Function(model.Q_thk,name='delta')
output_misfit = df.Function(model.Q_thk,name='misfit')
output_Bstd = df.Function(model.Q_thk,name='B_std')

initdir = f'{data_dir}/{prefix}/time/states/'
initfile = 'state_003.p'
with open(f'{initdir}/{initfile}','rb') as fi:
    data = pickle.load(fi)
    z_beta_ref.data[:],z_beta_t.data[:],z_B.data[:],z_adot.data[:],z_nse.data[:,:36] = data[:5]
    #Ubar_steady.data[:],Udef_steady.data[:],B_steady.data[:],H_steady.data[:] = [torch.from_numpy(x) for x in data[5]]
 
Ubar_prev = torch.from_numpy(model.Ubar0.dat.data[:])
Udef_prev = torch.from_numpy(model.Udef0.dat.data[:])
S_prev = (S_mean + S_map @ z_S).detach()


bed_map = BedMap(f'{data_dir}/bed/bed_basis.p')
surf_map = SurfaceMap(f'{data_dir}/surface/time_series/map_cop30.p')
beta_map = BetaMap(f'{data_dir}/beta/beta_basis.p')
adot_map_ = AdotMap(f'{data_dir}/adot/adot_basis.p')

if random_sample:
    laplace_ = LaplaceFromSamples([f'{data_dir}/{prefix}/uncertainty/hvps/hvp_{k}.p' for k in range(30)],bed_map=bed_map,beta_map=beta_map,adot_map=adot_map_,method='onepass',maxrank=None)

    n_B = B_map.shape[1]
    n_beta = beta_map_x.shape[1]
    n_adot = adot_map.shape[1]
    n_nse = laplace_.Ubar.shape[0] - (n_B + n_beta + n_adot)

    Delta = laplace_.sample()
    if run_number==0:
        Delta[:] = 0.


    delta_B = Delta[:n_B]
    delta_beta = Delta[n_B:n_B + n_beta]
    #delta_beta_t = Delta[n_B + n_beta:n_B + n_beta + n_beta_t].reshape(z_beta_t.shape)
    delta_adot = Delta[n_B +n_beta:n_B+n_beta+n_adot]
    delta_nse = Delta[n_B+n_beta+n_adot:].reshape(z_nse[:,:36].shape)

    z_B += delta_B
    if not linearize_beta:
        z_beta_ref += delta_beta
        #z_beta_t += delta_beta_t
    z_adot += delta_adot

    z_nse[:,:36] += delta_nse
    if run_number != 0:
        print('run number is not 0',run_number)
        z_nse[:,36:] += torch.randn(z_nse[:,36:].shape)

B = B_mean + B_map @ z_B
S0 = S_mean + S_map @ z_S
S0 = torch.maximum(B+thklim,S0)

H0 = (S0-B)

z_ = adot_mean + adot_map @ z_adot
z_ref = z_[:L0x.shape[1]]
z_dif = z_[L0x.shape[1]:]

adot_ref = L0x @ z_ref
adot_nse = L2x @ z_nse @ L2t.T

adot_dif = adot_ref.reshape(-1,1) + L1x @ z_dif.T.reshape(2,4).T @ L1t.T + L2x @ z_nse @ L2t.T
adot_hist = adot_ref.reshape(-1,1) + L1x @ z_dif.T.reshape(2,4).T @ L1t_hist.T

adot = adot_hist[:,0]

if linearize_beta:
    log_beta_ref = beta_map_x @ z_beta_ref
    log_beta_eps = beta_map_x @ delta_beta/2.0
    log_beta_t = beta_map_x @ (z_beta_t @ beta_map_t.T)
    beta2_ref = torch.exp(log_beta_ref)*(1 + log_beta_eps)
    beta2_ref[beta2_ref<0.001] = 0.001
    beta2 = beta2_ref.reshape(-1,1)*torch.exp(log_beta_t)

else:
    log_beta_ref = beta_map_x @ z_beta_ref
    log_beta_t = beta_map_x @ (z_beta_ref.reshape(-1,1) + z_beta_t @ beta_map_t.T)
    beta2_ref = torch.exp(log_beta_ref)
    beta2 = torch.exp(log_beta_t)

model.calving_factor.assign(0/vel_scale)
model.l.assign(100/thk_scale)

### initialization ###
Ubar,Udef,H0 = fm.apply(H0,B,beta2_ref,adot,Ubar_prev,Udef_prev,model,adjoint,0.0,5,solver_args)
Ubar,Udef,H0 = fm.apply(H0,B,beta2_ref,adot,Ubar,Udef,model,adjoint,0.0,10,solver_args)
Ubar,Udef,H0 = fm.apply(H0,B,beta2_ref,adot,Ubar,Udef,model,adjoint,0.0,20,solver_args)
Ubar,Udef,H0= fm.apply(H0,B,beta2_ref,adot,Ubar,Udef,model,adjoint,0.0,40,solver_args)

n_steps = 25
for i in range(n_steps):
    Ubar,Udef,H0 = fm.apply(H0,B,beta2_ref,adot,Ubar,Udef,model,adjoint,0.0,50,solver_args)

save_initial_state=True
if save_initial_state:
    Ubar,Udef,H0 = fm.apply(H0,B,beta2_ref,adot,Ubar,Udef,model,adjoint,0.0,.01,solver_args)
    with open(f'{results_dir}/run_{run_number}/data_1915.p','wb') as fi:
        data = [H0,B,beta2_ref,adot,Ubar,Udef]
        pickle.dump(data,fi)  # returns data as a bytes object

for k in range(n_hist_steps):
    Ubar,Udef,H0 = fm.apply(H0,B,beta2_ref,adot_hist[:,k],Ubar,Udef,model,adjoint,0.0,hist_length/n_hist_steps,solver_args)

U_series = df.File(f'{results_dir}/run_{run_number}/adjoint/U_series.pvd')
f_series = df.File(f'{results_dir}/run_{run_number}/adjoint/f_series.pvd')
H_series = df.File(f'{results_dir}/run_{run_number}/adjoint/H_series.pvd')
S_series = df.File(f'{results_dir}/run_{run_number}/adjoint/S_series.pvd')
beta_series = df.File(f'{results_dir}/run_{run_number}/adjoint/beta_series.pvd')
adot_series = df.File(f'{results_dir}/run_{run_number}/adjoint/adot_series.pvd')
D_series = df.File(f'{results_dir}/run_{run_number}/adjoint/D_series.pvd')

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
fm = FenicsModel

#model.calving_factor.assign(250/vel_scale)
model.calving_factor.assign(0/vel_scale)
model.l.assign(50/thk_scale)

years = range(1985,1985+36*10) 
 
years_to_record = [y for y in range(1985,2023)] + [y for y in range(2023,1985+360,5)] + [1985+360] 

surfs = []

for j,y in enumerate(years):
    beta2_i = beta2[:,(y-1985)%36]
    adot_i = adot_dif[:,min(y,2500)-1985] + adot_nse[:,y-1985]
    #adot_i = adot_dif[:,y-1985]
    Ubar,Udef,H0 = fm.apply(H0,B,beta2_i,adot_i,Ubar,Udef,model,adjoint,0.0,1.0,solver_args)
    S = (B + H0)

    surfs.append(S.detach().numpy())
    
    model.project_surface_velocity()
    U_series.write(model.U_s,time=y)
    beta_series.write(model.beta2,time=y)
    output_S.interpolate(model.S)
    S_series.write(output_S,time=y)
    H_series.write(model.H0,time=y) 
    adot_series.write(model.adot,time=y)
    output_f.interpolate(model.floating)
    f_series.write(output_f,time=y)

    if y in years_to_record:
        with open(f'{results_dir}/run_{run_number}/data_{y}.p','wb') as fi:
            data = [H0,B,beta2_i,adot_i,Ubar,Udef]
            pickle.dump(data,fi)  # returns data as a bytes object
        print(y)

output_delta = df.Function(model.Q_thk,name='delta')
for j,(S_tilde,y) in enumerate(zip(surfs,years)):
    output_delta.dat.data[:] = S_tilde - surfs[years.index(2013)]
    D_series.write(output_delta,time=y)
