import sys
sys.path.append('../..')
import firedrake as df
import numpy as np
from speceis_dg.hybrid import CoupledModel


class ISMIP_HOM_C:
    def __init__(self,results_dir):
        for l in [5000,10000,20000,40000,80000,160000]:
            mesh = df.PeriodicUnitSquareMesh(50,50,diagonal='crossed',name='mesh')
            model = self.model = CoupledModel(mesh,velocity_function_space='MTW',periodic=True,sia=False,ssa=False,vel_scale=100,thk_scale=1e3,len_scale=l,beta_scale=1e3,time_scale=1,g=9.81,rho_i=917.,rho_w=1000.0,n=3.0,A=1e-16,eps_reg=1e-8,thklim=1e-3,theta=1.0,alpha=0,p=4,membrane_degree=2,shear_degree=3)

            X = df.SpatialCoordinate(mesh)
            x,y = X

            alpha_ismip = df.Constant(np.deg2rad(0.1))
            omega_ismip = df.Constant(2*np.pi)

            S_exp = -x*df.tan(alpha_ismip)/model.delta
            S_lin = -x*df.tan(alpha_ismip)/model.delta
            S_grad_lin = df.diff(S_lin,X)

            B_exp = S_exp - 1
            B_lin = S_lin - 1
            B_grad_lin = df.diff(B_lin,X)

            beta_exp = 1 + df.sin(omega_ismip*x)*df.sin(omega_ismip*y)

            model.H0.interpolate(S_exp - B_exp)
            model.B.interpolate(B_exp)

            model.S_lin.interpolate(S_lin)
            model.B_lin.interpolate(B_lin)

            model.S_grad_lin.assign(S_grad_lin)
            model.B_grad_lin.assign(B_grad_lin)

            model.beta2.interpolate(beta_exp)

            Q_cg2 = df.VectorFunctionSpace(model.mesh,"CG",1)

            with df.CheckpointFile(f'{results_dir}/ismipc-L-{l}.h5', 'w') as afile:
                afile.save_mesh(mesh)  # optional
                model.step(0,1e-10)
                U_s = self.U_s = df.interpolate(model.Ubar0 - 0.25*model.Udef0,Q_cg2)
                afile.save_function(model.H0, idx=0, name='H')
                afile.save_function(U_s, idx=0, name='U_s')

if __name__=='__main__':
    ismipa = ISMIP_HOM_C('./results/')
