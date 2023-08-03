import sys
sys.path.append('../..')
import firedrake as df
import numpy as np
from speceis_dg.hybrid_flux_formulation import CoupledModel

class ISMIP_HOM_F:
    def __init__(self,results_dir):
        for l in ['000', '001']:

            mesh = df.PeriodicUnitSquareMesh(50,50,diagonal='crossed',name='mesh')
            model = CoupledModel(mesh,velocity_function_space='RT',sia=False,ssa=False,vel_scale=100,thk_scale=1e3,len_scale=1e5,beta_scale=1e3,time_scale=1,g=9.81,rho_i=917.,rho_w=1000.0,n=1.0,A=2.14e-7,eps_reg=1e-8,thklim=1e-3,theta=1.0,alpha=0,p=2,membrane_degree=2,shear_degree=3)

            X = df.SpatialCoordinate(mesh)
            x,y = X

            alpha_ismip = df.Constant(np.deg2rad(3.0))

            S_exp = -x*df.tan(alpha_ismip)/model.delta
            S_lin = -x*df.tan(alpha_ismip)/model.delta
            S_grad_lin = df.diff(S_lin,X)

            B_exp = S_exp - 1 + 0.1*df.exp(-((x-0.5)**2 + (y-0.5)**2)/(0.1**2))
            B_lin = S_lin - 1
            B_grad_lin = df.diff(B_lin,X)

            if l=='001':
                beta_exp = df.Constant(4.673)
            else:
                beta_exp = df.Constant(1000)
            
            model.H0.interpolate(S_exp - B_exp)
            model.B.interpolate(B_exp)

            model.S_lin.interpolate(S_lin)
            model.B_lin.interpolate(B_lin)

            model.S_grad_lin.assign(S_grad_lin)
            model.B_grad_lin.assign(B_grad_lin)

            model.beta2.interpolate(beta_exp)

            S_file = df.File(f'{results_dir}/S.pvd')
            Us_file = df.File(f'{results_dir}/U_s.pvd')
            S_out = df.Function(model.Q_thk)

            t = 0
            t_end = 250
            dt = 25
            model.F0.vector()[:]=0.1
            
            Q_cg2 = df.VectorFunctionSpace(model.mesh,"CG",1)
            U_s = self.U_s = df.Function(Q_cg2)
            count = 0

            with df.CheckpointFile(f'{results_dir}/ismipf-L-{l}.h5', 'w') as afile:
                afile.save_mesh(mesh)  # optional
                while t<t_end:
                    print(t)
                    model.step(t,dt)
                    print(model.H0.vector()[:].min())
                    t += dt
                    
                    count += 1

                    S_out.vector()[:] = model.H0.vector()[:] + model.B.vector()[:] - model.S_lin.vector()[:]
                    U_s.interpolate(model.Ubar0 - 0.25*model.Udef0)
                    afile.save_function(S_out, idx=count, name='S')
                    afile.save_function(U_s, idx=count, name='U_s')
                    S_file.write(S_out,time=t)
                    Us_file.write(U_s,time=t)

if __name__=='__main__':
    ismipf = ISMIP_HOM_F('./results/')
