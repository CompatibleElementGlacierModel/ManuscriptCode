import sys
sys.path.append('../..')
import firedrake as df
from firedrake.petsc import PETSc
import numpy as np
from speceis_dg.hybrid import CoupledModel

class EISMINT:
    def __init__(self,results_dir):
        mesh = df.UnitSquareMesh(25,25,diagonal='crossed')
        model = CoupledModel(mesh,velocity_function_space='RT',periodic=True,sia=True,ssa=False,vel_scale=100,thk_scale=1e3,len_scale=7.5e5,beta_scale=1e3,time_scale=1,g=9.81,rho_i=917.,rho_w=1000.0,n=3.0,A=1e-16,eps_reg=1e-6,thklim=1e-3,theta=1.0,alpha=1000.0,p=4,membrane_degree=2,shear_degree=3)

        X = df.SpatialCoordinate(mesh)
        x,y = X

        thklim = 1e-3
        len_scale = 7.5e5
        thk_scale = 1000


        R = (x**2 + y**2)**0.5
        R_el = 4.5e5/len_scale
        s = 1e-2*len_scale/thk_scale
        adot_exp = df.min_value(0.5,s*(R_el - R))/thk_scale

        model.H0.interpolate(df.Constant(1e-3))

        model.adot.interpolate(adot_exp)
        print(model.adot.vector().max(),model.adot.vector()[:].min())
        model.beta2.interpolate(df.Constant(1000.0))

        S_file = df.File(f'{results_dir}/S.pvd')
        B_file = df.File(f'{results_dir}/B.pvd')
        Us_file = df.File(f'{results_dir}/U_s.pvd')
        H_file = df.File(f'{results_dir}/H.pvd')
        H_true_file = df.File(f'{results_dir}/H_true.pvd')
        S_out = df.Function(model.Q_thk)
        U_s = df.Function(model.Q_vel)

        t = 0.0
        t_end = 20000
        dt = (t_end - t)/20
        H_true = df.Function(model.Q_thk)
        
        U_s.vector()[:] = model.Ubar0.vector()[:] - 1./4*model.Udef0.vector()[:]
        S_file.write(S_out,time=t)
        H_file.write(model.H0,time=t)
        B_file.write(model.B,time=t)
        Us_file.write(U_s,time=t)

        while t<t_end:
            model.step(t,dt,picard_tol=1e-4,max_iter=1000,momentum=0.0)
            t += dt
            print (t,model.H0.vector().max())
            S_out.vector()[:] = model.H0.vector()[:] + model.B.vector()[:] - model.S_lin.vector()[:]
            U_s.vector()[:] = model.Ubar0.vector()[:] - 1./4*model.Udef0.vector()[:]
            S_file.write(S_out,time=t)
            H_file.write(model.H0,time=t)
            B_file.write(model.B,time=t)
            Us_file.write(U_s,time=t)

if __name__=='__main__':
    eis = EISMINT('./results/')
