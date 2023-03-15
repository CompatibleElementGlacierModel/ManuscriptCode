import sys
sys.path.append('../..')
import firedrake as df
import numpy as np
from speceis_dg.hybrid import CoupledModel

class JaroschSchoofAnslow:
    def __init__(self,results_dir):
        for M in [50]:
            mesh = df.PeriodicRectangleMesh(M,3,1,3/M,direction='y',name='mesh',diagonal='crossed')
            model = CoupledModel(mesh,velocity_function_space='RT',periodic=True,sia=True,ssa=False,vel_scale=100,thk_scale=1e3,len_scale=2.5e4,beta_scale=1e3,time_scale=1,g=9.81,rho_i=917.,rho_w=1000.0,n=3.0,A=1e-16,eps_reg=1e-6,thklim=1e-6,theta=1.0,alpha=1000.0,p=4,membrane_degree=2,shear_degree=3,flux_type='lax-friedrichs')

            X = df.SpatialCoordinate(mesh)
            x,y = X


            a = df.Constant(1000)
            b_0 = df.Constant(0.5)
            x_m = df.Constant(0.8)
            x_s = df.Constant(7000/25000)
            m_0 = df.Constant(2./1000)
            n = df.Constant(3.0)

            m_dot = n*m_0/(x_m**(2*n - 1))*x**(n-1)*abs(x_m - x)**(n-1)*(x_m - 2*x)

            model.adot.interpolate(m_dot)

            B_exp = b_0*(df.Constant(1)/(df.Constant(1) + df.exp(-a*(x_s-x))))       

            model.H0.interpolate(df.Constant(1e-3))
            model.B.interpolate(B_exp)

            model.beta2.interpolate(df.Constant(10000.0))

            S_file = df.File(f'{results_dir}/S.pvd')
            B_file = df.File(f'{results_dir}/B.pvd')
            Us_file = df.File(f'{results_dir}/U_s.pvd')
            H_file = df.File(f'{results_dir}/H.pvd')
            H_true_file = df.File(f'{results_dir}/H_true.pvd')
            S_out = df.Function(model.Q_thk)
            U_s = df.Function(model.Q_vel)

            t = 0.0
            t_end = 100000
            dt = (t_end - t)/10000
            H_true = df.Function(model.Q_thk)
            
            U_s.vector()[:] = model.Ubar0.vector()[:] - 1./4*model.Udef0.vector()[:]
            S_file.write(S_out,time=t)
            H_file.write(model.H0,time=t)
            B_file.write(model.B,time=t)
            Us_file.write(U_s,time=t)

            i = 0
            with df.CheckpointFile(f"{results_dir}/functions_{M}_ho.h5", 'w') as afile:
                afile.save_mesh(mesh)
                while t<t_end:
                    model.step(t,dt,picard_tol=1e-3,max_iter=20,momentum=0.5)
                    afile.save_function(model.H0, idx=i)
                    t += dt
                    print (t,model.H0.vector().max(),df.assemble(model.H0*df.dx)*M/3*25000*1000/4.546e6)
                    S_out.vector()[:] = model.H0.vector()[:] + model.B.vector()[:]
                    U_s.vector()[:] = model.Ubar0.vector()[:] - 1./4*model.Udef0.vector()[:]
                    S_file.write(S_out,time=t)
                    H_file.write(model.H0,time=t)
                    B_file.write(model.B,time=t)
                    Us_file.write(U_s,time=t)
                    i+=1

if __name__=='__main__':
    jsa = JaroschSchoofAnslow('./results/')
