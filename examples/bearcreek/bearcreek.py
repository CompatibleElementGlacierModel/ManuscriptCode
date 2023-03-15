import sys
sys.path.append('../..')
import firedrake as df
from firedrake.petsc import PETSc
import numpy as np
from speceis_dg.hybrid import CoupledModel

class BearCreek:
    def __init__(self,results_dir,data_dir,dam_break=False,init_dir=None):
        import pickle
        if dam_break:
            with df.CheckpointFile(f"{init_dir}/functions.h5", 'r') as afile:
                mesh = afile.load_mesh('mesh')
        else:    
            mesh = df.Mesh(f'{data_dir}/mesh.msh',name='mesh')
        model = self.model = CoupledModel(mesh,velocity_function_space='MTW',periodic=False,sia=False,ssa=False,sliding_law='Budd',vel_scale=100,thk_scale=1e3,len_scale=108000.,beta_scale=1e3,time_scale=1,g=9.81,rho_i=917.,rho_w=1000.0,n=3.0,A=1e-16,eps_reg=1e-6,thklim=1e-3,theta=1.0,alpha=1000.0,p=4,c=0.7,z_sea=1.5,membrane_degree=2,shear_degree=3,flux_type='lax-friedrichs',calve=True)


        interpolant = pickle.load(open(f'{data_dir}/interpolant.pkl','rb'))
         
        E_dg2 = df.VectorElement(model.E_thk)
        v_dg = df.VectorFunctionSpace(mesh,model.E_thk)
        X = df.interpolate(mesh.coordinates,v_dg)
        model.B.dat.data[:] = interpolant(X.dat.data_ro[:,0],X.dat.data_ro[:,1],grid=False)/1000.0
        model.H0.interpolate(df.Constant(0.001))

        print(model.adot.vector().max(),model.adot.vector()[:].min())
        model.beta2.interpolate(df.Constant(100.0))

        S_file = df.File(f'{results_dir}/S.pvd')
        B_file = df.File(f'{results_dir}/B.pvd')
        Us_file = df.File(f'{results_dir}/U_s.pvd')
        H_file = df.File(f'{results_dir}/H.pvd')
        N_file = df.File(f'{results_dir}/N.pvd')
        adot_file = df.File(f'{results_dir}/adot.pvd')
        S_out = df.Function(model.Q_thk,name='S')
        N_out = df.Function(model.Q_thk,name='N')

        Q_cg2 = df.VectorFunctionSpace(mesh,"CG",3)
        U_s = df.Function(Q_cg2,name='U_s')

        t = 0.0
        t_end = 2000
        dt = 10.0
        H_true = df.Function(model.Q_thk)
        z_ela = 2.1
        if dam_break:
            lapse_rate=0.0
            time_step_factor = 1.01
            max_step = 20
        else:
            lapse_rate = 5/1000
            time_step_factor = 1.05
            max_step = 10.0


        if dam_break:
            with df.CheckpointFile(f"{init_dir}/functions.h5", 'r') as afile:
                H_in = afile.load_function(mesh, "H0", idx=399)
                model.H0.assign(H_in)

        S_out.interpolate(model.S)
        N_out.interpolate(model.N)
        U_s.interpolate(model.Ubar0 - 1./4*model.Udef0)
        model.adot.dat.data[:] = ((model.B.dat.data[:] + model.H0.dat.data[:]) - z_ela)*lapse_rate
        S_file.write(S_out,time=t)
        H_file.write(model.H0,time=t)
        B_file.write(model.B,time=t)
        Us_file.write(U_s,time=t)
        adot_file.write(model.adot,time=t)

        with df.CheckpointFile(f"{results_dir}/functions.h5", 'w') as afile:
            afile.save_mesh(mesh)  # optional

            i = 0
            while t<t_end:
                dt = min(dt*time_step_factor,max_step)

                model.adot.dat.data[:] = ((model.B.dat.data[:] + model.H0.dat.data[:]) - z_ela)*lapse_rate
                converged = model.step(t,dt,picard_tol=2e-3,momentum=0.5,max_iter=20,convergence_norm='l2')
                if not converged:
                    dt*=0.5
                    continue
                t += dt
                PETSc.Sys.Print(t,dt,df.assemble(model.H0*df.dx))
                S_out.interpolate(model.S)
                N_out.interpolate(model.N)
                U_s.interpolate(model.Ubar0 - 1./4*model.Udef0)

                afile.save_function(model.H0, idx=i)
                afile.save_function(S_out,idx=i)
                afile.save_function(N_out,idx=i)
                afile.save_function(U_s,idx=i)

                S_file.write(S_out,time=t)
                H_file.write(model.H0,time=t)
                B_file.write(model.B,time=t)
                Us_file.write(U_s,time=t)
                N_file.write(N_out,time=t)
                adot_file.write(model.adot,time=t)
                i += 1

if __name__=='__main__':
    bc = BearCreek('./results/','./data/',dam_break=False,init_dir='./results/')
