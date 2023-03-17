import os
import sys
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.append('../..')
import firedrake as df
import pickle
from firedrake.petsc import PETSc
from speceis_dg.hybrid import CoupledModel

class BearCreek:

    def interpolate_bed_from_pickle(self,interpolant_path):
        interpolant = pickle.load(open(interpolant_path,'rb'))

        v_dg = df.VectorFunctionSpace(self.model.mesh,self.model.E_thk)
        X = df.interpolate(self.model.mesh.coordinates,v_dg)
        self.model.B.dat.data[:] = (interpolant(
            X.dat.data_ro[:,0],X.dat.data_ro[:,1],grid=False)/1000.0)
        self.model.H0.interpolate(df.Constant(0.001))

    def __init__(self,results_dir,data_dir,conservation_test=False,init_dir=None):
        if conservation_test:
            with df.CheckpointFile(f"{init_dir}/functions.h5", 'r') as afile:
                mesh = afile.load_mesh('mesh')
        else:    
            mesh = df.Mesh(f'{data_dir}/mesh.msh',name='mesh')

        config = {'solver_type': 'gmres',
                  'sliding_law': 'Budd',
                  'vel_scale': 100.,
                  'thk_scale': 1000.,
                  'len_scale': 108000.,
                  'beta_scale': 1000.,
                  'theta': 1.0,
                  'thklim': 1e-3,
                  'alpha': 1000.0,
                  'z_sea': 1.5,
                  'calve': True}
          
        model = self.model = CoupledModel(mesh,**config)
        self.interpolate_bed_from_pickle(f'{data_dir}/interpolant.pkl')

        if conservation_test:
            with df.CheckpointFile(f"{init_dir}/functions.h5", 'r') as afile:
                H_in = afile.load_function(mesh, "H0", idx=399)
                model.H0.assign(H_in)
         
        model.beta2.interpolate(df.Constant(100.0))

        z_ela = 2.1

        if conservation_test:
            lapse_rate=0.0
            time_step_factor = 1.01
        else:
            lapse_rate = 5/1000
            time_step_factor = 1.05
        
        model.adot.dat.data[:] = (((model.B.dat.data[:] + model.H0.dat.data[:]) 
                                  - z_ela)*lapse_rate)

        S_file = df.File(f'{results_dir}/S.pvd')
        B_file = df.File(f'{results_dir}/B.pvd')
        Us_file = df.File(f'{results_dir}/U_s.pvd')
        H_file = df.File(f'{results_dir}/H.pvd')
        N_file = df.File(f'{results_dir}/N.pvd')
        adot_file = df.File(f'{results_dir}/adot.pvd')

        Q_cg2 = df.VectorFunctionSpace(mesh,"CG",3)
        S_out = df.Function(model.Q_thk,name='S')
        N_out = df.Function(model.Q_thk,name='N')
        U_s = df.Function(Q_cg2,name='U_s')

        S_out.interpolate(model.S)
        N_out.interpolate(model.N)
        U_s.interpolate(model.Ubar0 - 1./4*model.Udef0)

        S_file.write(S_out,time=0.)
        H_file.write(model.H0,time=0.)
        B_file.write(model.B,time=0.)
        Us_file.write(U_s,time=0.)
        adot_file.write(model.adot,time=0.)

        t = 0.0
        t_end = 2000
        dt = 10.0
        max_step = 10.0

        with df.CheckpointFile(f"{results_dir}/functions.h5", 'w') as afile:
            
            afile.save_mesh(mesh)

            i = 0
            while t<t_end:
                dt = min(dt*time_step_factor,max_step)

                model.adot.dat.data[:] = (((model.B.dat.data[:] + model.H0.dat.data[:])
                                          - z_ela)*lapse_rate)

                converged = model.step(t,
                                       dt,
                                       picard_tol=2e-3,
                                       momentum=0.5,
                                       max_iter=20,
                                       convergence_norm='l2')

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
    bc = BearCreek('./results/','./data/',conservation_test=False,init_dir='./results/')
