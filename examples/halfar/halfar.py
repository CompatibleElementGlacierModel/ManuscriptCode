import sys
sys.path.append('../..')
import firedrake as df
from firedrake.petsc import PETSc
import numpy as np
from speceis_dg.hybrid import CoupledModel

class Halfar:
    def __init__(self,results_dir,dry_run=True,fspaces=['RT'],Ns=[64],log_profile=False):
        self.H_2norms = {}
        self.H_profile = {}

        for fspace in fspaces:
            self.H_profile[fspace] = {}
            self.H_2norms[fspace] = {}
            for N in Ns:

                mesh = df.UnitSquareMesh(N,N,diagonal='crossed')
                model = self.model = CoupledModel(mesh,velocity_function_space=fspace,periodic=True,sia=True,ssa=False,vel_scale=100,thk_scale=1e3,len_scale=6e5,beta_scale=1e3,time_scale=1,g=9.81,rho_i=917.,rho_w=1000.0,n=3.0,A=1e-16,eps_reg=1e-8,thklim=1e-6,theta=1.0,alpha=1000.0,p=4,membrane_degree=2,shear_degree=3,flux_type='lax-friedrichs')

                X = df.SpatialCoordinate(mesh)
                x,y = X

                thklim = 1e-3
                len_scale = 6e5
                thk_scale = 1000
                rho = 917.
                g = 9.81
                lamda_halfar = 0.0
                n_halfar = 3.0
                A_halfar = 1e-16
                alpha_halfar = (2 - (n_halfar+1)*lamda_halfar)/(5*n_halfar + 3)
                beta_halfar = (1 + (2*n_halfar + 1)*lamda_halfar)/(5*n_halfar + 3)

                f = 0.0
                Rhat = 5./6*len_scale
                Hhat = 3.0*thk_scale
                gamma_halfar = 2*A_halfar*(rho*g)**n_halfar/(n_halfar+2)
                t0 = beta_halfar/gamma_halfar*((2*n_halfar + 1)/((1-f)*(n_halfar+1)))**n_halfar*Rhat**(n_halfar+1)/(Hhat**(2*n_halfar + 1))
                R = (x**2 + y**2)**0.5

                t_init = df.Constant(t0)
                dome_height = Hhat/thk_scale*(t_init/t0)**(-alpha_halfar)
                H_halfar = df.max_value(dome_height*(1 - ((t_init/t0)**(-beta_halfar)*R*len_scale/Rhat)**((n_halfar+1)/n_halfar))**(n_halfar/(2*n_halfar+1)),thklim)

                model.H0.interpolate(H_halfar)
                model.B.interpolate(df.Constant(0.0))

                model.beta2.interpolate(df.Constant(1000.0))

                t = t0
                t_end = 10*t0
                dt = (t_end - t)/100

                H_true = df.Function(model.Q_thk)
                H_true_smooth = df.Function(model.Q_cg1)
                H_smooth = df.Function(model.Q_cg1)
                
                while t<t_end:
                    model.step(t,dt,picard_tol=1e-4,max_iter=100)
                    t += dt
                    t_init.assign(t)

                    H_true.interpolate(H_halfar)
                    H_true_smooth.interpolate(H_halfar)
                    H_smooth.project(model.H0)

                    H_l_2 = df.errornorm(H_halfar,H_smooth)
                    PETSc.Sys.Print(t,model.H0([0,0])-dome_height(0),H_l_2,df.assemble(model.H0*df.dx))

                x = np.linspace(0,1,1001)
                X = np.c_[x,x]/np.sqrt(2)
                H_profile_dg = model.H0(X)
                H_profile_cg = H_smooth(X)
                H_profile_true_dg = H_true(X)
                H_profile_true_cg = H_true_smooth(X)
                self.H_profile[fspace][N] = np.c_[x,H_profile_dg,H_profile_cg,H_profile_true_dg,H_profile_true_cg]
                self.H_2norms[fspace][N] = H_l_2

        if dry_run==False:
            pickle.dump([self.H_2norms],open(f'{results_dir}/norms_static.p','wb'))
            pickle.dump([self.H_profile],open(f'{results_dir}/H_profiles.p','wb'))

if __name__=='__main__':
    halfar = Halfar('./results/',dry_run=False,fspaces=['RT','MTW'],Ns=[4,8,16,32,64])
