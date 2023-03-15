import sys
sys.path.append('../..')
import firedrake as df
import numpy as np
from speceis_dg.hybrid import CoupledModel
import pickle

class MMSForcing:
    def __init__(self,model):
        X = df.SpatialCoordinate(model.mesh)
        x,y = X

        delta = model.delta
        gamma = model.gamma
        omega = model.omega
        zeta = model.zeta
        eps_reg = model.eps_reg
        n = model.n
        
        T = df.Constant(25.0)
        alpha_u = df.Constant(0.1)
        ft = df.exp(-model.t/T)
        C_H = df.Constant(2.0)
        self.B_true = B_true = df.cos(2*np.pi*x)*df.cos(2*np.pi*y)
        self.S_true = S_true = ft*df.cos(2*np.pi*x)*df.cos(2*np.pi*y) + C_H
        self.H_true = H_true = S_true - B_true
        self.u_true = u_true = ft*alpha_u*df.sin(2*np.pi*x)*df.cos(2*np.pi*y)
        self.v_true = v_true = ft*alpha_u*df.cos(2*np.pi*x)*df.sin(2*np.pi*y)
        
        dHdt = df.diff(S_true,model.t)
        dHdx,dHdy = df.diff(H_true,X)
        dSdx,dSdy = df.diff(S_true,X)
        dudx,dudy = df.diff(u_true,X)
        dvdx,dvdy = df.diff(v_true,X)
        
        qx = H_true*u_true
        qy = H_true*v_true
        dqxdx,dqxdy = df.diff(qx,X)
        dqydx,dqydy = df.diff(qy,X)
        divq = dqxdx + dqydy
        
        eta_true = 0.5*(delta**2*(dudx**2 + dvdy**2 + dudx*dvdy + 0.25*(dudy + dvdx)**2) + eps_reg)**((1-n)/(2*n))
        tau_xx = 2*df.diff(delta**2*H_true*eta_true*(2*dudx + dvdy),X)[0]
        tau_yx,tau_xy = 2*df.diff(delta**2*H_true*eta_true*(0.5*dudy + 0.5*dvdx),X)
        tau_yy = 2*df.diff(delta**2*H_true*eta_true*(dudx + 2*dvdy),X)[1]
        tau_bx = -gamma*model.beta2*u_true
        tau_by = -gamma*model.beta2*v_true
        tau_dx = omega*H_true*dSdx
        tau_dy = omega*H_true*dSdy

        F_u = tau_xx + tau_xy + tau_bx - tau_dx
        F_v = tau_yx + tau_yy + tau_by - tau_dy
        self.F_U = df.as_vector([F_u,F_v])
        self.F_H = dHdt + zeta*divq

    def __call__(self,model):
        model.F_U.project(self.F_U)
        model.F_H.project(self.F_H)

class MMSTime:
    def __init__(self,results_dir,dry_run=True):
        self.H_inorms = {}
        self.H_2norms = {}
        self.U_2norms = {}

        t_end = 25

        for N in [4,8,16,32,64]:
            self.H_inorms[N] = {}
            self.H_2norms[N] = {}
            self.U_2norms[N] = {}
            mesh = df.PeriodicUnitSquareMesh(N,N,diagonal='crossed')
            with df.CheckpointFile(f'{results_dir}/mms_{t_end}_{N}.h5', 'w') as outfile:
                outfile.save_mesh(mesh)
                #for M in [16]:
                for M in [1,2,4,8,16,32,64]:
                    self.H_inorms[N][M] = []
                    self.H_2norms[N][M] = []
                    self.U_2norms[N][M] = []

                    model = self.model = CoupledModel(mesh,velocity_function_space='MTW',periodic=False,sia=False,ssa=True,vel_scale=100,thk_scale=1e2,len_scale=1e4,beta_scale=1e3,time_scale=1,g=9.81,rho_i=917.,rho_w=1000.0,n=3.0,A=1e-16,eps_reg=1e-5,thklim=1e-3,theta=1.0,alpha=0.0,p=4,membrane_degree=1,shear_degree=1,mms=True)

                    mms_forcing = MMSForcing(model)

                    model.H0.interpolate(mms_forcing.H_true)
                    model.B.interpolate(mms_forcing.B_true)
                    model.beta2.interpolate(df.Constant(0.1))

                    S_file = df.File(f'{results_dir}/S.pvd')
                    B_file = df.File(f'{results_dir}/B.pvd')
                    Us_file = df.File(f'{results_dir}/U_s.pvd')
                    H_file = df.File(f'{results_dir}/H.pvd')
                    S_out = df.Function(model.Q_thk)
                    U_s = df.Function(model.Q_vel)

                    t = 0
                    dt = t_end/M
                    #dt = 1./M#(t_end-t)/M
                    
                    U_s.vector()[:] = model.Ubar0.vector()[:] - 1./4.*model.Udef0.vector()[:]
                    S_file.write(S_out,time=t)
                    H_file.write(model.H0,time=t)
                    B_file.write(model.B,time=t)
                    Us_file.write(U_s,time=t)
                    count = 0 
                    U_true = df.as_vector([mms_forcing.u_true,mms_forcing.v_true])
                    while count<M:
                        print(t)
                        model.step(t,dt,picard_tol=1e-6,forcing=mms_forcing)
                        t += dt
                        S_out.vector()[:] = model.H0.vector()[:] + model.B.vector()[:] - model.S_lin.vector()[:]
                        U_s.vector()[:] = model.Ubar0.vector()[:] - 1./4*model.Udef0.vector()[:]
                        S_file.write(S_out,time=t)
                        H_file.write(model.H0,time=t)
                        B_file.write(model.B,time=t)
                        Us_file.write(U_s,time=t)
                        count += 1

                        H_true_dg0 = df.interpolate(mms_forcing.H_true,model.Q_thk)
                        H_true_cg1 = df.interpolate(mms_forcing.H_true,model.Q_cg1)

                        l_inf_H = abs(H_true_dg0.vector()[:] - model.H0.vector()[:]).max()

                        l_2_H = df.errornorm(mms_forcing.H_true,df.project(model.H0,model.Q_cg1),norm_type='L2')
                        l_2_U = df.errornorm(U_true,model.Ubar0,norm_type='L2')

                        H_mag = df.norm(H_true_cg1)
                        U_mag = df.norm(U_true)

                        self.H_inorms[N][M].append([t,l_inf_H,(H_true_dg0.vector()[:]).max()])
                        self.H_2norms[N][M].append([t,l_2_H,H_mag])
                        self.U_2norms[N][M].append([t,l_2_U,U_mag])

                        print(t,l_inf_H,l_2_H,l_2_U,H_mag,U_mag)
                    
        if dry_run==False:
            pickle.dump([self.H_inorms,self.H_2norms,self.U_2norms],open(f'{results_dir}/norms_time_dependent.p','wb'))

if __name__=='__main__':
    mms_time = MMSTime('./results/time/',dry_run=False)
