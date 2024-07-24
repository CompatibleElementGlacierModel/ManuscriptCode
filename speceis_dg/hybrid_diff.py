"""
This module contains base physics and helper classes for solving hybrid, SSA, and SIA
approximations to isothermal Stokes' flow for ice sheets using either 
Raviart-Thomas or Mardal-Tai-Winther finite elements for velocities, 
and DG0 elements for ice thickness.  This code is experimental and is still
lacking a few key elements, including full support for boundary conditions.
"""


import os
os.environ['OMP_NUM_THREADS'] = '1'


import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import firedrake as df
    from firedrake.petsc import PETSc
import numpy as np
import time
import torch



def full_quad(order):
    points,weights = np.polynomial.legendre.leggauss(order)
    points = (points+1)/2.
    weights /= 2.
    return points,weights

class VerticalBasis(object):
    def __init__(self,u,H,S_grad,B_grad,p=4,ssa=False):
        self.u = u
        if ssa:
            self.coef = [lambda s: 1.0]
            self.dcoef = [lambda s: 0.0]
        else:
            self.coef = [lambda s:1.0, lambda s:1./p*((p+1)*s**p - 1)]
            self.dcoef = [lambda s:0, lambda s:(p+1)*s**(p-1)]
        
        self.H = H
        self.S_grad = S_grad
        self.B_grad = B_grad

    def __call__(self,s):
        return sum([u*c(s) for u,c in zip(self.u,self.coef)])

    def ds(self,s):
        return sum([u*c(s) for u,c in zip(self.u,self.dcoef)])

    def dz(self,s):
        return self.ds(s)*self.dsdz(s)

    def dx_(self,s,x):
        return sum([u.dx(x)*c(s) for u,c in zip(self.u,self.coef)])

    def dx(self,s,x):
        return self.dx_(s,x) + self.ds(s)*self.dsdx(s,x)

    def dsdx(self,s,x):
        return 1./self.H*(self.S_grad[x] - s*(self.S_grad[x] - self.B_grad[x]))

    def dsdz(self,x):
        return -1./self.H

class VerticalIntegrator(object):
    def __init__(self,points,weights):
        self.points = points
        self.weights = weights

    def integral_term(self,f,s,w):
        return w*f(s)

    def intz(self,f):
        return sum([self.integral_term(f,s,w) 
                    for s,w in zip(self.points,self.weights)])  

class CoupledModel:
    def __init__(
            self, mesh,
            velocity_function_space='MTW',
            solver_type='direct',
            sia=False, ssa=False,
            sliding_law='linear',
            vel_scale=100, thk_scale=1e3,
            len_scale=1e4, beta_scale=1e3,
            time_scale=1, pressure_scale=1,
            g=9.81, rho_i=917., rho_w=1000.0,
            n=3.0, A=1e-16, eps_reg=1e-6,
            thklim=1e-3, theta=1.0, alpha=0,
            p=4, c=0.7, z_sea=-1000,
            membrane_degree=2, shear_degree=3,
            flux_type='lax-friedrichs', boundary_markers=None,
            calve=None):
            
        self.mesh = mesh
        nhat = df.FacetNormal(mesh)
        self.solver_type=solver_type

        E_cg1 = self.E_cg1 = df.FiniteElement('CG',mesh.ufl_cell(),1)
        E_thk = self.E_thk = df.FiniteElement('DG',mesh.ufl_cell(),0)
        if velocity_function_space=='MTW':
            E_bar = self.E_bar = df.FiniteElement('MTW',mesh.ufl_cell(),3)
            E_def = self.E_def = df.FiniteElement('RT',mesh.ufl_cell(),1)
            E_grd = self.E_grd = df.FiniteElement('RT',mesh.ufl_cell(),1)
        elif velocity_function_space=='RT' or velocity_function_space=='BDM':
            E_bar = self.E_bar = df.FiniteElement(velocity_function_space,
                                                  mesh.ufl_cell(),1)
            E_def = self.E_def = df.FiniteElement(velocity_function_space,
                                                  mesh.ufl_cell(),1)
            E_grd = self.E_grd = df.FiniteElement('RT',mesh.ufl_cell(),1)
        elif velocity_function_space=='CG' or velocity_function_space=='CR':
            E_0 = df.FiniteElement(velocity_function_space,mesh.ufl_cell(),1)
            E_bar = self.E_bar = df.VectorElement(E_0)
            E_def = self.E_def = df.VectorElement(E_0)
            E_grd = self.E_grd = df.VectorElement(E_0)
        elif velocity_function_space=='CGFB':
            E_0_cg = df.FiniteElement('CG',mesh.ufl_cell(),1)
            E_0_b = df.FiniteElement('FacetBubble',mesh.ufl_cell(),2)
            E_0 = E_0_cg + E_0_b
            E_bar = self.E_bar = df.VectorElement(E_0)
            E_def = self.E_def = df.VectorElement(E_0_cg)
            E_grd = self.E_grd = df.VectorElement(E_0_cg)
        else:
            print('Unsupported Element')


        E = self.E = df.MixedElement(E_bar,E_def,E_thk)
        
        Q_cg1 = self.Q_cg1 = df.FunctionSpace(mesh,E_cg1)
        Q_bar = self.Q_bar = df.FunctionSpace(mesh,E_bar)
        Q_def = self.Q_def = df.FunctionSpace(mesh,E_def)
        Q_thk = self.Q_thk = df.FunctionSpace(mesh,E_thk)
        Q_grd = self.Q_grd = df.FunctionSpace(mesh,E_grd)
        V = self.V = df.FunctionSpace(mesh,E)

        

        self.one = df.Function(Q_thk)
        self.one.assign(1.0)
        self.area = df.assemble(self.one*df.dx)

        theta = self.theta = df.Constant(theta)
        self.t = df.Constant(0.0)
        dt = self.dt = df.Constant(1.0)

        g = self.g = df.Constant(g)
        rho_i = self.rho_i = df.Constant(rho_i)
        rho_w = self.rho_w = df.Constant(rho_w)
        n = self.n = df.Constant(n)
        A = self.A = df.Constant(A)
        eps_reg = self.eps_reg = df.Constant(eps_reg)
        thklim = self.thklim = df.Constant(thklim)
        z_sea = self.z_sea = df.Constant(z_sea)

        vel_scale = self.vel_scale = df.Constant(vel_scale)
        thk_scale = self.thk_scale = df.Constant(thk_scale)
        len_scale = self.len_scale = df.Constant(len_scale)
        beta_scale = self.beta_scale = df.Constant(beta_scale)
        time_scale = self.time_scale = df.Constant(time_scale) 

        eta_star = self.eta_star = df.Constant(A**(-1./n)
                                               * (vel_scale/thk_scale)**((1-n)/n))

        delta = self.delta = df.Constant(thk_scale/len_scale)

        gamma = self.gamma = df.Constant(beta_scale*thk_scale/eta_star)

        omega = self.omega = df.Constant(rho_i*g*thk_scale**3
                                         / (eta_star*len_scale*vel_scale))

        zeta = self.zeta = df.Constant(time_scale*vel_scale/len_scale)

        if sia:
            delta.assign(0.0)

        W = self.W = df.Function(V,name='W')
        W_i = self.W_i = df.Function(V)
        Psi = self.Psi = df.TestFunction(V)
        dW = self.dW = df.TrialFunction(V)

        Ubar,Udef,H = df.split(W)
        ubar,vbar = Ubar
        udef,vdef = Udef

        Ubar_i,Udef_i,H_i = df.split(W_i)
        ubar_i,vbar_i = Ubar_i
        udef_i,vdef_i = Udef_i

        Phibar,Phidef,xsi = df.split(Psi)
        phibar_x,phibar_y = Phibar
        phidef_x,phidef_y = Phidef

        S_grad = self.S_grad = df.Function(Q_grd)
        B_grad = self.B_grad = df.Function(Q_grd)
        Chi = df.TestFunction(Q_grd)
        dS = df.TrialFunction(Q_grd)

        self.Ubar0 = df.Function(Q_bar)
        self.Udef0 = df.Function(Q_def)
        
        H0 = self.H0 = df.Function(Q_thk,name='H0')
        B = self.B = df.Function(Q_thk,name='B')

        S_lin = self.S_lin = df.Function(Q_thk)  
        B_lin = self.B_lin = df.Function(Q_thk)  
        S_grad_lin = self.S_grad_lin = df.Constant([0.0,0.0])
        B_grad_lin = self.B_grad_lin = df.Constant([0.0,0.0])

        adot = self.adot = df.Function(Q_thk) 
        beta2 = self.beta2 = df.Function(Q_cg1)
        alpha = self.alpha = df.Constant(alpha)

        Hmid = theta*H + (1-theta)*H0
        Hmid_i = theta*H_i + (1-theta)*H0

        Bhat = self.Bhat =  df.max_value(B, z_sea - rho_i/rho_w*Hmid_i)

        S = self.S = Bhat + H
        S0 = Bhat + H0
        
        Smid = theta*S + (1-theta)*S0

        self.F_U = F_U = df.Function(Q_bar)
        self.F_H = F_H = df.Function(Q_thk)

        u = VerticalBasis([ubar,udef],Hmid_i,S_grad,B_grad,p=p,ssa=ssa)
        v = VerticalBasis([vbar,vdef],Hmid_i,S_grad,B_grad,p=p,ssa=ssa)
        u_i = VerticalBasis([ubar_i,udef_i],Hmid_i,S_grad,B_grad,p=p,ssa=ssa)
        v_i = VerticalBasis([vbar_i,vdef_i],Hmid_i,S_grad,B_grad,p=p,ssa=ssa)
        phi_x = VerticalBasis([phibar_x,phidef_x],Hmid_i,S_grad,B_grad,p=p,ssa=ssa)
        phi_y = VerticalBasis([phibar_y,phidef_y],Hmid_i,S_grad,B_grad,p=p,ssa=ssa)

        U_b = df.as_vector([u(1),v(1)])
        Phi_b = df.as_vector([phi_x(1),phi_y(1)])

        vi_x = VerticalIntegrator(*full_quad(membrane_degree))
        vi_z = VerticalIntegrator(*full_quad(shear_degree)) 

        def eps_i_II(s):
            return (delta**2*(u_i.dx(s,0))**2 
                        + delta**2*(v_i.dx(s,1))**2 
                        + delta**2*(u_i.dx(s,0))*(v_i.dx(s,1)) 
                        + delta**2*0.25*((u_i.dx(s,1)) + (v_i.dx(s,0)))**2 
                        +0.25*(u_i.dz(s))**2 + 0.25*(v_i.dz(s))**2 
                        + eps_reg)

        def eta(s):
            return 0.5*eps_i_II(s)**((1-n)/(2*n))

        def phi_grad_membrane(s):
            return np.array([[delta*phi_x.dx(s,0), delta*phi_x.dx(s,1)],
                             [delta*phi_y.dx(s,0), delta*phi_y.dx(s,1)]])

        def phi_grad_shear(s):
            return np.array([[phi_x.dz(s)],
                             [phi_y.dz(s)]])

        def phi_outer_membrane(s):
            return np.array([[delta*phi_x(s)*nhat[0],delta*phi_x(s)*nhat[1]],
                             [delta*phi_y(s)*nhat[0],delta*phi_y(s)*nhat[1]]])

        def eps_membrane(s):
            return np.array([[2*delta*u.dx(s,0) + delta*v.dx(s,1), 
                              0.5*delta*u.dx(s,1) + 0.5*delta*v.dx(s,0)],
                             [0.5*delta*u.dx(s,1) + 0.5*delta*v.dx(s,0),
                              delta*u.dx(s,0) + 2*delta*v.dx(s,1)]])

        def eps_shear(s):
            return np.array([[0.5*u.dz(s)],
                            [0.5*v.dz(s)]])

        def membrane_form(s):
            return (2*eta(s)*(eps_membrane(s)
                    * phi_grad_membrane(s)).sum()*Hmid_i*df.dx(degree=9))

        def shear_form(s):
            return (2*eta(s)*(eps_shear(s)
                    * phi_grad_shear(s)).sum()*Hmid_i*df.dx(degree=9))

        def membrane_boundary_form_nopen(s):
            un = u(s)*nhat[0] + v(s)*nhat[1]
            if boundary_markers is not None:
                return alpha*(phi_x(s)*un*nhat[0] + phi_y(s)*un*nhat[1])*df.ds(boundary_markers[0])#(degree=4)
            else:
                return alpha*(phi_x(s)*un*nhat[0] + phi_y(s)*un*nhat[1])*df.ds#(degree=4)

        def membrane_boundary_form_nat(s):
            return 2*eta(s)*(phi_outer_membrane(s)*eps_membrane(s)).sum()*Hmid_i*df.ds#(degree=4)

        def membrane_boundary_form_pressure(s):
            return s*omega*Hmid_i*(phi_x(s)*nhat[0] + phi_y(s)*nhat[1])*df.ds#(degree=4)

        if ssa:
            membrane_stress = -(vi_x.intz(membrane_form) 
                                + vi_x.intz(membrane_boundary_form_nopen))
        else:
            membrane_stress = -(vi_x.intz(membrane_form) 
                                + vi_z.intz(shear_form) 
                                + vi_x.intz(membrane_boundary_form_nopen))

        if sliding_law == 'linear':
            basal_stress = -gamma*beta2*df.dot(U_b,Phi_b)*df.dx#degree=4)
        elif sliding_law == 'Budd':
            self.N = (df.min_value(c*Hmid_i + df.Constant(1e-2),
                          Hmid_i-rho_w/rho_i*(z_sea - Bhat)) 
                          + df.Constant(1e-4))
            basal_stress = -gamma*beta2*self.N**df.Constant(1./3.)*df.dot(U_b,Phi_b)*df.dx#(degree=4)

        driving_stress = (omega*Hmid*df.dot(S_grad_lin,Phibar)*df.dx#(degree=4)
                          - omega*df.div(Phibar*Hmid)*(Bhat - S_lin)*df.dx#(degree=4)
                          - omega*df.div(Phibar*Hmid_i)*Hmid*df.dx#(degree=4) 
                          + omega*df.jump(Phibar*Hmid,nhat)*df.avg(Bhat - S_lin)*df.dS 
                          + omega*df.jump(Phibar*Hmid_i,nhat)*df.avg(Hmid)*df.dS 
                          + omega*df.dot(Phibar*Hmid,nhat)*(Bhat - S_lin)*df.ds 
                          + omega*df.dot(Phibar*Hmid_i,nhat)*(Hmid)*df.ds)

        #forcing_stress = df.dot(Phibar,F_U)*df.dx(degree

        R_stress = membrane_stress + basal_stress - driving_stress# - forcing_stress

        # This ensures non-singularity when SSA simplification is used.
        if ssa:
            R_stress += df.dot(Phidef,Udef)*df.dx

        if velocity_function_space=='CR':
            R_stress += df.Constant(1.0)/df.FacetArea(self.mesh)*df.jump(Ubar,nhat)*df.jump(Phibar,nhat)*df.dS

        H_avg = 0.5*(Hmid_i('+') + Hmid_i('-'))
        H_jump = Hmid('+')*nhat('+') + Hmid('-')*nhat('-')
        xsi_jump = xsi('+')*nhat('+') + xsi('-')*nhat('-')

        unorm_i = df.dot(Ubar_i,Ubar_i)**0.5


        # Lax-Friederichs flux
        if flux_type=='centered':
            uH = df.avg(Ubar)*H_avg

        elif flux_type=='lax-friedrichs':
            uH = df.avg(Ubar)*H_avg + df.Constant(0.5)*df.avg(unorm_i)*H_jump

        elif flux_type=='upwind':
            uH = df.avg(Ubar)*H_avg + 0.5*abs(df.dot(df.avg(Ubar_i),nhat('+')))*H_jump

        else:
            print('Invalid flux')

        if boundary_markers is not None:
            #R_transport = ((H - H0)/dt - adot)*xsi*df.dx + zeta*df.dot(uH,xsi_jump)*df.dS + zeta*df.dot(Ubar*H,nhat)*df.ds(boundary_markers[1])#(#- xsi*F_H*df.dx
            R_transport = ((H - H0)/dt - adot)*xsi*df.dx + zeta*df.dot(uH,xsi_jump)*df.dS + df.Constant(10000.0)*xsi*df.dot((H-thklim)*nhat,nhat)*df.ds(boundary_markers[1])#(#- xsi*F_H*df.dx
        else:
            R_transport = ((H - H0)/dt - adot)*xsi*df.dx + zeta*df.dot(uH,xsi_jump)*df.dS#(#- xsi*F_H*df.dx

        #if velocity_function_space=='CG':
        #    R_transport += df.Constant(1e-5)*df.dot(df.jump(Hmid,nhat),df.jump(xsi,nhat))*df.dS
        
        self.l = df.Constant(100./thk_scale)
        z = self.rho_w/self.rho_i*(self.z_sea - self.B) - Hmid_i
        self.floating = df.min_value(df.max_value(z/self.l,0),1)
        
        self.calving_factor = df.Constant(0.0)
        if calve=='a':
            #floating = df.conditional(
            #    df.lt(Hmid_i, self.rho_w/self.rho_i*(self.z_sea - self.B)),
            #    df.Constant(1.0),df.Constant(0.0))
            #self.floating = df.Constant(1.0)/(df.Constant(1.0) + df.exp(-(z/self.l - self.o)))
            R_transport += xsi*self.floating*self.calving_factor*Hmid*df.dx

        if calve=='b':
            indicator = self.floating('+')*self.floating('-')
            u_c = self.calving_factor*nhat
            R_transport += zeta*indicator*2*df.avg(df.dot(u_c,nhat)*Hmid*xsi)*df.dS

        R = self.R = R_stress + R_transport

        R_lin = self.R_lin = df.replace(R,{W:dW})

        R_S = (df.dot(Chi,dS)*df.dx 
              - df.dot(Chi,S_grad_lin)*df.dx 
              + df.div(Chi)*(Smid - S_lin)*df.dx 
              - df.dot(Chi,nhat)*(Smid - S_lin)*df.ds)

        R_B = (df.dot(Chi,dS)*df.dx 
              - df.dot(Chi,B_grad_lin)*df.dx 
              + df.div(Chi)*(Bhat - B_lin)*df.dx
              - df.dot(Chi,nhat)*(Bhat - B_lin)*df.ds)

        coupled_problem = df.LinearVariationalProblem(df.lhs(R_lin),df.rhs(R_lin),W)

        if solver_type=='direct':
            coupled_parameters = {"ksp_type": "preonly",
                                  "pmat_type":"aij",
                                  "pc_type": "lu",  
                                  "pc_factor_mat_solver_type": "mumps"} 
        else:
            coupled_parameters = {'ksp_type': 'gmres',
                                  'pc_type':'ilu',
                                  "ksp_rtol":1e-6,
                                  'ksp_initial_guess_nonzero': True}

        self.coupled_solver = df.LinearVariationalSolver(
            coupled_problem,
            solver_parameters=coupled_parameters)
        
        projection_parameters = {'ksp_type':'cg','mat_type':'matfree'}
        S_grad_problem = df.LinearVariationalProblem(df.lhs(R_S),df.rhs(R_S),S_grad)
        self.S_grad_solver = df.LinearVariationalSolver(
            S_grad_problem,
            solver_parameters=projection_parameters)

        B_grad_problem = df.LinearVariationalProblem(df.lhs(R_B),df.rhs(R_B),B_grad)
        self.B_grad_solver = df.LinearVariationalSolver(
            B_grad_problem,
            solver_parameters=projection_parameters)

        self.H_temp = df.Function(self.Q_thk)
        

        V_proj = self.Q_bar#df.df.VectorFunctionSpace(self.mesh,'CG',3)
        self.U_s = df.Function(V_proj)
        self.Phi_s = df.TestFunction(V_proj)
        self.dU_s = df.TrialFunction(V_proj)

        R_proj = df.dot(self.dU_s - (Ubar - 0.25*Udef),self.Phi_s)*df.dx
        proj_problem = df.LinearVariationalProblem(df.lhs(R_proj),df.rhs(R_proj),self.U_s)
        self.proj_solver = df.LinearVariationalSolver(
                proj_problem,
                solver_parameters=projection_parameters)

    def step(
            self,
            t,
            dt,
            picard_tol=1e-6,
            max_iter=50,
            momentum=0.0,
            error_on_nonconvergence=False,
            convergence_norm='linf',
            forcing=None,
            update=True,
            enforce_positivity=True):

        self.W.sub(0).assign(self.Ubar0)
        self.W.sub(1).assign(self.Udef0)
        self.W.sub(2).assign(self.H0)

        self.W_i.assign(self.W)
        self.dt.assign(dt)

        eps = 1.0
        i = 0
        
        self.t.assign(t + self.theta(0)*dt)
        if forcing:
            forcing(self)
        while eps>picard_tol and i<max_iter:
            t_ = time.time()
            self.S_grad_solver.solve()
            self.B_grad_solver.solve()
            self.coupled_solver.solve()
            if enforce_positivity:
                self.H_temp.interpolate(df.max_value(self.W.sub(2),self.thklim))
                self.W.sub(2).assign(self.H_temp)
            
            if convergence_norm=='linf':
                with self.W_i.dat.vec_ro as w_i:
                    with self.W.dat.vec_ro as w:
                        eps = abs(w_i - w).max()[1]
            else:
                eps = (np.sqrt(
                       df.assemble((self.W_i.sub(2) - self.W.sub(2))**2*df.dx))
                       / self.area)

            PETSc.Sys.Print(i,eps,time.time()-t_)


            self.W_i.assign((1-momentum)*self.W + momentum*self.W_i)
            i+=1

        if i==max_iter and eps>picard_tol:
            converged=False
        else:
            converged=True

        if error_on_nonconvergence and not converged:
            return converged
            
        self.t.assign(t+dt)

        if update:
            self.Ubar0.assign(self.W.sub(0))
            self.Udef0.assign(self.W.sub(1))
            self.H0.assign(self.W.sub(2))

        return converged

    def project_surface_velocity(self):
        self.proj_solver.solve()     

class CoupledModelAdjoint:
    def __init__(self,model):
        self.model = model

        Lambda = self.Lambda = df.Function(model.V)
        delta = self.delta = df.Function(model.V)

        w_H0 = df.TestFunction(model.Q_thk)
        w_B = df.TestFunction(model.Q_thk)
        w_beta = df.TestFunction(model.Q_cg1)
        w_adot = df.TestFunction(model.Q_thk)

        R_full = self.R_full = df.replace(self.model.R,{model.W_i:model.W, 
                                                        model.Psi:Lambda})

        R_adjoint = df.derivative(R_full,model.W,model.Psi)
        R_adjoint_linear = df.replace(R_adjoint,{Lambda:model.dW})
        self.A_adjoint = df.lhs(R_adjoint_linear)
        self.b_adjoint = df.rhs(R_adjoint_linear)

        G_H0 = self.G_H0 = df.derivative(R_full,model.H0,w_H0)
        G_B = self.G_B = df.derivative(R_full,model.B,w_B)
        G_beta = self.G_beta = df.derivative(R_full,model.beta2,w_beta)
        G_adot = self.G_adot = df.derivative(R_full,model.adot,w_adot)

        self.g_H0 = None
        self.g_B = None
        self.g_beta = None
        self.g_adot = None


        self.ksp = PETSc.KSP().create()
        if self.model.solver_type=='direct':
            self.ksp.setType('preonly')
            pc = self.ksp.getPC()
            pc.setType('lu')
            pc.setFactorSolverType('mumps')
        else:
            self.ksp.setType('gmres')
            pc = self.ksp.getPC()
            self.ksp.setInitialGuessNonzero(True)
            pc.setType('ilu')
            self.ksp.setTolerances(1e-6)

        self.A = None

    def backward(self,delta):
        t1 = time.time()
        if self.A:
            df.assemble(self.A_adjoint,tensor=self.A,mat_type='aij')
        else:
            self.A = df.assemble(self.A_adjoint,tensor=self.A,mat_type='aij')
            self.ksp.setOperators(self.A.M.handle)
        print(f'assemble: {time.time() - t1}')

        for d in self.delta.dat.data[:]:
            d*=-1

        with self.delta.dat.vec_ro as vec:
            with self.Lambda.dat.vec as sol:
                t1 = time.time()
                self.ksp.solve(vec,sol)
                print(f'adjoint solve: {time.time() - t1}')

        
        t1 = time.time()
        if self.g_H0:
            df.assemble(self.G_H0,tensor=self.g_H0)
            df.assemble(self.G_B,tensor=self.g_B)
            df.assemble(self.G_beta,tensor=self.g_beta)
            df.assemble(self.G_adot,tensor=self.g_adot)
        else:
            self.g_H0 = df.assemble(self.G_H0)
            self.g_B = df.assemble(self.G_B)
            self.g_beta = df.assemble(self.G_beta)
            self.g_adot = df.assemble(self.G_adot)
        print(f'gradient: {time.time() - t1}')


class FenicsModel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H0, B, beta2, adot, Ubar0, Udef0, model, adjoint, t, dt, kwargs):
        ctx.H0 = H0
        ctx.B = B
        ctx.beta2 = beta2
        ctx.adot = adot
        ctx.Ubar0 = Ubar0
        ctx.Udef0 = Udef0

        ctx.model = model
        ctx.adjoint = adjoint
        ctx.t = t
        ctx.dt = dt

        model.Ubar0.dat.data[:] = Ubar0
        model.Udef0.dat.data[:] = Udef0
        model.H0.dat.data[:] = H0
        model.B.dat.data[:] = B
        model.beta2.dat.data[:] = beta2
        model.adot.dat.data[:] = adot

        model.step(t,dt,**kwargs)

        ctx.Ubar = torch.tensor(model.W.dat.data[:][0])
        ctx.Udef = torch.tensor(model.W.dat.data[:][1])
        ctx.H = torch.tensor(model.W.dat.data[:][2])

        return ctx.Ubar.clone().detach(),ctx.Udef.clone().detach(),ctx.H.clone().detach()
    
    @staticmethod
    def backward(ctx,delta_Ubar,delta_Udef,delta_H):
        model = ctx.model
        adjoint = ctx.adjoint
        
        model.H0.dat.data[:] = ctx.H0
        model.B.dat.data[:] = ctx.B
        model.beta2.dat.data[:] = ctx.beta2
        model.adot.dat.data[:] = ctx.adot
        model.Ubar0.dat.data[:] = ctx.Ubar0
        model.Udef0.dat.data[:] = ctx.Udef0

        model.dt.assign(ctx.dt)
        model.t.assign(ctx.t)

        model.W.dat.data[0][:] = ctx.Ubar
        model.W.dat.data[1][:] = ctx.Udef
        model.W.dat.data[2][:] = ctx.H

        adjoint.delta.dat.data[0][:] = delta_Ubar
        adjoint.delta.dat.data[1][:] = delta_Udef
        adjoint.delta.dat.data[2][:] = delta_H*(ctx.H>(model.thklim(0.0)+1e-5)).to(torch.float64)

        adjoint.backward(adjoint.delta)

        return torch.tensor(adjoint.g_H0.dat.data[:]),torch.tensor(adjoint.g_B.dat.data[:]),torch.tensor(adjoint.g_beta.dat.data[:]),torch.tensor(adjoint.g_adot.dat.data[:]),None,None,None,None,None,None,None


class SurfaceIntegral:
    def __init__(self,model,p=2):
        self.model = model
        self.S_obs = df.Function(model.Q_thk)
        self.S = df.Function(model.Q_thk)
        self.w = df.TestFunction(model.Q_thk)

        self.I = 1./p*abs(self.S - self.S_obs)**p*df.dx
        self.J = df.derivative(self.I,self.S,self.w)

class SurfaceCost(torch.autograd.Function):
    @staticmethod
    def forward(ctx,S,S_obs,surface_integral):
        ctx.surface_integral = surface_integral
        ctx.S = torch.tensor(S)
        ctx.S_obs = torch.tensor(S_obs)
        surface_integral.S_obs.dat.data[:] = S_obs
        surface_integral.S.dat.data[:] = S
        return torch.tensor(df.assemble(surface_integral.I))

    @staticmethod
    def backward(ctx, grad_output):
        surface_integral = ctx.surface_integral
        S = ctx.S
        S_obs = ctx.S_obs
        surface_integral.S_obs.dat.data[:] = S_obs
        surface_integral.S.dat.data[:] = S
        j = df.assemble(surface_integral.J)
        return torch.tensor(j.dat.data[:])*grad_output, None, None

class VelocityIntegral:
    def __init__(self,model,mode='lin',gamma=1.0,nu=10.,p=2,velocity_space_degree=3):
        self.model = model
        self.Q = df.FunctionSpace(model.mesh,"CG",velocity_space_degree)
        #self.Q_dg = self.Q#df.FunctionSpace(model.mesh,"DG",0)
        self.Q_dg = df.FunctionSpace(model.mesh,"DG",0)
        self.V = df.VectorFunctionSpace(model.mesh,"CG",velocity_space_degree)
        self.U_obs = df.Function(self.V)
        self.tau_obs = df.Function(self.Q)
        self.mask = df.Function(self.Q_dg)
        self.U_bar = df.Function(model.Q_bar)
        self.U_def = df.Function(model.Q_def)
        self.w_bar = df.TestFunction(model.Q_bar)
        self.w_def = df.TestFunction(model.Q_def)
        self.U_s = U_s = self.U_bar - 0.25*self.U_def
        self.U_mag = df.sqrt(df.dot(U_s,U_s) + df.Constant(gamma))
        self.U_obs_mag = df.sqrt(df.dot(self.U_obs,self.U_obs) + df.Constant(gamma))
        nu = df.Constant(nu)

        if mode=='log':
            self.I = self.mask*self.tau_obs*df.ln(self.U_mag/self.U_obs_mag)**2*df.dx
        elif mode=='t':
            r = self.U_s - self.U_obs
            self.I = (nu+1)/2.*df.ln(df.Constant(1) + self.tau_obs/nu*df.dot(r,r))*df.dx
        else:    
            r = self.U_s - self.U_obs
            self.I = (1./p)*self.mask*self.tau_obs**(p/2.)*df.dot(r,r)**(p/2.)*df.dx
        self.J_bar = df.derivative(self.I,self.U_bar,self.w_bar)
        self.J_def = df.derivative(self.I,self.U_def,self.w_def)

class VelocityCost(torch.autograd.Function):
    @staticmethod
    def forward(ctx,U_bar,U_def,U_obs,tau_obs,mask,velocity_integral):
        ctx.velocity_integral = velocity_integral
        ctx.U_bar = U_bar
        ctx.U_def = U_def
        ctx.U_obs = U_obs
        ctx.tau_obs = tau_obs
        ctx.mask = mask
        velocity_integral.U_obs.dat.data[:] = U_obs
        velocity_integral.U_bar.dat.data[:] = U_bar
        velocity_integral.U_def.dat.data[:] = U_def
        velocity_integral.tau_obs.dat.data[:] = tau_obs
        velocity_integral.mask.dat.data[:] = mask
        return torch.tensor(df.assemble(velocity_integral.I))

    @staticmethod
    def backward(ctx, grad_output):
        velocity_integral = ctx.velocity_integral
        U_bar = ctx.U_bar
        U_def = ctx.U_def
        U_obs = ctx.U_obs
        tau_obs = ctx.tau_obs
        mask = ctx.mask
        velocity_integral.U_obs.dat.data[:] = U_obs
        velocity_integral.U_bar.dat.data[:] = U_bar
        velocity_integral.U_def.dat.data[:] = U_def
        velocity_integral.tau_obs.dat.data[:] = tau_obs
        velocity_integral.mask.dat.data[:] = mask
        j_bar = df.assemble(velocity_integral.J_bar)
        j_def = df.assemble(velocity_integral.J_def)
        return torch.tensor(j_bar.dat.data[:])*grad_output, torch.tensor(j_def.dat.data[:])*grad_output, None, None, None, None

class Residual:
    def __init__(self,model):
        self.model = model
        self.V = df.VectorFunctionSpace(model.mesh,"DG",0)
        self.Q = df.FunctionSpace(model.mesh,"DG",0)
        self.V_obs = df.VectorFunctionSpace(model.mesh,"CG",3)
        self.V_mod = df.FunctionSpace(model.mesh,"RT",1)

        self.r = df.Function(self.V)
        self.delta = df.Function(self.V)
        self.lamda = df.Function(self.V)
        self.w = df.TestFunction(self.V)
        
        self.dw = df.TrialFunction(self.V)
        self.U_obs = df.Function(self.V_obs)
        self.U = df.Function(self.V_mod)
        self.Phi = df.TestFunction(self.V_mod)

        self.f = df.TestFunction(self.Q)
        self.M = df.assemble(self.f*df.dx).dat.data[:,np.newaxis]

        self.fwd = df.dot(self.w,self.dw - (self.U - self.U_obs))*df.dx
        self.adj = -df.dot(self.w,self.dw)*df.dx# + df.dot(self.w,self.delta)*df.dx
        self.grad = -df.dot(self.lamda,self.Phi)*df.dx

        self.ksp = PETSc.KSP().create()
        self.ksp.setType('gmres')
        pc = self.ksp.getPC()
        pc.setType('ilu')
        self.ksp.setTolerances(1e-6)


class Projector(torch.autograd.Function):
    @staticmethod
    def forward(ctx,U,U_obs,residual):
        ctx.residual = residual
        ctx.U = U
        ctx.U_obs = U_obs
        residual.U_obs.dat.data[:] = U_obs
        residual.U.dat.data[:] = U

        df.solve(df.lhs(residual.fwd)==df.rhs(residual.fwd),residual.r)
        return torch.tensor(residual.r.dat.data[:])

    @staticmethod
    def backward(ctx, grad_output):
        residual = ctx.residual
        U = ctx.U
        U_obs = ctx.U_obs
        residual.U_obs.dat.data[:] = U_obs
        residual.U.dat.data[:] = U
        residual.delta.dat.data[:] = grad_output.detach().numpy()

        A = df.assemble(residual.adj,mat_type='aij')
        residual.ksp.setOperators(A.M.handle)

        with residual.delta.dat.vec_ro as vec:
            with residual.lamda.dat.vec as sol:
                residual.ksp.solve(vec,sol)

        J_bar = df.assemble(residual.grad)
        return torch.tensor(J_bar.dat.data[:]), None, None


class VelocityToUbFiredrake:
    def __init__(self,model):
        self.model = model
        self.V = df.VectorFunctionSpace(model.mesh,"CG",1)

        self.r = df.Function(self.V)
        self.delta = df.Function(self.V)
        self.lamda = df.Function(self.V)
        self.w = df.TestFunction(self.V)
        
        self.dw = df.TrialFunction(self.V)

        self.Ubar = df.Function(model.Q_bar)
        self.Udef = df.Function(model.Q_def)

        self.Phi_bar = df.TestFunction(model.Q_bar)
        self.Phi_def = df.TestFunction(model.Q_def)

        self.fwd = df.dot(self.w,self.dw - (self.Ubar + self.Udef))*df.dx
        self.adj = -df.dot(self.w,self.dw)*df.dx
        self.gradbar = -df.dot(self.lamda,self.Phi_bar)*df.dx
        self.graddef = -df.dot(self.lamda,self.Phi_def)*df.dx

        self.ksp = PETSc.KSP().create()
        self.ksp.setType('gmres')
        pc = self.ksp.getPC()
        pc.setType('ilu')
        self.ksp.setTolerances(1e-6)


class VelocityToUb(torch.autograd.Function):
    @staticmethod
    def forward(ctx,Ubar,Udef,f):
        ctx.f = f
        ctx.Ubar = Ubar
        ctx.Udef = Udef
        f.Ubar.dat.data[:] = Ubar
        f.Udef.dat.data[:] = Udef

        df.solve(df.lhs(f.fwd)==df.rhs(f.fwd),f.r)
        return torch.tensor(f.r.dat.data[:])

    @staticmethod
    def backward(ctx, grad_output):
        f = ctx.f
        Ubar = ctx.Ubar
        Udef = ctx.Udef
        f.Ubar.dat.data[:] = Ubar
        f.Udef.dat.data[:] = Udef
        f.delta.dat.data[:] = grad_output.detach().numpy()

        A = df.assemble(f.adj,mat_type='aij')
        f.ksp.setOperators(A.M.handle)

        with f.delta.dat.vec_ro as vec:
            with f.lamda.dat.vec as sol:
                f.ksp.solve(vec,sol)

        J_bar = df.assemble(f.gradbar)
        J_def = df.assemble(f.graddef)
        return torch.tensor(J_bar.dat.data[:]), torch.tensor(J_def.dat.data[:]), None


