import torch
import pickle
import numpy as np

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

def ravel_f(a):
    return a.T.ravel()

def reshape_f(a,shape):
    return a.reshape(shape[1],shape[0]).T

class SurfaceMap:
    def __init__(self,filename):
        with open(filename,'rb') as fi:
            self.data = pickle.load(fi)

        self.n_train = self.data['observation_basis']['coeff_map'].shape[0]
        self.n_test = self.data['model_basis']['coeff_map'].shape[0]
        self.m = self.data['model_basis']['coeff_map'].shape[1]

        self.L_test = self.data['model_basis']['coeff_map']
        self.L_train = self.data['observation_basis']['coeff_map']
        
        self.h_test = self.data['model_basis']['mean_map']
        self.h_train = self.data['observation_basis']['mean_map']

        self.w_post = self.data['coefficients']['post_mean']
        self.beta_post = self.data['coefficients']['mean_coeff']

        self.X = self.data['plotting'][0]
        self.X_train = self.data['data']['x_obs']
        self.Z_train = self.data['data']['z_obs']

    def evaluate(self,z,mode='test'):
        if mode=='test':
            return self.h_test @ self.beta_post + self.L_test @ z
        else:
            return self.h_train @ self.beta_post + self.L_train @ z

    def map_point(self,mode='test'):
        return self.evaluate(self.w_post,mode=mode)

class BedMap:
    def __init__(self,filename):
        with open(filename,'rb') as fi:
            self.data = pickle.load(fi)

        self.n_train = self.data['observation_basis']['coeff_map'].shape[0]
        self.n_test = self.data['model_basis']['coeff_map'].shape[0]
        self.m = self.data['model_basis']['coeff_map'].shape[1]

        self.L_test = self.data['model_basis']['coeff_map']
        self.L_train = self.data['observation_basis']['coeff_map']
        
        self.h_test = self.data['model_basis']['mean_map']
        self.h_train = self.data['observation_basis']['mean_map']

        self.w_post = self.data['coefficients']['post_mean']
        self.beta_post = self.data['coefficients']['mean_coeff']
        self.G = self.data['coefficients']['post_cov_root']

        self.L_post = self.L_test @ self.G

        self.X = self.data['data']['x_test']
        self.X_train = self.data['data']['x_obs']
        self.Z_train = self.data['data']['z_obs']

        self.Uhat = None

    def evaluate(self,z,mode='test'):
        if mode=='test':
            return self.h_test @ self.beta_post + self.L_test @ (self.w_post + self.G @ z)
        else:
            return self.h_train @ self.beta_post + self.L_train @ (self.w_post + self.G @ z)

    def map_point(self,mode='test'):
        return self.evaluate(torch.zeros(self.G.shape[1]),mode=mode)

    def marginal_variance(self,mode='prior'):
        if mode=='prior':
            return ((self.L_test)**2).sum(axis=1)
        elif mode=='posterior_observation':
            return ((self.L_post)**2).sum(axis=1)
        else:
            return ((self.L_post)**2).sum(axis=1) - ((self.L_post @ self.Uhat)**2).sum(axis=1)

class BetaMap:
    def __init__(self,filename):
        with open(filename,'rb') as fi:
            self.beta_map_x,self.beta_map_t = pickle.load(fi)

        self.n_space = self.beta_map_x.shape[0]
        self.n_t = self.beta_map_t.shape[0]
        self.m_space = self.beta_map_x.shape[1]
        self.m_time = self.beta_map_t.shape[1]

        self.Uhat = None

    def evaluate(self,z=None,Z=None):
        if z is not None and Z is not None:
            return self.beta_map_x @ z, self.beta_map_x @ Z @ self.beta_map_t.T
        if z is not None:
            return self.beta_map_x @ z
        if Z is not None:
            return self.beta_map_x @ Z @ self.beta_map_t.T

    def marginal_variance(self,mode='prior'):
        if mode=='prior':
            return (self.beta_map_x**2).sum(axis=1)
        else:
            return (self.beta_map_x**2).sum(axis=1) - ((self.beta_map_x @ self.Uhat)**2).sum(axis=1)

class AdotMap:
    def __init__(self,filename):
        with open(filename,'rb') as fi:
            #self.adot_mean,self.adot_map,self.mu_post,self.G,self.f = pickle.load(fi)
            self.L0x,self.L1x,self.adot_mean,self.adot_map = pickle.load(fi)

        self.n = self.adot_map.shape[0]
        self.m = self.adot_map.shape[1]

        self.Uhat = None

    def evaluate(self,z):
        return self.adot_mean + self.adot_map @ z

    def marginal_variance(self,mode='prior'):
        if mode=='prior':
            return (self.adot_map**2).sum(axis=1)
        else:
            return (self.adot_map**2).sum(axis=1) - ((self.adot_map @ self.Uhat)**2).sum(axis=1)

class LaplaceFromSamples:
    def __init__(self,filenames,bed_map=None,beta_map=None,adot_map=None,dif_map=None,method='onepass',maxrank=None):


        if method=='eig':
            with open(filename,'rb') as fi:
                Om,Q = pickle.load(fi)
            print('here')
            B_ = Q.T @ Om
            B = 0.5*(B_ + B_.T)
            Lamda,U = torch.linalg.eigh(B)
            Lamda[Lamda<1e-10] = 1e-10
        elif method=='onepass':
       
            Ys = []
            Omegas = []
            for f in filenames:
                with open(f,'rb') as fi:
                    G,g,Omega = pickle.load(fi)
                Y = (G - g.reshape(-1,1))/1e-3
                Ys.append(Y)
                Omegas.append(Omega)

            Y = torch.hstack(Ys)
            Omega = torch.hstack(Omegas)

            if maxrank is not None:
                Y = Y[:,:maxrank]
                Omega = Omega[:,:maxrank]

            Q,R = torch.linalg.qr(Y)
            f = torch.linalg.solve(Omega.T @ Q, R.T)
            u,s,v = torch.linalg.svd(f,full_matrices=False)
            Lamda = s
            Ubar = Q @ u
            
        else:
            with open(filename,'rb') as fi:
                Om,Q = pickle.load(fi)
            U,S,_ = torch.linalg.svd(Om.T)
            Lamda = S#**2
            Ubar = Q @ U

        self.Lamda = Lamda
        self.Ubar = Ubar

        # Diagonal factor (Eq. 5.4 Bui-Thanh)
        D = 1./(1/Lamda + 1)

        self.Uhat = Ubar * torch.sqrt(D)
        # Posterior Covariance (full size)
        #S_bar = torch.eye(z_B.shape[0]) - Ubar * D @ Ubar.T

        # Matrix root (Bui-Thanh Appendix).  This should be used only for mat-vec products.
        self.P = 1./torch.sqrt(Lamda + 1) - 1
        #L_bar = Ubar * P @ Ubar.T + torch.eye(Ubar.shape[0])

        if bed_map is not None:
            bed_map.Uhat = self.Uhat[:bed_map.m]
        if beta_map is not None:
            beta_map.Uhat = self.Uhat[bed_map.m:bed_map.m+beta_map.m_space]
        if adot_map is not None:
            adot_map.Uhat = self.Uhat[bed_map.m+beta_map.m_space:bed_map.m+beta_map.m_space+adot_map.m]

    def sample(self):
        eps = torch.randn(self.Ubar.shape[0])
        return eps + (self.Ubar * self.P) @ (self.Ubar.T @ eps) 


