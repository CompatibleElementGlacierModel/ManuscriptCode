import torch
import numpy as np

def batch_vec(M):
    return torch.permute(M,[0,2,1]).reshape(M.shape[0],-1)
    
def batch_mat(v,shape):
    return torch.permute(v.reshape(v.shape[0],shape[1],shape[0]),[0,2,1]) 

def batch_mm(matrix, vector_batch):
    return matrix.mm(vector_batch.T).T

def batch_kron(A,B,z):
    return batch_vec(B @ batch_mat(z,(B.shape[1],A.shape[0])) @ A.T)

def bpcg(A,b,device,P_inv=lambda x: x,tol=1e-3,max_iter=10000):
    with torch.no_grad():
        x = torch.zeros_like(b,device=device)
        r = b - A(x)
        z = P_inv(r)
        p = z
        counter = 0
        tol = tol
        t = t0 = torch.linalg.norm(r,axis=1)
        while ((t/t0).max() > tol) & (counter<max_iter):
            Ap = A(p)
            alpha = torch.nan_to_num((r*z).sum(axis=-1)/(((p * Ap).sum(axis=-1))))
            x += alpha.unsqueeze(-1)*p
            r_plus = r - alpha.unsqueeze(-1)*Ap
            z_plus = P_inv(r_plus)
            beta = torch.nan_to_num((r_plus*z_plus).sum(axis=-1)/(((r*z).sum(axis=-1))))
            p = z_plus + beta.unsqueeze(-1)*p
            r = r_plus
            z = z_plus
            counter += 1
            t = torch.linalg.norm(r,axis=1)
        print(counter)
    return x

class KroneckerRandomizedMarginalLikelihood(torch.autograd.Function):
    @staticmethod
    def forward(ctx,Kx,Ky,sigma2,mu,W_train,W_train_t,y_train,Omega,n_train,device):
       
        ctx.Kx = Kx
        ctx.Ky = Ky
        ctx.sigma2 = sigma2
        ctx.mu = mu
        ctx.W_train = W_train
        ctx.W_train_t = W_train_t
        sigma = torch.sqrt(sigma2)
        sigma_inv = 1./sigma
        
        def mv_ident(z):
            r_0 = z*sigma_inv
            r_1 = batch_mm(W_train_t,r_0)
            r_2 = batch_kron(Kx,Ky,r_1)
            r_3 = batch_mm(W_train,r_2)
            r_4 = r_3*sigma_inv
            return r_4    
        
        def mv(z):
            r_0 = z * sigma
            r_1 = mv_ident(r_0) + r_0
            r_2 = r_1 * sigma
            return r_2
               
        Y = mv_ident(Omega)
        del Omega
        q,_ = torch.linalg.qr(Y.T)
        del Y
        R = mv_ident(q.T).T
        B = q.T @ R
        del R

        u,s,v = torch.linalg.svd(B)
        s[s<0] = 0.
        L = q @ (u * torch.sqrt(s))
        del q
        mat = torch.linalg.inv(L.T @ L + torch.eye(L.shape[1],device=device))
        
        log_det_loss = torch.log(s + 1.0).sum() + torch.log(sigma2).sum()

        def P_inv(z):
            r0 = z*sigma_inv
            r1 = batch_mm(L.T,r0)
            r2 = batch_mm(mat,r1)
            r3 = batch_mm(L,r2)
            r4 = r0 - r3
            return r4*sigma_inv   
        
        ctx.alpha = bpcg(mv,y_train-mu,device,P_inv=P_inv)
        
        ctx.fixed_noise = torch.randn(100,n_train,device=device)
        ctx.z_noise = bpcg(mv,ctx.fixed_noise,device,P_inv=P_inv)
        
        del L
        
        data_loss = ((y_train-mu)*ctx.alpha).sum()
        print(data_loss/n_train,log_det_loss/n_train)
        return data_loss + log_det_loss, mv(ctx.alpha)
        
    @staticmethod
    def backward(ctx,delta,na):
        
        Kx = ctx.Kx
        Ky = ctx.Ky
        
        n_x = Kx.shape[1]
        n_y = Ky.shape[1]
        
        z = batch_mm(ctx.W_train_t,ctx.alpha)
        Z = batch_mat(z,(n_y,n_x)).squeeze()
        
        dLdKy = -Z @ Kx @ Z.T
        dLdKx = -Z.T @ Ky @ Z
        dLdSigma = -(ctx.alpha**2)
        dLdmu = -2*ctx.alpha
        
        z_ = batch_mm(ctx.W_train_t,ctx.fixed_noise)
        Z = batch_mat(z_,(n_y,n_x))
        u = batch_mm(ctx.W_train_t,ctx.z_noise)
        U = batch_mat(u,(n_y,n_x))
        dDdKx = (U.permute(0,2,1) @ (ctx.Ky @ Z)).mean(axis=0)
        dDdKy = (U @ (ctx.Kx @ Z.permute(0,2,1))).mean(axis=0)
        dDdSigma = (ctx.fixed_noise*ctx.z_noise).mean(axis=0)
        
        return (dLdKx + 0.5*(dDdKx + dDdKx.T))*delta,(dLdKy + 0.5*(dDdKy + dDdKy.T))*delta, (dLdSigma + dDdSigma)*delta, dLdmu*delta, None,None,None,None,None,None


class StructuredGridInterpolant:

    def __init__(self,mesh,dem,X,log_l,log_sigma2,log_amplitude,w_0,s_x,s_y,device,function_space_type='DG',function_space_degree=0,Z_loc=0.0,Z_scale=1.,X_loc=(0.0,0.0),X_scale=1.0,target_resolution=400,nx=251,ny=251,nodatavalue=-9999):
        self.mesh = mesh
        self.dem = dem
        self.device = device
        
        self.log_l = torch.tensor(log_l,device=device,requires_grad=True,dtype=torch.float)
        self.log_sigma2 = torch.tensor(log_sigma2,device=device,requires_grad=True,dtype=torch.float)
        self.log_amplitude = torch.tensor(log_amplitude,device=device,requires_grad=True,dtype=torch.float)
        self.w_0 = torch.tensor(w_0,device=device,requires_grad=True,dtype=torch.float)
        self.s_x = torch.tensor(s_x,device=device,requires_grad=True,dtype=torch.float)
        self.s_y = torch.tensor(s_y,device=device,requires_grad=True,dtype=torch.float)
        
        self.X = X

        self.X -= X_loc
        self.X /= X_scale

        self.X_scale = X_scale
        self.X_loc = X_loc
        self.Z_scale = Z_scale
        self.Z_loc = Z_loc
        
        self.target_resolution = target_resolution

        self.X_train,self.Z_train = self.get_training_data_from_bbox(nodatavalue)

        self.n_train = len(self.Z_train)
        self.n_test = len(self.X)

        self.X_train = torch.from_numpy(self.X_train).to(torch.float)
        self.Z_train = torch.from_numpy(self.Z_train).to(torch.float).unsqueeze(0)
        self.X = torch.from_numpy(self.X).to(torch.float)

        self.x_ = torch.linspace(-1.1,1.1,nx)
        self.y_ = torch.linspace(-1.1,1.1,ny) 

        X_,Y_ = torch.meshgrid([self.x_,self.y_])
        self.X_ = torch.vstack((batch_vec(X_.unsqueeze(0)),batch_vec(Y_.unsqueeze(0)))).T

        self.W_test,self.W_test_t = self.build_interpolation_matrix(self.X,self.X_,self.x_,self.y_)
        self.W_train,self.W_train_t = self.build_interpolation_matrix(self.X_train,self.X_,self.x_,self.y_)

        self.X_train = self.X_train.to(device)
        self.Z_train = self.Z_train.to(device)

        self.x_ = self.x_.to(device)
        self.y_ = self.y_.to(device)

        self.W_train = self.W_train.to(device)
        self.W_train_t = self.W_train_t.to(device)

        self.W_test = self.W_test.to(device)
        self.W_test_t = self.W_test_t.to(device)

    def optimize_hyperparameters(self,n_iterations=0,n_samples=100,lr=3e-2):

        log_l = self.log_l
        log_sigma2 = self.log_sigma2
        log_amplitude = self.log_amplitude
        w_0 = self.w_0
        s_x = self.s_x
        s_y = self.s_y

        X_train = self.X_train
        Z_train = self.Z_train
        W_train = self.W_train
        W_train_t = self.W_train_t

        n_train = self.n_train

        optimizer = torch.optim.Adam([log_l,log_sigma2,log_amplitude,w_0,s_x,s_y],lr=lr)
        loss = KroneckerRandomizedMarginalLikelihood()


        for i in range(n_iterations):
            optimizer.zero_grad()
            Omega = torch.randn(n_samples,n_train,device=self.device)
            
            mu_vec = w_0 + X_train[:,0]*s_x + X_train[:,1]*s_y
            sigma2 = torch.exp(log_sigma2)
            sigma2_vec = sigma2*torch.ones(n_train,device=device)
            
            l = torch.exp(log_l)
            amplitude = torch.exp(log_amplitude)
            Kx = self.k(self.x_,self.x_,l,amplitude)
            Ky = self.k(self.y_,self.y_,l,amplitude)   

            l_0,y_pred = loss.apply(Kx,Ky,sigma2_vec,mu_vec,W_train,W_train_t,Z_train,Omega,n_train)
            l_0 = l_0/n_train
            
            l_0.backward()
            optimizer.step()
            print(f"i: {i}, L: {l_0.item():.2f}, l: {l.item():.4f}, s2: {sigma2.item():.6f}, a: {amplitude.item():.3f}, w_0: {w_0.item():.3f}, s_x: {s_x.item():.3f}, s_y: {s_y.item():.3f}")

    def predict(self):

        
        log_l = self.log_l
        log_sigma2 = self.log_sigma2
        log_amplitude = self.log_amplitude
        w_0 = self.w_0
        s_x = self.s_x
        s_y = self.s_y

        X_train = self.X_train
        Z_train = self.Z_train
        X_test = self.X
        
        W_train = self.W_train
        W_train_t = self.W_train_t
        W_test = self.W_test
        W_test_t = self.W_test_t

        n_train = self.n_train
        n_test = self.n_test
            
        mu_vec = w_0 + X_train[:,0]*s_x + X_train[:,1]*s_y
        sigma2 = torch.exp(log_sigma2)
        sigma2_vec = sigma2*torch.ones(n_train,device=self.device)
        sigma = torch.sqrt(sigma2)
        sigma_inv = 1./sigma

            
        l = torch.exp(log_l)
        amplitude = torch.exp(log_amplitude)
        Kx = self.k(self.x_,self.x_,l,amplitude)
        Ky = self.k(self.y_,self.y_,l,amplitude)   

        def mv_ident(z):
            r_0 = z*sigma_inv
            r_1 = batch_mm(W_train_t,r_0)
            r_2 = batch_kron(Kx,Ky,r_1)
            r_3 = batch_mm(W_train,r_2)
            r_4 = r_3*sigma_inv
            return r_4    

        def mv(z):
            r_0 = z * sigma
            r_1 = mv_ident(r_0) + r_0
            r_2 = r_1 * sigma
            return r_2

        def mv_post(z):
            r_0 = batch_mm(W_test_t.to(self.device),z)
            r_1 = batch_kron(Kx,Ky,r_0)
            r_2 = batch_mm(W_train,r_1)
            r_3 = bpcg(mv,r_2,self.device,max_iter=1000)
            r_4 = batch_mm(W_train_t,r_3)
            r_5 = batch_kron(Kx,Ky,r_4)
            r_6 = batch_mm(W_test.to(self.device),r_1 - r_5)

            return r_6
            
        with torch.no_grad():
            alpha = bpcg(mv,Z_train - mu_vec,self.device,max_iter=10000)

            H = torch.vstack([torch.ones(n_train,device=self.device),X_train[:,0],X_train[:,1]])
            H_test = torch.vstack([torch.ones(n_test,device=self.device),X_test[:,0].cuda(),X_test[:,1].cuda()])
            z = bpcg(mv,H,self.device,max_iter=10000)
            middle_inv = torch.linalg.inv(H @ z.T)
            r1 = batch_mm(W_train_t,z)
            r2 = batch_kron(Kx,Ky,r1)
            R = H_test - batch_mm(W_test,r2)
            

            Omega_test = torch.randn(1000,n_test,device=self.device)

            Y_2 = (R.T @ (middle_inv @ (R @ Omega_test.T))).T
            Y_1 = mv_post(Omega_test)

            q,_ = torch.linalg.qr((Y_1 + Y_2).T)
            R = mv_post(q.T).T
            B = q.T @ R
            u,s,v = torch.linalg.svd(B)   

        U = q @ u
        L = U * torch.sqrt(s)
        Sigma_post_diag = (L**2).sum(axis=1)

        mu_test = w_0 + s_x*X_test[:,0].cuda() + s_y*X_test[:,1].cuda()
        y_test = batch_mm(W_test.cuda(),batch_kron(Kx,Ky,batch_mm(W_train_t,alpha))) + mu_test
        return y_test.detach().cpu(),Sigma_post_diag.detach().cpu()
                
    def get_training_data_from_bbox(self,nodatavalue):

        # Get DEM elevations
        Z_dem = self.dem.read().squeeze()[::-1].astype(float)

        # Get edge coordinates
        x_dem_ = np.linspace(self.dem.bounds.left,self.dem.bounds.right,self.dem.width+1)
        y_dem_ = np.linspace(self.dem.bounds.bottom,self.dem.bounds.top,self.dem.height+1)

        # Get cell center coordinates
        x_dem = 0.5*(x_dem_[:-1] + x_dem_[1:])
        y_dem = 0.5*(y_dem_[:-1] + y_dem_[1:])

        # Get extremal locations
        x_max,y_max = self.mesh.coordinates().max(axis=0)
        x_min,y_min = self.mesh.coordinates().min(axis=0)

        # 
        x_in = (x_dem > (x_min-self.target_resolution)) & (x_dem < (x_max+self.target_resolution))
        y_in = (y_dem > (y_min-self.target_resolution)) & (y_dem < (y_max+self.target_resolution))

        # Keep valid locations
        x_dem = x_dem[x_in]
        y_dem = y_dem[y_in]
        Z_dem = Z_dem[y_in][:,x_in]

        # Downsample DEM
        dx = abs(x_dem[1] - x_dem[0])
        dy = abs(y_dem[1] - y_dem[0])

        skip_x = int(self.target_resolution // dx)
        skip_y = int(self.target_resolution // dy)

        x_ = (x_dem[::skip_x].reshape(-1,1) - self.X_loc[0])/self.X_scale
        y_ = (y_dem[::skip_y].reshape(-1,1) - self.X_loc[1])/self.X_scale

        X_dem,Y_dem = np.meshgrid(x_,y_)
        Z_dem = Z_dem[::skip_y,::skip_x]

        z = Z_dem.ravel(order='F')
        inds = (z!=nodatavalue)
        Z_train = z[inds]

        Z_train -= self.Z_loc
        Z_train /= self.Z_scale

        X_train = np.c_[X_dem.ravel(order='F'),Y_dem.ravel(order='F')][inds]
        return X_train,Z_train

    @staticmethod
    def build_interpolation_matrix(X,X_,x_,y_):
        rows = []
        cols = []
        vals = []
        
        #nx = len(x_)
        #ny = len(y_)
        delta_x = x_[1] - x_[0]
        delta_y = y_[1] - y_[0]

        nx = len(x_)
        ny = len(y_)
        m = nx*ny
        
        xmin = x_.min()
        #xmax = xmin + (nx + 1)*delta_x
        
        ymin = y_.min()
        #ymax = ymin + (ny + 1)*delta_y

        for ii,xx in enumerate(X):
            
            x_low = int(torch.floor((xx[0] - xmin)/delta_x))
            x_high = x_low + 1

            y_low = int(torch.floor((xx[1] - ymin)/delta_y))
            y_high = y_low + 1
            
            #print(xx,x[x_low],x[x_high],y[y_low],y[y_high])

            ll = x_low + y_low*nx
            ul = x_low + y_high*nx
            lr = x_high + y_low*nx
            ur = x_high + y_high*nx
            bbox = [ll,ul,lr,ur]

            dist = torch.sqrt(((xx - X_[bbox])**2).sum(axis=1))

            w = 1./dist
            w/=w.sum()

            rows.append(torch.ones((4))*ii)
            cols.append(torch.tensor(bbox))
            vals.append(w) 

        inds = torch.vstack((torch.hstack(rows),torch.hstack(cols)))
        tens = torch.sparse_coo_tensor(inds,torch.hstack(vals),(X.shape[0],m))
        return tens,torch.transpose(tens,1,0)
    
    @staticmethod
    def k(x1,x2,l,amplitude):
        D = torch.cdist(x1.unsqueeze(-1),x2.unsqueeze(-1))
        return amplitude*torch.exp(-D**2/(l**2))


