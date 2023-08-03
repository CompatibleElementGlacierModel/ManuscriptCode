import numpy as np
import pickle
import matplotlib.pyplot as plt
import firedrake as df

results_dir = './results/'
fig,ax = plt.subplots(nrows=1)
M = 200

Hs = []
with df.CheckpointFile(f"{results_dir}/functions_{M}_RT.h5", 'r') as afile:
    mesh = afile.load_mesh("mesh")
    for i in range(0,10000,400):
        Hs.append(afile.load_function(mesh, "H0",idx=i))

Hs_mtw = []
with df.CheckpointFile(f"{results_dir}/functions_{M}_MTW.h5", 'r') as afile:
    mesh = afile.load_mesh("mesh")
    for i in range(0,10000,400):
        Hs_mtw.append(afile.load_function(mesh, "H0",idx=i))

X = df.SpatialCoordinate(mesh)
x,y = X
a = df.Constant(1000/25000)
b_0 = df.Constant(500)
x_m = df.Constant(20000)
x_s = df.Constant(7000)
m_0 = df.Constant(2.)
n = df.Constant(3.0)
rho = df.Constant(917.0)
g = df.Constant(9.81)
A = df.Constant(1e-16)

m_dot = n*m_0/(x_m**(2*n - 1))*x**(n-1)*abs(x_m - x)**(n-1)*(x_m - 2*x)

x = 25000*x
B_exp = b_0*(df.Constant(1)/(df.Constant(1) + df.exp(-a*(x_s-x))))
h_right = ((2*n + 2)*(n+2)**(1./n)*m_0**(1./n)/(2**(1./n)*6*n*A**(1./n)*rho*g*x_m**((2*n-1)/n))*(x_m + 2*x)*(x_m - x)**2)**(n/(2*n + 2))

h_plus = ((2*n + 2)*(n+2)**(1./n)*m_0**(1./n)/(2**(1./n)*6*n*A**(1./n)*rho*g*x_m**((2*n-1)/n))*(x_m + 2*x_s)*(x_m - x_s)**2)**(n/(2*n + 2))

h_minus = df.Max(df.Constant(0.0),h_plus - b_0)
h_left = (h_minus**((2*n+2)/n) - h_plus**((2*n+2)/n) + (2*n + 2)*(n+2)**(1./n)*m_0**(1./n)/(2**(1./n)*6*n*A**(1./n)*rho*g*x_m**((2*n-1)/n))*(x_m + 2*x)*(x_m - x)**2)**(n/(2*n + 2))

Q_thk = df.FunctionSpace(mesh,'DG',0)
Q_smth = Q_thk#df.FunctionSpace(mesh,'CG',1)
B = df.interpolate(B_exp,Q_thk)
H_r = df.interpolate(h_right,Q_smth)
H_l = df.interpolate(h_left,Q_smth)

x_ = np.linspace(0,0.8,1001)


X = np.c_[x_,np.zeros_like(x_)*1.5/M]
X_l = X[X[:,0]<x_s(0,0)/25000]
X_r = X[X[:,0]>=x_s(0,0)/25000]
B_profile = np.array(B(X))


Hr_profile = np.array(H_r(X_r))
Hl_profile = np.array(H_l(X_l))
H_true = np.hstack((Hl_profile,Hr_profile))

#for i,H in enumerate(Hs):
#    H_profile = np.array(H(X))*1000
#    if i<len(Hs)-1:
#        ax.plot(x_*25,B_profile+H_profile,'b-',alpha=0.5,linewidth=1)
#    else:
#        ax.plot(x_*25,B_profile+H_profile,'b-',alpha=0.5,linewidth=1,label='SIA')

H_profile = np.array(Hs[-1](X))*1000
sia_line, = ax.plot(x_*25,B_profile+H_profile,'k-',linewidth=2,label='SIA')

H_mtw_profile = np.array(Hs_mtw[-1](X))*1000
mtw_line, = ax.plot(x_*25,B_profile+H_mtw_profile,'g-',linewidth=2,label='BPA')

exact_line, = ax.plot(x_*25,B_profile+H_true,'r-',linewidth=2,label='Exact')
ax.plot(x_*25,B_profile,'k-',linewidth=2)

ax2 = ax.twinx()
diff_line, = ax2.plot(x_*25,H_profile-H_true,'b-',linewidth=0.5,label='SIA - Exact')

ax.legend(handles=[sia_line,mtw_line,exact_line],loc='lower left')
fig.set_size_inches(4,4)
ax.set_xlabel('$x$ (km)')
ax.set_ylabel('Elevation (m)')
ax2.set_ylabel('$H_{jsa} - H_{sia}$ (m)')
ax2.spines['right'].set_color('blue')
ax2.tick_params(axis='y', colors='blue')
ax2.yaxis.label.set_color('blue')
fig.savefig('figures/steep_topography.pdf',bbox_inches='tight')



