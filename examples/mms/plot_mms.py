import numpy as np
import pickle
import matplotlib.pyplot as plt

results_dir = './results/'
fig,axs = plt.subplots(nrows=2)

H_inf, H_2, U_2 = pickle.load(open(f'{results_dir}/time/norms_time_dependent.p','rb'))
#H_2 = H_inf


N = np.array([v for v in H_2.keys()])
M = np.array([v for v in H_2[N[0]].keys()])

dxs = 1./N
dts = 1./M

for i,n in enumerate(N):
    eps_H_final = []
    eps_U_final = []
    for j,m in enumerate(M):
        X_H = np.array(H_2[n][m])
        X_U = np.array(U_2[n][m])
        eps_H_final.append(X_H[-1,1])
        eps_U_final.append(X_U[-1,1])
    c = i/len(N)
    cmap = plt.cm.viridis
    eps_H_final = np.array(eps_H_final)
    eps_U_final = np.array(eps_U_final)

    axs[0].loglog(dts,eps_H_final,'^-',base=10,c=cmap(c),label=f'$\Delta x = 1/{n}$')

H_inf, H_2, U_2 = pickle.load(open(f'{results_dir}/static/norms_static.p','rb'))
N = np.array([v for v in H_inf.keys()])
M = np.array([v for v in H_inf[N[0]].keys()])
dxs = 1./N
eps_U = []
for i,n in enumerate(N):
    for j,m in enumerate(M):
        eps_U.append(U_2[n][m][0][1])
eps_U = np.array(eps_U)

axs[1].loglog(dxs,eps_U,'k^-',base=10,label='$\Delta t=0$')


range2 = np.log2(dts.max()) - np.log2(dts.min())

axs[0].loglog([dts.min(),dts.max()],[2**-12,2**(-12+range2)],'r-',base=10,alpha=0.3,label='$\mathcal{O}(\Delta t)$')
axs[0].set_xlabel('$\Delta t$')
axs[0].set_ylabel('$\| H_{mms}- H\|_{L_2}$')

axs[0].legend()
axs[0].minorticks_on()
axs[0].grid(visible=True,which='major')

range2 = np.log2(dxs.max()) - np.log2(dxs.min())
axs[1].loglog([dxs.min(),dxs.max()],[2**-12,2**(-12+2*range2)],'r-',base=10,alpha=0.3,label='$\mathcal{O}(\Delta x^2)$')
axs[1].set_xlabel('$\Delta x$')
axs[1].set_ylabel('$\| \\bar{\mathbf{u}}_{mms}- \\bar{\mathbf{u}} \|_{L_2}$')
axs[1].legend()
axs[1].minorticks_on()
axs[1].grid(visible=True,which='major')

axs[0].text(0.95, 0.9, 'a', horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes,color='black',fontsize=20)
axs[1].text(0.95, 0.9, 'b', horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes,color='black',fontsize=20)
fig.set_size_inches(4,8)
fig.savefig('./figures/mms_results.pdf',bbox_inches='tight')










