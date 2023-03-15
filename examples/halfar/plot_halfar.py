import numpy as np
import pickle
import matplotlib.pyplot as plt

results_dir = '../../results/halfar/'
fig,axs = plt.subplots(nrows=2)

H_2 = pickle.load(open(f'{results_dir}/norms_static.p','rb'))[0]

models = np.array([v for v in H_2.keys()])
M = np.array([v for v in H_2[models[0]].keys()])

dxs = 1./M*6e5

for i,model in enumerate(models):
    eps_H = []
    for j,m in enumerate(M):
        eps_H.append(H_2[model][m])
    
    eps_H = np.array(eps_H)*1000

    if model=='RT':
        axs[0].loglog(dxs,eps_H,'^-',base=10,c='black',label='Raviart-Thomas')
    else:
        axs[0].loglog(dxs,eps_H,'^-',base=10,c='green',label='Mardal-Tai-Winther')

axs[0].plot([10**4,10**5],[10**0.9,10**1.9],'r-',alpha=0.5,label='$\mathcal{O}(\Delta x)$')


axs[0].set_xlabel('$\Delta x$')
axs[0].set_ylabel('$\| H_{Halfar}- H\|_{L_2}$')

axs[0].legend()
axs[0].minorticks_on()
axs[0].grid(visible=True,which='both')

results_dir = '../../results/halfar_profile/'

H_profile = pickle.load(open(f'{results_dir}/H_profiles.p','rb'))[0]
H_profile_onestep = pickle.load(open(f'../../results/halfar_onestep/H_profiles.p','rb'))[0]
x = H_profile['RT'][64][:,0]
H_model = H_profile['RT'][64][:,1]
H_onestep = H_profile_onestep['RT'][64][:,1]
H_true = H_profile['RT'][64][:,3]

axs[1].plot(x*6e2,H_model*1000,'k-',label='Raviart-Thomas')
axs[1].plot(x*6e2,H_true*1000,'r--',label='Exact')
axs[1].plot(x*6e2,H_onestep*1000,'g:',label='Single step',alpha=0.5)
#axs[1].grid()
axs[1].set_xlabel('$x$ (km)')
axs[1].set_ylabel('$H$ (m)')
ax2 = axs[1].twinx()
ax2.plot(x*6e2,(H_true - H_model)*1000,color='blue')
ax2.set_ylabel('$H_{Halfar} - H$ (m)')
ax2.spines['right'].set_color('blue')
ax2.tick_params(axis='y', colors='blue')
ax2.yaxis.label.set_color('blue')
ax2.grid()
axs[1].grid(axis='x')
axs[1].legend()

axs[0].text(0.95, 0.9, 'a', horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes,color='black',fontsize=20)
axs[1].text(0.95, 0.9, 'b', horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes,color='black',fontsize=20)

fig.set_size_inches(4,6)
fig.savefig('./figures/halfar_convergence.pdf',bbox_inches='tight')










