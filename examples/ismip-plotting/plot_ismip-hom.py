import matplotlib
import matplotlib.pylab as plt
import firedrake as df

import numpy as np
from scipy.interpolate import interp2d
#import netCDF4

FS = ['aas1','aas2','cma1','fpa2','ghg1','jvj1','mmr1','oga1','rhi1','rhi3','spr1','ssu1','yko1']
BP = ['ahu1', 'ahu2', 'bds1', 'cma2', 'fpa1', 'fsa1', 'mbr1', 'rhi2', 'tpa1']
ALL = ['aas1','ahu1','bds1','cma2','fpa1','fsa1','lpe1','mmr1','oga1','rhi1','rhi3','rhi5','ssu1','yuv1','aas2','ahu2','cma1','dpo1','fpa2','jvj1','mbr1','mtk1','oso1','rhi2','rhi4','spr1','tpa1']

models = {"FS" : FS, "BP" : BP, "ALL" : ALL}

# Set to "True" to remove models that appear to be obviously wrong or produce poor-quality
# results (oscillations).
remove_outliers = True

outliers = {"a" : [],
            "b" : [],
            "c" : ["mbr1"],
            "d" : ["rhi1", "rhi2", "rhi3"],
            "f" : []}

def plot_experiment(ax, experiment, length_scale, model_type, N_samples=501, color="blue", plot_models=True):

    filename = "ismip-hom-{exp}-{length}.npz".format(exp=experiment, length=length_scale)

    raw_data = np.load(filename)

    participating_models = [model for model in models[model_type] if model in raw_data]

    if remove_outliers:
        for model in outliers[experiment]:
            if model in participating_models:
                participating_models.remove(model)

    data = np.array([raw_data[model] for model in participating_models])

    xs = raw_data["x"]

    if plot_models:
        for model in participating_models:
            v = raw_data[model]
            ax.plot(xs, v, "-", lw=1, color=color, alpha=0.75)

    vx_mean = np.mean(data, axis=0)
    vx_std = np.std(data, axis=0)

    #ax.plot(xs, vx_mean,
    #        label="{} mean".format(model_type),
    #        lw=2,
    #        color=color)

    ax.fill_between(xs,vx_mean-2*vx_std,vx_mean+2*vx_std,alpha=0.5)

def plot(experiment, length_scales, axs):
    for length_scale, ax in zip(length_scales, axs):

        #ax.set_title("{} km".format(int(length_scale)))

        #ax.set_xlabel('x (normalized)')
        #ax.set_ylabel('vx (m / year)')

        models = True
        plot_experiment(ax, experiment, length_scale, "ALL", color="gray", plot_models=False)
        plot_hdiv(ax, experiment, length_scale)

        #ax.legend()


def plot_hdiv(ax, experiment, length_scale):
    "Plot PISM's ISMIP-HOM results"
    if experiment in 'abcd':
        filename = f'../ismip-hom-{experiment}/results/ismip{experiment}-L-{int(length_scale)*1000}.h5'
        print(filename)

        with df.CheckpointFile(filename, 'r') as afile:
            mesh = afile.load_mesh("mesh")
            U_s = afile.load_function(mesh, "U_s", idx=0)

        samples = np.c_[np.linspace(0.0001,0.9999,100),0.25*np.ones(100)]
        v = np.array(U_s(samples))[:,0]*100

        ax.plot(samples[:,0],v,color='red',lw=2,label='MTW')
    else:
        filename = f'../ismip-hom-{experiment}/results/ismipf-L-{length_scale}.h5'
        print(filename)
        with df.CheckpointFile(filename, 'r') as afile:
            mesh = afile.load_mesh("mesh")
            S = afile.load_function(mesh, "S", idx=9)

        samples = np.c_[np.linspace(0.0001,0.9999,1000),0.5*np.ones(1000)]
        S = np.array(S(samples))*1000

        ax.plot(samples[:,0],S,color='red',lw=2,label='MTW')


def grid_plot(experiment_name):
    if experiment_name in 'abcd':

        fig, axs = plt.subplots(2, 3)
        fig.dpi = 100
        fig.set_size_inches(12, 8)
        # fig.suptitle("ISMIP HOM Experiment {}".format(experiment_name.upper()))
        fig.tight_layout(h_pad=4)
        fig.subplots_adjust(top=0.9, bottom=0.1)

        row1 = plot(experiment_name, ["005", "010", "020"], axs[0])
        row2 = plot(experiment_name, ["040", "080", "160"], axs[1])

    else:
        fig,axs = plt.subplots(2,1)

        fig.dpi = 100
        fig.set_size_inches(12, 8)
        # fig.suptitle("ISMIP HOM Experiment {}".format(experiment_name.upper()))
        fig.tight_layout(h_pad=4)
        fig.subplots_adjust(top=0.9, bottom=0.1)

        row1 = plot(experiment_name, ["000", "001"], axs)

        fig.savefig("ismiphom-{}.png".format(experiment_name))
    

if __name__ == "__main__":

    fig,axs = plt.subplots(3,1,sharex=True)
    fig.set_size_inches(4.5,9)
    plot('a',['040'],[axs[0]])
    plot('c',['040'],[axs[1]])
    plot('f',['001'],[axs[2]])

    axs[2].set_xlim(0,1)
    axs[2].set_xlabel('Normalized $x$')
    axs[2].set_ylabel('$\Delta S$ (m)')

    axs[0].set_ylabel('$u_x$ (m a$^{-1}$)')
    axs[1].set_ylabel('$u_x$ (m a$^{-1}$)')

    axs[0].text(0.95, 0.9, 'a', horizontalalignment='center', verticalalignment='center', transform=axs[0].transAxes,color='black',fontsize=20)
    axs[1].text(0.95, 0.9, 'b', horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes,color='black',fontsize=20)
    axs[2].text(0.95, 0.9, 'c', horizontalalignment='center', verticalalignment='center', transform=axs[2].transAxes,color='black',fontsize=20)


    fig.subplots_adjust(hspace=0.05)
    fig.savefig('./figures/ismip_summary.pdf',bbox_inches='tight')

    #for ex in "f":
    #    grid_plot(ex)
