
import os
import sys
import pickle
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.append('../../..')

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("once",category=DeprecationWarning)
    import firedrake as df
    from firedrake.petsc import PETSc
    from speceis_dg.hybrid_diff import CoupledModel, CoupledModelAdjoint, FenicsModel, SurfaceIntegral, SurfaceCost, VelocityIntegral, VelocityCost
    from speceis_dg.Bases import BedMap, SurfaceMap, BetaMap, AdotMap, LaplaceFromSamples
import logging
logging.captureWarnings(True)
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

import matplotlib.pyplot as plt
from matplotlib import colors
import cmasher
import pyproj
import shapefile as shp  # Requires the pyshp package


def project_array(coordinates, from_epsg=4326, to_epsg=3338, always_xy=True):
    """
    Project a numpy (n,2) array from <from_epsg> to <to_epsg>
    Returns the projected numpy (n,2) array.
    """
    tform = pyproj.Transformer.from_crs(crs_from=from_epsg, crs_to=to_epsg, always_xy=always_xy)
    fx, fy = tform.transform(coordinates[:,0], coordinates[:,1])
    # Re-create (n,2) coordinates
    return np.dstack([fx, fy])[0]


def refine(x,n_times=1):
    for i in range(n_times):
        x_avg = (x[1:] + x[:-1])/2
        x_new = np.zeros((2*x.shape[0] - 1, x.shape[1]))
        x_new[::2] = x
        x_new[1::2] = x_avg
        x = x_new
    return x

len_scale = 50000
thk_scale = 5000
vel_scale = 100

X_scale = 49038.977309852955
X_loc = np.array([ 743600.34864157, 1204809.11692685])

data_dir = '../meshes/mesh_2200/'
prefix = 'v3'
results_dir = f'{data_dir}/{prefix}/' 
ensemble_dir = 'ensemble_linear'
experiment_type = 'present_climate_calve'
sample_dir = f'{data_dir}/{prefix}/uncertainty/'

mesh = df.Mesh(f'{data_dir}/mesh.msh',name='mesh')
Q_dg = df.FunctionSpace(mesh,"DG",0)
Q_rt = df.FunctionSpace(mesh,"RT",1)
Q_mtw = df.FunctionSpace(mesh,"MTW",3)
Q_cg2 = df.VectorFunctionSpace(mesh,"CG",3)
Q_cg = df.FunctionSpace(mesh,"CG",1)
Q_cg_3 = df.FunctionSpace(mesh,'CG',3)

bed_map = BedMap(f'{data_dir}/bed/bed_basis.p')
surf_map = SurfaceMap(f'{data_dir}/surface/time_series/map_cop30.p')
beta_map = BetaMap(f'{data_dir}/beta/beta_basis.p')
adot_map = AdotMap(f'{data_dir}/adot/adot_basis.p')
laplace_ = LaplaceFromSamples([f'{sample_dir}/hvps/hvp_{k}.p' for k in range(30)],bed_map=bed_map,beta_map=beta_map,adot_map=adot_map,method='onepass',maxrank=None)
#laplace = LaplaceFromSamples([f'{sample_dir}/hvps/hvp_{k}.p' for k in range(26)],bed_map=bed_map,beta_map=beta_map,adot_map=adot_map,method='onepass',maxrank=None)

z_B = torch.zeros(bed_map.m)
z_beta_ref = torch.randn(beta_map.m_space)
z_beta_t = torch.randn(beta_map.m_space,beta_map.m_time)
z_adot = torch.zeros(adot_map.m)

initfile = 'state_009.p'#max(os.listdir(f'{data_dir}/{prefix}/time/states'))
with open(f'{data_dir}/{prefix}/time/states/{initfile}','rb') as fi:
    z_beta_ref.data[:],z_beta_t.data[:],z_B.data[:],z_adot.data[:],z_nse,_ = pickle.load(fi)

B = df.Function(Q_dg)
B_std = df.Function(Q_dg)
log_beta = df.Function(Q_cg)
log_beta_std = df.Function(Q_cg)
H = df.Function(Q_dg)
H_pres = df.Function(Q_dg)
H_frac = df.Function(Q_dg)
H_frac_cg = df.Function(Q_cg)
U_s = df.Function(Q_mtw)

U_bar = df.Function(Q_mtw)
U_def = df.Function(Q_rt)

U_obs = df.Function(Q_cg2)
U_temp = df.Function(Q_cg2)
S = df.Function(Q_dg)

n_runs = 25

boundary = np.loadtxt('../preprocessing/boundary.csv',delimiter=',')
boundary *=X_scale
boundary += X_loc
boundary[:,0] -= 0.07*X_scale

x_samples = np.array(pickle.load(open('x_samples_center_branch.p','rb')))
x_samples = np.vstack((x_samples,x_samples[-1]-np.array([0,2000])))
r = np.cumsum(np.hstack(([0],np.linalg.norm(x_samples[1:] - x_samples[:-1],axis=1))))/1000

sf = shp.Reader("../data/boundary/geology/ice_margin.shp")
vg = shp.Reader("../data/boundary/geology/veg_line.shp")

for shape in sf.shapeRecords():
    xx = np.array([i[0] for i in shape.shape.points[:]])
    yy = np.array([i[1] for i in shape.shape.points[:]])
    X_boundary = project_array(np.vstack((xx,yy)).T,from_epsg=32607)

for shape in vg.shapeRecords():
    xx = np.array([i[0] for i in shape.shape.points[:]])
    yy = np.array([i[1] for i in shape.shape.points[:]])
    X_veg = project_array(np.vstack((xx,yy)).T,from_epsg=32607)

plot_adot_comparison=False
plot_posterior_bed=False
plot_posterior_bed_profiles=False
plot_posterior_bed_profiles_validation=False
plot_posterior_thickness=False
plot_profile=False
plot_misfit=False
plot_delta=False
plot_volumes=True
plot_velocity_posterior=False
plot_velocity_appendix=False
plot_surface_mass_balance=False
plot_talk_graphic=False

if plot_posterior_bed:

    #colors_undersea = plt.cm.terrain(np.linspace(0, 0.17, 256))
    #colors_land = plt.cm.terrain(np.linspace(0.25, 1, 256))
    colors_undersea = cmasher.ocean_r(np.linspace(0, 1, 256))
    colors_land = cmasher.savanna(np.linspace(0, 1, 256))
    all_colors = np.vstack((colors_undersea, colors_land))

    terrain_map = colors.LinearSegmentedColormap.from_list(
        'terrain_map', all_colors)

    class MidpointNormalize(colors.Normalize):
        def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
            self.vcenter = vcenter
            super().__init__(vmin, vmax, clip)

        def __call__(self, value, clip=None):
            # I'm ignoring masked values and all kinds of edge cases to make a    log_beta_ref = beta_map_x @ z_beta_ref
            # simple example...
            # Note also that we must extrapolate beyond vmin/vmax
            x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1.]
            return np.ma.masked_array(np.interp(value, x, y,
                                                left=-np.inf, right=np.inf))
        def inverse(self, value):
            y, x = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
            return np.interp(value, x, y, left=-np.inf, right=np.inf)


    midnorm = MidpointNormalize(vmin=-500., vcenter=-10, vmax=4000)
    fig,axs = plt.subplots(nrows=2,ncols=2)
    
    B.dat.data[:] = bed_map.evaluate(z_B)*thk_scale
    B_std.dat.data[:] = (bed_map.marginal_variance(mode='posterior')**0.5)*thk_scale


    c1 = df.tripcolor(B,axes=axs[1,0],rasterized=True,norm=midnorm,cmap=terrain_map)
    c2 = df.tripcolor(B_std,axes=axs[1,1],vmin=0,vmax=300,rasterized=True,cmap=cmasher.torch)
    
    B.dat.data[:] = bed_map.evaluate(torch.zeros_like(z_B))*thk_scale
    B_std.dat.data[:] = (bed_map.marginal_variance(mode='posterior_observation')**0.5)*thk_scale
    
    df.tripcolor(B,axes=axs[0,0],rasterized=True,norm=midnorm,cmap=terrain_map)
    c0 = df.tripcolor(B_std,axes=axs[0,1],vmin=0,vmax=1000,rasterized=True,cmap=cmasher.torch)


    for ax in axs.ravel():
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    cb1 = plt.colorbar(c1,ax=axs[1,0],location='bottom',shrink=0.8,extend='both',pad=0.02,label='Bed Elev. (m)')
    cb1.set_ticks([-500, 0, 2000, 4000])
    cb2 = plt.colorbar(c1,ax=axs[0,0],location='bottom',shrink=0.8,extend='both',pad=0.02,label='Bed Elev. (m)')
    cb2.set_ticks([-500, 0, 2000, 4000])

    plt.colorbar(c2,ax=axs[1,1],location='bottom',shrink=0.8,extend='both',pad=0.02,label='Bed Std. Dev. (m)')
    plt.colorbar(c0,ax=axs[0,1],location='bottom',shrink=0.8,extend='both',pad=0.02,label='Bed Std. Dev. (m)')

    for ax in axs.ravel():
        ax.plot(*X_boundary.T,'w--')

    axs[1,0].plot(*x_samples.T,'r-')
    axs[1,0].plot([7.473e5,7.616e5],[1.2404e6,1.2276e6],'w-')
    axs[1,0].text(7.473e5,1.2404e6,'A',ha='left',va='bottom',color='white')
    axs[1,0].text(7.616e5,1.2276e6,'A\'',ha='left',va='bottom',color='white')
    #axs[1,1].plot([7.473e5,7.616e5],[1.2404e6,1.2276e6],'g-')

    axs[0,0].text(0.01, 0.99, 'a',
     horizontalalignment='left',
     verticalalignment='top',
     transform = axs[0,0].transAxes,fontweight='bold')
    axs[0,1].text(0.01, 0.99, 'b',
     horizontalalignment='left',
     verticalalignment='top',
     transform = axs[0,1].transAxes,fontweight='bold')
    axs[1,0].text(0.01, 0.99, 'c',
     horizontalalignment='left',
     verticalalignment='top',
     transform = axs[1,0].transAxes,fontweight='bold')
    axs[1,1].text(0.01, 0.99, 'd',
     horizontalalignment='left',
     verticalalignment='top',
     transform = axs[1,1].transAxes,fontweight='bold')

    fig.subplots_adjust(wspace=0,hspace=0)
    fig.set_size_inches(9.0,13)
    fig.savefig(f'{results_dir}/plots/posterior_bed_map.pdf',dpi=300,bbox_inches='tight')


if plot_adot_comparison:
    melt_meas = np.array([[59.871034,-140.336906],[59.826082,-140.671635],[59.834383,-140.775957],[60.003848,-141.061608]])
    x_melt = project_array(melt_meas[:,::-1])
    obs_melt = np.array([-6.24, -4.94, -4.83, -4.82])
    sites = ['Mal_E_L','Mal_C_L','Mal_Ch','Aga_L']
    elevs = [310,366,304,308]

    adot_sample = df.Function(Q_dg)
    adot_exp = []
    for i in range(n_runs):
        with open(f'{data_dir}/{prefix}/{ensemble_dir}/projected_climate_calve/run_{i}/data_2023.p','rb') as fi:
            data = pickle.load(fi)
            adot_sample.dat.data[:] = data[3]*thk_scale
        
        adot_exp.append(np.array([adot_sample.at(x_melt)]))


    fig,axs=plt.subplots(nrows=2,ncols=2,sharex=True,sharey=True)
    adot_exp = np.array(adot_exp).squeeze()
    bins = np.linspace(-8,0,9)
    for j,(ax,s,z) in enumerate(zip(axs.ravel(),sites,elevs)):
        ax.hist(adot_exp[:,j],bins,histtype='step',color='black',linewidth=3.0,label='$P(\dot{a})$',density=True)
        ax.axvline(obs_melt[j],color='red',linewidth=3.0,label='Obs. 2022')
        ax.axvline(adot_exp[0,j],color='black',linewidth=3.0,linestyle='--',label='MAP')
        if j==0:
            ax.legend(loc='upper left')
        ax.text(0.98,0.98,f'{s}, {z}m',ha='right',va='top',transform=ax.transAxes)  

    axs[-1,0].set_xlabel('$\dot{a}$ (m a$^{-1}$)')
    axs[-1,0].set_ylabel('$P(\dot{a})$')
    fig.subplots_adjust(hspace=0,wspace=0)
    fig.set_size_inches(9,5)
    fig.savefig('plots/melt_comparison.pdf',bbox_inches='tight')


if plot_posterior_bed_profiles:

    B.dat.data[:] = bed_map.evaluate(z_B)*thk_scale
    B_std.dat.data[:] = (bed_map.marginal_variance(mode='posterior')**0.5)*thk_scale

    p1 = [B.at(list(x),dont_raise=True) for x in x_samples]
    B_profile = np.array([x if x is not None else np.nan for x in p1])

    p1 = [B_std.at(list(x),dont_raise=True) for x in x_samples]
    B_std_profile = np.array([x if x is not None else np.nan for x in p1])

    fig,axs = plt.subplots(nrows=3,sharex=True)
    years = [2013,2073,2173]
    S_sample = df.Function(Q_dg)
    B_sample = df.Function(Q_dg)
    adot_sample = df.Function(Q_dg)
    experiment_types_2013 = ['projected_climate_calve']
    experiment_types_2073 = ['projected_climate_calve','present_climate_calve']
    experiment_types_2173 = ['projected_climate_calve','present_climate_calve']
    #experiment_types_2013 = ['projected_climate_no_calve']
    #experiment_types_2073 = ['projected_climate_no_calve']#,'present_climate_calve']
    #experiment_types_2173 = ['projected_climate_no_calve']#,'present_climate_calve']
    experiment_types = [experiment_types_2013,experiment_types_2073,experiment_types_2173]
    colors = ['b','c']

    for y,ax,typ in zip(years,axs,experiment_types):
        ax2 = ax.twinx()
       
        for i in range(n_runs):
            adot_run = []
            for jj,t in enumerate(typ):

                with open(f'{data_dir}/{prefix}/{ensemble_dir}/{t}/run_{i}/data_{y}.p','rb') as fi:
                    data = pickle.load(fi)
                    S_0 = torch.maximum((data[0] + data[1])*thk_scale,(1-0.917)*data[0]*thk_scale)
                    B_0 = torch.maximum(torch.from_numpy(B.dat.data[:]),-0.917*data[0]*thk_scale)
                    S_sample.dat.data[:] = S_0
                    B_sample.dat.data[:] = B_0
                    adot_sample.dat.data[:] = data[3]*thk_scale

                p1 = [S_sample.at(list(x),dont_raise=True) for x in x_samples]
                S_profile = np.array([x if x is not None else np.nan for x in p1])
                p1 = [B_sample.at(list(x),dont_raise=True) for x in x_samples]
                Base_profile = np.array([x if x is not None else np.nan for x in p1])
                p1 = [adot_sample.at(list(x),dont_raise=True) for x in x_samples]
                adot_profile = np.array([x if x is not None else np.nan for x in p1])

                set_to_zero = (S_profile - Base_profile) < 3
                S_profile[set_to_zero] = np.nan
                Base_profile[set_to_zero] = np.nan
                
                if i==0:
                    ax.plot(r,S_profile,f'{colors[jj]}-')
                    ax.plot(r,Base_profile,f'{colors[jj]}-')
                    ax2.plot(r,adot_profile,f'{colors[jj]}:')
                else:
                    ax.plot(r,S_profile,f'{colors[jj]}-',alpha=0.1)
                    ax.plot(r,Base_profile,f'{colors[jj]}-',alpha=0.1)
                    ax2.plot(r,adot_profile,f'{colors[jj]}:',alpha=0.1)
            #adot_run.append(adot_exp)
        #adot_point.append(adot_run)
                
        S.dat.data[:] = surf_map.map_point()*thk_scale
        p1 = [S.at(list(x),dont_raise=True) for x in x_samples]
        Surface_profile = np.array([x if x is not None else np.nan for x in p1])

        ax.plot(r,B_profile,'k-')
        ax.fill_between(r,B_profile - 2*B_std_profile,B_profile + 2*B_std_profile,color='black',alpha=0.2)
        ax.plot(r,Surface_profile,'r--')
        ax.set_xlim(0,r.max())
        ax.axhline(0,linestyle='--',color='black')
        ax.axvline(79,linestyle='-.',color='black')
        ax2.axhline(0,linestyle=':',color='black')
        ax2.set_ylim(-15,3)
        ax2.set_ylabel('SMB (m a$^{-1}$)')
        ax.set_xlabel('Arc Length (km)')
        ax.set_ylabel('Elevation (m)')
        ax.text(0.99,0.99,f'{y}',ha='right',va='top',transform=ax.transAxes)

    fig.subplots_adjust(hspace=0)
    fig.set_size_inches(9,13)
    fig.savefig(f'{results_dir}/plots/posterior_bed_profiles.pdf')
 
    profiles = [5,8,9,14,15,16,18]
    fig,axs = plt.subplots(nrows=3,ncols=3,sharex=True,sharey=True)
    gs = axs[1,2].get_gridspec()
    for ax in axs[1:,-1]:
        ax.remove()
    axbig = fig.add_subplot(gs[1:,-1])

    axg = [axs[0,0],axs[0,1],axs[0,2],axs[1,0],axs[1,1],axs[2,0],axs[2,1]]

    HH = df.Function(Q_dg)
    HH.dat.data[:] = S.dat.data[:] - B.dat.data[:]
    df.tripcolor(HH,axes=axbig,vmin=0,vmax=1500,rasterized=True)
    for j,(ax,p) in enumerate(zip(axg,profiles)):
        data = np.loadtxt(f'../data/bed/gbr/profile{p}.csv',delimiter=',',skiprows=1)
        H_pro = data[:,-1]
        coords = project_array(data[:,1:-1][:,::-1])
        r_pro = np.hstack(([0],np.cumsum(np.linalg.norm(coords[1:] - coords[:-1],axis=1))))
        S_mod = np.array(S.at(coords))
        B_mod = np.array(B.at(coords))
        Bstd_mod = np.array(B_std.at(coords))
        ax.plot(r_pro,S_mod,'b-')
        ax.plot(r_pro,B_mod,'k-')
        ax.fill_between(r_pro,B_mod - 3*Bstd_mod,B_mod + 3*Bstd_mod, color='black',alpha=0.2)
        ax.plot(r_pro,S_mod - H_pro,'o',color=plt.cm.bwr(j/7.))
        ax.axhline((S_mod - H_pro).mean(),linestyle='--',color=plt.cm.bwr(j/7))
        ax.axhline((B_mod).mean(),linestyle='--',color='black')

        axbig.scatter(*coords.T,c=plt.cm.bwr(j/7),vmin=0,vmax=1500)
        #ax.text(0.99,0.99,f'Profil}',ha='right',va='top',transform=ax.transAxes)

   
        #ax.set_title(f'Profile {p}, H_mod={int((S_mod-B_mod).mean())}, H_obs={int((H_pro.mean()))}')

    axs[-1,0].set_xlabel('Dist. (m)')
    axs[-1,0].set_ylabel('Elev. (m)')

    axbig.set_xlim(740000,760000)
    axbig.set_ylim(1.22e6,1.26e6)
    axbig.set_xticks([])
    axbig.set_yticks([])
    
    fig.set_size_inches(9,9)
    fig.subplots_adjust(wspace=0,hspace=0)
    fig.savefig('plots/model_to_gbr_comparison.pdf',bbox_inches='tight')


if plot_posterior_bed_profiles_validation:

    B.dat.data[:] = bed_map.evaluate(z_B)*thk_scale
    B_std.dat.data[:] = (bed_map.marginal_variance(mode='posterior')**0.5)*thk_scale


    bed_map_v = BedMap(f'../meshes/mesh_2201/bed/bed_basis_checkerboard.p')
    laplace_v = LaplaceFromSamples([f'../meshes/mesh_2201/v5/uncertainty/hvps/hvp_{k}.p' for k in range(27)],bed_map=bed_map_v,beta_map=beta_map,adot_map=adot_map,method='onepass',maxrank=None)

    B_o = df.Function(Q_dg)
    B_o_std = df.Function(Q_dg)
    initfile = 'state_009.p'#max(os.listdir(f'{data_dir}/{prefix}/time/states'))
    with open(f'../meshes/mesh_2201/v5/time/states/{initfile}','rb') as fi:
        _,_,z_B_o,_,_,_ = pickle.load(fi)
    B_o.dat.data[:] = bed_map_v.evaluate(z_B_o)*thk_scale
    B_o_std.dat.data[:] = (bed_map_v.marginal_variance(mode='posterior')**0.5)*thk_scale

    xs = np.linspace(-0.5,0.5,6)*X_scale + X_loc[0]
    
    y_g = np.linspace(mesh.coordinates.dat.data[:,1].min(),mesh.coordinates.dat.data[:,1].max(),301)
    
    from scipy.spatial.kdtree import KDTree
    fig,axs = plt.subplots(nrows=len(xs),sharex=True,sharey=False)
    fig2,ax2 = plt.subplots()
    for n,(xx,ax) in enumerate(zip(xs,axs)):

        x_g = np.ones_like(y_g)*xx

        picks = (np.vstack((x_g,y_g)).T - X_loc)/X_scale
        tree = KDTree(picks)
        ds,inds = tree.query(bed_map.data['data']['x_obs'][-1887:])
        
        bb = B.at(bed_map.data['data']['x_obs'][-1887:]*X_scale + X_loc,dont_raise=True)
        bb = np.array(bb)[ds<1e-2].astype(float)

        #inds = inds[ds<4e-2]

        subs = 10
        start = np.linspace(-1,1,subs+1)
        mask = np.ones(picks.shape[0])
        for i,_ in enumerate(start[:-1]):
            for j,_ in enumerate(start[:-1]):
                if i%2==j%2:
                    pass
                else:
                    x_upper = start[i+1]
                    x_lower = start[i]
                    y_upper = start[j+1]
                    y_lower = start[j]
                    l_mask = (picks[:,0]<x_upper)*(picks[:,0]>x_lower)*(picks[:,1]>y_lower)*(picks[:,1]<y_upper)
                    mask[l_mask] = 0
        mask = mask.astype(bool)

        p1 = [B.at(list(x),dont_raise=True) for x in zip(x_g,y_g)]
        B_profile = np.array([x if x is not None else np.nan for x in p1])

        p1 = [B_std.at(list(x),dont_raise=True) for x in zip(x_g,y_g)]
        B_std_profile = np.array([x if x is not None else np.nan for x in p1])  

        p1 = [B_o.at(list(x),dont_raise=True) for x in zip(x_g,y_g)]
        B_o_profile = np.array([x if x is not None else np.nan for x in p1])
        
        p1 = [B_o_std.at(list(x),dont_raise=True) for x in zip(x_g,y_g)]
        B_o_std_profile = np.array([x if x is not None else np.nan for x in p1])

        z = B_profile-B_profile
        ax.plot(y_g,z,'k-')
        BB = B_o_profile-B_profile
        ax.plot(y_g,BB,'r-')
        ax.fill_between(y_g,z- 3*B_std_profile,z + 3*B_std_profile,color='black',alpha=0.2)
        ax.fill_between(y_g,BB - 3*B_o_std_profile,BB + 3*B_o_std_profile,color='red',alpha=0.2)


        p1 = [B.at(list(x),dont_raise=True) for x in zip(x_g,y_g)]
        B_profile = np.array([x if x is not None else np.nan for x in p1])

        ax.plot(y_g[inds[ds<1e-2]],bed_map.data['data']['z_obs'][-1887:][ds<1e-2]*5000 - bb,'ro')


        ax.fill_between(y_g,-1000,1000,where=~mask,alpha=0.1)
        ax.set_xlim(y_g.min(),y_g.max())
        ax.set_ylim(-1000,1000)
        ax.text(0.99,0.99,f'{chr(n+97)}',ha='right',va='top',transform=ax.transAxes)
        if n!=5:
            ax.set_yticklabels([])
        


        df.tripcolor(B_o_std,vmin=0,vmax=300,axes=ax2)
        ax2.plot(x_g,y_g,'r-')
        ax2.plot(*(bed_map.data['data']['x_obs'][-1887:][ds<1e-2]*X_scale + X_loc).T,'ro')
        ax2.text(x_g.min(),y_g.min(),chr(n+97),horizontalalignment='center',verticalalignment='top',fontsize=12,color='red')
        ax2.text(x_g.max(),y_g.max(),chr(n+97)+'\'',horizontalalignment='center',verticalalignment='bottom',fontsize=12,color='red')
        ax2.set_aspect('equal')
        ax2.set_xticks([])
        ax2.set_yticks([])
    axs[-1].set_xlabel('Northing (m)')
    axs[-1].set_ylabel('Rel. Elev. (m)')

    fig.subplots_adjust(hspace=0.0)
    fig.set_size_inches(9,6)
    fig.savefig('plots/validation_cross_section.pdf',bbox_inches='tight')
    
    melt_meas = np.array([[59.871034,-140.336906],[59.826082,-140.671635],[59.834383,-140.775957],[60.003848,-141.061608]])
    x_melt = project_array(melt_meas[:,::-1])
    sites = ['Mal_E_L','Mal_C_L','Mal_Ch','Aga_L']
    alignments=['bottom','top','bottom','bottom']

    for x,n,al in zip(x_melt,sites,alignments):
        ax2.plot(x[0],x[1],'^',color='white')
        ax2.text(x[0],x[1],n,horizontalalignment='left',verticalalignment=al,fontsize=10,color='white')

    fig2.set_size_inches(5,5)
    fig2.savefig('plots/validation_map.pdf',bbox_inches='tight')


if plot_posterior_thickness:

    fig,axs = plt.subplots(nrows=1,ncols=2)
    years = [1915,2023]

    nr = [n_runs,n_runs,n_runs,n_runs]
    for y,ax,n in zip(years,axs.ravel(),nr):
        H_mask_proj = torch.zeros(Q_dg.dim())
        H_mask_pres = torch.zeros(Q_dg.dim())
        for i in range(n):
            with open(f'{data_dir}/{prefix}/{ensemble_dir}/projected_climate_no_calve/run_{i}/data_{y}.p','rb') as fi:
                data = pickle.load(fi)

            H.dat.data[:] = data[0]*thk_scale
            if i==0:
                c1 = df.tripcolor(H,axes=ax,vmin=0,vmax=1200,rasterized=True,cmap=cmasher.arctic)

            H_cg = df.project(H,Q_cg)
            df.tricontour(H_cg,[5],axes=ax,colors='red',linewidth=2.0,alpha=0.2)
            
        ax.text(0.99,0.99,f'{y}',ha='right',va='top',transform=ax.transAxes)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    axs[1].plot(*x_samples.T,'r-')

    plt.colorbar(c1,ax=axs[0],location='bottom',shrink=0.8,extend='both',pad=0.02,label='Thickness (m)')
    c = plt.colorbar(c1,ax=axs[1],location='bottom',shrink=0.8,extend='both',pad=0.02,label=' ',alpha=0)
    c.ax.remove()

    axs[0].text(0.01, 0.99, 'a',
     horizontalalignment='left',
     verticalalignment='top',
     transform = axs[0].transAxes,fontweight='bold')
    axs[1].text(0.01, 0.99, 'b',
     horizontalalignment='left',
     verticalalignment='top',
     transform = axs[1].transAxes,fontweight='bold')

    for ax in axs.ravel():
        ax.plot(*X_boundary.T,'w--')
        #ax.plot(*X_veg.T,'g--')

    fig.subplots_adjust(wspace=0,hspace=0)
    fig.set_size_inches(9,13)
    fig.savefig(f'{results_dir}/plots/thickness_extent.pdf',bbox_inches='tight')

    years = [2073,2173]

    scenarios = ['present_climate_no_calve','present_climate_calve','projected_climate_no_calve','projected_climate_calve']
    for y in years:
        fig,axs = plt.subplots(nrows=2,ncols=2)
        for s,ax in zip(scenarios,axs.ravel()):
            for i in range(n):
                with open(f'{data_dir}/{prefix}/{ensemble_dir}/{s}/run_{i}/data_{y}.p','rb') as fi:
                    data = pickle.load(fi)

                H.dat.data[:] = data[0]*thk_scale
                if i==0:
                    c1 = df.tripcolor(H,axes=ax,vmin=0,vmax=1200,rasterized=True,cmap=cmasher.arctic)

                H_cg = df.project(H,Q_cg)
                df.tricontour(H_cg,[5],axes=ax,colors='red',linewidth=2.0,alpha=0.2)
                
            ax.set_ylim(1.1554e6,1.2e6)
            ax.set_aspect('equal')

            ax.set_xticks([])
            ax.set_yticks([])


        #plt.colorbar(c1,ax=axs[1,0],location='bottom',shrink=0.8,extend='both',pad=0.1,label='Thickness (m)')
        #c = plt.colorbar(c1,ax=axs[1,1],location='bottom',shrink=0.8,extend='both',pad=0.1,label=' ',alpha=0)
        #c.ax.remove()

        axs[0,0].text(0.01, 0.99, 'a',
         horizontalalignment='left',
         verticalalignment='top',
         transform = axs[0,0].transAxes,fontweight='bold')
        axs[0,1].text(0.01, 0.99, 'b',
         horizontalalignment='left',
         verticalalignment='top',
         transform = axs[0,1].transAxes,fontweight='bold')
        axs[1,0].text(0.01, 0.99, 'c',
         horizontalalignment='left',
         verticalalignment='top',
         transform = axs[1,0].transAxes,fontweight='bold')
        axs[1,1].text(0.01, 0.99, 'd',
         horizontalalignment='left',
         verticalalignment='top',
         transform = axs[1,1].transAxes,fontweight='bold')

        for ax in axs.ravel():
            ax.plot(*X_boundary.T,'w--')
            #ax.plot(*X_veg.T,'g--')

        axs[0,0].set_ylabel('Frozen Climate')
        axs[1,0].set_ylabel('Projected Climate')
        axs[1,0].set_xlabel('No Calving')
        axs[1,1].set_xlabel('Calving')



        fig.subplots_adjust(wspace=0,hspace=0)
        fig.set_size_inches(9,4.7)
        fig.savefig(f'{results_dir}/plots/thickness_extent_{y}.pdf',bbox_inches='tight')


if plot_profile:
    years = [1995,2000,2007,2010,2013,2018]
    names = ['1995','2000','2007','2010','cop30','2018']

    data_array = np.zeros((len(years),len(r)))
    model_array = np.zeros((len(years),n_runs,len(r)))

    for j,(y,name) in enumerate(zip(years,names)):
        with open(f'{data_dir}/surface/time_series/map_{name}.p','rb') as fi:
            surf = pickle.load(fi)

        X = surf['data']['x_obs']
        d = surf['data']['z_obs']
        X*=X_scale
        X+=X_loc
        d*=thk_scale

        close_indices = []
        for x in x_samples:
            close_indices.append(np.argmin(np.linalg.norm(X - x,axis=1)))


        data_array[j,:] = d[close_indices].numpy()
        for i in range(n_runs):
            with open(f'{data_dir}/{prefix}/{ensemble_dir}/{experiment_type}/run_{i}/data_{y}.p','rb') as fi:
                data = pickle.load(fi)
            S.dat.data[:] = (data[0] + data[1])*thk_scale
            p1 = [S.at(list(x),dont_raise=True) for x in x_samples]
            profile = np.array([x if x is not None else np.nan for x in p1])
            model_array[j,i,:] = profile

    fig,axs = plt.subplots(nrows=len(years),sharex=True)
    cmap = plt.cm.coolwarm
    for j,y in enumerate(years):
        ax = axs[j]
        offset = model_array[4,0]
        offset_ensemble = offset#model_array[4,:].mean(axis=0)
        offset_data = data_array[4]
        #ax.plot(r,data_array[j] - offset_data,color=cmap((y-1995)/(2018-1995)),linestyle='-')
        #ax.plot(r,model_array[j,0] - offset,color=cmap((y-1995)/(2018-1995)),linestyle='--')
        ax.plot(r,data_array[j] - offset_data,color='k',linestyle='-')
        ax.plot(r,model_array[j,0] - offset,color='k',linestyle='--')
        std = model_array[j,1:,:].std(axis=0)
        lb = np.min(model_array[j],axis=0) - offset_ensemble
        ub = np.max(model_array[j],axis=0) - offset_ensemble
        #lb = np.quantile(model_array[j],0.05,axis=0) - offset
        #ub = np.quantile(model_array[j],0.95,axis=0) - offset
        #lb = model_array[j,0] - offset - 2*std
        #ub = model_array[j,0] - offset + 2*std
        #ax.plot(r,(model_array[j,:] - offset).T,color=cmap((y-1995)/(2018-1995)),linestyle='-',alpha=0.05)
        ax.plot(r,(model_array[j,:] - offset_ensemble).T,color='k',linestyle='-',alpha=0.05)
        #ax.fill_between(r,lb,ub,color=cmap((y-1995)/(2018-1995)),alpha=0.2)
        ax.fill_between(r,lb,ub,color='k',alpha=0.2)
        ax.set_xlim(0,r.max())
        ax.set_ylim(-50,50)
        ax.axhline(0,color='red',linestyle=':')
        ax.text(0.01, 0.99, f'{y}',horizontalalignment='left',verticalalignment='top',fontsize='large',transform = ax.transAxes)
    fig.subplots_adjust(hspace=0)
    axs[-1].set_xlabel('Arc Distance (km)')
    axs[-1].set_ylabel('$\Delta$ Elevation')
    for ax in axs[:-1]:
        ax.set_yticks([])

    fig.set_size_inches(9,9)
    fig.savefig(f'{results_dir}/plots/profile_elevation_change.pdf',bbox_inches='tight')

if plot_misfit:
    years = [1995,2000,2003,2007,2009,2010,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021]
    names = []
    for y in years:
        if y!=2013:
            names.append(f'{y}')
        else:
            names.append('cop30')
    fig,axs = plt.subplots(nrows=5,ncols=4,gridspec_kw={'height_ratios': [10,10,10,10,0.5]})
    axs = axs.ravel()

    for j,(y,name,ax) in enumerate(zip(years,names,axs)):
        print(y)
        with open(f'{data_dir}/surface/time_series/map_{name}.p','rb') as fi:
            surf = pickle.load(fi)

        X = surf['data']['x_obs']
        d = surf['data']['z_obs']
        X*=X_scale
        X+=X_loc
        d*=thk_scale
        d = d.numpy()

        with open(f'{data_dir}/{prefix}/{ensemble_dir}/{experiment_type}/run_0/data_{y}.p','rb') as fi:
            data = pickle.load(fi)

        S.dat.data[:] = (data[0] + data[1])*thk_scale

        p1 = S.at(X,dont_raise=True)
        Z_mod = np.array([x if x is not None else np.nan for x in p1])

        #df.triplot(Delta
        c = ax.scatter(*X.T,c=Z_mod - d,s=5,alpha=1.0,cmap=plt.cm.coolwarm,vmin=-20,vmax=20,rasterized=True)
        ax.plot(*boundary.T,'k-')
        ax.set_xticks([])
        ax.set_yticks([])
        #ax.text(*boundary.max(axis=0),f'{y}')
        ax.text(0.01,0.99,f'{y}',ha='left',va='top',transform=ax.transAxes)
        #ax.plot(*X_boundary.T,'k--')


    axs[-2].remove()
    axs[-3].remove()
    axs[-4].remove()

    plt.colorbar(c,cax=axs[-1],orientation='horizontal',label='Misfit (m)',extend='both',shrink=0.8)

    fig.set_size_inches(7.5,11)
    fig.subplots_adjust(wspace=0,hspace=0)
    fig.savefig(f'{results_dir}/plots/misfit_all_years.pdf',bbox_inches='tight',dpi=300)

if plot_delta:
    fig,axs_ = plt.subplots(nrows=5,ncols=4,gridspec_kw={'height_ratios': [10,10,10,10,0.5]})
    axs = axs_.reshape(-1,2)

    with open(f'{data_dir}/{prefix}/{ensemble_dir}/{experiment_type}/run_0/data_2013.p','rb') as fi:
        data = pickle.load(fi)
    S_ref = (data[0] + data[1])*thk_scale

    Delta = df.Function(Q_dg)
    years = [1995,2003,2009,2012,2015,2016,2018,2021]
    names = []
    for y in years:
        if y!=2013:
            names.append(f'{y}')
        else:
            names.append('cop30')

    for j,(y,name,ax) in enumerate(zip(years,names,axs)):
        print(y)


        with open(f'{data_dir}/{prefix}/{ensemble_dir}/{experiment_type}/run_0/data_{y}.p','rb') as fi:
            data = pickle.load(fi)

        S.dat.data[:] = (data[0] + data[1])*thk_scale
        Delta.dat.data[:] = (data[0] + data[1])*thk_scale - S_ref

        df.tripcolor(Delta,axes=ax[0],cmap=plt.cm.coolwarm,vmin=-50,vmax=50,rasterized=True)

        try:
            with open(f'../meshes/mesh_1899/surface/time_series/map_rel_{name}.p','rb') as fi:
                surf = pickle.load(fi)

            X = surf['data']['x_obs']
            d = surf['data']['z_obs']
            X*=X_scale
            X+=X_loc
            d*=thk_scale
            d = d.numpy()
        
            c = ax[1].scatter(*X.T,c=d,s=5,alpha=1.0,cmap=plt.cm.coolwarm,vmin=-50,vmax=50,rasterized=True)
        except FileNotFoundError:
            pass

        #ax.plot(*boundary.T,'k-')
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_xlim(*ax[0].get_xlim())
        ax[1].set_ylim(*ax[0].get_ylim())
        ax[0].text(0.01,0.99,f'{y}',ha='left',va='top',transform=ax[0].transAxes)
        ax[0].plot(*boundary.T,'k-')
        ax[1].plot(*boundary.T,'k-')

    axs_[4,0].remove()
    axs_[4,1].remove()
    axs_[4,2].remove()

    plt.colorbar(c,cax=axs_[4,3],orientation='horizontal',label='$\Delta$ (m)',extend='both',shrink=0.8)


    fig.set_size_inches(8.5,11)
    fig.subplots_adjust(wspace=0,hspace=0)
    fig.savefig(f'{results_dir}/plots/delta_all_years.pdf',bbox_inches='tight',dpi=300)

if plot_volumes:
    years = [y for y in range(1985,2023)] + [y for y in range(2023,1985+360,5)] + [2343] 
    volumes = np.zeros((len(years),n_runs))
    fig,axs = plt.subplots(ncols=2,sharey=False)
    experiment_types = ['projected_climate_no_calve','projected_climate_calve','present_climate_calve','present_climate_no_calve']
    labels = ['Proj. climate, no calving','Proj. climate, calving','Pres. climate, no calving', 'Pres. climate, calving']
    colors = ['g','b','r','c']
    for typ,c,l in zip(experiment_types,colors,labels):
        for j in range(n_runs):
            for i,y in enumerate(years):
                with open(f'{data_dir}/{prefix}/{ensemble_dir}/{typ}/run_{j}/data_{y}.p','rb') as fi:
                    data = pickle.load(fi)
                H.dat.data[:] = data[0]*thk_scale
                volumes[i,j] = df.assemble(H*df.dx)

        volumes/=1e9

        delta = (volumes - volumes[0])#/volumes[0]*100

        #ax.plot(years,delta[:,1:],f'{c}-',alpha=0.2)
        b = 46
        axs[0].plot(years[:b+1],delta[:b+1,0],f'{c}-',label=l)
        axs[0].fill_between(years[:b+1],delta[:b+1].min(axis=1),delta[:b+1].max(axis=1),facecolor=f'{c}',alpha=0.2)
        axs[1].plot(years[b:],delta[b:,0],f'{c}-',label=l)
        axs[1].fill_between(years[b:],delta[b:].min(axis=1),delta[b:].max(axis=1),facecolor=f'{c}',alpha=0.2)
    axs[0].set_xlim(1985,years[b])
    axs[1].set_xlim(years[b],max(years))
    axs[0].set_xlabel('Year')
    axs[0].set_ylabel('Volume Change (km$^3$)')
    axs[1].set_yticks([])
    axs[0].set_ylim(-1200,50)
    axs[1].set_ylim(-1200,50)
    axs[1].legend()
    fig.subplots_adjust(wspace=0)
    fig.set_size_inches(8.5,4)
    fig.savefig(f'{results_dir}/plots/ensemble_prediction.pdf',bbox_inches='tight')

if plot_velocity_posterior:
    pts = np.array([(739617.9902570323, 1231665.3260523882),(745805.4929643238, 1199818.6539656834),(745152.3669084265, 1174011.4032284238)])
    #pts = np.array([(739617.9902570323, 1231665.3260523882),(742790.,1194540),(745152.3669084265, 1174011.4032284238)])
    #pts = np.array([(739617.9902570323, 1231665.3260523882),(742790.,1194540),(745152.3669084265, 1174011.4032284238)])
    profile_average = True
    startpts = np.array([[739550.725385059, 1232611.7700044673],
                        [744304.0860066848, 1199988.2482293672],
                        [739966.0710462283, 1174104.392941417]])
                        
            
    endpts = np.array([[740064.8290611646, 1228807.4028012853],
                       [747854.5489453411, 1199912.7064647148],
                       [749900.7281401302, 1174104.392941417]])
                       


    beta_scale = 1000
    log_beta.dat.data[:] = beta_map.evaluate(z=z_beta_ref)# + np.log(beta_scale)
    log_beta_std.dat.data[:] = beta_map.marginal_variance(mode='posterior')
    
    with open(f'{data_dir}/velocity/velocity.p','rb') as fi:
        [v_avg,v_mask] = pickle.load(fi)
     
    v_avg[np.isnan(v_avg)] = 0.0
    U_obs.dat.data[:] = v_avg

    years = [y for y in range(1985,2019)]

    Ub = 0
    Ud = 0
    u_obs = np.zeros((pts.shape[0],len(years)))
    u_mod = np.zeros((pts.shape[0],len(years),n_runs))
    for i,y in enumerate(years):
        for j in range(n_runs):
            with open(f'{data_dir}/{prefix}/{ensemble_dir}/{experiment_type}/run_{j}/data_{y}.p','rb') as fi:
                data = pickle.load(fi)
            if j==0:
                Ub += (data[-2])*len_scale*vel_scale
                Ud += (data[-1])*len_scale*vel_scale
            U_bar.dat.data[:] = data[-2]*len_scale*vel_scale
            U_def.dat.data[:] = data[-1]*len_scale*vel_scale
            if j%1==0:
                UU = df.project(U_bar - 0.25*U_def,Q_cg2)
                if profile_average:
                    for q in range(3):
                        print("here")
                        x0 = np.linspace(startpts[q,0],endpts[q,0],100)
                        y0 = np.linspace(startpts[q,1],endpts[q,1],100)
                        u_mod[q,i,j] = np.linalg.norm(np.vstack([UU.at(np.copy(x)) for x in np.vstack((x0,y0)).T]).mean(axis=0))
                else:
                    u_mod[0,i,j] = np.linalg.norm(UU(pts[0]))
                    u_mod[1,i,j] = np.linalg.norm(UU(pts[1]))
                    u_mod[2,i,j] = np.linalg.norm(UU(pts[2]))

            else:
                u_mod[0,i,j] = np.linalg.norm(U_s(pts[0]))
                u_mod[1,i,j] = np.linalg.norm(U_s(pts[1]))
                u_mod[2,i,j] = np.linalg.norm(U_s(pts[2]))

        with open(f'{data_dir}/velocity/itslive_annual/velocity_{y}.p','rb') as fi:
            v = pickle.load(fi)
            U_temp.dat.data[:] = v
            if profile_average:
                for q in range(3):
                    print("here")
                    x0 = np.linspace(startpts[q,0],endpts[q,0],100)
                    y0 = np.linspace(startpts[q,1],endpts[q,1],100)
                    u_obs[q,i] = np.linalg.norm(np.vstack([U_temp.at(np.copy(x)) for x in np.vstack((x0,y0)).T]).mean(axis=0))
            else:
                u_obs[0,i] = np.linalg.norm(U_temp(pts[0]))
                u_obs[1,i] = np.linalg.norm(U_temp(pts[1]))
                u_obs[2,i] = np.linalg.norm(U_temp(pts[2]))

    U_bar.dat.data[:] = Ub/len(years)
    U_def.dat.data[:] = Ud/len(years)
    UU = df.project(U_bar - 0.25*U_def,Q_cg2)


    fig,axs = plt.subplots(ncols=2,nrows=2)
    cb = df.tripcolor(log_beta,axes=axs[0,0])
    plt.colorbar(cb,ax=axs[0,0],location='bottom',shrink=0.8,extend='both',pad=0.02,label='Log $\\beta$ (log Pa a m$^{-1}$)')

    cbs = df.tripcolor(log_beta_std,axes=axs[0,1])
    plt.colorbar(cbs,ax=axs[0,1],location='bottom',shrink=0.8,extend='both',pad=0.02,label='Std. Log $\\beta$ (log Pa a m$^{-1}$)')

    cvo = df.tripcolor(U_obs,axes=axs[1,0],vmin=0,vmax=2000)
    plt.colorbar(cvo,ax=axs[1,0],location='bottom',shrink=0.8,extend='both',pad=0.02,label='Obs. Speed (m a$^{-1}$)')
    
    cvm = df.tripcolor(UU,axes=axs[1,1],vmin=0,vmax=2000)
    plt.colorbar(cvm,ax=axs[1,1],location='bottom',shrink=0.8,extend='both',pad=0.02,label='Mod. Speed (m a$^{-1}$)')


    allpts = np.dstack((startpts,endpts))
    if profile_average:
        #axs[1,0].plot(allpts[0,0,:],allpts[0,1,:],'-',color=plt.cm.Spectral(0.0),linewidth=3.0)
        #axs[1,0].plot(allpts[1,0,:],allpts[1,1,:],'-',color=plt.cm.Spectral(0.5),linewidth=3.0)
        #axs[1,0].plot(allpts[2,0,:],allpts[2,1,:],'-',color=plt.cm.Spectral(1.0),linewidth=3.0)
        axs[1,0].plot(allpts[0,0,:],allpts[0,1,:],'-',color='green',linewidth=4.0)
        axs[1,0].plot(allpts[1,0,:],allpts[1,1,:],'-',color='orange',linewidth=4.0)
        axs[1,0].plot(allpts[2,0,:],allpts[2,1,:],'-',color='red',linewidth=4.0)

    else:
        axs[1,0].plot(pts[0,0],pts[0,1],'x',color=plt.cm.cividis(0))
        axs[1,0].plot(pts[1,0],pts[1,1],'x',color=plt.cm.cividis(0.5))
        axs[1,0].plot(pts[2,0],pts[2,1],'x',color=plt.cm.cividis(1.0))


    for ax in axs.ravel():
        ax.plot(*X_boundary.T,'w--')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

    axs[0,0].text(0.01, 0.99, 'a',
     horizontalalignment='left',
     verticalalignment='top',
     transform = axs[0,0].transAxes,fontweight='bold')
    axs[0,1].text(0.01, 0.99, 'b',
     horizontalalignment='left',
     verticalalignment='top',
     transform = axs[0,1].transAxes,fontweight='bold')
    axs[1,0].text(0.01, 0.99, 'c',
     horizontalalignment='left',
     verticalalignment='top',
     transform = axs[1,0].transAxes,fontweight='bold')
    axs[1,1].text(0.01, 0.99, 'd',
     horizontalalignment='left',
     verticalalignment='top',
     transform = axs[1,1].transAxes,fontweight='bold')

    


    fig.subplots_adjust(hspace=0,wspace=0)
    fig.set_size_inches(9,13)
    fig.savefig(f'{results_dir}/plots/velocity_multipanel.pdf',bbox_inches='tight')

    fig,ax = plt.subplots()

    U_mean = u_mod[:,:,0].mean(axis=1)
    U_std = ((u_mod - u_mod[:,:,0].reshape(3,34,1))**2).sum(axis=1).sum(axis=1)**0.5/(34*50)**0.5
    

    #for j in range(3):
    #    ax.axhline(np.nanmean(u_obs[j]),linestyle='--',color='k')
    #    ax.axhline(u_mod.mean(axis=1).max(axis=1)[j],color='k')
    #    ax.axhline(u_mod.mean(axis=1).min(axis=1)[j],color='k')
    #    #ax.axhline(U_mean[j] - U_std[j]*3,color='k')

    colors = ['green','orange','red']
    for k,c,colr in zip(range(pts.shape[0]),np.linspace(0,1,pts.shape[0]),colors):
        #ax.plot(years,u_obs[k],'--',color=plt.cm.Spectral(c))
        #ax.plot(years,u_mod[k,:,0],'-',color=plt.cm.Spectral(c))
        #ax.plot(years,u_obs[k],'--',color=colr)
        ax.errorbar(years,u_obs[k],ls='',marker='o',markersize=3,yerr=100,color='black')
        ax.plot(years,u_mod[k,:,0],'-',color=colr)
        
        #for p in range(n_runs):
        #    ax.plot(years,u_mod[k,:,p],'-',color=plt.cm.cividis(c),alpha=0.1)

        lb = u_mod[k].min(axis=1)
        ub = u_mod[k].max(axis=1)
        #ax.fill_between(years,lb,ub,color=plt.cm.Spectral(c),alpha=0.2)
        ax.fill_between(years,lb,ub,color=colr,alpha=0.5)

    ax.set_xlim(1985,2018)
    ax.set_xlabel('Year')
    ax.set_ylabel('Speed (m a$^-1$)')

    fig.set_size_inches(4.5,4.5)
    fig.savefig(f'{results_dir}/plots/velocity_time.pdf',bbox_inches='tight')

if plot_velocity_appendix:
    years = [y for y in range(1985,2019)]

    with open(f'{data_dir}/velocity/velocity.p','rb') as fi:
        [v_avg,v_mask] = pickle.load(fi)

    velocities_0 = [None for i in range(len(years))]
    for i,y in enumerate(years):
        try:
            with open(f'{data_dir}/velocity/itslive_annual/velocity_{y}.p','rb') as fi:
                v = pickle.load(fi)
            velocities_0[i] = v
        except FileNotFoundError:
            pass

    v_avg = np.nanmean(velocities_0[:-2],axis=0)
    v_avg[np.isnan(v_avg)] = 0.0

    U_avg = df.Function(Q_cg2)
    U_avg.dat.data[:] = v_avg
    U_avg_mag = (U_avg.dat.data[:]**2).sum(axis=1)**0.5

    U_t = df.Function(Q_cg2)

    U_mod_dif = df.Function(Q_cg_3)
    U_obs_dif = df.Function(Q_cg_3)

    U_mods = []

    for i,y in enumerate(years):
        with open(f'{data_dir}/{prefix}/{ensemble_dir}/{experiment_type}/run_0/data_{y}.p','rb') as fi:
            data = pickle.load(fi)
        U_bar.dat.data[:] = data[-2]*len_scale*vel_scale
        U_def.dat.data[:] = data[-1]*len_scale*vel_scale
        U_s_2 = df.project(U_bar - 0.25*U_def,Q_cg2)

        U_mods.append(U_s_2.dat.data[:])

    U_mod_avg = np.mean(U_mods,axis=0)
    
    figs = [plt.subplots(nrows=4,ncols=4) for q in range(5)]
    for i,y in enumerate(years):
        axs = figs[i//8][1].ravel()
        U_mod_dif.dat.data[:] = np.linalg.norm(U_mods[i],axis=1) - np.linalg.norm(U_mod_avg,axis=1)
        U_obs_dif.dat.data[:] = np.linalg.norm(velocities_0[i],axis=1) - np.linalg.norm(v_avg,axis=1)
        
        ax1 = axs[2*(i%8)]
        ax2 = axs[2*(i%8) + 1]
        ax1.text(0.99,0.99,f'{y}',ha='right',va='top',transform=ax1.transAxes)
        #df.tripcolor(U_mod_dif,axes=ax1,vmin=-300,vmax=300,cmap=plt.cm.coolwarm)
        #df.tripcolor(U_obs_dif,axes=ax2,vmin=-300,vmax=300,cmap=plt.cm.coolwarm)
        df.tripcolor(U_mod_dif,axes=ax1,vmin=-200,vmax=200,cmap=plt.cm.coolwarm)
        df.tripcolor(U_obs_dif,axes=ax2,vmin=-200,vmax=200,cmap=plt.cm.coolwarm)

        for ax in axs.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            ax.plot(*boundary.T,'k-')

    for ax in figs[-1][1].ravel()[4:]:
        ax.remove()

    for j,(fig,axs) in enumerate(figs):
        fig.set_size_inches(9,10.51)
        fig.subplots_adjust(wspace=0,hspace=0)
        fig.savefig(f'{results_dir}/plots/velocity_all_{j}.pdf',bbox_inches='tight')


if plot_talk_graphic:
    
    fig,ax = plt.subplots(nrows=1,ncols=1)

    df.triplot(mesh,axes=ax,boundary_kw={'colors':'black'})
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_aspect('equal')


    fig.savefig('../../../../../Documents/Talks/centennial/figs/mesh.eps')

    fig,ax = plt.subplots(nrows=1,ncols=1)
    H = df.Function(Q_dg)
    with open(f'{data_dir}/{prefix}/{ensemble_dir}/projected_climate_calve/run_{0}/data_2023.p','rb') as fi:
        data = pickle.load(fi)
        H.dat.data[:] = data[0]
    
    df.tripcolor(H,axes=ax,cmap=cmasher.arctic,vmin=0,vmax=0.2)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_aspect('equal')

    fig.savefig('../../../../../Documents/Talks/centennial/figs/thk.eps')
  






        


    


   
    





