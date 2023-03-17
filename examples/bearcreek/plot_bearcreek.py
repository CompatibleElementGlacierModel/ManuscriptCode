import numpy as np
import firedrake as df
import matplotlib.pyplot as plt
import pickle

from pysheds.grid import Grid
from matplotlib.colors import ListedColormap,LogNorm,LightSource

blues = plt.cm.get_cmap('Blues', 256)
newcolors = blues(np.linspace(0, 1, 256))
newcolors[:20, 3] = 0.0
newcmp = ListedColormap(newcolors)


buildup_directory = '../../results/bearcreek_slide/'
dambreak_directory = '../../results/bearcreek_slide_dam_break_hacking/'

interpolant = pickle.load(open('../../data/bitterroot_mesh/interpolant.pkl','rb'))

grid = Grid.from_raster('../../data/output_SRTMGL1.tif')
dem = grid.read_raster('../../data/output_SRTMGL1.tif')

ls = LightSource(azdeg=315,altdeg=45)
hs = ls.hillshade(dem)

x = np.linspace(dem.extent[0],dem.extent[1],dem.shape[1])
y = np.linspace(dem.extent[2],dem.extent[3],dem.shape[0])
        
fig,axs = plt.subplots(nrows=3,ncols=2,sharex=True,sharey=True)

t = np.linspace(0,1000,400)
vol_buildup = []
with df.CheckpointFile(f"{buildup_directory}/functions.h5", 'r') as afile:
    mesh = afile.load_mesh("mesh")
    for i in range(400):
        H = afile.load_function(mesh, "H0",idx=i)
        U_s = afile.load_function(mesh, "U_s",idx=i)
        vol_buildup.append(df.assemble(H*df.dx))

    H_midpoint = afile.load_function(mesh,"H0",idx=100)
    U_midpoint = afile.load_function(mesh,"U_s",idx=100)

v_dg = df.VectorFunctionSpace(mesh,"DG",0)
Q_dg = df.FunctionSpace(mesh,"DG",0)
X = df.interpolate(mesh.coordinates,v_dg)
B = df.Function(Q_dg)
B.dat.data[:] = interpolant(X.dat.data_ro[:,0],X.dat.data_ro[:,1],grid=False)/1000.0

axs[0,0].imshow(hs,extent=grid.extent,cmap='gray')
axs[0,1].imshow(hs,extent=grid.extent,cmap='gray')
axs[1,0].imshow(hs,extent=grid.extent,cmap='gray')
axs[1,1].imshow(hs,extent=grid.extent,cmap='gray')
axs[2,0].imshow(hs,extent=grid.extent,cmap='gray')
axs[2,1].imshow(hs,extent=grid.extent,cmap='gray')
df.tripcolor(H_midpoint,axes=axs[0,0],cmap=newcmp,alpha=None,vmin=0,vmax=1.0,rasterized=True)
df.quiver(U_midpoint,axes=axs[0,1],norm=LogNorm(vmin=1e-1, vmax=3),scale=30,rasterized=True)
df.tripcolor(H,axes=axs[1,0],cmap=newcmp,alpha=None,vmin=0,vmax=1.0,rasterized=True)
df.quiver(U_s,axes=axs[1,1],norm=LogNorm(vmin=1e-1,vmax=3),scale=30,rasterized=True)


vol_dambreak = []
with df.CheckpointFile(f"{dambreak_directory}/functions.h5", 'r') as afile:
    mesh = afile.load_mesh("mesh")
    for i in range(621):
        H = afile.load_function(mesh, "H0",idx=i)
        U_s = afile.load_function(mesh, "U_s",idx=i)
        vol_dambreak.append(df.assemble(H*df.dx))
    U_s = afile.load_function(mesh,"U_s",idx=100)
    H = afile.load_function(mesh,"H0",idx=100)
df.tripcolor(H,axes=axs[2,0],cmap=newcmp,alpha=None,vmin=0,vmax=1,rasterized=True)
df.quiver(U_s,axes=axs[2,1],norm=LogNorm(vmin=1e-1,vmax=3),scale=30,rasterized=True)

fig.subplots_adjust(wspace=0,hspace=0)
axs[2,1].set_xlim(-114.4161,-114.175)
axs[2,1].set_ylim(46.3341,46.4620)
axs[2,0].tick_params(axis='x', labelrotation = 45)
axs[2,1].tick_params(axis='x', labelrotation = 45)

#axs[2,0].set_xlabel('Longitude')
#axs[2,0].set_ylabel('Latitude')



plt.setp(axs[0,0].get_yticklabels()[:],visible=False)
plt.setp(axs[1,0].get_yticklabels()[:],visible=False)
plt.setp(axs[2,1].get_xticklabels()[:],visible=False)


axs[1,0].plot([-114.346],[46.388],'r.')
axs[1,1].plot([-114.254],[46.385],'r.')
axs[1,0].annotate('750m',(-114.346,46.388),color='red')
axs[1,1].annotate('350m/a',(-114.254,46.385),color='red')


axs[0,0].text(0.95, 0.9, 'a', horizontalalignment='center', verticalalignment='center', transform=axs[0,0].transAxes,color='white',fontsize=20)
axs[0,1].text(0.95, 0.9, 'b', horizontalalignment='center', verticalalignment='center', transform=axs[0,1].transAxes,color='white',fontsize=20)
axs[1,0].text(0.95, 0.9, 'c', horizontalalignment='center', verticalalignment='center', transform=axs[1,0].transAxes,color='white',fontsize=20)
axs[1,1].text(0.95, 0.9, 'd', horizontalalignment='center', verticalalignment='center', transform=axs[1,1].transAxes,color='white',fontsize=20)
axs[2,0].text(0.95, 0.9, 'e', horizontalalignment='center', verticalalignment='center', transform=axs[2,0].transAxes,color='white',fontsize=20)
axs[2,1].text(0.95, 0.9, 'f', horizontalalignment='center', verticalalignment='center', transform=axs[2,1].transAxes,color='white',fontsize=20)
#-114.346,46.388 750
#-114.259,46.385 350

fig.set_size_inches(6.5,5.25)
fig.savefig('figures/bearcreek_fields.pdf',bbox_inches='tight',dpi=300)

V_scale = 108000**2*1000/1e9

t_1 = np.linspace(0,1000,400)
t_2 = np.cumsum(np.minimum(2.5*1.01**np.linspace(0,620,621),20)) + 1000.0

fig,axs = plt.subplots(ncols=2)
fig.subplots_adjust(wspace=0)
fig.set_size_inches(4,2)
axs[0].plot(t_1,np.array(vol_buildup)*V_scale,'k-')
ax_ = axs[1].twinx()
ax_.semilogx(t_2,np.array(vol_dambreak)*V_scale,'k--')
axs[0].set_xlim(0,1000)
ax_.set_xlim(1000,10000)
axs[0].set_xlabel('t (a)')
ax_.set_xlabel('t (a)')

#ax_.get_yaxis().get_major_formatter().set_useOffset(False)
axs[0].set_ylabel('Total Volume (m$^3$)')
ax_.set_ylabel('Total Volume (m$^3$)')

#plt.gcf().canvas.draw()

plt.setp(axs[1].get_xticklabels()[2], visible=False)
plt.setp(axs[1].get_yticklabels()[:], visible=False)
axs[1].set_yticks([])

fig.savefig('./figures/bearcreek_massconservation.pdf',bbox_inches='tight')


