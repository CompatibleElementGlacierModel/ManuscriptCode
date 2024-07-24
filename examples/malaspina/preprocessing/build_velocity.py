import netCDF4 as nc
import numpy as np
import scipy.interpolate as si
import firedrake as df
from pyproj import Transformer
import matplotlib.pyplot as plt
import pickle
import rasterio
from scipy.ndimage import gaussian_filter

velocity_data = nc.Dataset(f'../data/velocity/ALA_G0240_0000.nc')
cop_path = '../data/dem/cop30_3338.tif'
dem = rasterio.open(cop_path)
Z_dem = dem.read()[0]
Z_dem[Z_dem==Z_dem.max()]=0.0
x_dem_ = np.linspace(dem.bounds.left,dem.bounds.right,dem.width+1)
y_dem_ = np.linspace(dem.bounds.bottom,dem.bounds.top,dem.height+1)

# Get cell center coordinates
x_dem = 0.5*(x_dem_[:-1] + x_dem_[1:])
y_dem = 0.5*(y_dem_[:-1] + y_dem_[1:])

v_x = np.array(velocity_data['vx'][:])
v_y = np.array(velocity_data['vy'][:])
v_x[v_x<-5000] = np.nan
v_y[v_y<-5000] = np.nan
gx,gy = np.gradient(Z_dem,25)
gnorm = gaussian_filter((gx**2 + gy**2)**0.5,10)
grad_interpolant = si.RegularGridInterpolator((x_dem,y_dem),gnorm[::-1].T,method='nearest')


ice = np.array(velocity_data['ice'][:])

x_v = np.array(velocity_data.variables['x'][:])
y_v = np.array(velocity_data.variables['y'][:])


vx_interpolant = si.RegularGridInterpolator((x_v,y_v[::-1]),v_x[::-1].T,method='nearest')
vy_interpolant = si.RegularGridInterpolator((x_v,y_v[::-1]),v_y[::-1].T,method='nearest')
ice_interpolant = si.RegularGridInterpolator((x_v,y_v[::-1]),ice[::-1].T,method='nearest')


mesh_directory = '../meshes/mesh_2201/'
mesh = df.Mesh(f'{mesh_directory}/mesh.msh',name='mesh')

E = df.FiniteElement('CG',mesh.ufl_cell(),3)
Q = df.FunctionSpace(mesh,E)
V = df.VectorFunctionSpace(mesh,E)
X = np.array(df.interpolate(mesh.coordinates,V).dat.data_ro)

E_dg = df.FiniteElement('DG',mesh.ufl_cell(),0)
Q_dg = df.FunctionSpace(mesh,E_dg)
V_dg = df.VectorFunctionSpace(mesh,E_dg)
X_dg = np.array(df.interpolate(mesh.coordinates,V_dg).dat.data_ro)

from_epsg = 3338
to_epsg = 3413
transformer = Transformer.from_crs(from_epsg,to_epsg)

transformer_inv = Transformer.from_crs(to_epsg,from_epsg)
X_3413 = np.c_[transformer.transform(X[:,0],X[:,1])]
X_dg_3413 = np.c_[transformer.transform(X_dg[:,0],X_dg[:,1])]

u_x = vx_interpolant(X_3413)
u_y = vy_interpolant(X_3413)
u_xdg = vx_interpolant(X_dg_3413)
u_ydg = vy_interpolant(X_dg_3413)
U_dg = np.sqrt(u_xdg**2 + u_ydg**2)

ice = ice_interpolant(X_dg_3413)
steepness = np.rad2deg(np.arctan(grad_interpolant(X_dg)))
steepness_threshold = 15.

mask = (steepness<steepness_threshold)*ice
#mask = (U_dg>10)
mask_f = df.Function(Q_dg)
mask_f.dat.data[:] = mask

u = np.c_[u_x,u_y]

eps = 1e-3
X_plus = X_3413 + eps*u
X_minus = X_3413 - eps*u
X_bar_plus = np.c_[transformer_inv.transform(X_plus[:,0],X_plus[:,1])]
X_bar_minus = np.c_[transformer_inv.transform(X_minus[:,0],X_minus[:,1])]

u_bar = (X_bar_plus - X_bar_minus)/(2*eps)
u_norm = np.sqrt(u_bar[:,0]**2 + u_bar[:,1]**2)
fig,ax = plt.subplots()
ax.quiver(X[:,0],X[:,1],u_bar[:,0]/u_norm,u_bar[:,1]/u_norm,np.log10(np.sqrt(u_bar[:,0]**2 + u_bar[:,1]**2)),scale=100,clim=(0,3))
#plt.triplot(*mesh.coordinates.dat.data.T,mesh.cells())
#ax.scatter(*X_dg.T,c=mask)
fig.set_size_inches(4,4)
ax.set_title(f'avg')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('equal')
fig.savefig(f'./plots/velocities/vel_mean.png')


pickle.dump([u_bar,mask],open(f'{mesh_directory}/velocity/velocity.p','wb'))

years = [i for i in range(1985,2019)]
for y in years:
    velocity_data = nc.Dataset(f'../data/velocity/itslive_annual/ALA_G0240_{y}.nc')
    v_x = np.array(velocity_data['vx'][:])
    v_y = np.array(velocity_data['vy'][:])
    v_x[v_x<-5000] = np.nan
    v_y[v_y<-5000] = np.nan
    x_v = np.array(velocity_data.variables['x'][:])
    y_v = np.array(velocity_data.variables['y'][:])

    vx_interpolant = si.RegularGridInterpolator((x_v,y_v[::-1]),v_x[::-1].T,method='nearest')
    vy_interpolant = si.RegularGridInterpolator((x_v,y_v[::-1]),v_y[::-1].T,method='nearest')

    E = df.FiniteElement('CG',mesh.ufl_cell(),3)
    Q = df.FunctionSpace(mesh,E)
    V = df.VectorFunctionSpace(mesh,E)
    X = np.array(df.interpolate(mesh.coordinates,V).dat.data_ro)

    from_epsg = 3338
    to_epsg = 3413
    transformer = Transformer.from_crs(from_epsg,to_epsg)

    transformer_inv = Transformer.from_crs(to_epsg,from_epsg)
    X_3413 = np.c_[transformer.transform(X[:,0],X[:,1])]

    u_x = vx_interpolant(X_3413)
    u_y = vy_interpolant(X_3413)

    u = np.c_[u_x,u_y]

    eps = 1e-3
    X_plus = X_3413 + eps*u
    X_minus = X_3413 - eps*u
    X_bar_plus = np.c_[transformer_inv.transform(X_plus[:,0],X_plus[:,1])]
    X_bar_minus = np.c_[transformer_inv.transform(X_minus[:,0],X_minus[:,1])]

    u_bar = (X_bar_plus - X_bar_minus)/(2*eps)
    u_norm = np.sqrt(u_bar[:,0]**2 + u_bar[:,1]**2)
    fig,ax = plt.subplots()
    ax.quiver(X[:,0],X[:,1],u_bar[:,0]/u_norm,u_bar[:,1]/u_norm,np.log10(np.sqrt(u_bar[:,0]**2 + u_bar[:,1]**2)),scale=100,clim=(0,3))
#plt.triplot(*mesh.coordinates.dat.data.T,mesh.cells())
    fig.set_size_inches(4,4)
    ax.set_title(f'{y}')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('equal')
    fig.savefig(f'./plots/velocities/vel_{y}.png')
    pickle.dump(u_bar,open(f'{mesh_directory}/velocity/itslive_annual/velocity_{y}.p','wb'))

