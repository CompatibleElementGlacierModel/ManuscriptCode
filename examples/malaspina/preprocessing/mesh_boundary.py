# Read elevation raster
# ----------------------------
from pysheds.grid import Grid
from scipy.interpolate import RectBivariateSpline
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

import pygmsh
import gmsh
from shapely.geometry import Point, Polygon
from scipy.interpolate import interp1d
import fiona
import pyproj

import os

import pickle

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

# Resample a boundary and delete intersecting segments (doesn't exactly work all the time... note sure why)
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def resample_boundary(X,boundary_resolution=2000):
    r = np.hstack((0,np.cumsum(np.linalg.norm(X[1:] - X[:-1],axis=1))))
    x_spline = interp1d(r,X[:,0])
    y_spline = interp1d(r,X[:,1])
    n_segments = int(r.max()//boundary_resolution)
    sample_r = np.linspace(0,r.max(),n_segments)    
    X = np.vstack((x_spline(sample_r),y_spline(sample_r))).T
    
    converged = False
    while not converged:
        keep = np.ones(X.shape[0]).astype(bool)
        try:
            for i in range(len(X)-1):
                for j in range(i,len(X)-1):
                    if j>i+1:
                        if intersect(X[i],X[i+1],X[j],X[j+1]):
                            keep[i+1] = False
            X = X[keep]
        except:
            pass
        converged = keep.all()
    return X


grid = Grid.from_raster('../data/dem/cop30_3338.tif')
dem = grid.read_raster('../data/dem/cop30_3338.tif')
x = np.linspace(dem.extent[0],dem.extent[1],dem.shape[1])
y = np.linspace(dem.extent[2],dem.extent[3],dem.shape[0])
#spl = RectBivariateSpline(x,y,dem[::-1,:].T)
g = fiona.open('../data/boundary/margin_shapefile/ice_margin_ACT_v2.shp')
arr = [x for x in g]
X = np.array(arr[0]['geometry']['coordinates'])
transformer = pyproj.Transformer.from_crs("EPSG:26907","EPSG:3338",always_xy=True)
#X = np.array(transformer.transform(*X.T)).T
X = pickle.load(open('lower_margin.p','rb'))


fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_alpha(0)

from matplotlib.colors import LightSource
ls = LightSource(azdeg=315,altdeg=45)
plt.imshow(ls.hillshade(dem, vert_exag=1, dx=25, dy=25),extent=grid.extent,cmap='gray')
plt.imshow(dem, extent=grid.extent, cmap='terrain', zorder=1,vmin=0,vmax=5000,alpha=0.5)
plt.plot(X[:,0],X[:,1])
plt.colorbar(label='Elevation (m)')
plt.grid(zorder=0)
plt.title('Digital elevation map', size=14)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()


adaptive_size_from_boundary = False # Adapt mesh size away from boundary (slower)
boundary_resolution = 1000   # Resolution to mesh the boundary
max_size = 5000                    # Maximum element size
min_size = boundary_resolution     # Minimium element size
gradient = 0.2                     # speed at which element size can change
internal_boundary_area_threshold = 5e5   # area at which to delete internal boundaries
meshing_algorithm = 1 # 1=meshadapt, 6=frontal

output_directory = f'../meshes/mesh_{boundary_resolution}/'
#coords = pickle.load(open('../data/boundary/malaspina_watershed.p','rb'))
#coords -= np.array([ 740119.33782147, 1204883.47857356])
#coords /= 50000


coords = pickle.load(open('../data/boundary/malaspina_watershed_expanded.p','rb'))

boundary_external = resample_boundary(coords,boundary_resolution=boundary_resolution)

with open(output_directory+'mesh.geo','w') as fi:
    n_points = 1

    point_indices = [j for j in range(1,len(boundary_external)+1)]
    for i,point in zip(point_indices,boundary_external):
        fi.write(f'Point({i}) = {{{point[0]}, {point[1]}, 0, {boundary_resolution} }};\n')

    line_indices = point_indices
    for j,i in enumerate(line_indices):  
        fi.write(f'Line({i}) = {{{point_indices[j-1]}, {point_indices[j]}}};\n')

    curve_index = line_indices[-1] + 1
    internal_index = line_indices[-1] + 2
    front_index = line_indices[-1] + 3
    cline = f'Curve Loop({curve_index}) = {{'
    for l in line_indices[:-1]:
        cline+=f'{l}, '
    cline += f'{line_indices[-1]}}};\n'
    fi.write(cline)

    fi.write(f'Plane Surface(1) = {{{curve_index}}};\n')

    internal_lines = line_indices[:-78]
    front_lines = line_indices[-78:]

    
    pline = f'Physical Curve("Internal", {1000}) = {{'
    for l in internal_lines[:-1]:
        pline+=f'{l}, '
    pline += f'{internal_lines[-1]}}};\n'
    fi.write(pline)

    pline = f'Physical Curve("Front", {1001}) = {{'
    for l in front_lines[:-1]:
        pline+=f'{l}, '
    pline += f'{front_lines[-1]}}};\n'
    fi.write(pline)

    pline = f'Physical Surface("Ice", 1) = {{1}};'
    fi.write(pline)



"""
#plt.scatter(*boundary_external.T,c=range(len(boundary_external)))
# Build gmsh geometry
with pygmsh.geo.Geometry() as geom:

    points = []
    for point in boundary_external:
        p_in = geom.add_point(point,mesh_size=boundary_resolution)
        points.append(p_in)

    lines = [geom.add_line(points[i],points[i+1]) for i in range(-1,len(points)-1)]

    loop = geom.add_curve_loop(lines)
    surface = geom.add_plane_surface(loop)
    
    # Generate and write the mesh to vtu format
    geom.add_physical(loop.curves,'s')
    #geom.add_physical(lines[:-133],label='internal')
    geom.save_geometry(output_directory+'mesh.geo_unrolled')
    mesh = geom.generate_mesh(dim=2,algorithm=meshing_algorithm)
    mesh.write(output_directory+'mesh.msh',file_format='gmsh22')


"""




