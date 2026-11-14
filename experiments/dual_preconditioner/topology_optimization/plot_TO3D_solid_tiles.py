"""Opaque solid (rho >= 0.5) of a 3D TO design: one periodic unit cell (closed at the faces) and a 2 x 2 x 2 tiling.
    <pyvista python> plot_TO3D_solid_tiles.py <final.nc> <out.png> [title]
"""
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pyvista as pv
from skimage import measure

fn, out = sys.argv[1], sys.argv[2]
title = sys.argv[3] if len(sys.argv) > 3 else ''
with netCDF4.Dataset(fn) as d:
    v = d['phase_field_solution'] if 'phase_field_solution' in d.variables else d['phase_field_iterate']
    rho = np.asarray(v[-1, 0], dtype=float)                 # a _history.nc gives its last iterate
N = rho.shape[0]


def solid(field, n):
    v, f, nrm, _ = measure.marching_cubes(np.pad(field, 1, constant_values=0.0), 0.5)
    m = pv.PolyData((v - 0.5) / n, np.hstack([np.full((len(f), 1), 3), f]).astype(np.int64))
    m.point_data['Normals'] = -nrm
    return m


def shot(mesh, extent, size=(1500, 1500)):
    p = pv.Plotter(off_screen=True, window_size=size, lighting='three lights')
    p.set_background('white')
    p.add_mesh(mesh, color='#6f8fb8', smooth_shading=True, specular=0.35, specular_power=18)
    p.add_mesh(pv.Box((0, extent, 0, extent, 0, extent)).extract_all_edges(), color='#8c8c8c', line_width=2)
    p.enable_anti_aliasing('ssaa')
    c = extent / 2
    a, e, dist = np.radians(-60), np.radians(24), 3.5 * extent
    p.camera.focal_point = (c, c, c)
    p.camera.position = (c + dist * np.cos(e) * np.cos(a), c + dist * np.cos(e) * np.sin(a), c + dist * np.sin(e))
    p.camera.up = (0, 0, 1)
    p.camera.view_angle = 30.0
    img = p.screenshot(return_img=True)
    p.close()
    return img


one = shot(solid(rho, N), 1.0)
tiled = np.tile(rho, (2, 2, 2))
two = shot(solid(tiled, N), 2.0)          # tiled grid: 2N voxels over 2 cells, voxel size 1/N
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 15})
fig, axs = plt.subplots(1, 2, figsize=(14, 7.4), layout='constrained')
for ax, img, t in zip(axs, (one, two), ('one periodic unit cell', r'$2 \times 2 \times 2$ cells')):
    ax.imshow(img), ax.set_axis_off(), ax.set_title(t)
fig.suptitle(title, fontsize=17)
fig.savefig(out, dpi=120)
print('wrote', out)
