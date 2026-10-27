"""
Video of a bubble_inflation.py run: the deformed adapted grid (every element drawn as its deformed quadrilateral,
x = F_bar (X + d_g) + u~), tiled TILES x TILES periodically; solid grey, the pressurised bubble red, the other
bubbles white. Bottom: bubble area and the cell strains H_bar vs the pressure.

usage:  python bubble_inflation_video.py <run folder> [--tiles 2]   -> <run folder>/inflation.mp4 and inflation_last.png
"""
import argparse
import os
import shutil
import subprocess

import numpy as np
import netCDF4
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

parser = argparse.ArgumentParser()
parser.add_argument('folder')
parser.add_argument('--tiles', type=int, default=None, help='periodic copies per direction (default: 1 for the free plate, 2 for the cell)')
args = parser.parse_args()

ds = netCDF4.Dataset(os.path.join(args.folder, 'contact_fracture.nc'))
n = ds['phase'].shape[-1]
free_plate = bool(getattr(ds, 'free_plate', 0))
BUBBLE = np.asarray(getattr(ds, 'bubble_centre', (0.25, 0.25)))
RADIUS = float(getattr(ds, 'bubble_radius', 0.2))
args.tiles = args.tiles or (1 if free_plate else 2)
clamp_left = bool(getattr(ds, 'clamp_left', 0))
DAMAGE_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    'damage', [(0, '#8c8c94'), (0.3, '#8c8c94'), (0.6, '#e08a3c'), (0.8, '#b2182b'), (1, '#3d0008')])
grid_kind = 'adapted grid' if int(getattr(ds, 'adapted_grid', 1)) else 'regular grid'
fv = {k: np.asarray(ds[k][:]) for k in ('pressure', 'bubble_area', 'H_bar', 'damage_max', 'converged')}
with_damage = fv['damage_max'].max() > 0
nb = len(fv['pressure'])
phase = np.asarray(ds['phase'][0]).reshape(n, n)
d_g = np.asarray(ds['grid_displacement'][0]).reshape(2, n, n)
X = np.stack(np.meshgrid(np.arange(n) / n, np.arange(n) / n, indexing='ij'))
centre = X + 0.5 / n
dist = np.sqrt(sum(((centre[k] - BUBBLE[k] + 0.5) % 1.0 - 0.5) ** 2 for k in range(2)))
balloon = str(getattr(ds, 'geometry', '')).startswith('balloon')
bubble = (phase == 0) & (dist < (1.25 if balloon else 1.15) * RADIUS)
colour = np.where(phase > 0, 0, np.where(bubble, 1, 2)).ravel()
FACE = np.array([(0.55, 0.55, 0.58, 1.0), (0.85, 0.2, 0.15, 0.85), (1.0, 1.0, 1.0, 0.0)])
i, j = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
frames = os.path.join(args.folder, 'frames')
shutil.rmtree(frames, ignore_errors=True)
os.makedirs(frames)


def nodes(f):
    """deformed node positions on the (n+1) x (n+1) extended grid of one cell"""
    F_bar = np.eye(2) + fv['H_bar'][f]
    u = np.asarray(ds['u_fluc'][f]).reshape(2, n, n)
    x = np.einsum('ij,jxy->ixy', F_bar, X + d_g) + u
    ext = np.zeros((2, n + 1, n + 1))
    for a in (0, 1):
        for b in (0, 1):
            ii, jj = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing='ij')
            ext[:, ii, jj] = x[:, ii % n, jj % n] + (F_bar @ np.array([ii // n, jj // n]).reshape(2, -1)).reshape(2, n + 1, n + 1)
    return ext, F_bar


def face_colours(f):
    """solid coloured by the element damage (mean of its 4 nodes), broken elements (d > 0.9) hidden"""
    face = FACE[colour].copy()
    if with_damage:
        d_n = np.asarray(ds['damage'][f]).reshape(n, n)
        d = 0.25 * (d_n + np.roll(d_n, -1, 0) + np.roll(d_n, -1, 1) + np.roll(d_n, (-1, -1), (0, 1))).ravel()
        solid = colour == 0
        face[solid] = DAMAGE_CMAP(np.clip(d[solid], 0, 1))
        face[solid & (d > 0.9)] = (1, 1, 1, 0)
    return face


def all_polygons(f):
    ext, F_bar = nodes(f)
    quad = np.stack([ext[:, i + a, j + b] for a, b in corners], axis=-1)          # [2, n, n, 4]
    quad = quad.reshape(2, -1, 4).transpose(1, 2, 0)                              # [n*n, 4, 2]
    polys, cols = [], []
    for ta in range(args.tiles):
        for tb in range(args.tiles):
            polys.append(quad + F_bar @ np.array([ta, tb]))
            cols.append(colour)
    return np.concatenate(polys), np.concatenate(cols), F_bar


_p = np.concatenate([all_polygons(f)[0].reshape(-1, 2) for f in (0, nb - 1)])
lower, upper = _p.min(axis=0) - 0.03, _p.max(axis=0) + 0.03     # the tiled cells of the first and last frame
fig = plt.figure(figsize=(12, 7.2))
for f in range(nb):
    fig.clf()
    polys, cols, F_bar = all_polygons(f)
    ax = fig.add_axes((0.01, 0.02, 0.55, 0.86))
    edges = np.where((cols == 2)[:, None], (0, 0, 0, 0), (0, 0, 0, 0.25))   # no grid lines in the empty bubbles
    pc = PolyCollection(polys, facecolors=np.tile(face_colours(f), (args.tiles ** 2, 1)), edgecolors=edges, linewidths=0.3)
    ax.add_collection(pc)
    if clamp_left:
        ax.plot([1.0 / n - 0.008] * 2, [1.0 / n, 1 - 1.0 / n], color='k', lw=4)
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'p = {fv["pressure"][f]:.1f}  (E = 100),  cell strain '
                 f'[{fv["H_bar"][f][0, 0]:+.3f}, {fv["H_bar"][f][1, 1]:+.3f}, {fv["H_bar"][f][0, 1]:+.3f}]', fontsize=11)
    second = (('damage_max', 'max damage', fv['damage_max']) if free_plate else
              ('H_bar', 'cell strain $\\bar H_{xx}$ = $\\bar H_{yy}$', fv['H_bar'][:, 0, 0]))
    for k, (key, label, data) in enumerate((('bubble_area', 'area of the inflated bubble', fv['bubble_area']), second)):
        ax = fig.add_axes((0.64, 0.58 - 0.47 * k, 0.33, 0.34))
        ax.plot(fv['pressure'], data, color='0.75', lw=1)
        ax.plot(fv['pressure'][:f + 1], data[:f + 1], color='#b2182b', lw=2)
        ax.plot(fv['pressure'][f], data[f], 'o', color='#b2182b', ms=8, mfc='w', mew=2)
        ax.set_xlabel('pressure p')
        ax.set_ylabel(label)
        ax.grid(True, color='0.9')
    fig.suptitle((f'balloon in a round cavity ({n}$^2$, {grid_kind}), third-medium contact, ' if balloon else
                  f'plate with 2 x 2 bubbles ({n}$^2$, {grid_kind}), one bubble inflated, ')
                 + ('left edge clamped' if clamp_left else 'free edges') + (', AT2 fracture' if with_damage else '')
                 if free_plate else f'bubble_array cell ({n}$^2$, {grid_kind}), one bubble inflated, stress-free '
                 f'periodic cell ({args.tiles} x {args.tiles} cells shown)', fontsize=12)
    fig.savefig(os.path.join(frames, f'g{f:04d}.png'), dpi=110)
import imageio_ffmpeg
out = os.path.join(args.folder, 'inflation.mp4')
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-framerate', '10', '-i',
                os.path.join(frames, 'g%04d.png'), '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p', out],
               check=True)
shutil.copy(os.path.join(frames, f'g{nb - 1:04d}.png'), out.replace('.mp4', '_last.png'))
print(f'wrote {out} ({nb} frames)')
