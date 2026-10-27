"""
Video of a hertz_contact.py run, one frame per increment.

    left    the whole cell, deformed (x = X + d_g + u, every element drawn), block grey, cap blue, third medium shaded by
            its compression (white: J = 1, dark: squeezed), the void row dark
    middle  zoom on the contact, same colours, with the element edges
    right   top: surface pressure p(x) (= -P_yy in the block's top pixel row) with Hertz at the same P;
            bottom: force per unit length P vs the cap-top displacement

usage:  python hertz_video.py <run folder>      -> <run folder>/hertz.mp4 and hertz_last.png
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import netCDF4
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.collections import PolyCollection

folder = sys.argv[1]
n_env = None
tmp = os.path.join(tempfile.mkdtemp(), 'c.nc')            # a copy: the run may still be writing
shutil.copy(os.path.join(folder, 'contact_fracture.nc'), tmp)
ds = netCDF4.Dataset(tmp)
n = ds['phase'].shape[-1]
os.environ['NNN'] = str(n)                                 # hertz_geometry: the void row is one pixel
os.environ['GRADE'] = str(ds.getncattr('grade') if 'grade' in ds.ncattrs() else 0)   # graded grid: the same map
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import hertz_geometry as hz  # noqa: E402

E_STAR = 100.0 / (2 * (1 - 0.3 ** 2))
R = float(ds.getncattr('R'))
adapted = int(ds.getncattr('adapted_grid'))
fv = {k: np.asarray(ds[k][:]) for k in ('crown_u', 'crown_force', 'p_surface')}
nb = len(fv['crown_u'])
phase = np.asarray(ds['phase'][0]).reshape(n, n)
d_g = np.asarray(ds['grid_displacement'][0]).reshape(2, n, n)
X = np.stack(np.meshgrid(np.arange(n) / n, np.arange(n) / n, indexing='ij'))
BASE = np.array([colors.to_rgba(c) for c in ('#ffffff', '#b8b8bd', '#3d7fd0', '#4a4a4a')])
SQUEEZE = colors.LinearSegmentedColormap.from_list('squeeze', ['#ffffff', '#fdd49e', '#d7301f', '#7f0000'])
i, j = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
# physical pixel centres (mean of the corner nodes; graded / adapted grids are not uniform), x of the block's top row
_ext = np.pad(X + d_g, ((0, 0), (0, 1), (0, 1)), mode='wrap')
_ext[0, -1, :] += 1.0
_ext[1, :, -1] += 1.0
centre = 0.25 * (_ext[:, :-1, :-1] + _ext[:, 1:, :-1] + _ext[:, :-1, 1:] + _ext[:, 1:, 1:])
x_axis = centre[0, :, np.max(np.nonzero(((phase > 0) & (centre[1] < hz.BLOCK_TOP + 0.5 * hz.GAP)).any(axis=0))[0])] \
    - hz.CENTRE_X
lab = np.where(phase > 0, np.where(centre[1] < hz.BLOCK_TOP + 0.5 * hz.GAP, 1, 2), 0)
lab = np.where(hz.frame(centre) & (phase == 0), 3, lab).ravel()
frames = os.path.join(folder, 'frames_hertz')
shutil.rmtree(frames, ignore_errors=True)
os.makedirs(frames)


def polygons(f):
    x = X + d_g + np.asarray(ds['u_fluc'][f]).reshape(2, n, n)
    ext = np.pad(x, ((0, 0), (0, 1), (0, 1)), mode='wrap')
    ext[0, -1, :] += 1.0
    ext[1, :, -1] += 1.0
    quad = np.stack([ext[:, i + a, j + b] for a, b in corners], axis=-1)
    return quad.reshape(2, -1, 4).transpose(1, 2, 0)


def faces(f):
    face = BASE[lab].copy()
    J = np.asarray(ds['detF'][f]).reshape(n, n).ravel()
    medium = lab == 0
    face[medium] = SQUEEZE(np.clip((1.0 - J[medium]) / 0.9, 0, 1))
    return face


zoom = (hz.CENTRE_X - 0.15, hz.CENTRE_X + 0.15, hz.BLOCK_TOP - 0.06, hz.BLOCK_TOP + 0.07)
fig = plt.figure(figsize=(17, 7.5))
for f in range(nb):
    fig.clf()
    polys, face = polygons(f), faces(f)
    for k, (box, lw) in enumerate((((0, 1, 0, 1), 0.0), (zoom, 0.3))):
        ax = fig.add_axes((0.01 + 0.31 * k, 0.05, 0.3, 0.82))
        ax.add_collection(PolyCollection(polys, facecolors=face, edgecolors=(0, 0, 0, 0.35) if lw else 'none',
                                         linewidths=lw))
        ax.set_xlim(box[0], box[1])
        ax.set_ylim(box[2], box[3])
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        if k == 0:
            ax.add_patch(plt.Rectangle((zoom[0], zoom[2]), zoom[1] - zoom[0], zoom[3] - zoom[2], fill=False,
                                       ec='#b2182b', lw=1.2, ls='--'))
            ax.set_title(f'whole cell, cap top moved down {fv["crown_u"][f]:.4f} (gap {hz.GAP:g})', fontsize=10)
        else:
            ax.set_title('zoom on the contact: third medium shaded by its compression 1 - J', fontsize=10)
    P = fv['crown_force'][f]
    ax = fig.add_axes((0.66, 0.55, 0.32, 0.36))
    ax.plot(x_axis, fv['p_surface'][f], 'o-', color='#c4441a' if adapted else '#3d7fd0', ms=3, lw=1.5,
            label=f'{"adapted" if adapted else "regular"} grid')
    if P > 1e-3:
        a = np.sqrt(4 * P * R / (np.pi * E_STAR))
        xs = np.linspace(-a, a, 200)
        ax.plot(xs, 2 * P / (np.pi * a ** 2) * np.sqrt(a ** 2 - xs ** 2), color='k', lw=1.2, label='Hertz, same P')
    ax.set(xlim=(-0.2, 0.2), ylim=(0, 1.1 * max(fv['p_surface'].max(), 1e-6)), xlabel='x (relative to the cap axis)',
           ylabel='surface pressure p')
    ax.grid(True, color='0.9')
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    ax = fig.add_axes((0.66, 0.08, 0.32, 0.33))
    ax.plot(fv['crown_u'], fv['crown_force'], color='0.75', lw=1)
    ax.plot(fv['crown_u'][:f + 1], fv['crown_force'][:f + 1], color='#b2182b', lw=2)
    ax.plot(fv['crown_u'][f], P, 'o', color='#b2182b', ms=8, mfc='w', mew=2)
    ax.set(xlabel='cap-top displacement', ylabel='force per unit length P')
    ax.grid(True, color='0.9')
    fig.suptitle(f'2D Hertz test ({n}$^2$, {"adapted" if adapted else "regular"} grid): cylinder cap R = {R:g} pressed '
                 f'onto a block,  P = {P:.3g}', fontsize=12)
    fig.savefig(os.path.join(frames, f'g{f:04d}.png'), dpi=100)
import imageio_ffmpeg  # noqa: E402
out = os.path.join(folder, 'hertz.mp4')
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-framerate', '6', '-i',
                os.path.join(frames, 'g%04d.png'), '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p', out],
               check=True)
shutil.copy(os.path.join(frames, f'g{nb - 1:04d}.png'), out.replace('.mp4', '_last.png'))
print(f'wrote {out} ({nb} frames)')
