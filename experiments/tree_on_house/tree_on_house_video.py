"""
Video of a tree_on_house.py run: every pixel of the solid drawn as its deformed quadrilateral (x = X + u), coloured
by its part (ground, walls, roof, pole) and blended into the damage colours from d = 0.2 on; broken pixels (d > 0.9)
hidden, the third medium not drawn. Right: horizontal crown force vs crown displacement, and the max damage.

usage:  python tree_on_house_video.py <run folder>      -> <run folder>/tree_on_house.mp4 and tree_on_house_last.png
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import netCDF4
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.collections import PolyCollection

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import tree_house_geometry as tree  # noqa: E402

folder = sys.argv[1]
ds = netCDF4.Dataset(os.path.join(folder, 'contact_fracture.nc'))
n = ds['phase'].shape[-1]
fv = {k: np.asarray(ds[k][:]) for k in ('crown_u', 'crown_force', 'damage_max', 'nb_hessp')}
nb = len(fv['crown_u'])
X = np.stack(np.meshgrid(np.arange(n) / n, np.arange(n) / n, indexing='ij'))
lab = tree.labels(X + 0.5 / n).ravel()
PART = np.array([(1, 1, 1, 0), colors.to_rgba('#7a6a55'), colors.to_rgba('#b8b8bd'), colors.to_rgba('#c98a8a'),
                 colors.to_rgba('#3d7a2f')])
DAMAGE = colors.LinearSegmentedColormap.from_list('damage', [(0, '#e08a3c'), (0.5, '#b2182b'), (1, '#3d0008')])
i, j = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
solid = lab > 0
frames = os.path.join(folder, 'frames_tree')
shutil.rmtree(frames, ignore_errors=True)
os.makedirs(frames)


def polygons(f):
    x = X + np.asarray(ds['u_fluc'][f]).reshape(2, n, n)
    ext = np.pad(x, ((0, 0), (0, 1), (0, 1)), mode='wrap')
    ext[:, -1, :] += np.array([1.0, 0.0])[:, None]       # periodic images of the first column / row
    ext[:, :, -1] += np.array([0.0, 1.0])[:, None]
    quad = np.stack([ext[:, i + a, j + b] for a, b in corners], axis=-1)
    return quad.reshape(2, -1, 4).transpose(1, 2, 0)


fig = plt.figure(figsize=(13, 7.2))
for f in range(nb):
    fig.clf()
    d_n = np.asarray(ds['damage'][f]).reshape(n, n)
    d = 0.25 * (d_n + np.roll(d_n, -1, 0) + np.roll(d_n, -1, 1) + np.roll(d_n, (-1, -1), (0, 1))).ravel()
    face = PART[lab].copy()
    w = np.clip((d - 0.2) / 0.3, 0, 1)[:, None]
    face = np.where(solid[:, None], (1 - w) * face + w * DAMAGE(np.clip(d, 0, 1)), face)
    show = solid & (d <= 0.9)
    ax = fig.add_axes((0.01, 0.03, 0.58, 0.86))
    ax.add_collection(PolyCollection(polygons(f)[show], facecolors=face[show], edgecolors=(0, 0, 0, 0.15),
                                     linewidths=0.3))
    ax.plot([tree.FRAME, 1 - tree.FRAME], [tree.GROUND[0] - 0.006] * 2, color='k', lw=4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.85)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'crown displacement {fv["crown_u"][f]:.3f},  max damage {fv["damage_max"][f]:.2f}', fontsize=11)
    for k, (key, label) in enumerate((('crown_force', 'horizontal crown force'), ('damage_max', 'max damage'))):
        ax = fig.add_axes((0.67, 0.58 - 0.47 * k, 0.3, 0.34))
        ax.plot(fv['crown_u'], fv[key], color='0.75', lw=1)
        ax.plot(fv['crown_u'][:f + 1], fv[key][:f + 1], color='#b2182b', lw=2)
        ax.plot(fv['crown_u'][f], fv[key][f], 'o', color='#b2182b', ms=8, mfc='w', mew=2)
        ax.set_xlabel('crown displacement')
        ax.set_ylabel(label)
        ax.grid(True, color='0.9')
    fig.suptitle(f'pole pushed onto a house ({n}$^2$, regular grid): finite strain, third-medium contact, AT2 fracture',
                 fontsize=12)
    fig.savefig(os.path.join(frames, f'g{f:04d}.png'), dpi=110)
import imageio_ffmpeg  # noqa: E402
out = os.path.join(folder, 'tree_on_house.mp4')
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-framerate', '10', '-i',
                os.path.join(frames, 'g%04d.png'), '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p', out],
               check=True)
shutil.copy(os.path.join(frames, f'g{nb - 1:04d}.png'), out.replace('.mp4', '_last.png'))
print(f'wrote {out} ({nb} frames)')
