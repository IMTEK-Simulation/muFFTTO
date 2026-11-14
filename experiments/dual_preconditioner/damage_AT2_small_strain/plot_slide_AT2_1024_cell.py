"""
Slide figure: final cracked state of the 1024^2 polydisperse-pores run (l = 1/32, dual), drawn as the last frame of
make_AT2_video.py (deformed cell at true scale, exact pixels, pores white, damage colour map fading to transparent from
d = 0.6 to 0.95) but with the serif LaTeX fonts of plot_slide_AT2_1024(_study).py; vertical colour bar on the left.
    python plot_slide_AT2_1024_cell.py          ->  figures/slide_AT2_1024_cell.{png,pdf} (+ talk figures/)
    VIDEO=1 python plot_slide_AT2_1024_cell.py  ->  the same for every load step (run + restart segment joined, as in
        make_AT2_video.py), last frame held 2 s  ->  figures/slide_AT2_1024_cell.mp4, _initial.png, _final.png
        (+ talk figures/videos/)
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import imageio_ffmpeg
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from muFFTTO import microstructure_library

plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
HERE = os.path.dirname(os.path.realpath(__file__))
TALK = os.path.expanduser('~/Projects/Talks/Slides-Green-Jacobi-topology-optimization-NANTES/figures')
TAG = 'AT2_N1024_dual_polydisperse_pores_l0.03125'
DAMAGE_CMAP = LinearSegmentedColormap.from_list('damage_only', ['#e9e8e4', '#f4b183', '#d9622b', '#8c2a0c', '#2b0a02'])
FADE_START, FADE_END = 0.6, 0.95                        # as make_AT2_video.py
VIDEO = os.environ.get('VIDEO', '0') == '1'
FPS = 10                                                # as make_AT2_video.py

first = netCDF4.Dataset(os.path.join(HERE, 'exp_data', f'{TAG}.nc'))
second = netCDF4.Dataset(os.path.join(HERE, 'exp_data', f'{TAG}_restart.nc'))
# frames: the load steps of the first part before the first step of the restart, then the restart (make_AT2_video.JoinedRun)
n_first = int(np.sum(np.asarray(first['step'][:]) < np.asarray(second['step'][:]).min()))
FRAMES = [(first, k) for k in range(n_first)] + [(second, k) for k in range(second['step'].shape[0])]
micro = getattr(second, 'microstructure', None) or getattr(first, 'microstructure', 'polydisperse_pores')
N = second['damage'].shape[-1]
X = np.stack(np.meshgrid(np.arange(N + 1) / N, np.arange(N + 1) / N, indexing='ij'))
pore = microstructure_library.get_geometry(nb_voxels=(N, N), microstructure_name=micro, coordinates=X[:, :-1, :-1]) == 0


def frame(i):
    """deformed corners and RGBA pixel colours of frame i"""
    ds, k = FRAMES[i]
    nodal = np.asarray(ds['damage'][k, 0], float)
    u = np.asarray(ds['u_fluctuation'][k, :, 0], float)
    damage = 0.25 * (nodal + np.roll(nodal, -1, 0) + np.roll(nodal, -1, 1) + np.roll(np.roll(nodal, -1, 0), -1, 1))
    x = X + np.pad(u, ((0, 0), (0, 1), (0, 1)), mode='wrap')
    x[0] += float(ds['load'][k]) * X[0]                   # loaded in xx: deformed corners x = X + E X + u
    rgba = DAMAGE_CMAP(np.clip(damage, 0, 1))
    rgba[..., 3] = np.clip((FADE_END - damage) / (FADE_END - FADE_START), 0, 1)
    rgba[pore, 3] = 0.0
    return x, rgba


plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 13})
fig, ax = plt.subplots(figsize=(4.6, 3.9), layout='constrained')
ax.set_aspect('equal'), ax.axis('off')
x_last, _ = frame(len(FRAMES) - 1)
x_first, _ = frame(0)
pad = 0.005
ax.set_xlim(min(x_first[0].min(), x_last[0].min()) - pad, max(x_first[0].max(), x_last[0].max()) + pad)
ax.set_ylim(min(x_first[1].min(), x_last[1].min()) - pad, max(x_first[1].max(), x_last[1].max()) + pad)
levels = np.linspace(0, 1, 256)                           # colour bar as seen on white: faded part blended
alpha = np.clip((FADE_END - levels) / (FADE_END - FADE_START), 0, 1)[:, None]
colours = DAMAGE_CMAP(levels)[:, :3] * alpha + (1 - alpha)
cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=ListedColormap(colours)), ax=ax,
                  location='left', fraction=0.05, pad=0.02, aspect=22, shrink=0.75)
cb.outline.set_visible(False)
cb.ax.tick_params(length=0)
cb.ax.set_title('damage $d$', fontsize=13, pad=14)


def draw(i):
    for m in list(ax.collections):
        m.remove()
    x, rgba = frame(i)
    mesh = ax.pcolormesh(x[0], x[1], np.zeros((N, N)), shading='flat', rasterized=True, edgecolors='none', linewidth=0,
                         antialiased=False)
    mesh.set_array(None)
    mesh.set_facecolor(rgba.reshape(-1, 4))


out = os.path.join(HERE, 'figures', 'slide_AT2_1024_cell')
if VIDEO:
    # 4.6 x 3.9 in at 400 dpi = 1840 x 1560 px: > 1.5 video pixels per simulation pixel
    writer = animation.FFMpegWriter(fps=FPS, bitrate=12000, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
    with writer.saving(fig, out + '.mp4', dpi=400):
        for i in list(range(len(FRAMES))) + [len(FRAMES) - 1] * (2 * FPS):
            if i == 0 or i != getattr(draw, 'last', None):
                draw(i)
                draw.last = i
            writer.grab_frame(facecolor='white')
            if i == 0:
                fig.savefig(out + '_initial.png', dpi=400, facecolor='white')
    fig.savefig(out + '_final.png', dpi=400, facecolor='white')
    for f in ('.mp4', '_initial.png', '_final.png'):
        shutil.copy(out + f, os.path.join(TALK, 'videos'))
    print('wrote', out + '.mp4', len(FRAMES), 'load steps')
else:
    draw(len(FRAMES) - 1)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}.{ext}', dpi=900)          # > 2 output pixels per simulation pixel: exact pixels
        shutil.copy(f'{out}.{ext}', TALK)
    print('wrote', out)
