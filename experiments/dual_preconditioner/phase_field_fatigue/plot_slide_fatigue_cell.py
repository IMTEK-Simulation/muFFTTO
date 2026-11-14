"""Slide video (left half of the fatigue slide): damage of the 2D 1024^2 NEMO fatigue run (Carrara 2020 AT2 fatigue,
glass beads, pre-crack, dual run -- the solution is the same for all preconditioners), with a cycle counter; serif
LaTeX fonts as the right-hand figure (plot_slide_fatigue_study.py) and the AT2 slides (plot_slide_AT2_1024_cell.py).

    python plot_slide_fatigue_cell.py   -> figures/slide_fatigue_cell.mp4, _initial.png, _final.png (+ talk figures/videos/)

Damage on its exact pixels: each simulation pixel = 2 x 2 video pixels (2048 x 2048 image in a 2760 x 2344 frame at
600 dpi, i.e. 4.6 x 3.9 in like the right-hand figure); glass beads tinted as in make_FAT_slide_video.py; snapshots are
stored every 5th cycle + the final state, the field shown is the latest stored snapshot <= the current cycle.
"""
import os
import shutil
import sys

import imageio_ffmpeg
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
TALK = os.path.expanduser('~/Projects/Talks/Slides-Green-Jacobi-topology-optimization-NANTES/figures/videos')
N = 1024
TAG = '_incl11.6667_l0.015625_k0.0001_nemo2D_precrack_asymptotic_amp0.002_cyc1500'
W_PX, H_PX, DPI = 2760, 2344, 600                       # 4.6 x 3.9 in; multiples of 8 for the encoder
S = 2 * N                                                # exact pixels: magnification 2
FPS, HOLD = 10, 25                                       # as make_FAT_slide_video.py
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 13})

base = os.path.join(HERE, 'exp_data', f'FAT_N{N}_dual{TAG}')
nc = netCDF4.Dataset(base + '.nc')
snap_steps = list(np.asarray(nc['step'][:]).astype(int))
snaps = [np.asarray(nc['damage'][k, 0]) for k in range(len(snap_steps))]
fin = np.load(base + '_damage.npz')
if int(fin['steps'][-1]) > snap_steps[-1]:
    snap_steps.append(int(fin['steps'][-1]))
    snaps.append(np.asarray(fin['damage'][-1]))
bead = fin['phase'] < 0.5                                # phase = 1 matrix, 0 glass bead
snap_steps = np.array(snap_steps)
ncyc = len(np.genfromtxt(os.path.join(HERE, 'exp_data', f'FAT_N{N}_dual{TAG}_cycles.csv'), delimiter=',', names=True))

cmap = plt.get_cmap('inferno')
bead_rgb = np.array([0.55, 0.70, 0.88])


def rgb(dmg):
    img = cmap(np.clip(dmg, 0, 1))[..., :3]
    img[bead] = 0.45 * img[bead] + 0.55 * bead_rgb
    return np.transpose(img, (1, 0, 2))                  # x horizontal, y vertical (origin lower)


fig = plt.figure(figsize=(W_PX / DPI, H_PX / DPI), dpi=DPI)
fig.patch.set_facecolor('white')
px = lambda x0, y0, w, h: [x0 / W_PX, y0 / H_PX, w / W_PX, h / H_PX]
X0, Y0 = W_PX - S - 24, (H_PX - S) // 2
ax_d = fig.add_axes(px(X0, Y0, S, S))
im = ax_d.imshow(rgb(snaps[0]), origin='lower', interpolation='nearest', resample=False, extent=(0, 1, 0, 1))
ax_d.set_xticks([]), ax_d.set_yticks([])
ax_cb = fig.add_axes(px(X0 - 150, Y0 + S * 0.15, 60, S * 0.7))
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)), cax=ax_cb)
cb.set_ticks([0, 0.5, 1]), cb.set_ticklabels(['0', '0.5', '1'])
cb.outline.set_visible(False)
ax_cb.yaxis.set_ticks_position('left'), ax_cb.tick_params(length=0)
ax_cb.yaxis.set_label_position('left'), cb.set_label('damage $d$', labelpad=6)   # vertical, left of the ticks
ax_d.text(0.02, 0.98, f'${N}^2$, $\\ell/h = {N // 64}$', transform=ax_d.transAxes, ha='left', va='top', color='white')
head = ax_d.text(0.02, 0.02, '', transform=ax_d.transAxes, ha='left', va='bottom', color='white', fontsize=14)


def frame(n):
    """state after cycle n (1-based)"""
    k = int(np.searchsorted(snap_steps, n - 1, side='right') - 1)
    im.set_data(rgb(snaps[k]))
    head.set_text(f'cycle {n}')


def grab():
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


out = os.path.join(HERE, 'figures', 'slide_fatigue_cell')
frames_list = list(range(1, ncyc + 1)) + [ncyc] * HOLD
w = imageio_ffmpeg.write_frames(out + '.mp4', (W_PX, H_PX), fps=FPS, quality=9, codec='libx264', pix_fmt_out='yuv420p',
                                macro_block_size=8)
w.send(None)
for i, n in enumerate(frames_list):
    if i == 0 or n != frames_list[i - 1]:
        frame(n)
        buf = grab()
        if i == 0:
            plt.imsave(out + '_initial.png', buf)
    w.send(buf)
w.close()
plt.imsave(out + '_final.png', buf)
if os.path.isdir(TALK):
    for f in ('.mp4', '_initial.png', '_final.png'):
        shutil.copy(out + f, TALK)
print('saved', out + '.mp4', len(frames_list), 'frames,', ncyc, 'cycles')
