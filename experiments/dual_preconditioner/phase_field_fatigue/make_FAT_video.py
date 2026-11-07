"""MP4 of the damage evolution of one fatigue run, one frame per cycle (see README.md).

    python make_FAT_video.py [N] [PREC] [TAG] [CG_TAG]
        N, PREC     the run (default 128 dual)
        TAG         run tag of the .nc with every cycle (default: the Carrara glass-bead runs, NC_EVERY=1)
        CG_TAG      run tag of the G / GJ / dual runs for the CG panel (default: TAG)
    -> figures/FAT_video_N<N>_<PREC><TAG>.mp4

left: damage on the exact pixels (nodal values, one per pixel) with the inclusion outline, cycle number and peak stress
(MPa, E_matrix = 6 GPa) in the title; right: mean CG iterations per elastic solve vs cycle for Green / Green-Jacobi /
dual, moving marker at the current cycle.
"""
import os
import sys

import imageio_ffmpeg
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from matplotlib import animation

plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
plt.rcParams.update({'font.size': 13})
HERE = os.path.dirname(os.path.realpath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
PREC = sys.argv[2] if len(sys.argv) > 2 else 'dual'
TAG = sys.argv[3] if len(sys.argv) > 3 else '_incl11.6667_l0.0625_k0.0001_beads16_precrack_asymptotic_amp0.0028_cyc1500'
CG_TAG = sys.argv[4] if len(sys.argv) > 4 else TAG
E_MPA = float(os.environ.get('E_MPA', 6000.0))       # stress unit
PRECS = [('Green', 'Green', 'tab:green'), ('Green_Jacobi', 'Green-Jacobi', 'tab:blue'), ('dual', 'dual', 'tab:red')]
FPS = 12

data = netCDF4.Dataset(os.path.join(HERE, 'exp_data', f'FAT_N{N}_{PREC}{TAG}.nc'))
steps = np.asarray(data['step'][:]).astype(int)
sigma = np.asarray(data['macro_stress'][:, 1, 1]) * E_MPA
damage = data['damage']
phase = np.load(os.path.join(HERE, 'exp_data', f'FAT_N{N}_{PREC}{TAG}_damage.npz'))['phase']
xc = np.linspace(0.5 / N, 1 - 0.5 / N, N)

fig, (ax_d, ax_c) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={'width_ratios': [1, 1.25]})
im = ax_d.imshow(np.asarray(damage[0, 0]).T, origin='lower', extent=(0, 1, 0, 1), vmin=0, vmax=1, cmap='inferno',
                 interpolation='nearest')
ax_d.contour(xc, xc, phase.T, levels=[0.5], colors='c', linewidths=0.8)
ax_d.set_xlabel('x / L')
ax_d.set_ylabel('y / L')
fig.colorbar(im, ax=ax_d, fraction=0.046, label='damage d')
title = ax_d.set_title('', fontsize=13)
markers = []
ymax = 0
for P, label, c in PRECS:
    f = os.path.join(HERE, 'exp_data', f'FAT_N{N}_{P}{CG_TAG}_cycles.csv')
    if not os.path.exists(f):
        continue
    d = np.genfromtxt(f, delimiter=',', names=True)
    y = d['cg_u_step'] / d['staggered_its']
    ax_c.plot(d['step'] + 1, y, color=c, lw=1.5, label=label)
    ymax = max(ymax, np.percentile(y, 99.5))
    m, = ax_c.plot([], [], 'o', color=c, ms=9)
    markers.append((m, d['step'] + 1, y))
cursor = ax_c.axvline(1, color='0.4', lw=1)
ax_c.set_xlabel('cycle N')
ax_c.set_ylabel('CG iterations per elastic solve')
ax_c.set_title(f'{N}$^2$: elastic CG per solve (dual: 2 K per iteration)', fontsize=13)
ax_c.set_xlim(0, None)
ax_c.set_ylim(0, 1.1 * ymax)
ax_c.grid(alpha=0.3)
ax_c.legend(loc='upper left')
fig.tight_layout()


def frame(k):
    im.set_data(np.asarray(damage[k, 0]).T)
    cyc = steps[k] + 1
    title.set_text(f'cycle {cyc}, peak stress {sigma[k]:.1f} MPa')
    cursor.set_xdata([cyc, cyc])
    for m, c, y in markers:
        j = np.searchsorted(c, cyc)
        if j < len(c):
            m.set_data([c[j]], [y[j]])
    return [im, title, cursor] + [m for m, _, _ in markers]


out = os.path.join(HERE, 'figures', f'FAT_video_N{N}_{PREC}{TAG}.mp4')
anim = animation.FuncAnimation(fig, frame, frames=len(steps), blit=False)
anim.save(out, writer=animation.FFMpegWriter(fps=FPS, bitrate=4000), dpi=120)
frame(len(steps) - 1)
fig.savefig(out.replace('.mp4', '_final.png'), dpi=120)
print('saved', out)
