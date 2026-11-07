"""MP4 video of a core-shell battery run: Li filling and damage side by side (exact pixels, core and shell outlines),
time and state of lithiation in the title, and the CG iterations per elastic solve of Green / Green-Jacobi / dual vs
time with a moving marker.

    python make_battery_video.py [N=128] [preconditioner=dual]       # Tu et al. series (run_battery_tu.sh)
    env: SUFFIX (default _tu), L_CELL_UM (6), R_CORE_UM (2), H_BAR (0.2), FPS (8)

Needs dense snapshots (SNAP_EVERY=2 in the run).
"""
import os
import sys

import imageio_ffmpeg
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Circle

plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
plt.rcParams.update({'font.size': 13, 'axes.labelsize': 14, 'axes.titlesize': 14, 'legend.fontsize': 12})
HERE = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
PREC = sys.argv[2] if len(sys.argv) > 2 else 'dual'
SUFFIX = os.environ.get('SUFFIX', '_tu')
L_CELL = float(os.environ.get('L_CELL_UM', 6.0))
R_CORE = float(os.environ.get('R_CORE_UM', 2.0))
R_OUT = R_CORE * (1.0 + float(os.environ.get('H_BAR', 0.2)))
FPS = int(os.environ.get('FPS', 8))
T_SCALE = (L_CELL * 1e-6) ** 2 / 3.26e-14
PRECS = ['Green', 'Green_Jacobi', 'dual']
COLOR = {'Green': 'tab:green', 'Green_Jacobi': 'tab:blue', 'dual': 'tab:red'}
LABEL = {'Green': 'Green', 'Green_Jacobi': 'Green-Jacobi', 'dual': 'dual'}


def base(P):
    return os.path.join(HERE, 'exp_data', f'battery_N{N}_{P}_ms0.01{SUFFIX}')


def steps_of(P):
    if not os.path.exists(base(P) + '_steps.csv'):
        return None
    return np.atleast_1d(np.genfromtxt(base(P) + '_steps.csv', delimiter=',', names=True, dtype=None, encoding=None))


z = np.load(base(PREC) + '_snapshots.npz')
times, conc, dmg, particle = z['time'], z['concentration'], z['damage'], z['particle']
steps = steps_of(PREC)
others = {P: steps_of(P) for P in PRECS}
print(f'{len(times)} frames, N = {N}, {PREC}')

ext = (0, L_CELL, 0, L_CELL)
fig = plt.figure(figsize=(16, 5.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.35)
ax_c, ax_d, ax_g = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
im_c = ax_c.imshow(np.where(particle > 0, conc[0], np.nan).T, origin='lower', interpolation='nearest', extent=ext,
                   vmin=0.1, vmax=1.0, cmap='viridis')
im_d = ax_d.imshow(dmg[0].T, origin='lower', interpolation='nearest', extent=ext, vmin=0, vmax=1, cmap='inferno')
ax_c.set_facecolor('#d9d9d9')
for ax, lw_col in ((ax_c, 'white'), (ax_d, 'white')):
    for R in (R_CORE, R_OUT):
        ax.add_patch(Circle((L_CELL / 2, L_CELL / 2), R, fill=False, ec=lw_col, lw=0.8, ls='--'))
    ax.set_xlabel(r'x [$\mu$m]')
    ax.set_ylabel(r'y [$\mu$m]')
ax_c.set_title('Li filling c / c$_{max}$')
ax_d.set_title('damage d')
fig.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04)
fig.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)
ymax = 0
for P in PRECS:
    s = others[P]
    if s is None:
        continue
    ax_g.plot(s['time'] * T_SCALE, s['cg_u_mean'], color=COLOR[P], lw=1.3, label=LABEL[P])
    ymax = max(ymax, float(np.max(s['cg_u_mean'])))
marker = ax_g.axvline(times[0] * T_SCALE, color='k', lw=1.2)
ax_g.set_xlim(0, steps['time'][-1] * T_SCALE)
ax_g.set_ylim(0, 1.05 * ymax)
ax_g.set_xlabel('time [s]')
ax_g.set_ylabel('mean CG iterations per elastic solve')
ax_g.set_title(f'elasticity, {N}x{N}')
ax_g.grid(alpha=0.3)
ax_g.legend(loc='upper left')
title = fig.suptitle('')


def draw(k):
    im_c.set_data(np.where(particle > 0, conc[k], np.nan).T)
    im_d.set_data(dmg[k].T)
    marker.set_xdata([times[k] * T_SCALE] * 2)
    i = int(np.argmin(np.abs(steps['time'] - times[k])))
    title.set_text(f'core-shell particle (NMC811 @ NMC532), 1C lithiation, {LABEL[PREC]} run:  '
                   f't = {times[k] * T_SCALE:.0f} s,  SOL = {steps["sol"][i]:.3f},  '
                   f'broken fraction = {steps["a_c"][i]:.3f}')


out = os.path.join(HERE, 'figures', f'battery_video_N{N}_{PREC}{SUFFIX}.mp4')
writer = animation.FFMpegWriter(fps=FPS, bitrate=6000, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
with writer.saving(fig, out, dpi=110):
    for k in range(len(times)):
        draw(k)
        writer.grab_frame()
draw(len(times) - 1)
fig.savefig(out.replace('.mp4', '_last.png'), dpi=110, bbox_inches='tight')
print('saved', out)
if os.environ.get("NO_OPEN") != "1":
    os.system(f"xdg-open {out} >/dev/null 2>&1 &")
