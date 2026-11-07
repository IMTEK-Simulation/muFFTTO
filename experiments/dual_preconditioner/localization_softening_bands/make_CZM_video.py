"""MP4 of the PF-CZM crack evolution: phase field d (dual run, exact pixels, aggregate outlines) + stress-strain curve
+ mean CG per elastic solve of Green / Green-Jacobi / dual vs load, with a moving marker.

python make_CZM_video.py [LOAD_DIR=xx] [N=128] [K=0.001]   -> figures/CZM_video_N<N>_<dir>.mp4
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
import imageio_ffmpeg

plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
plt.rcParams.update({'font.size': 13})
here = os.path.dirname(os.path.abspath(__file__))
load_dir = sys.argv[1] if len(sys.argv) > 1 else 'xx'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 128
k = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3
tag = f'_asr_{load_dir}_cornelissen_b5mm_k{k:g}'
E_UNIT, CELL = 30e3, 80.0
precs = ['Green', 'Green_Jacobi', 'dual']
color = {'Green': 'tab:green', 'Green_Jacobi': 'tab:blue', 'dual': 'tab:red'}
label = {'Green': 'Green', 'Green_Jacobi': 'Green-Jacobi', 'dual': 'dual'}
comp = r'\bar\varepsilon_{xx}' if load_dir == 'xx' else r'\bar\gamma_{xy}'
scomp = r'\bar\sigma_{xx}' if load_dir == 'xx' else r'\bar\sigma_{xy}'

f = np.load(os.path.join(here, 'exp_data', f'CZM_N{N}_dual{tag}_fields.npz'))
d, loads, stress, matrix = f['d'], f['load'], f['stress'] * E_UNIT, f['phase']
steps = {P: np.genfromtxt(os.path.join(here, 'exp_data', f'CZM_N{N}_{P}{tag}_steps.csv'), delimiter=',', names=True)
         for P in precs if os.path.exists(os.path.join(here, 'exp_data', f'CZM_N{N}_{P}{tag}_steps.csv'))}

fig = plt.figure(figsize=(14, 6.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1])
ax_d = fig.add_subplot(gs[:, 0])
ax_s = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 1])
im = ax_d.imshow(d[0].T, origin='lower', extent=(0, CELL, 0, CELL), vmin=0, vmax=1, cmap='inferno',
                 interpolation='nearest')
xc = np.linspace(0, CELL, N, endpoint=False) + CELL / (2 * N)
ax_d.contour(xc, xc, matrix.T, [0.5], colors='cyan', linewidths=0.7)
ax_d.set_xlabel('x (mm)')
ax_d.set_ylabel('y (mm)')
fig.colorbar(im, ax=ax_d, shrink=0.85, label='phase field d')
ax_s.plot(np.r_[0, loads] * 1e3, np.r_[0, stress], '-', color='tab:red', lw=1.6)
mark_s, = ax_s.plot([], [], 'o', color='k', ms=8)
ax_s.set_xlim(0)
ax_s.set_ylim(0)
ax_s.set_xlabel(rf'${comp}$ ($\times 10^{{-3}}$)')
ax_s.set_ylabel(rf'${scomp}$ (MPa)')
ax_s.grid(alpha=0.3)
marks_c = {}
for P, st in steps.items():
    ax_c.plot(st['load'] * 1e3, st['mean_cg_u'], '-', color=color[P], label=label[P], lw=1.4)
    marks_c[P], = ax_c.plot([], [], 'o', color=color[P], ms=7, mec='k')
ax_c.set_xlim(0)
ax_c.set_ylim(0)
ax_c.set_xlabel(rf'${comp}$ ($\times 10^{{-3}}$)')
ax_c.set_ylabel('mean CG per elastic solve')
ax_c.legend(fontsize=11, loc='upper left')
ax_c.grid(alpha=0.3)
fig.tight_layout()

out = os.path.join(here, 'figures', f'CZM_video_N{N}_{load_dir}.mp4')
writer = animation.FFMpegWriter(fps=12, bitrate=6000, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
with writer.saving(fig, out, dpi=100):
    for i in range(len(loads)):
        im.set_data(d[i].T)
        ax_d.set_title(rf'PF-CZM {N}², ${comp}$ = {loads[i] * 1e3:.3f}e-3, ${scomp}$ = {stress[i]:.2f} MPa', fontsize=13)
        mark_s.set_data([loads[i] * 1e3], [stress[i]])
        for P, st in steps.items():
            j = min(i, len(st['load']) - 1)
            marks_c[P].set_data([st['load'][j] * 1e3], [st['mean_cg_u'][j]])
        writer.grab_frame()
print('saved', out)
