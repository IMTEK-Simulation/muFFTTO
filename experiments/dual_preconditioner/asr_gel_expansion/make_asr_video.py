"""
MP4 of the damage evolution vs gel expansion (one frame per load step) with the CG per elastic solve of
Green / Green-Jacobi / dual and a moving marker.
    python make_asr_video.py [N] [PREC]      env: LENGTH_SCALE (0.084375), SUFFIX ('_real')
-> figures/asr_damage_video_N<N>_<PREC><SUFFIX>_l<l>.mp4
"""
import os
import sys
import imageio_ffmpeg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
plt.rcParams.update({'font.size': 13, 'axes.labelsize': 14, 'axes.titlesize': 14})
here = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
P = sys.argv[2] if len(sys.argv) > 2 else 'dual'
l = float(os.environ.get('LENGTH_SCALE', 0.084375))
suffix = os.environ.get('SUFFIX', '_real')
PREC = {'Green': ('Green', 'tab:green', '-'), 'Green_Jacobi': ('Green-Jacobi', 'tab:blue', '--'),
        'dual': ('dual', 'tab:red', '-.')}


def base(Q):
    return os.path.join(here, 'exp_data', f'ASR_N{N}_{Q}_s1_l{l:g}{suffix}')


st = np.load(base(P) + '_steps.npz')
label, dvv, stiff, damage = st['label'], st['dVV'], st['stiffness_ratio'], st['damage']
cg_curves = {}
for Q in PREC:
    if not os.path.exists(base(Q) + '_iterations.csv'):
        continue
    it = np.atleast_1d(np.genfromtxt(base(Q) + '_iterations.csv', delimiter=',', names=True, dtype=None,
                                     encoding=None))
    cg = (it['cg_gel'] + it['cg_unit_xx'] + it['cg_unit_yy'] + it['cg_unit_xy']) / 4.0
    steps = np.unique(it['step'])
    cg_curves[Q] = (np.array([it['dVV'][it['step'] == s][0] for s in steps]),
                    np.array([cg[it['step'] == s].mean() for s in steps]))

fig, (ax_d, ax_c) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={'width_ratios': [1, 1.15]}, layout='constrained')
fig.get_layout_engine().set(w_pad=0.25, h_pad=0.1)
centres = (np.arange(N) + 0.5) / N * 100
im = ax_d.imshow(np.zeros((N, N)).T, origin='lower', cmap='inferno', vmin=0, vmax=1, interpolation='nearest',
                 extent=(0, 100, 0, 100))
ax_d.contour(centres, centres, (label == 1).T.astype(float), levels=[0.5], colors='w', linewidths=0.7)
ax_d.contour(centres, centres, (label == 2).T.astype(float), levels=[0.5], colors='#40c0ff', linewidths=0.9)
ax_d.set_xlabel('x [mm]')
ax_d.set_ylabel('y [mm]')
fig.colorbar(im, ax=ax_d, shrink=0.85, label='damage d')
markers = {}
for Q, (x, y) in cg_curves.items():
    lab, col, ls = PREC[Q]
    ax_c.plot(100 * x, y, color=col, ls=ls, lw=1.8, label=lab)
    markers[Q], = ax_c.plot([], [], 'o', color=col, ms=9)
ax_c.set_xlim(0, 100 * dvv.max() * 1.02)
ax_c.set_ylim(0, None)
ax_c.set_xlabel(r'gel area expansion $\Delta V/V$ [%]')
ax_c.set_ylabel('CG iterations per elastic solve')
ax_c.set_title(f'{N}$^2$, l = {l * 100:.2f} mm')
ax_c.grid(alpha=0.3)
ax_c.legend(loc='upper left')


def frame(k):            # k = 0: undamaged; damage[k - 1] belongs to dvv[k]
    im.set_data((np.zeros((N, N)) if k == 0 else np.clip(damage[k - 1], 0, 1)).T)
    ax_d.set_title(f'{P} run: $\\Delta V/V$ = {dvv[k]:.3f},  $\\|C\\|/\\|C_0\\|$ = {stiff[k]:.3f}')
    for Q, (x, y) in cg_curves.items():
        j = np.argmin(np.abs(x - dvv[k])) if k > 0 else None
        markers[Q].set_data(([] if j is None else [100 * x[j]]), ([] if j is None else [y[j]]))
    return [im]


out = os.path.join(here, 'figures', f'asr_damage_video_N{N}_{P}{suffix}_l{l:g}.mp4')
anim = animation.FuncAnimation(fig, frame, frames=len(dvv), blit=False)
anim.save(out, writer=animation.FFMpegWriter(fps=6, bitrate=4000, codec='libx264', extra_args=['-pix_fmt', 'yuv420p']),
          dpi=120)
print('saved', out)
