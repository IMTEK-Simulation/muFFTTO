"""Video: phase-field TO without (w = 0) and with (w = 0.2) the phase-field term, same grid and sine start
(logs/TO_no_pf, plot_TO_no_pf.py). Top: the designs on exact pixels (solid black, void white); bottom: the objective of
both over the L-BFGS iterations (log axis: w = 0 falls to ~1e-9). History every 10 iterations; each stored iterate is
held (no blending); the run that converged first stays on its final design.
    python make_TO_no_pf_video.py [N=64]  ->  figures/movie_TO_no_pf_N<N>.mp4 (+ _final.png)
"""
import glob
import os
import sys

import imageio_ffmpeg
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
D = os.path.join(HERE, 'data', 'example_2D_elasticity_TO')
N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
W, H, FPS, HOLD = 2400, 1600, 30, 6
ENC = ['-profile:v', 'high', '-level', '5.1', '-crf', '17', '-g', '30', '-keyint_min', '30', '-preset', 'slow',
       '-movflags', '+faststart']


def load(w):
    f = glob.glob(os.path.join(D, f'N{N}_soft0.0001_energy-warm_Green_Jacobi_eta_*_pfw{w}_l_bfgs_nopf_test_history.nc'))[0]
    with netCDF4.Dataset(f) as d:
        return np.asarray(d['phase_field_iterate'][:, 0]), np.asarray(d['objective'][:])


R0, F0 = load('0')
R1, F1 = load('0.2')
n = max(len(F0), len(F1))
it = 10 * np.arange(n)
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 22})
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.42], left=0.08, right=0.97, top=0.9, bottom=0.08, hspace=0.32,
                      wspace=0.12)
axs = [fig.add_subplot(gs[0, j]) for j in range(2)]
ims = []
for ax, R, t in zip(axs, (R0, R1), (r'\textbf{no phase-field term} ($w = 0$)', r'with phase field ($w = 0.2$)')):
    ims.append(ax.imshow(R[0].T, origin='lower', cmap='Greys', vmin=0, vmax=1, interpolation='nearest'))
    ax.set_xticks([]), ax.set_yticks([]), ax.set_title(t, fontsize=26)
axf = fig.add_subplot(gs[1, :])
axf.semilogy(10 * np.arange(len(F0)), F0, color='0.8', lw=2)
axf.semilogy(10 * np.arange(len(F1)), F1, color='0.8', lw=2)
l0, = axf.semilogy([], [], 'tab:red', lw=3, label=r'no phase field ($w = 0$)')
l1, = axf.semilogy([], [], 'tab:blue', lw=3, label=r'phase field ($w = 0.2$)')
axf.set(xlabel='L-BFGS iteration', ylabel='objective', xlim=(0, it[-1]), ylim=(1e-10, 20))
axf.set_yticks([1e-9, 1e-6, 1e-3, 1]), axf.set_yticklabels([r'$10^{-9}$', r'$10^{-6}$', r'$10^{-3}$', r'$1$'])
axf.grid(alpha=0.3), axf.legend(loc='lower left', fontsize=20)
title = fig.text(0.5, 0.955, '', ha='center', va='center', fontsize=30)


def draw(k):
    k0, k1 = min(k, len(F0) - 1), min(k, len(F1) - 1)
    ims[0].set_data(R0[k0].T), ims[1].set_data(R1[k1].T)
    l0.set_data(10 * np.arange(k0 + 1), F0[:k0 + 1]), l1.set_data(10 * np.arange(k1 + 1), F1[:k1 + 1])
    title.set_text(rf'Topology optimization ${N}^2$, contrast $10^4$, same start: iteration {10 * k}')
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


out = os.path.join(HERE, 'figures', f'movie_TO_no_pf_N{N}.mp4')
w = imageio_ffmpeg.write_frames(out, (W, H), fps=FPS, quality=None, codec='libx264', pix_fmt_out='yuv420p',
                                macro_block_size=8, output_params=ENC)
w.send(None)
for k in range(n):
    img = draw(k)
    for _ in range(HOLD if k else 2 * FPS):                       # hold the start 2 s
        w.send(img)
for _ in range(3 * FPS):                                           # hold the end 3 s
    w.send(img)
w.close()
plt.imsave(out.replace('.mp4', '_final.png'), img)
print('wrote', out, f'{n} iterates')
