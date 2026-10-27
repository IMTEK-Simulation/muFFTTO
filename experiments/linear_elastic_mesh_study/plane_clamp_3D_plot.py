"""
Figure of run_plane_3D.sh (plane_clamp_3D.py): 3D cantilever, corrected Green with the clamp plane inverted by a 2D FFT.

    1. outer CG iterations vs n_H: plain Green against the corrected Green (all S solves give the same count)
    2. total Green applications (setup + solve) vs n_H: dense S (solid face / full plane), inner CG, 2D FFT

usage:  python plane_clamp_3D_plot.py      -> exp_data/plane_clamp_3D.png
"""
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = os.path.dirname(os.path.realpath(__file__))
RUNS = os.path.join(HERE, 'exp_data', 'runs3d_plane')
NS = (4, 8, 16, 32)
plt.rcParams.update({'font.size': 12})


def load(n, variant):
    f = os.path.join(RUNS, f'n_H{n}_{variant}.npz')
    return {k: v[()] for k, v in np.load(f).items()} if os.path.exists(f) else None


def series(ax, variant, value, colour, text, ls='-', numbers=False, end_dy=-4):
    pts = [(n, load(n, variant)) for n in NS]
    pts = [(n, value(r)) for n, r in pts if r is not None]
    ax.plot([n for n, _ in pts], [v for _, v in pts], ls=ls, marker='o', color=colour, lw=2, ms=7)
    for n, v in (pts if numbers else []):
        ax.annotate(f'{v:.0f}', (n, v), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=9,
                    color=colour)
    ax.annotate(f'{text}: {pts[-1][1]:.0f}', pts[-1], textcoords='offset points', xytext=(12, end_dy), fontsize=11,
                color=colour)


total = lambda r: int(r['green_apps']) + int(r['setup_apps'])
fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout='constrained')
ax = axes[0]
series(ax, 'green', lambda r: r['cg_its'], 'tab:blue', 'plain Green', numbers=True)
series(ax, 'fft_plane', lambda r: r['cg_its'], 'tab:orange', 'corrected Green', numbers=True)
ax.set(ylim=(0, 115), ylabel='outer CG iterations')
ax.set_title('outer CG iterations\n(corrected Green: the same for every solve of S)', fontsize=13)
ax = axes[1]
series(ax, 'dense_plane', total, 'k', 'dense S, full plane', end_dy=4)
series(ax, 'dense_face', total, '0.55', 'dense S, solid face only', ls='--', end_dy=-12)
series(ax, 'cg_plane', total, 'tab:blue', 'inner CG 1e-8', ls=':')
series(ax, 'fft_plane', total, 'tab:orange', '2D FFT over the plane', numbers=True)
ax.set(yscale='log', ylim=(40, 2e4), ylabel='Green applications (setup + solve)')
ax.set_title('cost of the corrected Green\n(Green applications, setup + solve)', fontsize=13)
for ax in axes:
    ax.set_xscale('log')
    ax.set_xticks(NS, [str(n) for n in NS])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(3.4, 110)
    ax.set_xlabel('$n_H$ (voxels over the beam height)')
    ax.grid(True, color='0.92', which='both')
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('3D cantilever, clamped plane: S inverted by a 2D FFT over the plane (exact to 1e-15, never assembled).\n'
             'Same compliance for every variant (to 1e-10).', fontsize=13)
fig.savefig(os.path.join(HERE, 'exp_data', 'plane_clamp_3D.png'), dpi=110)
print('wrote exp_data/plane_clamp_3D.png')
