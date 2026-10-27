"""
Figure of run_box_3D.sh (box_clamp_3D.py): cube clamped on all six faces.

    1. outer CG iterations vs N: plain Green, corrected Green (dense S), the family FFT applied once
    2. total Green applications (setup + solve) vs N
    3. inner steps per outer iteration: plain inner CG against inner PCG with the family 2D FFT

usage:  python box_clamp_3D_plot.py      -> exp_data/box_clamp_3D.png
"""
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = os.path.dirname(os.path.realpath(__file__))
RUNS = os.path.join(HERE, 'exp_data', 'runs_box3d')
NS = (8, 16, 32, 64)
plt.rcParams.update({'font.size': 12})
DENSE, PLAIN, FFT, ONCE, GREEN = 'k', 'tab:blue', 'tab:orange', 'tab:red', 'tab:green'
EDGE = {1: plt.get_cmap('Purples')(0.55), 2: plt.get_cmap('Purples')(0.85)}


def load(n, variant):
    f = os.path.join(RUNS, f'N{n}_{variant}.npz')
    return {k: v[()] for k, v in np.load(f).items()} if os.path.exists(f) else None


def series(ax, variant, value, colour, text, ls='-', numbers=False, end_dy=-4):
    pts = [(n, load(n, variant)) for n in NS]
    pts = [(n, value(r)) for n, r in pts if r is not None]
    if not pts:
        return
    ax.plot([n for n, _ in pts], [v for _, v in pts], ls=ls, marker='o', color=colour, lw=2, ms=7)
    for n, v in (pts if numbers else []):
        ax.annotate(f'{v:.0f}', (n, v), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=9,
                    color=colour)
    ax.annotate(f'{text}: {pts[-1][1]:.0f}', pts[-1], textcoords='offset points', xytext=(12, end_dy), fontsize=11,
                color=colour)


total = lambda r: int(r['green_apps']) + int(r['setup_apps'])
outer = lambda r: abs(int(r['cg_its']))
fig, axes = plt.subplots(1, 3, figsize=(20, 6), layout='constrained')
ax = axes[0]
series(ax, 'green', outer, GREEN, 'plain Green', numbers=True)
series(ax, 'family1e-06', outer, FFT, 'corrected Green', numbers=True, end_dy=-14)
series(ax, 'familydirect', outer, ONCE, 'corrected, family FFT once', ls='-.', end_dy=6)
ax.set(ylabel='outer CG iterations', ylim=(0, None))
ax.set_title('outer CG iterations\n(corrected Green: the same for dense S, inner CG, inner PCG)', fontsize=13)
ax = axes[1]
series(ax, 'dense', total, DENSE, 'dense S')
series(ax, 'cg1e-08', total, PLAIN, 'inner CG 1e-8', ls=':', end_dy=-16)
series(ax, 'family1e-06', total, FFT, 'inner PCG, family FFT', numbers=True)
series(ax, 'familydirect', total, ONCE, 'family FFT once', ls='-.')
series(ax, 'green', total, GREEN, 'plain Green', ls='--', end_dy=-12)
for k in (1, 2):
    series(ax, f'family1e-06_patch{k}bal', total, EDGE[k], f'+ edge strips $\\pm{k}$', end_dy=14 if k == 2 else 0)
ax.set(yscale='log', ylabel='Green applications (setup + solve)')
ax.set_title('total cost: Green applications\n(setup + solve)', fontsize=13)
ax = axes[2]
series(ax, 'cg1e-08', lambda r: r['inner_its'].mean(), PLAIN, 'inner CG 1e-8', ls=':')
series(ax, 'family1e-06', lambda r: r['inner_its'].mean(), FFT, 'inner PCG, family FFT', numbers=True, end_dy=12)
for k in (1, 2):
    series(ax, f'family1e-06_patch{k}bal', lambda r: r['inner_its'].mean(), EDGE[k],
           f'+ edge strips $\\pm{k}$ (3 Green apps/step)', end_dy=-16 if k == 2 else -2)
ax.set(ylabel='inner steps per outer iteration', ylim=(0, None))
ax.set_title('inner steps per outer iteration', fontsize=13)
for ax in axes:
    ax.set_xscale('log')
    ax.set_xticks(NS, [str(n) for n in NS])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(6.5, 200)
    ax.set_xlabel('N (voxels over the cube)')
    ax.grid(True, color='0.92', which='both')
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('3D cube with 8 holes, all six faces clamped (planes through the void), body force: corrected Green with '
             'S solved by inner PCG,\nthe 3 families of parallel planes each inverted by a 2D FFT', fontsize=13)
fig.savefig(os.path.join(HERE, 'exp_data', 'box_clamp_3D.png'), dpi=110)
print('wrote exp_data/box_clamp_3D.png')
