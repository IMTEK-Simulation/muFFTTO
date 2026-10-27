"""Cost estimate on a formed design (JCTAM setup, kappa_void = 1e-3): every run starts from frame 500 of the dual run on
the same grid and does 6 L-BFGS iterations (run_JCTAM_soft3_from_dual.sh). The first evaluation (CG from zero) is left out.
    (a) mean PCG iterations per solve over the grid, mechanical (x) and adjoint (triangle)
    (b) PCG iterations of every solve over the evaluations on 512^2 (3 load cases per evaluation)

    python plot_JCTAM_fromdual.py      -> figures/JCTAM_fromdual.pdf, .png
"""
import os
import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.realpath(__file__))
GRIDS = [128, 256, 512]
PRECS = [('Green', 'Green', '#1a8a1a', (0, (6, 2, 1, 2))), ('Green_Jacobi', 'Green--Jacobi', 'black', '-'),
         ('dual', 'dual', '#2a78d6', '-')]
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 9, 'axes.titlesize': 10})


def counts(N, p):
    text = open(os.path.join(HERE, 'logs', f'JCTAM_N{N}_soft3_{p}_fromdual.log')).read()
    m = [int(x) for x in re.findall(r'nb_ steps CG mech +=(\d+)', text)][3:]       # without the first evaluation
    a = [int(x) for x in re.findall(r'nb_steps CG adjoint +=(\d+)', text)][3:]
    return np.array(m), np.array(a)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.4))
for p, label, c, ls in PRECS:
    data = [counts(N, p) for N in GRIDS]
    for k, marker in enumerate(('x', '^')):
        ax1.plot(GRIDS, [d[k].mean() for d in data], color=c, ls=ls, lw=1.8 if k == 0 else 1.3,
                 marker=marker, ms=6 if marker == 'x' else 5, mew=1.5)
    m, a = data[-1]
    solves = np.arange(len(m)) / 3 + 2                 # evaluation 2, 3, ...; its 3 load cases at +0, +1/3, +2/3
    ax2.plot(solves, m, color=c, ls=ls, lw=1.5, marker='x', ms=4)
    ax2.plot(solves, a, color=c, ls=ls, lw=1.2, marker='^', ms=3.5)
ax1.set_xscale('log', base=2)
ax1.set_xticks(GRIDS, [f'${N}^2$' for N in GRIDS])
ax1.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax1.set_xlim(100, 650)
ax1.set_xlabel('Grid size')
ax1.set_title(r'mean over 6 evaluations')
ax2.set_xlabel(r'evaluation (from frame 500), $512^2$')
ax2.set_title(r'every solve, $512^2$')
for ax, tag in ((ax1, '(a)'), (ax2, '(b)')):
    ax.set_yscale('log')
    ax.set_ylim(10, 1000)
    ax.grid(axis='y', which='major', color='0.85', lw=0.6)
    ax.set_ylabel(r'\# PCG iterations per solve')
    ax.text(0.01, 1.02, rf'\textbf{{{tag}}}', transform=ax.transAxes, ha='left', va='bottom')
handles = ([Line2D([], [], color=c, ls=ls, lw=1.8) for _, _, c, ls in PRECS]
           + [Line2D([], [], ls='none', marker='x', color='0.3', mew=1.5), Line2D([], [], ls='none', marker='^', color='0.3')])
fig.legend(handles, [l for _, l, _, _ in PRECS] + ['mechanical', 'adjoint'], loc='lower center', ncol=5, frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=2)
fig.savefig(os.path.join(HERE, 'figures', 'JCTAM_fromdual.pdf'))
fig.savefig(os.path.join(HERE, 'figures', 'JCTAM_fromdual.png'), dpi=200)
