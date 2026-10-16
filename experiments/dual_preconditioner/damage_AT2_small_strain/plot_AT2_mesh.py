"""Mesh dependence of the small-strain AT2 runs: total elastic CG iterations and wall time over the grid for the three
elastic preconditioners (logs/AT2_N<n>_<prec>.log) -> <paper>/figures/part2_AT2_mesh.pdf and figures/AT2_mesh.png

    python plot_AT2_mesh.py [TAG]      (TAG e.g. _l0.03125: the series with a fixed physical length scale, 32^2-512^2)
"""
import os
import re
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
TAG = sys.argv[1] if len(sys.argv) > 1 else ''
GRIDS = [32, 64, 128, 256, 512] if TAG else [64, 128, 256]
PRECS = [('Green', 'Green', '#2a78d6', 'o'), ('Green_Jacobi', 'Green-Jacobi', '#eb6834', 's'),
         ('dual', 'dual', '#1baf7a', '^')]
plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.5, 'legend.fontsize': 6})


def totals(N, p):
    text = open(os.path.join(HERE, 'logs', f'AT2_N{N}_{p}{TAG}.log')).read()
    cg = re.search(r'total CG iterations elasticity = (\d+)', text)
    t = re.search(r'Elapsed time: ([0-9.]+) s', text)
    return int(cg[1]), float(t[1])


fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.3))
for p, label, c, m in PRECS:
    r = [totals(N, p) for N in GRIDS]
    for ax, values in zip(axes, ([x[0] for x in r], [x[1] for x in r])):
        ax.plot(GRIDS, values, color=c, marker=m, ms=3.5, lw=1.3, label=label)
        dy = {'Green': 4, 'dual': -4}.get(p, 0) if ax is axes[1] else 0     # Green and dual times nearly coincide
        ax.annotate(f'{values[-1]:,.0f}', (GRIDS[-1], values[-1]), textcoords='offset points', xytext=(5, dy),
                    fontsize=5.5, va='center', color='0.25')
for ax, ylabel in zip(axes, ['elastic CG iterations (total)', 'wall time [s]']):
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(GRIDS, [f'${N}^2$' for N in GRIDS])
    ax.set_xlim(GRIDS[0] / 1.3, GRIDS[-1] * 1.6)
    ax.set_xlabel('grid')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, which='both')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
axes[0].legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.expanduser('~/PycharmProjects/Paper-Double-Green-Preconditioner/figures/part2_AT2_mesh' + TAG.replace('.', 'p') + '.pdf'))
fig.savefig(os.path.join(HERE, 'figures', f'AT2_mesh{TAG}.png'), dpi=250)
for p, *_ in PRECS:
    print(p, [totals(N, p) for N in GRIDS])
