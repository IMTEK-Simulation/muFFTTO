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
PRECS = [('Green', 'Green', '#1baf7a', 'o'), ('Green_Jacobi', 'Green-Jacobi', '#2a78d6', 's'),
         ('dual', 'dual', '#d6402a', '^')]
plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.5, 'legend.fontsize': 6})


def totals(N, p):
    path = os.path.join(HERE, 'logs', f'AT2_N{N}_{p}{TAG}.log')
    if not os.path.exists(path):
        return None
    text = open(path).read()
    cg = re.search(r'total CG iterations elasticity = (\d+)', text)
    t = re.search(r'Elapsed time: ([0-9.]+) s', text)
    # one elastic solve per staggered iteration
    solves = sum(int(m) for m in re.findall(r'staggered its = +(\d+)', text))
    return (int(cg[1]), int(cg[1]) / solves, float(t[1])) if cg else None      # None: run not finished (yet)


fig, axes = plt.subplots(1, 3, figsize=(5.2, 2.3))
ends = [[], [], []]         # per panel: (grid, value) at the end of each curve, labelled below
for p, label, c, m in PRECS:
    done = [(N, x) for N in GRIDS if (x := totals(N, p)) is not None]
    grids = [N for N, _ in done]
    for i, (ax, values) in enumerate(zip(axes, ([x[0] for _, x in done], [x[1] for _, x in done],
                                                [x[2] for _, x in done]))):
        ax.plot(grids, values, color=c, marker=m, ms=3.5, lw=1.3, label=label)
        ends[i].append((grids[-1], values[-1]))
# end labels; two curves ending on the same grid within 25 %: the lower label moved down, the upper one up
for ax, panel in zip(axes, ends):
    for N, v in panel:
        close = [w for M, w in panel if M == N and w != v and max(v, w) / min(v, w) < 1.25]
        dy = 0 if not close else (4 if v > close[0] else -4)
        ax.annotate(f'{v:,.0f}', (N, v), textcoords='offset points', xytext=(5, dy), fontsize=5, va='center',
                    color='0.25')
for ax, ylabel in zip(axes, ['elastic CG iterations (total)', 'mean CG iterations\nper elastic solve', 'wall time [s]']):
    ax.set_xscale('log', base=2)
    ax.set_ylim(bottom=0)
    ax.set_xticks(GRIDS, [str(N) if N in (32, 128, 512) or not TAG else '' for N in GRIDS])
    ax.minorticks_off() if False else ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(GRIDS[0] / 1.3, GRIDS[-1] * 2.2)
    ax.set_xlabel(r'grid $N$ ($N^2$ pixels)')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v / 1000:g}k' if v else '0'))
fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=3, frameon=False)
if TAG:     # the 512^2 runs ran on the GPU, the others on the CPU (several runs sharing the cores)
    axes[2].axvspan(GRIDS[-1] / 1.25, GRIDS[-1] * 1.25, color='0.92', zorder=0)
    axes[2].text(GRIDS[-1], 0.95 * axes[2].get_ylim()[1], 'GPU', ha='center', fontsize=6, color='0.3')
fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.6)
if 'polydisperse' not in TAG:      # the polydisperse series is not in the paper (yet)
    fig.savefig(os.path.expanduser('~/PycharmProjects/Paper-Double-Green-Preconditioner/figures/part2_AT2_mesh' + TAG.replace('.', 'p') + '.pdf'))
fig.savefig(os.path.join(HERE, 'figures', f'AT2_mesh{TAG}.png'), dpi=250)
for p, *_ in PRECS:
    print(p, [totals(N, p) for N in GRIDS])
