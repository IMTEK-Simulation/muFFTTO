"""Mesh dependence of the elastic CG iterations of the small-strain AT2 runs in the phases of the fracture process:
    intact            before the first crack breaks through (first drop of sigma_xx by more than 5 %)
    partly cracked    cracks through, the cell still carries load (only with several pores)
    broken            the cell is cut through: sigma_xx < 10 % of the peak
The break steps (sigma_xx drops by more than 5 %, many staggered iterations) are left out. Per load step the log holds
the CG count of the last staggered iteration (warm start); markers are the mean over the load steps of each phase,
bars their min-max; a phase that is empty in every run is not drawn; open markers: run not finished yet (load steps
so far) -> figures/AT2_mesh_phases<TAG>.png, .pdf

    python plot_AT2_mesh_phases.py [TAG]      (default _l0.03125: one pore, fixed physical length scale;
                                               _polydisperse_pores_l0.03125: pores of different sizes)
"""
import os
import re
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
TAG = sys.argv[1] if len(sys.argv) > 1 else '_l0.03125'
PRECS_NAMES = ('Green', 'Green_Jacobi', 'dual')


def finished(N, p):
    path = os.path.join(HERE, 'logs', f'AT2_N{N}_{p}{TAG}.log')
    return os.path.exists(path) and 'Elapsed time' in open(path).read()


GRIDS = [N for N in (32, 64, 128, 256, 512, 1024)
         if any(os.path.exists(os.path.join(HERE, 'logs', f'AT2_N{N}_{p}{TAG}.log')) for p in PRECS_NAMES)]
PRECS = [('Green', 'Green', '#1baf7a', 'o'), ('Green_Jacobi', 'Green-Jacobi', '#2a78d6', 's'),
         ('dual', 'dual', '#d6402a', '^')]
STEP = re.compile(r'sigma_xx = (\S+) +max d = (\S+) +staggered its = +\d+ +CG its \(u, d\) = \((\d+), \d+\)')
PHASES = ['intact', 'partly cracked', 'broken']
plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.5, 'legend.fontsize': 6})


def phases(N, p):
    """elastic CG iterations per load step in each of PHASES (arrays, possibly empty; all empty: no run)"""
    if not os.path.exists(os.path.join(HERE, 'logs', f'AT2_N{N}_{p}{TAG}.log')):
        return [np.zeros(0)] * 3
    sigma, d, cg = np.array(STEP.findall(open(os.path.join(HERE, 'logs', f'AT2_N{N}_{p}{TAG}.log')).read()),
                            dtype=float).T
    drop = np.r_[False, sigma[1:] < 0.95 * sigma[:-1]]
    phase = np.maximum.accumulate(drop).astype(int)
    phase[(phase == 1) & (sigma < 0.1 * sigma.max())] = 2        # not the first load steps, also small sigma
    return [cg[(phase == i) & ~drop] for i in range(3)]


data = {(N, p): phases(N, p) for N in GRIDS for p, *_ in PRECS}
shown = [i for i in range(3) if any(data[N, p][i].size for N in GRIDS for p, *_ in PRECS)]
fig, axes = plt.subplots(1, len(shown), figsize=(2.0 * len(shown), 2.3), sharey=True, squeeze=False)
axes = axes[0]
for ax, i in zip(axes, shown):
    title = PHASES[i]
    for j, (p, label, c, m) in enumerate(PRECS):
        grids = [N for N in GRIDS if data[N, p][i].size]
        values = [data[N, p][i] for N in grids]
        mean = np.array([v.mean() for v in values])
        lo, hi = mean - [v.min() for v in values], [v.max() for v in values] - mean
        if not grids:
            continue
        x = np.array(grids) * 1.04 ** (j - 1)          # small offset so the bars do not overlap
        ax.errorbar(x, mean, yerr=(lo, hi), color=c, marker=m, ms=3.5, lw=1.3, elinewidth=0.6, capsize=1.5,
                    label=label)
        running = [k for k, N in enumerate(grids) if not finished(N, p)]
        ax.plot(x[running], mean[running], ls='none', marker=m, ms=2.2, color='white', zorder=3)
        ax.annotate(f'{mean[-1]:.0f}', (x[-1], mean[-1]), textcoords='offset points', xytext=(6, 0), fontsize=5,
                    va='center', color='0.25')
    ax.set_title(title)
    ax.set_xscale('log', base=2)
    ax.set_ylim(bottom=0)
    ax.set_xticks(GRIDS, [str(N) for N in GRIDS])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(GRIDS[0] / 1.3, GRIDS[-1] * 1.9)
    ax.set_xlabel(r'grid $N$ ($N^2$ pixels)')
    ax.grid(alpha=0.3)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel('elastic CG iterations\nper load step')
fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.8)
out = os.path.join(HERE, 'figures', f'AT2_mesh_phases{TAG}')
fig.savefig(out + '.png', dpi=250)
fig.savefig(out + '.pdf')
for p, *_ in PRECS:
    print(p, [(N, *[round(float(data[N, p][i].mean()), 1) if data[N, p][i].size else None for i in range(3)]) for N in GRIDS])
