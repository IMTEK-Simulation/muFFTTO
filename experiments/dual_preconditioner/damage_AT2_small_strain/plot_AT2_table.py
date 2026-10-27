"""The per-run totals of the AT2 polydisperse-pore series over the grid size, one panel per quantity:
load steps, elastic CG iterations (total), elastic CG iterations per solve, damage CG iterations (total), wall time.

Data: exp_data/AT2_N<N>_<prec><TAG>_iterations.csv (joined with a _restart continuation, as in plot_AT2_performance.py)
and logs/AT2_N<N>_<prec><TAG>[_restart].log (wall time). Hollow markers: runs on the GPU (512^2 and up); the wall time
of the 1024^2 run is the restarted part only (the stopped first part logged no time).

    python plot_AT2_table.py [TAG]      (default _polydisperse_pores_l0.03125)
-> figures/AT2_table<TAG>.png, .pdf
"""
import os
import re
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
TAG = sys.argv[1] if len(sys.argv) > 1 else '_polydisperse_pores_l0.03125'
PRECS = [('Green', 'Green', '#1baf7a', 'o'), ('Green_Jacobi', 'Green-Jacobi', '#2a78d6', 's'),
         ('dual', 'dual', '#d6402a', '^')]
GRIDS = (32, 64, 128, 256, 512, 1024)
GPU_FROM = 512                        # these grids ran on the GPU
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8.5, 'legend.fontsize': 7.5})


def run(N, p):
    """load steps, elastic CG total, elastic CG per solve, damage CG total, wall time [s]; None if not finished"""
    name = f'AT2_N{N}_{p}{TAG}'
    csv, log = (os.path.join(HERE, 'exp_data', name + '_iterations.csv'), os.path.join(HERE, 'logs', name + '.log'))
    if not (os.path.exists(csv) and os.path.exists(log)):
        return None
    it = np.genfromtxt(csv, delimiter=',', names=True, ndmin=1)
    last_log = log
    if os.path.exists(log.replace('.log', '_restart.log')):
        last_log = log.replace('.log', '_restart.log')
        r = np.genfromtxt(csv.replace('_iterations', '_restart_iterations'), delimiter=',', names=True, ndmin=1)
        it = np.concatenate([it[it['step'] < r['step'].min()], r])
    elapsed = re.search(r'Elapsed time: ([\d.]+)', open(last_log).read())
    if elapsed is None:
        return None
    return (len(np.unique(it['step'])), it['cg_u'].sum(), it['cg_u'].mean(), it['cg_d'].sum(), float(elapsed[1]))


PANELS = ['load steps', 'elastic CG iterations\n(total)', 'elastic CG iterations\nper elastic solve',
          'damage CG iterations\n(total)', 'wall time [s]']
LOG_Y = [False, True, False, True, True]
fig, axes = plt.subplots(1, 5, figsize=(10, 2.5))
for p, label, colour, marker in PRECS:
    rows = [(N, r) for N in GRIDS if (r := run(N, p)) is not None]
    grids = np.array([N for N, _ in rows])
    gpu = grids >= GPU_FROM
    for k, ax in enumerate(axes):
        y = np.array([r[k] for _, r in rows])
        ax.plot(grids, y, color=colour, lw=1.6, label=label)
        ax.plot(grids[~gpu], y[~gpu], ls='none', marker=marker, ms=5, color=colour)
        ax.plot(grids[gpu], y[gpu], ls='none', marker=marker, ms=5, mfc='white', mec=colour, mew=1.3)
for ax, title, log_y in zip(axes, PANELS, LOG_Y):
    ax.set_title(title)
    ax.set_xscale('log', base=2)
    ax.set_xticks(GRIDS, [f'{N}' for N in GRIDS])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel(r'grid $N$ ($N^2$ pixels)')
    if log_y:
        ax.set_yscale('log')
    else:
        ax.set_ylim(0, 1.12 * ax.get_ylim()[1])
    ax.grid(alpha=0.3, which='major')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis='x', labelsize=7)
handles = axes[0].get_lines()[0::3]
gpu_handle = matplotlib.lines.Line2D([], [], ls='none', marker='o', ms=5, mfc='white', mec='0.35', mew=1.3)
fig.legend(list(handles) + [gpu_handle], [h.get_label() for h in handles] + ['GPU run (else CPU)'],
           loc='lower center', ncol=4, frameon=False)
fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=1.2)
out = os.path.join(HERE, 'figures', f'AT2_table{TAG}')
fig.savefig(out + '.png', dpi=250)
fig.savefig(out + '.pdf')
print(out + '.png')
