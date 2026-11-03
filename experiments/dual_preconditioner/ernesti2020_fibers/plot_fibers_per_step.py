"""
Convergence of the elastic preconditioners over the load path (fibre runs, seed 0): mean CG iterations per elastic
solve in every load step (lines), stress (thin grey, right axis scaled), staggered iterations of the step (bars);
bottom row: cumulative elastic work (dual counts 2 per iteration).
usage:  python plot_fibers_per_step.py          writes figures/fibers_per_step.png
"""
import os
import re

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
PRECS = [('Green', 'tab:green', '--', 1, 'Green'), ('Green_Jacobi', 'tab:blue', ':', 1, 'Green-Jacobi'),
         ('dual', 'tab:red', '-', 2, 'dual')]
GRIDS = (64, 128, 256)
PAT = re.compile(r'^step\s+(\d+)\s+E_xx = ([\d.]+)\s+sigma_xx = ([\d.e+-]+).*?staggered its =\s+(\d+).*?'
                 r'step CG \(u, d\) = \((\d+),', re.M)


def load(n, p):
    f = os.path.join(HERE, 'logs', f'AT2_fibers_s0_N{n}_{p}.log')
    if not os.path.exists(f):
        return None
    a = np.array([[float(v) for v in m] for m in PAT.findall(open(f).read())])
    return None if a.size == 0 else a          # step, E, sigma, staggered, CG u of the step


grids = [n for n in GRIDS if load(n, 'Green') is not None]
plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(2, len(grids), figsize=(7 * len(grids), 10), layout='constrained', squeeze=False)
for col, n in enumerate(grids):
    ax, axc = axes[0, col], axes[1, col]
    ax2 = ax.twinx()
    ref = None
    for p, c, ls, w, lab in PRECS:
        a = load(n, p)
        if a is None:
            continue
        ref = a
        E = 100 * a[:, 1]
        ax.plot(E, a[:, 4] / a[:, 3], color=c, ls=ls, lw=2, marker='o', ms=4, label=lab)
        axc.plot(E, w * np.cumsum(a[:, 4]) / 1e3, color=c, ls=ls, lw=2, marker='o', ms=4, label=lab)
    E = 100 * ref[:, 1]
    ax2.bar(E, ref[:, 3], width=0.06, color='0.75', zorder=0)
    ax2.plot(E, ref[:, 3].max() * ref[:, 2] / ref[:, 2].max(), color='0.4', lw=1)
    ax2.set_ylabel('staggered iterations of the step (bars)\nthin line: stress (scaled)')
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_ylim(0, None)
    ax.set_xlabel('E_xx [%]')
    ax.set_ylabel('CG iterations per elastic solve (mean over the step)')
    ax.set_title(f'{n}² (h = {512 / n:g} um, l/h = {6 * n / 512:g})')
    axc.set_ylim(0, None)
    axc.set_xlabel('E_xx [%]')
    axc.set_ylabel('cumulative elastic work [thousands] (dual 2 per it.)')
    for a_ in (ax, axc):
        a_.grid(True, color='0.9')
    if col == 0:
        ax.legend(frameon=False, loc='upper left')
fig.suptitle('Ernesti et al. 2020 fibres (seed 0): convergence of the elastic preconditioners over the load path',
             fontsize=14)
out = os.path.join(HERE, 'figures', 'fibers_per_step.png')
fig.savefig(out, dpi=150)
print(f'wrote {out}')
