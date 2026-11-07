"""
Green-Jacobi vs dual in the phase-field TO (NEMO eta / contrast study, figures/nemo/contrast_summary.csv; work per
linear solve, dual counted x 2): GJ work / dual work against eta/h, one line per grid, one panel per contrast.
> 1: dual cheaper, < 1: Green-Jacobi cheaper.     python plot_TO_resolution_gj_vs_dual.py
-> figures/TO_resolution_gj_vs_dual.png
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
W = {}
for r in csv.DictReader(open(os.path.join(HERE, 'figures', 'nemo', 'contrast_summary.csv'))):
    W[(int(float(r['contrast'])), int(r['N']), int(float(r['eta_h'])), r['precond'])] = float(r['work'])
GRIDS = [32, 64, 128, 256, 512, 1024]
EH = [1, 2, 4, 8]
KAPPA = [100, 1000, 10000, 100000]
cN = plt.cm.viridis(np.linspace(0, 0.9, len(GRIDS)))
plt.rcParams.update({'font.size': 13})
fig, axs = plt.subplots(2, 2, figsize=(16, 12), layout='constrained', sharey=True)
for ax, k in zip(axs.flat, KAPPA):
    for n, col in zip(GRIDS, cN):
        y = [W[(k, n, e, 'Green_Jacobi')] / W[(k, n, e, 'dual')] if (k, n, e, 'Green_Jacobi') in W
             and (k, n, e, 'dual') in W else np.nan for e in EH]
        ax.plot(EH, y, 's-', color=col, lw=2.2, ms=8, label=f'{n}$^2$')
    ax.set_xscale('log', base=2), ax.set_yscale('log')
    ax.axhline(1, color='k', lw=1.5)
    ax.axvspan(0.8, 2, color='0.9', zorder=0)
    ax.set_xticks(EH), ax.set_xticklabels([str(e) for e in EH]), ax.minorticks_off()
    yt = [0.2, 0.5, 1, 2, 5, 10]
    ax.set_yticks(yt), ax.set_yticklabels([f'{v:g}' for v in yt])
    ax.set_xlim(0.8, 10), ax.set_ylim(0.15, 15)
    ax.set(title=f'contrast {k:.0e}', xlabel='eta / h  (grey: < 2 under-resolved)',
           ylabel='Green-Jacobi work / dual work (x 2)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11, loc='upper right', framealpha=0.95, ncol=2)
fig.suptitle('Phase-field TO: Green-Jacobi vs dual (above 1: dual cheaper, below 1: Green-Jacobi cheaper)',
             fontsize=15)
out = os.path.join(HERE, 'figures', 'TO_resolution_gj_vs_dual.png')
fig.savefig(out, dpi=120)
print('wrote', out)
