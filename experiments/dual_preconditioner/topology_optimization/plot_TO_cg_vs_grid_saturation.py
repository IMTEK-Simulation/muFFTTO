"""
Does the dual saturate with the grid like Green?  Phase-field TO (NEMO eta / contrast study,
figures/nemo/contrast_summary.csv): mean CG iterations per linear solve (raw, no cost factor) against the grid N at
fixed eta/h, one panel per contrast. Green green, Green-Jacobi blue, dual red; eta/h = 1, 2, 4, 8 = line styles.
Green was not run at 1024^2. Black x: design ended grey (no valid design; as the open markers of
figures/nemo/TO_contrast_work.png, which shows the same data as work = dual x 2, panels per eta/h).        python plot_TO_cg_vs_grid_saturation.py -> figures/TO_cg_vs_grid_saturation.png
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
W, GREY = {}, set()
for r in csv.DictReader(open(os.path.join(HERE, 'figures', 'nemo', 'contrast_summary.csv'))):
    W[(int(float(r['contrast'])), int(r['N']), int(float(r['eta_h'])), r['precond'])] = float(r['cg'])
    if r['design'] != 'True':
        GREY.add((int(float(r['contrast'])), int(r['N']), int(float(r['eta_h'])), r['precond']))
GRIDS = [32, 64, 128, 256, 512, 1024]
EH = [(1, '-', 'o'), (2, '--', 's'), (4, '-.', '^'), (8, ':', 'D')]
PRE = [('Green', 'Green', 'tab:green'), ('Green_Jacobi', 'Green-Jacobi', 'tab:blue'), ('dual', 'dual', 'tab:red')]
plt.rcParams.update({'font.size': 13})
fig, axs = plt.subplots(2, 2, figsize=(16, 12), layout='constrained')
for ax, k in zip(axs.flat, [100, 1000, 10000, 100000]):
    for P, lab, col in PRE:
        for e, ls, mk in EH:
            y = [W.get((k, n, e, P), np.nan) for n in GRIDS]
            ax.plot(GRIDS, y, mk + ls, color=col, lw=2, ms=7, mfc='white' if e > 2 else col,
                    label=f'{lab}, eta = {e} h' if k == 100 else None)
            g = [(n, W[(k, n, e, P)]) for n in GRIDS if (k, n, e, P) in GREY]
            if g:
                ax.plot(*zip(*g), 'x', color='k', ms=11, mew=2, zorder=5,
                        label='design ended grey' if (k, P, e) == (100, 'Green', 8) or not ax.get_legend_handles_labels()[1].count('design ended grey') and k == 100 else None)
    ax.set_xscale('log', base=2), ax.set_yscale('log')
    ax.set_xticks(GRIDS), ax.set_xticklabels([str(n) for n in GRIDS]), ax.minorticks_off()
    yt = [20, 50, 100, 200, 500, 1000]
    ax.set_yticks(yt), ax.set_yticklabels([str(v) for v in yt])
    ax.set_ylim(15, 1500)
    ax.set(title=f'contrast {k:.0e}', xlabel='grid N (N$^2$), eta/h fixed', ylabel='CG iterations per linear solve')
    ax.grid(alpha=0.3)
h, l = axs.flat[0].get_legend_handles_labels()
u = dict(zip(l, h))
fig.legend(u.values(), u.keys(), loc='outside lower center', ncol=7, fontsize=11)
fig.suptitle('Phase-field TO: CG per solve against the grid at fixed eta/h (raw iterations, dual not x 2)', fontsize=15)
out = os.path.join(HERE, 'figures', 'TO_cg_vs_grid_saturation.png')
fig.savefig(out, dpi=120)
print('wrote', out)
