"""
Green on the final TO designs (l = 2h), diffuse vs thresholded two-phase (threshold_designs_study.py ->
exp_data/threshold_designs_l2h.csv): CG per solve (cold start, energy stop 1e-6) against the grid, one shade per
contrast; diffuse solid with filled circles, two-phase dashed with open squares. Slide style (log y 20 - 2000).
    python plot_threshold_designs.py  ->  figures/threshold_designs_l2h.{png,pdf}
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
R = {}
for r in csv.DictReader(open(os.path.join(HERE, 'exp_data', 'threshold_designs_l2h.csv'))):
    if r['precond'] == 'Green':
        R[(int(float(r['contrast'])), r['material'], int(r['N']))] = float(r['cg_per_solve'])
GRIDS = [32, 64, 128, 256, 512, 1024]
plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots(figsize=(6.4, 4.8), layout='constrained')
shades = ['#9be29b', '#4fbf4f', '#228b22', '#0b4d0b']
for kap, col in zip([100, 1000, 10000, 100000], shades):
    for mat, ls, mk, mfc in (('diffuse', '-', 'o', col), ('two-phase', '--', 's', 'white')):
        pts = [(n, R[(kap, mat, n)]) for n in GRIDS if (kap, mat, n) in R]
        if pts:
            ax.plot(*zip(*pts), ls + mk, color=col, lw=2.6, ms=8, mfc=mfc, mew=2, clip_on=False, zorder=5)
    pts = [(n, R[(kap, 'diffuse', n)]) for n in GRIDS if (kap, 'diffuse', n) in R]
    above = kap in (100, 1000)
    ax.text(300, pts[-1][1] * 1.2 if above else pts[-1][1] / 1.22, f'contrast $10^{{{int(np.log10(kap))}}}$',
            color=col, ha='center', va='bottom' if above else 'top', fontsize=13)
tp = [R[k] for k in R if k[1] == 'two-phase']
ax.text(300, min(tp) / 1.25, 'two-phase (thresholded), all contrasts', color='k', ha='center', va='top', fontsize=12)
ax.plot([], [], 'k-o', lw=2.6, ms=8, label='diffuse design')
ax.plot([], [], 'k--s', lw=2.6, ms=8, mfc='white', mew=2, label='thresholded at $\\rho$ = 0.5')
ax.set_xscale('log', base=2)
ax.set_xticks(GRIDS), ax.set_xticklabels([f'{n}$^2$' for n in GRIDS]), ax.minorticks_off()
ax.set_xlim(32, 1024)
ax.set_yscale('log')
yt = [20, 50, 100, 200, 500, 1000, 2000]
ax.set_yticks(yt), ax.set_yticklabels([str(v) for v in yt]), ax.minorticks_off()
ax.set_ylim(20, 2000)
ax.set(xlabel='grid size', ylabel='CG iterations per solve (Green)')
ax.grid(alpha=0.3)
fig.legend(loc='outside upper center', ncol=2, fontsize=12, frameon=False)
out = os.path.join(HERE, 'figures', 'threshold_designs_l2h')
for ext in ('png', 'pdf'):
    fig.savefig(f'{out}.{ext}', dpi=150)
print('wrote', out)
