"""
Companion of plot_slide_green_grid_independence.py for Green-Jacobi (same style and axes): phase-field TO, CG iterations
per linear solve against the grid at l = 2h, one line per phase contrast (NEMO eta / contrast study,
figures/nemo/contrast_summary.csv). Green-Jacobi was run up to 1024^2 (contrast 1e5 only to 512^2), so no estimates.
Open markers: design ended grey (coarse grids).
    python plot_slide_gj_grid_dependence.py  ->  figures/slide_gj_grid_dependence.{png,pdf}
    PREC=dual python plot_slide_gj_grid_dependence.py: the dual (red shades), iterations counted x2 (one dual
        iteration ~ two Green iterations in 2D) -> figures/slide_dual_grid_dependence_x2.{png,pdf}
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
PREC = os.environ.get('PREC', 'Green_Jacobi')
FAC = 2.0 if PREC == 'dual' else 1.0
ETA = int(os.environ.get('ETA', 2))                                # l = ETA * h; ETA != 2 -> output name + _l<ETA>h
W, OK = {}, {}
for r in csv.DictReader(open(os.path.join(HERE, 'figures', 'nemo', 'contrast_summary.csv'))):
    if r['precond'] == PREC and int(float(r['eta_h'])) == ETA:
        k = (int(float(r['contrast'])), int(r['N']))
        W[k], OK[k] = FAC * float(r['cg']), r['design'] == 'True'
GRIDS = [32, 64, 128, 256, 512, 1024]
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 13})
fig, ax = plt.subplots(figsize=(4.4, 3.3), layout='constrained')
shades = ['#9ecae1', '#4292c6', '#2171b5', '#08306b'] if PREC != 'dual' else ['#fcae91', '#fb6a4a', '#de2d26', '#a50f15']
for (kap, col) in zip([100, 1000, 10000, 100000], shades):
    pts = [(n, W[(kap, n)]) for n in GRIDS if (kap, n) in W]
    ax.plot(*zip(*pts), '-', color=col, lw=2.2, clip_on=False, label=f'contrast $10^{{{int(np.log10(kap))}}}$')
    for n, v in pts:
        ax.plot(n, v, 'o', color=col, ms=6, mfc=col if OK.get((kap, n), True) else 'white', mew=1.6, clip_on=False,
                zorder=5)
    print(kap, [round(v) for _, v in pts])
ax.set_xscale('log', base=2)
ax.set_xticks(GRIDS), ax.set_xticklabels([f'{n}$^2$' for n in GRIDS]), ax.minorticks_off()
ax.set_xlim(32, 1024)
ax.set_yscale('log')
yt = [20, 50, 100, 200, 500, 1000, 2000]
ax.set_yticks(yt), ax.set_yticklabels([str(v) for v in yt]), ax.minorticks_off()
ax.set_ylim(20, 2000)
ax.set(xlabel='grid size', ylabel=r'CG per solve (dual $\times 2$)' if PREC == 'dual' else 'CG iterations per solve')
ax.grid(alpha=0.3)
ax.legend(loc='upper left', fontsize=10 if PREC == 'dual' else 11, ncol=2 if PREC == 'dual' else 1, columnspacing=0.8, handlelength=1.5, framealpha=0.95)
out = os.path.join(HERE, 'figures', ('slide_gj_grid_dependence' if PREC != 'dual' else 'slide_dual_grid_dependence_x2') + (f'_l{ETA}h' if ETA != 2 else ''))
for ext in ('png', 'pdf'):
    fig.savefig(f'{out}.{ext}', dpi=150)
print('wrote', out)
