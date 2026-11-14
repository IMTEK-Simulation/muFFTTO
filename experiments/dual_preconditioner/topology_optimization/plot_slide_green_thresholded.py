"""
Twin of plot_slide_green_grid_independence.py (same style, axes, colours) for the THRESHOLDED designs: the final TO
designs at l = 2h made two-phase (rho_q > 0.5 -> C_0, else k C_0), Green CG per solve from zero for the 3 load cases
(threshold_designs_study.py -> exp_data/threshold_designs_l2h.csv). The four contrast lines lie on top of each other,
so one label. 1024^2 uses the dual run's design (Green TO was not run there).
    python plot_slide_green_thresholded.py  ->  figures/slide_green_thresholded.{png,pdf}
    PREC=dual python plot_slide_green_thresholded.py [diffuse|two-phase]: the dual (red shades), its iterations
        counted x2 (one dual iteration ~ two Green iterations in 2D) -> figures/slide_dual_{thresholded,diffuse_cold}_x2
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
import sys
MATERIAL = sys.argv[1] if len(sys.argv) > 1 else 'two-phase'      # 'two-phase' or 'diffuse'
PREC = os.environ.get('PREC', 'Green')
FAC = 2.0 if PREC == 'dual' else 1.0
R = {}
for r in csv.DictReader(open(os.path.join(HERE, 'exp_data', 'threshold_designs_l2h.csv'))):
    if r['precond'] == PREC and r['material'] == MATERIAL:
        R[(int(float(r['contrast'])), int(r['N']))] = FAC * float(r['cg_per_solve'])
GRIDS = [32, 64, 128, 256, 512, 1024]
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 13})
fig, ax = plt.subplots(figsize=(4.4, 3.3), layout='constrained')
shades = ['#9be29b', '#4fbf4f', '#228b22', '#0b4d0b'] if PREC == 'Green' else ['#fcae91', '#fb6a4a', '#de2d26', '#a50f15']
lo = []
for kap, col in zip([100, 1000, 10000, 100000], shades):
    pts = [(n, R[(kap, n)]) for n in GRIDS if (kap, n) in R]
    ax.plot(*zip(*pts), '-', color=col, lw=2.2, clip_on=False)
    ax.plot(*zip(*pts), 'o', color=col, ms=6, mew=1.6, clip_on=False, zorder=5)
    lo.append(min(v for _, v in pts))
    print(kap, [round(v) for _, v in pts])
    if MATERIAL == 'diffuse':
        ax.text(430, np.sqrt(pts[-1][1] * pts[-2][1]) / 1.22, f'$\\kappa_{{\\mathrm{{void}}}} = 10^{{-{int(np.log10(kap))}}}$', color=col,
                ha='center', va='top', fontsize=12)
if MATERIAL == 'two-phase':
    ax.text(430, min(lo) / 1.22, r'$\kappa_{\mathrm{void}} = 10^{-2}$ -- $10^{-5}$', color=shades[2], ha='center', va='top', fontsize=12)
ax.set_xscale('log', base=2)
ax.set_xticks(GRIDS), ax.set_xticklabels([f'{n}$^2$' for n in GRIDS]), ax.minorticks_off()
ax.set_xlim(32, 1024)
ax.set_yscale('log')
yt = [20, 50, 100, 200, 500, 1000, 2000]
ax.set_yticks(yt), ax.set_yticklabels([str(v) for v in yt]), ax.minorticks_off()
ax.set_ylim(20, 2000)
ax.set(xlabel='grid size', ylabel=r'CG per solve (dual $\times 2$)' if PREC == 'dual' else 'CG iterations per solve')
ax.grid(alpha=0.3)
out = os.path.join(HERE, 'figures', 'slide_green_thresholded' if MATERIAL == 'two-phase' else 'slide_green_diffuse_cold')
if PREC == 'dual':
    out = os.path.join(HERE, 'figures', 'slide_dual_thresholded_x2' if MATERIAL == 'two-phase' else 'slide_dual_diffuse_cold_x2')
for ext in ('png', 'pdf'):
    fig.savefig(f'{out}.{ext}', dpi=150)
print('wrote', out)
