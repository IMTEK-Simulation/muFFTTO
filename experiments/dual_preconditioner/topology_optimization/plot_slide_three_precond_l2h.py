"""All three preconditioners in one plot: phase-field TO at l = 2h (NEMO eta / contrast study,
figures/nemo/contrast_summary.csv), mean CG per solve against the grid, contrast 10^3 (solid) and 10^4 (dashed).
Green green, Green-Jacobi blue, dual red (dual counted x2: one dual iteration ~ two Green iterations in 2D).
Open markers: design ended grey (coarse grids). Green was not run at 1024^2.
    python plot_slide_three_precond_l2h.py  ->  figures/slide_three_precond_l2h.{png,pdf} (+ copy to the talk figures/)
"""
import csv
import os
import shutil

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
TALK = os.path.expanduser('~/Projects/Talks/Slides-Green-Jacobi-topology-optimization-NANTES/figures')
PRE = [('Green', 'Green', 'tab:green', 'o', 1), ('Green_Jacobi', 'Green-Jacobi', 'tab:blue', 's', 1),
       ('dual', r'dual ($\times 2$)', 'tab:red', '^', 2)]
GRIDS = [32, 64, 128, 256, 512, 1024]
W, OK = {}, {}
for r in csv.DictReader(open(os.path.join(HERE, 'figures', 'nemo', 'contrast_summary.csv'))):
    if int(float(r['eta_h'])) == 2:
        k = (r['precond'], int(float(r['contrast'])), int(r['N']))
        W[k], OK[k] = float(r['cg']), r['design'] == 'True'
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 13})
fig, ax = plt.subplots(figsize=(5.2, 3.9), layout='constrained')
for P, lab, col, mk, fac in PRE:
    for kap, ls in ((1000, '-'), (10000, '--')):
        pts = [(n, fac * W[(P, kap, n)]) for n in GRIDS if (P, kap, n) in W]
        ax.plot(*zip(*pts), ls, color=col, lw=2.2, clip_on=False, label=lab if kap == 1000 else None)
        for n, v in pts:
            ax.plot(n, v, mk, color=col, ms=6, mfc=col if OK[(P, kap, n)] else 'white', mew=1.6, clip_on=False,
                    zorder=5)
        print(f'{P:13s} contrast {kap:6d}', [(n, round(v)) for n, v in pts])
ax.plot([], [], 'k-', lw=2.2, label='contrast $10^3$')
ax.plot([], [], 'k--', lw=2.2, label='contrast $10^4$')
ax.set_xscale('log', base=2)
ax.set_xticks(GRIDS), ax.set_xticklabels([f'${n}^2$' for n in GRIDS]), ax.minorticks_off()
ax.set_xlim(32, 1024), ax.set_ylim(0, None)
ax.set(xlabel='grid size', ylabel=r'CG per solve (dual $\times 2$)')
ax.grid(alpha=0.3)
ax.legend(loc='upper left', fontsize=10, ncol=2, framealpha=0.95)
out = os.path.join(HERE, 'figures', 'slide_three_precond_l2h')
for ext in ('png', 'pdf'):
    fig.savefig(f'{out}.{ext}', dpi=150)
    shutil.copy(f'{out}.{ext}', TALK)
print('wrote', out)
