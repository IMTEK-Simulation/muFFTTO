"""
Slide figure (GAMM Freiburg 2026): the problem of the Green preconditioner in 2D topology optimization.
Green alone, contrast 1e2 ... 1e5, eta = 4h, CG iterations per linear solve averaged over the whole optimisation
(figures/nemo/contrast_summary.csv, NEMO runs).

    left    iterations against the grid, one line per contrast: flat (grid independent) ...
    right   ... but the level grows with the contrast (512^2), fitted slope ~ contrast^beta

usage:  python plot_slide_TO_green_problem.py     writes <talk>/figures/slide_TO_green_problem.{pdf,png}
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
OUT = os.path.expanduser('~/Projects/Talks/Slides-Green-Jacobi-topology-optimization-NANTES/figures/slide_TO_green_problem')
ETA_H = 4.0
CONTRASTS = [1e2, 1e3, 1e4, 1e5]
SHADES = plt.cm.Greens(np.linspace(0.45, 1.0, len(CONTRASTS)))
plt.rcParams.update({'font.size': 17})

cg = {}
with open(os.path.join(HERE, 'figures', 'nemo', 'contrast_summary.csv')) as f:
    for r in csv.DictReader(f):
        if r['precond'] == 'Green' and float(r['eta_h']) == ETA_H:
            cg[(float(r['contrast']), int(r['N']))] = float(r['cg'])
grids = sorted({n for _, n in cg})

fig, (a, b) = plt.subplots(1, 2, figsize=(13.3, 5.6), layout='constrained', gridspec_kw={'width_ratios': [1.25, 1]})
for c, col in zip(CONTRASTS, SHADES):
    ns = [n for n in grids if (c, n) in cg]
    a.plot(ns, [cg[(c, n)] for n in ns], 'o-', color=col, lw=3, ms=9, label=f'contrast $10^{{{int(np.log10(c))}}}$')
a.set_xscale('log', base=2)
a.set_xticks(grids), a.set_xticklabels([f'{n}$^2$' for n in grids])
a.minorticks_off()
a.set_ylim(0, None)
a.set(xlabel='grid', ylabel='CG iterations per solve')
a.set_title('grid independent ...')
a.legend(loc='upper left', fontsize=15, framealpha=0.95)
a.grid(alpha=0.3)

N = 512
cs = np.array([c for c in CONTRASTS if (c, N) in cg]); its = np.array([cg[(c, N)] for c in cs])
beta, logk = np.polyfit(np.log10(cs), np.log10(its), 1)
b.plot(cs, its, 'o', color=SHADES[-1], ms=12, zorder=3)
cc = np.logspace(2, 5, 50)
b.plot(cc, 10 ** logk * cc ** beta, '--', color='0.4', lw=2, label=f'~ contrast$^{{{beta:.2f}}}$')
b.set_xscale('log'), b.set_yscale('log')
b.set_xticks(CONTRASTS), b.set_xticklabels(['$10^2$', '$10^3$', '$10^4$', '$10^5$'])
yt = [50, 100, 200, 500, 1000]
b.set_yticks(yt), b.set_yticklabels([str(t) for t in yt]); b.minorticks_off()
b.set(xlabel='phase contrast', ylabel='CG iterations per solve')
b.set_title(f'... but contrast dependent ({N}$^2$)')
b.legend(loc='upper left', fontsize=15)
b.grid(alpha=0.3)
fig.suptitle('Green preconditioner in topology optimization ($\\eta$ = 4h)', fontsize=20)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}.{ext}', dpi=150)
print(f'beta = {beta:.3f}', dict(zip(cs, its)), '\nwrote', OUT + '.{pdf,png}')
