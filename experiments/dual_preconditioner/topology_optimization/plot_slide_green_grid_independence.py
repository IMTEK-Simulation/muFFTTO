"""
Slide 'Green's operator preconditioning -> grid size independence', right panel (draft): phase-field TO, Green only,
CG iterations per linear solve against the grid at l = 2h, one line per phase contrast (NEMO eta / contrast study,
figures/nemo/contrast_summary.csv). Message: independent of the grid size, set by the phase contrast.
Open markers: design ended grey (coarse grids). 1024^2 (Green not run): dashed, open diamond = mean of 128^2-512^2.
    python plot_slide_green_grid_independence.py  ->  figures/slide_green_grid_independence.{png,pdf}
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
ETA = int(os.environ.get('ETA', 2))                                # l = ETA * h; ETA != 2 -> output name + _l<ETA>h
W, OK = {}, {}
for r in csv.DictReader(open(os.path.join(HERE, 'figures', 'nemo', 'contrast_summary.csv'))):
    if r['precond'] == 'Green' and int(float(r['eta_h'])) == ETA:
        k = (int(float(r['contrast'])), int(r['N']))
        W[k], OK[k] = float(r['cg']), r['design'] == 'True'
GRIDS = [32, 64, 128, 256, 512]
TICKS = GRIDS + [1024]
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 13})
fig, ax = plt.subplots(figsize=(4.4, 3.3), layout='constrained')
shades = ['#9be29b', '#4fbf4f', '#228b22', '#0b4d0b']
for (kap, col) in zip([100, 1000, 10000, 100000], shades):
    y = [W.get((kap, n), np.nan) for n in GRIDS]
    ax.plot(GRIDS, y, '-', color=col, lw=2.2, clip_on=False)
    for n, v in zip(GRIDS, y):
        ax.plot(n, v, 'o', color=col, ms=6, mfc=col if OK.get((kap, n), True) else 'white', mew=1.6, clip_on=False, zorder=5)
    est = np.mean(y[2:])                       # 1024^2 not run: plateau estimate = mean of 128^2-512^2
    ax.plot([512, 1024], [y[-1], est], '-', color=col, lw=2.2, clip_on=False)
    ax.plot(1024, est, 'o', color=col, ms=6, mfc='white', mew=1.6, clip_on=False, zorder=5)
    ax.text(430, np.sqrt(y[-1] * y[-2]) / 1.22, f'contrast $10^{{{int(np.log10(kap))}}}$', color=col, ha='center',
            va='top', fontsize=12)
ax.set_xscale('log', base=2)
ax.set_xticks(TICKS), ax.set_xticklabels([f'{n}$^2$' for n in TICKS]), ax.minorticks_off()
ax.set_xlim(32, 1024)
ax.set_yscale('log')
yt = [20, 50, 100, 200, 500, 1000, 2000]
ax.set_yticks(yt), ax.set_yticklabels([str(v) for v in yt]), ax.minorticks_off()
ax.set_ylim(20, 2000)
ax.set(xlabel='grid size', ylabel='CG iterations per solve')
ax.grid(alpha=0.3)
out = os.path.join(HERE, 'figures', 'slide_green_grid_independence' + (f'_l{ETA}h' if ETA != 2 else ''))
for ext in ('png', 'pdf'):
    fig.savefig(f'{out}.{ext}', dpi=150)
print('wrote', out)
