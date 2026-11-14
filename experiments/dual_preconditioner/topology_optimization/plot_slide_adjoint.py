"""
Adjoint solves of the phase-field TO (l = 2h, NEMO eta / contrast study): mean CG per ADJOINT solve against the grid,
one line per contrast, in the style of the slide figures (plot_slide_green_grid_independence.py,
plot_slide_gj_grid_dependence.py). Data: figures/nemo/mech_adjoint_l2h.csv, parsed from the run logs
($WS/runs/TO_study/logs/TO_N<N>[_soft1e-e]_eta<2/N>_<P>[_seg*].log: 'nb_steps CG adjoint' lines).
The Green-Jacobi 1024^2 contrast 1e5 run (71 solves, stalled, cancelled) is left out.
    python plot_slide_adjoint.py  ->  figures/slide_adjoint_{Green,Green_Jacobi,dual}.{png,pdf}
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
D = {}
for r in csv.DictReader(open(os.path.join(HERE, 'figures', 'nemo', 'mech_adjoint_l2h.csv'))):
    if int(r['n_adj']) < 100:
        continue
    D[(r['precond'], int(float(r['contrast'])), int(r['N']))] = (float(r['mech']), float(r['adjoint']))
GRIDS = [32, 64, 128, 256, 512, 1024]
SHADES = {'Green': ['#9be29b', '#4fbf4f', '#228b22', '#0b4d0b'],
          'Green_Jacobi': ['#9ecae1', '#4292c6', '#2171b5', '#08306b'],
          'dual': ['#fcae91', '#fb6a4a', '#de2d26', '#7f0000']}
NAME = {'Green': 'Green', 'Green_Jacobi': 'Green-Jacobi', 'dual': 'dual'}
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 13})
for P in ('Green', 'Green_Jacobi', 'dual'):
    fig, ax = plt.subplots(figsize=(4.4, 3.3), layout='constrained')
    for kap, col in zip([100, 1000, 10000, 100000], SHADES[P]):
        pts = [(n, D[(P, kap, n)][1]) for n in GRIDS if (P, kap, n) in D]
        if not pts:
            continue
        ax.plot(*zip(*pts), '-o', color=col, lw=2.2, ms=6, clip_on=False, zorder=5,
                label=f'contrast $10^{{{int(np.log10(kap))}}}$')
        print(P, kap, [round(v) for _, v in pts], 'adjoint/mech',
              [round(D[(P, kap, n)][1] / D[(P, kap, n)][0], 3) for n, _ in pts])
    ax.set_xscale('log', base=2)
    ax.set_xticks(GRIDS), ax.set_xticklabels([f'${n}^2$' for n in GRIDS]), ax.minorticks_off()
    ax.set_xlim(32, 1024)
    ax.set_yscale('log')
    yt = [20, 50, 100, 200, 500, 1000, 2000]
    ax.set_yticks(yt), ax.set_yticklabels([str(v) for v in yt]), ax.minorticks_off()
    ax.set_ylim(20, 2000)
    ax.set(xlabel='grid size', ylabel='CG iterations per adjoint solve')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    out = os.path.join(HERE, 'figures', f'slide_adjoint_{P}')
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}.{ext}', dpi=150)
    plt.close(fig)
print('wrote figures/slide_adjoint_*.png/.pdf')
