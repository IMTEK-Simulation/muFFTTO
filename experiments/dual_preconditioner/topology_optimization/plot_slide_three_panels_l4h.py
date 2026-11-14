"""Slides 'J-FFT in topology optimization' at l = 4h (ETA=2: l = 2h): three separate panels (Green, Green-Jacobi, dual counted x2), same
style and axes as the l = 2h plots, sized to stand next to each other. Phase-field TO, CG iterations per linear solve
against the grid, one line per phase contrast (NEMO eta / contrast study, figures/nemo/contrast_summary.csv).
Open markers: design ended grey (coarse grids). Green was not run at 1024^2 (no estimate drawn).
    python plot_slide_three_panels_l4h.py  ->  figures/slide_{green,gj,dual}_grid_l<ETA>h.{png,pdf} (+ copy to the talk figures/)
"""
import csv
import os
import shutil

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
TALK = os.path.expanduser('~/Projects/Talks/Slides-Green-Jacobi-topology-optimization-NANTES/figures')
PANELS = [('Green', 'green', 1, ['#9be29b', '#4fbf4f', '#228b22', '#0b4d0b']),
          ('Green_Jacobi', 'gj', 1, ['#9ecae1', '#4292c6', '#2171b5', '#08306b']),
          ('dual', 'dual', 2, ['#fcae91', '#fb6a4a', '#de2d26', '#a50f15'])]
ETA = int(os.environ.get('ETA', 4))                                # l = ETA * h (2 or 4)
GRIDS = [32, 64, 128, 256, 512, 1024]
ROWS = list(csv.DictReader(open(os.path.join(HERE, 'figures', 'nemo', 'contrast_summary.csv'))))
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 12})
for P, tag, fac, shades in PANELS:
    W, OK = {}, {}
    for r in ROWS:
        if r['precond'] == P and int(float(r['eta_h'])) == ETA:
            k = (int(float(r['contrast'])), int(r['N']))
            W[k], OK[k] = fac * float(r['cg']), r['design'] == 'True'
    # same plot box (in inches) in all panels; only the left panel keeps the margin for the y axis. Include with equal height
    AW, AH, B, R, FH, L = 2.2, 1.7, 0.73, 0.16, 3.45, 0.775 if P == 'Green' else 0.2
    FW = L + AW + R
    fig = plt.figure(figsize=(FW, FH))
    ax = fig.add_axes([L / FW, B / FH, AW / FW, AH / FH])
    fig.text((L + AW / 2) / FW, 0.965, {'Green': r'\textbf{Green}', 'Green_Jacobi': r'\textbf{Green-Jacobi}',
             'dual': r'\textbf{dual} (counted $\times 2$)'}[P], ha='center', va='top', fontsize=13)
    for kap, col in zip([100, 1000, 10000, 100000], shades):
        pts = [(n, W[(kap, n)]) for n in GRIDS if (kap, n) in W]
        ax.plot(*zip(*pts), '-', color=col, lw=2.2, clip_on=False, label=f'$10^{{{int(np.log10(kap))}}}$')
        for n, v in pts:
            ax.plot(n, v, 'o', color=col, ms=5, mfc=col if OK[(kap, n)] else 'white', mew=1.4, clip_on=False, zorder=5)
        print(P, kap, [round(v) for _, v in pts])
    ax.set_xscale('log', base=2)
    ax.set_xticks(GRIDS), ax.set_xticklabels([f'{n}$^2$' for n in GRIDS], fontsize=12, rotation=45), ax.minorticks_off()
    ax.set_xlim(32, 1024)
    ax.set_yscale('log')
    yt = [20, 50, 100, 200, 500, 1000, 2000]
    ax.set_yticks(yt), ax.set_yticklabels([str(v) for v in yt]), ax.minorticks_off()
    ax.set_ylim(20, 2000)
    ax.set(xlabel='grid size', ylabel='CG iterations per solve' if P == 'Green' else '')   # y label on the left panel only
    ax.grid(alpha=0.3)
    if P != 'Green':                                               # one y axis (left panel) is enough
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax.legend(title='contrast', loc='lower center', bbox_to_anchor=(0.5, 1.0), fontsize=12, title_fontsize=12, ncol=4,
              columnspacing=0.7, handlelength=1.0, handletextpad=0.3, frameon=False)
    out = os.path.join(HERE, 'figures', f'slide_{tag}_grid_l{ETA}h')
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}.{ext}', dpi=200)
        shutil.copy(f'{out}.{ext}', TALK)
    print('wrote', out)
