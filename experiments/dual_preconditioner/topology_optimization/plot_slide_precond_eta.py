"""
Slide plots in the style of the J-FFT slide (plot_slide_green_grid_independence.py / plot_slide_gj_grid_dependence.py)
for every preconditioner and every l = eta/h of the NEMO eta / contrast study (figures/nemo/contrast_summary.csv,
warm-started optimisation runs): CG iterations per linear solve against the grid, one line per contrast.
Measured points only (no 1024^2 estimate for Green); open markers: design ended grey.
    python plot_slide_precond_eta.py  ->  figures/slide_<Green|Green_Jacobi|dual>_l<1|2|4|8>h.{png,pdf}
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
W, OK = {}, {}
for r in csv.DictReader(open(os.path.join(HERE, 'figures', 'nemo', 'contrast_summary.csv'))):
    k = (r['precond'], int(float(r['eta_h'])), int(float(r['contrast'])), int(r['N']))
    W[k], OK[k] = float(r['cg']), r['design'] == 'True'
GRIDS = [32, 64, 128, 256, 512, 1024]
SHADES = {'Green': ['#9be29b', '#4fbf4f', '#228b22', '#0b4d0b'],
          'Green_Jacobi': ['#9ecae1', '#4292c6', '#2171b5', '#08306b'],
          'dual': ['#fcae91', '#fb6a4a', '#de2d26', '#7f0000']}
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 13})
for P in ('Green', 'Green_Jacobi', 'dual'):
    for eh in (1, 2, 4, 8):
        fig, ax = plt.subplots(figsize=(4.4, 3.3), layout='constrained')
        for kap, col in zip([100, 1000, 10000, 100000], SHADES[P]):
            pts = [(n, W[(P, eh, kap, n)]) for n in GRIDS if (P, eh, kap, n) in W]
            if not pts:
                continue
            ax.plot(*zip(*pts), '-', color=col, lw=2.2, clip_on=False,
                    label=f'contrast $10^{{{int(np.log10(kap))}}}$')
            for n, v in pts:
                ax.plot(n, v, 'o', color=col, ms=6, mew=1.6, clip_on=False, zorder=5,
                        mfc=col if OK[(P, eh, kap, n)] else 'white')
            if P == 'Green' and len(pts) >= 2:
                ax.text(np.sqrt(pts[-1][0] * pts[-2][0]), np.sqrt(pts[-1][1] * pts[-2][1]) / 1.22,
                        f'contrast $10^{{{int(np.log10(kap))}}}$', color=col, ha='center', va='top', fontsize=11)
        ax.set_xscale('log', base=2)
        ax.set_xticks(GRIDS), ax.set_xticklabels([f'${n}^2$' for n in GRIDS]), ax.minorticks_off()
        ax.set_xlim(32, 1024)
        ax.set_yscale('log')
        yt = [20, 50, 100, 200, 500, 1000, 2000]
        ax.set_yticks(yt), ax.set_yticklabels([str(v) for v in yt]), ax.minorticks_off()
        ax.set_ylim(20, 2000)
        ax.set(xlabel=f'grid size  ($l = {eh}h$)' if eh != 1 else 'grid size  ($l = h$)',
               ylabel='CG iterations per solve')
        ax.grid(alpha=0.3)
        if P != 'Green':
            ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
        out = os.path.join(HERE, 'figures', f'slide_{P}_l{eh}h')
        for ext in ('png', 'pdf'):
            fig.savefig(f'{out}.{ext}', dpi=150)
        plt.close(fig)
print('wrote figures/slide_<P>_l<eh>h.png/.pdf')
