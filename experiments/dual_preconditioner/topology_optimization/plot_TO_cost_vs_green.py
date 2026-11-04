"""
TO: cost of Green-Jacobi and dual relative to Green against the grid, one panel per phase contrast.

    y = CG work per linear solve / Green's work per linear solve (dual counted x2); 1 = Green (black line),
        below 1: cheaper than Green
    x = grid N^2 (32^2 ... 1024^2)
    blue = Green-Jacobi, red = dual; eta / h by shade, line style and marker (eta = 4 h thick: recommended)
Green was not run at 1024^2; its work per solve does not depend on the grid at fixed eta/h (flat from 64^2 on), so the
1024^2 points use Green's 512^2 value as baseline and are drawn open.

usage:  python plot_TO_cost_vs_green.py [contrast_summary.csv]
        default figures/nemo/contrast_summary.csv; writes TO_cost_vs_green.png next to it
"""
import csv
import os
import sys

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

HERE = os.path.dirname(os.path.realpath(__file__))
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'figures', 'nemo', 'contrast_summary.csv')
GRIDS = [32, 64, 128, 256, 512, 1024]
KAPPA = [100, 1000, 10000, 100000]
EH = {1: (':', 'v', 0.45, 1.4), 2: ('-.', 's', 0.65, 1.6), 4: ('-', 'o', 1.0, 3.0), 8: ('--', 'D', 0.8, 1.8)}
FAM = {'Green_Jacobi': plt.get_cmap('Blues'), 'dual': plt.get_cmap('Reds')}
LAB = {'Green_Jacobi': 'Green-Jacobi', 'dual': 'dual (x2)'}

work = {}
for r in csv.DictReader(open(CSV)):
    work[(int(r['N']), int(float(r['eta_h'])), int(float(r['contrast'])), r['precond'])] = float(r['work'])

plt.rcParams.update({'font.size': 13})
fig, axes = plt.subplots(1, len(KAPPA), figsize=(24, 6.8), sharey=True, layout='constrained')
for ax, k in zip(axes, KAPPA):
    ax.axhline(1, color='k', lw=2.5)
    ax.text(26, 1.06, 'Green', fontsize=13, fontweight='bold')
    for p, cmap in FAM.items():
        for eh, (ls, mk, shade, lw) in EH.items():
            col = cmap(0.35 + 0.6 * shade)
            xs, ys, est = [], [], []
            for n in GRIDS:
                v = work.get((n, eh, k, p))
                g = work.get((n, eh, k, 'Green'))
                e = False
                if g is None:
                    g, e = work.get((512, eh, k, 'Green')), True
                if v is None or g is None:
                    continue
                xs.append(n), ys.append(v / g), est.append(e)
            if not xs:
                continue
            ax.plot(xs, ys, ls=ls, color=col, lw=lw, zorder=3 if eh == 4 else 2)
            for x, y, e in zip(xs, ys, est):
                ax.plot(x, y, mk, color=col, ms=9 if eh == 4 else 7, mfc='w' if e else col, mec=col, mew=1.8,
                        zorder=4)
    ax.set_xscale('log', base=2), ax.set_yscale('log')
    ax.xaxis.set_major_locator(FixedLocator(GRIDS))
    ax.xaxis.set_major_formatter(FixedFormatter([f'{n}$^2$' for n in GRIDS]))
    ax.xaxis.set_minor_locator(NullLocator())
    yt = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
    ax.yaxis.set_major_locator(FixedLocator(yt)), ax.yaxis.set_major_formatter(FixedFormatter([f'{t:g}' for t in yt]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(0.04, 25), ax.set_xlim(24, 1400)
    ax.set_xlabel('grid N$^2$')
    ax.set_title(f'contrast $\\kappa$ = 1e{int(np.log10(k))}', fontsize=15)
    ax.grid(True, color='0.9')
    ax.text(0.97, 0.03, 'cheaper than Green', transform=ax.transAxes, ha='right', fontsize=12, color='0.4')
    ax.text(0.97, 0.95, 'more expensive than Green', transform=ax.transAxes, ha='right', va='top', fontsize=12,
            color='0.4')
axes[0].set_ylabel('CG work per solve / Green\'s')

handles, labels = [], []
for p, cmap in FAM.items():
    for eh, (ls, mk, shade, lw) in EH.items():
        handles.append(Line2D([], [], ls=ls, marker=mk, color=cmap(0.35 + 0.6 * shade), lw=lw, ms=8))
        labels.append(f'{LAB[p]}, $\\eta$ = {eh} h' + ('  (recommended)' if eh == 4 else ''))
handles.append(Line2D([], [], ls='', marker='o', mfc='w', mec='0.3', ms=9, mew=1.8))
labels.append('1024$^2$: Green baseline = its 512$^2$ value')
fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=12, bbox_to_anchor=(0.5, -0.17))
fig.suptitle('TO: cost of Green-Jacobi and dual relative to Green (= 1) against the grid', fontsize=16)
out = os.path.join(os.path.dirname(CSV), 'TO_cost_vs_green.png')
fig.savefig(out, dpi=120, bbox_inches='tight')
print('wrote', out)
