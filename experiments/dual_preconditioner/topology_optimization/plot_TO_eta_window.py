"""
Acceptable eta for a grid N^2 (unit cell, h = 1/N):  4/N <= eta <= 0.035/P   (P = perimeter of the design; ~4.5 here)

    lower bound  eta >= 4 h = 4/N   interface resolved, design grid-converged (2/N: hard minimum)
    upper bound  eta <= 0.035 / P   interface band 4.4 eta P <~ 15 % of the cell: F_pf = perimeter / 3
The runs of the study (contrast 1e4, design_convergence.csv) are drawn on top: green inside the window, orange
under-resolved (eta < 4 h), purple band too thick (eta > 0.035 / P), grey cross: design ended grey.
See NOTES_eta_h_choice.md.

usage:  python plot_TO_eta_window.py [design_convergence.csv] [P]
        default figures/nemo/design_convergence.csv, P = 4.5; writes TO_eta_window.png next to the CSV
"""
import csv
import os
import sys

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

HERE = os.path.dirname(os.path.realpath(__file__))
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'figures', 'nemo', 'design_convergence.csv')
P = float(sys.argv[2]) if len(sys.argv) > 2 else 4.5
ETA_MAX = 0.035 / P
SOFT_EXP = 4

NS = np.array([32, 64, 128, 256, 512, 1024, 2048])
n = np.geomspace(16, 4096, 200)

plt.rcParams.update({'font.size': 14})
fig, ax = plt.subplots(figsize=(14, 8), layout='constrained')

# acceptable region and the two bounds
ax.fill_between(n, 4 / n, ETA_MAX, where=4 / n <= ETA_MAX, color='C2', alpha=0.18, lw=0)
ax.plot(n, 4 / n, color='C1', lw=2.5, label='lower bound  $\\eta$ = 4/N   (interface resolved)')
ax.plot(n, 2 / n, color='C1', lw=1.5, ls='--', label='hard minimum  $\\eta$ = 2/N')
ax.axhline(ETA_MAX, color='C4', lw=2.5,
           label=f'upper bound  $\\eta$ = 0.035/P $\\approx$ 1/128   (band < 15 % of cell, P = {P:g})')
ax.text(2000, ETA_MAX * 0.72, 'acceptable', color='C2', fontsize=16, fontweight='bold', ha='center')
ax.text(30, 1 / 1250, 'below 4/N: interface not resolved (design depends on the grid)', color='C1', fontsize=13)
ax.text(2700, 1 / 40, 'above 0.035/P: interface band too thick\n(F$_{pf}$ is not the perimeter)', color='C4', fontsize=13,
        ha='right')

# runs of the study
rows = [r for r in csv.DictReader(open(CSV)) if int(float(r['soft_exp'])) == SOFT_EXP]
done = set()
for r in rows:
    N, eta, eh = int(r['N']), float(r['eta']), float(r['eta_h'])
    if r['design'] != 'True':
        ax.plot(N, eta, 'x', color='0.5', ms=12, mew=2.5)
        continue
    col = 'C2' if eh >= 4 and eta <= ETA_MAX * 1.01 else ('C1' if eh < 4 else 'C4')
    ax.plot(N, eta, 'o', color=col, mec='k', ms=11, zorder=3)
    ax.annotate(f'R={float(r["R"]):.2f}', (N, eta), textcoords='offset points', xytext=(9, 5), fontsize=10,
                color='0.3')
for N, eta in ((512, 1 / 128), (1024, 1 / 256)):
    ax.plot(N, eta, 'o', ms=22, mfc='none', mec='k', mew=2, zorder=4)
ax.annotate('recommended:\n512$^2$, $\\eta$ = 1/128\n1024$^2$, $\\eta$ = 1/256', (1024, 1 / 256),
            textcoords='offset points', xytext=(-210, -95), fontsize=12,
            arrowprops=dict(arrowstyle='->', color='k'))

ax.set_xscale('log', base=2), ax.set_yscale('log', base=2)
ax.xaxis.set_major_locator(FixedLocator(NS)), ax.xaxis.set_major_formatter(FixedFormatter([f'{v}$^2$' for v in NS]))
ax.xaxis.set_minor_locator(NullLocator())
ET = [1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256, 1 / 512, 1 / 1024]
ax.yaxis.set_major_locator(FixedLocator(ET)), ax.yaxis.set_major_formatter(FixedFormatter([f'1/{round(1 / e)}' for e in ET]))
ax.yaxis.set_minor_locator(NullLocator())
ax.set_xlim(24, 2900), ax.set_ylim(1 / 1500, 1 / 6)
ax.set_xlabel('grid N$^2$  (h = 1/N, unit cell)')
ax.set_ylabel('phase-field length $\\eta$')
ax.grid(True, color='0.9')
h_run = [ax.plot([], [], 'o', color=c, mec='k', ms=10)[0] for c in ('C2', 'C1', 'C4')] + \
        [ax.plot([], [], 'x', color='0.5', ms=10, mew=2.5)[0]]
leg1 = ax.legend(loc='upper right', fontsize=12)
ax.add_artist(leg1)
ax.legend(h_run, ['run inside the window', 'run with $\\eta$ < 4h', 'run with band too thick', 'design ended grey'],
          loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=12, title=f'runs (contrast 1e{SOFT_EXP}; R = 3F$_{{pf}}$/P)', title_fontsize=12)
ax.set_title(f'Acceptable phase-field length:   4/N  $\\leq$  $\\eta$  $\\leq$  0.035/P  ($\\approx$ 1/128)'
             f'   $\\Rightarrow$   N $\\geq$ 115 P $\\approx$ {115 * P:.0f}', fontsize=15)
out = os.path.join(os.path.dirname(CSV), 'TO_eta_window.png')
fig.savefig(out, dpi=130)
print('wrote', out)
