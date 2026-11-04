"""
How stiff must the 'void' be? Effect of the phase contrast k (stiff / soft phase) on the TO result and on the cost.

    left    stiffness error |C - C*| / |C*| of the final design against k (colour: physical eta; solid eta = 4 h,
            dashed eta = 8 h -- the same eta on two grids coincide: grid-converged)
    middle  perimeter P of the final design (length of the rho = 0.5 contour, cell side = 1) against k
    right   CG work per linear solve (dual counted x2) against k at eta = 4 h, 512^2 and 1024^2:
            Green (green), Green-Jacobi (blue), dual (red); Green was not run at 1024^2
Open markers: design ended grey. Shaded: contrasts at which the design no longer changes (>= 1e4).

usage:  python plot_TO_contrast_effect.py [folder with design_convergence.csv and contrast_summary.csv]
        default figures/nemo (copies from $MUFFTTO_WS/runs/TO_study/analysis_contrast on NEMO);
        writes TO_contrast_effect.png into that folder.  Notes: NOTES_eta_h_choice.md
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
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'figures', 'nemo')
K = [1e2, 1e3, 1e4, 1e5]

conv = list(csv.DictReader(open(os.path.join(D, 'design_convergence.csv'))))
summ = list(csv.DictReader(open(os.path.join(D, 'contrast_summary.csv'))))

plt.rcParams.update({'font.size': 13})
fig, (ax_c, ax_p, ax_w) = plt.subplots(1, 3, figsize=(21, 6.4), layout='constrained')

# left / middle: final design against contrast; colour = physical eta, line style = eta/h (grid convergence)
eta_col = {32: 'C4', 64: 'C1', 128: 'C5', 256: 'C9'}
eh_ls = {4: ('-', 'o'), 8: ('--', 's')}
for n in (256, 512, 1024):
    for eh, (ls, mk) in eh_ls.items():
        s = sorted((r for r in conv if int(r['N']) == n and float(r['eta_h']) == eh),
                   key=lambda r: int(float(r['soft_exp'])))
        if len(s) < 2:
            continue
        inv = round(n / eh)
        col = eta_col[inv]
        k = [10 ** int(float(r['soft_exp'])) for r in s]
        for ax, key in ((ax_c, 'C_err'), (ax_p, 'P')):
            y = [float(r[key]) for r in s]
            ax.plot(k, y, ls=ls, marker=mk, color=col, lw=2.5 if eh == 4 else 1.8, ms=9 if eh == 4 else 7,
                    label=f'$\\eta$ = 1/{inv}: {n}$^2$ ({eh} h)')
            for ki, yi, r in zip(k, y, s):
                if r['design'] != 'True':
                    ax.plot(ki, yi, marker=mk, ms=11, mfc='w', mec=col, mew=2)

# right: cost per solve at eta = 4 h; preconditioner colours, grid by line style
PREC = [('Green', 'C2', 'Green'), ('Green_Jacobi', 'C0', 'Green-Jacobi'), ('dual', 'C3', 'dual (x2)')]
for n, ls, mk in ((512, '--', 's'), (1024, '-', 'o')):
    for p, col, lab in PREC:
        s = sorted((r for r in summ if int(r['N']) == n and float(r['eta_h']) == 4 and r['precond'] == p),
                   key=lambda r: float(r['contrast']))
        if len(s) < 2:
            continue
        ax_w.plot([float(r['contrast']) for r in s], [float(r['work']) for r in s], ls=ls, marker=mk, color=col,
                  lw=2, ms=8, label=f'{lab}, {n}$^2$')

for ax in (ax_c, ax_p, ax_w):
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(FixedLocator(K)), ax.xaxis.set_major_formatter(FixedFormatter(['1e2', '1e3', '1e4', '1e5']))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(60, 1.6e5)
    ax.axvspan(7e3, 1.6e5, color='C2', alpha=0.07, zorder=0)
    ax.set_xlabel('phase contrast k = E$_{stiff}$ / E$_{soft}$')
    ax.grid(True, color='0.9')

ax_c.set_ylim(0, 0.4)
ax_c.set_ylabel('stiffness error |C - C*| / |C*| of the design')
ax_c.set_title('Does the design depend on how soft the "void" is?\n1e2: yes (soft phase carries load); from 1e4: no')
ax_c.text(1.1e4, 0.37, 'design converged in k', color='C2', fontsize=12)
ax_c.legend(fontsize=11, loc='lower left', ncol=2, title='colour: $\\eta$; solid 4 h, dashed 8 h',
             title_fontsize=11)
ax_p.set_ylim(3.6, 5.2)
ax_p.set_ylabel('perimeter P of the design (cell side = 1)')
ax_p.set_title('Perimeter of the design\n1e2: differs by up to 20 %; 1e3 - 1e5 within 2 %')
ax_w.set_yscale('log')
yt = [50, 100, 200, 500, 1000, 2000]
ax_w.yaxis.set_major_locator(FixedLocator(yt)), ax_w.yaxis.set_major_formatter(FixedFormatter([f'{t}' for t in yt]))
ax_w.yaxis.set_minor_locator(NullLocator())
ax_w.set_ylim(40, 2000)
ax_w.set_ylabel('CG work per linear solve')
ax_w.set_title('Cost of a stiffer contrast ($\\eta$ = 4 h)\nall preconditioners pay; 1e5 costs 2-3x of 1e4')
ax_w.legend(fontsize=11, loc='upper left', ncol=2)
fig.suptitle('Effect of the phase contrast in the phase-field TO: contrast 1e4 is converged (1e3 nearly), 1e5 only '
             'costs more', fontsize=15)
out = os.path.join(D, 'TO_contrast_effect.png')
fig.savefig(out, dpi=130)
print('wrote', out)
