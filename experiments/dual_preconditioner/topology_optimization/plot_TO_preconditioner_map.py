"""
Which preconditioner is cheapest for a given phase-field length eta and phase contrast kappa?

One panel per grid (256^2, 512^2, 1024^2); in each panel the columns are eta (= 1, 2, 4, 8 h) and the rows the
contrast kappa = 1e2 ... 1e5. Every cell shows the CG work per linear solve (dual counted x2) of Green (G),
Green-Jacobi (GJ) and dual (D); the cell is tinted with the colour of the cheapest one (Green green, Green-Jacobi blue,
dual red), and 'x1.4' is how much the second best costs more. Thick black frame: the recommended setting
(4/N <= eta <= 1/128 and kappa = 1e4, see NOTES_eta_h_choice.md).
Green was not run at 1024^2; its work per solve does not depend on the grid at fixed eta/h, so the 512^2 value is
used there, marked '~'.

usage:  python plot_TO_preconditioner_map.py [contrast_summary.csv]
        default figures/nemo/contrast_summary.csv; writes TO_preconditioner_map.png next to it
"""
import csv
import os
import sys

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.realpath(__file__))
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'figures', 'nemo', 'contrast_summary.csv')
GRIDS = (256, 512, 1024)
EH = (1, 2, 4, 8)
KAPPA = (100, 1000, 10000, 100000)
PREC = [('Green', 'G', 'C2'), ('Green_Jacobi', 'GJ', 'C0'), ('dual', 'D', 'C3')]
ETA_MAX = 1 / 128

work = {}
for r in csv.DictReader(open(CSV)):
    work[(int(r['N']), int(float(r['eta_h'])), int(float(r['contrast'])), r['precond'])] = float(r['work'])

plt.rcParams.update({'font.size': 13})
fig, axes = plt.subplots(1, len(GRIDS), figsize=(22, 8.2), layout='constrained')
for ax, n in zip(axes, GRIDS):
    for i, eh in enumerate(EH):
        for j, k in enumerate(KAPPA):
            vals, est = {}, {}
            for p, short, col in PREC:
                v = work.get((n, eh, k, p))
                est[p] = False
                if v is None and p == 'Green':
                    v = work.get((512, eh, k, p))      # Green: flat in N at fixed eta/h
                    est[p] = v is not None
                if v is not None:
                    vals[p] = v
            order = sorted(vals, key=vals.get)
            if len(order) < 3:            # incomplete cell (runs not finished / cancelled): show what exists, no winner
                for m, (p, short, c) in enumerate(PREC):
                    txt = f'{short} {"~" if est.get(p) else ""}{vals[p]:.0f}' if p in vals else f'{short} -'
                    ax.text(i + 0.5, j + 0.78 - 0.2 * m, txt, ha='center', va='center', fontsize=12.5, color=c)
                ax.text(i + 0.5, j + 0.14, 'not complete', ha='center', va='center', fontsize=11, color='0.3')
                continue
            best = order[0]
            col = dict((p, c) for p, _, c in PREC)[best]
            ax.add_patch(Rectangle((i, j), 1, 1, color=col, alpha=0.22, lw=0))
            if est[best] or est.get(order[1]):   # the winner depends on an estimated (512^2) Green value
                ax.add_patch(Rectangle((i, j), 1, 1, fill=False, hatch='//', ec='0.55', lw=0))
            for m, (p, short, c) in enumerate(PREC):
                if p not in vals:
                    continue
                txt = f'{short} {"~" if est[p] else ""}{vals[p]:.0f}'
                ax.text(i + 0.5, j + 0.78 - 0.2 * m, txt, ha='center', va='center', fontsize=12.5, color=c,
                        fontweight='bold' if p == best else 'normal')
            ratio = vals[order[1]] / vals[best]
            ax.text(i + 0.5, j + 0.14, f'next x{ratio:.1f}', ha='center', va='center', fontsize=11, color='0.3')
            eta = eh / n
            if eh >= 4 and eta <= ETA_MAX * 1.001 and k == 10000:
                ax.add_patch(Rectangle((i + 0.03, j + 0.03), 0.94, 0.94, fill=False, ec='k', lw=3.5))
    for i in range(len(EH) + 1):
        ax.axvline(i, color='w', lw=2)
    for j in range(len(KAPPA) + 1):
        ax.axhline(j, color='w', lw=2)
    ax.set_xlim(0, len(EH)), ax.set_ylim(0, len(KAPPA))
    ax.set_xticks(np.arange(len(EH)) + 0.5)
    ax.set_xticklabels([f'{eh} h\n$\\eta$ = 1/{n // eh}' for eh in EH])
    ax.set_yticks(np.arange(len(KAPPA)) + 0.5)
    ax.set_yticklabels(['1e2', '1e3', '1e4', '1e5'])
    ax.set_xlabel('phase-field length $\\eta$')
    ax.set_ylabel('phase contrast $\\kappa$')
    ax.set_title(f'{n}$^2$' + ('   (Green ~ = 512$^2$ value)' if n == 1024 else ''), fontsize=15)
    ax.tick_params(length=0)
handles = [Rectangle((0, 0), 1, 1, color=c, alpha=0.35) for _, _, c in PREC] + \
          [Rectangle((0, 0), 1, 1, fill=False, ec='k', lw=3), Rectangle((0, 0), 1, 1, fill=False, hatch='//', ec='0.55', lw=0)]
fig.legend(handles, ['Green cheapest', 'Green-Jacobi cheapest', 'dual cheapest',
                     'recommended setting ($\\kappa$ = 1e4, 4/N $\\leq$ $\\eta$ $\\leq$ 1/128)',
                     'ranking uses the estimated Green (~)'],
           loc='lower center', ncol=5, fontsize=13, bbox_to_anchor=(0.5, -0.07))
fig.suptitle('TO: cheapest preconditioner by $\\eta$ and contrast  (CG work per linear solve, dual counted x2; '
             'bold = cheapest; "next" = second best / best)', fontsize=16)
out = os.path.join(os.path.dirname(CSV), 'TO_preconditioner_map.png')
fig.savefig(out, dpi=120, bbox_inches='tight')
print('wrote', out)
