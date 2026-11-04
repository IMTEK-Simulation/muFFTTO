"""
When is the phase-field term of the TO objective a perimeter penalty, and how to choose eta and h
(readable version of TO_perimeter_indicator.png of analyze_TO_design_convergence.py; notes: NOTES_eta_h_choice.md).

Phase-field term  F_pf = eta int |grad rho|^2 + 1/eta int rho^2 (1-rho)^2.  For a thin, resolved two-phase design
F_pf -> P/3 (Modica-Mortola; P = length of the rho = 0.5 contour) and the 0.1-0.9 band is 2 ln 9 eta = 4.39 eta wide.

    left    R = 3 F_pf / P against the interface band area 4.39 eta P (fraction of the unit cell): R = 1 means F_pf is a
            perimeter penalty; the condition is a THIN band (eta small against the feature size)
    middle  band width / eta against the same: the profile has its natural width only in thin bands
    right   design difference to the finest grid of the same eta, RMS(rho_N - rho_finest), against eta / h: the
            condition is a RESOLVED band (eta >= 4 h)
One marker per final design (all contrasts 1e2..1e5, all grids 32^2..1024^2, eta 1/4..1/1024); grey crosses: designs
that ended grey (no two-phase split), left out of the trends.

usage:  python plot_TO_perimeter_indicator.py [design_convergence.csv]
        default: figures/nemo/design_convergence.csv (copied from $MUFFTTO_WS/runs/TO_study/analysis_contrast on NEMO);
        writes TO_perimeter_indicator_readable.png next to the CSV
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
W_THEORY = 2 * np.log(9)

rows = []
for r in csv.DictReader(open(CSV)):
    q = {k: (float(v) if k not in ('precond', 'design') else v) for k, v in r.items()}
    q['design'] = r['design'] == 'True'
    q['band'] = 100 * W_THEORY * q['eta'] * q['P']          # interface band area, % of the unit cell
    rows.append(q)
good = [r for r in rows if r['design'] and np.isfinite(r['R'])]
grey = [r for r in rows if not r['design'] and np.isfinite(r['R']) and r['P'] > 0]

RES = [1, 2, 4, 8, 16, 32]                                  # eta / h
res_col = dict(zip(RES, plt.get_cmap('viridis')(np.linspace(0.0, 0.9, len(RES)))))
CON = {2: 'v', 3: 's', 4: 'D', 5: 'o'}
ETAS = sorted({r['eta'] for r in rows}, reverse=True)
eta_col = dict(zip(ETAS, plt.get_cmap('plasma')(np.linspace(0.0, 0.85, len(ETAS)))))

plt.rcParams.update({'font.size': 13})
fig, (ax_r, ax_w, ax_d) = plt.subplots(1, 3, figsize=(21, 6.6), layout='constrained')

for ax, key in ((ax_r, 'R'), (ax_w, 'w')):
    ax.axvspan(0, 16, color='C2', alpha=0.08, zorder=0)
    for r in good:
        y = r['R'] if key == 'R' else r['w_n'] / r['eta']
        ax.scatter(r['band'], y, s=60, marker=CON[int(r['soft_exp'])], color=res_col[int(round(r['eta_h']))],
                   edgecolors='k', linewidths=0.5, zorder=3)
    for r in grey:
        y = r['R'] if key == 'R' else r['w_n'] / r['eta']
        ax.scatter(r['band'], y, s=60, marker='x', color='0.6', zorder=2)
    ax.set_xscale('log')
    ticks = [1, 2, 5, 10, 20, 50, 100]
    ax.xaxis.set_major_locator(FixedLocator(ticks)), ax.xaxis.set_major_formatter(FixedFormatter([f'{t}' for t in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(0.5, 200)
    ax.set_xlabel('interface band area 4.4 $\\eta$ P  [% of the cell]')
    ax.grid(True, color='0.9')
ax_r.axhline(1, color='k', ls='--', lw=1.2)
ax_r.axhspan(0.98, 1.02, color='k', alpha=0.06)
ax_r.set_ylim(0.7, 1.2)
ax_r.set_ylabel('R = 3 F$_{pf}$ / P   (1 = perimeter penalty)')
ax_r.set_title('Is the phase-field term a perimeter penalty?\nyes (R = 1 $\\pm$ 2 %) when the band covers < ~16 % of the cell')
ax_r.text(0.7, 1.17, 'thin band', color='C2', fontsize=13)
ax_w.axhline(W_THEORY, color='k', ls='--', lw=1.2)
ax_w.text(0.6, W_THEORY + 0.05, 'theory 2 ln 9 = 4.39', fontsize=12)
ax_w.set_ylim(2.5, 4.7)
ax_w.set_ylabel('interface width (0.1 - 0.9) / $\\eta$')
ax_w.set_title('Does the interface keep its natural width?\nonly in thin bands; wide bands are squeezed')

# right: grid convergence of the design at fixed eta (resolution condition)
for eta in ETAS:
    ser = sorted((r for r in rows if r['eta'] == eta and int(r['soft_exp']) == 5 and np.isfinite(r['d_ref'])
                  and r['d_ref'] > 1e-12 and r['design']), key=lambda r: r['N'])
    if not ser or any(not r['design'] for r in rows if r['eta'] == eta and int(r['soft_exp']) == 5):
        continue                  # eta whose designs (or reference) ended grey: the difference means nothing
    x = [r['eta_h'] for r in ser]
    y = [r['d_ref'] for r in ser]
    ax_d.plot(x, y, '-o', color=eta_col[eta], lw=2, ms=8, label=f'$\\eta$ = 1/{round(1 / eta)}')
ax_d.axvspan(2, 64, color='C2', alpha=0.08, zorder=0)
ax_d.set_xscale('log', base=2), ax_d.set_yscale('log')
ax_d.xaxis.set_major_locator(FixedLocator(RES)), ax_d.xaxis.set_major_formatter(FixedFormatter([f'{t}' for t in RES]))
ax_d.xaxis.set_minor_locator(NullLocator())
yt = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
ax_d.yaxis.set_major_locator(FixedLocator(yt)), ax_d.yaxis.set_major_formatter(FixedFormatter([f'{t:g}' for t in yt]))
ax_d.yaxis.set_minor_locator(NullLocator())
ax_d.set_xlim(0.8, 40), ax_d.set_ylim(0.01, 0.5)
ax_d.set_xlabel('$\\eta$ / h  (pixels per $\\eta$)')
ax_d.set_ylabel('RMS($\\rho_N$ - $\\rho_{finest}$)  (same $\\eta$)')
ax_d.set_title('Is the design grid-converged? (contrast 1e5)\n'
               '1 h: off by 10-30 %; 2-4 h: 2-9 %; 8 h: 2 %')
ax_d.grid(True, color='0.9'), ax_d.legend(fontsize=12, loc='upper right')
ax_d.text(2.15, 0.012, 'resolved', color='C2', fontsize=13)

# shared legends for the left/middle panels: colour = eta/h, marker = contrast
h_res = [ax_r.scatter([], [], s=60, marker='o', color=res_col[k], edgecolors='k', linewidths=0.5) for k in RES]
h_con = [ax_r.scatter([], [], s=60, marker=CON[e], color='w', edgecolors='k') for e in CON]
h_grey = ax_r.scatter([], [], s=60, marker='x', color='0.6')
ax_r.legend(h_res, [f'$\\eta$/h = {k}' for k in RES], title='colour', fontsize=11, title_fontsize=11,
            loc='lower left', ncol=2)
ax_w.legend(h_con + [h_grey], [f'contrast 1e{e}' for e in CON] + ['ended grey'], title='marker', fontsize=11,
            title_fontsize=11, loc='lower left', ncol=2)
fig.suptitle('Choosing $\\eta$ and h for the phase-field TO: thin band ($\\eta$ P small) AND resolved band '
             '($\\eta$ $\\geq$ 2-4 h)   [one marker per final design]', fontsize=15)
out = os.path.join(os.path.dirname(CSV), 'TO_perimeter_indicator_readable.png')
fig.savefig(out, dpi=130)
print('wrote', out)

# table used in NOTES_eta_h_choice.md: contrast 1e5, finest grid of every eta
print(f'{"eta":>7s} {"N":>5s} {"eta/h":>5s} {"etaP":>6s} {"band%":>6s} {"R":>6s} {"w/eta":>6s}')
for eta in ETAS:
    s = [r for r in good if r['eta'] == eta and int(r['soft_exp']) == 5]
    if s:
        r = max(s, key=lambda r: r['N'])
        print(f'1/{round(1 / eta):<5d} {int(r["N"]):5d} {r["eta_h"]:5g} {eta * r["P"]:6.3f} {r["band"]:6.1f} '
              f'{r["R"]:6.3f} {r["w_n"] / eta:6.2f}')
