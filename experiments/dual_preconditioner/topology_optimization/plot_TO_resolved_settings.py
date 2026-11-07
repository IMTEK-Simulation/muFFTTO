"""
Phase-field TO, only the properly resolved settings (NOTES_eta_h_choice.md: 4 h <= eta <= 0.035 / P ~ 1/128):
512^2 eta = 4h (1/128), 1024^2 eta = 4h (1/256), 1024^2 eta = 8h (1/128).  NEMO eta / contrast study
(figures/nemo/contrast_summary.csv), CG work per linear solve (dual counted x 2), one panel per contrast.
Green was not run at 1024^2: hatched bar = its 512^2 value at the same eta/h (Green is flat in N from ~128^2).
    python plot_TO_resolved_settings.py   ->  figures/TO_resolved_settings.png
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
W, OK = {}, {}
for r in csv.DictReader(open(os.path.join(HERE, 'figures', 'nemo', 'contrast_summary.csv'))):
    key = (int(float(r['contrast'])), int(r['N']), int(float(r['eta_h'])), r['precond'])
    W[key], OK[key] = float(r['work']), r['design'] == 'True'
SETTINGS = [(512, 4, '512$^2$\neta = 4h = 1/128'), (1024, 4, '1024$^2$\neta = 4h = 1/256'),
            (1024, 8, '1024$^2$\neta = 8h = 1/128')]
PRE = [('Green', 'Green', 'tab:green'), ('Green_Jacobi', 'Green-Jacobi', 'tab:blue'), ('dual', 'dual (x 2)', 'tab:red')]
plt.rcParams.update({'font.size': 13})
fig, axs = plt.subplots(1, 4, figsize=(22, 6.5), layout='constrained')
x = np.arange(len(SETTINGS))
wb = 0.26
for ax, k in zip(axs, [100, 1000, 10000, 100000]):
    top = 0
    for j, (P, lab, col) in enumerate(PRE):
        for i, (n, e, _) in enumerate(SETTINGS):
            key = (k, n, e, P)
            est = key not in W and P == 'Green'
            v = W.get(key, W.get((k, 512, e, P), np.nan) if est else np.nan)
            if np.isnan(v):
                continue
            top = max(top, v)
            ax.bar(x[i] + (j - 1) * wb, v, wb, color='white' if est else col, edgecolor=col, lw=2,
                   hatch='//' if est else None, label=(lab if i == 0 else None))
            ax.text(x[i] + (j - 1) * wb, v, f'{v:.0f}' + ('*' if est else '') + ('' if OK.get(key, True) else '\ngrey'),
                    ha='center', va='bottom', fontsize=10.5)
    ax.set_xticks(x), ax.set_xticklabels([s[2] for s in SETTINGS], fontsize=11.5)
    ax.set_ylim(0, 1.15 * top)
    ax.set(title=f'contrast {k:.0e}', ylabel='CG work per linear solve (dual x 2)')
    ax.grid(axis='y', alpha=0.3)
axs[0].legend(loc='upper left', fontsize=12)
fig.suptitle('Phase-field TO, properly resolved settings only (4h <= eta <= 1/128):  * = Green not run at 1024$^2$, '
             'its 512$^2$ value shown (hatched)', fontsize=15)
out = os.path.join(HERE, 'figures', 'TO_resolved_settings.png')
fig.savefig(out, dpi=120)
print('wrote', out)
