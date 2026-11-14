"""
3D scaling on NEMO (scaling_case_3d.py): MG (option (a)) vs Green-Jacobi / Green / dual on the new Cahn-Hilliard
geometries, 32^3 / 64^3 (l40s) / 128^3 (h200). Table of iterations / GPU solve time; figure for ch_s12: columns
eta/h = 0, 0.5, 1 (fixed ratio) and fixed physical eta = 1/32 (eta/h = 1, 2, 4), rows iterations / GPU time.

    ./sync_nemo_scaling3d.sh && python plot_scaling_3d.py
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
METHODS = [('green', 'Green', '#2ca02c', 'o', '-'), ('green_jacobi', 'Green-Jacobi', '#1f77b4', 's', '-'),
           ('dual', 'dual', '#d62728', '^', '-'), ('mg', 'multigrid (3D, option a)', '#222222', 'D', '--')]
rows = []
for f in sorted((HERE / 'exp_data' / 'nemo_scaling3d' / 'csv').glob('mg3d_scaling_*.csv')):
    with open(f) as fh:
        rows += list(csv.DictReader(fh))
data = {(r['geometry'], int(r['N']), float(r['eta_over_h']), r['method']): r for r in rows}
geoms = sorted({k[0] for k in data})
grids = sorted({k[1] for k in data})
print(f'{"geometry":9s}{"N":>5s}{"eta/h":>7s}' + ''.join(f'{lab.split(" ")[0]:>20s}' for _, lab, *_ in METHODS)
      + '   (iterations / GPU solve s)')
for g in geoms:
    for eta in sorted({k[2] for k in data if k[0] == g}):
        for N in grids:
            if not any((g, N, eta, m) in data for m, *_ in METHODS):
                continue
            cells = [f'{data[g, N, eta, m]["iterations"]:>7s} /{float(data[g, N, eta, m]["t_solve"]):8.2f} s'
                     if (g, N, eta, m) in data else f'{"-":>20s}' for m, *_ in METHODS]
            print(f'{g:9s}{N:5d}{eta:7g}' + ''.join(f'{c:>20s}' for c in cells))

G = 'ch_s12'
cols = [('η = 0 (sharp)', [(N, 0.0) for N in grids]), ('η / h = 0.5', [(N, 0.5) for N in grids]),
        ('η / h = 1', [(N, 1.0) for N in grids]),
        ('fixed η = 1/32 (η/h = 1, 2, 4)', [(N, N / 32) for N in grids])]
plt.rcParams.update({'font.size': 13})
fig, axes = plt.subplots(2, len(cols), figsize=(4.4 * len(cols), 8.4), sharex=True)
x = np.arange(len(grids))
for c, (title, cases) in enumerate(cols):
    for r, (key, ylabel) in enumerate([('iterations', 'CG iterations'), ('t_solve', 'GPU solve time [s]')]):
        ax = axes[r, c]
        for m, lab, col, mk, ls in METHODS:
            y = [float(data[G, N, e, m][key]) if (G, N, e, m) in data else np.nan for N, e in cases]
            ax.plot(x, y, color=col, marker=mk, ls=ls, lw=2, ms=8, label=lab)
        ax.set_ylim(bottom=0)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{N}³' for N in grids])
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if r == 0:
            ax.set_title(title)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            ax.set_xlabel('grid')
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=4, fontsize=12, frameon=False)
fig.suptitle(f'3D, Cahn-Hilliard geometry {G}, contrast 1e-4', y=0.995, fontsize=12, color='#444444')
fig.tight_layout(rect=(0, 0, 1, 0.93))
out = HERE / 'figures' / 'nemo_scaling_3d.png'
fig.savefig(out, dpi=150)
print(out)
