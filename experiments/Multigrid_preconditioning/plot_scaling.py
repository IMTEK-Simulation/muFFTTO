"""
NEMO scaling study (64^2 ... 1024^2, one L40S): per interface width, CG iterations and GPU solve time vs grid,
Green / Green-Jacobi / dual / multigrid.  Library CG, energy stop rtol 1e-6, adaptive delay (d_min = 0).

    ./sync_nemo_scaling.sh && python plot_scaling.py [--etas 0 0.25 1 2] [--metric iterations|t_solve|work]
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--etas', nargs='+', type=float, default=[0, 0.25, 1, 2])
parser.add_argument('--fixed-physical', action='store_true', default=True,
                    help='add a panel for eta = 1/128 (eta/h = N/128)')
args = parser.parse_args()

METHODS = [('green', 'Green', '#2ca02c', 'o', '-'), ('green_jacobi', 'Green-Jacobi', '#1f77b4', 's', '-'),
           ('dual', 'dual', '#d62728', '^', '-'), ('mg', 'multigrid', '#222222', 'D', '--'),
           ('sgs-green', 'smooth-Green-smooth', '#74c476', 'v', '-.')]
rows = []
for f in sorted((HERE / 'exp_data' / 'nemo_scaling' / 'csv').glob('mg_scaling_N*.csv')):
    with open(f) as fh:
        rows += list(csv.DictReader(fh))
data = {(int(r['N']), float(r['eta_over_h']), r['method']): r for r in rows}
grids = sorted({k[0] for k in data})

print(f'{"N":>5s}{"eta/h":>7s}' + ''.join(f'{lab:>22s}' for _, lab, *_ in METHODS) + '   (iterations / GPU solve s)')
for eta in sorted({k[1] for k in data}):
    for N in grids:
        if any((N, eta, m) in data for m, *_ in METHODS):
            cells = []
            for m, *_ in METHODS:
                r = data.get((N, eta, m))
                cells.append(f'{r["iterations"]:>8s} /{float(r["t_solve"]):9.2f} s' if r else f'{"-":>22s}')
            print(f'{N:5d}{eta:7g}' + ''.join(f'{c:>22s}' for c in cells))

columns = [(f'η / h = {e:g}' if e else 'η = 0 (sharp)', [(N, e) for N in grids]) for e in args.etas]
if args.fixed_physical:
    columns.append(('fixed η = 1/128', [(N, N / 128) for N in grids]))
plt.rcParams.update({'font.size': 13})
fig, axes = plt.subplots(2, len(columns), figsize=(4.3 * len(columns), 8.4), sharex=True, squeeze=False)
x = np.arange(len(grids))
for col, (title, cases) in enumerate(columns):
    for row, (key, ylabel) in enumerate([('iterations', 'CG iterations'), ('t_solve', 'GPU solve time [s]')]):
        ax = axes[row, col]
        for m, lab, c, mk, ls in METHODS:
            y = [float(data[(N, e, m)][key]) if (N, e, m) in data else np.nan for N, e in cases]
            ax.plot(x, y, color=c, marker=mk, ls=ls, lw=2, ms=8, label=lab)
        ax.set_ylim(bottom=0)
        if key == 'iterations':
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xticks(x)
        ax.set_xticklabels([f'{N}²' for N in grids], rotation=45)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if row == 0:
            ax.set_title(title)
        else:
            ax.set_xlabel('grid')
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=5, fontsize=12, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = HERE / 'figures' / 'nemo_scaling.png'
fig.savefig(out, dpi=150)
print(out)
