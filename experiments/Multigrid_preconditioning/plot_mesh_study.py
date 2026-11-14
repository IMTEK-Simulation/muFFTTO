"""
Mesh study 64/128/256 on the phase-field TO design (k = 1e-4, E_xx): iterations to the true relative energy error
1e-12 and solve work (fine-grid operator applications + 0.2 Green + 0.05 transfers).
Columns: fixed eta/h = 0 (sharp), 1, 2; and fixed physical width eta = 1/32 (eta/h = 2 @ 64, 4 @ 128, 8 @ 256).

    python plot_mesh_study.py
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
GRIDS = [64, 128, 256]
ETAS = [0, 0.5, 1, 2, 4, 8]
METHODS = [('green', 'Green', '#2ca02c', 'o', '-'), ('green_jacobi', 'Green-Jacobi', '#1f77b4', 's', '-'),
           ('dual', 'dual', '#d62728', '^', '-'), ('mg', 'multigrid (Jacobi smoother, Galerkin)', '#222222', 'D', '--')]

it, work = {}, {}
for N in GRIDS:
    for eta in ETAS:
        f = HERE / 'exp_data' / f'phasefield_twogrid_N{N}_eta{eta:g}_mesh.npz'
        if not f.exists():
            continue
        res = np.load(f, allow_pickle=True)['results'].item()
        for m, *_ in METHODS:
            w, e = res[m]
            it[N, eta, m] = len(w) - 1
            work[N, eta, m] = w[-1]

print('iterations / work')
print(f'{"N":>5s}{"eta/h":>7s}' + ''.join(f'{lab.split(" ")[0]:>16s}' for _, lab, *_ in METHODS))
for eta in ETAS:
    for N in GRIDS:
        if (N, eta, 'mg') in it:
            print(f'{N:5d}{eta:7g}' + ''.join(f'{it[N, eta, m]:8d} /{work[N, eta, m]:6.0f}' for m, *_ in METHODS))

COLUMNS = [('η = 0 (sharp)', [(N, 0) for N in GRIDS]),
           ('η / h = 1', [(N, 1) for N in GRIDS]),
           ('η / h = 2', [(N, 2) for N in GRIDS]),
           ('fixed η = 1/32  (η/h = 2, 4, 8)', [(64, 2), (128, 4), (256, 8)])]

plt.rcParams.update({'font.size': 13})
fig, axes = plt.subplots(2, 4, figsize=(17, 8.4), sharex=True)
x = np.arange(len(GRIDS))
for col, (title, cases) in enumerate(COLUMNS):
    for row, (data, ylabel) in enumerate([(it, 'CG iterations to 10⁻¹²'),
                                          (work, 'work [operator applications]')]):
        ax = axes[row, col]
        for m, lab, c, mk, ls in METHODS:
            y = [data.get((N, eta, m), np.nan) for N, eta in cases]
            ax.plot(x, y, color=c, marker=mk, ls=ls, lw=2, ms=8, label=lab)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xticks(x)
        ax.set_xticklabels([f'{N}²' for N in GRIDS])
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if row == 0:
            ax.set_title(title)
        else:
            ax.set_xlabel('grid')
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=4, fontsize=12, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = HERE / 'figures' / 'mesh_study_64_128_256.png'
fig.savefig(out, dpi=150)
print(out)
