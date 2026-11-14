"""
eta sweep at 256^2 (phase-field TO design, k = 1e-4, E_xx): CG iterations to the true relative energy error 1e-12
and solve work (fine-grid operator applications + 0.2 Green + 0.05 transfers) vs interface width eta / h.

    python plot_eta_sweep.py
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
ETAS = [0, 0.25, 0.5, 1, 1.5, 2, 3, 4]
METHODS = [('green', 'Green', '#2ca02c', 'o', '-'), ('green_jacobi', 'Green-Jacobi', '#1f77b4', 's', '-'),
           ('dual', 'dual', '#d62728', '^', '-'), ('mg', 'multigrid (Jacobi smoother, Galerkin)', '#222222', 'D', '--')]

it = {m: [] for m, *_ in METHODS}
work = {m: [] for m, *_ in METHODS}
for eta in ETAS:
    d = np.load(HERE / 'exp_data' / f'phasefield_twogrid_N256_eta{eta:g}_sweep.npz', allow_pickle=True)
    res = d['results'].item()
    for m, *_ in METHODS:
        w, e = res[m]
        it[m].append(len(w) - 1)
        work[m].append(w[-1])

print(f'{"eta/h":>6s}' + ''.join(f'{lab.split(" ")[0]:>14s}' for _, lab, *_ in METHODS) + '   (iterations / work)')
for i, eta in enumerate(ETAS):
    print(f'{eta:6g}' + ''.join(f'{it[m][i]:6d} /{work[m][i]:6.0f}' for m, *_ in METHODS))

plt.rcParams.update({'font.size': 13})
fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
for ax, data, ylabel in [(axes[0], it, 'CG iterations to ||e||²_K / ||e₀||²_K ≤ 10⁻¹²'),
                         (axes[1], work, 'solve work [fine-grid operator applications]')]:
    for m, lab, c, mk, ls in METHODS:
        ax.plot(ETAS, data[m], color=c, marker=mk, ls=ls, lw=2, ms=8, label=lab)
    ax.set_xlabel('interface width η / h')
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.1, 4.1)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.3)
axes[0].set_title('iterations')
axes[1].set_title('cost (serial CPU weights)')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=4, fontsize=12, frameon=False)
fig.suptitle('256² phase-field TO design, contrast 10⁻⁴', y=0.90, fontsize=13, color='#444444')
fig.tight_layout(rect=(0, 0, 1, 0.86))
out = HERE / 'figures' / 'eta_sweep_N256.png'
fig.savefig(out, dpi=150)
print(out)
