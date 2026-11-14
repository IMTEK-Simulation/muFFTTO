"""
Design-optimizer test (run_design_opt_test.sh): objective vs objective evaluations (each = 6 linear solves) for
NuMPI L-BFGS, L-BFGS with Helmholtz H_0, projected PR+ NCG plain / with Helmholtz; 32^2 and 64^2.
Table: evaluations to the stop, final objective, evaluations to reach within 1 % / 0.1 % of the best objective
found by any variant on that grid.

    python plot_design_opt.py
"""
import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

HERE = Path(__file__).resolve().parent
GRIDS = [32, 64]
VARIANTS = [('lbfgs', 'L-BFGS (NuMPI)', '#c2a5cf', '--', 'o'), ('lbfgs_helm', 'L-BFGS + Helmholtz H₀', '#762a83', '-', 's'),
            ('ncg', 'NCG (PR+)', '#fdb863', '--', '^'), ('ncg_helm', 'NCG (PR+) + Helmholtz', '#b35806', '-', 'D')]

data = {}
for N in GRIDS:
    for v, *_ in VARIANTS:
        f = HERE / 'logs' / f'dopt_N{N}_{v}.log'
        if f.exists():
            ev = re.findall(r'^EVAL (\d+) f ([\d.eE+-]+) wall ([\d.]+)', f.read_text(), flags=re.M)
            if ev:
                data[N, v] = np.array([[float(a), float(b), float(c)] for a, b, c in ev])

plt.rcParams.update({'font.size': 13})
fig, axes = plt.subplots(2, len(GRIDS), figsize=(14, 9.5), squeeze=False)
print(f'{"N":>4s} {"variant":24s}{"evals":>7s}{"final f":>13s}{"to 1%":>8s}{"to 0.1%":>9s}{"wall s":>9s}')
for col, N in enumerate(GRIDS):
    runs = {v: d for (n, v), d in data.items() if n == N}
    if not runs:
        continue
    f_best = min(d[:, 1].min() for d in runs.values())
    for v, lab, c, ls, mk in VARIANTS:
        if v not in runs:
            continue
        d = runs[v]
        best_so_far = np.minimum.accumulate(d[:, 1])
        gap = (best_so_far - f_best) / f_best
        axes[0, col].plot(d[:, 0], d[:, 1], color=c, ls=ls, lw=2, label=lab, marker=mk, markevery=max(1, len(d) // 8),
                          ms=7)
        axes[1, col].semilogy(d[:, 0], np.maximum(gap, 1e-6), color=c, ls=ls, lw=2, label=lab, marker=mk,
                              markevery=max(1, len(d) // 8), ms=7)

        def first(th):
            idx = np.nonzero(gap <= th)[0]
            return int(d[idx[0], 0]) if idx.size else -1
        print(f'{N:4d} {lab:24s}{int(d[-1, 0]):7d}{d[-1, 1]:13.6e}{first(1e-2):8d}{first(1e-3):9d}{d[-1, 2]:9.1f}')
    axes[0, col].set_title(f'{N}², η = 4h, Cahn-Hilliard start')
    axes[0, col].set_ylabel('objective')
    axes[0, col].set_ylim(bottom=0)
    axes[1, col].set_ylabel('gap to best objective found')
    axes[1, col].yaxis.set_major_locator(LogLocator(base=10, numticks=7))
    axes[1, col].yaxis.set_minor_locator(NullLocator())
    axes[1, col].yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{100 * y:g} %'))
    axes[1, col].set_ylim(1e-5, 2)
    for r in range(2):
        axes[r, col].set_xlabel('objective evaluations (6 linear solves each)')
        axes[r, col].set_xlim(left=0)
        axes[r, col].grid(alpha=0.3)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=4, fontsize=12, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = HERE / 'figures' / 'design_optimizers.png'
fig.savefig(out, dpi=150)
print(out)
