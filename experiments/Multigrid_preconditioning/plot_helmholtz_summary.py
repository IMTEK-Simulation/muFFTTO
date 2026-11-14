"""
How much the Helmholtz H_0 helped on NEMO (2D TO, CH start bi_s0, multigrid mechanics): objective evaluations and
GPU time to the converged design vs grid, plain NuMPI L-BFGS vs Helmholtz c = 1 and c = 4; production setup
(Green-Jacobi + NuMPI L-BFGS) as the time reference. Rows: evaluations / time; columns: eta = 2h, 4h.

    ./sync_nemo_TO.sh && python plot_helmholtz_summary.py
"""
import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
L = HERE / 'logs' / 'nemo_TO'
GRIDS = [64, 128, 256, 512]
SERIES = [('', 'MG', 'plain L-BFGS (MG mechanics)', '#c2a5cf', '--', 'o'),
          ('_lbfgs_helm', 'MG', 'Helmholtz c = 1', '#9970ab', '-', 's'),
          ('_lbfgs_helm_c4', 'MG', 'Helmholtz c = 4', '#40004b', '-', 'D'),
          ('', 'Green_Jacobi', 'production: Green-Jacobi + plain L-BFGS', '#1f77b4', ':', '^')]


def final(path):
    if not path.exists():
        return None
    t = path.read_text()
    ev = re.findall(r'EVAL (\d+) f ([\d.eE+-]+) wall ([\d.]+)', t)
    if not ev or 'TOTAL wall' not in t:
        return None
    return int(ev[-1][0]), float(ev[-1][2]) / 60, float(ev[-1][1])


plt.rcParams.update({'font.size': 13})
fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), sharex=True)
x = np.arange(len(GRIDS))
print(f'{"eta":>4s} {"series":42s}' + ''.join(f'{N:>16d}' for N in GRIDS) + '   (evaluations / minutes)')
for col, E in enumerate([2, 4]):
    for tag, P, lab, c, ls, mk in SERIES:
        vals = [final(L / f'TO_N{N}_{P}_bi_s0_eta{E}h{tag}.log') for N in GRIDS]
        ev = [v[0] if v else np.nan for v in vals]
        tm = [v[1] if v else np.nan for v in vals]
        print(f'{E:3d}h {lab:42s}' + ''.join(f'{e:7.0f} /{t:6.1f} m' if v else f'{"-":>16s}'
                                               for e, t, v in zip(ev, tm, vals)))
        if P == 'MG':
            axes[0, col].plot(x, ev, color=c, ls=ls, marker=mk, lw=2.2, ms=8, label=lab)
        axes[1, col].plot(x, tm, color=c, ls=ls, marker=mk, lw=2.2, ms=8, label=lab)
    axes[0, col].set_title(f'η = {E}h')
    axes[0, col].set_ylabel('objective evaluations to the stop')
    axes[1, col].set_ylabel('GPU time to the converged design [min]')
    for r in range(2):
        ax = axes[r, col]
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xticks(x)
        ax.set_xticklabels([f'{N}²' for N in GRIDS])
        ax.grid(alpha=0.3)
    axes[1, col].set_xlabel('grid')
h, l = axes[1, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=2, fontsize=12, frameon=False)
fig.suptitle('2D TO from the CH start, NEMO L40S', y=0.905, fontsize=12, color='#444444')
fig.tight_layout(rect=(0, 0, 1, 0.9))
out = HERE / 'figures' / 'helmholtz_summary_nemo.png'
fig.savefig(out, dpi=150)
print(out)
