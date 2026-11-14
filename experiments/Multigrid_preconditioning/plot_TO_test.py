"""
TO test (run_TO_test.sh): CG iterations per solve along the optimization and the objective, per preconditioner,
plus totals (CG iterations, work = iterations x cost per iteration in operator applications, laptop wall time).
Cost per CG iteration (serial CPU, timing.py): Green / GJ 1 K + 1 FFT pair = 1.2, dual 2.4, MG ~6.5.

    python plot_TO_test.py [--grid 64]
"""
import argparse
import re
from pathlib import Path

import numpy as np
import netCDF4
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--grid', type=int, default=64)
args = parser.parse_args()
N = args.grid
PRECS = [('Green', 'Green', '#2ca02c', '-', 1.2), ('Green_Jacobi', 'Green-Jacobi', '#1f77b4', '-', 1.2),
         ('dual', 'dual', '#d62728', '-', 2.4), ('MG', 'multigrid', '#222222', '--', 6.5)]

plt.rcParams.update({'font.size': 13})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.6))
print(f'{"preconditioner":14s}{"evals":>7s}{"CG total":>10s}{"CG/solve":>10s}{"work":>10s}{"wall s":>9s}{"final f":>12s}')
for P, lab, c, ls, w_it in PRECS:
    log = HERE / 'logs' / f'TO_N{N}_{P}.log'
    if not log.exists():
        continue
    text = log.read_text()
    its = [int(v) for v in re.findall(r'nb_?\s?steps CG (?:mech|adjoint)\s*=\s*(\d+)', text)]
    if not its:
        continue
    per_eval = np.array(its[:len(its) // 6 * 6]).reshape(-1, 6).mean(axis=1)
    ax1.plot(np.arange(1, len(per_eval) + 1), per_eval, color=c, ls=ls, lw=2, label=lab)
    hist = sorted((HERE / 'data' / 'example_2D_elasticity_TO').glob(f'N{N}_*{P}_eta_*mgstudy_history.nc'))
    f_final = np.nan
    if hist:
        with netCDF4.Dataset(hist[0]) as ds:
            obj = np.asarray(ds['objective'][:])
        ax2.plot(np.arange(len(obj)), obj, color=c, ls=ls, lw=2, label=lab)
        f_final = obj[-1]
    m = re.search(r'TOTAL wall ([\d.]+) s', text)
    wall = float(m.group(1)) if m else np.nan
    print(f'{lab:14s}{len(per_eval):7d}{sum(its):10d}{np.mean(its):10.1f}{sum(its) * w_it:10.0f}{wall:9.1f}{f_final:12.4e}')
ax1.set_xlabel('objective evaluation')
ax1.set_ylabel('CG iterations per solve (mean of 6)')
ax1.set_ylim(bottom=0)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.set_title(f'{N}² TO, η = 4h, contrast 10⁴, warm-started energy CG')
ax2.set_xlabel('L-BFGS iteration')
ax2.set_ylabel('objective')
ax2.set_title('objective (same problem, own trajectories)')
ax2.set_yscale('log')
ax2.grid(alpha=0.3)
ax1.grid(alpha=0.3)
h, l = ax1.get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=4, fontsize=12, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.92))
out = HERE / 'figures' / f'TO_test_N{N}.png'
fig.savefig(out, dpi=150)
print(out)
