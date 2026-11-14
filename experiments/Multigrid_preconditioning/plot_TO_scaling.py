"""
NEMO TO scaling (sbatch_TO_mg.sh): Green-Jacobi vs multigrid, 64^2 ... 512^2, eta = 2h and 4h, Cahn-Hilliard start
bi_s0, NEMO production settings, CG_DELAY = 0.  Parses the logs (fetched by sync_nemo_TO.sh to logs/nemo_TO/):
CG iterations per solve (6 per evaluation), GPU wall time per evaluation and to the end, final objective.

    ./sync_nemo_TO.sh && python plot_TO_scaling.py
"""
import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
GRIDS = [64, 128, 256, 512]
PRECS = [('Green', 'Green', '#2ca02c', 'o', '-'), ('Green_Jacobi', 'Green-Jacobi', '#1f77b4', 's', '-'),
         ('MG', 'multigrid', '#222222', 'D', '--')]
ETAS = [1, 2, 4]          # 1h: Green vs MG only (user, 2026-11-13); 2h, 4h: GJ vs MG


def parse(path):
    text = path.read_text()
    its = [int(v) for v in re.findall(r'nb_?\s?steps CG (?:mech|adjoint)\s*=\s*(\d+)', text)]
    evals = re.findall(r'EVAL (\d+) f ([\d.eE+-]+) wall ([\d.]+)', text)
    if not evals:
        return None
    n_eval = int(evals[-1][0])
    wall = float(evals[-1][2])
    first = float(evals[0][2])
    done = 'TOTAL wall' in text
    return {'cg_per_solve': np.mean(its) if its else np.nan, 'evals': n_eval, 'f': float(evals[-1][1]),
            'wall': wall, 't_eval': (wall - first) / max(n_eval - 1, 1), 'done': done}


res = {}
for eta in ETAS:
    for N in GRIDS:
        for P, *_ in PRECS:
            f = HERE / 'logs' / 'nemo_TO' / f'TO_N{N}_{P}_bi_s0_eta{eta}h.log'
            if f.exists():
                r = parse(f)
                if r:
                    res[eta, N, P] = r

print(f'{"eta":>4s}{"N":>5s} {"precond":13s}{"evals":>7s}{"CG/solve":>10s}{"s/eval":>9s}{"wall s":>9s}{"final f":>13s}  status')
for (eta, N, P), r in sorted(res.items()):
    print(f'{eta:3d}h{N:5d} {P:13s}{r["evals"]:7d}{r["cg_per_solve"]:10.1f}{r["t_eval"]:9.2f}{r["wall"]:9.0f}'
          f'{r["f"]:13.5e}  {"done" if r["done"] else "running"}')

plt.rcParams.update({'font.size': 13})
fig, axes = plt.subplots(2, len(ETAS), figsize=(6.2 * len(ETAS), 9), sharex=True)
x = np.arange(len(GRIDS))
for col, eta in enumerate(ETAS):
    for row, (key, ylabel) in enumerate([('cg_per_solve', 'CG iterations per solve'),
                                         ('t_eval', 'GPU time per objective evaluation [s]')]):
        ax = axes[row, col]
        for P, lab, c, mk, ls in PRECS:
            y = [res[eta, N, P][key] if (eta, N, P) in res else np.nan for N in GRIDS]
            ax.plot(x, y, color=c, marker=mk, ls=ls, lw=2, ms=8, label=lab)
        ax.set_ylim(bottom=0)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{N}²' for N in GRIDS])
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if row == 0:
            ax.set_title(f'TO, η = {eta}h')
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            ax.set_xlabel('grid')
h, l = [], []
for ax in axes[0]:
    for hh, ll in zip(*ax.get_legend_handles_labels()):
        if ll not in l and not np.all(np.isnan(hh.get_ydata())):
            h.append(hh); l.append(ll)
fig.legend(h, l, loc='upper center', ncol=3, fontsize=12, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = HERE / 'figures' / 'TO_scaling_nemo.png'
fig.savefig(out, dpi=150)
print(out)
