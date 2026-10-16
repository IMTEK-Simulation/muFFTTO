"""Summary of finite_strain_P123.py runs: mean CG iterations per Newton step for every case (geometry, contrast),
preconditioner and grid -> figures/finite_strain_P123.png and a table on stdout. A run whose load path stopped
(tangent not positive definite) is drawn hatched, with the mean over the Newton steps done.

    python summarize.py [GRIDS ...]       (default: 32 64)
"""
import os
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
GRIDS = [int(g) for g in sys.argv[1:]] or [32, 64]
CASES = [('P1', -3), ('P1', 3), ('P2', -3), ('P2', 3), ('P3', -3)]
PRECS = [('Green', 'Green (C_0)', '#2a78d6'), ('Green_Jacobi', 'Green-Jacobi (2μI)', '#eb6834'),
         ('dual', 'dual, full A⁺ (cutoff 1e-6)', '#1baf7a'), ('dual_full0.01', 'dual, full A⁺ (cutoff 1e-2)', '#8fd9b6'),
         ('dual_sym', 'dual, sym (A_ss)⁻¹', '#0b6e4a')]


def load(geometry, kexp, N, prec):
    f = os.path.join(HERE, 'data', f'{geometry}_N{N}_kappa1e{kexp:+d}_{prec}.npz')
    if not os.path.exists(f):
        return None
    d = np.load(f, allow_pickle=True)
    cg = np.asarray(d['cg_per_newton_flat'])
    return dict(total=int(cg.sum()), newton=len(cg), mean=cg.mean() if len(cg) else np.nan,
                increments=int(d['completed_increments']), ninc=int(d['ninc']), failed=str(d['loading_failed']),
                time=float(d['elapsed_time']))


print('| case | grid | ' + ' | '.join(label for _, label, _ in PRECS) + ' |')
print('|---|---|' + '---|' * len(PRECS))
fig, axes = plt.subplots(1, len(GRIDS), figsize=(7.5 * len(GRIDS), 4.4), facecolor='#fcfcfb', sharey=True,
                         squeeze=False)
for ax, N in zip(axes[0], GRIDS):
    width = 0.17
    for j, (p, label, c) in enumerate(PRECS):
        for i, (g, k) in enumerate(CASES):
            r = load(g, k, N, p)
            if r is None:
                continue
            ax.bar(i + (j - 2) * width, r['mean'], width * 0.92, color=c, label=label if i == 0 else None,
                   hatch='///' if r['failed'] else None, edgecolor='#fcfcfb' if not r['failed'] else '0.2', lw=0.5)
            ax.annotate(f'{r["mean"]:.0f}', (i + (j - 2) * width, r['mean']), textcoords='offset points',
                        xytext=(0, 2), ha='center', fontsize=6, color='0.25', rotation=90)
    for g, k in CASES:
        cells = []
        for p, *_ in PRECS:
            r = load(g, k, N, p)
            cells.append('running / not run' if r is None else
                         f'{r["total"]:,} CG / {r["newton"]} Newton = {r["mean"]:.0f}'
                         + (f' (stopped at increment {r["increments"] + 1})' if r['failed'] else ''))
        print(f'| {g}, κ = 1e{k:+d} | {N}² | ' + ' | '.join(cells) + ' |')
    ax.set_xticks(range(len(CASES)), [f'{g}\nκ = 1e{k:+d}' for g, k in CASES])
    ax.set_yscale('log')
    ax.set_title(f'{N}²', loc='left', fontsize=10)
    ax.grid(alpha=0.3, axis='y', which='both')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
axes[0, 0].set_ylabel('mean CG iterations per Newton step')
axes[0, 0].bar(0, np.nan, color='white', edgecolor='0.2', hatch='///', label='load path stopped (tangent indefinite)')
axes[0, 0].legend(frameon=False, fontsize=8, loc='upper left')
fig.suptitle('Finite strain, neo-Hookean, Newton-CG: geometries P1-P3, shear 0.2 + stretch 0.2 in 10 increments, '
             'CG energy stop 1e-6', x=0.01, ha='left', fontweight='bold', fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.93))
os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
out = os.path.join(HERE, 'figures', 'finite_strain_P123.png')
fig.savefig(out, dpi=140, facecolor='#fcfcfb')
print(out)
