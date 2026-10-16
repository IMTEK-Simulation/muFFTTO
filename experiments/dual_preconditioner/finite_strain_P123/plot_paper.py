"""Paper figure and table rows of the finite-strain runs (finite_strain_P123.py), drawn at the text width 4.8 in:
mean CG iterations per Newton step over the grid for P1 (kappa = 1e-3, 1e3) and P3 (kappa = 1e-3), Green,
Green-Jacobi, dual (full pseudo-inverse) and dual (sym)  ->  part4_finite_strain.pdf, LaTeX rows on stdout.

    python plot_paper.py [OUT_DIR]      (default: ~/PycharmProjects/Paper-Double-Green-Preconditioner/figures)
"""
import os
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    '~/PycharmProjects/Paper-Double-Green-Preconditioner/figures')
GRIDS = [32, 64, 128, 256]
CASES = [('P1', -3), ('P1', 3), ('P3', -3)]
PRECS = [('Green', r'Green', '#2a78d6', 'o', '-'), ('Green_Jacobi', r'Green-Jacobi', '#eb6834', 's', '-'),
         ('dual', r'dual, $\mathbb{A}^{+}$', '#1baf7a', '^', '--'),
         ('dual_sym', r'dual, $(\mathbb{A}_{ss})^{-1}$', '#0b6e4a', 'v', '-'),
         ('dual_inv', r'dual, $\mathbb{A}^{-1}$', '#8fd9b6', 'D', ':')]
plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.5, 'legend.fontsize': 6})


def load(g, k, N, p):
    f = os.path.join(HERE, 'data', f'{g}_N{N}_kappa1e{k:+d}_{p}.npz')
    if not os.path.exists(f):
        return None
    d = np.load(f, allow_pickle=True)
    cg = np.asarray(d['cg_per_newton_flat'])
    return dict(mean=cg.mean(), total=int(cg.sum()), newton=len(cg), failed=str(d['loading_failed']))


fig, axes = plt.subplots(1, 3, figsize=(4.8, 2.4), sharey=True)
for ax, (g, k) in zip(axes, CASES):
    for p, label, c, m, ls in PRECS:
        pts = [(N, load(g, k, N, p)) for N in GRIDS]
        pts = [(N, r) for N, r in pts if r is not None]
        ax.plot([N for N, _ in pts], [r['mean'] for _, r in pts], color=c, marker=m, ls=ls, ms=3.5, lw=1.2,
                label=label)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(GRIDS, [f'${N}^2$' for N in GRIDS])
    ax.set_xlim(GRIDS[0] / 1.3, GRIDS[-1] * 1.3)
    ax.set_title(f'{g}, $\\kappa=10^{{{k}}}$')
    ax.set_xlabel('grid')
    ax.grid(alpha=0.3, which='both')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel('mean CG iterations\nper Newton step')
fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=5, frameon=False)
fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig(os.path.join(OUT, 'part4_finite_strain.pdf'))
plt.close(fig)

print('% case & grid & Newton & Green & Green-Jacobi & dual (A+) & dual (sym): mean CG per Newton step')
for g, k in CASES:
    for N in GRIDS:
        rs = [load(g, k, N, p) for p, *_ in PRECS]
        newton = next(r['newton'] for r in rs if r is not None)
        print(f'{g}, $10^{{{k}}}$ & ${N}^2$ & {newton} & '
              + ' & '.join('--' if r is None else f'{r["mean"]:.0f}' for r in rs) + r' \\')
print(OUT)
