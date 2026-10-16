"""
Figures and tables of the test problems P1-P3 for the paper (Paper-Double-Green-Preconditioner), from the runs of
run_all.sh / run_all_elasticity.sh / run_grids_elasticity.sh (exp_data/, exp_data_elasticity/):

    part1_iterations_vs_contrast.pdf   k* (e_k <= 1e-10) and the production stop over the contrast, 256^2
    part1_grid_study.pdf               k* over the grid 256^2 ... 1024^2, kappa = 1e-3
    part1_grid_study_stiff.pdf         elasticity, k* over the grid, kappa = 1e3 (P1, P2)
and the LaTeX tables on stdout.

    python plot_paper.py [OUT_DIR]      (default: ~/PycharmProjects/Paper-Double-Green-Preconditioner/figures)
"""
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / 'PycharmProjects' / 'Paper-Double-Green-Preconditioner' \
    / 'figures'
DATA = {'conductivity': HERE / 'exp_data', 'elasticity': HERE / 'exp_data_elasticity'}
PRECS = ['green', 'green_jacobi', 'dual']
LABEL = {'green': r'Green $\mathbf{G}$', 'green_jacobi': r'Green-Jacobi $\mathbf{D}^{-1/2}\mathbf{G}_{I}\mathbf{D}^{-1/2}$',
         'dual': r'dual $\mathbf{G}\,\mathbf{K}(\mathbf{A}_0\mathbf{A}^{-1}\mathbf{A}_0)\,\mathbf{G}$'}
COLOR = {'green': '#2a78d6', 'green_jacobi': '#eb6834', 'dual': '#1baf7a'}
MARKER = {'green': 'o', 'green_jacobi': 's', 'dual': '^'}
KAPPAS = {'P1': [-3, -2, -1, 1, 2, 3], 'P2': [-3, -2, -1, 1, 2, 3], 'P3': [-3, -2, -1]}
GRIDS = [256, 512, 1024]
plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.5, 'legend.fontsize': 6})   # drawn at the text width 4.8 in


def load(physics, problem, N, kexp, prec):
    f = DATA[physics] / f'{problem}_N{N}_kappa1e{kexp:+d}_{prec}.npz'
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    return {'k_star': int(d['k_star']), 'k_production': int(d['k_production'])}


def style(ax):
    ax.grid(alpha=0.3, which='both')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


# 1. iterations over the contrast ------------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(4.8, 3.8), sharey=True)
for row, physics in zip(axes, ['conductivity', 'elasticity']):
    for ax, problem in zip(row, ['P1', 'P2', 'P3']):
        for prec in PRECS:
            pts = [(10.0 ** k, load(physics, problem, 256, k, prec)) for k in KAPPAS[problem]]
            pts = [(k, h) for k, h in pts if h is not None]
            for side in (-1, 1):                              # no line across kappa = 1
                part = [(k, h) for k, h in pts if np.sign(np.log10(k)) == side]
                if not part:
                    continue
                kk = [k for k, _ in part]
                ax.loglog(kk, [h['k_star'] for _, h in part], c=COLOR[prec], marker=MARKER[prec], ms=3, lw=1.1,
                          label=LABEL[prec] if side == -1 else None)
                ax.loglog(kk, [h['k_production'] for _, h in part], c=COLOR[prec], marker=MARKER[prec], ms=3,
                          lw=0.8, ls='--', mfc='none')
        ax.set_title(f'{physics}, {problem}')
        ax.set_ylim(2, 1e4)
        if physics == 'elasticity':
            ax.set_xlabel(r'phase contrast $\kappa$')
        style(ax)
    row[0].set_ylabel('PCG iterations')
axes[0, 0].legend(loc='upper left', frameon=False)
style_handles = [axes[0, 2].plot([], [], c='0.3', lw=1.4)[0], axes[0, 2].plot([], [], c='0.3', lw=0.8, ls='--')[0]]
axes[0, 2].legend(style_handles, [r'$k^*$: $\|\mathbf{e}_k\|_\mathbf{K}^2\leq 10^{-10}$', 'production stop'],
                  loc='upper right', frameon=False)
fig.tight_layout()
fig.savefig(OUT / 'part1_iterations_vs_contrast.pdf')
plt.close(fig)


# 2. grid studies ----------------------------------------------------------------------------------------------------
GRID_LINEAR = False      # True: linear iteration axis with the counts written next to the points


def grid_panel(ax, physics, problem, kexp):
    for prec in PRECS:
        pts = [(N, load(physics, problem, N, kexp, prec)) for N in GRIDS]
        pts = [(N, h) for N, h in pts if h is not None]
        if not pts:
            continue
        ax.plot([N for N, _ in pts], [h['k_star'] for _, h in pts], c=COLOR[prec], marker=MARKER[prec],
                ms=3.5, lw=1.3, label=LABEL[prec])
        if GRID_LINEAR:
            for N, h in pts:
                ax.annotate(str(h['k_star']), (N, h['k_star']), textcoords='offset points', xytext=(0, 5),
                            ha='center', fontsize=5, color='0.25')
    ax.set_xscale('log', base=2)
    ax.set_xticks(GRIDS, [f'${N}^2$' for N in GRIDS])
    ax.set_xlim(GRIDS[0] / 1.3, GRIDS[-1] * 1.3)
    if GRID_LINEAR:
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    else:
        ax.set_yscale('log')
        ax.set_ylim(2, 1e4)
    ax.set_title(f'{problem}, $\\kappa=10^{{{kexp}}}$')
    style(ax)


fig, axes = plt.subplots(2, 3, figsize=(4.8, 3.8), sharey=not GRID_LINEAR)
for row, physics in zip(axes, ['conductivity', 'elasticity']):
    for ax, problem in zip(row, ['P1', 'P2', 'P3']):
        grid_panel(ax, physics, problem, -3)
        if physics == 'elasticity':
            ax.set_xlabel('grid')
    row[0].set_ylabel(f'{physics}\nPCG iterations $k^*$')
axes[0, 0].legend(loc='upper left', frameon=False)
fig.tight_layout()
fig.savefig(OUT / 'part1_grid_study.pdf')
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(3.6, 2.4), sharey=not GRID_LINEAR)
for ax, problem in zip(axes, ['P1', 'P2']):
    grid_panel(ax, 'elasticity', problem, 3)
    ax.set_xlabel('grid')
axes[0].set_ylabel('elasticity\nPCG iterations $k^*$')
fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2, frameon=False)   # below the panels
fig.tight_layout(rect=(0, 0.17, 1, 1))
fig.savefig(OUT / 'part1_grid_study_stiff.pdf')
plt.close(fig)


# 3. LaTeX tables ----------------------------------------------------------------------------------------------------
def fmt(h):
    return '--' if h is None else str(h['k_star'])


for physics in ['conductivity', 'elasticity']:
    print(f'% {physics}: k* at 256^2, Green / Green-Jacobi / dual')
    for problem in ['P1', 'P2', 'P3']:
        cells = []
        for k in [-3, -2, -1, 1, 2, 3]:
            hs = [load(physics, problem, 256, k, p) for p in PRECS]
            cells.append('--' if all(h is None for h in hs) else ' / '.join(fmt(h) for h in hs))
        print(f'{problem} & ' + ' & '.join(cells) + r' \\')
    print(f'% {physics}: k* at kappa = 1e-3 over the grid, Green / Green-Jacobi / dual')
    for problem in ['P1', 'P2', 'P3']:
        print(f'{problem} & ' + ' & '.join(' / '.join(fmt(load(physics, problem, N, -3, p)) for p in PRECS)
                                          for N in GRIDS) + r' \\')
print(OUT)
