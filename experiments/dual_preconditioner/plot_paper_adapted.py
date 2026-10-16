"""Paper figures of the adapted-grid studies (conductivity_adapted_grid/, elasticity_adapted_grid/), drawn at the text
width 4.8 in: CG iterations over the grid, uniform grid (filled, solid) and adapted grid (hollow, dashed), one panel
per contrast  ->  part3_adapted_conductivity.pdf, part3_adapted_elasticity.pdf, and the LaTeX table rows on stdout.

    python plot_paper_adapted.py [OUT_DIR]      (default: ~/PycharmProjects/Paper-Double-Green-Preconditioner/figures)
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
GRIDS = [16, 32, 64, 128]
KAPPAS = [-3, 3]
PRECS = [('Green', 'Green', '#2a78d6', 'o'), ('Green_Jacobi', 'Green-Jacobi', '#eb6834', 's'),
         ('dual', 'dual', '#1baf7a', '^')]
plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.5, 'legend.fontsize': 6})


def load(study, grid, N, k, p):
    f = os.path.join(HERE, study, 'data', f'bubble_array_{grid}_N{N}_kappa1e{k:+d}_{p}.npz')
    return dict(np.load(f)) if os.path.exists(f) else None


for study, name, prop in (('conductivity_adapted_grid', 'part3_adapted_conductivity', 'A_eff'),
                          ('elasticity_adapted_grid', 'part3_adapted_elasticity', 'C_eff_voigt')):
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.6), sharey=True)
    for ax, k in zip(axes, KAPPAS):
        for p, label, c, m in PRECS:
            for grid, ls, mfc in (('regular', '-', c), ('adapted', '--', 'white')):
                pts = [(N, load(study, grid, N, k, p)) for N in GRIDS]
                pts = [(N, r) for N, r in pts if r is not None]
                ax.plot([N for N, _ in pts], [np.mean(r['iterations']) for _, r in pts], color=c, ls=ls, marker=m,
                        mfc=mfc, ms=3.5, lw=1.2, label=f'{label}, {"uniform" if grid == "regular" else "adapted"}')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xticks(GRIDS, [f'${N}^2$' for N in GRIDS])
        ax.set_title(f'$\\kappa=10^{{{k}}}$')
        ax.set_xlabel('grid')
        ax.grid(alpha=0.3, which='both')
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel('PCG iterations')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(os.path.join(OUT, f'{name}.pdf'))
    plt.close(fig)

    print(f'% {study}: kappa & grid & N & min J & g_max/g_min & Green & Green-Jacobi & dual & property')
    for k in KAPPAS:
        for grid in ('regular', 'adapted'):
            for N in GRIDS:
                rs = {p: load(study, grid, N, k, p) for p, *_ in PRECS}
                r0 = rs['dual']
                value = r0[prop][0, 0]
                ratio = float(r0['g_max'] / r0['g_min'])
                exponent = int(np.floor(np.log10(ratio)))
                print(f'$10^{{{k}}}$ & {"uniform" if grid == "regular" else "adapted"} & ${N}^2$ & '
                      f'{float(r0["min_J"]):.3f} & ${ratio / 10 ** exponent:.1f}\\cdot10^{{{exponent}}}$ & '
                      + ' & '.join(f'{np.mean(rs[p]["iterations"]):.0f}' for p, *_ in PRECS)
                      + f' & {value:.4f} \\\\')
print(OUT)
