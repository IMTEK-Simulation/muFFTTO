"""Summary of elasticity_adapted_grid.py: CG iterations (mean of the three load cases) over the grid for the regular
(solid) and the adapted grid (dashed), one panel per contrast, and the effective stiffness C_1111 over the grid
-> figures/elasticity_adapted_grid.png and a table on stdout.

    python summarize.py
"""
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
GRIDS = [16, 32, 64, 128]
KAPPAS = [-3, 3]
PRECS = [('Green', 'Green (C_0)', '#2a78d6', 'o'), ('Green_Jacobi', 'Green-Jacobi (2μI)', '#eb6834', 's'),
         ('dual', 'dual G K(C_0 C̃⁺ C_0) G', '#1baf7a', '^')]


def load(grid, N, k, p):
    f = os.path.join(HERE, 'data', f'bubble_array_{grid}_N{N}_kappa1e{k:+d}_{p}.npz')
    return dict(np.load(f)) if os.path.exists(f) else None


fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), facecolor='#fcfcfb')
for ax, k in zip(axes[:2], KAPPAS):
    for p, label, c, m in PRECS:
        for grid, ls, mfc in (('regular', '-', c), ('adapted', '--', '#fcfcfb')):
            pts = [(N, load(grid, N, k, p)) for N in GRIDS]
            pts = [(N, r) for N, r in pts if r is not None]
            ax.plot([N for N, _ in pts], [np.mean(r['iterations']) for _, r in pts], color=c, ls=ls, marker=m,
                    mfc=mfc, mew=1.5, ms=6, lw=1.8, label=f'{label}, {grid}')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(GRIDS, [f'{N}²' for N in GRIDS])
    ax.set_title(f'bubble stiffness κ C_0, κ = 1e{k:+d}', loc='left', fontsize=10)
    ax.set_xlabel('grid')
    ax.set_ylabel('CG iterations (energy stop 1e-6)')
for k, ls in zip(KAPPAS, ('-', ':')):
    for grid, c in (('regular', '0.15'), ('adapted', '#b5179e')):
        pts = [(N, load(grid, N, k, 'dual')) for N in GRIDS]
        pts = [(N, r) for N, r in pts if r is not None]
        axes[2].plot([N for N, _ in pts], [r['C_eff_voigt'][0, 0] for _, r in pts], color=c, ls=ls, marker='o', ms=5,
                     label=f'{grid}, κ = 1e{k:+d}')
axes[2].set_xscale('log', base=2)
axes[2].set_yscale('log')
axes[2].set_xticks(GRIDS, [f'{N}²' for N in GRIDS])
axes[2].set_title('effective stiffness C_eff,1111', loc='left', fontsize=10)
axes[2].set_xlabel('grid')
for ax in axes:
    ax.grid(alpha=0.3, which='both')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
axes[0].legend(frameon=False, fontsize=7, ncol=2)
axes[2].legend(frameon=False, fontsize=8)
fig.suptitle('Small-strain elasticity, bubble array: uniform grid vs grid adapted to the bubbles (pull-back '
             'C̃ = J F⁻¹·C·F⁻ᵀ, Ẽ = E·F), Q1', x=0.01, ha='left', fontweight='bold', fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93))
os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
out = os.path.join(HERE, 'figures', 'elasticity_adapted_grid.png')
fig.savefig(out, dpi=140, facecolor='#fcfcfb')

print('| κ | grid | N | min J | g_max/g_min | Green | Green-Jacobi | dual | C_eff,1111 |')
print('|---|---|---|---|---|---|---|---|---|')
for k in KAPPAS:
    for grid in ('regular', 'adapted'):
        for N in GRIDS:
            rs = {p: load(grid, N, k, p) for p, *_ in PRECS}
            r0 = rs['dual']
            print(f'| 1e{k:+d} | {grid} | {N}² | {float(r0["min_J"]):.3f} | {float(r0["g_max"] / r0["g_min"]):.2g} | '
                  + ' | '.join(f'{np.mean(rs[p]["iterations"]):.0f}' for p, *_ in PRECS)
                  + f' | {r0["C_eff_voigt"][0, 0]:.5f} |')
print(out)
