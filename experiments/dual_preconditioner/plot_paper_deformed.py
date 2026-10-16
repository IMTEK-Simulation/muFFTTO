"""Paper figures of the elasticity geometries and deformed states, drawn at the text width 4.8 in. Every pixel is drawn
as the quadrilateral of its four (deformed) corner nodes, coloured by its material, without smoothing.

    part2_elasticity_deformed.pdf      small strain P1-P3, kappa = 1e-3, E = e_1 x e_1 (displacements scaled to a
                                       mean strain of 10 %)       <- paper1_problems/solutions/
    part3_adapted_elasticity_grids.pdf bubble array on the uniform and the adapted 32^2 grid, undeformed and deformed
                                       (E = e_1 x e_1, scaled to 10 %)   <- elasticity_adapted_grid/solutions/
    part4_finite_strain_deformed.pdf   finite strain P1 (kappa = 1e-3, 1e3) and P3, true deformation after 3 of 10
                                       increments (30 % of the load)     <- finite_strain_P123/solutions/

    python plot_paper_deformed.py [OUT_DIR]   (default: ~/PycharmProjects/Paper-Double-Green-Preconditioner/figures)
"""
import os
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.realpath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    '~/PycharmProjects/Paper-Double-Green-Preconditioner/figures')
plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.5})


def wrap(a):
    """nodal field [c, N, N] -> [c, N+1, N+1], periodic"""
    return np.concatenate([np.concatenate([a, a[:, :1]], axis=1), np.concatenate([a, a[:, :1]], axis=1)[:, :, :1]],
                          axis=2)


def node_positions(u, H, u_grid=None):
    """deformed node positions [2, N+1, N+1]: reference node X (moved by the grid map u_grid, physical position
    x0 = X + u_grid), then x = x0 + H x0 + u with the periodic fluctuation u"""
    N = u.shape[-1]
    X = np.stack(np.meshgrid(np.arange(N + 1) / N, np.arange(N + 1) / N, indexing='ij'))
    x0 = X + (wrap(u_grid) if u_grid is not None else 0.0)
    return x0 + np.einsum('ij,j...->i...', H, x0) + wrap(u)


def draw(ax, x, colour, edges=False, **kw):
    ax.pcolormesh(x[0], x[1], colour, shading='flat', edgecolors='0.35' if edges else 'face',
                  linewidth=0.15 if edges else 0.0, rasterized=True, **kw)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


# 1. small strain P1-P3 -----------------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(4.8, 1.85))
amplification = 0.1
for ax, g in zip(axes, ['P1', 'P2', 'P3']):
    d = np.load(os.path.join(HERE, 'paper1_problems', 'solutions', f'{g}_N128_kappa1e-3.npz'))
    C = d['C1111'] / d['C1111'].max()
    x = node_positions(amplification * d['u'], amplification * d['E'])
    draw(ax, x, C, cmap='Greys', norm=LogNorm(1e-3, 2.0))
    ax.set_title(g)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'part2_elasticity_deformed.pdf'), dpi=300)
plt.close(fig)

# 2. adapted grid -----------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(3.6, 3.7))
for row, grid in zip(axes, ['regular', 'adapted']):
    d = np.load(os.path.join(HERE, 'elasticity_adapted_grid', 'solutions', f'bubble_array_{grid}_N32_kappa1e-3.npz'))
    colour = np.where(d['phase'] > 0, 1.0, 0.15)
    for ax, (u, H, title) in zip(row, [(0 * d['u'], 0 * d['E'], 'undeformed'),
                                       (amplification * d['u'], amplification * d['E'], 'deformed')]):
        draw(ax, node_positions(u, H, d['u_phi'] if grid == 'adapted' else None), colour, edges=True, cmap='Greys',
             vmin=0, vmax=1)
        ax.set_title(f'{"uniform" if grid == "regular" else "adapted"} grid, {title}')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'part3_adapted_elasticity_grids.pdf'), dpi=300)
plt.close(fig)

# 3. finite strain ----------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(4.8, 1.95))
for ax, (g, k) in zip(axes, [('P1', -3), ('P1', 3), ('P3', -3)]):
    d = np.load(os.path.join(HERE, 'finite_strain_P123', 'solutions', f'{g}_N64_kappa1e{k:+d}_inc3.npz'))
    x = node_positions(d['u'], d['H'])
    draw(ax, x, d['scale'], cmap='Greys', norm=LogNorm(min(1e-3, d['scale'].min()), max(1.0, d['scale'].max())))
    ax.set_title(f'{g}, $\\kappa=10^{{{k}}}$')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'part4_finite_strain_deformed.pdf'), dpi=300)
plt.close(fig)
print(OUT)
