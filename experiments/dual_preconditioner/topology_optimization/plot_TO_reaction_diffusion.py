"""Reaction-diffusion level set (sharp two-phase quadrature points) vs phase field and Hamilton-Jacobi level set, 64^2:
    top       final designs as the CG solves see them: the density at the 2 quadrature points (triangles) of each pixel
              (exact geometry, no smoothing); grey = intermediate stiffness
    bottom    stress mismatch (best so far) and CG iterations per solve over the objective evaluations; total CG
              iterations and wall time of each optimization
-> figures/TO_reaction_diffusion_comparison_N64.png

    python plot_TO_reaction_diffusion.py
"""
import os

import matplotlib
import netCDF4
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

HERE = os.path.dirname(os.path.realpath(__file__))
DATA = os.path.join(HERE, 'data')
PF = os.path.join(DATA, 'example_2D_elasticity_TO', 'N64_soft1e-05_energy-warm_{}_eta_0.03125_w_5.0_l_bfgs')
LS = os.path.join(DATA, 'example_2D_elasticity_TO_level_set',
                  'N64_soft1e-05_energy_{}_w_5.0_p1_eps2-1-0.5-0_alpha2_holes4_noreinit')
RD = os.path.join(DATA, 'example_2D_elasticity_TO_reaction_diffusion', 'N64_soft1e-05_energy_{}_w_5.0_quad_')
BLUE, ORANGE = '#2a78d6', '#eb6834'
RUNS = [  # label, file, kind, colour, line style
    ('phase field, Green', PF.format('Green'), 'pf', BLUE, ':'),
    ('phase field, Green-Jacobi', PF.format('Green_Jacobi'), 'pf', ORANGE, ':'),
    ('HJ level set ε 2→0, Green', LS.format('Green'), 'ls', BLUE, '--'),
    ('HJ level set ε 2→0, Green-Jacobi', LS.format('Green_Jacobi'), 'ls', ORANGE, '--'),
    ('RD sharp (c 0.05, L 1 px), Green', RD.format('Green') + 'L1_c0.05_all_initphase', 'rd', BLUE, '-'),
    ('RD sharp (c 0.05, L 1 px), Green-Jacobi', RD.format('Green_Jacobi') + 'L1_c0.05_all_initphase', 'rd', ORANGE, '-'),
    ('RD sharp (c 0.2, L 0.5 px), Green', RD.format('Green') + 'L0.5_c0.2_all_initphase', 'rd', BLUE, (0, (5, 1))),
    ('RD sharp from solid, Green', RD.format('Green') + 'L2-0.5_c0.1_anneal30_all_initsolid', 'rd', BLUE, '-.'),
    ('RD topology → HJ ε 2→0,\nGreen-Jacobi', LS.format('Green_Jacobi').replace('holes4_noreinit', 'initpf_final')
     + '_noreinit_from_RD', 'ls', ORANGE, '-.'),
]


def quad_values(field):
    """Field at the 2 quadrature points of each pixel (linear triangles, as discretization.apply_N_operator_mugrid)."""
    f_10, f_01 = np.roll(field, -1, 0), np.roll(field, -1, 1)
    return np.stack([(field + f_10 + f_01) / 3, (f_10 + f_01 + np.roll(f_10, -1, 1)) / 3])


def triangulation(n):
    """Triangles of the n x n cell, nodes at the pixel corners: lower-left and upper-right triangle of each pixel."""
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing='ij')
    node = lambda a, b: a * (n + 1) + b
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    lower = np.stack([node(ii, jj), node(ii + 1, jj), node(ii, jj + 1)], -1).reshape(-1, 3)
    upper = np.stack([node(ii + 1, jj), node(ii, jj + 1), node(ii + 1, jj + 1)], -1).reshape(-1, 3)
    return Triangulation(i.ravel() / n, j.ravel() / n, np.concatenate([lower, upper]))


SHORT = {'phase field, Green': 'PF G', 'phase field, Green-Jacobi': 'PF GJ', 'HJ level set ε 2→0, Green': 'HJ G',
         'HJ level set ε 2→0, Green-Jacobi': 'HJ GJ', 'RD sharp (c 0.05, L 1 px), Green': 'RD G',
         'RD sharp (c 0.05, L 1 px), Green-Jacobi': 'RD GJ', 'RD sharp (c 0.2, L 0.5 px), Green': 'RD G best',
         'RD sharp from solid, Green': 'RD G solid', 'RD topology → HJ ε 2→0,\nGreen-Jacobi': 'RD→HJ GJ'}
runs = []
for label, path, kind, color, style in RUNS:
    if not os.path.exists(path + '_final.nc'):  # unfinished run
        print(f'skipped (not finished): {label}')
        continue
    final = netCDF4.Dataset(path + '_final.nc')
    cg = (np.asarray(final.num_iteration_mech, dtype=float) +
          np.asarray(final.num_iteration_adjoint, dtype=float)).reshape(-1, 3).sum(1)
    if kind == 'pf':
        density = quad_values(np.asarray(final['phase_field_solution'][:])[0, 0])
        mismatch = float(final.weight) / 3 * np.asarray(final.norms_sigma)
        evaluations = np.arange(1, len(mismatch) + 1)
    else:
        cg = cg[:-1]  # the last evaluation re-sets the final material (post-processing)
        history = netCDF4.Dataset(path + '_history.nc')
        mismatch = np.asarray(history['objective'][:])
        evaluations = np.asarray(history['nb_evaluations'][:]).astype(int)
        nodal = np.asarray(final['phase_field_solution'][:])[0, 0]
        if kind == 'rd':  # sharp threshold of the interpolated level set
            density = (quad_values(np.asarray(final['level_set_solution'][:])[0, 0]) > 0).astype(float)
        else:
            density = quad_values(nodal)
    C = np.asarray(final.homogenized_C_voigt).reshape(3, 3)
    # J of the final design: the best one for RD (kept) and the phase field (L-BFGS), the last (sharp) one for HJ
    final_J = mismatch[-1] if kind == 'ls' else min(mismatch)
    runs.append(dict(label=label, short=SHORT.get(label, label), final_J=final_J, color=color, style=style, density=density, cg=cg, mismatch=mismatch,
                     evaluations=evaluations, time=float(final.elapsed_time), C=C,
                     grey=np.mean((density > 1e-3) & (density < 1 - 1e-3))))

n = runs[0]['density'].shape[-1]
mesh = triangulation(n)
fig = plt.figure(figsize=(2.6 * len(runs), 8.6), dpi=150)
grid_top = fig.add_gridspec(1, len(runs), left=0.02, right=0.98, top=0.86, bottom=0.58, wspace=0.08)
grid_bottom = fig.add_gridspec(1, 4, left=0.06, right=0.98, top=0.47, bottom=0.1, wspace=0.32,
                               width_ratios=[1.2, 1.2, 0.8, 0.8])
for k, r in enumerate(runs):
    ax = fig.add_subplot(grid_top[0, k])
    ax.tripcolor(mesh, facecolors=r['density'].reshape(2, -1).ravel(), cmap='Greys_r', vmin=0, vmax=1,
                 edgecolors='none')
    ax.set_aspect('equal')
    ax.set_xticks([]), ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(r['color'])
        spine.set_linewidth(2.5)
        spine.set_linestyle(r['style'] if r['style'] != ':' else (0, (1, 1)))  # dense dots, visible on a frame
    ax.set_title(f'{r["label"]}\nJ = {r["final_J"]:.3g}, {100 * r["grey"]:.0f}% grey\n'
                 f'C12 = {r["C"][0, 1]:.3f}  (target -0.096)', fontsize=8.5)

ax_J, ax_cg, ax_total, ax_time = (fig.add_subplot(grid_bottom[0, i]) for i in range(4))
for r in runs:
    ax_J.semilogy(r['evaluations'], np.minimum.accumulate(r['mismatch']), color=r['color'], ls=r['style'], lw=1.5,
                  label=r['label'])
    ax_cg.semilogy(np.arange(1, len(r['cg']) + 1), r['cg'] / 6, color=r['color'], ls=r['style'], lw=1.2)
x = np.arange(len(runs))
for ax, values, title in ((ax_total, [r['cg'].sum() for r in runs], 'CG iterations, whole optimization'),
                          (ax_time, [r['time'] / 60 for r in runs], 'wall time [min]')):
    bars = ax.bar(x, values, color=[r['color'] for r in runs], edgecolor=[r['color'] for r in runs])
    for bar, r in zip(bars, runs):
        if r['style'] != '-':
            bar.set_facecolor('white')
            bar.set_hatch({':': '...', '--': '///', '-.': 'xx'}.get(r['style'], '\\\\'))
    ax.bar_label(bars, labels=[f'{v / 1e3:.0f}k' if v > 100 else f'{v:.1f}' for v in values], fontsize=7)
    ax.set_xticks(x, [r['short'] for r in runs], fontsize=7.5, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_yscale('log')
    ax.margins(y=0.15)
    ax.set_title(title, fontsize=10)
ax_J.set(xlabel='objective evaluation', ylabel='stress mismatch J (best so far)', title='fit of the target')
ax_cg.set(xlabel='objective evaluation', ylabel='CG iterations per solve', title='cost per solve')
ax_J.legend(fontsize=7, loc='upper right')
for ax in (ax_J, ax_cg, ax_total, ax_time):
    ax.grid(True, alpha=0.3)
fig.suptitle('Sharp two-phase reaction-diffusion level set (RD) vs phase field and Hamilton-Jacobi level set (HJ), '
             '64²  —  designs as the solver sees them (2 triangles per pixel)', fontsize=12)
figure_name = os.path.join(HERE, 'figures', 'TO_reaction_diffusion_comparison_N64.png')
fig.savefig(figure_name, bbox_inches='tight')
print(f'figure: {figure_name}')
