"""Level set with vs without reinitialization during the optimization (Green-Jacobi, eps continuation 2-1-0.5-0,
64^2): objective J and CG iterations per solve over the objective evaluations, final designs on their exact pixels
-> figures/TO_level_set_reinit_comparison_N64.png

    python plot_TO_level_set_reinit_comparison.py
"""
import os

import matplotlib
import netCDF4
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
RUN = os.path.join(HERE, 'data', 'example_2D_elasticity_TO_level_set',
                   'N64_soft1e-05_energy_Green_Jacobi_w_5.0_p1_eps2-1-0.5-0_alpha2_holes4')
RUNS = [('reinitialized every 10 iterations', RUN, '#9a9994'), ('no reinitialization', RUN + '_noreinit', '#eb6834')]

fig = plt.figure(figsize=(15, 8.5), dpi=150)
grid = fig.add_gridspec(2, 4, height_ratios=[1, 1.1], hspace=0.35, wspace=0.3)
ax_J = fig.add_subplot(grid[0, :2])
ax_cg = fig.add_subplot(grid[0, 2:])
for i, (label, path, color) in enumerate(RUNS):
    history = netCDF4.Dataset(path + '_history.nc')
    final = netCDF4.Dataset(path + '_final.nc')
    evaluations = np.asarray(history['nb_evaluations'][:]).astype(int)
    objective = np.asarray(history['objective'][:])
    eps = np.asarray(history['eps'][:])
    cg = (np.asarray(final.num_iteration_mech, dtype=float) +
          np.asarray(final.num_iteration_adjoint, dtype=float)).reshape(-1, 3).sum(1)[:-1]  # last: post-processing
    ax_J.semilogy(evaluations, objective, color=color, lw=1.5, label=f'{label}: J = {objective[-1]:.3g}')
    for switch in np.flatnonzero(np.diff(eps)) + 1:
        ax_J.axvline(evaluations[switch], color=color, ls=':', lw=1)
    ax_cg.plot(np.arange(1, len(cg) + 1), cg / 6, color=color, lw=1.2,
               label=f'{label}: {len(cg)} evaluations, {int(cg.sum()):,} CG in total')
    ax = fig.add_subplot(grid[1, 2 * i:2 * i + 2])
    ax.imshow(np.tile(np.asarray(final['phase_field_solution'][:])[0, 0], (2, 2)).T, origin='lower',
              cmap='Greys_r', vmin=0, vmax=1, interpolation='nearest', extent=[0, 2, 0, 2])
    ax.set_xticks([]), ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2.5)
    C = np.asarray(final.homogenized_C_voigt).reshape(3, 3)
    ax.set_title(f'{label}  (2x2 cells)\nfinal J = {objective[-1]:.3g},  C11 = {C[0, 0]:.3f}, C22 = {C[1, 1]:.3f}, '
                 f'C12 = {C[0, 1]:.3f}, C66 = {C[2, 2]:.3f}', fontsize=10)
target = np.asarray(netCDF4.Dataset(RUN + '_final.nc').target_C_voigt).reshape(3, 3)
ax_J.set(xlabel='objective evaluation', ylabel='objective J  (stress mismatch)',
         title='fit of the target (dotted: switch to the next ε)')
ax_cg.set(xlabel='objective evaluation', ylabel='CG iterations per solve', title='cost per solve (Green-Jacobi)')
ax_cg.set_ylim(0, None)
for ax in (ax_J, ax_cg):
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
fig.suptitle(f'Level set, 64², ε = 2 → 1 → 0.5 → sharp: with vs without reinitialization   '
             f'(target C11 = C22 = {target[0, 0]:.3f}, C12 = {target[0, 1]:.3f}, C66 = {target[2, 2]:.3f})',
             fontsize=12)
figure_name = os.path.join(HERE, 'figures', 'TO_level_set_reinit_comparison_N64.png')
fig.savefig(figure_name, bbox_inches='tight')
print(f'figure: {figure_name}')
