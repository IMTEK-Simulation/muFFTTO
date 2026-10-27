"""Optimizer study on the phase-field formulation (run_phase_field_optimizer_study.sh): L-BFGS-B, GCMMA and plain MMA
on the objective of example_2D_elasticity_TO.py (64^2, void 1e-5, eta 1/32, Green-Jacobi), 5 start designs each:
    left      final phase fields thresholded at the quadrature points > 0.5 (exactly two-phase, 2 triangles per pixel,
              exact geometry), one row per optimizer, one column per start; J of each in the title
    right     per optimizer, every start as a dot and the median as a bar: J of the thresholded design, J of the
              phase-field design, design evaluations, total CG iterations, wall time; and the thresholded J of the
              SIMP study (same starts, same optimizers) next to the phase-field one
-> figures/TO_phase_field_optimizer_study_N64.png   (prints the table as well)

    python plot_TO_phase_field_optimizer_study.py
"""
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

HERE = os.path.dirname(os.path.realpath(__file__))
DATA = os.path.join(HERE, 'data', 'example_2D_elasticity_TO_phase_field_optimizers')
DATA_SIMP = os.path.join(HERE, 'data', 'example_2D_elasticity_TO_simp_mma')
TAG_SIMP = 'p3_r2px64_32b1-32b2-32b4-64b4-64b8-64b16-64b32'
OPTIMIZERS = [('lbfgs', 'L-BFGS-B', '#2a78d6'), ('gcmma', 'GCMMA', '#1baa7d'), ('mma', 'MMA (plain)', '#eb6834')]
SEEDS = [0, 1, 2, 3, 4]


def triangulation(n):
    """Lower-left (quadrature point 0) and upper-right (1) triangle of each pixel, nodes at the pixel corners."""
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing='ij')
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    node = lambda a, b: a * (n + 1) + b
    lower = np.stack([node(ii, jj), node(ii + 1, jj), node(ii, jj + 1)], -1).reshape(-1, 3)
    upper = np.stack([node(ii + 1, jj), node(ii, jj + 1), node(ii + 1, jj + 1)], -1).reshape(-1, 3)
    return Triangulation(i.ravel() / n, j.ravel() / n, np.concatenate([lower, upper]))


results, simp = {}, {}
for opt, _, _ in OPTIMIZERS:
    for seed in SEEDS:
        path = os.path.join(DATA, f'N64_soft1e-05_Green_Jacobi_{opt}_eta0.03125_dw1_seed{seed}.npz')
        if os.path.exists(path):
            d = np.load(path)
            results[opt, seed] = dict(J_sharp=float(d['J_sharp']), J_grey=float(d['J_final']), sharp=d['sharp_quad'],
                                      evaluations=len(d['evaluations']), cg=d['evaluations'][:, 2].sum(),
                                      time=float(d['elapsed_time']) / 60, iterations=len(d['F']) - 1)
        path = os.path.join(DATA_SIMP, f'soft1e-05_Green_Jacobi_{opt}_{TAG_SIMP}_seed{seed}.npz')
        if os.path.exists(path):
            simp[opt, seed] = float(np.load(path)['J_sharp'])

METRICS = [('J_sharp', 'J, thresholded\n(two-phase)', True), ('J_grey', 'J, phase field\n(continuous)', True),
           ('evaluations', 'design\nevaluations', False), ('cg', 'CG iterations\n(total)', True),
           ('time', 'wall time\n[min]', False)]
print(f'{"optimizer":12s} {"seed":>4s} {"J two-phase":>12s} {"J pf":>10s} {"iters":>6s} {"evals":>6s} {"CG":>8s} '
      f'{"min":>6s} {"J SIMP":>10s}')
for opt, label, _ in OPTIMIZERS:
    for seed in SEEDS:
        if (opt, seed) in results:
            r = results[opt, seed]
            print(f'{label:12s} {seed:4d} {r["J_sharp"]:12.3e} {r["J_grey"]:10.2e} {r["iterations"]:6d} '
                  f'{r["evaluations"]:6d} {r["cg"]:8.0f} {r["time"]:6.2f} {simp.get((opt, seed), np.nan):10.2e}')
    values = [results[opt, s] for s in SEEDS if (opt, s) in results]
    if values:
        print(f'{label:12s} {"med":>4s} ' + ' '.join(f'{np.median([v[k] for v in values]):{w}.3g}' for k, w in
                                                     (('J_sharp', 12), ('J_grey', 10), ('iterations', 6),
                                                      ('evaluations', 6), ('cg', 8), ('time', 6))))

mesh = triangulation(64)
fig = plt.figure(figsize=(19, 6.4), dpi=150)
grid_designs = fig.add_gridspec(len(OPTIMIZERS), len(SEEDS), left=0.04, right=0.42, top=0.88, bottom=0.04,
                                wspace=0.06, hspace=0.22)
grid_metrics = fig.add_gridspec(1, len(METRICS) + 1, left=0.48, right=0.99, top=0.84, bottom=0.12, wspace=0.75)
for i, (opt, label, color) in enumerate(OPTIMIZERS):
    for j, seed in enumerate(SEEDS):
        ax = fig.add_subplot(grid_designs[i, j])
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_aspect('equal')
        for spine in ax.spines.values():
            spine.set_edgecolor(color), spine.set_linewidth(2)
        if (opt, seed) not in results:
            ax.text(0.5, 0.5, 'missing', ha='center', va='center', transform=ax.transAxes)
            continue
        r = results[opt, seed]
        ax.tripcolor(mesh, facecolors=r['sharp'].reshape(-1), cmap='Greys_r', vmin=0, vmax=1, edgecolors='none')
        ax.set_xlim(0, 1), ax.set_ylim(0, 1)
        ax.set_title(f'J = {r["J_sharp"]:.1e}', fontsize=8.5)
        if j == 0:
            ax.set_ylabel(label, fontsize=10, color='#222222')
        if i == 0:
            ax.text(0.5, 1.32, f'start {seed}', ha='center', transform=ax.transAxes, fontsize=9, color='#52514e')
for k, (key, title, log) in enumerate(METRICS):
    ax = fig.add_subplot(grid_metrics[0, k])
    for i, (opt, label, color) in enumerate(OPTIMIZERS):
        values = np.array([results[opt, s][key] for s in SEEDS if (opt, s) in results])
        if len(values) == 0:
            continue
        ax.bar(i, np.median(values), width=0.6, color=color, alpha=0.25, edgecolor=color, lw=1.5)
        ax.plot(i + np.linspace(-0.15, 0.15, len(values)), values, 'o', color=color, ms=5, mec='white', mew=0.8)
    ax.set_xticks(range(len(OPTIMIZERS)), [label for _, label, _ in OPTIMIZERS], rotation=45, ha='right',
                  rotation_mode='anchor', fontsize=8.5)
    if log:
        ax.set_yscale('log')
    else:
        ax.set_ylim(0, None)
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
# thresholded J, phase field (filled) vs SIMP (open), paired by start
ax = fig.add_subplot(grid_metrics[0, len(METRICS)])
for i, (opt, label, color) in enumerate(OPTIMIZERS):
    for s in SEEDS:
        if (opt, s) in results and (opt, s) in simp:
            ax.plot([i - 0.17, i + 0.17], [results[opt, s]['J_sharp'], simp[opt, s]], color=color, lw=0.7, alpha=0.5)
    ax.plot(np.full(5, i - 0.17), [results[opt, s]['J_sharp'] for s in SEEDS if (opt, s) in results][:5] +
            [np.nan] * (5 - sum((opt, s) in results for s in SEEDS)), 'o', color=color, ms=5, mec='white', mew=0.8)
    ax.plot(np.full(5, i + 0.17), [simp.get((opt, s), np.nan) for s in SEEDS], 'o', mfc='white', mec=color, ms=5,
            mew=1.2)
ax.set_yscale('log')
ax.set_xticks(range(len(OPTIMIZERS)), [label for _, label, _ in OPTIMIZERS], rotation=45, ha='right',
              rotation_mode='anchor', fontsize=8.5)
ax.set_title('J two-phase:\n● phase field  ○ SIMP', fontsize=10)
ax.grid(True, axis='y', alpha=0.3)
fig.text(0.735, 0.93, 'dots: the 5 starts;  bars: median;  lines in the last panel pair the same start',
         ha='center', fontsize=9.5, color='#52514e')
fig.text(0.735, 0.0, 'wall time with 6 runs sharing the machine (2 threads each); CG iterations (forward + adjoint) '
         'are the machine-independent cost', ha='center', fontsize=8.5, color='#52514e')
fig.text(0.23, 0.955, 'final designs, thresholded at the quadrature points > 0.5 (two-phase)', ha='center',
         fontsize=10.5)
fig.suptitle('Optimizer study, phase field (p = 2, η = 1/32, double well), 64², void 10⁻⁵, Green-Jacobi: identical '
             'settings, the 5 start designs of the SIMP study', fontsize=12.5, y=1.0)
figure_name = os.path.join(HERE, 'figures', 'TO_phase_field_optimizer_study_N64.png')
fig.savefig(figure_name, bbox_inches='tight')
print(f'figure: {figure_name}')
