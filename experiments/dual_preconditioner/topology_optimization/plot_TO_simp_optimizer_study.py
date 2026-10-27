"""Optimizer study (run_simp_optimizer_study.sh): L-BFGS-B, GCMMA and plain MMA on the identical SIMP problem
(example_2D_elasticity_TO_simp_mma.py, 64^2, void 1e-5, Green-Jacobi), 5 start designs each:
    left      final designs thresholded rho > 0.5 (exactly two-phase, exact pixels), one row per optimizer, one column
              per start; J of each in the title
    right     per optimizer, every start as a dot and the median as a bar: J of the thresholded design, J of the
              projected (grey) design, number of design evaluations, total CG iterations, wall time
-> figures/TO_simp_optimizer_study_N64.png   (prints the table as well)

    python plot_TO_simp_optimizer_study.py
"""
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
DATA = os.path.join(HERE, 'data', 'example_2D_elasticity_TO_simp_mma')
TAG = 'p3_r2px64_32b1-32b2-32b4-64b4-64b8-64b16-64b32'
OPTIMIZERS = [('lbfgs', 'L-BFGS-B', '#2a78d6'), ('gcmma', 'GCMMA', '#1baa7d'), ('mma', 'MMA (plain)', '#eb6834')]
SEEDS = [0, 1, 2, 3, 4]

results = {}
for opt, _, _ in OPTIMIZERS:
    for seed in SEEDS:
        path = os.path.join(DATA, f'soft1e-05_Green_Jacobi_{opt}_{TAG}_seed{seed}.npz')
        if os.path.exists(path):
            d = np.load(path)
            results[opt, seed] = dict(J_sharp=float(d['J_sharp']), J_grey=float(d['J'][-1]), sharp=d['sharp_design'],
                                      evaluations=len(d['evaluations']), cg=3 * d['evaluations'][:, 2].sum(),
                                      time=float(d['elapsed_time']) / 60, C12=float(d['C_sharp_voigt'][0, 1]))

METRICS = [('J_sharp', 'J, thresholded\n(two-phase)', True), ('J_grey', 'J, projected\n(grey)', True),
           ('evaluations', 'design\nevaluations', False), ('cg', 'CG iterations\n(total)', True),
           ('time', 'wall time\n[min]', False)]
print(f'{"optimizer":12s} {"seed":>4s} {"J two-phase":>12s} {"J grey":>10s} {"evals":>6s} {"CG":>8s} {"min":>6s}')
for opt, label, _ in OPTIMIZERS:
    for seed in SEEDS:
        if (opt, seed) in results:
            r = results[opt, seed]
            print(f'{label:12s} {seed:4d} {r["J_sharp"]:12.3e} {r["J_grey"]:10.2e} {r["evaluations"]:6d} '
                  f'{r["cg"]:8.0f} {r["time"]:6.2f}')
    values = [results[opt, s] for s in SEEDS if (opt, s) in results]
    if values:
        print(f'{label:12s} {"med":>4s} ' + ' '.join(f'{np.median([v[k] for v in values]):{w}.3g}' for k, w in
                                                     (('J_sharp', 12), ('J_grey', 10), ('evaluations', 6),
                                                      ('cg', 8), ('time', 6))))

fig = plt.figure(figsize=(17, 6.4), dpi=150)
grid_designs = fig.add_gridspec(len(OPTIMIZERS), len(SEEDS), left=0.05, right=0.47, top=0.88, bottom=0.04,
                                wspace=0.06, hspace=0.22)
grid_metrics = fig.add_gridspec(1, len(METRICS), left=0.53, right=0.99, top=0.84, bottom=0.12, wspace=0.75)
for i, (opt, label, color) in enumerate(OPTIMIZERS):
    for j, seed in enumerate(SEEDS):
        ax = fig.add_subplot(grid_designs[i, j])
        ax.set_xticks([]), ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color), spine.set_linewidth(2)
        if (opt, seed) not in results:
            ax.text(0.5, 0.5, 'missing', ha='center', va='center', transform=ax.transAxes)
            continue
        r = results[opt, seed]
        ax.imshow(r['sharp'].T, origin='lower', cmap='Greys_r', vmin=0, vmax=1, interpolation='nearest')
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
        jitter = np.linspace(-0.15, 0.15, len(values))
        ax.plot(i + jitter, values, 'o', color=color, ms=5, mec='white', mew=0.8)
    ax.set_xticks(range(len(OPTIMIZERS)), [label for _, label, _ in OPTIMIZERS], rotation=45, ha='right',
                  rotation_mode='anchor', fontsize=8.5)
    if log:
        ax.set_yscale('log')
    else:
        ax.set_ylim(0, None)
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
fig.text(0.76, 0.93, 'dots: the 5 starts;  bars: median', ha='center', fontsize=9.5, color='#52514e')
fig.text(0.76, 0.0, 'wall time with 6 runs sharing the machine (2 threads each); CG iterations are the '
         'machine-independent cost', ha='center', fontsize=8.5, color='#52514e')
fig.text(0.26, 0.955, 'final designs, thresholded ρ > 0.5 (two-phase)', ha='center', fontsize=10.5)
fig.suptitle('Optimizer study, SIMP + filter + projection, 64², void 10⁻⁵, Green-Jacobi: identical settings, '
             '5 start designs per optimizer', fontsize=12.5, y=1.0)
figure_name = os.path.join(HERE, 'figures', 'TO_simp_optimizer_study_N64.png')
fig.savefig(figure_name, bbox_inches='tight')
print(f'figure: {figure_name}')
