"""MMA vs L-BFGS-B on the identical SIMP problem (example_2D_elasticity_TO_simp_mma.py: filter 2 px, Heaviside
projection beta 1 -> 32, grid 32^2 -> 64^2, same gradient, same stage stop), each with Green and Green-Jacobi, 64^2,
void 1e-5:
    top       final designs thresholded rho > 0.5 (exactly two-phase; one value per pixel, exact pixels) with their J
              (all final J verified independently by verify_TO_design.py)
    bottom    J of the projected (grey) design over the accepted iterations; CG iterations per solve; J of the
              thresholded design; total CG iterations of all evaluations (rejected MMA steps included); wall time
-> figures/TO_simp_optimizer_comparison_N64.png

    python plot_TO_simp_optimizer_comparison.py
"""
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
DATA = os.path.join(HERE, 'data', 'example_2D_elasticity_TO_simp_mma')
TAG = 'p3_r2px64_32b1-32b2-32b4-64b4-64b8-64b16-64b32'
BLUE, ORANGE = '#2a78d6', '#eb6834'
RUNS = [  # label, optimizer, preconditioner, colour, line style
    ('L-BFGS-B, Green-Jacobi', 'lbfgs', 'Green_Jacobi', ORANGE, '-'),
    ('L-BFGS-B, Green', 'lbfgs', 'Green', BLUE, '-'),
    ('MMA, Green-Jacobi', 'mma', 'Green_Jacobi', ORANGE, '--'),
    ('MMA, Green', 'mma', 'Green', BLUE, '--'),
]

runs = []
for label, optimizer, prec, color, style in RUNS:
    d = np.load(os.path.join(DATA, f'soft1e-05_{prec}_{optimizer}_{TAG}.npz'))
    runs.append(dict(label=label, color=color, style=style, J=d['J'], cg=d['cg'], n=d['n'], beta=d['beta'],
                     sharp=d['sharp_design'], J_sharp=float(d['J_sharp']), C=d['C_sharp_voigt'],
                     cg_total=3 * d['evaluations'][:, 2].sum(), nb_evaluations=len(d['evaluations']),
                     time=float(d['elapsed_time'])))

fig = plt.figure(figsize=(14, 8.6), dpi=150)
grid_top = fig.add_gridspec(1, len(runs), left=0.04, right=0.96, top=0.87, bottom=0.55, wspace=0.12)
grid_bottom = fig.add_gridspec(1, 4, left=0.06, right=0.98, top=0.44, bottom=0.12, wspace=0.38,
                               width_ratios=[1.3, 1.3, 0.75, 0.75])
for k, r in enumerate(runs):
    ax = fig.add_subplot(grid_top[0, k])
    ax.imshow(r['sharp'].T, origin='lower', extent=[0, 1, 0, 1], cmap='Greys_r', vmin=0, vmax=1,
              interpolation='nearest')
    ax.set_xticks([]), ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(r['color']), spine.set_linewidth(2.5), spine.set_linestyle(r['style'])
    ax.set_title(f'{r["label"]}\nthresholded J = {r["J_sharp"]:.2e}\n'
                 f'C12 = {r["C"][0, 1]:.4f}  (target -0.0964)', fontsize=9.5)

ax_J, ax_cg, ax_sharp, ax_total = (fig.add_subplot(grid_bottom[0, i]) for i in range(4))
for r in runs:
    it = np.arange(len(r['J']))
    ax_J.semilogy(it, r['J'], color=r['color'], ls=r['style'], lw=1.5, label=r['label'])
    ax_cg.semilogy(it, r['cg'], color=r['color'], ls=r['style'], lw=1.2)
# stage boundaries differ per run: a vertical line at the grid change of each run
for r in runs:
    change = np.argmax(r['n'] == 64)
    ax_J.axvline(change, color=r['color'], ls=r['style'], lw=0.8, alpha=0.6)
ax_J.set(xlabel='accepted iteration (all stages)', ylabel='J of the projected (grey) design',
         title='objective during the optimization\n(vertical lines: grid switch 32² → 64²)')
ax_cg.set(xlabel='accepted iteration (all stages)', ylabel='CG iterations per solve',
          title='cost per solve\n(of the accepted design)')
ax_J.legend(fontsize=7.5, loc='lower left')
x = np.arange(len(runs))
for ax, values, title, fmt in ((ax_sharp, [r['J_sharp'] for r in runs], 'final J, thresholded\n(two-phase)',
                                lambda v: f'{v:.1e}'),
                               (ax_total, [r['cg_total'] for r in runs], 'CG iterations, all\nevaluations',
                                lambda v: f'{v / 1e3:.0f}k')):
    bars = ax.bar(x, values, color=[r['color'] for r in runs], edgecolor=[r['color'] for r in runs], lw=1.5)
    for bar, r in zip(bars, runs):
        if r['style'] == '--':
            bar.set_facecolor('white'), bar.set_hatch('///')
    ax.bar_label(bars, labels=[fmt(v) for v in values], fontsize=7)
    ax.set_yscale('log')
    ax.margins(y=0.2)
    ax.set_xticks(x, ['L-BFGS GJ', 'L-BFGS G', 'MMA GJ', 'MMA G'], fontsize=7.5, rotation=45, ha='right',
                  rotation_mode='anchor')
    ax.set_title(title, fontsize=10)
for ax in (ax_J, ax_cg, ax_sharp, ax_total):
    ax.grid(True, alpha=0.3)
fig.text(0.5, 0.015, 'wall time: ' + ',  '.join(f'{r["label"]} {r["time"] / 60:.1f} min ({r["nb_evaluations"]} '
                                               f'evaluations)' for r in runs), ha='center', fontsize=8.5)
fig.suptitle('SIMP + filter + Heaviside projection, 64², void 10⁻⁵: MMA vs L-BFGS-B, Green vs Green-Jacobi '
             '(one run each, same start)', fontsize=12)
figure_name = os.path.join(HERE, 'figures', 'TO_simp_optimizer_comparison_N64.png')
fig.savefig(figure_name, bbox_inches='tight')
print(f'figure: {figure_name}')
