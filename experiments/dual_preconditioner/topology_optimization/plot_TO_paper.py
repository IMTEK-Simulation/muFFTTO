"""Paper figures of the topology optimization (eta = 1/32, warm-start energy stop), drawn at the text width 4.8 in:
    part2_TO_mesh_dependence.pdf   mean CG iterations per objective evaluation over the grid, soft phase 1e-3 / 1e-5
    part2_TO_designs.pdf           final designs (dual runs) on the finest finished grid, both soft phases
and the LaTeX table rows on stdout.

    python plot_TO_paper.py [OUT_DIR]      (default: ~/PycharmProjects/Paper-Double-Green-Preconditioner/figures)
"""
import os
import re
import sys

import matplotlib
import netCDF4
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    '~/PycharmProjects/Paper-Double-Green-Preconditioner/figures')
DATA = os.path.join(HERE, 'data', 'example_2D_elasticity_TO')
ETA = 0.03125
GRIDS = [32, 64, 128, 256]
SOFTS = [1e-3, 1e-5]
PRECS = [('Green', r'Green $\mathbf{G}$', '#2a78d6', 'o'),
         ('Green_Jacobi', r'Green-Jacobi $\mathbf{D}^{-1/2}\mathbf{G}_{I}\mathbf{D}^{-1/2}$', '#eb6834', 's'),
         ('dual', r'dual $\mathbf{G}\,\mathbf{K}(\mathbb{C}_0\mathbb{C}^{-1}\mathbb{C}_0)\,\mathbf{G}$', '#1baf7a', '^')]
plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.5, 'legend.fontsize': 6})


def name(N, soft, prec, kind):
    return os.path.join(DATA, f'N{N}_soft{soft:g}_energy-warm_{prec}_eta_{ETA}_w_5.0_l_bfgs_{kind}.nc')


def cg_from_log(N, soft, prec):
    log = os.path.join(HERE, 'logs', f'TO_N{N}_soft1e-{-int(round(np.log10(soft)))}_eta1-32_warm_{prec}.log')
    if not os.path.exists(log):
        return None
    per_eval, total = [], 0
    for line in open(log):
        if 'nb_ steps CG mech' in line or 'nb_steps CG adjoint' in line:
            total += int(line.split('=')[1].split(',')[0])
        elif re.match(r'^ *\d+ +[0-9.]+e[+-]\d+ ', line):
            per_eval.append(total)
            total = 0
    return np.array(per_eval, dtype=float) if per_eval else None


def result(N, soft, prec):
    """dict(mean, evaluations, total, objective, finished) or None"""
    if os.path.exists(name(N, soft, prec, 'final')):
        f = netCDF4.Dataset(name(N, soft, prec, 'final'))
        cg = (np.asarray(f.num_iteration_mech, dtype=float) + np.asarray(f.num_iteration_adjoint, dtype=float))
        cg = cg.reshape(-1, 3).sum(1)
        objective = float(netCDF4.Dataset(name(N, soft, prec, 'history'))['objective'][-1])
        return dict(mean=cg.mean(), evaluations=len(cg), total=cg.sum(), objective=objective, finished=True)
    cg = cg_from_log(N, soft, prec)
    return None if cg is None else dict(mean=cg.mean(), evaluations=len(cg), total=cg.sum(), objective=np.nan,
                                        finished=False)


fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.5), sharey=True)
for ax, soft in zip(axes, SOFTS):
    for p, label, c, m in PRECS:
        pts = [(N, result(N, soft, p)) for N in GRIDS]
        pts = [(N, r) for N, r in pts if r is not None and r['finished']]
        if pts:
            ax.plot([N for N, _ in pts], [r['mean'] for _, r in pts], color=c, marker=m, ms=3.5, lw=1.3, label=label)
        stopped = [(N, r) for N, r in [(N, result(N, soft, p)) for N in GRIDS]
                   if r is not None and not r['finished'] and N in (128,)]
        for N, r in stopped:                                    # stopped run: hollow marker, mean so far
            ax.plot(N, r['mean'], color=c, marker=m, ms=3.5, mfc='white', lw=0)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    finished_grids = [N for N in GRIDS if any(result(N, s, p) is not None and result(N, s, p)['finished']
                                              for s in SOFTS for p, *_ in PRECS)]
    ax.set_xticks(finished_grids, [f'${N}^2$' for N in finished_grids])
    ax.set_xlim(finished_grids[0] / 1.3, finished_grids[-1] * 1.3)
    ax.set_title(f'soft phase $10^{{{int(round(np.log10(soft)))}}}\\,\\mathbb{{C}}_0$')
    ax.set_xlabel('grid')
    ax.grid(alpha=0.3, which='both')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel('mean CG iterations\nper objective evaluation')
fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2, frameon=False)
fig.tight_layout(rect=(0, 0.16, 1, 1))
fig.savefig(os.path.join(OUT, 'part2_TO_mesh_dependence.pdf'))
plt.close(fig)

# final designs of the dual runs on the finest grid that finished for both soft phases
N_design = max(N for N in GRIDS if all(os.path.exists(name(N, s, 'dual', 'final')) for s in SOFTS))
fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.9))
for ax, soft in zip(axes, SOFTS):
    f = netCDF4.Dataset(name(N_design, soft, 'dual', 'final'))
    design = np.asarray(f.variables[list(f.variables)[-1]][:]).squeeze()
    ax.pcolormesh(design.T, cmap='Greys', vmin=0, vmax=1, rasterized=True)          # exact pixels
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    objective = float(netCDF4.Dataset(name(N_design, soft, 'dual', 'history'))['objective'][-1])
    ax.set_title(f'soft phase $10^{{{int(round(np.log10(soft)))}}}\\,\\mathbb{{C}}_0$\nobjective {objective:.3f}')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'part2_TO_designs.pdf'), dpi=300)
plt.close(fig)

print(f'% designs: N = {N_design}')
print('% mean CG per evaluation (evaluations), Green / Green-Jacobi / dual')
for soft in SOFTS:
    for N in GRIDS:
        cells = []
        for p, *_ in PRECS:
            r = result(N, soft, p)
            cells.append('--' if r is None else (f'{r["mean"]:.0f}' + ('' if r['finished'] else r'$^\dagger$')))
        print(f'{soft:g} & ${N}^2$ & ' + ' & '.join(cells) + r' \\')
