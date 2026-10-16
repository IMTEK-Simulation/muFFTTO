"""Mesh dependence of the three preconditioners along the topology optimization (eta = 1/32 fixed, warm-start energy
stop): mean CG iterations per objective evaluation (forward + adjoint, 3 load cases) over the grid, one panel per soft
phase -> figures/TO_energy-warm_mesh_dependence_eta0.03125.png and a table on stdout.
Finished runs: filled markers (counts from the final file). Runs without a final file (stopped or still running):
hollow markers, mean over the evaluations done so far (from the log), which over-weights the expensive early designs.

    python plot_TO_mesh_dependence.py
"""
import os
import re

import matplotlib
import netCDF4
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
DATA = os.path.join(HERE, 'data', 'example_2D_elasticity_TO')
ETA = 0.03125
GRIDS = [32, 64, 128, 256, 512, 1024]
SOFTS = [1e-3, 1e-5]
PRECS = [('Green', 'Green (C_0)', '#2a78d6', 'o'), ('Green_Jacobi', 'Green-Jacobi (2μI)', '#eb6834', 's'),
         ('dual', 'dual (C_0)', '#1baf7a', '^')]


def cg_from_log(N, soft, prec):
    log = os.path.join(HERE, 'logs', f'TO_N{N}_soft1e-{-int(round(np.log10(soft)))}_eta1-{int(round(1 / ETA))}_warm_'
                                     f'{prec}.log')
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


def mean_cg(N, soft, prec):
    """(mean CG per evaluation, number of evaluations, finished) or None"""
    final = os.path.join(DATA, f'N{N}_soft{soft:g}_energy-warm_{prec}_eta_{ETA}_w_5.0_l_bfgs_final.nc')
    if os.path.exists(final):
        f = netCDF4.Dataset(final)
        cg = (np.asarray(f.num_iteration_mech, dtype=float) + np.asarray(f.num_iteration_adjoint, dtype=float))
        cg = cg.reshape(-1, 3).sum(1)
        return cg.mean(), len(cg), True
    cg = cg_from_log(N, soft, prec)
    return None if cg is None else (cg.mean(), len(cg), False)


fig, axes = plt.subplots(1, len(SOFTS), figsize=(11, 4.2), facecolor='#fcfcfb', sharey=True)
print('| soft | preconditioner | ' + ' | '.join(f'{N}²' for N in GRIDS) + ' |')
print('|---|---|' + '---|' * len(GRIDS))
grids_used = set()
for ax, soft in zip(axes, SOFTS):
    for p, label, c, m in PRECS:
        pts = [(N, mean_cg(N, soft, p)) for N in GRIDS]
        pts = [(N, r) for N, r in pts if r is not None]
        grids_used.update(N for N, _ in pts)
        print(f'| {soft:g} | {label} | ' + ' | '.join(
            next((f'{r[0]:.0f}' + ('' if r[2] else f' ({r[1]} ev., unfinished)') for n, r in pts if n == N), '')
            for N in GRIDS) + ' |')
        if not pts:
            continue
        ax.plot([N for N, _ in pts], [r[0] for _, r in pts], color=c, lw=2, label=label)
        for N, (mean, n_ev, finished) in pts:
            ax.plot(N, mean, marker=m, ms=8, color=c, mfc=c if finished else '#fcfcfb', mew=2)
            ax.annotate(f'{mean:.0f}', (N, mean), textcoords='offset points', xytext=(7, -3), fontsize=8,
                        color='0.25')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_title(f'soft phase {soft:g} C_0', loc='left', fontsize=10)
    ax.set_xlabel('grid')
    ax.grid(alpha=0.3, which='both')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
for ax in axes:
    used = sorted(grids_used)
    ax.set_xticks(used, [f'{N}²' for N in used])
    ax.set_xlim(used[0] / 1.3, used[-1] * 1.5)
axes[0].set_ylabel('mean CG iterations per objective evaluation\n(forward + adjoint, 3 load cases)')
axes[0].plot([], [], 'k^', mfc='#fcfcfb', mew=2, label='stopped / running (mean so far)')
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle('Topology optimization: mesh dependence, eta = 1/32, warm start + energy stop 1e-6', x=0.01, ha='left',
             fontweight='bold')
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = os.path.join(HERE, 'figures', f'TO_energy-warm_mesh_dependence_eta{ETA}.png')
fig.savefig(out, dpi=140, facecolor='#fcfcfb')
print(out)
