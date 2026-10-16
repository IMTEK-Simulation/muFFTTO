"""Summary of the overnight topology optimizations (run_TO_queue.py): for every finished (grid, soft phase) the three
preconditioners side by side -> figures/TO_energy_<N>_soft<s>.png and a table on stdout.

    python plot_TO_preconditioner_comparison.py
"""
import glob
import os
import re

import matplotlib
import netCDF4
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
# the local copy of example_2D_elasticity_TO.py and its data (this folder)
TO = os.path.dirname(os.path.realpath(__file__))
DATA = os.path.join(TO, 'data', 'example_2D_elasticity_TO')
OUT = os.path.join(HERE, 'figures')
PRECS = [('Green', 'Green (C_0)', '#2a78d6'), ('Green_Jacobi', 'Green-Jacobi (2μI)', '#eb6834'),
         ('dual', 'dual (C_0)', '#1baf7a')]

# CG_TAG: 'energy' = cold starts (runs of 2026-10-15/16), 'energy-warm' = warm starts (2026-10-16 on);
# ETA (optional): only runs with this eta, e.g. ETA=0.0625
CG_TAG = os.environ.get('CG_TAG', 'energy')
ETA_FILTER = os.environ.get('ETA')
runs = {}
for path in glob.glob(os.path.join(DATA, f'N*_soft*_{CG_TAG}_*_final.nc')):
    m = re.match(rf'N(\d+)_soft([0-9.e-]+)_{CG_TAG}_(Green_Jacobi|Green|dual)_eta_([0-9.e-]+)_w_5.0_l_bfgs_final',
                 os.path.basename(path))
    if m is None or (ETA_FILTER is not None and float(m[4]) != float(ETA_FILTER)):
        continue
    f = netCDF4.Dataset(path)
    mech = np.asarray(f.num_iteration_mech, dtype=float).reshape(-1, 3).sum(1)
    adj = np.asarray(f.num_iteration_adjoint, dtype=float).reshape(-1, 3).sum(1)
    h = netCDF4.Dataset(path.replace('_final.nc', '_history.nc'))
    runs[(int(m[1]), float(m[2]), m[3])] = dict(
        cg=mech + adj, mech=mech.sum(), adj=adj.sum(), time=float(f.elapsed_time),
        objective=np.asarray(h['objective'][:]), evaluations=np.asarray(h['nb_evaluations'][:]),
        design=np.asarray(f.variables[list(f.variables)[-1]][:]).squeeze())

print('| grid | soft | preconditioner | evaluations | CG total | CG / evaluation | final objective | wall time |')
print('|---|---|---|---|---|---|---|---|')
for N, soft in sorted({(N, s) for N, s, _ in runs}):
    present = [(p, label, c) for p, label, c in PRECS if (N, soft, p) in runs]
    for p, label, c in present:
        r = runs[(N, soft, p)]
        print(f'| {N}² | {soft:g} | {label} | {len(r["cg"])} | {int(r["cg"].sum()):,} | {r["cg"].mean():.0f} | '
              f'{r["objective"][-1]:.4g} | {r["time"]:.0f} s |')
    fig = plt.figure(figsize=(15, 8.5), facecolor='#fcfcfb')
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
    ax_cg, ax_cum, ax_obj = (fig.add_subplot(gs[0, i]) for i in range(3))
    for i, (p, label, c) in enumerate(present):
        r = runs[(N, soft, p)]
        ev = np.arange(1, len(r['cg']) + 1)
        ax_cg.semilogy(ev, r['cg'], color=c, lw=1.2, label=label)
        ax_cum.plot(ev, np.cumsum(r['cg']), color=c, lw=2, label=label)
        ax_obj.semilogy(r['evaluations'], r['objective'], color=c, lw=2, label=label)
        ax = fig.add_subplot(gs[1, i])
        ax.pcolormesh(r['design'].T, cmap='Greys', vmin=0, vmax=1)            # exact pixels
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'{label}: objective {r["objective"][-1]:.4g}, {len(r["cg"])} evaluations', fontsize=10)
    ends = sorted((runs[(N, soft, p)]['cg'].sum(), len(runs[(N, soft, p)]['cg']), runs[(N, soft, p)]['time'])
                  for p, _, _ in present)
    top = ends[-1][0]
    for i, (total, n_ev, wall) in enumerate(ends):           # end labels, pushed apart when the totals are close
        dy = 0
        if i > 0 and total - ends[i - 1][0] < 0.12 * top:
            dy = 9
        if i < len(ends) - 1 and ends[i + 1][0] - total < 0.12 * top:
            dy = -9
        ax_cum.annotate(f'{int(total):,}\n{wall:.0f} s', (n_ev, total), textcoords='offset points', xytext=(4, dy),
                        fontsize=8, va='center')
    for ax, t in ((ax_cg, 'CG per evaluation (3 load cases, forward + adjoint)'), (ax_cum, 'cumulative CG'),
                  (ax_obj, 'objective')):
        ax.set_title(t, loc='left', fontsize=10)
        ax.set_xlabel('objective evaluation')
        ax.grid(alpha=0.3)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    ax_cg.legend(frameon=False, fontsize=9)
    start = 'warm start' if CG_TAG == 'energy-warm' else 'cold start'
    fig.suptitle(f'Topology optimization {N}², soft phase {soft:g} C_0, {start} + energy stop 1e-6, '
                 f'eta = {ETA_FILTER or "1/64"}',
                 x=0.01, ha='left', fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f'TO_{CG_TAG}_{N}_soft{soft:g}' + (f'_eta{ETA_FILTER}' if ETA_FILTER else '') + '.png'),
                dpi=130, facecolor='#fcfcfb')
    plt.close(fig)
