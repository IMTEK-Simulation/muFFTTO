"""All runs of one grid on one page: a row per soft phase with CG per evaluation, cumulative CG and the final design of
every preconditioner -> figures/TO_<CG_TAG>_<N>_soft_phases[_eta<ETA>].png

    NPIX=32 CG_TAG=energy-warm ETA=0.0625 python plot_TO_soft_phases.py
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
DATA = os.path.join(HERE, 'data', 'example_2D_elasticity_TO')
N = int(os.environ.get('NPIX', 32))
CG_TAG = os.environ.get('CG_TAG', 'energy-warm')
ETA = os.environ.get('ETA')
PRECS = [('Green', 'Green (C_0)', '#2a78d6'), ('Green_Jacobi', 'Green-Jacobi (2μI)', '#eb6834'),
         ('dual', 'dual (C_0)', '#1baf7a')]


def cg_from_log(soft, prec, eta):
    """CG iterations per evaluation (forward + adjoint, 3 load cases) from the run log of a run without a final
    file (stopped before the end); logs/TO_N<N>_soft1e-<e>_eta1-<1/eta>_warm_<prec>.log"""
    log = os.path.join(HERE, 'logs', f'TO_N{N}_soft1e-{-int(round(np.log10(soft)))}_eta1-{int(round(1 / eta))}_warm_'
                                     f'{prec}.log')
    per_eval, total = [], 0
    for line in open(log):
        if 'nb_ steps CG mech' in line or 'nb_steps CG adjoint' in line:
            total += int(line.split('=')[1].split(',')[0])
        elif re.match(r'^ *\d+ +[0-9.]+e[+-]\d+ ', line):
            per_eval.append(total)
            total = 0
    return np.array(per_eval, dtype=float)


runs = {}
for path in glob.glob(os.path.join(DATA, f'N{N}_soft*_{CG_TAG}_*_history.nc')):
    m = re.match(rf'N{N}_soft([0-9.e-]+)_{CG_TAG}_(Green_Jacobi|Green|dual)_eta_([0-9.e-]+)_w_5.0_l_bfgs_history',
                 os.path.basename(path))
    if m is None or (ETA is not None and float(m[3]) != float(ETA)):
        continue
    h = netCDF4.Dataset(path)
    final = path.replace('_history.nc', '_final.nc')
    if os.path.exists(final):
        f = netCDF4.Dataset(final)
        cg = (np.asarray(f.num_iteration_mech, dtype=float) + np.asarray(f.num_iteration_adjoint, dtype=float))
        runs[(float(m[1]), m[2])] = dict(cg=cg.reshape(-1, 3).sum(1), time=float(f.elapsed_time), stopped=False,
                                         objective=float(h['objective'][-1]),
                                         design=np.asarray(f.variables[list(f.variables)[-1]][:]).squeeze())
    elif CG_TAG == 'energy-warm' and ETA is not None:          # stopped run: log + last stored iterate
        runs[(float(m[1]), m[2])] = dict(cg=cg_from_log(float(m[1]), m[2], float(ETA)), time=np.nan, stopped=True,
                                         objective=float(h['objective'][-1]),
                                         design=np.asarray(h['phase_field_iterate'][-1, 0]).squeeze())

softs = sorted({s for s, _ in runs}, reverse=True)
fig, axes = plt.subplots(len(softs), 5, figsize=(21, 4.3 * len(softs)), facecolor='#fcfcfb',
                         gridspec_kw={'width_ratios': [1.5, 1.5, 1, 1, 1]}, squeeze=False)
for row, soft in zip(axes, softs):
    ends = []
    for p, label, c in PRECS:
        if (soft, p) not in runs:
            continue
        r = runs[(soft, p)]
        ev = np.arange(1, len(r['cg']) + 1)
        row[0].semilogy(ev, r['cg'], color=c, lw=1.2, label=label)
        row[1].plot(ev, np.cumsum(r['cg']), color=c, lw=2, label=label)
        ends.append((r['cg'].sum(), ev[-1], r['time']))
    ends.sort()
    top = ends[-1][0]
    for i, (total, n_ev, wall) in enumerate(ends):          # end labels, pushed apart when the totals are close
        dy = 9 if i > 0 and total - ends[i - 1][0] < 0.12 * top else 0
        row[1].annotate(f'{int(total):,}\n' + ('stopped' if np.isnan(wall) else f'{wall:.0f} s'), (n_ev, total),
                        textcoords='offset points', xytext=(4, dy),
                        fontsize=8, va='center')
    for ax, title in ((row[0], f'soft phase {soft:g} C_0: CG per evaluation'), (row[1], 'cumulative CG')):
        ax.set_title(title, loc='left', fontsize=10)
        ax.set_xlabel('objective evaluation')
        ax.grid(alpha=0.3)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    row[0].legend(frameon=False, fontsize=8)
    for ax, (p, label, c) in zip(row[2:], PRECS):
        ax.set_xticks([])
        ax.set_yticks([])
        if (soft, p) not in runs:
            continue
        r = runs[(soft, p)]
        ax.pcolormesh(r['design'].T, cmap='Greys', vmin=0, vmax=1)          # exact pixels
        ax.set_aspect('equal')
        ax.set_title(f'{label}\nobjective {r["objective"]:.4g}, {len(r["cg"])} evaluations'
                     + (' (stopped)' if r['stopped'] else ''), fontsize=9)
start = 'warm start' if CG_TAG == 'energy-warm' else 'cold start'
fig.suptitle(f'Topology optimization {N}², {start} + energy stop 1e-6' + (f', eta = {ETA}' if ETA else ''),
             x=0.01, ha='left', fontweight='bold')
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = os.path.join(HERE, 'figures', f'TO_{CG_TAG}_{N}_soft_phases' + (f'_eta{ETA}' if ETA else '') + '.png')
fig.savefig(out, dpi=130, facecolor='#fcfcfb')
print(out)
