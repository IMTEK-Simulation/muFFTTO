"""Level set vs phase field (64^2, soft 1e-5, energy CG stop with warm starts): CG iterations per solve and in total
for the Green and Green-Jacobi preconditioners -> figures/TO_level_set_vs_phase_field_N64.png and a table on stdout.

    python plot_TO_level_set_vs_phase_field.py
"""
import os

import matplotlib
import netCDF4
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
PF = os.path.join(HERE, 'data', 'example_2D_elasticity_TO', 'N64_soft1e-05_energy-warm_{}_eta_0.03125_w_5.0_l_bfgs')
LS = os.path.join(HERE, 'data', 'example_2D_elasticity_TO_level_set',
                  'N64_soft1e-05_energy_{}_w_5.0_p1_eps2_alpha2_holes4')
PRECS = [('Green', 'Green (C_0)', '#2a78d6'), ('Green_Jacobi', 'Green-Jacobi (2μI)', '#eb6834')]
LS_PF_START = LS.replace('_holes4', '_initphase')  # level set started from the phase-field start (LS_INIT=phase)
METHODS = [('phase field', PF, '-'), ('level set', LS, '--'), ('level set, PF start', LS_PF_START, ':'),
           ('level set from PF result', LS.replace('_holes4', '_initpf_final'), '-.')]  # LS_INIT=pf_final

runs = []
for method, template, style in METHODS:
    for prec, label, color in PRECS:
        path = template.format(prec) + '_final.nc'
        if not os.path.exists(path):
            print(f'missing: {path}')
            continue
        f = netCDF4.Dataset(path)
        mech = np.asarray(f.num_iteration_mech, dtype=float)       # one entry per solve, 3 load cases per evaluation
        adj = np.asarray(f.num_iteration_adjoint, dtype=float)
        if method.startswith('level set'):  # the last evaluation re-sets the final material (post-processing)
            mech, adj = mech[:-3], adj[:-3]
        runs.append(dict(method=method, prec=prec, label=f'{method}, {label}', color=color, style=style,
                         mech=mech, adj=adj, per_evaluation=(mech + adj).reshape(-1, 3).sum(1),
                         # stress mismatch sum_lc w f_sigma (the phase-field J also contains the regularization)
                         mismatch=(5.0 / 3) * float(np.atleast_1d(f.norms_sigma)[-1]) if method == 'phase field'
                         else float(np.atleast_1d(f.objective_history)[-1]),
                         design=np.asarray(f['phase_field_solution'][:]).squeeze(),
                         C_voigt=np.asarray(f.homogenized_C_voigt).reshape(3, 3), time=float(f.elapsed_time)))

print('| method | preconditioner | evaluations | solves | CG / solve (mech, adjoint) | CG total | final mismatch | C12 |')
print('|---|---|---|---|---|---|---|---|')
for r in runs:
    print(f'| {r["method"]} | {r["prec"]} | {len(r["per_evaluation"])} | {len(r["mech"]) + len(r["adj"])} | '
          f'{np.mean(np.r_[r["mech"], r["adj"]]):.0f} ({r["mech"].mean():.0f}, {r["adj"].mean():.0f}) | '
          f'{int(r["mech"].sum() + r["adj"].sum()):,} | {r["mismatch"]:.3g} | {r["C_voigt"][0, 1]:.3f} |')

fig = plt.figure(figsize=(max(15, 3.4 * len(runs)), 8.8), dpi=150, facecolor='white')
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.0])
gs_designs = gs[1, :].subgridspec(1, len(runs))
ax_solve = fig.add_subplot(gs[0, 0:2])
ax_cum = fig.add_subplot(gs[0, 2])
ax_bar = fig.add_subplot(gs[0, 3])
for r in runs:
    ev = np.arange(1, len(r['per_evaluation']) + 1)
    ax_solve.semilogy(ev, r['per_evaluation'] / 6, color=r['color'], ls=r['style'], lw=1.5, label=r['label'])
    ax_cum.semilogy(ev, np.cumsum(r['per_evaluation']), color=r['color'], ls=r['style'], lw=2, label=r['label'])
ax_solve.set(xlabel='objective evaluation (3 forward + 3 adjoint solves)', ylabel='CG iterations per solve (mean)',
             title='CG iterations per solve')
ax_cum.set(xlabel='objective evaluation', ylabel='cumulative CG iterations', title='CG iterations in total')
ax_solve.legend(frameon=False, fontsize=9)
for ax in (ax_solve, ax_cum):
    ax.grid(True, which='major', alpha=0.3)

x = np.arange(len(runs))
totals = [r['mech'].sum() + r['adj'].sum() for r in runs]
bars = ax_bar.bar(x, totals, color=[r['color'] for r in runs], edgecolor=[r['color'] for r in runs],
                  hatch=None)
for bar, r in zip(bars, runs):
    if r['method'].startswith('level set'):
        bar.set_facecolor('white')
        bar.set_hatch({'level set': '///', 'level set, PF start': '...'}.get(r['method'], 'xx'))
ax_bar.set_yscale('log')
ax_bar.bar_label(bars, labels=[f'{int(t):,}' for t in totals], fontsize=8)
ax_bar.set_xticks(x, [f'{ {"phase field": "PF", "level set": "LS", "level set, PF start": "LS, PF start"}.get(r["method"], "LS from PF") }\n{r["prec"].replace("_", "-")}'
                      for r in runs], fontsize=8, rotation=35, ha="right", rotation_mode="anchor")
ax_bar.set(ylabel='CG iterations, whole optimization', title='total')
ax_bar.margins(y=0.15)

for i, r in enumerate(runs):
    ax = fig.add_subplot(gs_designs[0, i])
    ax.imshow(np.tile(r['design'], (2, 2)).T, origin='lower', cmap='Greys', vmin=0, vmax=1,
              interpolation='nearest', extent=[0, 2, 0, 2])
    ax.set_title(f'{r["label"]}\n{len(r["per_evaluation"])} evaluations, mismatch = {r["mismatch"]:.3g}, '
                 f'C12 = {r["C_voigt"][0, 1]:.3f}', fontsize=8.5)
    ax.set_xticks([]), ax.set_yticks([])

fig.tight_layout()
out = os.path.join(HERE, 'figures', 'TO_level_set_vs_phase_field_N64.png')
fig.savefig(out)
print(f'figure: {out}')
