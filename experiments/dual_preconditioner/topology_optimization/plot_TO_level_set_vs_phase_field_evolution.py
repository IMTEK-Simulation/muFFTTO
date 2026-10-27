"""Design evolution, level set vs phase field (the runs of plot_TO_level_set_vs_phase_field.py): one row per run,
columns at the same numbers of objective evaluations; each snapshot labelled with the mean CG iterations per solve of
that evaluation -> figures/TO_level_set_vs_phase_field_evolution_N64.png

    python plot_TO_level_set_vs_phase_field_evolution.py
    STIFFNESS=1 python ...  -> the stiffness the CG solves see instead of the density: log10 of
                               s = rho^p (1 - soft) + soft, C(x) = s C_0 (..._evolution_stiffness_N64.png)
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

SNAPSHOT_EVALUATIONS = [2, 5, 10, 20, 40, 80]  # + the final design
STIFFNESS = os.environ.get('STIFFNESS', '0') == '1'

rows = []
for method, template, _ in METHODS:
    for prec, label, color in PRECS:
        if not os.path.exists(template.format(prec) + '_final.nc'):  # unfinished run
            continue
        f = netCDF4.Dataset(template.format(prec) + '_final.nc')
        h = netCDF4.Dataset(template.format(prec) + '_history.nc')
        cg_per_solve = (np.asarray(f.num_iteration_mech, dtype=float) +
                        np.asarray(f.num_iteration_adjoint, dtype=float)).reshape(-1, 3).sum(1) / 6
        designs = np.asarray(h['phase_field_iterate'][:])[:, 0]
        if STIFFNESS:
            soft, p = float(f.soft_phase), float(f.p)
            designs = np.log10(np.clip(designs, 0, 1) ** p * (1 - soft) + soft)
        rows.append(dict(label=f'{method}\n{label}', color=color, cg_per_solve=cg_per_solve, designs=designs,
                         evaluations=np.asarray(h['nb_evaluations'][:]).astype(int)))

nb_cols = len(SNAPSHOT_EVALUATIONS) + 1
fig, axes = plt.subplots(len(rows), nb_cols, figsize=(2.1 * nb_cols + 0.8, 2.35 * len(rows)), dpi=150,
                         squeeze=False)
for row, r in zip(axes, rows):
    for col, ax in enumerate(row):
        # frame: the last accepted iterate reached within the evaluation budget
        if col < len(SNAPSHOT_EVALUATIONS):
            frame = max(np.searchsorted(r['evaluations'], SNAPSHOT_EVALUATIONS[col], side='right') - 1, 0)
        else:
            frame = len(r['evaluations']) - 1
        evaluation = r['evaluations'][frame]
        image = ax.imshow(r['designs'][frame].T, origin='lower', cmap='viridis' if STIFFNESS else 'Greys',
                          vmin=-5 if STIFFNESS else 0, vmax=0 if STIFFNESS else 1, interpolation='nearest',
                          extent=[0, 1, 0, 1])
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f'{"final, " if col == nb_cols - 1 else ""}eval. {evaluation}\n'
                     f'CG/solve {r["cg_per_solve"][evaluation - 1]:.0f}', fontsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(r['color'])
            spine.set_linewidth(2)
    row[0].set_ylabel(r['label'], fontsize=9)
fig.tight_layout()
if STIFFNESS:
    fig.colorbar(image, ax=axes, shrink=0.6, pad=0.01, label='log10 stiffness  (C = s C_0)')
out = os.path.join(HERE, 'figures', 'TO_level_set_vs_phase_field_evolution'
                   + ('_stiffness' if STIFFNESS else '') + '_N64.png')
fig.savefig(out)
print(f'figure: {out}')
