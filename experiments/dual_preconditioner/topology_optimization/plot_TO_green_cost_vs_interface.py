"""Why Green likes sharp interfaces: mean CG iterations per solve against the width of the interface band of
intermediate stiffness, Green vs Green-Jacobi, 64^2. Data: the Hamilton-Jacobi level set with the eps continuation
2 -> 1 -> 0.5 -> 0 (per stage; sharp = nodal [phi > 0], stiffness 1/3, 2/3 at the quadrature points along the
interface), the reaction-diffusion level set (sharp at the quadrature points: exactly two stiffness values), and the
phase field (diffuse interface, eta = 2 pixels)
-> figures/TO_green_cost_vs_interface_N64.png

    python plot_TO_green_cost_vs_interface.py
"""
import os

import matplotlib
import netCDF4
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
DATA = os.path.join(HERE, 'data')
PF = os.path.join(DATA, 'example_2D_elasticity_TO', 'N64_soft1e-05_energy-warm_{}_eta_0.03125_w_5.0_l_bfgs')
LS = os.path.join(DATA, 'example_2D_elasticity_TO_level_set',
                  'N64_soft1e-05_energy_{}_w_5.0_p1_eps2-1-0.5-0_alpha2_holes4_noreinit')
RD = os.path.join(DATA, 'example_2D_elasticity_TO_reaction_diffusion', 'N64_soft1e-05_energy_{}_w_5.0_quad_{}')
RD_RUNS = {'Green': ['L0.5_c0.2_all_initphase', 'L1_c0.2_all_initphase', 'L1_c0.05_all_initphase'],
           'Green_Jacobi': ['L1_c0.05_all_initphase']}
PRECS = [('Green', 'Green (C_ref = solid)', '#2a78d6'), ('Green_Jacobi', 'Green-Jacobi (2μI)', '#eb6834')]
CATEGORIES = ['phase field\n(diffuse, η = 2 px)', 'level set\nε = 2 px', 'ε = 1 px', 'ε = 0.5 px',
              'sharp at nodes\n(stiffness 0, ⅓, ⅔, 1)', 'sharp at quad. points\n(stiffness 0 or 1)']


def cg_per_evaluation(final, drop_last):
    cg = (np.asarray(final.num_iteration_mech, dtype=float) +
          np.asarray(final.num_iteration_adjoint, dtype=float)).reshape(-1, 3).sum(1) / 6
    return cg[:-1] if drop_last else cg


costs = {}  # (prec, category index) -> list of CG per solve of every evaluation
for prec, _, _ in PRECS:
    pf = netCDF4.Dataset(PF.format(prec) + '_final.nc')
    costs[prec, 0] = list(cg_per_evaluation(pf, False))
    final = netCDF4.Dataset(LS.format(prec) + '_final.nc')
    history = netCDF4.Dataset(LS.format(prec) + '_history.nc')
    cg = cg_per_evaluation(final, True)
    frame_eps = np.asarray(history['eps'][:])
    frame_evaluations = np.asarray(history['nb_evaluations'][:]).astype(int)
    for evaluation, value in enumerate(cg, start=1):  # stage of an evaluation: that of the next written frame
        frame = min(np.searchsorted(frame_evaluations, evaluation), len(frame_eps) - 1)
        category = {2.0: 1, 1.0: 2, 0.5: 3, 0.0: 4}[float(frame_eps[frame])]
        costs.setdefault((prec, category), []).append(value)
    for run in RD_RUNS[prec]:
        path = RD.format(prec, run) + '_final.nc'
        if os.path.exists(path):
            costs.setdefault((prec, 5), []).extend(cg_per_evaluation(netCDF4.Dataset(path), True))

fig, ax = plt.subplots(figsize=(11, 5.2), dpi=150)
width = 0.38
for k, (prec, label, color) in enumerate(PRECS):
    x = np.arange(len(CATEGORIES)) + (k - 0.5) * width
    means = [np.mean(costs.get((prec, c), [np.nan])) for c in range(len(CATEGORIES))]
    bars = ax.bar(x, means, width, color=color, label=label)
    ax.bar_label(bars, labels=[f'{m:.0f}' if np.isfinite(m) else 'n/a' for m in means], fontsize=8)
ax.set_yscale('log')
ax.set_xticks(np.arange(len(CATEGORIES)), CATEGORIES, fontsize=8.5)
ax.set_ylabel('CG iterations per solve (mean over the run / stage)')
ax.set_title('Interface band of intermediate stiffness vs preconditioner, 64², contrast 10⁵\n'
             '(each bar: mean over all solves of that run or stage)', fontsize=11)
ax.grid(True, axis='y', alpha=0.3)
ax.legend()
ax.margins(y=0.15)
figure_name = os.path.join(HERE, 'figures', 'TO_green_cost_vs_interface_N64.png')
fig.savefig(figure_name, bbox_inches='tight')
print(f'figure: {figure_name}')
for key, values in sorted(costs.items()):
    print(key, len(values), f'{np.mean(values):.0f}')
