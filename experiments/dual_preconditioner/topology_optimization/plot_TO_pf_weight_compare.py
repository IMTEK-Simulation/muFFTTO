"""
Effect of the perimeter weight w (PF_NORMALIZED=1, PF_WEIGHT) on the TO design: w = 0.2 (= the objective of the 2026-11
eta / contrast study) against w = 1 (muTopOpt's default), 128^2, contrast 1e4, Green-Jacobi, eta = 2 h and 4 h.

    top: final designs (black = stiff phase), one column per (eta, w)
    bottom: perimeter P, stiffness error |C - C*| / |C*|, R = 3 F_pf / P, grey fraction, L-BFGS iterations, CG per solve

usage:  python plot_TO_pf_weight_compare.py      (runs of logs_pfw/, launched by hand 2026-11-04, see NOTES_eta_h_choice.md)
        writes figures/TO_pf_weight_compare.png
"""
import glob
import os

import numpy as np
import netCDF4
import contourpy
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
DATA = os.path.join(HERE, 'data', 'example_2D_elasticity_TO')
N, SOFT, PREC = 128, '0.0001', 'Green_Jacobi'
ETAS = [0.03125, 0.015625]
WS = [0.2, 1]


def metrics(phase, eta):
    """P (0.5 contour length, cell side 1), F_pf (raw bracket, P1 forward differences + lumped double well), R, band"""
    n = phase.shape[0]
    h = 1.0 / n
    dx = np.roll(phase, -1, 0) - phase
    dy = np.roll(phase, -1, 1) - phase
    f_pf = eta * np.sum(dx ** 2 + dy ** 2) + h * h * np.sum(phase ** 2 * (1 - phase) ** 2) / eta
    z = np.pad(phase, ((0, 1), (0, 1)), mode='wrap')
    xs = np.linspace(0, 1, n + 1)
    P = sum(np.sum(np.linalg.norm(np.diff(l, axis=0), axis=1))
            for l in contourpy.contour_generator(xs, xs, z.T).lines(0.5))
    return P, 3 * f_pf / P


rows = []
for eta in ETAS:
    for w in WS:
        f = os.path.join(DATA, f'N{N}_soft{SOFT}_energy-warm_{PREC}_eta_{eta}_pfw{w:g}_l_bfgs_final.nc')
        with netCDF4.Dataset(f) as d:
            phase = np.asarray(d['phase_field_solution'][:]).reshape(N, N)
            C, Ct = np.asarray(d.getncattr('homogenized_C_voigt')), np.asarray(d.getncattr('target_C_voigt'))
            its = int(d.getncattr('nb_iterations'))
            cg = np.r_[d.getncattr('num_iteration_mech'), d.getncattr('num_iteration_adjoint')]
        P, R = metrics(phase, eta)
        rows.append(dict(eta=eta, w=w, phase=phase, P=P, R=R, C_err=np.linalg.norm(C - Ct) / np.linalg.norm(Ct),
                         grey=np.mean((phase > 0.1) & (phase < 0.9)), its=its, cg=cg.mean()))

plt.rcParams.update({'font.size': 13})
fig = plt.figure(figsize=(18, 9.5), layout='constrained')
gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1])
for k, r in enumerate(rows):
    ax = fig.add_subplot(gs[0, k])
    ax.imshow(1 - r['phase'].T, cmap='gray', origin='lower', vmin=0, vmax=1, interpolation='nearest',
              extent=(0, 1, 0, 1))
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(f'$\\eta$ = 1/{round(1 / r["eta"])} ({r["eta"] * N:g} h),  w = {r["w"]:g}\n'
                 f'P = {r["P"]:.2f},  stiffness error = {r["C_err"]:.3f}', fontsize=13)

labels = [f'$\\eta$ = {r["eta"] * N:g} h\nw = {r["w"]:g}' for r in rows]
cols = ['C1' if r['w'] == 0.2 else 'C4' for r in rows]
hatch = ['' if r['eta'] == ETAS[0] else '//' for r in rows]
for k, (key, title) in enumerate((('P', 'perimeter P (cell side = 1)'), ('C_err', 'stiffness error |C - C*| / |C*|'),
                                  ('its', 'L-BFGS iterations'), ('cg', 'CG iterations per solve'))):
    ax = fig.add_subplot(gs[1, k])
    vals = [r[key] for r in rows]
    bars = ax.bar(range(len(rows)), vals, color=cols, hatch=hatch, edgecolor='k')
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f'{v:.3g}', ha='center', va='bottom', fontsize=12)
    ax.set_xticks(range(len(rows))), ax.set_xticklabels(labels, fontsize=12)
    ax.set_title(title), ax.set_ylim(0, 1.18 * max(vals)), ax.grid(True, axis='y', color='0.9')
fig.suptitle(f'Perimeter weight w: 0.2 (study objective) vs 1 (muTopOpt default);  {N}$^2$, contrast 1e4, '
             f'Green-Jacobi;  R = 3F$_{{pf}}$/P: ' + ', '.join(f'{r["R"]:.2f}' for r in rows), fontsize=15)
out = os.path.join(HERE, 'figures', 'TO_pf_weight_compare.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=120)
print('wrote', out)
for r in rows:
    print(f"eta={r['eta']:.6g} ({r['eta'] * N:g}h) w={r['w']:g}: P={r['P']:.3f} C_err={r['C_err']:.3f} R={r['R']:.3f} "
          f"grey={r['grey']:.3f} its={r['its']} cg/solve={r['cg']:.0f}")
