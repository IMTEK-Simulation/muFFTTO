"""
Effect of the objective weights on the hinges of the 2D TO designs (Cahn-Hilliard starts bi_s0, bi_s3):
  128^2 (eta 1/32), from the Cahn-Hilliard starts: perimeter weight w = PF_WEIGHT 0.2 / 0.5 / 1, stress weight a = 1
  256^2 (eta 1/64), from the 256^2 optima (a = 1): stress weight a = STRESS_WEIGHT 1 / 5 / 25, w = 0.2
Larger w (or smaller a) penalises interfaces more: do the thin stiff necks between the slit tips (hinges) disappear?
Hinge measure: soft pieces (rho < 0.5, periodic) and the smallest stiff gap between two of them, in units of eta
(a gap of ~1 eta or less is a hinge thinner than the interface, not a resolved ligament).
Figure figures/TO_weights_hinges.png (designs as 2 x 2 periodic copies); table printed.
usage:  python plot_TO_weights.py
"""
import glob
import os

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from scipy import ndimage

from elastic_properties import properties

HERE = os.path.dirname(os.path.realpath(__file__))
DATA = os.path.join(HERE, 'data', 'example_2D_elasticity_TO')
plt.rcParams.update({'font.size': 13})
STARTS = ['bi_s0', 'bi_s3']
CASES = [  # (title, N, eta, w, a, file suffix after l_bfgs)
    ('128$^2$, w 0.2', 128, '0.03125', 0.2, 1, '_start_{L}'),
    ('128$^2$, w 0.5', 128, '0.03125', 0.5, 1, '_start_{L}'),
    ('128$^2$, w 1', 128, '0.03125', 1.0, 1, '_start_{L}'),
    ('256$^2$, a 1', 256, '0.015625', 0.2, 1, '_start_{L}_from128'),
    ('256$^2$, a 5', 256, '0.015625', 0.2, 5, '_start_{L}_from256'),
    ('256$^2$, a 25', 256, '0.015625', 0.2, 25, '_start_{L}_from256'),
]


def final(N, eta, w, a, sfx):
    tag = f'pfw{w:g}' + (f'_sw{a:g}' if a != 1 else '')
    hits = glob.glob(os.path.join(DATA, f'N{N}_*Green_Jacobi_eta_{eta}_{tag}_l_bfgs{sfx}_final.nc'))
    return hits[0] if hits else None


def hinges(rho):
    N = rho.shape[0]
    lab, _ = ndimage.label(np.tile(rho < 0.5, (3, 3)))
    ids = [i for i in np.unique(lab[N:2 * N, N:2 * N]) if i > 0]
    gaps = []
    for i in ids:
        dist = ndimage.distance_transform_edt(lab != i)
        other = (lab > 0) & (lab != i)
        if other.any():
            gaps.append(dist[other].min() / N)
    return len(ids), min(gaps) if gaps else np.nan


fig, axs = plt.subplots(len(STARTS), len(CASES), figsize=(3.4 * len(CASES), 3.9 * len(STARTS) + 0.8))
print(f'{"start":>6} {"case":>14} {"its":>5} {"P":>5} {"grey":>5}  C11    C22    C12    C33    err   nu12    mu   A_Z  pieces  gap/eta')
for r, L in enumerate(STARTS):
    for c, (title, N, eta, w, a, sfx) in enumerate(CASES):
        ax = axs[r, c]
        ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
        f = final(N, eta, w, a, sfx.format(L=L))
        if f is None:
            ax.set_title(f'{L}, {title}\nnot available', fontsize=12)
            continue
        with netCDF4.Dataset(f) as d:
            rho = np.array(d['phase_field_solution'][0, 0])
            C = np.array(d.getncattr('homogenized_C_voigt')).reshape(3, 3)
            its = int(d.getncattr('nb_iterations'))
            pf = np.atleast_1d(d.getncattr('norms_pf'))[-1]
        Ct = np.array([[0.28929, -0.09643, 0], [-0.09643, 0.28929, 0], [0, 0, 0.19286]])
        err = np.linalg.norm(C - Ct) / np.linalg.norm(Ct)
        pieces, gap = hinges(rho)
        pr = properties(C)
        P = pf / w
        grey = np.mean((rho > 0.1) & (rho < 0.9))
        print(f'{L:>6} {title.replace("$^2$", "^2"):>14} {its:5d} {P:5.2f} {grey:5.2f}  {C[0, 0]:.3f}  {C[1, 1]:.3f} '
              f'{C[0, 1]:+.3f}  {C[2, 2]:.3f}  {err:.3f}  {pr["nu12"]:+.3f} {pr["mu"]:.3f} {pr["A_Z"]:.2f}  {pieces:5d}  {gap / float(eta):7.1f}')
        ax.imshow(np.tile(rho, (2, 2)).T, origin='lower', cmap='Greys', vmin=0, vmax=1, interpolation='nearest',
                  extent=(0, 2, 0, 2))
        ax.set_title(f'{L}, {title}\n$\\nu_{{12}}$ {pr["nu12"]:+.3f}, err {err:.2f}, gap {gap / float(eta):.1f} $\\eta$',
                     fontsize=12)
fig.suptitle('Objective weights and hinges: perimeter weight w (128$^2$, $\\eta$ 1/32) and stress weight a '
             '(256$^2$, $\\eta$ 1/64); target $\\nu_{12}$ = -1/3; gap = narrowest stiff neck between two slits',
             fontsize=14)
fig.tight_layout()
out = os.path.join(HERE, 'figures', 'TO_weights_hinges.png')
fig.savefig(out, dpi=100)
print('saved', out)


# ---- trends: Poisson ratio, stiffness error, Zener ratio and perimeter across the weight variants ----
fig2, axs2 = plt.subplots(1, 4, figsize=(20, 5.2))
names = [c[0].replace('$^2$', '²') for c in CASES]
for L, mk in zip(STARTS, ('o', 's')):
    vals = {k: [] for k in ('nu12', 'err', 'A_Z', 'P')}
    for title, N, eta, w, a, sfx in CASES:
        f = final(N, eta, w, a, sfx.format(L=L))
        if f is None:
            for k in vals:
                vals[k].append(np.nan)
            continue
        with netCDF4.Dataset(f) as d:
            C = np.array(d.getncattr('homogenized_C_voigt')).reshape(3, 3)
            pf = np.atleast_1d(d.getncattr('norms_pf'))[-1]
        Ct = np.array([[0.28929, -0.09643, 0], [-0.09643, 0.28929, 0], [0, 0, 0.19286]])
        pr = properties(C)
        vals['nu12'].append(pr['nu12']); vals['A_Z'].append(pr['A_Z']); vals['P'].append(pf / w)
        vals['err'].append(np.linalg.norm(C - Ct) / np.linalg.norm(Ct))
    for ax, k in zip(axs2, ('nu12', 'err', 'A_Z', 'P')):
        ax.plot(range(len(CASES)), vals[k], marker=mk, ms=9, lw=2, label=L)
for ax, k, tgt, lab in zip(axs2, ('nu12', 'err', 'A_Z', 'P'), (-1 / 3, 0, 1, None),
                           ('Poisson ratio $\\nu_{12}$', 'stiffness error', 'Zener ratio $A_Z$', 'perimeter P')):
    if tgt is not None:
        ax.axhline(tgt, color='k', ls=':', lw=2, label='target')
    ax.axvline(2.5, color='grey', lw=1)
    ax.set_xticks(range(len(CASES))); ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_title(lab); ax.grid(alpha=0.3); ax.legend(fontsize=11)
fig2.suptitle('2D TO: effect of the objective weights. Left of the grey line: perimeter weight w at 128$^2$ '
              '(stress weight a = 1); right: stress weight a at 256$^2$ (w = 0.2)', fontsize=14)
fig2.tight_layout()
out2 = os.path.join(HERE, 'figures', 'TO_weights_trends.png')
fig2.savefig(out2, dpi=100)
print('saved', out2)
