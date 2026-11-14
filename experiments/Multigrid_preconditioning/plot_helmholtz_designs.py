"""
Final TO designs of the Helmholtz strength sweep (laptop, CH start bi_s0, eta = 4h, Green-Jacobi mechanics):
rows 32^2 / 64^2, columns c = 0 (plain L-BFGS) ... 64.  Exact pixels (nearest), solid = dark.
Per design: final objective, evaluations, solid fraction (rho > 0.5), grey fraction (0.1 < rho < 0.9) and the
overlap with the plain design: IoU of the solid phase maximised over all periodic shifts and the 8 symmetries of
the square (the objective is invariant under them, so a raw IoU mistakes translated copies for different designs).

    python plot_helmholtz_designs.py
"""
import re
from pathlib import Path

import numpy as np
import netCDF4
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data' / 'example_2D_elasticity_TO'
CS = [0, 0.015625, 0.0625, 0.25, 1, 4, 16, 64]
LABELS = ['plain', '1/64', '1/16', '1/4', '1', '4', '16', '64']


def aligned_iou(a, b):
    A, best = a > 0.5, 0.0
    for k in range(4):
        for flip in (False, True):
            B = np.rot90(b > 0.5, k)
            B = B.T if flip else B
            inter = np.real(np.fft.ifft2(np.fft.fft2(A) * np.conj(np.fft.fft2(B)))).max()
            best = max(best, inter / (A.sum() + B.sum() - inter))
    return best


def names(N, c):
    tag = 'dopt_lbfgs' if c == 0 else ('dopt_lbfgs_helm' if c == 1 else f'dopt_lbfgs_helm_c{c:g}')
    nc = DATA / f'N{N}_soft0.0001_energy-warm_Green_Jacobi_eta_{4 / N}_pfw0.2_l_bfgs_{tag}_final.nc'
    log = HERE / 'logs' / f'dopt_N{N}_{tag.replace("dopt_", "")}.log'
    return nc, log


plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(2, len(CS), figsize=(2.3 * len(CS), 5.6))
print(f'{"N":>4s} {"c":>6s} {"evals":>6s} {"f":>9s} {"solid":>6s} {"grey":>6s} {"IoU vs plain":>13s}')
for row, N in enumerate([32, 64]):
    plain = None
    for col, (c, lab) in enumerate(zip(CS, LABELS)):
        ax = axes[row, col]
        ax.set_xticks([]); ax.set_yticks([])
        nc, log = names(N, c)
        if not nc.exists():
            ax.set_title(f'c = {lab}\n(missing)', fontsize=10)
            continue
        rho = np.asarray(netCDF4.Dataset(nc)['phase_field_solution'][:]).reshape(N, N)
        ev = re.findall(r'^EVAL (\d+) f ([\d.eE+-]+)', log.read_text(), flags=re.M)
        n_ev, f = int(ev[-1][0]), float(ev[-1][1])
        solid = rho > 0.5
        if c == 0:
            plain = rho
        iou = aligned_iou(plain, rho)
        grey = ((rho > 0.1) & (rho < 0.9)).mean()
        print(f'{N:4d} {lab:>6s} {n_ev:6d} {f:9.5f} {solid.mean():6.3f} {grey:6.3f} {iou:13.3f}')
        ax.imshow(rho.T, origin='lower', cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(f'c = {lab}\nf {f:.4f}, {n_ev} ev', fontsize=10)
        ax.set_xlabel(f'IoU {iou:.2f}', fontsize=10)
    axes[row, 0].set_ylabel(f'{N}²', fontsize=14)
fig.suptitle('Final designs vs Helmholtz strength c (ℓ = √c η, η = 4h); dark = solid; '
             'IoU = overlap with the plain L-BFGS design (best periodic shift / symmetry)', fontsize=12)
fig.tight_layout()
out = HERE / 'figures' / 'helmholtz_designs.png'
fig.savefig(out, dpi=200)
print(out)
