"""
Helmholtz-preconditioned L-BFGS at different interface widths eta = 1h, 2h, 4h (laptop, CH start, Green-Jacobi
mechanics, production settings): plain L-BFGS vs Helmholtz c = 4 / 16 / 64 (smoothing length l = sqrt(c) eta).
4h from run_helm_c_test.sh / run_design_opt_test.sh, 2h and 1h from run_helm_eta_test.sh.
Table: evaluations to the stop, final objective, grey fraction, aligned IoU of the solid with the plain design.
Figure: final designs (exact pixels, dark = solid), rows (eta, N), columns plain / c = 4 / 16 / 64.

    python plot_helmholtz_eta.py
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
COLS = [0, 4, 16, 64]


def files(N, E, c):
    if E == 4:
        tag = 'dopt_lbfgs' if c == 0 else ('dopt_lbfgs_helm' if c == 1 else f'dopt_lbfgs_helm_c{c:g}')
        log = HERE / 'logs' / f'dopt_N{N}_{tag[5:]}.log'
    else:
        tag = f'dopt_eta{E}h_' + ('plain' if c == 0 else f'helm_c{c:g}')
        log = HERE / 'logs' / f'dopt_eta{E}h_N{N}_{tag.split("_", 2)[2]}.log'
    return DATA / f'N{N}_soft0.0001_energy-warm_Green_Jacobi_eta_{E / N}_pfw0.2_l_bfgs_{tag}_final.nc', log


def aligned_iou(a, b):
    A, best = a > 0.5, 0.0
    for k in range(4):
        for flip in (False, True):
            B = np.rot90(b > 0.5, k)
            B = B.T if flip else B
            inter = np.real(np.fft.ifft2(np.fft.fft2(A) * np.conj(np.fft.fft2(B)))).max()
            best = max(best, inter / (A.sum() + B.sum() - inter))
    return best


ROWS = [(E, N) for E in (4, 2, 1) for N in (32, 64)]
plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(len(ROWS), len(COLS), figsize=(2.6 * len(COLS), 2.7 * len(ROWS)))
print(f'{"eta":>4s}{"N":>4s} {"variant":>8s} {"evals":>6s} {"f":>9s} {"grey":>6s} {"IoU plain":>10s} {"evals saved":>12s}')
for r, (E, N) in enumerate(ROWS):
    plain_rho, plain_ev = None, None
    for c_i, c in enumerate(COLS):
        ax = axes[r, c_i]
        ax.set_xticks([]); ax.set_yticks([])
        nc, log = files(N, E, c)
        if not (nc.exists() and log.exists()):
            ax.axis('off')
            continue
        rho = np.asarray(netCDF4.Dataset(nc)['phase_field_solution'][:]).reshape(N, N)
        ev = re.findall(r'^EVAL (\d+) f ([\d.eE+-]+)', log.read_text(), flags=re.M)
        n_ev, f = int(ev[-1][0]), float(ev[-1][1])
        if c == 0:
            plain_rho, plain_ev = rho, n_ev
        iou = aligned_iou(plain_rho, rho) if plain_rho is not None else np.nan
        grey = ((rho > 0.1) & (rho < 0.9)).mean()
        lab = 'plain' if c == 0 else f'c={c}'
        print(f'{E:3d}h{N:4d} {lab:>8s} {n_ev:6d} {f:9.5f} {grey:6.3f} {iou:10.3f} '
              f'{(plain_ev / n_ev if plain_ev else np.nan):11.2f}x')
        ax.imshow(rho.T, origin='lower', cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
        ell = '' if c == 0 else f', ℓ={np.sqrt(c) * E:g}h'
        ax.set_title(f'{lab}{ell}\nf {f:.4f}, {n_ev} ev', fontsize=10)
        ax.set_xlabel(f'IoU {iou:.2f}, grey {grey:.2f}', fontsize=9)
    axes[r, 0].set_ylabel(f'η = {E}h, {N}²', fontsize=12)
fig.suptitle('Helmholtz L-BFGS vs interface width: final designs (dark = solid; IoU vs plain, best shift/symmetry)',
             fontsize=12)
fig.tight_layout()
out = HERE / 'figures' / 'helmholtz_eta.png'
fig.savefig(out, dpi=170)
print(out)
