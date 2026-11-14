"""
Why the Helmholtz preconditioner for the TO design update (64^2, eta = 4h, production settings).

1. Curvature per wavelength: the objective's Hessian H (central differences of the adjoint gradient, step 1e-3) applied
   to plane waves v_k = cos(2 pi k x) at the converged design (DESIGN_OPT=lbfgs_helm run), k = 0 ... N/2:
       plain L-BFGS sees      R(k)   = v^T H v / v^T v                 (H_0 = gamma I)
       Helmholtz L-BFGS sees  R_M(k) = v^T H v / v^T M^-1 v,  M^-1 = I + eta^2 (-Lap_h)
   The spread max R / min R over k is what a quasi-Newton method has to learn from its 20 (s, y) pairs.
2. The search direction at the Cahn-Hilliard start: raw gradient vs Helmholtz-smoothed gradient.
numpy.fft only here (offline analysis); the optimizer applies M with the muFFTTO FFT.

    python why_helmholtz.py
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
N = 64
os.environ.update({'NPIX': str(N), 'ETA': str(4 / N), 'SOFT_EXP': '4', 'PF_NORMALIZED': '1', 'PF_WEIGHT': '0.2',
                   'CG_MODE': 'energy', 'CG_DELAY': '0', 'SHOW_PLOTS': '0', 'JAX_PLATFORMS': 'cpu'})
sys.argv = [sys.argv[0], 'Green_Jacobi']
sys.path.insert(0, str(HERE))

import numpy as np
import netCDF4
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

import example_2D_elasticity_TO as TO          # module-level setup only (objective, discretization), no optimization

eta, h = 4 / N, 1 / N
DATA = HERE / 'data' / 'example_2D_elasticity_TO'
design = np.asarray(netCDF4.Dataset(DATA / f'N{N}_soft0.0001_energy-warm_Green_Jacobi_eta_{eta}_pfw0.2_l_bfgs'
                                           f'_dopt_lbfgs_helm_final.nc')['phase_field_solution'][:]).reshape(-1)
start = np.asarray(netCDF4.Dataset(HERE / 'data' / 'CH_start_bi_s0.nc')['phase_field_solution'][:]).squeeze()
start = start.reshape(N, 256 // N, N, 256 // N).mean(axis=(1, 3)).reshape(-1)   # 256^2 start -> 64^2 (as INIT_FROM)


def gradient(x):
    return np.asarray(TO.objective_function_multiple_load_cases(x)[1]).ravel()


def neg_lap_symbol(kx, ky):
    return 4 / h ** 2 * (np.sin(np.pi * kx / N) ** 2 + np.sin(np.pi * ky / N) ** 2)


# ---------------------------------------------------------------- 1. curvature per wavenumber
xg = np.arange(N) / N
eps = 1e-3
ks = np.arange(0, N // 2 + 1)
R, RM = [], []
gradient(design)                                               # warm the CG start fields
for k in ks:
    v = np.repeat(np.cos(2 * np.pi * k * xg)[:, None], N, axis=1).reshape(-1)
    Hv = (gradient(design + eps * v) - gradient(design - eps * v)) / (2 * eps)
    vHv = float(v @ Hv)
    R.append(vHv / float(v @ v))
    RM.append(vHv / float(v @ v) / (1 + eta ** 2 * neg_lap_symbol(k, 0)))
    print(f'k {k:3d}  R {R[-1]: .4e}  R_M {RM[-1]: .4e}', flush=True)
R, RM = np.array(R), np.array(RM)

# ---------------------------------------------------------------- 2. search directions at the start
g = gradient(start).reshape(N, N)
kx = np.fft.fftfreq(N, 1 / N)
KX, KY = np.meshgrid(kx, kx, indexing='ij')
Mg = np.real(np.fft.ifft2(np.fft.fft2(g) / (1 + eta ** 2 * neg_lap_symbol(KX, KY))))

np.savez(HERE / 'exp_data' / f'why_helmholtz_N{N}.npz', ks=ks, R=R, RM=RM, g=g, Mg=Mg, start=start.reshape(N, N))

plt.rcParams.update({'font.size': 13})
fig = plt.figure(figsize=(17, 5.4))
ax1 = fig.add_subplot(1, 3, 1)
pos = R > 0
spread = R[pos].max() / R[pos].min()
spread_M = RM[pos].max() / RM[pos].min()
ax1.semilogy(ks[pos], R[pos] / R[pos].max(), color='#c2a5cf', lw=2.5, marker='o', ms=6,
             label=f'plain L-BFGS sees  (spread {spread:,.0f}×)')
ax1.semilogy(ks[pos], RM[pos] / RM[pos].max(), color='#762a83', lw=2.5, marker='s', ms=6,
             label=f'Helmholtz L-BFGS sees  (spread {spread_M:,.0f}×)')
ax1.set_xlabel('wavenumber k of the design change')
ax1.set_ylabel('curvature  vᵀHv / vᵀv  (normalised)')
ax1.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
ax1.yaxis.set_minor_locator(NullLocator())
ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
ax1.set_title('objective curvature per wavelength')
ax1.legend(fontsize=11, loc='lower right')
ax1.grid(alpha=0.3)
for i, (field, title) in enumerate([(start.reshape(N, N), 'Cahn-Hilliard start ρ₀'),
                                    (g, 'raw gradient (plain L-BFGS step 1)'),
                                    (Mg, 'Helmholtz-smoothed gradient')]):
    if i == 0:
        continue
    ax = fig.add_subplot(1, 3, i + 1)
    lim = np.abs(field).max()
    im = ax.imshow(field.T, origin='lower', cmap='RdBu_r', vmin=-lim, vmax=lim, interpolation='nearest')
    ax.contour(start.reshape(N, N).T, levels=[0.5], colors='k', linewidths=0.8)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
out = HERE / 'figures' / f'why_helmholtz_N{N}.png'
fig.savefig(out, dpi=150)
print(out, f'spread plain {spread:.0f}, Helmholtz {spread_M:.1f}')
