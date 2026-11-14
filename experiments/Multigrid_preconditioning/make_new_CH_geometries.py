"""
New random Cahn-Hilliard (spinodal) structures for the overnight runs of 2026-11-13 (user: "new / random
initialization of Cahn-Hilliard geometries so we have interestingly new structures").

    dc/dt = Laplace( c^3 - c - eps^2 Laplace c ),  periodic, semi-implicit spectral stepping
    (as experiments/dual_preconditioner/topology_optimization/make_TO_CH_starts.py (2D) and
     damage_AT2_small_strain/schneider2025_notch/make_cahn_hilliard_geometry_3D.py (3D), new seeds)

2D TO starts, 256^2, eps 0.025: data/CH_starts_new/CH_start_<label>.nc (phase_field_solution, read by INIT_FROM),
    bi_s11, bi_s12 (mean c 0, bicontinuous, rho = (c+1)/2), drop_s11 (mean c -0.3, soft droplets, rho = (1-c)/2)
3D geometries, 128^3, eps 0.02: data/CH3D/ch3d_<label>.npy (int8, 1 = stiff, 0 = soft),
    bi_s11 (mean c 0), ch_s12 (mean c -0.12, as the AT2 geometry), ch_s13 (mean c -0.12)
Figures: figures/CH_starts_new_2D.png, figures/CH3D_new.png.  Runs on the GPU (cupy) if available.

    python make_new_CH_geometries.py
"""
import os
from pathlib import Path

import numpy as np
import netCDF4
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import cupy as xp
    xp.zeros(1)
except Exception:                    # no GPU: numpy
    xp = np
to_np = (lambda a: a.get()) if xp is not np else (lambda a: a)
HERE = Path(__file__).resolve().parent


def cahn_hilliard(N, dim, eps, c0, seed, dt, steps):
    k = 2 * np.pi * np.fft.fftfreq(N, 1.0 / N)
    grids = np.meshgrid(*([k] * dim), indexing='ij')
    k2 = xp.asarray(sum(g ** 2 for g in grids))
    rng = np.random.default_rng(seed)
    c = xp.asarray(c0 + 0.05 * rng.standard_normal((N,) * dim))
    c_hat = xp.fft.fftn(c)
    denom = 1.0 + dt * eps ** 2 * k2 ** 2
    for _ in range(steps):
        c_hat = (c_hat - dt * k2 * xp.fft.fftn(c ** 3 - c)) / denom
        c = xp.real(xp.fft.ifftn(c_hat))
    return to_np(c)


# ---------------------------------------------------------------- 2D TO starts
out2 = HERE / 'data' / 'CH_starts_new'
out2.mkdir(parents=True, exist_ok=True)
N2, EPS2 = 256, 0.025
DT2, STEPS2 = 4e-6 * (EPS2 / 0.008) ** 2, 3000
starts = [('bi_s11', 0.0, 11), ('bi_s12', 0.0, 12), ('drop_s11', -0.3, 11)]
fig, axs = plt.subplots(1, len(starts), figsize=(4 * len(starts), 4.4))
for ax, (label, c0, seed) in zip(axs, starts):
    c = cahn_hilliard(N2, 2, EPS2, c0, seed, DT2, STEPS2)
    rho = np.clip((c + 1) / 2 if c0 == 0 else (1 - c) / 2, 0, 1)
    path = out2 / f'CH_start_{label}.nc'
    with netCDF4.Dataset(path, 'w', format='NETCDF3_64BIT_DATA') as d:
        d.createDimension('frame', None)
        d.createDimension('tensor_dim__phase_field_solution-0', 1)
        d.createDimension('nx', N2)
        d.createDimension('ny', N2)
        v = d.createVariable('phase_field_solution', 'f8', ('frame', 'tensor_dim__phase_field_solution-0', 'nx', 'ny'))
        v[0, 0] = rho
        d.description = f'Cahn-Hilliard start, eps {EPS2}, mean c {c0}, seed {seed}'
        d.setncattr('muFFTTO_layout_phase_field_solution', np.array([1, 1], dtype=np.int32))
    ax.imshow(rho.T, origin='lower', cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
    ax.set_title(f'{label}: stiff {100 * np.mean(rho > 0.5):.0f} %')
    ax.set_xticks([]); ax.set_yticks([])
    print(f'2D {label}: stiff fraction {np.mean(rho > 0.5):.2f} -> {path}', flush=True)
fig.suptitle('new Cahn-Hilliard TO starts (black = stiff)')
fig.tight_layout()
(HERE / 'figures').mkdir(exist_ok=True)
fig.savefig(HERE / 'figures' / 'CH_starts_new_2D.png', dpi=120)

# ---------------------------------------------------------------- 3D geometries
out3 = HERE / 'data' / 'CH3D'
out3.mkdir(parents=True, exist_ok=True)
N3, EPS3 = 128, 0.02
DT3, STEPS3 = 2.5e-5 * (EPS3 / 0.02) ** 2, 4000
geoms = [('bi_s11', 0.0, 11), ('ch_s12', -0.12, 12), ('ch_s13', -0.12, 13)]
fig, axs = plt.subplots(2, len(geoms), figsize=(4 * len(geoms), 8.2))
for col, (label, c0, seed) in enumerate(geoms):
    c = cahn_hilliard(N3, 3, EPS3, c0, seed, DT3, STEPS3)
    phase = np.where(c > 0, 0, 1).astype(np.int8)            # 1: stiff matrix, 0: soft phase (c > 0)
    np.save(out3 / f'ch3d_{label}.npy', phase)
    for row, (sl, name) in enumerate([(phase[:, :, N3 // 2], 'z = 1/2'), (phase[N3 // 2], 'x = 1/2')]):
        axs[row, col].imshow(sl.T, origin='lower', cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
        axs[row, col].set_title(f'{label} ({name}): stiff {100 * phase.mean():.0f} %')
        axs[row, col].set_xticks([]); axs[row, col].set_yticks([])
    print(f'3D {label}: stiff fraction {phase.mean():.2f} -> {out3 / f"ch3d_{label}.npy"}', flush=True)
fig.suptitle('new 3D Cahn-Hilliard geometries, 128^3 (black = stiff)')
fig.tight_layout()
fig.savefig(HERE / 'figures' / 'CH3D_new.png', dpi=110)
