"""
Cahn-Hilliard (spinodal) structures as initial designs for the 2D TO (example_2D_elasticity_TO.py, option INIT_FROM):
does the optimizer reach the same design from different starts, or different local optima?

    dc/dt = Laplace( c^3 - c - eps^2 Laplace c ),   periodic unit cell, semi-implicit spectral time stepping

as in damage_AT2_small_strain/schneider2025_notch/make_cahn_hilliard_geometry.py, with a larger EPS (a few domains per
cell, like the TO designs). Start design rho_0 = clip((c + 1) / 2, 0, 1): smooth interfaces, no imposed symmetry.
Mean concentration C0: 0 -> 50/50 bicontinuous (rho = (c+1)/2); -0.3 -> droplets of c > 0 in a c < 0 matrix,
mapped to a stiff matrix with soft droplets (rho = (1-c)/2).
Written as data/TO_CH_starts/CH_start_<label>.nc (variable phase_field_solution, read by INIT_FROM) and a figure
figures/TO_CH_starts.png.
usage:  python make_TO_CH_starts.py
"""
import os

import matplotlib
import netCDF4
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
OUT = os.path.join(HERE, 'data', 'TO_CH_starts')
os.makedirs(OUT, exist_ok=True)
N = 256
EPS = 0.025
DT, STEPS = 4e-6 * (EPS / 0.008) ** 2, 3000
STARTS = [(f'bi_s{s}', 0.0, s) for s in range(4)] + [(f'drop_s{s}', -0.3, s) for s in range(2)]

k = 2 * np.pi * np.fft.fftfreq(N, 1.0 / N)
k2 = k[:, None] ** 2 + k[None, :] ** 2
fig, axs = plt.subplots(1, len(STARTS), figsize=(3.6 * len(STARTS), 4.2))
for ax, (label, c0, seed) in zip(axs, STARTS):
    rng = np.random.default_rng(seed)
    c = c0 + 0.05 * rng.standard_normal((N, N))
    c_hat = np.fft.fft2(c)
    for _ in range(STEPS):
        c_hat = (c_hat - DT * k2 * np.fft.fft2(c ** 3 - c)) / (1.0 + DT * EPS ** 2 * k2 ** 2)
        c = np.real(np.fft.ifft2(c_hat))
    rho = np.clip((c + 1) / 2 if c0 == 0 else (1 - c) / 2, 0, 1)   # droplets (c > 0, minority) = soft
    path = os.path.join(OUT, f'CH_start_{label}.nc')
    with netCDF4.Dataset(path, 'w', format='NETCDF3_64BIT_DATA') as d:
        d.createDimension('frame', None)
        d.createDimension('tensor_dim__phase_field_solution-0', 1)
        d.createDimension('nx', N)
        d.createDimension('ny', N)
        v = d.createVariable('phase_field_solution', 'f8', ('frame', 'tensor_dim__phase_field_solution-0', 'nx', 'ny'))
        v[0, 0] = rho
        d.description = f'Cahn-Hilliard start, eps {EPS}, mean c {c0}, seed {seed}'
        d.setncattr('muFFTTO_layout_phase_field_solution', np.array([1, 1], dtype=np.int32))   # io_utils.read_file
    ax.imshow(rho.T, origin='lower', cmap='Greys', vmin=0, vmax=1, interpolation='nearest', extent=(0, 1, 0, 1))
    ax.set_title(f'{label}\nstiff {100 * np.mean(rho > 0.5):.0f} %', fontsize=13)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    print(label, f'stiff fraction {np.mean(rho > 0.5):.2f}', path)
fig.suptitle('Cahn-Hilliard initial designs for the 2D TO (black = stiff, rho = 1)', fontsize=15)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'figures', 'TO_CH_starts.png'), dpi=100)
