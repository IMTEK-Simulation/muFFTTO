"""
3D Cahn-Hilliard (spinodal) structures as initial designs for example_3D_elasticity_TO.py (INIT_FROM), the 3D version
of make_TO_CH_starts.py: no imposed symmetry, unlike the default start (sin 4 pi x + sin 4 pi y + sin 4 pi z + 3) / 6,
which repeats every half cell and biased the designs (2D test: all Cahn-Hilliard starts reached a 25-30 % lower
objective than the sine start).

    dc/dt = Laplace( c^3 - c - eps^2 Laplace c ),   periodic unit cube, semi-implicit spectral time stepping

Bicontinuous (mean concentration 0), rho_0 = clip((c + 1) / 2, 0, 1); generated on 64^3, INIT_FROM interpolates it
onto any grid. Written as data/TO3D_CH_starts/CH3D_start_bi_s<seed>.nc and figures/TO3D_CH_starts.png (mid-plane
slices).
usage:  python make_TO3D_CH_starts.py
"""
import os

import matplotlib
import netCDF4
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
OUT = os.path.join(HERE, 'data', 'TO3D_CH_starts')
os.makedirs(OUT, exist_ok=True)
N = 64
EPS = 0.03
DT, STEPS = 2.5e-5 * (EPS / 0.02) ** 2, 3000
SEEDS = (0, 1)

k = 2 * np.pi * np.fft.fftfreq(N, 1.0 / N)
k2 = k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2
denom = 1.0 + DT * EPS ** 2 * k2 ** 2
fig, axs = plt.subplots(len(SEEDS), 3, figsize=(11, 3.8 * len(SEEDS)))
for row, seed in enumerate(SEEDS):
    rng = np.random.default_rng(seed)
    c = 0.05 * rng.standard_normal((N, N, N))
    c_hat = np.fft.fftn(c)
    for _ in range(STEPS):
        c_hat = (c_hat - DT * k2 * np.fft.fftn(c ** 3 - c)) / denom
        c = np.real(np.fft.ifftn(c_hat))
    rho = np.clip((c + 1) / 2, 0, 1)
    path = os.path.join(OUT, f'CH3D_start_bi_s{seed}.nc')
    with netCDF4.Dataset(path, 'w', format='NETCDF3_64BIT_DATA') as d:
        d.createDimension('frame', None)
        d.createDimension('tensor_dim__phase_field_solution-0', 1)
        for a in 'xyz':
            d.createDimension('n' + a, N)
        v = d.createVariable('phase_field_solution', 'f8',
                             ('frame', 'tensor_dim__phase_field_solution-0', 'nx', 'ny', 'nz'))
        v[0, 0] = rho
        d.setncattr('muFFTTO_layout_phase_field_solution', np.array([1, 1], dtype=np.int32))   # io_utils.read_file
        d.description = f'3D Cahn-Hilliard start, eps {EPS}, seed {seed}'
    for c_, (nm, sl) in enumerate([('x', rho[N // 2]), ('y', rho[:, N // 2]), ('z', rho[:, :, N // 2])]):
        ax = axs[row, c_]
        ax.imshow(sl.T, origin='lower', cmap='Greys', vmin=0, vmax=1, interpolation='nearest', extent=(0, 1, 0, 1))
        ax.set_title(f'seed {seed}: slice {nm} = 1/2', fontsize=13)
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    print(f'seed {seed}: stiff fraction {np.mean(rho > 0.5):.2f}', path)
fig.suptitle(f'3D Cahn-Hilliard initial designs ({N}$^3$, black = stiff)', fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'figures', 'TO3D_CH_starts.png'), dpi=100)
