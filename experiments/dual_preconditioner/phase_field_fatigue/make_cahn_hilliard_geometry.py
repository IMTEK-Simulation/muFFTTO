"""
Two-phase periodic microstructure by spinodal decomposition (Cahn-Hilliard), resembling the synthetic microstructure
of Schneider & Kaestner (2025), Fig. 12 (from Seibert et al. 2024): the stiff phase 1 (matrix) with a connected / blobby
soft phase 2.

    dc/dt = Laplace( c^3 - c - eps^2 Laplace c ),   periodic unit cell, semi-implicit spectral time stepping
            (linear terms implicit, c^3 explicit), random initial field around the mean concentration c0 (fixed seed)

The phase is c > 0 -> phase 2 (soft), else phase 1; the field is generated once on a fine grid and sampled (nearest
pixel) on the coarser simulation grids, so every grid sees the same geometry.

usage:  python make_cahn_hilliard_geometry.py    writes geometry/cahn_hilliard_512.npy (1 = phase 1, 0 = phase 2) and
                                                 figures/cahn_hilliard_geometry.png
"""
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
N_FINE = 512
EPS = 0.008            # interface width (unit cell = 1): sets the coarsening length
C0 = -0.12             # mean concentration: phase 2 (c > 0) fraction ~ 0.4, near-symmetric -> connected
DT, STEPS = 4e-6, 4000
SEED = 0

k = 2 * np.pi * np.fft.fftfreq(N_FINE, 1.0 / N_FINE)
k2 = k[:, None] ** 2 + k[None, :] ** 2
rng = np.random.default_rng(SEED)
c = C0 + 0.05 * rng.standard_normal((N_FINE, N_FINE))
c_hat = np.fft.fft2(c)
for _ in range(STEPS):
    nonlin_hat = np.fft.fft2(c ** 3 - c)
    c_hat = (c_hat - DT * k2 * nonlin_hat) / (1.0 + DT * EPS ** 2 * k2 ** 2)
    c = np.real(np.fft.ifft2(c_hat))
phase = np.where(c > 0, 0, 1).astype(np.int8)            # 1: phase 1 (stiff matrix), 0: phase 2 (soft)
os.makedirs(os.path.join(HERE, 'geometry'), exist_ok=True)
np.save(os.path.join(HERE, 'geometry', f'cahn_hilliard_{N_FINE}.npy'), phase)
print(f'phase 2 fraction {1 - phase.mean():.3f}')

fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), layout='constrained', dpi=130)
for ax, n in zip(axes, (N_FINE, 128, 64)):
    idx = (np.arange(n) * N_FINE) // n
    ax.imshow(phase[np.ix_(idx, idx)].T, origin='lower', cmap=matplotlib.colors.ListedColormap(['#f2d633', '#14204a']),
              interpolation='nearest', extent=(0, 1, 0, 1))
    ax.set_title(f'{n}$^2$ (dark: phase 1 stiff, yellow: phase 2 soft)', fontsize=12)
    ax.set_xticks([]), ax.set_yticks([])
out = os.path.join(HERE, 'figures', 'cahn_hilliard_geometry.png')
fig.savefig(out)
print(f'wrote {out}')
