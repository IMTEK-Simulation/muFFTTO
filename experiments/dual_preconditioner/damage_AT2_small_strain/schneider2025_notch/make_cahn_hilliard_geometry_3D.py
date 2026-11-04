"""
3D version of make_cahn_hilliard_geometry.py: periodic two-phase spinodal (Cahn-Hilliard) microstructure on the unit
cube, the stiff phase 1 (matrix) with a connected soft phase 2, for the 3D confirmation of the 2D Cahn-Hilliard
preconditioner scaling (fixed l, grids 32^3 / 64^3 / 128^3).

    dc/dt = Laplace( c^3 - c - eps^2 Laplace c ),   semi-implicit spectral time stepping as in 2D

EPS is larger than in 2D (0.008 there): the 3D grids are coarser, and the features must stay resolved on 32^3 and be
larger than the crack length scale l = 1/16. Generated once on N_FINE^3 and sampled (nearest voxel) on the simulation
grids, so every grid sees the same geometry.

usage:  python make_cahn_hilliard_geometry_3D.py    writes geometry/cahn_hilliard_3D_128.npy (1 = phase 1, 0 = phase 2)
                                                    and figures/cahn_hilliard_geometry_3D.png
"""
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
N_FINE = 128
EPS = float(os.environ.get('EPS', 0.02))
C0 = -0.12                      # mean concentration as in 2D
DT = 2.5e-5 * (EPS / 0.02) ** 2
STEPS = int(os.environ.get('STEPS', 4000))
SEED = 0

k = 2 * np.pi * np.fft.fftfreq(N_FINE, 1.0 / N_FINE)
k2 = k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2
rng = np.random.default_rng(SEED)
c = C0 + 0.05 * rng.standard_normal((N_FINE,) * 3)
c_hat = np.fft.fftn(c)
denom = 1.0 + DT * EPS ** 2 * k2 ** 2
for _ in range(STEPS):
    c_hat = (c_hat - DT * k2 * np.fft.fftn(c ** 3 - c)) / denom
    c = np.real(np.fft.ifftn(c_hat))
phase = np.where(c > 0, 0, 1).astype(np.int8)            # 1: phase 1 (stiff matrix), 0: phase 2 (soft)
os.makedirs(os.path.join(HERE, 'geometry'), exist_ok=True)
np.save(os.path.join(HERE, 'geometry', f'cahn_hilliard_3D_{N_FINE}.npy'), phase)


def n_components(mask):
    """Number of connected components with periodic 6-neighbour connectivity (union-find on the voxels)."""
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    parent = list(range(n + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for ax in range(3):
        a = np.take(lab, 0, axis=ax).ravel(); b = np.take(lab, -1, axis=ax).ravel()
        for x, y in zip(a[(a > 0) & (b > 0)], b[(a > 0) & (b > 0)]):
            parent[find(x)] = find(y)
    return len({find(i) for i in range(1, n + 1)})


frac2 = 1 - phase.mean()
print(f'EPS {EPS}, {STEPS} steps: phase-2 fraction {frac2:.3f}, components: phase 2 {n_components(phase == 0)}, '
      f'phase 1 {n_components(phase == 1)}')
fig, axs = plt.subplots(1, 4, figsize=(17, 4.6))
for ax, (nm, sl) in zip(axs[:3], [('x = 1/2', phase[N_FINE // 2]), ('y = 1/2', phase[:, N_FINE // 2]),
                                  ('z = 1/2', phase[:, :, N_FINE // 2])]):
    ax.imshow(sl.T, origin='lower', cmap='Greys_r', interpolation='nearest', extent=(0, 1, 0, 1), vmin=0, vmax=1)
    ax.set_title(f'slice {nm}', fontsize=13)
# the same slice z = 1/2 as seen by the 32^3 grid (nearest-voxel sampling), with the crack length l = 1/16 for scale
s = N_FINE // 32
axs[3].imshow(phase[::s, ::s, N_FINE // 2].T, origin='lower', cmap='Greys_r', interpolation='nearest',
              extent=(0, 1, 0, 1), vmin=0, vmax=1)
axs[3].plot([0.05, 0.05 + 1 / 16], [0.05, 0.05], 'r-', lw=4)
axs[3].set_title('z = 1/2 on 32$^3$ (red: l = 1/16)', fontsize=13)
for ax in axs:
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
fig.suptitle(f'3D Cahn-Hilliard microstructure {N_FINE}$^3$ (white: stiff phase 1, black: phase 2, '
             f'fraction {frac2:.2f})', fontsize=14)
fig.tight_layout()
os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
fig.savefig(os.path.join(HERE, 'figures', 'cahn_hilliard_geometry_3D.png'), dpi=110)
