"""
Checks of SYMMETRY=cubic in to3d_large.py (WP3):
  1. for a symmetric design the 2-load-case objective equals the 6-load-case objective, and the projected 2-load-case
     gradient equals the 6-load-case gradient;
  2. finite-difference check of the projected gradient along symmetric directions.
    python fable_check_symmetry.py [N, default 8] [cubic | cubic_chiral]       mpirun -n 2 python fable_check_symmetry.py 16
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from mpi4py import MPI

import to3d_large
from cubic_symmetry import chiral_mean, cubic_mean

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
symmetry = sys.argv[2] if len(sys.argv) > 2 else 'cubic'          # or cubic_chiral
mpi = MPI.COMM_WORLD
settings = dict(element='trilinear_hexahedron_1Q_hg', objective='energy', target_scale=0.3, cg_tol=1e-12, verbose=False)
full = to3d_large.TopologyOptimization(N, 2 / N, **settings)
symmetric = to3d_large.TopologyOptimization(N, 2 / N, symmetry=symmetry, **settings)
fft = full.disc.fft
window = tuple(slice(int(l), int(l) + int(n)) for l, n in zip(fft.subdomain_locations, fft.nb_subdomain_grid_pts))
dot = lambda a, b: mpi.allreduce(float(np.dot(a, b)))


def symmetric_field(seed, smooth):
    values = np.random.default_rng(seed).standard_normal((N, N, N))
    if smooth:                                       # a few neighbour averages (periodic)
        for _ in range(3):
            values = sum(np.roll(values, shift, axis) for axis in range(3) for shift in (-1, 1)) / 6
    return (chiral_mean if symmetry == 'cubic_chiral' else cubic_mean)(np, values)


design = symmetric_field(1, smooth=True)
design = 0.5 + 0.35 * design / np.abs(design).max()
x = design[window].ravel()
f6, g6 = full.evaluate(x)
g6 = g6.copy()
f2, g2 = symmetric.evaluate(x)
g2 = g2.copy()
if mpi.rank == 0:
    print(f'{symmetry} N {N}, {mpi.size} rank(s): f 6 load cases {f6:.14e}, 2 load cases {f2:.14e}, rel diff {abs(f6 - f2) / abs(f6):.1e}')
    print(f'  homogenised stresses, max difference: {np.abs(full.homogenized_stresses - symmetric.homogenized_stresses).max():.1e}')
norm = np.sqrt(dot(g6, g6))
difference = np.sqrt(dot(g6 - g2, g6 - g2))
if mpi.rank == 0:
    print(f'  gradient: rel diff {difference / norm:.1e}')
for trial in range(2):
    d = symmetric_field(10 + trial, smooth=False)[window].ravel()
    step = 1e-4
    fd = (symmetric.evaluate(x + step * d)[0] - symmetric.evaluate(x - step * d)[0]) / (2 * step)
    ad = dot(g2, d)
    if mpi.rank == 0:
        print(f'  symmetric direction {trial}: adjoint {ad: .8e}  FD {fd: .8e}  rel diff {abs(ad - fd) / abs(fd):.1e}')
