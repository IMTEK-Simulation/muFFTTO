"""
Random cubic-symmetric start designs (WP3): smooth random fields (sum of a few low Fourier modes, built from cosines
and sines directly), projected onto the cubic-symmetric (sym) or the rotation-symmetric (chiral) fields and rescaled to [0.05, 0.95]
around the mean 0.5 -> data/CH3D/{sym,chiral}_start_rnd<k>_s<seed>.nc
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np

from muFFTTO import domain, io_utils
from cubic_symmetry import chiral_mean, cubic_mean

N = 32
x = np.arange(N) / N
X = np.meshgrid(x, x, x, indexing='ij')
cell = domain.PeriodicUnitCell(domain_size=[1, 1, 1], problem_type='elasticity')
disc = domain.Discretization(cell=cell, nb_of_pixels_global=(N,) * 3, discretization_type='finite_element',
                             element_type='trilinear_hexahedron')
field = disc.get_scalar_field(name='phase_field_solution')
for k_max in (2, 3):
    for seed in range(4):
        rng = np.random.default_rng(100 * k_max + seed)
        rho = np.zeros((N, N, N))
        for k in np.ndindex(k_max + 1, k_max + 1, k_max + 1):
            if 0 < sum(k) and max(k) <= k_max:
                phase = 2 * np.pi * sum(ka * Xa for ka, Xa in zip(k, X))
                rho += rng.standard_normal() * np.cos(phase) + rng.standard_normal() * np.sin(phase)
        for name, mean in (('sym', cubic_mean), ('chiral', chiral_mean)):       # mirror-symmetric / rotations only
            start = mean(np, rho)
            start = np.clip(0.5 + 0.9 * (start - start.mean()) / np.abs(start - start.mean()).max(), 0.0, 1.0)
            field.s[0, 0] = mean(np, start)
            io_utils.save_fields(f'data/CH3D/{name}_start_rnd{k_max}_s{seed}.nc', [field])
