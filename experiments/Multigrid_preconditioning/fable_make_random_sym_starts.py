"""
Random cubic-symmetric start designs (WP3): smooth random fields (sum of a few low Fourier modes, built from cosines
and sines directly), symmetrised, rescaled to [0.05, 0.95] around the mean 0.5 -> data/CH3D/sym_start_rnd<k>_s<seed>.nc
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np

from muFFTTO import domain, io_utils
from cubic_symmetry import cubic_mean

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
        rho = cubic_mean(np, rho)
        rho = np.clip(0.5 + 0.45 * (rho - rho.mean()) / np.abs(rho - rho.mean()).max() * 2.0, 0.0, 1.0)
        field.s[0, 0] = cubic_mean(np, rho)
        io_utils.save_fields(f'data/CH3D/sym_start_rnd{k_max}_s{seed}.nc', [field])
