"""
muTopOpt's default start design (`filtered_random`, seed 0, volume fraction 0.5, correlation length 3 eta) as a start
file of to3d_large.py, so that both codes optimise from the same field:
    python fable_make_mutopopt_start.py <N> [eta in voxels, default 2]   -> data/CH3D/mutopopt_start_N<N>.nc
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from muTopOpt.optimize import initial_density

from muFFTTO import domain, io_utils

N = int(sys.argv[1])
eta = (float(sys.argv[2]) if len(sys.argv) > 2 else 2.0) / N
rho = initial_density((N, N, N), kind='filtered_random', volume_fraction=0.5, seed=0, length=3.0 * eta,
                      grid_spacing=(1.0 / N,) * 3)
cell = domain.PeriodicUnitCell(domain_size=[1, 1, 1], problem_type='elasticity')
disc = domain.Discretization(cell=cell, nb_of_pixels_global=(N,) * 3, discretization_type='finite_element',
                             element_type='trilinear_hexahedron_1Q_hg')
field = disc.get_scalar_field(name='phase_field_solution')
field.s[0, 0] = rho
io_utils.save_fields(f'data/CH3D/mutopopt_start_N{N}.nc', [field])
print(f'N {N}: mean {rho.mean():.4f}, min {rho.min():.3f}, max {rho.max():.3f}')
