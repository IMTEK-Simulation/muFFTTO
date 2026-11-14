"""
Cubic-symmetric start designs for SYMMETRY=cubic runs (WP3) -> data/CH3D/sym_start_<name>.nc (32^3, nodal).
    python fable_make_sym_starts.py [final.nc of an unsymmetric run, symmetrised as start 'fromfull']
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np

from muFFTTO import domain, io_utils
from cubic_symmetry import cubic_mean

N = 32
x = np.arange(N) / N
X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
c = lambda k, t: np.cos(2 * np.pi * k * t)
starts = {
    'cos2': (c(2, X) + c(2, Y) + c(2, Z) + 3) / 6,                                   # the script's sine start, even
    'cos1': (c(1, X) + c(1, Y) + c(1, Z) + 3) / 6,
    'prod1': 0.5 + 0.5 * c(1, X) * c(1, Y) * c(1, Z),
    'edges': np.clip(0.5 + 0.5 * (c(1, X) * c(1, Y) + c(1, Y) * c(1, Z) + c(1, Z) * c(1, X)) / 1.5, 0, 1),
}
if len(sys.argv) > 1:
    full = np.asarray(io_utils.read_file(sys.argv[1]).fields['phase_field_solution'][-1]).reshape(N, N, N)
    starts['fromfull'] = cubic_mean(np, full)
    sharp = cubic_mean(np, full)
    starts['fromfull_sharp'] = np.clip(0.5 + 2.0 * (sharp - sharp.mean()), 0, 1)     # contrast restored after averaging
cell = domain.PeriodicUnitCell(domain_size=[1, 1, 1], problem_type='elasticity')
disc = domain.Discretization(cell=cell, nb_of_pixels_global=(N,) * 3, discretization_type='finite_element',
                             element_type='trilinear_hexahedron')
field = disc.get_scalar_field(name='phase_field_solution')
for name, rho in starts.items():
    assert np.abs(rho - cubic_mean(np, rho)).max() < 1e-12, name
    field.s[0, 0] = rho
    io_utils.save_fields(f'data/CH3D/sym_start_{name}.nc', [field])
    print(name, 'mean', round(float(rho.mean()), 3), 'min', round(float(rho.min()), 3), 'max', round(float(rho.max()), 3))
