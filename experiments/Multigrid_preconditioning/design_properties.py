"""
Effective elastic properties of 2D TO designs (final .nc): SIMP material s = rho_q^2 (1 - k) + k (rho_q = muFFTTO N
operator), the 3 unit load cases solved with MG-PCG (library CG, energy stop 1e-8), homogenised stresses -> Voigt
stiffness -> elastic_properties.properties (copied from dual_preconditioner/topology_optimization): nu12 = -S_1122 /
S_1111, mu = C_1212, Zener ratio. Target of the TO (K_0 = 1, mu_0 = 0.5, plane strain): nu = -0.5 -> nu12 = -1/3,
mu* = 0.15 E_0 = 0.193.

    python design_properties.py <final.nc> [...] [--soft 1e-4]
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import numpy as np
import netCDF4

from muFFTTO import material_models, solvers

import mg_lib
import elastic_properties

parser = argparse.ArgumentParser()
parser.add_argument('files', nargs='+')
parser.add_argument('--soft', type=float, default=1e-4)
args = parser.parse_args()
C_0 = material_models.get_elastic_material_tensor(dim=2, K=1.0, mu=0.5, kind='linear')
cases = [np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([[0.0, 0.0], [0.0, 1.0]]), np.array([[0.0, 0.5], [0.5, 0.0]])]

for path in args.files:
    rho = np.asarray(netCDF4.Dataset(path)['phase_field_solution'][:]).squeeze()
    N = rho.shape[-1]
    hier = mg_lib.Hierarchy(np.ones((2, N, N)), C_0, n_coarsest=4, scalar=True)
    fine = hier.levels[0]
    disc = fine.disc
    ph, phq = disc.get_scalar_field(name='phase'), disc.get_quad_field_scalar(name='phase_q')
    ph.s[0, 0] = rho
    disc.apply_N_operator_mugrid(ph, phq)
    hier.update_material(np.asarray(phq.s[0, 0]) ** 2 * (1 - args.soft) + args.soft)
    hier.setup('mg')
    B = hier.preconditioner('mg')
    macro = disc.get_gradient_size_field(name='macro')
    rhs = disc.get_unknown_size_field(name='rhs')
    x = disc.get_unknown_size_field(name='x')
    stresses = []
    for E in cases:
        disc.get_macro_gradient_field_mugrid(macro_gradient_ij=E, macro_gradient_field_ijqxyz=macro)
        rhs.s.fill(0)
        disc.get_rhs_mugrid(material_data_field_ijklqxyz=fine.material, macro_gradient_field_ijqxyz=macro, rhs_inxyz=rhs)
        x.s[...] = 0
        solvers.conjugate_gradients_mugrid_experimental(
            comm=disc.communicator, fc=disc.field_collection, hessp=lambda a, b: fine._apply_K(fine.material, a, b),
            b=rhs, x=x, P=lambda r, z: z.s.__setitem__(Ellipsis, B(r.s)), tol=1e-8, maxiter=2000,
            stop_crit_norm='energy_lower_estim', energy_rtol=True, d_min=0)
        stresses.append(disc.get_homogenized_stress_mugrid(material_data_field_ijklqxyz=fine.material,
                                                           displacement_field_inxyz=x, macro_gradient_field_ijqxyz=macro,
                                                           formulation='small_strain'))
    C = elastic_properties.voigt_from_stresses(np.array(stresses))
    p = elastic_properties.properties(C)
    print(f'{os.path.basename(path)[:80]:80s} N {N:4d}  solid {np.mean(rho > 0.5):.2f}  nu12 {p["nu12"]:+.3f}  '
          f'nu21 {p["nu21"]:+.3f}  mu {p["mu"]:.3f}  C11 {C[0, 0]:.3f}  A_Z {p["A_Z"]:.2f}', flush=True)
