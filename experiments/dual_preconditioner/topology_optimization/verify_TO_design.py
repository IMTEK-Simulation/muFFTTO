"""Independent check of the objective of a final design of example_2D_elasticity_TO_simp_mma.py: the homogenized
stresses recomputed from scratch with the library routines only (domain.get_rhs_mugrid, get_homogenized_stress_mugrid,
topology_optimization.compute_stress_equivalence_potential -- none of the driver's own formulas), CG from zero with
the residual test |r| <= 1e-10 |b|, a full material field per quadrature point (no ScaledMaterial).

    python verify_TO_design.py data/example_2D_elasticity_TO_simp_mma/<name>.npz [...]

Also for example_2D_elasticity_TO_phase_field_optimizers.py results (nodal phase field interpolated to the quadrature
points with the library N operator; p from the file; threshold at the quadrature points).

Prints, for the final projected design (rho^p SIMP) and its threshold rho > 0.5 (two-phase): J, the homogenized
tangent (Voigt), the number of distinct stiffness values, and J as stored by the driver.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np

from muFFTTO import domain, material_models, solvers, topology_optimization

dim, weight, nb_load_cases = 2, 5.0, 3
K_0, G_0 = 1.0, 0.5
C_0 = material_models.get_elastic_material_tensor(dim=dim, K=K_0, mu=G_0, kind='linear')
E_0 = 9 * K_0 * G_0 / (3 * K_0 + G_0)  # target as in example_2D_elasticity_TO.py
G_aux, poisson = (3 / 20) * E_0, -0.5
K_t, G_t = material_models.get_bulk_and_shear_modulus(E=2 * G_aux * (1 + poisson), poisson=poisson)
C_target = material_models.get_elastic_material_tensor(dim=dim, K=K_t, mu=G_t, kind='linear')
macro = [np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([[0.0, 0.0], [0.0, 1.0]]), np.array([[0.0, 0.5], [0.5, 0.0]])]


def objective(stiffness_scale, soft_phase):
    """J of the pixel stiffness scale s (material C_void + s (C_0 - C_void)), and the homogenized tangent."""
    n = stiffness_scale.shape[-1]
    disc = domain.Discretization(cell=domain.PeriodicUnitCell(domain_size=[1, 1], problem_type='elasticity'),
                                 nb_of_pixels_global=(n, n), discretization_type='finite_element',
                                 element_type='linear_triangles')
    C_void = soft_phase * C_0
    material = disc.get_material_data_size_field_mugrid(name='verify_material')
    scale = stiffness_scale.reshape((1,) * (7 - stiffness_scale.ndim) + stiffness_scale.shape)  # [x, y] or [q, x, y]
    material.s[...] = C_void[..., None, None, None] + (C_0 - C_void)[..., None, None, None] * scale
    green = disc.get_preconditioner_Green_mugrid(reference_material_data_ijkl=C_0, name='verify_green')

    def M(x, Px):
        disc.fft.communicate_ghosts(x)
        disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green, input_nodal_field_fnxyz=x,
                                         output_nodal_field_fnxyz=Px)

    def K(x, Kx):
        disc.apply_system_matrix_mugrid(material_data_field=material, input_field_inxyz=x, output_field_inxyz=Kx,
                                        formulation='small_strain')

    J, columns = 0.0, []
    for a in range(nb_load_cases):
        E = disc.get_gradient_size_field(name='verify_E')
        disc.get_macro_gradient_field_mugrid(macro_gradient_ij=macro[a], macro_gradient_field_ijqxyz=E)
        b = disc.get_unknown_size_field(name='verify_b')
        b.s.fill(0)
        disc.get_rhs_mugrid(material_data_field_ijklqxyz=material, macro_gradient_field_ijqxyz=E, rhs_inxyz=b)
        u = disc.get_unknown_size_field(name=f'verify_u{a}')
        u.s.fill(0)
        b_norm = np.sqrt(float(np.sum(np.asarray(b.s) ** 2)))
        solvers.conjugate_gradients_mugrid(comm=disc.communicator, fc=disc.field_collection, hessp=K, b=b, x=u, P=M,
                                           tol=1e-10 * b_norm, maxiter=200000)
        stress = disc.get_homogenized_stress_mugrid(material_data_field_ijklqxyz=material, displacement_field_inxyz=u,
                                                    macro_gradient_field_ijqxyz=E, formulation='small_strain')
        target = np.einsum('ijkl,kl->ij', C_target, macro[a])
        J += weight / nb_load_cases * topology_optimization.compute_stress_equivalence_potential(
            actual_stress_ij=stress, target_stress_ij=target)
        columns.append([stress[0, 0], stress[1, 1], stress[0, 1]])
    return J, np.array(columns).T


def phase_field_at_quad_points(phase):
    """The nodal phase field interpolated to the quadrature points (library N operator) [q, x, y]."""
    n = phase.shape[0]
    disc = domain.Discretization(cell=domain.PeriodicUnitCell(domain_size=[1, 1], problem_type='elasticity'),
                                 nb_of_pixels_global=(n, n), discretization_type='finite_element',
                                 element_type='linear_triangles')
    nodal, quad = disc.get_scalar_field(name='verify_phase'), disc.get_quad_field_scalar(name='verify_phase_q')
    nodal.s[0, 0] = phase
    disc.apply_N_operator_mugrid(nodal, quad)
    return np.array(quad.s[0, 0])


for path in sys.argv[1:]:
    data = np.load(path)
    if 'sharp_quad' in data.files:  # example_2D_elasticity_TO_phase_field_optimizers.py
        rho_q, p = phase_field_at_quad_points(data['phase_field']), float(data['p'])
        print(f'== {os.path.basename(path)}   (threshold recomputed matches stored: '
              f'{np.array_equal((rho_q > 0.5).astype(float), data["sharp_quad"])})')
        for label, scale, stored in (('phase field', rho_q ** p, float(data['J_final'])),
                                     ('thresholded at quad > 0.5', (rho_q > 0.5).astype(float), float(data['J_sharp']))):
            J, C = objective(scale, float(data['soft_phase']))
            print(f'  {label:26s} J = {J:.6e}   (driver: {stored:.6e})   distinct stiffness values: '
                  f'{len(np.unique(np.round(scale, 12)))}')
            print('   C_H (Voigt) =', np.array2string(C, precision=5).replace('\n', '\n' + ' ' * 17))
        continue
    soft, n_final = float(data['soft_phase']), max(int(k[8:]) for k in data.files if k.startswith('design_N'))
    rho = data[f'design_N{n_final}'][-1]
    p = 3.0 if '_p3_' in path else float(path.split('_p')[1].split('_')[0])
    print(f'== {os.path.basename(path)}')
    for label, scale, stored in (('projected', rho ** p, data['J'][-1]),
                                 ('thresholded rho > 0.5', (rho > 0.5).astype(float), float(data['J_sharp']))):
        J, C = objective(scale, soft)
        print(f'  {label:22s} J = {J:.6e}   (driver: {stored:.6e})   distinct stiffness values: '
              f'{len(np.unique(np.round(scale, 12)))}')
        print('   C_H (Voigt) =', np.array2string(C, precision=5).replace('\n', '\n' + ' ' * 17))
