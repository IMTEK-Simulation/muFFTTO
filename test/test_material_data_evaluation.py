"""Strong tests for the evaluation of material data and the contractions built on it.

The older tests mostly use isotropic or diagonal materials, for which an index
mix-up (e.g. ``C_ijkl eps_lk`` instead of ``C_ijkl eps_kl``) gives the same
numbers. These tests use general anisotropic tensors and non-symmetric
gradients, so every contraction must be exactly the documented one:

* ``sigma_ij = C_ijkl eps_kl`` and ``q_i = A_ij g_j`` (docs/theory.md, Section 2.3);
* finite-strain tangent ``A_ijkl = dP_ij / dF_kl``;
* ``K = B^T W C B`` and ``f = -B^T W C E``;
* the symmetry checks of ``Discretization.assert_material_symmetry``;
* adjoint sensitivities with fully anisotropic materials vs finite differences.
"""
import warnings

import numpy as np
import pytest

from muFFTTO import domain, material_models, solvers, tensor_operations
from muFFTTO import topology_optimization as to
from muFFTTO import topology_optimization_conductivity as toc

warnings.filterwarnings('ignore', category=RuntimeWarning)

ELEMENTS = [('linear_triangles', (3, 4), (1.3, 0.9)),
            ('bilinear_rectangle', (3, 3), (1.1, 0.7)),
            ('trilinear_hexahedron', (2, 2, 3), (1.0, 0.8, 1.2))]


def make_discretization(problem_type, element_type, nb_pixels, domain_size):
    cell = domain.PeriodicUnitCell(domain_size=list(domain_size), problem_type=problem_type)
    return domain.Discretization(cell=cell, nb_of_pixels_global=list(nb_pixels),
                                 discretization_type='finite_element', element_type=element_type)


def anisotropic_stiffness(dim, rng):
    """Random positive definite stiffness with major and both minor symmetries."""
    pairs = [(i, j) for i in range(dim) for j in range(i, dim)]
    A = rng.normal(size=(len(pairs), len(pairs)))
    mandel = A @ A.T + len(pairs) * np.eye(len(pairs))
    C = np.zeros((dim,) * 4)
    for a, (i, j) in enumerate(pairs):
        for b, (k, l) in enumerate(pairs):
            scale = (1.0 if i == j else np.sqrt(2)) * (1.0 if k == l else np.sqrt(2))
            for I, J in {(i, j), (j, i)}:
                for K, L in {(k, l), (l, k)}:
                    C[I, J, K, L] = mandel[a, b] / scale
    return C


def anisotropic_conductivity(dim, rng):
    """Random symmetric positive definite conductivity."""
    A = rng.normal(size=(dim, dim))
    return A @ A.T + dim * np.eye(dim)


def spread(tensor, field):
    """Broadcast a constant tensor over the trailing (q, x, y[, z]) axes of ``field``."""
    extra = field.s.ndim - tensor.ndim
    return np.broadcast_to(tensor[(...,) + (np.newaxis,) * extra], field.s.shape)


# ---------------------------------------------------------------------------
# point-wise material evaluation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('element_type, nb_pixels, domain_size', ELEMENTS)
def test_elastic_material_contraction_is_C_ijkl_eps_kl(element_type, nb_pixels, domain_size):
    """apply_material_data_mugrid with a general (no symmetry) C and a non-symmetric gradient."""
    disc = make_discretization('elasticity', element_type, nb_pixels, domain_size)
    disc.check_material_symmetry = False  # deliberately non-symmetric data
    rng = np.random.default_rng(0)
    C = disc.get_material_data_size_field_mugrid(name='C_general')
    C.s[...] = rng.normal(size=C.s.shape)
    g = disc.get_gradient_size_field(name='g_general')
    g.s[...] = rng.normal(size=g.s.shape)

    expected = np.zeros_like(g.s)
    d = disc.domain_dimension
    for i, j, k, l in np.ndindex(d, d, d, d):
        expected[i, j] += C.s[i, j, k, l] * g.s[k, l]

    disc.apply_material_data_mugrid(material_data=C, gradient_field=g)
    np.testing.assert_allclose(g.s, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize('element_type, nb_pixels, domain_size', ELEMENTS)
def test_conductivity_contraction_is_A_ij_g_j(element_type, nb_pixels, domain_size):
    disc = make_discretization('conductivity', element_type, nb_pixels, domain_size)
    disc.check_material_symmetry = False
    rng = np.random.default_rng(1)
    A = disc.get_material_data_size_field_mugrid(name='A_general')
    A.s[...] = rng.normal(size=A.s.shape)
    g = disc.get_gradient_size_field(name='g_cond_general')
    g.s[...] = rng.normal(size=g.s.shape)

    expected = np.zeros_like(g.s)
    d = disc.domain_dimension
    for i, j in np.ndindex(d, d):
        expected[0, i] += A.s[i, j] * g.s[0, j]

    disc.apply_material_data_mugrid(material_data=A, gradient_field=g)
    np.testing.assert_allclose(g.s, expected, rtol=1e-13, atol=1e-13)


def test_tensor_operations_ddot_contractions():
    rng = np.random.default_rng(2)
    disc = make_discretization('elasticity', 'linear_triangles', (2, 3), (1.0, 1.0))
    A4 = disc.get_material_data_size_field_mugrid(name='A4')
    B4 = disc.get_material_data_size_field_mugrid(name='B4')
    C4 = disc.get_material_data_size_field_mugrid(name='C4')
    B2 = disc.get_gradient_size_field(name='B2')
    C2 = disc.get_gradient_size_field(name='C2')
    for f in (A4, B4, B2):
        f.s[...] = rng.normal(size=f.s.shape)
    tensor_operations.ddot42(A4, B2, C2)
    np.testing.assert_allclose(C2.s, np.einsum('ijklq...,klq...->ijq...', A4.s, B2.s))
    tensor_operations.ddot44(A4, B4, C4)
    np.testing.assert_allclose(C4.s, np.einsum('ijklq...,klmnq...->ijmnq...', A4.s, B4.s))


@pytest.mark.parametrize('element_type, nb_pixels, domain_size',
                         [ELEMENTS[0], ELEMENTS[2]])
def test_neo_hookean_tangent_is_dP_ij_dF_kl(element_type, nb_pixels, domain_size):
    """Stored tangent equals the finite-difference dP_ij/dF_kl component by component."""
    disc = make_discretization('elasticity', element_type, nb_pixels, domain_size)
    d = disc.domain_dimension
    lam = disc.get_quad_field_scalar(name='lam_nh')
    mu = disc.get_quad_field_scalar(name='mu_nh')
    rng = np.random.default_rng(3)
    lam.s[...] = rng.uniform(1.0, 3.0, lam.s.shape)
    mu.s[...] = rng.uniform(0.5, 2.0, mu.s.shape)
    material = material_models.NeoHookean(discretization=disc, lam_1qxyz=lam, mu_1qxyz=mu)

    F = disc.get_strain_sized_field(name='F_nh')
    F.s[...] = spread(np.eye(d), F) + rng.normal(0, 0.15, F.s.shape)  # non-symmetric F
    A = disc.get_material_data_size_field_mugrid(name='A_nh')
    material.get_algorithmic_tangent(F, A)

    P = disc.get_stress_sized_field(name='P_nh')
    F_shift = disc.get_strain_sized_field(name='F_shift_nh')
    h = 1e-6
    for k, l in np.ndindex(d, d):
        F_shift.s[...] = F.s
        F_shift.s[k, l] += h
        material.get_stress(F_shift, P)
        P_plus = P.s.copy()
        F_shift.s[k, l] -= 2 * h
        material.get_stress(F_shift, P)
        dP_dFkl = (P_plus - P.s) / (2 * h)
        np.testing.assert_allclose(A.s[:, :, k, l], dP_dFkl, atol=1e-7)

    # hyperelastic tangent: major symmetry A_ijkl = A_klij
    np.testing.assert_allclose(A.s, np.einsum('ijkl...->klij...', A.s), atol=1e-12)


def test_linear_elastic_tangent_has_all_symmetries():
    disc = make_discretization('elasticity', 'trilinear_hexahedron', (2, 2, 2), (1.0, 1.0, 1.0))
    lam = disc.get_quad_field_scalar(name='lam_le')
    mu = disc.get_quad_field_scalar(name='mu_le')
    lam.s[...] = 1.3
    mu.s[...] = 0.7
    material = material_models.LinearElastic(discretization=disc, lam_1qxyz=lam, mu_1qxyz=mu)
    eps = disc.get_strain_sized_field(name='eps_le')
    C = disc.get_material_data_size_field_mugrid(name='C_le')
    material.get_algorithmic_tangent(eps, C)
    disc.assert_material_symmetry(C, minor=True)  # must not raise


# ---------------------------------------------------------------------------
# assembled operators
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('element_type, nb_pixels, domain_size', ELEMENTS)
def test_system_matrix_and_rhs_with_anisotropic_material(element_type, nb_pixels, domain_size):
    """K u = B^T W (C : B u), f = -B^T W (C : E), and K is symmetric."""
    disc = make_discretization('elasticity', element_type, nb_pixels, domain_size)
    d = disc.domain_dimension
    rng = np.random.default_rng(4)
    C = disc.get_material_data_size_field_mugrid(name='C_aniso')
    C.s[...] = spread(anisotropic_stiffness(d, rng), C) * rng.uniform(0.5, 2.0, C.s.shape[4:])

    u = disc.get_displacement_sized_field(name='u_aniso')
    v = disc.get_displacement_sized_field(name='v_aniso')
    u.s[...] = rng.normal(size=u.s.shape)
    v.s[...] = rng.normal(size=v.s.shape)
    Ku = disc.get_displacement_sized_field(name='Ku_aniso')
    Kv = disc.get_displacement_sized_field(name='Kv_aniso')
    disc.apply_system_matrix_mugrid(material_data_field=C, input_field_inxyz=u,
                                    output_field_inxyz=Ku, formulation='small_strain')
    disc.apply_system_matrix_mugrid(material_data_field=C, input_field_inxyz=v,
                                    output_field_inxyz=Kv, formulation='small_strain')

    # explicit B^T W (C : sym(B u))
    g = disc.get_gradient_size_field(name='g_explicit')
    disc.apply_gradient_operator_symmetrized_mugrid(u_inxyz=u, grad_u_ijqxyz=g)
    g.s[...] = np.einsum('ijkl...,kl...->ij...', C.s, g.s)
    expected = disc.get_displacement_sized_field(name='Ku_explicit')
    disc.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=g, div_u_fnxyz=expected,
                                                   apply_weights=True)
    np.testing.assert_allclose(Ku.s, expected.s, rtol=1e-12, atol=1e-12)

    # symmetry <K u, v> = <u, K v>
    assert np.vdot(Ku.s, v.s) == pytest.approx(np.vdot(u.s, Kv.s), rel=1e-12)

    # right-hand side f = -B^T W (C : E) for a symmetric macro strain
    E = rng.normal(size=(d, d))
    E = 0.5 * (E + E.T)
    E_field = disc.get_gradient_size_field(name='E_aniso')
    disc.get_macro_gradient_field_mugrid(macro_gradient_ij=E, macro_gradient_field_ijqxyz=E_field)
    f = disc.get_displacement_sized_field(name='f_aniso')
    disc.get_rhs_mugrid(material_data_field_ijklqxyz=C, macro_gradient_field_ijqxyz=E_field, rhs_inxyz=f)
    g.s[...] = np.einsum('ijkl...,kl...->ij...', C.s, E_field.s)
    disc.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=g, div_u_fnxyz=expected,
                                                   apply_weights=True)
    np.testing.assert_allclose(f.s, -expected.s, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('element_type, nb_pixels, domain_size', ELEMENTS)
def test_homogenized_stress_of_homogeneous_cell_is_column_kl(element_type, nb_pixels, domain_size):
    """For a homogeneous anisotropic cell and E = e_k (x) e_l the average stress is C[:, :, k, l]."""
    disc = make_discretization('elasticity', element_type, nb_pixels, domain_size)
    d = disc.domain_dimension
    C_const = anisotropic_stiffness(d, np.random.default_rng(5))
    C = disc.get_material_data_size_field_mugrid(name='C_homog')
    C.s[...] = spread(C_const, C)
    u = disc.get_displacement_sized_field(name='u_homog')  # zero fluctuation solves the cell problem
    E_field = disc.get_gradient_size_field(name='E_homog')
    for k, l in np.ndindex(d, d):
        E = np.zeros((d, d))
        E[k, l] = 1.0
        disc.get_macro_gradient_field_mugrid(macro_gradient_ij=E, macro_gradient_field_ijqxyz=E_field)
        stress = disc.get_homogenized_stress_mugrid(C, u, E_field, formulation='finite_strain')
        np.testing.assert_allclose(stress, C_const[:, :, k, l], rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# symmetry checks
# ---------------------------------------------------------------------------

def test_symmetry_check_rejects_and_accepts():
    rng = np.random.default_rng(6)
    disc = make_discretization('elasticity', 'linear_triangles', (3, 3), (1.0, 1.0))
    C = disc.get_material_data_size_field_mugrid(name='C_check')
    C.s[...] = spread(anisotropic_stiffness(2, rng), C)
    disc.assert_material_symmetry(C, minor=True)  # valid: no error

    broken_major = C.s.copy()
    broken_major[0, 0, 1, 1] += 0.1  # C_0011 != C_1100
    with pytest.raises(ValueError, match='major'):
        disc.assert_material_symmetry(broken_major)

    broken_minor = C.s.copy()
    broken_minor[0, 1, 0, 0] += 0.1  # C_0100 != C_1000 ...
    broken_minor[0, 0, 0, 1] += 0.1  # ... but keep major symmetry
    disc.assert_material_symmetry(broken_minor)  # major only: fine
    with pytest.raises(ValueError, match='minor'):
        disc.assert_material_symmetry(broken_minor, minor=True)

    disc.check_material_symmetry = False
    disc.assert_material_symmetry(broken_major)  # switched off: no error


def test_solve_entry_points_check_symmetry():
    disc = make_discretization('conductivity', 'linear_triangles', (3, 3), (1.0, 1.0))
    A = disc.get_material_data_size_field_mugrid(name='A_nonsym')
    A.s[...] = spread(np.array([[2.0, 0.5], [0.1, 1.0]]), A)
    E_field = disc.get_gradient_size_field(name='E_nonsym')
    rhs = disc.get_unknown_size_field(name='rhs_nonsym')
    with pytest.raises(ValueError, match='A_ij = A_ji'):
        disc.get_rhs_mugrid(material_data_field_ijklqxyz=A, macro_gradient_field_ijqxyz=E_field,
                            rhs_inxyz=rhs)
    with pytest.raises(ValueError):
        disc.get_preconditioner_Green_mugrid(reference_material_data_ijkl=np.array([[2.0, 0.5], [0.1, 1.0]]))


# ---------------------------------------------------------------------------
# adjoint sensitivities with fully anisotropic materials
# ---------------------------------------------------------------------------

def finite_difference_gradient(fun, x0, eps=1e-5):
    grad = np.zeros_like(x0)
    x = x0.copy()
    for i in range(x0.size):
        original = x.flat[i]
        x.flat[i] = original + eps
        f_plus = fun(x)
        x.flat[i] = original - eps
        f_minus = fun(x)
        x.flat[i] = original
        grad.flat[i] = (f_plus - f_minus) / (2 * eps)
    return grad


def identity_preconditioner(x, Px):
    Px.s[...] = x.s


def simp_material(disc, rho_values, base, void, p, name):
    rho = disc.get_scalar_field(name='rho_simp_tmp')
    rho.s[...] = rho_values
    rho_q = disc.get_quad_field_scalar(name='rho_q_simp_tmp')
    disc.apply_N_operator_mugrid(rho, rho_q)
    material = disc.get_material_data_size_field_mugrid(name=name)
    to.material_interpolation_simp(rho_q, material, p, void, base, disc.domain_dimension)
    return material


@pytest.mark.parametrize('element_type, nb_pixels, domain_size', [ELEMENTS[0], ELEMENTS[2]])
def test_elasticity_sensitivity_with_anisotropic_material(element_type, nb_pixels, domain_size):
    disc = make_discretization('elasticity', element_type, nb_pixels, domain_size)
    d = disc.domain_dimension
    rng = np.random.default_rng(7)
    C1 = anisotropic_stiffness(d, rng)
    C0 = 0.05 * anisotropic_stiffness(d, rng)
    E = rng.normal(size=(d, d))
    E = 0.5 * (E + E.T)
    target = rng.normal(size=(d, d))
    target = 0.5 * (target + target.T)
    p, weight = 3, 2.5

    E_field = disc.get_gradient_size_field(name='E_sens')
    disc.get_macro_gradient_field_mugrid(macro_gradient_ij=E, macro_gradient_field_ijqxyz=E_field)
    rho = disc.get_scalar_field(name='rho_sens')
    rho.s[...] = 0.2 + 0.6 * rng.random(rho.s.shape)

    def solve_state(rho_values):
        C = simp_material(disc, rho_values, C1, C0, p, 'C_state')

        def K(x, Ax):
            disc.apply_system_matrix_mugrid(material_data_field=C, input_field_inxyz=x,
                                            output_field_inxyz=Ax, formulation='small_strain')

        rhs = disc.get_unknown_size_field(name='rhs_state')
        disc.get_rhs_mugrid(C, E_field, rhs)
        u = disc.get_displacement_sized_field(name='u_state')
        u.s.fill(0)
        solvers.conjugate_gradients_mugrid(comm=disc.communicator, fc=disc.field_collection, hessp=K,
                                           b=rhs, x=u, P=identity_preconditioner, tol=1e-13, maxiter=20000)
        stress = disc.get_homogenized_stress_mugrid(C, u, E_field, formulation='small_strain')
        return K, u, stress

    def objective(rho_values):
        return weight * to.compute_stress_equivalence_potential(solve_state(rho_values)[2], target)

    K, u, stress = solve_state(rho.s.copy())
    adjoint = disc.get_displacement_sized_field(name='lambda_sens')
    sensitivity = to.sensitivity_stress_and_adjoint_FE_NEW(
        disc, C1, C0, u, adjoint, E_field, rho, target, stress,
        identity_preconditioner, K, p, weight, cg_tol=1e-13)[0]
    reference = finite_difference_gradient(objective, rho.s.copy())
    np.testing.assert_allclose(np.asarray(sensitivity), reference,
                               rtol=1e-6, atol=1e-8 * np.abs(reference).max())


@pytest.mark.parametrize('element_type, nb_pixels, domain_size', [ELEMENTS[0], ELEMENTS[2]])
def test_conductivity_sensitivity_with_anisotropic_material(element_type, nb_pixels, domain_size):
    disc = make_discretization('conductivity', element_type, nb_pixels, domain_size)
    d = disc.domain_dimension
    rng = np.random.default_rng(8)
    A1 = anisotropic_conductivity(d, rng)
    A0 = 0.05 * anisotropic_conductivity(d, rng)
    E = rng.normal(size=(1, d))
    target = rng.normal(size=(1, d))
    p, weight = 3, 2.5

    E_field = disc.get_gradient_size_field(name='E_sens_c')
    disc.get_macro_gradient_field_mugrid(macro_gradient_ij=E, macro_gradient_field_ijqxyz=E_field)
    rho = disc.get_scalar_field(name='rho_sens_c')
    rho.s[...] = 0.2 + 0.6 * rng.random(rho.s.shape)

    def solve_state(rho_values):
        A = simp_material(disc, rho_values, A1, A0, p, 'A_state')

        def K(x, Ax):
            disc.apply_system_matrix_mugrid(material_data_field=A, input_field_inxyz=x,
                                            output_field_inxyz=Ax)

        rhs = disc.get_unknown_size_field(name='rhs_state_c')
        disc.get_rhs_mugrid(A, E_field, rhs)
        u = disc.get_unknown_size_field(name='u_state_c')
        u.s.fill(0)
        solvers.conjugate_gradients_mugrid(comm=disc.communicator, fc=disc.field_collection, hessp=K,
                                           b=rhs, x=u, P=identity_preconditioner, tol=1e-13, maxiter=20000)
        return K, u, disc.get_homogenized_stress_mugrid(A, u, E_field)

    def objective(rho_values):
        return weight * toc.compute_flux_equivalence_potential(solve_state(rho_values)[2], target)

    K, u, flux = solve_state(rho.s.copy())
    adjoint = disc.get_unknown_size_field(name='lambda_sens_c')
    sensitivity = toc.sensitivity_flux_and_adjoint(
        disc, A1, A0, u, adjoint, E_field, rho, target, flux,
        identity_preconditioner, K, p, weight, cg_tol=1e-13)[0]
    reference = finite_difference_gradient(objective, rho.s.copy())
    np.testing.assert_allclose(np.asarray(sensitivity), reference,
                               rtol=1e-6, atol=1e-8 * np.abs(reference).max())
