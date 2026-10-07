"""Regression tests for bugs found while documenting the library.

Each test pins down one fixed bug (see ``docs/known_issues.md``), so that it
cannot come back unnoticed.
"""
import warnings

import numpy as np
import pytest

from muFFTTO import domain
from muFFTTO import material_models
from muFFTTO import microstructure_library
from muFFTTO import solvers
from muFFTTO import solvers_nonlinear


def _make_discretization(problem_type, nb_pixels=(4, 5), domain_size=(1.0, 1.0),
                         element_type='linear_triangles'):
    cell = domain.PeriodicUnitCell(domain_size=list(domain_size), problem_type=problem_type)
    return domain.Discretization(cell=cell,
                                 nb_of_pixels_global=list(nb_pixels),
                                 discretization_type='finite_element',
                                 element_type=element_type)


# ---------------------------------------------------------------------------
# domain.py
# ---------------------------------------------------------------------------

def test_flux_field_uses_full_conductivity_matrix():
    """q_i = A_ij (E + grad T)_j also for a non-diagonal (anisotropic) A."""
    disc = _make_discretization('conductivity')
    rng = np.random.default_rng(0)

    A = np.array([[2.0, 0.7],
                  [0.3, 1.5]])  # deliberately non-diagonal (and non-symmetric)
    material = disc.get_material_data_size_field_mugrid(name='A_regression')
    material.s[...] = A[..., np.newaxis, np.newaxis, np.newaxis]

    temperature = disc.get_unknown_size_field(name='T_regression')
    temperature.s[...] = rng.random(temperature.s.shape)
    macro_gradient = disc.get_gradient_size_field(name='E_regression')
    disc.get_macro_gradient_field_mugrid(macro_gradient_ij=np.array([[1.0, -0.5]]),
                                         macro_gradient_field_ijqxyz=macro_gradient)

    flux = disc.get_gradient_size_field(name='q_regression')
    disc.get_flux_field_mugrid(material_data_field_ijqxyz=material,
                               temperature_field_inxyz=temperature,
                               macro_gradient_field_ijqxyz=macro_gradient,
                               output_flux_field_ijqxyz=flux)

    gradient = disc.get_gradient_size_field(name='g_regression')
    disc.apply_gradient_operator_mugrid(u_inxyz=temperature, grad_u_ijqxyz=gradient)
    total_gradient = gradient.s + macro_gradient.s
    expected = np.einsum('ij...,uj...->ui...', material.s, total_gradient)
    np.testing.assert_allclose(flux.s, expected, rtol=1e-12, atol=1e-12)


def test_operators_reject_ndarray_with_type_error():
    """Passing a NumPy array instead of a muGrid field gives a clear TypeError."""
    disc = _make_discretization('conductivity')
    gradient = disc.get_gradient_size_field(name='g_ndarray_regression')
    with pytest.raises(TypeError, match='does not support ndarray'):
        disc.apply_gradient_operator_mugrid(u_inxyz=np.zeros((1, 1, 4, 5)),
                                            grad_u_ijqxyz=gradient)


def test_integrate_field_sums_all_spatial_axes_in_3d():
    rng = np.random.default_rng(1)
    field = rng.random((3, 3, 2, 4, 5, 6))  # [i, j, q, x, y, z]
    weights = np.array([0.25, 0.75])
    result = domain.integrate_field(field, weights)
    assert result.shape == (3, 3)
    np.testing.assert_allclose(result, np.einsum('ijqxyz,q->ij', field, weights))


# ---------------------------------------------------------------------------
# solvers_nonlinear.py
# ---------------------------------------------------------------------------

def test_finite_strain_solver_starts_from_identity():
    """Homogeneous Neo-Hookean cell: F = I + H everywhere, P = P(I + H)."""
    disc = _make_discretization('elasticity', nb_pixels=(4, 4))
    lam, mu = 2.0, 1.0
    lam_field = disc.get_quad_field_scalar(name='lam_regression')
    mu_field = disc.get_quad_field_scalar(name='mu_regression')
    lam_field.s[...] = lam
    mu_field.s[...] = mu
    material = material_models.NeoHookean(discretization=disc,
                                          lam_1qxyz=lam_field,
                                          mu_1qxyz=mu_field,
                                          name='neo_hookean_regression')

    H = np.array([[0.02, 0.01],
                  [0.0, -0.01]])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        results = solvers_nonlinear.solve_finite_strain_newton_cg(
            discretization=disc, material=material, macro_gradient_ij=H,
            ninc=1, verbose=False)

    F = np.eye(2) + H
    F_field = results['total_strain_field'].s
    np.testing.assert_allclose(F_field, np.broadcast_to(F[:, :, None, None, None], F_field.shape),
                               atol=1e-10)

    # analytical first Piola-Kirchhoff stress of the Neo-Hookean model
    F_inv_T = np.linalg.inv(F).T
    P = lam * np.log(np.linalg.det(F)) * F_inv_T + mu * (F - F_inv_T)
    stress = results['stress_field'].s
    np.testing.assert_allclose(stress, np.broadcast_to(P[:, :, None, None, None], stress.shape),
                               atol=1e-10)


# ---------------------------------------------------------------------------
# solvers.py
# ---------------------------------------------------------------------------

def _diagonal_system(disc, scale):
    """SPD diagonal system A x = b with exact solution x = 1 and b = scale * a."""
    rng = np.random.default_rng(2)
    diagonal = rng.uniform(1.0, 5.0, disc.get_unknown_size_field(name='d_reg').s.shape)
    b = disc.get_unknown_size_field(name='b_reg')
    x = disc.get_unknown_size_field(name='x_reg')
    b.s[...] = scale * diagonal
    x.s[...] = 0.0

    def hessp(p, Ap):
        Ap.s[...] = diagonal * p.s

    def precond(r, z):
        z.s[...] = r.s

    return b, x, hessp, precond


def test_cg_relative_tolerance_ignores_absolute_scale():
    """With rtol=True a tiny right-hand side must still be solved accurately."""
    disc = _make_discretization('conductivity')
    scale = 1e-12  # initial residual is far below the absolute tol**2
    b, x, hessp, precond = _diagonal_system(disc, scale)
    x = solvers.conjugate_gradients_mugrid(disc.communicator, disc.field_collection,
                                           hessp, b, x, precond, tol=1e-8, rtol=True)
    np.testing.assert_allclose(x.s, scale, rtol=1e-6)


def test_experimental_cg_relative_tolerance_ignores_absolute_scale():
    disc = _make_discretization('conductivity')
    scale = 1e-12
    b, x, hessp, precond = _diagonal_system(disc, scale)
    x, _ = solvers.conjugate_gradients_mugrid_experimental(
        disc.communicator, disc.field_collection, hessp, b, x, precond,
        tol=1e-8, rtol=True)
    np.testing.assert_allclose(x.s, scale, rtol=1e-6)


def test_findS_accepts_lists():
    curve = [10.0, 4.0, 1.0, 0.5]
    Delta = [6.0, 3.0, 0.5, 0.5]
    S = solvers.findS(curve, Delta, 1)
    assert S == pytest.approx(max(c / d for c, d in zip(curve[:-1], Delta[:-1])))


def test_adam_without_callback_minimises_quadratic():
    target = np.array([1.0, -2.0, 0.5])
    x, phi, _ = solvers.adam(f=lambda x: float(np.sum((x - target) ** 2)),
                             df=lambda x: 2 * (x - target),
                             x0=np.zeros(3), n_iter=5000, alpha=0.05,
                             beta1=0.9, beta2=0.999, gtol=1e-6, ftol=0.0)
    np.testing.assert_allclose(x, target, atol=1e-3)


# ---------------------------------------------------------------------------
# material_models.py
# ---------------------------------------------------------------------------

def test_orthotropic_tensor_matches_inverted_compliance():
    """Reduced (plane-stress) orthotropic stiffness = inverse of the compliance."""
    E1, E2, G12, nu12 = 10.0, 2.0, 1.5, 0.3
    C = material_models.get_orthotropic_stiffness_tensor_plane_strain(E1, E2, G12, nu12)
    compliance = np.array([[1 / E1, -nu12 / E1],
                           [-nu12 / E1, 1 / E2]])
    stiffness = np.linalg.inv(compliance)
    np.testing.assert_allclose(C[0, 0, 0, 0], stiffness[0, 0])
    np.testing.assert_allclose(C[1, 1, 1, 1], stiffness[1, 1])
    np.testing.assert_allclose(C[0, 0, 1, 1], stiffness[0, 1])
    np.testing.assert_allclose(C[1, 1, 0, 0], stiffness[1, 0])
    np.testing.assert_allclose(C[0, 1, 0, 1], G12)


def test_voigt_notation_rejects_unsupported_dimension():
    with pytest.raises(ValueError):
        material_models.compute_Voigt_notation_4order(np.ones((1, 1, 1, 1)))


# ---------------------------------------------------------------------------
# microstructure_library.py / visualization_utils.py
# ---------------------------------------------------------------------------

def test_unimplemented_geometry_name_is_rejected():
    nb_voxels = np.array([4, 4])
    coordinates = np.asarray(np.meshgrid(*[np.arange(n) / n for n in nb_voxels], indexing='ij'))
    with pytest.raises(ValueError):
        microstructure_library.get_geometry(nb_voxels=nb_voxels,
                                            microstructure_name='uniform_x1',
                                            coordinates=coordinates)


def test_visualize_voxels_with_given_figure():
    import matplotlib.pyplot as plt
    from muFFTTO import visualization_utils
    figure = plt.figure()
    fig, ax = visualization_utils.visualize_voxels(np.ones((2, 2, 2)), figure=figure)
    assert fig is figure
    plt.close(fig)
