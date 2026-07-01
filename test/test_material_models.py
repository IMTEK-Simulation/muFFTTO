import pytest
import numpy as np

from muFFTTO import material_models
from muFFTTO import domain


@pytest.fixture()
def discretization_fixture(domain_size, element_type, nb_pixels):
    problem_type = 'elasticity'
    element_types = ['linear_triangles', 'linear_triangles_tilled', 'trilinear_hexahedron']
    my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                      problem_type=problem_type)
    discretization_type = 'finite_element'
    discretization = domain.Discretization(cell=my_cell,
                                           nb_of_pixels_global=nb_pixels,
                                           discretization_type=discretization_type,
                                           element_type=element_types[element_type])
    return discretization


discretization_cases = pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
    # --- 2D: linear_triangles ---
    ([2, 2], 0, [2, 2]),
    ([2, 3], 0, [2, 3]),
    ([2, 4], 0, [2, 4]),
    ([3, 2], 0, [3, 2]),
    ([3, 3], 0, [3, 3]),
    ([3, 4], 0, [3, 4]),
    ([4, 2], 0, [4, 2]),
    ([4, 3], 0, [4, 3]),
    ([4, 4], 0, [4, 4]),
    # --- 2D: linear_triangles_tilled ---
    ([2, 2], 1, [2, 2]),
    ([2, 3], 1, [2, 3]),
    ([2, 4], 1, [2, 4]),
    ([3, 2], 1, [3, 2]),
    ([3, 3], 1, [3, 3]),
    ([3, 4], 1, [3, 4]),
    ([4, 2], 1, [4, 2]),
    ([4, 3], 1, [4, 3]),
    ([4, 4], 1, [4, 4]),
    # --- 3D: trilinear_hexahedron ---
    # cubic domain, uniform pixels
    ([2, 2, 2], 2, (2, 2, 2)),
    ([3, 3, 3], 2, (3, 3, 3)),
    ([4, 4, 4], 2, (4, 4, 4)),
    # non-cubic domain, uniform pixels (isolates aspect-ratio effects from grid resolution)
    ([2, 3, 4], 2, (3, 3, 3)),
    ([4, 2, 3], 2, (3, 3, 3)),
    ([3, 4, 2], 2, (3, 3, 3)),
    # non-cubic domain, matching non-uniform pixels per axis
    ([2, 3, 4], 2, (2, 3, 4)),
    ([4, 2, 3], 2, (4, 2, 3)),
    ([3, 4, 2], 2, (3, 4, 2)),
    # mismatched domain/pixel ratios (domain_size shape != nb_pixels shape)
    ([4, 3, 5], 2, (3, 4, 2)),
    ([2, 4, 3], 2, (4, 2, 3)),
    # heavier resolution case (as originally requested)
    ([4, 3, 5], 2, (16, 16, 16)),
])


@discretization_cases
def test_discretization_init(discretization_fixture):
    assert hasattr(discretization_fixture, "cell")
    assert hasattr(discretization_fixture, "domain_dimension")
    assert hasattr(discretization_fixture, "B_grad_at_pixel_dqnijk")
    assert hasattr(discretization_fixture, "quadrature_weights")
    assert hasattr(discretization_fixture, "nb_quad_points_per_pixel")
    assert hasattr(discretization_fixture, "nb_nodes_per_pixel")


@discretization_cases
def test_linear_isotropic_elasticity_FD(discretization_fixture):
    """
    FD check that sigma = stress_from_strain_lame(eps, lam, mu) is the
    gradient of W(eps) = 0.5 * sigma:eps, where W is built from the
    model's own stress output (self-consistency check), evaluated over
    the whole grid (all quad points, all pixels).
    """
    rng = np.random.default_rng(0)
    dim = discretization_fixture.domain_dimension

    strain_ijqxyz = discretization_fixture.get_strain_sized_field(name='strain_temp')
    stress_ijqxyz = discretization_fixture.get_stress_sized_field(name='stress_temp')
    lam_1qxyz = discretization_fixture.get_quad_field_scalar(name='lam_temp')
    mu_1qxyz = discretization_fixture.get_quad_field_scalar(name='mu_temp')

    shape = strain_ijqxyz.s[...].shape
    point_shape = shape[2:]

    raw = rng.normal(size=shape)
    strain = 0.5 * (raw + np.swapaxes(raw, 0, 1))
    strain_ijqxyz.s[...] = strain

    lam_1qxyz.s[...] = rng.uniform(1.0, 5.0, size=(1,) + point_shape)
    mu_1qxyz.s[...] = rng.uniform(1.0, 5.0, size=(1,) + point_shape)
    lam = lam_1qxyz.s[...]
    mu = mu_1qxyz.s[...]

    def energy_density(eps):
        strain_perturbed_ijqxyz = discretization_fixture.get_strain_sized_field(name='strain_perturbed_ijqxyz')
        strain_perturbed_ijqxyz.s[...] = eps
        material_models.linear_isotropic_elasticity_stress_from_strain_lame(strain_perturbed_ijqxyz, lam_1qxyz, mu_1qxyz, stress_ijqxyz)
        sigma = stress_ijqxyz.s[...]
        return 0.5 * np.einsum('ij...,ij...->...', sigma, eps)

    # analytic stress
    material_models.linear_isotropic_elasticity_stress_from_strain_lame(strain_ijqxyz, lam_1qxyz, mu_1qxyz, stress_ijqxyz)
    sigma_analytic = stress_ijqxyz.s[...].copy()

    h = 1e-6
    sigma_fd = np.zeros_like(sigma_analytic)

    for i in range(dim):
        for j in range(dim):
            dstrain = np.zeros_like(strain)
            if i == j:
                dstrain[i, j, ...] = h
            else:
                dstrain[i, j, ...] = h / 2
                dstrain[j, i, ...] = h / 2

            W_plus = energy_density(strain + dstrain)
            W_minus = energy_density(strain - dstrain)

            sigma_fd[i, j, ...] = (W_plus - W_minus) / (2 * h)

    np.testing.assert_allclose(sigma_analytic, sigma_fd, rtol=1e-5, atol=1e-8)


@discretization_cases
def test_linear_isotropic_elasticity_FD_convergence(discretization_fixture):
    """
    Plots FD error (in the sigma = dW/deps check) vs. step size h,
    on a log-log scale, to confirm the expected O(h^2) convergence of the
    centered finite-difference scheme, and to show where floating-point
    round-off error starts to dominate for very small h.
    """
    rng = np.random.default_rng(0)
    dim = discretization_fixture.domain_dimension

    strain_ijqxyz = discretization_fixture.get_strain_sized_field(name='strain_temp')
    stress_ijqxyz = discretization_fixture.get_stress_sized_field(name='stress_temp')
    lam_1qxyz = discretization_fixture.get_quad_field_scalar(name='lam_temp')
    mu_1qxyz = discretization_fixture.get_quad_field_scalar(name='mu_temp')

    shape = strain_ijqxyz.s[...].shape
    point_shape = shape[2:]

    raw = rng.normal(size=shape)
    strain = 0.5 * (raw + np.swapaxes(raw, 0, 1))
    strain_ijqxyz.s[...] = strain

    lam_1qxyz.s[...] = rng.uniform(1.0, 5.0, size=(1,) + point_shape)
    mu_1qxyz.s[...] = rng.uniform(1.0, 5.0, size=(1,) + point_shape)

    def energy_density(eps):
        strain_perturbed_ijqxyz = discretization_fixture.get_strain_sized_field(name='strain_perturbed_ijqxyz')
        strain_perturbed_ijqxyz.s[...] = eps
        material_models.linear_isotropic_elasticity_stress_from_strain_lame(strain_perturbed_ijqxyz, lam_1qxyz, mu_1qxyz, stress_ijqxyz)
        sigma = stress_ijqxyz.s[...]
        return 0.5 * np.einsum('ij...,ij...->...', sigma, eps)

    # analytic stress (reference, independent of h)
    material_models.linear_isotropic_elasticity_stress_from_strain_lame(strain_ijqxyz, lam_1qxyz, mu_1qxyz, stress_ijqxyz)
    sigma_analytic = stress_ijqxyz.s[...].copy()

    # --- sweep h over several decades ---
    h_values = np.logspace(-1, -8, 15)
    errors = []

    for h in h_values:
        sigma_fd = np.zeros_like(sigma_analytic)
        for i in range(dim):
            for j in range(dim):
                dstrain = np.zeros_like(strain)
                if i == j:
                    dstrain[i, j, ...] = h
                else:
                    dstrain[i, j, ...] = h / 2
                    dstrain[j, i, ...] = h / 2

                W_plus = energy_density(strain + dstrain)
                W_minus = energy_density(strain - dstrain)
                sigma_fd[i, j, ...] = (W_plus - W_minus) / (2 * h)

        err = np.linalg.norm((sigma_fd - sigma_analytic).ravel())
        errors.append(err)

    errors = np.array(errors)

    # --- sanity check ---
    # Since W is an exact quadratic form, FD matches the analytic stress to
    # near machine precision for any h in the well-conditioned regime
    # (not too small). We check that, rather than checking a convergence
    # order (there is no truncation error to converge).
    well_conditioned = slice(2, 8)  # roughly h in [1e-2, 1e-5]
    norm_ref = np.linalg.norm(sigma_analytic) + 1e-30
    rel_errors = errors[well_conditioned] / norm_ref

    assert np.all(rel_errors < 1e-6), (
        f"FD stress does not match analytic stress to near machine precision "
        f"in the well-conditioned regime: rel_errors={rel_errors}"
    )
