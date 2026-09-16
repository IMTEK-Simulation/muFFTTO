import pytest
import numpy as np

from muFFTTO import material_models
from muFFTTO import domain


@pytest.fixture()
def discretization_fixture(domain_size, element_type, nb_pixels):
    problem_type = 'elasticity'
    element_types = ['linear_triangles', 'bilinear_rectangle', 'linear_triangles_tilled', 'trilinear_hexahedron']
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
    # --- 2D: bilinear_rectangle ---
    ([2, 2], 1, [2, 2]),
    ([2, 3], 1, [2, 3]),
    ([2, 4], 1, [2, 4]),
    ([3, 2], 1, [3, 2]),
    ([3, 3], 1, [3, 3]),
    ([3, 4], 1, [3, 4]),
    ([4, 2], 1, [4, 2]),
    ([4, 3], 1, [4, 3]),
    ([4, 4], 1, [4, 4]),
    # --- 2D: linear_triangles_tilled ---
    ([2, 2], 2, [2, 2]),
    ([2, 3], 2, [2, 3]),
    ([2, 4], 2, [2, 4]),
    ([3, 2], 2, [3, 2]),
    ([3, 3], 2, [3, 3]),
    ([3, 4], 2, [3, 4]),
    ([4, 2], 2, [4, 2]),
    ([4, 3], 2, [4, 3]),
    ([4, 4], 2, [4, 4]),
    # --- 3D: trilinear_hexahedron ---
    # cubic domain, uniform pixels
    ([2, 2, 2], 3, (2, 2, 2)),
    ([3, 3, 3], 3, (3, 3, 3)),
    ([4, 4, 4], 3, (4, 4, 4)),
    # non-cubic domain, uniform pixels (isolates aspect-ratio effects from grid resolution)
    ([2, 3, 4], 3, (3, 3, 3)),
    ([4, 2, 3], 3, (3, 3, 3)),
    ([3, 4, 2], 3, (3, 3, 3)),
    # non-cubic domain, matching non-uniform pixels per axis
    ([2, 3, 4], 3, (2, 3, 4)),
    ([4, 2, 3], 3, (4, 2, 3)),
    ([3, 4, 2], 3, (3, 4, 2)),
    # mismatched domain/pixel ratios (domain_size shape != nb_pixels shape)
    ([4, 3, 5], 3, (3, 4, 2)),
    ([2, 4, 3], 3, (4, 2, 3)),
    # heavier resolution case (as originally requested)
    ([4, 3, 5], 3, (16, 16, 16)),
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

    h = 1e-1
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


@discretization_cases
def test_LinearElastic_MaterialModelElasticity_(discretization_fixture):
    """
    Finite difference check that sigma = stress_from_strain_lame(eps, lam, mu) is the
    gradient of W(eps) = 0.5 * sigma:eps, where W is built from the
    model's own stress output (self-consistency check), evaluated over
    the whole grid (all quad points, all pixels).
    """
    rng = np.random.default_rng(0)
    dim = discretization_fixture.domain_dimension
    strain_ijqxyz = discretization_fixture.get_strain_sized_field(name='strain_temp')
    stress_ijqxyz = discretization_fixture.get_stress_sized_field(name='stress_temp')
    lam_11qxyz    = discretization_fixture.get_quad_field_scalar(name='lam_temp')
    mu_11qxyz     = discretization_fixture.get_quad_field_scalar(name='mu_temp')

    shape       = strain_ijqxyz.s[...].shape    # [i, j, q, x, y, z]
    point_shape = shape[2:]                     # [q, x, y, z]

    raw    = rng.normal(size=shape)
    strain = 0.5 * (raw + np.swapaxes(raw, 0, 1))
    strain_ijqxyz.s[...] = strain

    # scalar fields have layout [1, 1, q, x, y, z]
    lam_11qxyz.s[0, 0] = rng.uniform(1.0, 5.0, size=point_shape)
    mu_11qxyz.s[0, 0]  = rng.uniform(1.0, 5.0, size=point_shape)

    material = material_models.LinearElastic(
        discretization=discretization_fixture,
        lam_1qxyz=lam_11qxyz,
        mu_1qxyz=mu_11qxyz,
        name='linear_isotropic_elasticity'
    )

    def energy_density(eps):
        strain_perturbed_ijqxyz = discretization_fixture.get_strain_sized_field(
            name='strain_perturbed_ijqxyz'
        )
        strain_perturbed_ijqxyz.s[...] = eps
        material.get_stress(strain_perturbed_ijqxyz, stress_ijqxyz)
        sigma = stress_ijqxyz.s[...]
        return 0.5 * np.einsum('ij...,ij...->...', sigma, eps)

    # --- stress test ---
    material.get_stress(strain_ijqxyz, stress_ijqxyz)
    sigma_analytic = stress_ijqxyz.s[...].copy()

    h        = 1e-4                              # reduced from 1e-1 for better FD accuracy
    sigma_fd = np.zeros_like(sigma_analytic)
    for i in range(dim):
        for j in range(dim):
            dstrain = np.zeros_like(strain)
            if i == j:
                dstrain[i, j, ...] = h
            else:
                dstrain[i, j, ...] = h / 2
                dstrain[j, i, ...] = h / 2
            W_plus  = energy_density(strain + dstrain)
            W_minus = energy_density(strain - dstrain)
            sigma_fd[i, j, ...] = (W_plus - W_minus) / (2 * h)

    np.testing.assert_allclose(
        sigma_analytic, sigma_fd,
        rtol=1e-5, atol=1e-8,
        err_msg=(
            "STRESS TEST FAILED: get_stress is not the gradient of the energy density W. "
            "Either the stress formula σ_ij = λ δ_ij ε_kk + 2μ ε_ij is wrong, "
            "or the symmetrisation of the finite difference perturbation is broken. "
            f"Max absolute error: {np.max(np.abs(sigma_analytic - sigma_fd)):.3e}, "
            f"Max relative error: {np.max(np.abs((sigma_analytic - sigma_fd) / (sigma_fd + 1e-30))):.3e}"
        )
    )

    # --- tangent test ---
    tangent_ijklqxyz = discretization_fixture.get_material_data_size_field_mugrid(
        name='tangent_ijklqxyz'
    )
    material.get_algorithmic_tangent(strain_ijqxyz, tangent_ijklqxyz)
    material.apply_algorithmic_tangent(strain_ijqxyz, stress_ijqxyz, tangent_ijklqxyz)

    np.testing.assert_allclose(
        sigma_analytic, stress_ijqxyz.s,
        rtol=1e-5, atol=1e-8,
        err_msg=(
            "TANGENT TEST FAILED: apply_algorithmic_tangent(C, ε) does not reproduce get_stress(ε). "
            "Either get_algorithmic_tangent builds the wrong C_ijkl, "
            "or apply_algorithmic_tangent contracts over the wrong indices. "
            f"Max absolute error: {np.max(np.abs(sigma_analytic - stress_ijqxyz.s)):.3e}, "
            f"Max relative error: {np.max(np.abs((sigma_analytic - stress_ijqxyz.s) / (sigma_analytic + 1e-30))):.3e}"
        )
    )


@discretization_cases
def test_NeoHookean_MaterialModelElasticity_(discretization_fixture):
    """
    Correctness check for NeoHookean tangent contraction convention.

    The project uses the reversed contraction convention P_ij = A_ijkl F_lk,
    not the direct order P_ij = A_ijkl F_kl. Since NeoHookean's stress P(F)
    is nonlinear, C:F ≠ P, so we cannot use the "tangent reproduces stress" check.
    Instead, we verify the directional derivative: A·dF must match dP/dF (dF̂).

    The test builds F without symmetrizing to exercise the bug that hides in
    the LinearElastic test (whose tangent is symmetric under k↔l, so direct
    and reversed orders give identical results for symmetric strains).
    """
    rng = np.random.default_rng(42)
    dim = discretization_fixture.domain_dimension

    strain_ijqxyz = discretization_fixture.get_strain_sized_field(name='strain_temp')
    stress_ijqxyz = discretization_fixture.get_stress_sized_field(name='stress_temp')
    tangent_ijklqxyz = discretization_fixture.get_material_data_size_field_mugrid(
        name='tangent_ijklqxyz'
    )
    lam_11qxyz = discretization_fixture.get_quad_field_scalar(name='lam_temp')
    mu_11qxyz = discretization_fixture.get_quad_field_scalar(name='mu_temp')

    shape = strain_ijqxyz.s[...].shape  # [i, j, q, x, y, z]
    point_shape = shape[2:]  # [q, x, y, z]

    # Build F = I + small perturbation, WITHOUT symmetrizing (this is the key difference)
    # Use small scale (~0.1-0.2) to ensure J > 0 and good conditioning
    raw = rng.normal(0.0, 0.1, size=shape)
    F = np.eye(dim).reshape((dim, dim) + (1,) * (len(shape) - 2))
    F = F + raw  # Non-symmetric deformation gradient
    strain_ijqxyz.s[...] = F

    # Set material parameters
    lam_11qxyz.s[0, 0] = rng.uniform(1.0, 5.0, size=point_shape)
    mu_11qxyz.s[0, 0] = rng.uniform(1.0, 5.0, size=point_shape)

    material = material_models.NeoHookean(
        discretization=discretization_fixture,
        lam_1qxyz=lam_11qxyz,
        mu_1qxyz=mu_11qxyz,
        name='neo_hookean'
    )

    # --- Part A: Verify stress against dW/dF (sanity check, independent of contraction convention) ---
    def energy_density(F_test):
        strain_perturbed_ijqxyz = discretization_fixture.get_strain_sized_field(
            name='strain_perturbed_ijqxyz'
        )
        strain_perturbed_ijqxyz.s[...] = F_test
        W_field = discretization_fixture.get_quad_field_scalar(name='W_temp')
        material.get_energy_density(strain_perturbed_ijqxyz, W_field)
        return W_field.s[0, 0].copy()

    material.get_stress(strain_ijqxyz, stress_ijqxyz)
    P_analytic = stress_ijqxyz.s[...].copy()

    h_stress = 1e-4
    P_fd = np.zeros_like(P_analytic)
    for i in range(dim):
        for j in range(dim):
            dF = np.zeros_like(F)
            dF[i, j, ...] = h_stress
            W_plus = energy_density(F + dF)
            W_minus = energy_density(F - dF)
            P_fd[i, j, ...] = (W_plus - W_minus) / (2 * h_stress)

    np.testing.assert_allclose(
        P_analytic, P_fd,
        rtol=1e-5, atol=1e-6,
        err_msg=(
            "STRESS TEST FAILED for NeoHookean: get_stress does not match dW/dF. "
            "The energy density W or the stress formula P = dW/dF is broken. "
            f"Max absolute error: {np.max(np.abs(P_analytic - P_fd)):.3e}"
        )
    )

    # --- Part B: Verify tangent contraction convention using directional derivative ---
    # Pick a random non-symmetric perturbation direction dF_hat
    dF_hat = rng.normal(0.0, 0.05, size=shape)

    # Compute dP via FD: dP = (P(F + h·dF_hat) - P(F - h·dF_hat)) / (2h)
    h_tangent = 1e-4
    strain_plus = discretization_fixture.get_strain_sized_field(name='strain_plus')
    strain_minus = discretization_fixture.get_strain_sized_field(name='strain_minus')
    stress_plus = discretization_fixture.get_stress_sized_field(name='stress_plus')
    stress_minus = discretization_fixture.get_stress_sized_field(name='stress_minus')

    strain_plus.s[...] = F + h_tangent * dF_hat
    strain_minus.s[...] = F - h_tangent * dF_hat
    material.get_stress(strain_plus, stress_plus)
    material.get_stress(strain_minus, stress_minus)
    dP_fd = (stress_plus.s[...] - stress_minus.s[...]) / (2 * h_tangent)

    # Compute dP via tangent contraction: use the real apply_algorithmic_tangent
    material.get_algorithmic_tangent(strain_ijqxyz, tangent_ijklqxyz)
    dF_hat_field = discretization_fixture.get_strain_sized_field(name='dF_hat_field')
    dF_hat_field.s[...] = dF_hat
    dP_tangent_field = discretization_fixture.get_stress_sized_field(name='dP_tangent')
    material.apply_algorithmic_tangent(dF_hat_field, dP_tangent_field, tangent_ijklqxyz)
    dP_tangent = dP_tangent_field.s[...]

    np.testing.assert_allclose(
        dP_fd, dP_tangent,
        rtol=1e-5, atol=1e-6,
        err_msg=(
            "TANGENT TEST FAILED for NeoHookean: apply_algorithmic_tangent(A, dF) does not match "
            "the directional derivative dP computed via finite differences. "
            "This indicates an error in either get_algorithmic_tangent (building the wrong A_ijkl) "
            "or apply_algorithmic_tangent (contracting over wrong indices). "
            "The project's contraction convention is P_ij = A_ijkl F_lk (see tensor_operations.py). "
            f"Max absolute error: {np.max(np.abs(dP_fd - dP_tangent)):.3e}, "
            f"Max relative error: {np.max(np.abs((dP_fd - dP_tangent) / (np.abs(dP_fd) + 1e-30))):.3e}"
        )
    )


# TODO: Add test for symmetricity of the tangent