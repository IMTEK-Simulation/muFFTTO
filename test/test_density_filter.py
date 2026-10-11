"""
Tests of :mod:`muFFTTO.density_filter` (Helmholtz density filter and Heaviside projection).

Serial tests check the filter against its definition (Fourier modes, a direct sparse solve of
(I + r^2 L) rho_tilde = rho), its properties (constants, mean, symmetry, bounds) in 2D and 3D, the projection and
its derivative, and the chain rule of :class:`FilterAndProjection` by finite differences.
``test_mpi_matches_serial`` runs the filter with ``mpirun -n 2`` (skipped without mpirun) and compares it with the
serial result.
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from muFFTTO import domain
from muFFTTO import density_filter


def make_discretization(nb_pixels, domain_size, element_type='linear_triangles'):
    cell = domain.PeriodicUnitCell(domain_size=domain_size, problem_type='elasticity')
    return domain.Discretization(cell=cell, nb_of_pixels_global=nb_pixels, discretization_type='finite_element',
                                 element_type=element_type)


def nodal_coordinates(discretization):
    """Coordinates of the nodes, shape [d, *nb_pixels] (fft.coords are fractional)."""
    dim = len(discretization.nb_of_pixels_global)
    return np.asarray(discretization.fft.coords) * np.asarray(discretization.domain_size).reshape((dim,) + (1,) * dim)


def periodic_laplacian(nb_pixels, pixel_size):
    """Sparse periodic second-order FD Laplacian (5-point in 2D), C order of the nodes."""
    operator = None
    for d, (n, h) in enumerate(zip(nb_pixels, pixel_size)):
        one_d = sparse.diags([2.0 * np.ones(n), -np.ones(n - 1), -np.ones(n - 1)], [0, 1, -1], format='lil')
        one_d[0, n - 1] = one_d[n - 1, 0] = -1.0
        one_d = one_d.tocsr() / h ** 2
        factors = [sparse.identity(m, format='csr') for m in nb_pixels]
        factors[d] = one_d
        term = factors[0]
        for factor in factors[1:]:
            term = sparse.kron(term, factor, format='csr')
        operator = term if operator is None else operator + term
    return operator


# ----------------------------------------------------------------------------------------------------------------
# Helmholtz filter
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('element_type', ['linear_triangles', 'bilinear_rectangle'])
def test_constants_are_reproduced(element_type):
    discretization = make_discretization((12, 8), [1.0, 1.5], element_type)
    filter_ = density_filter.HelmholtzFilter(discretization, radius=0.3)
    field = np.full((1, 1, 12, 8), 0.37)
    np.testing.assert_allclose(filter_.apply(field), field, rtol=0, atol=1e-14)


def test_mean_is_preserved_and_field_is_smoothed():
    discretization = make_discretization((16, 12), [1.0, 0.7])
    rho = np.random.default_rng(0).random((1, 1, 16, 12))
    rho_tilde = density_filter.HelmholtzFilter(discretization, radius=0.2).apply(rho)
    assert np.mean(rho_tilde) == pytest.approx(np.mean(rho), abs=1e-14)
    assert np.var(rho_tilde) < 0.5 * np.var(rho)


def test_fourier_mode_is_damped_by_the_symbol():
    """cos(2 pi (m x / Lx + n y / Ly)) is an eigenvector: factor 1 / (1 + r^2 lambda), lambda of the FD Laplacian."""
    nb, size = (24, 18), [2.0, 1.5]
    discretization = make_discretization(nb, size)
    x = nodal_coordinates(discretization)
    radius = 0.25
    r = density_filter.helmholtz_filter_radius_to_length(radius)
    h = np.asarray(size) / np.asarray(nb)
    for m, n in [(1, 0), (0, 2), (3, 1), (12, 9)]:      # (12, 9): Nyquist in both directions
        mode = np.cos(2 * np.pi * (m * x[0] / size[0] + n * x[1] / size[1]))[None, None]
        eigenvalue = 4 * np.sin(np.pi * m / nb[0]) ** 2 / h[0] ** 2 + 4 * np.sin(np.pi * n / nb[1]) ** 2 / h[1] ** 2
        filtered = density_filter.HelmholtzFilter(discretization, radius).apply(mode)
        np.testing.assert_allclose(filtered, mode / (1 + r ** 2 * eigenvalue), rtol=0, atol=1e-12)


def test_matches_direct_sparse_solve():
    """rho_tilde solves (I + r^2 L) rho_tilde = rho with the periodic 5-point Laplacian."""
    nb, size = (10, 7), [1.0, 0.9]
    discretization = make_discretization(nb, size)
    rho = np.random.default_rng(1).random(nb)
    radius = 0.3
    r = density_filter.helmholtz_filter_radius_to_length(radius)
    operator = sparse.identity(rho.size, format='csr') + r ** 2 * periodic_laplacian(nb, np.asarray(size) / nb)
    expected = sparse_linalg.spsolve(operator.tocsc(), rho.ravel()).reshape(nb)
    filtered = density_filter.HelmholtzFilter(discretization, radius).apply(rho[None, None])
    np.testing.assert_allclose(filtered[0, 0], expected, rtol=0, atol=1e-12)


def test_filter_is_symmetric():
    discretization = make_discretization((14, 9), [1.0, 1.0])
    rng = np.random.default_rng(2)
    a, b = rng.standard_normal((2, 1, 1, 14, 9))
    filter_ = density_filter.HelmholtzFilter(discretization, radius=0.2)
    assert np.sum(filter_.apply(a) * b) == pytest.approx(np.sum(a * filter_.apply(b)), rel=1e-13)


def test_bounds_are_preserved():
    """(I + r^2 L)^-1 has non-negative entries and unit row sums: 0 <= rho <= 1 -> 0 <= rho_tilde <= 1."""
    discretization = make_discretization((20, 20), [1.0, 1.0])
    rho = (np.random.default_rng(3).random((1, 1, 20, 20)) > 0.5).astype(float)
    filtered = density_filter.HelmholtzFilter(discretization, radius=0.15).apply(rho)
    assert filtered.min() >= -1e-14 and filtered.max() <= 1 + 1e-14


def test_zero_radius_is_the_identity_and_flat_input_works():
    discretization = make_discretization((8, 6), [1.0, 1.0])
    rho = np.random.default_rng(4).random(48)
    np.testing.assert_allclose(density_filter.HelmholtzFilter(discretization, 0.0).apply(rho), rho, atol=1e-14)
    assert density_filter.HelmholtzFilter(discretization, 0.2).apply(rho).shape == (48,)


def test_muGrid_field_input():
    discretization = make_discretization((8, 6), [1.0, 1.0])
    field = discretization.get_scalar_field(name='rho_test_input')
    field.s[...] = np.random.default_rng(5).random(field.s.shape)
    filter_ = density_filter.HelmholtzFilter(discretization, 0.2)
    np.testing.assert_allclose(filter_.apply(field), filter_.apply(field.s.copy()), atol=1e-15)


def test_3d_fourier_mode():
    nb, size = (8, 6, 10), [1.0, 0.75, 1.25]
    cell = domain.PeriodicUnitCell(domain_size=size, problem_type='elasticity')
    discretization = domain.Discretization(cell=cell, nb_of_pixels_global=nb, discretization_type='finite_element',
                                           element_type='trilinear_hexahedron')
    x = nodal_coordinates(discretization)
    radius = 0.3
    r = density_filter.helmholtz_filter_radius_to_length(radius)
    h = np.asarray(size) / np.asarray(nb)
    m = (1, 2, 3)
    mode = np.sin(2 * np.pi * sum(m[d] * x[d] / size[d] for d in range(3)))[None, None]
    eigenvalue = sum(4 * np.sin(np.pi * m[d] / nb[d]) ** 2 / h[d] ** 2 for d in range(3))
    filtered = density_filter.HelmholtzFilter(discretization, radius).apply(mode)
    np.testing.assert_allclose(filtered, mode / (1 + r ** 2 * eigenvalue), rtol=0, atol=1e-12)


# ----------------------------------------------------------------------------------------------------------------
# Heaviside projection
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('beta', [0.5, 1.0, 8.0, 64.0])
@pytest.mark.parametrize('threshold', [0.3, 0.5, 0.7])
def test_projection_end_points_and_monotonicity(beta, threshold):
    x = np.linspace(0, 1, 201)
    projected = density_filter.heaviside_projection(x, beta, threshold)
    assert projected[0] == pytest.approx(0.0, abs=1e-14)
    assert projected[-1] == pytest.approx(1.0, abs=1e-14)
    assert np.all(np.diff(projected) >= -1e-15)                            # saturates to 0 / 1 for large beta
    derivative = density_filter.heaviside_projection_derivative(x, beta, threshold)
    assert np.all(derivative >= 0)                                          # underflows to 0 far from threshold
    assert np.all(derivative[(x > threshold - 0.1) & (x < threshold + 0.1)] > 0)


def test_projection_limits():
    x = np.linspace(0, 1, 101)
    np.testing.assert_allclose(density_filter.heaviside_projection(x, 0.0), x)
    np.testing.assert_allclose(density_filter.heaviside_projection(x, 1e-6, 0.4), x, atol=1e-6)
    steep = density_filter.heaviside_projection(x, 1000.0, 0.4)
    np.testing.assert_allclose(steep[x < 0.39], 0.0, atol=1e-12)
    np.testing.assert_allclose(steep[x > 0.41], 1.0, atol=1e-12)
    assert density_filter.heaviside_projection(0.5, 8.0, 0.5) == pytest.approx(0.5)


@pytest.mark.parametrize('beta', [0.0, 1.0, 16.0])
def test_projection_derivative(beta):
    x = np.linspace(0.01, 0.99, 37)
    h = 1e-6
    finite_difference = (density_filter.heaviside_projection(x + h, beta, 0.4)
                         - density_filter.heaviside_projection(x - h, beta, 0.4)) / (2 * h)
    np.testing.assert_allclose(density_filter.heaviside_projection_derivative(x, beta, 0.4), finite_difference,
                               rtol=1e-7, atol=1e-7)


def test_measure_of_non_discreteness():
    assert density_filter.measure_of_non_discreteness(np.array([0.0, 1.0, 1.0, 0.0])) == 0.0
    assert density_filter.measure_of_non_discreteness(np.full(10, 0.5)) == pytest.approx(1.0)


# ----------------------------------------------------------------------------------------------------------------
# filter + projection: chain rule
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('radius, beta', [(0.15, 1.0), (0.15, 8.0), (0.0, 4.0), (0.2, 0.0)])
def test_chain_rule_by_finite_differences(radius, beta):
    """d/drho of f(rho_bar) = sum w rho_bar^3 via backward() vs central differences along random directions."""
    discretization = make_discretization((12, 10), [1.0, 0.8])
    density = density_filter.FilterAndProjection(discretization, radius=radius, beta=beta, threshold=0.45)
    rng = np.random.default_rng(6)
    rho = rng.random((1, 1, 12, 10))
    weights = rng.random((1, 1, 12, 10))

    def f(x):
        return np.sum(weights * density.forward(x) ** 3)

    rho_bar = density.forward(rho)
    gradient = density.backward(3 * weights * rho_bar ** 2)
    for _ in range(3):
        direction = rng.standard_normal(rho.shape)
        errors = []
        for h in [1e-4, 1e-5]:
            finite_difference = (f(rho + h * direction) - f(rho - h * direction)) / (2 * h)
            # relative to |g| |d| (a random direction can be nearly orthogonal to the gradient)
            errors.append(abs(finite_difference - np.sum(gradient * direction))
                          / (np.linalg.norm(gradient) * np.linalg.norm(direction)))
        assert errors[-1] < 1e-7                                       # truncation error ~ beta^3 h^2
        assert errors[1] < errors[0] / 30 or errors[1] < 1e-9          # O(h^2) until round-off


def test_backward_needs_forward_and_threshold_is_checked():
    discretization = make_discretization((6, 6), [1.0, 1.0])
    with pytest.raises(RuntimeError):
        density_filter.FilterAndProjection(discretization, 0.2).backward(np.ones((1, 1, 6, 6)))
    with pytest.raises(ValueError):
        density_filter.FilterAndProjection(discretization, 0.2, threshold=1.0)


# ----------------------------------------------------------------------------------------------------------------
# MPI: two ranks give the serial result
# ----------------------------------------------------------------------------------------------------------------
MPI_SCRIPT = r'''
import sys, numpy as np
sys.path.insert(0, sys.argv[1])
from mpi4py import MPI
from muFFTTO import domain, density_filter
nb = (16, 12)
cell = domain.PeriodicUnitCell(domain_size=[1.0, 0.75], problem_type='elasticity')
discretization = domain.Discretization(cell=cell, nb_of_pixels_global=nb, discretization_type='finite_element',
                                       element_type='linear_triangles')
rho_global = np.random.default_rng(7).random(nb)
loc = tuple(slice(o, o + n) for o, n in zip(discretization.fft.subdomain_locations,
                                           discretization.fft.nb_subdomain_grid_pts))
density = density_filter.FilterAndProjection(discretization, radius=0.2, beta=4.0)
rho_bar = density.forward(rho_global[loc][None, None])
gradient = density.backward(np.ones_like(rho_bar))
result = np.zeros((2,) + nb)
result[(0,) + loc] = rho_bar[0, 0]
result[(1,) + loc] = gradient[0, 0]
result = MPI.COMM_WORLD.allreduce(result)
if MPI.COMM_WORLD.rank == 0:
    np.save(sys.argv[2], result)
'''


def test_mpi_matches_serial(tmp_path):
    mpirun = shutil.which('mpirun') or shutil.which('mpiexec')
    if mpirun is None:
        pytest.skip('mpirun not available')
    script = tmp_path / 'filter_mpi.py'
    script.write_text(MPI_SCRIPT)
    # importing mpi4py initialised MPI in this process; its environment must not leak into the child mpirun
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(('OMPI_', 'PMIX_', 'PMI_'))}
    results = []
    for ranks in (1, 2):
        out = tmp_path / f'result_{ranks}.npy'
        subprocess.run([mpirun, '-n', str(ranks), sys.executable, str(script), parent_dir, str(out)], check=True,
                       timeout=300, capture_output=True, env=environment)
        results.append(np.load(out))
    np.testing.assert_allclose(results[1], results[0], rtol=0, atol=1e-13)

    # and both equal the in-process serial computation
    discretization = make_discretization((16, 12), [1.0, 0.75])
    density = density_filter.FilterAndProjection(discretization, radius=0.2, beta=4.0)
    rho_bar = density.forward(np.random.default_rng(7).random((16, 12))[None, None])
    np.testing.assert_allclose(results[0][0], rho_bar[0, 0], rtol=0, atol=1e-13)


# ----------------------------------------------------------------------------------------------------------------
# robust formulation
# ----------------------------------------------------------------------------------------------------------------
def test_robust_designs_are_ordered_and_match_single_projections():
    """eroded <= intermediate <= dilated pointwise; each equals FilterAndProjection with that threshold."""
    discretization = make_discretization((16, 12), [1.0, 0.75])
    rho = np.random.default_rng(8).random((1, 1, 16, 12))
    robust = density_filter.RobustFilterAndProjection(discretization, radius=0.15, beta=6.0)
    eroded, intermediate, dilated = robust.forward(rho)
    assert np.all(eroded <= intermediate + 1e-15) and np.all(intermediate <= dilated + 1e-15)
    for threshold, design in zip((0.75, 0.5, 0.25), (eroded, intermediate, dilated)):
        single = density_filter.FilterAndProjection(discretization, radius=0.15, beta=6.0, threshold=threshold)
        np.testing.assert_allclose(design, single.forward(rho), rtol=0, atol=1e-14)


@pytest.mark.parametrize('beta', [1.0, 8.0])
def test_robust_chain_rule_by_finite_differences(beta):
    """Gradient of sum_i w_i f_i(rho_bar_i), f_i = sum c_i rho_bar_i^3, vs central differences."""
    discretization = make_discretization((12, 10), [1.0, 0.8])
    robust = density_filter.RobustFilterAndProjection(discretization, radius=0.15, beta=beta)
    rng = np.random.default_rng(9)
    rho = rng.random((1, 1, 12, 10))
    c = rng.random((3, 1, 1, 12, 10))
    weights = np.array([0.5, 1.0, 2.0])

    def f(x):
        return sum(w * np.sum(ci * d ** 3) for w, ci, d in zip(weights, c, robust.forward(x)))

    designs = robust.forward(rho)
    gradient = robust.backward([3 * ci * d ** 2 for ci, d in zip(c, designs)], weights=weights)
    for _ in range(3):
        direction = rng.standard_normal(rho.shape)
        errors = []
        for h in [1e-4, 1e-5]:
            finite_difference = (f(rho + h * direction) - f(rho - h * direction)) / (2 * h)
            # relative to |g| |d| (a random direction can be nearly orthogonal to the gradient)
            errors.append(abs(finite_difference - np.sum(gradient * direction))
                          / (np.linalg.norm(gradient) * np.linalg.norm(direction)))
        assert errors[-1] < 1e-7
        assert errors[1] < errors[0] / 30 or errors[1] < 1e-9


def test_robust_removes_thin_members():
    """A one-pixel bar survives the plain projection (threshold 0.5) of a weak filter but not the eroded design."""
    discretization = make_discretization((32, 32), [1.0, 1.0])
    rho = np.zeros((1, 1, 32, 32))
    rho[0, 0, 16, :] = 1.0                       # one-pixel-wide bar along y
    rho[0, 0, 4:12, :] = 1.0                     # eight-pixel-wide bar along y
    robust = density_filter.RobustFilterAndProjection(discretization, radius=3 / 32, beta=64.0)
    eroded, intermediate, dilated = robust.forward(rho)
    assert eroded[0, 0, 16].max() < 0.01         # the thin bar is gone in the eroded design
    assert eroded[0, 0, 8].min() > 0.99          # the thick bar survives
    assert dilated[0, 0, 16].min() > 0.5         # and the dilated design still sees the thin bar


def test_robust_argument_checks():
    discretization = make_discretization((6, 6), [1.0, 1.0])
    robust = density_filter.RobustFilterAndProjection(discretization, 0.2)
    with pytest.raises(RuntimeError):
        robust.backward([np.ones((1, 1, 6, 6))] * 3)
    robust.forward(np.ones((1, 1, 6, 6)))
    with pytest.raises(ValueError):
        robust.backward([np.ones((1, 1, 6, 6))] * 2)
    with pytest.raises(ValueError):
        density_filter.RobustFilterAndProjection(discretization, 0.2, thresholds=(0.0, 0.5))


@pytest.mark.parametrize('beta', [1.0, 8.0])
def test_length_scale_constraints_gradient_by_finite_differences(beta):
    """Geometric length scale constraints (Zhou et al. 2015): values >= 0 and gradients by central differences."""
    rng = np.random.default_rng(5)
    n = 24
    X, Y = np.meshgrid(np.arange(n) / n, np.arange(n) / n, indexing='ij')
    x = np.clip(0.5 + 0.45 * np.sin(2 * np.pi * X) * np.cos(4 * np.pi * Y) + 0.05 * rng.standard_normal((n, n)),
                0.01, 0.99)[np.newaxis, np.newaxis]
    g_s, g_v, dg_s, dg_v = density_filter.length_scale_constraints(x, radius_pixels=2.0, beta=beta)
    assert g_s > 0 and g_v > 0
    d = rng.standard_normal(x.shape)
    h = 1e-6
    for index, dg in ((0, dg_s), (1, dg_v)):
        plus = density_filter.length_scale_constraints(x + h * d, radius_pixels=2.0, beta=beta)[index]
        minus = density_filter.length_scale_constraints(x - h * d, radius_pixels=2.0, beta=beta)[index]
        assert (plus - minus) / (2 * h) == pytest.approx(np.sum(dg * d), rel=1e-6)


def test_length_scale_constraints_vanish_for_thick_members():
    """A thick bar (filtered: 0/1 core, smooth edges) satisfies both constraints; a one-pixel bar violates g_solid."""
    n = 40
    thick = np.zeros((n, n))
    thick[:, 10:30] = 1.0
    thin = np.zeros((n, n))
    thin[:, 20] = 1.0
    discretization = domain.Discretization(cell=domain.PeriodicUnitCell(domain_size=[1.0, 1.0],
                                                                         problem_type='conductivity'),
                                           nb_of_pixels_global=(n, n), discretization_type='finite_element',
                                           element_type='linear_triangles')
    helmholtz = density_filter.HelmholtzFilter(discretization, radius=3.0 / n)
    g_thick = density_filter.length_scale_constraints(helmholtz.apply(thick), radius_pixels=3.0, beta=16.0)
    g_thin = density_filter.length_scale_constraints(helmholtz.apply(thin), radius_pixels=3.0, beta=1.0)
    assert g_thick[0] < 1e-6 and g_thick[1] < 1e-6
    assert g_thin[0] > 1e-4
