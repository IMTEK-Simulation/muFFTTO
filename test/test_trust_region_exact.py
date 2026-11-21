"""muFFTTO.solvers_nonlinear.trust_region_exact: convergence to a local minimum, negative curvature, the hard case,
fixed unknowns, every available direct solver backend."""
import importlib.util

import numpy as np
import pytest
import scipy.sparse as sparse
from scipy.optimize import rosen, rosen_der, rosen_hess

from muFFTTO import solvers_nonlinear

BACKENDS = ['scipy'] + [pytest.param('mumps', marks=pytest.mark.skipif(importlib.util.find_spec('mumps') is None,
                                                                        reason='mumps not installed'))]


@pytest.mark.parametrize('backend', BACKENDS)
def test_rosenbrock(backend):
    """Nonconvex, indefinite Hessian away from the minima: converges to a local minimum (the chained Rosenbrock in
    6D has two: x = 1 and one near x_0 = -1), stable (0 negative eigenvalues), with a lower energy."""
    x0 = np.array([-1.2, 1.0, -1.2, 1.0, 0.5, -0.5])
    result = solvers_nonlinear.trust_region_exact(lambda x: (rosen(x), rosen_der(x)), x0,
                                                  lambda x: sparse.csr_matrix(rosen_hess(x)), gtol=1e-10,
                                                  maxiter=200, delta0=1.0, delta_max=10.0, scaling=np.ones(6),
                                                  solver=backend)
    assert result.success, result.message
    assert np.abs(rosen_der(result.x)).max() <= 1e-10
    assert np.all(np.linalg.eigvalsh(rosen_hess(result.x)) > 0) and result.negative_eigenvalues == 0
    assert result.fun < rosen(x0)


def quartic(A, b):
    """f = x.A x / 2 + sum x^4 / 4 - b.x: bounded below, A indefinite -> nonconvex."""
    def fun(x):
        return 0.5 * x @ (A @ x) + 0.25 * np.sum(x ** 4) - b @ x, A @ x + x ** 3 - b

    def hess(x):
        return sparse.csr_matrix(A + np.diag(3 * x ** 2))
    return fun, hess


@pytest.mark.parametrize('backend', BACKENDS)
def test_hard_case_reaches_a_stable_minimum(backend):
    """A = diag(-2, 1, 3, ...), b orthogonal to the negative eigenvector e_0: at x0 = 0 the Hessian is indefinite
    and g has no component along e_0 (hard case). The method must leave the saddle direction and end at a local
    minimum (positive definite Hessian), with a lower energy than the start."""
    n = 6
    A = np.diag([-2.0, 1.0, 3.0, 1.5, 2.0, 4.0])
    b = np.array([0.0, 0.3, -0.2, 0.1, 0.4, -0.1])
    fun, hess = quartic(A, b)
    result = solvers_nonlinear.trust_region_exact(fun, np.zeros(n), hess, gtol=1e-10, maxiter=100, delta0=0.5,
                                                  delta_max=5.0, scaling=np.ones(n), solver=backend)
    assert result.success, result.message
    assert result.negative_eigenvalues == 0
    assert np.all(np.linalg.eigvalsh(hess(result.x).toarray()) > 0)
    assert result.fun < fun(np.zeros(n))[0]
    assert abs(abs(result.x[0]) - np.sqrt(2.0)) < 1e-6           # x_0 = +-sqrt(2): out of the saddle


def test_fixed_unknowns_stay():
    n = 6
    A = sparse.diags([-np.ones(n - 1), 2.5 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1]).toarray()
    fun, hess = quartic(A, np.linspace(-1, 1, n))
    x0 = np.full(n, 0.3)
    result = solvers_nonlinear.trust_region_exact(fun, x0, hess, gtol=1e-10, fixed=[0, 3], scaling=np.ones(n))
    assert result.success
    assert result.x[0] == 0.3 and result.x[3] == 0.3
    g = fun(result.x)[1]
    assert np.abs(np.delete(g, [0, 3])).max() <= 1e-10


def test_inadmissible_points_are_rejected():
    """f = inf outside x > -0.5 (like an inverted element): the radius shrinks, the solution is still found."""
    def fun(x):
        if np.any(x <= -0.5):
            return np.inf, np.zeros_like(x)
        return np.sum((x + 0.4) ** 2 - np.log(x + 0.5)), 2 * (x + 0.4) - 1 / (x + 0.5)

    def hess(x):
        return sparse.diags(2 + 1 / (x + 0.5) ** 2).tocsr()
    result = solvers_nonlinear.trust_region_exact(fun, np.full(3, 2.0), hess, gtol=1e-10, delta0=10.0,
                                                  delta_max=10.0, scaling=np.ones(3))
    assert result.success
    np.testing.assert_allclose(result.x, (-0.9 + np.sqrt(0.01 + 2)) / 2 - 0.0, atol=1e-8)
