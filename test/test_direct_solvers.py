"""muFFTTO.direct_solvers: the inertia (number of negative eigenvalues) and the solve of every available backend."""
import importlib.util

import numpy as np
import pytest
import scipy.sparse as sparse

from muFFTTO import direct_solvers


def available(backend):
    module = {'scipy': 'scipy', 'mumps': 'mumps', 'cudss': 'nvmath'}[backend]
    if importlib.util.find_spec(module) is None:
        return False
    if backend == 'cudss':
        try:
            import cupy
            return cupy.cuda.runtime.getDeviceCount() > 0
        except Exception:
            return False
    return True


BACKENDS = [pytest.param(b, marks=pytest.mark.skipif(not available(b), reason=f'{b} not installed'))
            for b in ('scipy', 'mumps', 'cudss')]


def laplacian_2d(n):
    """Periodic-free 2D 5-point Laplacian (n x n grid, Dirichlet): symmetric positive definite, known spectrum."""
    T = sparse.diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1])
    return (sparse.kron(T, sparse.eye(n)) + sparse.kron(sparse.eye(n), T)).tocsr()


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('shift', [0.0, 0.3, 1.1])
def test_inertia_and_solve(backend, shift):
    """K = Laplacian - shift I: positive definite (shift 0) and indefinite with 2 / 9 negative eigenvalues."""
    K = (laplacian_2d(12) - shift * sparse.eye(144)).tocsr()
    expected = int(np.count_nonzero(np.linalg.eigvalsh(K.toarray()) < 0))
    solver = direct_solvers.symmetric_solver(backend)
    negative = solver.factorise(K)
    if backend == 'scipy' and expected > 0:
        assert negative in (expected, -1)                       # without pivoting: exact, or 'not positive definite'
    else:
        assert negative == expected
    if negative >= 0:
        b = np.random.default_rng(0).standard_normal(144)
        x = solver.solve(b)
        assert np.abs(K @ x - b).max() <= 1e-10 * np.abs(b).max()


@pytest.mark.parametrize('backend', BACKENDS)
def test_refactorisation_same_pattern(backend):
    """A second matrix with the same pattern (analysis reused) gives its own inertia and solution."""
    solver = direct_solvers.symmetric_solver(backend)
    K0 = laplacian_2d(10)
    assert solver.factorise(K0) == 0
    K1 = (K0 - 0.5 * sparse.eye(100)).tocsr()
    expected = int(np.count_nonzero(np.linalg.eigvalsh(K1.toarray()) < 0))
    negative = solver.factorise(K1)
    assert negative == expected or (backend == 'scipy' and negative == -1)
    K2 = (K0 + 0.5 * sparse.eye(100)).tocsr()
    assert solver.factorise(K2) == 0
    b = np.ones(100)
    np.testing.assert_allclose(K2 @ solver.solve(b), b, atol=1e-11)


def test_auto_backend():
    assert direct_solvers.symmetric_solver('auto').name in ('scipy', 'mumps')
