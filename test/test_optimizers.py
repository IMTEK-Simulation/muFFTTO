"""
Tests of the optimizers of :mod:`muFFTTO.solvers_nonlinear`: :func:`mma_bounded` and the common front end
:func:`minimize`, on problems with known solutions.

* separable quadratic with the minimum partly outside the box -> clip(c, 0, 1);
* the same with a volume constraint <a, x> <= V / = V -> Euclidean projection clip(c - mu a, 0, 1);
* an ill-conditioned coupled quadratic (Laplacian-like) -> same minimum as NuMPI's L-BFGS (box only) and
  SciPy's SLSQP (with the volume constraint).
"""
import os
import sys

import numpy as np
import pytest

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from NuMPI.Optimization import LinearConstraint, l_bfgs_bounded
from scipy.optimize import minimize as scipy_minimize

from muFFTTO.solvers_nonlinear import minimize, mma_bounded


def separable_quadratic(c):
    return lambda x: (0.5 * np.sum((x - c) ** 2), x - c)


def laplacian_matrix(n):
    return 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1) + 1e-2 * np.eye(n)


def laplacian_quadratic(n, seed=0):
    A = laplacian_matrix(n)
    b = np.random.default_rng(seed).normal(size=n)
    return lambda x: (0.5 * x @ A @ x - b @ x, A @ x - b)


def test_box_only_quadratic():
    c = np.random.default_rng(0).uniform(-0.5, 1.5, 400)
    res = mma_bounded(separable_quadratic(c), np.full(400, 0.5), gtol=1e-8, maxiter=200)
    assert res.success
    np.testing.assert_allclose(res.x, np.clip(c, 0, 1), atol=1e-7)


@pytest.mark.parametrize('constraint_type, volume_fraction', [('ineq', 0.3), ('ineq', 0.9), ('eq', 0.3),
                                                              ('eq', 0.7)])
def test_volume_constrained_quadratic(constraint_type, volume_fraction):
    n = 400
    c = np.random.default_rng(1).uniform(-0.5, 1.5, n)
    a = np.full(n, 1.0 / n)  # <a, x> = mean(x)
    res = mma_bounded(separable_quadratic(c), np.full(n, 0.5), constraint_weights=a,
                      constraint_target=volume_fraction, constraint_type=constraint_type, gtol=1e-7,
                      maxiter=500)
    assert res.success, res.message
    if constraint_type == 'ineq' and np.mean(np.clip(c, 0, 1)) <= volume_fraction:
        expected = np.clip(c, 0, 1)  # constraint not active
    else:
        expected = LinearConstraint(a, volume_fraction).project(c, 0.0, 1.0)
    np.testing.assert_allclose(res.x, expected, atol=1e-5)
    if constraint_type == 'eq':
        assert abs(np.mean(res.x) - volume_fraction) < 1e-9
    else:
        assert np.mean(res.x) <= volume_fraction + 1e-9


def test_iterates_stay_feasible():
    n = 200
    fun = laplacian_quadratic(n)
    a = np.full(n, 1.0 / n)
    volumes = []
    mma_bounded(fun, np.full(n, 0.9), constraint_weights=a, constraint_target=0.4, maxiter=50,
                callback=lambda x: volumes.append(np.mean(x)))
    assert max(volumes) <= 0.4 + 1e-9


def test_laplacian_matches_lbfgs():
    n = 200
    fun = laplacian_quadratic(n)
    mma = mma_bounded(fun, np.full(n, 0.5), gtol=1e-6, maxiter=3000)
    lbfgs = l_bfgs_bounded(fun, np.full(n, 0.5), jac=True, bounds_lo=0., bounds_hi=1., gtol=1e-6, maxiter=3000)
    assert mma.success and lbfgs.success
    assert mma.fun == pytest.approx(lbfgs.fun, rel=1e-6)


@pytest.mark.parametrize('constraint_type', ['ineq', 'eq'])
def test_laplacian_constrained_matches_slsqp(constraint_type):
    # reference: SciPy SLSQP (NuMPI's l_bfgs_projected stalls on this problem at f = -51.5503)
    n = 200
    fun = laplacian_quadratic(n)
    a = np.full(n, 1.0 / n)
    mma = mma_bounded(fun, np.full(n, 0.5), constraint_weights=a, constraint_target=0.4,
                      constraint_type=constraint_type, gtol=1e-6, maxiter=3000)
    reference = scipy_minimize(fun, np.full(n, 0.5), jac=True, method="SLSQP", bounds=[(0, 1)] * n,
                         constraints=[{'type': constraint_type, 'fun': lambda x: 0.4 - a @ x if constraint_type == 'ineq'
                                       else a @ x - 0.4, 'jac': lambda x: -a if constraint_type == 'ineq' else a}],
                         options={'ftol': 1e-14, 'maxiter': 1000})
    assert mma.success, mma.message
    assert reference.success, reference.message
    assert mma.fun == pytest.approx(reference.fun, rel=1e-9)
    assert a @ mma.x <= 0.4 + 1e-9


# --- minimize: one front end for all methods ------------------------------------------------------------------------

@pytest.mark.parametrize('method', ['l_bfgs', 'mma', 'tr_newton'])
def test_minimize_box_only_same_minimum(method):
    n = 200
    fun = laplacian_quadratic(n)
    A = laplacian_matrix(n)
    reference = l_bfgs_bounded(fun, np.full(n, 0.5), jac=True, bounds_lo=0., bounds_hi=1., gtol=1e-8,
                               maxiter=5000)
    result = minimize(fun, np.full(n, 0.5), method=method, bounds=(0., 1.),
                      hessp=(lambda x, v: A @ v) if method == 'tr_newton' else None, gtol=1e-6, maxiter=3000)
    assert result.success, result.message
    assert result.method == method
    assert result.nfev > 0
    assert (result.nhessp > 0) == (method == 'tr_newton')
    assert result.fun == pytest.approx(reference.fun, rel=1e-6)


@pytest.mark.parametrize('method', ['l_bfgs', 'mma'])
def test_minimize_volume_constraint(method):
    n = 400
    c = np.random.default_rng(1).uniform(-0.5, 1.5, n)
    a = np.full(n, 1.0 / n)
    result = minimize(separable_quadratic(c), np.full(n, 0.5), method=method, bounds=(0., 1.),
                      constraint_weights=a, constraint_target=0.3, constraint_type='eq', gtol=1e-7, maxiter=500)
    assert result.success, result.message
    np.testing.assert_allclose(result.x, LinearConstraint(a, 0.3).project(c, 0.0, 1.0), atol=1e-5)


@pytest.mark.parametrize('method, kwargs', [
    ('tr_newton', {}),                                                     # no hessp
    ('tr_newton', {'constraint_weights': np.ones(4), 'constraint_target': 1., 'hessp': lambda x, v: v}),
    ('l_bfgs', {'constraint_weights': np.ones(4), 'constraint_target': 1., 'constraint_type': 'ineq'}),
    ('mma', {'bounds': None}),
    ('newton', {}),
])
def test_minimize_rejects_unsupported(method, kwargs):
    with pytest.raises(ValueError):
        minimize(separable_quadratic(np.zeros(4)), np.zeros(4), method=method, **kwargs)
