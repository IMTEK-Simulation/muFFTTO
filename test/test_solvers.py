import pytest

import numpy as np
import scipy as sc
import time

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import topology_optimization
from muFFTTO import microstructure_library


params = np.array([1, 100])
x0 = np.array([3.0, 4.0])

def gradf(x):
    [a, b] = params
    return np.array([-2 * (a - x[0]) - 4 * b * (x[1] - x[0] * x[0]) * x[0], 2 * b * (x[1] - x[0] * x[0])])


def f(x):
    [a, b] = params
    return (np.power((a - x[0]), 2) + b * np.power((x[1] - x[0] * x[0]), 2))






def test_cg_solver_small_matrix( ):
    print('========= TEST  Optimizing the Rosenbrock function =========')
    print('xmim=', np.array([1.0, 1.0]))
    print('fmim=', 0.0)
    x_expected = np.array([1.0, 1.0])  # Hypothetical expected solution
    [x_computed, fmin, Niter] = solvers.optimize_fire(x0, f, gradf, params=params, atol=1e-6)
    np.testing.assert_allclose(x_computed, x_expected, rtol=1e-4)