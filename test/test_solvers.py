import pytest

import numpy as np
import scipy as sc
import time

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import topology_optimization
from muFFTTO import microstructure_library
from muFFTTO import method_of_moving_asymptotes_

params = np.array([1, 100])
x0 = np.array([1.0, 5.0])


def gradf(x):
    [a, b] = params
    return np.array([-2 * (a - x[0]) - 4 * b * (x[1] - x[0] * x[0]) * x[0], 2 * b * (x[1] - x[0] * x[0])])


def f(x):
    [a, b] = params
    return np.power((a - x[0]), 2) + b * np.power((x[1] - x[0] * x[0]), 2)


def gradf_mma(x):
    return np.array([2 * x[0] + 2, 2 * x[1] - 2])


def f_mma(x):
    return np.power(x[0] + 1, 2) + np.power(x[1] - 1, 2)


def test_optimizer():
    print('========= TEST  Optimizing the Rosenbrock function =========')
    print('xmim=', np.array([1.0, 1.0]))
    print('fmim=', 0.0)
    x_expected = np.array([-1.0, 1.0])  # np.array([1.0, 1.0])  # Hypothetical expected solution

    nb_constr = 1
    parameters = {'nb_unknown': np.size(x0),
                  'nb_constrains': nb_constr,
                  'xmin': -5.0,
                  'xmax': 5.0,
                  'move': 0.5,
                  'maxoutit': 1000,
                  'a0': 1,
                  'ai': np.zeros((nb_constr, 1)),
                  'ci': 100000 * np.ones((nb_constr, 1)),
                  'di': np.ones((nb_constr, 1)),
                  }


    [x_computed_mma, fmin_mma, Niter_mma] = method_of_moving_asymptotes_.mma_loop(x0=x0, f=f , gradf=gradf,
                                                                                  atol=1e-6, **parameters)
    print(x_computed_mma[:,0])
    print(fmin_mma)
    print(Niter_mma)


    [x_fire_1, fmin_1, Niter_1] = solvers.optimize_fire(x0, f , gradf , params=params, atol=1e-10)

    print(x_fire_1)

    np.testing.assert_allclose(x_fire_1, x_computed_mma[:,0], rtol=1e-4)
    # np.testing.assert_allclose(x_computed_mma, x_expected, rtol=1e-4)

# '''
# 'a0': 100,
# 'ai': np.zeros((nb_constr, 1)),
# 'ci': 100000*np.ones((nb_constr, 1)),
# 'di':  np.ones((nb_constr, 1)),
# '''
