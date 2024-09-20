from scipy import optimize
import numpy as np

import NuMPI
from NuMPI import Optimization
# Matrix to be used in the function definitions
A = np.array([[1., 1.], [1., 3.]])


# Objectve function to be minimized: f = x^T A x
def f(x):
    return np.dot(x.T, np.dot(A, x)+35*x)


# gradient of the objective function, df = 2*A*x
def fp(x):
    return 2 * np.dot(A, x)


# Initial value of x
x0 = np.array([1., 2.])

# Try with BFGS
xopt = optimize.minimize(f, x0, method='bfgs', jac=fp, options={'disp': 1})

result =  Optimization.l_bfgs(fun=f, x= x0,jac=fp)

# l-bfgs-b algorithm local optimization of a convex function
from scipy.optimize import minimize
from numpy.random import rand


# objective function
def objective(x):
    return x[0] ** 2.0 + x[1] ** 2.0


# define range for input
r_min, r_max = -5.0, 5.0
# define the starting point as a random sample from the domain
pt = r_min + rand(2) * (r_max - r_min)
# perform the l-bfgs-b algorithm search
result = minimize(objective, pt, method='L-BFGS-B')
# summarize the result
print('Status : %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])
# evaluate solution
solution = result['x']
evaluation = objective(solution)
print('Solution: f(%s) = %.5f' % (solution, evaluation))
