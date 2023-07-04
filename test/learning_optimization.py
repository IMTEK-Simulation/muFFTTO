from scipy import optimize
import numpy as np

# Matrix to be used in the function definitions
A = np.array([[1.,1.],[1.,3.]])

# Objectve function to be minimized: f = x^T A x
def f(x):
    return np.dot(x.T,np.dot(A,x))

# gradient of the objective function, df = 2*A*x
def fp(x):
    return 2*np.dot(A,x)

# Initial value of x
x0 = np.array([1.,2.])

# Try with BFGS
xopt = optimize.minimize(f,x0,method='bfgs',jac=fp,options={'disp':1})






