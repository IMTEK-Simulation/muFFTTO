import numpy as np
import scipy.sparse.linalg as sp
import itertools

# ----------------------------------- GRID ------------------------------------
def solve_sparse(A, b, M=None):
    num_iters = 0

    def callback(xk):
        nonlocal num_iters
        num_iters += 1

    x, status = sp.cg(A, b, M=M, tol=1e-6,maxiter=1000, callback=callback)
    return x, status, num_iters

# PARAMETERS ##############################################################
ndim = 2  # number of dimensions (works for 2D and 3D)
N = ndim * (1023,)  # number of voxels (assumed equal for all directions)

# auxiliary values
prodN = np.prod(np.array(N))  # number of grid points
ndof = ndim * prodN  # number of degrees-of-freedom
vec_shape = (ndim,) + N  # shape of the vector for storing DOFs

# PROBLEM DEFINITION ######################################################
mat_contrast = 1.
phase = mat_contrast * np.ones(N)
inc_contrast = 10.
# phase[10:30,10:30]=phase[10:30,10:30]*inc_contrast
phase[phase.shape[0] * 1 // 4:phase.shape[0] * 3 // 4,
phase.shape[1] * 1 // 4:phase.shape[1] * 3 // 4] *= inc_contrast

A = np.einsum('ij,...->ij...', np.eye(ndim), phase)  # material coefficients

E = np.zeros(vec_shape);
E[0] = 1.  # set macroscopic loading

# PROJECTION IN FOURIER SPACE #############################################
Ghat = np.zeros((ndim, ndim) + N)  # zero initialize
freq = [np.arange(-(N[ii] - 1) / 2., +(N[ii] + 1) / 2.) for ii in range(ndim)]
for i, j in itertools.product(range(ndim), repeat=2):
    for ind in itertools.product(*[range(n) for n in N]):
        q = np.empty(ndim)
        for ii in range(ndim):
            q[ii] = freq[ii][ind[ii]]  # frequency vector
        if not q.dot(q) == 0:  # zero freq. -> mean
            Ghat[i, j][ind] = -(q[i] * q[j]) / (q.dot(q))

# OPERATORS ###############################################################
dot21 = lambda A, v: np.einsum('ij...,j...  ->i...', A, v)

fft = lambda V: np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(V), N))
ifft = lambda V: np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(V), N))
G_fun = lambda V: np.real(ifft(dot21(Ghat, fft(V)))).reshape(-1)
A_fun = lambda v: dot21(A, v.reshape(vec_shape))
GA_fun = lambda v: G_fun(A_fun(v))

# CONJUGATE GRADIENT SOLVER ###############################################
b = -GA_fun(E)  # right-hand side
#e, _ = sp.cg(A=sp.LinearOperator(shape=(ndof, ndof), matvec=GA_fun, dtype='float'), b=b)
e,  status, num_iters = solve_sparse(A=sp.LinearOperator(shape=(ndof, ndof), matvec=GA_fun, dtype='float'), b=b)
print('Number of steps = {}'.format(num_iters))

aux = e + E.reshape(-1)

print('homogenised properties A11 = {}'.format(np.inner(A_fun(aux).reshape(-1), aux) / prodN))
print('END')
