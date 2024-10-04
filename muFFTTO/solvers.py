import numpy as np

from NuMPI.Tools import Reduction
from mpi4py import MPI


def PCG(Afun, B, x0, P, steps=int(500), toler=1e-6):
    # print('I am in PCG')
    """
    Conjugate gradients solver.

    Parameters
    ----------
    Afun : Matrix, LinOper, or numpy.array of shape (n, n)
        it stores the matrix data of linear system and provides a matrix by
        vector multiplication
    B : VecTri or numpy.array of shape (n,)
        it stores a right-hand side of linear system
    x0 : VecTri or numpy.array of shape (n,)
        initial approximation of solution of linear system
    """
    if x0 is None:
        x0 = np.zeros(B.shape)

    norms = dict()
    norms['residual_rr'] = []
    norms['residual_rz'] = []
    ##
    k = 0
    x_k = np.copy(x0)
    ##
    Ax = Afun(x0)

    r_0 = B - Ax
    z_0 = P(r_0)

    scalar_product = lambda a, b: np.sum(a * b)

    r_0z_0 = scalar_product_mpi(r_0, z_0)

    norms['residual_rr'].append(scalar_product(r_0, r_0))
    norms['residual_rz'].append(r_0z_0)
    p_0 = np.copy(z_0)

    for k in np.arange(1, steps):
        Ap_0 = Afun(p_0)

        alpha = float(r_0z_0 / scalar_product_mpi(p_0, Ap_0))
        x_k = x_k + alpha * p_0

        # if xCG.val.mean() > 1e-10:
        #     print('iteration left zero-mean space {} \n {}'.format(xCG.name, xCG.val.mean()))

        r_0 = r_0 - alpha * Ap_0

        z_0 = P(r_0)

        r_1z_1 = scalar_product_mpi(r_0, z_0)

        norms['residual_rr'].append(scalar_product_mpi(r_0, r_0))
        norms['residual_rz'].append(r_1z_1)

        if (np.sqrt(r_1z_1)) < toler:  # TODO[Solver] check out stopping criteria
            break

        beta = r_1z_1 / r_0z_0
        p_0 = z_0 + beta * p_0

        r_0z_0 = r_1z_1

    return x_k, norms


def scalar_product_mpi(a, b):
    return Reduction(MPI.COMM_WORLD).sum(a * b)
