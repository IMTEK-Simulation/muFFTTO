"""
Preconditioned conjugate gradients with the energy-error stopping test, for large grids.

The iteration and the stopping rule are those of the library solver
``muFFTTO.solvers.conjugate_gradients_mugrid_experimental(stop_crit_norm='energy_lower_estim', d_min=..., tau=0.25,
**topology_optimization.energy_stop_kwargs(...))`` -- the Meurant-Papez-Tichy lower estimate of ||x - x_k||_K^2 with
adaptive delay, compared with (tol ||x||_K)^2 -- so the iterates and iteration counts are the same. Local copy
(the library CG keeps no variants) with two differences that matter on large grids:

* vector updates and dot products are muGrid ``linalg`` kernels on the fields (``axpy``, ``axpby``, ``vecdot``):
  no temporaries, where ``np.dot(r.s.ravel(), ...)`` and ``x.s += alpha * p.s`` copy whole vectors on the GPU;
* the product K x_0 needed for the initial residual also gives the energy of the warm start (the reference of the
  stopping test), and only (r, z) is reduced per iteration -- one K product and one dot product less per solve.
"""
import numpy as np
from muGrid import linalg

from muFFTTO.solvers import findS


def conjugate_gradients_energy(comm, K, M, b, x, work, tol=1e-6, d_min=0, tau=0.25, maxiter=100000):
    """Solve K x = b by PCG from the start x until the energy error is below tol * ||x||_K.

    Parameters
    ----------
    comm : muGrid.Communicator
    K, M : callable
        ``K(v, Kv)`` and ``M(r, z)`` write the operator / preconditioner product into the second field.
    b, x : muGrid Field
        Right-hand side and start vector (overwritten with the solution).
    work : tuple of 4 muGrid Fields
        Residual r, preconditioned residual z, search direction p and product K p (any content).
    tol : float
        Relative energy error: stop when the estimate of ||x - x_k||_K is below ``tol ||x_0||_K`` (warm start) or
        ``tol ||x - x_0||_K`` (start x_0 = 0, where ||x_0||_K = 0).
    d_min : int
        Minimal delay of the error estimate (0: plain adaptive delay).

    Returns
    -------
    int
        Number of iterations (K products in the loop).
    """
    r, z, p, Kp = work

    def dot(u, v):
        return comm.sum(np.float64(float(linalg.vecdot(u, v))))

    K(x, Kp)
    start_energy = dot(x, Kp)                    # ||x_0||_K^2: the scale of the stopping test for a warm start
    linalg.copy(b, r)
    linalg.axpy(-1.0, Kp, r)
    if dot(r, r) == 0.0:
        return 0
    M(r, z)
    linalg.copy(z, p)
    rz = dot(r, z)

    first, delay = 0, 0                          # iteration whose error is estimated next, and its current delay
    Delta, tail_sums = [], []                    # Delta[j] = ||x_{j+1} - x_j||_K^2, tail_sums[j] = sum_{i >= j} Delta[i]
    estimate = None                              # latest accepted estimate of ||x - x_first||_K^2
    for iteration in range(1, maxiter + 1):
        K(p, Kp)
        pKp = dot(p, Kp)
        if pKp <= 0:
            raise RuntimeError('conjugate gradients: operator is not positive definite')
        alpha = rz / pKp
        linalg.axpy(alpha, p, x)
        linalg.axpy(-alpha, Kp, r)
        M(r, z)
        rz_new = dot(r, z)
        linalg.axpby(1.0, z, rz_new / rz, p)     # p = z + beta p

        Delta.append(alpha * rz)
        tail_sums = [value + Delta[-1] for value in tail_sums] + [Delta[-1]]
        if iteration > 2:
            safety = findS(tail_sums, Delta, first)
            numerator, denominator = safety * Delta[-1], float(np.sum(Delta[first:-1]))
            while (delay >= 0 and denominator > 0 and numerator / denominator <= tau
                   and len(Delta) - first >= d_min):
                estimate = denominator + Delta[-1]
                first, delay = first + 1, delay - 1
                denominator = float(np.sum(Delta[first:-1]))
            delay += 1
        rz = rz_new

        threshold = tol ** 2 * (start_energy if start_energy > 0 else tail_sums[0])
        if estimate is not None and estimate < threshold:
            return iteration
    return maxiter
