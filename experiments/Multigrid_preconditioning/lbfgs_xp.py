"""
Bound-constrained L-BFGS with a preconditioned initial inverse Hessian, for CPU (numpy, MPI) and GPU (cupy) arrays.
Experiment-local (2026-11-15); NuMPI and muFFTTO are untouched.

The algorithm is exactly NuMPI.Optimization.l_bfgs_bounded with `precond` (same masks, two-loop recursion with
H_0 = gamma P M P, gamma = <s, y> / <y, P M P y>, curvature test s.y > 1e-12 |s||y|, projected Armijo backtracking,
the steepest-descent retry, xtol / ftol / gtol tests and callback), so the iterates agree with NuMPI's to round-off.
What differs is only how the vectors are handled:

* The arrays stay in the namespace of x0: numpy -> everything on the host; cupy -> everything on the GPU (then `fun`
  must accept a device iterate and return the gradient on the device, `precond` must map device to device arrays).
  Only scalars cross the PCIe bus / the network (one MPI allreduce per inner product, as in NuMPI).
* Inner products with BLAS / cuBLAS dots, in-place updates, no temporaries in the two-loop recursion (NuMPI's
  pnp.sum(s * q) allocates and traverses a full temporary per product). Measured on the laptop at the per-rank size
  of 512^3 on 4 GPUs (33.5 M values): NuMPI 12 s of host work per iteration with a full memory.
* reduction='sum' computes as NuMPI does (inner products as the sum of a temporary product, updates without fused
  multiply-add): bitwise NuMPI's iterates on the CPU, for checks; 'dot' (default) is faster but rounds differently, so near-ties (a slope <r, d> ~ 0, an
  Armijo test at equality) can go the other way and the iterates then drift apart.
* pair_store='host' (with device arrays): the curvature pairs live in pinned host memory and are streamed to the GPU
  in the two-loop recursion (2 x 2 x maxcor transfers per iteration) -- for grids whose 2 maxcor pair vectors do not
  fit on the GPU next to the solver (1024^3 on 8 H200).

    result = l_bfgs_bounded_xp(fun, x0, bounds_lo=0.0, bounds_hi=1.0, precond=M, comm=MPI.COMM_WORLD, ...)
result: dict-like (x, fun, jac, nit, message, success, max_grad, nfev), x and jac in the namespace and shape of x0.
"""
import time

import numpy as np

try:
    import cupy
except ImportError:                                          # CPU-only environments
    cupy = None


def _namespace(a):
    return cupy if (cupy is not None and isinstance(a, cupy.ndarray)) else np


class _Ops:
    """the vector operations of the optimiser in one array namespace, with MPI reductions over `comm`"""

    def __init__(self, xp, comm, reduction='dot'):
        self.xp, self.comm, self.reduction = xp, comm, reduction
        self.seconds = 0.0
        if xp is np:
            try:
                from scipy.linalg.blas import daxpy
                self._daxpy = daxpy
            except ImportError:
                self._daxpy = None
        else:
            self._axpy_kernel = cupy.ElementwiseKernel('float64 a, float64 x', 'float64 y', 'y += a * x', 'lbfgs_axpy')

    def _allreduce(self, value, op):
        if self.comm is None or self.comm.size == 1:
            return value
        from mpi4py import MPI
        return self.comm.allreduce(value, op=MPI.SUM if op == 'sum' else MPI.MAX)

    def dot(self, a, b):
        if self.reduction == 'sum':                         # NuMPI's arithmetic (temporary + pairwise sum): for checks
            return self._allreduce(float(self.xp.sum(a * b)), 'sum')
        return self._allreduce(float(self.xp.dot(a, b)), 'sum')

    def max_abs(self, a):
        return self._allreduce(float(self.xp.max(self.xp.abs(a))) if a.size else 0.0, 'max')

    def axpy(self, alpha, x, y):
        """y += alpha x in place"""
        if self.reduction == 'sum':                         # NuMPI's arithmetic (two roundings, no FMA)
            y += alpha * x
        elif self.xp is np:
            if self._daxpy is not None and y.flags.c_contiguous and x.flags.c_contiguous:
                self._daxpy(x, y, a=alpha)                    # in place on y (BLAS, no temporary)
            else:
                y += alpha * x
        else:
            self._axpy_kernel(alpha, x, y)


class _PairStore:
    """curvature pairs, newest first; 'device' keeps them in the array namespace, 'host' in pinned host memory"""

    def __init__(self, xp, where, maxcor):
        self.xp, self.where, self.maxcor = xp, where, maxcor
        self.S, self.Y, self.rho, self.spare = [], [], [], []
        self.buffers = None                                   # device buffers for streamed host pairs

    def __len__(self):
        return len(self.S)

    def clear(self):
        if self.where == 'host':                              # pinned buffers are expensive: keep them for reuse
            self.spare += list(zip(self.S, self.Y))
        self.S, self.Y, self.rho = [], [], []

    def push(self, s, y, rho):
        if self.where == 'host':
            if self.spare:
                hs, hy = self.spare.pop()
            else:
                import cupyx
                hs, hy = cupyx.empty_pinned(s.shape, s.dtype), cupyx.empty_pinned(y.shape, y.dtype)
            s.get(out=hs)
            y.get(out=hy)
            s, y = hs, hy
        self.S.insert(0, s)
        self.Y.insert(0, y)
        self.rho.insert(0, rho)
        if len(self.S) > self.maxcor:
            oldest = (self.S.pop(), self.Y.pop())
            self.rho.pop()
            if self.where == 'host':
                self.spare.append(oldest)

    def get(self, k):
        """(s_k, y_k) as arrays of the working namespace"""
        if self.where != 'host':
            return self.S[k], self.Y[k]
        if self.buffers is None:
            self.buffers = (cupy.empty(self.S[k].shape, self.S[k].dtype), cupy.empty(self.Y[k].shape, self.Y[k].dtype))
        bs, by = self.buffers
        bs.set(self.S[k])
        by.set(self.Y[k])
        return bs, by


def l_bfgs_bounded_xp(fun, x0, bounds_lo=0.0, bounds_hi=1.0, precond=None, gtol=1e-5, xtol=0.0, ftol=0.0,
                      maxiter=500, maxcor=20, c1=1e-4, max_halvings=40, callback=None, disp=False, comm=None,
                      pair_store='device', tol_box=1e-12, reduction='dot'):
    xp = _namespace(x0)
    ops = _Ops(xp, comm, reduction)
    rank0 = comm is None or comm.rank == 0
    shape = x0.shape
    lo, hi = bounds_lo, bounds_hi
    pairs = _PairStore(xp, pair_store if xp is not np else 'device', maxcor)
    nfev = [0]
    timing = {'optimizer': 0.0, 'fun': 0.0, 'precond': 0.0}

    def fun_grad(x):
        t = time.time()
        nfev[0] += 1
        phi, g = fun(x.reshape(shape))
        g = xp.asarray(g, dtype=float).ravel()
        timing['fun'] += time.time() - t
        return float(phi), g

    def project(y):
        return xp.clip(y, lo, hi)

    def active_mask(grad, x):
        """True where x sits on a bound and the gradient points into it (NuMPI's _kkt_residual / _free_mask)"""
        return ((x <= lo + tol_box) & (grad >= 0.0)) | ((x >= hi - tol_box) & (grad <= 0.0))

    def kkt(grad, x):
        return xp.where(active_mask(grad, x), 0.0, grad)

    def finish(success, x, phi, grad, nit, residual, message):
        return {'success': bool(success), 'x': x.reshape(shape), 'fun': phi, 'jac': grad.reshape(shape),
                'nit': int(nit), 'message': message, 'max_grad': float(residual), 'nfev': nfev[0],
                'timing': timing}

    def armijo(x, d, phi, gd, alpha):
        for _ in range(max_halvings):
            x_try = project(x + alpha * d)
            phi_try, grad_try = fun_grad(x_try)
            if phi_try <= phi + c1 * alpha * gd:
                return x_try, phi_try, grad_try, alpha
            alpha *= 0.5
        return None

    x = project(xp.asarray(x0, dtype=float).ravel())
    phi, grad = fun_grad(x)
    r = kkt(grad, x)
    r_norm = ops.max_abs(r)
    if disp and rank0:
        print(f"{'iter':<5} {'f':<14} {'|r|inf_free':<14} {'alpha':<8}")
        print("-" * 46)
        print(f"{0:<5d} {phi:<14.6e} {r_norm:<14.4e} {'-':<8}", flush=True)
    if r_norm < gtol:
        return finish(True, x, phi, grad, 0, r_norm, "CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_GTOL")

    for iteration in range(1, maxiter + 1):
        t_it = time.time()
        fun_seconds = timing['fun']
        if precond is not None:
            active = active_mask(grad, x)

            def H0(v, active=active):
                t = time.time()
                z = xp.asarray(precond(xp.where(active, 0.0, v).reshape(shape)), dtype=float).ravel()
                z = xp.where(active, 0.0, z)
                timing['precond'] += time.time() - t
                return z
        else:
            H0 = None

        # two-loop recursion (newest -> oldest, H_0, oldest -> newest), in place on q
        q = r.copy()
        n = len(pairs)
        a = np.empty(n)
        for k in range(n):
            s_k, y_k = pairs.get(k)
            a[k] = pairs.rho[k] * ops.dot(s_k, q)
            ops.axpy(-a[k], y_k, q)
        if H0 is not None:
            if n > 0:
                s_0, y_0 = pairs.get(0)
                gamma = ops.dot(s_0, y_0) / ops.dot(y_0, H0(y_0))
                q = H0(q)
                q *= gamma
            else:
                q = H0(q)
        elif n > 0:
            s_0, y_0 = pairs.get(0)
            q *= ops.dot(s_0, y_0) / ops.dot(y_0, y_0)
        for k in range(n - 1, -1, -1):
            s_k, y_k = pairs.get(k)
            beta = pairs.rho[k] * ops.dot(y_k, q)
            ops.axpy(a[k] - beta, s_k, q)
        d = q
        d *= -1.0

        gd = ops.dot(r, d)
        quasi_newton_step = n > 0
        if gd >= 0.0:                                         # stale curvature: reset
            pairs.clear()
            d = -r if H0 is None else -H0(r)
            gd = ops.dot(r, d)
        timing['optimizer'] += time.time() - t_it - (timing['fun'] - fun_seconds)
        step = armijo(x, d, phi, gd, 1.0)
        if step is None:
            pairs.clear()
            d = -r if H0 is None else -H0(r)
            gd = ops.dot(r, d)
            d_norm = np.sqrt(ops.dot(d, d))
            step = armijo(x, d, phi, gd, 1.0 / d_norm if d_norm > 0 else 1.0)
        if step is None:
            return finish(False, x, phi, grad, iteration, r_norm, "CONVERGENCE: line-search did not converge")
        x_new, phi_new, grad_new, alpha = step
        t_upd = time.time()
        r_new = kkt(grad_new, x_new)

        s = x_new - x
        y = r_new - r
        sy = ops.dot(s, y)
        s_norm, y_norm = np.sqrt(ops.dot(s, s)), np.sqrt(ops.dot(y, y))
        if sy > 1e-12 * (s_norm * y_norm + np.finfo(float).eps):
            pairs.push(s, y, 1.0 / sy)
        if xtol > 0 and quasi_newton_step and ops.max_abs(x_new - x) < xtol:
            return finish(True, x, phi, grad, iteration, r_norm, "CONVERGENCE: NORM_OF_VARIABLE_STEP_<=_XTOL")
        if ftol > 0 and abs(phi_new - phi) < ftol * max(1, abs(phi_new), abs(phi)):
            return finish(True, x, phi, grad, iteration, r_norm, "CONVERGENCE: REL_REDUCTION_OF_F_<=_FTOL")
        x, phi, grad, r = x_new, phi_new, grad_new, r_new
        r_norm = ops.max_abs(r)
        timing['optimizer'] += time.time() - t_upd
        if disp and rank0:
            print(f"{iteration:<5d} {phi:<14.6e} {r_norm:<14.4e} {alpha:<8.2e}", flush=True)
        if callback is not None and callback(x.reshape(shape)) is True:
            return finish(True, x, phi, grad, iteration, r_norm, "CONVERGENCE: NORM_OF_custom_stop_crit_<= _csc_tol")
        if r_norm < gtol:
            return finish(True, x, phi, grad, iteration, r_norm, "CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_GTOL")
    return finish(False, x, phi, grad, maxiter, r_norm, "NO CONVERGENCE: MAXITERATIONS REACHED")
