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
* FIXED MEMORY: every vector is allocated once, by `allocate(name)` (default: an empty array of the namespace; the TO
  driver passes muGrid fields, so on the GPU the optimiser's memory comes from muGrid's allocator, not from the cupy
  pool), and all updates are in place: 9 work vectors, 2 candidate pair vectors, 2 maxcor pair vectors (or 2 staging
  vectors with pair_store='host') and a few boolean masks. `fun` may return a view of its own gradient field (it is
  copied at once). `precond(v, out=...)` is used when the preconditioner accepts `out` (it must then fill `out`).
* Inner products with BLAS / cuBLAS dots, axpy without temporaries. Laptop, 33.5 M values (512^3 per rank on 4 GPUs),
  vector work per iteration with a full memory: NuMPI ~11 s, here CPU ~5 s, GPU ~0.3 s (pairs on the device) / ~0.9 s
  (pairs streamed from pinned host memory).
* reduction='sum' computes as NuMPI does (inner products as the sum of a temporary product, updates without fused
  multiply-add): bitwise NuMPI's iterates on the CPU, for checks; 'dot' (default) is faster but rounds differently,
  so near-ties (a slope <r, d> ~ 0, an Armijo test at equality) can go the other way and the iterates then drift.
* pair_store='host' (with device arrays): the curvature pairs live in pinned host memory and are streamed through two
  device staging vectors in the two-loop recursion -- for grids whose 2 maxcor pair vectors do not fit on the GPU
  next to the solver (1024^3 on 8 H200).

    result = l_bfgs_bounded_xp(fun, x0, bounds_lo=0.0, bounds_hi=1.0, precond=M, comm=MPI.COMM_WORLD, ...)
result: dict (x, fun, jac, nit, message, success, max_grad, nfev, timing); x and jac are copies in the namespace and
shape of x0.
"""
import inspect
import time

import numpy as np

try:
    import cupy
except ImportError:                                          # CPU-only environments
    cupy = None


def _namespace(a):
    return cupy if (cupy is not None and isinstance(a, cupy.ndarray)) else np


class _Ops:
    """the vector operations in one array namespace, with MPI reductions over `comm`"""

    def __init__(self, xp, comm, reduction='dot'):
        self.xp, self.comm, self.reduction = xp, comm, reduction
        self._daxpy = None
        if xp is np:
            try:
                from scipy.linalg.blas import daxpy
                self._daxpy = daxpy
            except ImportError:
                pass
        else:
            self._axpy_kernel = cupy.ElementwiseKernel('float64 a, float64 x', 'float64 y', 'y += a * x', 'lbfgs_axpy')

    def _allreduce(self, value, op):
        if self.comm is None or self.comm.size == 1:
            return value
        from mpi4py import MPI
        return self.comm.allreduce(value, op=MPI.SUM if op == 'sum' else MPI.MAX)

    def dot(self, a, b):
        if self.reduction == 'sum':                         # NuMPI's arithmetic: for checks
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
                self._daxpy(x, y, a=alpha)                  # in place on y (BLAS)
            else:
                y += alpha * x
        else:
            self._axpy_kernel(alpha, x, y)


def _pointer(a):
    return a.data.ptr if (cupy is not None and isinstance(a, cupy.ndarray)) else a.__array_interface__['data'][0]


def l_bfgs_bounded_xp(fun, x0, bounds_lo=0.0, bounds_hi=1.0, precond=None, gtol=1e-5, xtol=0.0, ftol=0.0,
                      maxiter=500, maxcor=20, c1=1e-4, max_halvings=40, callback=None, disp=False, comm=None,
                      pair_store='device', tol_box=1e-12, reduction='dot', allocate=None):
    xp = _namespace(x0)
    ops = _Ops(xp, comm, reduction)
    rank0 = comm is None or comm.rank == 0
    shape, n = x0.shape, x0.size
    lo, hi = float(bounds_lo), float(bounds_hi)
    host_pairs = pair_store == 'host' and xp is not np
    alloc = allocate if allocate is not None else (lambda name: xp.empty(n, dtype=float))
    timing = {'optimizer': 0.0, 'fun': 0.0, 'precond': 0.0}
    nfev = [0]
    precond_out = precond is not None and 'out' in inspect.signature(precond).parameters

    # ---- the optimiser's vectors, allocated once
    V = {name: alloc(f'lbfgs_{name}') for name in ('x', 'g', 'r', 'q', 'xt', 'gt', 'rt', 'hv', 'hy')}
    spare = [alloc('lbfgs_s_new'), alloc('lbfgs_y_new')]     # candidate pair (s, y) of the current iteration
    act = xp.empty(n, dtype=bool)                            # box-active nodes of the current iterate (for H_0)
    act_kkt, m1, m2 = (xp.empty(n, dtype=bool) for _ in range(3))
    S, Y, RHO = [], [], []                                   # newest first (device arrays or pinned host arrays)
    free_pairs = []                                          # recycled pair buffers
    staging = [alloc('lbfgs_s_stage'), alloc('lbfgs_y_stage')] if host_pairs else None
    allocated_pairs = [0]

    def pair(k):
        if not host_pairs:
            return S[k], Y[k]
        staging[0].set(S[k])
        staging[1].set(Y[k])
        return staging[0], staging[1]

    def push_pair(s, y, rho):
        """store the candidate (s, y) as the newest pair; recycle the buffers of the pair that drops out"""
        if host_pairs:
            if free_pairs:
                hs, hy = free_pairs.pop()
            else:
                import cupyx
                hs, hy = cupyx.empty_pinned((n,), dtype=float), cupyx.empty_pinned((n,), dtype=float)
            s.get(out=hs)
            y.get(out=hy)
            stored = (hs, hy)
        else:
            stored = (s, y)
        S.insert(0, stored[0])
        Y.insert(0, stored[1])
        RHO.insert(0, rho)
        if len(S) > maxcor:
            free_pairs.append((S.pop(), Y.pop()))
            RHO.pop()
        if not host_pairs:                                    # new candidate buffers
            if free_pairs:
                spare[:] = list(free_pairs.pop())
            else:
                allocated_pairs[0] += 1
                spare[:] = [alloc(f'lbfgs_s{allocated_pairs[0]}'), alloc(f'lbfgs_y{allocated_pairs[0]}')]

    def clear_pairs():
        free_pairs.extend(zip(S, Y))
        S.clear(), Y.clear(), RHO.clear()

    def evaluate(x, g_out):
        t = time.time()
        nfev[0] += 1
        phi, g = fun(x.reshape(shape))
        xp.copyto(g_out, xp.asarray(g, dtype=float).reshape(-1))
        timing['fun'] += time.time() - t
        return float(phi)

    def active_mask(grad, x, out):
        """True where x sits on a bound and the gradient points into it (NuMPI's _kkt_residual / _free_mask)"""
        xp.less_equal(x, lo + tol_box, out=out)
        xp.greater_equal(grad, 0.0, out=m1)
        xp.logical_and(out, m1, out=out)
        xp.greater_equal(x, hi - tol_box, out=m1)
        xp.less_equal(grad, 0.0, out=m2)
        xp.logical_and(m1, m2, out=m1)
        xp.logical_or(out, m1, out=out)
        return out

    def kkt(grad, x, out):
        active_mask(grad, x, act_kkt)
        xp.copyto(out, grad)
        xp.copyto(out, 0.0, where=act_kkt)
        return out

    def H0(v, out):
        """out = P M P v (P: zero on the box-active nodes of the current iterate); out may be v"""
        t = time.time()
        hv = V['hv']
        xp.copyto(hv, v)
        xp.copyto(hv, 0.0, where=act)
        if precond_out:
            z = precond(hv.reshape(shape), out=out.reshape(shape))
            if z is not None and _pointer(z) != _pointer(out):
                xp.copyto(out, xp.asarray(z, dtype=float).reshape(-1))
        else:
            xp.copyto(out, xp.asarray(precond(hv.reshape(shape)), dtype=float).reshape(-1))
        xp.copyto(out, 0.0, where=act)
        timing['precond'] += time.time() - t
        return out

    def steepest(r, out):
        """out = -H0 r (or -r without a preconditioner)"""
        if precond is None:
            xp.copyto(out, r)
        else:
            H0(r, out)
        out *= -1.0
        return out

    def armijo(phi, gd, alpha):
        x, d, xt = V['x'], V['q'], V['xt']
        for _ in range(max_halvings):
            xp.multiply(d, alpha, out=xt)                     # NuMPI's rounding: alpha d first, then + x
            xt += x
            xp.clip(xt, lo, hi, out=xt)
            phi_try = evaluate(xt, V['gt'])
            if phi_try <= phi + c1 * alpha * gd:
                return phi_try, alpha
            alpha *= 0.5
        return None

    def finish(success, phi, nit, residual, message):
        return {'success': bool(success), 'x': V['x'].reshape(shape).copy(), 'fun': phi,
                'jac': V['g'].reshape(shape).copy(), 'nit': int(nit), 'message': message,
                'max_grad': float(residual), 'nfev': nfev[0], 'timing': timing}

    # ---- start
    xp.copyto(V['x'], xp.asarray(x0, dtype=float).reshape(-1))
    xp.clip(V['x'], lo, hi, out=V['x'])
    phi = evaluate(V['x'], V['g'])
    kkt(V['g'], V['x'], V['r'])
    r_norm = ops.max_abs(V['r'])
    if disp and rank0:
        print(f"{'iter':<5} {'f':<14} {'|r|inf_free':<14} {'alpha':<8}")
        print("-" * 46)
        print(f"{0:<5d} {phi:<14.6e} {r_norm:<14.4e} {'-':<8}", flush=True)
    if r_norm < gtol:
        return finish(True, phi, 0, r_norm, "CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_GTOL")

    for iteration in range(1, maxiter + 1):
        t_it, fun_before = time.time(), timing['fun']
        x, g, r, q = V['x'], V['g'], V['r'], V['q']
        if precond is not None:
            active_mask(g, x, act)

        # two-loop recursion, in place on q
        xp.copyto(q, r)
        npairs = len(S)
        a = np.empty(npairs)
        for k in range(npairs):
            s_k, y_k = pair(k)
            a[k] = RHO[k] * ops.dot(s_k, q)
            ops.axpy(-a[k], y_k, q)
        if precond is not None:
            if npairs > 0:
                s_0, y_0 = pair(0)
                gamma = ops.dot(s_0, y_0) / ops.dot(y_0, H0(y_0, V['hy']))
                H0(q, q)
                q *= gamma
            else:
                H0(q, q)
        elif npairs > 0:
            s_0, y_0 = pair(0)
            q *= ops.dot(s_0, y_0) / ops.dot(y_0, y_0)
        for k in range(npairs - 1, -1, -1):
            s_k, y_k = pair(k)
            beta = RHO[k] * ops.dot(y_k, q)
            ops.axpy(a[k] - beta, s_k, q)
        q *= -1.0                                             # q is now the direction d

        gd = ops.dot(r, q)
        quasi_newton_step = npairs > 0
        if gd >= 0.0:                                         # stale curvature: reset
            clear_pairs()
            steepest(r, q)
            gd = ops.dot(r, q)
        timing['optimizer'] += time.time() - t_it - (timing['fun'] - fun_before)
        step = armijo(phi, gd, 1.0)
        if step is None:
            clear_pairs()
            steepest(r, q)
            gd = ops.dot(r, q)
            d_norm = np.sqrt(ops.dot(q, q))
            step = armijo(phi, gd, 1.0 / d_norm if d_norm > 0 else 1.0)
        if step is None:
            return finish(False, phi, iteration, r_norm, "CONVERGENCE: line-search did not converge")
        phi_new, alpha = step
        t_upd = time.time()
        xt, gt, rt = V['xt'], V['gt'], V['rt']
        kkt(gt, xt, rt)

        s, y = spare
        xp.subtract(xt, x, out=s)
        xp.subtract(rt, r, out=y)
        sy = ops.dot(s, y)
        s_norm, y_norm = np.sqrt(ops.dot(s, s)), np.sqrt(ops.dot(y, y))
        step_max = ops.max_abs(s) if (xtol > 0 and quasi_newton_step) else np.inf
        if sy > 1e-12 * (s_norm * y_norm + np.finfo(float).eps):
            push_pair(s, y, 1.0 / sy)
        if step_max < xtol:
            return finish(True, phi, iteration, r_norm, "CONVERGENCE: NORM_OF_VARIABLE_STEP_<=_XTOL")
        if ftol > 0 and abs(phi_new - phi) < ftol * max(1, abs(phi_new), abs(phi)):
            return finish(True, phi, iteration, r_norm, "CONVERGENCE: REL_REDUCTION_OF_F_<=_FTOL")
        V['x'], V['xt'] = xt, x                               # accept: swap the roles of the buffers
        V['g'], V['gt'] = gt, g
        V['r'], V['rt'] = rt, r
        phi = phi_new
        r_norm = ops.max_abs(V['r'])
        timing['optimizer'] += time.time() - t_upd
        if disp and rank0:
            print(f"{iteration:<5d} {phi:<14.6e} {r_norm:<14.4e} {alpha:<8.2e}", flush=True)
        if callback is not None and callback(V['x'].reshape(shape)) is True:
            return finish(True, phi, iteration, r_norm, "CONVERGENCE: NORM_OF_custom_stop_crit_<= _csc_tol")
        if r_norm < gtol:
            return finish(True, phi, iteration, r_norm, "CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_GTOL")
    return finish(False, phi, maxiter, r_norm, "NO CONVERGENCE: MAXITERATIONS REACHED")
