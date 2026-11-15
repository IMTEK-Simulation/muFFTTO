"""
Local design optimizers for the TO test (2026-11-13; NuMPI and muFFTTO untouched): bound-constrained L-BFGS with a
preconditioned initial Hessian, and projected preconditioned nonlinear CG.  Serial (no MPI reductions).

Both use the conventions of NuMPI.Optimization.l_bfgs_bounded (box-masked "free KKT" gradient r_free, projected
Armijo backtracking along the clipping arc, xtol test on max |x_new - x| skipped in the first iteration), so with
precond=None `l_bfgs_bounded_precond` reproduces NuMPI's iterates.

precond(v): an SPD smoother M (here the Helmholtz (I + c eta^2 (-Lap_h))^-1 of the design gradient).  It is applied
restricted to the free subspace, P M P with P = 0 on box-active nodes, so the direction never pushes into an active
bound and stays a descent direction.

    l_bfgs_bounded_precond   two-loop recursion with H_0 = gamma P M P, gamma = <s, y> / <y, P M P y>
    ncg_bounded              d = -z + beta d_prev, z = P M P r_free, beta = PR+ (Polak-Ribiere-Polyak, >= 0);
                             restart (d = -z) when <r, d> >= 0, when > active_change_tol of the nodes change
                             their active/free status, Powell's test
                             |<r_new, z_old>| >= 0.2 <r_new, z_new>, or every `restart_every` iterations.
                             Step: initial alpha = alpha_prev <r_prev, d_prev> / <r, d> (Nocedal & Wright 3.60;
                             first iteration: largest change of x = step0), one quadratic-interpolation trial, then
                             projected Armijo halving.
"""
import numpy as np

from NuMPI.Optimization._lbfgs_helpers import _armijo, _kkt_residual
from NuMPI.Optimization.Result import OptimizeResult


def _project_fn(lo, hi):
    def project(y):
        return np.clip(np.asarray(y, dtype=float), lo, hi)
    return project


def _free_mask(x, grad, lo, hi, tol=1e-12):
    """1 where x may move: interior, or on a bound with the gradient pointing back into the box"""
    at_lo = (x <= lo + tol) & (grad >= 0)                      # the same test as NuMPI's _kkt_residual
    at_hi = (x >= hi - tol) & (grad <= 0)
    return (~(at_lo | at_hi)).astype(float)


def _result(success, x, phi, grad, nit, residual, message, nfev):
    return OptimizeResult({'success': bool(success), 'x': x, 'fun': phi, 'jac': grad, 'nit': int(nit),
                           'message': message, 'max_grad': float(residual), 'nfev': nfev})


class _Counted:
    def __init__(self, fun, shape):
        self.fun, self.shape, self.n = fun, shape, 0

    def __call__(self, x):
        self.n += 1
        phi, g = self.fun(x.reshape(self.shape))
        return phi, np.asarray(g).ravel()


def l_bfgs_bounded_precond(fun, x0, bounds_lo=0.0, bounds_hi=1.0, precond=None, gtol=1e-5, xtol=0.0, maxiter=500,
                           maxcor=20, c1=1e-4, max_halvings=40, callback=None, disp=False):
    shape = np.asarray(x0).shape
    fg = _Counted(fun, shape)
    project = _project_fn(bounds_lo, bounds_hi)
    x = project(np.asarray(x0, dtype=float).ravel())
    phi, grad = fg(x)
    r = _kkt_residual(grad, x, lo=bounds_lo, hi=bounds_hi)
    r_norm = np.max(np.abs(r))
    S, Y, RHO = [], [], []

    def H0(v, mask):
        return v if precond is None else mask * precond(mask * v)

    for it in range(1, maxiter + 1):
        mask = _free_mask(x, grad, bounds_lo, bounds_hi)
        # two-loop recursion with H_0 = gamma P M P
        q = r.copy()
        alphas = []
        for s, y, rho in zip(S, Y, RHO):
            a = rho * np.sum(s * q)
            alphas.append(a)
            q -= a * y
        if S:
            Hy = H0(Y[0], mask)
            q = (np.sum(S[0] * Y[0]) / np.sum(Y[0] * Hy)) * H0(q, mask)
        else:
            q = H0(q, mask)
        for (s, y, rho), a in zip(reversed(list(zip(S, Y, RHO))), reversed(alphas)):
            b = rho * np.sum(y * q)
            q += (a - b) * s
        d = -q
        gd = np.sum(r * d)
        quasi_newton = len(S) > 0
        if gd >= 0:
            S.clear(); Y.clear(); RHO.clear()
            d = -H0(r, mask)
            gd = np.sum(r * d)
        alpha_init = 1.0 if (quasi_newton or precond is not None) else 1.0
        step = _armijo(fg, x, d, phi, gd, project, c1=c1, max_halvings=max_halvings, alpha_init=alpha_init)
        if step is None:
            S.clear(); Y.clear(); RHO.clear()
            d = -H0(r, mask)
            gd = np.sum(r * d)
            dn = np.max(np.abs(d))
            step = _armijo(fg, x, d, phi, gd, project, c1=c1, max_halvings=max_halvings,
                           alpha_init=1.0 / dn if dn > 0 else 1.0)
        if step is None:
            return _result(False, x.reshape(shape), phi, grad.reshape(shape), it, r_norm,
                           'line search did not converge', fg.n)
        x_new, phi_new, grad_new, alpha = step
        grad_new = np.asarray(grad_new)
        r_new = _kkt_residual(grad_new, x_new, lo=bounds_lo, hi=bounds_hi)
        s, y = x_new - x, r_new - r
        sy = np.sum(s * y)
        if disp:                                           # diagnostics of the curvature pair and of the direction
            Hy = H0(y, mask)
            print(f'L-BFGS pair it {it:4d}: cos(s, y) {sy / (np.linalg.norm(s) * np.linalg.norm(y) + 1e-300):+.3e}  '
                  f'gamma = sy / yHy {sy / (np.sum(y * Hy) + 1e-300):+.3e}  mean|d| {np.mean(np.abs(d)):.3e}  '
                  f'mean|s| {np.mean(np.abs(s)):.3e}  free nodes changed {int(np.sum(_free_mask(x_new, grad_new, bounds_lo, bounds_hi) != mask))}',
                  flush=True)
        if sy > 1e-12 * (np.linalg.norm(s) * np.linalg.norm(y) + np.finfo(float).eps):
            S.insert(0, s.copy()); Y.insert(0, y.copy()); RHO.insert(0, 1.0 / sy)
            if len(S) > maxcor:
                S.pop(); Y.pop(); RHO.pop()
        if xtol > 0 and quasi_newton and np.max(np.abs(x_new - x)) < xtol:
            return _result(True, x.reshape(shape), phi, grad.reshape(shape), it, r_norm, 'XTOL', fg.n)
        x, phi, grad, r = x_new, phi_new, grad_new, r_new
        r_norm = np.max(np.abs(r))
        if disp:
            print(f'L-BFGS{"-M" if precond else ""} it {it:4d}  f {phi:.6e}  |r|inf {r_norm:.3e}  alpha {alpha:.2e}  '
                  f'nfev {fg.n}', flush=True)
        if callback is not None and callback(x.reshape(shape)) is True:
            return _result(True, x.reshape(shape), phi, grad.reshape(shape), it, r_norm, 'callback', fg.n)
        if r_norm < gtol:
            return _result(True, x.reshape(shape), phi, grad.reshape(shape), it, r_norm, 'GTOL', fg.n)
    return _result(False, x.reshape(shape), phi, grad.reshape(shape), maxiter, r_norm, 'MAXITER', fg.n)


def ncg_bounded(fun, x0, bounds_lo=0.0, bounds_hi=1.0, precond=None, gtol=1e-5, xtol=0.0, maxiter=500, c1=1e-4,
                max_halvings=40, step0=0.1, restart_every=50, active_change_tol=1e-3, callback=None, disp=False):
    shape = np.asarray(x0).shape
    fg = _Counted(fun, shape)
    project = _project_fn(bounds_lo, bounds_hi)
    M = (lambda v, m: v) if precond is None else (lambda v, m: m * precond(m * v))
    x = project(np.asarray(x0, dtype=float).ravel())
    phi, grad = fg(x)
    r = _kkt_residual(grad, x, lo=bounds_lo, hi=bounds_hi)
    r_norm = np.max(np.abs(r))
    mask = _free_mask(x, grad, bounds_lo, bounds_hi)
    z = M(r, mask)
    d = -z
    rz = np.sum(r * z)
    alpha_prev, gd_prev = None, None
    since_restart = 0

    for it in range(1, maxiter + 1):
        gd = np.sum(r * d)
        if gd >= 0:                                            # not a descent direction -> restart
            d, gd, since_restart = -z, -rz, 0
        if alpha_prev is None:
            alpha0 = step0 / max(np.max(np.abs(d)), 1e-300)    # first step: largest change of x = step0
        else:
            alpha0 = alpha_prev * gd_prev / gd                 # Nocedal & Wright (3.60)
        # one quadratic-interpolation trial from f(0), f'(0) = gd and f(alpha0), then projected Armijo halving
        x_t = project(x + alpha0 * d)
        phi_t, grad_t = fg(x_t)
        denom = phi_t - phi - gd * alpha0
        if denom > 0:
            a_q = -gd * alpha0 ** 2 / (2 * denom)
            if 0.1 * alpha0 <= a_q <= 10 * alpha0 and abs(a_q - alpha0) > 0.05 * alpha0:
                x_q = project(x + a_q * d)
                phi_q, grad_q = fg(x_q)
                if phi_q < phi_t:
                    alpha0, x_t, phi_t, grad_t = a_q, x_q, phi_q, grad_q
        if phi_t <= phi + c1 * alpha0 * gd:
            x_new, phi_new, grad_new, alpha = x_t, phi_t, np.asarray(grad_t), alpha0
        else:
            step = _armijo(fg, x, d, phi, gd, project, c1=c1, max_halvings=max_halvings, alpha_init=alpha0 / 2)
            if step is None:
                d, gd = -z, -rz                                 # restart along -z once
                step = _armijo(fg, x, d, phi, gd, project, c1=c1, max_halvings=max_halvings,
                               alpha_init=step0 / max(np.max(np.abs(d)), 1e-300))
                since_restart = 0
            if step is None:
                return _result(False, x.reshape(shape), phi, grad.reshape(shape), it, r_norm,
                               'line search did not converge', fg.n)
            x_new, phi_new, grad_new, alpha = step
            grad_new = np.asarray(grad_new)
        r_new = _kkt_residual(grad_new, x_new, lo=bounds_lo, hi=bounds_hi)
        mask_new = _free_mask(x_new, grad_new, bounds_lo, bounds_hi)
        z_new = M(r_new, mask_new)
        rz_new = np.sum(r_new * z_new)
        if xtol > 0 and it > 1 and np.max(np.abs(x_new - x)) < xtol:
            return _result(True, x.reshape(shape), phi, grad.reshape(shape), it, r_norm, 'XTOL', fg.n)
        beta = max(0.0, np.sum(r_new * (z_new - z)) / rz)       # PR+ in the M inner product
        since_restart += 1
        if (np.mean(mask_new != mask) > active_change_tol or since_restart >= restart_every
                or abs(np.sum(r_new * z)) >= 0.2 * rz_new):     # active set changed / periodic / Powell
            beta, since_restart = 0.0, 0
        d = -z_new + beta * d
        alpha_prev, gd_prev = alpha, gd
        x, phi, grad, r, z, rz, mask = x_new, phi_new, grad_new, r_new, z_new, rz_new, mask_new
        r_norm = np.max(np.abs(r))
        if disp:
            print(f'NCG{"-M" if precond else ""} it {it:4d}  f {phi:.6e}  |r|inf {r_norm:.3e}  alpha {alpha:.2e}  '
                  f'beta {beta:.3f}  nfev {fg.n}', flush=True)
        if callback is not None and callback(x.reshape(shape)) is True:
            return _result(True, x.reshape(shape), phi, grad.reshape(shape), it, r_norm, 'callback', fg.n)
        if r_norm < gtol:
            return _result(True, x.reshape(shape), phi, grad.reshape(shape), it, r_norm, 'GTOL', fg.n)
    return _result(False, x.reshape(shape), phi, grad.reshape(shape), maxiter, r_norm, 'MAXITER', fg.n)
