"""Preconditioned conjugate gradients with residual history and cost counters.

Stopping test (same for all preconditioners): ||r_k||_2 <= tol ||r_0||_2, r_k = f - K u_k (unpreconditioned,
recursively updated; the true residual of the returned solution is reported). The returned solution, iteration count
and cost counters belong to this residual stop.

Additionally (track_energy=True) the iteration continues until the Hestenes-Stiefel lower estimate of the energy
error, sum_{j=k}^{k+delay-1} alpha_j (r_j, z_j), drops below energy_tol^2 times the estimated initial energy error
sum_j alpha_j (r_j, z_j); that iteration k is reported as 'it_energy' (a criterion that does not depend on the
preconditioner). The extra iterations are not counted in 'counter'.
"""
import time

import numpy as np

from fem import apply_K


def pcg(C, f, M, counter, tol=1e-5, maxiter=2000, energy_tol=1e-5, delay=10, track_energy=True, stop='residual'):
    """stop='residual' (default, see module docstring) or 'energy': stop as soon as the delayed energy-error estimate
    is below energy_tol^2 times the estimated initial energy error (x, counters at that iteration)."""
    start = time.perf_counter()
    x = np.zeros_like(f)
    r = f.copy()
    z = M(r)
    p = z.copy()
    rz = np.vdot(r, z)
    r0 = np.linalg.norm(r)
    if r0 == 0.0:                                        # f = 0: u = 0 is the solution
        return {'x': x, 'iterations': 0, 'converged': True, 'time': time.perf_counter() - start,
                'counter': dict(counter), 'it_energy': 0, 'history': np.array([0.0]), 'true_residual': 0.0}
    history = [r0]
    deltas = []
    result = None
    it_energy = None
    for k in range(1, maxiter + 1):
        Kp = apply_K(p, C)
        if result is None:
            counter['K'] += 1
        alpha = rz / np.vdot(p, Kp)
        deltas.append(alpha * rz)
        x += alpha * p
        r -= alpha * Kp
        if result is None:
            history.append(np.linalg.norm(r))
            if stop == 'residual' and history[-1] <= tol * r0:
                result = {'x': x.copy(), 'iterations': k, 'converged': True, 'time': time.perf_counter() - start,
                          'counter': dict(counter)}
        if track_energy and it_energy is None and len(deltas) > delay:
            j = len(deltas) - delay - 1                     # estimate for iteration j is complete
            if sum(deltas[j:j + delay]) <= energy_tol ** 2 * sum(deltas):
                it_energy = j
                if stop == 'energy' and result is None:
                    result = {'x': x.copy(), 'iterations': k, 'converged': True,
                              'time': time.perf_counter() - start, 'counter': dict(counter)}
        if result is not None and (not track_energy or it_energy is not None):
            break
        z = M(r) if result is None else _uncounted(M, r, counter)
        rz_new = np.vdot(r, z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    if result is None:
        result = {'x': x.copy(), 'iterations': maxiter, 'converged': False, 'time': time.perf_counter() - start,
                  'counter': dict(counter)}
    result['it_energy'] = it_energy
    result['history'] = np.array(history)
    result['true_residual'] = np.linalg.norm(f - apply_K(result['x'], C)) / r0
    return result


def _uncounted(M, r, counter):
    saved = dict(counter)
    z = M(r)
    counter.update(saved)
    return z
