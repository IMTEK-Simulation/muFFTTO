"""T7' Newton for power-law elasticity (2D version of Ladecky et al. 2026, Sec. 4.5): compliant linear inclusion
(0.5 x 0.25 rectangle, volume fraction 1/8, pixel-aligned on every grid -> sharp edges) in a power-law matrix,
single-step macroscopic shear eps6 = 0.025, Newton tolerance ||B du|| <= 1e-5 ||E||.

Preconditioners: G (C_ref = inclusion), GJ (C_ref = identity), PD with C_ref = inclusion ('PD_inc'), matrix tangent
at the first Newton iterate ('PD_mat0') or identity ('PD_I'). Records PCG iterations per Newton step, totals, 2D FFT
count and time.

usage: python t7b_newton.py OMEGA N   -> appends one JSON line per method to results/t7b_newton.jsonl
"""
import json
import sys
import time

import numpy as np

from common import RESULTS
from fem import B, BT, weight
from nonlinear import linear_isotropic, stress_and_tangent
from pcg import pcg
from precond import make_green, make_green_jacobi, make_primal_dual_tangent, new_counter

MU0 = 1.0
C_INC = linear_isotropic(2.0 * MU0, MU0)
PARAMETERS = dict(K=20.0 * MU0, sigma0=5.0 * MU0, eps0=0.01)
E = np.array([0.0, 0.0, 0.025])
METHODS = ('G', 'GJ', 'PD_inc', 'PD_mat0', 'PD_I')


def solve_newton(n, omega, method, newton_tol=1e-5, max_newton=60):
    d = 2
    x = (np.arange(n) + 0.5) / n
    x1, x2 = np.meshgrid(x, x, indexing='ij')
    matrix = ~((np.abs(x1 - 0.5) < 0.25) & (np.abs(x2 - 0.5) < 0.125))
    w = weight(n)
    E_field = np.broadcast_to(E[None, :, None, None], (2, 3, n, n))
    # matrix tangent at the first Newton iterate (uniform strain E)
    _, C0_t = stress_and_tangent(E_field, matrix, omega=omega, C_inclusion=C_INC, **PARAMETERS)
    C_mat0 = C0_t[0, :, :, 0, 0] if matrix[0, 0] else None
    Cref = {'G': C_INC, 'GJ': np.eye(3), 'PD_inc': C_INC, 'PD_mat0': C_mat0, 'PD_I': np.eye(3)}[method]
    u = np.zeros((d, n, n))
    cg, cg_energy, ffts, steps = [], [], [], []
    start = time.perf_counter()
    for it in range(max_newton):
        sigma, C_t = stress_and_tangent(E_field + B(u), matrix, omega=omega, C_inclusion=C_INC, **PARAMETERS)
        f = -BT(w * sigma)
        counter = new_counter()
        if method == 'G':
            M = make_green(Cref, d, n, counter)
        elif method == 'GJ':
            M = make_green_jacobi(C_t, Cref, d, n, counter)
        else:
            M = make_primal_dual_tangent(C_t, Cref, d, n, counter)
        result = pcg(C_t, f, M, counter, tol=1e-5, maxiter=10000)
        u += result['x']
        cg.append(result['iterations'])
        cg_energy.append(result['it_energy'])
        ffts.append(result['counter']['fft'])
        steps.append(float(np.linalg.norm(B(result['x'])) / np.linalg.norm(E_field)))
        if steps[-1] <= newton_tol:
            break
    sigma, _ = stress_and_tangent(E_field + B(u), matrix, omega=omega, C_inclusion=C_INC, **PARAMETERS)
    return dict(omega=omega, n=n, method=method, newton_iterations=len(cg), cg=cg, cg_energy=cg_energy,
                cg_total=sum(cg), ffts_total=sum(ffts), time_s=time.perf_counter() - start, newton_steps=steps,
                mean_sigma6=float(sigma[:, 2].mean()))


if __name__ == '__main__':
    omega, n = int(sys.argv[1]), int(sys.argv[2])
    for method in METHODS:
        out = solve_newton(n, omega, method)
        with open(f'{RESULTS}/t7b_newton.jsonl', 'a') as f:
            f.write(json.dumps(out) + '\n')
        print(method, out['newton_iterations'], out['cg_total'], out['cg'], f"{out['time_s']:.1f}s", flush=True)
