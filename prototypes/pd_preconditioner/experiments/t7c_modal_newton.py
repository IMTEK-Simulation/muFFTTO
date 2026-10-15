"""T7'' Newton for power-law elasticity with the modally matched PD (same problem as t7b_newton.py: rectangle
inclusion, volume fraction 1/8, shear 0.025 in one step, Newton tol 1e-5).

Preconditioners: G (C_ref = inclusion), GJ (C_ref = I), PD_pointwise (S = C_ref C_t^-1 C_ref, C_ref = inclusion),
PD_modal (isotropic S matching the acoustic moduli; reference = inclusion shear mu_r = 1 with kappa_r raised to
max kappa/mu of the tangent so that S stays positive), PD_modal_clip (same, mu clipped at 1e-2 mu_max in S).
CG stop: 'residual' (||r|| <= 1e-5 ||r_0||) or 'energy' (energy-error estimate <= 1e-5, delay 10).

usage: python t7c_modal_newton.py OMEGA N STOP   -> appends JSON lines to results/t7c_modal_newton.jsonl
"""
import json
import sys
import time

import numpy as np

from common import RESULTS
from fem import B, BT, weight
from nonlinear import linear_isotropic, stress_and_tangent
from pcg import pcg
from precond import (make_green, make_green_jacobi, make_primal_dual_modal, make_primal_dual_tangent,
                     new_counter)

C_INC = linear_isotropic(2.0, 1.0)
PARAMETERS = dict(K=20.0, sigma0=5.0, eps0=0.01)
E = np.array([0.0, 0.0, 0.025])
METHODS = ('G', 'GJ', 'PD_pointwise', 'PD_modal', 'PD_modal_clip')


def solve_newton(n, omega, method, stop, newton_tol=1e-5, max_newton=60):
    d = 2
    x = (np.arange(n) + 0.5) / n
    x1, x2 = np.meshgrid(x, x, indexing='ij')
    matrix = ~((np.abs(x1 - 0.5) < 0.25) & (np.abs(x2 - 0.5) < 0.125))
    w = weight(n)
    E_field = np.broadcast_to(E[None, :, None, None], (2, 3, n, n))
    u = np.zeros((d, n, n))
    cg, ffts, steps = [], [], []
    start = time.perf_counter()
    for it in range(max_newton):
        sigma, C_t = stress_and_tangent(E_field + B(u), matrix, omega=omega, C_inclusion=C_INC, **PARAMETERS)
        f = -BT(w * sigma)
        counter = new_counter()
        if method == 'G':
            M = make_green(C_INC, d, n, counter)
        elif method == 'GJ':
            M = make_green_jacobi(C_t, np.eye(3), d, n, counter)
        elif method == 'PD_pointwise':
            M = make_primal_dual_tangent(C_t, C_INC, d, n, counter)
        elif method == 'PD_modal':
            M = make_primal_dual_modal(C_t, C_INC, d, n, counter, auto_reference=True)
        else:
            M = make_primal_dual_modal(C_t, C_INC, d, n, counter, auto_reference=True, clip=1e-2)
        result = pcg(C_t, f, M, counter, tol=1e-5, maxiter=20000, stop=stop)
        u += result['x']
        cg.append(result['iterations'])
        ffts.append(result['counter']['fft'])
        steps.append(float(np.linalg.norm(B(result['x'])) / np.linalg.norm(E_field)))
        if steps[-1] <= newton_tol:
            break
    sigma, _ = stress_and_tangent(E_field + B(u), matrix, omega=omega, C_inclusion=C_INC, **PARAMETERS)
    return dict(omega=omega, n=n, method=method, stop=stop, newton_iterations=len(cg), cg=cg, cg_total=sum(cg),
                ffts_total=sum(ffts), time_s=time.perf_counter() - start, newton_steps=steps,
                mean_sigma6=float(sigma[:, 2].mean()))


if __name__ == '__main__':
    omega, n, stop = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    for method in METHODS:
        out = solve_newton(n, omega, method, stop)
        with open(f'{RESULTS}/t7c_modal_newton.jsonl', 'a') as f:
            f.write(json.dumps(out) + '\n')
        print(method, out['newton_iterations'], out['cg_total'], out['ffts_total'], f"{out['time_s']:.1f}s",
              out['cg'][:10], flush=True)
