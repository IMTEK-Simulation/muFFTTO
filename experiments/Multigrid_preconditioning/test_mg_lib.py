"""
Checks of mg_lib on a small grid (N = 16):
  1. R = P^T                                   (P u, r) = (u, R r)
  2. Galerkin coarsening                       R K_h P u = K_H u, K_H = muFFTTO stiffness with coarsened material
  3. every preconditioner is symmetric         (y, B x) = (x, B y)
  4. and positive definite on zero-mean x      (x, B x) > 0
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import mg_lib
import problems

N = 16
rng = np.random.default_rng(0)

# random SPD material per quad point: congruence of C_0 with random matrices -> anisotropic, heterogeneous
T = rng.standard_normal((2, 2, 2, N, N)) * 0.4 + np.eye(2)[:, :, None, None, None]
s = 10.0 ** rng.uniform(-3, 3, size=(2, N, N))
C_q = s * np.einsum('iaqxy,jbqxy,kcqxy,ldqxy,abcd->ijklqxy', T, T, T, T, problems.C_0)

hier = mg_lib.Hierarchy(C_q, problems.C_0, n_coarsest=4)
fine, coarse = hier.levels[0], hier.levels[1]
print('levels:', [lev.N for lev in hier.levels], ' field shape', fine.shape)


def zero_mean(x):
    return x - x.mean(axis=(-2, -1), keepdims=True)


u_H = rng.standard_normal(coarse.shape)
r_h = rng.standard_normal(fine.shape)
lhs, rhs = np.sum(mg_lib.prolong(u_H) * r_h), np.sum(u_H * mg_lib.restrict(r_h))
print(f'1. adjointness         (Pu, r) = {lhs:.12f}  (u, Rr) = {rhs:.12f}  rel diff {abs(lhs - rhs) / abs(lhs):.1e}')

galerkin = mg_lib.restrict(fine.K(mg_lib.prolong(u_H)))
direct = coarse.K(u_H)
print(f'2. Galerkin            |R K_h P u - K_H u| / |K_H u| = '
      f'{np.linalg.norm(galerkin - direct) / np.linalg.norm(direct):.1e}')

methods = ['jacobi', 'green', 'green_jacobi', 'dual', 'mg', 'mgS-green', 'mgS-green_jacobi', 'mgS-dual',
           'bnn-green', 'bnn-green_jacobi', 'bnn-dual', 'add-green_jacobi', 'sgs-green']
print('3./4. symmetry and positivity')
for method in methods:
    hier.setup(method)
    B = hier.preconditioner(method)
    x, y = zero_mean(rng.standard_normal(fine.shape)), zero_mean(rng.standard_normal(fine.shape))
    yBx, xBy = np.sum(y * B(x)), np.sum(x * B(y))
    xBx = min(np.sum(v * B(v)) for v in [zero_mean(rng.standard_normal(fine.shape)) for _ in range(5)])
    print(f'   {method:18s} asym {abs(yBx - xBy) / max(abs(yBx), 1e-300):.1e}   min (x, Bx) = {xBx:.3e}')
