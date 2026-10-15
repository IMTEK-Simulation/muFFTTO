"""T5b mean drift with voids / sharp data: GJ and PD without mean subtraction inside the preconditioner,
error vs Green at the same tolerance (cosine chi = inf p = 16, cosine chi = 1e4 p = 256, sharp disk chi = 1e4)."""
import sys

import numpy as np

from common import solve, write_csv
from fem import B
from materials import cosine, disk

d = int(sys.argv[1]) if len(sys.argv) > 1 else 1
cases = {'cosine chi=inf p=16': cosine(256, 16, np.inf), 'cosine chi=1e4 p=256': cosine(256, 256, 1e4),
         'sharp disk chi=1e4': disk(256, 1e4, 0)}
rows = []
for case, rho in cases.items():
    for tol in (1e-4, 1e-6, 1e-8, 1e-10):
        green = solve(rho, d, 'G', tol=tol, maxiter=20000, track_energy=False)['x']
        for method in ('GJ', 'PD'):
            x = solve(rho, d, method, tol=tol, maxiter=20000, subtract_mean=False, track_energy=False)['x']
            mean = x.mean(axis=(1, 2), keepdims=True)
            rows.append(dict(d=d, case=case, tol=tol, method=method,
                             displacement_error=f'{np.linalg.norm(x - green) / np.linalg.norm(green):.3e}',
                             displacement_error_mean_removed=f'{np.linalg.norm(x - mean - green) / np.linalg.norm(green):.3e}',
                             strain_error=f'{np.linalg.norm(B(x) - B(green)) / np.linalg.norm(B(green)):.3e}',
                             mean_relative=f'{np.abs(mean).max() / np.abs(green).max():.3e}'))
            print(rows[-1], flush=True)
kind = 'scalar' if d == 1 else 'elastic'
write_csv(f't5b_mean_drift_voids_{kind}.csv', rows)
