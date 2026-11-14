"""
OPERATOR=fused in to3d_large.py (muGrid's fused trilinear-hexahedron kernel, one material value per voxel):
  1. finite-difference check of the gradient (double- and single-precision multigrid);
  2. the objective does not depend on the multigrid precision (the cycle is only a preconditioner);
  3. how far it is from the one-point element with hourglass control at the same design (different elements).
    python fable_check_fused.py [N, default 8] [objective, default energy]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import to3d_large

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
objective = sys.argv[2] if len(sys.argv) > 2 else 'energy'
rng = np.random.default_rng(5)
values = rng.standard_normal((N, N, N))
for _ in range(2):                                   # smooth periodic random design in (0.15, 0.85)
    values = sum(np.roll(values, shift, axis) for axis in range(3) for shift in (-1, 1)) / 6
x = (0.5 + 0.35 * values / np.abs(values).max()).ravel()
settings = dict(objective=objective, target_scale=0.3, cg_tol=1e-12, verbose=False)
results = {}
for label, options in (('fused, double multigrid', dict(operator='fused')),
                       ('fused, single multigrid', dict(operator='fused', mg_precision='single')),
                       ('one-point + hourglass (stencils)', dict())):
    to = to3d_large.TopologyOptimization(N, 2 / N, **settings, **options)
    f, g = to.evaluate(x)
    g = g.copy()
    results[label] = (f, g, sum(to.cg_iterations['mechanical']) + sum(to.cg_iterations['adjoint']))
    print(f'{label}: f = {f:.12e}, CG iterations {results[label][2]}', flush=True)
    if 'fused' in label:
        for trial in range(2):
            d = np.random.default_rng(trial).standard_normal(N ** 3)
            step = 1e-4
            fd = (to.evaluate(x + step * d)[0] - to.evaluate(x - step * d)[0]) / (2 * step)
            ad = float(np.dot(g, d))
            print(f'  direction {trial}: adjoint {ad: .8e}  FD {fd: .8e}  rel diff {abs(ad - fd) / abs(fd):.1e}')
(f_d, g_d, _), (f_s, g_s, _) = results['fused, double multigrid'], results['fused, single multigrid']
f_h, g_h, _ = results['one-point + hourglass (stencils)']
print(f'double vs single multigrid: f rel diff {abs(f_d - f_s) / abs(f_d):.1e}, gradient rel diff '
      f'{np.linalg.norm(g_d - g_s) / np.linalg.norm(g_d):.1e}')
print(f'fused vs one-point element: f rel diff {abs(f_d - f_h) / abs(f_h):.1e}, gradient rel diff '
      f'{np.linalg.norm(g_d - g_h) / np.linalg.norm(g_h):.1e}  (different elements)')
