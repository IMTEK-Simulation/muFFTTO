"""
to3d_large.TopologyOptimization against the reference script example_3D_elasticity_TO.py: objective and gradient at
the same design (N^3, CPU, near-exact solves), and a finite-difference check of the gradient.
    python fable_check_to3d_large.py [N, default 8] [element, default trilinear_hexahedron_1Q_hg] [objective]
"""
import os
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
element = sys.argv[2] if len(sys.argv) > 2 else 'trilinear_hexahedron_1Q_hg'
objective = sys.argv[3] if len(sys.argv) > 3 else 'energy'
os.environ.update({'NPIX': str(N), 'ETA': str(2 / N), 'SOFT_EXP': '3', 'PF_NORMALIZED': '1', 'PF_WEIGHT': '0.2',
                   'CG_MODE': 'energy', 'CG_DELAY': '0', 'SHOW_PLOTS': '0', 'JAX_PLATFORMS': 'cpu', 'TARGET_SCALE': '0.3',
                   'ELEMENT': element, 'HG_EPS': '0.3', 'OBJECTIVE': objective})
sys.argv = [sys.argv[0], 'MG']
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import example_3D_elasticity_TO as reference
import to3d_large

reference.cg_setup['cg_tol'] = 1e-12
rng = np.random.default_rng(5)
k = np.fft.fftfreq(N, 1 / N)
KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
spectrum = (np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2) <= 2) * (rng.standard_normal((N,) * 3) + 1j * rng.standard_normal((N,) * 3))
smooth = np.real(np.fft.ifftn(spectrum))          # test design only (not an operator of the method)
x = (0.5 + 0.35 * smooth / np.abs(smooth).max()).ravel()

to = to3d_large.TopologyOptimization(N, 2 / N, element=element, objective=objective, target_scale=0.3, cg_tol=1e-12,
                                     verbose=False)
f_ref, g_ref = reference.objective_function_multiple_load_cases(x)
f, g = to.evaluate(x)
g = g.copy()
print(f'{element} {objective} N {N}: f reference {f_ref:.14e}  to3d_large {f:.14e}  rel diff {abs(f - f_ref) / abs(f_ref):.1e}')
print(f'  gradient: rel diff {np.linalg.norm(g - g_ref) / np.linalg.norm(g_ref):.1e}')
for trial in range(2):
    d = rng.standard_normal(N ** 3)
    step = 1e-4
    fd = (to.evaluate(x + step * d)[0] - to.evaluate(x - step * d)[0]) / (2 * step)
    ad = float(np.dot(g, d))
    print(f'  direction {trial}: adjoint {ad: .8e}  FD {fd: .8e}  rel diff {abs(ad - fd) / abs(fd):.1e}')
