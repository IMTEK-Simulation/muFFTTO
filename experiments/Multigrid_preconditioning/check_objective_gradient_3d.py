"""
Finite-difference check of the 3D TO gradient (copy of example_3D_elasticity_TO.py) for ELEMENT (env):
trilinear_hexahedron (control) and trilinear_hexahedron_1Q_hg (one-point + hourglass control, whose design dependence
enters the adjoint term through muFFTTO.hourglass_control.design_derivative). g.d vs central differences along random
directions, smooth random design in (0.15, 0.85), 8^3, contrast 1e3, MG mechanics.
    ELEMENT=trilinear_hexahedron_1Q_hg python check_objective_gradient_3d.py
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
N = 8
os.environ.update({'NPIX': str(N), 'ETA': str(2 / N), 'SOFT_EXP': '3', 'PF_NORMALIZED': '1', 'PF_WEIGHT': '0.2',
                   'CG_MODE': 'energy', 'CG_DELAY': '0', 'SHOW_PLOTS': '0', 'JAX_PLATFORMS': 'cpu', 'TARGET_SCALE': '0.3'})
sys.argv = [sys.argv[0], 'MG']
sys.path.insert(0, str(HERE))
import numpy as np
import example_3D_elasticity_TO as TO
if os.environ.get("TIGHT") == "1":
    TO.cg_setup["cg_tol"] = 1e-12          # near-exact solves (FD check)

rng = np.random.default_rng(5)
k = np.fft.fftfreq(N, 1 / N)
KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
spec = (np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2) <= 2) * (rng.standard_normal((N, N, N)) + 1j * rng.standard_normal((N, N, N)))
f0 = np.real(np.fft.ifftn(spec))
x = (0.5 + 0.35 * f0 / np.abs(f0).max()).ravel()
fun = lambda z: TO.objective_function_multiple_load_cases(z)
f, g = fun(x)
print(f'ELEMENT {TO.element_type}: f = {f:.8e}', flush=True)
for trial in range(2):
    d = rng.standard_normal(N ** 3)
    eps = 1e-4
    fd = (fun(x + eps * d)[0] - fun(x - eps * d)[0]) / (2 * eps)
    ad = float(np.dot(g, d))
    print(f'  direction {trial}: adjoint {ad: .8e}  FD {fd: .8e}  rel diff {abs(ad - fd) / abs(fd):.1e}', flush=True)
