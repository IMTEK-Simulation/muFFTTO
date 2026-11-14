"""
Finite-difference check of the TO gradient (2D copy) for OBJECTIVE = energy and stress: g.d vs central differences of
f along random directions d, smooth random design rho in (0.15, 0.85) on 32^2, contrast 1e3, MG mechanics.
    OBJECTIVE=energy python check_objective_gradient.py
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.update({'NPIX': '32', 'ETA': str(2 / 32), 'SOFT_EXP': '3', 'PF_NORMALIZED': '1', 'PF_WEIGHT': '0.2',
                   'CG_MODE': 'energy', 'CG_DELAY': '0', 'SHOW_PLOTS': '0', 'JAX_PLATFORMS': 'cpu',
                   'TARGET_SCALE': os.environ.get('TARGET_SCALE', '0.3')})
sys.argv = [sys.argv[0], 'MG']
sys.path.insert(0, str(HERE))
import numpy as np
import example_2D_elasticity_TO as TO

N = 32
rng = np.random.default_rng(3)
k = np.fft.fftfreq(N, 1 / N)
KX, KY = np.meshgrid(k, k, indexing='ij')
f0 = np.real(np.fft.ifft2((np.hypot(KX, KY) <= 3) * (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))))
x = (0.5 + 0.35 * f0 / np.abs(f0).max()).ravel()
fun = lambda z: TO.objective_function_multiple_load_cases(z)
f, g = fun(x)
print(f'OBJECTIVE {TO.objective_mode}: f = {f:.8e}')
for trial in range(3):
    d = rng.standard_normal(N * N)
    for eps in (1e-3, 1e-4):
        fp, _ = fun(x + eps * d)
        fm, _ = fun(x - eps * d)
        fd = (fp - fm) / (2 * eps)
        ad = float(np.dot(g, d))
        print(f'  direction {trial} eps {eps:.0e}: adjoint {ad: .8e}  FD {fd: .8e}  rel diff {abs(ad - fd) / abs(fd):.1e}', flush=True)
