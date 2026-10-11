"""
Finite-difference check of the objective gradient of example_2D_finite_strain_TO.py (default) or of
another example of this folder with the same interface (objective, rho_nodal, ...):
    python check_sensitivity.py [example_2D_finite_strain_TO_curve]

Compares the adjoint directional derivative  g . v  with the central difference
(F(rho + h v) - F(rho - h v)) / (2 h)  for a random design rho and a random direction v,
for several h (the error should fall like h^2 until the solver tolerances are reached).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
import numpy as np

# module-level setup only (the optimizer runs under __main__)
to = importlib.import_module(sys.argv[1] if len(sys.argv) > 1 else 'example_2D_finite_strain_TO')

# tight state and adjoint solves for the check (1e-12 is at round-off: CG breaks down)
to.adjoint_rtol = 1e-10
if hasattr(to, 'newton_rtol'):
    to.newton_rtol = 1e-11                         # Newton-CG examples
if hasattr(to, 'tr_gtol'):
    to.tr_gtol = 1e-11                             # trust-region examples

rng = np.random.default_rng(1)
shape = to.rho_nodal.s.shape
rho = 0.3 + 0.6 * rng.random(shape).ravel()       # away from the bounds
direction = rng.standard_normal(shape).ravel()

f0, gradient = to.objective(rho)
derivative = to.discretization.communicator.sum(np.dot(gradient, direction))
to.root_print(f'\nadjoint directional derivative {derivative:.10e}')
for h in [1e-2, 1e-3, 1e-4, 1e-5]:
    f_plus = to.objective(rho + h * direction)[0]
    f_minus = to.objective(rho - h * direction)[0]
    fd = (f_plus - f_minus) / (2 * h)
    to.root_print(f'h {h:.0e}: finite difference {fd:.10e}, relative error {abs(fd - derivative) / abs(derivative):.2e}')
