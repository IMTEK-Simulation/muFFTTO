"""T10 linear test of the PD applicability condition C(x) = rho(x) C_ref: sharp disk (n = 256, radius 0.25),
inclusion (kappa, mu) = (2, 1) inside, matrix outside with (a) (20, 2000): different Poisson ratio, as in the Newton
tangents, (b) (4000, 2000): same kappa/mu as the inclusion, i.e. C = rho C_inc. Macroscopic load E = [1, 1, 1] and a
random mean-free rhs. Prediction: PD ~ Green (~80) in (a), PD back to 5-10 in (b)."""
import numpy as np

from common import write_csv
from fem import rhs
from nonlinear import linear_isotropic
from pcg import pcg
from precond import (make_green, make_green_jacobi, make_primal_dual_modal, make_primal_dual_tangent,
                     new_counter)

n, d = 256, 2
x = (np.arange(n) + 0.5) / n
x1, x2 = np.meshgrid(x, x, indexing='ij')
inside = (x1 - 0.5) ** 2 + (x2 - 0.5) ** 2 < 0.25 ** 2
C_inc = linear_isotropic(2.0, 1.0)
rows = []
for label, (kappa, mu) in (('matrix (20, 2000): different Poisson ratio', (20.0, 2000.0)),
                           ('matrix (4000, 2000): same Poisson ratio', (4000.0, 2000.0))):
    C_mat = linear_isotropic(kappa, mu)
    C = np.where(inside[None, None], C_inc[:, :, None, None], C_mat[:, :, None, None])
    f_macro = rhs(C, np.array([1.0, 1.0, 1.0]), d, n)
    f_random = np.random.default_rng(0).normal(size=(d, n, n))
    f_random -= f_random.mean(axis=(1, 2), keepdims=True)
    for load, f in (('macro E', f_macro), ('random rhs', f_random)):
        out = {}
        for method in ('G', 'GJ', 'PD_pointwise', 'PD_modal'):
            c = new_counter()
            M = {'G': lambda: make_green(C_inc, d, n, c), 'GJ': lambda: make_green_jacobi(C, np.eye(3), d, n, c),
                 'PD_pointwise': lambda: make_primal_dual_tangent(C, C_inc, d, n, c),
                 'PD_modal': lambda: make_primal_dual_modal(C, C_inc, d, n, c, auto_reference=True)}[method]()
            result = pcg(C, f, M, c, maxiter=5000)
            out[method] = result['iterations']
            rows.append(dict(material=label, load=load, method=method, iterations=result['iterations'],
                             it_energy=result['it_energy'], ffts=result['counter']['fft']))
        print(f'{label:45s} {load:10s} ' + ', '.join(f'{m} {v}' for m, v in out.items()), flush=True)
write_csv('t10_poisson_ratio.csv', rows)
