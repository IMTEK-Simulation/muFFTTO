"""T1b laminate with a random mean-free right-hand side (fixed seed) instead of the macroscopic load: tests
prediction 2 at the operator level (M^-1 K = I - rank <= d would give <= 3 / ~5 iterations for ANY rhs)."""
import sys

import numpy as np

from common import METHODS, RESULTS, iteration_grids, problem, row, write_csv
from materials import laminate
from pcg import pcg
from precond import make_green, make_green_jacobi, make_primal_dual, new_counter

d = int(sys.argv[1]) if len(sys.argv) > 1 else 1
rows = []
for chi in (10.0, 1e4):
    for n in (16, 32, 64, 128, 256):
        rng = np.random.default_rng(0)
        f = rng.normal(size=(d, n, n))
        f -= f.mean(axis=(1, 2), keepdims=True)
        p = 4
        while p <= n:
            rho = laminate(n, p, chi)
            C0, C, _ = problem(rho, d)
            for method in METHODS:
                counter = new_counter()
                M = {'G': lambda: make_green(C0, d, n, counter),
                     'GJ': lambda: make_green_jacobi(C, C0, d, n, counter),
                     'PD': lambda: make_primal_dual(rho, C0, C0, d, n, counter)}[method]()
                rows.append(row(pcg(C, f, M, counter), case='laminate random rhs', d=d, n=n, p=p, chi=chi,
                                method=method))
                print(rows[-1], flush=True)
            p *= 2
kind = 'scalar' if d == 1 else 'elastic'
write_csv(f't1b_laminate_random_rhs_{kind}.csv', rows)
iteration_grids(rows, f'{RESULTS}/t1b_laminate_random_rhs_{kind}.png',
                f'T1b laminate with a random right-hand side, {kind}: PCG iterations',
                'same laminate as T1, rhs = fixed random mean-free vector; ||r_k|| <= 1e-5 ||r_0||; colour log scale',
                'chi', (10.0, 1e4))
