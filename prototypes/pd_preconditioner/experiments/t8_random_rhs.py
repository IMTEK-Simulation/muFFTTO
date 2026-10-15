"""T8 random mean-free right-hand side (fixed seed) on the cosine (chi = 1e4, all p) and on the disk (chi = 1e4,
smoothing 0 and 8): is PD flat in n for a general rhs (proxy for Newton residuals)?"""
import sys

import numpy as np

from common import METHODS, RESULTS, iteration_grids, problem, row, write_csv
from materials import cosine, disk
from pcg import pcg
from precond import make_green, make_green_jacobi, make_primal_dual, new_counter

d = int(sys.argv[1]) if len(sys.argv) > 1 else 1


def run(rho, f):
    n = rho.shape[0]
    C0, C, _ = problem(rho, d)
    out = {}
    for method in METHODS:
        counter = new_counter()
        M = {'G': lambda: make_green(C0, d, n, counter), 'GJ': lambda: make_green_jacobi(C, C0, d, n, counter),
             'PD': lambda: make_primal_dual(rho, C0, C0, d, n, counter)}[method]()
        out[method] = pcg(C, f, M, counter)
    return out


rows = []
for n in (16, 32, 64, 128, 256, 512):
    f = np.random.default_rng(0).normal(size=(d, n, n))
    f -= f.mean(axis=(1, 2), keepdims=True)
    p = 4
    while p <= n:
        for method, result in run(cosine(n, p, 1e4), f).items():
            rows.append(row(result, case='cosine random rhs', d=d, n=n, p=p, chi=1e4, method=method))
            print(rows[-1], flush=True)
        p *= 2
    for i in (0, 8):
        for method, result in run(disk(n, 1e4, i), f).items():
            rows.append(row(result, case=f'disk i={i} random rhs', d=d, n=n, p='', chi=1e4, method=method))
            print(rows[-1], flush=True)
kind = 'scalar' if d == 1 else 'elastic'
write_csv(f't8_random_rhs_{kind}.csv', rows)
cos_rows = [r for r in rows if r['case'] == 'cosine random rhs']
iteration_grids(cos_rows, f'{RESULTS}/t8_cosine_random_rhs_{kind}.png',
                f'T8 cosine with a random right-hand side, {kind}: PCG iterations',
                'chi = 1e4; rhs = fixed random mean-free vector; ||r_k|| <= 1e-5 ||r_0||', 'chi', (1e4,))
