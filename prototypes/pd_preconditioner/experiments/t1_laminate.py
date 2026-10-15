"""T1 laminate: iterations of G, GJ, PD over n, p, chi (scalar: d=1, elasticity: d=2)."""
import sys

from common import METHODS, RESULTS, iteration_grids, row, solve, write_csv
from materials import laminate

d = int(sys.argv[1]) if len(sys.argv) > 1 else 1
rows = []
for chi in (10.0, 1e4):
    for n in (16, 32, 64, 128, 256, 512):
        p = 4
        while p <= n:
            rho = laminate(n, p, chi)
            for method in METHODS:
                rows.append(row(solve(rho, d, method), case='laminate', d=d, n=n, p=p, chi=chi, method=method))
                print(rows[-1], flush=True)
            p *= 2
kind = 'scalar' if d == 1 else 'elastic'
write_csv(f't1_laminate_{kind}.csv', rows)
iteration_grids(rows, f'{RESULTS}/t1_laminate_{kind}.png', f'T1 laminate, {kind}: PCG iterations',
                'rho varies along x1 in p layers, chi ... 1; ||r_k|| <= 1e-5 ||r_0||; colour log scale',
                'chi', (10.0, 1e4))
