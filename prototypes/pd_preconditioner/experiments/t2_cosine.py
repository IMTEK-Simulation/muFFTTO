"""T2 cosine geometry with stiff, matrix and soft/void phases; chi = 1e4 and inf."""
import sys

import numpy as np

from common import METHODS, RESULTS, iteration_grids, row, solve, write_csv
from materials import cosine

d = int(sys.argv[1]) if len(sys.argv) > 1 else 1
rows = []
for chi in (1e4, np.inf):
    for n in (16, 32, 64, 128, 256, 512):
        p = 4
        while p <= n:
            rho = cosine(n, p, chi)
            for method in METHODS:
                rows.append(row(solve(rho, d, method), case='cosine', d=d, n=n, p=p, chi=chi, method=method))
                print(rows[-1], flush=True)
            p *= 2
kind = 'scalar' if d == 1 else 'elastic'
write_csv(f't2_cosine_{kind}.csv', rows)
iteration_grids(rows, f'{RESULTS}/t2_cosine_{kind}.png', f'T2 cosine, {kind}: PCG iterations',
                'rho = 0.5 + 0.25 (cos 2pi(x1-x2) + cos 2pi(x1+x2)) + 1/chi on p x p data pixels; colour log scale',
                'chi', (1e4, np.inf))
