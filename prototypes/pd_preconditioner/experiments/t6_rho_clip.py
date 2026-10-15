"""T6 rho_clip sensitivity of PD on the cosine geometry with true voids (chi = inf)."""
import sys

import numpy as np

from common import RESULTS, row, solve, write_csv
from materials import cosine

d = int(sys.argv[1]) if len(sys.argv) > 1 else 1
rows = []
for n, p in ((64, 4), (256, 4), (256, 16), (256, 64), (256, 256)):
    rho = cosine(n, p, np.inf)
    for clip in (1e-4, 1e-6, 1e-8, 1e-10):
        rows.append(row(solve(rho, d, 'PD', rho_clip=clip), case='cosine', d=d, n=n, p=p, chi='inf',
                        method='PD', rho_clip=clip))
        print(rows[-1], flush=True)
    for method in ('G', 'GJ'):                     # for reference
        rows.append(row(solve(rho, d, method), case='cosine', d=d, n=n, p=p, chi='inf', method=method, rho_clip=''))
        print(rows[-1], flush=True)
kind = 'scalar' if d == 1 else 'elastic'
write_csv(f't6_rho_clip_{kind}.csv', rows)
