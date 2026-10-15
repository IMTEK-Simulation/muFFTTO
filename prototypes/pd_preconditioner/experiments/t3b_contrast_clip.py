"""T3' disk, n = 256: contrast chi in {1e4, 1e6, 1e8, 1e12} x smoothing i x rho_clip for PD (G and GJ for reference).
rho_clip is relative to max(rho) = 1; 'none' = 1e-300 (no clipping)."""
import sys

from common import RESULTS, row, solve, write_csv
from materials import disk

d = int(sys.argv[1]) if len(sys.argv) > 1 else 1
chis = [float(c) for c in sys.argv[2:]] or [1e4, 1e6, 1e8, 1e12]
n = 256
clips = (1e-2, 1e-3, 1e-4, 1e-6, 1e-300)
rows = []
for chi in chis:
    for i in (0, 1, 2, 4, 8, 16, 32, 64):
        rho = disk(n, chi, i)
        for method in ('G', 'GJ'):
            rows.append(row(solve(rho, d, method), d=d, n=n, chi=chi, smoothing=i, method=method, rho_clip=''))
            print(rows[-1], flush=True)
        for clip in clips:
            rows.append(row(solve(rho, d, 'PD', rho_clip=clip), d=d, n=n, chi=chi, smoothing=i, method='PD',
                            rho_clip='none' if clip < 1e-100 else clip))
            print(rows[-1], flush=True)
kind = 'scalar' if d == 1 else 'elastic'
write_csv(f't3b_contrast_clip_{kind}_' + '_'.join(f'{c:g}' for c in chis) + '.csv', rows)
