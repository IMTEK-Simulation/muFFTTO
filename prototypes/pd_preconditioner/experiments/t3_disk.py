"""T3 smoothed disk: n = 256, chi = 1e4, iterations vs number of smoothing steps i."""
import sys

import numpy as np

from common import COLORS, METHODS, NAMES, RESULTS, SURFACE, TEXT_1, TEXT_2, plt, row, solve, style, write_csv
from materials import disk

d = int(sys.argv[1]) if len(sys.argv) > 1 else 1
n, chi = 256, 1e4
steps = (0, 1, 2, 4, 8, 16, 32, 64, 128)
rows = []
for i in steps:
    rho = disk(n, chi, i)
    for method in METHODS:
        rows.append(row(solve(rho, d, method), case='disk', d=d, n=n, chi=chi, smoothing=i, method=method,
                        rho_min=f'{rho.min():.3e}'))
        print(rows[-1], flush=True)
kind = 'scalar' if d == 1 else 'elastic'
write_csv(f't3_disk_{kind}.csv', rows)

fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
fig.subplots_adjust(left=0.1, right=0.97, top=0.82, bottom=0.12)
fig.text(0.1, 0.94, f'T3 smoothed disk, {kind}: PCG iterations', fontsize=14, fontweight='bold', color=TEXT_1)
fig.text(0.1, 0.89, f'n = {n}, chi = {chi:g}, radius 0.25; disk smoothed i times with [1 2 1]^T[1 2 1]/16',
         fontsize=10, color=TEXT_2)
x = np.arange(len(steps))
for method in METHODS:
    y = [r['iterations'] for r in rows if r['method'] == method]
    ax.plot(x, y, color=COLORS[method], linewidth=2, marker='o', markersize=6, label=NAMES[method])
ax.set_xticks(x, [str(i) for i in steps])
ax.set_yscale('log')
ax.grid(axis='y', color='#e6e5e1', linewidth=0.8)
style(ax, None, 'smoothing steps i', 'PCG iterations (log)')
ax.legend(frameon=False, fontsize=10)
fig.savefig(f'{RESULTS}/t3_disk_{kind}.png', dpi=150, facecolor=SURFACE)
