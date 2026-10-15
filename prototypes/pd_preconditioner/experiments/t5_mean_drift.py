"""T5 mean drift: cosine n = 256, p = 16, chi = 1e4. GJ and PD without mean subtraction in the preconditioner,
compared with Green at the same tolerance; relative errors of the displacement (raw and mean-subtracted) and of
the strain (B u)."""
import sys

import numpy as np

from common import COLORS, NAMES, RESULTS, SURFACE, TEXT_1, TEXT_2, plt, solve, style, write_csv
from fem import B
from materials import cosine

d = int(sys.argv[1]) if len(sys.argv) > 1 else 1
rho = cosine(256, 16, 1e4)
tolerances = (1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10)
rows = []


def norm(a):
    return np.linalg.norm(a)


for tol in tolerances:
    green = solve(rho, d, 'G', tol=tol, maxiter=20000, track_energy=False)['x']
    for method in ('GJ', 'PD'):
        x = solve(rho, d, method, tol=tol, maxiter=20000, subtract_mean=False, track_energy=False)['x']
        mean = x.mean(axis=(1, 2), keepdims=True)
        rows.append(dict(d=d, tol=tol, method=method,
                         displacement_error=f'{norm(x - green) / norm(green):.3e}',
                         displacement_error_mean_removed=f'{norm(x - mean - green) / norm(green):.3e}',
                         strain_error=f'{norm(B(x) - B(green)) / norm(B(green)):.3e}',
                         mean_of_solution=f'{np.abs(mean).max():.3e}'))
        print(rows[-1], flush=True)
kind = 'scalar' if d == 1 else 'elastic'
write_csv(f't5_mean_drift_{kind}.csv', rows)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=SURFACE)
fig.subplots_adjust(left=0.06, right=0.98, top=0.8, bottom=0.12, wspace=0.22)
fig.text(0.06, 0.94, f'T5 mean drift, {kind}: error relative to Green at the same tolerance', fontsize=14,
         fontweight='bold', color=TEXT_1)
fig.text(0.06, 0.89, 'cosine n = 256, p = 16, chi = 1e4; no mean subtraction inside GJ / PD', fontsize=10,
         color=TEXT_2)
for ax, key, title in zip(axes, ('displacement_error', 'displacement_error_mean_removed', 'strain_error'),
                          ('displacement', 'displacement, mean removed afterwards', 'strain B u')):
    for method in ('GJ', 'PD'):
        sel = [r for r in rows if r['method'] == method]
        ax.plot([r['tol'] for r in sel], [float(r[key]) for r in sel], color=COLORS[method], linewidth=2,
                marker='o', label=NAMES[method])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.invert_xaxis()
    ax.grid(axis='y', color='#e6e5e1', linewidth=0.8)
    style(ax, title, 'PCG tolerance', 'relative error' if key == 'displacement_error' else None)
axes[0].legend(frameon=False, fontsize=9.5)
fig.savefig(f'{RESULTS}/t5_mean_drift_{kind}.png', dpi=150, facecolor=SURFACE)
