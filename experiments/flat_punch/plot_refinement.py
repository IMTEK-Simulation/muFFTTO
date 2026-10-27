"""
Mesh refinement of the flat punch (same physical geometry, gap one pixel, the same push beyond closure): at the last
increment of every run

    left    p / P per pixel of the block's top row, with the exact surface pressure (pointwise)
    middle  simulation / exact per pixel (exact = -sigma_yy of the half-space at the Gauss points of the top pixel row),
            vs x / B
    right   per mesh: error of the corner pixel, of the pixel beyond the corner, and of the force carried outside the
            punch in the top row

usage:  python plot_refinement.py <run folder> ...      -> punch_refinement.png
"""
import importlib
import os
import shutil
import sys
import tempfile

import numpy as np
import netCDF4
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
SHADES = {32: dict(color='#9ecae1', marker='s', ls=':'), 64: dict(color='#3d7fd0', marker='D', ls='--'),
          128: dict(color='#08306b', marker='o', ls='-'), 256: dict(color='#000000', marker='^', ls='-')}

rows = []
fig, axes = plt.subplots(1, 3, figsize=(19, 6), layout='constrained')
for folder in sys.argv[1:]:
    tmp = os.path.join(tempfile.mkdtemp(), 'c.nc')
    shutil.copy(os.path.join(folder, 'contact_fracture.nc'), tmp)
    with netCDF4.Dataset(tmp) as ds:
        n = ds['phase'].shape[-1]
        os.environ['NNN'] = str(n)
        os.environ['B_PX'] = str(int(ds.getncattr('B_px')))
        os.environ['GAP_PX'] = str(int(ds.getncattr('gap_px')))
        P = float(ds['crown_force'][-1])
        p = np.asarray(ds['p_surface'][-1]) / P
    import punch_geometry
    pg = importlib.reload(punch_geometry)
    h = 1.0 / n
    edges = (np.arange(n + 1) - n / 2) * h
    centres = 0.5 * (edges[:-1] + edges[1:])
    exact = pg.top_row_average(edges, 1.0)
    st = SHADES[n]
    axes[0].stairs(p, edges, color=st['color'], lw=1.6, ls=st['ls'], label=f'{n}$^2$ (P = {P:.3f})')
    near = np.abs(centres) < pg.B + 2 * h
    axes[1].plot(centres[near] / pg.B, p[near] / exact[near], color=st['color'], marker=st['marker'], ls=st['ls'], ms=4,
                 label=f'{n}$^2$')
    k_corner = np.argmin(np.abs(centres - (pg.B - 0.5 * h)))
    out = np.abs(centres) > pg.B
    rows.append((n, p[k_corner] / exact[k_corner] - 1, p[k_corner + 1] / exact[k_corner + 1] - 1,
                 (p[out] * h).sum() - (exact[out] * h).sum(), np.median(np.abs(p / exact - 1)[np.abs(centres) < 0.5 * pg.B])))
xs = np.linspace(-pg.B * 0.999, pg.B * 0.999, 600)
axes[0].plot(xs, 1.0 / (np.pi * np.sqrt(pg.B ** 2 - xs ** 2)), color='k', lw=1, label='exact surface pressure')
axes[0].set(xlim=(-1.6 * pg.B, 1.6 * pg.B), ylim=(0, 9), xlabel='x (relative to the punch axis)', ylabel='p / P',
            title='pressure in the block\'s top pixel row, last increment')
axes[0].legend(frameon=False, fontsize=9, loc='upper center')
axes[1].axhline(1, color='k', lw=1)
axes[1].axvspan(-1, 1, color='0.95', zorder=0)
axes[1].set(xlabel='x / B (grey: under the punch)', ylabel='simulation / exact (as measured)', ylim=(0, 2.4),
            title='per-pixel ratio near the punch (exact: half-space $-\\sigma_{yy}$ at the top-row Gauss points)')
axes[1].legend(frameon=False, fontsize=9)
r = np.array(sorted(rows))
ax = axes[2]
for col, c, name in ((1, '#b2182b', 'corner pixel (under the punch)'), (2, '#ef8a62', 'pixel beyond the corner'),
                     (3, '#3d7fd0', 'force outside the punch (/ P)'), (4, '#4d9221', 'central pixels (median)')):
    ax.loglog(r[:, 0], np.abs(r[:, col]), 'o-', color=c, lw=2, label=name)
ax.set_xticks(r[:, 0], [f'{int(v)}' for v in r[:, 0]])
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
ax.set(xlabel='pixels per side', ylabel='|error| (relative; force: fraction of P)', title='error vs mesh')
ax.grid(True, color='0.9', which='both')
ax.legend(frameon=False, fontsize=9)
fig.suptitle(f'Flat rigid punch, mesh refinement: 2B = {2 * pg.B:.3f}, block depth {pg.BLOCK_TOP:g}, gap 1 pixel, '
             f'push 0.00875 beyond closure', fontsize=13)
out = os.path.join(HERE, 'punch_refinement.png')
fig.savefig(out, dpi=115)
print(f'wrote {out}')
for row in r:
    print(f'N {int(row[0]):4d}: corner {row[1]:+.3f}  beyond {row[2]:+.3f}  outside force {row[3]:+.4f} P  centre {row[4]:.4f}')
