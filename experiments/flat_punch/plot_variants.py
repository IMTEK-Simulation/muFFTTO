"""
Flat punch variants on one mesh (smoothing strength alpha, gap thickness): per-pixel ratio simulation / exact near the
punch at the last increment (exact = half-space -sigma_yy at the Gauss points of the block's top pixel row), and p / P.

usage:  python plot_variants.py <run folder> ...      -> punch_variants_N<n>.png
"""
import importlib
import os
import re
import shutil
import sys
import tempfile

import numpy as np
import netCDF4
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
STYLES = [dict(color='#08306b', marker='o', ls='-'), dict(color='#c4441a', marker='s', ls='--'),
          dict(color='#4d9221', marker='D', ls=':'), dict(color='#7b3294', marker='^', ls='-.')]

fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout='constrained')
for folder, st in zip(sys.argv[1:], STYLES):
    tmp = os.path.join(tempfile.mkdtemp(), 'c.nc')
    shutil.copy(os.path.join(folder, 'contact_fracture.nc'), tmp)
    with netCDF4.Dataset(tmp) as ds:
        n = ds['phase'].shape[-1]
        os.environ['NNN'] = str(n)
        os.environ['B_PX'] = str(int(ds.getncattr('B_px')))
        os.environ['GAP_PX'] = str(int(ds.getncattr('gap_px')))
        P = float(ds['crown_force'][-1])
        p = np.asarray(ds['p_surface'][-1]) / P
        alpha = float(ds.getncattr('alpha'))
    import punch_geometry
    pg = importlib.reload(punch_geometry)
    h = 1.0 / n
    edges = (np.arange(n + 1) - n / 2) * h
    centres = 0.5 * (edges[:-1] + edges[1:])
    exact = pg.top_row_average(edges, 1.0)
    label = f'B = {pg.B:.3f} (B / depth {pg.B / pg.BLOCK_TOP:.2f}), $\\alpha$ = {alpha:g}, gap {pg.GAP_PX} px'
    near = np.abs(centres) < pg.B + 2 * h
    r = p / exact
    centre_err = np.median(np.abs(r - 1)[np.abs(centres) < 0.5 * pg.B])
    k = np.argmin(np.abs(centres - (pg.B - 0.5 * h)))
    axes[0].plot(centres[near] / pg.B, r[near], marker=st['marker'], color=st['color'], ls=st['ls'], ms=4,
                 label=f'{label}: centre {centre_err:.1%}, corner {r[k] - 1:+.0%}, beyond {r[k + 1] - 1:+.0%}')
    axes[1].stairs(p * pg.B, edges / pg.B, color=st['color'], ls=st['ls'], lw=1.6, label=f'{label} (P = {P:.3f})')
    print(f'{label}: P {P:.3f}  centre {centre_err:.4f}  corner {r[k] - 1:+.3f}  beyond {r[k + 1] - 1:+.3f}  '
          f'centre ratio {r[n // 2]:.4f}')
axes[0].axhline(1, color='k', lw=1)
axes[0].axvspan(-1, 1, color='0.95', zorder=0)
axes[0].set(xlabel='x / B (grey: under the punch)', ylabel='simulation / exact (as measured)', ylim=(0.6, 1.2),
            title='per-pixel ratio (corner pixels beyond the axis range)')
axes[0].legend(frameon=False, fontsize=9, loc='lower center')
xs = np.linspace(-0.999, 0.999, 600)
axes[1].plot(xs, 1.0 / (np.pi * np.sqrt(1 - xs ** 2)), color='k', lw=1, label='exact surface pressure')
axes[1].set(xlim=(-1.5, 1.5), ylim=(0, 1.6), xlabel='x / B', ylabel='p B / P',
            title='pressure in the block\'s top pixel row (dimensionless)')
axes[1].legend(frameon=False, fontsize=9, loc='upper center')
fig.suptitle(f'Flat punch, {n}$^2$: variants (punch width, smoothing strength, gap thickness)', fontsize=13)
out = os.path.join(HERE, f'punch_variants_N{n}.png')
fig.savefig(out, dpi=115)
print(f'wrote {out}')
