"""
Disc between two rigid plates: errors vs the load P, from every increment of one run (finite-strain effect at larger
loads vs discretization): ellipse-fit contact half-width a and peak pressure p0 vs Hertz, centre stresses vs the
Brazilian solution.

usage:  python plot_disc_sweep.py <run folder>      -> <run folder>/disc_sweep.png
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import netCDF4
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

folder = sys.argv[1]
tmp = os.path.join(tempfile.mkdtemp(), 'c.nc')
shutil.copy(os.path.join(folder, 'contact_fracture.nc'), tmp)
ds = netCDF4.Dataset(tmp)
n = ds['phase'].shape[-1]
os.environ['NNN'] = str(n)
os.environ['GRADE'] = str(float(ds.getncattr('grade')))
os.environ['GAP'] = str(float(ds.getncattr('gap')))
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import disc_geometry as dg  # noqa: E402

P = np.asarray(ds['crown_force'][:])
x_face = np.asarray(ds.getncattr('x_face')) - dg.CENTRE_X
p_bottom = np.asarray(ds['f_bottom'][:]) / np.asarray(ds.getncattr('width_face'))
rows = [n // 2 - 1, n // 2]
k0 = n // 2                                               # pixel column right of the centre (x = h'/2)
rows_out = []
for f in np.nonzero(P > 0.03)[0]:
    a_h, p0_h = dg.hertz(P[f])
    m = p_bottom[f] > 0.05 * p_bottom[f].max()
    if m.sum() < 3:
        continue
    (p0, a, _), _ = curve_fit(lambda x, p0, a, x0: p0 * np.sqrt(np.clip(1 - ((x - x0) / a) ** 2, 0, None)),
                              x_face[m], p_bottom[f][m], p0=[p0_h, a_h, 0])
    sxx = np.asarray(ds['sxx'][f]).reshape(n, n)[k0, rows].mean()
    syy = np.asarray(ds['syy'][f]).reshape(n, n)[k0, rows].mean()
    ex, ey = dg.brazilian_horizontal(np.array([0.0]), P[f])
    rows_out.append((P[f], a / a_h - 1, p0 / p0_h - 1, sxx / ex[0] - 1, syy / ey[0] - 1, p0_h / 100.0,
                     np.sum(np.abs(x_face) < a_h)))
r = np.array(rows_out)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), layout='constrained')
ax = axes[0]
ax.plot(r[:, 0], 100 * r[:, 1], 'o-', color='#c4441a', ms=4, label='contact half-width a (ellipse fit)')
ax.plot(r[:, 0], 100 * r[:, 2], 's--', color='#3d7fd0', ms=4, mfc='white', label='peak pressure $p_0$ (ellipse fit)')
ax2 = ax.twinx()
ax2.plot(r[:, 0], r[:, 6], color='0.6', lw=1, ls=':', drawstyle='steps-mid')
ax2.set_ylabel('nodes across 2a (dotted)', color='0.4')
ax.axhline(0, color='k', lw=0.8)
ax.set(xlabel='force per unit length P', ylabel='error vs Hertz (%)', title='contact vs Hertz (rigid plate)')
ax.legend(frameon=False, fontsize=9, loc='upper right')
ax.grid(True, color='0.9')
ax = axes[1]
ax.plot(r[:, 0], 100 * r[:, 3], 'o-', color='#b2182b', ms=4, label='$\\sigma_{xx}$ at the centre')
ax.plot(r[:, 0], 100 * r[:, 4], 's--', color='#3d7fd0', ms=4, mfc='white', label='$\\sigma_{yy}$ at the centre')
ax.axhline(0, color='k', lw=0.8)
ax.set(xlabel='force per unit length P', ylabel='error vs Brazilian (%)',
       title='disc centre vs Brazilian (pixel next to the centre)')
ax.legend(frameon=False, fontsize=9)
ax.grid(True, color='0.9')
fig.suptitle(f'Disc between rigid plates, {n}$^2$: errors vs load (one run, every increment)', fontsize=13)
out = os.path.join(folder, 'disc_sweep.png')
fig.savefig(out, dpi=115)
print(f'wrote {out}')
for row in r[::3]:
    print(f'P {row[0]:.3f}  a {row[1]:+.3f}  p0 {row[2]:+.3f}  sxx {row[3]:+.3f}  syy {row[4]:+.3f}  '
          f'p0/E {row[5]:.3f}  nodes {int(row[6])}')
