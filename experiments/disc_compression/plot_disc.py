"""
Disc between two rigid plates against the exact solutions.

    left    force vs total approach of the plates (top and bottom plate forces: equal by equilibrium)
    middle  contact pressure on both plate faces from the nodal reactions (exactly at the surface), at three loads, with
            Hertz (rigid plate) at the same P; the ellipse fit (a, p0) of the last one is quoted
    right   sigma_xx and sigma_yy along the horizontal diameter (the two pixel rows next to it) at the last load, with
            the Brazilian solution for the same P

usage:  python plot_disc.py <run folder>      -> <run folder>/disc_vs_exact.png
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

u = np.asarray(ds['crown_u'][:])
P = np.asarray(ds['crown_force'][:])
fb, ft = np.asarray(ds['f_bottom'][:]), np.asarray(ds['f_top'][:])
x_face = np.asarray(ds.getncattr('x_face')) - dg.CENTRE_X
width = np.asarray(ds.getncattr('width_face'))
p_bottom, p_top = fb / width, -ft / width
P_bottom, P_top = fb.sum(axis=1), -ft.sum(axis=1)
# physical pixel centres
d_g = np.asarray(ds['grid_displacement'][0]).reshape(2, n, n)
X = np.stack(np.meshgrid(np.arange(n) / n, np.arange(n) / n, indexing='ij'))
ext = np.pad(X + d_g, ((0, 0), (0, 1), (0, 1)), mode='wrap')
ext[0, -1, :] += 1.0
ext[1, :, -1] += 1.0
centre = 0.25 * (ext[:, :-1, :-1] + ext[:, 1:, :-1] + ext[:, :-1, 1:] + ext[:, 1:, 1:])
phase = np.asarray(ds['phase'][0]).reshape(n, n)
f_last = len(P) - 1
order = np.argsort(x_face)


def ellipse(x, p0, a, x0):
    return p0 * np.sqrt(np.clip(1 - ((x - x0) / a) ** 2, 0, None))


fig, axes = plt.subplots(1, 3, figsize=(19, 6), layout='constrained')
ax = axes[0]
ax.plot(u, P_top, 'o-', color='#3d7fd0', ms=3, label='top plate')
ax.plot(u, P_bottom, 's--', color='#c4441a', ms=3, mfc='white', label='bottom plate')
ax.axvline(2 * dg.GAP, color='0.6', ls=':')
ax.text(2 * dg.GAP, 0.9 * P.max(), ' both gaps closed', fontsize=9, color='0.4')
ax.set(xlabel='total approach of the plates', ylabel='force per unit length P',
       title=f'load-displacement: |P_top - P_bottom| / P = {abs(P_top[-1] - P_bottom[-1]) / P[-1]:.1e}')
ax.legend(frameon=False, fontsize=9)
ax.grid(True, color='0.9')

ax = axes[1]
contact = np.nonzero(P > 0.05 * P.max())[0]
SHADES = ['#9ecae1', '#3d7fd0', '#08306b']
for f, c in zip(contact[np.linspace(0, len(contact) - 1, 3).astype(int)], SHADES):
    a, p0 = dg.hertz(P[f])
    xs = np.linspace(-a, a, 200)
    ax.plot(xs, 2 * P[f] / (np.pi * a ** 2) * np.sqrt(a ** 2 - xs ** 2), color='k', lw=1, alpha=0.7)
    ax.plot(x_face[order], p_bottom[f][order], 'o-', color=c, ms=4, lw=1.4, label=f'bottom, P = {P[f]:.3f}')
    ax.plot(x_face[order], p_top[f][order], 'x', color=c, ms=7, label=f'top, P = {P[f]:.3f}')
a_h, p0_h = dg.hertz(P[f_last])
m = p_bottom[f_last] > 0.05 * p_bottom[f_last].max()
(p0_fit, a_fit, _), _ = curve_fit(ellipse, x_face[m], p_bottom[f_last][m], p0=[p0_h, a_h, 0])
ax.set(xlim=(-2.2 * a_h, 2.2 * a_h), xlabel='x (relative to the axis)', ylabel='contact pressure p (nodal reactions)',
       title=f'contact pressure at both plates (black: Hertz, rigid plate)\nlast load, ellipse fit: '
             f'a {a_fit / a_h - 1:+.1%}, $p_0$ {p0_fit / p0_h - 1:+.1%} vs Hertz')
ax.legend(frameon=False, fontsize=8)
ax.grid(True, color='0.9')

ax = axes[2]
rows = [n // 2 - 1, n // 2]                                # the two pixel rows next to the horizontal diameter
xc = centre[0][:, rows].mean(axis=1) - dg.CENTRE_X
inside = (phase[:, rows] > 0).all(axis=1) & (np.abs(xc) < dg.R)
sxx = np.asarray(ds['sxx'][f_last]).reshape(n, n)[:, rows].mean(axis=1)
syy = np.asarray(ds['syy'][f_last]).reshape(n, n)[:, rows].mean(axis=1)
xs = np.linspace(-dg.R, dg.R, 300)
ex, ey = dg.brazilian_horizontal(xs, P[f_last])
ax.plot(xs, ex, color='#b2182b', lw=1.2, label='$\\sigma_{xx}$ Brazilian')
ax.plot(xs, ey, color='#3d7fd0', lw=1.2, label='$\\sigma_{yy}$ Brazilian')
ax.plot(xc[inside], sxx[inside], 'o', color='#b2182b', ms=5, mfc='white', label='$\\sigma_{xx}$ simulation')
ax.plot(xc[inside], syy[inside], 's', color='#3d7fd0', ms=5, mfc='white', label='$\\sigma_{yy}$ simulation')
ex0, ey0 = dg.brazilian_horizontal(np.array([0.0]), P[f_last])
k0 = np.argmin(np.abs(xc))
ax.axhline(0, color='0.5', lw=0.8)
ax.set(xlabel='x on the horizontal diameter', ylabel='Cauchy stress',
       title=f'Brazilian stress, last load (P = {P[f_last]:.3f}): centre $\\sigma_{{xx}}$ {sxx[k0] / ex0[0] - 1:+.1%}, '
             f'$\\sigma_{{yy}}$ {syy[k0] / ey0[0] - 1:+.1%}')
ax.legend(frameon=False, fontsize=9)
ax.grid(True, color='0.9')
fig.suptitle(f'Disc between two rigid plates vs exact, {n}$^2$, fitted grid' +
             (f', graded $\\beta$ = {dg.GRADE:g}' if dg.GRADE else '') + f': R = {dg.R:.3f}, gap {dg.GAP:g}', fontsize=13)
out = os.path.join(folder, 'disc_vs_exact.png')
fig.savefig(out, dpi=115)
print(f'wrote {out}')
print(f'P {P[-1]:.4f} top {P_top[-1]:.4f} bottom {P_bottom[-1]:.4f}; a fit {a_fit / a_h - 1:+.3f} p0 fit {p0_fit / p0_h - 1:+.3f}; '
      f'centre sxx {sxx[k0]:.4f} vs {ex0[0]:.4f}, syy {syy[k0]:.4f} vs {ey0[0]:.4f}; x of centre pixel {xc[k0]:.4f}')
