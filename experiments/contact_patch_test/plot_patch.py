"""
Patch test against the exact solution: two blocks in series, free sides (sigma_xx = 0), plane strain.

    top     left: force P vs the top displacement u, with the exact line P = E' W (u - u_c) / H (E' = E / (1 - nu^2),
            W the block width, H the total block height, u_c where the gap is closed); middle: the same, log scale (force
            before contact = parasitic stiffness of the medium); right: max stress error in the blocks vs u
    bottom  stress errors per pixel at the last increment (exact pixels): sigma_yy / (-p) - 1, sigma_xx / p, sigma_xy / p

usage:  python plot_patch.py <run folder>      -> <run folder>/patch_vs_exact.png
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

folder = sys.argv[1]
tmp = os.path.join(tempfile.mkdtemp(), 'c.nc')
shutil.copy(os.path.join(folder, 'contact_fracture.nc'), tmp)
ds = netCDF4.Dataset(tmp)
n = ds['phase'].shape[-1]
h = 1.0 / n
os.environ['NNN'] = str(n)
os.environ['GAP_PX'] = str(int(ds.getncattr('gap_px')))
os.environ['E_RATIO'] = str(float(ds.getncattr('E_ratio')))
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import patch_geometry as pg  # noqa: E402

E, NU = 100.0, 0.3
u = np.asarray(ds['crown_u'][:])
P = np.asarray(ds['crown_force'][:])
s = {k: np.asarray(ds[k][:]) for k in ('sxx', 'syy', 'sxy')}
centres = np.stack(np.meshgrid((np.arange(n) + 0.5) * h, (np.arange(n) + 0.5) * h, indexing='ij'))
lab = pg.labels(centres)
solid = lab > 0
W = 1.0 - 2 * pg.SIDE
H_low = pg.LOWER_TOP
H_up = pg.UPPER_TOP - pg.LOWER_TOP - pg.GAP
E_low, E_up = E / (1 - NU ** 2), pg.E_RATIO * E / (1 - NU ** 2)    # plane strain, free sides
k_series = W / (H_low / E_low + H_up / E_up)                       # force per unit closure
contact = P > 1e-2 * P.max()
# u_c from the last two increments (the exact line has slope k_series); compare the measured slope with k_series
slope = (P[-1] - P[-2]) / (u[-1] - u[-2])
u_c = u[-1] - P[-1] / k_series
# exact Cauchy stress: the force over the current (widened) width of each block row; first-PK p = P / W
ux = np.asarray(ds['u_fluc'][:]).reshape(len(u), 2, n, n)[:, 0]
left, right = int(round(pg.SIDE / h)), n - int(round(pg.SIDE / h))     # the block's edge node columns
row_width = W + 0.5 * (ux[:, right, :] + np.roll(ux[:, right, :], -1, axis=-1)
                       - ux[:, left, :] - np.roll(ux[:, left, :], -1, axis=-1))   # [frame, pixel row]
p = P / W
p_cauchy = P[:, None] / row_width                                         # [frame, pixel row]
exact = {'syy': -p_cauchy[:, None, :], 'sxx': 0.0, 'sxy': 0.0}
err = {k: np.array([(np.abs(s[k][f] - (exact[k][f] if k == 'syy' else 0.0)) / max(p[f], 1e-30))[solid].max()
                    for f in range(len(u))]) for k in s}

fig = plt.figure(figsize=(18, 11))
ax = fig.add_axes((0.05, 0.55, 0.26, 0.34))
ax.plot(u, P, 'o', color='#3d7fd0', ms=4, label='simulation')
uu = np.linspace(u_c, u[-1], 50)
ax.plot(uu, k_series * (uu - u_c), color='k', lw=1.2, label=f"exact slope E'W/H = {k_series:.1f}")
ax.axvline(pg.GAP, color='0.6', ls=':', lw=1)
ax.text(pg.GAP, 0.9 * P.max(), f' gap {pg.GAP:.4f}', fontsize=9, color='0.4')
ax.set(xlabel='top displacement u', ylabel='force per unit length P',
       title=f'load-displacement: last slope {slope:.1f} vs linear {k_series:.1f} ({slope / k_series - 1:+.1%})\n'
             f'(neo-Hooke stiffens in compression: ~ +2 % at 1 % strain)')
ax.legend(frameon=False, fontsize=9)
ax = fig.add_axes((0.37, 0.55, 0.26, 0.34))
ax.semilogy(u, np.maximum(P, 1e-12), 'o-', color='#3d7fd0', ms=3)
ax.axvline(pg.GAP, color='0.6', ls=':', lw=1)
ax.axvline(u_c, color='k', ls='--', lw=1)
ax.text(u_c, P.max(), f' closure at u = {u_c:.4f}\n (medium left: {(pg.GAP - u_c) / pg.GAP:.0%} of the gap)'
        if u_c < pg.GAP else f' closure at u = {u_c:.4f}', fontsize=9, va='top')
ax.set(xlabel='top displacement u', ylabel='P (log)', title='force before contact = parasitic stiffness of the medium')
ax = fig.add_axes((0.69, 0.55, 0.29, 0.34))
for k, c, name in (('syy', '#b2182b', '$|\\sigma_{yy} - \\sigma_{yy}^{exact}| / p$'), ('sxx', '#3d7fd0', '$|\\sigma_{xx}| / p$'),
                   ('sxy', '#4d9221', '$|\\sigma_{xy}| / p$')):
    ax.semilogy(u[contact], err[k][contact], 'o-', ms=3, color=c, label=name)
ax.set(xlabel='top displacement u', ylabel='max over all block pixels',
       title='stress error in the blocks, after contact\n(exact Cauchy: $\\sigma_{yy}$ = -P / current width, '
             '$\\sigma_{xx}$ = $\\sigma_{xy}$ = 0)')
ax.grid(True, color='0.9', which='both')
ax.legend(frameon=False, fontsize=10)
for k, (key, name) in enumerate((('syy', '$(\\sigma_{yy} - \\sigma_{yy}^{exact}) / p$'), ('sxx', '$\\sigma_{xx} / p$'),
                                 ('sxy', '$\\sigma_{xy} / p$'))):
    ax = fig.add_axes((0.03 + 0.33 * k, 0.05, 0.28, 0.44))
    e = (s[key][-1] - (exact[key][-1] if key == 'syy' else 0.0)) / p[-1]
    e = np.where(solid, e, np.nan)
    v = np.nanmax(np.abs(e))
    im = ax.imshow(e.T, origin='lower', extent=(0, 1, 0, 1), cmap='RdBu_r', vmin=-v, vmax=v, interpolation='nearest')
    ax.set(xticks=[], yticks=[], title=f'{name} at the last step (max |.| {v:.1e})')
    fig.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle(f'Contact patch test vs exact solution ({n}$^2$, gap {pg.GAP_PX} px, E_upper/E_lower = {pg.E_RATIO:g}, '
             f'$K_V$ = {float(ds.getncattr("k_v")):g}): final P = {P[-1]:.3f}, p = P/W = {p[-1]:.3f}', fontsize=13)
out = os.path.join(folder, 'patch_vs_exact.png')
fig.savefig(out, dpi=110)
print(f'wrote {out}')
print(f'slope {slope:.2f} exact {k_series:.2f}; u_c {u_c:.5f} (gap {pg.GAP:.5f}); max errors at the end: ' +
      ', '.join(f'{k} {err[k][-1]:.2e}' for k in err) + f'; P at u = gap/2: {np.interp(pg.GAP / 2, u, P):.2e}')
