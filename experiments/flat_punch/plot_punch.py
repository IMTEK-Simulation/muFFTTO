"""
Flat punch against the exact (Sadowsky) pressure.

    left    p per pixel of the block's top row (stairs) at three loads, each normalised by its P, with the exact pressure
            averaged per pixel and pointwise
    middle  ratio simulation / exact per pixel under the punch, at the same loads
    right   force P vs the punch displacement (log): force before contact, closure

usage:  python plot_punch.py <run folder>      -> <run folder>/punch_vs_exact.png
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
os.environ['B_PX'] = str(int(ds.getncattr('B_px')))
os.environ['GAP_PX'] = str(int(ds.getncattr('gap_px')))
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import punch_geometry as pg  # noqa: E402

u = np.asarray(ds['crown_u'][:])
P = np.asarray(ds['crown_force'][:])
p = np.asarray(ds['p_surface'][:])
edges = (np.arange(n + 1) - n / 2) * h                     # pixel edges relative to the punch axis
centres = 0.5 * (edges[:-1] + edges[1:])
under = np.abs(centres) < pg.B + h                 # under the punch and the pixel beside each corner
contact = np.nonzero(P > 0.05 * P.max())[0]
frames = contact[np.linspace(0, len(contact) - 1, 3).astype(int)]
SHADES = ['#9ecae1', '#3d7fd0', '#08306b']

fig, axes = plt.subplots(1, 3, figsize=(19, 6), layout='constrained')
ax = axes[0]
xs = np.linspace(-pg.B * 0.999, pg.B * 0.999, 400)
ax.plot(xs, 1.0 / (np.pi * np.sqrt(pg.B ** 2 - xs ** 2)), color='k', lw=1, label='exact, pointwise')
surface = pg.pixel_average(edges[:-1], edges[1:], 1.0)
ax.stairs(surface, edges, color='0.6', lw=1.5, ls=':', label='exact surface pressure, averaged per pixel')
# what the simulation measures: -sigma_yy at the Gauss points of the block's top pixel row (depth h/2 -+ ...)
exact = pg.top_row_average(edges, 1.0)
ax.stairs(exact, edges, color='k', lw=2.2, ls='--', label='exact $-\\sigma_{yy}$ at the Gauss points of the\ntop pixel row (as measured)')
for f, c in zip(frames, SHADES):
    ax.stairs(p[f] / P[f], edges, color=c, lw=1.6, label=f'simulation, P = {P[f]:.3f}')
ax.set(xlim=(-1.8 * pg.B, 1.8 * pg.B), ylim=(0, 1.15 * max(exact.max(), (p[frames] / P[frames, None]).max())),
       xlabel='x (relative to the punch axis)', ylabel='p / P', title='pressure under the punch, normalised by P')
ax.legend(frameon=False, fontsize=9, loc='upper center')
ax = axes[1]
for f, c in zip(frames, SHADES):
    ax.plot(centres[under], (p[f] / P[f])[under] / exact[under], 'o-', color=c, ms=5, label=f'P = {P[f]:.3f}')
ax.axhline(1, color='k', lw=1)
ax.axvspan(-pg.B, pg.B, color='0.95', zorder=0)
ax.set(xlabel='pixel centre x (grey: under the punch)', ylabel='simulation / exact (as measured)',
       title='per-pixel ratio, punch and one pixel beyond each corner (1 = exact)')
ax.legend(frameon=False, fontsize=9)
ax = axes[2]
ax.semilogy(u, np.maximum(P, 1e-12), 'o-', color='#3d7fd0', ms=3)
ax.axvline(pg.GAP, color='0.6', ls=':')
ax.text(pg.GAP, 2e-5, f' gap {pg.GAP:.4f}', fontsize=9, color='0.4')
ax.set(xlabel='punch displacement u', ylabel='force P (log)', title='force vs displacement: before contact ~ medium')
outside = np.array([(p[f][np.abs(centres) > pg.B] * h).sum() / P[f] for f in range(len(P))])
exact_outside = (exact[np.abs(centres) > pg.B] * h).sum()
fig.suptitle(f'Flat rigid punch vs exact (Sadowsky, half-space), {n}$^2$, 2B = {2 * pg.B:.3f} ({2 * pg.B_PX} px), '
             f'block depth {pg.BLOCK_TOP:.3f}: row force outside the punch {outside[-1]:.3f} P (exact at that depth {exact_outside:.3f} P)', fontsize=13)
out = os.path.join(folder, 'punch_vs_exact.png')
fig.savefig(out, dpi=115)
print(f'wrote {out}')
f = frames[-1]
r = (p[f] / P[f])[under] / exact[under]
print('ratio per pixel (left to right):', np.round(r, 3), f'; outside {outside[-1]:+.2e} P; '
      f'exact outside {exact_outside:.3f}; sum p h / P {(p[f] * h).sum() / P[f]:.4f}; P(u = gap/2) {np.interp(pg.GAP / 2, u, P):.1e}')
