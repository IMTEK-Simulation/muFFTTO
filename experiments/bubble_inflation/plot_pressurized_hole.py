"""Plot a pressurized_hole.py run: deformed plate on its exact pixels (colour = pixel det F) at a few pressures,
pressure vs hole area, CG (Hessian products) per increment.  Usage: python plot_pressurized_hole.py <run folder>"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

plt.rcParams.update({'font.size': 13})
run = sys.argv[1]
d = np.load(os.path.join(run, 'state.npz'))
N = int(d['ncell']) if 'ncell' in d else int(d['nnn'])          # cell pixels per direction
h = float(d['h']) if 'h' in d else 1.0 / N
solid, hole, fixed = d['solid_px'], d['hole_px'], d['fixed']
CLR = {'GJ': 'tab:blue', 'G': 'tab:green', 'G_corr': 'tab:green', 'G_wall': 'tab:green', 'D': 'tab:red'}[str(d['precond'])]
p, area, U = d['p'], d['area'], d['u'].copy()                # U [inc, 2, x, y]
# DISPLAY ONLY: the hole-interior nodes are pinned (u = 0, they carry nothing); give them the harmonic extension of the
# hole-boundary displacement so the gas region is drawn as the deformed hole, not as a star of pinned nodes
_void_node = np.all([np.roll(~solid, (a, b), (0, 1)) for a in (0, 1) for b in (0, 1)], axis=0)
_hole_node = _void_node & np.any([np.roll(hole, (a, b), (0, 1)) for a in (0, 1) for b in (0, 1)], axis=0)
if 'clamp_node' in d:                                       # clamped nodes stay where they are
    _hole_node &= ~d['clamp_node']
for k in np.unique(np.linspace(0, len(p) - 1, 4).round().astype(int)):   # only the plotted frames
    for _ in range(4 * N * N):
        avg = 0.25 * sum(np.roll(U[k], s, ax) for s in (1, -1) for ax in (1, 2))
        U[k] = np.where(_hole_node, avg, U[k])


def pixel_polygons(u, mask):
    ii, jj = np.nonzero(mask)
    polys = []
    for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
        a, b = (ii + di) % N, (jj + dj) % N
        polys.append(np.stack([(ii + di) * h + u[0, a, b], (jj + dj) * h + u[1, a, b]], axis=-1))
    polys = np.stack(polys, axis=1)                           # [pixel, 4, 2]
    x, y = polys[..., 0], polys[..., 1]
    J = 0.5 * np.sum(x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y, axis=1) / h ** 2   # = pixel mean det F (Q1)
    return polys, J


frames = np.unique(np.linspace(0, len(p) - 1, 4).round().astype(int))
fig = plt.figure(figsize=(4.6 * len(frames), 9.4))
gs = fig.add_gridspec(2, len(frames), height_ratios=[1.25, 1])
Jall = np.concatenate([pixel_polygons(U[k], solid)[1] for k in frames])
norm = matplotlib.colors.Normalize(vmin=min(Jall.min(), 0.8), vmax=max(Jall.max(), 1.2))
lim = [np.min([pixel_polygons(U[k], solid | hole)[0][..., c].min() for k in frames]) for c in (0, 1)], \
      [np.max([pixel_polygons(U[k], solid | hole)[0][..., c].max() for k in frames]) for c in (0, 1)]
for col, k in enumerate(frames):
    ax = fig.add_subplot(gs[0, col])
    ph, _ = pixel_polygons(U[k], hole)
    ax.add_collection(PolyCollection(ph, facecolor='#e06c5a', edgecolor='none'))
    ps, J = pixel_polygons(U[k], solid)
    pc = PolyCollection(ps, array=J, cmap='viridis', norm=norm, edgecolor='k', linewidth=0.15)
    ax.add_collection(pc)
    ii, jj = np.nonzero(fixed[0, 0] & ~_void_node)                    # supports / clamped nodes on the solid
    ax.plot(ii * h + U[k][0, ii, jj], jj * h + U[k][1, ii, jj], 'k^' if len(ii) < 4 else 'ks', ms=8 if len(ii) < 4 else 3)
    ax.set_xlim(lim[0][0] - 0.02, lim[1][0] + 0.02)
    ax.set_ylim(lim[0][1] - 0.02, lim[1][1] + 0.02)
    ax.set_aspect('equal')
    ax.set_title(f'p = {p[k]:.2f}  (inc {k})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
cb = fig.colorbar(pc, ax=fig.axes[:len(frames)], shrink=0.8, pad=0.01)
cb.set_label('det F (pixel mean)')

ax = fig.add_subplot(gs[1, :len(frames) // 2] if len(frames) > 1 else gs[1, 0])
ax.plot(area / area[0], p, 'o-', color='k')
ax.set_xlabel('hole area / initial hole area')
ax.set_ylabel('pressure p   (E = 100)')
ax.set_xlim(0, None)
ax.set_ylim(0, None)
ax.grid(alpha=0.3)
ax = fig.add_subplot(gs[1, len(frames) // 2:])
ax.plot(p[1:], d['hessp'][1:], 'o-', color=CLR, label='Hessian products (CG)')
ax.plot(p[1:], 50 * d['nit'][1:], 's--', color=CLR, alpha=0.5, mfc='none', label='Newton iterations x 50')
ax.set_xlabel('pressure p')
ax.set_ylabel('per accepted increment')
ax.set_xlim(0, None)
ax.set_ylim(0, None)
ax.grid(alpha=0.3)
ax.legend()
NAMES = {'GJ': 'Green-Jacobi', 'G': 'projected Green', 'G_corr': 'corrected Green (dense)', 'G_wall': 'corrected Green (wall FFT)', 'D': 'dual'}
ax.set_title(f'{NAMES[str(d["precond"])]}, h = 1/{int(round(1 / h))}' + (', left + right clamped' if 'clamp_node' in d and d['clamp_node'].any() else ', free'))
out = os.path.join(run, 'pressurized_hole.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
print(out)
