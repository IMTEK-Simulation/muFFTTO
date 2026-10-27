"""
Setup sketch of the flat rigid punch at N^2 pixels (exact pixels, every pixel outlined): the whole periodic cell with the
void frame and the boundary conditions, a zoom on the punch corner, and the exact pressure (pointwise and per pixel).

usage:  N=32 python plot_setup.py      -> setup_punch_N<N>.png
"""
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle, Circle, Polygon
from matplotlib.collections import PolyCollection

N = int(os.environ.get('N', 32))
os.environ['NNN'] = str(N)
import punch_geometry as pg  # noqa: E402

HERE = os.path.dirname(os.path.realpath(__file__))
h = 1.0 / N
COLOURS = ['#eef3fa', '#b8b8bd', '#3d7fd0', '#4a4a4a']
NAMES = [f'third medium: the gap below the punch, {pg.GAP_PX} pixel row', 'block (E = 100, $\\nu$ = 0.3)',
         'rigid punch: every node prescribed', 'true void, zero stiffness']
centres = np.stack(np.meshgrid((np.arange(N) + 0.5) * h, (np.arange(N) + 0.5) * h, indexing='ij'))
lab = pg.labels(centres)
lab = np.where(pg.frame(centres) & (lab == 0), 3, lab)
i, j = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
quads = np.stack([np.stack([(i + a) * h, (j + b) * h]) for a, b in ((0, 0), (1, 0), (1, 1), (0, 1))], axis=-1)
quads = quads.reshape(2, -1, 4).transpose(1, 2, 0)


def draw(ax, lw):
    ax.add_collection(PolyCollection(quads, facecolors=np.array(COLOURS)[lab.ravel()], edgecolors='0.55', linewidths=lw))
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])


def roller(ax, x, y, size):
    ax.add_patch(Polygon([(x, y), (x - 0.5 * size, y - size), (x + 0.5 * size, y - size)], closed=True, fc='white',
                         ec='k', lw=1, zorder=5))
    ax.add_patch(Circle((x, y - 1.3 * size), 0.25 * size, fc='white', ec='k', lw=1, zorder=5))


fig = plt.figure(figsize=(17, 9.5))
ax = fig.add_axes((0.01, 0.06, 0.45, 0.84))
draw(ax, 0.3)
ax.set(xlim=(-0.04, 1.04), ylim=(-0.08, 1.06))
for x in np.linspace(pg.SIDE + 2 * h, 1 - pg.SIDE - 2 * h, 9):
    roller(ax, x, 0.0, 0.025)
ax.plot(0.5, 0.0, 'o', color='#b2182b', ms=8, zorder=6)
ax.text(0.53, -0.06, 'one node pinned in x', fontsize=9, color='#b2182b')
ax.text(0.03, -0.065, 'bottom row: $u_y$ = 0, $u_x$ free', fontsize=10)
for x in np.linspace(pg.CENTRE_X - pg.B + h, pg.CENTRE_X + pg.B - h, 4):
    ax.annotate('', (x, pg.PUNCH_TOP - 0.02), (x, pg.PUNCH_TOP + 0.05),
                arrowprops=dict(arrowstyle='-|>', color='#f4a582', lw=2.5), zorder=6)
ax.text(0.5, 1.035, 'every punch node: $u_x$ = 0, $u_y$ = $-\\lambda U_{max}$ (rigid punch, reaction = P)',
        ha='center', fontsize=10, color='#d6604d', fontweight='bold')
ax.annotate('', (pg.CENTRE_X - pg.B, pg.BLOCK_TOP + pg.GAP + 0.03), (pg.CENTRE_X + pg.B, pg.BLOCK_TOP + pg.GAP + 0.03),
            arrowprops=dict(arrowstyle='<->', color='w', lw=1.2))
ax.text(pg.CENTRE_X, pg.BLOCK_TOP + pg.GAP + 0.04, f'2B = {2 * pg.B:.3f} ({2 * pg.B_PX} px)', color='w', ha='center',
        fontsize=10)
z = (pg.CENTRE_X + pg.B - 4 * h, pg.CENTRE_X + pg.B + 4 * h, pg.BLOCK_TOP - 3 * h, pg.BLOCK_TOP + pg.GAP + 3 * h)
ax.add_patch(Rectangle((z[0], z[2]), z[1] - z[0], z[3] - z[2], fill=False, ec='#b2182b', lw=1.5, ls='--'))
ax.set_title(f'whole periodic cell, {N}$^2$ pixels (regular grid): block depth {pg.BLOCK_TOP:.3f}, '
             f'B / depth = {pg.B / pg.BLOCK_TOP:.2f}', fontsize=11)

ax = fig.add_axes((0.49, 0.5, 0.22, 0.4))
draw(ax, 0.6)
ax.set(xlim=z[:2], ylim=z[2:])
for s in ax.spines.values():
    s.set_edgecolor('#b2182b')
    s.set_linestyle('--')
ax.set_title('zoom on the punch corner: medium only under the\npunch, void beside it (block surface free)', fontsize=10)

ax = fig.add_axes((0.76, 0.5, 0.22, 0.4))
P = 1.0
xs = np.linspace(-pg.B * 0.999, pg.B * 0.999, 400)
ax.plot(xs, P / (np.pi * np.sqrt(pg.B ** 2 - xs ** 2)), color='k', lw=1.5, label='exact (Sadowsky)')
edges = (np.arange(N + 1) - N / 2) * h
avg = pg.pixel_average(edges[:-1], edges[1:], P)
ax.stairs(avg, edges, color='#c4441a', lw=2, label='exact, averaged per pixel\n(what a pixel row can show)')
ax.set(xlim=(-1.6 * pg.B, 1.6 * pg.B), ylim=(0, 3.2 * avg.max() / 2), xlabel='x (relative to the punch axis)',
       ylabel='p / P', title='exact pressure under the punch (P = 1)')
ax.grid(True, color='0.9')
ax.legend(frameon=False, fontsize=9, loc='upper center')
fig.legend(handles=[Patch(fc=c, ec='0.5', label=n) for c, n in zip(COLOURS, NAMES)], loc='lower left',
           bbox_to_anchor=(0.49, 0.25), fontsize=10, frameon=False)
fig.text(0.74, 0.07,
         'exact (rigid frictionless punch on a half-space):\n'
         '   p(x) = P / ($\\pi \\sqrt{B^2 - x^2}$), infinite at the corners\n'
         'checks:\n'
         '   - p per pixel vs the pixel average of the exact p\n'
         '     (the corner pixel carries the singularity)\n'
         '   - P carried only under the punch (none outside)\n'
         '   - force before contact ~ 0\n'
         'caveat: finite block (depth 0.625, width 0.94):\n'
         '   B / depth = 0.3 changes p slightly vs the half-space',
         fontsize=9.5, va='bottom', bbox=dict(fc='white', ec='0.7'))
fig.suptitle('Flat rigid punch with third medium: sharp corners, displacement driven, finite strain, small load',
             fontsize=13)
out = os.path.join(HERE, f'setup_punch_N{N}.png')
fig.savefig(out, dpi=120)
print(f'wrote {out}')
