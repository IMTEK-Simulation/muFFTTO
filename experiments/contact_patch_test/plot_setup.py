"""
Setup sketch of the contact patch test at N^2 pixels (exact pixels, every pixel outlined): the whole periodic cell with
the void frame, the boundary conditions, and a zoom on the gap.

usage:  N=32 python plot_setup.py      -> setup_patch_N<N>.png
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
import patch_geometry as pg  # noqa: E402

HERE = os.path.dirname(os.path.realpath(__file__))
h = 1.0 / N
COLOURS = ['#eef3fa', '#b8b8bd', '#3d7fd0', '#4a4a4a']
NAMES = ['third medium: the gap, one pixel row', 'lower block (E = 100, $\\nu$ = 0.3)',
         f'upper block (E = {100 * pg.E_RATIO:g}, $\\nu$ = 0.3)',
         'true void, zero stiffness: top row and one column\nat each side (blocks free at the sides)']
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


def roller(ax, x, y, up, size):
    """roller support at (x, y): triangle + wheel, pointing up (below the body) or down (above it)"""
    s = 1 if up else -1
    ax.add_patch(Polygon([(x, y), (x - 0.5 * size, y - s * size), (x + 0.5 * size, y - s * size)], closed=True,
                         fc='white', ec='k', lw=1, zorder=5))
    ax.add_patch(Circle((x, y - s * 1.3 * size), 0.25 * size, fc='white', ec='k', lw=1, zorder=5))


fig = plt.figure(figsize=(17, 9))
ax = fig.add_axes((0.01, 0.06, 0.47, 0.84))
draw(ax, 0.3)
ax.set(xlim=(-0.04, 1.04), ylim=(-0.08, 1.06))
for x in np.linspace(pg.SIDE + 2 * h, 1 - pg.SIDE - 2 * h, 9):          # bottom rollers: u_y = 0, u_x free
    roller(ax, x, 0.0, True, 0.025)
ax.plot(0.5, 0.0, 'o', color='#b2182b', ms=8, zorder=6)
ax.annotate('one node pinned in x\n(rigid-body mode)', (0.5, 0.0), (0.62, -0.07), fontsize=9, color='#b2182b',
            arrowprops=dict(arrowstyle='->', color='#b2182b'))
ax.text(0.03, -0.065, 'bottom row: $u_y$ = 0, $u_x$ free', fontsize=10)
for x in np.linspace(pg.SIDE + 2 * h, 1 - pg.SIDE - 2 * h, 9):          # top: pushed down, u_x free
    ax.annotate('', (x, pg.UPPER_TOP), (x, pg.UPPER_TOP + 0.055), arrowprops=dict(arrowstyle='-|>', color='#f4a582',
                                                                                  lw=2.5), zorder=6)
ax.plot(0.5, pg.UPPER_TOP, 'o', color='#b2182b', ms=8, zorder=7)
ax.text(0.515, pg.UPPER_TOP - 0.035, 'one node pinned in x\n(else held only by the soft medium)', fontsize=9,
        color='#b2182b', va='top')
ax.text(0.5, 1.035, 'top row of the upper block: $u_y$ = $-\\lambda U_{max}$, $u_x$ free (no friction at the plate)',
        ha='center', fontsize=10, color='#d6604d', fontweight='bold')
z = (0.3, 0.7, pg.LOWER_TOP - 4 * h, pg.LOWER_TOP + pg.GAP + 4 * h)
ax.add_patch(Rectangle((z[0], z[2]), z[1] - z[0], z[3] - z[2], fill=False, ec='#b2182b', lw=1.5, ls='--'))
ax.set_title(f'whole periodic cell, {N}$^2$ pixels (regular grid)', fontsize=11)

ax = fig.add_axes((0.52, 0.47, 0.46, 0.42))
draw(ax, 0.6)
ax.set(xlim=z[:2], ylim=z[2:])
for s in ax.spines.values():
    s.set_edgecolor('#b2182b')
    s.set_linestyle('--')
ax.annotate('', (0.36, pg.LOWER_TOP), (0.36, pg.LOWER_TOP + pg.GAP), arrowprops=dict(arrowstyle='<->', lw=1.2))
ax.text(0.365, pg.LOWER_TOP + 0.5 * pg.GAP, f' gap = {pg.GAP_PX} pixel = {pg.GAP:.4f}', va='center', fontsize=10)
ax.set_title('zoom on the gap: flat faces on pixel edges, exact on the regular grid', fontsize=10)
fig.legend(handles=[Patch(fc=c, ec='0.5', label=n) for c, n in zip(COLOURS, NAMES)], loc='lower left',
           bbox_to_anchor=(0.52, 0.27), fontsize=10, frameon=False)
fig.text(0.735, 0.07,
         'exact solution once the gap is closed:\n'
         '   $\\sigma_{yy}$ = $-p$,  $\\sigma_{xx}$ = $\\sigma_{xy}$ = 0  in both blocks,  p = P / width\n'
         'checks:\n'
         '   - stress uniform (max deviation from p, everywhere)\n'
         '   - force before contact ~ 0 (medium parasitic stiffness)\n'
         '   - P vs closure: two blocks in series after contact\n'
         'variants: $K_V$, gap 1 / 2 px, E_upper / E_lower $\\neq$ 1\n'
         '   (the blocks then slide sideways at the interface:\n'
         '    tests that the medium transmits no shear)',
         fontsize=9.5, va='bottom', bbox=dict(fc='white', ec='0.7'))
fig.suptitle('Contact patch test with third medium: two flat blocks pressed together, displacement driven, '
             'finite strain, small load', fontsize=13)
out = os.path.join(HERE, f'setup_patch_N{N}.png')
fig.savefig(out, dpi=120)
print(f'wrote {out}')
