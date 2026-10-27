"""
Setup sketch of the disc between two rigid plates at N^2 pixels: the fitted (and optionally graded) grid with every
element drawn, the boundary conditions, a zoom on the bottom contact, and the exact solutions.

usage:  N=32 GRADE=0.6 python plot_setup.py      -> setup_disc_N<N>[_grade<beta>].png
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.collections import PolyCollection

N = int(os.environ.get('N', 32))
os.environ['NNN'] = str(N)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import disc_geometry as dg  # noqa: E402
from muFFTTO import grid_adaptation  # noqa: E402

HERE = os.path.dirname(os.path.realpath(__file__))
COLOURS = ['#eef3fa', '#b8b8bd', '#3d7fd0', '#3d7fd0', '#4a4a4a']
NAMES = ['third medium: the two gaps near the contacts', 'disc (E = 100, $\\nu$ = 0.3)',
         'rigid plates: every node prescribed', None, 'true void, zero stiffness']
X = np.array(np.meshgrid(*(np.arange(N) / N,) * 2, indexing='ij'))
d_ref, phase, info = grid_adaptation.adapt_grid(N, lambda p: (dg.labels(dg.grade(p)) > 0).astype(float),
                                                min_jacobian=0.01, verbose=False)
x = dg.grade(X + d_ref)
c = dg.grade(X + 0.5 / N)
lab = np.where(phase > 0, dg.labels(c), 0)
lab = np.where(phase > 0, np.where(lab == 0, 1, lab), np.where(dg.frame(c), 4, 0))


def polygons(x):
    ext = np.pad(x, ((0, 0), (0, 1), (0, 1)), mode='wrap')
    ext[0, -1, :] += 1.0
    ext[1, :, -1] += 1.0
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    quad = np.stack([ext[:, i + a, j + b] for a, b in ((0, 0), (1, 0), (1, 1), (0, 1))], axis=-1)
    return quad.reshape(2, -1, 4).transpose(1, 2, 0)


def draw(ax, lw):
    ax.add_collection(PolyCollection(polygons(x), facecolors=np.array(COLOURS)[lab.ravel()], edgecolors='0.45',
                                     linewidths=lw))
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])


fig = plt.figure(figsize=(17, 9.5))
ax = fig.add_axes((0.01, 0.06, 0.45, 0.84))
draw(ax, 0.35)
ax.set(xlim=(-0.04, 1.04), ylim=(-0.07, 1.07))
t = np.linspace(0, 2 * np.pi, 400)
ax.plot(dg.CENTRE_X + dg.R * np.cos(t), dg.CENTRE_Y + dg.R * np.sin(t), color='#b2182b', lw=0.8, ls=(0, (3, 2)))
for xa in np.linspace(0.25, 0.75, 5):
    ax.annotate('', (xa, dg.TOP_VOID - 0.005), (xa, dg.TOP_VOID + 0.06),
                arrowprops=dict(arrowstyle='-|>', color='#f4a582', lw=2.5), zorder=6)
    ax.annotate('', (xa, 0.005), (xa, -0.06), arrowprops=dict(arrowstyle='-|>', color='#f4a582', lw=2.5), zorder=6)
ax.text(0.5, 1.045, 'top plate (every node): $u$ = (0, $-\\lambda U_{max}/2$)', ha='center', fontsize=10,
        color='#d6604d', fontweight='bold')
ax.text(0.5, -0.068, 'bottom plate (every node): $u$ = (0, $+\\lambda U_{max}/2$)', ha='center', fontsize=10,
        color='#d6604d', fontweight='bold')
ax.plot(dg.CENTRE_X, dg.CENTRE_Y, 'o', color='#b2182b', ms=7, zorder=7)
ax.text(dg.CENTRE_X + 0.015, dg.CENTRE_Y + 0.015, 'centre node pinned\n(symmetric solution)', fontsize=9,
        color='#b2182b')
ax.plot([dg.CENTRE_X - dg.R, dg.CENTRE_X + dg.R], [dg.CENTRE_Y] * 2, color='#4d9221', lw=1.5, ls='--')
ax.text(dg.CENTRE_X - dg.R + 0.01, dg.CENTRE_Y - 0.035, 'horizontal diameter:\nBrazilian $\\sigma_{xx}, \\sigma_{yy}$',
        fontsize=9, color='#4d9221')
z = (dg.CENTRE_X - 0.16, dg.CENTRE_X + 0.16, dg.BOTTOM_FACE - 0.04, dg.BOTTOM_FACE + 0.09)
ax.add_patch(Rectangle((z[0], z[2]), z[1] - z[0], z[3] - z[2], fill=False, ec='#b2182b', lw=1.5, ls='--'))
grade_txt = f', graded $\\beta$ = {dg.GRADE:g}' if dg.GRADE else ''
ax.set_title(f'whole cell, {N}$^2$ elements, fitted grid{grade_txt}: disc R = {dg.R:.3f}, gap {dg.GAP:g}',
             fontsize=11)

ax = fig.add_axes((0.49, 0.47, 0.25, 0.43))
draw(ax, 0.6)
ax.set(xlim=z[:2], ylim=z[2:])
xs = np.linspace(z[0], z[1], 300)
ax.plot(xs, dg.CENTRE_Y - np.sqrt(np.maximum(dg.R ** 2 - (xs - dg.CENTRE_X) ** 2, 0)), color='#b2182b', lw=1,
        ls=(0, (3, 2)))
a_end, _ = dg.hertz(1.0)
ax.axvspan(dg.CENTRE_X - a_end, dg.CENTRE_X + a_end, ymax=0.05, color='#c4441a', alpha=0.7)
spacing = np.diff(np.unique(np.round(x[0][:, N // 4], 9))).min()
n_cols = int(np.sum(np.abs(np.unique(np.round(x[0][:, 2], 9)) - dg.CENTRE_X) < a_end))
ax.set_title(f'zoom on the bottom contact (dashed: exact circle)\norange: Hertz width 2a at P = 1 (a = {a_end:.3f}): '
             f'{n_cols} node columns', fontsize=10)
for s in ax.spines.values():
    s.set_edgecolor('#b2182b')
    s.set_linestyle('--')
fig.legend(handles=[Patch(fc=col, ec='0.5', label=nm) for col, nm in zip(COLOURS, NAMES) if nm], loc='lower left',
           bbox_to_anchor=(0.49, 0.3), fontsize=10, frameon=False)
fig.text(0.76, 0.47,
         'exact (small strain, plane strain):\n\n'
         'each contact: Hertz with a rigid plate\n'
         '   E* = E / (1 - $\\nu^2$) = 109.9\n'
         '   a = $\\sqrt{4PR / (\\pi E^*)}$\n'
         '   p(x) = $\\frac{2P}{\\pi a^2}\\sqrt{a^2 - x^2}$\n\n'
         'inside: Brazilian disc (D = 2R)\n'
         '   centre: $\\sigma_{xx}$ = 2P/($\\pi$D) (tension)\n'
         '              $\\sigma_{yy}$ = -6P/($\\pi$D)\n'
         '   whole horizontal diameter: closed form\n\n'
         'measured:\n'
         '   - P vs approach\n'
         '   - p(x) at each plate from the nodal reactions\n'
         '     of the plate face (exactly at the surface)\n'
         '   - $\\sigma$ along the horizontal diameter\n'
         '   - both contacts identical (symmetry)',
         fontsize=9.5, va='top', bbox=dict(fc='white', ec='0.7'))
fig.suptitle('Disc between two rigid plates with third medium: two contacts + Brazilian stress, displacement driven, '
             'finite strain, small load', fontsize=13)
out = os.path.join(HERE, f'setup_disc_N{N}' + (f'_grade{dg.GRADE:g}' if dg.GRADE else '') + '.png')
fig.savefig(out, dpi=120)
print(f'wrote {out}')
