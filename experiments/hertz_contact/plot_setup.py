"""
Setup sketch of the 2D Hertz test at N^2 pixels (exact pixels, every pixel outlined): the whole periodic cell with the
void frame that cuts the periodicity, and a zoom on the contact for the regular and the adapted (zecevic) grid.

usage:  N=128 python plot_setup.py      -> setup_hertz_N<N>.png
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.collections import PolyCollection

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import hertz_geometry as hz  # noqa: E402
from muFFTTO import grid_adaptation  # noqa: E402

N = int(os.environ.get('N', 128))
HERE = os.path.dirname(os.path.realpath(__file__))
COLOURS = ['#eef3fa', '#b8b8bd', '#3d7fd0', '#4a4a4a']
NAMES = ['third medium (contact): only in the gap below the cap', 'block', 'cylinder cap (R = 1)',
         'true void, zero stiffness: top row and one column\nat each side (cuts all periodic links)']
centres = np.stack(np.meshgrid((np.arange(N) + 0.5) / N, (np.arange(N) + 0.5) / N, indexing='ij'))


def label_of(points, solid):
    """0 medium, 1 block, 2 cap, 3 frame; solid from the (regular or adapted) element phase"""
    lab = np.where(solid, np.where(points[1] < hz.BLOCK_TOP + 0.5 * hz.GAP, 1, 2), 0)
    return np.where(hz.frame(points) & ~solid, 3, lab)


lab_regular = label_of(centres, hz.labels(centres) > 0)
d_g, phase_adapted, _ = grid_adaptation.adapt_grid(N, lambda p: (hz.labels(p) > 0).astype(float), method='zecevic',
                                                   min_jacobian=0.01, verbose=False)
lab_adapted = label_of(centres, phase_adapted > 0)


def polygons(d):
    X = np.stack(np.meshgrid(np.arange(N + 1) / N, np.arange(N + 1) / N, indexing='ij'))
    X[:, :N, :N] += d
    X[:, N, :N] += d[:, 0, :]
    X[:, :N, N] += d[:, :, 0]
    X[:, N, N] += d[:, 0, 0]
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    quad = np.stack([X[:, i + a, j + b] for a, b in ((0, 0), (1, 0), (1, 1), (0, 1))], axis=-1)
    return quad.reshape(2, -1, 4).transpose(1, 2, 0)


def draw(ax, d, lab, lw):
    ax.add_collection(PolyCollection(polygons(d), facecolors=np.array(COLOURS)[lab.ravel()], edgecolors='0.55',
                                     linewidths=lw))
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])


fig = plt.figure(figsize=(17, 9))
ax = fig.add_axes((0.01, 0.04, 0.47, 0.88))
draw(ax, np.zeros_like(d_g), lab_regular, 0.15)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.plot([0, 1], [0.002] * 2, color='k', lw=3)
for x in np.linspace(hz.CENTRE_X - 0.3, hz.CENTRE_X + 0.3, 5):
    ax.annotate('', (x, hz.CAP_TOP - 0.09), (x, hz.CAP_TOP - 0.01),
                arrowprops=dict(arrowstyle='-|>', color='#f4a582', lw=2.5))
ax.text(hz.CENTRE_X, hz.CAP_TOP - 0.13, 'cap top row moved down rigidly: $u_x$ = 0, $u_y$ = $-\\lambda U_{max}$  '
        '(reaction = P)', ha='center', fontsize=10, color='#f4a582', fontweight='bold')
ax.text(0.5, 0.012, 'block bottom row clamped', ha='center', fontsize=10)
z = (hz.CENTRE_X - 0.16, hz.CENTRE_X + 0.16, hz.BLOCK_TOP - 0.07, hz.BLOCK_TOP + 0.09)
ax.add_patch(Rectangle((z[0], z[2]), z[1] - z[0], z[3] - z[2], fill=False, ec='#b2182b', lw=1.5, ls='--'))
ax.set_title(f'whole periodic cell, {N}$^2$ pixels (regular grid)', fontsize=11)
fig.legend(handles=[Patch(fc=c, ec='0.5', label=n) for c, n in zip(COLOURS, NAMES)], loc='lower left',
           bbox_to_anchor=(0.49, 0.04), fontsize=10, ncol=1, frameon=False)
for k, (d, lab, name) in enumerate(((np.zeros_like(d_g), lab_regular, 'regular grid: the cap is a staircase, '
                                     'its lowest step is flat'),
                                    (d_g, lab_adapted, 'adapted (zecevic) grid: element edges follow the cap'))):
    ax = fig.add_axes((0.5 + 0.25 * k, 0.33, 0.24, 0.6))
    draw(ax, d, lab, 0.4)
    ax.set_xlim(z[0], z[1])
    ax.set_ylim(z[2], z[3])
    xs = np.linspace(z[0], z[1], 200)
    ax.plot(xs, hz.BLOCK_TOP + hz.GAP + hz.R - np.sqrt(hz.R ** 2 - (xs - hz.CENTRE_X) ** 2), color='#b2182b', lw=1,
            ls=(0, (3, 2)))
    ax.set_title(name, fontsize=10)
    for s in ax.spines.values():
        s.set_edgecolor('#b2182b')
        s.set_linestyle('--')
E_star = 100.0 / (2 * (1 - 0.3 ** 2))
a = 0.06
P = np.pi * E_star * a ** 2 / (4 * hz.R)
fig.text(0.755, 0.06, f'Hertz (plane strain, same material): E* = E / (2(1-$\\nu^2$)) = {E_star:.1f}\n'
         f'a = $\\sqrt{{4PR / (\\pi E^*)}}$,   p(x) = $\\frac{{2P}}{{\\pi a^2}}\\sqrt{{a^2 - x^2}}$\n'
         f'e.g. a = {a:g} ({a * N:.0f} px) needs P = {P:.3f}, $p_0$ = {2 * P / (np.pi * a):.2f}\n'
         f'gap {hz.GAP:g} = {hz.GAP * N:.1f} px;  p(x) = $-P_{{yy}}$ in the block\'s top pixel row\n'
         f'dashed red: the exact circle',
         fontsize=9.5, va='bottom', bbox=dict(fc='white', ec='0.7'))
fig.suptitle('2D Hertz test of third-medium contact: cylinder cap (R = 1) pressed onto a block, E = 100, '
             '$\\nu$ = 0.3, displacement driven', fontsize=13)
out = os.path.join(HERE, f'setup_hertz_N{N}.png')
fig.savefig(out, dpi=120)
print(f'wrote {out}')
