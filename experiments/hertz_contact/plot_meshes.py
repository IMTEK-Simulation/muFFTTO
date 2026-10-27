"""
Meshes of the 2D Hertz test at N^2 pixels: interface-fitted (zecevic) grids without and with grading (nodes clustered
towards the contact, hertz_geometry.grade), each as the whole cell and a zoom on the contact (every element drawn).

usage:  N=32 GRADES=0,0.6 python plot_meshes.py      -> meshes_hertz_N<N>.png
"""
import importlib
import os
import sys

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.collections import PolyCollection

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from muFFTTO import grid_adaptation  # noqa: E402

N = int(os.environ.get('N', 32))
GRADES = [float(g) for g in os.environ.get('GRADES', '0,0.6').split(',')]
HERE = os.path.dirname(os.path.realpath(__file__))
COLOURS = ['#eef3fa', '#b8b8bd', '#3d7fd0', '#4a4a4a']
NAMES = ['third medium', 'block', 'cylinder cap (R = 1)', 'true void']
os.environ['NNN'] = str(N)


def mesh(grade):
    os.environ['GRADE'] = str(grade)
    import hertz_geometry
    hz = importlib.reload(hertz_geometry)                 # FRAME / SIDE follow the graded edge pixels
    X = np.array(np.meshgrid(*(np.arange(N) / N,) * 2, indexing='ij'))
    d_ref, phase, info = grid_adaptation.adapt_grid(N, lambda p: (hz.labels(hz.grade(p)) > 0).astype(float),
                                                    min_jacobian=0.01, verbose=False)
    x = hz.grade(X + d_ref)
    c = hz.grade(X + 0.5 / N)
    solid = phase > 0
    lab = np.where(solid, np.where(c[1] < hz.BLOCK_TOP + 0.5 * hz.GAP, 1, 2), 0)
    lab = np.where(hz.frame(c) & ~solid, 3, lab)
    return hz, x, lab, info


def polygons(x):
    ext = np.pad(x, ((0, 0), (0, 1), (0, 1)), mode='wrap')
    ext[0, -1, :] += 1.0
    ext[1, :, -1] += 1.0
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    quad = np.stack([ext[:, i + a, j + b] for a, b in ((0, 0), (1, 0), (1, 1), (0, 1))], axis=-1)
    return quad.reshape(2, -1, 4).transpose(1, 2, 0)


fig = plt.figure(figsize=(6.2 * len(GRADES) + 1, 12.5))
for k, grade in enumerate(GRADES):
    hz, x, lab, info = mesh(grade)
    z = (hz.CENTRE_X - 0.16, hz.CENTRE_X + 0.16, hz.BLOCK_TOP - 0.07, hz.BLOCK_TOP + 0.09)
    for row, (box, lw) in enumerate((((0, 1, 0, 1), 0.4), (z, 0.6))):
        ax = fig.add_axes((0.02 + k / len(GRADES) * 0.96, 0.5 - 0.47 * row + 0.03, 0.96 / len(GRADES) - 0.02, 0.4))
        p = PolyCollection(polygons(x), facecolors=np.array(COLOURS)[lab.ravel()],
                           edgecolors='0.45', linewidths=lw)
        ax.add_collection(p)
        ax.set(xlim=box[:2], ylim=box[2:], aspect='equal', xticks=[], yticks=[])
        if row == 0:
            ax.add_patch(Rectangle((z[0], z[2]), z[1] - z[0], z[3] - z[2], fill=False, ec='#b2182b', lw=1.5, ls='--'))
            spacing = (1 - grade) / N
            ax.set_title(('fitted grid, no grading' if grade == 0 else f'fitted + graded grid ($\\beta$ = {grade:g})')
                         + f'\nsmallest spacing {spacing:.4f} (h/{1 / (1 - grade):.2g}), gap {hz.GAP:g} = '
                         f'{hz.GAP / spacing:.1f} elements; min J/h$^2$ {info["min_gauss_J"]:.2f}', fontsize=11)
        else:
            xs = np.linspace(z[0], z[1], 300)
            ax.plot(xs, hz.BLOCK_TOP + hz.GAP + hz.R - np.sqrt(hz.R ** 2 - (xs - hz.CENTRE_X) ** 2), color='#b2182b',
                    lw=1, ls=(0, (3, 2)))
            a = 0.12                                         # Hertz half-width at the final load (~P = 0.6)
            ax.axvspan(hz.CENTRE_X - a, hz.CENTRE_X + a, ymax=0.04, color='#c4441a', alpha=0.6)
            ax.set_title(f'zoom on the contact; orange bar: final Hertz contact width 2a (a $\\approx$ 0.12),\n'
                         f'{np.sum(np.abs(np.unique(np.round(x[0][:, N // 2], 9)) - hz.CENTRE_X) < a)} '
                         f'node columns across it', fontsize=10)
            for s in ax.spines.values():
                s.set_edgecolor('#b2182b')
                s.set_linestyle('--')
fig.legend(handles=[Patch(fc=c, ec='0.5', label=n) for c, n in zip(COLOURS, NAMES)], loc='lower center', ncol=4,
           fontsize=10, frameon=False)
fig.suptitle(f'Hertz test meshes, {N}$^2$ elements (dashed red: exact circle)', fontsize=13)
out = os.path.join(HERE, f'meshes_hertz_N{N}.png')
fig.savefig(out, dpi=120)
print(f'wrote {out}')
