"""
Figures of a box_pressure.py run (inflation, release, suction).

    1. snapshots along the load path, every pixel at its deformed position, the third medium coloured by det F
       (log scale: dark = squeezed, light = inflated); solid in yellow
    2. gap at x = L/2, area of the medium and length of the box vs. the pressure, coloured by the stage of the path

usage:  python plot_box_pressure.py [--run n40_gamma0.0001_alpha1e-06_path0_-0.8_0_1]
reads / writes exp_data/box_pressure/<run>*
"""
import argparse
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm

parser = argparse.ArgumentParser()
parser.add_argument('--run', default='n40_gamma0.0001_alpha1e-06_path0_-0.8_0_1')
args = parser.parse_args()

folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'exp_data', 'box_pressure')
data = np.load(os.path.join(folder, f'{args.run}.npz'))
displacement = data['displacement']  # [increment, component, x, y]
solid, medium = data['solid_pixels'], data['medium_pixels']
h = float(data['pixel_size'])
n = data['number_of_pixels']
vac = int(data['vacuum_layer'])
P, stage, gap, area, length = data['P'], data['stage'], data['gap'], data['area'], data['length']
gamma = float(data['gamma'])
path = data['pressure_path']


def deformed_nodes(u):
    up = np.pad(u, ((0, 0), (0, 1), (0, 1)), mode='wrap')  # periodic copy nodes
    X = np.stack(np.meshgrid(np.arange(n[0] + 1), np.arange(n[1] + 1), indexing='ij')) * h + up
    return X - vac * h  # box corner at the origin


def pixel_J(u):
    """det F at the pixel centres (bilinear element, centre point)."""
    up = np.pad(u, ((0, 0), (0, 1), (0, 1)), mode='wrap')
    c = lambda a, b: up[:, a:n[0] + a, b:n[1] + b]
    dx = (c(1, 0) + c(1, 1) - c(0, 0) - c(0, 1)) / (2 * h)
    dy = (c(0, 1) + c(1, 1) - c(0, 0) - c(1, 0)) / (2 * h)
    return (1 + dx[0]) * (1 + dy[1]) - dy[0] * dx[1]


def draw(ax, u, norm):
    X = deformed_nodes(u)
    quad = lambda i, j: [X[:, i, j], X[:, i + 1, j], X[:, i + 1, j + 1], X[:, i, j + 1]]
    ax.add_collection(PolyCollection([quad(i, j) for i, j in np.argwhere(solid)], facecolors='#efef9f',
                                     edgecolors='#7a7a5a', linewidths=0.25))
    J = pixel_J(u)
    pc = PolyCollection([quad(i, j) for i, j in np.argwhere(medium)], array=np.clip(J[medium], norm.vmin, norm.vmax),
                        cmap='magma', norm=norm, edgecolors='0.35', linewidths=0.15)
    ax.add_collection(pc)
    ax.set_aspect('equal')
    return pc


# snapshots: the end of every stage plus the middle of the inflation and of the suction
labels = {1: 'inflation', 2: 'release', 3: 'suction'}
if len(path) == 2:  # a single stage: a quarter, half, three quarters and all of it, named by the sign of P
    targets = [(1, f * path[1]) for f in (0.25, 0.5, 0.75, 1.0)]
    labels = {1: 'inflation' if path[1] < 0 else 'suction'}
else:
    targets = [(1, path[1] / 2), (1, path[1]), (2, path[2]), (3, 0.1), (3, path[-1])]
picks = [int(np.argmin(np.abs(P - p) + 100 * (stage != s))) for s, p in targets if s <= stage.max()]
norm = LogNorm(vmin=1e-2, vmax=1e1)
fig, axes = plt.subplots(1, len(picks), figsize=(3.6 * len(picks), 3.6), layout='constrained')
for ax, k in zip(np.atleast_1d(axes), picks):
    pc = draw(ax, displacement[k], norm)
    ax.autoscale()
    ax.set_title(f'{labels.get(int(stage[k]), "")}: P = {P[k]:+.2f}\ngap g = {gap[k]:.2e}', fontsize=10)
    ax.set_xlabel('x')
np.atleast_1d(axes)[0].set_ylabel('y')
fig.colorbar(pc, ax=axes, shrink=0.85, pad=0.01).set_label('det F of the third medium')
fig.suptitle(f'box under internal pressure (P < 0) and suction (P > 0), γ = {gamma:g}, h = {h:g}', fontsize=11)
fig.savefig(os.path.join(folder, f'{args.run}_snapshots.png'), dpi=170, bbox_inches='tight')

# curves vs the pressure, one colour per stage
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), layout='constrained')
colours = {1: '#2a78d6', 2: '#1baf7a', 3: '#eb6834'}
for s in range(1, int(stage.max()) + 1):
    sel = (stage == s) | (np.arange(len(stage)) == np.flatnonzero(stage == s)[0] - 1)  # connect to the previous stage
    for ax, y in zip(axes, (gap, area, length)):
        ax.plot(P[sel], y[sel], 'o-', ms=3, color=colours[s], label=labels[s])
axes[0].set_yscale('log')
axes[0].set_ylabel('gap g at x = L/2')
axes[1].set_ylabel('area of the medium / initial area')
axes[1].set_yscale('log')
axes[2].set_ylabel('box length (lower corners)')
for ax in axes:
    ax.set_xlabel('pressure P  (< 0 inflates, > 0 suction)')
    ax.axvline(0, color='0.7', lw=0.8)
    ax.grid(True, color='0.9', lw=0.5)
axes[0].legend(frameon=False, fontsize=8)
fig.savefig(os.path.join(folder, f'{args.run}_response.png'), dpi=150)
print(f'wrote {folder}/{args.run}_snapshots.png, _response.png')
