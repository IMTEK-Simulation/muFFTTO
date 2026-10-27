"""
Figures of a box_point_load.py run: how the third medium transmits the contact.

    1. snapshots of the loading, every pixel at its deformed position, the third medium coloured by det F
       (log scale: dark = squeezed); solid in yellow
    2. close-up of the contact zone at x = L/2 in the last state
    3. gap, reaction force and smallest det F of the medium vs. the prescribed displacement

usage:  python plot_box_point_load.py [--run n40_gamma0.0002_alpha1e-06_v1]
reads / writes exp_data/box_point_load/<run>*
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
parser.add_argument('--run', default='n40_gamma0.0002_alpha1e-06_v1')
args = parser.parse_args()

folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'exp_data', 'box_point_load')
data = np.load(os.path.join(folder, f'{args.run}.npz'))
displacement = data['displacement']  # [increment, component, x, y]
solid, medium = data['solid_pixels'], data['medium_pixels']
h = float(data['pixel_size'])
n = data['number_of_pixels']
vac = int(data['vacuum_layer'])
v, gap, reaction, min_J = data['v'], data['gap'], data['reaction'], data['min_J_medium']
gamma = float(data['gamma'])


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


def draw(ax, u, norm, edge=0.25):
    X = deformed_nodes(u)
    quad = lambda i, j: [X[:, i, j], X[:, i + 1, j], X[:, i + 1, j + 1], X[:, i, j + 1]]
    ax.add_collection(PolyCollection([quad(i, j) for i, j in np.argwhere(solid)], facecolors='#efef9f',
                                     edgecolors='#7a7a5a', linewidths=edge))
    J = pixel_J(u)
    cells = np.argwhere(medium)
    pc = PolyCollection([quad(i, j) for i, j in cells], array=np.clip(J[medium], 1e-3, 1), cmap='magma', norm=norm,
                        edgecolors='0.35', linewidths=edge * 0.6)
    ax.add_collection(pc)
    ax.set_aspect('equal')
    return pc


norm = LogNorm(vmin=1e-2, vmax=1.0)

# 1. snapshots
targets = [0.0, 0.25, 0.5, 0.75, 1.0]
frames = [int(np.argmin(np.abs(v - t))) for t in targets]
fig, axes = plt.subplots(1, len(frames), figsize=(17, 3.1), sharey=True, layout='constrained')
for ax, k in zip(axes, frames):
    pc = draw(ax, displacement[k], norm)
    ax.set_xlim(-0.1, 2.1)
    ax.set_ylim(-0.8, 0.6)
    ax.set_title(f'v = {v[k]:.2f},  g = {gap[k]:.2e}', fontsize=10)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([-0.5, 0, 0.5])
    ax.set_xlabel('x')
axes[0].set_ylabel('y')
cbar = fig.colorbar(pc, ax=axes, shrink=0.9, pad=0.01)
cbar.set_label('det F of the third medium')
fig.suptitle(f'box under a point load on the upper flange (γ = {gamma:g}, h = {h:g}); solid yellow, third medium '
             f'coloured by det F', fontsize=11)
fig.savefig(os.path.join(folder, f'{args.run}_snapshots.png'), dpi=170, bbox_inches='tight')

# 2. close-up of the contact zone in the last state
fig, ax = plt.subplots(figsize=(8.6, 4.6), layout='constrained')
pc = draw(ax, displacement[-1], norm, edge=0.4)
X = deformed_nodes(displacement[-1])
centre = X[:, n[0] // 2, n[1] // 2]
ax.set_xlim(centre[0] - 0.3, centre[0] + 0.3)
ax.set_ylim(centre[1] - 0.15, centre[1] + 0.13)
fig.colorbar(pc, ax=ax, shrink=0.9, pad=0.08).set_label('det F of the third medium')
ax.set_title(f'contact zone at x = L/2, v = {v[-1]:g}\nthe medium between the flanges is squeezed to '
             f'det F ≈ {min_J[-1]:.2f}, gap g = {gap[-1]:.2e}', fontsize=10)
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.savefig(os.path.join(folder, f'{args.run}_contact_zone.png'), dpi=200, bbox_inches='tight')

# 3. curves
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
axes[0].plot(v, gap, 'o-', color='#2a78d6', ms=3)
axes[0].set_yscale('log')
axes[0].set_ylabel('gap g at x = L/2')
axes[1].plot(v, reaction, 'o-', color='#2a78d6', ms=3)
axes[1].set_ylabel('force at the load point')
axes[2].plot(v, min_J, 'o-', color='#2a78d6', ms=3)
axes[2].set_yscale('log')
axes[2].set_ylabel('smallest det F of the third medium')
for a in axes:
    a.set_xlabel('prescribed displacement v')
    a.grid(True, color='0.9', lw=0.5)
fig.tight_layout()
fig.savefig(os.path.join(folder, f'{args.run}_response.png'), dpi=150)
print(f'wrote {folder}/{args.run}_snapshots.png, _contact_zone.png, _response.png')
