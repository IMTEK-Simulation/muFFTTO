"""
Video of a box_point_load.py (paper Fig. 3) or cbox_point_load.py (Figs. 11-13) run: the structure under the
prescribed displacement v of the load point.

Left: the box, every pixel at its deformed position (fixed axes), solid yellow, third medium coloured by det F (log
scale: dark = squeezed). Right: gap between the inner flange faces at x = L/2 and force at the load point vs. v, and
the CG iterations of every increment. One frame per increment.

usage:  python make_box_point_load_video.py [--experiment box_point_load] [--run n40_gamma0.0002_alpha1e-06_v1] [--fps 8]
        python make_box_point_load_video.py --experiment cbox_point_load --run n40_gamma1e-06_alpha1e-06_v1
reads / writes exp_data/<experiment>/<run>.npz / .mp4
"""
import argparse
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm

try:  # ffmpeg shipped with imageio-ffmpeg, if there is none on the PATH
    import imageio_ffmpeg

    plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', choices=['box_point_load', 'cbox_point_load'], default='box_point_load')
parser.add_argument('--run', default='n40_gamma0.0002_alpha1e-06_v1')
parser.add_argument('--fps', type=int, default=8)
args = parser.parse_args()

folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'exp_data', args.experiment)
data = np.load(os.path.join(folder, f'{args.run}.npz'))
displacement = data['displacement']
solid, medium = data['solid_pixels'], data['medium_pixels']
h = float(data['pixel_size'])
n = data['number_of_pixels']
vac = int(data['vacuum_layer'])
v, gap, reaction, nb_cg = data['v'], data['gap'], data['reaction'], data['nb_cg']
gamma = float(data['gamma'])
nb_frames = len(v)


def deformed_nodes(u):
    up = np.pad(u, ((0, 0), (0, 1), (0, 1)), mode='wrap')
    X = np.stack(np.meshgrid(np.arange(n[0] + 1), np.arange(n[1] + 1), indexing='ij')) * h + up
    return X - vac * h


def pixel_J(u):
    up = np.pad(u, ((0, 0), (0, 1), (0, 1)), mode='wrap')
    c = lambda a, b: up[:, a:n[0] + a, b:n[1] + b]
    dx = (c(1, 0) + c(1, 1) - c(0, 0) - c(0, 1)) / (2 * h)
    dy = (c(0, 1) + c(1, 1) - c(0, 0) - c(1, 0)) / (2 * h)
    return (1 + dx[0]) * (1 + dy[1]) - dy[0] * dx[1]


solid_cells, medium_cells = np.argwhere(solid), np.argwhere(medium)


def polygons(X, cells):
    i, j = cells[:, 0], cells[:, 1]
    return np.stack([X[:, i, j], X[:, i + 1, j], X[:, i + 1, j + 1], X[:, i, j + 1]], axis=1).transpose(2, 1, 0)


frames_X = [deformed_nodes(u) for u in displacement]
all_points = np.concatenate([X[:, vac:n[0] - vac + 1, vac:n[1] - vac + 1].reshape(2, -1) for X in frames_X], axis=1)
lo, hi = all_points.min(axis=1), all_points.max(axis=1)
margin = 0.05 * (hi - lo).max()

fig = plt.figure(figsize=(13, 6))
grid = fig.add_gridspec(3, 2, width_ratios=[1.3, 1])
ax_box = fig.add_subplot(grid[:, 0])
ax_gap = fig.add_subplot(grid[0, 1])
ax_force = fig.add_subplot(grid[1, 1])
ax_cg = fig.add_subplot(grid[2, 1])

norm = LogNorm(vmin=1e-2, vmax=1.0)
solid_pc = PolyCollection(polygons(frames_X[0], solid_cells), facecolors='#efef9f', edgecolors='#7a7a5a',
                          linewidths=0.25)
medium_pc = PolyCollection(polygons(frames_X[0], medium_cells), array=np.ones(len(medium_cells)), cmap='magma',
                           norm=norm, edgecolors='0.35', linewidths=0.15)
ax_box.add_collection(solid_pc)
ax_box.add_collection(medium_pc)
ax_box.set_xlim(lo[0] - margin, hi[0] + margin)
ax_box.set_ylim(lo[1] - margin, hi[1] + margin)
ax_box.set_aspect('equal')
ax_box.set_xlabel('x')
ax_box.set_ylabel('y')
fig.colorbar(medium_pc, ax=ax_box, shrink=0.7, pad=0.02).set_label('det F of the third medium')
load_marker, = ax_box.plot([], [], 'v', color='#e34948', ms=10)
title = ax_box.set_title(' ', fontsize=11)

ax_gap.plot(v, gap, '-', color='#2a78d6', lw=1.2, alpha=0.5)
gap_marker, = ax_gap.plot([], [], 'o', color='k', ms=6)
ax_gap.set_yscale('log')
ax_gap.set_ylabel('gap g at x = L/2' if args.experiment == 'box_point_load' else 'smallest gap g')
ax_force.plot(v, reaction, '-', color='#2a78d6', lw=1.2, alpha=0.5)
force_marker, = ax_force.plot([], [], 'o', color='k', ms=6)
ax_force.set_ylabel('load-point force')
for ax in (ax_gap, ax_force):
    ax.set_xlabel('prescribed displacement v')
bars = ax_cg.bar(np.arange(1, nb_frames), nb_cg[1:], width=0.85, color='0.85')
ax_cg.set_xlabel('increment')
ax_cg.set_ylabel('CG its (Green-Jacobi)')
cg_title = ax_cg.set_title(' ', fontsize=9)
for ax in (ax_gap, ax_force, ax_cg):
    ax.grid(True, color='0.9', lw=0.5)
fig.suptitle(('box under a point load on the upper flange' if args.experiment == 'box_point_load' else
              'C-box under an end load') + f' (third medium, γ = {gamma:g}, h = {h:g})', fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))
load_column, load_row = data['load_node'] if 'load_node' in data.files else (n[0] // 2, n[1] - vac)  # load point


def draw(frame):
    X = frames_X[frame]
    solid_pc.set_verts(polygons(X, solid_cells))
    medium_pc.set_verts(polygons(X, medium_cells))
    medium_pc.set_array(np.clip(pixel_J(displacement[frame])[medium], norm.vmin, norm.vmax))
    load_marker.set_data([X[0, load_column, load_row]], [X[1, load_column, load_row] + 0.03])
    title.set_text(f'v = {v[frame]:.3f},  gap g = {gap[frame]:.2e}')
    gap_marker.set_data([v[frame]], [gap[frame]])
    force_marker.set_data([v[frame]], [reaction[frame]])
    for k, bar in enumerate(bars, start=1):
        bar.set_color('#2a78d6' if k <= frame else '0.85')
    cg_title.set_text(f'total {nb_cg[:frame + 1].sum()} CG iterations')


frames = list(range(nb_frames)) + [nb_frames - 1] * (2 * args.fps)  # hold the last state for 2 s
writer = animation.FFMpegWriter(fps=args.fps, bitrate=4000, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
output_path = os.path.join(folder, f'{args.run}.mp4')
with writer.saving(fig, output_path, dpi=120):
    for frame in frames:
        draw(frame)
        writer.grab_frame()
print(f'wrote {output_path}')
