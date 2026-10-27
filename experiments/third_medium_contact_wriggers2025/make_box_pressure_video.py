"""
Video of a box_pressure.py run along its pressure path (inflation, release, suction).

Left: the box, every pixel at its deformed position (fixed axes over the whole path), solid yellow, third medium
coloured by det F (log scale: dark = squeezed, light = inflated). Right: the pressure along the path, the gap between
the inner flange faces (at x = L/2 and smallest) and the area of the medium vs. the pressure. One frame per increment.

usage:  python make_box_pressure_video.py [--run n40_gamma0.0001_alpha1e-06_path0_-0.8_0_1_sym] [--fps 12]
reads / writes exp_data/box_pressure/<run>.npz / .mp4
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
parser.add_argument('--run', default='n40_gamma0.0001_alpha1e-06_path0_-0.8_0_1_sym')
parser.add_argument('--fps', type=int, default=12)
args = parser.parse_args()

folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'exp_data', 'box_pressure')
data = np.load(os.path.join(folder, f'{args.run}.npz'))
displacement = data['displacement']
solid, medium = data['solid_pixels'], data['medium_pixels']
h = float(data['pixel_size'])
n = data['number_of_pixels']
vac = int(data['vacuum_layer'])
P, stage, gap, area = data['P'], data['stage'], data['gap'], data['area']
gap_min = data['gap_min'] if 'gap_min' in data.files else gap
gamma = float(data['gamma'])
nb_frames = len(P)
labels = {1: 'inflation', 2: 'release', 3: 'suction'}
if len(data['pressure_path']) == 2:  # a single stage: name it by the sign of the pressure
    labels = {1: 'inflation' if data['pressure_path'][1] < 0 else 'suction'}
colours = {1: '#2a78d6', 2: '#1baf7a', 3: '#eb6834'}


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
grid = fig.add_gridspec(3, 2, width_ratios=[1.25, 1], height_ratios=[1, 1, 1])
ax_box = fig.add_subplot(grid[:, 0])
ax_path = fig.add_subplot(grid[0, 1])
ax_gap = fig.add_subplot(grid[1, 1])
ax_area = fig.add_subplot(grid[2, 1])

norm = LogNorm(vmin=1e-2, vmax=1e1)
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
title = ax_box.set_title(' ', fontsize=11)

steps = np.arange(nb_frames)
for s in np.unique(stage[1:]):
    sel = (stage == s) | (steps == np.flatnonzero(stage == s)[0] - 1)
    ax_path.plot(steps[sel], P[sel], '-', color=colours[s], lw=1.5, label=labels[s])
    ax_gap.plot(P[sel], gap[sel], '-', color=colours[s], lw=1.2, alpha=0.6)
    ax_gap.plot(P[sel], gap_min[sel], ':', color=colours[s], lw=1.2, alpha=0.6)
    ax_area.plot(P[sel], area[sel], '-', color=colours[s], lw=1.2, alpha=0.6)
path_marker, = ax_path.plot([], [], 'o', color='k', ms=6)
gap_marker, = ax_gap.plot([], [], 'o', color='k', ms=6, label='at x = L/2')
gap_min_marker, = ax_gap.plot([], [], 's', color='0.4', ms=5, label='smallest along the box (dotted)')
area_marker, = ax_area.plot([], [], 'o', color='k', ms=6)
ax_path.axhline(0, color='0.7', lw=0.8)
ax_path.set_xlabel('increment')
ax_path.set_ylabel('pressure P')
ax_path.legend(frameon=False, fontsize=8, loc='best')
ax_gap.set_yscale('log')
ax_gap.set_ylabel('gap g')
ax_gap.legend(frameon=False, fontsize=8, loc='lower left')
ax_area.set_yscale('log')
ax_area.set_ylabel('medium area / initial')
for ax in (ax_gap, ax_area):
    ax.set_xlabel('pressure P  (< 0 inflates, > 0 suction)')
    ax.axvline(0, color='0.7', lw=0.8)
for ax in (ax_path, ax_gap, ax_area):
    ax.grid(True, color='0.9', lw=0.5)
fig.suptitle(f'box under internal pressure and suction (third medium, γ = {gamma:g}, h = {h:g})', fontsize=12)
fig.tight_layout()


def draw(frame):
    solid_pc.set_verts(polygons(frames_X[frame], solid_cells))
    medium_pc.set_verts(polygons(frames_X[frame], medium_cells))
    medium_pc.set_array(np.clip(pixel_J(displacement[frame])[medium], norm.vmin, norm.vmax))
    title.set_text(f'{labels.get(int(stage[frame]), "start")}:  P = {P[frame]:+.3f},  gap g = {gap[frame]:.2e}')
    path_marker.set_data([frame], [P[frame]])
    gap_marker.set_data([P[frame]], [gap[frame]])
    gap_min_marker.set_data([P[frame]], [gap_min[frame]])
    area_marker.set_data([P[frame]], [area[frame]])


frames = list(range(nb_frames)) + [nb_frames - 1] * (2 * args.fps)  # hold the last state for 2 s
writer = animation.FFMpegWriter(fps=args.fps, bitrate=4000, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
output_path = os.path.join(folder, f'{args.run}.mp4')
with writer.saving(fig, output_path, dpi=120):
    for frame in frames:
        draw(frame)
        writer.grab_frame()
print(f'wrote {output_path}')
