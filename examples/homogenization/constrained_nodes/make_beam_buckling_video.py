"""
Video of the buckling beam from example_2D_elasticity_beam_buckling.py.

Left: the deformed beam (true scale, beam pixels only). Right: load vs. tip deflection with the exact elastica,
and the CG iterations of every load increment. One frame per load increment.

usage:
    python make_beam_buckling_video.py [--fps 12] [--preconditioner Green]
reads  exp_data/example_2D_elasticity_beam_buckling/beam_buckling_<preconditioner>.npz
writes exp_data/example_2D_elasticity_beam_buckling/beam_buckling_<preconditioner>.mp4
"""
import argparse
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import PolyCollection
from scipy.special import ellipk, ellipe

try:  # ffmpeg shipped with imageio-ffmpeg, if there is none on the PATH
    import imageio_ffmpeg

    plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass

parser = argparse.ArgumentParser(description='video of the buckling beam')
parser.add_argument('--fps', type=int, default=12)
parser.add_argument('--preconditioner', choices=['Green', 'Green_Jacobi'], default='Green')
args = parser.parse_args()

data_folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'exp_data',
                                'example_2D_elasticity_beam_buckling')
data = np.load(os.path.join(data_folder_path, f'beam_buckling_{args.preconditioner}.npz'))
displacement = data['displacement']  # [increment, component, x, y]
beam_pixels = data['beam_pixels']
pixel_size = float(data['pixel_size'])
beam_x, beam_y = data['beam_x'], data['beam_y']
beam_length = float(data['beam_length'])
P_ratio = data['P'] / float(data['P_euler'])
tip_u_x = data['tip_u_x'] / beam_length
cg_per_solve = data['cg_per_solve']  # CG iterations of every inner solve
solve_increment = data['solve_increment']  # load increment of every solve
nb_frames = displacement.shape[0]

# deformed pixel corners of the beam, every increment: [increment, pixel, corner, xy]
nx, ny = displacement.shape[2:]
X, Y = np.meshgrid(np.arange(nx + 1) * pixel_size, np.arange(ny + 1) * pixel_size, indexing='ij')
pixels = np.argwhere(beam_pixels)
i, j = pixels[:, 0], pixels[:, 1]
polygons = []
for u in displacement:
    up = np.pad(u, ((0, 0), (0, 1), (0, 1)), mode='wrap')  # periodic copy nodes
    Xd, Yd = X + up[0], Y + up[1]
    corners = [(Xd[i, j], Yd[i, j]), (Xd[i + 1, j], Yd[i + 1, j]), (Xd[i + 1, j + 1], Yd[i + 1, j + 1]),
               (Xd[i, j + 1], Yd[i, j + 1])]
    polygons.append(np.stack([np.stack(c, axis=-1) for c in corners], axis=1))
polygons = np.array(polygons)
margin = 0.15 * beam_length
lo = polygons.reshape(-1, 2).min(axis=0) - margin
hi = polygons.reshape(-1, 2).max(axis=0) + margin

# exact inextensible elastica, clamped-free, vertical dead load
p = np.linspace(1e-4, np.sin(np.radians(150) / 2), 400)
K_p, E_p = ellipk(p ** 2), ellipe(p ** 2)
elastica_P, elastica_u_x = (2 * K_p / np.pi) ** 2, 2 * p / K_p
keep = elastica_P <= P_ratio.max()

fig = plt.figure(figsize=(11, 5.4))
grid = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], height_ratios=[1.6, 1])
ax_beam = fig.add_subplot(grid[:, 0])
ax_curve = fig.add_subplot(grid[0, 1])
ax_cg = fig.add_subplot(grid[1, 1])

beam = PolyCollection(polygons[0], facecolors='#86b6ef', edgecolors='#184f95', linewidths=0.25)
ax_beam.add_collection(beam)
ax_beam.plot(np.array([beam_x[0] - 2, beam_x[1] + 2]) * pixel_size, [beam_y[0] * pixel_size] * 2, 'k', lw=3)
arrow = ax_beam.annotate('', xy=(0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='#eb6834', lw=2))
ax_beam.set_xlim(lo[0], hi[0])
ax_beam.set_ylim(lo[1], hi[1])
ax_beam.set_aspect('equal')
ax_beam.set_xticks([])
ax_beam.set_yticks([])
title = ax_beam.set_title('P / P_cr = 0.00', fontsize=11)  # text set before tight_layout reserves its space

ax_curve.plot([0, 0] + list(elastica_u_x[keep]), [0, 1] + list(elastica_P[keep]), color='#eb6834', lw=2,
              label='elastica (exact)')
ax_curve.plot(tip_u_x, P_ratio, color='#2a78d6', lw=1, alpha=0.35)
marker, = ax_curve.plot([], [], 'o', color='#2a78d6', ms=8, label='FE')
ax_curve.axhline(1.0, color='0.5', lw=1, ls='--')
ax_curve.set_xlabel(r'tip deflection $u_x / L$')
ax_curve.set_ylabel(r'$P / P_{cr}^{Euler}$')
ax_curve.legend(frameon=False, fontsize=9, loc='lower right')
ax_curve.grid(True, color='0.9', lw=0.5)

solves = np.arange(1, len(cg_per_solve) + 1)
bars = ax_cg.bar(solves, cg_per_solve, width=0.85, color='0.85')
first_buckled = solves[P_ratio[solve_increment] >= 1.0][0]  # first solve at P >= P_cr
ax_cg.axvline(first_buckled - 0.5, color='0.5', lw=1, ls='--')
ax_cg.set_xlabel('inner CG solve (trust-region iteration)')
ax_cg.set_ylabel('CG its per solve')
ax_cg.set_title(f'{args.preconditioner}: {len(solves)} solves, {cg_per_solve.sum()} CG iterations', fontsize=9)
ax_cg.grid(True, axis='y', color='0.9', lw=0.5)
fig.tight_layout()

# top centre node, for the load arrow
top_centre = np.argmin(np.abs(pixels[:, 0] - (beam_x[0] + beam_x[1]) / 2 + 0.5) + np.abs(pixels[:, 1] - (beam_y[1] - 1)))


def draw(frame):
    beam.set_verts(polygons[frame])
    tip = polygons[frame][top_centre, 3]  # upper-left corner of the top centre pixel
    arrow.xy = (tip[0], tip[1])
    arrow.set_position((tip[0], tip[1] + 0.25 * beam_length * P_ratio[frame] / P_ratio.max() + 1e-9))
    title.set_text(f'P / P_cr = {P_ratio[frame]:.2f}')
    marker.set_data([tip_u_x[frame]], [P_ratio[frame]])
    for bar, increment in zip(bars, solve_increment):
        bar.set_color('#2a78d6' if increment <= frame else '0.85')


frames = list(range(nb_frames)) + [nb_frames - 1] * (2 * args.fps)  # hold the final state for 2 s
writer = animation.FFMpegWriter(fps=args.fps, bitrate=4000, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
output_path = os.path.join(data_folder_path, f'beam_buckling_{args.preconditioner}.mp4')
with writer.saving(fig, output_path, dpi=120):
    for frame in frames:
        draw(frame)
        writer.grab_frame()
print(f'wrote {output_path}')
