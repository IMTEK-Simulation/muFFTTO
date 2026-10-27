"""
Video of the twisted 3D beam from example_3D_elasticity_beam_torsion.py.

Left: the deformed beam surface (true scale, fixed 3D camera), the four side faces in different colours to show the
twist. Right: torque vs. twist with Saint-Venant torsion under a dead-load torque, and the CG iterations of every
inner solve. One frame per load increment.

usage:
    python make_beam_torsion_3D_video.py [--fps 12] [--preconditioner Green]
reads  exp_data/example_3D_elasticity_beam_torsion/beam_torsion_3D_<preconditioner>.npz
writes exp_data/example_3D_elasticity_beam_torsion/beam_torsion_3D_<preconditioner>.mp4
"""
import argparse
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:  # ffmpeg shipped with imageio-ffmpeg, if there is none on the PATH
    import imageio_ffmpeg

    plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass

parser = argparse.ArgumentParser(description='video of the twisted 3D beam')
parser.add_argument('--fps', type=int, default=8)
parser.add_argument('--preconditioner', choices=['Green', 'Green_Jacobi'], default='Green')
args = parser.parse_args()

data_folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'exp_data',
                                'example_3D_elasticity_beam_torsion')
data = np.load(os.path.join(data_folder_path, f'beam_torsion_3D_{args.preconditioner}.npz'))
displacement = data['displacement']  # [increment, component, x, y, z]
beam_pixels = data['beam_pixels']
pixel_size = float(data['pixel_size'])
number_of_pixels = data['number_of_pixels']
beam_length = float(data['beam_length'])
T_ratio = data['T'] / float(data['torsional_stiffness'])
twist = np.degrees(data['twist'])
cg_per_solve = data['cg_per_solve']  # CG iterations of every inner solve
solve_increment = data['solve_increment']  # load increment of every solve
nb_frames = displacement.shape[0]


def outer_faces(beam_voxels):
    """Corner node indices [face, 4, 3] and outward direction (2 d + side) of the voxel faces on the beam surface."""
    faces, directions = [], []
    for d in range(3):
        a, b = [k for k in range(3) if k != d]
        for side in (0, 1):
            neighbour = np.roll(beam_voxels, -1 if side else 1, axis=d)
            for voxel in np.argwhere(beam_voxels & ~neighbour):
                corners = []
                for oa, ob in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    node = voxel.copy()
                    node[d] += side
                    node[a] += oa
                    node[b] += ob
                    corners.append(node)
                faces.append(corners)
                directions.append(2 * d + side)
    return np.array(faces), np.array(directions)


# deformed surface polygons of every increment: [increment, face, corner, xyz]
faces, directions = outer_faces(beam_pixels)
side_colours = np.array(['#86b6ef', '#eda100', '#1baf7a', '#e87ba4', '#b7d3f6', '#b7d3f6'])[directions]
grid_nodes = np.stack(np.meshgrid(*[np.arange(n + 1) for n in number_of_pixels], indexing='ij')) * pixel_size
polygons = []
for u in displacement:
    up = np.pad(u, ((0, 0), (0, 1), (0, 1), (0, 1)), mode='wrap')  # periodic copy nodes
    X = grid_nodes + up
    polygons.append(np.moveaxis(X[:, faces[..., 0], faces[..., 1], faces[..., 2]], 0, -1))
polygons = np.array(polygons)

# Saint-Venant torsion with the dead-load moment: T L / (G J) = phi / cos(phi)
phi = np.linspace(0, np.radians(min(80.0, 1.15 * twist.max())), 300)

fig = plt.figure(figsize=(10, 6))
grid = fig.add_gridspec(2, 2, width_ratios=[0.8, 1], height_ratios=[1.6, 1])
ax_beam = fig.add_subplot(grid[:, 0], projection='3d')
ax_curve = fig.add_subplot(grid[0, 1])
ax_cg = fig.add_subplot(grid[1, 1])

surface = Poly3DCollection(polygons[0], facecolors=side_colours, edgecolors='0.3', linewidths=0.1)
ax_beam.add_collection3d(surface)
# limits of the beam over all frames, equal scale on all axes (true shape)
lo = polygons.reshape(-1, 3).min(axis=0) - 0.05 * beam_length
hi = polygons.reshape(-1, 3).max(axis=0) + 0.05 * beam_length
ax_beam.set_xlim(lo[0], hi[0])
ax_beam.set_ylim(lo[1], hi[1])
ax_beam.set_zlim(lo[2], hi[2])
ax_beam.set_box_aspect(hi - lo, zoom=1.1)
ax_beam.view_init(elev=20, azim=-60)
ax_beam.set_axis_off()
title = ax_beam.set_title('twist 0.0 deg', fontsize=11)  # text set before tight_layout reserves its space

ax_curve.plot(np.degrees(phi), phi / np.cos(phi), color='#eb6834', lw=2,
              label=r'Saint-Venant, dead load: $T\cos\varphi = GJ\varphi/L$')
ax_curve.plot(np.degrees(phi), phi, color='0.6', lw=1, ls='--', label=r'linear: $T = GJ\varphi/L$')
ax_curve.plot(twist, T_ratio, color='#2a78d6', lw=1, alpha=0.35)
marker, = ax_curve.plot([], [], 'o', color='#2a78d6', ms=8, label='FE 3D')
ax_curve.set_xlim(0, np.degrees(phi[-1]))
ax_curve.set_ylim(0, 1.1 * T_ratio.max())
ax_curve.set_xlabel(r'twist of the top face $\varphi$ [deg]')
ax_curve.set_ylabel(r'$T L / (G J)$')
ax_curve.legend(frameon=False, fontsize=8, loc='upper left')
ax_curve.grid(True, color='0.9', lw=0.5)

solves = np.arange(1, len(cg_per_solve) + 1)
bars = ax_cg.bar(solves, cg_per_solve, width=0.85, color='0.85')
ax_cg.set_xlabel('inner CG solve (trust-region iteration)')
ax_cg.set_ylabel('CG its per solve')
ax_cg.set_title(f'{args.preconditioner}: {len(solves)} solves, {cg_per_solve.sum()} CG iterations', fontsize=9)
ax_cg.grid(True, axis='y', color='0.9', lw=0.5)
fig.tight_layout()


def draw(frame):
    surface.set_verts(polygons[frame])
    title.set_text(f'twist {twist[frame]:.1f} deg,   T L / (G J) = {T_ratio[frame]:.2f}')
    marker.set_data([twist[frame]], [T_ratio[frame]])
    for bar, increment in zip(bars, solve_increment):
        bar.set_color('#2a78d6' if increment <= frame else '0.85')


frames = list(range(nb_frames)) + [nb_frames - 1] * (2 * args.fps)  # hold the final state for 2 s
writer = animation.FFMpegWriter(fps=args.fps, bitrate=4000, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
output_path = os.path.join(data_folder_path, f'beam_torsion_3D_{args.preconditioner}.mp4')
with writer.saving(fig, output_path, dpi=120):
    for frame in frames:
        draw(frame)
        writer.grab_frame()
print(f'wrote {output_path}')
