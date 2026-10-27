"""
Video of the 3D buckling beam from example_3D_elasticity_beam_buckling.py.

Left: the deformed beam surface (true scale, fixed 3D camera). Right: load vs. tip deflection with the exact elastica,
and the CG iterations of every inner solve. One frame per load increment.

usage:
    python make_beam_buckling_3D_video.py [--fps 12] [--preconditioner Green]
reads  exp_data/example_3D_elasticity_beam_buckling/beam_buckling_3D_<preconditioner>.npz
writes exp_data/example_3D_elasticity_beam_buckling/beam_buckling_3D_<preconditioner>.mp4
"""
import argparse
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.special import ellipk

try:  # ffmpeg shipped with imageio-ffmpeg, if there is none on the PATH
    import imageio_ffmpeg

    plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass

parser = argparse.ArgumentParser(description='video of the 3D buckling beam')
parser.add_argument('--fps', type=int, default=12)
parser.add_argument('--preconditioner', choices=['Green', 'Green_Jacobi'], default='Green')
args = parser.parse_args()

data_folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'exp_data',
                                'example_3D_elasticity_beam_buckling')
data = np.load(os.path.join(data_folder_path, f'beam_buckling_3D_{args.preconditioner}.npz'))
displacement = data['displacement']  # [increment, component, x, y, z]
beam_pixels = data['beam_pixels']
pixel_size = float(data['pixel_size'])
number_of_pixels = data['number_of_pixels']
beam_lo, beam_hi = data['beam_lo'], data['beam_hi']
beam_length = float(data['beam_length'])
P_ratio = data['P'] / float(data['P_euler'])
tip_u_x = data['tip_u_x'] / beam_length
cg_per_solve = data['cg_per_solve']  # CG iterations of every inner solve
solve_increment = data['solve_increment']  # load increment of every solve
nb_frames = displacement.shape[0]


def outer_faces(beam_voxels):
    """Corner node indices [face, 4, 3] of the voxel faces on the surface of the beam."""
    faces = []
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
    return np.array(faces)


# deformed surface polygons of every increment: [increment, face, corner, xyz]
faces = outer_faces(beam_pixels)
grid_nodes = np.stack(np.meshgrid(*[np.arange(n + 1) for n in number_of_pixels], indexing='ij')) * pixel_size
polygons = []
for u in displacement:
    up = np.pad(u, ((0, 0), (0, 1), (0, 1), (0, 1)), mode='wrap')  # periodic copy nodes
    X = grid_nodes + up
    polygons.append(np.moveaxis(X[:, faces[..., 0], faces[..., 1], faces[..., 2]], 0, -1))
polygons = np.array(polygons)

# exact inextensible elastica, clamped-free, axial dead load
p = np.linspace(1e-4, np.sin(np.radians(150) / 2), 400)
K_p = ellipk(p ** 2)
elastica_P, elastica_u_x = (2 * K_p / np.pi) ** 2, 2 * p / K_p
keep = elastica_P <= P_ratio.max()

fig = plt.figure(figsize=(11, 5.4))
grid = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], height_ratios=[1.6, 1])
ax_beam = fig.add_subplot(grid[:, 0], projection='3d')
ax_curve = fig.add_subplot(grid[0, 1])
ax_cg = fig.add_subplot(grid[1, 1])

surface = Poly3DCollection(polygons[0], facecolors='#86b6ef', edgecolors='#184f95', linewidths=0.15)
ax_beam.add_collection3d(surface)
# limits of the beam over all frames, equal scale on all axes (true shape)
lo = polygons.reshape(-1, 3).min(axis=0) - 0.05 * beam_length
hi = polygons.reshape(-1, 3).max(axis=0) + 0.05 * beam_length
ax_beam.set_xlim(lo[0], hi[0])
ax_beam.set_ylim(lo[1], hi[1])
ax_beam.set_zlim(lo[2], hi[2])
ax_beam.set_box_aspect(hi - lo, zoom=1.25)
ax_beam.view_init(elev=15, azim=-60)
ax_beam.set_axis_off()
title = ax_beam.set_title('P / P_cr = 0.00', fontsize=11)  # text set before tight_layout reserves its space

ax_curve.plot([0, 0] + list(elastica_u_x[keep]), [0, 1] + list(elastica_P[keep]), color='#eb6834', lw=2,
              label='elastica (exact)')
ax_curve.plot(tip_u_x, P_ratio, color='#2a78d6', lw=1, alpha=0.35)
marker, = ax_curve.plot([], [], 'o', color='#2a78d6', ms=8, label='FE 3D')
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


def draw(frame):
    surface.set_verts(polygons[frame])
    title.set_text(f'P / P_cr = {P_ratio[frame]:.2f}')
    marker.set_data([tip_u_x[frame]], [P_ratio[frame]])
    for bar, increment in zip(bars, solve_increment):
        bar.set_color('#2a78d6' if increment <= frame else '0.85')


frames = list(range(nb_frames)) + [nb_frames - 1] * (2 * args.fps)  # hold the final state for 2 s
writer = animation.FFMpegWriter(fps=args.fps, bitrate=4000, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
output_path = os.path.join(data_folder_path, f'beam_buckling_3D_{args.preconditioner}.mp4')
with writer.saving(fig, output_path, dpi=120):
    for frame in frames:
        draw(frame)
        writer.grab_frame()
print(f'wrote {output_path}')
