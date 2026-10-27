"""
Video of the 3D beam twisted with third-medium self-contact, from example_3D_elasticity_beam_twist_contact.py.

Left: the deformed beam surface (true scale, fixed 3D camera; the medium is not drawn), the four side faces in
different colours to show the twist. Right, top: reaction torque vs. twist, with linear Saint-Venant and, if present,
the run without contact (example_3D_elasticity_beam_twist.py, beam_twist_3D_Green_1440deg.npz). Right, middle: the
smallest det F in the third medium (it drops where the medium is squeezed). Right, third: the smallest distance between
far-apart parts of the beam centreline over the beam width (below 1 the beam passes through itself, i.e. the contact
failed). Right, bottom: CG iterations of every accepted increment (log scale). One frame per accepted increment.

usage:
    python make_beam_twist_contact_3D_video.py [--fps 8] [--preconditioner Green_Jacobi] [--run rotated_nu0.45_hw8]
reads  exp_data/example_3D_elasticity_beam_twist_contact/beam_twist_contact_3D_<preconditioner>_<run>.npz
writes exp_data/example_3D_elasticity_beam_twist_contact/beam_twist_contact_3D_<preconditioner>_<run>.mp4
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

parser = argparse.ArgumentParser(description='video of the 3D beam twisted with third-medium contact')
parser.add_argument('--fps', type=int, default=8)
parser.add_argument('--preconditioner', choices=['Green', 'Green_Jacobi'], default='Green_Jacobi')
parser.add_argument('--run', default='rotated_nu0.45_hw8',
                    help='<medium top>_nu<medium Poisson ratio>_hw<medium half width>, as in the data file name')
args = parser.parse_args()

here = os.path.dirname(os.path.realpath(__file__))
data_folder_path = os.path.join(here, 'exp_data', 'example_3D_elasticity_beam_twist_contact')
data = np.load(os.path.join(data_folder_path, f'beam_twist_contact_3D_{args.preconditioner}_{args.run}.npz'))
displacement = data['displacement']  # [increment, component, x, y, z]
beam_pixels = data['beam_pixels']
pixel_size = float(data['pixel_size'])
number_of_pixels = data['number_of_pixels']
beam_length = float(data['beam_length'])
T_ratio = data['T'] / float(data['torsional_stiffness'])
twist = np.degrees(data['twist'])
min_J_medium = data['min_J_medium']
nb_cg = data['nb_cg']  # CG iterations of every accepted increment (incl. its trust-region iterations)
nit = data['nit']  # trust-region iterations of every accepted increment
nb_frames = displacement.shape[0]

no_contact_file = os.path.join(here, 'exp_data', 'example_3D_elasticity_beam_twist', 'beam_twist_3D_Green_1440deg.npz')
no_contact = np.load(no_contact_file) if os.path.exists(no_contact_file) else None


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

# self-penetration check, independent of the medium: deformed beam centreline = centroid of the beam nodes of every
# node layer z; the smallest distance between two layers that are more than 2 b apart along the beam, divided by the
# beam width b. Below 1 the beam passes through itself (the contact failed); about 1 means two strands touch.
beam_lo, beam_hi = data['beam_lo'], data['beam_hi']
beam_width = (beam_hi[0] - beam_lo[0]) * pixel_size
layers = np.arange(beam_lo[2], beam_hi[2] + 1)
far_apart = np.abs(layers[:, None] - layers[None, :]) * pixel_size > 2 * beam_width
min_distance = []
for u in displacement:
    up = np.pad(u, ((0, 0), (0, 1), (0, 1), (0, 1)), mode='wrap')
    X = (grid_nodes + up)[:, beam_lo[0]:beam_hi[0] + 1, beam_lo[1]:beam_hi[1] + 1, layers]
    centreline = X.mean(axis=(1, 2)).T  # [layer, xyz]
    distance = np.linalg.norm(centreline[:, None] - centreline[None, :], axis=-1)
    min_distance.append(distance[far_apart].min() / beam_width)
min_distance = np.array(min_distance)

fig = plt.figure(figsize=(11, 9.2))
grid = fig.add_gridspec(4, 2, width_ratios=[0.9, 1], height_ratios=[1.4, 0.9, 0.9, 0.9])
ax_beam = fig.add_subplot(grid[:, 0], projection='3d')
ax_curve = fig.add_subplot(grid[0, 1])
ax_J = fig.add_subplot(grid[1, 1])
ax_gap = fig.add_subplot(grid[2, 1])
ax_cg = fig.add_subplot(grid[3, 1])

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
title = ax_beam.set_title('twist 0 deg', fontsize=11)  # text set before tight_layout reserves its space

twist_max = twist.max() if no_contact is None else max(twist.max(), np.degrees(no_contact['twist']).max())
ax_curve.plot([0, twist_max], [0, np.radians(twist_max)], color='0.6', lw=1, ls='--',
              label=r'linear Saint-Venant')
if no_contact is not None:
    ax_curve.plot(np.degrees(no_contact['twist']), no_contact['T'] / no_contact['torsional_stiffness'],
                  color='#eda100', lw=1.5, label='no contact, 8x8x64 beam (passes through itself)')
ax_curve.plot(twist, T_ratio, color='#2a78d6', lw=1, alpha=0.35)
marker, = ax_curve.plot([], [], 'o', color='#2a78d6', ms=8, label='third-medium contact')
ax_curve.set_xlim(0, twist_max)
ax_curve.set_xlabel(r'twist of the top face $\theta$ [deg]')
ax_curve.set_ylabel(r'$T L / (G J)$')
ax_curve.legend(frameon=False, fontsize=8, loc='upper left')
ax_curve.grid(True, color='0.9', lw=0.5)

ax_J.plot(twist, min_J_medium, color='#2a78d6', lw=1, alpha=0.35)
J_marker, = ax_J.plot([], [], 'o', color='#2a78d6', ms=7)
ax_J.set_xlim(0, twist_max)
ax_J.set_ylim(0, 1.05)
ax_J.set_xlabel(r'twist of the top face $\theta$ [deg]')
ax_J.set_ylabel('min det F, medium')
ax_J.set_title('smallest det F in the third medium', fontsize=9)
ax_J.grid(True, color='0.9', lw=0.5)

ax_gap.plot(twist, min_distance, color='#2a78d6', lw=1, alpha=0.35)
gap_marker, = ax_gap.plot([], [], 'o', color='#2a78d6', ms=7)
ax_gap.axhline(1.0, color='#e34948', lw=1, ls='--', label='strands touch (distance = b)')
ax_gap.set_yscale('log')
ax_gap.set_xlim(0, twist_max)
ax_gap.set_ylim(0.3, max(20.0, 1.5 * min_distance.max()))
ax_gap.set_xlabel(r'twist of the top face $\theta$ [deg]')
ax_gap.set_ylabel('min distance / b')
ax_gap.set_title('closest far-apart parts of the beam centreline (below 1: passes through itself)', fontsize=9)
ax_gap.legend(frameon=False, fontsize=8, loc='lower left')
ax_gap.grid(True, color='0.9', lw=0.5)

increments = np.arange(1, nb_frames)
bars = ax_cg.bar(increments, nb_cg[1:], width=0.85, color='0.85')
ax_cg.set_yscale('log')
ax_cg.set_ylim(500, max(2e4, 2 * nb_cg.max()))  # fixed decades: pre- and post-buckling increments compare fairly
ax_cg.set_xlim(0.3, nb_frames - 0.3)
ax_cg.set_xlabel('accepted increment')
ax_cg.set_ylabel('CG its per increment')
cg_title = ax_cg.set_title(' ', fontsize=9)
ax_cg.grid(True, axis='y', color='0.9', lw=0.5)
fig.tight_layout()


def draw(frame):
    surface.set_verts(polygons[frame])
    title.set_text(f'twist {twist[frame]:.0f} deg,   T L / (G J) = {T_ratio[frame]:.2f}')
    marker.set_data([twist[frame]], [T_ratio[frame]])
    J_marker.set_data([twist[frame]], [min_J_medium[frame]])
    gap_marker.set_data([twist[frame]], [min_distance[frame]])
    for k, bar in enumerate(bars, start=1):
        bar.set_color('#2a78d6' if k <= frame else '0.85')
    cg_title.set_text(f'{args.preconditioner}: increment {frame}: {nb_cg[frame]} CG its in {nit[frame]} '
                      f'trust-region its;  total {nb_cg[:frame + 1].sum()} CG its')


frames = list(range(nb_frames)) + [nb_frames - 1] * (2 * args.fps)  # hold the final state for 2 s
writer = animation.FFMpegWriter(fps=args.fps, bitrate=4000, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
output_path = os.path.join(data_folder_path, f'beam_twist_contact_3D_{args.preconditioner}_{args.run}.mp4')
with writer.saving(fig, output_path, dpi=120):
    for frame in frames:
        draw(frame)
        writer.grab_frame()
print(f'wrote {output_path}')
