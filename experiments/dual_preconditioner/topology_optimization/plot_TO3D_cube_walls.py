"""
Anatomy of a 3D TO design with a rotating-cubes mechanism: the soft phase forms flat walls on two planes per axis
(half a cell apart), the stiff phase 2 x 2 x 2 cubes per cell, joined to their neighbours through openings in the walls
(bridges). The cell is shifted periodically so that the walls lie on the cell faces and mid-planes.
  left: the 8 cubes (stiff phase cut at the walls), coloured as a 3D checkerboard (the design repeats under a shift by
        half a cell along the diagonal: the two colours are the two kinds of cube);
  right: the walls seen face-on, black = soft wall, white = stiff bridge (majority over the wall thickness).
usage:  python plot_TO3D_cube_walls.py <design.nc> [out.png]
"""
import os
import sys

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

plt.rcParams.update({'font.size': 13})
path = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures',
                                                         'TO3D_cube_walls.png')
with netCDF4.Dataset(path) as d:
    name = 'phase_field_solution' if 'phase_field_solution' in d.variables else 'phase_field_iterate'
    rho = np.array(d[name][-1, 0])
N = rho.shape[0]
H = N // 2
soft = rho < 0.5
# wall position per axis: the plane (mod half a cell) with the most soft phase
pos = []
for a in range(3):
    prof = soft.mean(axis=tuple(i for i in range(3) if i != a))
    pos.append(int(np.argmax(prof[:H] + prof[H:])))
rho_c = np.roll(rho, [-p for p in pos], axis=(0, 1, 2))     # walls now at 0 and N/2
soft_c = rho_c < 0.5
half_t = max(2, int(round(0.055 * N)))                       # half the wall thickness (~0.11 of the cell)

fig = plt.figure(figsize=(20, 9.5))
ax = fig.add_subplot(2, 4, (1, 6), projection='3d')
step = max(1, N // 64)
colours = [plt.cm.Blues, plt.cm.Oranges]
for i in range(2):
    for j in range(2):
        for k in range(2):
            block = rho_c[i * H:(i + 1) * H, j * H:(j + 1) * H, k * H:(k + 1) * H]
            v, f, _, _ = measure.marching_cubes(np.pad(block, 1, mode='constant', constant_values=0.0), 0.5,
                                                step_size=step)
            v = (v - 1) / N + np.array([i, j, k]) * 0.5
            shrink = 0.04                                     # pull the cubes apart a little, to see them
            v = (v - (np.array([i, j, k]) * 0.5 + 0.25)) * (1 - shrink) + np.array([i, j, k]) * 0.5 + 0.25 \
                + (np.array([i, j, k]) - 0.5) * 0.06
            mesh = Poly3DCollection(v[f], linewidth=0)
            nrm = np.cross(v[f][:, 1] - v[f][:, 0], v[f][:, 2] - v[f][:, 0])
            shade = 0.35 + 0.65 * np.abs(nrm @ np.array([0.4, 0.5, 0.77])) / (np.linalg.norm(nrm, axis=1) + 1e-30)
            mesh.set_facecolor(colours[(i + j + k) % 2](0.3 + 0.6 * shade))
            ax.add_collection3d(mesh)
ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05), zlim=(-0.05, 1.05), xlabel='x', ylabel='y', zlabel='z')
ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=22, azim=35)
ax.set_title('stiff phase of one cell: 2 x 2 x 2 cubes (pulled apart slightly),\n'
             'blue / orange = the two kinds of cube (3D checkerboard)')
for a, nm in enumerate('xyz'):
    other = [c for i, c in enumerate('xyz') if i != a]
    for w, at in enumerate((0, H)):
        idx = [(at + s) % N for s in range(-half_t, half_t)]
        wall = np.take(soft_c, idx, axis=a).mean(axis=a) > 0.5
        ax2 = fig.add_subplot(2, 8, 8 * w + 5 + a + (1 if a else 0) * 0)
        ax2.imshow(np.where(wall, 1.0, 0.0).T, origin='lower', cmap='Greys', vmin=0, vmax=1,
                   interpolation='nearest', extent=(0, 1, 0, 1))
        ax2.set_xlabel(other[0]); ax2.set_ylabel(other[1])
        ax2.set_title(f'wall _|_ {nm} at {nm}={((pos[a] + at) % N) / N:.2f}\nwall {100 * wall.mean():.0f} %',
                      fontsize=12)
        ax2.set_xticks([0, 0.5, 1]); ax2.set_yticks([0, 0.5, 1])
fig.suptitle(f'3D TO design {N}$^3$: rotating-cubes mechanism. Stiff cubes (72 %) separated by soft walls '
             '(28 %, thickness ~0.11); black = wall, white = stiff bridge joining two cubes', fontsize=15)
fig.tight_layout()
fig.savefig(out, dpi=105)
print('saved', out, 'wall planes', [p / N for p in pos], '+ 0.5')
