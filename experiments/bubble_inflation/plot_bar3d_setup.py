"""Setup sketch of the 3D hollow bar (bar3d_geometry.py) on its exact voxels: 3D cut-away view + three mid-sections.
Usage: python plot_bar3d_setup.py [nnn]   (default 16 voxels per unit length for the 3D view; sections at 32)"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from bar3d_geometry import bar_geometry, DEFAULTS  # noqa: E402

plt.rcParams.update({'font.size': 13})
HERE = os.path.dirname(os.path.realpath(__file__))
n3d = int(sys.argv[1]) if len(sys.argv) > 1 else 16
SOLID, GAS, VOID, CLAMP = '#7f7f86', '#e06c5a', '#f4f4f8', 'k'

fig = plt.figure(figsize=(21, 6.6))
# 3D cut-away: the half y < Ly/2 removed, so the cavity and the caps are visible
g = bar_geometry(n3d)
ax = fig.add_subplot(1, 4, 1, projection='3d')
keep = np.zeros(g['N'], bool)
keep[:, g['N'][1] // 2:, :] = True
colors = np.empty(g['N'], object)
colors[g['solid']] = SOLID
colors[g['cavity']] = GAS
ax.voxels(*np.meshgrid(*[np.arange(n + 1) * g['h'] for n in g['N']], indexing='ij'),
          (g['solid'] | g['cavity']) & keep, facecolors=colors, edgecolor='#00000030', linewidth=0.3)
for xc in g['clamp_x']:                                                  # clamp planes (bar extent)
    x = xc * g['h']
    t = DEFAULTS['VOID_T']
    yy, zz = np.meshgrid([t, g['L'][1] - t], [t, g['L'][2] - t])
    ax.plot_surface(np.full_like(yy, x), yy, zz, color='k', alpha=0.25)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_box_aspect(g['L'])
ax.view_init(elev=22, azim=-58)
ax.set_title(f'cut-away (half y > L/2 shown), h = 1/{n3d}')

# mid-sections at h = 1/32 on exact voxels
g = bar_geometry(32)
phase = np.where(g['solid'], 0, np.where(g['cavity'], 1, 2))
cmap = ListedColormap([SOLID, GAS, VOID])
Nx, Ny, Nz = g['N']
Lx, Ly, Lz = g['L']
sections = [('cross-section z = Lz/2  (x-y)', phase[:, :, Nz // 2].T, (0, Lx, 0, Ly), 'x', 'y', 'x'),
            ('longitudinal y = Ly/2  (x-z)', phase[:, Ny // 2, :].T, (0, Lx, 0, Lz), 'x', 'z', 'x'),
            ('longitudinal x = Lx/2  (y-z)', phase[Nx // 2, :, :].T, (0, Ly, 0, Lz), 'y', 'z', None)]
for k, (title, img, ext, xl, yl, clamp_axis) in enumerate(sections):
    ax = fig.add_subplot(1, 4, k + 2)
    ax.imshow(img, origin='lower', extent=ext, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
    if clamp_axis == 'x':
        for xc in g['clamp_x']:
            x = xc * g['h']
            ax.plot([x, x], [ext[2], ext[3]], color=CLAMP, lw=4)
    ax.set_xticks(np.arange(0, ext[1] + 1e-9, 0.25))
    ax.set_yticks(np.arange(0, ext[3] + 1e-9, 0.25))
    ax.grid(color='w', lw=0.4, alpha=0.6)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(title)
fig.legend(handles=[Patch(color=SOLID, label='solid (neo-Hookean, E = 100, nu = 0.3)'),
                    Patch(color=GAS, label='cavity: gas pressure p, zero stiffness'),
                    Patch(facecolor=VOID, edgecolor='0.6', label='void (zero stiffness), t = 1/16'),
                    Patch(color=CLAMP, label='clamped plane u = 0 (x = t and x = Lx - t, full plane)')],
           loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.04))
d = DEFAULTS
fig.suptitle(f'3D hollow bar: cross-section {d["PLATE"]} x {d["PLATE"]}, hole R {d["HOLE_R"]}, length {d["BAR_L"]} '
             f'with sealing caps {d["CAP_T"]}; cell {Lx:g} x {Ly:g} x {Lz:g} ({Nx} x {Ny} x {Nz} voxels at h = 1/32)')
out = os.path.join(HERE, 'bar3d_setup.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
print(out)
