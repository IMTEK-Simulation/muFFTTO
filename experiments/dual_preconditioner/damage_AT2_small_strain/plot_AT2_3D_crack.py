"""
3D crack surfaces of the l/h = 2 runs at 64^3 (final state, cell cut through): the voxels with d >= 0.9 (a voxel takes
the damage of its lower-left node, exact voxels, no smoothing), coloured by x (the load direction), with all
balls of the microstructure drawn as translucent spheres (periodic images clipped to the cell).

    top     stiff tough inclusions (10x)   AT2_N64_dual_polydisperse_pores_incl10_l0.03125_3D
    bottom  pores                          AT2_N64_Green_Jacobi_polydisperse_pores_l0.03125_3D
    left: oblique view; right: the crack surface face-on as a map over (y, z) of its position x (the node of maximal
    damage along x), blank where no voxel reaches the threshold (the crack passes through a pore), with the outlines
    of the balls cut by the plane x = mean crack position

usage:  python plot_AT2_3D_crack.py      writes figures/AT2_3D_crack.png
"""
import os
import sys

import numpy as np
import netCDF4
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '../../..')))
from muFFTTO import geometry  # noqa: E402

CASES = [('stiff inclusions (10x)', 'AT2_N64_dual_polydisperse_pores_incl10_l0.03125_3D.nc', '0.6'),
         ('pores', 'AT2_N64_Green_Jacobi_polydisperse_pores_l0.03125_3D.nc', 'C0')]
THRESHOLD = 0.9
centres, radii = geometry.polydisperse_disk_centres_and_radii(3, 0.25, (0.05, 0.14), 0.04, 0)
u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi, 20))
sphere = np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)])

fig = plt.figure(figsize=(16, 15), layout='constrained', dpi=130)
for row, (title, name, ball_color) in enumerate(CASES):
    with netCDF4.Dataset(os.path.join(HERE, 'exp_data', name)) as ds:
        d = np.asarray(ds['damage'][-1, 0])
        load = float(ds['load'][-1])
    n = d.shape[0]
    crack = d >= THRESHOLD
    x = np.arange(n + 1) / n
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    colors = plt.cm.viridis((np.arange(n)[:, None, None] / n + 0 * d - 0.35) / 0.4)
    for col, (elev, azim, view) in enumerate(((22, -55, 'oblique view'),)):
        ax = fig.add_subplot(2, 2, 2 * row + col + 1, projection='3d')
        ax.voxels(X, Y, Z, crack, facecolors=colors, edgecolor='none', shade=True)
        for c, r in zip(centres, radii):
            for shift in np.ndindex(3, 3, 3):
                cc = c + np.array(shift) - 1
                if np.any(cc + r < 0) or np.any(cc - r > 1):
                    continue
                s = cc[:, None, None] + r * sphere
                s = np.where((s >= 0) & (s <= 1), s, np.nan)
                ax.plot_surface(*s, color=ball_color, alpha=0.18, linewidth=0, shade=True)
        ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.set_zlim(0, 1), ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel('x (load)', fontsize=12), ax.set_ylabel('y', fontsize=12), ax.set_zlabel('z', fontsize=12)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f'3D {title}, {n}$^3$, l/h = 2, $\\bar\\varepsilon_{{xx}}$ = {load:.3f}\n'
                     f'crack: voxels with d $\\geq$ {THRESHOLD} ({crack.sum()} voxels), {view}', fontsize=13)
    ax = fig.add_subplot(2, 2, 2 * row + 2)
    x_crack = np.where(d.max(axis=0) >= THRESHOLD, d.argmax(axis=0) / n, np.nan)        # [y, z]
    im = ax.imshow(x_crack.T, origin='lower', extent=(0, 1, 0, 1), interpolation='nearest', cmap='viridis',
                   vmin=np.nanmin(x_crack), vmax=np.nanmax(x_crack))
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label('crack position x', fontsize=12), cb.ax.tick_params(labelsize=11)
    x0 = np.nanmean(x_crack)
    t = np.linspace(0, 2 * np.pi, 200)
    for c, r in zip(centres, radii):
        for shift in np.ndindex(3, 3, 3):
            cc = c + np.array(shift) - 1
            if abs(cc[0] - x0) < r:
                rc = np.sqrt(r ** 2 - (cc[0] - x0) ** 2)
                ax.plot(cc[1] + rc * np.cos(t), cc[2] + rc * np.sin(t), '-', color='w' if row == 1 else 'k', lw=1)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.set_aspect('equal')
    ax.set_xlabel('y', fontsize=12), ax.set_ylabel('z', fontsize=12), ax.tick_params(labelsize=11)
    ax.set_title(f'crack surface face-on: position x over (y, z)\nblank: no voxel with d $\\geq$ {THRESHOLD}; '
                 f'circles: balls cut at x = {x0:.2f}', fontsize=13)
out = os.path.join(HERE, 'figures', 'AT2_3D_crack.png')
fig.savefig(out)
print(f'wrote {out}')
