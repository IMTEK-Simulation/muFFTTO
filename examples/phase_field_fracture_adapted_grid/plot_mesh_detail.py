"""
Large view of saved grids (.npz from plot_adapted_grids.py) with zooms.

    python plot_mesh_detail.py output.png grid_a.npz [grid_b.npz ...]

One row per grid: the whole cell and three zooms (a cusp of the central void, a cusp at the
outer junction, the left medium wedge); red: exact interface, green: detected corners.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from muFFTTO import grid_adaptation, geometry  # noqa: E402

output, files = sys.argv[1], sys.argv[2:]
name = 'sinusoidal_ligaments'
fine = 1024
centres = (np.arange(fine) + 0.5) / fine
exact = geometry.get(name, np.array(np.meshgrid(centres, centres, indexing='ij')))
zooms = [('central void, upper left cusp', (0.22, 0.44), (0.60, 0.82)),
         ('outer junction, lower left cusp', (0.14, 0.36), (0.06, 0.28)),
         ('left medium wedge', (-0.02, 0.24), (0.52, 0.80))]
fig, axes = plt.subplots(len(files), 1 + len(zooms), figsize=(7.5 + 5.2 * len(zooms), 7.6 * len(files)),
                         gridspec_kw={'width_ratios': [1.45] + [1] * len(zooms)}, squeeze=False)
for row, file in zip(axes, files):
    grid = np.load(file)
    label = str(grid['method']) if 'method' in grid.files else os.path.basename(file)
    for k, ax in enumerate(row):
        grid_adaptation.plot_grid(ax, grid['d_g'], grid['phase'], linewidth=0.5 if k == 0 else 0.9)
        ax.contour(centres, centres, exact.T, levels=[0.5], colors='r', linewidths=1.0 if k == 0 else 1.6)
        if k == 0 or True:                                   # boundary nodes off the interface
            if 'off_nodes' not in locals() or off_nodes[0] is not file:
                n = grid['d_g'].shape[1]
                h = 1.0 / n
                X = np.array(np.meshgrid(np.arange(n) * h, np.arange(n) * h, indexing='ij'))
                v = grid['phase'] > 0.5
                edge_x, edge_y = v != np.roll(v, 1, 1), v != np.roll(v, 1, 0)
                b = edge_x | np.roll(edge_x, 1, 0) | edge_y | np.roll(edge_y, 1, 1)
                interface = grid_adaptation._Interface(name, {}, 512)
                xb = (X + grid['d_g'])[:, b]
                dist = np.linalg.norm(interface.project(xb, h) - xb, axis=0) / h
                off_nodes = (file, xb[:, dist > 0.01], dist[dist > 0.01])
            sc = ax.scatter(*np.mod(off_nodes[1], 1.0), c=off_nodes[2], cmap='cool', vmin=0, vmax=0.3,
                            s=12 if k == 0 else 40, zorder=5)
        if 'corners' in grid.files and grid['corners'].size:
            ax.plot(*grid['corners'].T, 'o', mfc='none', mec='lime', ms=7 if k == 0 else 12, mew=1.5)
        if k == 0:
            ax.set_xlim(-0.01, 1.01)
            ax.set_ylim(-0.01, 1.01)
            ax.set_title(f'{label}: misassigned area {float(grid["misassigned_area"]):.4f}, '
                         f'solid {float(grid["solid_fraction"]):.4f} (exact {float(grid["exact_fraction"]):.4f})',
                         fontsize=11)
            for title, xlim, ylim in zooms:
                ax.add_patch(plt.Rectangle((xlim[0], ylim[0]), xlim[1] - xlim[0], ylim[1] - ylim[0],
                                           fill=False, ec='orange', lw=1.5))
        else:
            title, xlim, ylim = zooms[k - 1]
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_title(f'{label}: {title}', fontsize=11)
fig.tight_layout()
fig.colorbar(sc, ax=axes[:, -1].tolist(), label='boundary node: distance to interface / h (> 0.01 shown)', shrink=0.6)
fig.savefig(output, dpi=100)
print('wrote', output)
