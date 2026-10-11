"""
Project the nodes of a saved grid (.npz from plot_adapted_grids.py) that are closer than
radius h to the interface onto it, save the result next to it and print the quality.

    python project_saved_grid.py grid.npz [radius]
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np  # noqa: E402

from muFFTTO import grid_adaptation  # noqa: E402

file = sys.argv[1]
radius = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
name = 'sinusoidal_ligaments'
grid = np.load(file)
d_g, info = grid_adaptation.project_nodes_to_interface(grid['d_g'], name, radius=radius)
quality = grid_adaptation.grid_quality(d_g, grid['phase'], name)
print('quality: ' + ', '.join(f'{k} {v:.4g}' if isinstance(v, float) else f'{k} {v}' for k, v in quality.items()))
output = os.path.splitext(file)[0] + f'_projected{radius:g}_{time.strftime("%Y%m%d-%H%M%S")}.npz'   # never overwrite
method = str(grid['method']) if 'method' in grid.files else 'grid'
np.savez(output, d_g=d_g, phase=grid['phase'], method=f'{method} + projection r={radius:g}h',
         corners=grid['corners'] if 'corners' in grid.files else np.zeros((0, 2)),
         **{k: np.asarray(v) for k, v in {**info, **quality}.items()})
print('wrote', output)
