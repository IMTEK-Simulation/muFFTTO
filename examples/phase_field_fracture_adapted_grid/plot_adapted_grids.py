"""
Regular vs adapted grids (muFFTTO.grid_adaptation), side by side, with quality numbers.

    python plot_adapted_grids.py [nnn] [geometry_name] [output.png] [method ...]

Default: 32, sinusoidal_ligaments, methods zecevic and snap; the figure goes to exp_data/grids/.
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from muFFTTO import grid_adaptation, geometry  # noqa: E402

nnn = int(sys.argv[1]) if len(sys.argv) > 1 else 32
name = sys.argv[2] if len(sys.argv) > 2 else 'sinusoidal_ligaments'
output = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'exp_data', 'grids',
    f'grids_{name}_N{nnn}_{time.strftime("%Y%m%d-%H%M%S")}.png')
methods = sys.argv[4:] or ['zecevic', 'snap']

h = 1.0 / nnn
X = np.array(np.meshgrid(np.arange(nnn) * h, np.arange(nnn) * h, indexing='ij'))
regular_phase = geometry.get(name, X + 0.5 * h)
grids = [('regular', np.zeros_like(X), regular_phase,
          grid_adaptation.grid_quality(np.zeros_like(X), regular_phase, name))]
for method in methods:
    d_g, phase, info = grid_adaptation.adapt_grid(nnn, name, method=method)
    grids.append((method, d_g, phase, info))
    np.savez(os.path.splitext(output)[0] + f'_{method}.npz', d_g=d_g, phase=phase,
             **{key: np.asarray(value) for key, value in info.items()})

fine = 1024
centres = (np.arange(fine) + 0.5) / fine
exact = geometry.get(name, np.array(np.meshgrid(centres, centres, indexing='ij')))
fig, axes = plt.subplots(1, len(grids), figsize=(6.4 * len(grids), 6.9))
for ax, (label, d_g, phase, info) in zip(axes, grids):
    grid_adaptation.plot_grid(ax, d_g, phase)
    ax.contour(centres, centres, exact.T, levels=[0.5], colors='r', linewidths=0.8)
    if 'corners' in info:
        ax.plot(*info['corners'].T, 'o', mfc='none', mec='C2', ms=6)
    ax.set_title(f'{label} {nnn}$^2$: solid {info["solid_fraction"]:.4f} (exact {info["exact_fraction"]:.4f})\n'
                 f'misassigned area {info["misassigned_area"]:.4f}, cut elements {info["nb_cut"]}, '
                 f'min J$_g$/h$^2$ {info["min_gauss_J"]:.2f}', fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
fig.tight_layout()
fig.savefig(output, dpi=110)
print('wrote', output)
