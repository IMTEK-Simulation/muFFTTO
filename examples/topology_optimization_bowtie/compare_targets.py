"""
Overview of bow-tie TO runs: the initial bow-tie and the optimized designs of several runs (3 x 3 cells each),
with the target and achieved Poisson ratio, the grey fraction (share of nodes with 0.1 < rho < 0.9) and the
minimum solid / void feature size in pixels (feature_size.py).

    python compare_targets.py data/<run 1> data/<run 2> ...

Output: data/compare_targets_<time>.png
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from muFFTTO import io_utils
from feature_size import feature_sizes

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def engineering_constants(C):
    S = np.linalg.inv(C)
    return {'E_x': 1 / S[0, 0], 'E_y': 1 / S[1, 1], 'nu_xy': -S[1, 0] / S[0, 0], 'nu_yx': -S[0, 1] / S[1, 1]}


runs = []
for folder in sys.argv[1:]:
    data = io_utils.read_file(os.path.join(folder, 'final.nc'))
    rho = np.asarray(data.fields['rho_final']).reshape(data.attributes['nb_of_pixels'].astype(int))
    C_0 = np.asarray(data.attributes['C_initial_voigt']).reshape(3, 3)
    C_t = np.asarray(data.attributes['C_target_voigt']).reshape(3, 3)
    C_opt = np.asarray(data.attributes['C_optimized_voigt']).reshape(3, 3)
    runs.append((folder, rho, C_0, C_t, C_opt, data.attributes))

fig, axes = plt.subplots(1, len(runs) + 1, figsize=(3.6 * (len(runs) + 1), 4.2))
from muFFTTO import geometry
a = runs[0][5]
n = int(np.asarray(a['nb_of_pixels']).ravel()[0])
x = (np.arange(n) / n)
coords = np.array(np.meshgrid(x, x, indexing='ij'))
rho_0 = geometry.get('reentrant_honeycomb', coords, theta=float(np.ravel(a['theta'])[0]),
                     thickness=float(np.ravel(a['thickness'])[0]))
p0 = engineering_constants(runs[0][2])
s0, v0 = feature_sizes(rho_0)
panels = [('initial bow-tie', rho_0, f'nu_xy {p0["nu_xy"]:.2f}, nu_yx {p0["nu_yx"]:.2f}\n'
                                     f'min solid {s0} px, min void {v0} px')]
for folder, rho, C_0, C_t, C_opt, attributes in runs:
    pt, po = engineering_constants(C_t), engineering_constants(C_opt)
    grey = np.mean((rho > 0.1) & (rho < 0.9))
    solid_size, void_size = feature_sizes(rho)
    panels.append((f'target nu_xy {pt["nu_xy"]:.2f}', rho,
                   f'achieved nu_xy {po["nu_xy"]:.3f}, nu_yx {po["nu_yx"]:.3f}\n'
                   f'C11 {C_opt[0, 0]:.4f} ({C_t[0, 0]:.4f}), C22 {C_opt[1, 1]:.4f} ({C_t[1, 1]:.4f})\n'
                   f'grey {100 * grey:.0f} %, min solid {solid_size} px, min void {void_size} px'))
for ax, (title, rho, text) in zip(axes, panels):
    ax.imshow(np.tile(rho, (3, 3)).T, origin='lower', cmap='Greys', vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(text, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                   'compare_targets_' + time.strftime('%Y%m%d-%H%M%S') + '.png')
fig.savefig(out, dpi=110)
print('wrote', out)
