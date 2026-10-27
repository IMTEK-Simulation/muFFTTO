"""
Deformation of the clamped cube (box_clamp_3D.py, N <= 32 stores the displacement) and CG convergence.

The outer faces are clamped, so the cube's surface does not move; the deformation is shown on cuts through it:
    1. horizontal cut z = 1/4 through the centres of 4 holes (the in-plane displacement u_x, u_y)
    2. horizontal cut z = 1/2 between the holes
    3. vertical cut x = 1/4 through 4 holes (in-plane u_y, u_z)
each the voxel faces of that layer drawn on their exact pixels with the displaced nodes (magnified), coloured by |u|;
    4. CG convergence (energy-error estimate) of plain Green, corrected Green with inner PCG, and the family FFT once

usage:  N=32 python box_clamp_3D_deformation.py      -> exp_data/box_clamp_3D_deformation_N<n>.png
"""
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

HERE = os.path.dirname(os.path.realpath(__file__))
RUNS = os.path.join(HERE, 'exp_data', 'runs_box3d')
n = int(os.environ.get('N', 32))
plt.rcParams.update({'font.size': 12})


def load(variant):
    f = os.path.join(RUNS, f'N{n}_{variant}.npz')
    return {k: v[()] for k, v in np.load(f).items()} if os.path.exists(f) else None


r = load('family1e-06')
u, solid = r['u_cube'], r['solid_cube']  # (3, n+1, n+1, n+1), (n, n, n)
h = 1.0 / n
umax = np.abs(u).max()
magnify = 0.06 / umax  # the largest displacement drawn as 6 % of the side
u_mag = np.sqrt((u ** 2).sum(axis=0))


def cut(ax, axis, k, title):
    """voxel layer k normal to axis: its faces at the node plane k, nodes displaced in-plane"""
    a, b = [d for d in range(3) if d != axis]  # in-plane axes (drawn as horizontal, vertical)
    u_plane = np.take(u, k, axis=axis + 1)  # (3, n+1, n+1) over (a, b)
    s_layer = np.take(solid, min(k, n - 1), axis=axis)
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing='ij')
    X = i * h + magnify * u_plane[a]
    Y = j * h + magnify * u_plane[b]
    polys, values = [], []
    mag = np.take(u_mag, k, axis=axis)
    for p in range(n):
        for q in range(n):
            if not s_layer[p, q]:
                continue
            corners = [(p, q), (p + 1, q), (p + 1, q + 1), (p, q + 1)]
            polys.append([(X[c], Y[c]) for c in corners])
            values.append(np.mean([mag[c] for c in corners]))
    pc = PolyCollection(polys, array=np.array(values), cmap='viridis', edgecolors='face', linewidths=0.2)
    pc.set_clim(0, u_mag.max())
    ax.add_collection(pc)
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color='0.4', lw=1, ls='--')
    ax.set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03), aspect='equal', title=title,
           xlabel='xyz'[a], ylabel='xyz'[b])
    return pc


fig, axes = plt.subplots(1, 4, figsize=(23, 6), layout='constrained')
pc = cut(axes[0], 2, n // 4, f'cut z = 1/4 (through 4 holes)')
cut(axes[1], 2, n // 2, 'cut z = 1/2 (between the holes)')
cut(axes[2], 0, n // 4, 'cut x = 1/4 (through 4 holes)')
fig.colorbar(pc, ax=axes[:3], label='|u|', shrink=0.8, pad=0.01)

ax = axes[3]
for variant, colour, ls, label in (('green', 'tab:green', '--', 'plain Green'),
                                   ('family1e-06', 'tab:orange', '-', 'corrected Green (inner PCG, family FFT)'),
                                   ('familydirect', 'tab:red', '-.', 'corrected Green, family FFT once'),
                                   ('dense', 'k', ':', 'corrected Green, dense S')):
    rv = load(variant)
    if rv is None or 'energy_error_sq' not in rv:
        continue
    e = np.sqrt(np.abs(rv['energy_error_sq']))
    ax.semilogy(e / e[0], ls=ls, color=colour, lw=2, label=f'{label}: {abs(int(rv["cg_its"]))} its')
ax.set(xlabel='outer CG iteration k', ylabel='energy error of iterate k (relative)',
       title='outer CG convergence\n(the error of iterate k is estimated 10 its later)')
ax.grid(True, color='0.92', which='both')
ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon=False, fontsize=10)
fig.suptitle(f'Cube with 8 holes, all six faces clamped, body force in -y, N = {n}: deformation on cuts '
             f'(magnified x{magnify:.0f}; the clamped faces do not move)', fontsize=13)
out = os.path.join(HERE, 'exp_data', f'box_clamp_3D_deformation_N{n}.png')
fig.savefig(out, dpi=110)
print('wrote', out)
