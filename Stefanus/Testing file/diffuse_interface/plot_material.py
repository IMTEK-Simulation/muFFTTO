"""mu and lambda per pixel (log10) of the sharp and the diffuse interfaces (exp_data/material_N<n>_w{0,1,2}.npz from
stefanus_diffuse.py --dump_material), drawn on their exact pixels, plus a profile along y = 0 (through a void and the
inclusion).  -> figures/material_diffuse.png"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
widths = tuple(int(w) for w in sys.argv[2].split(",")) if len(sys.argv) > 2 else (0, 1, 2)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
fig, axes = plt.subplots(3, 3, figsize=(12, 12.5), layout='constrained', height_ratios=[1, 1, 0.55])
for col, w in enumerate(widths):
    d = np.load(os.path.join(HERE, 'exp_data', f'material_N{N}_w{w}.npz'))
    Lx, Ly = d['domain_size'] * float(d['R_MM'])
    nx, ny = d['mu'].shape
    for row, (key, label, vmin, vmax) in enumerate((('mu', '$\\log_{10}\\,\\mu/\\mu_m$', -4, 1),
                                                    ('lam', '$\\log_{10}\\,\\lambda/\\mu_m$', -4, 3))):
        ax = axes[row, col]
        im = ax.imshow(np.log10(d[key]).T, origin='lower', extent=(0, Lx, 0, Ly), interpolation='none',
                       cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set(title=f'{label}, {N} px, width {w} px' if w else f'{label}, {N} px, sharp', xlabel='x [mm]', ylabel='y [mm]')
        if col == 2:
            fig.colorbar(im, ax=ax, shrink=0.85)
    ax = axes[2, col]
    xs = (np.arange(nx) + 0.5) * Lx / nx
    ax.step(xs, np.log10(d['mu'][:, 0]), where='mid', color='tab:blue', label='$\\log_{10}\\mu$')
    ax.step(xs, np.log10(d['lam'][:, 0]), where='mid', color='tab:orange', label='$\\log_{10}\\lambda$')
    ax.set(xlabel='x [mm] (row y = 0)', title='profile through void and inclusion', ylim=(-4.5, 3.5))
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
out = os.path.join(HERE, 'figures', f'material_diffuse_N{N}' + ('' if len(sys.argv) < 3 else '_w' + sys.argv[2].replace(',', '-')) + '.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=130)
print('saved', out)
