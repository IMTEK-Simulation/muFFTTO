"""
The 8 fibre realisations (seeds 0-7) at N = 256 with the latest damage field of a run (dual by default: the further
one); fibres grey, damage d in the matrix (inferno, d >= 0.05 shown), exact pixels. Cf. the paper's Fig. 13.
usage:  python plot_fibers_gallery.py [N] [precond]        writes figures/fibers_gallery_N<N>.png
"""
import os
import sys

import numpy as np
import netCDF4
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import fiber_geometry as geo

HERE = os.path.dirname(os.path.realpath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 256
P = sys.argv[2] if len(sys.argv) > 2 else 'dual'

plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(2, 4, figsize=(18, 9.4), layout='constrained')
for seed, ax in enumerate(axes.ravel()):
    ph = geo.phase(N, seed)
    ax.imshow(np.where(ph == 0, 0.55, 1.0).T, origin='lower', extent=(0, 512, 0, 512), interpolation='nearest',
              cmap='Greys_r', vmin=0, vmax=1)
    title = f'#{seed}'
    path = os.path.join(HERE, 'exp_data', f'AT2_N{N}_{P}_fibers_s{seed}_l0.0117188_k0.0001.nc')
    if os.path.exists(path):
        with netCDF4.Dataset(path) as ds:
            d = np.asarray(ds['damage'][-1, 0])
            e = float(ds['load'][-1])
        im = ax.imshow(np.ma.masked_where((d < 0.05) | (ph == 0), d).T, origin='lower', extent=(0, 512, 0, 512),
                       interpolation='nearest', cmap='inferno_r', vmin=0, vmax=1)
        title += f'   damage at E_xx = {100 * e:.1f} %'
    ax.set_title(title)
    ax.set_xlabel('x [um]')
    ax.set_ylabel('y [um]')
fig.colorbar(im, ax=axes, shrink=0.6, label='damage d (matrix)')
fig.suptitle(f'16 fibres, 45 %, N = {N}: fibres grey, latest damage of the {P} run (load in x)', fontsize=14)
out = os.path.join(HERE, 'figures', f'fibers_gallery_N{N}.png')
fig.savefig(out, dpi=150)
print(f'wrote {out}')
