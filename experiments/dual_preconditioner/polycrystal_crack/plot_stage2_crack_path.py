"""
Stage 2: final crack (d >= 0.9, last frame) on top of the grains and their boundaries, constant Gc against weak grain
boundaries (Gc x GC_GB on the 2-pixel boundary band); exact pixels (nearest), grain colour = Young modulus.
usage:  python plot_stage2_crack_path.py           writes figures/polycrystal_stage2_crack_path.png
"""
import os

import numpy as np
import netCDF4
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import polycrystal_geometry as geo

HERE = os.path.dirname(os.path.realpath(__file__))
N = 128
CASES = [(f'AT2_N{N}_Green_poly_l0.03125', 'constant Gc, l = 0.031 (l/h = 4)'),
         (f'AT2_N{N}_Green_poly_gb0.3_l0.015625', 'grain boundaries Gc x 0.3, l = 0.016 (l/h = 2)')]

ids = geo.grain_ids(N)
gb = np.zeros(ids.shape, dtype=bool)
for ax_ in (0, 1):
    for sh in (1, -1):
        gb |= np.roll(ids, sh, axis=ax_) != ids
E = geo.grain_moduli()[ids]
crack0, void = geo.crack_and_void(N)

plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(1, len(CASES), figsize=(7 * len(CASES), 6.6), layout='constrained')
for ax, (name, title) in zip(axes, CASES):
    path = os.path.join(HERE, 'exp_data', name + '.nc')
    if not os.path.exists(path):
        ax.set_title(f'{title}\n(not run)')
        continue
    with netCDF4.Dataset(path) as ds:
        d = np.asarray(ds['damage'][-1, 0])
    ax.imshow(E.T, origin='lower', extent=(0, 1, 0, 1), interpolation='nearest', cmap='Greys', vmin=0, vmax=6)
    ax.imshow(np.ma.masked_where(~gb, gb).T, origin='lower', extent=(0, 1, 0, 1), interpolation='nearest',
              cmap=ListedColormap(['tab:blue']), alpha=0.35)
    ax.imshow(np.ma.masked_where(~(crack0 | void), crack0 | void).T, origin='lower', extent=(0, 1, 0, 1),
              interpolation='nearest', cmap=ListedColormap(['k']))
    cr = d >= 0.9
    ax.imshow(np.ma.masked_where(~cr, cr).T, origin='lower', extent=(0, 1, 0, 1), interpolation='nearest',
              cmap=ListedColormap(['tab:red']))
    on_gb = (cr & gb & ~(crack0 | void)).sum() / max((cr & ~(crack0 | void)).sum(), 1)
    ax.set_title(f'{title}\ncrack pixels on grain boundaries: {on_gb:.0%} (boundary band: {gb.mean():.0%} of cell)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
fig.suptitle(f'final crack (red, d >= 0.9) over grains (grey: E) and grain-boundary band (blue), N = {N}; '
             'black: pre-crack + void layer', fontsize=13)
out = os.path.join(HERE, 'figures', 'polycrystal_stage2_crack_path.png')
fig.savefig(out, dpi=200)
print(f'wrote {out}')
