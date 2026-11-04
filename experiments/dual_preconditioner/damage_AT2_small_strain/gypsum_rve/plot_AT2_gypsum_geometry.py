"""
Gypsum RVE geometry (25.6 x 25.6 mm, nine disks, radii 1.3 - 3.6 mm) as the solver sees it on 64^2 / 128^2 / 256^2
(exact pixels: a pixel takes the phase of its lower-left node), with a bar of the length scale l = 0.5 mm, and the final
damage of the dual runs on 128^2 and 256^2 for both variants (pores, stiff grains).

usage:  python plot_AT2_gypsum_geometry.py      writes figures/AT2_gypsum_geometry.png
"""
import os
import sys

import numpy as np
import netCDF4
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '../../../..')))
from muFFTTO import microstructure_library  # noqa: E402

L_MM, CELL_MM = 0.5, 25.6


def phase(n):
    x = np.arange(n) / n
    return microstructure_library.get_geometry(nb_voxels=(n, n), microstructure_name='polydisperse_pores',
                                               coordinates=np.array(np.meshgrid(x, x, indexing='ij')))


def final_damage(n, tag):
    f = os.path.join(HERE, 'exp_data', f'AT2_N{n}_dual_polydisperse_pores{tag}_l0.0195915_gypsum.nc')
    with netCDF4.Dataset(f) as ds:
        d = np.asarray(ds['damage'][-1, 0])
    return 0.25 * (d + np.roll(d, -1, 0) + np.roll(d, -1, 1) + np.roll(np.roll(d, -1, 0), -1, 1))


fig, axes = plt.subplots(2, 3, figsize=(16, 11), layout='constrained', dpi=140)
for ax, n in zip(axes[0], (64, 128, 256)):
    ax.imshow(phase(n).T, origin='lower', extent=(0, CELL_MM, 0, CELL_MM), interpolation='nearest',
              cmap=matplotlib.colors.ListedColormap(['#e8c440', '#3d4a6b']))
    ax.plot([1, 1 + L_MM], [1, 1], 'w-', lw=4), ax.text(1, 1.6, 'l = 0.5 mm', color='w', fontsize=11)
    ax.set_title(f'{n}$^2$: h = {CELL_MM / n:.2f} mm, l/h = {L_MM / (CELL_MM / n):g}\n'
                 '(dark: gypsum matrix, yellow: pore or grain)', fontsize=12)
    ax.set_xlabel('x [mm]', fontsize=11), ax.set_ylabel('y [mm]', fontsize=11)
for ax, (n, tag, name) in zip(axes[1], ((128, '_holes0.0001', 'pores'), (256, '_holes0.0001', 'pores'),
                                        (256, '_incl10', 'stiff grains'))):
    d = final_damage(n, tag)
    img = plt.cm.inferno_r(np.clip(d, 0, 1) * 0.9)
    img[phase(n) == 0] = (0.55, 0.75, 0.95, 1) if 'pores' in name else (0.6, 0.6, 0.6, 1)
    ax.imshow(np.transpose(img, (1, 0, 2)), origin='lower', extent=(0, CELL_MM, 0, CELL_MM), interpolation='nearest')
    ax.set_title(f'final damage, {name}, {n}$^2$ (l/h = {L_MM / (CELL_MM / n):g})\n'
                 f'({"pores light blue" if "pores" in name else "grains grey"}, dark = cracked)', fontsize=12)
    ax.set_xlabel('x [mm]', fontsize=11), ax.set_ylabel('y [mm]', fontsize=11)
out = os.path.join(HERE, 'figures', 'AT2_gypsum_geometry.png')
fig.savefig(out)
print(f'wrote {out}')
