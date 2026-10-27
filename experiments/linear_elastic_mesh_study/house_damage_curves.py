"""
Load paths of the cracking house runs (house_damage_3D.py) from their NetCDF files: wind vs. eaves displacement,
cracked masonry volume, and the CG iterations per elastic solve, with and without self-weight.

usage:  H=0.125 python house_damage_curves.py       writes exp_data/house_damage_curves_h<h>.png
"""
import os

import numpy as np
import netCDF4
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

h = float(os.environ.get('H', 0.125))
HERE = os.path.dirname(os.path.realpath(__file__))
RUNS = [(f'damage_h{h:g}_ft0.2', 'with self-weight', 'C3'), (f'damage_h{h:g}_ft0.2_nograv', 'without self-weight', 'C0')]

fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), layout='constrained')
for case, label, c in RUNS:
    path = os.path.join(HERE, 'exp_data', 'house', f'{case}.nc')
    if not os.path.exists(path):
        continue
    with netCDF4.Dataset(path) as ds:
        fv = {k: np.asarray(ds[k][:]) for k in ('converged', 'load', 'u_eaves', 'cracked_volume', 'cg_u')}
    conv = fv['converged'] > 0
    axes[0].plot(np.r_[0, 1e3 * fv['u_eaves'][conv]], np.r_[0, fv['load'][conv]], 'o-', ms=2.5, color=c, label=label)
    axes[0].plot(1e3 * fv['u_eaves'], fv['load'], '-', lw=0.6, color=c, alpha=0.4)  # incl. staggered iterations
    axes[1].plot(1e3 * fv['u_eaves'][conv], fv['cracked_volume'][conv], 'o-', ms=2.5, color=c, label=label)
    nb_solves = 1 if case.endswith('nograv') else 2  # without self-weight its solve is zero (0 iterations)
    axes[2].plot(fv['cg_u'] / nb_solves, color=c, lw=1, label=label)
axes[0].set_xlabel('mean horizontal eaves displacement (controlled) [mm]')
axes[0].set_ylabel('wind pressure q [kPa]')
axes[0].set_title('load path (thin: the staggered iterations of the snaps)')
axes[1].set_xlabel('mean horizontal eaves displacement [mm]')
axes[1].set_ylabel('cracked masonry (voxel mean d > 0.9) [m$^3$]')
axes[1].set_title('crack growth')
axes[2].set_xlabel('staggered iteration (all load steps)')
axes[2].set_ylabel('CG iterations per elastic solve')
axes[2].set_title('elastic CG, corrected Green')
for ax in axes:
    ax.grid(True, color='0.9')
    ax.legend(frameon=False)
fig.suptitle(f'cracking house, voxel {h * 100:g} cm, f$_t$ = 0.2 MPa', fontsize=12)
out = os.path.join(HERE, 'exp_data', f'house_damage_curves_h{h:g}.png')
fig.savefig(out, dpi=140)
print(f'wrote {out}')
