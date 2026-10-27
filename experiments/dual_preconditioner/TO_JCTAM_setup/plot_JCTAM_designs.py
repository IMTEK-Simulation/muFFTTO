"""Designs of the JCTAM-setup TO runs (example_2D_elasticity_TO_JCTAM.py): final phase field (or the latest iterate of a
run still going, from _history.nc), rows: kappa_void x preconditioner, columns: grid. 2 x 2 periodic cells, exact pixels.

    python plot_JCTAM_designs.py      -> figures/JCTAM_designs.png
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
DATA = os.path.join(HERE, 'data', 'example_2D_elasticity_TO_JCTAM')
GRIDS = [16, 32, 64, 128, 256]
ROWS = [('0', 'Green'), ('0', 'Green_Jacobi'), ('1e-05', 'Green'), ('1e-05', 'Green_Jacobi'), ('1e-05', 'dual')]


def design(N, soft, p):
    # 64^2, kappa 1e-5 has the default name without the N/soft prefix
    prefix = '' if (N == 64 and soft == '1e-05') else f'N{N}_soft{soft}_residual_'
    base = os.path.join(DATA, f'{prefix}{p}_eta_0.01_w_20.0_l_bfgs')
    if os.path.exists(base + '_final.nc'):
        d = netCDF4.Dataset(base + '_final.nc')
        a = d.__dict__
        return np.asarray(d.variables['phase_field_solution'][:]).squeeze(), int(a['nb_iterations']), True
    if os.path.exists(base + '_history.nc'):
        d = netCDF4.Dataset(base + '_history.nc')
        x = d['phase_field_iterate']
        return np.asarray(x[-1]).squeeze(), x.shape[0], False
    return None


fig, axes = plt.subplots(len(ROWS), len(GRIDS), figsize=(2.2 * len(GRIDS) + 1.2, 2.35 * len(ROWS)))
for i, (soft, p) in enumerate(ROWS):
    for j, N in enumerate(GRIDS):
        ax = axes[i, j]
        ax.set_xticks([]); ax.set_yticks([])
        r = design(N, soft, p)
        if r is None:
            ax.text(0.5, 0.5, '--', ha='center', va='center', transform=ax.transAxes, color='0.5')
            for s in ax.spines.values(): s.set_color('0.85')
        else:
            x, it, done = r
            ax.imshow(np.tile(x.T, (2, 2)), origin='lower', cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
            ax.set_title(f'{it} it.' + ('' if done else ' (stopped)'), fontsize=8)
        if i == 0:
            ax.text(0.5, 1.25, f'${N}^2$', ha='center', transform=ax.transAxes, fontsize=11)
    axes[i, 0].set_ylabel(('$\\kappa=0$' if soft == '0' else '$\\kappa=10^{-5}$') + '\n' + p.replace('_', '-'), fontsize=9)
fig.tight_layout()
os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
fig.savefig(os.path.join(HERE, 'figures', 'JCTAM_designs.png'), dpi=180)
