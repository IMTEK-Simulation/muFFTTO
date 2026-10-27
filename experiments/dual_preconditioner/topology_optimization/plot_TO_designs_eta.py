"""Final TO designs over the grid for the two interface widths eta = 1/32 and eta = 0.014 (D = 1), soft phase 1e-3 and
1e-5, both preconditioners: one cell per panel on its exact pixels, with the stress misfit, the volume fraction, the
share of grey pixels (0.1 < rho < 0.9) and the effective C_11 / C_12 / C_33 (Voigt) in the title.

    python plot_TO_designs_eta.py [PRECONDITIONER]     (default dual)  -> figures/TO_designs_eta_<prec>.png
"""
import glob
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
PREC = sys.argv[1] if len(sys.argv) > 1 else 'dual'
GRIDS = [128, 256, 512]
ROWS = [('0.03125', 3), ('0.014', 3), ('0.03125', 5), ('0.014', 5)]
SOFT = {3: '0.001', 5: '1e-05'}
TARGET = (0.2893, -0.0964, 0.1929)

fig, axes = plt.subplots(len(ROWS), len(GRIDS), figsize=(2.6 * len(GRIDS) + 0.9, 2.95 * len(ROWS)))
for i, (eta, e) in enumerate(ROWS):
    for j, N in enumerate(GRIDS):
        ax = axes[i, j]
        ax.set_xticks([])
        ax.set_yticks([])
        f = glob.glob(os.path.join(HERE, 'data', 'example_2D_elasticity_TO',
                                   f'N{N}_soft{SOFT[e]}_energy-warm_{PREC}_eta_{eta}_w_5.0_l_bfgs_final.nc'))
        if not f:
            ax.text(0.5, 0.5, 'running' if eta == '0.014' and N == 512 else '--', ha='center', va='center',
                    transform=ax.transAxes, color='0.4')
            for side in ax.spines.values():
                side.set_color('0.85')
            continue
        d = netCDF4.Dataset(f[0])
        x = np.asarray(d.variables['phase_field_solution'][:]).squeeze()
        a = d.__dict__
        C = np.asarray(a['homogenized_C_voigt']).reshape(3, 3)
        ax.imshow(x.T, origin='lower', cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(f'${N}^2$: misfit {np.asarray(a["norms_sigma"])[-1]:.3f}, vol {x.mean():.2f}, '
                     f'grey {100 * np.mean((x > 0.1) & (x < 0.9)):.0f}%\n'
                     f'$C_{{11}}$ {C[0, 0]:.3f}  $C_{{12}}$ {C[0, 1]:.3f}  $C_{{33}}$ {C[2, 2]:.3f}', fontsize=7.5)
    axes[i, 0].set_ylabel(rf'$\eta={"1/32" if eta == "0.03125" else eta}$, soft $10^{{-{e}}}$', fontsize=10)
fig.suptitle(f'Final designs, {PREC.replace("_", "-")} preconditioner (one cell, exact pixels); target $C_{{11}}$ '
             f'{TARGET[0]:.3f}, $C_{{12}}$ {TARGET[1]:.3f}, $C_{{33}}$ {TARGET[2]:.3f}', fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.975))
fig.savefig(os.path.join(HERE, 'figures', f'TO_designs_eta_{PREC}.png'), dpi=250)
