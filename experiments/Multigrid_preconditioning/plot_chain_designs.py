"""
Final designs of the overnight 2D TO chains (MG + Helmholtz c = 4, eta = 4h hierarchy, new Cahn-Hilliard starts):
overview figures/TO_chains_new_CH.png (rows: starts bi_s11, bi_s12, drop_s11; columns: CH start, 256^2, 512^2,
1024^2, 2048^2; nearest-neighbour, dark = stiff) and exact-pixel images figures/TO_chain_<start>_N2048.png
(one image pixel = one grid pixel).

    python plot_chain_designs.py
"""
import re
from pathlib import Path

import numpy as np
import netCDF4
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
D = HERE / 'data' / 'nemo_designs'
STARTS = ['bi_s11', 'bi_s12', 'drop_s11']
GRIDS = [256, 512, 1024, 2048]


def design(N, lab):
    f = D / f'N{N}_soft0.0001_energy-warm_MG_eta_{4 / N}_pfw0.2_l_bfgs_mgTO_{lab}_lbfgs_helm_c4_final.nc'
    return np.asarray(netCDF4.Dataset(f)['phase_field_solution'][:]).reshape(N, N) if f.exists() else None


def final_f(N, lab):
    log = HERE / 'logs' / 'nemo_TO' / f'TO_N{N}_MG_{lab}_eta4h_lbfgs_helm_c4.log'
    ev = re.findall(r'EVAL (\d+) f ([\d.eE+-]+) wall ([\d.]+)', log.read_text()) if log.exists() else []
    return (float(ev[-1][1]), int(ev[-1][0]), float(ev[-1][2]) / 3600) if ev else (np.nan, 0, np.nan)


plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(len(STARTS), len(GRIDS) + 1, figsize=(4 * (len(GRIDS) + 1), 4.3 * len(STARTS)))
for r, lab in enumerate(STARTS):
    start = np.asarray(netCDF4.Dataset(HERE / 'data' / 'CH_starts_new' / f'CH_start_{lab}.nc')['phase_field_solution'][:])
    start = start.reshape(256, 256)
    axes[r, 0].imshow(start.T, origin='lower', cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
    axes[r, 0].set_title(f'{lab}: Cahn-Hilliard start')
    for c, N in enumerate(GRIDS):
        ax = axes[r, c + 1]
        rho = design(N, lab)
        if rho is None:
            ax.axis('off')
            continue
        f, n_ev, hours = final_f(N, lab)
        ax.imshow(rho.T, origin='lower', cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(f'{N}²: f {f:.4f}\n{n_ev} evaluations, {hours:.2f} h')
        if N == 2048:
            plt.imsave(HERE / 'figures' / f'TO_chain_{lab}_N2048.png', rho.T[::-1], cmap='gray_r', vmin=0, vmax=1)
    for ax in axes[r]:
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('2D TO, η = 4h hierarchy (each grid starts from the previous design), multigrid + Helmholtz L-BFGS; '
             'dark = stiff', fontsize=13)
fig.tight_layout()
out = HERE / 'figures' / 'TO_chains_new_CH.png'
fig.savefig(out, dpi=200)
print(out)
