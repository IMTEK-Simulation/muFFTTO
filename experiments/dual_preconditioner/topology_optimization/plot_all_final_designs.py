"""All final designs of example_2D_elasticity_TO.py in data/example_2D_elasticity_TO/ on one page: rows = (grid, soft
phase), columns = preconditioner -> figures/all_final_designs.png

    python plot_all_final_designs.py
"""
import glob
import os
import re

import matplotlib
import netCDF4
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
# the local copy of example_2D_elasticity_TO.py and its data (this folder)
TO = os.path.dirname(os.path.realpath(__file__))
DATA = os.path.join(TO, 'data', 'example_2D_elasticity_TO')
PRECS = [('Green', 'Green (C_0)'), ('Green_Jacobi', 'Green-Jacobi (2μI)'), ('dual', 'dual (C_0)')]

runs = {}
for path in glob.glob(os.path.join(DATA, '*_final.nc')):
    name = os.path.basename(path)
    m = re.match(r'N(\d+)_soft([0-9.e-]+)_energy_(Green_Jacobi|Green|dual)_', name)
    if m:
        row = (int(m[1]), float(m[2]), 'energy stop')
        p = m[3]
    else:                                                  # first 64^2 runs: soft 1e-5, residual stop, warm start
        p = re.match(r'(Green_Jacobi|Green|dual)_', name)[1]
        row = (64, 1e-5, 'residual stop, warm start')
    f = netCDF4.Dataset(path)
    h = netCDF4.Dataset(path.replace('_final.nc', '_history.nc'))
    runs[row, p] = (np.asarray(f.variables[list(f.variables)[-1]][:]).squeeze(), float(h['objective'][-1]),
                    int(h['nb_evaluations'][-1]))

rows = sorted({r for r, _ in runs}, key=lambda r: (r[2] != 'energy stop', r[0], -r[1]))
fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3.1 * len(rows)), facecolor='#fcfcfb')
for i, row in enumerate(rows):
    for j, (p, label) in enumerate(PRECS):
        ax = axes[i, j]
        ax.set_xticks([])
        ax.set_yticks([])
        if (row, p) not in runs:
            for side in ax.spines.values():
                side.set_visible(False)
            ax.text(0.5, 0.5, 'running / not run', ha='center', va='center', color='0.5', transform=ax.transAxes)
            continue
        design, objective, evaluations = runs[row, p]
        ax.pcolormesh(design.T, cmap='Greys', vmin=0, vmax=1)          # exact pixels
        ax.set_aspect('equal')
        ax.set_title(f'{label}\nobjective {objective:.4g}, {evaluations} evaluations', fontsize=8)
    axes[i, 0].set_ylabel(f'{row[0]}², soft {row[1]:g}' + ('' if row[2] == 'energy stop' else f'\n({row[2]})'),
                          fontsize=10, fontweight='bold')
fig.tight_layout()
out = os.path.join(HERE, 'figures', 'all_final_designs.png')
fig.savefig(out, dpi=110, facecolor='#fcfcfb')
print(out, len(runs), 'designs')
