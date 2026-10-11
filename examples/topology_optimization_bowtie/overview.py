"""
Overview of several methods (rows) over several Poisson targets (columns): the optimized design (2 x 2 cells) and,
under it, the reached nu_xy, the worst relative error of C11, C22, C12, C66, the minimum solid / void feature size,
the L-BFGS iterations, objective evaluations and the run time.

    python overview.py --row "label" 'data/<glob of the runs of method 1>' --row "label" '<glob 2>' ... [--name overview]

Red: off target (|nu - target| >= 0.02 or a C entry off by >= 5 %) or hinges (min solid or void < 3 px).
Each glob may match several runs; the newest run per target nu is shown. Written to data/<name>_<time>.png.
"""
import argparse
import glob
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402

import feature_size  # noqa: E402

here = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description='overview of methods x Poisson targets')
parser.add_argument('--row', nargs=2, action='append', metavar=('LABEL', 'GLOB'), required=True)
parser.add_argument('--nu', type=float, nargs='+', default=[-1.0, -0.5, 0.0, 0.5])
parser.add_argument('--title', default='')
parser.add_argument('--name', default='overview')
arguments = parser.parse_args()


def nu_xy(C):
    S = np.linalg.inv(C)
    return -S[0, 1] / S[0, 0]


def load(run):
    with netCDF4.Dataset(os.path.join(run, 'final.nc')) as d:
        rho = np.array(d.variables['rho_final'][:]).squeeze()
        a = {name: d.getncattr(name) for name in d.ncattrs()}
    return rho, a


rows, columns = len(arguments.row), len(arguments.nu)
fig, axes = plt.subplots(rows, columns, figsize=(3.1 * columns + 1.6, 3.9 * rows), squeeze=False)
for i, (label, pattern) in enumerate(arguments.row):
    runs = [r for r in glob.glob(pattern) if os.path.exists(os.path.join(r, 'final.nc'))]
    for j, nu in enumerate(arguments.nu):
        ax = axes[i, j]
        ax.set_xticks([])
        ax.set_yticks([])
        matching = sorted([r for r in runs if abs(float(load(r)[1]['nu_target']) - nu) < 1e-9])
        if j == 0:
            ax.set_ylabel(label, fontsize=10)
        if i == 0:
            ax.set_title(f'target nu_xy = {nu:g}', fontsize=11)
        if not matching:
            ax.text(0.5, 0.5, 'not run / not finished', ha='center', va='center', transform=ax.transAxes)
            continue
        run = matching[-1]
        rho, a = load(run)
        C = np.reshape(a['C_optimized_voigt'], (3, 3))
        C_t = np.reshape(a['C_target_voigt'], (3, 3))
        mask = rho > 0.5
        solid = feature_size.minimum_feature_size(mask)
        void = feature_size.minimum_feature_size(~mask)
        entries = [(0, 0), (1, 1), (0, 1), (2, 2)]
        on_target = abs(nu_xy(C) - nu) < 0.02
        if a.get('objective', 'stiffness') == 'poisson':
            fit = f'|nu - target| {abs(nu_xy(C) - nu):.1e}'
        else:
            error = max(abs(C[e] - C_t[e]) / abs(C_t[e]) if abs(C_t[e]) > 1e-12 else abs(C[e]) / abs(C_t[0, 0])
                        for e in entries)
            fit = f'max C error {error * 100:.2g} %'
            on_target = on_target and error < 0.05
        ax.imshow(np.tile(rho, (2, 2)).T, origin='lower', cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
        good = solid >= 3 and void >= 3 and on_target
        problems = ', '.join(([] if on_target else ['off target']) + ([] if solid >= 3 and void >= 3 else ['hinges']))
        ax.set_xlabel(f'nu_xy {nu_xy(C):.3f}   {fit}\n'
                      f'min solid / void {solid} / {void} px{f"  ({problems})" if problems else ""}\n'
                      f'L-BFGS it {int(a["nb_iterations"])}, evals {len(np.atleast_1d(a["log_objective"]))}, '
                      f'{float(a["elapsed"]) / 60:.0f} min', fontsize=8.5, color='black' if good else 'firebrick')
fig.suptitle(arguments.title, fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.985))
out = os.path.join(here, 'data', f'{arguments.name}_{time.strftime("%Y%m%d-%H%M%S")}.png')
fig.savefig(out, dpi=110)
print(f'wrote {out}')
