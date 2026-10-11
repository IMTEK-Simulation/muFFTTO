"""
Overview report of finished runs of example_2D_elasticity_TO_bowtie.py, one row per run:

  initial design | optimized design (2 x 2 cells) | convergence (mismatch, grey measure, beta stages) |
  relative error of C_11, C_22, C_12, C_66 (initial vs optimized) | summary (nu, iterations, time, feature size)

    python report.py [--logs DIR] [--title TEXT] [--name NAME] data/<run> [data/<run> ...]

``--logs DIR``: directory of the console logs; a log that names the run folder ("wrote <folder>") gives the
iterations at which the beta stages ended (dashed lines). The figure is written to data/<NAME>_<time>.png.
"""
import argparse
import glob
import os
import re
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from muFFTTO import geometry  # noqa: E402
from feature_size import feature_sizes  # noqa: E402

parser = argparse.ArgumentParser(description='overview report of bow-tie TO runs')
parser.add_argument('runs', nargs='+')
parser.add_argument('--logs', default=None, help='directory with the console logs (beta stage ends)')
parser.add_argument('--title', default='')
parser.add_argument('--name', default='report')
arguments = parser.parse_args()


def attribute(dataset, name, default=None):
    return dataset.getncattr(name) if name in dataset.ncattrs() else default


def engineering(C):
    S = np.linalg.inv(C)
    return {'E_x': 1 / S[0, 0], 'E_y': 1 / S[1, 1], 'G_xy': 1 / S[2, 2], 'nu_xy': -S[1, 0] / S[0, 0],
            'nu_yx': -S[0, 1] / S[1, 1]}


def initial_design(dataset, n):
    """The initial design of the run, rebuilt as in the example (bow-tie or the guess of the original example)."""
    x = np.arange(n) / n
    coords = np.array(np.meshgrid(x, x, indexing='ij'))
    if attribute(dataset, 'init', 'bowtie') == 'original':
        np.random.seed(0)
        rho = (np.sin(4 * np.pi * coords[0]) + np.sin(4 * np.pi * coords[1]) + 2) / 4 + 0.5 * np.random.rand(n, n)
        return np.clip(rho, 0, 1), 'initial guess (original example)'
    theta, thickness = float(attribute(dataset, 'theta')), float(attribute(dataset, 'thickness'))
    return (geometry.get('reentrant_honeycomb', coords, theta=theta, thickness=thickness).astype(float),
            f'bow-tie, theta {theta:g}, t {thickness:g}')


def stage_ends(folder):
    """Iteration counts at the end of the beta stages, from the console log of the run (if found)."""
    if not arguments.logs:
        return []
    for log in glob.glob(os.path.join(arguments.logs, '*.log')):
        text = open(log, errors='replace').read()
        if os.path.basename(os.path.normpath(folder)) in text:
            return [(float(b), int(i)) for b, i in re.findall(r'^beta ([\d.]+): (\d+) iterations so far', text, re.M)]
    return []


def grey(rho):
    return 4 * np.mean(rho * (1 - rho))


def method_name(dataset):
    if attribute(dataset, 'regularization') == 'filter':
        kind = 'robust filter' if int(attribute(dataset, 'robust', 0)) else 'filter'
        return f'{kind}, R {float(attribute(dataset, "filter_radius_pixels")):g} px'
    return f'phase field, eta {float(attribute(dataset, "eta")) * 64:g} px, dw {float(attribute(dataset, "double_well_depth")):g}'


entries = [('C11', 0, 0), ('C22', 1, 1), ('C12', 0, 1), ('C66', 2, 2)]
nb = len(arguments.runs)
height = 4.3 * nb + 0.6
fig = plt.figure(figsize=(22, height))
grid = fig.add_gridspec(nb, 5, width_ratios=[1, 1, 1.5, 1.2, 1.25], wspace=0.45, hspace=0.38,
                       top=1 - 0.9 / height, bottom=0.4 / height)
for row, folder in enumerate(arguments.runs):
    folder = os.path.normpath(folder)
    final = netCDF4.Dataset(os.path.join(folder, 'final.nc'))
    history = netCDF4.Dataset(os.path.join(folder, 'history.nc'))
    n = int(np.asarray(attribute(final, 'nb_of_pixels'))[0])
    rho_final = np.asarray(final.variables['rho_final'][:]).reshape(n, n)
    rho_0, init_label = initial_design(final, n)
    C_0 = np.asarray(attribute(final, 'C_initial_voigt')).reshape(3, 3)
    C_t = np.asarray(attribute(final, 'C_target_voigt')).reshape(3, 3)
    C_opt = np.asarray(attribute(final, 'C_optimized_voigt')).reshape(3, 3)
    mismatch = np.asarray(history.variables['mismatch'][:])
    grey_history = [grey(np.asarray(history.variables['rho_physical'][k]).squeeze())
                    for k in range(history.variables['rho_physical'].shape[0])]
    nb_iterations = int(attribute(final, 'nb_iterations'))
    nb_evaluations = len(np.atleast_1d(attribute(final, 'log_objective')))
    elapsed = float(attribute(final, 'elapsed'))
    method = method_name(final)

    # 1. initial design
    ax = fig.add_subplot(grid[row, 0])
    ax.imshow(np.tile(rho_0.T, (2, 2)), cmap='Greys', vmin=0, vmax=1, origin='lower')
    ax.set_xticks([]), ax.set_yticks([])
    p0 = engineering(C_0)
    ax.set_title(f'{init_label}\nnu_xy {p0["nu_xy"]:.3f}', fontsize=9)

    # 2. optimized design
    ax = fig.add_subplot(grid[row, 1])
    ax.imshow(np.tile(rho_final.T, (2, 2)), cmap='Greys', vmin=0, vmax=1, origin='lower')
    ax.set_xticks([]), ax.set_yticks([])
    p = engineering(C_opt)
    ax.set_title(f'optimized ({method})\nnu_xy {p["nu_xy"]:.3f}, target {float(attribute(final, "nu_target")):g}',
                 fontsize=9)

    # 3. convergence
    ax = fig.add_subplot(grid[row, 2])
    iterations = np.arange(1, len(mismatch) + 1)
    ax.semilogy(iterations, mismatch, color='C0', label='stress mismatch')
    ax.set_xlabel('L-BFGS iteration'), ax.set_ylabel('mismatch', color='C0')
    ax.grid(alpha=0.3)
    previous = 0
    for beta, end in stage_ends(folder):
        ax.axvline(end, color='gray', ls='--', lw=0.7)
        ax.text((previous + end) / 2, 1.02, f'β{beta:g}', transform=ax.get_xaxis_transform(), ha='center',
                fontsize=7, color='gray')
        previous = end
    twin = ax.twinx()
    twin.plot(iterations, grey_history, color='C3', lw=1, label='grey measure')
    twin.set_ylim(0, max(0.05, 1.1 * max(grey_history))), twin.set_ylabel('grey measure 4<ρ(1-ρ)>', color='C3')

    # 4. relative error of the stiffness entries
    ax = fig.add_subplot(grid[row, 3])
    scale = np.max(np.abs(C_t))
    error_0 = [abs(C_0[i, j] - C_t[i, j]) / max(abs(C_t[i, j]), 1e-3 * scale) for _, i, j in entries]
    error_opt = [abs(C_opt[i, j] - C_t[i, j]) / max(abs(C_t[i, j]), 1e-3 * scale) for _, i, j in entries]
    position = np.arange(len(entries))
    ax.bar(position - 0.2, np.maximum(error_0, 1e-7), 0.4, color='lightgray', label='initial')
    ax.bar(position + 0.2, np.maximum(error_opt, 1e-7), 0.4, color='C2', label='optimized')
    ax.set_yscale('log'), ax.set_ylim(1e-7, 20), ax.axhline(0.05, color='k', ls=':', lw=0.8)
    ax.set_xticks(position, [name for name, _, _ in entries])
    ax.legend(fontsize=7, loc='upper right'), ax.grid(alpha=0.3, axis='y')
    ax.set_title('fit: |C - C_target| / |C_target| (dotted: 5 %)', fontsize=9)

    # 5. summary
    ax = fig.add_subplot(grid[row, 4])
    ax.axis('off')
    solid, void = feature_sizes(rho_final)
    lines = [f'{os.path.basename(folder)[:60]}', '',
             f'{"":8}{"target":>10}{"optimized":>11}']
    lines += [f'{name:8}{C_t[i, j]:10.4f}{C_opt[i, j]:11.4f}' for name, i, j in entries]
    pt = engineering(C_t)
    lines += [f'{"nu_xy":8}{pt["nu_xy"]:10.3f}{p["nu_xy"]:11.3f}', f'{"nu_yx":8}{pt["nu_yx"]:10.3f}{p["nu_yx"]:11.3f}',
              '', f'iterations {nb_iterations}, evaluations {nb_evaluations}',
              f'time {elapsed / 60:.0f} min, final mismatch {mismatch[-1]:.2e}',
              f'grey measure {grey(rho_final):.3f}, solid fraction {np.mean(rho_final > 0.5):.2f}',
              f'min solid {solid} px, min void {void} px']
    ax.text(0, 1, '\n'.join(lines), va='top', family='monospace', fontsize=8.2, transform=ax.transAxes)
    final.close(), history.close()

if arguments.title:
    fig.suptitle(arguments.title, fontsize=13, y=1 - 0.2 / height)
out = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data',
                   f'{arguments.name}_{time.strftime("%Y%m%d-%H%M%S")}.png')
fig.savefig(out, dpi=85, bbox_inches='tight')
print(f'wrote {out}')
