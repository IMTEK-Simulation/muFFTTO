"""Shared setup: build the problem for a density field and solve it with G, GJ or PD."""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
RESULTS = os.path.join(HERE, '..', 'results')
os.makedirs(RESULTS, exist_ok=True)

from fem import mandel_isotropic, material_field, rhs  # noqa: E402
from pcg import pcg  # noqa: E402
from precond import make_green, make_green_jacobi, make_primal_dual, new_counter  # noqa: E402

METHODS = ('G', 'GJ', 'PD')
C0_ELASTIC = mandel_isotropic(2.0, 1.0)          # mu0 = 1, K0 = 2 mu0
E_MACRO = {1: np.array([1.0, 0.0]), 2: np.array([1.0, 1.0, 1.0])}


def problem(rho, d):
    C0 = np.eye(2) if d == 1 else C0_ELASTIC     # scalar: c = rho, c_ref = 1; elastic: C = rho C0, C_ref = C0
    C = material_field(rho, C0)
    n = rho.shape[0]
    return C0, C, rhs(C, E_MACRO[d], d, n)


def solve(rho, d, method, tol=1e-5, maxiter=2000, rho_clip=1e-8, subtract_mean=True, track_energy=True):
    n = rho.shape[0]
    C0, C, f = problem(rho, d)
    counter = new_counter()
    if method == 'G':
        M = make_green(C0, d, n, counter, subtract_mean)
    elif method == 'GJ':
        M = make_green_jacobi(C, C0, d, n, counter, subtract_mean)
    else:
        M = make_primal_dual(rho, C0, C0, d, n, counter, rho_clip=rho_clip, subtract_mean=subtract_mean)
    return pcg(C, f, M, counter, tol=tol, maxiter=maxiter, track_energy=track_energy)


def row(result, **keys):
    c = result['counter']
    return dict(keys, iterations=result['iterations'], converged=result['converged'],
                it_energy=result['it_energy'] if result['it_energy'] is not None else '',
                true_residual=f"{result['true_residual']:.2e}", K_applications=c['K'], M_applications=c['M'],
                ffts=c['fft'], time_s=f"{result['time']:.3f}")


def write_csv(name, rows):
    path = os.path.join(RESULTS, name)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


# ---- plotting -------------------------------------------------------------------------------------------------------
import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

SURFACE, TEXT_1, TEXT_2, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e1', '#a3a29c'
COLORS = {'G': '#2a78d6', 'GJ': '#eb6834', 'PD': '#1baf7a'}
NAMES = {'G': 'Green', 'GJ': 'Green-Jacobi', 'PD': 'primal-dual Green'}
BLUE_RAMP = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']   # light -> dark
ITERATION_CMAP = LinearSegmentedColormap.from_list('iterations', BLUE_RAMP)


def style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    if title:
        ax.set_title(title, loc='left', fontsize=11.5, color=TEXT_1, pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_2, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_2, fontsize=10)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=TEXT_2, labelsize=9.5)


def iteration_grids(rows, out, title, subtitle, contrast_key, contrasts):
    """Paper-style tables: iterations over grid size n (x) and data resolution p (y); one panel per method and
    contrast. Cells: iteration count, colour on a shared log scale; 2000 = not converged."""
    ns = sorted({r['n'] for r in rows})
    ps = sorted({r['p'] for r in rows})
    vmax = max(r['iterations'] for r in rows)
    norm = LogNorm(vmin=1, vmax=max(vmax, 10))
    fig, axes = plt.subplots(len(contrasts), 3, figsize=(15, 4.6 * len(contrasts)), squeeze=False, facecolor=SURFACE)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86 if len(contrasts) > 1 else 0.78, bottom=0.08,
                        hspace=0.45, wspace=0.18)
    fig.text(0.06, 0.965, title, fontsize=15, fontweight='bold', color=TEXT_1)
    fig.text(0.06, 0.935 if len(contrasts) > 1 else 0.9, subtitle, fontsize=10.5, color=TEXT_2)
    for i, chi in enumerate(contrasts):
        for j, method in enumerate(METHODS):
            ax = axes[i, j]
            grid = np.full((len(ps), len(ns)), np.nan)
            for r in rows:
                if r['method'] == method and r[contrast_key] == chi:
                    grid[ps.index(r['p']), ns.index(r['n'])] = r['iterations']
            ax.imshow(np.ma.masked_invalid(grid), origin='lower', cmap=ITERATION_CMAP, norm=norm, aspect='auto')
            for (a, b), value in np.ndenumerate(grid):
                if not np.isnan(value):
                    text = '>2000' if value >= 2000 else f'{int(value)}'
                    ax.text(b, a, text, ha='center', va='center', fontsize=8.5,
                            color='white' if norm(value) > 0.55 else TEXT_1)
            ax.set_xticks(range(len(ns)), [str(n) for n in ns])
            ax.set_yticks(range(len(ps)), [str(p) for p in ps])
            chi_label = '∞' if np.isinf(chi) else f'{chi:g}'
            style(ax, f'{NAMES[method]}, χ = {chi_label}', 'grid n (nodes per direction)',
                  'data resolution p' if j == 0 else None)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
