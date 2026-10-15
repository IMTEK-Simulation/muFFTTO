"""T11 PD on density-based topology optimization designs (elastic, C = rho_mat C0, three macroscopic loads e1, e2, e3
in Mandel notation, residual stop 1e-5, u0 = 0). Iterations and 2D FFTs summed over the three loads.

Preconditioners: G, GJ, PD (pointwise S) with rho_clip 1e-2, 1e-4 and 'none' (= 1e-15, the floor of Part A; a literal
no-clip would put 1e300 into S on exact voids).

Part A (also with the optimizer's own law rho^2 (1 - 1e-5) + 1e-5): designs along a muFFTTO phase-field TO run at 128^2 (example_2D_elasticity_TO.py, random U(0,1) start, L-BFGS,
eta = one pixel): iterations k = 0, 50, 200, 250 and the last one. The optimizer's material law is SIMP with p = 2, so
rho_mat = max(rho^2, 1e-15). Also the thresholded final design: rho_mat = 1 where rho >= 0.5, else 1e-4 / else 0.

Part B: filtered random noise on a design grid p in {32, 64}, SIMP rho_mat = rho^3 floored at 1e-4, and the thresholded
two-phase design with voids (1 / 0), analyzed at n in {128, 256} (n > p).

usage: python t11_topology_optimization.py A HISTORY.nc   or   python t11_topology_optimization.py B
"""
import sys

import netCDF4
import numpy as np

from common import C0_ELASTIC, RESULTS, SURFACE, TEXT_1, TEXT_2, plt, write_csv
from fem import material_field, rhs
from materials import expand, smooth
from pcg import pcg
from precond import make_green, make_green_jacobi, make_primal_dual, new_counter

d = 2
LOADS = np.eye(3)
METHODS = [('G', None), ('GJ', None), ('PD', 1e-2), ('PD', 1e-4), ('PD', 1e-15)]


def label(method, clip):
    return method if clip is None else f"PD clip {'none' if clip == 1e-15 else f'{clip:g}'}"


def solve_design(rho):
    n = rho.shape[0]
    C = material_field(rho, C0_ELASTIC)
    out = {}
    for method, clip in METHODS:
        iterations, ffts, converged = 0, 0, True
        for E in LOADS:
            f = rhs(C, E, d, n)
            counter = new_counter()
            if method == 'G':
                M = make_green(C0_ELASTIC, d, n, counter)
            elif method == 'GJ':
                M = make_green_jacobi(C, C0_ELASTIC, d, n, counter)
            else:
                M = make_primal_dual(rho, C0_ELASTIC, C0_ELASTIC, d, n, counter, rho_clip=clip)
            result = pcg(C, f, M, counter, tol=1e-5, maxiter=5000, track_energy=False)
            iterations += result['iterations']
            ffts += result['counter']['fft']
            converged &= result['converged']
        out[label(method, clip)] = (iterations, ffts, converged)
    return out


def record(rows, part, case, n, rho, out):
    for name, (iterations, ffts, converged) in out.items():
        rows.append(dict(part=part, case=case, n=n, method=name, iterations=iterations, ffts=ffts,
                         converged=converged, rho_min=f'{rho.min():.1e}'))
    best = min(out, key=lambda k: out[k][1])
    print(f"{part} {case:38s} n={n:3d} | " + ' | '.join(f'{k} {v[0]}/{v[1]}' for k, v in out.items())
          + f' | lowest FFT: {best}', flush=True)


rows, designs = [], {}
part = sys.argv[1]
soft_only = len(sys.argv) > 3 and sys.argv[3] == '_soft'      # Part A: only the optimizer's own material law
# ---- Part A: optimization path -------------------------------------------------------------------------------------
if part == 'A':
    history = np.asarray(netCDF4.Dataset(sys.argv[2])['phase_field_iterate'][:])[:, 0]     # [k, n, n]
    last = history.shape[0] - 1
    for k in sorted({0, 50, 200, 250, last}):
        if k > last:
            continue
        rho_design = np.clip(history[k], 0.0, 1.0)
        designs[f'A: k = {k}'] = rho_design
        if not soft_only:
            rho = np.maximum(rho_design ** 2, 1e-15)
            record(rows, 'A', f'path k = {k}', rho.shape[0], rho, solve_design(rho))
        # the optimizer's own law: SIMP p = 2 with soft phase 1e-5 (soft_phase_exponent = 5)
        rho = rho_design ** 2 * (1 - 1e-5) + 1e-5
        record(rows, 'A', f'path k = {k}, soft phase 1e-5', rho.shape[0], rho, solve_design(rho))
    final = np.clip(history[last], 0.0, 1.0)
    for void in (() if soft_only else (1e-4, 0.0)):
        rho = np.where(final >= 0.5, 1.0, void)
        record(rows, 'A', f'final thresholded, void {void:g}', rho.shape[0], rho, solve_design(rho))

# ---- Part B: coarse design grid ------------------------------------------------------------------------------------
for p in ((32, 64) if part == 'B' else ()):
    noise = smooth(np.random.default_rng(p).normal(size=(p, p)), 4)
    rho_design = (noise - noise.min()) / (noise.max() - noise.min())
    designs[f'B: p = {p}'] = rho_design
    for n in (128, 256):
        rho = expand(np.maximum(rho_design ** 3, 1e-4), n)
        record(rows, 'B', f'p = {p} SIMP rho^3 floor 1e-4', n, rho, solve_design(rho))
        rho = expand(np.where(rho_design >= 0.5, 1.0, 0.0), n)
        record(rows, 'B', f'p = {p} thresholded 1 / 0', n, rho, solve_design(rho))

write_csv(f't11_topology_optimization_{part}{sys.argv[3] if len(sys.argv) > 3 else ""}.csv', rows)

# designs on their exact pixels
fig, axes = plt.subplots(1, len(designs), figsize=(2.6 * len(designs), 3.2), facecolor=SURFACE)
fig.subplots_adjust(left=0.02, right=0.98, top=0.78, bottom=0.03, wspace=0.08)
fig.text(0.02, 0.93, 'T11 designs (design density rho, white 0 ... black 1)', fontsize=12, fontweight='bold',
         color=TEXT_1)
for ax, (name, rho) in zip(np.atleast_1d(axes), designs.items()):
    m = rho.shape[0]
    ax.pcolormesh(np.arange(m + 1), np.arange(m + 1), rho.T, shading='flat', cmap='Greys', vmin=0, vmax=1)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(name, fontsize=9.5, color=TEXT_2)
fig.savefig(f'{RESULTS}/t11_designs_{part}.png', dpi=200, facecolor=SURFACE)
