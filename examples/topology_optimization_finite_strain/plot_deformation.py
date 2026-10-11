"""
Deformation mechanism of an optimized design of example_2D_finite_strain_TO_curve.py.

Loads the final design of a run, loads every load case along F = I + tau H from tau = 0 (path following
with the trust-region solver of the example) and plots 2 x 2 periodic cells, deformed, at a few strain
levels: one row per load case. Also writes a movie (--movie) of the whole loading path.

    python plot_deformation.py data/<run folder> [--movie]

Output (in the run folder): deformation_mechanism.png [, deformation_mechanism.mp4]
"""
import os
import sys

import numpy as np

folder = os.path.abspath(sys.argv[1])
make_movie = '--movie' in sys.argv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from muFFTTO import io_utils

# the example's module-level setup, on the grid of the run
run = io_utils.read_file(os.path.join(folder, 'final.nc'))
nb_pixels = int(np.asarray(run.attributes['nb_of_pixels']).ravel()[0])
contact = int(np.asarray(run.attributes.get('third_medium_contact', 0)).ravel()[0])
sys.argv = [sys.argv[0], '-n', str(nb_pixels)] + (['--contact'] if contact else [])
# the target curve and strain levels of the run (if stored; older runs: defaults of the example)
if 'target_E1' in run.attributes:
    sys.argv += ['--target'] + [str(float(np.asarray(run.attributes[k]).ravel()[0]))
                                for k in ('target_E1', 'target_E2', 'target_contact_strain')]
if 'levels' in run.attributes:
    run_levels = np.asarray(run.attributes['levels']).ravel()
    sys.argv += ['--levels', str(run_levels.size), '--max-strain', str(float(np.max(np.abs(run_levels))))]
import example_2D_finite_strain_TO_curve as to

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# the design
rho = to.discretization.get_scalar_field(name='rho_final')
io_utils.load_fields(os.path.join(folder, 'final.nc'), [rho])
to.set_design(rho.s.reshape(-1).copy())

# load path: small steps from 0 to 1.2 x the largest optimized level; keep u at every step
tau_end = 1.2 * to.levels[-1]
taus = np.linspace(0.0, tau_end, 37)
snapshots = [0.25, 0.5, 0.75, 1.0]                      # columns: fractions of tau_end (plus tau = 0)
paths = []                                              # [load case][step] -> (tau, u, P_h)
for l in range(to.nb_cases):
    to.discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=to.directions[l],
                                                      macro_gradient_field_ijqxyz=to.direction_field)
    u = to.discretization.get_unknown_size_field(name='u_path_plot')
    u.s[...] = 0.0
    path = []
    for n, tau in enumerate(taus):
        to.predict_fluctuation(u, to.directions[l], taus[n - 1] if n else 0.0, tau)
        ok = to.minimize(u, tau)[0]
        if not ok:
            print(f'load case {l}: no equilibrium at tau = {tau:.3f}, path stops')
            break
        to.set_deformation_gradient(u, tau)
        to.material.get_stress(to.F_field, to.P_field)
        path.append((tau, u.s[:, 0].copy(), to.cell_average(to.P_field)))
    paths.append(path)

nx, ny = to.number_of_pixels
h = to.pixel_size
i, j = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), indexing='ij')
X = np.array([i * h[0], j * h[1]])                       # pixel corners of one cell
rho_pixels = rho.s[0, 0]


def draw_cells(ax, l, tau, u_nodal, n_cells=2):
    """n_cells x n_cells deformed cells: x = (I + tau H)(X + shift) + u(X) (u periodic)."""
    F_macro = np.eye(2) + tau * to.directions[l]
    u = u_nodal[:, i % nx, j % ny]
    for a in range(n_cells):
        for b in range(n_cells):
            shift = np.array([a * to.domain_size[0], b * to.domain_size[1]])[:, None, None]
            x = np.einsum('ab,b...->a...', F_macro, X + shift) + u
            ax.pcolormesh(x[0], x[1], rho_pixels, cmap='Greys', vmin=0, vmax=1, shading='flat')
    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], float).T * np.asarray(to.domain_size)[:, None]
    for a in range(n_cells):
        for b in range(n_cells):
            c = F_macro @ (corners + np.array([a, b])[:, None] * np.asarray(to.domain_size)[:, None])
            ax.plot(c[0], c[1], color='C3', lw=0.6)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])


# figure: rows = load cases, columns = tau = 0 and the snapshots, last column = stress-strain curve
ncol = len(snapshots) + 2
fig, axes = plt.subplots(to.nb_cases, ncol, figsize=(3.2 * ncol, 3.4 * to.nb_cases), squeeze=False)
for l, path in enumerate(paths):
    steps = [0] + [int(np.argmin(np.abs(taus[:len(path)] - s * tau_end))) for s in snapshots]
    for c, n in enumerate(steps):
        tau, u_nodal, P = path[n]
        draw_cells(axes[l, c], l, tau, u_nodal)
        axes[l, c].set_title(f'{to.load_cases[l]["name"]}, tau = {tau:.3f}', fontsize=8)
    ax = axes[l, -1]
    t = np.array([p[0] for p in path])
    P = np.array([p[2] for p in path])
    for (a, b) in to.load_cases[l]['match']:
        ax.plot(t, P[:, a, b], '-', color='C0', label=f'P{a}{b}')
        ax.plot(t, [to.target_stress(l, s)[a, b] for s in t], '--', color='C0', label='target')
        ax.plot(to.levels, to.target_stresses[l, :, a, b], 'o', color='C0', mfc='none')
    ax.plot(t, P[:, 0, 1], ':', color='C2', label='P01 (free)')
    for n in steps:
        ax.axvline(t[n], color='0.85', lw=0.8, zorder=0)
    ax.set_xlabel('tau')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
fig.suptitle(f'{os.path.basename(folder)}: 2 x 2 cells (red: cell boundaries, F_macro applied), black = solid',
             fontsize=9)
fig.tight_layout()
out = os.path.join(folder, 'deformation_mechanism.png')
if os.path.exists(out):                                 # never overwrite an earlier figure
    import time
    out = os.path.join(folder, 'deformation_mechanism_' + time.strftime('%Y%m%d-%H%M%S') + '.png')
fig.savefig(out, dpi=110)
print('wrote', out)

if make_movie:
    from matplotlib import animation
    fig, axes = plt.subplots(1, to.nb_cases, figsize=(4.5 * to.nb_cases, 4.8), squeeze=False)

    def frame(n):
        for l, path in enumerate(paths):
            ax = axes[0, l]
            ax.clear()
            tau, u_nodal, _ = path[min(n, len(path) - 1)]
            draw_cells(ax, l, tau, u_nodal)
            ax.set_xlim(-0.1, 2.0 * (1 + abs(tau_end)) + 0.1)
            ax.set_ylim(-0.1, 2.0 * (1 + abs(tau_end)) + 0.1)
            ax.set_title(f'{to.load_cases[l]["name"]}, tau = {tau:.3f}', fontsize=9)

    movie = animation.FuncAnimation(fig, frame, frames=max(len(p) for p in paths), interval=150)
    out = os.path.join(folder, 'deformation_mechanism.mp4')
    movie.save(out, writer=animation.FFMpegWriter(fps=6), dpi=100)
    print('wrote', out)
