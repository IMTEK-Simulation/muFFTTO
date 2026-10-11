"""
Initial geometry vs optimized design of a run of example_2D_finite_strain_TO_curve.py.

The initial geometry is the one the optimization started from: the --init geometry of the run, or, for a
continuation (--init-file), the --init geometry of the first run of the chain. Both designs are loaded along
every load case (path following over the strain levels of the run); the figure shows the designs (3 x 3 cells),
the deformed 2 x 2 cells at the largest strain and the stress-strain curves of both against the target.

    python compare_designs.py data/<run folder>

The optimized design is final.nc if the run has finished, else the last iterate of history.nc.
Output (in the run folder): compare_designs_<time>.png
"""
import os
import sys
import time

import numpy as np

folder = os.path.abspath(sys.argv[1])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from muFFTTO import io_utils, geometry


def scalar(attributes, key):
    return np.asarray(attributes[key]).ravel()[0]


# the run: final design if finished, else the last iterate
finished = os.path.exists(os.path.join(folder, 'final.nc'))
run_file = os.path.join(folder, 'final.nc' if finished else 'history.nc')
run = io_utils.read_file(run_file)
attributes = run.attributes

# the initial geometry: follow the continuation chain (init_file) back to the first run
root = attributes
while str(root.get('init_file', '')) not in ('', 'None'):
    previous = str(root['init_file'])
    if os.path.isdir(previous):
        previous = os.path.join(previous, 'final.nc')
    root = io_utils.read_file(previous).attributes
init_name = str(root['init'])

# the example's module-level setup with the parameters of the run
levels = np.asarray(attributes['levels']).ravel()
sys.argv = [sys.argv[0], '-n', str(int(scalar(attributes, 'nb_of_pixels'))),
            '--target'] + [str(float(scalar(attributes, k))) for k in ('target_E1', 'target_E2', 'target_contact_strain')] + [
            '--levels', str(levels.size), '--max-strain', str(float(np.max(np.abs(levels))))]
if int(scalar(attributes, 'third_medium_contact')):
    sys.argv.append('--contact')
import example_2D_finite_strain_TO_curve as to

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# the two designs (nodal densities)
rho_field = to.discretization.get_scalar_field(name='rho_final' if finished else 'rho_iterate')
io_utils.load_fields(run_file, [rho_field])
rho_optimized = rho_field.s[0, 0].copy()
coords = np.asarray(to.discretization.fft.coords) / np.asarray(to.domain_size)[:, None, None]
rho_initial = np.clip(geometry.get(init_name, coords).astype(float), 0.0, 1.0)
designs = [('initial (' + init_name + ')', rho_initial),
           ('optimized (' + ('final' if finished else f'iterate {run.nb_frames}') + ')', rho_optimized)]

taus = np.concatenate([[0.0], to.levels])


def load_path(rho):
    """Curves P_ll(tau) of every load case and the fluctuation at the largest strain."""
    to.set_design(rho.reshape(-1).copy())
    curves, last_states = [], []
    for l in range(to.nb_cases):
        to.discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=to.directions[l],
                                                          macro_gradient_field_ijqxyz=to.direction_field)
        u = to.discretization.get_unknown_size_field(name='u_compare')
        u.s[...] = 0.0
        stresses, tau_previous = [], 0.0
        for tau in taus:
            to.predict_fluctuation(u, to.directions[l], tau_previous, tau)
            tau_previous = tau
            ok = to.minimize(u, tau)[0]
            to.set_deformation_gradient(u, tau)
            to.material.get_stress(to.F_field, to.P_field)
            stresses.append(to.cell_average(to.P_field)[l, l] if ok else np.nan)
        curves.append(np.array(stresses))
        last_states.append(u.s[:, 0].copy())
    return curves, last_states


results = [load_path(rho) for _, rho in designs]

# figure: rows = designs, columns = design, deformed cells per load case, curve per load case
nx, ny = to.number_of_pixels
i, j = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), indexing='ij')
X = np.array([i * to.pixel_size[0], j * to.pixel_size[1]])
ncol = 1 + 2 * to.nb_cases
fig, axes = plt.subplots(2, ncol, figsize=(3.6 * ncol, 7.4))
for r, ((name, rho), (curves, states)) in enumerate(zip(designs, results)):
    ax = axes[r, 0]
    ax.imshow(np.tile(rho, (3, 3)).T, origin='lower', cmap='Greys', vmin=0, vmax=1)
    ax.set_title(f'{name}, 3 x 3 cells', fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for l in range(to.nb_cases):
        ax = axes[r, 1 + l]
        u = states[l][:, i % nx, j % ny]
        for a in range(2):
            for b in range(2):
                x = np.einsum('ab,b...->a...', np.eye(2) + taus[-1] * to.directions[l],
                              X + np.array([a * to.domain_size[0], b * to.domain_size[1]])[:, None, None]) + u
                ax.pcolormesh(x[0], x[1], rho, cmap='Greys', vmin=0, vmax=1, shading='flat')
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'{name.split(" ")[0]}: {to.load_cases[l]["name"]}, strain {taus[-1]:.2f}', fontsize=9)

# curves: one panel per load case (both rows share the comparison; bottom row: relative error)
sign = -1.0 if taus[-1] < 0 else 1.0                    # plot compression as positive
for l in range(to.nb_cases):
    ax = axes[0, 1 + to.nb_cases + l]
    fine = np.linspace(0.0, taus[-1], 100)
    ax.plot(sign * fine, [sign * to.target_stress(l, t)[l, l] for t in fine], 'k--', label='target')
    ax.plot(sign * to.levels, sign * to.target_stresses[l, :, l, l], 'ko', mfc='none', ms=4)
    for (name, _), (curves, _), style in zip(designs, results, ['C1-', 'C0-']):
        ax.plot(sign * taus, sign * curves[l], style, label=name.split(' ')[0])
    ax.set_title(f'{to.load_cases[l]["name"]}: P{l}{l}', fontsize=9)
    ax.set_xlabel('strain |tau|')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1, 1 + to.nb_cases + l]
    target = to.target_stresses[l, :, l, l]
    for (name, _), (curves, _), style in zip(designs, results, ['C1o-', 'C0o-']):
        ax.plot(sign * to.levels, sign * (curves[l][1:] - target) / np.abs(target).max(), style, ms=3,
                label=name.split(' ')[0])
    ax.axhline(0.0, color='k', lw=0.8)
    ax.set_title(f'{to.load_cases[l]["name"]}: (P - P_target) / max|P_target| (> 0: too stiff)', fontsize=9)
    ax.set_xlabel('strain |tau|')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
fig.suptitle(os.path.basename(folder), fontsize=9)
fig.tight_layout()
out = os.path.join(folder, 'compare_designs_' + time.strftime('%Y%m%d-%H%M%S') + '.png')
fig.savefig(out, dpi=100)
print('wrote', out)
