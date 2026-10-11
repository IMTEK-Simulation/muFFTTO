"""
What the CG solver sees: for one finished run, the design variable rho is pushed through the filter and the
projection at every beta of the continuation, rho_bar = H_beta(F rho), interpolated to the quadrature points
(rho_q, linear triangles: 2 per pixel) and turned into the SIMP stiffness factor rho_q^p, i.e. C(rho_q) / C_solid
(void 1e-5). For each beta: the stiffness factor field, a profile across a member, nu_xy of the design seen at that
beta, and the number of CG iterations of the state solves (3 load cases) and of the adjoint solves, cold started
(zero initial guess) with the tolerance of the optimization.

    python what_cg_sees.py data/<run> [--y 32]      # writes data/<run>/what_cg_sees_<time>.png
"""
import argparse
import importlib.util
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402

here = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(here, '../..')))
from muFFTTO import solvers  # noqa: E402

parser = argparse.ArgumentParser(description='stiffness field seen by CG at every beta, CG iteration counts')
parser.add_argument('run')
parser.add_argument('--y', type=int, default=None, help='row of the profile (default: middle)')
arguments = parser.parse_args()

with netCDF4.Dataset(os.path.join(arguments.run, 'final.nc')) as d:
    attributes = {name: d.getncattr(name) for name in d.ncattrs()}
    rho_design = np.array(d.variables['rho_design'][:]).squeeze()
n = rho_design.shape[0]

# example module configured like the run
options = ['-n', str(n), '--simp-p', f'{float(attributes["simp_p"]):g}', '--weight', f'{float(attributes["weight"]):g}',
           f'--nu={float(attributes["nu_target"]):g}', '--thickness', f'{float(attributes["thickness"]):g}',
           '--regularization', 'filter', '--filter-radius', f'{float(attributes["filter_radius_pixels"]):g}']
if str(attributes.get('objective', 'stiffness')) == 'poisson':
    options += ['--objective', 'poisson']
sys.argv = ['example', *options]
spec = importlib.util.spec_from_file_location('example', os.path.join(here, 'example_2D_elasticity_TO_bowtie.py'))
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)
example.root_print = lambda *a, **k: None
C_t = np.reshape(attributes['C_target_voigt'], (3, 3))
for k in range(example.nb_load_cases):
    example.target_stresses[k] = example.tensor_stress(example.strain_amplitude * C_t[:, k])

# count CG iterations (state and adjoint call the same solver)
counter = {'n': 0}
original_cg = solvers.conjugate_gradients_mugrid


def counting_cg(*args, callback=None, **kwargs):
    def count(*a):
        counter['n'] += 1
        if callback is not None:
            callback(*a)
    return original_cg(*args, callback=count, **kwargs)


solvers.conjugate_gradients_mugrid = counting_cg
example.solvers.conjugate_gradients_mugrid = counting_cg

betas = [float(b) for b in np.atleast_1d(attributes['beta_schedule'])]
y = n // 2 if arguments.y is None else arguments.y
p = float(attributes['simp_p'])
rows = []
for beta in betas:
    example.density.beta = beta
    rho_bar = example.density.forward(rho_design.reshape(-1))
    rho_tilde = example.density._rho_tilde.reshape(n, n)
    example.set_design(rho_bar)
    stiffness = np.power(example.rho_quad.s[0, 0], p)            # [quad, x, y]: C(rho_q) / C_solid (minus void)
    for k in range(example.nb_load_cases):                       # cold start
        example.displacements[k].s[...] = 0.0
        example.adjoints[k].s[...] = 0.0
    counter['n'] = 0
    stresses = [example.solve_load_case(k, example.cg_tol) for k in range(example.nb_load_cases)]
    state_iterations = counter['n']
    counter['n'] = 0
    example.mismatch_and_gradient(rho_bar)                       # states converged (warm) + 3 adjoints (cold)
    adjoint_iterations = counter['n']
    C = np.array([example.voigt_stress(s) for s in stresses]).T / example.strain_amplitude
    S = np.linalg.inv(C)
    grey = np.mean((stiffness > 1e-3) & (stiffness < 0.999))
    rows.append(dict(beta=beta, rho_tilde=rho_tilde, rho_bar=rho_bar.reshape(n, n), stiffness=stiffness,
                     state=state_iterations, adjoint=adjoint_iterations, nu=-S[0, 1] / S[0, 0], grey=grey))
    print(f'beta {beta:4g}: CG state {state_iterations} (3 load cases), adjoint {adjoint_iterations}, '
          f'nu_xy {-S[0, 1] / S[0, 0]:.4f}, quadrature points with grey stiffness {100 * grey:.1f} %', flush=True)

# figure: top row stiffness factor (mean of the two triangles of a pixel) per beta; bottom: profiles and CG counts
fig = plt.figure(figsize=(3.2 * len(betas), 8.4))
grid = fig.add_gridspec(2, len(betas), height_ratios=[1.15, 1])
for i, r in enumerate(rows):
    ax = fig.add_subplot(grid[0, i])
    ax.imshow(r['stiffness'].mean(axis=0).T, origin='lower', cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
    ax.axhline(y, color='tab:red', lw=0.6)
    ax.set_title(f'beta {r["beta"]:g}: stiffness seen by CG\nnu_xy {r["nu"]:.3f}, grey quad. points '
                 f'{100 * r["grey"]:.0f} %', fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
ax = fig.add_subplot(grid[1, :len(betas) // 2])
x = np.arange(n)
ax.plot(x, rows[0]['rho_tilde'][:, y], color='black', lw=2, label='filtered F rho (same for all beta)')
for r, color in zip(rows, plt.cm.viridis(np.linspace(0, 0.9, len(rows)))):
    ax.step(x, r['stiffness'].mean(axis=0)[:, y], where='mid', color=color, lw=1, label=f'CG stiffness, beta {r["beta"]:g}')
ax.set_xlabel(f'x [pixels] along the red line (y = {y})')
ax.set_ylabel('C(rho_q) / C_solid')
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
ax = fig.add_subplot(grid[1, len(betas) // 2:])
index = np.arange(len(rows))
ax.bar(index - 0.2, [r['state'] for r in rows], 0.4, label='state (3 load cases)')
ax.bar(index + 0.2, [r['adjoint'] for r in rows], 0.4, label='adjoint (3 load cases)')
ax.set_xticks(index, [f'beta {r["beta"]:g}' for r in rows])
ax.set_ylabel('CG iterations (cold start)')
ax.set_title(f'tolerance |r| <= {example.cg_tol:g} |b|, Green-Jacobi preconditioner', fontsize=9)
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis='y')
fig.suptitle(f'{os.path.basename(arguments.run)}\nfinal design variable rho viewed through filter + projection at each '
             f'beta (stiffness factor rho_q^{p:g}, mean of the 2 triangles per pixel)', fontsize=10)
fig.tight_layout()
out = os.path.join(arguments.run, f'what_cg_sees_{time.strftime("%Y%m%d-%H%M%S")}.png')
fig.savefig(out, dpi=110)
print(f'wrote {out}')
