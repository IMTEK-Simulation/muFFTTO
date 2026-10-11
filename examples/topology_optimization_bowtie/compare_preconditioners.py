"""
Compare runs of the same problem with different CG preconditioners (--preconditioner): CG iterations of the state
and adjoint solves per objective evaluation, cumulative CG iterations, wall time, and the final designs.

    python compare_preconditioners.py data/<run 1> data/<run 2> ... [--name compare_preconditioners]
"""
import argparse
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402

here = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description='compare CG preconditioners')
parser.add_argument('runs', nargs='+')
parser.add_argument('--name', default='compare_preconditioners')
arguments = parser.parse_args()


def nu_xy(C):
    S = np.linalg.inv(C)
    return -S[0, 1] / S[0, 0]


runs = []
for run in arguments.runs:
    with netCDF4.Dataset(os.path.join(run, 'final.nc')) as d:
        a = {name: d.getncattr(name) for name in d.ncattrs()}
        rho = np.array(d.variables['rho_final'][:]).squeeze()
    runs.append(dict(name=str(a.get('preconditioner', 'green-jacobi')), rho=rho, a=a,
                     state=np.atleast_1d(a['log_cg_state']).astype(float),
                     adjoint=np.atleast_1d(a['log_cg_adjoint']).astype(float),
                     mismatch=np.atleast_1d(a['log_mismatch']).astype(float)))

fig = plt.figure(figsize=(16, 9))
grid = fig.add_gridspec(2, max(3, len(runs)))
colors = dict(zip([r['name'] for r in runs], ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']))
ax = fig.add_subplot(grid[0, 0])
for r in runs:
    nonzero = r['state'] > 0                       # repeated evaluations at the same design need no iteration
    ax.plot(np.flatnonzero(nonzero), r['state'][nonzero], '.', ms=3, color=colors[r['name']], label=f'{r["name"]} state')
    ax.plot(np.flatnonzero(nonzero), r['adjoint'][nonzero], 'x', ms=3, color=colors[r['name']], alpha=0.5,
            label=f'{r["name"]} adjoint')
ax.set_yscale('log')
ax.set_xlabel('objective evaluation')
ax.set_ylabel('CG iterations (3 load cases, warm start)')
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.3)
ax = fig.add_subplot(grid[0, 1])
for r in runs:
    ax.plot(np.cumsum(r['state'] + r['adjoint']), color=colors[r['name']], label=r['name'])
ax.set_xlabel('objective evaluation')
ax.set_ylabel('cumulative CG iterations (state + adjoint)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax = fig.add_subplot(grid[0, 2])
for r in runs:
    ax.semilogy(r['mismatch'], color=colors[r['name']], label=r['name'], lw=1)
ax.set_xlabel('objective evaluation')
ax.set_ylabel('Poisson / stress mismatch')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
for i, r in enumerate(runs):
    ax = fig.add_subplot(grid[1, i])
    ax.imshow(np.tile(r['rho'], (2, 2)).T, origin='lower', cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    C = np.reshape(r['a']['C_optimized_voigt'], (3, 3))
    used = r['state'] + r['adjoint']
    ax.set_title(f'{r["name"]}: nu_xy {nu_xy(C):.4f}\nL-BFGS it {int(r["a"]["nb_iterations"])}, evals {len(used)}, '
                 f'time {float(r["a"]["elapsed"]):.0f} s\nCG total {int(used.sum())}, per evaluation '
                 f'{used[used > 0].mean():.0f} (state {r["state"][r["state"] > 0].mean():.0f}, adjoint '
                 f'{r["adjoint"][r["adjoint"] > 0].mean():.0f})', fontsize=9)
a = runs[0]['a']
fig.suptitle(f'Preconditioner comparison, {int(np.atleast_1d(a["nb_of_pixels"])[0])}^2, SIMP p {float(a["simp_p"]):g}, '
             f'void {float(a["void_contrast"]):g}, CG |r| <= tol |b| (inexact, warm started)', fontsize=11)
fig.tight_layout()
out = os.path.join(here, 'data', f'{arguments.name}_{time.strftime("%Y%m%d-%H%M%S")}.png')
fig.savefig(out, dpi=110)
print(f'wrote {out}')
