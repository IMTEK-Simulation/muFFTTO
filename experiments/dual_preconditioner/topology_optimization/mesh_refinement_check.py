"""
Mesh refinement of the preconditioner comparison on a fixed topology-optimization design: the final design of
example_2D_elasticity_TO.py (64^2) on 32^2 (2 x 2 average), 64^2, 128^2, 256^2 (pixel replication), so the geometry is
the same in physical units. Green (C_0), Green-Jacobi (2 mu I), dual G K(C_0 A^-1 C_0) G; cold start, CG stopped on
the energy-error estimate (relative 1e-6, delay 10), 3 load cases; soft phase 1e-3 and 1e-5.

    python mesh_refinement_check.py [--history HISTORY.nc] [--grids 32 64 128 256] [--softs 1e-3 1e-5]
    python mesh_refinement_check.py --plot          (only the figure, from all results/mesh_refinement_*.json)
Each run writes figures/mesh_refinement_<grids>_<softs>.json; the figure merges all of them.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import matplotlib
import netCDF4
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from muFFTTO import domain, material_models, solvers

here = os.path.dirname(os.path.realpath(__file__))
# the local copy of example_2D_elasticity_TO.py and its data (this folder)
TO = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--history', default=os.path.join(TO, 'data', 'example_2D_elasticity_TO',
                                                       'dual_eta_0.015625_w_5.0_l_bfgs_history.nc'))
parser.add_argument('--grids', type=int, nargs='+', default=[32, 64, 128, 256])
parser.add_argument('--softs', type=float, nargs='+', default=[1e-3, 1e-5])
parser.add_argument('--plot', action='store_true', help='only merge the stored results into the figure')
args = parser.parse_args()
design_64 = np.asarray(netCDF4.Dataset(args.history)['phase_field_iterate'][-1, 0])
GRIDS, SOFTS = args.grids, args.softs
out_folder = os.path.join(here, 'figures')
K_0, G_0, p, dim = 1.0, 0.5, 2, 2                          # as in example_2D_elasticity_TO.py
C_0 = material_models.get_elastic_material_tensor(dim=dim, K=K_0, mu=G_0, kind='linear')
macro_gradients = [np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([[0.0, 0.0], [0.0, 1.0]]),
                   np.array([[0.0, 0.5], [0.5, 0.0]])]


def resample(design, N):
    n0 = design.shape[0]
    if N >= n0:
        return np.kron(design, np.ones((N // n0, N // n0)))
    f = n0 // N
    return design.reshape(N, f, N, f).mean(axis=(1, 3))


results = {}                                                 # (soft, preconditioner) -> {N: (iterations, time)}
for N in ([] if args.plot else GRIDS):
    cell = domain.PeriodicUnitCell(domain_size=[1, 1], problem_type='elasticity')
    disc = domain.Discretization(cell=cell, nb_of_pixels_global=(N, N), discretization_type='finite_element',
                                 element_type='linear_triangles')
    green_C0 = disc.get_preconditioner_Green_mugrid(reference_material_data_ijkl=C_0, name='green_C0')
    green_2muI = disc.get_preconditioner_Green_mugrid(
        reference_material_data_ijkl=2 * G_0 * np.einsum('ik,jl->ijkl', np.eye(dim), np.eye(dim)), name='green_2muI')
    phase = disc.get_scalar_field(name='phase')
    phase_q = disc.get_quad_field_scalar(name='phase_q')
    A = disc.get_material_data_size_field_mugrid(name='A')
    A_dual = disc.get_material_data_size_field_mugrid(name='A_dual')
    macro = disc.get_gradient_size_field(name='macro')
    rhs = disc.get_unknown_size_field(name='rhs')
    u = disc.get_unknown_size_field(name='u')
    t1 = disc.get_unknown_size_field(name='t1')
    t2 = disc.get_unknown_size_field(name='t2')
    phase.s[0, 0] = resample(design_64, N)
    disc.apply_N_operator_mugrid(phase, phase_q)

    def K_fun(x, Ax):
        disc.apply_system_matrix_mugrid(material_data_field=A, input_field_inxyz=x, output_field_inxyz=Ax,
                                        formulation='small_strain')
        disc.fft.communicate_ghosts(Ax)

    def green(G):
        def apply(x, Px):
            disc.fft.communicate_ghosts(x)
            disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=G, input_nodal_field_fnxyz=x,
                                             output_nodal_field_fnxyz=Px)
        return apply

    def green_jacobi(x, Px):
        t1.s[...] = jacobi.s * x.s
        green(green_2muI)(t1, Px)
        Px.s[...] *= jacobi.s

    def dual(x, Px):
        green(green_C0)(x, t1)
        disc.apply_system_matrix_mugrid(material_data_field=A_dual, input_field_inxyz=t1, output_field_inxyz=t2,
                                        formulation='small_strain')
        disc.fft.communicate_ghosts(t2)
        green(green_C0)(t2, Px)

    precs = {'Green (C_0)': green(green_C0), 'Green-Jacobi (2μI)': green_jacobi, 'dual (C_0)': dual}
    for soft in SOFTS:
        s = np.power(phase_q.s[0, 0], p) * (1 - soft) + soft          # A = s C_0
        A.s[...] = C_0[..., None, None, None] * s
        A_dual.s[...] = C_0[..., None, None, None] / s
        jacobi = disc.get_preconditioner_Jacobi_mugrid(material_data_field_ijklqxyz=A)
        for name, M in precs.items():
            total, t0 = 0, time.time()
            for E in macro_gradients:
                disc.get_macro_gradient_field_mugrid(macro_gradient_ij=E, macro_gradient_field_ijqxyz=macro)
                rhs.s.fill(0)
                disc.get_rhs_mugrid(material_data_field_ijklqxyz=A, macro_gradient_field_ijqxyz=macro, rhs_inxyz=rhs)
                u.s.fill(0)
                norms = solvers.conjugate_gradients_mugrid_experimental(
                    comm=disc.communicator, fc=disc.field_collection, hessp=K_fun, b=rhs, x=u, P=M, tol=1e-6,
                    maxiter=50000, stop_crit_norm='energy_lower_estim', energy_rtol=True, d_min=10)[1]
                total += norms['stop_iteration']['energy_lower_estim']
            results.setdefault(f'{soft:g}|{name}', {})[N] = (total, time.time() - t0)
            print(f'N {N:4d}  soft {soft:g}  {name:20s} CG {total:6d}  {time.time() - t0:7.1f} s', flush=True)
    json.dump(results, open(os.path.join(out_folder, 'mesh_refinement_' + '-'.join(map(str, GRIDS)) + '_'
                                         + '-'.join(f'{s:g}' for s in SOFTS) + '.json'), 'w'), indent=1)

# merge all stored runs (the earlier 32^2 ... 256^2 file is mesh_refinement.json)
import glob
results = {}
for path in sorted(glob.glob(os.path.join(out_folder, 'mesh_refinement*.json'))):
    for key, by_n in json.load(open(path)).items():
        results.setdefault(key, {}).update({int(n): v for n, v in by_n.items()})
SOFTS = sorted({float(k.split('|')[0]) for k in results}, reverse=True)
GRIDS = sorted({n for by_n in results.values() for n in by_n})

colours = {'Green (C_0)': '#2a78d6', 'Green-Jacobi (2μI)': '#eb6834', 'dual (C_0)': '#1baf7a'}
fig, axes = plt.subplots(1, 2 * len(SOFTS) + 1, figsize=(19, 4),
                         gridspec_kw={'width_ratios': [1.6] * (2 * len(SOFTS)) + [1]})
for j, soft in enumerate(SOFTS):
    for k, (what, ylabel) in enumerate([(0, 'CG iterations, 3 load cases'), (1, 'wall time [s]')]):
        ax = axes[2 * j + k]
        ends = []
        for name, c in colours.items():
            r = results[f'{soft:g}|{name}']
            grids = [N for N in GRIDS if N in r]
            ax.loglog(grids, [r[N][what] for N in grids], base=2, color=c, marker='o', lw=2, label=name)
            ends.append((r[grids[-1]][what], grids[-1]))
        ends.sort()
        for i, (v, N) in enumerate(ends):                 # end labels, pushed apart when the values are close
            dy = 0
            if i > 0 and v / ends[i - 1][0] < 1.3:
                dy = 5
            if i < len(ends) - 1 and ends[i + 1][0] / v < 1.3:
                dy = -5
            ax.annotate(f'{v:.0f}', (N, v), textcoords='offset points', xytext=(5, dy), fontsize=8, va='center')
        ax.set_xticks(GRIDS, [f'{N}²' for N in GRIDS])
        ax.set_xlim(GRIDS[0] / 1.3, GRIDS[-1] * 1.6)
        ax.set_title(f'soft phase {soft:g} C_0: {ylabel.split(",")[0]}', loc='left', fontsize=10)
        ax.set_xlabel('grid')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, which='both')
axes[0].legend(frameon=False, fontsize=8)
axes[-1].pcolormesh(design_64.T, cmap='Greys', vmin=0, vmax=1)          # exact pixels
axes[-1].set_aspect('equal')
axes[-1].set_xticks([])
axes[-1].set_yticks([])
axes[-1].set_title('design (64²)', fontsize=10)
fig.tight_layout()
out = os.path.join(here, 'figures', 'mesh_refinement.png')
fig.savefig(out, dpi=150)
print(out)
