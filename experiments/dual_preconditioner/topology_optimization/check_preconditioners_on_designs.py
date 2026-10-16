"""
Preconditioners on fixed designs along a topology-optimization path (example_2D_elasticity_TO.py, history file):
every design is solved from zero for all load cases with Green (C_0), Green-Jacobi (2 mu I) and dual
G K(C_0 A^-1 C_0) G, CG stopped on the energy-error estimate (relative 1e-6, delay 10). Separates the effect of the
design from the warm starts and loose residual tolerance of the optimization loop.

    python check_preconditioners_on_designs.py [HISTORY.nc] [STRIDE] [SOFT_PHASE]
"""
import os
import sys

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
history = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    here, 'data', 'example_2D_elasticity_TO', 'dual_eta_0.015625_w_5.0_l_bfgs_history.nc')
stride = int(sys.argv[2]) if len(sys.argv) > 2 else 25
soft = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5      # stiffness of the void, C_void = soft C_0

h = netCDF4.Dataset(history)
designs = np.asarray(h['phase_field_iterate'][:])[:, 0]
iterates = list(range(0, len(designs), stride)) + ([len(designs) - 1] if (len(designs) - 1) % stride else [])
N = designs.shape[1:]

cell = domain.PeriodicUnitCell(domain_size=[1, 1], problem_type='elasticity')
disc = domain.Discretization(cell=cell, nb_of_pixels_global=N, discretization_type='finite_element',
                             element_type='linear_triangles')
dim = 2
K_0, G_0, p = 1.0, 0.5, 2                                  # as in example_2D_elasticity_TO.py
C_0 = material_models.get_elastic_material_tensor(dim=dim, K=K_0, mu=G_0, kind='linear')
green_C0 = disc.get_preconditioner_Green_mugrid(reference_material_data_ijkl=C_0, name='green_C0')
green_2muI = disc.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=2 * G_0 * np.einsum('ik,jl->ijkl', np.eye(dim), np.eye(dim)), name='green_2muI')
macro_gradients = [np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([[0.0, 0.0], [0.0, 1.0]]),
                   np.array([[0.0, 0.5], [0.5, 0.0]])]

phase = disc.get_scalar_field(name='phase')
phase_q = disc.get_quad_field_scalar(name='phase_q')
A = disc.get_material_data_size_field_mugrid(name='A')
A_dual = disc.get_material_data_size_field_mugrid(name='A_dual')
macro = disc.get_gradient_size_field(name='macro')
rhs = disc.get_unknown_size_field(name='rhs')
u = disc.get_unknown_size_field(name='u')
t1 = disc.get_unknown_size_field(name='t1')
t2 = disc.get_unknown_size_field(name='t2')


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


PRECS = {'Green (C_0)': green(green_C0), 'Green-Jacobi (2μI)': green_jacobi, 'dual (C_0)': dual}
results = {name: [] for name in PRECS}
for k in iterates:
    phase.s[0, 0] = designs[k]
    disc.apply_N_operator_mugrid(phase, phase_q)
    s = np.power(phase_q.s[0, 0], p) * (1 - soft) + soft          # A = s C_0
    A.s[...] = C_0[..., None, None, None] * s
    A_dual.s[...] = C_0[..., None, None, None] / s
    jacobi = disc.get_preconditioner_Jacobi_mugrid(material_data_field_ijklqxyz=A)
    for name, M in PRECS.items():
        total = 0
        for E in macro_gradients:
            disc.get_macro_gradient_field_mugrid(macro_gradient_ij=E, macro_gradient_field_ijqxyz=macro)
            rhs.s.fill(0)
            disc.get_rhs_mugrid(material_data_field_ijklqxyz=A, macro_gradient_field_ijqxyz=macro, rhs_inxyz=rhs)
            u.s.fill(0)
            norms = solvers.conjugate_gradients_mugrid_experimental(
                comm=disc.communicator, fc=disc.field_collection, hessp=K_fun, b=rhs, x=u, P=M, tol=1e-6,
                maxiter=20000, stop_crit_norm='energy_lower_estim', energy_rtol=True, d_min=10)[1]
            total += norms['stop_iteration']['energy_lower_estim']
        results[name].append(total)
    print(f'iterate {k:4d}: ' + '  '.join(f'{n} {r[-1]:5d}' for n, r in results.items()), flush=True)

fig, axes = plt.subplots(1, 1 + 4, figsize=(17, 3.8), gridspec_kw={'width_ratios': [2.2, 1, 1, 1, 1]})
colours = {'Green (C_0)': '#2a78d6', 'Green-Jacobi (2μI)': '#eb6834', 'dual (C_0)': '#1baf7a'}
for name, r in results.items():
    axes[0].semilogy(iterates, r, color=colours[name], marker='o', ms=4, lw=2, label=name)
axes[0].set_xlabel('optimization iterate (design)')
axes[0].set_ylabel('CG iterations, 3 load cases\n(cold start, energy stop 1e-6)')
axes[0].set_title(f'soft phase {soft:g} C_0', loc='left', fontsize=10)
axes[0].grid(alpha=0.3, which='both')
axes[0].legend(frameon=False, fontsize=9)
for ax, k in zip(axes[1:], [iterates[0], iterates[len(iterates) // 3], iterates[2 * len(iterates) // 3], iterates[-1]]):
    ax.pcolormesh(designs[k].T, cmap='Greys', vmin=0, vmax=1)        # exact pixels
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'iterate {k}', fontsize=10)
fig.tight_layout()
out = os.path.join(here, 'figures', f'preconditioners_on_designs_soft{soft:g}.png')
fig.savefig(out, dpi=150)
print(out)
