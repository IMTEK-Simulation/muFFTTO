"""
Conductivity homogenization of the bubble array of the slide problems on the regular grid and on the grid adapted to
the bubbles (Zecevic, examples/phase_field_fracture_adapted_grid), with the preconditioners Green, Green-Jacobi and
dual (paper, Section "Discrete problem and preconditioners").

The adapted grid x = xi + d(xi) is treated as in the paper: the problem on the regular grid (same B, W) with the
pulled-back conductivity  A~_q = J_q F_q^-1 A_q F_q^-T,  F = I + grad_xi d,  J = det F,  and the right-hand side
b = -B^T W (J F^-1 A E). The effective conductivity is the mean physical flux  A (E + F^-T grad_xi u) over the
deformed cell (weights w_q J_q).

Material: solid (1) A = I, bubbles (0) A = kappa I. Reference A_0 = I (the solid) for Green and dual, I for
Green-Jacobi. Elements: bilinear quadrilaterals (Q1), as the slide runs. CG from zero, stopped on the energy-error
estimate relative to the initial error (1e-6, delay 10); the true relative energy error at the stop is recorded.
--spectrum (small grids): eigenvalues of P K on the mean-free vectors and g_min, g_max of A_0^-1 A~_q (Propositions
1 and 2).

    python conductivity_adapted_grid.py -n 32 -k -3 -p dual --grid adapted [--spectrum]
writes data/bubble_array_<grid>_N<n>_kappa1e<k>_<preconditioner>.npz
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from muFFTTO import domain
from muFFTTO import microstructure_library
from muFFTTO import solvers

HERE = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('-n', '--nb_pixel', type=int, default=32)
parser.add_argument('-k', '--kappa_exp', type=int, required=True, help='bubble conductivity kappa = 10^kappa_exp')
parser.add_argument('-p', '--preconditioner', choices=['Green', 'Green_Jacobi', 'dual'], required=True)
parser.add_argument('--grid', choices=['regular', 'adapted'], default='regular')
parser.add_argument('--cg_tol', type=float, default=1e-6)
parser.add_argument('--cg_delay', type=int, default=10)
parser.add_argument('--spectrum', action='store_true', help='dense spectrum of P K (serial, small grids)')
args = parser.parse_args()
nnn, kappa, preconditioner_type = args.nb_pixel, 10.0 ** args.kappa_exp, args.preconditioner
dim = 2
number_of_pixels = (nnn, nnn)

# two discretizations of the same grid and quadrature: scalar (the problem) and vector (the grid displacement)
cell_c = domain.PeriodicUnitCell(domain_size=[1, 1], problem_type='conductivity')
disc = domain.Discretization(cell=cell_c, nb_of_pixels_global=number_of_pixels, discretization_type='finite_element',
                             element_type='bilinear_rectangle')
disc_v = domain.Discretization(cell=domain.PeriodicUnitCell(domain_size=[1, 1], problem_type='elasticity'),
                               nb_of_pixels_global=number_of_pixels, discretization_type='finite_element',
                               element_type='bilinear_rectangle')
comm = disc.communicator
coords = np.asarray(disc.fft.coords)
local_index = tuple(np.rint(coords[k] * nnn).astype(int) % nnn for k in range(dim))

# ============================================================================
# grid and phase (copies of the grid files of the slide problems in grids/)
# ============================================================================
if args.grid == 'adapted':
    grid = np.load(os.path.join(HERE, 'grids', f'grid_bubble_array_N{nnn}_zecevic_min_jacobian0.01.npz'))
    d_g = grid['d_g'][(slice(None),) + local_index]                 # nodal grid displacement [2, x, y]
    phase_pixel = grid['phase'][local_index]                        # phase at the deformed element centroid
else:
    d_g = np.zeros((dim,) + coords.shape[1:])
    phase_pixel = microstructure_library.get_geometry(nb_voxels=disc.nb_of_pixels, microstructure_name='bubble_array',
                                                      coordinates=disc.fft.coords)
solid = phase_pixel > 0

grid_displacement = disc_v.get_unknown_size_field(name='grid_displacement')
grid_displacement.s[:, 0] = d_g
disc_v.fft.communicate_ghosts(grid_displacement)
grid_gradient = disc_v.get_gradient_size_field(name='grid_gradient')
disc_v.apply_gradient_operator_mugrid(u_inxyz=grid_displacement, grad_u_ijqxyz=grid_gradient)
F = grid_gradient.s + np.eye(dim).reshape(dim, dim, 1, 1, 1)                 # [i, j, q, x, y]
J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]                                     # [q, x, y]
if not comm.min(float(J.min())) > 0:
    raise ValueError('grid map not invertible')
F_inv = np.stack([np.stack([F[1, 1], -F[0, 1]]), np.stack([-F[1, 0], F[0, 0]])]) / J

# physical conductivity at the quadrature points and its pull-back
A_phys = np.zeros((dim, dim) + J.shape)
A_phys[0, 0] = A_phys[1, 1] = np.where(solid[None], 1.0, kappa)
M = J * np.einsum('ik...,kj...->ij...', F_inv, A_phys)                        # J F^-1 A   (flux map)
A_tilde = np.einsum('ik...,jk...->ij...', M, F_inv)                           # J F^-1 A F^-T
A_0 = np.eye(dim)

material = disc.get_material_data_size_field_mugrid(name='A_tilde')
material.s[...] = A_tilde
material_dual = disc.get_material_data_size_field_mugrid(name='A_dual')       # A_0 A~^-1 A_0
material_dual.s[...] = np.moveaxis(np.linalg.inv(np.moveaxis(A_tilde, (0, 1), (-2, -1))), (-2, -1), (0, 1))
g = np.linalg.eigvalsh(np.moveaxis(A_tilde, (0, 1), (-2, -1)))               # A_0 = I
g_min, g_max = comm.min(float(g.min())), comm.max(float(g.max()))


def K_fun(x, Ax):
    disc.apply_system_matrix_mugrid(material_data_field=material, input_field_inxyz=x, output_field_inxyz=Ax)
    disc.fft.communicate_ghosts(Ax)


green_0 = disc.get_preconditioner_Green_mugrid(reference_material_data_ijkl=A_0, name='green_A0')
jacobi = disc.get_preconditioner_Jacobi_mugrid(material_data_field_ijklqxyz=material, name='jacobi_A')
temp_1 = disc.get_unknown_size_field(name='temp_1')
temp_2 = disc.get_unknown_size_field(name='temp_2')


def apply_green(x, Px):
    disc.fft.communicate_ghosts(x)
    disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_0, input_nodal_field_fnxyz=x,
                                     output_nodal_field_fnxyz=Px)


def M_Green(x, Px):
    apply_green(x, Px)


def M_Green_Jacobi(x, Px):                    # reference I = A_0 here, so the same Green operator
    temp_1.s[...] = jacobi.s * x.s
    apply_green(temp_1, Px)
    Px.s[...] *= jacobi.s


def M_dual(x, Px):
    apply_green(x, temp_1)
    disc.apply_system_matrix_mugrid(material_data_field=material_dual, input_field_inxyz=temp_1,
                                    output_field_inxyz=temp_2)
    disc.fft.communicate_ghosts(temp_2)
    apply_green(temp_2, Px)


PRECS = {'Green': M_Green, 'Green_Jacobi': M_Green_Jacobi, 'dual': M_dual}

# ============================================================================
# solves: E = e_1, e_2
# ============================================================================
flux = disc.get_gradient_size_field(name='flux')
grad_u = disc.get_gradient_size_field(name='grad_u')
rhs = disc.get_unknown_size_field(name='rhs')
u = disc.get_unknown_size_field(name='u')
u_ref = disc.get_unknown_size_field(name='u_ref')
err = disc.get_unknown_size_field(name='err')
K_err = disc.get_unknown_size_field(name='K_err')
weights = np.asarray(disc.quadrature_weights).reshape((-1, 1, 1))


def energy(x):
    K_fun(x, K_err)
    return comm.sum(float(np.dot(x.s.ravel(), K_err.s.ravel())))


iterations, rel_errors, A_eff = [], [], np.zeros((dim, dim))
t0 = time.time()
for i in range(dim):
    E = np.zeros(dim)
    E[i] = 1.0
    flux.s[...] = np.einsum('ij...,j->i...', M, E)[None]                       # J F^-1 A E   [1, i, q, x, y]
    disc.fft.communicate_ghosts(flux)
    disc.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=flux, div_u_fnxyz=rhs, apply_weights=True)
    rhs.s[...] *= -1
    u_ref.s.fill(0)                                                             # reference: dual, tight tolerance
    solvers.conjugate_gradients_mugrid_experimental(comm=comm, fc=disc.field_collection, hessp=K_fun, b=rhs, x=u_ref,
                                                    P=M_dual, tol=1e-13, maxiter=20000, stop_crit_norm='rz')
    u.s.fill(0)
    norms = solvers.conjugate_gradients_mugrid_experimental(
        comm=comm, fc=disc.field_collection, hessp=K_fun, b=rhs, x=u, P=PRECS[preconditioner_type], tol=args.cg_tol,
        maxiter=20000, stop_crit_norm='energy_lower_estim', energy_rtol=True, d_min=args.cg_delay)[1]
    iterations.append(int(norms['stop_iteration']['energy_lower_estim']))
    err.s[...] = u_ref.s - u.s
    rel_errors.append(np.sqrt(energy(err) / energy(u_ref)))
    # effective conductivity: mean physical flux A (E + F^-T grad_xi u) over the deformed cell
    disc.fft.communicate_ghosts(u)
    disc.apply_gradient_operator_mugrid(u_inxyz=u, grad_u_ijqxyz=grad_u)
    grad_x = E.reshape(dim, 1, 1, 1) + np.einsum('ki...,k...->i...', F_inv, grad_u.s[0])
    q_phys = np.einsum('ij...,j...->i...', A_phys, grad_x)
    vol = comm.sum(float((weights * J).sum()))
    A_eff[:, i] = [comm.sum(float((weights * J * q_phys[a]).sum())) / vol for a in range(dim)]
elapsed = time.time() - t0

spectrum = None
if args.spectrum:
    import scipy.linalg as sl
    n = rhs.s.size
    e, o = disc.get_unknown_size_field(name='spec_e'), disc.get_unknown_size_field(name='spec_o')

    def dense(op):
        out = np.zeros((n, n))
        for col in range(n):
            e.s[...] = 0.0
            e.s.flat[col] = 1.0
            disc.fft.communicate_ghosts(e)
            op(e, o)
            out[:, col] = o.s.ravel()
        return out

    Q = sl.null_space(np.ones((1, n)))                                         # mean-free vectors
    Kd = Q.T @ dense(K_fun) @ Q
    spectrum = {p: np.sort(np.real(sl.eigvals(Q.T @ dense(PRECS[p]) @ Q @ Kd))) for p in PRECS}

if comm.rank == 0:
    os.makedirs(os.path.join(HERE, 'data'), exist_ok=True)
    out = os.path.join(HERE, 'data', f'bubble_array_{args.grid}_N{nnn}_kappa1e{args.kappa_exp:+d}_'
                                     f'{preconditioner_type}.npz')
    np.savez(out, grid=args.grid, N=nnn, kappa=kappa, preconditioner=preconditioner_type, iterations=iterations,
             rel_errors=rel_errors, A_eff=A_eff, g_min=g_min, g_max=g_max, min_J=comm.min(float(J.min())),
             max_J=comm.max(float(J.max())), elapsed=elapsed,
             **({f'spectrum_{p}': ev for p, ev in spectrum.items()} if spectrum else {}))
    print(f'{args.grid:8s} N{nnn} kappa 1e{args.kappa_exp:+d} {preconditioner_type:12s}: CG {iterations} '
          f'(rel. energy error {max(rel_errors):.1e}), A_eff_11 {A_eff[0, 0]:.6f}, A_eff_22 {A_eff[1, 1]:.6f}, '
          f'g in [{g_min:.3g}, {g_max:.3g}], J in [{comm.min(float(J.min())):.3f}, {comm.max(float(J.max())):.3f}], '
          f'{elapsed:.1f} s')
    if spectrum:
        for p, ev in spectrum.items():
            print(f'  spectrum {p:12s} [{ev[0]:.4g}, {ev[-1]:.4g}]  cond {ev[-1] / ev[0]:.3g}')
