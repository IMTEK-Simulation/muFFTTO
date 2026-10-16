"""
Test problems P1-P3 of Paper 1 (Ladecky, Papez, Tichy, Pastewka, *Guaranteed Bounds on
the Iterative Error of CG-Accelerated FFT-Based Homogenization*, Section 6) in muFFTTO,
solved with three preconditioners:

    green         G = (B^T W B)^-1                    reference material I
    green_jacobi  D^-1/2 G D^-1/2,  D = diag(K)
    dual          G K(A^-1) G,  K(A^-1) = B^T W A^-1 B  (system matrix with the inverse material data)

Spectral bounds of the preconditioned operator, used by the Gauss-Radau upper bound:
    green         [min eig A, max eig A]
    dual          [1, max eig A / min eig A]   (lambda_min >= 1 by the complementary energy:
                  f.K^-1 f = min_{B^T W q = f} q.W A^-1 q <= f.P f with q = B G f)
    green_jacobi  no guaranteed bound -> no Gauss-Radau bound

Every case runs three solves:
    reference   dual preconditioner, until (r, z) < 1e-26 (energy error <= (r, z) / 1)
    monitored   the paper's run: until the residual, the GR upper bound and the PT upper
                estimate are all below 1e-10, with the true energy error at every iteration
    production  the stopping test of all muFFTTO experiments: PT lower estimate of the energy
                error relative to the initial one, tol 1e-6, minimal delay 10

    python run_experiment.py --problem P2 --kappa-exp -3 --preconditioner dual
    python run_experiment.py --problem P1 --kappa-exp 3 --grid 1024 --preconditioner green
"""
import sys
import os
import argparse
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
from mpi4py import MPI

from muFFTTO import domain
from muFFTTO import solvers

HERE = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--problem', choices=['P1', 'P2', 'P3'], required=True)
parser.add_argument('--kappa-exp', type=int, required=True, help='total phase contrast kappa = 10^kappa_exp')
parser.add_argument('--preconditioner', choices=['green', 'green_jacobi', 'dual'], required=True)
parser.add_argument('--grid', type=int, default=256, help='grid N x N, a multiple of 256 for P2 and P3')
parser.add_argument('--tau', type=float, default=0.25, help='tolerance of the adaptive delay')
parser.add_argument('--tol-monitored', type=float, default=1e-10, help='squared, as in the paper')
parser.add_argument('--tol-production', type=float, default=1e-6, help='relative energy tolerance')
parser.add_argument('--d-min', type=int, default=10, help='minimal delay of the production run')
parser.add_argument('--maxiter', type=int, default=5000)
args = parser.parse_args()

kappa = 10.0 ** args.kappa_exp
N = args.grid
comm_world = MPI.COMM_WORLD

# ============================================================================
# DISCRETIZATION
# ============================================================================
cell = domain.PeriodicUnitCell(domain_size=[1, 1], problem_type='conductivity')
discretization = domain.Discretization(cell=cell,
                                       nb_of_pixels_global=(N, N),
                                       discretization_type='finite_element',
                                       element_type='linear_triangles')
loc = discretization.subdomain_locations_no_buffers
nb = discretization.nb_of_pixels
local = (slice(loc[0], loc[0] + nb[0]), slice(loc[1], loc[1] + nb[1]))

# ============================================================================
# MATERIAL  (the settings of the paper-1 notebooks)
# ============================================================================
I = np.eye(2)
A_global = np.zeros((2, 2, N, N))                     # conductivity per pixel, both quadrature points
if args.problem == 'P1':
    # square inclusion [0.25, 0.75)^2, A_mat = I, A_inc = kappa I; exact A_eff of Obnosov
    x_c = (np.arange(N) + 0.5) / N
    inside = np.abs(x_c - 0.5) < 0.25
    phase = 1 - np.outer(inside, inside)              # 1 matrix, 0 inclusion
    A_global = np.einsum('ij,xy->ijxy', I, phase + kappa * (1 - phase))
    A_eff_ref = np.sqrt((1 + 3 * kappa) / (3 + kappa))
elif args.problem == 'P2':
    # 58 circles, two rotated anisotropic phases kappa diag(1, 2) and kappa diag(2, 1)
    phase = np.load(HERE / 'data' / 'p2_circles_256.npz')['phase']
    phase = np.kron(phase, np.ones((N // 256, N // 256)))
    for value, A in [(0, I), (1, kappa * np.diag([1., 2.])), (2, kappa * np.diag([2., 1.]))]:
        A_global[:, :, phase == value] = A[:, :, None]
    A_eff_ref = 0.15383360605988938 if args.kappa_exp == -3 else np.nan   # 4096^2 solution of the paper
elif args.problem == 'P3':
    # phase field from topology optimization, A = rho I + (1 - rho) kappa I
    rho = np.load(HERE / 'data' / 'p3_density_256.npz')['density']
    rho = np.kron(rho, np.ones((N // 256, N // 256)))
    A_global = np.einsum('ij,xy->ijxy', I, rho + (1 - rho) * kappa)
    A_eff_ref = 0.5793392277572645 if args.kappa_exp == -3 else np.nan    # 4096^2 solution of the paper

eig = np.linalg.eigvalsh(A_global.transpose(2, 3, 0, 1))
eig_min, eig_max = eig.min(), eig.max()

material = discretization.get_material_data_size_field_mugrid(name='conductivity')
material.s[...] = A_global[:, :, None][(..., *local)]
material_inv = discretization.get_material_data_size_field_mugrid(name='conductivity_inverse')
material_inv.s[...] = np.moveaxis(np.linalg.inv(np.moveaxis(material.s, (0, 1), (-2, -1))), (-2, -1), (0, 1))

# ============================================================================
# SYSTEM
# ============================================================================
macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=np.array([1.0, 0.0]),
                                               macro_gradient_field_ijqxyz=macro_gradient_field)
rhs = discretization.get_unknown_size_field(name='rhs')
discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material,
                              macro_gradient_field_ijqxyz=macro_gradient_field,
                              rhs_inxyz=rhs)


def K_fun(x, Ax):
    discretization.apply_system_matrix_mugrid(material_data_field=material,
                                              input_field_inxyz=x, output_field_inxyz=Ax)
    discretization.fft.communicate_ghosts(Ax)


def K_inv_fun(x, Ax):
    discretization.apply_system_matrix_mugrid(material_data_field=material_inv,
                                              input_field_inxyz=x, output_field_inxyz=Ax)
    discretization.fft.communicate_ghosts(Ax)


green = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=I)


def G_fun(x, Px):
    discretization.fft.communicate_ghosts(x)
    discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green,
                                               input_nodal_field_fnxyz=x, output_nodal_field_fnxyz=Px)


tmp_1 = discretization.get_unknown_size_field(name='prec_tmp_1')
tmp_2 = discretization.get_unknown_size_field(name='prec_tmp_2')


def dual_fun(x, Px):
    """P = G K(A^-1) G"""
    G_fun(x, tmp_1)
    K_inv_fun(tmp_1, tmp_2)
    G_fun(tmp_2, Px)


if args.preconditioner == 'green_jacobi':
    jacobi = discretization.get_preconditioner_Jacobi_mugrid(material_data_field_ijklqxyz=material)


def green_jacobi_fun(x, Px):
    """P = D^-1/2 G D^-1/2"""
    tmp_1.s[...] = jacobi.s * x.s
    G_fun(tmp_1, Px)
    Px.s[...] *= jacobi.s


M_fun, lam_min, lam_max = {'green': (G_fun, eig_min, eig_max),
                           'green_jacobi': (green_jacobi_fun, None, None),
                           'dual': (dual_fun, 1.0, eig_max / eig_min)}[args.preconditioner]


def energy(x):
    return discretization.get_homogenized_energy_mugrid(material_data_field_ijklqxyz=material,
                                                        displacement_field_inxyz=x,
                                                        macro_gradient_field_ijqxyz=macro_gradient_field)


def cg(x, P, lambda_min, tol, stop, cb=None, **kw):
    out = solvers.conjugate_gradients_mugrid_experimental(
        comm=discretization.communicator, fc=discretization.field_collection,
        hessp=K_fun, b=rhs, x=x, P=P, tol=tol, maxiter=args.maxiter, callback=cb,
        lambda_min=lambda_min, stop_crit_norm=stop, tau=args.tau, **kw)
    return out[1] if isinstance(out, tuple) else {}


# ============================================================================
# 1. REFERENCE SOLUTION u_h  (dual preconditioner, lambda_min = 1: e <= (r, z) < 1e-26)
# ============================================================================
t0 = time.time()
u_h = discretization.get_unknown_size_field(name='u_h')
cg(u_h, dual_fun, None, 1e-13, 'rz')
A_h = energy(u_h)

# ============================================================================
# 2. MONITORED RUN (the paper's setting) with the true energy error
# ============================================================================
err = discretization.get_unknown_size_field(name='err')
K_err = discretization.get_unknown_size_field(name='K_err')
hist = {'rr': [], 'rz': [], 'error': []}


def record(it, x, r, p, z, stop_crit):
    hist['rr'].append(discretization.communicator.sum(np.dot(r.ravel(), r.ravel())))
    hist['rz'].append(discretization.communicator.sum(np.dot(r.ravel(), z.ravel())))
    err.s[...] = u_h.s - x
    K_fun(err, K_err)
    hist['error'].append(discretization.communicator.sum(np.dot(err.s.ravel(), K_err.s.ravel())))


u_k = discretization.get_unknown_size_field(name='u_k')
t1 = time.time()
norms = cg(u_k, M_fun, lam_min, np.sqrt(args.tol_monitored), 'all', cb=record)
t_monitored = time.time() - t1

# ============================================================================
# 3. PRODUCTION RUN: PT lower estimate, relative tolerance, minimal delay 10
# ============================================================================
prod = {'error': []}


def record_error(it, x, r, p, z, stop_crit):
    err.s[...] = u_h.s - x
    K_fun(err, K_err)
    prod['error'].append(discretization.communicator.sum(np.dot(err.s.ravel(), K_err.s.ravel())))


u_p = discretization.get_unknown_size_field(name='u_p')
t1 = time.time()
cg(u_p, M_fun, lam_min, args.tol_production, 'energy_lower_estim', energy_rtol=True, d_min=args.d_min)
t_production = time.time() - t1
u_p.s[...] = 0
norms_prod = cg(u_p, M_fun, lam_min, args.tol_production, 'energy_lower_estim', cb=record_error,
                energy_rtol=True, d_min=args.d_min)
k_production = len(prod['error']) - 1

# ============================================================================
# SAVE
# ============================================================================
error = np.array(hist['error'])
k_star = int(np.argmax(error <= args.tol_monitored)) if np.any(error <= args.tol_monitored) else None
if comm_world.rank == 0:
    out_dir = HERE / 'exp_data'
    out_dir.mkdir(exist_ok=True)
    stem = f'{args.problem}_N{N}_kappa1e{args.kappa_exp:+d}_{args.preconditioner}'
    np.savez(out_dir / f'{stem}.npz',
             problem=args.problem, N=N, kappa=kappa, preconditioner=args.preconditioner, tau=args.tau,
             lam_min=np.nan if lam_min is None else lam_min, lam_max=np.nan if lam_max is None else lam_max,
             rr=hist['rr'], rz=hist['rz'], error=error,
             GR=np.array(norms.get('energy_upper_bound', [])),
             PT_LB=np.array(norms.get('energy_lower_estim', [])),
             stop_iteration=np.array(norms.get('stop_iteration', {}), dtype=object),
             k_star=-1 if k_star is None else k_star,
             k_production=k_production, error_production=prod['error'][-1],
             error0_production=prod['error'][0],
             A_h=A_h, A_eff_ref=A_eff_ref,
             t_monitored=t_monitored, t_production=t_production)
    print(f'{stem}: k* = {k_star} (e_k <= {args.tol_monitored:g}), production stop at k = {k_production} '
          f'(e_k / e_0 = {prod["error"][-1] / prod["error"][0]:.2e}), {t_production:.2f} s; '
          f'A_h = {A_h:.10f}; total {time.time() - t0:.1f} s')
