"""
Geometries P1-P3 of Paper 1 (Ladecky, Papez, Tichy, Pastewka, *Guaranteed Bounds on the Iterative Error of
CG-Accelerated FFT-Based Homogenization*, Section 6) for small-strain linear elasticity (plane strain) in muFFTTO,
solved with three preconditioners:

    green         G = (B^T W C_0 B)^-1                reference material C_0 (the matrix phase)
    green_jacobi  D^-1/2 G_I D^-1/2,  D = diag(K)     reference 2 mu I (= I_ijkl for mu = 1/2)
    dual          G K(C_0 A^-1 C_0) G                 A^-1 on the symmetric strains (Mandel 3 x 3)

Materials (C_0 = isotropic, K = 1, mu = 1/2, as in the topology optimization examples):
    P1  square inclusion [0.25, 0.75)^2:  C_0 in the matrix, kappa C_0 in the inclusion
    P2  58 circles, two orthotropic phases kappa T C_0 T with T = diag(1, 2^1/4) and diag(2^1/4, 1) on every index
        (C_yyyy = 2 C_xxxx and the other way round, the analogue of kappa diag(1, 2) and kappa diag(2, 1))
    P3  topology-optimization phase field:  A = rho C_0 + (1 - rho) kappa C_0
Load: macroscopic strain E = e_x (x) e_x.

Spectral bounds of the preconditioned operator (g = eigenvalues of C_0^-1 A over all pixels):
    green         [min g, max g]
    dual          [1, max g / min g]   (lambda_min >= 1 by the complementary energy, as in conductivity)
    green_jacobi  no guaranteed bound -> no Gauss-Radau bound

Every case runs the three solves of run_experiment.py (reference, monitored to 1e-10, production with the PT lower
estimate, relative 1e-6, minimal delay 10) and writes exp_data_elasticity/<problem>_N<N>_kappa1e<k>_<prec>.npz.

    python run_experiment_elasticity.py --problem P2 --kappa-exp -3 --preconditioner dual
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
from muFFTTO import material_models
from muFFTTO import solvers

HERE = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--problem', choices=['P1', 'P2', 'P3'], required=True)
parser.add_argument('--kappa-exp', type=int, required=True, help='total phase contrast kappa = 10^kappa_exp')
parser.add_argument('--preconditioner', choices=['green', 'green_jacobi', 'dual'], required=True)
parser.add_argument('--grid', type=int, default=256,
                    help='grid N x N; P2, P3: a multiple or a divisor of 256 (data on 256^2 replicated / coarsened)')
parser.add_argument('--tau', type=float, default=0.25, help='tolerance of the adaptive delay')
parser.add_argument('--tol-monitored', type=float, default=1e-10, help='squared, as in the paper')
parser.add_argument('--tol-production', type=float, default=1e-6, help='relative energy tolerance')
parser.add_argument('--d-min', type=int, default=10, help='minimal delay of the production run')
parser.add_argument('--maxiter', type=int, default=20000)
parser.add_argument('--save_solution', action='store_true',
                    help='store the reference solution u_h (E = e_1 x e_1) and C_1111 per pixel in solutions/ (serial)')
args = parser.parse_args()

kappa = 10.0 ** args.kappa_exp
N = args.grid
comm_world = MPI.COMM_WORLD

# ============================================================================
# DISCRETIZATION
# ============================================================================
cell = domain.PeriodicUnitCell(domain_size=[1, 1], problem_type='elasticity')
discretization = domain.Discretization(cell=cell,
                                       nb_of_pixels_global=(N, N),
                                       discretization_type='finite_element',
                                       element_type='linear_triangles')
loc = discretization.subdomain_locations_no_buffers
nb = discretization.nb_of_pixels
local = (slice(loc[0], loc[0] + nb[0]), slice(loc[1], loc[1] + nb[1]))

# ============================================================================
# MATERIAL
# ============================================================================
dim = 2
C_0 = material_models.get_elastic_material_tensor(dim=dim, K=1.0, mu=0.5, kind='linear')
MANDEL = [(0, 0), (1, 1), (0, 1)]
W_MANDEL = np.array([1.0, 1.0, np.sqrt(2.0)])


def to_mandel(C):
    """[2, 2, 2, 2, ...] -> [3, 3, ...]"""
    return np.array([[W_MANDEL[a] * W_MANDEL[b] * C[i, j, k, l] for b, (k, l) in enumerate(MANDEL)]
                     for a, (i, j) in enumerate(MANDEL)])


def from_mandel(M):
    """[3, 3, ...] -> [2, 2, 2, 2, ...] with both minor symmetries"""
    C = np.zeros((2, 2, 2, 2) + M.shape[2:])
    for a, (i, j) in enumerate(MANDEL):
        for b, (k, l) in enumerate(MANDEL):
            value = M[a, b] / (W_MANDEL[a] * W_MANDEL[b])
            for ii, jj in {(i, j), (j, i)}:
                for kk, ll in {(k, l), (l, k)}:
                    C[ii, jj, kk, ll] = value
    return C


def rotate_index(C, T):
    """T_ia T_jb T_kc T_ld C_abcd: congruence, keeps C symmetric positive definite"""
    return np.einsum('ia,jb,kc,ld,abcd->ijkl', T, T, T, T, C)


C_global = np.zeros((2, 2, 2, 2, N, N))                    # stiffness per pixel, both quadrature points
if args.problem == 'P1':
    x_c = (np.arange(N) + 0.5) / N
    inside = np.abs(x_c - 0.5) < 0.25
    phase = 1 - np.outer(inside, inside)                     # 1 matrix, 0 inclusion
    C_global = np.einsum('ijkl,xy->ijklxy', C_0, phase + kappa * (1 - phase))
elif args.problem == 'P2':
    phase = np.load(HERE / 'data' / 'p2_circles_256.npz')['phase']
    if N >= 256:
        phase = np.kron(phase, np.ones((N // 256, N // 256)))
    else:                                                    # coarser grid: sample the pixel centres
        f = 256 // N
        phase = phase[f // 2::f, f // 2::f]
    t = 2 ** 0.25
    for value, C in [(0, C_0), (1, kappa * rotate_index(C_0, np.diag([1., t]))),
                     (2, kappa * rotate_index(C_0, np.diag([t, 1.])))]:
        C_global[..., phase == value] = C[..., None]
elif args.problem == 'P3':
    rho = np.load(HERE / 'data' / 'p3_density_256.npz')['density']
    if N >= 256:
        rho = np.kron(rho, np.ones((N // 256, N // 256)))
    else:                                                    # coarser grid: block average of the density
        f = 256 // N
        rho = rho.reshape(N, f, N, f).mean(axis=(1, 3))
    C_global = np.einsum('ijkl,xy->ijklxy', C_0, rho + (1 - rho) * kappa)

M_0 = to_mandel(C_0)
M_global = to_mandel(C_global)                                # [3, 3, N, N]
L_0_inv = np.linalg.inv(np.linalg.cholesky(M_0))
g = np.linalg.eigvalsh(np.einsum('ab,bcxy,dc->xyad', L_0_inv, M_global, L_0_inv))   # eigenvalues of C_0^-1 A
g_min, g_max = g.min(), g.max()
# C_0 A^-1 C_0 on the symmetric strains
M_dual = np.einsum('ab,xybc,cd->adxy', M_0, np.linalg.inv(M_global.transpose(2, 3, 0, 1)), M_0)
C_dual_global = from_mandel(M_dual)

material = discretization.get_material_data_size_field_mugrid(name='stiffness')
material.s[...] = C_global[:, :, :, :, None][(..., *local)]
material_dual = discretization.get_material_data_size_field_mugrid(name='stiffness_dual')
material_dual.s[...] = C_dual_global[:, :, :, :, None][(..., *local)]

# ============================================================================
# SYSTEM
# ============================================================================
macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=np.array([[1.0, 0.0], [0.0, 0.0]]),
                                               macro_gradient_field_ijqxyz=macro_gradient_field)
rhs = discretization.get_unknown_size_field(name='rhs')
discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material,
                              macro_gradient_field_ijqxyz=macro_gradient_field,
                              rhs_inxyz=rhs)


def K_fun(x, Ax):
    discretization.apply_system_matrix_mugrid(material_data_field=material, input_field_inxyz=x,
                                              output_field_inxyz=Ax, formulation='small_strain')
    discretization.fft.communicate_ghosts(Ax)


def K_dual_fun(x, Ax):
    discretization.apply_system_matrix_mugrid(material_data_field=material_dual, input_field_inxyz=x,
                                              output_field_inxyz=Ax, formulation='small_strain')
    discretization.fft.communicate_ghosts(Ax)


green_C0 = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=C_0, name='green_C0')
green_I = discretization.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=np.einsum('ik,jl->ijkl', np.eye(dim), np.eye(dim)),     # 2 mu I, mu = 1/2
    name='green_2muI')


def green_apply(G):
    def apply(x, Px):
        discretization.fft.communicate_ghosts(x)
        discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=G,
                                                   input_nodal_field_fnxyz=x, output_nodal_field_fnxyz=Px)
    return apply


G_fun = green_apply(green_C0)
tmp_1 = discretization.get_unknown_size_field(name='prec_tmp_1')
tmp_2 = discretization.get_unknown_size_field(name='prec_tmp_2')


def dual_fun(x, Px):
    """P = G K(C_0 A^-1 C_0) G"""
    G_fun(x, tmp_1)
    K_dual_fun(tmp_1, tmp_2)
    G_fun(tmp_2, Px)


if args.preconditioner == 'green_jacobi':
    jacobi = discretization.get_preconditioner_Jacobi_mugrid(material_data_field_ijklqxyz=material)


def green_jacobi_fun(x, Px):
    """P = D^-1/2 G_I D^-1/2"""
    tmp_1.s[...] = jacobi.s * x.s
    green_apply(green_I)(tmp_1, Px)
    Px.s[...] *= jacobi.s


M_fun, lam_min, lam_max = {'green': (G_fun, g_min, g_max),
                           'green_jacobi': (green_jacobi_fun, None, None),
                           'dual': (dual_fun, 1.0, g_max / g_min)}[args.preconditioner]


def energy(x):
    return discretization.get_homogenized_energy_mugrid(material_data_field_ijklqxyz=material,
                                                        displacement_field_inxyz=x,
                                                        macro_gradient_field_ijqxyz=macro_gradient_field,
                                                        formulation='small_strain')


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
if args.save_solution and comm_world.size == 1:
    (HERE / 'solutions').mkdir(exist_ok=True)
    np.savez(HERE / 'solutions' / f'{args.problem}_N{N}_kappa1e{args.kappa_exp:+d}.npz', u=np.array(u_h.s[:, 0]),
             C1111=C_global[0, 0, 0, 0], E=np.array([[1.0, 0.0], [0.0, 0.0]]))

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
    out_dir = HERE / 'exp_data_elasticity'
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
             A_h=A_h, A_eff_ref=np.nan,
             t_monitored=t_monitored, t_production=t_production)
    print(f'{stem}: k* = {k_star} (e_k <= {args.tol_monitored:g}), production stop at k = {k_production} '
          f'(e_k / e_0 = {prod["error"][-1] / prod["error"][0]:.2e}), {t_production:.2f} s; '
          f'A_h = {A_h:.10f}; total {time.time() - t0:.1f} s')
