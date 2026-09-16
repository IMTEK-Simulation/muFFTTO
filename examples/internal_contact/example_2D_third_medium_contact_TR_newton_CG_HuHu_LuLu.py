"""
2D third-medium contact with HuHu-LuLu regularization (Frederiksen et al.,
CMAME 436 (2025) 117595, Eq. 4), solved as an energy minimization with
NuMPI's bound-constrained trust-region Newton-CG.

    Pi(u) = int_Omega W(F_bar + grad u) dx
          + k_r/2 int_Omega ( Hu : Hu - (1/dim) Lu . Lu ) dx

    grad Pi = G^T P        + R u
    hess Pi = G^T C G      + R        with  R = k_r (H^T H - L^T L/dim)

Flags of interest
-----------------
--precond
    Precondition the inner Steihaug CG with  M = G^T C_ref G + R, i.e. the
    Green operator built from the reference material *plus* the same
    regularization. The trust region is then measured in the M-norm, so
    --delta0 changes units; --calibrate_delta reports the factor.

--inner_tol T
    Fixed relative tolerance for the inner CG. Default (omitted) uses the
    superlinear forcing sequence min(0.5, sqrt(||g||_inf)). Set 1e-6 to
    reproduce the stopping criterion of the old conjugate_gradients_mugrid
    driver.

--tr_mode {safeguard,dormant}
    'safeguard' (default): finite delta0/delta_max, so the trust region also
    limits the step length.
    'dormant': delta0 = delta_max = 1e6, so the boundary is never reached
    and Steihaug reduces exactly to (preconditioned) CG. The energy-based
    acceptance test still applies -- a step that inverts an element gives
    Pi = +inf, hence rho = -inf, hence rejection and delta *= 0.25 -- so the
    mechanism that keeps det F > 0 survives.

Benchmark decomposition of the cost (N=16, 100 increments):

    unpreconditioned, forcing sequence   220040 hessp   205 s
    preconditioned,   forcing sequence    49730 hessp   148 s
    preconditioned,   --inner_tol 1e-6         ?             ?
"""

import time
import os
import sys
import argparse

import numpy as np
from mpi4py import MPI
from NuMPI.IO import save_npy
from matplotlib import pyplot as plt

try:
    from NuMPI.Optimization import tr_newton_bounded
except ImportError:
    from NuMPI.Optimization import BoundedTRNewtonCG
    tr_newton_bounded = BoundedTRNewtonCG.tr_newton_bounded

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from muFFTTO import domain, tensor_operations
from muFFTTO import microstructure_library
from muFFTTO import material_models
from muFFTTO import visualization_utils

# ============================================================================
# argument parsing
# ============================================================================
parser = argparse.ArgumentParser(
    prog='example_2D_third_medium_contact_TR_newton_CG_HuHu_LuLu.py',
    description='Finite strain neo-Hookean TMC in 2D, trust-region Newton-CG'
)
parser.add_argument('-n', '--nb_pixel', default='32')
parser.add_argument('-inc', '--nb_increments', default='100')
parser.add_argument('--save_per_it', action='store_true')
parser.add_argument('--check_derivatives', action='store_true',
                    help='Finite-difference checks of grad/hessp, then exit')
parser.add_argument('--precond', action='store_true',
                    help='Precondition the inner CG with M = G^T C_ref G + R')
parser.add_argument('--calibrate_delta', action='store_true',
                    help='Print the M-norm/Euclidean delta ratio, then exit')
parser.add_argument('--inner_tol', type=float, default=None,
                    help='Fixed relative tolerance for the inner CG. Default '
                         'None = superlinear forcing sequence.')
parser.add_argument('--tr_mode', choices=['safeguard', 'dormant'],
                    default='safeguard',
                    help="'dormant' sets delta0 = delta_max = 1e6 so the "
                         'trust-region boundary is never active and the inner '
                         'solve is plain (preconditioned) CG.')
parser.add_argument('--delta0', type=float, default=None,
                    help='Override the initial trust-region radius '
                         '(ignored with --tr_mode dormant)')
parser.add_argument('--gtol_rel', type=float, default=1e-4,
                    help='Newton tolerance, relative to ||grad Pi||_inf at '
                         'the start of each increment')
parser.add_argument('--disp', action='store_true',
                    help='Per-iteration trust-region table')
parser.add_argument('--plot_figures', action='store_true',
                    help='Plot per-increment meshes and the final response')

script_name = os.path.splitext(os.path.basename(__file__))[0]
args = parser.parse_args()
nnn = int(args.nb_pixel)
ninc = int(args.nb_increments)
save_per_it = args.save_per_it
use_precond = args.precond
plot_figures = args.plot_figures

# ============================================================================
# problem setup
# ============================================================================
number_of_pixels = (nnn, nnn)
domain_size = [1, 1]
dim = len(domain_size)
problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'bilinear_rectangle'
formulation = 'finite_strain'

# --- trust-region controls ---------------------------------------------------
# 'dormant': the boundary can never be reached, so Steihaug == CG from the
#   first inner iteration. No growth transient, and the `not last_on_boundary`
#   convergence guard never costs an extra outer iteration.
# 'safeguard': finite radius. WITHOUT preconditioning delta is in
#   RMS-per-component units of the displacement fluctuation (1e-3 validated
#   over the full load path). WITH preconditioning it is an M-norm; use
#   --calibrate_delta and pass --delta0.
if args.tr_mode == 'dormant':
    tr_delta0 = 1e6
    tr_delta_max = 1e6
else:
    if use_precond:
        tr_delta0 = 5e-3
        tr_delta_max = 5e-2
    else:
        tr_delta0 = 1e-3
        tr_delta_max = 1e-2
    if args.delta0 is not None:
        tr_delta0 = args.delta0
        tr_delta_max = 10.0 * args.delta0

tr_gtol_rel = args.gtol_rel
tr_maxiter = 200
tr_inner_maxiter = 2000 if args.inner_tol is not None else 600

_info = {
    'problem_type': problem_type,
    'discretization_type': discretization_type,
    'element_type': element_type,
    'formulation': formulation,
    'solver': 'tr_newton_bounded',
    'preconditioned': use_precond,
    'tr_mode': args.tr_mode,
    'nb_of_pixels': number_of_pixels,
    'domain_size': domain_size,
    'ninc': ninc,
    'tr_delta0': tr_delta0,
    'tr_delta_max': tr_delta_max,
    'tr_gtol_rel': tr_gtol_rel,
    'inner_tol': -1.0 if args.inner_tol is None else args.inner_tol,
    'inner_maxiter': tr_inner_maxiter,
}

# ============================================================================
# discretization
# ============================================================================
my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)

comm = MPI.COMM_WORLD
rank = discretization.communicator.rank


def root_print(*a, **kw):
    if rank == 0:
        print(*a, **kw)


# ============================================================================
# output folders
# ============================================================================
tag = ('precond' if use_precond else 'noprecond')
tag += '_' + args.tr_mode
tag += '_itol' + ('forcing' if args.inner_tol is None else f'{args.inner_tol:g}')

file_folder_path = os.path.dirname(os.path.realpath(__file__))
data_folder_path = (file_folder_path + '/exp_data/' + script_name + '/'
                    + f'Nx={nnn}Ny={nnn}_{tag}/')
figure_folder_path = (file_folder_path + '/figures/' + script_name + '/'
                      + f'Nx={nnn}Ny={nnn}_{tag}/')

if rank == 0:
    os.makedirs(data_folder_path, exist_ok=True)
    os.makedirs(figure_folder_path, exist_ok=True)

# ============================================================================
# material parameters
# ============================================================================
E_matrix = 100.0
nu_matrix = 0.3
lam_matrix = E_matrix * nu_matrix / ((1 + nu_matrix) * (1 - 2 * nu_matrix))
mu_matrix = E_matrix / (2 * (1 + nu_matrix))
K, G = material_models.get_bulk_and_shear_modulus(E_matrix, nu_matrix)

k_v = 1e-5
E_void = k_v * E_matrix
nu_void = nu_matrix
lam_void = E_void * nu_void / ((1 + nu_void) * (1 - 2 * nu_void))
mu_void = E_void / (2 * (1 + nu_void))

alpha = 1e-6
k_r = alpha * domain_size[0] ** 2 * (K + G * 4 / 3)
inv_tr_I = 1.0 / dim  # = 1/Tr(I), Eq. (4)

# reference material for the Green preconditioner
_i = np.eye(dim)
II = np.einsum('ij,kl->ijkl', _i, _i)
I4rt = np.einsum('ik,jl->ijkl', _i, _i)
I4s = 0.5 * (I4rt + np.einsum('il,jk->ijkl', _i, _i))
ref_mat = lam_matrix * II + 2.0 * mu_matrix * I4s

_info.update(lam_matrix=lam_matrix, mu_matrix=mu_matrix,
             lam_inc=lam_void, mu_inc=mu_void, k_r=k_r)

# ============================================================================
# geometry
# ============================================================================
phase_field = discretization.get_scalar_field(name='phase_field')
phase_field.s[0, 0] = microstructure_library.get_geometry(
    nb_voxels=discretization.nb_of_pixels,
    microstructure_name='contact_test_geometry_1',
    coordinates=discretization.fft.coords
)

matrix_mask = phase_field.s[0, 0] > 0
inc_mask = phase_field.s[0, 0] == 0

# ============================================================================
# material fields
# ============================================================================
lam_field = discretization.get_quad_field_scalar(name='lam_field')
mu_field = discretization.get_quad_field_scalar(name='mu_field')

lam_field.s[0, 0, :, matrix_mask] = lam_matrix
lam_field.s[0, 0, :, inc_mask] = lam_void
mu_field.s[0, 0, :, matrix_mask] = mu_matrix
mu_field.s[0, 0, :, inc_mask] = mu_void

root_print(f'lam min/max : {lam_field.s.min():.4e} / {lam_field.s.max():.4e}')
root_print(f'mu  min/max : {mu_field.s.min():.4e} / {mu_field.s.max():.4e}')

material = material_models.NeoHookean(
    discretization=discretization,
    lam_1qxyz=lam_field,
    mu_1qxyz=mu_field,
    name='neo_hookean_two_phase'
)

# ============================================================================
# fields
# ============================================================================
macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
displacement_fluctuation_field = discretization.get_unknown_size_field(name='u_fluc')
grad_u_field = discretization.get_displacement_gradient_sized_field(name='grad_u')
total_strain_field = discretization.get_displacement_gradient_sized_field(name='F')
stress_field = discretization.get_displacement_gradient_sized_field(name='P')
tangent_field = discretization.get_material_data_size_field_mugrid(name='C')
energy_field = discretization.get_quad_field_scalar(name='energy_density')
J_1qxyz = discretization.get_quad_field_scalar(name='J')

x_field = discretization.get_unknown_size_field(name='x_field')
grad_field = discretization.get_unknown_size_field(name='grad_field')
Ru_field = discretization.get_unknown_size_field(name='Ru_field')

v_field = discretization.get_unknown_size_field(name='v_field')
Bv_field = discretization.get_unknown_size_field(name='Bv_field')

# preconditioner scratch (kept separate from the hessp path)
r_field = discretization.get_unknown_size_field(name='r_field')
Pv_field = discretization.get_unknown_size_field(name='Pv_field')

hess_u_ijkqxyz = discretization.get_displacement_hessian_size_field(name='hess_u')
HtH_field = discretization.get_unknown_size_field(name='HtH')
lap_u_inxyz = discretization.get_displacement_laplacian_at_quad_field(name='lap_u')
LtL_field = discretization.get_displacement_sized_field(name='LtL')

u_shape = np.asarray(displacement_fluctuation_field.s).shape
_qw = np.asarray(discretization.quadrature_weights).reshape((-1,) + (1,) * dim)


# ============================================================================
# reductions
# ============================================================================
def global_sum(a):
    return comm.allreduce(float(a), op=MPI.SUM)


def global_min(a):
    return comm.allreduce(float(a), op=MPI.MIN)


def global_max(a):
    return comm.allreduce(float(a), op=MPI.MAX)


def dot_global(a, b):
    return global_sum(np.dot(np.asarray(a).ravel(), np.asarray(b).ravel()))


# ============================================================================
# regularization operator  R = k_r (H^T H - L^T L / dim)
# ============================================================================
def add_regularization(x, out, scale=1.0):
    """out += scale * R x. Ghosts of x must already be current."""
    discretization.apply_hessian_operator_to_vector_field_mugrid(
        u_inxyz=x, hess_u_ijkqxyz=hess_u_ijkqxyz)
    discretization.apply_hessian_operator_transposed_to_vector_field_mugrid(
        hess_u_ijkqxyz=hess_u_ijkqxyz, nodal_field_inxyz=HtH_field,
        apply_weights=True)

    # discretization.laplacian.apply(nodal_field=x,
    #                                quadrature_point_field=lap_u_inxyz)
    # discretization.laplacian.transpose(quadrature_point_field=lap_u_inxyz,
    #                                    nodal_field=LtL_field,
    #                                    weights=discretization.quadrature_weights)

    out.s[...] += scale * k_r * (HtH_field.s - inv_tr_I * LtL_field.s)


# ============================================================================
# preconditioner:  M = G^T C_ref G + R,  applied as M^-1
# ============================================================================
def operator_for_preconditioner(input_field_inxyz, output_field_inxyz):
    """The operator whose Fourier symbol the Green blocks invert."""
    discretization.fft.communicate_ghosts(field=input_field_inxyz)
    discretization.apply_system_matrix_mugrid(
        material_data_field=ref_mat,
        input_field_inxyz=input_field_inxyz,
        output_field_inxyz=output_field_inxyz,
        formulation=formulation)
    add_regularization(input_field_inxyz, output_field_inxyz)


preconditioner = None
precond = None

if use_precond or args.calibrate_delta:
    preconditioner = discretization.get_preconditioner_Green_mugrid(
        reference_material_data_ijkl=ref_mat,
        operator=operator_for_preconditioner,
    )

    def precond(v):
        """Return M^-1 v."""
        r_field.s[...] = v
        discretization.fft.communicate_ghosts(r_field)
        discretization.apply_preconditioner_mugrid(
            preconditioner_Fourier_fnfnqks=preconditioner,
            input_nodal_field_fnxyz=r_field,
            output_nodal_field_fnxyz=Pv_field)
        return np.array(Pv_field.s, copy=True)


# ============================================================================
# objective, gradient, Hessian-vector product
# ============================================================================
_tangent_x = None
_eval_counts = {'fun': 0, 'hessp': 0}


def _set_F_from(x):
    """Load x, exchange ghosts, build F = F_bar + grad u; return global min J."""
    x_field.s[...] = x
    discretization.fft.communicate_ghosts(x_field)
    discretization.apply_gradient_operator_mugrid(
        u_inxyz=x_field, grad_u_ijqxyz=grad_u_field)
    total_strain_field.s[...] = macro_gradient_field.s + grad_u_field.s
    tensor_operations.det2(total_strain_field, J_1qxyz)
    return global_min(np.min(J_1qxyz.s))


def fun_grad(x):
    """(Pi, grad Pi). Pi is globally reduced; +inf on element inversion."""
    _eval_counts['fun'] += 1

    J_min = _set_F_from(x)
    if not np.isfinite(J_min) or J_min <= 0.0:
        return np.inf, np.zeros(u_shape)

    material.get_energy_density(total_strain_field, energy_field)
    Pi = global_sum(np.sum(_qw * energy_field.s[0, 0]))
    if not np.isfinite(Pi):
        return np.inf, np.zeros(u_shape)

    # regularization energy 1/2 <u, R u>, reusing R u for the gradient
    Ru_field.s[...] = 0.0
    add_regularization(x_field, Ru_field)
    Pi += 0.5 * dot_global(x_field.s, Ru_field.s)

    material.get_stress(total_strain_field, stress_field)
    discretization.fft.communicate_ghosts(stress_field)
    discretization.apply_gradient_transposed_operator_mugrid(
        gradient_field_ijqxyz=stress_field, div_u_fnxyz=grad_field,
        apply_weights=True)
    grad_field.s[...] += Ru_field.s

    return Pi, np.array(grad_field.s, copy=True)


def hessp(x, v):
    """B v = (G^T C G + R) v, with C evaluated at x (cached per outer step)."""
    global _tangent_x
    _eval_counts['hessp'] += 1

    if _tangent_x is None or not np.array_equal(_tangent_x, x):
        _set_F_from(x)
        material.get_algorithmic_tangent(total_strain_field, tangent_field)
        _tangent_x = np.array(x, copy=True)

    v_field.s[...] = v
    discretization.fft.communicate_ghosts(v_field)
    discretization.apply_system_matrix_mugrid(
        material_data_field=tangent_field,
        input_field_inxyz=v_field,
        output_field_inxyz=Bv_field,
        formulation=formulation)
    add_regularization(v_field, Bv_field)
    discretization.fft.communicate_ghosts(Bv_field)

    return np.array(Bv_field.s, copy=True)


# ============================================================================
# one-off diagnostics
# ============================================================================
def check_derivatives():
    rng = np.random.default_rng(0)
    macro_gradient_field.s[...] = 0.0
    for d in range(dim):
        macro_gradient_field.s[d, d] = 1.0

    u0 = 1e-3 * rng.standard_normal(u_shape)
    v = rng.standard_normal(u_shape)
    v /= np.sqrt(dot_global(v, v))

    Pi0, g0 = fun_grad(u0)
    gv = dot_global(g0, v)
    Bv = hessp(u0, v)

    root_print('\n  eps        |grad err|      |hess err|')
    root_print('  ' + '-' * 40)
    for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        Pi_p, g_p = fun_grad(u0 + eps * v)
        Pi_m, g_m = fun_grad(u0 - eps * v)
        e_g = abs((Pi_p - Pi_m) / (2 * eps) - gv) / max(abs(gv), 1e-300)
        H_fd = (g_p - g_m) / (2 * eps)
        e_H = np.sqrt(dot_global(H_fd - Bv, H_fd - Bv)) \
            / max(np.sqrt(dot_global(Bv, Bv)), 1e-300)
        root_print(f'  {eps:.0e}    {e_g:12.4e}    {e_H:12.4e}')

    w = rng.standard_normal(u_shape)
    num = dot_global(w, hessp(u0, v))
    asym = abs(num - dot_global(v, hessp(u0, w)))
    root_print(f'\n  relative asymmetry of hessp : {asym / (abs(num) + 1e-300):.3e}')

    Ru_field.s[...] = 0.0
    v_field.s[...] = v
    discretization.fft.communicate_ghosts(v_field)
    add_regularization(v_field, Ru_field)
    root_print(f'  <v, R v> / <v, v>           : '
               f'{dot_global(v, Ru_field.s) / dot_global(v, v):+.3e}'
               '   (must be >= 0)')


def calibrate_delta():
    """||s||_M / ||s||_2 for a Newton-like step, to convert delta0 to the
    M-norm. Uses ||z||_M^2 = z^T M z = z^T g, free since M z = g.

    The ratio is resolution dependent (the quadrature weights carry h^2),
    so re-run this at every N rather than reusing a value."""
    macro_gradient_field.s[...] = 0.0
    for d in range(dim):
        macro_gradient_field.s[d, d] = 1.0
    macro_gradient_field.s[1, 0] += -0.003

    u0 = np.zeros(u_shape)
    _, g = fun_grad(u0)
    z = precond(g)
    n2 = np.sqrt(dot_global(z, z))
    nM = np.sqrt(abs(dot_global(z, g)))
    ratio = nM / max(n2, 1e-300)
    root_print(f'\n  ||s||_M / ||s||_2      = {ratio:.4e}')
    root_print(f'  Euclidean delta0 1e-3 -> M-norm delta0 ~ {1e-3 * ratio:.4e}')
    root_print(f'  Euclidean delta_max 1e-2 -> M-norm      ~ {1e-2 * ratio:.4e}')
    root_print('\n  Pass roughly 2x the first value to --delta0 (it only needs')
    root_print('  to exceed the step length; the radius then stays inactive).')


if args.check_derivatives:
    check_derivatives()
    sys.exit(0)

if args.calibrate_delta:
    calibrate_delta()
    sys.exit(0)

# ============================================================================
# macroscopic loading
# ============================================================================
macro_gradient_inc = np.zeros((dim, dim))
macro_gradient_inc[0, 0] -= 0.3/ float(ninc)
# macro_gradient_inc[1, 0] += -0.3 / float(ninc)

_info['macro_gradient_inc'] = macro_gradient_inc
for key in ('mean_stress_field', 'total_macro_gradient', 'nb_hessp',
            'nb_precond', 'nb_outer', 'min_J', 'energy', 'delta_end',
            'grad_inf'):
    _info[key] = []

root_print('=' * 70)
root_print(f'preconditioning : {"ON  (M = G^T C_ref G + R)" if use_precond else "OFF"}')
root_print('inner tolerance : ' + ('forcing sequence min(0.5, sqrt(|g|_inf))'
                                   if args.inner_tol is None
                                   else f'fixed {args.inner_tol:g}'))
root_print(f'tr_mode         : {args.tr_mode}')
root_print(f'delta0 / max    : {tr_delta0:.3e} / {tr_delta_max:.3e}'
           + ('   (boundary never active)' if args.tr_mode == 'dormant'
              else ('   (M-norm units)' if use_precond
                    else '   (Euclidean, RMS/component)')))
root_print(f'gtol_rel        : {tr_gtol_rel:.1e}')

# ============================================================================
# incremental loading
# ============================================================================
sum_hessp = 0
sum_precond = 0
sum_outer = 0
iteration_total = 0
start_time = time.time()

u = np.zeros(u_shape)

for inc in range(ninc):
    total_macro_gradient = np.eye(dim) + (inc + 1) * macro_gradient_inc
    discretization.get_macro_gradient_field_mugrid(
        macro_gradient_ij=total_macro_gradient,
        macro_gradient_field_ijqxyz=macro_gradient_field)

    _tangent_x = None  # cache refers to the previous increment's energy

    root_print('=' * 70)
    root_print(f'Increment {inc}   load F_bar[1,0] = '
               f'{total_macro_gradient[1, 0]:+.4f}')

    Pi0, g0 = fun_grad(u)
    g0_inf = global_max(np.max(np.abs(g0))) if np.isfinite(Pi0) else np.inf
    gtol = max(tr_gtol_rel * g0_inf, 1e-14)
    root_print(f'  Pi_0 = {Pi0:12.6e}   |grad|_inf = {g0_inf:10.3e}'
               f'   gtol = {gtol:10.3e}')

    def tr_callback(x_accepted):
        global iteration_total
        iteration_total += 1
        if not save_per_it:
            return
        x_field.s[...] = x_accepted
        save_npy(
            data_folder_path + f'u_fluc_it{iteration_total}.npy',
            x_field.s.mean(axis=1),
            tuple(discretization.subdomain_locations_no_buffers),
            tuple(discretization.nb_of_pixels_global),
            MPI.COMM_WORLD)

    res = tr_newton_bounded(
        fun=fun_grad,
        x0=u,
        hessp=hessp,
        precond=precond,            # None -> unpreconditioned Steihaug
        jac=True,
        gtol=gtol,
        maxiter=tr_maxiter,
        delta0=tr_delta0,
        delta_max=tr_delta_max,
        inner_tol=args.inner_tol,   # None -> forcing sequence
        inner_maxiter=tr_inner_maxiter,
        comm=comm,
        callback=tr_callback,
        disp=args.disp,
    )

    u = np.asarray(res.x)
    sum_hessp += res.nb_hessp
    sum_precond += res.get('nb_precond', 0)
    sum_outer += res.nit

    J_min = _set_F_from(u)
    material.get_stress(total_strain_field, stress_field)
    material.get_algorithmic_tangent(total_strain_field, tangent_field)
    material.get_energy_density(total_strain_field, energy_field)
    _tangent_x = np.array(u, copy=True)

    delta_end = res.delta_history[-1] if res.delta_history else float('nan')

    root_print(f'  {res.message}')
    root_print(f'  outer its {res.nit:4d} | hessp {res.nb_hessp:6d} '
               f'| M^-1 {res.get("nb_precond", 0):6d} '
               f'| |grad|_inf {res.max_grad:10.3e} | Pi {res.fun:12.6e}')
    root_print(f'  min(det F) {J_min:10.3e} | delta_end {delta_end:9.3e}')
    root_print(f'  P_xx {stress_field.s[0, 0].mean():10.3e} | '
               f'P_yx {stress_field.s[1, 0].mean():10.3e}')

    displacement_fluctuation_field.s[...] = u
    _info['mean_stress_field'].append(stress_field.s[1, 0].mean())
    _info['total_macro_gradient'].append(total_macro_gradient)
    _info['nb_hessp'].append(res.nb_hessp)
    _info['nb_precond'].append(res.get('nb_precond', 0))
    _info['nb_outer'].append(res.nit)
    _info['min_J'].append(J_min)
    _info['energy'].append(res.fun)
    _info['delta_end'].append(delta_end)
    _info['grad_inf'].append(res.max_grad)

    if save_per_it:
        save_npy(data_folder_path + f'stress_field_inc{inc}.npy',
                 stress_field.s.mean(axis=2),
                 tuple(discretization.subdomain_locations_no_buffers),
                 tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD)
        save_npy(data_folder_path + f'energy_field_inc{inc}.npy',
                 energy_field.s[0, 0].mean(axis=0),
                 tuple(discretization.subdomain_locations_no_buffers),
                 tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD)

    if discretization.communicator.size == 1 and plot_figures:
        x_plot_ixyz = visualization_utils.get_deformed_grid_coords_two_dim(
            discretization,
            macro_gradient_ij=total_macro_gradient - np.eye(dim),
            displacement_fluctuation=displacement_fluctuation_field)
        visualization_utils.plot_field_on_grid(
            coordinates_for_plot=x_plot_ixyz,
            field_to_plot=phase_field.s[0, 0],
            name=f'phase, load increment {inc}')

    if not res.success:
        root_print(f'  STOP: increment {inc} did not converge '
                   f'({res.message}). Reduce the increment size or --delta0.')
        break

# ============================================================================
# response
# ============================================================================
if rank == 0 and plot_figures and len(_info['mean_stress_field']) > 1:
    F_plot = np.asarray(_info['total_macro_gradient'])[..., 0, 0]
    P_plot = np.asarray(_info['mean_stress_field'])

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot(F_plot, P_plot, '-o', ms=3, color='k')
    ax[0].set_xlabel(r'$\bar{F}_{10}$')
    ax[0].set_ylabel(r'$\bar{P}_{10}$')
    ax[0].grid(alpha=.3)

    ax[1].semilogy(_info['min_J'], '-o', ms=3, color='C3')
    ax[1].set_xlabel('increment')
    ax[1].set_ylabel(r'$\min \det F$')
    ax[1].grid(alpha=.3)

    ax[2].plot(_info['nb_hessp'], '-o', ms=3, color='C0')
    ax[2].set_xlabel('increment')
    ax[2].set_ylabel('Hessian products')
    ax[2].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(figure_folder_path + 'response.png', dpi=150)
    plt.show()

# ============================================================================
# summary
# ============================================================================
elapsed_time = time.time() - start_time
_info.update(sum_hessp=sum_hessp, sum_precond=sum_precond,
             sum_outer=sum_outer, elapsed_time=elapsed_time,
             nb_fun_evals=_eval_counts['fun'])

if rank == 0:
    print('=' * 70)
    print(f'element_type       : {element_type}')
    print(f'number_of_pixels   : {number_of_pixels}')
    print(f'preconditioned     : {use_precond}')
    print(f'tr_mode            : {args.tr_mode}')
    print('inner_tol          : ' + ('forcing' if args.inner_tol is None
                                     else f'{args.inner_tol:g}'))
    print(f'Total hessp        : {sum_hessp}')
    print(f'Total M^-1 applies : {sum_precond}')
    print(f'Total outer its    : {sum_outer}')
    print(f'Total fun evals    : {_eval_counts["fun"]}')
    print(f'Elapsed time       : {elapsed_time:.2f} s '
          f'({elapsed_time / 60:.2f} min)')
    if sum_hessp:
        print(f'hessp / outer it   : {sum_hessp / max(sum_outer, 1):.1f}')
        print(f'ms / hessp         : {1e3 * elapsed_time / sum_hessp:.3f}')
    np.savez(data_folder_path + 'info_log_final.npz', **_info)