"""
2D third-medium contact with full HuHu-LuLu regularization, solved as an
ENERGY MINIMIZATION with NuMPI's trust-region Newton-CG, plus explicit
control of the imposed deformation gradient along the load path.

    Pi(u) = int_Omega W(F) dx
          + k_r/2 int_Omega ( Hu : Hu - (1/dim) Lu . Lu ) dx

    grad Pi = G^T P   + R u
    hess Pi = G^T C G + R        with  R = k_r (H^T H - L^T L/dim)

Why not plain Newton-CG
-----------------------
Under a prescribed macroscopic F the buckling of the third-medium ligaments is
a *bifurcation*: past the critical load the Hessian G^T C G + R acquires a
negative eigenvalue.  Two things then break in the plain Newton-CG driver:

  * linear CG structurally requires a positive definite operator, so
    `solvers.conjugate_gradients_mugrid` raises 'Hessian is not positive
    definite';
  * more fundamentally, the unbuckled symmetric configuration is *still an
    exact equilibrium* -- it has merely turned from a minimum of Pi into a
    saddle.  A residual-based solver has no preference between the two, so
    patching CG would only make it converge onto the unstable branch.

Minimizing Pi instead fixes both.  Trust-region Steihaug handles indefinite
Hessians by design: when it meets a direction d with d^T B d <= 0 it walks to
the trust-region boundary along it (both intersections are evaluated and the
lower model value is kept) and reports 'negative curvature'.  The count of
those inner terminations is reported per increment -- it is the direct
numerical evidence of buckling.

What is kept from the plain-Newton step-control script
------------------------------------------------------
  * `set_F`: F is rebuilt from scratch from (lam, u) every evaluation, never
    accumulated, so a load step can be halved or rolled back exactly;
  * `max_admissible_scale`: the exact 2D fraction-to-the-boundary scale that
    keeps det F > 0, used to cap the LOAD increment at the predictor;
  * the full HuHu-LuLu regularization and 'biquadratic_rectangle'.

The per-Newton-step cap of the plain driver is deliberately NOT kept: the
trust region owns the step length now, and admissibility inside the solve is
enforced by the energy itself (an inversion-producing trial step gets the
INADMISSIBLE_ENERGY penalty, hence rho << 0, hence rejection and delta/4).
Keeping both controls would make them fight each other.

NOTE on the element type: the discrete Laplacian is identically zero for Q1
elements on axis-aligned pixels (see the RuntimeWarning in
discretization_library.py), which would silently reduce HuHu-LuLu to
HuHu-only.  The LuLu term therefore requires 'biquadratic_rectangle'.

Flags of interest
-----------------
--check_derivatives
    Finite-difference check that Pi and grad Pi are the same functional, that
    hessp is symmetric, and that <v, R v> >= 0.  RUN THIS FIRST: rho, and
    therefore every accept/reject decision, is meaningless if the energy and
    the gradient disagree.  The <v, R v> test also settles whether an observed
    negative curvature is physics or an overshooting LuLu subtraction.

--precond
    Precondition the inner Steihaug CG with M = G^T C_ref G + R.  The trust
    region is then measured in the M-norm, so --delta0 changes units;
    --calibrate_delta reports the conversion factor (resolution dependent,
    re-run per -n).

--imperfection X
    Seed u at lam = 0 with a fixed-seed random field of RMS
    X * max|H_macro| per component.  A perfectly symmetric discretization has
    no reason to leave the symmetric branch even after it has become a
    saddle; the imperfection selects a branch.  --imperfection 0 reproduces
    the symmetric path for comparison.

Requires the patched NuMPI BoundedTRNewtonCG.py that supports `precond`
(see the import guard below).
"""

import time
import os
import sys
import argparse
import inspect

import numpy as np
from mpi4py import MPI
from NuMPI.IO import save_npy
from matplotlib import pyplot as plt

from NuMPI.Optimization import tr_newton_bounded

# ---------------------------------------------------------------------------
# External-boundary validation.  `precond` and the M-norm Steihaug path exist
# only in a hand-patched BoundedTRNewtonCG.py; NuMPI.__version__ still reports
# 0.15.1 either way, so nothing warns you.  Fail loudly and early instead of
# with a bare TypeError several hundred lines down.
# ---------------------------------------------------------------------------
if 'precond' not in inspect.signature(tr_newton_bounded).parameters:
    raise RuntimeError(
        'Installed NuMPI has no `precond` support; this script needs the '
        'patched BoundedTRNewtonCG.py. Stock NuMPI 0.15.1 will not work.')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from muFFTTO import domain, tensor_operations
from muFFTTO import microstructure_library
from muFFTTO import material_models
from muFFTTO import visualization_utils

# ============================================================================
# argument parsing
# ============================================================================
parser = argparse.ArgumentParser(
    prog='example_2D_third_medium_contact_TR_newton_CG_HuHu_LuLu_stepctrl.py',
    description='Finite strain neo-Hookean TMC in 2D, trust-region Newton-CG '
                'energy minimization with deformation gradient step control'
)
parser.add_argument('-n', '--nb_pixel', default='32')
parser.add_argument('-inc', '--nb_increments', default='100')
parser.add_argument('--save_per_it', action='store_true')
parser.add_argument('--check_derivatives', action='store_true',
                    help='Finite-difference checks of grad/hessp and of the '
                         'regularizer sign, then exit')
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
                         'trust-region boundary is never active. NOTE: with a '
                         'negative eigenvalue present the model is unbounded '
                         'below, so a dormant region has nothing to stop the '
                         'negative-curvature step -- use it only below the '
                         'critical load.')
parser.add_argument('--delta0', type=float, default=None,
                    help='Override the initial trust-region radius '
                         '(ignored with --tr_mode dormant)')
parser.add_argument('--gtol_rel', type=float, default=1e-4,
                    help='Newton tolerance, relative to ||grad Pi||_inf at '
                         'the start of each increment')
parser.add_argument('--imperfection', type=float, default=1e-4,
                    help='Symmetry-breaking seed amplitude at lam = 0, as a '
                         'fraction of max|H_macro|. 0 = symmetric path.')
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
element_type = 'biquadratic_rectangle'  # required for a nonzero LuLu term
formulation = 'finite_strain'

# --- load step control (OUTSIDE the inner solver) ---------------------------
LOAD_SAFETY = 0.5      # fraction of the way to det F = 0 for the load step
MAX_INCREMENTS = 10 * ninc
MAX_CUTS = 8           # increment halvings before giving up
DLAM_MIN = 1e-9        # below this the load path is considered stalled

# Finite on purpose: np.inf would make ared = inf - inf = nan whenever the
# *current* iterate is also inadmissible, and a nan rho fails all three
# comparisons in the radius update -- the step is rejected but delta is never
# shrunk, so the solver spins to maxiter.  A large finite value gives
# rho << 0, hence delta *= 0.25, and keeps the diagnostics readable.
INADMISSIBLE_ENERGY = 1e30

# --- trust-region controls --------------------------------------------------
# delta is in RMS-per-component units (the solver uses delta*sqrt(n_global)).
# WITHOUT preconditioning that is the displacement fluctuation itself; WITH
# preconditioning it is an M-norm, so use --calibrate_delta and --delta0.
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
    'LOAD_SAFETY': LOAD_SAFETY,
    'MAX_CUTS': MAX_CUTS,
    'imperfection': args.imperfection,
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
tag += f'_imp{args.imperfection:g}'

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
# matrix: soft neo-Hookean
E_matrix = 100.0
nu_matrix = 0.3
lam_matrix = E_matrix * nu_matrix / ((1 + nu_matrix) * (1 - 2 * nu_matrix))
mu_matrix = E_matrix / (2 * (1 + nu_matrix))
K, G = material_models.get_bulk_and_shear_modulus(E_matrix, nu_matrix)

# third medium ("void"): same neo-Hookean as the matrix, k_v times softer
k_v = 1e-5  # TMC contrast, Table 1
E_void = k_v * E_matrix
nu_void = nu_matrix  # keep the solid's Poisson ratio
lam_void = E_void * nu_void / ((1 + nu_void) * (1 - 2 * nu_void))
mu_void = E_void / (2 * (1 + nu_void))

# HuHu-LuLu regularization
alpha = 1e-6  # the HuHu coefficient
k_r = alpha * domain_size[0] ** 2 * (K + G * 4 / 3)
inv_tr_I = 1.0 / dim  # = 1/Tr(I), Eq. (4)

# reference material for the Green preconditioner
_i = np.eye(dim)
II = np.einsum('ij,kl->ijkl', _i, _i)
I4rt = np.einsum('ik,jl->ijkl', _i, _i)
I4s = 0.5 * (I4rt + np.einsum('il,jk->ijkl', _i, _i))
ref_mat = lam_matrix * II + 2.0 * mu_matrix * I4s

_info.update(lam_matrix=lam_matrix, mu_matrix=mu_matrix,
             lam_inc=lam_void, mu_inc=mu_void, k_r=k_r, k_v=k_v)

# ============================================================================
# geometry
# ============================================================================
geometry_name = 'contact_test_geometry_2'

phase_field = discretization.get_scalar_field(name='phase_field')
phase_field.s[0, 0] = microstructure_library.get_geometry(
    nb_voxels=discretization.nb_of_pixels,
    microstructure_name=geometry_name,
    coordinates=discretization.fft.coords
)

matrix_mask = phase_field.s[0, 0] > 0
inc_mask = phase_field.s[0, 0] == 0

_info['geometry'] = geometry_name

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
root_print(f'k_r (HuHu)  : {k_r:.6e}')

material = material_models.NeoHookean(
    discretization=discretization,
    lam_1qxyz=lam_field,
    mu_1qxyz=mu_field,
    name='neo_hookean_two_phase'
)

# ============================================================================
# fields
#
# CONVENTION: macro_gradient_full_field holds H_macro WITHOUT the identity.
# `set_F` adds I.  The load cap needs dF/dlam = H_macro, which is only true
# with the identity kept out of the macro field.
# ============================================================================
macro_gradient_full_field = discretization.get_gradient_size_field(
    name='macro_gradient_full_field')
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
# deformation gradient bookkeeping and admissibility
# ============================================================================
def min_J(F_in):
    """Global min over all quadrature points of det F."""
    tensor_operations.det2(F_in, J_1qxyz)
    return global_min(np.min(J_1qxyz.s))


def set_F(lam, u_field, out):
    """F = I + lam * H_macro + grad(u~).

    Rebuilt from scratch, never accumulated, so the imposed macro gradient is
    always exactly lam * H_macro and a load step can be halved or rolled back.
    """
    discretization.fft.communicate_ghosts(u_field)
    discretization.apply_gradient_operator_mugrid(u_inxyz=u_field,
                                                  grad_u_ijqxyz=grad_u_field)
    out.s[...] = grad_u_field.s + lam * macro_gradient_full_field.s
    for d in range(dim):
        out.s[d, d] += 1.0


def max_admissible_scale(F_in, dF_in):
    """Largest s > 0 keeping det(F + s*dF) > 0 at every quadrature point.

    In 2D this is exact rather than a bound: for a 2x2 F,

        det(F + s dF) = det F + s*b + s^2 * det dF
        b = F00 dF11 + dF00 F11 - F01 dF10 - dF01 F10

    so the first crossing of zero is the smallest positive root of a quadratic
    and needs no bisection.  Returns +inf when the ray never crosses (the step
    is unconstrained) and 0.0 if F is already inadmissible.

    Here it serves the LOAD step only (dF = H_macro, u frozen, so F is affine
    in the load parameter).  The Newton step length is the trust region's job.
    """
    F, D = F_in.s, dF_in.s

    c = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    b = (F[0, 0] * D[1, 1] + D[0, 0] * F[1, 1]
         - F[0, 1] * D[1, 0] - D[0, 1] * F[1, 0])
    a = D[0, 0] * D[1, 1] - D[0, 1] * D[1, 0]

    if global_min(np.min(c)) <= 0.0:
        return 0.0

    with np.errstate(divide='ignore', invalid='ignore'):
        # degenerate (rank-one or uniform) dF: det dF = 0, so det F is linear
        s_linear = np.where(b < 0.0, -c / b, np.inf)

        disc = b * b - 4.0 * a * c
        root = np.sqrt(np.maximum(disc, 0.0))
        r_minus = (-b - root) / (2.0 * a)
        r_plus = (-b + root) / (2.0 * a)
        first_positive = np.minimum(np.where(r_minus > 0.0, r_minus, np.inf),
                                    np.where(r_plus > 0.0, r_plus, np.inf))
        s_quadratic = np.where(disc >= 0.0, first_positive, np.inf)

    s_qxy = np.where(np.abs(a) <= 1e-30 * (np.abs(b) + 1.0),
                     s_linear, s_quadratic)

    return global_min(np.min(s_qxy))


# ============================================================================
# regularization operator  R = k_r (H^T H - L^T L / dim)
# ============================================================================
def add_regularization(x, out, scale=1.0):
    """out += scale * R x.  Ghosts of x must already be current."""
    discretization.apply_hessian_operator_to_vector_field_mugrid(
        u_inxyz=x, hess_u_ijkqxyz=hess_u_ijkqxyz)
    discretization.apply_hessian_operator_transposed_to_vector_field_mugrid(
        hess_u_ijkqxyz=hess_u_ijkqxyz, nodal_field_inxyz=HtH_field,
        apply_weights=True)

    discretization.laplacian.apply(nodal_field=x,
                                   quadrature_point_field=lap_u_inxyz)
    discretization.laplacian.transpose(quadrature_point_field=lap_u_inxyz,
                                       nodal_field=LtL_field,
                                       weights=discretization.quadrature_weights)

    out.s[...] += scale * k_r * (HtH_field.s - inv_tr_I * LtL_field.s)


# ============================================================================
# preconditioner:  M = G^T C_ref G + R,  applied as M^-1
#
# Built ONCE from the fixed reference material, not refreshed per increment:
# with --precond the preconditioner also defines the trust-region metric, so
# changing M mid-run would silently change the meaning of delta.
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
#
# The load parameter is read from the module-level `lam_current`, which the
# increment loop sets before each solve.  fun_grad/hessp therefore see
# F = I + lam_current * H_macro + grad u.
# ============================================================================
lam_current = 0.0

_tangent_x = None
_eval_counts = {'fun': 0, 'hessp': 0, 'neg_curv': 0, 'inadmissible': 0}


def _set_F_from(x):
    """Load x, exchange ghosts, build F = I + lam*H + grad u; return min J."""
    x_field.s[...] = x
    set_F(lam_current, x_field, total_strain_field)
    return min_J(total_strain_field)


def fun_grad(x):
    """(Pi, grad Pi).  Pi is globally reduced.

    On element inversion returns INADMISSIBLE_ENERGY (large but FINITE, see
    the constant's comment) and a zero gradient.  A zero gradient at an
    inadmissible point would look like convergence to the outer loop, so the
    increment loop verifies admissibility of the predictor before every solve;
    inside the solve only trial points can be inadmissible, and those are
    rejected by the rho test before their gradient is ever adopted.
    """
    _eval_counts['fun'] += 1

    J_min = _set_F_from(x)
    if not np.isfinite(J_min) or J_min <= 0.0:
        _eval_counts['inadmissible'] += 1
        return INADMISSIBLE_ENERGY, np.zeros(u_shape)

    material.get_energy_density(total_strain_field, energy_field)
    Pi = global_sum(np.sum(_qw * energy_field.s[0, 0]))
    if not np.isfinite(Pi):
        _eval_counts['inadmissible'] += 1
        return INADMISSIBLE_ENERGY, np.zeros(u_shape)

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
    """B v = (G^T C G + R) v, with C evaluated at x (cached per outer step).

    Also counts negative curvature.  `_steihaug_cg` calls hessp exactly once
    per inner iteration, always as hessp(x, d) for the current CG direction,
    and terminates immediately with status 'negative curvature' when
    <d, B d> <= 0.  (The one other call site -- the re-evaluation after a
    box projection clips the step -- cannot fire here, because we pass no
    bounds and no zero_mask, so the projection is the identity.)  Counting
    <v, B v> <= 0 here is therefore *exactly* the number of inner solves that
    ended on negative curvature, which the solver does not otherwise report.
    """
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

    if dot_global(v, Bv_field.s) <= 0.0:
        _eval_counts['neg_curv'] += 1

    return np.array(Bv_field.s, copy=True)


# ============================================================================
# macroscopic loading
#
# H_macro is the TOTAL target macro gradient (F - I).  The load parameter lam
# runs 0 -> 1, so the imposed deformation gradient is exactly I + lam*H_macro
# at every point of the path.
# ============================================================================
H_macro = np.zeros((dim, dim))
H_macro[1, 0] = -0.6
# H_macro[0, 0] = -0.3

discretization.get_macro_gradient_field_mugrid(
    macro_gradient_ij=H_macro,
    macro_gradient_field_ijqxyz=macro_gradient_full_field
)

dlam_nominal = 1.0 / float(ninc)
H_macro_scale = float(np.max(np.abs(H_macro)))

_info['H_macro'] = H_macro


# ============================================================================
# one-off diagnostics
# ============================================================================
def check_derivatives():
    """Pi and grad Pi must be the same functional, or rho -- and therefore
    every accept/reject decision -- is meaningless.

    If |grad err| plateaus at a constant factor instead of falling with eps,
    the quadrature weighting of the energy is wrong: compare against the
    W_MAT calibration in
    example_2D_homogenization_third_medium_contact_trust_region_newton.py,
    which exists for precisely this reason.
    """
    global lam_current
    lam_current = 0.0

    rng = np.random.default_rng(0)
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
    root_print(f'\n  relative asymmetry of hessp : {asym / (abs(num) + 1e-300):.3e}'
               '   (must be < 1e-10)')

    # If the LuLu subtraction overshoots, R itself supplies the negative
    # curvature and no solver change is the right fix.
    Ru_field.s[...] = 0.0
    v_field.s[...] = v
    discretization.fft.communicate_ghosts(v_field)
    add_regularization(v_field, Ru_field)
    vRv = dot_global(v, Ru_field.s) / dot_global(v, v)
    root_print(f'  <v, R v> / <v, v>           : {vRv:+.3e}'
               '   (must be >= 0)')
    if vRv < 0.0:
        root_print('\n  FAIL: the HuHu-LuLu regularizer is NOT positive '
                   'semi-definite on this\n        element/resolution.  Any '
                   'negative curvature seen in the solve is\n        the '
                   'regularizer, not buckling.')


def calibrate_delta():
    """||s||_M / ||s||_2 for a Newton-like step, to convert delta0 to the
    M-norm.  Uses ||z||_M^2 = z^T M z = z^T g, free since M z = g.

    The ratio is resolution dependent (the quadrature weights carry h^2),
    so re-run this at every N rather than reusing a value."""
    global lam_current
    # a small but nonzero load, so that g != 0: max|lam*H| = 0.003
    lam_current = (0.003 / H_macro_scale) if H_macro_scale > 0.0 else 0.0

    u0 = np.zeros(u_shape)
    _, g = fun_grad(u0)
    z = precond(g)
    n2 = np.sqrt(dot_global(z, z))
    nM = np.sqrt(abs(dot_global(z, g)))
    ratio = nM / max(n2, 1e-300)
    root_print(f'\n  probe load  max|lam*H| = {lam_current * H_macro_scale:.4e}')
    root_print(f'  ||s||_M / ||s||_2      = {ratio:.4e}')
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
# reporting containers
# ============================================================================
for key in ('mean_stress_field', 'mean_stress_xx', 'total_macro_gradient',
            'nb_hessp', 'nb_precond', 'nb_outer', 'nb_neg_curv', 'min_J',
            'energy', 'delta_end', 'grad_inf', 'load_scale', 'lam', 'nb_cuts'):
    _info[key] = []

root_print('=' * 70)
root_print(f'geometry        : {geometry_name}')
root_print(f'element_type    : {element_type}')
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
root_print(f'imperfection    : {args.imperfection:.3e} * max|H_macro|')

# ============================================================================
# initial state: lam = 0, u = imperfection seed
#
# A perfectly symmetric discretization has no reason to leave the symmetric
# branch even after it has turned into a saddle: the gradient stays in the
# symmetric subspace, and so does every Steihaug direction built from it.  A
# fixed-seed random seed (reproducible) selects a branch.  The mean is removed
# because a uniform shift is a rigid translation, which lives in the null
# space of both G and the Green preconditioner.
# ============================================================================
u = np.zeros(u_shape)
if args.imperfection != 0.0:
    _rng_imp = np.random.default_rng(20240917)
    u = (args.imperfection * max(H_macro_scale, 1e-300)
         * _rng_imp.standard_normal(u_shape))
    u -= u.mean(axis=tuple(range(1, u.ndim)), keepdims=True)

lam_current = 0.0
J0 = _set_F_from(u)
root_print(f'min J at lam = 0 : {J0:.6e}')
if J0 <= 0.0:
    raise RuntimeError(f'initial state already inadmissible (min J = {J0:.3e}); '
                       'reduce --imperfection')

material.get_algorithmic_tangent(total_strain_field, tangent_field)
root_print(f'tangent min/max  : {tangent_field.s.min():.4e} / '
           f'{tangent_field.s.max():.4e}')

# ============================================================================
# incremental loading -- trust-region energy minimization with load control
# ============================================================================
sum_hessp = 0
sum_precond = 0
sum_outer = 0
sum_neg_curv = 0
iteration_total = 0
start_time = time.time()

lam = 0.0
inc = -1
stop_reason = 'load path completed'

while lam < 1.0 - 1e-12 and inc < MAX_INCREMENTS:
    inc += 1

    # ---- roll-back state, and how much load the worst quad point can take --
    # u is frozen, so F is affine in dlam:  F + dlam * H_macro.
    lam_backup = lam
    u_backup = np.array(u, copy=True)

    lam_current = lam_backup
    _set_F_from(u_backup)
    s_star_load = max_admissible_scale(total_strain_field,
                                       macro_gradient_full_field)
    dlam = min(dlam_nominal, LOAD_SAFETY * s_star_load, 1.0 - lam_backup)

    n_cuts = 0
    converged = False
    res = None

    # ---- increment, with cutback on failure --------------------------------
    while True:
        if dlam < DLAM_MIN:
            stop_reason = f'load increment underflow at lam = {lam_backup:.6f}'
            break

        lam_try = lam_backup + dlam
        lam_current = lam_try
        u = np.array(u_backup, copy=True)
        _tangent_x = None  # the cache refers to the previous (lam, u)

        root_print('=' * 70)
        root_print(f'Increment {inc}'
                   + (f'   cutback {n_cuts}/{MAX_CUTS}' if n_cuts else ''))
        root_print(f'  lam {lam_try:.6f}   dlam {dlam:.3e} '
                   f'(nominal {dlam_nominal:.3e})')
        root_print(f'  max admissible dlam {s_star_load:.3e}   '
                   f'F_bar = I + lam*H, lam*H_10 = '
                   f'{lam_try * H_macro[1, 0]:+.5f}, lam*H_00 = '
                   f'{lam_try * H_macro[0, 0]:+.5f}')

        # ---- predictor admissibility --------------------------------------
        # The load cap should guarantee this; check anyway, because fun_grad
        # returns a zero gradient at an inadmissible point and the outer loop
        # would read that as convergence.
        J_pred = _set_F_from(u)
        Pi0, g0 = fun_grad(u)
        if J_pred <= 0.0 or Pi0 >= INADMISSIBLE_ENERGY:
            root_print(f'  predictor inadmissible (min J = {J_pred:.3e}); '
                       f'cutback')
            n_cuts += 1
            dlam *= 0.5
            if n_cuts > MAX_CUTS:
                stop_reason = f'predictor inadmissible after {MAX_CUTS} cutbacks'
                break
            continue

        g0_inf = global_max(np.max(np.abs(g0)))
        gtol = max(tr_gtol_rel * g0_inf, 1e-14)
        root_print(f'  min J after load step {J_pred:10.3e}')
        root_print(f'  Pi_0 = {Pi0:12.6e}   |grad|_inf = {g0_inf:10.3e}'
                   f'   gtol = {gtol:10.3e}')

        neg_curv_0 = _eval_counts['neg_curv']

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

        nb_neg_curv = _eval_counts['neg_curv'] - neg_curv_0

        if res.success:
            converged = True
            lam = lam_try
            u = np.asarray(res.x)
            break

        root_print(f'  {res.message}  ->  cutback')
        n_cuts += 1
        dlam *= 0.5
        if n_cuts > MAX_CUTS:
            stop_reason = (f'increment {inc} did not converge after '
                           f'{MAX_CUTS} cutbacks ({res.message})')
            break

    if not converged:
        # restore the last good state so the summary and the plots describe it
        lam = lam_backup
        u = np.array(u_backup, copy=True)
        lam_current = lam
        _set_F_from(u)
        root_print('=' * 70)
        root_print(f'STOP: {stop_reason}')
        break

    # ---- increment accepted: refresh the constitutive state ----------------
    sum_hessp += res.nb_hessp
    sum_precond += res.get('nb_precond', 0)
    sum_outer += res.nit
    sum_neg_curv += nb_neg_curv

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
    root_print(f'  min(det F) {J_min:10.3e} | delta_end {delta_end:9.3e} '
               f'| neg. curvature exits {nb_neg_curv:4d}'
               + ('   <-- indefinite Hessian' if nb_neg_curv else ''))
    root_print(f'  P_xx {stress_field.s[0, 0].mean():10.3e} | '
               f'P_yx {stress_field.s[1, 0].mean():10.3e}')

    total_macro_gradient = lam * H_macro
    displacement_fluctuation_field.s[...] = u

    _info['mean_stress_field'].append(float(stress_field.s[1, 0].mean()))
    _info['mean_stress_xx'].append(float(stress_field.s[0, 0].mean()))
    _info['total_macro_gradient'].append(total_macro_gradient)
    _info['nb_hessp'].append(res.nb_hessp)
    _info['nb_precond'].append(res.get('nb_precond', 0))
    _info['nb_outer'].append(res.nit)
    _info['nb_neg_curv'].append(nb_neg_curv)
    _info['min_J'].append(J_min)
    _info['energy'].append(res.fun)
    _info['delta_end'].append(delta_end)
    _info['grad_inf'].append(res.max_grad)
    _info['load_scale'].append((lam - lam_backup) / dlam_nominal)
    _info['lam'].append(lam)
    _info['nb_cuts'].append(n_cuts)

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
            macro_gradient_ij=total_macro_gradient,
            displacement_fluctuation=displacement_fluctuation_field)
        visualization_utils.plot_field_on_grid(
            coordinates_for_plot=x_plot_ixyz,
            field_to_plot=phase_field.s[0, 0],
            name=f'phase, load increment {inc}   lam = {lam:.4f}')

# `stop_reason` still holds its initial value unless a break set one, so an
# unchanged value with lam short of 1 means the loop ran out of increments.
if lam >= 1.0 - 1e-12:
    stop_reason = 'load path completed'
elif stop_reason == 'load path completed':
    stop_reason = f'increment cap MAX_INCREMENTS = {MAX_INCREMENTS} reached'
    root_print('=' * 70)
    root_print(f'STOP: {stop_reason}')

# ============================================================================
# response curves
# ============================================================================
if rank == 0 and plot_figures and len(_info['mean_stress_field']) > 1:
    F_plot = np.asarray(_info['total_macro_gradient'])[..., 1, 0]
    P_plot = np.asarray(_info['mean_stress_field'])

    fig, ax = plt.subplots(1, 4, figsize=(19, 4))

    ax[0].plot(F_plot, P_plot, '-o', ms=3, color='k')
    ax[0].set_xlabel(r'$\bar{F}_{10}$')
    ax[0].set_ylabel(r'$\bar{P}_{10}$')
    ax[0].grid(alpha=.3)

    ax[1].semilogy(_info['lam'], np.asarray(_info['min_J']), '-o', ms=3,
                   color='C3')
    ax[1].set_xlabel(r'$\lambda$')
    ax[1].set_ylabel(r'$\min \det F$')
    ax[1].grid(alpha=.3)

    ax[2].plot(_info['lam'], np.asarray(_info['load_scale']), '-o', ms=3,
               color='C0')
    ax[2].set_xlabel(r'$\lambda$')
    ax[2].set_ylabel('load step / nominal')
    ax[2].grid(alpha=.3)

    ax[3].plot(_info['lam'], np.asarray(_info['nb_neg_curv']), '-o', ms=3,
               color='C2')
    ax[3].set_xlabel(r'$\lambda$')
    ax[3].set_ylabel('negative-curvature CG exits')
    ax[3].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(figure_folder_path + 'response.png', dpi=150)
    plt.show()

# ============================================================================
# summary
# ============================================================================
elapsed_time = time.time() - start_time
_info.update(sum_hessp=sum_hessp, sum_precond=sum_precond,
             sum_outer=sum_outer, sum_neg_curv=sum_neg_curv,
             elapsed_time=elapsed_time, nb_fun_evals=_eval_counts['fun'],
             nb_inadmissible=_eval_counts['inadmissible'],
             lam_applied=lam, stop_reason=stop_reason)

if rank == 0:
    print('=' * 70)
    print(f'geometry           : {geometry_name}')
    print(f'element_type       : {element_type}')
    print(f'number_of_pixels   : {number_of_pixels}')
    print(f'preconditioned     : {use_precond}')
    print(f'tr_mode            : {args.tr_mode}')
    print('inner_tol          : ' + ('forcing' if args.inner_tol is None
                                     else f'{args.inner_tol:g}'))
    print(f'imperfection       : {args.imperfection:g}')
    print(f'reached            : lam = {lam:.6f} of 1.0   ({100 * lam:.1f}%)')
    print(f'stop reason        : {stop_reason}')
    if _info['min_J']:
        print(f'final min J        : {_info["min_J"][-1]:.6e}')
    print(f'increments         : {len(_info["lam"])}')
    print(f'Total hessp        : {sum_hessp}')
    print(f'Total M^-1 applies : {sum_precond}')
    print(f'Total outer its    : {sum_outer}')
    print(f'Total fun evals    : {_eval_counts["fun"]}')
    print(f'  inadmissible     : {_eval_counts["inadmissible"]}')
    print(f'Neg. curvature CG exits : {sum_neg_curv}'
          + ('   (indefinite Hessian: buckling)' if sum_neg_curv else ''))
    print(f'Elapsed time       : {elapsed_time:.2f} s '
          f'({elapsed_time / 60:.2f} min)')
    if sum_hessp:
        print(f'hessp / outer it   : {sum_hessp / max(sum_outer, 1):.1f}')
        print(f'ms / hessp         : {1e3 * elapsed_time / sum_hessp:.3f}')
    np.savez(data_folder_path + 'info_log_final.npz', **_info)
