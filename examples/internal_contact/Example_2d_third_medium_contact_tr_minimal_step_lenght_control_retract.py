"""
2D third-medium contact, neo-Hookean finite strain, solved as an ENERGY
MINIMIZATION with NuMPI's trust-region Newton-CG.

    Pi(u) = int_Omega W(F) dx
          + k_r/2 int_Omega ( Hu : Hu - (1/dim) Lu . Lu ) dx

    grad Pi = G^T P   + R u
    hess Pi = G^T C G + R        with  R = k_r (H^T H - L^T L/dim)

The inner Steihaug CG is preconditioned with the Green operator
M = G^T C_ref G + R, built once from a fixed reference material.

PER-PIXEL LOAD THROTTLING
-------------------------
The imposed deformation is accumulated in a FIELD, H_imposed, rather than
carried by a scalar lam:

    F = I + H_imposed(q,x,y) + grad(u~)

Each increment offers every quadrature point the same dH = dlam * H_macro,
then scales it down point by point wherever that would drive det F to zero:

    theta(q,x,y) = min(1, PIXEL_SAFETY * s(q,x,y) / dlam)
    H_imposed   += theta * dlam * H_macro

with s the exact largest admissible scale at that point (root of a quadratic
in 2D).  Points with room take the full step; only the ones about to invert
are held back.

Consequence, as intended: H_imposed is no longer uniform, so <F> drifts away
from I + lam*H_macro and the imposed part is generally not a gradient.  Both
are measured and printed every increment -- the mean imposed driven
component against the
nominal, and the throttled-point count.
"""

import os
import sys
import inspect

import numpy as np
from mpi4py import MPI
from matplotlib import pyplot as plt

from NuMPI.Optimization import tr_newton_bounded

# `precond` exists only in a hand-patched BoundedTRNewtonCG.py; NuMPI reports
# version 0.15.1 either way, so fail here instead of with a bare TypeError
# several hundred lines down.
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
# problem setup
# ============================================================================
nnn = 32
ninc = 100

# plotting: deformed mesh every `plot_every` increments (0 = only the final
# response curves).  Serial runs only -- the fields are distributed under MPI.
plot_every = 1

# per-pixel load throttling: fraction of each point's own admissible scale
PIXEL_SAFETY = 0.5

number_of_pixels = (nnn, nnn)
domain_size = [1, 1]
dim = len(domain_size)
problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'bilinear_rectangle'
formulation = 'finite_strain'

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
# material parameters
# ============================================================================
# matrix: soft neo-Hookean
E_matrix = 100.0
nu_matrix = 0.3
lam_matrix = E_matrix * nu_matrix / ((1 + nu_matrix) * (1 - 2 * nu_matrix))
mu_matrix = E_matrix / (2 * (1 + nu_matrix))
K, G = material_models.get_bulk_and_shear_modulus(E_matrix, nu_matrix)

# third medium ("void"): same neo-Hookean, k_v times softer
k_v = 1e-5
E_void = k_v * E_matrix
nu_void = nu_matrix
lam_void = E_void * nu_void / ((1 + nu_void) * (1 - 2 * nu_void))
mu_void = E_void / (2 * (1 + nu_void))

# HuHu-LuLu regularization
alpha = 1e-6
k_r = alpha * domain_size[0] ** 2 * (K + G * 4 / 3)
inv_tr_I = 1.0 / dim

# reference material for the Green preconditioner
_i = np.eye(dim)
II = np.einsum('ij,kl->ijkl', _i, _i)
I4rt = np.einsum('ik,jl->ijkl', _i, _i)
I4s = 0.5 * (I4rt + np.einsum('il,jk->ijkl', _i, _i))
ref_mat = lam_matrix * II + 2.0 * mu_matrix * I4s

# ============================================================================
# geometry
# ============================================================================
geometry_name = 'contact_test_geometry_1'

phase_field = discretization.get_scalar_field(name='phase_field')
phase_field.s[0, 0] = microstructure_library.get_geometry(
    nb_voxels=discretization.nb_of_pixels,
    microstructure_name=geometry_name,
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

material = material_models.NeoHookean(
    discretization=discretization,
    lam_1qxyz=lam_field,
    mu_1qxyz=mu_field,
    name='neo_hookean_two_phase'
)

# ============================================================================
# fields
#
# CONVENTION: macro_gradient_full_field holds H_macro WITHOUT the identity,
# as the per-unit-dlam load DIRECTION.  imposed_field holds what has actually
# been applied so far, also without the identity; `set_F` adds I.
# ============================================================================
macro_gradient_full_field = discretization.get_gradient_size_field(
    name='macro_gradient_full_field')
imposed_field = discretization.get_gradient_size_field(name='H_imposed')
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

# preconditioner scratch, kept separate from the hessp path
r_field = discretization.get_unknown_size_field(name='r_field')
Pv_field = discretization.get_unknown_size_field(name='Pv_field')

hess_u_ijkqxyz = discretization.get_displacement_hessian_size_field(name='hess_u')
HtH_field = discretization.get_unknown_size_field(name='HtH')
lap_u_inxyz = discretization.get_displacement_laplacian_at_quad_field(name='lap_u')
LtL_field = discretization.get_displacement_sized_field(name='LtL')

displacement_fluctuation_field = discretization.get_unknown_size_field(
    name='u_fluc')

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


def global_mean(a):
    a = np.asarray(a)
    return global_sum(np.sum(a)) / max(global_sum(float(a.size)), 1.0)


# ============================================================================
# deformation gradient
# ============================================================================
def min_J(F_in):
    tensor_operations.det2(F_in, J_1qxyz)
    return global_min(np.min(J_1qxyz.s))


def set_F(u_field, out):
    """F = I + H_imposed + grad(u~).

    H_imposed is a FIELD now, accumulated by the throttled load step, so it
    is read as-is rather than rebuilt from a scalar load parameter.
    """
    discretization.fft.communicate_ghosts(u_field)
    discretization.apply_gradient_operator_mugrid(u_inxyz=u_field,
                                                  grad_u_ijqxyz=grad_u_field)
    out.s[...] = grad_u_field.s + imposed_field.s
    for d in range(dim):
        out.s[d, d] += 1.0


def admissible_scale_field(F, D):
    """Per-quadrature-point largest s > 0 keeping det(F + s*D) > 0.

    Takes raw arrays, so the caller can pass a SIGNED direction: on
    retraction the step is along -H_macro, and the room available in that
    direction is not the room available along +H_macro.

    Returns the LOCAL field, unreduced: entry (q, x, y) is how far that one
    point can travel along D before its own det F reaches zero, +inf if it
    never does, 0 if it is already inverted.

    In 2D this is exact rather than a bound: for a 2x2 F,

        det(F + s D) = det F + s*b + s^2 * det D
        b = F00 D11 + D00 F11 - F01 D10 - D01 F10

    so the first crossing of zero is the smallest positive root of a
    quadratic and needs no bisection.
    """
    c = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    b = (F[0, 0] * D[1, 1] + D[0, 0] * F[1, 1]
         - F[0, 1] * D[1, 0] - D[0, 1] * F[1, 0])
    a = D[0, 0] * D[1, 1] - D[0, 1] * D[1, 0]

    with np.errstate(divide='ignore', invalid='ignore'):
        # degenerate (rank-one or uniform) dF: det dF = 0, det F is linear
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

    return np.where(c <= 0.0, 0.0, s_qxy)


def apply_throttled_load(dlam):
    """H_imposed += theta * dlam * H_macro, point by point.

    `dlam` is SIGNED: positive while loading, negative while retracting.
    The admissible scale is measured along the direction actually being
    taken, sign(dlam) * H_macro, and theta scales the magnitude |dlam|.

    theta = 1 wherever the point can take the whole increment, and
    PIXEL_SAFETY * s / |dlam| wherever it cannot.  Returns
    (n_throttled, theta_min) for reporting.
    """
    direction = np.sign(dlam) * macro_gradient_full_field.s
    s_qxy = admissible_scale_field(total_strain_field.s, direction)

    with np.errstate(divide='ignore', invalid='ignore'):
        theta = np.minimum(1.0, PIXEL_SAFETY * s_qxy / abs(dlam))
    theta = np.where(np.isfinite(theta), theta, 1.0)   # s = inf -> full step

    # broadcast (q,x,y) over the (i,j) tensor axes of the gradient field
    imposed_field.s[...] += (theta[None, None, ...] * dlam
                             * macro_gradient_full_field.s)

    n_throttled = int(global_sum(float(np.count_nonzero(theta < 1.0))))
    return n_throttled, global_min(np.min(theta))


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
# Green preconditioner:  M = G^T C_ref G + R,  applied as M^-1
#
# Built ONCE from the fixed reference material.  With a preconditioner the
# trust region is measured in the M-norm, so M must not change mid-run or the
# meaning of the radius changes with it.
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
# The load is read from the module-level `imposed_field`, which the increment
# loop advances before each solve.
# ============================================================================

# Large but FINITE: np.inf would make ared = inf - inf = nan, and a nan rho is
# never rejected *and* never shrinks the radius, so the solver spins to
# maxiter.  A finite penalty gives rho << 0, hence rejection and delta/4.
INADMISSIBLE_ENERGY = 1e30


def _set_F_from(x):
    x_field.s[...] = x
    set_F(x_field, total_strain_field)
    return min_J(total_strain_field)


def fun_grad(x):
    """(Pi, grad Pi), globally reduced."""
    J_min = _set_F_from(x)
    if not np.isfinite(J_min) or J_min <= 0.0:
        return INADMISSIBLE_ENERGY, np.zeros(u_shape)

    material.get_energy_density(total_strain_field, energy_field)
    Pi = global_sum(np.sum(_qw * energy_field.s[0, 0]))
    if not np.isfinite(Pi):
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
    """B v = (G^T C G + R) v, with C evaluated at x."""
    _set_F_from(x)
    material.get_algorithmic_tangent(total_strain_field, tangent_field)

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
# macroscopic loading:  target F_bar = I + lam * H_macro,  lam: 0 -> 1
# ============================================================================
H_macro = np.zeros((dim, dim))
H_macro[0, 0] = -0.3

# The component the load path drives, taken from H_macro itself, so the
# reporting follows whatever is set above instead of assuming shear.  With
# several non-zero entries this picks the largest; the others still load,
# they are just not the one named in the log.
load_ij = np.unravel_index(np.argmax(np.abs(H_macro)), H_macro.shape)
load_lbl = f'{load_ij[0]}{load_ij[1]}'
H_drv = H_macro[load_ij]
if H_drv == 0.0:
    raise ValueError('H_macro is identically zero: nothing to load')

discretization.get_macro_gradient_field_mugrid(
    macro_gradient_ij=H_macro,
    macro_gradient_field_ijqxyz=macro_gradient_full_field
)

# Load path: a list of waypoints in units of H_macro, and how many
# increments to take on each leg.  Any path works -- monotonic, cycles,
# partial unloads, overshoot past 1, negative lam:
#
#   [0, 1, 0]           load and retract          (needs 2 leg counts)
#   [0, 1, 0.3, 1]      load, partial unload, reload
#   [0, -0.5, 1.5]      reverse first, then push past the nominal end
#
# Legs may have different resolution; `per_leg` is broadcast if a single
# integer is given.
waypoints = [0.0, 1.0, 0.0]
per_leg = ninc

per_leg = ([per_leg] * (len(waypoints) - 1) if np.isscalar(per_leg)
           else list(per_leg))
if len(per_leg) != len(waypoints) - 1:
    raise ValueError(f'{len(waypoints)} waypoints need '
                     f'{len(waypoints) - 1} leg counts, got {len(per_leg)}')

schedule = []
leg_of_inc = []          # which leg each increment belongs to, for plotting
for leg, (a, b, n) in enumerate(zip(waypoints[:-1], waypoints[1:], per_leg)):
    if n < 1:
        raise ValueError(f'leg {leg} has {n} increments')
    schedule += [(b - a) / n] * n
    leg_of_inc += [leg] * n

leg_bounds = np.cumsum(per_leg)[:-1]     # increment numbers where legs change
dlam_max = max(abs(d) for d in schedule)

# ============================================================================
# incremental loading
# ============================================================================
u = np.zeros(u_shape)
imposed_field.s[...] = 0.0

hist_lam = []
hist_F_target = []
hist_F_actual = []
hist_P_drv = []
hist_P_all = []           # full mean P, so nothing is lost by naming one
hist_minJ = []
hist_energy = []
hist_throttled = []
hist_branch = []          # +1 loading, -1 retracting

serial = discretization.communicator.size == 1

root_print('=' * 70)
root_print(f'geometry     : {geometry_name}')
root_print(f'element_type : {element_type}')
root_print(f'pixels       : {number_of_pixels}')
root_print('load path    : lam ' + ' -> '.join(f'{w:g}' for w in waypoints)
           + f'   (driven F_{load_lbl}, scale {H_drv:+.3f})')
root_print(f'               legs {per_leg}   {len(schedule)} increments total')
root_print(f'PIXEL_SAFETY : {PIXEL_SAFETY}')

lam_nominal = waypoints[0]
if lam_nominal != 0.0:
    raise ValueError('the path must start at lam = 0: the initial state is '
                     'u = 0 with no imposed deformation')

for inc, dlam in enumerate(schedule, start=1):
    lam_nominal += dlam
    leg = leg_of_inc[inc - 1]
    branch = f'leg {leg} {"+" if dlam > 0 else "-"}'

    # ---- throttled load step ----------------------------------------------
    # F at the converged previous state sets each point's own room to move,
    # measured along the direction this step actually goes.
    _set_F_from(u)
    n_throttled, theta_min = apply_throttled_load(dlam)

    J_pred = _set_F_from(u)

    res = tr_newton_bounded(
        fun=fun_grad,
        x0=u,
        hessp=hessp,
        precond=precond,
        jac=True,
        gtol=1e-4,
        maxiter=500,
        inner_tol=1e-6,
        inner_maxiter=None,
        comm=comm,
    )

    u = np.asarray(res.x)

    J_min = _set_F_from(u)
    material.get_stress(total_strain_field, stress_field)

    F_target = lam_nominal * H_drv
    F_actual = global_mean(imposed_field.s[load_ij])
    P_mean = np.array([[global_mean(stress_field.s[i, j])
                        for j in range(dim)] for i in range(dim)])

    root_print('-' * 70)
    root_print(f'increment {inc:4d}  [{branch}]  lam_nominal = '
               f'{lam_nominal:.4f}')
    root_print(f'  throttled points {n_throttled:5d}   theta_min '
               f'{theta_min:8.4f}   min J after load {J_pred:10.3e}')
    root_print(f'  imposed F_{load_lbl}: target {F_target:+.5f}  '
               f'actual (mean) {F_actual:+.5f}  '
               f'drift {F_actual - F_target:+.3e}')
    root_print(f'  {res.message}')
    root_print(f'  outer its {res.nit:4d} | hessp {res.nb_hessp:6d} '
               f'| |grad|_inf {res.max_grad:10.3e} | Pi {res.fun:12.6e}')
    root_print(f'  min(det F) {J_min:10.3e} | P_{load_lbl} '
               f'{P_mean[load_ij]:10.3e} | mean P '
               + ' '.join(f'{P_mean[i, j]:+.3e}'
                          for i in range(dim) for j in range(dim)))

    hist_lam.append(lam_nominal)
    hist_F_target.append(F_target)
    hist_F_actual.append(F_actual)
    hist_P_drv.append(float(P_mean[load_ij]))
    hist_P_all.append(P_mean)
    hist_minJ.append(J_min)
    hist_energy.append(float(res.fun))
    hist_throttled.append(n_throttled)
    hist_branch.append(1 if dlam > 0 else -1)

    # ---- deformed mesh -----------------------------------------------------
    # NOTE: plotted with the MEAN imposed gradient, since there is no single
    # macro gradient any more.  The picture is therefore indicative.
    if serial and plot_every and inc % plot_every == 0:
        displacement_fluctuation_field.s[...] = u
        H_mean = np.array([[global_mean(imposed_field.s[i, j])
                            for j in range(dim)] for i in range(dim)])
        x_plot_ixyz = visualization_utils.get_deformed_grid_coords_two_dim(
            discretization,
            macro_gradient_ij=H_mean,
            displacement_fluctuation=displacement_fluctuation_field)
        visualization_utils.plot_field_on_grid(
            coordinates_for_plot=x_plot_ixyz,
            field_to_plot=phase_field.s[0, 0],
            name=f'phase, increment {inc} [{branch.strip()}]   '
                 f'lam_nom = {lam_nominal:.4f}')

root_print('=' * 70)
root_print(f'final min J : {min_J(total_strain_field):.6e}')

# ---- did it come back? -----------------------------------------------------
# Only meaningful when the path ends where it started.  The throttle is
# one-way: a point held back on the way out does not catch up, and the
# return leg subtracts the full nominal step, so the imposed field need not
# return to zero even when lam_nominal does.
root_print('-' * 70)
if abs(lam_nominal) < 1e-12:
    root_print('RETURN TO ORIGIN')
    root_print(f'  lam_nominal      {lam_nominal:+.6e}   (target 0)')
    root_print(f'  imposed F_{load_lbl}     {hist_F_actual[-1]:+.6e}   (target 0)')
    root_print(f'  residual Pi      {hist_energy[-1]:.6e}   (target 0)')
    root_print(f'  residual P_{load_lbl}    {hist_P_drv[-1]:+.6e}   (target 0)')
    root_print(f'  residual |u|_inf {global_max(np.max(np.abs(u))):.6e}')
else:
    root_print(f'END OF PATH   lam_nominal {lam_nominal:+.6f}')
    root_print(f'  imposed F_{load_lbl}     {hist_F_actual[-1]:+.6e}  '
               f'(nominal {hist_F_target[-1]:+.6e})')
    root_print(f'  Pi               {hist_energy[-1]:.6e}')
    root_print(f'  P_{load_lbl}             {hist_P_drv[-1]:+.6e}')

# ============================================================================
# response curves
# ============================================================================
if rank == 0 and len(hist_lam) > 1:
    F10a = np.asarray(hist_F_actual)
    P10 = np.asarray(hist_P_drv)
    lamn = np.asarray(hist_lam)
    legs = np.asarray(leg_of_inc)
    incs = np.arange(1, len(hist_minJ) + 1)

    fig, ax = plt.subplots(1, 4, figsize=(19, 4))

    for leg in range(len(per_leg)):
        m = legs == leg
        sgn = '+' if schedule[int(np.argmax(m))] > 0 else '-'
        ax[0].plot(F10a[m], P10[m], '-o', ms=3, color=f'C{leg % 10}',
                   label=f'leg {leg} ({sgn})')
    ax[0].set_xlabel(rf'$\bar{{F}}_{{{load_lbl}}}$ (actual mean)')
    ax[0].set_ylabel(rf'$\bar{{P}}_{{{load_lbl}}}$')
    ax[0].legend(fontsize=8)

    ax[1].plot(incs, F10a - np.asarray(hist_F_target), '-o', ms=3, color='C0')
    ax[1].set_xlabel('increment')
    ax[1].set_ylabel(rf'imposed $F_{{{load_lbl}}}$ drift')

    ax[2].semilogy(incs, hist_minJ, '-o', ms=3, color='C3')
    ax[2].set_xlabel('increment')
    ax[2].set_ylabel(r'$\min \det F$')

    ax[3].plot(incs, hist_throttled, '-o', ms=3, color='C2')
    ax[3].set_xlabel('increment')
    ax[3].set_ylabel('throttled points')

    for a in ax[1:]:
        for b in leg_bounds:
            a.axvline(b + 0.5, color='k', lw=0.8, ls='--')
    for a in ax:
        a.grid(alpha=.3)

    fig.tight_layout()
    plt.show()