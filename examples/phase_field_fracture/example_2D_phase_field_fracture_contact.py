"""
Phase-field fracture with third-medium contact (2D, finite strain)
=================================================================

A solid cell with an S-shaped cavity between two interlocking teeth
(geometry 'contact_fracture_s_gap') is sheared. The cavity is filled with a
very soft "third medium": when the channel between the teeth closes, the
medium is crushed, its energy blows up as det F -> 0 and it transmits the
contact pressure. The load then concentrates at the teeth and the solid
cracks (AT2 phase field). Only the solid can be damaged.

Energy, with F = I + lam H_macro + grad u~ (u~ periodic):

    Pi(u~, d) = int  g_s(d) W_iso(F) + g_v(d, F) W_vol(F)  dx            elastic
              + k_r/2 int  Hu~ : Hu~ - (1/dim) Lu~ . Lu~  dx              HuHu-LuLu
              + Gc/2 int  d^2/l + l |grad d|^2  dx                        AT2 crack

    W_iso = mu/2 (J^-2/3 (I1 + 1) - 3)        isochoric (plane strain), I1 = F:F
    W_vol = kappa/2 ln(J)^2                   volumetric
    g(d)  = (1 - d)^2 + k                     degradation
    g_s   = g(d) in the solid, 1 in the third medium
    g_v   = g_s for J > 1 (expansion), 1 for J <= 1 (compression)

The split keeps compression undamaged, so crack faces and the crushed medium
still resist interpenetration. Only the solid can be damaged: the damage problem
is assembled over the solid elements only (natural boundary grad d . n = 0 at the
cavity surface), nodes touching only the medium stay d = 0
(damage_in_medium = True solves it on the whole cell). The third medium has the same law with
mu, kappa scaled by k_v. The second-gradient term (Faltus et al.) stabilizes
the collapsing medium. With Q2 elements ('biquadratic_rectangle') both its
Hessian and its Laplacian part act; with the default Q1 elements the
Laplacian of a bilinear field vanishes and only the mixed second derivative
is penalized.

Algorithm, per load increment lam -> lam + dlam (staggered):

    1. u~: minimize Pi(., d) with NuMPI's trust-region Newton-CG; the inner
       CG is preconditioned with a Green operator of the undamaged solid plus
       the regularization. det F <= 0 is rejected (large finite energy).
    2. H = max(H_{n-1}, psi^+(F)),  psi^+ = W_iso + <J > 1> W_vol  (solid only)
    3. d: [Gc l B^T W B + N^T W (Gc/l + 2H) N] d = N^T W 2H  (linear, CG with a
       Green(-Jacobi) preconditioner, as in example_2D_phase_field_fracture_AT2)
    repeat 1-3 until max|d - d_old| < staggered_tol, then H_{n-1} <- H.

The load step is adaptive. Each increment starts from the linear
extrapolation of the last two converged fluctuations (or from the last one if
that is not admissible); the step is halved when no admissible start exists
(det F <= 0 somewhere), the trust-region solve fails or the staggered loop
does not converge within max_staggered_iterations, and grows back after a
success.

Math and scheme: docs/examples/phase_field_fracture_contact.md.
Output: NetCDF via muFFTTO.io_utils (damage, u~, H, det F and phase, plus the
increment history); plot it with plot_phase_field_fracture_contact_output.py.
Needs the patched NuMPI with `precond` in tr_newton_bounded (see
docs/examples/internal_contact.md).
"""
import sys
import os
import time
import inspect

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np  # noqa: E402
from mpi4py import MPI  # noqa: E402
from muGrid import Solvers  # noqa: E402
from NuMPI.Optimization import tr_newton_bounded  # noqa: E402

from muFFTTO import domain  # noqa: E402
from muFFTTO import microstructure_library  # noqa: E402
from muFFTTO import io_utils  # noqa: E402
from muFFTTO import tensor_operations  # noqa: E402

if 'precond' not in inspect.signature(tr_newton_bounded).parameters:
    raise RuntimeError('this example needs the patched NuMPI with `precond` in tr_newton_bounded')

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
nnn = 32                                   # pixels per direction
# where the fields live: 'cpu' or 'gpu' (GPU needs muGrid built with CUDA + cupy)
device = 'cpu'
number_of_pixels = (nnn, nnn)
domain_size = [1.0, 1.0]
# 'bilinear_rectangle' (Q1) or 'biquadratic_rectangle' (Q2). With Q1 on pixels the
# discrete Laplacian vanishes, so only the HuHu part of the regularization acts.
element_type = 'bilinear_rectangle'
geometry_name = 'contact_fracture_s_gap'    # any name from muFFTTO.geometry.available()
geometry_parameters = {}                   # passed to the geometry, e.g. {'notch_length': 0.1}

# load: simple shear F_bar = I + lam H_macro, lam: 0 -> 1 (H_10 > 0 closes the channel)
H_macro = np.array([[0.0, 0.0],
                    [0.3, 0.0]])
nb_increments = 50                         # nominal increments (largest step 1/nb_increments)
min_step = 1.0 / nb_increments / 64        # smallest step after halving

# solid (neo-Hookean) and third medium
E_solid, nu_solid = 100.0, 0.3
k_v = 1e-5                                 # third-medium stiffness ratio
alpha = 1e-6                               # HuHu-LuLu regularization strength

# phase-field fracture (AT2)
Gc = 0.1                                   # fracture toughness
# fixed in physical units (not in pixels), so a finer grid refines the same problem:
# 1 px at 32^2, 2 px at 64^2, 4 px at 128^2
length_scale = domain_size[0] / 32
k_residual = 1e-4                          # residual stiffness in g(d)
staggered_tol = 1e-3                       # max |d - d_old|
max_staggered_iterations = 300            # not converged by then: the increment is retried with half the step
damage_preconditioner = 'Green_Jacobi'     # or 'Green'
# False: the damage problem lives on the solid only (natural boundary grad d . n = 0 at the
# solid / third-medium interface; nodes that touch only the medium are pinned to d = 0).
# True: solved on the whole cell, d diffuses into the medium (old behaviour, for comparison)
damage_in_medium = False
# mechanics (inner CG of the trust region): 'Green' = fixed Green operator of the undamaged
# solid; 'Green_Jacobi' = D^-1/2 G D^-1/2 with D = diag(B^T A B + R) of the current tangent,
# recomputed at the start of every trust-region solve (M stays fixed within a solve)
mechanics_preconditioner = 'Green'

# trust-region Newton-CG. The solver works on the energy divided by the force scale
# s = E_solid h^(dim-1) (h = pixel size): a nodal force for a unit strain error, so the
# tolerances below do not change with the resolution (a nodal force ~ stress x h^(dim-1)).
# Scaling the energy, gradient and Hessian by a constant leaves all steps unchanged.
SOLVER_GTOL = 3e-6                         # max |nodal force| / s  (~ residual stress / E)
SOLVER_MAXITER = 500
# relative tolerance of the inner CG; None = inexact Newton (NuMPI forcing
# min(0.5, sqrt(max|g| / s)): loose far from the solution, tight close to it)
SOLVER_INNER_TOL = None
INADMISSIBLE_ENERGY = 1e30                 # finite, so the trust region shrinks (inf - inf = nan)

# Output: every run gets its own, new folder (earlier runs are never overwritten)
#   <output_root>/<geometry>_<load>_<element>_N<nnn>_Gc<Gc>_<date-time>/
# with contact_fracture.nc (one frame per accepted increment), run.log and, at
# the end, response.png and contact_fracture.mp4. output_root = None: no output.
output_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'exp_data', 'phase_field_fracture_contact')
make_figures = True                        # response figure and MP4 at the end (needs netCDF4)
# Restart: contact_fracture.nc of an earlier run of this script with the same grid, geometry,
# element, material and load. The run continues from its last frame (u~, d, H, lam, dlam) in a
# new folder; the earlier frames are copied into the new file first, so that it holds the
# whole load path. The file is only read. None: start at lam = 0.
restart_file = None

# ----------------------------------------------------------------------------
# Discretizations: vector (u~) and scalar (d) on the same grid and element
# ----------------------------------------------------------------------------
disc_u = domain.Discretization(cell=domain.PeriodicUnitCell(domain_size=domain_size, problem_type='elasticity'),
                               nb_of_pixels_global=number_of_pixels, discretization_type='finite_element',
                               element_type=element_type, device=device)
disc_d = domain.Discretization(cell=domain.PeriodicUnitCell(domain_size=domain_size, problem_type='conductivity'),
                               nb_of_pixels_global=number_of_pixels, discretization_type='finite_element',
                               element_type=element_type, device=device)

# field data (.s) are numpy arrays on the CPU and cupy arrays on the GPU
if device == 'gpu':
    import cupy as xp
else:
    xp = np
to_host = tensor_operations.to_host
comm = MPI.COMM_WORLD
rank = comm.rank
dim = 2
pixel_size = (np.prod(domain_size) / np.prod(number_of_pixels)) ** (1.0 / dim)


log_file = None                            # run.log in the run folder, set below


def root_print(*args):
    if rank == 0:
        print(*args, flush=True)
        if log_file is not None:
            with open(log_file, 'a') as log:
                print(*args, file=log)


def global_sum(a):
    return comm.allreduce(float(a), op=MPI.SUM)


def global_min(a):
    return comm.allreduce(float(a), op=MPI.MIN)


def global_max(a):
    return comm.allreduce(float(a), op=MPI.MAX)


def global_mean(a):
    return global_sum(a.sum()) / max(global_sum(float(a.size)), 1.0)


def dot_global(a, b):
    return global_sum((a.ravel() * b.ravel()).sum())


# ----------------------------------------------------------------------------
# Geometry and material parameters at the quadrature points
# ----------------------------------------------------------------------------
phase_pixel = microstructure_library.get_geometry(nb_voxels=disc_u.nb_of_pixels,
                                                  microstructure_name=geometry_name,
                                                  coordinates=disc_u.fft.coords,
                                                  **geometry_parameters)           # 1 solid, 0 medium
nb_q = disc_u.nb_quad_points_per_pixel
solid_q = xp.broadcast_to(xp.asarray(phase_pixel > 0), (nb_q, *phase_pixel.shape))  # [q, x, y]

mu_solid = E_solid / (2 * (1 + nu_solid))
lam_solid = E_solid * nu_solid / ((1 + nu_solid) * (1 - 2 * nu_solid))
kappa_solid = lam_solid + 2.0 * mu_solid / 3.0
mu_q = xp.where(solid_q, mu_solid, k_v * mu_solid)
kappa_q = xp.where(solid_q, kappa_solid, k_v * kappa_solid)
k_r = alpha * domain_size[0] ** 2 * (kappa_solid + 4.0 * mu_solid / 3.0)
quad_weights = xp.asarray(disc_u.quadrature_weights).reshape((-1,) + (1,) * dim)

root_print(f'mechanics preconditioner {mechanics_preconditioner}, damage preconditioner {damage_preconditioner}')
root_print(f'grid {number_of_pixels}, {element_type}, ranks {comm.size}, solid fraction '
           f'{global_mean(phase_pixel):.3f}, l = {length_scale:.4f}, Gc = {Gc}, k_r = {k_r:.2e}')


def split_neo_hookean(F, mu, kappa, g, tangent=False):
    """W, psi^+, P = dW/dF (and A = dP/dF) of the split neo-Hookean law, pointwise.

    F: [2, 2, ...]; mu, kappa, g: [...]. See the module docstring.
    """
    xp_ = tensor_operations.get_array_module(F)   # numpy or cupy, where F lives
    J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    FiT = xp_.stack([xp_.stack([F[1, 1], -F[1, 0]]), xp_.stack([-F[0, 1], F[0, 0]])]) / J       # F^-T
    lnJ = xp_.log(J)
    I1 = xp_.einsum('ij...,ij...->...', F, F)
    a = J ** (-2.0 / 3.0)
    b = (I1 + 1.0) / 3.0
    W_iso = 0.5 * mu * (a * (I1 + 1.0) - 3.0)
    W_vol = 0.5 * kappa * lnJ ** 2
    g_vol = xp_.where(lnJ > 0, g, 1.0)
    psi_plus = W_iso + xp_.where(lnJ > 0, W_vol, 0.0)
    W = g * W_iso + g_vol * W_vol
    P = g * mu * a * (F - b * FiT) + g_vol * kappa * lnJ * FiT
    if not tangent:
        return W, psi_plus, P
    identity = xp_.eye(2)
    dd = xp_.einsum('ik,jl->ijkl', identity, identity).reshape((2, 2, 2, 2) + (1,) * (F.ndim - 2))
    FiT_il_FiT_kj = xp_.einsum('il...,kj...->ijkl...', FiT, FiT)
    A_iso = (-2.0 / 3.0 * xp_.einsum('kl...,ij...->ijkl...', FiT, F - b * FiT) + dd
             - 2.0 / 3.0 * xp_.einsum('kl...,ij...->ijkl...', F, FiT) + b * FiT_il_FiT_kj)
    A_vol = xp_.einsum('ij...,kl...->ijkl...', FiT, FiT) - lnJ * FiT_il_FiT_kj
    return W, psi_plus, P, g * mu * a * A_iso + g_vol * kappa * A_vol


# ----------------------------------------------------------------------------
# Fields
# ----------------------------------------------------------------------------
macro_gradient = disc_u.get_gradient_size_field(name='macro_gradient')          # H_macro (without I)
disc_u.get_macro_gradient_field_mugrid(macro_gradient_ij=H_macro, macro_gradient_field_ijqxyz=macro_gradient)
grad_u = disc_u.get_displacement_gradient_sized_field(name='grad_u')
F_field = disc_u.get_displacement_gradient_sized_field(name='F')
stress_field = disc_u.get_displacement_gradient_sized_field(name='P')
tangent_field = disc_u.get_material_data_size_field_mugrid(name='A')
u_fluc = disc_u.get_unknown_size_field(name='u_fluc')
x_field = disc_u.get_unknown_size_field(name='x_field')
grad_field = disc_u.get_unknown_size_field(name='grad_field')
Ru_field = disc_u.get_unknown_size_field(name='Ru_field')
v_field = disc_u.get_unknown_size_field(name='v_field')
Bv_field = disc_u.get_unknown_size_field(name='Bv_field')
r_field = disc_u.get_unknown_size_field(name='r_field')
Pv_field = disc_u.get_unknown_size_field(name='Pv_field')
hess_u = disc_u.get_displacement_hessian_size_field(name='hess_u')
HtH_field = disc_u.get_unknown_size_field(name='HtH')
lap_u = disc_u.get_displacement_laplacian_at_quad_field(name='lap_u')
LtL_field = disc_u.get_displacement_sized_field(name='LtL')
u_shape = u_fluc.s.shape

damage = disc_d.get_unknown_size_field(name='damage')                 # d, nodal
damage_old = disc_d.get_unknown_size_field(name='damage_old')
rhs_d = disc_d.get_unknown_size_field(name='rhs_d')
damage_q = disc_d.get_quad_field_scalar(name='damage_q')
history = disc_d.get_quad_field_scalar(name='history')                # H, current iterate
history_n = disc_d.get_quad_field_scalar(name='history_converged')    # H of the last increment
reaction = disc_d.get_quad_field_scalar(name='reaction')              # Gc/l + 2H
grad_d = disc_d.get_gradient_size_field(name='grad_damage')
temp_q = disc_d.get_quad_field_scalar(name='temp_q')
temp_n = disc_d.get_unknown_size_field(name='temp_n')
x_scaled_d = disc_d.get_unknown_size_field(name='x_scaled_d')
# damage domain: weight 1 at solid quadrature points, 0 in the medium; free nodes = nodes of
# at least one solid element (the others get a unit row and stay d = 0)
damage_weight_q = xp.ones(solid_q.shape) if damage_in_medium else solid_q.astype(float)
temp_q.s[0, 0] = damage_weight_q
disc_d.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=temp_q, nodal_field_inxyz=temp_n, apply_weights=True)
damage_free_nodes = temp_n.s > 0                                      # [1, nodes, x, y]

# pixel-level output fields: phase and quadrature-averaged det F
phase_out = disc_u.field_collection.real_field('phase', [1], 'pixel')
phase_out.s[0, 0] = xp.asarray(phase_pixel)
detF_out = disc_u.field_collection.real_field('detF', [1], 'pixel')

g_q = xp.ones(solid_q.shape)               # g_s(d) at the quadrature points
# tangent A(F(x)) of the last state (x, lam) it was built for: x is fixed during a whole
# Steihaug-CG solve, so it is rebuilt once per trust-region iterate, not per Hessian product
tangent_cache = {'valid': False, 'x': None, 'lam': None}
lam_current = 0.0


def update_degradation():
    disc_d.apply_N_operator_mugrid(damage, damage_q)
    g_q[...] = xp.where(solid_q, (1.0 - damage_q.s[0, 0]) ** 2 + k_residual, 1.0)
    tangent_cache['valid'] = False         # the tangent depends on g(d)


# ----------------------------------------------------------------------------
# Elasticity: F, energy, gradient, Hessian-vector product, preconditioner
# ----------------------------------------------------------------------------
def set_F(lam, u_field):
    """F = I + lam H_macro + grad u~; returns the global min det F."""
    disc_u.fft.communicate_ghosts(u_field)
    disc_u.apply_gradient_operator_mugrid(u_inxyz=u_field, grad_u_ijqxyz=grad_u)
    F_field.s[...] = grad_u.s + lam * macro_gradient.s
    for k in range(dim):
        F_field.s[k, k] += 1.0
    F = F_field.s
    return global_min(np.min(F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]))


def add_regularization(x, out):
    """out += R x = k_r (H^T H - L^T L / dim) x; ghosts of x must be current."""
    disc_u.apply_hessian_operator_to_vector_field_mugrid(u_inxyz=x, hess_u_ijkqxyz=hess_u)
    disc_u.apply_hessian_operator_transposed_to_vector_field_mugrid(hess_u_ijkqxyz=hess_u,
                                                                    nodal_field_inxyz=HtH_field,
                                                                    apply_weights=True)
    out.s[...] += k_r * HtH_field.s
    if laplacian_acts:
        disc_u.laplacian.apply(nodal_field=x, quadrature_point_field=lap_u)
        disc_u.laplacian.transpose(quadrature_point_field=lap_u, nodal_field=LtL_field,
                                   weights=disc_u.quadrature_weights)
        out.s[...] -= (k_r / dim) * LtL_field.s


# the discrete Laplacian of a bilinear (Q1) field on axis-aligned pixels is identically zero,
# so the L^T L part is skipped when it has no effect (it acts for Q2)
LtL_field.s[...] = xp.asarray(np.random.default_rng(rank).random(LtL_field.s.shape))
disc_u.fft.communicate_ghosts(LtL_field)
disc_u.laplacian.apply(nodal_field=LtL_field, quadrature_point_field=lap_u)
laplacian_acts = global_max(np.max(np.abs(lap_u.s))) > 0.0


def field_to_optimizer(field):
    """Copy of the field data as a numpy array (host), for NuMPI's optimizer."""
    if device == 'gpu':
        return field.s.get()  # .get() already makes a host copy
    return np.array(field.s, copy=True)


def fun_grad(x):
    """(Pi_elastic + Pi_reg, gradient) at the fluctuation x, damage fixed."""
    x_field.s[...] = xp.asarray(x.reshape(u_shape))
    if not set_F(lam_current, x_field) > 0.0:
        return INADMISSIBLE_ENERGY, np.zeros(u_shape)
    W, _, P = split_neo_hookean(F_field.s, mu_q, kappa_q, g_q)
    Pi = global_sum(np.sum(quad_weights * W))
    if not np.isfinite(Pi):
        return INADMISSIBLE_ENERGY, np.zeros(u_shape)
    Ru_field.s[...] = 0.0
    add_regularization(x_field, Ru_field)
    Pi += 0.5 * dot_global(x_field.s, Ru_field.s)
    stress_field.s[...] = P
    disc_u.fft.communicate_ghosts(stress_field)
    disc_u.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=stress_field, div_u_fnxyz=grad_field,
                                                     apply_weights=True)
    grad_field.s[...] += Ru_field.s
    return Pi, field_to_optimizer(grad_field)


def update_tangent(lam, x):
    """Tangent A(F) at (lam, x) into tangent_field, unless it is already there (cached)."""
    # x comes from the optimizer (numpy) or from a field (numpy or cupy): compare on the host
    x = to_host(x)
    stale = (not tangent_cache['valid'] or tangent_cache['lam'] != lam
             or not np.array_equal(tangent_cache['x'], np.reshape(x, -1)))
    if global_max(float(stale)) == 0.0:    # the same decision on all ranks (set_F is collective)
        return
    x_field.s[...] = xp.asarray(x.reshape(u_shape))
    set_F(lam, x_field)
    tangent_field.s[...] = split_neo_hookean(F_field.s, mu_q, kappa_q, g_q, tangent=True)[3]
    tangent_cache.update(valid=True, lam=lam, x=np.array(x, copy=True).reshape(-1))


def hessp(x, v):
    """(B^T A B + R) v with the tangent A at x (rebuilt only when x, lam or d changed)."""
    update_tangent(lam_current, x)
    v_field.s[...] = xp.asarray(v.reshape(u_shape))
    disc_u.fft.communicate_ghosts(v_field)
    disc_u.apply_system_matrix_mugrid(material_data_field=tangent_field, input_field_inxyz=v_field,
                                      output_field_inxyz=Bv_field, formulation='finite_strain')
    add_regularization(v_field, Bv_field)
    disc_u.fft.communicate_ghosts(Bv_field)
    return field_to_optimizer(Bv_field)


# Green preconditioner of the undamaged solid at F = I, plus the regularization (built once)
reference_tangent = split_neo_hookean(np.eye(2)[:, :, None], np.array([mu_solid]), np.array([kappa_solid]),
                                      np.array([1.0]), tangent=True)[3][..., 0]


def operator_for_preconditioner(input_field_inxyz, output_field_inxyz):
    disc_u.fft.communicate_ghosts(field=input_field_inxyz)
    disc_u.apply_system_matrix_mugrid(material_data_field=reference_tangent, input_field_inxyz=input_field_inxyz,
                                      output_field_inxyz=output_field_inxyz, formulation='finite_strain')
    add_regularization(input_field_inxyz, output_field_inxyz)


green_u = disc_u.get_preconditioner_Green_mugrid(reference_material_data_ijkl=reference_tangent,
                                                 operator=operator_for_preconditioner)


jacobi_u = None                            # (D / d_ref)^-1/2 of the mechanics Hessian (Green_Jacobi only)
# The trust region is measured in the norm of the preconditioner, |s|_M with M = P^-1. For Green,
# M = K_ref: the energy norm of the undamaged solid. For Green_Jacobi, M = D^1/2 K_ref D^1/2 would
# be ~ d_ref times larger; with D normalized by the (uniform) diagonal d_ref of K_ref, M = K_ref
# wherever the tangent is the reference one, so both preconditioners use the same trust-region
# measure. A constant scaling does not change the CG iterates.
reference_diagonal = global_max(np.max(
    1.0 / disc_u.get_preconditioner_Jacobi_mugrid(operator=operator_for_preconditioner).s ** 2))


def update_mechanics_jacobi(lam, u_field):
    """D^-1/2 of B^T A(F) B + R at the state (lam, u_field), for the next trust-region solve."""
    global jacobi_u
    update_tangent(lam, u_field.s)

    def hessian_operator(input_field_inxyz, output_field_inxyz):
        disc_u.fft.communicate_ghosts(input_field_inxyz)
        disc_u.apply_system_matrix_mugrid(material_data_field=tangent_field, input_field_inxyz=input_field_inxyz,
                                          output_field_inxyz=output_field_inxyz, formulation='finite_strain')
        add_regularization(input_field_inxyz, output_field_inxyz)

    jacobi_u = float(np.sqrt(reference_diagonal)) * disc_u.get_preconditioner_Jacobi_mugrid(operator=hessian_operator).s


def precond(v):
    r_field.s[...] = xp.asarray(v.reshape(u_shape))
    if mechanics_preconditioner == 'Green_Jacobi':
        r_field.s[...] *= jacobi_u
    disc_u.fft.communicate_ghosts(r_field)
    disc_u.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_u, input_nodal_field_fnxyz=r_field,
                                       output_nodal_field_fnxyz=Pv_field)
    if mechanics_preconditioner == 'Green_Jacobi':
        Pv_field.s[...] *= jacobi_u
    return field_to_optimizer(Pv_field)


# ----------------------------------------------------------------------------
# Damage: [Gc l B^T W B + N^T W (Gc/l + 2H) N] d = N^T W 2H
# ----------------------------------------------------------------------------
def apply_damage_operator(input_field_inxyz, output_field_inxyz, reaction_q, weight_q=1.0):
    disc_d.apply_gradient_operator_mugrid(u_inxyz=input_field_inxyz, grad_u_ijqxyz=grad_d)
    grad_d.s[...] *= Gc * length_scale * weight_q
    disc_d.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=grad_d, div_u_fnxyz=output_field_inxyz,
                                                     apply_weights=True)
    disc_d.apply_N_operator_mugrid(input_field_inxyz, temp_q)
    temp_q.s[...] *= reaction_q * weight_q
    disc_d.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=temp_q, nodal_field_inxyz=temp_n,
                                              apply_weights=True)
    output_field_inxyz.s[...] += temp_n.s


def K_d(x, Ax):
    """Damage operator on the damage domain; unit rows (identity) for the pinned nodes."""
    apply_damage_operator(x, Ax, reaction.s, damage_weight_q)
    Ax.s[...] = xp.where(damage_free_nodes, Ax.s, x.s)


green_d = disc_d.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=None,
    operator=lambda input_field_inxyz, output_field_inxyz: apply_damage_operator(
        input_field_inxyz, output_field_inxyz, Gc / length_scale),
    invert_zero_mode=True)
jacobi_d = None


def M_d(x, Px):
    if damage_preconditioner == 'Green_Jacobi':
        x_scaled_d.s[...] = jacobi_d.s * x.s
        disc_d.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_d, input_nodal_field_fnxyz=x_scaled_d,
                                           output_nodal_field_fnxyz=Px)
        Px.s[...] *= jacobi_d.s
    else:
        disc_d.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_d, input_nodal_field_fnxyz=x,
                                           output_nodal_field_fnxyz=Px)
    Px.s[...] = xp.where(damage_free_nodes, Px.s, x.s)       # identity on the pinned nodes


class IterationCounter:
    def __init__(self):
        self.count = 0

    def __call__(self, iteration, fields):
        self.count = iteration


def solve_damage():
    """History update and one damage solve; returns (max |d - d_old|, CG iterations)."""
    global jacobi_d
    psi_plus = split_neo_hookean(F_field.s, mu_q, kappa_q, g_q)[1]
    history.s[0, 0] = xp.maximum(history_n.s[0, 0], xp.where(solid_q, psi_plus, 0.0))
    reaction.s[...] = Gc / length_scale + 2.0 * history.s
    temp_q.s[...] = 2.0 * history.s
    disc_d.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=temp_q, nodal_field_inxyz=rhs_d, apply_weights=True)
    rhs_d.s[...] *= damage_free_nodes                          # pinned nodes: d = 0
    damage_old.s[...] = damage.s
    if damage_preconditioner == 'Green_Jacobi':
        jacobi_d = disc_d.get_preconditioner_Jacobi_mugrid(
            operator=lambda input_field_inxyz, output_field_inxyz: K_d(input_field_inxyz, output_field_inxyz))
    counter = IterationCounter()
    Solvers.conjugate_gradients(comm=disc_d.communicator, fc=disc_d.field_collection, hessp=K_d, b=rhs_d,
                                x=damage, prec=M_d, rtol=1e-8, maxiter=5000, callback=counter)
    update_degradation()
    return global_max(np.max(np.abs(damage.s - damage_old.s))), counter.count


# ----------------------------------------------------------------------------
# One load increment (staggered); False if it has to be retried with a smaller step
# ----------------------------------------------------------------------------
force_scale = E_solid * pixel_size ** (dim - 1)    # s: the solver sees Pi / s, g / s, H / s


def scaled_fun_grad(x):
    value, gradient = fun_grad(x)
    if value >= 0.5 * INADMISSIBLE_ENERGY:
        return value, gradient
    return value / force_scale, gradient / force_scale


def scaled_hessp(x, v):
    return hessp(x, v) / force_scale


def solve_increment(lam):
    global lam_current
    lam_current = lam
    if not set_F(lam, u_fluc) > 0.0:
        return False, 'warm start not admissible (det F <= 0)', {}
    stats = {'nb_hessp': 0, 'nb_outer': 0, 'cg_damage': 0, 'staggered_iterations': 0}
    for stagger in range(1, max_staggered_iterations + 1):
        if mechanics_preconditioner == 'Green_Jacobi':
            update_mechanics_jacobi(lam, u_fluc)
        res = tr_newton_bounded(fun=scaled_fun_grad, x0=field_to_optimizer(u_fluc), hessp=scaled_hessp, precond=precond,
                                jac=True,
                                gtol=SOLVER_GTOL, maxiter=SOLVER_MAXITER, inner_tol=SOLVER_INNER_TOL,
                                inner_maxiter=None, comm=comm)
        stats['nb_hessp'] += int(res.nb_hessp)
        stats['nb_outer'] += int(res.nit)
        if not res.success or res.fun >= 0.5 * INADMISSIBLE_ENERGY:
            return False, f'trust region failed in staggered iteration {stagger}: {res.message}', stats
        u_fluc.s[...] = xp.asarray(np.asarray(res.x).reshape(u_shape))
        stats['min_J'] = set_F(lam, u_fluc)
        stats['energy'] = float(res.fun) * force_scale
        change, nb_cg = solve_damage()
        stats['cg_damage'] += nb_cg
        stats['staggered_iterations'] = stagger
        stats['damage_change'] = change
        if change < staggered_tol:
            break
    if change >= staggered_tol:
        return False, (f'staggered loop not converged after {max_staggered_iterations} iterations '
                       f'(max |d - d_old| = {change:.2e})'), stats
    return True, 'converged', stats


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
run_attributes = {'geometry': geometry_name, 'geometry_parameters': repr(geometry_parameters),
                  'element_type': element_type, 'nb_grid_pts': number_of_pixels,
                  'domain_size': np.asarray(domain_size), 'H_macro': H_macro, 'E_solid': E_solid,
                  'nu_solid': nu_solid, 'k_v': k_v, 'alpha': alpha, 'k_r': k_r, 'Gc': Gc,
                  'length_scale': length_scale, 'k_residual': k_residual, 'staggered_tol': staggered_tol,
                  'mechanics_preconditioner': mechanics_preconditioner,
                  'inner_tol': 'forcing' if SOLVER_INNER_TOL is None else SOLVER_INNER_TOL,
                  'gtol': SOLVER_GTOL, 'force_scale': force_scale,
                  'damage_in_medium': int(damage_in_medium),
                  'damage_preconditioner': damage_preconditioner,
                  'nb_increments': nb_increments,
                  'restart_from': '' if restart_file is None else os.path.abspath(restart_file),
                  'note': 'u_fluc and damage are nodal (Q2: sub-point 0 = pixel corner); '
                          'detF and phase are pixel values'}
frame_variables = {'increment': (), 'lam': (), 'dlam': (), 'F10': (), 'P10': (), 'Pxx': (),
                   'mean_P': (dim, dim), 'min_det_F': (), 'energy': (), 'damage_max': (),
                   'staggered_iterations': (), 'nb_outer': (), 'nb_hessp': (), 'cg_damage': (),
                   'converged': (), 'elapsed_time': ()}
output_fields = [damage, u_fluc, history, detF_out, phase_out]
# a restart must continue the same problem: these attributes have to match
restart_keys = ['geometry', 'geometry_parameters', 'element_type', 'nb_grid_pts', 'domain_size', 'H_macro',
                'E_solid', 'nu_solid', 'k_v', 'alpha', 'Gc', 'length_scale', 'k_residual', 'damage_in_medium']
writer = None
output_file = None
if output_root is not None:
    load_tag = '_'.join(f'F{i}{j}{H_macro[i, j]:+g}' for i, j in zip(*np.nonzero(H_macro)))
    run_name = (f'{geometry_name}_{load_tag}_{element_type.split("_")[0]}'
                f'_N{nnn}_Gc{Gc:g}_prec-{mechanics_preconditioner}'
                f'_inner-{"EW" if SOLVER_INNER_TOL is None else f"{SOLVER_INNER_TOL:g}"}'
                f'{"_dmg-all" if damage_in_medium else ""}'
                f'{"" if staggered_tol == 1e-3 else f"_stag{staggered_tol:g}"}'
                f'{"" if restart_file is None else "_restart"}_' + time.strftime('%Y%m%d-%H%M%S'))
    output_folder = None
    if rank == 0:                          # a new folder; never reuse an existing one
        for attempt in range(100):
            candidate = os.path.join(output_root, run_name + (f'_{attempt}' if attempt else ''))
            try:
                os.makedirs(candidate)
                output_folder = candidate
                break
            except FileExistsError:
                continue
    output_folder = comm.bcast(output_folder, root=0)
    output_file = os.path.join(output_folder, 'contact_fracture.nc')
    log_file = os.path.join(output_folder, 'run.log')
    root_print(f'run folder {output_folder}')
    writer = io_utils.FieldWriter(output_file, output_fields, attributes=run_attributes,
                                  frame_variables=frame_variables)

# ----------------------------------------------------------------------------
# Load stepping
# ----------------------------------------------------------------------------
start_time = time.time()
lam, dlam, increment = 0.0, 1.0 / nb_increments, 0
max_step = dlam
u_previous, dlam_previous = u_fluc.s.copy(), None     # last converged state before the current one
if restart_file is not None:
    previous, mismatch = None, None
    if rank == 0:
        import netCDF4
        with netCDF4.Dataset(restart_file, 'r') as dataset:
            stored = {key: dataset.getncattr(key) for key in dataset.ncattrs()}
            previous = {name: np.asarray(dataset.variables[name][:]) for name in frame_variables}
        mismatch = [key for key in restart_keys
                    if (str(stored.get(key)) != str(run_attributes[key]) if isinstance(run_attributes[key], str)
                        else not np.allclose(np.ravel(stored.get(key, np.nan)), np.ravel(run_attributes[key])))]
    previous, mismatch = comm.bcast((previous, mismatch), root=0)
    if mismatch:
        raise ValueError(f'{restart_file} is a different problem: {mismatch} differ')
    nb_previous = len(previous['lam'])
    for frame in range(nb_previous):         # copy the earlier load path into the new file
        io_utils.load_fields(restart_file, output_fields, frame=frame)
        if frame == nb_previous - 2:
            u_previous = u_fluc.s.copy()
        if writer is not None:
            writer.write(**{name: previous[name][frame] for name in frame_variables})
    # the fields now hold the last frame; H of that frame is the converged history
    history_n.s[...] = history.s
    lam, increment = float(previous['lam'][-1]), int(previous['increment'][-1])
    dlam_previous = float(previous['dlam'][-1])
    dlam = min(2.0 * dlam_previous, max_step)
    start_time -= float(previous['elapsed_time'][-1])   # elapsed_time continues
    root_print(f'restart from {restart_file}: frame {nb_previous - 1}, increment {increment}, lam {lam:.4f}')
update_degradation()
while lam < 1.0 - 1e-12:
    dlam = min(dlam, 1.0 - lam)
    u_saved, d_saved = u_fluc.s.copy(), damage.s.copy()
    # predictor: linear extrapolation in lam, if admissible (else the warm start u_n)
    if dlam_previous is not None:
        u_fluc.s[...] = u_saved + (dlam / dlam_previous) * (u_saved - u_previous)
        if not set_F(lam + dlam, u_fluc) > 0.0:
            u_fluc.s[...] = u_saved
    ok, message, stats = solve_increment(lam + dlam)
    if not ok:
        u_fluc.s[...], damage.s[...] = u_saved, d_saved
        update_degradation()
        root_print(f'  lam {lam + dlam:.5f}: {message} -> halve the step')
        dlam *= 0.5
        if dlam < min_step:
            root_print('step below min_step: stopping')
            break
        continue
    lam += dlam
    increment += 1
    u_previous, dlam_previous = u_saved, dlam
    history_n.s[...] = history.s
    F = F_field.s
    _, _, P = split_neo_hookean(F, mu_q, kappa_q, g_q)
    mean_P = np.array([[global_sum(np.sum(quad_weights * P[i, j])) / np.prod(domain_size) for j in range(dim)]
                       for i in range(dim)])
    J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    detF_out.s[0, 0] = J.mean(axis=0)
    d_max = global_max(np.max(damage.s))
    root_print(f'inc {increment:3d}  lam {lam:.4f} (dlam {dlam:.4f})  F10 {lam * H_macro[1, 0]:+.4f}  '
               f'P10 {mean_P[1, 0]:+.4e}  min J {stats["min_J"]:.2e}  max d {d_max:.3f}  '
               f'stagger {stats["staggered_iterations"]:3d}  hessp {stats["nb_hessp"]:6d}  {message}')
    if writer is not None:
        writer.write(increment=increment, lam=lam, dlam=dlam, F10=lam * H_macro[1, 0], P10=mean_P[1, 0],
                     Pxx=mean_P[0, 0], mean_P=mean_P, min_det_F=stats['min_J'], energy=stats['energy'],
                     damage_max=d_max, staggered_iterations=stats['staggered_iterations'],
                     nb_outer=stats['nb_outer'], nb_hessp=stats['nb_hessp'], cg_damage=stats['cg_damage'],
                     converged=int(message == 'converged'), elapsed_time=time.time() - start_time)
    dlam = min(2.0 * dlam, max_step)

if writer is not None:
    writer.close()
root_print(f'elapsed {time.time() - start_time:.1f} s, {increment} increments, final lam {lam:.4f}')
if writer is not None:
    root_print(f'wrote {output_file}')
    if make_figures and rank == 0:
        import subprocess
        plot_script = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   'plot_phase_field_fracture_contact_output.py')
        # a child started under mpirun must not see the MPI launcher variables, otherwise
        # its MPI_Init (muGrid import) tries to join this job and aborts
        plot_env = {k: v for k, v in os.environ.items()
                    if not k.startswith(('OMPI_', 'PMIX_', 'PMI_', 'ORTE_', 'HYDRA_', 'MPI_LOCALRANKID'))}
        for extra in ([], ['--animate']):
            subprocess.run([sys.executable, plot_script, output_file, '--save'] + extra,
                           check=False, env=plot_env)
