"""
3D twist of a slender beam to large rotations WITH SELF-CONTACT through a third medium, finite strain, solved as an
ENERGY MINIMIZATION with the trust-region Newton-CG method (NuMPI tr_newton_bounded through solvers_nonlinear.minimize).

    Pi(u) = int W(F) dx + 1/2 u . R u - f_imp . u,      F = I + grad u,   W neo-Hookean
    R = k_r H^T W H   (HuHu regularization of the third medium; the LuLu part vanishes for trilinear voxels)
    u = Q x + u_D(theta)

Without contact (example_3D_elasticity_beam_twist.py) the twisted beam buckles and then passes through itself, which
removes two turns of twist at every passage. Here the beam is embedded in a very soft "third medium" (stiffness
k_v x the beam, Wriggers et al.): when two parts of the beam approach, the medium between them is squeezed and
stiffens, which acts as contact. The HuHu regularization keeps the medium from distorting into zero-energy modes.

Geometry (all sizes in voxels of size h = 1/32):
    * a beam with a square cross-section along z, clamped at the bottom, twisted at the top;
    * the third medium fills a block around the beam that spans the whole cell except a layer of true vacuum (zero
      stiffness) on all six sides, which separates the block from its periodic images;
    * the bottom face of the block (beam and medium) is clamped. MEDIUM_TOP selects the top:
        'free'            only the beam end is rotated rigidly about the beam axis by theta (u_x, u_y prescribed,
                          u_z free); the medium top and sides are free, so the medium twists along with the beam
                          and nothing holds it back (no winding);
        'rotated' (default) the whole top face of the block is rotated rigidly. The outer medium is then pinned at its
                          original radius at both ends and, twisted by hundreds of degrees, contracts towards the
                          axis like a twisted bundle of strings (hourglass): det F of the medium -> 0 at mid-height
                          (0.02 at 460 deg), and the trust region stops converging.

A small horizontal force at mid-height (imperfection) breaks the symmetry. Every increment starts from the previous
state rotated by d_theta z / L; if the trust region does not converge, the increment is halved (cutback).

Preconditioner (PRECONDITIONER):
    'Green_Jacobi' (default)  D^-1/2 G D^-1/2,  G = (B^T 2 mu I B + R)^-1,  D = diag(B^T C(F) B + R) normalized by the
                              diagonal of the reference operator; rebuilt after every accepted trust-region iterate
    'Green'                   G = (B^T C_beam B + R)^-1, fixed

usage:  DEVICE=gpu python example_3D_elasticity_beam_twist_contact.py
        (TWIST_MAX_DEG, TWIST_STEP_DEG, BEAM_WIDTH, BEAM_LENGTH, MEDIUM_HALF_WIDTH, MEDIUM_TOP, K_V, POISSON_MEDIUM, ALPHA, PREDICTOR, RESTART, SAVE_EVERY, TR_MAXITER optional)
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import time
from mpi4py import MPI

from muFFTTO import domain
from muFFTTO import material_models
from muFFTTO import solvers_nonlinear
from muFFTTO import tensor_operations

device = os.environ.get('DEVICE', 'cpu')  # 'cpu' or 'gpu'
if device == 'gpu':
    import cupy as xp
else:
    xp = np

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'trilinear_hexahedron'
formulation = 'finite_strain'

# ----------------------------------------------------------------------
# geometry, in voxels of size h
pixel_size = 1 / 32
beam_width = int(os.environ.get('BEAM_WIDTH', 4))  # beam cross-section beam_width x beam_width voxels
beam_length_px = int(os.environ.get('BEAM_LENGTH', 64))  # beam length in voxels
medium_half_width = int(os.environ.get('MEDIUM_HALF_WIDTH', 12))  # third medium from the beam axis to its free face
medium_top = os.environ.get('MEDIUM_TOP', 'rotated')  # 'free': only the beam end is rotated; 'rotated': whole top face
vacuum_layer = 2  # true vacuum around the medium block, separates it from its periodic images
number_of_pixels = (2 * (medium_half_width + vacuum_layer),) * 2 + (beam_length_px + 2 * vacuum_layer,)
domain_size = [n * pixel_size for n in number_of_pixels]

my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type,
                                       device=device)
comm = discretization.communicator
dim = discretization.domain_dimension
start_time = time.time()

# voxels and nodes by GLOBAL index (MPI-safe); node (ix, iy, iz) is the lower corner of voxel (ix, iy, iz)
ix, iy, iz = discretization.fft.icoords
axis = medium_half_width + vacuum_layer  # beam axis at node index x = y = axis
block_lo = [vacuum_layer, vacuum_layer, vacuum_layer]  # medium block (beam included), first voxel
block_hi = [number_of_pixels[0] - vacuum_layer, number_of_pixels[1] - vacuum_layer,
            vacuum_layer + beam_length_px]  # last voxel + 1
beam_lo = [axis - beam_width // 2, axis - beam_width // 2, block_lo[2]]
beam_hi = [beam_lo[0] + beam_width, beam_lo[1] + beam_width, block_hi[2]]


def voxel_box(lo, hi):
    return ((ix >= lo[0]) & (ix < hi[0]) & (iy >= lo[1]) & (iy < hi[1]) & (iz >= lo[2]) & (iz < hi[2]))


def node_box(lo, hi):
    return ((ix >= lo[0]) & (ix <= hi[0]) & (iy >= lo[1]) & (iy <= hi[1]) & (iz >= lo[2]) & (iz <= hi[2]))


block_pixels_host = voxel_box(block_lo, block_hi)
beam_pixels_host = voxel_box(beam_lo, beam_hi)
medium_pixels_host = block_pixels_host & ~beam_pixels_host
block_pixels = xp.asarray(block_pixels_host)
beam_pixels = xp.asarray(beam_pixels_host)
medium_pixels = xp.asarray(medium_pixels_host)
empty_pixels = ~block_pixels  # true vacuum

block_nodes = node_box(block_lo, block_hi)
beam_nodes = node_box(beam_lo, beam_hi)
bottom_nodes = block_nodes & (iz == block_lo[2])  # bottom face of the block: clamped
beam_top_nodes = beam_nodes & (iz == beam_hi[2])
# rotated rigidly: the beam end, or the whole top face of the block
top_nodes = beam_top_nodes if medium_top == 'free' else block_nodes & (iz == block_hi[2])
beam_mid_nodes = beam_nodes & (iz == block_lo[2] + beam_length_px // 2)

# ----------------------------------------------------------------------
# material: neo-Hookean beam, k_v times softer neo-Hookean third medium, zero stiffness outside the block
E_young, poisson = 1.0, 0.2
k_v = float(os.environ.get('K_V', 1e-5))
lam, mu = material_models.get_lame_parameters(E=E_young, poisson=poisson)
K_bulk = lam + 2.0 * mu / 3.0
# medium Poisson ratio: a nearly incompressible medium cannot collapse in volume when it is twisted
poisson_medium = float(os.environ.get('POISSON_MEDIUM', 0.45))
lam_v, mu_v = material_models.get_lame_parameters(E=k_v * E_young, poisson=poisson_medium)

lam_field = discretization.get_quad_field_scalar(name='lam_field')
mu_field = discretization.get_quad_field_scalar(name='mu_field')
lam_field.s[...] = 0
mu_field.s[...] = 0
lam_field.s[..., beam_pixels] = lam
mu_field.s[..., beam_pixels] = mu
lam_field.s[..., medium_pixels] = lam_v
mu_field.s[..., medium_pixels] = mu_v

material = material_models.NeoHookean(discretization=discretization,
                                      lam_1qxyz=lam_field,
                                      mu_1qxyz=mu_field,
                                      name='neo_hookean_beam_and_medium')

# HuHu regularization of the third medium (as in examples/internal_contact): k_r = alpha (K + 4 G / 3) L_ref^2
alpha = float(os.environ.get('ALPHA', 1e-6))
k_r = alpha * 1.0 ** 2 * (K_bulk + 4.0 * mu / 3.0)

# ----------------------------------------------------------------------
# free dofs: nodes of the block without the clamped bottom face and without u_x, u_y of the rotated top face
free_dofs = xp.zeros(discretization.get_unknown_size_field(name='u').s.shape)
free_dofs[:, 0, xp.asarray(block_nodes & ~bottom_nodes)] = 1.0
free_dofs[:2, 0, xp.asarray(top_nodes)] = 0.0

beam_length = beam_length_px * pixel_size
beam_side = beam_width * pixel_size
torsional_stiffness = mu * 0.1406 * beam_side ** 4 / beam_length  # Saint-Venant G J / L of the square beam
theta_max = np.radians(float(os.environ.get('TWIST_MAX_DEG', 1440)))
theta_step = np.radians(float(os.environ.get('TWIST_STEP_DEG', 5)))

rx = (ix - axis) * pixel_size  # node positions relative to the beam axis
ry = (iy - axis) * pixel_size
z_fraction = np.clip((iz - block_lo[2]) / beam_length_px, 0.0, 1.0)  # 0 at the clamp, 1 at the top face

# prescribed displacement u_D: (u_x, u_y) of the top face = rigid rotation by theta
prescribed_field = discretization.get_unknown_size_field(name='prescribed_field')
prescribed_field.s[...] = 0


def set_prescribed_twist(theta):
    prescribed_field.s[0, 0, xp.asarray(top_nodes)] = xp.asarray(
        (np.cos(theta) * rx - np.sin(theta) * ry - rx)[top_nodes])
    prescribed_field.s[1, 0, xp.asarray(top_nodes)] = xp.asarray(
        (np.sin(theta) * rx + np.cos(theta) * ry - ry)[top_nodes])


# imperfection: a small horizontal force on the mid-height row of the beam, ~1e-2 of the Euler load scale E I / L^2
imperfection_force = 1e-2 * E_young * beam_side ** 4 / 12 / beam_length ** 2
imperfection_field = discretization.get_unknown_size_field(name='imperfection_field')
imperfection_field.s[...] = 0
imperfection_field.s[0, 0, xp.asarray(beam_mid_nodes)] = imperfection_force / np.sum(beam_mid_nodes)

# ----------------------------------------------------------------------
# fields
x_field = discretization.get_unknown_size_field(name='x_field')  # u = Q x + u_D
grad_field = discretization.get_unknown_size_field(name='grad_field')
Ru_field = discretization.get_unknown_size_field(name='Ru_field')
v_field = discretization.get_unknown_size_field(name='v_field')
Hv_field = discretization.get_unknown_size_field(name='Hv_field')
r_field = discretization.get_unknown_size_field(name='r_field')
Mr_field = discretization.get_unknown_size_field(name='Mr_field')
HtH_field = discretization.get_unknown_size_field(name='HtH_field')
hess_u = discretization.get_displacement_hessian_size_field(name='hess_u')
F_field = discretization.get_displacement_gradient_sized_field(name='F')
P_field = discretization.get_displacement_gradient_sized_field(name='P')
C_field = discretization.get_material_data_size_field_mugrid(name='C')
energy_field = discretization.get_quad_field_scalar(name='energy_density')
J_field = discretization.get_quad_field_scalar(name='J')

u_shape = x_field.s.shape
quad_weights = xp.asarray(discretization.quadrature_weights).reshape((-1,) + (1,) * dim)


def global_sum(a):
    return comm.sum(float(a))


def global_min(a):
    return float(discretization.mpi_reduction.min(np.asarray(float(a))))


def to_optimizer(array):
    """Flat numpy copy for the optimizer (host)."""
    return tensor_operations.to_host(array).ravel().copy()


def add_regularization(x, out, masked=True):
    """out += R x = k_r H^T W H x; masked: only in the block (zero in the true vacuum). Ghosts of x must be current."""
    discretization.apply_hessian_operator_to_vector_field_mugrid(u_inxyz=x, hess_u_ijkqxyz=hess_u)
    if masked:
        hess_u.s[:, :, :, :, empty_pixels] = 0.0
    discretization.apply_hessian_operator_transposed_to_vector_field_mugrid(hess_u_ijkqxyz=hess_u,
                                                                            nodal_field_inxyz=HtH_field,
                                                                            apply_weights=True)
    out.s[...] += k_r * HtH_field.s


def set_F(x):
    """u = Q x + u_D, F = I + grad u; returns (min det F in the beam, min det F in the medium)."""
    x_field.s[...] = xp.asarray(x).reshape(u_shape) * free_dofs + prescribed_field.s
    discretization.apply_gradient_operator_mugrid(u_inxyz=x_field, grad_u_ijqxyz=F_field)
    for d in range(dim):
        F_field.s[d, d] += 1.0
    tensor_operations.det2(F_field, J_field)
    return (global_min(J_field.s[0, 0][:, beam_pixels].min()), global_min(J_field.s[0, 0][:, medium_pixels].min()))


def evaluate_stress():
    """P(F) at the current F, zero in the true vacuum."""
    with np.errstate(invalid='ignore', divide='ignore'):  # ln J of (irrelevant) true-vacuum voxels
        material.get_stress(F_field, P_field)
    P_field.s[:, :, :, empty_pixels] = 0.0


def internal_force():
    """grad_field = B^T W P + R u at the current state (x_field, P_field)."""
    discretization.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=P_field,
                                                             div_u_fnxyz=grad_field,
                                                             apply_weights=True)
    discretization.fft.communicate_ghosts(x_field)
    Ru_field.s[...] = 0.0
    add_regularization(x_field, Ru_field)
    grad_field.s[...] += Ru_field.s


# ----------------------------------------------------------------------
# energy, gradient, Hessian-vector product
theta_current = 0.0  # current twist, set by the load loop
INADMISSIBLE_ENERGY = 1e30  # finite (not inf): the trust region then rejects the step and shrinks

# CG iterations of every inner (Steihaug) solve: NuMPI calls hessp once per CG iteration and evaluates fun_grad once
# after every solve (at the trial point), so the Hessian products between two fun_grad calls are one solve
cg_counter = {'current': 0, 'solves': []}


def fun_grad(x):
    """(Pi, Q grad Pi) at u = Q x + u_D."""
    if cg_counter['current'] > 0:
        cg_counter['solves'].append(cg_counter['current'])
        cg_counter['current'] = 0
    J_beam, J_medium = set_F(x)
    J_min = min(J_beam, J_medium)
    if not np.isfinite(J_min) or J_min <= 0.0:
        return INADMISSIBLE_ENERGY, np.zeros(x.size)
    with np.errstate(invalid='ignore', divide='ignore'):
        material.get_energy_density(F_field, energy_field)
    energy_field.s[0, 0][:, empty_pixels] = 0.0
    evaluate_stress()
    internal_force()
    Pi = (global_sum(xp.sum(quad_weights * energy_field.s[0, 0])) + 0.5 * global_sum(xp.sum(x_field.s * Ru_field.s))
          - global_sum(xp.sum(imperfection_field.s * x_field.s)))
    grad_field.s[...] = (grad_field.s - imperfection_field.s) * free_dofs
    return Pi, to_optimizer(grad_field.s)


tangent_cache = {'x': None, 'theta': None}


def update_tangent(x):
    set_F(x)
    with np.errstate(invalid='ignore', divide='ignore'):
        material.get_algorithmic_tangent(F_field, C_field)
    C_field.s[:, :, :, :, :, empty_pixels] = 0.0


def apply_hessian(input_field_inxyz, output_field_inxyz):
    """(B^T W C B + R) input, with the cached tangent."""
    discretization.fft.communicate_ghosts(input_field_inxyz)
    discretization.apply_system_matrix_mugrid(material_data_field=C_field,
                                              input_field_inxyz=input_field_inxyz,
                                              output_field_inxyz=output_field_inxyz,
                                              formulation=formulation)
    add_regularization(input_field_inxyz, output_field_inxyz)
    discretization.fft.communicate_ghosts(output_field_inxyz)


def hessp(x, v):
    """Q H(Q x + u_D) Q v; the tangent C(F) is recomputed only when x or theta changes."""
    cg_counter['current'] += 1
    if tangent_cache['x'] is None or tangent_cache['theta'] != theta_current or not np.array_equal(tangent_cache['x'],
                                                                                                   x):
        update_tangent(x)
        tangent_cache['x'] = np.array(x, copy=True)
        tangent_cache['theta'] = theta_current
    v_field.s[...] = xp.asarray(v).reshape(u_shape) * free_dofs
    apply_hessian(v_field, Hv_field)
    return to_optimizer(Hv_field.s * free_dofs)


# ----------------------------------------------------------------------
# preconditioner, applied as Q M Q
preconditioner_type = os.environ.get('PRECONDITIONER', 'Green_Jacobi')  # 'Green_Jacobi' or 'Green'
i2 = np.eye(dim)
I4rt = np.einsum('ik,jl->ijkl', i2, i2)
if preconditioner_type == 'Green_Jacobi':
    reference_tangent = 2 * mu * I4rt
else:
    reference_tangent = material_models.get_elastic_tensor_from_lame(dim=dim, lam=lam, mu=mu)


def reference_operator(input_field_inxyz, output_field_inxyz):
    """B^T W C_ref B + R on the whole (translation-invariant) cell: the operator the Green blocks invert."""
    discretization.fft.communicate_ghosts(field=input_field_inxyz)
    discretization.apply_system_matrix_mugrid(material_data_field=reference_tangent, input_field_inxyz=input_field_inxyz,
                                              output_field_inxyz=output_field_inxyz, formulation=formulation)
    add_regularization(input_field_inxyz, output_field_inxyz, masked=False)


green = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=reference_tangent,
                                                       operator=reference_operator, name='green_reference')
# Green-Jacobi: D normalized by the (uniform) diagonal d_ref of the reference operator, so M = K_ref wherever the
# tangent is the reference one (keeps the trust-region measure of the beam)
reference_diagonal = float(discretization.mpi_reduction.max(np.asarray(float(xp.max(
    1.0 / discretization.get_preconditioner_Jacobi_mugrid(operator=reference_operator, name='jacobi_reference').s ** 2)))))
jacobi_field = None


def update_preconditioner(x):
    """Green-Jacobi: D^-1/2 of B^T W C(F(x)) B + R; nodes with a zero diagonal (true vacuum) are not moved."""
    global jacobi_field
    if preconditioner_type != 'Green_Jacobi':
        return
    update_tangent(x)
    tangent_cache['x'] = None  # C_field was rebuilt outside hessp
    jacobi = discretization.get_preconditioner_Jacobi_mugrid(operator=apply_hessian, name='jacobi_current',
                                                             zero_threshold=0.0)
    jacobi_field = float(np.sqrt(reference_diagonal)) * jacobi.s


def precond(r):
    """Q M Q r."""
    r_field.s[...] = xp.asarray(r).reshape(u_shape) * free_dofs
    if preconditioner_type == 'Green_Jacobi':
        r_field.s[...] *= jacobi_field
    discretization.fft.communicate_ghosts(r_field)
    discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green,
                                               input_nodal_field_fnxyz=r_field,
                                               output_nodal_field_fnxyz=Mr_field)
    if preconditioner_type == 'Green_Jacobi':
        Mr_field.s[...] *= jacobi_field
    return to_optimizer(Mr_field.s * free_dofs)


# ----------------------------------------------------------------------
# measurements
nb_beam_top_nodes = discretization.mpi_reduction.sum(np.asarray(np.sum(beam_top_nodes)))


def measure(x):
    """Reaction torque about the axis on the top face (internal forces incl. R, deformed positions), mean axial
    displacement of the beam top, min det F in beam and medium."""
    J_beam, J_medium = set_F(x)
    evaluate_stress()
    internal_force()
    f = tensor_operations.to_host(grad_field.s)[:, 0]
    u_s = tensor_operations.to_host(x_field.s)[:, 0]
    X, Y = rx + u_s[0], ry + u_s[1]
    torque = discretization.mpi_reduction.sum(np.asarray(np.sum((X * f[1] - Y * f[0])[top_nodes])))
    u_z = discretization.mpi_reduction.sum(np.asarray(np.sum(u_s[2][beam_top_nodes]))) / nb_beam_top_nodes
    return float(torque), float(u_z), J_beam, J_medium


# ----------------------------------------------------------------------
# output
data_folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'exp_data',
                                os.path.splitext(os.path.basename(__file__))[0])
data_file = os.path.join(data_folder_path, f'beam_twist_contact_3D_{preconditioner_type}_{medium_top}_nu{poisson_medium:g}_hw{medium_half_width}.npz')
history = {'T': [0.0], 'twist': [0.0], 'top_u_z': [0.0], 'min_J_beam': [1.0], 'min_J_medium': [1.0], 'nit': [0],
           'nhessp': [0], 'nb_cg': [0], 'time': [0.0]}
solve_increment = []  # accepted increment of every inner CG solve (0: rejected / cut-back attempts)
displacements = [np.zeros(u_shape, dtype=np.float32)[:, 0]]


def save():
    if comm.size != 1:
        return
    os.makedirs(data_folder_path, exist_ok=True)
    np.savez(data_file, displacement=np.array(displacements, dtype=np.float32),
             beam_pixels=beam_pixels_host, block_pixels=block_pixels_host, pixel_size=pixel_size,
             number_of_pixels=number_of_pixels, beam_lo=beam_lo, beam_hi=beam_hi, beam_length=beam_length,
             torsional_stiffness=torsional_stiffness, k_v=k_v, alpha=alpha, k_r=k_r,
             cg_per_solve=np.array(cg_counter['solves']), solve_increment=np.array(solve_increment),
             preconditioner=preconditioner_type, device=device,
             **{key: np.array(value) for key, value in history.items()})


# ----------------------------------------------------------------------
# incremental twist with cutback
nb_dofs = int(np.prod(u_shape))
# predictor of every increment: 'rotate' rotates every cross-section of the converged state by d_theta z / L about the
# straight axis (good while the beam is straight); 'extrapolate' extrapolates linearly from the last two converged
# states (follows a buckled, coiling beam), falling back to 'rotate' when there is no previous state
predictor = os.environ.get('PREDICTOR', 'rotate')
if comm.rank == 0:
    print(f'device {device}, preconditioner {preconditioner_type}, predictor {predictor}, medium top {medium_top}, medium Poisson {poisson_medium:g}, beam {beam_width}x{beam_width}x{beam_length_px} '
          f'voxels, medium block {np.subtract(block_hi, block_lo)} voxels, grid {number_of_pixels} ({nb_dofs} dofs), '
          f'k_v = {k_v:g}, k_r = {k_r:.3e}, G J / L = {torsional_stiffness:.4e}, twist up to '
          f'{np.degrees(theta_max):.0f} deg in steps of {np.degrees(theta_step):g} deg', flush=True)

x = np.zeros(nb_dofs)
theta_done, step = 0.0, theta_step
increment = 0
x_previous, theta_previous = None, None
# trust-region iterations per increment: a snap-through (the buckled beam jumping to a new branch) needs many more
# than a regular increment, the radius first collapses and then has to grow along the long path
tr_maxiter = int(os.environ.get('TR_MAXITER', 300))
save_every = int(os.environ.get('SAVE_EVERY', 10))  # save the data file every save_every increments
restart_file = os.environ.get('RESTART')  # continue from the last increment stored in this data file
if restart_file:
    restart = np.load(restart_file)
    for key in history:
        history[key] = list(restart[key])
    displacements = list(restart['displacement'])
    cg_counter['solves'] = list(restart['cg_per_solve'])
    solve_increment = list(restart['solve_increment'])
    increment = len(displacements) - 1
    theta_done = float(restart['twist'][-1])
    x = (displacements[-1][:, None] * tensor_operations.to_host(free_dofs)).ravel().astype(np.float64)
    if len(displacements) > 1:
        x_previous = (displacements[-2][:, None] * tensor_operations.to_host(free_dofs)).ravel().astype(np.float64)
        theta_previous = float(restart['twist'][-2])
    start_time -= history['time'][-1]
    if comm.rank == 0:
        print(f'restart from {restart_file}: increment {increment}, twist {np.degrees(theta_done):.2f} deg',
              flush=True)
while theta_done < theta_max - 1e-12:
    theta_current = min(theta_done + step, theta_max)
    set_prescribed_twist(theta_current)
    # predictor: rotate every cross-section of the converged state by d_theta z / L about the axis
    u_s = (tensor_operations.to_host(xp.asarray(x).reshape(u_shape) * free_dofs)
           + tensor_operations.to_host(prescribed_field.s))[:, 0]
    X, Y = rx + u_s[0], ry + u_s[1]
    angle = (theta_current - theta_done) * z_fraction
    u_s[0] += (np.cos(angle) - 1) * X - np.sin(angle) * Y
    u_s[1] += np.sin(angle) * X + (np.cos(angle) - 1) * Y
    x_start = (u_s[:, None] * tensor_operations.to_host(free_dofs)).ravel()
    if predictor == 'extrapolate' and x_previous is not None:
        x_extrapolated = x + (x - x_previous) * (theta_current - theta_done) / (theta_done - theta_previous)
        if min(set_F(x_extrapolated)) > 0.0:  # after a snap the extrapolation can invert elements: keep 'rotate'
            x_start = x_extrapolated
    if min(set_F(x_start)) <= 0.0:  # an inverted start has no energy, the optimizer would stop at its zero gradient
        if comm.rank == 0:
            print(f'   cutback at twist {np.degrees(theta_current):7.2f} deg (step {np.degrees(step):.3g} deg): '
                  f'predictor inverts elements', flush=True)
        step /= 2
        if step < theta_step / 64:
            break
        continue
    nb_solves_before = len(cg_counter['solves'])

    update_preconditioner(x_start)
    result = solvers_nonlinear.minimize(fun=fun_grad, x0=x_start, method='tr_newton', hessp=hessp, precond=precond,
                                        gtol=1e-10, maxiter=tr_maxiter, inner_tol=1e-6, inner_maxiter=20000,
                                        comm=MPI.COMM_WORLD, callback=update_preconditioner)
    if cg_counter['current'] > 0:  # a last solve not followed by fun_grad
        cg_counter['solves'].append(cg_counter['current'])
        cg_counter['current'] = 0
    if not result.success:
        solve_increment.extend([0] * (len(cg_counter['solves']) - len(solve_increment)))
        if comm.rank == 0:
            print(f'   cutback at twist {np.degrees(theta_current):7.2f} deg (step {np.degrees(step):.3g} deg): '
                  f'{result.message}, TR its {result.nit}, max |grad| {result.max_grad:.2e} (gtol 1e-10), last '
                  f'radii {", ".join(f"{d:.1e}" for d in result.delta_history[-3:])}', flush=True)
        step /= 2
        if step < theta_step / 64:
            if comm.rank == 0:
                print('   step too small, stopping', flush=True)
            break
        continue

    x_previous, theta_previous = x, theta_done
    x = np.asarray(result.x)
    increment += 1
    theta_done = theta_current
    step = min(theta_step, 1.5 * step)
    solve_increment.extend([increment] * (len(cg_counter['solves']) - len(solve_increment)))
    torque, top_u_z, J_beam, J_medium = measure(x)
    nb_cg = int(np.sum(cg_counter['solves'][nb_solves_before:]))
    history['T'].append(torque)
    history['twist'].append(theta_done)
    history['top_u_z'].append(top_u_z)
    history['min_J_beam'].append(J_beam)
    history['min_J_medium'].append(J_medium)
    history['nit'].append(int(result.nit))
    history['nhessp'].append(int(result.nhessp))
    history['nb_cg'].append(nb_cg)
    history['time'].append(time.time() - start_time)
    displacements.append(tensor_operations.to_host(x_field.s)[:, 0].astype(np.float32))
    if comm.rank == 0:
        print(f'increment {increment:3d}  twist = {np.degrees(theta_done):7.2f} deg  '
              f'T L/(G J) = {torque / torsional_stiffness:7.3f}  top u_z = {top_u_z:+.4e}  '
              f'min det F beam {J_beam:.3f} medium {J_medium:.3f}  TR its {result.nit:3d}  CG its {nb_cg:6d}  '
              f't = {history["time"][-1]:7.1f} s', flush=True)
    if increment % save_every == 0:
        save()

save()
if comm.rank == 0:
    print(f'total TR iterations {sum(history["nit"])}, CG iterations {sum(cg_counter["solves"])} '
          f'in {len(cg_counter["solves"])} solves, elapsed time {time.time() - start_time:.1f} s', flush=True)
    print(f'wrote {data_file}')
