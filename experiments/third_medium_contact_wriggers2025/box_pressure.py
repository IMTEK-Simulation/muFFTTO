"""
Box under internal pressure and suction, solved with the muFFTTO third-medium approach.

Reproduces Sec. 4.1.2 (Figs. 6 and 7) of
    P. Wriggers, J. Korelc, P. Junker, A third medium approach for contact using first and second order finite
    elements, Comput. Methods Appl. Mech. Engrg. 436 (2025) 117740.
Only the problem is taken from the paper; the method is ours (see box_point_load.py).

Problem (paper):
    * the box of Fig. 2: [0, L] x [0, H], L = 2, H = 0.5, walls of thickness T = 0.1, the inside filled with the third
      medium; same material (K = 20, mu = 10, plane strain) and supports (lower left corner fixed, lower right corner
      u_y = 0); no point load;
    * a pressure P acts in the third medium through the energy W_P = int_medium P J dV (eq. (22)): P > 0 is suction,
      P < 0 an internal pressure;
    * part 1, inflation: P = -0.8, the box blows up to an almost circular balloon (Fig. 7);
    * part 2, decompression: the pressure is released and turned into suction up to P = 1; the box collapses until the
      upper and lower flange come into contact through the squeezed medium (Fig. 6, P = 0.1 and P = 1).
    The two parts run as one load path P: 0 -> -0.8 -> 0 -> 1 (PRESSURE_PATH).

Supports: the paper's (lower left corner fixed, lower right corner u_y = 0) by default. Under strong suction the
collapsed box can break the left-right symmetry (the upper flange slides sideways, the side walls fold over, which
encloses less area); SYMMETRIC=1 keeps the symmetric branch: u_x = 0 on the middle column x = L/2 and u_y = 0 at both
lower corners (the paper's supports up to a rigid horizontal shift, plus the symmetry).

Gap: the distance from the inner face of the upper flange to the inner face of the lower flange (a polyline through
its nodes): at x = L/2 (gap) and the smallest along the box (gap_min).

Our model: as in box_point_load.py (pixel Q1 elements, neo-Hookean solid and GAMMA-scaled medium, HuHu regularization
of the medium, trust-region Newton-CG with Green-Jacobi, cutback). The pressure adds to the medium
    energy P (J - 1),  stress P cof F,  tangent P d cof F / d F
(cof F = J F^{-T}; in 2D d cof F / d F is a constant tensor). Every increment starts from the last converged state
(plain trust-region Newton-CG); an increment that does not converge is halved (cutback).

usage:
    python box_pressure.py      # figures and data into exp_data/box_pressure/
    (NB_PER_UNIT, VACUUM_LAYER, MATERIAL, GAMMA, ALPHA, PRESSURE_PATH, PRESSURE_STEP, SYMMETRIC, PRECONDITIONER
     optional)
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpi4py import MPI

from muFFTTO import domain
from muFFTTO import material_models
from muFFTTO import solvers_nonlinear
from muFFTTO import tensor_operations

xp = np
problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'bilinear_rectangle'
formulation = 'finite_strain'

# ----------------------------------------------------------------------
# geometry (paper) and pixels
box_length, box_height, wall = 2.0, 0.5, 0.1
nb_per_unit = int(os.environ.get('NB_PER_UNIT', 40))  # pixels per unit length, h = 1 / nb_per_unit
vacuum_layer = int(os.environ.get('VACUUM_LAYER', 2))
pixel_size = 1.0 / nb_per_unit
box_px = (round(box_length * nb_per_unit), round(box_height * nb_per_unit))
wall_px = round(wall * nb_per_unit)
number_of_pixels = (box_px[0] + 2 * vacuum_layer, box_px[1] + 2 * vacuum_layer)
domain_size = [n * pixel_size for n in number_of_pixels]

my_cell = domain.PeriodicUnitCell(domain_size=domain_size, problem_type=problem_type)
discretization = domain.Discretization(cell=my_cell, nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type, element_type=element_type)
comm = discretization.communicator
dim = discretization.domain_dimension

# pixels and nodes by global index; node (ix, iy) is the lower left corner of pixel (ix, iy)
ix, iy = discretization.fft.icoords
x0, y0 = vacuum_layer, vacuum_layer  # node index of the box corner (0, 0)
box_pixels = (ix >= x0) & (ix < x0 + box_px[0]) & (iy >= y0) & (iy < y0 + box_px[1])
medium_pixels = ((ix >= x0 + wall_px) & (ix < x0 + box_px[0] - wall_px) &
                 (iy >= y0 + wall_px) & (iy < y0 + box_px[1] - wall_px))
solid_pixels = box_pixels & ~medium_pixels
empty_pixels = ~box_pixels  # true vacuum
box_nodes = (ix >= x0) & (ix <= x0 + box_px[0]) & (iy >= y0) & (iy <= y0 + box_px[1])

mid = x0 + box_px[0] // 2  # node column x = L/2
fixed_node = (ix == x0) & (iy == y0)  # lower left corner: u = 0
roller_node = (ix == x0 + box_px[0]) & (iy == y0)  # lower right corner: u_y = 0
gap_top_node = (ix == mid) & (iy == y0 + box_px[1] - wall_px)  # inner face of the upper flange at x = L/2
gap_bottom_node = (ix == mid) & (iy == y0 + wall_px)  # inner face of the lower flange at x = L/2

# ----------------------------------------------------------------------
# material
K_bulk, mu = 20.0, 10.0
lam = K_bulk - 2.0 * mu / 3.0
gamma = float(os.environ.get('GAMMA', 1e-4))
alpha = float(os.environ.get('ALPHA', 1e-6))
k_r = alpha * (K_bulk + 4.0 * mu / 3.0) * box_length ** 2

material_type = os.environ.get('MATERIAL', 'neo_hookean')  # 'neo_hookean' or 'decoupled' (the paper's energy)
first_field = discretization.get_quad_field_scalar(name='lam_or_K_field')  # lam (neo_hookean) or K (decoupled)
mu_field = discretization.get_quad_field_scalar(name='mu_field')
first_parameter = lam if material_type == 'neo_hookean' else K_bulk
first_field.s[...] = 0
mu_field.s[...] = 0
first_field.s[..., solid_pixels] = first_parameter
mu_field.s[..., solid_pixels] = mu
first_field.s[..., medium_pixels] = gamma * first_parameter
mu_field.s[..., medium_pixels] = gamma * mu
if material_type == 'neo_hookean':
    material = material_models.NeoHookean(discretization=discretization, lam_1qxyz=first_field, mu_1qxyz=mu_field,
                                          name='neo_hookean_box')
else:
    material = material_models.DecoupledNeoHookean(discretization=discretization, K_1qxyz=first_field,
                                                   mu_1qxyz=mu_field, name='decoupled_neo_hookean_box')

# ----------------------------------------------------------------------
# output
data_folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'exp_data',
                                os.path.splitext(os.path.basename(__file__))[0])
os.makedirs(data_folder_path, exist_ok=True)

# ----------------------------------------------------------------------
# constraints: u = Q x (lower left corner fixed, lower right corner u_y = 0)
free_dofs = np.zeros(discretization.get_unknown_size_field(name='u').s.shape)
symmetric = bool(int(os.environ.get('SYMMETRIC', 0)))
free_dofs[:, 0, box_nodes] = 1.0
free_dofs[1, 0, fixed_node] = 0.0
free_dofs[1, 0, roller_node] = 0.0
if symmetric:
    free_dofs[0, 0, box_nodes & (ix == mid)] = 0.0  # symmetry plane x = L/2
else:
    free_dofs[0, 0, fixed_node] = 0.0

# ----------------------------------------------------------------------
# fields
x_field = discretization.get_unknown_size_field(name='x_field')
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
quad_weights = np.asarray(discretization.quadrature_weights).reshape((-1,) + (1,) * dim)


def global_sum(a):
    return comm.sum(float(a))


def global_min(a):
    return float(discretization.mpi_reduction.min(np.asarray(float(a))))


def add_regularization(x, out, masked=True):
    """out += R x = k_r H^T W H x; masked: only in the third medium. Ghosts of x must be current."""
    discretization.apply_hessian_operator_to_vector_field_mugrid(u_inxyz=x, hess_u_ijkqxyz=hess_u)
    if masked:
        hess_u.s[:, :, :, :, ~medium_pixels] = 0.0
    discretization.apply_hessian_operator_transposed_to_vector_field_mugrid(hess_u_ijkqxyz=hess_u,
                                                                            nodal_field_inxyz=HtH_field,
                                                                            apply_weights=True)
    out.s[...] += k_r * HtH_field.s


def set_F(x):
    """u = Q x, F = I + grad u; returns (min det F in the solid, min det F in the medium)."""
    x_field.s[...] = np.asarray(x).reshape(u_shape) * free_dofs
    discretization.apply_gradient_operator_mugrid(u_inxyz=x_field, grad_u_ijqxyz=F_field)
    for d in range(dim):
        F_field.s[d, d] += 1.0
    tensor_operations.det2(F_field, J_field)
    return (global_min(J_field.s[0, 0][:, solid_pixels].min()), global_min(J_field.s[0, 0][:, medium_pixels].min()))


# pressure in the third medium, energy W_P = P (J - 1) per unit reference volume (paper eq. (22); P > 0 suction,
# P < 0 internal pressure that inflates). In 2D: d J / d F = cof F = [[F11, -F10], [-F01, F00]] and
# d cof F_ij / d F_kl is the constant tensor E below.
pressure_state = {'P': 0.0}
E_cof = np.zeros((2, 2, 2, 2))
E_cof[0, 0, 1, 1] = E_cof[1, 1, 0, 0] = 1.0
E_cof[0, 1, 1, 0] = E_cof[1, 0, 0, 1] = -1.0


def cofactor(F):
    return np.array([[F[1, 1], -F[1, 0]], [-F[0, 1], F[0, 0]]])


def internal_force():
    """grad_field = B^T W P + R u at the current state (P includes the pressure stress P cof F in the medium)."""
    with np.errstate(invalid='ignore', divide='ignore'):  # ln J of the (irrelevant) vacuum pixels
        material.get_stress(F_field, P_field)
    P_field.s[:, :, :, empty_pixels] = 0.0
    P_field.s[:, :, :, medium_pixels] += pressure_state['P'] * cofactor(F_field.s)[:, :, :, medium_pixels]
    discretization.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=P_field, div_u_fnxyz=grad_field,
                                                             apply_weights=True)
    discretization.fft.communicate_ghosts(x_field)
    Ru_field.s[...] = 0.0
    add_regularization(x_field, Ru_field)
    grad_field.s[...] += Ru_field.s


INADMISSIBLE_ENERGY = 1e30  # finite: the trust region rejects the step and shrinks
cg_counter = {'current': 0, 'solves': []}  # CG iterations of every inner solve (Hessian products between fun calls)


def fun_grad(x):
    """(Pi, Q grad Pi) at u = Q x."""
    if cg_counter['current'] > 0:
        cg_counter['solves'].append(cg_counter['current'])
        cg_counter['current'] = 0
    J_min = min(set_F(x))
    if not np.isfinite(J_min) or J_min <= 0.0:
        return INADMISSIBLE_ENERGY, np.zeros(x.size)
    with np.errstate(invalid='ignore', divide='ignore'):
        material.get_energy_density(F_field, energy_field)
    energy_field.s[0, 0][:, empty_pixels] = 0.0
    energy_field.s[0, 0][:, medium_pixels] += pressure_state['P'] * (J_field.s[0, 0][:, medium_pixels] - 1.0)
    internal_force()
    Pi = global_sum(np.sum(quad_weights * energy_field.s[0, 0])) + 0.5 * global_sum(np.sum(x_field.s * Ru_field.s))
    return Pi, (grad_field.s * free_dofs).ravel().copy()


tangent_cache = {'x': None, 'P': None}


def update_tangent(x):
    set_F(x)
    with np.errstate(invalid='ignore', divide='ignore'):
        material.get_algorithmic_tangent(F_field, C_field)
    C_field.s[:, :, :, :, :, empty_pixels] = 0.0
    C_field.s[:, :, :, :, :, medium_pixels] += pressure_state['P'] * E_cof[..., None, None]


def apply_hessian(input_field_inxyz, output_field_inxyz):
    """(B^T W C B + R) input, with the cached tangent."""
    discretization.fft.communicate_ghosts(input_field_inxyz)
    discretization.apply_system_matrix_mugrid(material_data_field=C_field, input_field_inxyz=input_field_inxyz,
                                              output_field_inxyz=output_field_inxyz, formulation=formulation)
    add_regularization(input_field_inxyz, output_field_inxyz)
    discretization.fft.communicate_ghosts(output_field_inxyz)


def hessp(x, v):
    """Q H(Q x) Q v; the tangent is recomputed only when x or the pressure changes."""
    cg_counter['current'] += 1
    if tangent_cache['x'] is None or tangent_cache['P'] != pressure_state['P'] or not np.array_equal(tangent_cache['x'],
                                                                                                     x):
        update_tangent(x)
        tangent_cache['x'] = np.array(x, copy=True)
        tangent_cache['P'] = pressure_state['P']
    v_field.s[...] = np.asarray(v).reshape(u_shape) * free_dofs
    apply_hessian(v_field, Hv_field)
    return (Hv_field.s * free_dofs).ravel().copy()


# ----------------------------------------------------------------------
# preconditioner, applied as Q M Q
preconditioner_type = os.environ.get('PRECONDITIONER', 'Green_Jacobi')  # 'Green_Jacobi' or 'Green'
I4rt = np.einsum('ik,jl->ijkl', np.eye(dim), np.eye(dim))
if preconditioner_type == 'Green_Jacobi':
    reference_tangent = 2 * mu * I4rt
else:
    reference_tangent = material_models.get_elastic_tensor_from_lame(dim=dim, lam=lam, mu=mu)


def reference_operator(input_field_inxyz, output_field_inxyz):
    """B^T W C_ref B + R on the whole cell: the operator the Green blocks invert."""
    discretization.fft.communicate_ghosts(field=input_field_inxyz)
    discretization.apply_system_matrix_mugrid(material_data_field=reference_tangent, input_field_inxyz=input_field_inxyz,
                                              output_field_inxyz=output_field_inxyz, formulation=formulation)
    add_regularization(input_field_inxyz, output_field_inxyz, masked=False)


green = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=reference_tangent,
                                                       operator=reference_operator, name='green_reference')
reference_diagonal = float(np.max(1.0 / discretization.get_preconditioner_Jacobi_mugrid(
    operator=reference_operator, name='jacobi_reference').s ** 2))
jacobi_field = None


def update_preconditioner(x):
    """Green-Jacobi: D^-1/2 of the current B^T W C B + R, normalized by the reference diagonal."""
    global jacobi_field
    if preconditioner_type != 'Green_Jacobi':
        return
    update_tangent(x)
    tangent_cache['x'] = None
    jacobi = discretization.get_preconditioner_Jacobi_mugrid(operator=apply_hessian, name='jacobi_current',
                                                             zero_threshold=0.0)
    jacobi_field = float(np.sqrt(reference_diagonal)) * jacobi.s


def precond(r):
    """Q M Q r."""
    r_field.s[...] = np.asarray(r).reshape(u_shape) * free_dofs
    if preconditioner_type == 'Green_Jacobi':
        r_field.s[...] *= jacobi_field
    discretization.fft.communicate_ghosts(r_field)
    discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green, input_nodal_field_fnxyz=r_field,
                                               output_nodal_field_fnxyz=Mr_field)
    if preconditioner_type == 'Green_Jacobi':
        Mr_field.s[...] *= jacobi_field
    return (Mr_field.s * free_dofs).ravel().copy()


# ----------------------------------------------------------------------
# load path in the pressure: P = path[0] -> path[1] -> ... in steps of at most PRESSURE_STEP
pressure_path = [float(p) for p in os.environ.get('PRESSURE_PATH', '0,-0.8,0,1').split(',')]
pressure_step = float(os.environ.get('PRESSURE_STEP', 0.05))
run_name = (f'n{nb_per_unit}_gamma{gamma:g}_alpha{alpha:g}_path{"_".join(f"{p:g}" for p in pressure_path)}'
            + ('_decoupled' if material_type == 'decoupled' else '') + ('_sym' if symmetric else ''))
gap_top_node_y = (box_px[1] - wall_px) * pixel_size
gap_bottom_node_y = wall_px * pixel_size
medium_area = float(np.sum(medium_pixels)) * pixel_size ** 2


inner_columns = np.arange(x0 + wall_px, x0 + box_px[0] - wall_px + 1)  # node columns of the inner faces
inner_top_row, inner_bottom_row = y0 + box_px[1] - wall_px, y0 + wall_px
mid_in_inner = int(np.flatnonzero(inner_columns == mid)[0])


def surface_gap(u):
    """Distance of every node of the upper inner face to the lower inner face (polyline): (at x = L/2, smallest)."""
    X = np.stack(np.meshgrid(np.arange(number_of_pixels[0]), np.arange(number_of_pixels[1]), indexing='ij')) \
        * pixel_size + u
    top = X[:, inner_columns, inner_top_row].T  # [node, xy]
    bottom = X[:, inner_columns, inner_bottom_row].T
    a, b = bottom[:-1], bottom[1:]  # segments of the lower face
    ab = b - a
    t = np.clip(np.einsum('psk,sk->ps', top[:, None] - a[None], ab) / np.einsum('sk,sk->s', ab, ab), 0.0, 1.0)
    distance = np.linalg.norm(top[:, None] - (a[None] + t[..., None] * ab[None]), axis=-1).min(axis=1)
    return float(distance[mid_in_inner]), float(distance.min())


def measure():
    """Gap (at x = L/2 and smallest), medium area / initial area, box length (distance of the two lower corners)."""
    u = x_field.s[:, 0]
    gap, gap_min = surface_gap(u)
    area = global_sum(np.sum((quad_weights * J_field.s[0, 0])[:, medium_pixels])) / medium_area
    length = box_length + float(u[0][roller_node].sum()) - float(u[0][fixed_node].sum())
    return gap, gap_min, area, length


nb_dofs = int(np.prod(u_shape))
print(f'box {box_px[0]} x {box_px[1]} pixels (h = 1/{nb_per_unit}), grid {number_of_pixels}, {nb_dofs} dofs, '
      f'material {material_type}, gamma = {gamma:g}, k_r = {k_r:.3e}, preconditioner {preconditioner_type}, '
      f'pressure path {pressure_path} in steps of {pressure_step:g}', flush=True)

start_time = time.time()
history = {'P': [0.0], 'stage': [0], 'gap': [(box_px[1] - 2 * wall_px) * pixel_size],
           'gap_min': [(box_px[1] - 2 * wall_px) * pixel_size], 'area': [1.0],
           'length': [box_length], 'min_J_solid': [1.0], 'min_J_medium': [1.0], 'nit': [0], 'nb_cg': [0],
           'time': [0.0]}
displacements = [np.zeros((dim,) + tuple(number_of_pixels))]
x = np.zeros(nb_dofs)
P_done = pressure_path[0]
for stage, P_target in enumerate(pressure_path[1:], start=1):
    step = pressure_step
    direction = np.sign(P_target - P_done)
    while direction * (P_target - P_done) > 1e-12:
        P_current = P_done + direction * min(step, abs(P_target - P_done))
        pressure_state['P'] = P_current
        x_start = x
        nb_solves_before = len(cg_counter['solves'])
        if min(set_F(x_start)) <= 0.0:
            print(f'   cutback at P = {P_current:+.4f} (step {step:.3g}): start inverts elements', flush=True)
            step /= 2
            continue
        update_preconditioner(x_start)
        result = solvers_nonlinear.minimize(fun=fun_grad, x0=x_start, method='tr_newton', hessp=hessp, precond=precond,
                                            gtol=1e-9, maxiter=300, inner_tol=1e-6, inner_maxiter=20000,
                                            comm=MPI.COMM_WORLD, callback=update_preconditioner)
        if cg_counter['current'] > 0:
            cg_counter['solves'].append(cg_counter['current'])
            cg_counter['current'] = 0
        if not result.success:
            print(f'   cutback at P = {P_current:+.4f} (step {step:.3g}): {result.message}, TR its {result.nit}',
                  flush=True)
            step /= 2
            if step < pressure_step / 64:
                print('   step too small, stopping', flush=True)
                break
            continue
        x = np.asarray(result.x)
        P_done = P_current
        step = min(pressure_step, 1.5 * step)
        J_solid, J_medium = set_F(x)
        fun_grad(x)
        gap, gap_min, area, length = measure()
        nb_cg = int(np.sum(cg_counter['solves'][nb_solves_before:]))
        for key, value in zip(history, (P_done, stage, gap, gap_min, area, length, J_solid, J_medium, int(result.nit), nb_cg,
                                        time.time() - start_time)):
            history[key].append(value)
        displacements.append(x_field.s[:, 0].copy())
        print(f'stage {stage}  P = {P_done:+.4f}  gap = {gap:.4e} (min {gap_min:.4e})  area/area_0 = {area:.4f}  length = {length:.4f}  '
              f'min det F solid {J_solid:.3f} medium {J_medium:.4f}  TR its {result.nit:3d}  CG its {nb_cg:6d}  '
              f't = {history["time"][-1]:6.1f} s', flush=True)
    else:
        continue
    break  # the inner loop stopped (step too small)

np.savez(os.path.join(data_folder_path, f'{run_name}.npz'), displacement=np.array(displacements),
         solid_pixels=solid_pixels, medium_pixels=medium_pixels, pixel_size=pixel_size,
         number_of_pixels=number_of_pixels, vacuum_layer=vacuum_layer, gamma=gamma, alpha=alpha, k_r=k_r,
         pressure_path=np.array(pressure_path), cg_per_solve=np.array(cg_counter['solves']),
         **{key: np.array(value) for key, value in history.items()})
print(f'total TR its {sum(history["nit"])}, CG its {sum(history["nb_cg"])}, time {time.time() - start_time:.1f} s')
print(f'wrote {data_folder_path}/{run_name}.npz')
