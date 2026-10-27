"""
C-box with self-contact under an end load, solved with the muFFTTO third-medium approach.

Reproduces Sec. 4.1.4 (Figs. 11-13) of
    P. Wriggers, J. Korelc, P. Junker, A third medium approach for contact using first and second order finite
    elements, Comput. Methods Appl. Mech. Engrg. 436 (2025) 117740 (benchmark of Bluhm et al.).
Only the problem is taken from the paper; the method is ours (see box_point_load.py).

Problem (paper):
    * two beams of length L = 1 and thickness T = 0.1 (y in [0, T] and [H - T, H], H = 0.5, initial gap g0 = 0.3),
      connected on the left by a wall x in [0, T]; the left face x = 0 is clamped;
    * the inside [T, L] x [T, H - T] and a column x in [L, L + 0.05] over the full height are third medium;
    * solid: K = 5/3, mu = 5/14 (E = 1, nu = 0.4), plane strain;
    * prescribed vertical displacement v = -0.5 lambda at the top right corner of the upper beam (x = L, y = H);
      lambda = 1 (Fig. 12, point contact) and lambda = 2 (Fig. 13, area contact); gamma = 1e-6.

Output: deformed shapes at lambda = 1 and 2, the smallest distance between the lower face of the upper beam and the
upper face of the lower beam (gap), the force at the load point, CG iterations.

usage:
    python cbox_point_load.py   # figures and data into exp_data/cbox_point_load/
    (NB_PER_UNIT, VACUUM_LAYER, MATERIAL, GAMMA, ALPHA, V_MAX, NB_INCREMENTS, PRECONDITIONER optional)
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
beam_length, box_height, wall, medium_column = 1.0, 0.5, 0.1, 0.05
box_length = beam_length + medium_column
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
beam_length_px = round(beam_length * nb_per_unit)
medium_pixels = (((ix >= x0 + wall_px) & (ix < x0 + beam_length_px) & (iy >= y0 + wall_px) &
                  (iy < y0 + box_px[1] - wall_px))  # between the beams
                 | ((ix >= x0 + beam_length_px) & (ix < x0 + box_px[0]) & (iy >= y0) & (iy < y0 + box_px[1])))
solid_pixels = box_pixels & ~medium_pixels
empty_pixels = ~box_pixels  # true vacuum
box_nodes = (ix >= x0) & (ix <= x0 + box_px[0]) & (iy >= y0) & (iy <= y0 + box_px[1])

clamped_nodes = box_nodes & (ix == x0)  # left face x = 0: u = 0
load_node = (ix == x0 + beam_length_px) & (iy == y0 + box_px[1])  # top right corner of the upper beam: u_y = v

# ----------------------------------------------------------------------
# material
K_bulk, mu = 5.0 / 3.0, 5.0 / 14.0
lam = K_bulk - 2.0 * mu / 3.0
gamma = float(os.environ.get('GAMMA', 1e-6))
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
# constraints: u = Q x + u_D
free_dofs = np.zeros(discretization.get_unknown_size_field(name='u').s.shape)
free_dofs[:, 0, box_nodes] = 1.0
free_dofs[:, 0, clamped_nodes] = 0.0
free_dofs[1, 0, load_node] = 0.0
prescribed_field = discretization.get_unknown_size_field(name='prescribed_field')
prescribed_field.s[...] = 0
v_max = float(os.environ.get('V_MAX', 1.0))  # = 0.5 lambda, lambda = 2
nb_increments = int(os.environ.get('NB_INCREMENTS', 40))


def set_load(v):
    prescribed_field.s[1, 0, load_node] = v


# ----------------------------------------------------------------------
# output
data_folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'exp_data',
                                os.path.splitext(os.path.basename(__file__))[0])
os.makedirs(data_folder_path, exist_ok=True)
run_name = f'n{nb_per_unit}_gamma{gamma:g}_alpha{alpha:g}_v{v_max:g}' + ('_decoupled' if material_type == 'decoupled'
                                                                         else '')


def plot_mesh(ax, u_nxy, title):
    """Pixels of the box drawn at their deformed corner positions: solid yellow, third medium red."""
    up = np.pad(u_nxy, ((0, 0), (0, 1), (0, 1)), mode='wrap')  # periodic copy nodes
    X = np.stack(np.meshgrid(np.arange(number_of_pixels[0] + 1), np.arange(number_of_pixels[1] + 1),
                             indexing='ij')) * pixel_size + up
    X = X - np.array([x0, y0])[:, None, None] * pixel_size  # box corner at the origin
    for pixels, colour in ((solid_pixels, '#efef9f'), (medium_pixels, '#efb99a')):
        polygons = [[X[:, i, j], X[:, i + 1, j], X[:, i + 1, j + 1], X[:, i, j + 1]] for i, j in np.argwhere(pixels)]
        ax.add_collection(PolyCollection(polygons, facecolors=colour, edgecolors='#5a5a8c', linewidths=0.3))
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)


if os.environ.get('SETUP_ONLY'):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    plot_mesh(ax, np.zeros((dim,) + tuple(number_of_pixels)), f'setup: {box_px[0]} x {box_px[1]} pixels in the box '
              f'(h = 1/{nb_per_unit}), grid {number_of_pixels[0]} x {number_of_pixels[1]} with {vacuum_layer} vacuum '
              f'pixels around')
    node_xy = lambda mask: (np.argwhere(mask)[0] - [x0, y0]) * pixel_size
    clamped_xy = (np.argwhere(clamped_nodes) - [x0, y0]) * pixel_size
    ax.plot(*clamped_xy.T, 's', color='#184f95', ms=4, label='clamped: u = 0')
    ax.plot(*node_xy(load_node), 'v', color='#e34948', ms=9, label=f'prescribed u_y = -{v_max:g} (lambda = 2)')
    ax.add_patch(plt.Rectangle((-vacuum_layer * pixel_size,) * 2, domain_size[0], domain_size[1], fill=False,
                               ls='--', ec='0.5', lw=1))
    ax.text(-vacuum_layer * pixel_size + 0.01, -vacuum_layer * pixel_size - 0.035, 'periodic cell (dashed)',
            color='0.4', fontsize=8, va='top')
    ax.autoscale()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(data_folder_path, 'setup.png'), dpi=200)
    print(f'wrote {os.path.join(data_folder_path, "setup.png")}')
    sys.exit()

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
    """u = Q x + u_D, F = I + grad u; returns (min det F in the solid, min det F in the medium)."""
    x_field.s[...] = np.asarray(x).reshape(u_shape) * free_dofs + prescribed_field.s
    discretization.apply_gradient_operator_mugrid(u_inxyz=x_field, grad_u_ijqxyz=F_field)
    for d in range(dim):
        F_field.s[d, d] += 1.0
    tensor_operations.det2(F_field, J_field)
    return (global_min(J_field.s[0, 0][:, solid_pixels].min()), global_min(J_field.s[0, 0][:, medium_pixels].min()))


def internal_force():
    """grad_field = B^T W P + R u at the current state."""
    with np.errstate(invalid='ignore', divide='ignore'):  # ln J of the (irrelevant) vacuum pixels
        material.get_stress(F_field, P_field)
    P_field.s[:, :, :, empty_pixels] = 0.0
    discretization.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=P_field, div_u_fnxyz=grad_field,
                                                             apply_weights=True)
    discretization.fft.communicate_ghosts(x_field)
    Ru_field.s[...] = 0.0
    add_regularization(x_field, Ru_field)
    grad_field.s[...] += Ru_field.s


INADMISSIBLE_ENERGY = 1e30  # finite: the trust region rejects the step and shrinks
cg_counter = {'current': 0, 'solves': []}  # CG iterations of every inner solve (Hessian products between fun calls)


def fun_grad(x):
    """(Pi, Q grad Pi) at u = Q x + u_D."""
    if cg_counter['current'] > 0:
        cg_counter['solves'].append(cg_counter['current'])
        cg_counter['current'] = 0
    J_min = min(set_F(x))
    if not np.isfinite(J_min) or J_min <= 0.0:
        return INADMISSIBLE_ENERGY, np.zeros(x.size)
    with np.errstate(invalid='ignore', divide='ignore'):
        material.get_energy_density(F_field, energy_field)
    energy_field.s[0, 0][:, empty_pixels] = 0.0
    internal_force()
    Pi = global_sum(np.sum(quad_weights * energy_field.s[0, 0])) + 0.5 * global_sum(np.sum(x_field.s * Ru_field.s))
    return Pi, (grad_field.s * free_dofs).ravel().copy()


tangent_cache = {'x': None, 'v': None}
current_load = {'v': 0.0}


def update_tangent(x):
    set_F(x)
    with np.errstate(invalid='ignore', divide='ignore'):
        material.get_algorithmic_tangent(F_field, C_field)
    C_field.s[:, :, :, :, :, empty_pixels] = 0.0


def apply_hessian(input_field_inxyz, output_field_inxyz):
    """(B^T W C B + R) input, with the cached tangent."""
    discretization.fft.communicate_ghosts(input_field_inxyz)
    discretization.apply_system_matrix_mugrid(material_data_field=C_field, input_field_inxyz=input_field_inxyz,
                                              output_field_inxyz=output_field_inxyz, formulation=formulation)
    add_regularization(input_field_inxyz, output_field_inxyz)
    discretization.fft.communicate_ghosts(output_field_inxyz)


def hessp(x, v):
    """Q H(Q x + u_D) Q v; the tangent is recomputed only when x or the load changes."""
    cg_counter['current'] += 1
    if tangent_cache['x'] is None or tangent_cache['v'] != current_load['v'] or not np.array_equal(tangent_cache['x'],
                                                                                                   x):
        update_tangent(x)
        tangent_cache['x'] = np.array(x, copy=True)
        tangent_cache['v'] = current_load['v']
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
# load stepping: every increment starts from the last converged state (plain trust-region Newton-CG); an increment
# whose start inverts elements or that does not converge is halved (cutback), the step then grows back
inner_columns = np.arange(x0 + wall_px, x0 + beam_length_px + 1)  # nodes of the inner faces of the two beams
inner_top_row, inner_bottom_row = y0 + box_px[1] - wall_px, y0 + wall_px


def surface_gap(u):
    """Smallest distance of a node of the lower face of the upper beam to the upper face of the lower beam."""
    X = np.stack(np.meshgrid(np.arange(number_of_pixels[0]), np.arange(number_of_pixels[1]), indexing='ij')) \
        * pixel_size + u
    top = X[:, inner_columns, inner_top_row].T
    bottom = X[:, inner_columns, inner_bottom_row].T
    a, b = bottom[:-1], bottom[1:]
    ab = b - a
    t = np.clip(np.einsum('psk,sk->ps', top[:, None] - a[None], ab) / np.einsum('sk,sk->s', ab, ab), 0.0, 1.0)
    return float(np.linalg.norm(top[:, None] - (a[None] + t[..., None] * ab[None]), axis=-1).min())


def measure():
    """Force at the loaded node, gap (current state in x_field / grad_field)."""
    reaction = -float(grad_field.s[1, 0][load_node].sum())
    return reaction, surface_gap(x_field.s[:, 0])


nb_dofs = int(np.prod(u_shape))
print(f'box {box_px[0]} x {box_px[1]} pixels (h = 1/{nb_per_unit}), grid {number_of_pixels}, {nb_dofs} dofs, '
      f'material {material_type}, gamma = {gamma:g}, k_r = {k_r:.3e}, preconditioner {preconditioner_type}, v_max = {v_max:g} in '
      f'{nb_increments} increments', flush=True)

start_time = time.time()
history = {'v': [0.0], 'reaction': [0.0], 'gap': [(box_px[1] - 2 * wall_px) * pixel_size], 'min_J_solid': [1.0],
           'min_J_medium': [1.0], 'nit': [0], 'nb_cg': [0], 'time': [0.0]}
displacements = [np.zeros((dim,) + tuple(number_of_pixels))]
x = np.zeros(nb_dofs)
v_done, step = 0.0, v_max / nb_increments
while v_done < v_max - 1e-12:
    v_current = min(v_done + step, v_max)
    set_load(-v_current)
    current_load['v'] = v_current
    x_start = x
    nb_solves_before = len(cg_counter['solves'])
    if min(set_F(x_start)) <= 0.0:  # moving the loaded node by the step inverts its elements: no energy there
        print(f'   cutback at v = {v_current:.4f} (step {step:.3g}): start inverts elements', flush=True)
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
        print(f'   cutback at v = {v_current:.4f} (step {step:.3g}): {result.message}, TR its {result.nit}', flush=True)
        step /= 2
        if step < v_max / nb_increments / 64:
            print('   step too small, stopping', flush=True)
            break
        continue
    x = np.asarray(result.x)
    v_done = v_current
    step = min(v_max / nb_increments, 1.5 * step)
    J_solid, J_medium = set_F(x)
    fun_grad(x)  # internal forces at the converged state
    reaction, gap = measure()
    nb_cg = int(np.sum(cg_counter['solves'][nb_solves_before:]))
    for key, value in zip(history, (v_done, reaction, gap, J_solid, J_medium, int(result.nit), nb_cg,
                                    time.time() - start_time)):
        history[key].append(value)
    displacements.append(x_field.s[:, 0].copy())
    print(f'v = {v_done:.4f}  reaction = {reaction:+.4e}  gap = {gap:.4e}  min det F solid {J_solid:.3f} medium '
          f'{J_medium:.4f}  TR its {result.nit:3d}  CG its {nb_cg:6d}  t = {history["time"][-1]:6.1f} s', flush=True)

np.savez(os.path.join(data_folder_path, f'{run_name}.npz'), displacement=np.array(displacements),
         solid_pixels=solid_pixels, medium_pixels=medium_pixels, pixel_size=pixel_size,
         number_of_pixels=number_of_pixels, vacuum_layer=vacuum_layer, gamma=gamma, alpha=alpha, k_r=k_r,
         load_node=np.argwhere(load_node)[0], cg_per_solve=np.array(cg_counter['solves']), **{key: np.array(value) for key, value in history.items()})

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# figures: deformed C-box at lambda = 1 (paper Fig. 12) and lambda = 2 (Fig. 13)
v_history = np.array(history['v'])
picks = [int(np.argmin(np.abs(v_history - v_target))) for v_target in (0.5 * v_max, v_max)]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout='constrained')
for ax, k in zip(axes, picks):
    plot_mesh(ax, displacements[k], f'lambda = {2 * history["v"][k]:.2f} (v = {history["v"][k]:.2f}): '
              f'gap = {history["gap"][k]:.2e}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
fig.savefig(os.path.join(data_folder_path, f'{run_name}_deformed.png'), dpi=200)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
axes[0].plot(history['v'], history['gap'], 'o-', color='#2a78d6', ms=3)
axes[0].set_yscale('log')
axes[0].set_xlabel('prescribed displacement v')
axes[0].set_ylabel('smallest gap between the beams')
axes[1].plot(history['v'], history['reaction'], 'o-', color='#2a78d6', ms=3)
axes[1].set_xlabel('prescribed displacement v')
axes[1].set_ylabel('reaction force at the load point')
axes[2].bar(np.arange(1, len(history['nb_cg'])), history['nb_cg'][1:], color='#2a78d6')
axes[2].set_xlabel('load increment')
axes[2].set_ylabel(f'CG its per increment ({preconditioner_type})')
for a in axes:
    a.grid(True, color='0.9', lw=0.5)
fig.tight_layout()
fig.savefig(os.path.join(data_folder_path, f'{run_name}_curves.png'), dpi=150)
print(f'total TR its {sum(history["nit"])}, CG its {sum(history["nb_cg"])}, time {time.time() - start_time:.1f} s')
print(f'wrote {data_folder_path}/{run_name}_*.png/.npz')
