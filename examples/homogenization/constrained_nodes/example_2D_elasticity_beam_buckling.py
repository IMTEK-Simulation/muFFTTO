"""
Buckling of a clamped beam under a compressive dead load, finite strain, solved as an ENERGY MINIMIZATION
with the trust-region Newton-CG method (NuMPI tr_newton_bounded through solvers_nonlinear.minimize).

    Pi(u) = int_beam W(F) dx - f . u,      F = I + grad u,   W neo-Hookean
    grad Pi = B^T W P(F) - f
    hess Pi = B^T W C(F) B

Geometry as in example_2D_elasticity_beam.py: a vertical beam surrounded on all four sides by zero-stiffness
(vacuum) pixels. The bottom row of beam nodes is clamped, a vertical load -P is applied on the top row. A small
horizontal load (imperfection) chooses the side to which the beam buckles.

Constraints: the optimizer only sees the free dofs. Q = projection onto the free dofs (beam nodes without the
clamped row) is applied inside the callbacks: fun returns Q grad, hessp returns Q H Q v and the preconditioner
Q M Q r (NuMPI does not combine its zero_mask with a preconditioner). Nodes that touch only vacuum pixels are
kept fixed as well (they carry no energy).

The trust-region method handles the indefinite Hessian of the straight beam above the critical load, where plain
Newton-CG fails. Euler critical load (clamped-free, plane strain): P_cr = pi^2 E' I / (4 L^2).
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import time
from mpi4py import MPI
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

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
element_type = 'bilinear_rectangle'
formulation = 'finite_strain'

# square pixels of fixed size; vacuum on all four sides
pixel_size = 1 / 32
nb_beam_pixels = (8, 64)  # beam width x height in pixels
vacuum_thickness = int(os.environ.get('VACUUM', 4))  # number of vacuum pixels on each side
number_of_pixels = (nb_beam_pixels[0] + 2 * vacuum_thickness, nb_beam_pixels[1] + 2 * vacuum_thickness)
domain_size = [number_of_pixels[0] * pixel_size, number_of_pixels[1] * pixel_size]

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

# beam pixels and nodes, by GLOBAL pixel index (MPI-safe); node (ix, iy) is the lower-left corner of pixel (ix, iy)
ix, iy = discretization.fft.icoords
beam_x = (vacuum_thickness, vacuum_thickness + nb_beam_pixels[0])  # first, last + 1 pixel in x
beam_y = (vacuum_thickness, vacuum_thickness + nb_beam_pixels[1])
beam_pixels_host = (ix >= beam_x[0]) & (ix < beam_x[1]) & (iy >= beam_y[0]) & (iy < beam_y[1])
beam_pixels = xp.asarray(beam_pixels_host)
vacuum_pixels = ~beam_pixels
beam_nodes = (ix >= beam_x[0]) & (ix <= beam_x[1]) & (iy >= beam_y[0]) & (iy <= beam_y[1])
bottom_nodes = beam_nodes & (iy == beam_y[0])
top_nodes = beam_nodes & (iy == beam_y[1])

# ----------------------------------------------------------------------
# material: neo-Hookean beam, zero stiffness in the vacuum
E_young, poisson = 1.0, 0.2
lam, mu = material_models.get_lame_parameters(E=E_young, poisson=poisson)

lam_field = discretization.get_quad_field_scalar(name='lam_field')
mu_field = discretization.get_quad_field_scalar(name='mu_field')
lam_field.s[...] = 0
mu_field.s[...] = 0
lam_field.s[..., beam_pixels] = lam
mu_field.s[..., beam_pixels] = mu

material = material_models.NeoHookean(discretization=discretization,
                                      lam_1qxyz=lam_field,
                                      mu_1qxyz=mu_field,
                                      name='neo_hookean_beam')

# ----------------------------------------------------------------------
# free dofs: beam nodes without the clamped bottom row (Q = multiplication by free_dofs)
free_dofs = xp.zeros(discretization.get_unknown_size_field(name='u').s.shape)
free_dofs[:, 0, xp.asarray(beam_nodes & ~bottom_nodes)] = 1.0

# dead load on the top row: consistent nodal forces of a uniform edge load (h/2 at the two end nodes)
beam_length = nb_beam_pixels[1] * pixel_size
beam_width = nb_beam_pixels[0] * pixel_size
E_plane_strain = E_young / (1 - poisson ** 2)
P_euler = np.pi ** 2 * E_plane_strain * beam_width ** 3 / 12 / (4 * beam_length ** 2)
P_max = 2.0 * P_euler  # largest compressive load
imperfection = 1e-3  # horizontal load / vertical load
nb_increments = 100

top_force_weights = np.where((ix == beam_x[0]) | (ix == beam_x[1]), 0.5, 1.0) / nb_beam_pixels[0]
unit_load_field = discretization.get_unknown_size_field(name='unit_load_field')  # nodal forces for P = 1
unit_load_field.s[...] = 0
unit_load_field.s[0, 0, xp.asarray(top_nodes)] = xp.asarray(imperfection * top_force_weights[top_nodes])
unit_load_field.s[1, 0, xp.asarray(top_nodes)] = xp.asarray(-1.0 * top_force_weights[top_nodes])

# ----------------------------------------------------------------------
# fields
x_field = discretization.get_unknown_size_field(name='x_field')
grad_field = discretization.get_unknown_size_field(name='grad_field')
v_field = discretization.get_unknown_size_field(name='v_field')
Hv_field = discretization.get_unknown_size_field(name='Hv_field')
r_field = discretization.get_unknown_size_field(name='r_field')
Mr_field = discretization.get_unknown_size_field(name='Mr_field')
F_field = discretization.get_displacement_gradient_sized_field(name='F')
P_field = discretization.get_displacement_gradient_sized_field(name='P')
C_field = discretization.get_material_data_size_field_mugrid(name='C')
energy_field = discretization.get_quad_field_scalar(name='energy_density')
J_field = discretization.get_quad_field_scalar(name='J')

u_shape = x_field.s.shape
quad_weights = xp.asarray(discretization.quadrature_weights).reshape((-1,) + (1,) * dim)


def global_sum(a):
    return comm.sum(float(a))


def to_optimizer(array):
    """Flat numpy copy for the optimizer (host)."""
    return tensor_operations.to_host(array).ravel().copy()


def set_F(x):
    """F = I + grad(Q x); returns min det F over the beam pixels (vacuum pixels are irrelevant)."""
    x_field.s[...] = xp.asarray(x).reshape(u_shape) * free_dofs
    discretization.apply_gradient_operator_mugrid(u_inxyz=x_field, grad_u_ijqxyz=F_field)
    for d in range(dim):
        F_field.s[d, d] += 1.0
    tensor_operations.det2(F_field, J_field)
    J_beam = J_field.s[0, 0][:, beam_pixels]
    return float(discretization.mpi_reduction.min(np.asarray(float(J_beam.min()))))


# ----------------------------------------------------------------------
# energy, gradient, Hessian-vector product, preconditioner
load_factor = 0.0  # current P, set by the load loop
INADMISSIBLE_ENERGY = 1e30  # finite (not inf): the trust region then rejects the step and shrinks


# CG iterations of every inner (Steihaug) solve: NuMPI calls hessp once per CG iteration and evaluates fun_grad once
# after every solve (at the trial point), so the Hessian products between two fun_grad calls are one solve
cg_counter = {'current': 0, 'solves': []}


def fun_grad(x):
    """(Pi, Q grad Pi) at Q x."""
    if cg_counter['current'] > 0:
        cg_counter['solves'].append(cg_counter['current'])
        cg_counter['current'] = 0
    J_min = set_F(x)
    if not np.isfinite(J_min) or J_min <= 0.0:
        return INADMISSIBLE_ENERGY, np.zeros(x.size)
    with np.errstate(invalid='ignore', divide='ignore'):  # ln J of (irrelevant) vacuum pixels
        material.get_energy_density(F_field, energy_field)
        material.get_stress(F_field, P_field)
    energy_field.s[0, 0][:, vacuum_pixels] = 0.0
    P_field.s[:, :, :, vacuum_pixels] = 0.0

    external_work = load_factor * global_sum(xp.sum(unit_load_field.s * x_field.s))
    Pi = global_sum(xp.sum(quad_weights * energy_field.s[0, 0])) - external_work

    discretization.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=P_field,
                                                             div_u_fnxyz=grad_field,
                                                             apply_weights=True)
    grad_field.s[...] = (grad_field.s - load_factor * unit_load_field.s) * free_dofs
    return Pi, to_optimizer(grad_field.s)


tangent_cache = {'x': None, 'load': None}


def hessp(x, v):
    """Q H(Q x) Q v; the tangent C(F) is recomputed only when x or the load changes."""
    cg_counter['current'] += 1
    if tangent_cache['x'] is None or tangent_cache['load'] != load_factor or not np.array_equal(tangent_cache['x'], x):
        set_F(x)
        with np.errstate(invalid='ignore', divide='ignore'):
            material.get_algorithmic_tangent(F_field, C_field)
        C_field.s[:, :, :, :, :, vacuum_pixels] = 0.0
        tangent_cache['x'] = np.array(x, copy=True)
        tangent_cache['load'] = load_factor
    v_field.s[...] = xp.asarray(v).reshape(u_shape) * free_dofs
    discretization.fft.communicate_ghosts(v_field)
    discretization.apply_system_matrix_mugrid(material_data_field=C_field,
                                              input_field_inxyz=v_field,
                                              output_field_inxyz=Hv_field,
                                              formulation=formulation)
    discretization.fft.communicate_ghosts(Hv_field)
    return to_optimizer(Hv_field.s * free_dofs)


# Preconditioner M of the inner (Steihaug) CG, applied as Q M Q:
#   'Green'        G = (B^T C_ref B)^-1, C_ref = beam tangent at F = I; fixed
#   'Green_Jacobi' D^-1/2 G_2muI D^-1/2, D = diag(B^T A B) of the current tangent A, reference 2 mu I
preconditioner_type = os.environ.get('PRECONDITIONER', 'Green')  # 'Green' or 'Green_Jacobi'
# when Green-Jacobi is rebuilt from the current tangent: 'increment' = at the start of every load increment
# (fixed within one trust-region solve, the trust region is measured in the M-norm), 'iterate' = also after every
# accepted trust-region iterate (optimizer callback, between two Steihaug solves, never inside one)
preconditioner_update = os.environ.get('PRECONDITIONER_UPDATE', 'iterate')

i2 = np.eye(dim)
I4rt = np.einsum('ik,jl->ijkl', i2, i2)
elastic_C_ref = material_models.get_elastic_tensor_from_lame(dim=dim, lam=lam, mu=mu)
# distinct names: two Green operators on one discretization must not share a field
green_C_ref = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=elastic_C_ref,
                                                             name='green_C_ref')
if preconditioner_type == 'Green_Jacobi':
    green_2muI = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=2 * mu * I4rt,
                                                                name='green_2muI')
jacobi_field = None


def update_preconditioner(x):
    """Green-Jacobi: D^-1/2 of the tangent at x."""
    global jacobi_field
    if preconditioner_type == 'Green':
        return
    set_F(x)
    with np.errstate(invalid='ignore', divide='ignore'):
        material.get_algorithmic_tangent(F_field, C_field)
    C_field.s[:, :, :, :, :, vacuum_pixels] = 0.0
    tangent_cache['x'] = None  # C_field was overwritten outside hessp
    # zero_threshold = 0: nodes with a zero diagonal (vacuum) are not moved by the preconditioner
    jacobi_field = discretization.get_preconditioner_Jacobi_mugrid(material_data_field_ijklqxyz=C_field,
                                                                   formulation=formulation, zero_threshold=0.0)


def apply_green(green, input_field, output_field):
    discretization.fft.communicate_ghosts(input_field)
    discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green,
                                               input_nodal_field_fnxyz=input_field,
                                               output_nodal_field_fnxyz=output_field)


def precond(r):
    """Q M Q r."""
    r_field.s[...] = xp.asarray(r).reshape(u_shape) * free_dofs
    if preconditioner_type == 'Green':
        apply_green(green_C_ref, r_field, Mr_field)
    else:  # Green-Jacobi
        r_field.s[...] *= jacobi_field.s
        apply_green(green_2muI, r_field, Mr_field)
        Mr_field.s[...] *= jacobi_field.s
    return to_optimizer(Mr_field.s * free_dofs)


# ----------------------------------------------------------------------
# incremental loading
nb_top_nodes = discretization.mpi_reduction.sum(np.asarray(np.sum(top_nodes)))
u = np.zeros(int(np.prod(u_shape)))
history = {'P': [0.0], 'tip_u_x': [0.0], 'tip_u_y': [0.0], 'min_J': [1.0], 'nit': [0], 'nhessp': [0], 'nb_cg': [0]}
solve_increment = []  # load increment of every inner CG solve
displacements = [np.zeros(u_shape)]  # host copy of u after every increment (serial plots / video)

if comm.rank == 0:
    print(f'preconditioner {preconditioner_type} (update: {preconditioner_update}), beam {nb_beam_pixels[0]}x{nb_beam_pixels[1]} pixels, '
          f'grid {number_of_pixels}, '
          f'Euler P_cr = {P_euler:.4e}, load up to {P_max:.4e} in {nb_increments} increments')
for increment in range(1, nb_increments + 1):
    load_factor = increment / nb_increments * P_max
    update_preconditioner(u)
    result = solvers_nonlinear.minimize(fun=fun_grad, x0=u, method='tr_newton', hessp=hessp, precond=precond,
                                        gtol=1e-10, maxiter=500, inner_tol=1e-6, comm=MPI.COMM_WORLD,
                                        callback=update_preconditioner if preconditioner_update == 'iterate' else None)
    u = np.asarray(result.x)
    if cg_counter['current'] > 0:  # a last solve not followed by fun_grad
        cg_counter['solves'].append(cg_counter['current'])
        cg_counter['current'] = 0
    solve_increment.extend([increment] * (len(cg_counter['solves']) - len(solve_increment)))

    min_J = set_F(u)
    u_s = tensor_operations.to_host(x_field.s)
    tip = [discretization.mpi_reduction.sum(np.asarray(np.sum(u_s[d, 0][top_nodes]))) / nb_top_nodes
           for d in range(dim)]
    history['P'].append(load_factor)
    history['tip_u_x'].append(tip[0])
    history['tip_u_y'].append(tip[1])
    history['min_J'].append(min_J)
    history['nit'].append(int(result.nit))
    history['nhessp'].append(int(result.nhessp))
    # NuMPI calls hessp once per inner (Steihaug) CG iteration (its m_cg column)
    history['nb_cg'].append(int(result.nhessp))
    displacements.append(u_s.copy())
    if comm.rank == 0:
        print(f'increment {increment:3d}  P/P_cr = {load_factor / P_euler:5.3f}  '
              f'tip u_x = {tip[0]:+.4e}  u_y = {tip[1]:+.4e}  min det F = {min_J:.4f}  '
              f'TR its {result.nit:3d}  CG its {history["nb_cg"][-1]:5d}  {"" if result.success else result.message}')

if comm.rank == 0:
    print(f'total TR iterations {sum(history["nit"])}, CG iterations {sum(history["nb_cg"])} '
          f'in {len(cg_counter["solves"])} solves (max {max(cg_counter["solves"])} per solve), '
          f'elapsed time {time.time() - start_time:.1f} s')

# load path and displacement of every increment, for make_beam_buckling_video.py (serial only)
if comm.size == 1:
    data_folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'exp_data',
                                    os.path.splitext(os.path.basename(__file__))[0])
    os.makedirs(data_folder_path, exist_ok=True)
    data_file = os.path.join(data_folder_path, f'beam_buckling_{preconditioner_type}.npz')
    np.savez(data_file, displacement=np.array(displacements)[:, :, 0],
             beam_pixels=beam_pixels_host, pixel_size=pixel_size, number_of_pixels=number_of_pixels,
             beam_x=beam_x, beam_y=beam_y, beam_length=beam_length, P_euler=P_euler,
             cg_per_solve=np.array(cg_counter['solves']), solve_increment=np.array(solve_increment),
             preconditioner=preconditioner_type, preconditioner_update=preconditioner_update,
             **{key: np.array(value) for key, value in history.items()})
    print(f'wrote {data_file}')

# ----------------------------------------------------------------------
# plots (serial: the fields are distributed under MPI)
if comm.size == 1:
    from scipy.special import ellipk, ellipe

    # exact inextensible elastica (clamped-free, vertical dead load): with p = sin(alpha / 2), alpha = tip angle,
    # P / P_cr = (2 K(p) / pi)^2,  u_x / L = 2 p / K(p),  -u_y / L = 2 - 2 E(p) / K(p)
    p = np.linspace(1e-4, np.sin(np.radians(150) / 2), 400)
    K_p, E_p = ellipk(p ** 2), ellipe(p ** 2)
    elastica = {'P': (2 * K_p / np.pi) ** 2, 'u_x': 2 * p / K_p, 'u_y': 2 - 2 * E_p / K_p}
    keep = elastica['P'] <= P_max / P_euler
    elastica = {key: np.concatenate([[0.0, 0.0] if key != 'P' else [0.0, 1.0], value[keep]])
                for key, value in elastica.items()}

    P_ratio = np.array(history['P']) / P_euler
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].plot(elastica['u_x'], elastica['P'], color='#eb6834', lw=2, label='elastica (exact, inextensible)')
    axes[0].plot(np.array(history['tip_u_x']) / beam_length, P_ratio, 'o', color='#2a78d6', ms=4,
                 label='FE, neo-Hookean, trust-region Newton')
    axes[0].set_xlabel(r'tip deflection $u_x / L$')
    axes[1].plot(elastica['u_y'], elastica['P'], color='#eb6834', lw=2)
    axes[1].plot(-np.array(history['tip_u_y']) / beam_length, P_ratio, 'o', color='#2a78d6', ms=4)
    axes[1].set_xlabel(r'tip shortening $-u_y / L$')
    axes[0].legend(frameon=False, fontsize=8, loc='lower right')
    for ax in axes:
        ax.axhline(1.0, color='0.5', lw=1, ls='--')
        ax.set_ylabel(r'$P / P_{cr}^{Euler}$')
        ax.grid(True, color='0.9', lw=0.5)
    axes[0].text(0.02, 1.03, 'Euler critical load', color='0.4', fontsize=8, transform=axes[0].get_yaxis_transform())
    fig.suptitle(f'Clamped beam {nb_beam_pixels[0]}x{nb_beam_pixels[1]} px, neo-Hookean, '
                 f'imperfection {imperfection:g}', fontsize=10)
    fig.tight_layout()

    # deformed beam at four loads, true scale, physical coordinates
    # (the beam may leave the periodic cell: the vacuum has no stiffness and there is no contact, so the
    # periodic images do not interact; only the beam pixels are drawn)
    X, Y = np.meshgrid(np.arange(number_of_pixels[0] + 1) * pixel_size,
                       np.arange(number_of_pixels[1] + 1) * pixel_size, indexing='ij')
    snapshots = {increment: displacements[increment] for increment in
                 (nb_increments // 4, nb_increments // 2, 3 * nb_increments // 4, nb_increments)}
    beam_polygons = {}
    for increment, u_s in snapshots.items():
        up = np.pad(u_s[:, 0], ((0, 0), (0, 1), (0, 1)), mode='wrap')
        Xd, Yd = X + up[0], Y + up[1]
        beam_polygons[increment] = np.array(
            [[(Xd[i, j], Yd[i, j]), (Xd[i + 1, j], Yd[i + 1, j]), (Xd[i + 1, j + 1], Yd[i + 1, j + 1]),
              (Xd[i, j + 1], Yd[i, j + 1])] for i, j in np.argwhere(beam_pixels_host)])
    corners = np.concatenate([polygons.reshape(-1, 2) for polygons in beam_polygons.values()])
    lo, hi = corners.min(axis=0) - 2 * beam_width, corners.max(axis=0) + 2 * beam_width
    fig2, axes2 = plt.subplots(1, len(snapshots),
                               figsize=(2.6 * len(snapshots), 2.6 * (hi[1] - lo[1]) / (hi[0] - lo[0]) + 0.8))
    for ax, (increment, polygons) in zip(np.atleast_1d(axes2), beam_polygons.items()):
        ax.add_collection(PolyCollection(polygons, facecolors='#86b6ef', edgecolors='#184f95', linewidths=0.2))
        ax.plot(np.array([beam_x[0] - 1, beam_x[1] + 1]) * pixel_size, [beam_y[0] * pixel_size] * 2, 'k', lw=2)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect('equal')
        ax.set_title(f'P/P_cr = {increment / nb_increments * P_max / P_euler:.2f}', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig2.suptitle('Deformed beam (true scale)', fontsize=10)
    fig2.tight_layout()

    # CG iterations of every inner solve, in the order of the solves
    fig3, ax3 = plt.subplots(figsize=(8, 3.4))
    solves = np.arange(1, len(cg_counter['solves']) + 1)
    ax3.bar(solves, cg_counter['solves'], width=0.85, color='#2a78d6')
    first_buckled = solves[np.array(solve_increment) >= nb_increments // 2][0]  # first solve at P >= P_cr
    ax3.axvline(first_buckled - 0.5, color='0.5', lw=1, ls='--')
    ax3.text(first_buckled, 0.97, r' $P \geq P_{cr}$', color='0.4', fontsize=8, va='top',
             transform=ax3.get_xaxis_transform())
    ax3.set_xlabel('inner CG solve (trust-region iteration), in load order')
    ax3.set_ylabel('CG iterations per solve')
    ax3.grid(True, axis='y', color='0.9', lw=0.5)
    ax3.set_title(f'{preconditioner_type}: {len(solves)} solves, {sum(cg_counter["solves"])} CG iterations',
                  fontsize=10)
    fig3.tight_layout()
    plt.show()
