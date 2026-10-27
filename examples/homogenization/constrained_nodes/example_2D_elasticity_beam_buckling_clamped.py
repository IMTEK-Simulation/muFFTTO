"""
Buckling of a beam clamped at both ends under prescribed end shortening, finite strain, solved as an ENERGY
MINIMIZATION with the trust-region Newton-CG method (NuMPI tr_newton_bounded through solvers_nonlinear.minimize).

    Pi(u) = int_beam W(F) dx - f_imp . u,      F = I + grad u,   W neo-Hookean
    u = Q x + u_D(delta):  bottom row of beam nodes u = (0, 0),  top row u = (0, -delta)

Geometry as in example_2D_elasticity_beam_buckling.py: a vertical beam surrounded on all four sides by
zero-stiffness (vacuum) pixels. The top is moved down in increments (displacement control); a tiny horizontal force
at mid-height (imperfection) chooses the side to which the beam buckles.

Constraints: the optimizer only sees the free dofs (beam nodes without the bottom and top rows). The displacement is
u = Q x + u_D with the prescribed values u_D on the top row; fun returns Q grad, hessp Q H Q v and the preconditioner
Q M Q r. Nodes that touch only vacuum pixels are kept fixed (they carry no energy).

Euler critical load (clamped-clamped, plane strain): P_cr = 4 pi^2 E' I / L^2, reached at the end shortening
delta_cr = P_cr L / (E' A).
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
mid_nodes = beam_nodes & (iy == beam_y[0] + nb_beam_pixels[1] // 2)

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
# free dofs: beam nodes without the clamped bottom and top rows (Q = multiplication by free_dofs)
free_dofs = xp.zeros(discretization.get_unknown_size_field(name='u').s.shape)
free_dofs[:, 0, xp.asarray(beam_nodes & ~bottom_nodes & ~top_nodes)] = 1.0

beam_length = nb_beam_pixels[1] * pixel_size
beam_width = nb_beam_pixels[0] * pixel_size
E_plane_strain = E_young / (1 - poisson ** 2)
P_euler = 4 * np.pi ** 2 * E_plane_strain * beam_width ** 3 / 12 / beam_length ** 2  # clamped-clamped
delta_euler = P_euler * beam_length / (E_plane_strain * beam_width)  # end shortening at P_cr
delta_max = 0.5  # largest end shortening
nb_increments = 100

# prescribed displacement u_D: top row moves down by delta (set per increment), bottom row fixed
prescribed_field = discretization.get_unknown_size_field(name='prescribed_field')
prescribed_field.s[...] = 0

# imperfection: a small horizontal force at mid-height, distributed over the row as a uniform edge load
imperfection = 1e-3  # total horizontal force / P_cr
mid_force_weights = np.where((ix == beam_x[0]) | (ix == beam_x[1]), 0.5, 1.0) / nb_beam_pixels[0]
imperfection_field = discretization.get_unknown_size_field(name='imperfection_field')
imperfection_field.s[...] = 0
imperfection_field.s[0, 0, xp.asarray(mid_nodes)] = xp.asarray(imperfection * P_euler * mid_force_weights[mid_nodes])

# ----------------------------------------------------------------------
# fields
u_field = discretization.get_unknown_size_field(name='u_field')  # u = Q x + u_D
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

u_shape = u_field.s.shape
quad_weights = xp.asarray(discretization.quadrature_weights).reshape((-1,) + (1,) * dim)


def global_sum(a):
    return comm.sum(float(a))


def to_optimizer(array):
    """Flat numpy copy for the optimizer (host)."""
    return tensor_operations.to_host(array).ravel().copy()


def set_F(x):
    """u = Q x + u_D, F = I + grad u; returns min det F over the beam pixels (vacuum pixels are irrelevant)."""
    u_field.s[...] = xp.asarray(x).reshape(u_shape) * free_dofs + prescribed_field.s
    discretization.apply_gradient_operator_mugrid(u_inxyz=u_field, grad_u_ijqxyz=F_field)
    for d in range(dim):
        F_field.s[d, d] += 1.0
    tensor_operations.det2(F_field, J_field)
    J_beam = J_field.s[0, 0][:, beam_pixels]
    return float(discretization.mpi_reduction.min(np.asarray(float(J_beam.min()))))


def evaluate_stress(x):
    """First Piola-Kirchhoff stress at u = Q x + u_D, zero in the vacuum; returns min det F of the beam."""
    J_min = set_F(x)
    with np.errstate(invalid='ignore', divide='ignore'):  # ln J of (irrelevant) vacuum pixels
        material.get_stress(F_field, P_field)
    P_field.s[:, :, :, vacuum_pixels] = 0.0
    return J_min


# ----------------------------------------------------------------------
# energy, gradient, Hessian-vector product, preconditioner
delta = 0.0  # current end shortening, set by the load loop
INADMISSIBLE_ENERGY = 1e30  # finite (not inf): the trust region then rejects the step and shrinks

# CG iterations of every inner (Steihaug) solve: NuMPI calls hessp once per CG iteration and evaluates fun_grad once
# after every solve (at the trial point), so the Hessian products between two fun_grad calls are one solve
cg_counter = {'current': 0, 'solves': []}


def fun_grad(x):
    """(Pi, Q grad Pi) at u = Q x + u_D."""
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

    external_work = global_sum(xp.sum(imperfection_field.s * u_field.s))
    Pi = global_sum(xp.sum(quad_weights * energy_field.s[0, 0])) - external_work

    discretization.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=P_field,
                                                             div_u_fnxyz=grad_field,
                                                             apply_weights=True)
    grad_field.s[...] = (grad_field.s - imperfection_field.s) * free_dofs
    return Pi, to_optimizer(grad_field.s)


tangent_cache = {'x': None, 'delta': None}


def hessp(x, v):
    """Q H(Q x + u_D) Q v; the tangent C(F) is recomputed only when x or delta changes."""
    cg_counter['current'] += 1
    if tangent_cache['x'] is None or tangent_cache['delta'] != delta or not np.array_equal(tangent_cache['x'], x):
        set_F(x)
        with np.errstate(invalid='ignore', divide='ignore'):
            material.get_algorithmic_tangent(F_field, C_field)
        C_field.s[:, :, :, :, :, vacuum_pixels] = 0.0
        tangent_cache['x'] = np.array(x, copy=True)
        tangent_cache['delta'] = delta
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
# incremental end shortening
nb_mid_nodes = discretization.mpi_reduction.sum(np.asarray(np.sum(mid_nodes)))
# predictor of every increment: uniform compression of the free nodes, -d_delta (y - y_bottom) / L in u_y
uniform_compression = np.zeros(u_shape)
uniform_compression[1, 0] = -(iy - beam_y[0]) / nb_beam_pixels[1]
uniform_compression = (uniform_compression * tensor_operations.to_host(free_dofs)).ravel()

x = np.zeros(int(np.prod(u_shape)))
history = {'delta': [0.0], 'force': [0.0], 'mid_u_x': [0.0], 'min_J': [1.0], 'nit': [0], 'nhessp': [0], 'nb_cg': [0]}
solve_increment = []  # load increment of every inner CG solve
displacements = [np.zeros(u_shape)]  # host copy of u after every increment (serial plots / video)

if comm.rank == 0:
    print(f'preconditioner {preconditioner_type} (update: {preconditioner_update}), beam {nb_beam_pixels[0]}x'
          f'{nb_beam_pixels[1]} pixels, grid {number_of_pixels}, Euler P_cr = {P_euler:.4e} at delta = '
          f'{delta_euler:.4e}, shortening up to {delta_max} in {nb_increments} increments')
for increment in range(1, nb_increments + 1):
    delta_old, delta = delta, increment / nb_increments * delta_max
    prescribed_field.s[1, 0, xp.asarray(top_nodes)] = -delta
    x = x + (delta - delta_old) * uniform_compression
    update_preconditioner(x)
    result = solvers_nonlinear.minimize(fun=fun_grad, x0=x, method='tr_newton', hessp=hessp, precond=precond,
                                        gtol=1e-10, maxiter=500, inner_tol=1e-6, comm=MPI.COMM_WORLD,
                                        callback=update_preconditioner if preconditioner_update == 'iterate' else None)
    x = np.asarray(result.x)
    if cg_counter['current'] > 0:  # a last solve not followed by fun_grad
        cg_counter['solves'].append(cg_counter['current'])
        cg_counter['current'] = 0
    solve_increment.extend([increment] * (len(cg_counter['solves']) - len(solve_increment)))

    # compressive force on the top: minus the vertical internal force B^T W P of the top nodes
    min_J = evaluate_stress(x)
    discretization.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=P_field,
                                                             div_u_fnxyz=grad_field,
                                                             apply_weights=True)
    internal_force = tensor_operations.to_host(grad_field.s)
    force = -discretization.mpi_reduction.sum(np.asarray(np.sum(internal_force[1, 0][top_nodes])))
    u_s = tensor_operations.to_host(u_field.s)
    mid_u_x = discretization.mpi_reduction.sum(np.asarray(np.sum(u_s[0, 0][mid_nodes]))) / nb_mid_nodes

    history['delta'].append(delta)
    history['force'].append(float(force))
    history['mid_u_x'].append(float(mid_u_x))
    history['min_J'].append(min_J)
    history['nit'].append(int(result.nit))
    history['nhessp'].append(int(result.nhessp))
    history['nb_cg'].append(int(result.nhessp))  # NuMPI calls hessp once per inner CG iteration (its m_cg column)
    displacements.append(u_s.copy())
    if comm.rank == 0:
        print(f'increment {increment:3d}  delta = {delta:.4f}  P/P_cr = {force / P_euler:6.3f}  '
              f'mid u_x = {mid_u_x:+.4e}  min det F = {min_J:.4f}  TR its {result.nit:3d}  '
              f'CG its {history["nb_cg"][-1]:5d}  {"" if result.success else result.message}')

if comm.rank == 0:
    print(f'total TR iterations {sum(history["nit"])}, CG iterations {sum(history["nb_cg"])} '
          f'in {len(cg_counter["solves"])} solves (max {max(cg_counter["solves"])} per solve), '
          f'elapsed time {time.time() - start_time:.1f} s')

# load path and displacement of every increment, for make_beam_buckling_clamped_video.py (serial only)
if comm.size == 1:
    data_folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'exp_data',
                                    os.path.splitext(os.path.basename(__file__))[0])
    os.makedirs(data_folder_path, exist_ok=True)
    data_file = os.path.join(data_folder_path, f'beam_buckling_clamped_{preconditioner_type}.npz')
    np.savez(data_file, displacement=np.array(displacements)[:, :, 0],
             beam_pixels=beam_pixels_host, pixel_size=pixel_size, number_of_pixels=number_of_pixels,
             beam_x=beam_x, beam_y=beam_y, beam_length=beam_length, beam_width=beam_width,
             P_euler=P_euler, delta_euler=delta_euler, E_plane_strain=E_plane_strain,
             cg_per_solve=np.array(cg_counter['solves']), solve_increment=np.array(solve_increment),
             preconditioner=preconditioner_type, preconditioner_update=preconditioner_update,
             **{key: np.array(value) for key, value in history.items()})
    print(f'wrote {data_file}')

# ----------------------------------------------------------------------
# plots (serial: the fields are distributed under MPI)
if comm.size == 1:
    from scipy.special import ellipk, ellipe

    # exact inextensible elastica, clamped-clamped = four clamped-free quarters of length L/4: with p = sin(alpha / 2),
    # alpha = largest slope angle, P / P_cr = (2 K(p) / pi)^2, shortening / L = 2 - 2 E(p) / K(p),
    # mid-height deflection / L = p / K(p). The axial compression P L / (E' A) of the beam is added to the shortening.
    p = np.linspace(1e-4, 0.95, 400)
    K_p, E_p = ellipk(p ** 2), ellipe(p ** 2)
    elastica_P = (2 * K_p / np.pi) ** 2
    elastica_delta = (2 - 2 * E_p / K_p) + elastica_P * delta_euler / beam_length
    elastica_mid = p / K_p
    keep = elastica_delta <= delta_max / beam_length
    elastica_P = np.concatenate([[0.0, 1.0], elastica_P[keep]])
    elastica_delta = np.concatenate([[0.0, delta_euler / beam_length], elastica_delta[keep]])
    elastica_mid = np.concatenate([[0.0, 0.0], elastica_mid[keep]])

    P_ratio = np.array(history['force']) / P_euler
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].plot(elastica_delta, elastica_P, color='#eb6834', lw=2,
                 label='elastica + axial compression')
    axes[0].plot(np.array(history['delta']) / beam_length, P_ratio, 'o', color='#2a78d6', ms=4,
                 label='FE, neo-Hookean, trust-region Newton')
    axes[0].set_xlabel(r'end shortening $\delta / L$')
    axes[1].plot(elastica_mid, elastica_P, color='#eb6834', lw=2)
    axes[1].plot(np.array(history['mid_u_x']) / beam_length, P_ratio, 'o', color='#2a78d6', ms=4)
    axes[1].set_xlabel(r'mid-height deflection $u_x / L$')
    axes[0].legend(frameon=False, fontsize=8, loc='lower right')
    for ax in axes:
        ax.axhline(1.0, color='0.5', lw=1, ls='--')
        ax.set_ylabel(r'$P / P_{cr}^{Euler}$')
        ax.grid(True, color='0.9', lw=0.5)
    fig.suptitle(f'Beam clamped at both ends, {nb_beam_pixels[0]}x{nb_beam_pixels[1]} px, neo-Hookean, '
                 f'imperfection {imperfection:g} P_cr', fontsize=10)
    fig.tight_layout()

    # deformed beam at four end shortenings, true scale, physical coordinates
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
        top_y = beam_y[1] * pixel_size - history['delta'][increment]
        ax.plot(np.array([beam_x[0] - 1, beam_x[1] + 1]) * pixel_size, [top_y] * 2, 'k', lw=2)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect('equal')
        ax.set_title(f'delta / L = {history["delta"][increment] / beam_length:.2f}', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig2.suptitle('Deformed beam (true scale)', fontsize=10)
    fig2.tight_layout()

    # CG iterations of every inner solve, in the order of the solves
    fig3, ax3 = plt.subplots(figsize=(8, 3.4))
    solves = np.arange(1, len(cg_counter['solves']) + 1)
    ax3.bar(solves, cg_counter['solves'], width=0.85, color='#2a78d6')
    buckled = np.flatnonzero(np.array(history['delta'])[np.array(solve_increment)] >= delta_euler)
    if buckled.size:
        ax3.axvline(solves[buckled[0]] - 0.5, color='0.5', lw=1, ls='--')
        ax3.text(solves[buckled[0]], 0.97, r' $\delta \geq \delta_{cr}$', color='0.4', fontsize=8, va='top',
                 transform=ax3.get_xaxis_transform())
    ax3.set_xlabel('inner CG solve (trust-region iteration), in load order')
    ax3.set_ylabel('CG iterations per solve')
    ax3.grid(True, axis='y', color='0.9', lw=0.5)
    ax3.set_title(f'{preconditioner_type}: {len(solves)} solves, {sum(cg_counter["solves"])} CG iterations',
                  fontsize=10)
    fig3.tight_layout()
    plt.show()
