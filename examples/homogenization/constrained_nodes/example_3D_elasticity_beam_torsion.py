"""
3D torsion of a clamped beam under a torque applied as dead forces on the top face, finite strain, solved as an ENERGY
MINIMIZATION with the trust-region Newton-CG method (NuMPI tr_newton_bounded through solvers_nonlinear.minimize).

    Pi(u) = int_beam W(F) dx - f . u,      F = I + grad u,   W neo-Hookean
    grad Pi = B^T W P(F) - f
    hess Pi = B^T W C(F) B

Geometry as in example_3D_elasticity_beam_buckling.py: a beam with a square cross-section along z, surrounded on all six
sides by zero-stiffness (vacuum) voxels, bottom face clamped. The top face carries tangential nodal forces
f_i = T w_i (-(y_i - y_c), x_i - x_c, 0) / sum_j w_j r_j^2  (w = consistent face weights), a pure torque T about the
beam axis (zero net force). The forces are DEAD loads: they keep their direction, so when the top face has rotated by
phi their moment about the axis is only T cos(phi) and the twist stays below 90 degrees.

Reference: Saint-Venant torsion of a square section, J = 0.1406 b^4, with the dead-load moment,
    T cos(phi) = G J phi / L      (rigid rotation of the top face, linear torsion, free warping).
The clamped bottom restrains warping, which makes the beam slightly stiffer near the clamp.

Constraints: the optimizer only sees the free dofs (beam nodes without the clamped face), Q is applied inside the
callbacks as in the buckling example.

usage:  DEVICE=gpu python example_3D_elasticity_beam_torsion.py      (PRECONDITIONER=Green_Jacobi optional)
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import time
from mpi4py import MPI
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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

# cubic voxels of fixed size; vacuum on all six sides
pixel_size = 1 / 32
nb_beam_pixels = (8, 8, 64)  # beam cross-section x, y and length z in voxels
vacuum_thickness = int(os.environ.get('VACUUM', 4))  # number of vacuum voxels on each side
number_of_pixels = tuple(n + 2 * vacuum_thickness for n in nb_beam_pixels)
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

# beam voxels and nodes, by GLOBAL voxel index (MPI-safe); node (ix, iy, iz) is the lower corner of voxel (ix, iy, iz)
ix, iy, iz = discretization.fft.icoords
beam_lo = [vacuum_thickness] * dim  # first beam voxel per direction
beam_hi = [vacuum_thickness + n for n in nb_beam_pixels]  # last beam voxel + 1 per direction
beam_pixels_host = ((ix >= beam_lo[0]) & (ix < beam_hi[0]) & (iy >= beam_lo[1]) & (iy < beam_hi[1])
                    & (iz >= beam_lo[2]) & (iz < beam_hi[2]))
beam_pixels = xp.asarray(beam_pixels_host)
vacuum_pixels = ~beam_pixels
beam_nodes = ((ix >= beam_lo[0]) & (ix <= beam_hi[0]) & (iy >= beam_lo[1]) & (iy <= beam_hi[1])
              & (iz >= beam_lo[2]) & (iz <= beam_hi[2]))
bottom_nodes = beam_nodes & (iz == beam_lo[2])
top_nodes = beam_nodes & (iz == beam_hi[2])

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
# free dofs: beam nodes without the clamped bottom face (Q = multiplication by free_dofs)
free_dofs = xp.zeros(discretization.get_unknown_size_field(name='u').s.shape)
free_dofs[:, 0, xp.asarray(beam_nodes & ~bottom_nodes)] = 1.0

# torque on the top face: tangential dead forces, consistent face weights (1/2 per boundary direction)
beam_length = nb_beam_pixels[2] * pixel_size
beam_width = nb_beam_pixels[0] * pixel_size
torsion_constant = 0.1406 * beam_width ** 4  # Saint-Venant J of a square section
torsional_stiffness = mu * torsion_constant / beam_length  # G J / L
T_max = 2.0 * torsional_stiffness  # linear twist of 2 rad; the dead-load moment T cos(phi) limits it to ~59 degrees
nb_increments = 50

face_weights = (np.where((ix == beam_lo[0]) | (ix == beam_hi[0]), 0.5, 1.0)
                * np.where((iy == beam_lo[1]) | (iy == beam_hi[1]), 0.5, 1.0))
centre = 0.5 * (beam_lo[0] + beam_hi[0]) * pixel_size  # beam axis at x = y = centre
rx, ry = ix * pixel_size - centre, iy * pixel_size - centre  # position of the nodes relative to the axis
polar_sum = np.sum((face_weights * (rx ** 2 + ry ** 2))[top_nodes])
unit_load_field = discretization.get_unknown_size_field(name='unit_load_field')  # nodal forces for T = 1
unit_load_field.s[...] = 0
unit_load_field.s[0, 0, xp.asarray(top_nodes)] = xp.asarray(-(face_weights * ry)[top_nodes] / polar_sum)
unit_load_field.s[1, 0, xp.asarray(top_nodes)] = xp.asarray((face_weights * rx)[top_nodes] / polar_sum)

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
    """F = I + grad(Q x); returns min det F over the beam voxels (vacuum voxels are irrelevant)."""
    x_field.s[...] = xp.asarray(x).reshape(u_shape) * free_dofs
    discretization.apply_gradient_operator_mugrid(u_inxyz=x_field, grad_u_ijqxyz=F_field)
    for d in range(dim):
        F_field.s[d, d] += 1.0
    tensor_operations.det2(F_field, J_field)
    J_beam = J_field.s[0, 0][:, beam_pixels]
    return float(discretization.mpi_reduction.min(np.asarray(float(J_beam.min()))))


# ----------------------------------------------------------------------
# energy, gradient, Hessian-vector product, preconditioner
load_factor = 0.0  # current torque T, set by the load loop
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
    with np.errstate(invalid='ignore', divide='ignore'):  # ln J of (irrelevant) vacuum voxels
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


def top_rotation(u_s):
    """Rotation angle of the top face about z (least squares over its nodes) and its mean axial displacement."""
    r0 = np.stack([rx[top_nodes], ry[top_nodes]])  # reference positions relative to the axis
    r = r0 + u_s[:2, 0][:, top_nodes]
    r = r - r.mean(axis=1, keepdims=True)  # relative to the deformed centroid
    cross = discretization.mpi_reduction.sum(np.asarray(np.sum(r0[0] * r[1] - r0[1] * r[0])))
    dot = discretization.mpi_reduction.sum(np.asarray(np.sum(r0[0] * r[0] + r0[1] * r[1])))
    u_z = discretization.mpi_reduction.sum(np.asarray(np.sum(u_s[2, 0][top_nodes])))
    return float(np.arctan2(cross, dot)), float(u_z) / nb_top_nodes


# ----------------------------------------------------------------------
# incremental loading
nb_top_nodes = discretization.mpi_reduction.sum(np.asarray(np.sum(top_nodes)))
u = np.zeros(int(np.prod(u_shape)))
history = {'T': [0.0], 'twist': [0.0], 'top_u_z': [0.0], 'min_J': [1.0], 'nit': [0], 'nhessp': [0], 'nb_cg': [0],
           'time': [0.0]}
solve_increment = []  # load increment of every inner CG solve
displacements = [np.zeros(u_shape)]  # host copy of u after every increment (serial plots / video)

if comm.rank == 0:
    print(f'device {device}, preconditioner {preconditioner_type} (update: {preconditioner_update}), beam '
          f'{nb_beam_pixels} voxels, grid {number_of_pixels} ({int(np.prod(u_shape))} dofs), '
          f'G J / L = {torsional_stiffness:.4e}, torque up to {T_max:.4e} in {nb_increments} increments')
for increment in range(1, nb_increments + 1):
    load_factor = increment / nb_increments * T_max
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
    twist, top_u_z = top_rotation(u_s)
    history['T'].append(load_factor)
    history['twist'].append(twist)
    history['top_u_z'].append(top_u_z)
    history['min_J'].append(min_J)
    history['nit'].append(int(result.nit))
    history['nhessp'].append(int(result.nhessp))
    history['nb_cg'].append(int(result.nhessp))  # NuMPI calls hessp once per inner CG iteration (its m_cg column)
    history['time'].append(time.time() - start_time)
    displacements.append(u_s.copy())
    if comm.rank == 0:
        print(f'increment {increment:3d}  T L/(G J) = {load_factor / torsional_stiffness:5.3f}  '
              f'twist = {np.degrees(twist):6.2f} deg  top u_z = {top_u_z:+.4e}  min det F = {min_J:.4f}  '
              f'TR its {result.nit:3d}  CG its {history["nb_cg"][-1]:5d}  t = {history["time"][-1]:6.1f} s  '
              f'{"" if result.success else result.message}')

if comm.rank == 0:
    print(f'total TR iterations {sum(history["nit"])}, CG iterations {sum(history["nb_cg"])} '
          f'in {len(cg_counter["solves"])} solves (max {max(cg_counter["solves"])} per solve), '
          f'elapsed time {time.time() - start_time:.1f} s')

# load path and displacement of every increment, for make_beam_torsion_3D_video.py (serial only)
if comm.size == 1:
    data_folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'exp_data',
                                    os.path.splitext(os.path.basename(__file__))[0])
    os.makedirs(data_folder_path, exist_ok=True)
    data_file = os.path.join(data_folder_path, f'beam_torsion_3D_{preconditioner_type}.npz')
    np.savez(data_file, displacement=np.array(displacements, dtype=np.float32)[:, :, 0],
             beam_pixels=beam_pixels_host, pixel_size=pixel_size, number_of_pixels=number_of_pixels,
             beam_lo=beam_lo, beam_hi=beam_hi, beam_length=beam_length, torsional_stiffness=torsional_stiffness,
             cg_per_solve=np.array(cg_counter['solves']), solve_increment=np.array(solve_increment),
             preconditioner=preconditioner_type, device=device,
             **{key: np.array(value) for key, value in history.items()})
    print(f'wrote {data_file}')


def outer_faces(beam_voxels):
    """Corner node indices [face, 4, 3] and outward direction (2 d + side) of the voxel faces on the beam surface."""
    faces, directions = [], []
    for d in range(3):
        a, b = [k for k in range(3) if k != d]
        for side in (0, 1):
            neighbour = np.roll(beam_voxels, -1 if side else 1, axis=d)
            for voxel in np.argwhere(beam_voxels & ~neighbour):
                corners = []
                for oa, ob in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    node = voxel.copy()
                    node[d] += side
                    node[a] += oa
                    node[b] += ob
                    corners.append(node)
                faces.append(corners)
                directions.append(2 * d + side)
    return np.array(faces), np.array(directions)


# ----------------------------------------------------------------------
# plots (serial: the fields are distributed under MPI)
if comm.size == 1:
    # Saint-Venant with the dead-load moment: T = (G J / L) phi / cos(phi)
    phi = np.linspace(0, np.radians(80), 300)
    reference_T = phi / np.cos(phi)
    T_ratio = np.array(history['T']) / torsional_stiffness

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].plot(np.degrees(phi), reference_T, color='#eb6834', lw=2,
                 label=r'Saint-Venant, dead load: $T\cos\varphi = GJ\varphi/L$')
    axes[0].plot(np.degrees(phi), phi, color='0.6', lw=1, ls='--', label=r'linear: $T = GJ\varphi/L$')
    axes[0].plot(np.degrees(history['twist']), T_ratio, 'o', color='#2a78d6', ms=4,
                 label='FE 3D, neo-Hookean, trust-region Newton')
    axes[0].set_xlabel(r'twist of the top face $\varphi$ [deg]')
    axes[0].set_ylabel(r'$T L / (G J)$')
    axes[0].set_xlim(0, np.degrees(max(history['twist'])) * 1.15)
    axes[0].set_ylim(0, T_ratio.max() * 1.1)
    axes[0].legend(frameon=False, fontsize=7.5, loc='upper left')
    axes[1].plot(np.degrees(history['twist']), np.array(history['top_u_z']) / beam_length, 'o-', color='#2a78d6',
                 ms=4, lw=1)
    axes[1].set_xlabel(r'twist of the top face $\varphi$ [deg]')
    axes[1].set_ylabel(r'axial displacement of the top $u_z / L$')
    for ax in axes:
        ax.grid(True, color='0.9', lw=0.5)
    fig.suptitle(f'3D torsion, beam {nb_beam_pixels[0]}x{nb_beam_pixels[1]}x{nb_beam_pixels[2]} voxels, '
                 f'neo-Hookean, torque as dead forces', fontsize=10)
    fig.tight_layout()

    # deformed beam surface at four torques, true scale; the four side faces in different colours show the twist
    faces, directions = outer_faces(beam_pixels_host)
    side_colours = np.array(['#86b6ef', '#eda100', '#1baf7a', '#e87ba4', '#b7d3f6', '#b7d3f6'])[directions]
    grid_nodes = np.stack(np.meshgrid(*[np.arange(n + 1) for n in number_of_pixels], indexing='ij')) * pixel_size
    snapshots = (nb_increments // 4, nb_increments // 2, 3 * nb_increments // 4, nb_increments)
    polygons = {}
    for increment in snapshots:
        up = np.pad(displacements[increment][:, 0], ((0, 0), (0, 1), (0, 1), (0, 1)), mode='wrap')
        X = grid_nodes + up  # deformed node coordinates [3, x, y, z]
        polygons[increment] = np.moveaxis(X[:, faces[..., 0], faces[..., 1], faces[..., 2]], 0, -1)  # [face, 4, 3]
    corners = np.concatenate([polygon.reshape(-1, 3) for polygon in polygons.values()])
    lo, hi = corners.min(axis=0) - 0.05 * beam_length, corners.max(axis=0) + 0.05 * beam_length
    fig2 = plt.figure(figsize=(1.9 * len(snapshots), 5.5))
    for k, increment in enumerate(snapshots):
        ax = fig2.add_subplot(1, len(snapshots), k + 1, projection='3d')
        ax.add_collection3d(Poly3DCollection(polygons[increment], facecolors=side_colours, edgecolors='0.3',
                                             linewidths=0.1))
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo, zoom=1.25)
        ax.view_init(elev=20, azim=-60)
        ax.set_axis_off()
        ax.set_title(f'twist {np.degrees(history["twist"][increment]):.0f} deg', fontsize=9)
    fig2.suptitle('Twisted beam (true scale)', fontsize=10)
    fig2.tight_layout()

    # CG iterations of every inner solve, in the order of the solves
    fig3, ax3 = plt.subplots(figsize=(8, 3.4))
    solves = np.arange(1, len(cg_counter['solves']) + 1)
    ax3.bar(solves, cg_counter['solves'], width=0.85, color='#2a78d6')
    ax3.set_xlabel('inner CG solve (trust-region iteration), in load order')
    ax3.set_ylabel('CG iterations per solve')
    ax3.grid(True, axis='y', color='0.9', lw=0.5)
    ax3.set_title(f'3D torsion, {preconditioner_type}: {len(solves)} solves, {sum(cg_counter["solves"])} CG '
                  f'iterations', fontsize=10)
    fig3.tight_layout()
    plt.show()
