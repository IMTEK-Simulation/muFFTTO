"""
Cantilever beam in a periodic cell.

A vertical beam is surrounded on all four sides by zero-stiffness (vacuum) pixels, so it is a free
body that does not touch its periodic images. The bottom row of beam nodes is clamped (prescribed
displacement, NodalConstraints) and a total force F is applied on the top row of beam nodes
(nodal forces added to the right-hand side). The macro gradient is zero, so the displacement
fluctuation is the total displacement.

Checks: tip deflection vs. Euler-Bernoulli / Timoshenko (plane strain), reaction force at the clamp = -F.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import time
from mpi4py import MPI

from muGrid import Solvers

from muFFTTO import constraints
from muFFTTO import domain
from muFFTTO import material_models
from muFFTTO import tensor_operations
from muFFTTO import visualization_utils

device = os.environ.get('DEVICE', 'cpu')  # 'cpu' or 'gpu'
if device == 'gpu':
    import cupy as xp
else:
    xp = np

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'bilinear_rectangle'
formulation = 'small_strain'

# Fixed structure: square pixels of fixed size; the beam has a fixed number of pixels, vacuum is added
# on all four sides, so the domain grows with the vacuum and the beam stays the same.
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
start_time = time.time()
print(f'{MPI.COMM_WORLD.rank:6} {MPI.COMM_WORLD.size:6} {str(discretization.fft.nb_domain_grid_pts):>15} '
      f'{str(discretization.fft.nb_subdomain_grid_pts):>15} {str(discretization.fft.subdomain_locations):>15}')

# beam pixels and nodes, by GLOBAL pixel index (MPI-safe); node (ix, iy) is the lower-left corner of pixel (ix, iy)
ix, iy = discretization.fft.icoords
beam_x = (vacuum_thickness, vacuum_thickness + nb_beam_pixels[0])  # first, last + 1 pixel in x
beam_y = (vacuum_thickness, vacuum_thickness + nb_beam_pixels[1])
beam_pixels = xp.asarray((ix >= beam_x[0]) & (ix < beam_x[1]) & (iy >= beam_y[0]) & (iy < beam_y[1]))
bottom_nodes = (ix >= beam_x[0]) & (ix <= beam_x[1]) & (iy == beam_y[0])
top_nodes = (ix >= beam_x[0]) & (ix <= beam_x[1]) & (iy == beam_y[1])

# material: beam with E, poisson (plane strain), vacuum with zero stiffness
E_young, poisson = 1.0, 0.2
lam, mu = material_models.get_lame_parameters(E=E_young, poisson=poisson)

lam_11qxyz = discretization.get_quad_field_scalar(name='lam_first_lame')
mu_11qxyz = discretization.get_quad_field_scalar(name='mu_second_lame')
lam_11qxyz.s[...] = 0
mu_11qxyz.s[...] = 0
lam_11qxyz.s[..., beam_pixels] = lam
mu_11qxyz.s[..., beam_pixels] = mu

material = material_models.LinearElastic(discretization=discretization,
                                         lam_1qxyz=lam_11qxyz,
                                         mu_1qxyz=mu_11qxyz,
                                         name='linear_isotropic_elasticity')

# clamp: both displacement components of the bottom row of beam nodes are fixed (u_D = 0)
constrained_dofs_field = discretization.get_displacement_sized_field(name='constrained_dofs_field')
constrained_dofs_field.s[:, 0, xp.asarray(bottom_nodes)] = 1
prescribed_displacement = discretization.get_displacement_sized_field(name='prescribed_displacement')
prescribed_displacement.s[...] = 0.0  # set nonzero values here to impose a displacement of the bottom

nodal_constraints = constraints.NodalConstraints.from_mask(discretization,
                                                           constrained_mask=constrained_dofs_field,
                                                           prescribed_values=prescribed_displacement)

# force on the top: total force F distributed as consistent nodal forces of a uniform edge load
# (weight h at inner nodes, h/2 at the two end nodes; the weights sum to the beam width)
total_force = np.array([1e-4, 0.0])  # (F_x, F_y), per unit thickness; tip deflection ~ 0.1 beam length
top_force_weights = np.where((ix == beam_x[0]) | (ix == beam_x[1]), 0.5, 1.0) / nb_beam_pixels[0]
force_field = discretization.get_unknown_size_field(name='force_field')
force_field.s[...] = 0
for d in range(discretization.domain_dimension):
    force_field.s[d, 0, xp.asarray(top_nodes)] = xp.asarray(total_force[d] * top_force_weights[top_nodes])


def K_fun(x, Ax):
    discretization.apply_system_matrix_mugrid_explicit_stress(constitutive=material.get_stress,
                                                              input_field_inxyz=x,
                                                              output_field_inxyz=Ax,
                                                              formulation='small_strain')
    discretization.fft.communicate_ghosts(Ax)


# Green preconditioner with the beam material as reference
elastic_C_ref = material_models.get_elastic_tensor_from_lame(dim=discretization.domain_dimension,
                                                             lam=lam,
                                                             mu=mu)
preconditioner = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=elastic_C_ref)


def M_fun(x, Px):
    discretization.fft.communicate_ghosts(x)
    discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                               input_nodal_field_fnxyz=x,
                                               output_nodal_field_fnxyz=Px)


rhs_field = discretization.get_unknown_size_field(name='rhs_field')
solution_field = discretization.get_unknown_size_field(name='solution')

nb_iterations = [0]


def callback(iteration, fields):
    nb_iterations[0] = iteration
    if discretization.communicator.rank == 0:
        print(f"{iteration:5} norm of residual = {fields['rr']:.5}")


# K u = f, no macro gradient: the right-hand side is the applied nodal force only
rhs_field.s[...] = force_field.s
# right-hand side with constraints: Q(f) - Q K P(u_D)
nodal_constraints.apply_to_rhs(rhs_field, K_fun)

Solvers.conjugate_gradients(
    comm=discretization.communicator,
    fc=discretization.field_collection,
    hessp=nodal_constraints.project_operator(K_fun),  # Q K Q
    b=rhs_field,
    x=solution_field,
    prec=nodal_constraints.project_preconditioner(M_fun),  # Q M Q
    rtol=1e-8,
    maxiter=5000,
    callback=callback)

# full displacement u = Q(u) + P(u_D)
nodal_constraints.impose_prescribed_values(solution_field)

# ----------------------------------------------------------------------
# checks
reduction = discretization.mpi_reduction
u_s = tensor_operations.to_host(solution_field.s)
# tip deflection: mean displacement of the top row of beam nodes
nb_top_nodes = reduction.sum(np.asarray(np.sum(top_nodes)))
tip_displacement = np.array([reduction.sum(np.asarray(np.sum(u_s[d, 0][top_nodes])))
                             for d in range(2)]) / nb_top_nodes

# reaction force of the clamp: R = (K u - f) on the constrained dofs (= K u there, f = 0 at the bottom)
Ku_field = discretization.get_unknown_size_field(name='Ku_field')
K_fun(solution_field, Ku_field)
Ku_s = tensor_operations.to_host(Ku_field.s - force_field.s)
reaction_force = np.array([reduction.sum(np.asarray(np.sum(Ku_s[d, 0][bottom_nodes]))) for d in range(2)])
# residual on the free dofs (should be ~0): K u - f with the constrained dofs removed
residual = Ku_field.s - force_field.s
residual[xp.asarray(constrained_dofs_field.s) != 0] = 0
max_free_residual = float(reduction.max(np.asarray(float(xp.max(xp.abs(residual))))))

# Euler-Bernoulli and Timoshenko (shear correction 5/6), plane strain, per unit thickness
beam_length = nb_beam_pixels[1] * pixel_size  # from the clamped row to the loaded row
beam_width = nb_beam_pixels[0] * pixel_size
E_plane_strain = E_young / (1 - poisson ** 2)
second_moment = beam_width ** 3 / 12
deflection_EB = total_force[0] * beam_length ** 3 / (3 * E_plane_strain * second_moment)
deflection_Timoshenko = deflection_EB + total_force[0] * beam_length / (5 / 6 * mu * beam_width)

if discretization.communicator.rank == 0:
    print(f'CG iterations                 = {nb_iterations[0]}')
    print(f'max free residual |Ku - f|    = {max_free_residual:.3e}')
    print(f'applied force F               = {total_force}')
    print(f'reaction force at the clamp R = {reaction_force}   (R + F = {reaction_force + total_force})')
    print(f'tip displacement (mean, top)  = {tip_displacement}')
    print(f'Euler-Bernoulli deflection    = {deflection_EB:.6e}   (FE / EB = {tip_displacement[0] / deflection_EB:.4f})')
    print(f'Timoshenko deflection         = {deflection_Timoshenko:.6e}   '
          f'(FE / Timoshenko = {tip_displacement[0] / deflection_Timoshenko:.4f})')

if discretization.communicator.size == 1:
    # bending stress sigma_yy, pixel average of the quadrature points
    strain_field = discretization.get_gradient_size_field(name='strain_field')
    stress_field = discretization.get_gradient_size_field(name='stress_field')
    discretization.apply_gradient_operator_symmetrized_mugrid(u_inxyz=solution_field, grad_u_ijqxyz=strain_field)
    material.get_stress(strain_field, stress_field)
    sigma_yy_xy = tensor_operations.to_host(stress_field.s[1, 1]).mean(axis=0)

    # displacement drawn at true scale (linear elasticity: scale the force to change the picture)
    deformation_scale = 1.0
    visualization_utils.plot_deformed_solid_two_dim(discretization,
                                                    macro_gradient_ij=np.zeros([2, 2]),
                                                    displacement_fluctuation=solution_field,
                                                    stiffness_field=mu_11qxyz,
                                                    field_to_plot=sigma_yy_xy,
                                                    name=r'cantilever, $\sigma_{yy}$',
                                                    deformation_scale=deformation_scale)

print("Elapsed time: ", time.time() - start_time)
