import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import time
from mpi4py import MPI

from muGrid import Solvers

from muFFTTO import constraints
from muFFTTO import domain
from muFFTTO import microstructure_library
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
element_type = 'bilinear_rectangle'  # 'biquadratic_rectangle'#'linear_triangles'
formulation = 'small_strain'

domain_size = [1, 1]
number_of_pixels = (32, 32)

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

# material distribution
geometry_ID = 'square_inclusion'

phase_field = discretization.get_scalar_field(name='phase_field')
phase_field.s[0, 0] = xp.asarray(microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                                     microstructure_name=geometry_ID,
                                                                     coordinates=tensor_operations.to_host(
                                                                         discretization.fft.coords)))
matrix_mask = phase_field.s[0, 0] > 0
inc_mask = phase_field.s[0, 0] == 0

# initialize material data

mat_contrast = 0
mat_contrast_2 = 1

lam, mu = material_models.get_lame_parameters(E=1, poisson=0.2)

lam_11qxyz = discretization.get_quad_field_scalar(name='lam_first_lame')
mu_11qxyz = discretization.get_quad_field_scalar(name='mu_second_lame')

# apply material distribution
lam_11qxyz.s[..., matrix_mask] = mat_contrast_2 * lam
lam_11qxyz.s[..., inc_mask] = mat_contrast * lam

mu_11qxyz.s[..., matrix_mask] = mat_contrast_2 * mu
mu_11qxyz.s[..., inc_mask] = mat_contrast * mu

material = material_models.LinearElastic(discretization=discretization,
                                         lam_1qxyz=lam_11qxyz,
                                         mu_1qxyz=mu_11qxyz,
                                         name='linear_isotropic_elasticity')


# P = projection onto constrained nodes, Q = I - P onto free nodes (François notation)
# Mark the constrained dofs in a displacement-sized field: 1 = constrained, 0 = free.
# Select nodes by their GLOBAL pixel index (fft.icoords), not by slicing .s:
# under MPI, .s holds only the local subdomain, so .s[..., 4:10, 4:10] would mark different nodes on every rank.
ix, iy = discretization.fft.icoords  # global pixel indices of the nodes of this rank
block = xp.asarray((ix >= 4) & (ix < 5) & (iy >= 4) & (iy < 20))  # nodes x, y in 4:10

constrained_dofs_field = discretization.get_displacement_sized_field(name='constrained_dofs_field')
constrained_dofs_field.s[:, 0, block] = 1  # both components u_x, u_y (use [0, 0, block] for u_x only)

# prescribed displacement u_D; only the values at constrained dofs are used
prescribed_displacement = discretization.get_displacement_sized_field(name='prescribed_displacement')
prescribed_displacement.s[:, 0, block] = 0.0
# True : prescribed_displacement is the TOTAL displacement E.x + u_fluctuation -> nodes fixed in space
#        (the macro gradient is cancelled there; no longer a pure homogenization problem)
# False: prescribed_displacement is the displacement fluctuation -> nodes follow the macro deformation E.x
fix_total_displacement = False

# from_mask reads the two fields once and stores the indices of the constrained dofs,
# so applying Q later costs O(nb constrained dofs), not O(grid)
nodal_constraints = constraints.NodalConstraints.from_mask(discretization,
                                                           constrained_mask=constrained_dofs_field,
                                                           prescribed_values=prescribed_displacement)


def K_fun(x, Ax):
    discretization.apply_system_matrix_mugrid_explicit_stress(constitutive=material.get_stress,
                                                              input_field_inxyz=x,
                                                              output_field_inxyz=Ax,
                                                              formulation='small_strain')
    discretization.fft.communicate_ghosts(Ax)


# preconditioner
elastic_C_ref = material_models.get_elastic_tensor_from_lame(dim=discretization.domain_dimension,
                                                             lam=lam,
                                                             mu=mu)
preconditioner = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=elastic_C_ref)


def M_fun(x, Px):
    """
    Function to compute the product of the Preconditioner matrix with a vector.
    The Preconditioner is represented by the convolution operator.
    """
    discretization.fft.communicate_ghosts(x)
    discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                               input_nodal_field_fnxyz=x,
                                               output_nodal_field_fnxyz=Px)


# Allocate fields
macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
rhs_field = discretization.get_unknown_size_field(name='rhs_field')
solution_field = discretization.get_unknown_size_field(name='solution')


def callback(iteration, fields):
    """
    Callback function to print the current solution, residual, and search direction.
    """
    norm_of_rr = fields['rr']
    if discretization.communicator.rank == 0:
        print(f"{iteration:5} norm of residual = {norm_of_rr:.5}")


dim = discretization.domain_dimension
homogenized_C_ijkl = np.zeros(np.array(4 * [dim, ]))
# compute whole homogenized elastic tangent
i = 0
j = 1
# set macroscopic gradient
macro_gradient_ij = np.zeros([dim, dim])
macro_gradient_ij[i, j] = 0.2
# Set up right hand side
discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient_ij,
                                               macro_gradient_field_ijqxyz=macro_gradient_field)
# Solve mechanical equilibrium constrain
discretization.get_rhs_explicit_stress_mugrid(stress_function=material.get_stress,
                                              gradient_field_ijqxyz=macro_gradient_field,
                                              rhs_inxyz=rhs_field)

if fix_total_displacement:
    # u_D = u_total - E.x on the constrained dofs (call again whenever the macro gradient changes)
    nodal_constraints.set_prescribed_total_displacement(prescribed_displacement, macro_gradient_ij)
# right-hand side with constraints: Q(b) - Q K P(u_D)
nodal_constraints.apply_to_rhs(rhs_field, K_fun)

Solvers.conjugate_gradients(
    comm=discretization.communicator,
    fc=discretization.field_collection,
    hessp=nodal_constraints.project_operator(K_fun),  # linear operator Q K Q
    b=rhs_field,  # right-hand side
    x=solution_field,
    prec=nodal_constraints.project_preconditioner(M_fun),  # preconditioner Q M Q
    tol=1e-6,
    maxiter=2000,
    callback=callback)

# full displacement fluctuation u = Q(u) + P(u_D)
nodal_constraints.impose_prescribed_values(solution_field)

if discretization.communicator.size == 1:
    # Plot the first two components of the solution field
    try:
        x_plot_ixyz = visualization_utils.get_deformed_grid_coords_two_dim(discretization,
                                                                           macro_gradient_ij=macro_gradient_ij,
                                                                           displacement_fluctuation=solution_field)

        # visualization_utils.plot_field_on_grid(
        #     coordinates_for_plot=x_plot_ixyz,
        #     field_to_plot=solution_field.s[0, 0],
        #     name = fr'$\tilde{{u}}_{{x}}$   ')
        # visualization_utils.plot_field_on_grid(
        #     coordinates_for_plot=x_plot_ixyz,
        #     field_to_plot=solution_field.s[1, 0],
        #     name=fr'$\tilde{{u}}_{{y}}$   ')
        visualization_utils.plot_field_on_grid(
            coordinates_for_plot=x_plot_ixyz,
            field_to_plot=tensor_operations.to_host(matrix_mask),
            name=fr'$\tilde{{u}}_{{y}}$   ')
    except:
        print(f"Plotting failed:  ")
# ----------------------------------------------------------------------
# compute homogenized stress field corresponding
homogenized_C_ijkl[:, :, i, j] = discretization.get_homogenized_stress_mugrid_explicit_stress(
    constitutive=material.get_stress,
    displacement_field_inxyz=solution_field,
    macro_gradient_field_ijqxyz=macro_gradient_field,
    formulation='small_strain')

if MPI.COMM_WORLD.rank == 0:
    print(
        "Homogenized elastic tangent =\n" +
        np.array2string(material_models.compute_Voigt_notation_4order(homogenized_C_ijkl),
                        formatter={'float_kind': lambda x: f"{x:0.8f}"})
    )
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)
