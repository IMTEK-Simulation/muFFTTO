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

# Fixed structure: the solid has a fixed number of square pixels of fixed size. Vacuum pixels are
# added on the left and right only, so the domain grows with the vacuum, the pixels stay square and
# the solid (with its inclusion) is the same for every vacuum_thickness.
pixel_size = 1 / 32
nb_solid_pixels = (30, 32)
vacuum_thickness = int(os.environ.get('VACUUM', 1))  # number of vacuum pixels on each side (left / right)
number_of_pixels = (nb_solid_pixels[0] + 2 * vacuum_thickness, nb_solid_pixels[1])
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

# material distribution
geometry_ID = 'square_inclusion'

ix, iy = discretization.fft.icoords  # global pixel indices of the nodes of this rank
# geometry in coordinates of the solid, [0, 1) over the solid (not over the whole domain),
# so the inclusion does not move or grow when vacuum is added
solid_coords = np.stack([(ix - vacuum_thickness) / nb_solid_pixels[0],
                         iy / nb_solid_pixels[1]])
phase_field = discretization.get_scalar_field(name='phase_field')
phase_field.s[0, 0] = xp.asarray(microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                                     microstructure_name=geometry_ID,
                                                                     coordinates=solid_coords))
matrix_mask = phase_field.s[0, 0] > 0
inc_mask = phase_field.s[0, 0] == 0

# pixels on the boundary of the whole domain (MPI-safe); choose any of 'left', 'right', 'bottom', 'top'
domain_boundary_mask = discretization.get_domain_boundary_pixel_mask(boundaries=['left',
                                                                                 'right',
                                                                                 'bottom',
                                                                                 'top'
                                                                                 ],
                                                                     width=vacuum_thickness)

# initialize material data

mat_contrast = 0 # inclusion = void (0 is fine as long as no force acts on nodes touching only void pixels)
mat_contrast_2 = 1  # matrix
boundary_contrast = 0  # pixels on the domain boundary

lam, mu = material_models.get_lame_parameters(E=1, poisson=0.2)

lam_11qxyz = discretization.get_quad_field_scalar(name='lam_first_lame')
mu_11qxyz = discretization.get_quad_field_scalar(name='mu_second_lame')

# apply material distribution
lam_11qxyz.s[..., matrix_mask] = mat_contrast_2 * lam
lam_11qxyz.s[..., inc_mask] = mat_contrast * lam

mu_11qxyz.s[..., matrix_mask] = mat_contrast_2 * mu
mu_11qxyz.s[..., inc_mask] = mat_contrast * mu

# domain boundary last, so it overrides matrix / inclusion there
lam_11qxyz.s[..., domain_boundary_mask] = boundary_contrast * lam
mu_11qxyz.s[..., domain_boundary_mask] = boundary_contrast * mu

material = material_models.LinearElastic(discretization=discretization,
                                         lam_1qxyz=lam_11qxyz,
                                         mu_1qxyz=mu_11qxyz,
                                         name='linear_isotropic_elasticity')


# P = projection onto constrained nodes, Q = I - P onto free nodes (François notation)
# Mark the constrained dofs in a displacement-sized field: 1 = constrained, 0 = free.
# Select nodes by their GLOBAL pixel index (fft.icoords), not by slicing .s:
# under MPI, .s holds only the local subdomain, so .s[..., 4:10, 4:10] would mark different nodes on every rank.
block =xp.asarray((ix >= vacuum_thickness) & (ix < vacuum_thickness) & (iy >= 0) & (iy < number_of_pixels[1]/2))  # nodes x, y in 4:10

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
# set macroscopic gradient
macro_gradient_ij = np.zeros([dim, dim])  # E = 0: the cell is loaded only by the pressure in the void
# Set up right hand side
discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient_ij,
                                               macro_gradient_field_ijqxyz=macro_gradient_field)
# Solve mechanical equilibrium constrain
discretization.get_rhs_explicit_stress_mugrid(stress_function=material.get_stress,
                                              gradient_field_ijqxyz=macro_gradient_field,
                                              rhs_inxyz=rhs_field)

# imposed forces: pressure p in the void, acting on the void boundary from inside.
# The void is treated as a fluid with stress sigma = -p I; get_rhs_explicit_stress_mugrid gives
# -B^T W sigma = int_void B^T p I = int_boundary N p n -> consistent nodal forces on the void boundary
# (sum to zero, no need to find boundary nodes or normals)
void_pressure = 0.1


def void_pressure_stress(gradient_ijqxyz, stress_ijqxyz):
    stress_ijqxyz.s[...] = 0
    for d in range(dim):
        stress_ijqxyz.s[d, d][:, inc_mask] = -void_pressure


pressure_force_field = discretization.get_unknown_size_field(name='pressure_force_field')
discretization.get_rhs_explicit_stress_mugrid(stress_function=void_pressure_stress,
                                              gradient_field_ijqxyz=macro_gradient_field,
                                              rhs_inxyz=pressure_force_field)
rhs_field.s[...] += pressure_force_field.s

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
    rtol=1e-6,
    maxiter=2000,
    callback=callback)

# full displacement fluctuation u = Q(u) + P(u_D)
nodal_constraints.impose_prescribed_values(solution_field)

if discretization.communicator.size == 1:
    # Plot the first two components of the solution field
    try:

        # visualization_utils.plot_field_on_grid(
        #     coordinates_for_plot=x_plot_ixyz,
        #     field_to_plot=solution_field.s[0, 0],
        #     name = fr'$\tilde{{u}}_{{x}}$   ')
        # visualization_utils.plot_field_on_grid(
        #     coordinates_for_plot=x_plot_ixyz,
        #     field_to_plot=solution_field.s[1, 0],
        #     name=fr'$\tilde{{u}}_{{y}}$   ')
        # draw only pixels with nonzero stiffness: nodes touching only zero-stiffness pixels (void interior,
        # the cut domain boundary) have undefined displacements, and the periodic copy nodes added by
        # get_deformed_grid_coords_two_dim repeat those of the (undefined) seam nodes x = 0, y = 0
        # physical coordinates, so the non-square cell (solid + vacuum) is drawn to scale
        visualization_utils.plot_deformed_solid_two_dim(discretization,
                                                        macro_gradient_ij=macro_gradient_ij,
                                                        displacement_fluctuation=solution_field,
                                                        stiffness_field=mu_11qxyz,
                                                        field_to_plot=tensor_operations.to_host(lam_11qxyz.s[0, 0, 0]),
                                                        name=f'deformed solid, vacuum_thickness = {vacuum_thickness}')
    except:
        print(f"Plotting failed:  ")
# ----------------------------------------------------------------------
# average stress in the solid, int sigma(u) dx / |cell| (without the pressure -p I of the fluid in the void)
average_stress_ij = discretization.get_homogenized_stress_mugrid_explicit_stress(
    constitutive=material.get_stress,
    displacement_field_inxyz=solution_field,
    macro_gradient_field_ijqxyz=macro_gradient_field,
    formulation='small_strain')

if MPI.COMM_WORLD.rank == 0:
    print(
        "Average stress =\n" +
        np.array2string(np.asarray(average_stress_ij),
                        formatter={'float_kind': lambda x: f"{x:0.8f}"})
    )
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)

# ----------------------------------------------------------------------
# eigenvalues of the preconditioned system (Q M Q)(Q K Q), restricted to the free dofs.
# Dense: both operators are applied to every unit vector -> small grids and one MPI rank only.
compute_eigenvalues = True
if compute_eigenvalues and discretization.communicator.size == 1:
    import scipy.linalg
    import matplotlib.pyplot as plt

    hessp = nodal_constraints.project_operator(K_fun)
    prec = nodal_constraints.project_preconditioner(M_fun)
    free_dofs = np.flatnonzero(tensor_operations.to_host(constrained_dofs_field.s).ravel() == 0)
    nb_free_dofs = free_dofs.size
    unit_field = discretization.get_unknown_size_field(name='unit_field')
    output_field = discretization.get_unknown_size_field(name='output_field')


    def assemble_dense(operator):
        """Dense matrix of operator(x, Ax) on the free dofs, one column per unit vector."""
        matrix = np.zeros([nb_free_dofs, nb_free_dofs])
        for column, dof in enumerate(free_dofs):
            unit_field.s[...] = 0
            unit_field.s[np.unravel_index(dof, unit_field.s.shape)] = 1
            discretization.fft.communicate_ghosts(unit_field)
            operator(unit_field, output_field)
            matrix[:, column] = tensor_operations.to_host(output_field.s).ravel()[free_dofs]
        return matrix


    K_ff = assemble_dense(hessp)
    M_ff = assemble_dense(prec)
    K_ff = (K_ff + K_ff.T) / 2  # remove round-off asymmetry
    M_ff = (M_ff + M_ff.T) / 2

    # M K is similar to the symmetric M^1/2 K M^1/2 (M is symmetric positive semi-definite) -> real eigenvalues
    w_M, V_M = scipy.linalg.eigh(M_ff)
    M_half = (V_M * np.sqrt(np.clip(w_M, 0, None))) @ V_M.T
    eigenvalues_MK = scipy.linalg.eigvalsh(M_half @ K_ff @ M_half)


    def summarize(name, eigenvalues):
        # eigenvalues below 1e-10 * max are the null space (zero-stiffness pixels, rigid body modes)
        is_zero = np.abs(eigenvalues) < 1e-10 * np.max(np.abs(eigenvalues))
        nonzero = eigenvalues[~is_zero]
        print(f'{name}: {nb_free_dofs} free dofs, {np.sum(is_zero)} zero eigenvalues, '
              f'nonzero in [{nonzero.min():.4e}, {nonzero.max():.4e}], '
              f'effective condition number = {nonzero.max() / nonzero.min():.4e}')
        return nonzero, np.sum(is_zero)


    nonzero_MK, nb_zero_MK = summarize('M K  ', eigenvalues_MK)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.semilogy(np.sort(nonzero_MK), color='#2a78d6', linewidth=2)
    ax.set_xlabel('index of nonzero eigenvalue (sorted)')
    ax.set_ylabel(r'$\lambda$')
    ax.set_title(f'Spectrum of Q M Q K Q (Green), {number_of_pixels[0]}x{number_of_pixels[1]} pixels, '
                 f'{nb_zero_MK} zero eigenvalues not shown', fontsize=9)
    ax.grid(True, which='major', color='0.9', linewidth=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f'eigenvalues_preconditioned_system_w{vacuum_thickness}.png'))
    plt.show()
