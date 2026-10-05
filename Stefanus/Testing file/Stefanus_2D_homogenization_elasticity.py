import sys
import os

from matplotlib import pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import time
from mpi4py import MPI

from muGrid import Solvers

from muFFTTO import domain
from muFFTTO import microstructure_library
from muFFTTO import material_models
from muFFTTO import visualization_utils

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'bilinear_rectangle'# 'biquadratic_rectangle'#'linear_triangles'
formulation = 'small_strain'

domain_size = [1, 1]
number_of_pixels = (16,16)

my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)
start_time = time.time()
print(f'{MPI.COMM_WORLD.rank:6} {MPI.COMM_WORLD.size:6} {str(discretization.fft.nb_domain_grid_pts):>15} '
      f'{str(discretization.fft.nb_subdomain_grid_pts):>15} {str(discretization.fft.subdomain_locations):>15}')


# material distribution
geometry_ID = 'geometry_stefanus'

phase_field = discretization.get_scalar_field(name='phase_field')
phase_field.s[0, 0] = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                          microstructure_name=geometry_ID,
                                                          coordinates=discretization.fft.coords)

plt.pcolormesh(discretization.fft.coords[0], discretization.fft.coords[1], phase_field.s[0, 0])

plt.show()

# ORIGINAL Setup
# matrix_mask = phase_field.s[0, 0] > 0
# inc_mask = phase_field.s[0, 0] == 0
matrix_mask = phase_field.s[0, 0] == 0
inc_soft_mask = phase_field.s[0, 0] == 1
inc_stiff_mask = phase_field.s[0, 0] == 2


# initialize material data

K_0, G_0 = material_models.get_bulk_and_shear_modulus(E=1, poisson=0.2)
K_1, G_1 = material_models.get_bulk_and_shear_modulus(E=0.001, poisson=0.2)
K_2, G_2 = material_models.get_bulk_and_shear_modulus(E=10, poisson=0.2)

lam_0, mu_0 = material_models.get_lame_parameters_from_bulk_and_shear(K_0,
                                                                      G_0,
                                                                      dim=discretization.domain_dimension)
lam_1, mu_1 = material_models.get_lame_parameters_from_bulk_and_shear(K_1,
                                                                      G_1,
                                                                      dim=discretization.domain_dimension)
lam_2, mu_2 = material_models.get_lame_parameters_from_bulk_and_shear(K_2,
                                                                      G_2,
                                                                      dim=discretization.domain_dimension)

lam_11qxyz = discretization.get_quad_field_scalar(name='lam_first_lame')
mu_11qxyz = discretization.get_quad_field_scalar(name='mu_second_lame')

# apply material distribution
lam_11qxyz.s[..., matrix_mask] = lam_0
lam_11qxyz.s[..., inc_soft_mask] = lam_1
lam_11qxyz.s[..., inc_stiff_mask] = lam_2

mu_11qxyz.s[..., matrix_mask] = mu_0
mu_11qxyz.s[..., inc_soft_mask] = mu_1
mu_11qxyz.s[..., inc_stiff_mask] = mu_2

material = material_models.LinearElastic(discretization=discretization,
                                         lam_1qxyz=lam_11qxyz,
                                         mu_1qxyz=mu_11qxyz,
                                         name='linear_isotropic_elasticity')

material_data_field_C_0 = discretization.get_material_data_size_field_mugrid(name='elastic_tensor')

# populate the field with C_0 material
total_strain_ijqxyz = discretization.get_strain_sized_field(name='total_strain_field')

material.get_algorithmic_tangent(total_strain_ijqxyz, material_data_field_C_0)




def K_fun(x, Ax):

    discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0,
                                              input_field_inxyz=x,
                                              output_field_inxyz=Ax,
                                              formulation='small_strain')
    discretization.fft.communicate_ghosts(Ax)


# preconditioner
elastic_C_ref = material_models.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                 K=K_0,
                                                 mu=G_0,
                                                 kind='linear')
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
displacement_fluctuation_field = discretization.get_unknown_size_field(name='solution')


def callback(iteration, fields):
    """
    Callback function to print the current solution, residual, and search direction.
    """
    norm_of_rr = fields['rr']
    if discretization.communicator.rank == 0:
        print(f"{iteration:5} norm of residual = {norm_of_rr:.5}")

dim = discretization.domain_dimension
homogenized_C_ijkl = np.zeros(np.array(4 * [dim, ]))

# set macroscopic gradient
macro_gradient_ij = np.zeros([dim, dim])
macro_gradient_ij[0, 0] = 0.1

# Set up right hand side
discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient_ij,
                                               macro_gradient_field_ijqxyz=macro_gradient_field)
# Solve mechanical equilibrium constrain
discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0,
                              macro_gradient_field_ijqxyz=macro_gradient_field,
                              rhs_inxyz=rhs_field)

Solvers.conjugate_gradients(
    comm=discretization.communicator,
    fc=discretization.field_collection,
    hessp=K_fun,  # linear operator
    b=rhs_field,  # right-hand side
    x=displacement_fluctuation_field,
    prec=M_fun,
    tol=1e-6,
    maxiter=2000,
    callback=callback)

# strain from displacement increment
strain_fluc_field = discretization.get_strain_sized_field(name='strain_fluc_field')
discretization.apply_gradient_operator_symmetrized_mugrid(
    u_inxyz=displacement_fluctuation_field,
    grad_u_ijqxyz=strain_fluc_field,
)

# update total strain and displacement
total_strain_ijqxyz.s[...] = macro_gradient_field.s[...] + strain_fluc_field.s[...]

# re-evaluate constitutive response
total_stress_field = discretization.get_strain_sized_field(name='total_stress_field')

material.get_stress(total_strain_ijqxyz, total_stress_field)

Von_Mises_1nxyz = discretization.get_scalar_field(name='Von_Mises')

s = total_stress_field.s  # (2, 2, 4, 16, 32)
nu = 0.3  # Poisson's ratio

sxx = s[0, 0]
syy = s[1, 1]
sxy = s[0, 1]

szz = nu * (sxx + syy)  # out-of-plane stress in plane strain

vm = np.sqrt(
    0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
    + 3.0 * sxy ** 2
)

#  I want to plot the field/ Von Mises stress. I have four quad points , but matplotlib plots only one number per pixel # shape: (4, 16, 32)
Von_Mises_1nxyz.s[...] = np.mean(vm, axis=0)

if discretization.communicator.size == 1:
    # Plot the first two components of the solution field
    try:
        x_plot_ixyz = visualization_utils.get_deformed_grid_coords_two_dim(discretization,
                                                                           macro_gradient_ij=macro_gradient_ij,
                                                                           displacement_fluctuation=displacement_fluctuation_field)

        visualization_utils.plot_field_on_grid(
            coordinates_for_plot=x_plot_ixyz,
            field_to_plot=Von_Mises_1nxyz.s[0, 0],
            name=fr'$\tilde{{u}}_{{x}}$   ',
            plot_grid=False)
    except:
        print(f"Plotting failed:  ")

        plt.pcolormesh(discretization.fft.coords[0], discretization.fft.coords[1], Von_Mises_1nxyz.s[0, 0])
        plt.title(fr'Von_Mises_1nxyz   ')
        plt.show()

        plt.pcolormesh(discretization.fft.coords[0], discretization.fft.coords[1],
                       total_strain_ijqxyz.s[0, 0].mean(axis=0))
        plt.title(fr'Total strain x,x    ')
        plt.show()
        plt.pcolormesh(discretization.fft.coords[0], discretization.fft.coords[1],
                       total_strain_ijqxyz.s[0, 1].mean(axis=0))
        plt.title(fr'Total strain x,y    ')
        plt.show()

        # ----------------------------------------------------------------------
        # compute homogenized stress field corresponding
        homogenized_C_ijkl[0, 0] = discretization.get_homogenized_stress_mugrid(
            material_data_field_ijklqxyz=material_data_field_C_0,
            displacement_field_inxyz=displacement_fluctuation_field,
            macro_gradient_field_ijqxyz=macro_gradient_field,
            formulation='small_strain')

if MPI.COMM_WORLD.rank == 0:
    print(
        "Homogenized elastic tangent =\n" +
        np.array2string(material_models.compute_Voigt_notation_4order(homogenized_C_ijkl), formatter={'float_kind': lambda x: f"{x:0.8f}"})
    )
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)
