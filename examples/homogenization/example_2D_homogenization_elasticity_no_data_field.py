import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import time
from mpi4py import MPI

from muGrid import Solvers

from muFFTTO import domain
from muFFTTO import microstructure_library
from muFFTTO import material_models

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'

domain_size = [1, 1]
number_of_pixels = (32, 32)

my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)
start_time = time.time()
print(f'{MPI.COMM_WORLD.rank:6} {MPI.COMM_WORLD.size:6} {str(discretization.fft.nb_domain_grid_pts):>15} '
      f'{str(discretization.fft.nb_subdomain_grid_pts):>15} {str(discretization.fft.subdomain_locations):>15}')

# initialize material data
lam_0, mu_0 = material_models.get_lame_parameters(E=1, poisson=0.2)
# create material data field
elastic_C_1 = material_models.get_elastic_tensor_from_lame(dim=discretization.domain_dimension,
                                                           lam=lam_0, mu=mu_0)
# create material data field
lam_1qxyz = discretization.get_quad_field_scalar(name='first_Lamé_parameter')
mu_1qxyz = discretization.get_quad_field_scalar(name='second_Lamé_parameter')

# material_data_field_C_0 = discretization.get_material_data_size_field_mugrid(name='elastic_tensor')


# material distribution
geometry_ID = 'square_inclusion'

phase_field = discretization.get_scalar_field(name='phase_field')
phase_field.s[0, 0] = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                          microstructure_name=geometry_ID,
                                                          coordinates=discretization.fft.coords)

mat_contrast = 1
mat_contrast_2 = 1e2
matrix_mask = phase_field.s[0, 0] > 0
inc_mask = phase_field.s[0, 0] == 0

# populate the field with  material data
lam_1qxyz.s[...] = lam_0
mu_1qxyz.s[...] = mu_0

lam_1qxyz.s[..., matrix_mask] = mat_contrast_2 * lam_0
mu_1qxyz.s[..., matrix_mask] = mat_contrast_2 * mu_0


def constitutive_model(strain, stress):
    material_models.linear_isotropic_elasticity_stress_from_strain_lame(strain_ijqxyz=strain,
                                                                        lam_1qxyz=lam_1qxyz,
                                                                        mu_1qxyz=mu_1qxyz,
                                                                        output_stress_ijqxyz=stress)


def K_fun(x, Ax):
    discretization.apply_system_matrix_mugrid_explicit_stress(constitutive=constitutive_model,
                                                              input_field_inxyz=x,
                                                              output_field_inxyz=Ax,
                                                              formulation='small_strain')

    discretization.fft.communicate_ghosts(Ax)


preconditioner = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=elastic_C_1)


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
for i in range(dim):
    for j in range(dim):
        # set macroscopic gradient
        macro_gradient_ij = np.zeros([dim, dim])
        macro_gradient_ij[i, j] = 1

        # Set up right hand side
        discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient_ij,
                                                       macro_gradient_field_ijqxyz=macro_gradient_field)
        # Solve mechanical equilibrium constrain
        discretization.get_rhs_explicit_stress_mugrid(stress_function=constitutive_model,
                                                      gradient_field_ijqxyz=macro_gradient_field,
                                                      rhs_inxyz=rhs_field)

        Solvers.conjugate_gradients(
            comm=discretization.communicator,
            fc=discretization.field_collection,
            hessp=K_fun,  # linear operator
            b=rhs_field,  # right-hand side
            x=solution_field,
            prec=M_fun,
            tol=1e-6,
            maxiter=2000,
            callback=callback)

        if discretization.communicator.size == 1:
            # Plot the first two components of the solution field
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 2, figsize=(12, 5))

            im0 = ax[0].pcolormesh(discretization.fft.coords[0],
                                   discretization.fft.coords[1],
                                   solution_field.s[0, 0],
                        vmin=-0.2, vmax=0.2)
            ax[0].set_title(f'Solution field $u_0$ - macro gradient {macro_gradient_ij} ')
            ax[0].set_xlabel('x  / L')
            ax[0].set_ylabel('y  / L')
            fig.colorbar(im0, ax=ax[0], label=rf'Displacement $u_{0}$')

            im1 = ax[1].pcolormesh(discretization.fft.coords[0],
                                   discretization.fft.coords[1],
                                   solution_field.s[1, 0],
                        vmin=-0.2, vmax=0.2)
            ax[1].set_title(f'Solution field $u_1$ - macro gradient {macro_gradient_ij} ')
            ax[1].set_xlabel('x  / L')
            ax[1].set_ylabel('y  / L')
            fig.colorbar(im1, ax=ax[1], label=rf'Displacement $u_{1}$')

            plt.tight_layout()
            plt.show()

        # ----------------------------------------------------------------------
        # compute homogenized stress field corresponding
        homogenized_C_ijkl[i, j] = discretization.get_homogenized_stress_mugrid_explicit_stress(
            constitutive=constitutive_model,
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
