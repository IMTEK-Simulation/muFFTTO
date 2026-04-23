import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import time
from mpi4py import MPI

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'

domain_size = [1, 1]
number_of_pixels = (128,128)

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
K_0, G_0 = domain.get_bulk_and_shear_modulus(E=1, poison=0.2)

# create material data field
elastic_C_1 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                 K=K_0,
                                                 mu=G_0,
                                                 kind='linear')
if discretization.communicator.rank == 0:
    print('elastic tangent = \n {}'.format(domain.compute_Voigt_notation_4order(elastic_C_1)))

material_data_field_C_0 = discretization.get_material_data_size_field_mugrid(name='elastic_tensor')

# populate the field with C_1 material
material_data_field_C_0.s[...] = elastic_C_1[:, :, :, :, np.newaxis, np.newaxis, np.newaxis]


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

# apply material distribution

material_data_field_C_0.s[..., matrix_mask] = mat_contrast_2 * material_data_field_C_0.s[..., matrix_mask]
material_data_field_C_0.s[..., inc_mask] = mat_contrast * material_data_field_C_0.s[..., inc_mask]


def K_fun(x, Ax):

    discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0,
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


def callback(it, x, r, p, z, stop_crit_norm):
    """
    Callback function to print the current solution, residual, and search direction.
    """
    norm_of_rr = discretization.communicator.sum(np.dot(r.ravel(), r.ravel()))
    if discretization.communicator.rank == 0:
        print(f"{it:5} norm of residual = {norm_of_rr:.5}")

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
        discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0,
                                      macro_gradient_field_ijqxyz=macro_gradient_field,
                                      rhs_inxyz=rhs_field)

        solvers.conjugate_gradients_mugrid(
            comm=discretization.communicator,
            fc=discretization.field_collection,
            hessp=K_fun,  # linear operator
            b=rhs_field,
            x=solution_field,
            P=M_fun,
            tol=1e-5,
            maxiter=1000,
            callback=callback,
        )
        if discretization.communicator.size == 1:
            # Plot the first two components of the solution field
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 2, figsize=(12, 5))

            im0 = ax[0].pcolormesh(discretization.fft.coords[0],
                                   discretization.fft.coords[1],
                                   solution_field.s[0, 0])
            ax[0].set_title(f'Solution field $u_0$ - macro gradient {macro_gradient_ij} ')
            ax[0].set_xlabel('x  / L')
            ax[0].set_ylabel('y  / L')
            fig.colorbar(im0, ax=ax[0], label=rf'Displacement $u_{0}$')

            im1 = ax[1].pcolormesh(discretization.fft.coords[0],
                                   discretization.fft.coords[1],
                                   solution_field.s[1, 0])
            ax[1].set_title(f'Solution field $u_1$ - macro gradient {macro_gradient_ij} ')
            ax[1].set_xlabel('x  / L')
            ax[1].set_ylabel('y  / L')
            fig.colorbar(im1, ax=ax[1], label=rf'Displacement $u_{1}$')

            plt.tight_layout()
            plt.show()

        # ----------------------------------------------------------------------
        # compute homogenized stress field corresponding
        homogenized_C_ijkl[i, j] = discretization.get_homogenized_stress_mugrid(
            material_data_field_ijklqxyz=material_data_field_C_0,
            displacement_field_inxyz=solution_field,
            macro_gradient_field_ijqxyz=macro_gradient_field,
            formulation='small_strain')

if MPI.COMM_WORLD.rank == 0:
    print(
        "Homogenized elastic tangent =\n" +
        np.array2string(domain.compute_Voigt_notation_4order(homogenized_C_ijkl), formatter={'float_kind': lambda x: f"{x:0.8f}"})
    )
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)
