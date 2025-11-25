import numpy as np
import time
import sys

sys.path.append('..')  # Add parent directory to path
from mpi4py import MPI

from muGrid import ConvolutionOperator
from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'trilinear_hexahedron'
formulation = 'small_strain'

domain_size = [4, 3, 5]
number_of_pixels = (64, 64,64)
geometry_ID = 'circle_inclusion'  # 'sine_wave_' #

my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)
start_time = time.time()
print(f'{MPI.COMM_WORLD.rank:6} {MPI.COMM_WORLD.size:6} {str(discretization.fft.nb_domain_grid_pts):>15} '
      f'{str(discretization.fft.nb_subdomain_grid_pts):>15} {str(discretization.fft.subdomain_locations):>15}')

# set macroscopic gradient
macro_gradient = np.array([[0.0, 0, 0], [0., 0.0, 0], [0, 0, 1.0]])

# create material data field
K_0, G_0 = domain.get_bulk_and_shear_modulus(E=1, poison=0.2)
mat_contrast = 1
mat_contrast_2 = 1e2

elastic_C_1 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                 K=K_0,
                                                 mu=G_0,
                                                 kind='linear')
print(domain.compute_Voigt_notation_4order(elastic_C_1))
material_data_field_C_0 = discretization.get_material_data_size_field_mugrid(name='elastic_tensor')

material_data_field_C_0.s = np.einsum('ijkl,qxyz->ijklqxyz', elastic_C_1,
                                      np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                        *discretization.nb_of_pixels])))
print('elastic tangent = \n {}'.format(domain.compute_Voigt_notation_4order(elastic_C_1)))

# material distribution
phase_field = discretization.get_scalar_field(name='phase_field')

phase_field.s[0,0] = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                    microstructure_name=geometry_ID,
                                                    coordinates=discretization.fft.coords)
# apply material distribution
matrix_mask = phase_field.s[0, 0] > 0
inc_mask = phase_field.s[0, 0] == 0
# material_data_field_C_0_rho = material_data_field_C_0[..., :, :, :] * np.power(phase_field+1, 1)
material_data_field_C_0.s[..., matrix_mask] = mat_contrast_2 * material_data_field_C_0.s[..., matrix_mask]
material_data_field_C_0.s[..., inc_mask] = mat_contrast * material_data_field_C_0.s[..., inc_mask]

# Set up right hand side
macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient,
                                               macro_gradient_field_ijqxyz=macro_gradient_field)
# macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)

# Solve mechanical equilibrium constrain
rhs_field = discretization.get_unknown_size_field(name='rhs_field')
discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0,
                             macro_gradient_field_ijqxyz=macro_gradient_field,
                             rhs_inxyz=rhs_field)

def K_fun(x, Ax):
    discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0,
                                              input_field_inxyz=x,
                                              output_field_inxyz=Ax,
                                              formulation='small_strain')
    discretization.fft.communicate_ghosts(Ax)
# M_fun = lambda x: 1 * x

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

def callback(it, x, r, p):
    """
    Callback function to print the current solution, residual, and search direction.
    """
    norm_of_rr = discretization.fft.communicator.sum(np.dot(r.ravel(), r.ravel()))
    if discretization.fft.communicator.rank == 0:
        print(f"{it:5} norm of residual = {norm_of_rr:.5}")

solution_field = discretization.get_unknown_size_field(name='solution')

solvers.conjugate_gradients_mugrid(
    comm=discretization.fft.communicator,
    fc=discretization.field_collection,
    hessp=K_fun,  # linear operator
    b=rhs_field,
    x=solution_field,
    P=M_fun,
    tol=1e-6,
    maxiter=2000,
    callback=callback,
)
# ----------------------------------------------------------------------
# compute homogenized stress field corresponding to displacement
homogenized_stress = discretization.get_homogenized_stress_mugrid(
    material_data_field_ijklqxyz=material_data_field_C_0,
    displacement_field_inxyz=solution_field,
    macro_gradient_field_ijqxyz=macro_gradient_field,
    formulation='small_strain')
print('homogenized stress = \n {}'.format(homogenized_stress))
print('homogenized stress in Voigt notation = \n {}'.format(domain.compute_Voigt_notation_2order(homogenized_stress)))

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)

quit()
start_time = time.time()
dim = discretization.domain_dimension
homogenized_C_ijkl = np.zeros(np.array(4 * [dim, ]))
# compute whole homogenized elastic tangent
for i in range(dim):
    for j in range(dim):
        # set macroscopic gradient
        macro_gradient = np.zeros([dim, dim])
        macro_gradient[i, j] = 1
        # Set up right hand side
        discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient,
                                                                       macro_gradient_field_ijqxyz=macro_gradient_field)

        # Solve mechanical equilibrium constrain
        discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0,
                                           macro_gradient_field_ijqxyz=macro_gradient_field,
                                           rhs_inxyz=rhs_field)

        solvers.conjugate_gradients_mugrid(
            comm=discretization.fft.communicator,
            fc=discretization.field_collection,
            hessp=K_fun,  # linear operator
            b=rhs_field,
            x=solution_field,
            P=M_fun,
            tol=1e-6,
            maxiter=2000,
            callback=callback,
        )
        # ----------------------------------------------------------------------
        # compute homogenized stress field corresponding
        homogenized_C_ijkl[i, j] = discretization.get_homogenized_stress_mugrid(
            material_data_field_ijklqxyz=material_data_field_C_0,
            displacement_field_inxyz=solution_field,
            macro_gradient_field_ijqxyz=macro_gradient_field,
            formulation='small_strain')

print('homogenized elastic tangent = \n {}'.format(domain.compute_Voigt_notation_4order(homogenized_C_ijkl)))
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)
