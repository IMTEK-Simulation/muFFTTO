import numpy as np
import time
# from netCDF4 import Dataset
import sys

sys.path.append('..')  # Add parent directory to path


from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library

problem_type = 'conductivity'
discretization_type = 'finite_element'
element_type = 'trilinear_hexahedron'

domain_size = [1, 1, 1]
number_of_pixels = 3 * (128,)

#geometry_ID = 'sine_wave_'
geometry_ID = 'square_inclusion'
# set up the system
my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)
start_time = time.time()

# create material data field
mat_contrast = 1
mat_contrast_2 = 1e2
conductivity_C_1 = 1 * np.array([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])

material_data_field_C_0 = discretization.get_material_data_size_field_mugrid(name=' conductivity_tensor')
material_data_field_C_0.s = np.einsum('ij,qxyz->ijqxyz', conductivity_C_1,
                                      np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                        *discretization.nb_of_pixels])))

# material distribution
phase_field = discretization.get_scalar_field(name='phase_field')

phase_field.s[0, 0] = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                  microstructure_name=geometry_ID,
                                                  coordinates=discretization.fft.coords)

matrix_mask = phase_field.s[0, 0] > 0
inc_mask = phase_field.s[0, 0] == 0

# apply material distribution
material_data_field_C_0.s[..., matrix_mask] = mat_contrast_2 * material_data_field_C_0.s[..., matrix_mask]
material_data_field_C_0.s[..., inc_mask] = mat_contrast * material_data_field_C_0.s[..., inc_mask]

# linear system
def K_fun(x, Ax):
    """
    Function to compute the product of the Hessian matrix with a vector.
    The Hessian is represented by the convolution operator.
    """

    discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0,
                                              input_field_inxyz=x,
                                              output_field_inxyz=Ax)
    discretization.fft.communicate_ghosts(Ax)



preconditioner = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=conductivity_C_1)

# preconditioner
def M_fun(x, Px):
    """
    Function to compute the product of the Preconditioner matrix with a vector.
    The Preconditioner is represented by the convolution operator.
    """
    discretization.fft.communicate_ghosts(x)
    discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                               input_nodal_field_fnxyz=x,
                                               output_nodal_field_fnxyz=Px)

dim = discretization.domain_dimension
homogenized_A_ij = np.zeros(np.array(2 * [dim, ]))
# compute whole homogenized elastic tangent

init_x_0 = discretization.get_unknown_size_field(name='init_solution')
solution_field = discretization.get_unknown_size_field(name='solution')
macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
rhs_field  = discretization.get_unknown_size_field(name='rhs_field')
for i in range(dim):
    # set macroscopic gradient
    macro_gradient = np.zeros([dim])
    macro_gradient[i] = 1
    # Set up right hand side
    macro_gradient_field.s.fill(0)
    discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient,
                                                   macro_gradient_field_ijqxyz=macro_gradient_field)
    discretization.fft.communicate_ghosts(field=macro_gradient_field)

    # Solve equilibrium constrain
    rhs_field.s.fill(0)
    discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0,
                                  macro_gradient_field_ijqxyz=macro_gradient_field,
                                  rhs_inxyz=rhs_field)
    # solver
    def callback(it, x, r, p):
        """
        Callback function to print the current solution, residual, and search direction.
        """
        norm_of_rr = discretization.fft.communicator.sum(np.dot(r.ravel(), r.ravel()))
        if discretization.fft.communicator.rank == 0:
            print(f"{it:5} norm of residual = {norm_of_rr:.5}")

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

    discretization.fft.communicate_ghosts(field=solution_field)

    # ----------------------------------------------------------------------
    # compute homogenized stress field corresponding to displacement
    homogenized_A_ij[i, :] = discretization.get_homogenized_stress_mugrid(
        material_data_field_ijklqxyz=material_data_field_C_0,
        displacement_field_inxyz=solution_field,
        macro_gradient_field_ijqxyz=macro_gradient_field)

print('homogenized elastic tangent = \n {}'.format(homogenized_A_ij))
end_time = time.time()
elapsed_time = end_time - start_time

print("Elapsed time: ", elapsed_time)
print("Elapsed time: ", elapsed_time / 60)
quit()
microstructure_library.visualize_voxels(phase_field_xyz=temperature_field.s[0, 0])

grad_field = discretization.get_gradient_size_field(name='solution_gradient')
discretization.apply_gradient_operator(u_inxyz=temperature_field,
                                       grad_u_ijqxyz=grad_field)

microstructure_library.visualize_voxels(phase_field_xyz=grad_field.s.mean(axis=2)[0, 0])
microstructure_library.visualize_voxels(phase_field_xyz=grad_field.s.mean(axis=2)[0, 1])
microstructure_library.visualize_voxels(phase_field_xyz=grad_field.s.mean(axis=2)[0, 2])
