import sys

sys.path.append('..')  # Add parent directory to path

import numpy as np
import time
# from netCDF4 import Dataset


# from muGrid import Solvers
from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library
from mpi4py import MPI

problem_type = 'conductivity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'  # #'linear_triangles'# linear_triangles_tilled
# formulation = 'small_strain'
geometry_ID = 'square_inclusion'  # 'sine_wave_' #

domain_size = [1, 1]
number_of_pixels = (1024, 1024)

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
conductivity_C_1 = np.array([[1., 0], [0, 1.0]])

material_data_field_C_0 = discretization.get_material_data_size_field_mugrid(name='conductivity_tensor')

# populate the field with C_1 material
material_data_field_C_0.s = np.einsum('ij,qxy->ijqxy', conductivity_C_1,
                                      np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                        *discretization.nb_of_pixels])))

# material distribution
phase_field_geom = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                       microstructure_name=geometry_ID,
                                                       coordinates=discretization.fft.coords)

phase_field = discretization.get_scalar_field(name='phase_field')
phase_field.s[0, 0] = phase_field_geom
matrix_mask = phase_field_geom > 0
inc_mask = phase_field_geom == 0

# apply material distribution
# print(matrix_mask)
# material_data_field_C_0.s = mat_contrast * material_data_field_C_0.s[..., :, :] * np.power(phase_field,
#                                                                                            1)
# material_data_field_C_0.s += mat_contrast_2 * material_data_field_C_0.s[..., :, :] * np.power(1 - phase_field, 1)
material_data_field_C_0.s[..., matrix_mask] = mat_contrast_2 * material_data_field_C_0.s[..., matrix_mask]
material_data_field_C_0.s[..., inc_mask] = mat_contrast * material_data_field_C_0.s[..., inc_mask]


# material_data_field_C_0.sg += mat_contrast_2 * material_data_field_C_0.sg[..., :, :, :] * (1-phase_field.sg[0, 0])
# Set up the equilibrium system


# K_fun = lambda x: discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0,
#                                                      displacement_field=x)


def K_fun(x, Ax):
    """
    Function to compute the product of the Hessian matrix with a vector.
    The Hessian is represented by the convolution operator.
    """

    discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0,
                                              input_field_inxyz=x,
                                              output_field_inxyz=Ax)
    discretization.fft.communicate_ghosts(Ax)


# xx__00 = np.zeros([1, discretization.nb_nodes_per_pixel, *discretization.nb_of_pixels])
# graddd = np.zeros(
#     [1, discretization.domain_dimension, discretization.nb_quad_points_per_pixel, *discretization.nb_of_pixels])
#
# graddd = discretization.apply_gradient_operator(u_inxyz=xx__00, grad_u_ijqxyz=graddd)
# aa = K_fun(xx__00)
# M_fun = lambda x: 1 * x
# K_matrix = discretization.get_system_matrix(material_data_field_C_0)
# preconditioner_old = discretization.get_preconditioner_DELETE(reference_material_data_field_ijklqxyz=material_data_field_C_0.s)

#preconditioner = discretization.get_preconditioner_Green_fast(reference_material_data_ijkl=conductivity_C_1)
preconditioner = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=conductivity_C_1)


# preconditioner_old = discretization.get_preconditioner(reference_material_data_field_ijklqxyz=material_data_field_C_0)

# M_fun = lambda x: discretization.apply_preconditioner_NEW(preconditioner, x)


def M_fun(x, Px):
    """
    Function to compute the product of the Preconditioner matrix with a vector.
    The Preconditioner is represented by the convolution operator.
    """
    discretization.fft.communicate_ghosts(x)
    discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                               input_nodal_field_fnxyz=x,
                                               output_nodal_field_fnxyz=Px)
    #Px.s[...] =10* x.s[...]
    #print()

#  M_fun = lambda x: 1 * x

# x_1 = K_fun(rhs)
# initial solution
# init_x_0 = discretization.get_unknown_size_field(name='init_solution')
solution_field = discretization.get_unknown_size_field(name='solution')
# solution_field.s=np.random.rand(*solution_field.s.shape)
macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
rhs_field = discretization.get_unknown_size_field(name='rhs_field')

dim = discretization.domain_dimension
homogenized_A_ij = np.zeros(np.array(2 * [dim, ]))




for i in range(1):
    # set macroscopic gradient
    macro_gradient = np.zeros([dim])
    macro_gradient[i] = 1

    macro_gradient_field.sg.fill(0)
    discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient,
                                            macro_gradient_field_ijqxyz=macro_gradient_field)

   # print('rank' f'{MPI.COMM_WORLD.rank:6} material_data_field_C_0.s[0,0,0] =' f'{material_data_field_C_0.s[0,0,0]}')
    #print('rank' f'{MPI.COMM_WORLD.rank:6} macro_gradient_field.s[0,0,0] =' f'{macro_gradient_field.s[0, 0,0]}')
    discretization.fft.communicate_ghosts(field=macro_gradient_field)
    # Solve mechanical equilibrium constrain
    rhs_field.sg.fill(0)
    discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0,
                                  macro_gradient_field_ijqxyz=macro_gradient_field,
                                  rhs_inxyz=rhs_field)

    #print('rank' f'{MPI.COMM_WORLD.rank:6} rhs_field =' f'{rhs_field.s}')

    #    solution_field.sg, norms = solvers.PCG(K_fun, rhs.sg, x0=init_x_0.sg, P=M_fun, steps=int(1500), toler=1e-6)
    # solution_field.sg, norms = solvers.PCG(K_fun, rhs.sg, x0=init_x_0.sg, P=M_fun, steps=int(1500), toler=1e-6)
    def callback(it, x, r, p):
        """
        Callback function to print the current solution, residual, and search direction.
        """
        norm_of_rr=discretization.fft.communicator.sum(np.dot(r.ravel(), r.ravel()))
        if discretization.fft.communicator.rank == 0:
            print(f"{it:5} norm of residual = {norm_of_rr:.5}")

      #  print(f"{it:5} {discretization.fft.communicator.sum(np.dot(r.ravel(), r.ravel())):.5}")


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

    sum_sol=discretization.mpi_reduction.sum(solution_field.s,
                           axis=tuple(range(-3, 0)))
    print('rank' f'{MPI.COMM_WORLD.rank:6} sum_sol =' f'{sum_sol}')

    # nb_it = len(norms['residual_rz'])
    # print(solution_field.s)
    # print(' nb_ steps CG =' f'{nb_it}')
    # compute homogenized stress field corresponding to displacement
    homogenized_A_ij[i, :] = discretization.get_homogenized_stress_mugrid(
        material_data_field_ijklqxyz=material_data_field_C_0,
        displacement_field_inxyz=solution_field,
        macro_gradient_field_ijqxyz=macro_gradient_field)

    # ----------------------------------------------------------------------
    print('homogenized conductivity tangent = \n {}'.format(homogenized_A_ij))

# compute homogenized stress field corresponding to displacement
homogenized_flux = discretization.get_homogenized_stress_mugrid(
    material_data_field_ijklqxyz=material_data_field_C_0,
    displacement_field_inxyz=solution_field,
    macro_gradient_field_ijqxyz=macro_gradient_field)

print(homogenized_flux)

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time, 'seconds')
print("Elapsed time: ", elapsed_time / 60, 'minutes')
J_eff = mat_contrast_2 * np.sqrt((mat_contrast_2 + 3 * mat_contrast) / (3 * mat_contrast_2 + mat_contrast))
print("J_eff : ", J_eff)

J_eff = mat_contrast * np.sqrt((mat_contrast + 3 * mat_contrast_2) / (3 * mat_contrast + mat_contrast_2))
print("J_eff : ", J_eff)

# nc = Dataset('temperatures.nc', 'w', format='NETCDF3_64BIT_OFFSET')
# nc.createDimension('coords', 1)
# nc.createDimension('number_of_dofs_x', number_of_pixels[0])
# nc.createDimension('number_of_dofs_y', number_of_pixels[1])
# nc.createDimension('number_of_dofs_per_pixel', 1)
# nc.createDimension('time', None)  # 'unlimited' dimension
# var = nc.createVariable('temperatures', 'f8',
#                         ('time', 'coords', 'number_of_dofs_per_pixel', 'number_of_dofs_x', 'number_of_dofs_y'))
# var[0, ...] = temperatute_field[0, ...]
#
# print(homogenized_flux)
# # var[0, ..., 0] = x
# # var[0, ..., 1] = y
