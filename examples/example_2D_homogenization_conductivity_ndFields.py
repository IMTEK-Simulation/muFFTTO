import numpy as np
import time
# from netCDF4 import Dataset

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
number_of_pixels = 2*(1024, )

my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)
start_time = time.time()

# create material data field
mat_contrast = 1
mat_contrast_2 = 1e-2
conductivity_C_1 = np.array([[1., 0], [0, 1.0]])

material_data_field_C_0 = discretization.get_material_data_size_field(name='conductivity_tensor')
material_data_field_C_0 = material_data_field_C_0.s
# populate the field with C_1 material
material_data_field_C_0 = np.einsum('ij,qxy->ijqxy', conductivity_C_1,
                                      np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                        *discretization.nb_of_pixels])))

# material distribution
phase_field = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                  microstructure_name=geometry_ID,
                                                  coordinates=discretization.fft.coords)

# apply material distribution

material_data_field_C_0 = mat_contrast * material_data_field_C_0[..., :, :] * np.power(phase_field,
                                                                                           1)
material_data_field_C_0 += mat_contrast_2 * material_data_field_C_0[..., :, :] * np.power(1 - phase_field, 2)

# Set up the equilibrium system


K_fun = lambda x: discretization.apply_system_matrix(material_data_field=material_data_field_C_0,
                                                     displacement_field=x)
# xx__00 = np.zeros([1, discretization.nb_nodes_per_pixel, *discretization.nb_of_pixels])
# graddd = np.zeros(
#     [1, discretization.domain_dimension, discretization.nb_quad_points_per_pixel, *discretization.nb_of_pixels])
#
# graddd = discretization.apply_gradient_operator(u_inxyz=xx__00, grad_u_ijqxyz=graddd)
# aa = K_fun(xx__00)
# M_fun = lambda x: 1 * x
# K_matrix = discretization.get_system_matrix(material_data_field_C_0)
# preconditioner_old = discretization.get_preconditioner_DELETE(reference_material_data_field_ijklqxyz=material_data_field_C_0.s)

preconditioner = discretization.get_preconditioner_NEW(reference_material_data_ijkl=conductivity_C_1)
# preconditioner_old = discretization.get_preconditioner(reference_material_data_field_ijklqxyz=material_data_field_C_0)

M_fun = lambda x: discretization.apply_preconditioner_NEW(preconditioner, x)

# x_1 = K_fun(rhs)
# initial solution
init_x_0 = discretization.get_unknown_size_field(name='init_solution')
init_x_0 = init_x_0.s
solution_field = discretization.get_unknown_size_field(name='solution')
solution_field = solution_field.s
macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
# macro_gradient_field = macro_gradient_field.s
rhs_field = discretization.get_unknown_size_field(name='rhs_field')
# rhs_field = rhs_field.s

dim = discretization.domain_dimension
homogenized_A_ij = np.zeros(np.array(2 * [dim, ]))

for i in range(dim):
    # set macroscopic gradient
    macro_gradient = np.zeros([dim])
    macro_gradient[i] = 1

    macro_gradient_field.s.fill(0)
    macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient_ij=macro_gradient,
                                                                   macro_gradient_field_ijqxyz=macro_gradient_field)

    # Solve mechanical equilibrium constrain
    rhs_field.s.fill(0)
    rhs = discretization.get_rhs(material_data_field_ijklqxyz=material_data_field_C_0,
                                 macro_gradient_field_ijqxyz=macro_gradient_field,
                                 rhs_inxyz=rhs_field)

    init_x_0.fill(0)
    solution_field, norms = solvers.PCG(K_fun, rhs, x0=init_x_0, P=M_fun, steps=int(500), toler=1e-12)
    nb_it = len(norms['residual_rz'])
    print(' nb_ steps CG =' f'{nb_it}')
    # compute homogenized stress field corresponding to displacement
    homogenized_A_ij[i, :] = discretization.get_homogenized_stress(
        material_data_field_ijklqxyz=material_data_field_C_0,
        displacement_field_inxyz=solution_field,
        macro_gradient_field_ijqxyz=macro_gradient_field)

# ----------------------------------------------------------------------
print('homogenized conductivity tangent = \n {}'.format(homogenized_A_ij))

# compute homogenized stress field corresponding to displacement
homogenized_flux = discretization.get_homogenized_stress(
    material_data_field_ijklqxyz=material_data_field_C_0,
    displacement_field_inxyz=solution_field,
    macro_gradient_field_ijqxyz=macro_gradient_field)

print(homogenized_flux)

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)
print("Elapsed time: ", elapsed_time / 60)
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
