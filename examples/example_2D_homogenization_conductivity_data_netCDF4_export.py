import numpy as np
import time
#from netCDF4 import Dataset

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library

problem_type = 'conductivity'
discretization_type = 'finite_element'
element_type = 'linear_triangles' #  #'linear_triangles'# linear_triangles_tilled
#formulation = 'small_strain'
geometry_ID = 'square_inclusion'

domain_size = [1, 1]
number_of_pixels = (128,128)

my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)
start_time = time.time()

# set macroscopic gradient
macro_gradient = np.array([1.0, 0])

# create material data field
mat_contrast=1
mat_contrast_2=5
conductivity_C_1 = np.array([[1., 0], [0, 1.0]])

material_data_field_C_0 = np.einsum('ij,qxy->ijqxy', conductivity_C_1,
                                    np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                      *discretization.nb_of_pixels])))

# material distribution
phase_field = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                  microstructure_name=geometry_ID,
                                                  coordinates=discretization.fft.coords)

# apply material distribution
material_data_field_C_0_rho = mat_contrast*material_data_field_C_0[..., :, :] * np.power(phase_field,
                                                                            1)
material_data_field_C_0_rho += mat_contrast_2*material_data_field_C_0[..., :, :] * np.power(1-phase_field, 2)

# Set up the equilibrium system
macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)

# Solve mechanical equilibrium constrain
rhs = discretization.get_rhs(material_data_field_C_0_rho, macro_gradient_field)

K_fun = lambda x: discretization.apply_system_matrix(material_data_field_C_0_rho, x)
#M_fun = lambda x: 1 * x


preconditioner = discretization.get_preconditioner_NEW(reference_material_data_field_ijklqxyz=material_data_field_C_0)
# preconditioner_old = discretization.get_preconditioner(reference_material_data_field_ijklqxyz=material_data_field_C_0)

M_fun = lambda x: discretization.apply_preconditioner_NEW(preconditioner, x)
#M_fun_old= lambda x: discretization.apply_preconditioner(preconditioner_old, x)


#M_fun_NONE = lambda x: 1 * x
# x_0=np.random.rand(*discretization.get_unknown_size_field().shape)
# x_00=np.copy(x_0)
# x_1=M_fun_old(np.copy(x_0))
# x_12=M_fun_old(np.copy(x_1))
#
# x_2=M_fun(x_0)
# x_22=M_fun(x_2)

temperatute_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(500), toler=1e-12)
nb_it=len(norms['residual_rz'])
print(' nb_ steps CG =' f'{nb_it}')

# ----------------------------------------------------------------------
# compute homogenized stress field corresponding to displacement
homogenized_flux = discretization.get_homogenized_stress(
    material_data_field_ijklqxyz=material_data_field_C_0_rho,
    displacement_field_inxyz=temperatute_field,
    macro_gradient_field_ijqxyz=macro_gradient_field)

print(homogenized_flux)

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)

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
