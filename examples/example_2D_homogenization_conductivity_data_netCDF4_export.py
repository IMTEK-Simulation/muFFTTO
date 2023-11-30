import numpy as np
import time
from netCDF4 import Dataset



from muFFTTO import domain
from muFFTTO import solvers

problem_type = 'conductivity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'

domain_size = [4, 3]
number_of_pixels = (24, 24)

my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       number_of_pixels=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)
start_time = time.time()

# set macroscopic gradient
macro_gradient = np.array([0.2, 0])

# create material data field
conductivity_C_1=100*np.array([[1.2, 0], [0, 1.0]])

material_data_field_C_0 = np.einsum('ij,qxy->ijqxy', conductivity_C_1,
                                    np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                      *discretization.nb_of_pixels])))

# material distribution
phase_field = np.random.rand(*discretization.get_scalar_sized_field().shape)  # set random distribution

# apply material distribution
material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * np.power(phase_field[0, 0],
                                                                            1)

# Set up the equilibrium system
macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)

# Solve mechanical equilibrium constrain
rhs = discretization.get_rhs(material_data_field_C_0_rho, macro_gradient_field)

K_fun = lambda x: discretization.apply_system_matrix(material_data_field_C_0_rho, x)
# M_fun = lambda x: 1 * x

preconditioner = discretization.get_preconditioner(reference_material_data_field=material_data_field_C_0)

M_fun = lambda x: discretization.apply_preconditioner(preconditioner, x)

tempetarute_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(500), toler=1e-12)

# ----------------------------------------------------------------------
# compute homogenized stress field corresponding to displacement
homogenized_flux = discretization.get_homogenized_stress(
    material_data_field_ijklqxyz=material_data_field_C_0_rho,
    displacement_field_fnxyz=tempetarute_field,
    macro_gradient_field_ijqxyz=macro_gradient_field)




print(homogenized_flux)

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)

nc = Dataset('temperatures.nc', 'w', format='NETCDF3_64BIT_OFFSET')
nc.createDimension('coords', 1)
nc.createDimension('number_of_dofs_x', number_of_pixels[0])
nc.createDimension('number_of_dofs_y', number_of_pixels[1])
nc.createDimension('number_of_dofs_per_pixel', 1)
nc.createDimension('time', None) # 'unlimited' dimension
var = nc.createVariable('temperatures', 'f8', ('time', 'coords','number_of_dofs_per_pixel','number_of_dofs_x', 'number_of_dofs_y'))
var[0, ...] = tempetarute_field[0,...]


print(homogenized_flux)
#var[0, ..., 0] = x
#var[0, ..., 1] = y



