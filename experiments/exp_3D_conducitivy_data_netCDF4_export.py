import numpy as np
import time
from netCDF4 import Dataset

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library

problem_type = 'conductivity'
discretization_type = 'finite_element'
element_type = 'trilinear_hexahedron'
# formulation = 'small_strain'

domain_size = [1, 1, 1]
number_of_pixels = 3 * (40,)
geometry_ID = 'geometry_III_2_3D'

# set up the system
my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       number_of_pixels=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)
start_time = time.time()

# set macroscopic gradient
macro_gradient = np.array([[.0, 0, 1]])

# create material data field
conductivity_C_1 = 100 * np.array([[1, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 1.0]])

material_data_field_C_0 = np.einsum('ij,qxyz->ijqxyz', conductivity_C_1,
                                    np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                      *discretization.nb_of_pixels])))

# material distribution
phase_field = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                  microstructure_name=geometry_ID)

microstructure_library.visualize_voxels(phase_field_xyz=phase_field)

# apply material distribution
material_data_field_C_0_rho = material_data_field_C_0[..., :, :, :] * np.power(phase_field, 1)

# Set up right hand side
macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient=macro_gradient)

# Solve mechanical equilibrium constrain
rhs = discretization.get_rhs(material_data_field_ijklqxyz=material_data_field_C_0_rho,
                             macro_gradient_field_ijqxyz=macro_gradient_field)

# linear system
K_fun = lambda x: discretization.apply_system_matrix(material_data_field=material_data_field_C_0_rho,
                                                     displacement_field=x,
                                                     formulation='small_strain')
preconditioner = discretization.get_preconditioner(reference_material_data_field_ijklqxyz=material_data_field_C_0)

# preconditioner
M_fun = lambda x: discretization.apply_preconditioner(preconditioner_Fourier_fnfnxyz=preconditioner,
                                                      nodal_field_fnxyz=x)

# solver
temperature_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(500), toler=1e-6)

#
# ----------------- Postprocessing ----------------------------.
dataset_name = f'muFFTTO_{problem_type}_{geometry_ID}_N{discretization.nb_of_pixels[0]}_all.nc'

# create dataset
nc = Dataset(filename='exp_data/' + dataset_name,
             mode='w',
             format='NETCDF3_64BIT_OFFSET')

# create dimensions -- nicknames with sizes
nc.createDimension(dimname='temperature_shape', size=1)  # for temperature this is 1!!!
nc.createDimension(dimname='dimension', size=discretization.domain_dimension)

nc.createDimension(dimname='nb_nodes_per_pixel', size=discretization.nb_nodes_per_pixel)
nc.createDimension(dimname='nb_quad_per_pixel', size=discretization.nb_quad_points_per_pixel)

nc.createDimension(dimname='nb_voxels_x', size=number_of_pixels[0])
nc.createDimension(dimname='nb_voxels_y', size=number_of_pixels[1])
nc.createDimension(dimname='nb_voxels_z', size=number_of_pixels[2])

# crate a variable
temperature_var = nc.createVariable(varname='temperature_field', datatype='f8',
                                    dimensions=('temperature_shape',
                                                'nb_nodes_per_pixel',
                                                'nb_voxels_x', 'nb_voxels_y', 'nb_voxels_z'))

gradient_var = nc.createVariable(varname='gradient_field', datatype='f8',
                                 dimensions=('temperature_shape', 'dimension',
                                             'nb_quad_per_pixel',
                                             'nb_voxels_x', 'nb_voxels_y', 'nb_voxels_z'))

material_data_var = nc.createVariable(varname='material_data_field', datatype='f8',
                                      dimensions=('dimension', 'dimension',
                                                  'nb_quad_per_pixel',
                                                  'nb_voxels_x', 'nb_voxels_y', 'nb_voxels_z'))
phase_field_var = nc.createVariable(varname='phase_field', datatype='f8',
                                    dimensions=('nb_voxels_x', 'nb_voxels_y', 'nb_voxels_z'))

# fill the variable with data
temperature_var[...] = temperature_field
gradient_var[...] = discretization.apply_gradient_operator(temperature_field)
material_data_var[...] = material_data_field_C_0_rho
phase_field_var[...] = phase_field

nc.close()

# # read dataset
# loaded_dataset = Dataset(dataset_name)
#
# displacemet = loaded_dataset.variables['displacement_field']
