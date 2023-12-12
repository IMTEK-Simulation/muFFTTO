
import numpy as np
import time
from netCDF4 import Dataset

from muFFTTO import domain

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'trilinear_hexahedron'
formulation = 'small_strain'

domain_size = [4, 3, 5]
number_of_pixels = (6, 6, 6)
geometry_ID = 'geometry_1_3D'

# set up the system
my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       number_of_pixels=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)


###
#dataset_name = f'muFFTTO_{problem_type}_{formulation}_{geometry_ID}_N{discretization.nb_of_pixels[0]}_all.nc'
dataset_name = 'exp_data/'+f'muFFTTO_elasticity_small_strain_geometry_1_3D_N6_all.nc'


# # read dataset
loaded_dataset = Dataset(dataset_name)
#
displacemet = loaded_dataset.variables['displacement_field']

displacemet = loaded_dataset.variables['phase_field']


print(displacemet)
