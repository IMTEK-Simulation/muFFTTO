import numpy as np
import scipy as sp
import matplotlib as mpl

import matplotlib.pyplot as plt

from NuMPI import Optimization
from mpi4py import MPI
from muFFT import FFT

plt.rcParams['text.usetex'] = True

import time

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library

problem_type = 'elasticity'
discretization_type = 'finite_element'
# element_type = 'linear_triangles'
element_type = 'trilinear_hexahedron'
#
formulation = 'small_strain'

domain_size = [4, 3, 5]
number_of_pixels = (8, 8, 8)

my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

if MPI.COMM_WORLD.rank == 0:
    print('  Rank   Size          Domain       Subdomain        Location')
    print('  ----   ----          ------       ---------        --------')
MPI.COMM_WORLD.Barrier()  # Barrier so header is printed first
print(f'{MPI.COMM_WORLD.rank:6} {MPI.COMM_WORLD.size:6}  ')

# set up the system
my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)

print(f'{MPI.COMM_WORLD.rank:6} {MPI.COMM_WORLD.size:6} {str(discretization.fft.nb_domain_grid_pts):>15} '
      f'{str(discretization.fft.nb_subdomain_grid_pts):>15} {str(discretization.fft.subdomain_locations):>15}')

start_time = time.time()

## set macroscopic gradient
macro_gradient = np.array([[1.0, 0, 0], [0., 0.0, 0], [0, 0, 0.0]])

# create material data field
K_0, G_0 = domain.get_bulk_and_shear_modulus(E=1, poison=0.2)

elastic_C_1 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                 K=K_0,
                                                 mu=G_0,
                                                 kind='linear')
print(domain.compute_Voigt_notation_4order(elastic_C_1))
material_data_field_C_0 = np.einsum('ijkl,qxyz->ijklqxyz', elastic_C_1,
                                    np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                      *discretization.nb_of_pixels])))

# material distribution
phase_field = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels
                                                  , microstructure_name='random_distribution')

# apply material distribution
material_data_field_C_0_rho = material_data_field_C_0[..., :, :, :] * np.power(phase_field, 1)

# Set up right hand side
macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)

# Solve mechanical equilibrium constrain
rhs = discretization.get_rhs(material_data_field_C_0_rho, macro_gradient_field)

K_fun = lambda x: discretization.apply_system_matrix(material_data_field_C_0_rho, x,
                                                     formulation='small_strain')
# M_fun = lambda x: 1 * x

preconditioner = discretization.get_preconditioner(reference_material_data_field_ijklqxyz=material_data_field_C_0)
preconditioner_NEW = discretization.get_preconditioner_NEW(
    reference_material_data_field_ijklqxyz=material_data_field_C_0)

#np.sum(preconditioner[...,0:int(number_of_pixels[0]/2+1),:,:]-preconditioner_NEW)

M_fun = lambda x: discretization.apply_preconditioner(preconditioner_Fourier_fnfnqks=preconditioner,
                                                      nodal_field_fnxyz=x)

M_fun_NEW = lambda x: discretization.apply_preconditioner_NEW(preconditioner_Fourier_fnfnqks=preconditioner_NEW,
                                                              nodal_field_fnxyz=x)


x_0=np.random.rand(*discretization.get_unknown_size_field().shape)
x_00=np.copy(x_0)
x_1=M_fun_NEW(np.copy(x_0))
x_12=M_fun_NEW(np.copy(x_1))

x_2=M_fun(x_0)
x_22=M_fun(np.copy(x_2))

displacement_field_NEW, norms_NEW = solvers.PCG(K_fun, rhs, x0=None, P=M_fun_NEW, steps=int(500), toler=1e-6)


displacement_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(500), toler=1e-6)

# ----------------------------------------------------------------------
# compute homogenized stress field corresponding to displacement
homogenized_stress = discretization.get_homogenized_stress(
    material_data_field_ijklqxyz=material_data_field_C_0_rho,
    displacement_field_fnxyz=displacement_field,
    macro_gradient_field_ijqxyz=macro_gradient_field,
    formulation='small_strain')

print(homogenized_stress)
print(domain.compute_Voigt_notation_2order(homogenized_stress))

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)

start_time = time.time()
dim = discretization.domain_dimension
homogenized_C_ijkl = np.zeros(np.array(4 * [dim, ]))
