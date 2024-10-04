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
discretization_type = 'finite_element'  # bilinear_rectangle
element_type = 'bilinear_rectangle'  # 'linear_triangles'

# element_type = 'trilinear_hexahedron'
#
formulation = 'small_strain'

domain_size = [1, 1]  # 4, 5
number_of_pixels = [1024,1024]  # 2 * (2,)
start_time =  MPI.Wtime()

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
                                       element_type=element_type,
                                       communicator=MPI.COMM_WORLD)

print(f'{MPI.COMM_WORLD.rank:6} {MPI.COMM_WORLD.size:6} {str(discretization.fft.nb_domain_grid_pts):>15} '
      f'{str(discretization.fft.nb_subdomain_grid_pts):>15} {str(discretization.fft.subdomain_locations):>15}')

## set macroscopic gradient
# macro_gradient = np.array([[1.0, 0, 0], [0., 0.0, 0], [0, 0, 0.0]])
macro_gradient = np.array([[1.0, 0], [0, 0.0]])
# create material data field
K_0, G_0 = domain.get_bulk_and_shear_modulus(E=1, poison=0.2)

elastic_C_1 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                 K=K_0,
                                                 mu=G_0,
                                                 kind='linear')
print(domain.compute_Voigt_notation_4order(elastic_C_1))
material_data_field_C_0 = np.einsum('ijkl,qxy...->ijklqxy...', elastic_C_1,
                                    np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                      *discretization.nb_of_pixels])))
K_1, G_1 = domain.get_bulk_and_shear_modulus(E=1e3, poison=0.4)

elastic_C_1 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                 K=K_1,
                                                 mu=G_1,
                                                 kind='linear')
material_data_field_C_1 = np.einsum('ijkl,qxy...->ijklqxy...', elastic_C_1,
                                    np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                      *discretization.nb_of_pixels])))
# material distribution

phase_field = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                  microstructure_name='circle_inclusion',
                                                  coordinates=discretization.fft.coords)
#phase_field[phase_field == 0]=1e-4
#phase_field[:, :phase_field.shape[1] // 2] = 1
#phase_field[:, phase_field.shape[1] // 2:] = 1
#print('rank' f'{MPI.COMM_WORLD.rank:6} phase_field=' f'{phase_field }')
# apply material distribution
material_data_field_C_0_rho = material_data_field_C_0[..., :, :, :] * np.power(phase_field,2)
material_data_field_C_0_rho += material_data_field_C_1[..., :, :, :] * np.power(1-phase_field, 2)


# Set up right hand side
macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)
#print('rank' f'{MPI.COMM_WORLD.rank:6} macro_gradient_field=' f'{macro_gradient_field }')
# Solve mechanical equilibrium constrain
rhs = discretization.get_rhs(material_data_field_C_0_rho, macro_gradient_field)
print(' Before apply K 'f'{MPI.COMM_WORLD.rank:6} {MPI.COMM_WORLD.size:6}  ')

K_fun = lambda x: discretization.apply_system_matrix(material_data_field=material_data_field_C_0_rho,
                                                     displacement_field=x,
                                                     formulation='small_strain')
# M_fun = lambda x: 1 * x
print('rank' f'{MPI.COMM_WORLD.rank:6}')
print(' Before getting K 'f'{MPI.COMM_WORLD.rank:6} {MPI.COMM_WORLD.size:6}  ')

#K_old = discretization.get_system_matrix(material_data_field=material_data_field_C_0_rho)
# print('rank' f'{MPI.COMM_WORLD.rank:6} K_old=' f'{K_old}')
#print('rank' f'{MPI.COMM_WORLD.rank:6} K_old shape=' f'{K_old.shape}')

#print(' Before Getting M 'f'{MPI.COMM_WORLD.rank:6} {MPI.COMM_WORLD.size:6}  ')

preconditioner_NEW = discretization.get_preconditioner_NEW(
    reference_material_data_field_ijklqxyz=material_data_field_C_0)

# print('rank' f'{MPI.COMM_WORLD.rank:6} preconditioner_NEW=' f'{preconditioner_NEW}')
# print('rank' f'{MPI.COMM_WORLD.rank:6} preconditioner_NEW=' f'{preconditioner_NEW.reshape(2,2,4)}')

#print('rank' f'{MPI.COMM_WORLD.rank:6} preconditioner_NEW shape=' f'{preconditioner_NEW.shape}')

# np.sum(preconditioner[...,0:int(number_of_pixels[0]/2+1),:,:]-preconditioner_NEW)


M_fun_NEW = lambda x: discretization.apply_preconditioner_NEW(preconditioner_Fourier_fnfnqks=preconditioner_NEW,
                                                              nodal_field_fnxyz=x)

#M_fun_NONE = lambda x: 1 * x
# x_0=np.random.rand(*discretization.get_unknown_size_field().shape)
# x_00=np.copy(x_0)
# x_1=M_fun_NEW(np.copy(x_0))
# x_12=M_fun_NEW(np.copy(x_1))
#
# x_2=M_fun_NEW(x_0)
# x_22=M_fun_NEW(x_2)
#print(' Before solving problem 'f'{MPI.COMM_WORLD.rank:6} {MPI.COMM_WORLD.size:6}  ')

displacement_field_NEW, norms_NEW = solvers.PCG(K_fun, rhs, x0=None, P=M_fun_NEW, steps=int(500), toler=1e-6)
#print('rank' f'{MPI.COMM_WORLD.rank:6} displacement_field_NEW =' f'{displacement_field_NEW}')
nb_it=len(norms_NEW['residual_rz'])
print('rank' f'{MPI.COMM_WORLD.rank:6} nb_ steps CG =' f'{nb_it}')

#print(' After solving problem 'f'{MPI.COMM_WORLD.rank:6} {MPI.COMM_WORLD.size:6}  ')

# compute homogenized stress field corresponding to displacement
homogenized_stress = discretization.get_homogenized_stress(
    material_data_field_ijklqxyz=material_data_field_C_0_rho,
    displacement_field_fnxyz=displacement_field_NEW,
    macro_gradient_field_ijqxyz=macro_gradient_field,
    formulation='small_strain')

print("homogenized_stress NEW: ", homogenized_stress)
#print("homogenized_stress NEW: ", domain.compute_Voigt_notation_2order(homogenized_stress))

end_time = MPI.Wtime()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)


