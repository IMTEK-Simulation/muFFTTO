import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import time
# from netCDF4 import Dataset

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library

from scipy import misc
import glob

name='muFFTTO_elasticity_random_init_N32_E_target_0.3_Poisson_0.25_w0.01_eta0.032_p2_bounds=False_FE_NuMPI4.npy'
fp='exp_data/'+name

#fp='exp_data/muFFTTO_elasticity_random_init_N64_E_target_0.3_Poisson_0.25_w0.01_eta0.01_p2_bounds=False_FE_NuMPI6.npy'
#
phase_field=np.load(fp)
plt.figure()
plt.imshow(phase_field)
#plt.show()

problem_type = 'conductivity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
# formulation = 'small_strain'
geometry_ID = 'linear'#,'sine_wave'

domain_size = [1, 1]
number_of_pixels = (4, 4)

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
mat_contrast = 1
mat_contrast_2 = 100
conductivity_C_0 = np.array([[1., 0], [0, 1.0]])
conductivity_C_1 = np.array([[1., 0], [0, 1.0]])
# mat_contrast_2 = 100
# conductivity_C_0 = np.array([[1000., 0], [0, 1.0]])
# conductivity_C_1 = np.array([[3., 5], [5, 80.0]])

material_data_field_C_0 = np.einsum('ij,qxy->ijqxy', conductivity_C_0,
                                    np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                      *discretization.nb_of_pixels])))
material_data_field_C_1 = np.einsum('ij,qxy->ijqxy', conductivity_C_1,
                                    np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                      *discretization.nb_of_pixels])))
# material distribution
phase_field = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                   microstructure_name=geometry_ID,
                                                   coordinates=discretization.fft.coords)
phase_field_jump = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                       microstructure_name='square_inclusion',
                                                       coordinates=discretization.fft.coords)
# apply material distribution
material_data_field_C_0_rho = mat_contrast * material_data_field_C_0[..., :, :] * np.power(phase_field, 2)
#material_data_field_C_0_rho[1,1,...] += mat_contrast_2 * material_data_field_C_1[1,1,:, :, :] * np.power(phase_field, 2)

#material_data_field_C_0_rho *= (phase_field_jump + 1)

# Set up the equilibrium system
macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)

# Solve mechanical equilibrium constrain
rhs = discretization.get_rhs(material_data_field_C_0_rho, macro_gradient_field)

K_fun = lambda x: discretization.apply_system_matrix(material_data_field_C_0_rho, x)
# M_fun = lambda x: 1 * x
K = discretization.get_system_matrix(material_data_field_C_0_rho)
M = discretization.get_system_matrix(material_data_field_C_0)

K_diag = np.diag(K).reshape(number_of_pixels)
M_diag = np.diag(M).reshape(number_of_pixels)

K_diag_alg=discretization.get_preconditioner_Jacoby(material_data_field_ijklqxyz=material_data_field_C_0_rho)
K_diag_alg_fast=discretization.get_preconditioner_Jacoby_fast(material_data_field_ijklqxyz=material_data_field_C_0_rho)


K_diag_inv_sym = K_diag ** (-1 / 2)

D_diag_inv_sym = (K_diag / M_diag) ** (-1 / 2)
# K_diag_inv_ = K_diag ** (-1)

M_inv = np.linalg.inv(M)
M_inv_sym = sc.linalg.fractional_matrix_power(M_inv, 0.5)

MK = np.matmul(np.matmul(M_inv_sym, K), M_inv_sym)
DMDK = np.matmul(np.matmul(np.diag(K_diag_inv_sym.reshape(-1)), K), np.diag(K_diag_inv_sym.reshape(-1)))

# Jsym = np.diag(sc.linalg.fractional_matrix_power(np.diag(K_diag), -0.5)).reshape(number_of_pixels)
# Jsymple = np.diag(sc.linalg.fractional_matrix_power(np.diag(K_diag), -0.5)).reshape(number_of_pixels)
eigvals_K = np.real(np.linalg.eigvals(K))
eigvals_MK = np.real(np.linalg.eigvals(MK))
eigvals_DMDK = np.real(np.linalg.eigvals(DMDK))


preconditioner = discretization.get_preconditioner_NEW(reference_material_data_field_ijklqxyz=material_data_field_C_0)
plt.figure()
eig_round = 12
# uniq_eig_K  = np.unique(np.round(eigvals_K , decimals=eig_round)).size
# plt.plot(np.arange(eigvals_K.__len__()), np.sort(eigvals_K),
#              label='K, N={}, uniq={}'.format(eigvals_K.size, uniq_eig_K))# preconditioner_old = discretization.get_preconditioner(reference_material_data_field_ijklqxyz=material_data_field_C_0)

uniq_eig_MK = np.unique(np.round(eigvals_MK, decimals=eig_round)).size

plt.plot(np.arange(eigvals_MK.__len__()), np.sort(eigvals_MK),
         label='LK, N={}, uniq={}'.format(eigvals_MK.size,
                                          uniq_eig_MK))  # preconditioner_old = discretization.get_preconditioner(reference_material_data_field_ijklqxyz=material_data_field_C_0)

uniq_eig_DMDK = np.unique(np.round(eigvals_DMDK, decimals=eig_round)).size

plt.plot(np.arange(eigvals_DMDK.__len__()), np.sort(eigvals_DMDK),
         label='DMDK , N={}, uniq={}'.format(eigvals_DMDK.size,
                                             uniq_eig_DMDK))  # preconditioner_old = discretization.get_preconditioner(reference_material_data_field_ijklqxyz=material_data_field_C_0)

plt.legend()

plt.show()
M_fun = lambda x: discretization.apply_preconditioner_NEW(preconditioner, x)
M_fun_combined = lambda x: K_diag_inv_sym * discretization.apply_preconditioner_NEW(preconditioner, K_diag_inv_sym * x)

M_fun_combined_IP = lambda x: D_diag_inv_sym * discretization.apply_preconditioner_NEW(preconditioner,
                                                                                       D_diag_inv_sym * x)

# M_fun_old= lambda x: discretization.apply_preconditioner(preconditioner_old, x)


# M_fun_NONE = lambda x: 1 * x
# x_0=np.random.rand(*discretization.get_unknown_size_field().shape)
# x_00=np.copy(x_0)
# x_1=M_fun_old(np.copy(x_0))
# x_12=M_fun_old(np.copy(x_1))
#
# x_2=M_fun(x_0)
# x_22=M_fun(x_2)
temperatute_field_comb, norms_comb = solvers.PCG(K_fun, rhs, x0=None, P=M_fun_combined, steps=int(1500), toler=1e-12)
nb_it_comb = len(norms_comb['residual_rz'])
print(' nb_ steps CG _comb =' f'{nb_it_comb}')
# ----------------------------------------------------------------------
# compute homogenized stress field corresponding to displacement
homogenized_flux = discretization.get_homogenized_stress(
    material_data_field_ijklqxyz=material_data_field_C_0_rho,
    displacement_field_fnxyz=temperatute_field_comb,
    macro_gradient_field_ijqxyz=macro_gradient_field)

print(homogenized_flux)

temperatute_field_comb_IP, norms_comb_IP = solvers.PCG(K_fun, rhs, x0=None, P=M_fun_combined_IP, steps=int(1500),
                                                       toler=1e-12)
nb_it_comb_IP = len(norms_comb_IP['residual_rz'])
print(' nb_ steps CG _comb =' f'{nb_it_comb_IP}')
# ----------------------------------------------------------------------
# compute homogenized stress field corresponding to displacement
homogenized_flux = discretization.get_homogenized_stress(
    material_data_field_ijklqxyz=material_data_field_C_0_rho,
    displacement_field_fnxyz=temperatute_field_comb_IP,
    macro_gradient_field_ijqxyz=macro_gradient_field)

print(homogenized_flux)

temperatute_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(1500), toler=1e-12)
nb_it = len(norms['residual_rz'])
print(' nb_ steps PCG Laplace =' f'{nb_it}')

# ----------------------------------------------------------------------
# compute homogenized stress field corresponding to displacement
homogenized_flux = discretization.get_homogenized_stress(
    material_data_field_ijklqxyz=material_data_field_C_0_rho,
    displacement_field_fnxyz=temperatute_field,
    macro_gradient_field_ijqxyz=macro_gradient_field)

print(homogenized_flux)

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)

J_eff = mat_contrast_2 * np.sqrt((mat_contrast_2 + 3 * mat_contrast) / (3 * mat_contrast_2 + mat_contrast))
print("J_eff: ", J_eff)

plt.figure()
plt.semilogy(np.arange(norms['residual_rz'].__len__()), norms['residual_rz'],
         label='MK rz')  # preconditioner_old = discretization.get_preconditioner(reference_material_data_field_ijklqxyz=material_data_field_C_0)
plt.semilogy(np.arange(norms['residual_rr'].__len__()), norms['residual_rr'],
         label='MK rr')  # preconditioner_old = discretization.get_preconditioner(reference_material_data_field_ijklqxyz=material_data_field_C_0)
plt.semilogy(np.arange(norms_comb['residual_rz'].__len__()), norms_comb['residual_rz'],
         label='DLDK  rz')  # preconditioner_old = discretization.get_preconditioner(reference_material_data_field_ijklqxyz=material_data_field_C_0)
plt.semilogy(np.arange(norms_comb['residual_rr'].__len__()), norms_comb['residual_rr'],
         label='DLDK  rr')  # preconditioner_old = discretization.get_preconditioner(reference_material_data_field_ijklqxyz=material_data_field_C_0)

plt.semilogy(np.arange(norms_comb_IP['residual_rz'].__len__()), norms_comb_IP['residual_rz'],
         label='DLDK IP rz')  # preconditioner_old = discretization.get_preconditioner(reference_material_data_field_ijklqxyz=material_data_field_C_0)
plt.semilogy(np.arange(norms_comb_IP['residual_rr'].__len__()), norms_comb_IP['residual_rr'],
         label='DLDK IP rr')  # preconditioner_old = discretization.get_preconditioner(reference_material_data_field_ijklqxyz=material_data_field_C_0)


plt.legend()

plt.show()

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
