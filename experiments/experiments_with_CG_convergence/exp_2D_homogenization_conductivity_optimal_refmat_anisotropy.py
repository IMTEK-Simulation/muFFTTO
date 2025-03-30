import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import matplotlib as mpl
import time
# from netCDF4 import Dataset

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library
from simple_CG import (get_ritz_values, plot_ritz_values, get_cg_polynomial, plot_cg_polynomial, plot_eigenvectors, \
                       plot_eigendisplacement)

problem_type = 'conductivity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'  # #'linear_triangles'# linear_triangles_tilled
# formulation = 'small_strain'
geometry_ID = 'square_inclusion_equal_volfrac'

domain_size = [1, 1]
number_of_pixels = (24, 24)  # (128,128)

my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)
start_time = time.time()

# set macroscopic gradient
macro_gradient = np.array([1.0, .0])

# create material data field
mat_contrast = 1
mat_contrast_2 = 1

conductivity_C_matrix = np.array([[1.0, 0], [0, 1.0]])
conductivity_C_inclusion = np.array([[100., 0], [0, 101.0]])
conductivity_C_ref = np.array([[1., 0], [0,  1.0]])

material_data_field_C_inclusion = np.einsum('ij,qxy->ijqxy', conductivity_C_inclusion,
                                            np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                              *discretization.nb_of_pixels])))
material_data_field_C_matrix = np.einsum('ij,qxy->ijqxy', conductivity_C_matrix,
                                         np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                           *discretization.nb_of_pixels])))

material_data_field_C_ref = np.einsum('ij,qxy->ijqxy', conductivity_C_ref,
                                         np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                           *discretization.nb_of_pixels])))

# material distribution
phase_indicator_field = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                            microstructure_name=geometry_ID,
                                                            coordinates=discretization.fft.coords)

matrix_ind = np.asarray(phase_indicator_field, dtype=bool)

inclusion_ind = np.asarray(1 - phase_indicator_field, dtype=bool)

# apply material distribution
material_data_field_C_0_rho = np.zeros_like(material_data_field_C_matrix)
material_data_field_C_0_rho[..., matrix_ind] = material_data_field_C_matrix[..., matrix_ind]
material_data_field_C_0_rho[
    ..., inclusion_ind] = material_data_field_C_inclusion[..., inclusion_ind]

# Set up the equilibrium system
macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)

# Solve mechanical equilibrium constrain
rhs = discretization.get_rhs(material_data_field_C_0_rho, macro_gradient_field)

# linear system matrix
K_fun = lambda x: discretization.apply_system_matrix(material_data_field_C_0_rho, x)
K = discretization.get_system_matrix(material_data_field_C_0_rho)
K[:, 0] = 0
K[0, :] = 0
K[0, 0] = 1

eig_K, eig_vect_K = sc.linalg.eig(a=K, b=None)  # , eigvals_only=True
eig_K = np.real(eig_K)
# eig_K[eig_K == 1.0] = 1
# Sort in descending order (largest eigenvalues first)
idx_K = np.argsort(eig_K)[::-1]  # Get indices of sorted eigenvalues
#


preconditioner_inc = discretization.get_preconditioner_NEW(
    reference_material_data_field_ijklqxyz=material_data_field_C_inclusion)
M_fun_inc = lambda x: discretization.apply_preconditioner_NEW(preconditioner_inc, x)

temperatute_field_inc, norms_inc = solvers.PCG(K_fun, rhs, x0=None, P=M_fun_inc, steps=int(500), toler=1e-12,
                                               norm_type='rz')
nb_it = len(norms_inc['residual_rz'])
print(' nb_ steps CG_inc =' f'{nb_it}')
nomr_rz_inc = norms_inc['residual_rz']
print('  residual_rz CG _inc  =' f'{nomr_rz_inc}')

# Greeen precond
M_inc = discretization.get_system_matrix(material_data_field_C_inclusion)
M_inc[:, 0] = 0
M_inc[0, :] = 0
M_inc[0, 0] = 1
eig_G_inc, eig_vect_G_inc = sc.linalg.eig(a=K, b=M_inc)  # , eigvals_only=True
idx_G_inc = np.argsort(-eig_G_inc)[::-1]  # Get indices of sorted eigenvalues

##
preconditioner_matrix = discretization.get_preconditioner_NEW(
    reference_material_data_field_ijklqxyz=material_data_field_C_matrix)
M_fun_matrix = lambda x: discretization.apply_preconditioner_NEW(preconditioner_matrix, x)

temperatute_field_matrix, norms_matrix = solvers.PCG(K_fun, rhs, x0=None, P=M_fun_matrix, steps=int(500), toler=1e-12,
                                                     norm_type='rz')
nb_it = len(norms_matrix['residual_rz'])
print(' nb_ steps CG _matrix  =' f'{nb_it}')
nomr_rz_matrix = norms_matrix['residual_rz']

# Greeen precond
M_matrix = discretization.get_system_matrix(material_data_field_C_matrix)
M_matrix[:, 0] = 0
M_matrix[0, :] = 0
M_matrix[0, 0] = 1
eig_G_matrix, eig_vect_G_matrix = sc.linalg.eig(a=K, b=M_matrix)  # , eigvals_only=True
idx_G_matrix = np.argsort(-eig_G_matrix)[::-1]  # Get indices of sorted eigenvalues


##
preconditioner_matrix = discretization.get_preconditioner_NEW(
    reference_material_data_field_ijklqxyz=material_data_field_C_ref)
M_fun_ref= lambda x: discretization.apply_preconditioner_NEW(preconditioner_matrix, x)

temperatute_field_ref, norms_ref = solvers.PCG(K_fun, rhs, x0=None, P=M_fun_ref, steps=int(500), toler=1e-12,
                                                     norm_type='rz')
nb_it = len(norms_ref['residual_rz'])
print(' nb_ steps CG _ref  =' f'{nb_it}')
nomr_rz_ref = norms_ref['residual_rz']

# Greeen precond
M_ref = discretization.get_system_matrix(material_data_field_C_ref)
M_ref[:, 0] = 0
M_ref[0, :] = 0
M_ref[0, 0] = 1
eig_G_ref, eig_vect_G_ref = sc.linalg.eig(a=K, b=M_ref)  # , eigvals_only=True
idx_G_ref = np.argsort(-eig_G_ref)[::-1]  # Get indices of sorted eigenvalues




# PLOTS PLOTS PLOTS
fig = plt.figure(figsize=(7, 7))
gs = fig.add_gridspec(3, 2, width_ratios=[1, 1])
ax_converg = fig.add_subplot(gs[0, :])
ax_eig_G_inc = fig.add_subplot(gs[1, 0])
ax_eig_G_matrix = fig.add_subplot(gs[1, 1])
ax_hist_G_inc = fig.add_subplot(gs[2, 0])
ax_hist_G_matrix = fig.add_subplot(gs[2, 1])

ax_converg.semilogy(nomr_rz_inc / nomr_rz_inc[0], color='Green',marker='|', label='C_REF = C_inclusion')
# /nomr_rz_inc [0]
ax_converg.semilogy(nomr_rz_matrix / nomr_rz_matrix[0], color='red', label='C_REF = C_matirx')
# /nomr_rz_matrix [0]
ax_converg.semilogy(nomr_rz_ref / nomr_rz_ref[0], color='blue',marker='o', label='C_REF = C_ref')
# /nomr_rz_matrix [0]
ax_converg.legend()
ax_converg.set_ylim([1e-14, 1e0])
ax_converg.set_xlim([0, 60])
ax_eig_G_inc.plot(eig_G_inc[idx_G_inc], color='Green', label=f'Green',
                  alpha=0.5, marker='.', linewidth=0, markersize=5)
ax_eig_G_matrix.plot(eig_G_matrix[idx_G_matrix], color='red', label=f'red',
                     alpha=0.5, marker='.', linewidth=0, markersize=5)

# Calculate the bin edges
num_bins = eig_G_inc.size
# Define the bin width
bin_width = 0.05
min_edge = np.min(eig_G_inc)
max_edge = np.max(eig_G_inc)
bins = np.arange(min_edge, max_edge + bin_width, bin_width)
bins = 100
# Create the histogram
ax_hist_G_inc.hist(eig_G_inc[idx_G_inc], bins=bins, color='Green', label=f'Green', edgecolor='Green',
                   alpha=0.99)  # , marker='.', linewidth=0, markersize=5)
ax_hist_G_matrix.hist(eig_G_matrix[idx_G_matrix], bins=bins, color='Red', label=f'Jacobi-Green', edgecolor='black',
                      alpha=0.99)

plt.show()
print('  residual_rz CG _matrix  =' f'{nomr_rz_matrix}')

# --print(' nb_ steps CG =' f'{nb_it}')--------------------------------------------------------------------
# compute homogenized stress field corresponding to displacement
homogenized_flux = discretization.get_homogenized_stress(
    material_data_field_ijklqxyz=material_data_field_C_0_rho,
    displacement_field_fnxyz=temperatute_field_inc,
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
