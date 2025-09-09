import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

# from netCDF4 import Dataset

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library

import numpy as np
import matplotlib.pyplot as plt

problem_type = 'conductivity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'  # #'linear_triangles'# linear_triangles_tilled
# formulation = 'small_strain'
geometry_ID = 'checkerboard'

domain_size = [1, 1]
dim = len(domain_size)
number_of_pixels = (128, 128)

my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)
start_time = time.time()

# set macroscopic gradient
macro_gradient = np.array([1.0, .0])

# create material data field # We have 4 squares
c_1 = 100  # inclusion
c_2, c_3, c_4 = 1., 1., 1.  # matrix  remaining 3 squares

conductivity_C_1 = c_1 * np.array([[1., 0], [0, 1.0]])
conductivity_C_2 = c_2 * np.array([[10., 0], [0, 1.0]])

conductivity_C_ref = np.array([[1., 0], [0, 1.0]])

# Analytical solution
ρ = [1 / c_1, 1 / c_2, 1 / c_3, 1 / c_4]

σ1 = sum(ρ)
σ2 = ρ[0] * ρ[2] - ρ[1] * ρ[3]
σ3 = ρ[0] * ρ[1] * ρ[2] + ρ[0] * ρ[1] * ρ[3] + ρ[0] * ρ[2] * ρ[3] + ρ[1] * ρ[2] * ρ[3]

# homogenized resistivity and conductivity
ρx = np.sqrt((ρ[1] + ρ[2]) * (ρ[3] + ρ[0]) / ((ρ[0] + ρ[1]) * (ρ[2] + ρ[3])) * (σ3 / σ1))
ρy = np.sqrt((ρ[0] + ρ[1]) * (ρ[2] + ρ[3]) / ((ρ[1] + ρ[2]) * (ρ[3] + ρ[0])) * (σ3 / σ1))
Chom = (1 / ρx, 1 / ρy)
J_eff = Chom
print("J_eff : ", J_eff)

# eigenvalues
eigen_C1 = sp.linalg.eigh(a=conductivity_C_1, b=conductivity_C_ref, eigvals_only=True)
eigen_C2 = sp.linalg.eigh(a=conductivity_C_2, b=conductivity_C_ref, eigvals_only=True)
eigen_LB = np.min([eigen_C1, eigen_C2])
# seigen_LB *=0.9
eigen_UB = np.max([eigen_C1, eigen_C2])
print(f'eigen_LB = {eigen_LB}')
print(f'eigen_UB = {eigen_UB}')

material_data_field = np.einsum('ij,qxy->ijqxy', conductivity_C_2,
                                np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                  *discretization.nb_of_pixels])))

material_data_field_C_1 = np.einsum('ij,qxy->ijqxy', conductivity_C_1,
                                    np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                      *discretization.nb_of_pixels])))

material_data_field_C_ref = np.einsum('ij,qxy->ijqxy', conductivity_C_ref,
                                      np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                        *discretization.nb_of_pixels])))
# material distribution

coordinates = discretization.fft.coords
r1 = 0.25
# apply material distribution
for center_point in [0.25]:  # , 0.75
    r_center = coordinates - center_point
    distances = 0
    distances += np.sum(np.abs(r_center[d]) for d in range(dim))
    mask = distances <= r1
    mask += np.rot90(np.fliplr(mask).T)
    material_data_field[..., :, mask] = material_data_field_C_1[..., mask]


plt.imshow(material_data_field[0, 0, 0], cmap="gray")
plt.show()

# Set up the equilibrium system
macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)

# Solve mechanical equilibrium constrain
rhs = discretization.get_rhs(material_data_field, macro_gradient_field)

K_fun = lambda x: discretization.apply_system_matrix(material_data_field, x)
# M_fun = lambda x: 1 * x

preconditioner = discretization.get_preconditioner_NEW(reference_material_data_field_ijklqxyz=material_data_field_C_ref)
# preconditioner_old = discretization.get_preconditioner(reference_material_data_field_ijklqxyz=material_data_field_C_0)

M_fun = lambda x: discretization.apply_preconditioner_NEW(preconditioner, x)


temperatute_field_precise, norms_precise = solvers.PCG(Afun=K_fun,
                                                       B=rhs,
                                                       x0=None,
                                                       P=M_fun,
                                                       steps=int(500),
                                                       toler=1e-14,
                                                       norm_energy_upper_bound=True,
                                                       lambda_min=eigen_LB)

parameters_CG = {'exact_solution': temperatute_field_precise,
                 'energy_lower_bound': True,
                 'tau': 0.25}

error_in_Aeff_00 = []
# compute homogenized stress field corresponding to displacement
J_eff_computed = discretization.get_homogenized_stress(
    material_data_field_ijklqxyz=material_data_field,
    displacement_field_inxyz=temperatute_field_precise,
    macro_gradient_field_ijqxyz=macro_gradient_field)[0, 0]


def my_callback(x_k):
    # compute homogenized stress field corresponding to displacement
    homogenized_flux = discretization.get_homogenized_stress(
        material_data_field_ijklqxyz=material_data_field,
        displacement_field_inxyz=x_k,
        macro_gradient_field_ijqxyz=macro_gradient_field)
    error_in_Aeff_00.append(homogenized_flux[0, 0] - J_eff_computed)  # J_eff_computed if J_eff is not available


temperatute_field, norms = solvers.PCG(Afun=K_fun,
                                       B=rhs,
                                       x0=None,
                                       P=M_fun,
                                       steps=int(500), toler=1e-10,
                                       norm_energy_upper_bound=True,
                                       lambda_min=eigen_LB,
                                       callback=my_callback,
                                       **parameters_CG)

nb_it = len(norms['residual_rz'])
print(' nb_ steps CG =' f'{nb_it}')
fig = plt.figure(figsize=(7, 4.5))
gs = fig.add_gridspec(1, 1, hspace=0.1, wspace=0.1, width_ratios=1 * (1,),
                      height_ratios=[1, ])
ax_norms = fig.add_subplot(gs[0])
ax_norms.semilogy(norms['energy_iter_error'],
                  label=r'$\overline{\varepsilon}^{T} (A_{h,k}^{\mathrm{eff}} -A^{\mathrm{eff}}_{h,\infty})\,\overline{\varepsilon} $',
                  color='red',
                  alpha=0.5, marker='x', linewidth=1, markersize=5, markevery=5)
ax_norms.semilogy(norms['energy_upper_bound'], label='Upper bound PT', color='Green',
                  alpha=0.5, marker='v', linewidth=1, markersize=5, markevery=5)
ax_norms.semilogy(norms['energy_lower_bound'], label='Lower estim PT', color='Red',
                  alpha=0.5, marker='^', linewidth=1, markersize=5, markevery=5)

ax_norms.semilogy(norms['residual_rz'] / eigen_LB,
                  label=r'Trivial   upper  bound  - $\frac{1}{\lambda_{min}}|| \mathbf{ r}_{k} ||_{\mathbf{G}}^2$',
                  color='Blue',
                  alpha=0.5, marker='v', linewidth=1, markersize=5, markevery=5)
ax_norms.semilogy(norms['residual_rz'] / eigen_UB,
                  label=r'Trivial lower  bound  - $\frac{1}{\lambda_{max}}|| \mathbf{ r}_{k} ||_{\mathbf{G}}^2$',
                  color='Blue',
                  alpha=0.5, marker='^', linewidth=1, markersize=5, markevery=5)

# ax_norms.semilogy(norms['residual_rr'], label='residual_rr', color='Black',
#                   alpha=0.5, marker='.', linewidth=1, markersize=5, markevery=5)
ax_norms.semilogy(error_in_Aeff_00,
                  label=r'hom prop $\overline{\varepsilon}^{T} (A_{h,k}^{\mathrm{eff}} -A^{\mathrm{eff}}_{h,\infty})\,\overline{\varepsilon} $',
                  color='Black',
                  alpha=0.5, marker='x', linewidth=1, markersize=5, markevery=1)

# plt.title('optimizer {}'.format(optimizer))
ax_norms.set_ylabel('Norms')
ax_norms.set_ylim(1e-14, 1e6)
# ax_norms.set_yticks([1, 34, 67, 100])
# ax_norms.set_yticklabels([1, 34, 67, 100])

ax_norms.set_xlabel('# CG iterations')

ax_norms.set_xlim([0, norms['residual_rr'].__len__() - 1])
# ax_norms.set_xticks([1, len(eig_G) // 2, len(eig_G)])
# ax_norms.set_xticklabels([1, len(eig_G) // 2, len(eig_G)])

plt.grid(True)

plt.legend()

plt.show()

true_e_error = np.asarray(norms['energy_iter_error'])
lower_estim = np.asarray(norms['energy_lower_bound'])
upper_estim = lower_estim / (1 - parameters_CG['tau'])
upper_bound = np.asarray(norms['energy_upper_bound'])
trivial_lower_bound = np.asarray(norms['residual_rz'] / eigen_UB)
trivial_upper_bound = np.asarray(norms['residual_rz'] / eigen_LB)

fig = plt.figure(figsize=(7, 4.5))
gs = fig.add_gridspec(1, 1, hspace=0.1, wspace=0.1, width_ratios=1 * (1,),
                      height_ratios=[1, ])
ax_norms = fig.add_subplot(gs[0])

tmp = min(len(lower_estim), len(upper_bound))
ax_norms.semilogy(lower_estim[0:tmp - 1] / true_e_error[0:tmp - 1],
                  label='Lower estim PT')
ax_norms.semilogy(upper_estim[0:tmp - 1] / true_e_error[0:tmp - 1],
                  label='Upper estim PT')

ax_norms.semilogy(upper_bound[0:tmp - 1] / true_e_error[0:tmp - 1],
                  label='Upper bound PT')

ax_norms.semilogy(trivial_lower_bound[0:tmp - 1] / true_e_error[0:tmp - 1],
                  label=r'Trivial lower  bound  - $\frac{1}{\lambda_{max}}|| \mathbf{ r}_{k} ||_{\mathbf{G}}^2$',
                  )
ax_norms.semilogy(trivial_upper_bound[0:tmp - 1] / true_e_error[0:tmp - 1],
                  label=r'Trivial   upper  bound  - $\frac{1}{\lambda_{min}}|| \mathbf{ r}_{k} ||_{\mathbf{G}}^2$',
                  )

# ax_norms.semilogy((1:tmp,upper_estim_M(1:tmp)./norm_ener_error_M(1:tmp))
# ax_norms.semilogy((1:tmp,estim_M_UB(1:tmp)./norm_ener_error_M(1:tmp))
ax_norms.semilogy(np.ones(tmp), 'k-')

ax_norms.set_title('effectivity indices')
ax_norms.set_xlabel('# CG iterations')

# hezci rozsah os, abychom videli efektivitu u jednicky
ax_norms.set_ylim(1e-4, 1e4)
ax_norms.legend(loc='best')
plt.show()
# ----------------------------------------------------------------------
# compute homogenized stress field corresponding to displacement
homogenized_flux = discretization.get_homogenized_stress(
    material_data_field_ijklqxyz=material_data_field,
    displacement_field_inxyz=temperatute_field,
    macro_gradient_field_ijqxyz=macro_gradient_field)

print(homogenized_flux)

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)

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
