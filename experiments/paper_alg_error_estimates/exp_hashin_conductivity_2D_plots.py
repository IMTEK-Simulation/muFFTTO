import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import os

# File that generates plots
script_name = 'exp_hashin_conductivity_2D'
file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
figure_folder_path = file_folder_path + '/figures/' + script_name + '/'

# from netCDF4 import Dataset

from NuMPI.IO import save_npy, load_npy

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library

from experiments.paper_alg_error_estimates import _labels, _colors, _markers, _linestyles

problem_type = 'conductivity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
geometry_ID = 'square_inclusion'

mat_contrast_1 = 1.  # inclusion

mat_contrast_2 = 1e-3  # inclusion
anisotropy = True
if anisotropy:
    a_ani = 10
else:
    a_ani = 1

# PLOT CONVERGENCE
# results set up
number_of_pixels = (64, 64)
results_name = f'N{number_of_pixels[0]}_rho_inc{mat_contrast_2}_mat{mat_contrast_1}_ani{a_ani}'

_info = np.load(data_folder_path + results_name + f'_log.npz', allow_pickle=True)
temp_field = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True)

true_e_error = _info['true_e_error']
lower_bound = _info['lower_bound']
upper_bound = _info['upper_bound']
upper_estim = _info['upper_estim']
trivial_lower_bound = _info['trivial_lower_bound']
trivial_upper_bound = _info['trivial_upper_bound']

total_phase_contrast = _info['total_phase_contrast']

homogenized_flux = _info['homogenized_flux']
Aeff_h_precise = _info['Aeff_h_precise']
A_eff = _info['A_eff']
Aeff_hk = _info['Aeff_hk']
error_in_Aeff_hk = _info['error_in_Aeff_hk']

fig = plt.figure(figsize=(7, 4.5))
gs = fig.add_gridspec(1, 1, hspace=0.1, wspace=0.1, width_ratios=1 * (1,),
                      height_ratios=[1, ])
ax_norms = fig.add_subplot(gs[0])
ax_norms.semilogy(true_e_error,
                  label=_labels['true_error'],
                  color=_colors['true_error'],
                  alpha=1.,
                  marker=_markers['true_error'],
                  linewidth=1, markersize=5, markevery=5)

ax_norms.semilogy(trivial_upper_bound,
                  label=_labels['trivial_upper_bound'],
                  color=_colors['trivial_upper_bound'],
                  alpha=0.5,
                  marker=_markers['trivial_upper_bound'],
                  linewidth=1, markersize=5, markevery=5)
ax_norms.semilogy(trivial_lower_bound,
                  label=_labels['trivial_lower_bound'],
                  color=_colors['trivial_lower_bound'],
                  alpha=0.5,
                  marker=_markers['trivial_lower_bound'],
                  linewidth=1, markersize=5, markevery=5)

ax_norms.semilogy(upper_bound,
                  label=_labels['PT_upper_bound'],
                  color=_colors['PT_upper_bound'],
                  alpha=0.5,
                  marker=_markers['PT_upper_bound'],
                  linewidth=1, markersize=5, markevery=5)

ax_norms.semilogy(lower_bound,
                  label=_labels['PT_lower_bound'],
                  color=_colors['PT_lower_bound'],
                  linestyle='--',
                  linewidth=1,
                  alpha=0.5,
                  marker=_markers['PT_lower_bound'],
                  markersize=5, markevery=5)
ax_norms.semilogy(upper_estim,
                  label=_labels['PT_upper_estimate'],
                  color=_colors['PT_upper_estimate'],
                  linestyle='--',
                  linewidth=1,
                  alpha=0.5,
                  marker=_markers['PT_upper_estimate'],
                  markersize=5, markevery=5)

# ax_norms.semilogy(norms['residual_rr'], label='residual_rr', color='Black',
#                   alpha=0.5, marker='.', linewidth=1, markersize=5, markevery=5)
# ax_norms.semilogy(error_in_Aeff_00,
#                   label=r'hom prop $\overline{\varepsilon}^{T} (A_{h,k}^{\mathrm{eff}} -A^{\mathrm{eff}}_{h,\infty})\,\overline{\varepsilon} $',
#                   color='Black',
#                   alpha=0.5, marker='x', linewidth=1, markersize=5, markevery=1)

# plt.title('optimizer {}'.format(optimizer))
# ax_norms.set_ylabel('Norms')
ax_norms.set_ylim(1e-14, 1e6)
# ax_norms.set_yticks([1, 34, 67, 100])
# ax_norms.set_yticklabels([1, 34, 67, 100])

ax_norms.set_xlabel(r'PCG iteration - $k$')

ax_norms.set_xlim([0, true_e_error.__len__() - 1])
# ax_norms.set_xticks([1, len(eig_G) // 2, len(eig_G)])
# ax_norms.set_xticklabels([1, len(eig_G) // 2, len(eig_G)])

plt.grid(True)

plt.legend()
fig_name = f'norm_evolution_kappa{total_phase_contrast}'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')

# fig.tight_layout()
fname = figure_folder_path + fig_name + '{}'.format('.pdf')
print(('create figure: {}'.format(fname)))
plt.savefig(fname, bbox_inches='tight'
            )
plt.show()
####################################### PLOT Efficiencies

number_of_pixels = (64, 64)
mat_contrast_1 = 1.  # inclusion
anisotropy = True
if anisotropy:
    a_ani = 10
else:
    a_ani = 1

fig = plt.figure(figsize=(7, 4.5))
gs = fig.add_gridspec(1, 1, hspace=0.1, wspace=0.1, width_ratios=1 * (1,),
                      height_ratios=[1, ])
ax_norms = fig.add_subplot(gs[0])
# multiple contrasts
rhos = [-1, -2, -3]
for i in np.arange(rhos.__len__()):
    rho = rhos[i]
    mat_contrast_2 = 10 ** rho  # coating

    # results_name = f'N{number_of_pixels[0]}_rho_inc{mat_contrast_1}_mat{mat_contrast_2}_ani{a_ani}'
#    results_name = f'N{number_of_pixels[0]}_rho_inc{mat_contrast_2}_mat{mat_contrast}_ani{a_ani}'
    results_name = f'N{number_of_pixels[0]}_rho_inc{mat_contrast_2}_mat{mat_contrast_1}_ani{a_ani}'

    _info = np.load(data_folder_path + results_name + f'_log.npz', allow_pickle=True)
    temp_field = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True)

    true_e_error = _info['true_e_error']
    lower_bound = _info['lower_bound']
    upper_bound = _info['upper_bound']
    upper_estim = _info['upper_estim']
    trivial_lower_bound = _info['trivial_lower_bound']
    trivial_upper_bound = _info['trivial_upper_bound']

    total_phase_contrast = _info['total_phase_contrast']

    homogenized_flux = _info['homogenized_flux']
    Aeff_h_precise = _info['Aeff_h_precise']
    A_eff = _info['A_eff']
    Aeff_hk = _info['Aeff_hk']
    error_in_Aeff_hk = _info['error_in_Aeff_hk']

    tmp = min(len(lower_bound), len(upper_bound))

    ax_norms.semilogy(trivial_upper_bound[0:tmp - 1] / true_e_error[0:tmp - 1],
                      label=_labels['trivial_upper_bound'],
                      color=_colors['trivial_upper_bound'],
                      marker=_markers['trivial_upper_bound'],
                      linestyle=_linestyles[f'{i}'],
                      alpha=0.5,
                      linewidth=1, markersize=5, markevery=10
                      )

    ax_norms.semilogy(trivial_lower_bound[0:tmp - 1] / true_e_error[0:tmp - 1],
                      label=_labels['trivial_lower_bound'],
                      color=_colors['trivial_lower_bound'],
                      alpha=0.5,
                      marker=_markers['trivial_lower_bound'],
                      linestyle=_linestyles[f'{i}'],
                      linewidth=1, markersize=5, markevery=10
                      )

    ax_norms.semilogy(upper_bound[0:tmp - 1] / true_e_error[0:tmp - 1],
                      label=_labels['PT_upper_bound'],
                      color=_colors['PT_upper_bound'],
                      alpha=1,
                      marker=_markers['PT_upper_bound'],
                      linestyle=_linestyles[f'{i}'],
                      linewidth=1, markersize=5, markevery=10)

    ax_norms.semilogy(lower_bound[0:tmp - 1] / true_e_error[0:tmp - 1],
                      label=_labels['PT_lower_bound'],
                      color=_colors['PT_lower_bound'],
                      linewidth=1,
                      alpha=1,
                      marker=_markers['PT_lower_bound'],
                      linestyle=_linestyles[f'{i}'],
                      markersize=5, markevery=10)

    ax_norms.semilogy(upper_estim[0:tmp - 1] / true_e_error[0:tmp - 1],
                      label=_labels['PT_upper_estimate'],
                      color=_colors['PT_upper_estimate'],
                      linewidth=1,
                      alpha=1,
                      marker=_markers['PT_upper_estimate'],
                      linestyle=_linestyles[f'{i}'],
                      markersize=5, markevery=10)

# ax_norms.semilogy((1:tmp,upper_estim_M(1:tmp)./norm_ener_error_M(1:tmp))
# ax_norms.semilogy((1:tmp,estim_M_UB(1:tmp)./norm_ener_error_M(1:tmp))
ax_norms.semilogy(np.ones(tmp), 'k-')

# ax_norms.set_title('effectivity indices')
ax_norms.set_xlabel(r'PCG iteration - $k$')

# hezci rozsah os, abychom videli efektivitu u jednicky
ax_norms.set_ylim(1e-4, 1e4)
ax_norms.set_xlim(0, 100)

# ax_norms.legend(loc='best', ncol=3)

fig_name = f'norm_efficiency_kappa{total_phase_contrast}'  # '  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')

fig.tight_layout()
fname = figure_folder_path + fig_name + '{}'.format('.pdf')
print(('create figure: {}'.format(fname)))
plt.savefig(fname, bbox_inches='tight'
            )

plt.show()

####################################### PLOT DISCRETIZATION ERRORS
fig = plt.figure(figsize=(7, 4.5))
gs = fig.add_gridspec(1, 1, hspace=0.1, wspace=0.1, width_ratios=1 * (1,),
                      height_ratios=[1, ])
ax_norms = fig.add_subplot(gs[0])

mat_contrast_1 = 1.  # matrix
mat_contrast_2 = 10**3  # inclusion
anisotropy = True
if anisotropy:
    a_ani = 10
else:
    a_ani = 1
#results_name = f'N{1024}_rho_inc{mat_contrast_2}_mat{mat_contrast}_ani{a_ani}'
results_name = f'N{128}_rho_inc{mat_contrast_2}_mat{mat_contrast_1}_ani{a_ani}'

_info = np.load(data_folder_path + results_name + f'_log.npz', allow_pickle=True)
temp_field = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True)

# true_e_error = _info['true_e_error']
# lower_bound = _info['lower_bound']
# upper_bound = _info['upper_bound']
# upper_estim = _info['upper_estim']
# trivial_lower_bound = _info['trivial_lower_bound']
# trivial_upper_bound = _info['trivial_upper_bound']
# homogenized_flux = _info['homogenized_flux']
# total_phase_contrast = _info['total_phase_contrast']

if anisotropy:
    A_eff_fine_grid = _info['Aeff_h_precise']
else:
    A_eff_fine_grid = _info['A_eff']

Aeff_fine_grid_k = _info['Aeff_hk']
# error_in_Aeff_00 = _info['error_in_Aeff_00']

# homogenized_flux = _info['homogenized_flux']
# Aeff_h_precise = _info['Aeff_h_precise']
# A_eff = _info['A_eff']
# Aeff_hk = _info['Aeff_hk']
# error_in_Aeff_hk = _info['error_in_Aeff_hk']

grids_sizes = [3, 5, 6]#[6, 7, 8]
for i in range(grids_sizes.__len__()):
    # results set up
    number_of_pixels = (2 ** grids_sizes[i], 2 ** grids_sizes[i])

    results_name = f'N{number_of_pixels[0]}_rho_inc{mat_contrast_2}_mat{mat_contrast_1}_ani{a_ani}'
    _info = np.load(data_folder_path + results_name + f'_log.npz', allow_pickle=True)
    temp_field = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True)

    # true_e_error = _info['true_e_error']
    # lower_bound = _info['lower_bound']
    # upper_bound = _info['upper_bound']
    # upper_estim = _info['upper_estim']
    # trivial_lower_bound = _info['trivial_lower_bound']
    # trivial_upper_bound = _info['trivial_upper_bound']
    # homogenized_flux = _info['homogenized_flux']
    Aeff_h_precise = _info['Aeff_h_precise']
    # J_eff = _info['J_eff']
    # total_phase_contrast = _info['total_phase_contrast']
    Aeff_h_k = _info['Aeff_hk']
    error_in_Aeff_hk = _info['error_in_Aeff_hk']

    # plot iterative error Aeff_h_k -Aeff_h_k[-1]
    e_iter = Aeff_h_k - Aeff_h_precise
    ax_norms.semilogy(e_iter,
                      # label=_labels['true_error'],
                      label=fr'e$^\mathrm{{iter}}_{{ {{{2 ** grids_sizes[i]}}},k}}$',
                      # label=fr' $\overline{{\varepsilon}}^{{T}} (A_{{ {{{2 ** grids_sizes[i]}}},k}}^{{\mathrm{{eff}}}} -A^{{\mathrm{{eff}}}}_{{ {{{2 ** grids_sizes[i]}}},\infty}})\,\overline{{\varepsilon}} $',
                      color=_colors[f'{i}'],
                      linestyle='-.',
                      alpha=1.,
                      marker=_markers[f'{i}'],
                      linewidth=1, markersize=4, markevery=5)

    # plot discretization error Aeff_h_inf-Aeff_fine_grid_inf $
    e_dis = Aeff_h_k[-1] - Aeff_fine_grid_k[-1]
    ax_norms.semilogy(np.arange(0, np.size(Aeff_h_k)),
                      e_dis * np.ones_like(Aeff_h_k),
                      # grids_sizes[i]
                      label=fr'e$^\mathrm{{dis}}_{{ {{{2 ** grids_sizes[i]}}}}}$',
                      # label=fr' $\overline{{\varepsilon}}^{{T}} (A^{{\mathrm{{eff}}}}_{{ {{{2 ** grids_sizes[i]}}},\infty}} -A^{{\mathrm{{eff}}}})\,\overline{{\varepsilon}} $',
                      # label=r' $\overline{\varepsilon}^{T} (A_{h,k}^{\mathrm{eff}} -A^{\mathrm{eff}}_{h,\infty})\,\overline{\varepsilon} $',
                      color=_colors[f'{i}'],
                      linestyle='--',
                      alpha=1, marker=_markers[f'{i}'], linewidth=1, markersize=4, markevery=5)

    # plot total error Aeff_h_k - Aeff_fine_grid_k[-1] $
    e_tot = Aeff_h_k - Aeff_fine_grid_k[-1]
    ax_norms.semilogy(e_tot,  # grids_sizes[i]
                      label=fr'e$^\mathrm{{tot}}_{{ {{{2 ** grids_sizes[i]}}},k}}$',
                      # label=fr' $\overline{{\varepsilon}}^{{T}} (A_{{ {{{2 ** grids_sizes[i]}}},k}}^{{\mathrm{{eff}}}} -A^{{\mathrm{{eff}}}})\,\overline{{\varepsilon}} $',
                      # label=r' $\overline{\varepsilon}^{T} (A_{h,k}^{\mathrm{eff}} -A^{\mathrm{eff}}_{h,\infty})\,\overline{\varepsilon} $',
                      color=_colors[f'{i}'],
                      alpha=1, marker=_markers[f'{i}'], linewidth=1, markersize=4, markevery=5)

# plt.title('optimizer {}'.format(optimizer))
# ax_norms.set_ylabel('Norms')
ax_norms.set_ylim(1e-6, 1e3)
# ax_norms.set_yticks([1, 34, 67, 100])
# ax_norms.set_yticklabels([1, 34, 67, 100])

ax_norms.set_xlabel(r'PCG iteration - $k$')

ax_norms.set_xlim([0, error_in_Aeff_hk.__len__() - 1])
# ax_norms.set_xticks([1, len(eig_G) // 2, len(eig_G)])
# ax_norms.set_xticklabels([1, len(eig_G) // 2, len(eig_G)])

plt.grid(True)

plt.legend(loc='best', ncol=3)
fig_name = f'norm_evolution_h{number_of_pixels[0]}'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')

# fig.tight_layout()
fname = figure_folder_path + fig_name + '{}'.format('.pdf')
print(('create figure: {}'.format(fname)))
plt.savefig(fname, bbox_inches='tight'
            )
plt.show()
