import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import os

# File that generates plots
script_name = 'exp_square_inclusion_for_DFG'
file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
figure_folder_path = file_folder_path + '/figures/' + script_name + '/'

from experiments.paper_alg_error_estimates import _labels, _colors, _markers, _linestyles

problem_type = 'conductivity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
geometry_ID = 'square_inclusion'

mat_contrast = 1  # matrix
mat_contrast_2 = 1e3  # inclusion
anisotropy = True
if anisotropy:
    a_ani = 10
else:
    a_ani = 1

# PLOT CONVERGENCE  /N32_rho_inc1000_mat1_ani1.npy
# results set up
fig = plt.figure(figsize=(7, 4.5))
gs = fig.add_gridspec(1, 1, hspace=0.1, wspace=0.1, width_ratios=1 * (1,),
                      height_ratios=[1, ])
ax_norms = fig.add_subplot(gs[0])

grids_sizes = [5, 7, 9]

# get most precise soluition
number_of_pixels = (2 ** 10, 2 ** 10)
results_name = f'N{number_of_pixels[0]}_rho_inc{mat_contrast_2:.0f}_mat{mat_contrast:.0f}_ani{a_ani:.0f}'

_info = np.load(data_folder_path + results_name + f'_log.npz', allow_pickle=True)
temp_field = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True)

homogenized_flux = _info['homogenized_flux']
Aeff_h_precise = _info['Aeff_h_precise'][0, 0]
counter = 0
for n in grids_sizes:
    counter += 1
    number_of_pixels = (2 ** n, 2 ** n)
    results_name = f'N{number_of_pixels[0]}_rho_inc{mat_contrast_2:.0f}_mat{mat_contrast:.0f}_ani{a_ani:.0f}'

    # Load necessary data for each grid size
    with np.load(data_folder_path + results_name + f'_log.npz', allow_pickle=True) as _info:
        Aeff_hk = _info['Aeff_hk']
        error_in_Aeff_hk = _info['error_in_Aeff_hk']
        lower_estim = _info['lower_estim']

    # Calculate precise error using pre-loaded Aeff_h_precise
    error_in_Aeff_hk_precise = Aeff_hk - Aeff_h_precise

    # Plot results with descriptive labels for each grid size
    if not anisotropy:
        ax_norms.semilogy(error_in_Aeff_hk,  # / error_in_Aeff_hk[0],
                          label=fr'Total Error (grid={number_of_pixels[0]}$^2$)',
                          color=_colors[str(counter)],
                          alpha=1.,
                          # marker=_markers['true_error'],
                          linewidth=1, markersize=5, markevery=5
                          )

    if anisotropy:
        ax_norms.semilogy(error_in_Aeff_hk_precise,  # / error_in_Aeff_hk_precise[0],
                          linestyle=':',
                          label=fr'Total Error (grid={number_of_pixels[0]}$^2$)',
                          color=_colors[str(counter)],
                          alpha=1.,
                          # marker=_markers['true_error'],
                          linewidth=1, markersize=5, markevery=5
                          )

    ax_norms.semilogy(lower_estim,  # / lower_estim[0],
                      linestyle='--',
                      color=_colors[str(counter)],

                      label=fr'Lower Est. (grid={number_of_pixels[0]}$^2$)',
                      )
# Finalize the plot outside the loop for efficiency
ax_norms.set_ylim(1e-6, 1e1)
ax_norms.set_xlim(0, 200)

ax_norms.set_xlabel(r'FFT- solver iteration - $k$')
ax_norms.set_ylabel('Error')
# plt.grid(True)
ax_norms.legend()
fig_name = f'norm_evolution_kappa{anisotropy}'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')

fig.tight_layout()
fname = figure_folder_path + fig_name + '{}'.format('.png')
print(('create figure: {}'.format(fname)))
plt.savefig(fname, bbox_inches='tight')
plt.show()
####################################### PLOT Efficiencies
quit()
