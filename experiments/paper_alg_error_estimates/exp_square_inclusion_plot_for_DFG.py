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

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

# results set up
fig = plt.figure(figsize=(7, 4.5))
gs = fig.add_gridspec(1, 1, hspace=0.1, wspace=0.1,
                      width_ratios=[1], height_ratios=[1])
ax_norms = fig.add_subplot(gs[0])

grids_sizes = [  5,  7, 9]

# get most precise solution
number_of_pixels = (2 ** 10, 2 ** 10)
results_name = f'N{number_of_pixels[0]}_rho_inc{mat_contrast_2:.0f}_mat{mat_contrast:.0f}_ani{a_ani:.0f}'

_info = np.load(data_folder_path + results_name + '_log.npz', allow_pickle=True)
Aeff_h_precise = _info['Aeff_h_precise'][0, 0]

# store handles for grouped legends
handles_per_grid = {}

counter = 0
for n in grids_sizes:
    counter += 1
    number_of_pixels = (2 ** n, 2 ** n)
    results_name = f'N{number_of_pixels[0]}_rho_inc{mat_contrast_2:.0f}_mat{mat_contrast:.0f}_ani{a_ani:.0f}'

    _info = np.load(data_folder_path + results_name + '_log.npz', allow_pickle=True)

    Aeff_hk = _info['Aeff_hk']
    error_in_Aeff_hk_precise = Aeff_hk - Aeff_h_precise
    lower_estim = _info['lower_estim']

    # plot curves and capture handles
    if anisotropy:
        line_total, = ax_norms.semilogy(
            error_in_Aeff_hk_precise,
            label='Total error',
            color=_colors[f'{counter}']
        )
    else:
        line_total, = ax_norms.semilogy(
            error_in_Aeff_hk,
            linestyle=':',
            label='Total error',
            color=_colors[f'{counter}']
        )

    line_iter, = ax_norms.semilogy(
        lower_estim,
        linestyle='--',
        label='Iterative error',
        color=_colors[f'{counter}']
    )

    # store handles for this grid size
    handles_per_grid[f'Grid = {number_of_pixels[0]}²'] = [line_total, line_iter]

# -----------------------------
# Create stacked legends
# -----------------------------

grid_labels = list(handles_per_grid.keys())

# first legend
legend1 = ax_norms.legend(
    handles_per_grid[grid_labels[0]],
    ['Total error', 'Iterative error'],
    title=grid_labels[0],
    loc='upper right',
    bbox_to_anchor=(0.4, 1.01)
)
ax_norms.add_artist(legend1)

# second legend
legend2 = ax_norms.legend(
    handles_per_grid[grid_labels[1]],
    ['Total error', 'Iterative error'],
    title=grid_labels[1],
    loc='upper right',
    bbox_to_anchor=(.7, 1.01)
)
ax_norms.add_artist(legend2)

# third legend
ax_norms.legend(
    handles_per_grid[grid_labels[2]],
    ['Total error', 'Iterative error'],
    title=grid_labels[2],
    loc='upper right',
    bbox_to_anchor=(1. , 1.01)
)

# -----------------------------
# Axes formatting
# -----------------------------
ax_norms.set_ylim(1e-6, 1e1)
ax_norms.set_xlim(1, 200)
ax_norms.set_xlabel(r'$k$-th iteration of FFT-based solver')
ax_norms.set_ylabel('Error')

fig.tight_layout()
fig_name = f'norm_evolution_kappa{anisotropy}'
fname = figure_folder_path + fig_name +f'{grids_sizes}'+ '{}'.format('.png')
print(('create figure: {}'.format(fname)))
plt.savefig(fname, bbox_inches='tight')
plt.show()

####################################### PLOT Efficiencies
quit()
