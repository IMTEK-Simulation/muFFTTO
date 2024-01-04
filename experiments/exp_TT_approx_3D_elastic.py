import numpy as np
from netCDF4 import Dataset
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter

# modify global setting
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'sans-serif'

from muFFTTO import microstructure_library
from muFFTTO import TT_tools

discretization_type = 'finite_element'
element_type = 'trilinear_hexahedron'
formulation = 'small_strain'
N = 40
number_of_pixels = 3 * (N,)
geometry_ID = 'geometry_III_1_3D'

#problem_type = 'elasticity'
#dataset_name = 'exp_data/' + f'muFFTTO_{problem_type}_{formulation}_{geometry_ID}_N{number_of_pixels[0]}_all.nc'

problem_type = 'conductivity'
dataset_name = 'exp_data/' + f'muFFTTO_{problem_type}_{geometry_ID}_N{number_of_pixels[0]}_all.nc'


# read dataset
loaded_dataset = Dataset(dataset_name)
#
# displacemet = loaded_dataset.variables['displacement_field']# displacement_field

# phase_field = loaded_dataset.variables['phase_field']

# microstructure_library.visualize_voxels(phase_field_xyz=geometry)
# routine
# field_name = 'phase_field' #
# field = np.asarray(loaded_dataset.variables[field_name])  #

# field_name = 'temperature_field'
# field = np.asarray(loaded_dataset.variables[field_name][0, 0])  #

field_name ='gradient_field'
field=np.asarray(loaded_dataset.variables[field_name]).mean(axis=2)[0,0]


field_norm = np.linalg.norm(field)
abs_error_norms = np.empty([number_of_pixels[0], number_of_pixels[0]])
rel_error_norms = np.empty([number_of_pixels[0], number_of_pixels[0]])
memory_lr = np.empty([number_of_pixels[0], number_of_pixels[0]])

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6),
                         gridspec_kw={'wspace': 0.5, 'hspace': 0.5})

fig_2, axes_2 = plt.subplots(nrows=1, ncols=4, figsize=(20, 6),
                             gridspec_kw={'wspace': 0.5, 'hspace': 0.5})
#microstructure_library.visualize_voxels(phase_field_xyz=field)
index = 0
for filter_par in [0, 0.5, 1, 2]:  # 0.1, 0.5, 1, 2, 4, 8
    if filter_par == 0:
        field_filtered = np.copy(field)
    else:
        field_filtered = gaussian_filter(field, sigma=filter_par)
    field_filtered_norm = np.linalg.norm(field_filtered)

    microstructure_library.visualize_voxels(phase_field_xyz=field_filtered )

    for rank_i in np.arange(0, N):
        rank_j = rank_i
        # for rank_j in np.arange(0, number_of_pixels[0]):

        ranks = np.asarray([1, rank_i + 1, rank_j + 1, 1])
        tt_field = TT_tools.tt_decompose_rank(tensor_xyz=field_filtered,
                                              ranks=ranks)

        tt_reconstructed_field = TT_tools.tt_to_full_format(tt_cores=tt_field)

        abs_error_norms[rank_i, rank_j] = (np.linalg.norm(field_filtered - tt_reconstructed_field))
        rel_error_norms[rank_i, rank_j] = (abs_error_norms[rank_i, rank_j] / field_filtered_norm)
        memory_lr[rank_i, rank_j] = (number_of_pixels[0] * rank_i +
                                     rank_i * number_of_pixels[1] * rank_j +
                                     number_of_pixels[-1] * rank_j)
        if rank_i % 5 == 0 and rank_i < 35:
            axes_2[index].plot(np.arange(0, number_of_pixels[0]),
                               tt_reconstructed_field[:, tt_reconstructed_field.shape[1] // 2, 0],
                               label='rank = {}'.format(rank_i + 1))

    axes_2[index].plot(np.arange(0, number_of_pixels[0]),
                       field[:, field.shape[1] // 2, 0],
                       label='origin')
    axes_2[index].legend(loc='upper right')
    axes_2[index].set_title('std = {}'.format(filter_par))

    #axes_2[index].set_ylim([-0.2, 1.2])
    index += 1
    # if rank_i == rank_j < 10:
    # microstructure_library.visualize_voxels(phase_field_xyz=tt_reconstructed_field)
    #   plot rank vs error
    axes[0, 0].semilogy(np.arange(0, number_of_pixels[0]), np.diag(rel_error_norms),
                        label='std = {}'.format(filter_par))
    axes[0, 0].grid(True)
    axes[0, 0].set_xlabel('Rank')
    axes[0, 0].set_ylabel('Rel. error')

    axes[1, 0].plot(np.arange(0, number_of_pixels[0]), field_filtered[:, field_filtered.shape[1] // 2, 0],
                    label='std = {}'.format(filter_par))
    axes[1, 0].set_xlabel('x coordinate')
    axes[1, 0].set_ylabel(field_name)

    # ,  # data
    # marker='o',  # each marker will be rendered as a circle
    # markersize=8,  # marker size
    # markerfacecolor='red',  # marker facecolor
    # markeredgecolor='black',  # marker edgecolor
    # markeredgewidth=2,  # marker edge width
    # linestyle='--',  # line style will be dash line
    # linewidth=3)  # line width
    # ax.set_zscale('log')
    # axes.set_xlabel('Ranks')
    # axes.set_ylabel('rel error')
#   plot rank vs memory
axes[0, 1].semilogy(np.arange(0, number_of_pixels[0]), np.diag(memory_lr) / N ** 3)
axes[0, 1].grid(True)
axes[0, 1].set_xlabel('Rank')
axes[0, 1].set_ylabel('Memory')

axes[0, 0].legend(loc='upper right')
axes[0, 1].legend(loc='upper right')
axes[1, 0].legend(loc='upper right')

plt.show()
# print(abs_error_norms)
# print(rel_error_norms)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 6),
                         gridspec_kw={'wspace': 0.5, 'hspace': 0.5})
axes[1].semilogy(np.arange(0, number_of_pixels[0]), np.diag(rel_error_norms),  # data
                 marker='o',  # each marker will be rendered as a circle
                 markersize=8,  # marker size
                 markerfacecolor='red',  # marker facecolor
                 markeredgecolor='black',  # marker edgecolor
                 markeredgewidth=2,  # marker edge width
                 linestyle='--',  # line style will be dash line
                 linewidth=3)  # line width

X, Y = np.meshgrid(np.arange(0, number_of_pixels[0]), np.arange(0, number_of_pixels[0]))

###### plot error with respec to two ranks
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')
# # Set scale of Z axis as logarithmic
# # ax.zaxis.set_major_formatter(ticker.LogFormatter())
#
# # Set logarithmic scale for the z-axis
# # ax.set_zscale('log')
# surf = ax.plot_surface(X, Y, rel_error_norms, cmap=plt.cm.cividis)
# # ax.set_zscale('log')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z (Log Scale)')
#
# # ax.set_xlabel('rank i')
# # ax.set_ylabel('rank j')
# # ax.set_zlabel('rel error norm')
# # ax.set_zticks([0,1e-4,1e-3,  1e-2,1e-1,1])
#
# ax.set_zlim(0, 1)
plt.show()
