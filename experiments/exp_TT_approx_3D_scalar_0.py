import numpy as np
from netCDF4 import Dataset
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter


def set_pars(mpl):
    #    mpl.rc['text.latex.preamble'] = [r"\usepackage{amsmath,bm,amsfonts}"]

    mpl.rcParams['text.usetex'] = True
    # mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}',  r'\usepackage{amssymb}']
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = 'Computer Modern'
    # mpl.verbose.level = 'debug-annoying'

    params = {'text.usetex': True,
              'font.family': 'serif',
              'font.size': 12,
              'legend.fontsize': 10,
              }
    mpl.rcParams.update(params)
    fig_par = {'dpi': 1000,
               'facecolor': 'w',
               'edgecolor': 'k',
               'linewidth': 0.01,
               'figsize': (4, 3),
               'figsize3D': (4, 4),
               'pad_inches': 0.02,
               }

    return fig_par


# modify global setting
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'sans-serif'

from muFFTTO import microstructure_library
from muFFTTO import tensor_train_tools

discretization_type = 'finite_element'
element_type = 'trilinear_hexahedron'
formulation = 'small_strain'
N = 130
number_of_pixels = 3 * (N,)
#number_of_pixels = np.asarray( (N,N,1.8*N), dtype=int)

domain_size = [1, 1, 1]
pixel_size = domain_size / np.asarray(number_of_pixels)

# geometry_ID = 'geometry_III_1_3D'
for geometry_ID in ['geometry_III_2_3D']:

    # ['geometry_I_1_3D',
    #  'geometry_I_2_3D',
    #  'geometry_I_3_3D',
    #  'geometry_I_4_3D',
    #  'geometry_I_5_3D',
    #  'geometry_II_1_3D',
    #  'geometry_II_4_3D',
    #  'geometry_III_2_3D']:
    # problem_type = 'elasticity'
    # dataset_name = 'exp_data/' + f'muFFTTO_{problem_type}_{formulation}_{geometry_ID}_N{number_of_pixels[0]}_all.nc'
    # ['geometry_I_1_3D']:

    filter_par = 1

    problem_type = 'conductivity'
    dataset_name = 'exp_data/' + f'muFFTTO_{problem_type}_{geometry_ID}_N{number_of_pixels[0]}_filt{filter_par}_all.nc'

    # read dataset
    loaded_dataset = Dataset(dataset_name)
    #
    # field_name = 'gradient_field'

    # displacemet = loaded_d ataset.variables['displacement_field']# displacement_field

    # phase_field = loaded_dataset.variables['phase_field']
    # field=np.asarray(loaded_dataset.variables[field_name]).mean(axis=2)[2,2]

    # microstructure_library.visualize_voxels(phase_field_xyz=geometry)

    # for field_name in ['phase_field','temperature_field','gradient_field']:

    field_geometry = np.asarray(loaded_dataset.variables['phase_field'])  #

    field_temperature = np.asarray(loaded_dataset.variables['temperature_field'][0, 0])  #

    field_gradient_x = np.asarray(loaded_dataset.variables['gradient_field']).mean(axis=2)[0, 0]
    field_gradient_y = np.asarray(loaded_dataset.variables['gradient_field']).mean(axis=2)[0, 1]
    field_gradient_z = np.asarray(loaded_dataset.variables['gradient_field']).mean(axis=2)[0, 2]

    fields = {'geometry': field_geometry,
              'temperature': field_temperature,
              'gradient_x': field_gradient_x,
              'gradient_y': field_gradient_y,
              'gradient_z': field_gradient_z}

    symbols = {'geometry': r'\rho',
               'temperature': 'u',
               'gradient_x': r'\nabla_x u',
               'gradient_y': r'\nabla_y u',
               'gradient_z': r'\nabla_z u'}
    # field_name ='gradient_field'
    # field=np.asarray(loaded_dataset.variables[field_name]).mean(axis=2)[0,2]

    # field_geometry_norm = np.linalg.norm(field_geometry)
    # field_temperature_norm = np.linalg.norm(field_temperature)
    # field_gradient_x_norm = np.linalg.norm(field_gradient_x)
    # field_gradient_y_norm = np.linalg.norm(field_gradient_y)
    # field_gradient_z_norm = np.linalg.norm(field_gradient_z)

    abs_error_norms = np.empty([number_of_pixels[0], number_of_pixels[0]])
    rel_error_norms = np.empty([number_of_pixels[0], number_of_pixels[0]])

    memory_lr = np.empty([number_of_pixels[0], number_of_pixels[0]])

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 3),
                             gridspec_kw={'wspace': 0.5, 'hspace': 0.5})
    fig_combined, axes_combined = plt.subplots(nrows=1, ncols=1, figsize=(6, 6),
                                               gridspec_kw={'wspace': 0.5, 'hspace': 0.5})
    index = 0

    # filter_pars= [0, 0.5, 1, 2]  # 0.1, 0.5, 1, 2, 4, 8

    for field_name in fields:
        print(field_name)
        field = fields[field_name]
        # field = gaussian_filter(field, sigma=filter_pars[1])

        for rank_i in np.arange(0, N):
            # if rank_i == 25:
            #    continue
            print(rank_i)
            rank_j = rank_i
            # for rank_j in np.arange(0, number_of_pixels[0]):

            ranks = np.asarray([1, rank_i + 1, rank_j + 1, 1])

            field_norm = np.linalg.norm(field)
            tt_cores = tensor_train_tools.tt_decompose_rank(tensor_xyz=field,
                                                            ranks=ranks)
            tt_reconstructed_field = tensor_train_tools.tt_to_full_format(tt_cores=tt_cores)

            abs_error_norms[rank_i, rank_j] = (np.linalg.norm(field - tt_reconstructed_field))
            rel_error_norms[rank_i, rank_j] = (abs_error_norms[rank_i, rank_j] / field_norm)
            memory_lr[rank_i, rank_j] = (number_of_pixels[0] * (rank_i + 1) +
                                         (rank_i + 1) * number_of_pixels[1] * (rank_j + 1) +
                                         number_of_pixels[-1] * (rank_j + 1))
            # if rank_i % 5 == 0 and rank_i < 35:
            #     axes_2[index].plot(np.arange(0, number_of_pixels[0]),
            #                        tt_reconstructed_field[:, tt_reconstructed_field.shape[1] // 2, 0],
            #                        label='rank = {}'.format(rank_i + 1))

        axes.semilogy(np.arange(0, number_of_pixels[0]), np.diag(rel_error_norms),
                      label=r'${}$'.format(symbols[field_name]))
        axes.grid(True)
        axes.set_xlabel('Ranks')
        axes.set_ylabel('Rel. error')

        axes_combined.semilogy(np.arange(1, number_of_pixels[0] + 1), np.diag(rel_error_norms),
                               label=r'${}$'.format(symbols[field_name]))

        axes_combined.grid(True)
        axes_combined.set_xlabel('Ranks')
        axes_combined.set_ylabel('Rel. error')
        axes_combined.set_ylim([1e-6, 1])
        # axes[0, 0].set_title(' {}'.format(field_name))

    axes.set_ylim([1e-6, 1])
    axes_combined.legend(loc='lower left')

    axesfig_combined_2 = axes_combined.twinx()
    axesfig_combined_2.semilogy(np.arange(1, number_of_pixels[0] + 1), np.diag(memory_lr) / N ** 3, color='k',
                                label='memory')
    axesfig_combined_2.grid(True)
    axesfig_combined_2.set_ylabel('Memory efficiency')
    axesfig_combined_2.legend(loc='lower right')
    axesfig_combined_2.set_ylim([1e-6, 1])

    # axes[0, 1].legend(loc='upper right')
    fig.suptitle(' {}'.format(geometry_ID))

    src = './figures/'  # source folder\
    fig_data_name = f'muFFTTO_{problem_type}_{geometry_ID}_N{number_of_pixels[0]}_filt{filter_par}'
    parf = set_pars(mpl)

    fname = src + fig_data_name + '{}'.format('.pdf')
    print(('create figure: {}'.format(fname)))  # axes[1, 0].legend(loc='upper right')
    plt.savefig(fname, dpi=parf['dpi'], pad_inches=parf['pad_inches'], bbox_inches='tight')
    print('END plot ')

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 3),
                             gridspec_kw={'wspace': 0.5, 'hspace': 0.5})
    axes.semilogy(np.arange(0, number_of_pixels[0]), np.diag(memory_lr) / N ** 3)
    axes.grid(True)
    axes.set_xlabel('Ranks')
    axes.set_ylabel('Memory')

    fname = src + fig_data_name + 'memory{}'.format('.pdf')
    print(('create figure: {}'.format(fname)))  # axes[1, 0].legend(loc='upper right')
    plt.savefig(fname, dpi=parf['dpi'], pad_inches=parf['pad_inches'], bbox_inches='tight', edgecolor=None)
    print('END mem plot ')

    fig_g, ax = microstructure_library.visualize_voxels(phase_field_xyz=field_geometry)
    fig_g.suptitle(' {}'.format(geometry_ID))

    fname = src + fig_data_name + '_geometry{}'.format('.pdf')
    print(('create figure: {}'.format(fname)))  # axes[1, 0].legend(loc='upper right')
    plt.savefig(fname, dpi=parf['dpi'], pad_inches=parf['pad_inches'], bbox_inches='tight',
                facecolor='auto', edgecolor='auto')
    print('END plot ')

    fig_dz, ax = microstructure_library.visualize_voxels(phase_field_xyz=field_gradient_z)
    fig_dz.suptitle(' {}'.format(geometry_ID))

    fname = src + fig_data_name + '_grad_z{}'.format('.pdf')
    print(('create figure: {}'.format(fname)))  # axes[1, 0].legend(loc='upper right')
    plt.savefig(fname, dpi=parf['dpi'], pad_inches=parf['pad_inches'], bbox_inches='tight')
    print('END plot ')

    # fig_2, axes_2 = plt.subplots(nrows=1, ncols=4, figsize=(20, 6),
    #                              gridspec_kw={'wspace': 0.5, 'hspace': 0.5})
    # axes_2[index].plot(np.arange(0, number_of_pixels[0]),
    #                    field_geometry[:, field_geometry.shape[1] // 2, 0],
    #                    label='origin')
    # axes_2[index].legend(loc='upper right')
    # axes_2[index].set_title(' {}'.format(field_name))
    #
    #
    #
    #
    # #axes_2[index].set_ylim([-0.2, 1.2])
    # index += 1
    # # if rank_i == rank_j < 10:
    # # microstructure_library.visualize_voxels(phase_field_xyz=tt_reconstructed_field)
    # #   plot rank vs error
    #
    #
    # axes[1, 0].plot(np.arange(0, number_of_pixels[0]), field_geometry[:, field_geometry.shape[1] // 2, 0],
    #                 label='std = {}'.format(0))
    # axes[1, 0].set_xlabel('x coordinate')
    # axes[1, 0].set_ylabel(field_name)
    #
    # # ,  # data
    #     # marker='o',  # each marker will be rendered as a circle
    #     # markersize=8,  # marker size
    #     # markerfacecolor='red',  # marker facecolor
    #     # markeredgecolor='black',  # marker edgecolor
    #     # markeredgewidth=2,  # marker edge width
    #     # linestyle='--',  # line style will be dash line
    #     # linewidth=3)  # line width
    #     # ax.set_zscale('log')
    #     # axes.set_xlabel('Ranks')
    #     # axes.set_ylabel('rel error')
    # #   plot rank vs memory
    # axes[0, 1].semilogy(np.arange(0, number_of_pixels[0]), np.diag(memory_lr) / N ** 3)
    # axes[0, 1].grid(True)
    # axes[0, 1].set_xlabel('Rank')
    # axes[0, 1].set_ylabel('Memory')
    #
    # axes[0, 0].legend(loc='upper right')
    # axes[0, 1].legend(loc='upper right')
    # axes[1, 0].legend(loc='upper right')
    #
    # plt.show()
    # # print(abs_error_norms)
    # # print(rel_error_norms)
    #
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 6),
    #                          gridspec_kw={'wspace': 0.5, 'hspace': 0.5})
    # axes[1].semilogy(np.arange(0, number_of_pixels[0]), np.diag(rel_error_norms),  # data
    #                  marker='o',  # each marker will be rendered as a circle
    #                  markersize=8,  # marker size
    #                  markerfacecolor='red',  # marker facecolor
    #                  markeredgecolor='black',  # marker edgecolor
    #                  markeredgewidth=2,  # marker edge width
    #                  linestyle='--',  # line style will be dash line
    #                  linewidth=3)  # line width
    #
    # X, Y = np.meshgrid(np.arange(0, number_of_pixels[0]), np.arange(0, number_of_pixels[0]))
    #
    # ###### plot error with respec to two ranks
    # # fig = plt.figure(figsize=(10, 10))
    # # ax = fig.add_subplot(111, projection='3d')
    # # # Set scale of Z axis as logarithmic
    # # # ax.zaxis.set_major_formatter(ticker.LogFormatter())
    # #
    # # # Set logarithmic scale for the z-axis
    # # # ax.set_zscale('log')
    # # surf = ax.plot_surface(X, Y, rel_error_norms, cmap=plt.cm.cividis)
    # # # ax.set_zscale('log')
    # # ax.set_xlabel('X')
    # # ax.set_ylabel('Y')
    # # ax.set_zlabel('Z (Log Scale)')
    # #
    # # # ax.set_xlabel('rank i')
    # # # ax.set_ylabel('rank j')
    # # # ax.set_zlabel('rel error norm')
    # # ax.set_zticks([0,1e-4,1e-3,  1e-2,1e-1,1])
    #
    # ax.set_zlim(0, 1)
plt.show()
