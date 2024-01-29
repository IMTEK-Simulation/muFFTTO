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
N = 90
number_of_pixels = 3 * (N,)
# number_of_pixels = np.asarray( (30,30,1.8*30), dtype=int)

geometry_ID = 'geometry_I_2_3D'

# problem_type = 'elasticity'
# dataset_name = 'exp_data/' + f'muFFTTO_{problem_type}_{formulation}_{geometry_ID}_N{number_of_pixels[0]}_all.nc'
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

# field_name ='gradient_field'
# field=np.asarray(loaded_dataset.variables[field_name]).mean(axis=2)[0,2]


# field_geometry_norm = np.linalg.norm(field_geometry)
# field_temperature_norm = np.linalg.norm(field_temperature)
# field_gradient_x_norm = np.linalg.norm(field_gradient_x)
# field_gradient_y_norm = np.linalg.norm(field_gradient_y)
# field_gradient_z_norm = np.linalg.norm(field_gradient_z)


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(4, 3),
                         gridspec_kw={'wspace': 0.5, 'hspace': 0.5})

index = 0

# filter_pars= [0, 0.5, 1, 2]  # 0.1, 0.5, 1, 2, 4, 8
error_levels = np.linspace(0, 1, 50)

abs_error_norms = np.empty([error_levels.__len__()])
rel_error_norms = np.empty([error_levels.__len__()])

memory_lr = np.empty([error_levels.__len__()])
ranks_i = np.empty([error_levels.__len__(),4])


for field_name in fields:
    print(field_name)
    field = fields[field_name]
    # field = gaussian_filter(field, sigma=filter_pars[1])

    for i in np.arange(error_levels.__len__()):
        # if rank_i == 84:
        #    continue
        # print(rank_i)
        # rank_j = rank_i
        # for rank_j in np.arange(0, number_of_pixels[0]):

        # ranks = np.asarray([1, rank_i + 1, rank_j + 1, 1])
        rel_error_norm_i = error_levels[i]
        field_norm = np.linalg.norm(field)
        tt_field, ranks = tensor_train_tools.tt_decompose_error(tensor_xyz=field,
                                                                rel_error_norm=rel_error_norm_i)

        tt_reconstructed_field = tensor_train_tools.tt_to_full_format(tt_cores=tt_field)

        abs_error_norms[i] = (np.linalg.norm(field - tt_reconstructed_field))
        rel_error_norms[i] = (abs_error_norms[i] / field_norm)
        memory_lr[i] = (number_of_pixels[0] * ranks[1] +
                                     ranks[1] * number_of_pixels[1] * ranks[2] +
                                     number_of_pixels[-1] * ranks[2])
        ranks_i[i,:]=ranks

        # if rank_i % 5 == 0 and rank_i < 35:
        #     axes_2[index].plot(np.arange(0, number_of_pixels[0]),
        #                        tt_reconstructed_field[:, tt_reconstructed_field.shape[1] // 2, 0],
        #                        label='rank = {}'.format(rank_i + 1))

    axes[0].semilogy(ranks_i[:,1] ,rel_error_norms,
                  label='{}'.format(field_name))
    axes[0].grid(True)
    axes[0].set_xlabel('Errors')
    axes[0].set_ylabel('ranks_i')
    #axes[0].set_title(' {}'.format(field_name))

    axes[1].semilogy(ranks_i[:,2] ,rel_error_norms,
              label='{}'.format(field_name))
    axes[1].grid(True)
    axes[1].set_xlabel('Errors')
    axes[1].set_ylabel('ranks_i')


#axes.set_ylim([1e-6, 1])

# axes[1].semilogy(np.arange(0, number_of_pixels[0]), np.diag(memory_lr) / N ** 3)
# axes[1].grid(True)
# axes[1].set_xlabel('Ranks')
# axes[1].set_ylabel('Memory')

axes[0].legend(loc='best')

# # axes[0, 1].legend(loc='upper right')
# fig.suptitle(' {}'.format(geometry_ID))
#
# src = './figures/'  # source folder\
# fig_data_name = f'muFFTTO_{problem_type}_{geometry_ID}_N{number_of_pixels[0]}_filt{filter_par}'
# parf = set_pars(mpl)
#
# fname = src + fig_data_name + '{}'.format('.pdf')
# print(('create figure: {}'.format(fname)))  # axes[1, 0].legend(loc='upper right')
# plt.savefig(fname, dpi=parf['dpi'], pad_inches=parf['pad_inches'], bbox_inches='tight')
# print('END plot ')
#
# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 3),
#                          gridspec_kw={'wspace': 0.5, 'hspace': 0.5})
# axes.semilogy(np.arange(0, number_of_pixels[0]), np.diag(memory_lr) / N ** 3)
# axes.grid(True)
# axes.set_xlabel('Ranks')
# axes.set_ylabel('Memory')
#
# fname = src + fig_data_name + 'memory{}'.format('.pdf')
# print(('create figure: {}'.format(fname)))  # axes[1, 0].legend(loc='upper right')
# plt.savefig(fname, dpi=parf['dpi'], pad_inches=parf['pad_inches'], bbox_inches='tight')
# print('END mem plot ')
#
# fig_g, ax = microstructure_library.visualize_voxels(phase_field_xyz=field_geometry)
# fig_g.suptitle(' {}'.format(geometry_ID))
#
# fname = src + fig_data_name + '_geometry{}'.format('.pdf')
# print(('create figure: {}'.format(fname)))  # axes[1, 0].legend(loc='upper right')
# plt.savefig(fname, dpi=parf['dpi'], pad_inches=parf['pad_inches'], bbox_inches='tight')
# print('END plot ')

plt.show()
