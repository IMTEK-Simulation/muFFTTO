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

domain_size = [1, 1, 1]
pixel_size = domain_size / np.asarray(number_of_pixels)

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

#field_gradient_x = np.asarray(loaded_dataset.variables['gradient_field']).mean(axis=2)[0, 0]
#field_gradient_y = np.asarray(loaded_dataset.variables['gradient_field']).mean(axis=2)[0, 1]
#field_gradient_z = np.asarray(loaded_dataset.variables['gradient_field']).mean(axis=2)[0, 2]

field_gradient_x = (np.roll(field_temperature, -1, axis=0) - field_temperature) / pixel_size[0]
field_gradient_y = (np.roll(field_temperature, -1, axis=1) - field_temperature) / pixel_size[1]
field_gradient_z = (np.roll(field_temperature, -1, axis=2) - field_temperature) / pixel_size[2]

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


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 3),
                         gridspec_kw={'wspace': 0.5, 'hspace': 0.5})

index = 0
abs_error_norms = np.empty([number_of_pixels[0]])
rel_error_norms = np.empty([number_of_pixels[0]])
rel_error_norms_Dx = np.empty([number_of_pixels[0]])
rel_error_norms_Dy = np.empty([number_of_pixels[0]])
rel_error_norms_Dz = np.empty([number_of_pixels[0]])

memory_lr = np.empty([number_of_pixels[0]])

# filter_pars= [0, 0.5, 1, 2]  # 0.1, 0.5, 1, 2, 4, 8

for field_name in ['temperature']:
    print(field_name)
    field = fields[field_name]
    # field = gaussian_filter(field, sigma=filter_pars[1])
    field_gradient_x = fields['gradient_x']
    field_gradient_y = fields['gradient_y']
    field_gradient_z = fields['gradient_z']

    for rank_i in np.arange(0, N):
        # if rank_i == 84:
        #  continue
        # print(rank_i)
        rank_j = rank_i
        # for rank_j in np.arange(0, number_of_pixels[0]):

        ranks = np.asarray([1, rank_i + 1, rank_j + 1, 1])

        field_norm = np.linalg.norm(field)
        tt_cores = tensor_train_tools.tt_decompose_rank(tensor_xyz=field,
                                                        ranks=ranks)

        tt_reconstructed_field = tensor_train_tools.tt_to_full_format(tt_cores=tt_cores)

        abs_error_norms[rank_i] = (np.linalg.norm(field - tt_reconstructed_field))
        rel_error_norms[rank_i] = (abs_error_norms[rank_i] / field_norm)
        memory_lr[rank_i] = (number_of_pixels[0] * rank_i +
                             rank_i * number_of_pixels[1] * rank_j +
                             number_of_pixels[-1] * rank_j)

        tt_cores_grad = tensor_train_tools.get_gradient_finite_difference(tt_cores=tt_cores,
                                                                          voxel_sizes=pixel_size)

        tt_dNx_field_FD = tensor_train_tools.tt_to_full_format(tt_cores=tt_cores_grad[0])
        tt_dNy_field_FD = tensor_train_tools.tt_to_full_format(tt_cores=tt_cores_grad[1])
        tt_dNz_field_FD = tensor_train_tools.tt_to_full_format(tt_cores=tt_cores_grad[2])

        rel_error_norms_Dx[rank_i] = np.linalg.norm(field_gradient_x - tt_dNx_field_FD) / np.linalg.norm(
            field_gradient_x)
        rel_error_norms_Dy[rank_i] = np.linalg.norm(field_gradient_y - tt_dNy_field_FD) / np.linalg.norm(
            field_gradient_y)
        rel_error_norms_Dz[rank_i] = np.linalg.norm(field_gradient_z - tt_dNz_field_FD) / np.linalg.norm(
            field_gradient_z)


    fig_dz, ax = microstructure_library.visualize_voxels(phase_field_xyz=field_gradient_z)
    fig_TTdz, axTT = microstructure_library.visualize_voxels(phase_field_xyz=tt_dNz_field_FD)


    axes.semilogy(np.arange(0, N), rel_error_norms, label='{}'.format(field_name))
    axes.semilogy(np.arange(0, N), rel_error_norms_Dx, label='{}'.format('gradient_x'))
    axes.semilogy(np.arange(0, N), rel_error_norms_Dy, label='{}'.format('gradient_y'))
    axes.semilogy(np.arange(0, N), rel_error_norms_Dz, label='{}'.format('gradient_z'))

    axes.grid(True)
    axes.set_xlabel('Ranks')
    axes.set_ylabel('Rel. error')
    # axes[0, 0].set_title(' {}'.format(field_name))

axes.set_ylim([1e-6, 1])

# axes[1].semilogy(np.arange(0, number_of_pixels[0]), np.diag(memory_lr) / N ** 3)
# axes[1].grid(True)
# axes[1].set_xlabel('Ranks')
# axes[1].set_ylabel('Memory')

axes.legend(loc='best')
# axes[0, 1].legend(loc='upper right')
fig.suptitle(' {}'.format(geometry_ID))

plt.show()
