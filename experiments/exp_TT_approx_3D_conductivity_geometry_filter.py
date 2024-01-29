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
    #mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}',  r'\usepackage{amssymb}']
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = 'Computer Modern'
    #mpl.verbose.level = 'debug-annoying'


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
N =60
number_of_pixels = 3 * (N,)
# number_of_pixels = np.asarray( (30,30,1.8*30), dtype=int)

#geometry_ID = 'geometry_I_2_3D'
geometry_ID = 'geometry_III_1_3D'
# problem_type = 'elasticity'
# dataset_name = 'exp_data/' + f'muFFTTO_{problem_type}_{formulation}_{geometry_ID}_N{number_of_pixels[0]}_all.nc'
filter_par=1

problem_type = 'conductivity'
dataset_name = 'exp_data/' + f'muFFTTO_{problem_type}_{geometry_ID}_N{number_of_pixels[0]}_filt{filter_par}_all.nc'

# read dataset
loaded_dataset = Dataset(dataset_name)
#
# field_name = 'gradient_field'

# displacemet = loaded_dataset.variables['displacement_field']# displacement_field

# phase_field = loaded_dataset.variables['phase_field']
# field=np.asarray(loaded_dataset.variables[field_name]).mean(axis=2)[2,2]


# microstructure_library.visualize_voxels(phase_field_xyz=geometry)

# for field_name in ['phase_field','temperature_field','gradient_field']:

field_geometry = np.asarray(loaded_dataset.variables['phase_field'])  #

field_temperature = np.asarray(loaded_dataset.variables['temperature_field'][0, 0])  #

field_gradient_x = np.asarray(loaded_dataset.variables['gradient_field']).mean(axis=2)[0, 0]
field_gradient_y = np.asarray(loaded_dataset.variables['gradient_field']).mean(axis=2)[0, 1]
field_gradient_z = np.asarray(loaded_dataset.variables['gradient_field']).mean(axis=2)[0, 2]

fields = {'geometry': field_geometry}

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

index = 0


filter_pars= [0, 0.5, 1, 2]  # 0.1, 0.5, 1, 2, 4, 8



for field_name in fields:

    print(field_name)
    field_org = fields[field_name]

    for filter_par in filter_pars:  # 0.1, 0.5, 1, 2, 4, 8
        if filter_par == 0:
            field = np.copy(field_org)
        else:
            field= gaussian_filter(field_org, sigma=filter_par)

        for rank_i in np.arange(0, N):
            #if rank_i == 10:
             #   continue
            rank_j = rank_i
            # for rank_j in np.arange(0, number_of_pixels[0]):

            ranks = np.asarray([1, rank_i + 1, rank_j + 1, 1])

            field_norm = np.linalg.norm(field)
            tt_field = tensor_train_tools.tt_decompose_rank(tensor_xyz=field,
                                                            ranks=ranks)

            tt_reconstructed_field = tensor_train_tools.tt_to_full_format(tt_cores=tt_field)

            abs_error_norms[rank_i, rank_j] = (np.linalg.norm(field - tt_reconstructed_field))
            rel_error_norms[rank_i, rank_j] = (abs_error_norms[rank_i, rank_j] / field_norm)
            memory_lr[rank_i, rank_j] = (number_of_pixels[0] * rank_i +
                                         rank_i * number_of_pixels[1] * rank_j +
                                         number_of_pixels[-1] * rank_j)
            # if rank_i % 5 == 0 and rank_i < 35:
            #     axes_2[index].plot(np.arange(0, number_of_pixels[0]),
            #                        tt_reconstructed_field[:, tt_reconstructed_field.shape[1] // 2, 0],
            #                        label='rank = {}'.format(rank_i + 1))

        axes.semilogy(np.arange(0, number_of_pixels[0]), np.diag(rel_error_norms),
                         label='std = {}'.format(filter_par))
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


src = './figures/'  # source folder\
fig_data_name = f'muFFTTO_{problem_type}_{geometry_ID}_N{number_of_pixels[0]}'
parf = set_pars(mpl)

fname = src + fig_data_name+'filtering{}'.format('.pdf')
print(('create figure: {}'.format(fname)))# axes[1, 0].legend(loc='upper right')
plt.savefig(fname, dpi=parf['dpi'], pad_inches=parf['pad_inches'], bbox_inches='tight')
print('END plot ')




fig_g, ax  = microstructure_library.visualize_voxels(phase_field_xyz=field_geometry )
fig_g.suptitle(' {}'.format(geometry_ID))





plt.show()




