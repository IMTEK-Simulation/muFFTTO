from cProfile import label

import numpy as np
import scipy as sc
import time
from mpi4py import MPI
from NuMPI.IO import save_npy, load_npy
import os

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library

script_name = 'exp_paper_JG_intro_circle'
folder_name = '../exp_data/'

file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
if not os.path.exists(file_folder_path):
    os.makedirs(file_folder_path)
data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
if not os.path.exists(data_folder_path):
    os.makedirs(data_folder_path)
figure_folder_path = file_folder_path + '/figures/' + script_name + '/'
if not os.path.exists(figure_folder_path):
    os.makedirs(figure_folder_path)

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'

domain_size = [1, 1]
nb_pix = 2 ** 8  # ,2,3,3,2,
number_of_pixels = (nb_pix, nb_pix)
tol_cg = 1e-3
contrast = 1e-4

ratios = np.arange(10000)

my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)
start_time = time.time()

# material distribution
geometry_ID = 'circle_inclusion'  # 'square_inclusion'#'circle_inclusion'#
phase_field_smooth = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                         microstructure_name=geometry_ID,
                                                         coordinates=discretization.fft.coords)
phase_field_smooth = np.abs(phase_field_smooth)

phase_field = np.abs(phase_field_smooth - 1)
phase_field = phase_field_smooth + contrast
for i in np.arange(ratios.size):
    nb_of_filters = ratios[i]
    _info = {}


    def apply_smoother(phase):
        # Define a 2D smoothing kernel
        kernel = np.array([[0.0625, 0.125, 0.0625],
                           [0.125, 0.25, 0.125],
                           [0.0625, 0.125, 0.0625]])
        # kernel = np.array([[0.0, 0.25, 0.0],
        #                    [0.25, 0., 0.25],
        #                    [0.0, 0.25, 0.0]])
        # Apply convolution for smoothing
        # phase[number_of_pixels[0] // 4 - 1:number_of_pixels[0] // 4 + 1, :] = 1e-4
        # phase[3 * number_of_pixels[0] // 4 - 1: 3 * number_of_pixels[0] // 4 + 1, :] = 1

        smoothed_arr = sc.signal.convolve2d(phase, kernel, mode='same', boundary='wrap')

        # smoothed_arr[number_of_pixels[0] // 4 - 1:number_of_pixels[0] // 4 + 1,:] = 1e-4
        # smoothed_arr[3*number_of_pixels[0] // 4 - 1: 3*number_of_pixels[0] // 4 + 1,:] =1
        return smoothed_arr


    def apply_smoother_log10(phase):
        # Define a 2D smoothing kernel
        kernel = np.array([[0.0625, 0.125, 0.0625],
                           [0.125, 0.25, 0.125],
                           [0.0625, 0.125, 0.0625]])
        # kernel = np.array([[0.0, 0.25, 0.0],
        #                    [0.25, 0., 0.25],
        #                    [0.0, 0.25, 0.0]])
        # Apply convolution for smoothing
        smoothed_arr = sc.signal.convolve2d(np.log10(phase), kernel, mode='same', boundary='wrap')
        # Fix bouarders
        smoothed_arr[0, :] = 0  # First row
        smoothed_arr[-1, :] = 0  # Last row
        smoothed_arr[:, 0] = 0  # First column
        smoothed_arr[:, -1] = 0  # Last column

        # Fix center point
        # smoothed_arr[number_of_pixels[0] // 2 - 1:number_of_pixels[0] // 2 + 1,
        # number_of_pixels[0] // 2 - 1:number_of_pixels[0] // 2 + 1] = -4

        # Fix boarders for laminate
        # smoothed_arr[number_of_pixels[0] // 4 - 1:number_of_pixels[0] // 4 + 1,:] = -4
        # smoothed_arr[3*number_of_pixels[0] // 4 - 1: 3*number_of_pixels[0] // 4 + 1,:] =0

        smoothed_arr = 10 ** smoothed_arr

        return smoothed_arr


    # if i == 0:
    # if nb_of_filters == 0:
    #
    if nb_of_filters > 0:
    #for aplication in np.arange(nb_of_filters):
        phase_field = apply_smoother_log10(phase_field)

    phase_fem = np.zeros([2, *number_of_pixels])
    phase_inxyz = discretization.get_scalar_field(name='phase_field')

    phase_inxyz.s[0, 0, ...] = np.copy(phase_field)

    results_name = (f'phase_field_N{number_of_pixels[0]}_F{nb_of_filters}_kappa{contrast}')

    save_npy(fn=data_folder_path + results_name + f'.npy', data=phase_inxyz.s.mean(axis=0)[0],
             subdomain_locations=tuple(discretization.subdomain_locations_no_buffers),
             nb_grid_pts=tuple(discretization.nb_of_pixels_global),
             components_are_leading=True,
             comm=MPI.COMM_WORLD)
    print(data_folder_path + results_name + f'.npy')


    if nb_of_filters in [62, 63, 64]:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(phase_inxyz.s[0, 0])
        plt.show()
