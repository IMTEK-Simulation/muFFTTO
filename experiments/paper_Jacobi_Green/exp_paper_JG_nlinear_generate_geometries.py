import time
import os
import sys
import argparse
import sys

sys.path.append("/home/martin/Programming/muFFTTO_paralellFFT_test/muFFTTO")
sys.path.append('../..')  # Add parent directory to path

from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library

if MPI.COMM_WORLD.rank == 0:
    print('  Rank   Size          Domain       Subdomain        Location')
    print('  ----   ----          ------       ---------        --------')
MPI.COMM_WORLD.Barrier()  # Barrier so header is printed first
script_name = os.path.splitext(os.path.basename(__file__))[0]

file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
if not os.path.exists(file_folder_path):
    os.makedirs(file_folder_path)
data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
if not os.path.exists(data_folder_path):
    os.makedirs(data_folder_path)
figure_folder_path = file_folder_path + '/figures/' + script_name + '/'
if not os.path.exists(figure_folder_path):
    os.makedirs(figure_folder_path)
src = '../figures/'

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'

domain_size = [1, 1]
# Variables to be set up

max_size=12
for nb_pixels_power in np.arange(2, max_size + 1):
    nb_laminates = 2 ** nb_pixels_power

    #
    number_of_pixels = (2 ** nb_pixels_power, 2 ** nb_pixels_power)

    geometry_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                            problem_type=problem_type)

    discretization = domain.Discretization(cell=geometry_cell,
                                           nb_of_pixels_global=number_of_pixels,
                                           discretization_type=discretization_type,
                                           element_type=element_type)

    # material distribution
    material_distribution = discretization.get_scalar_field(name='material_distribution')

    # x_coors = discretization.fft.coords[0]
    #
    # strip_size = x_coors.shape[0] // nb_laminates
    # phase_contrast = 10 ** 2
    # phases_single = phase_contrast + (1 - phase_contrast) / (1 - 1 / nb_laminates) * np.linspace(0, 1,
    #                                                                                              nb_laminates,
    #                                                                                              endpoint=False)
    # # print('phases_single = \n {}'.format(phases_single))
    #
    # # Repeat each element strip_size times
    # phases_repeated = np.repeat(phases_single, strip_size)

    steps_one_d = np.linspace(1, 2, nb_laminates, endpoint=True)

    # print('phases_repeated = \n {}'.format(phases_repeated))
    geometry = np.tile(steps_one_d[:, np.newaxis], (1, material_distribution.s.shape[-1]))

    results_name = (f'linear_geometry_pixels={nb_laminates}' + f'dof={nb_laminates}')
    to_save = np.copy(geometry)
    np.save(data_folder_path + results_name + f'.npy', to_save)

    for nb_of_disc_points in np.arange(nb_pixels_power, max_size):
        dof = 2 ** (nb_of_disc_points + 1)
        geometry = np.repeat(np.repeat(geometry, 2, axis=0), 2, axis=1)
        print('dof', dof)
        print('size geometry', geometry.shape)

        # save
        results_name = (f'linear_geometry_pixels={nb_laminates}' + f'dof={dof}')
        to_save = np.copy(geometry)
        np.save(data_folder_path + results_name + f'.npy', to_save)

        plt.figure()
        plt.imshow(geometry)
        plt.show()
