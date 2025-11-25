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

from mpi4py import MPI
from NuMPI.IO import save_npy, load_npy
from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library

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

# GENERATE GEOMETRIES
problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'

domain_size = [1, 1]
# Variables to be set up

for nb_pixels_power in np.arange(2, 10 + 1):
    nb_laminates = 2 ** nb_pixels_power

    #
    number_of_pixels = (2 ** nb_pixels_power, 2 ** nb_pixels_power)

    geometry_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                            problem_type=problem_type)

    discretization = domain.Discretization(cell=geometry_cell,
                                           nb_of_pixels_global=number_of_pixels,
                                           discretization_type=discretization_type,
                                           element_type=element_type)
    # geometry= discretization.get_scalar_field(name='geometry')
    x_coors = discretization.fft.coords
    geometry = (0.5 +
                0.25 * np.cos(2 * np.pi * x_coors[0] - 2 * np.pi * x_coors[1]) +
                0.25 * np.cos(2 * np.pi * x_coors[1] + 2 * np.pi * x_coors[0]))

    # save stress
    temp_max_size_ = {'nb_max_subdomain_grid_pts': discretization.nb_max_subdomain_grid_pts}
    results_name = (f'cos_geometry_pixels={nb_laminates}' + f'dof={nb_laminates}')
    to_save = np.copy(geometry)
    np.save(data_folder_path + results_name + f'.npy', to_save)
    for nb_of_disc_points in np.arange(nb_pixels_power, 10 ):
        dof = 2 ** (nb_of_disc_points+1)
        geometry = np.repeat(np.repeat(geometry, 2, axis=0), 2, axis=1)
        print('dof', dof)
        print('size geometry', geometry.shape)

        # save
        results_name = (f'cos_geometry_pixels={nb_laminates}' + f'dof={dof}')
        to_save = np.copy(geometry)
        np.save(data_folder_path + results_name + f'.npy', to_save)

        plt.figure()
        plt.imshow(geometry)
        plt.show()

quit()
