import numpy as np
import scipy as sp
import matplotlib as mpl
import time
import sys
import argparse
import gc
import tracemalloc

tracemalloc.start()
# run iterations


sys.path.append("/")
sys.path.append('../..')  # Add parent directory to path

import matplotlib.pyplot as plt

from NuMPI import Optimization
from NuMPI.IO import save_npy, load_npy

from mpi4py import MPI
from muGrid import FileIONetCDF, OpenMode, Communicator

plt.rcParams['text.usetex'] = True

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import topology_optimization
from muFFTTO import microstructure_library
from muGrid import ConvolutionOperator


parser = argparse.ArgumentParser(
    prog="exp_paper_JG_2D_elasticity_TO.py", description="Solve topology optimization of negative poison ratio"
)
parser.add_argument("-n", "--nb_pixels", default="16")
# Preconditioner type (string, choose from a set)
parser.add_argument(
    "-p", "--preconditioner_type",
    type=str,
    choices=["Green", "Jacobi", "Green_Jacobi"],  # example options
    default="Green_Jacobi",
    help="Type of preconditioner to use"
)
parser.add_argument(
    "-s", "--save_phases",
    action="store_true",
    help="Enable saving phases"
)
# Total phase contrast (integer)

args = parser.parse_args()
nb_pixels = int(args.nb_pixels)
preconditioner_type = args.preconditioner_type
save_data = args.save_phases

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'  # 'bilinear_rectangle'##'linear_triangles' # # linear_triangles_tilled
formulation = 'small_strain'

domain_size = [1, 1]  #
number_of_pixels = (nb_pixels, nb_pixels)
if MPI.COMM_WORLD.rank == 0:
    print(number_of_pixels)
dim = np.size(number_of_pixels)
my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)
start_time = time.time()

if MPI.COMM_WORLD.rank == 0:
    print('number_of_pixels = \n {} core {}'.format(number_of_pixels, MPI.COMM_WORLD.rank))
    print('  Rank   Size          Domain       Subdomain        Location')
    print('  ----   ----          ------       ---------        --------')
MPI.COMM_WORLD.Barrier()  # Barrier so header is printed first

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)

phase_field_1nxyz = discretization.get_scalar_field(name='phase_field_in_objective')
phase_field_at_quad_poits_1qxyz = discretization.get_quad_field_scalar(name='phase_field_at_quad_poits_1qxyz')



def memory_loop(phase_field_1nxyz):

    # # Phase field  in quadrature points
    discretization.fft.communicate_ghosts(field=phase_field_1nxyz)
    conv_op = ConvolutionOperator( [0,0 ], discretization.N_at_quad_points_qnijk)
    conv_op.apply(nodal_field=phase_field_1nxyz,
                                quadrature_point_field=phase_field_at_quad_poits_1qxyz)


if __name__ == '__main__':
    phase_field_0 = discretization.get_scalar_field(name='phase_field_in_initial_')

    for i in range(100000):
        if MPI.COMM_WORLD.rank == 0:
            print(f'iteration :{i}')
        phase_field_0.s += 0.5 * np.random.rand(*phase_field_0.s.shape) ** 1
        memory_loop(phase_field_0)


    plt.imshow(phase_field_0.s[0,0])
    plt.show()