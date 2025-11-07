# Solver for unit impulse response  using muFFT

import argparse

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None
import numpy as np
import muGrid
from muGrid.Solvers import conjugate_gradients
from muFFT import FFT

try:
    from mpi4py import MPI

    comm = muGrid.Communicator(MPI.COMM_WORLD)
except ImportError:
    comm = muGrid.Communicator()

from NuMPI.Testing.Subdivision import suggest_subdivisions

parser = argparse.ArgumentParser(
    prog="Poisson", description="Solve the Poisson equation"
)
parser.add_argument("-n", "--nb-grid-pts", default="3,32")
args = parser.parse_args()

nb_grid_pts = [int(x) for x in args.nb_grid_pts.split(",")]

fft = FFT(nb_grid_pts=nb_grid_pts,
          engine='fftwmpi',
          communicator=comm,
         nb_ghosts_left=[1, 1],
          nb_ghosts_right=[1, 1],
          )
print(fft.subdomain_slices)
fft.create_plan(1)
# s = suggest_subdivisions(len(nb_grid_pts), comm.size)

# decomposition = muGrid.CartesianDecomposition(comm, nb_grid_pts, s, (1, 1), (1, 1))
# fc = decomposition.collection
fc = fft.real_field_collection
fc.set_nb_sub_pts('quad_points', 2)
# fc.set_nb_sub_pts('nodal_points', 1)

# Get nodal field
nodal_field = fc.real_field("nodal-field", )

# Get quadrature field of shape (2, quad, nx, ny)
quad_field = fc.real_field("quad-field", (2,), "quad_points")
if MPI.COMM_WORLD.rank == 0:#MPI.COMM_WORLD.size-1:
    nodal_field.s[0, :,0] = 1

print(f' unit impulse before communication: nodal field with buffers in rank {MPI.COMM_WORLD.rank} \n ' + f'{nodal_field.sg}')
MPI.COMM_WORLD.Barrier()


fft.communicate_ghosts(nodal_field)
print(f' unit impuls after communication: nodal field with buffers in rank {MPI.COMM_WORLD.rank} \n ' + f'{nodal_field.sg}')

MPI.COMM_WORLD.Barrier()

# grid_spacing = 1 / np.array(nb_grid_pts)  # Grid spacing
#
# stencil = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # FD-stencil for the Laplacian
# laplace = muGrid.ConvolutionOperator([-1, -1], stencil)

# Derivative stencil of shape (2, quad, 2, 2)
gradient = np.array(
    [
        [  # Derivative in x-direction
            [[[-1, 0], [1, 0]]],  # Bottom-left triangle (first quadrature point)
            [[[0, -1], [0, 1]]],  # Top-right triangle (second quadrature point)
        ],
        [  # Derivative in y-direction
            [[[-1, 1], [0, 0]]],  # Bottom-left triangle (first quadrature point)
            [[[0, 0], [-1, 1]]],  # Top-right triangle (second quadrature point)
        ],
    ],
)
gradient_op = muGrid.ConvolutionOperator([0, 0], gradient)

# Apply the gradient operator to the nodal field and write result to the quad field
gradient_op.apply(nodal_field, quad_field)

fft.communicate_ghosts(quad_field)

gradient_op.transpose(quadrature_point_field=quad_field,
                      nodal_field=nodal_field,
                      weights=[1, 1]
                      )

fft.communicate_ghosts(nodal_field)

print(f'unit impuls response : nodal field with buffers in rank {MPI.COMM_WORLD.rank} \n ' + f'{nodal_field.s}')

# plt.figure()
# plt.plot(nodal_field.s[0, 0])
# plt.show()
