# Solver for unit impulse response  using muGrid

import argparse

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None
import numpy as np
import muGrid
from muGrid.Solvers import conjugate_gradients

try:
    from mpi4py import MPI

    comm = muGrid.Communicator(MPI.COMM_WORLD)
except ImportError:
    comm = muGrid.Communicator()

from NuMPI.Testing.Subdivision import suggest_subdivisions

parser = argparse.ArgumentParser(
    prog="Poisson", description="Compute unit impuls response"
)
parser.add_argument("-n", "--nb-grid-pts", default="3,32")
args = parser.parse_args()

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

nb_grid_pts = [int(x) for x in args.nb_grid_pts.split(",")]

s = suggest_subdivisions(len(nb_grid_pts), comm.size)
print(s)
s=[1,4]
decomposition = muGrid.CartesianDecomposition(comm, nb_grid_pts, s, (1, 1), (1, 1))
fc = decomposition.collection

fc.set_nb_sub_pts('quad_points', 2)
nodal_field = fc.real_field("nodal-field", )
quad_field = fc.real_field("quad-field", (2,), "quad_points")

if MPI.COMM_WORLD.rank == 0:
    nodal_field.s[0, :, 0] = 1

print(f' unit impulse before communication: nodal field with buffers in rank {MPI.COMM_WORLD.rank} \n ' + f'{nodal_field.sg}')
decomposition.communicate_ghosts(nodal_field)
print(f' unit impuls after communication: nodal field with buffers in rank {MPI.COMM_WORLD.rank} \n ' + f'{nodal_field.sg}')


# Apply the gradient operator to the nodal field and write result to the quad field
gradient_op.apply(nodal_field, quad_field)

decomposition.communicate_ghosts(quad_field)
gradient_op.transpose(quadrature_point_field=quad_field,
                      nodal_field=nodal_field,
                      weights=[1, 1]
                      )
decomposition.communicate_ghosts(nodal_field)

print(f'unit impuls response : nodal field with buffers in rank {MPI.COMM_WORLD.rank} \n ' +  f'{nodal_field.sg}')


plt.figure()
plt.plot(nodal_field.s[0, 0])
plt.show()
