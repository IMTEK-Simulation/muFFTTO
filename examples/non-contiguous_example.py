import numpy as np
from mpi4py import MPI
import muFFT
from muGrid import ConvolutionOperator

# Two dimensional grid
nx, ny = nb_grid_pts = [1024, 10]

left_ghosts = [1, 1]
right_ghosts = [1, 1]

engine = muFFT.FFT(nb_grid_pts=nb_grid_pts,
             engine='fftwmpi',
             communicator=muFFT.Communicator(MPI.COMM_WORLD),
             nb_ghosts_left=left_ghosts,
             nb_ghosts_right=right_ghosts,
             )

fc = engine.real_field_collection
fc.set_nb_sub_pts('quad_points', 2)
fc.set_nb_sub_pts('nodal_points', 1)

# Get nodal field
nodal_field = fc.real_field("nodal-field",(1,),"nodal_points" )

# Get quadrature field of shape (2, quad, nx, ny)
quad_field = fc.real_field("quad-field", (2,), "quad_points")
# # Get nodal field
# nodal_field = fc.real_field("nodal-field")
#
# # Get quadrature field of shape (2, quad, nx, ny)
# quad_field = fc.real_field("quad-field", (2,), "quad_points")


# Fill nodal field with a sine-wave
x, y = nodal_field.icoords
nodal_field.p[0] = np.sin(2 * np.pi * x / nx)

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
op = ConvolutionOperator([0, 0], gradient)

# Apply the gradient operator to the nodal field and write result to the quad field
op.apply(nodal_field, quad_field)

# Check that the quadrature field has the correct derivative
np.testing.assert_allclose(
    quad_field.s[0, 0], 2 * np.pi * np.cos(2 * np.pi * (x + 0.25) / nx) / nx, atol=1e-5
)

print('Piča piča ')

print(f'{MPI.COMM_WORLD.rank:6} {MPI.COMM_WORLD.size:6} {str(engine.nb_domain_grid_pts):>15} '
      f'{str(engine.nb_subdomain_grid_pts):>15} {str(engine.subdomain_locations):>15}')
