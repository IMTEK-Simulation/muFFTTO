
import os
import sys

sys.path.append("/home/martin/Programming/muFFTTO_paralellFFT_test/muFFTTO")
sys.path.append('../..')  # Add parent directory to path

from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

from muFFTTO import domain

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
element_type = 'trilinear_hexahedron'
formulation = 'small_strain'

domain_size = [1, 1, 1]
# Variables to be set up
import numpy as np

def coarsen_binary_round(arr, factor):
    """
    Deterministic coarsening of a binary 3D microstructure.
    For each block: compute mean, round to 0 or 1.
    """
    nx, ny, nz = arr.shape
    assert nx % factor == ny % factor == nz % factor == 0

    # reshape into blocks
    arr_b = arr.reshape(nx//factor, factor,
                        ny//factor, factor,
                        nz//factor, factor)

    # compute block means
    phi_block = arr_b.mean(axis=(1, 3, 5))

    # deterministic rounding
    return np.rint(phi_block).astype(arr.dtype)

# load fines geometry -- generated jsut finnest, and the coarsen others
N=256
results_name = (f'bubbles_' + f'dof={N}')
#to_save = np.copy(geometry)
finest_geometry=np.load(data_folder_path + results_name + f'.npy')

geom_128 = coarsen_binary_round(finest_geometry, factor=2)
results_name = (f'bubbles_' + f'dof={N//2}')
np.save(data_folder_path + results_name + f'.npy', geom_128)

geom_64  = coarsen_binary_round(finest_geometry, factor=4)
results_name = (f'bubbles_' + f'dof={N//4}')
np.save(data_folder_path + results_name + f'.npy', geom_64)

geom_32  = coarsen_binary_round(finest_geometry, factor=8)
results_name = (f'bubbles_' + f'dof={N//8}')
np.save(data_folder_path + results_name + f'.npy', geom_32)

geom_16  = coarsen_binary_round(finest_geometry, factor=16)
results_name = (f'bubbles_' + f'dof={N//16}')
np.save(data_folder_path + results_name + f'.npy', geom_16)

plt.figure()
plt.imshow(geom_128[...,64])
plt.show()

plt.imshow(geom_64[...,32])
plt.show()

plt.imshow(geom_32[...,16])
plt.show()

plt.imshow(geom_16[...,7])
plt.show()
max_size=6

generate_new=False
if generate_new:

    pixel_sizes =np.array([50,])
for nb_pixels_power in pixel_sizes:

   # nb_laminates = 2 ** nb_pixels_power
    #nb_laminates=200
    #
    #number_of_pixels = (2 ** nb_pixels_power, 2 ** nb_pixels_power, 2 ** nb_pixels_power)
    number_of_pixels =  (nb_pixels_power, nb_pixels_power, nb_pixels_power)

    geometry_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                            problem_type=problem_type)

    discretization = domain.Discretization(cell=geometry_cell,
                                           nb_of_pixels_global=number_of_pixels,
                                           discretization_type=discretization_type,
                                           element_type=element_type)

    # material distribution
    material_distribution = discretization.get_scalar_field(name='material_distribution')


    def generate_circular_inclusions(coords, num_inclusions, radius, seed=None):
        """
        Generate a boolean array with randomly distributed spherical inclusions
        with periodic boundary conditions.

        Parameters
        ----------
        coords : ndarray, shape [3, Nx, Ny, Nz]
            Grid coordinates
        num_inclusions : int
            Number of inclusions to place
        radius : float
            Radius of each inclusion
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        ndarray, shape [Nx, Ny, Nz]
            Boolean array where 1 indicates inclusion, 0 indicates matrix
        """
        if seed is not None:
            np.random.seed(seed)

        # Get domain bounds and size
        x_min, x_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        z_min, z_max = coords[2].min(), coords[2].max()

        # Domain lengths
        Lx = x_max - x_min
        Ly = y_max - y_min
        Lz = z_max - z_min

        # Initialize output array
        inclusions = np.zeros(coords.shape[1:], dtype=np.int32)

        # Generate random center positions
        centers_x = np.random.uniform(x_min, x_max, num_inclusions)
        centers_y = np.random.uniform(y_min, y_max, num_inclusions)
        centers_z = np.random.uniform(z_min, z_max, num_inclusions)

        # Mark points inside each inclusion with periodic images
        for cx, cy, cz in zip(centers_x, centers_y, centers_z):
            # Compute minimum distance considering periodic images
            dx = coords[0] - cx
            dy = coords[1] - cy
            dz = coords[2] - cz

            # Apply minimum image convention
            dx = dx - Lx * np.round(dx / Lx)
            dy = dy - Ly * np.round(dy / Ly)
            dz = dz - Lz * np.round(dz / Lz)

            distance_sq = dx ** 2 + dy ** 2 + dz ** 2
            inclusions[distance_sq <= radius ** 2] = 1

        return inclusions


    geometry = generate_circular_inclusions(
        discretization.fft.coords,
        num_inclusions=10,
        radius=0.2,
        seed=42
    )

    results_name = (f'bubbles_' + f'dof={nb_pixels_power}')
    to_save = np.copy(geometry)
    np.save(data_folder_path + results_name + f'.npy', to_save)


    def visualize_inclusions_voxels(inclusions, color='blue', edgecolor='k', alpha=0.9, figsize=(8, 8)):
        """
        Visualize 3D inclusion geometry using voxels.

        Parameters
        ----------
        inclusions : ndarray, shape [Nx, Ny, Nz]
            Boolean array of inclusions
        color : str
            Color of the inclusions
        edgecolor : str
            Edge color of voxels ('k' for black, None for no edges)
        alpha : float
            Transparency (0-1)
        figsize : tuple
            Figure size
        """
        # Create boolean array
        voxelarray = inclusions.astype(bool)

        # Set colors
        colors = np.empty(voxelarray.shape, dtype=object)
        colors[voxelarray] = color

        # Plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')
        ax.voxels(voxelarray, facecolors=colors, edgecolor=edgecolor, alpha=alpha)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Inclusions')

        plt.tight_layout()
        plt.show()

        # Print volume fraction
        vf = inclusions.sum() / inclusions.size
        print(f"Volume fraction: {vf:.2%}")


    # visualize_inclusions_voxels(geometry)

    # Or with custom options
    #visualize_inclusions_voxels(geometry, color='red', edgecolor=None, alpha=0.7)