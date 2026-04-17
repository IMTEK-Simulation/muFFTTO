import warnings
import numpy as np
import matplotlib.pyplot as plt

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import



def get_geometry(nb_voxels,
                 microstructure_name='random_distribution',
                 coordinates=None,
                 parameter=None,
                 contrast=None,
                 **kwargs):
    if not microstructure_name in ['random_distribution', 'square_inclusion', 'circle_inclusion', 'circle_inclusions',
                                   'sine_wave', 'sine_wave_', 'linear', 'bilinear', 'tanh', 'sine_wave_inv', 'abs_val',
                                   'right_cluster_x3', 'left_cluster_x3', 'uniform_x1', 'n_laminate', 'circles',
                                   '2_circles',
                                   'symmetric_linear', 'hashin_inclusion_2D',
                                   'square_inclusion_equal_volfrac', 'sine_wave_rapid', 'n_squares',
                                   'laminate', 'laminate2', 'laminate_log',
                                   'geometry_I_1_3D', 'geometry_I_2_3D', 'geometry_I_3_3D', 'geometry_I_4_3D',
                                   'geometry_I_5_3D',
                                   'geometry_II_0_3D', 'geometry_II_1_3D', 'geometry_II_3_3D', 'geometry_II_4_3D',
                                   'geometry_III_1_3D', 'geometry_III_2_3D', 'geometry_III_3_3D', 'geometry_III_4_3D',
                                   'geometry_III_5_3D'
                                   ]:
        raise ValueError('Unrecognised microstructure_name {}'.format(microstructure_name))
    # if not nb_voxels[0] > 19 and nb_voxels[1] > 19 and nb_voxels[2] > 19 and nb_voxels[0]//5!=0 and nb_voxels[1]//5!=0 and nb_voxels[2]//5!=0:
    #     raise ValueError('Microstructure_name {} is implemented only when Size of any dimension is more than 10 and it is multiple of 5'.format(microstructure_name))

    # TODO [Bharat] put this condition into proper positions!
    # if nb_voxels[0] != nb_voxels[1] != nb_voxels[2]:
    #     raise ValueError(
    #         'Microstructure_name {} is implemented only in Nx=Ny=Nz grids'.format(microstructure_name))
    # if not (nb_voxels[0] > 19 and nb_voxels[1] > 19 and nb_voxels[2] > 19):
    #     # and nb_voxels[0] % 5 == 0 and nb_voxels[1] % 5 == 0 and nb_voxels[2] % 5 == 0
    #     raise ValueError('Microstructure_name {} is implemented only when Size '
    #                      'of any dimension is more than 19 and it is a multiple of 5'.format(
    #         microstructure_name))

    match microstructure_name:
        case 'random_distribution':
            seed = kwargs['seed']
            np.random.seed(seed)
            phase_field = np.random.rand(*nb_voxels)

        case 'square_inclusion':

            phase_field = np.ones(nb_voxels)
            if len(nb_voxels) == 2:
                phase_field[np.logical_and(np.logical_and(coordinates[0] < 0.75, coordinates[1] < 0.75),
                                           np.logical_and(coordinates[0] >= 0.25, coordinates[1] >= 0.25))] = 0
            elif len(nb_voxels) ==3:
                phase_field[np.logical_and(np.logical_and(coordinates[0] >= 0.25, coordinates[0] < 0.75),
                                       np.logical_and(np.logical_and(coordinates[1] < 0.75, coordinates[2] < 0.75),
                                                      np.logical_and(coordinates[1] >= 0.25,
                                                                     coordinates[2] >= 0.25)))] = 0

        case 'hashin_inclusion_2D':
            r1 = kwargs['rad_1']
            r2 = kwargs['rad_2']

            phase_field = np.ones(nb_voxels)
            phase_field[np.logical_and(np.logical_and(coordinates[0] < 0.75, coordinates[1] < 0.75),
                                       np.logical_and(coordinates[0] >= 0.25, coordinates[1] >= 0.25))] = 0

        case 'n_squares':

            phase_field = np.ones(nb_voxels)
            phase_field[np.logical_and(np.logical_and(coordinates[0] < 0.75, coordinates[1] < 0.75),
                                       np.logical_and(coordinates[0] >= 0.25, coordinates[1] >= 0.25))] = 0
        case 'circles':
            phase_field = np.ones(nb_voxels)
            x_lim = coordinates[0][-1, -1]
            y_lim = coordinates[1][-1, -1]
            # Define circle parameters (center coordinates and radius)
            circles = [
                (x_lim / 4, y_lim / 4, y_lim / 8),  # Circle 1
                (x_lim / 4, 3 * y_lim / 4, y_lim / 8),  # Circle 2
                (3 * x_lim / 4, 3 * y_lim / 4, y_lim / 8),  # Circle 3
                (3 * x_lim / 4, y_lim / 4, y_lim / 8)  # Circle 4
            ]
            # Apply circle masks
            for cx, cy, r in circles:
                mask = (coordinates[0] - cx) ** 2 + (coordinates[1] - cy) ** 2 <= r ** 2
                phase_field[mask] = 0  # Set pixels inside the circle to 1
        case '2_circles':
            phase_field = np.ones(nb_voxels)
            x_lim = coordinates[0][-1, -1]
            y_lim = coordinates[1][-1, -1]
            # Define circle parameters (center coordinates and radius)
            circles = [
                (x_lim / 6, 3 * y_lim / 4, y_lim / 10),  # Circle 1
                (3 * x_lim / 6, 4 * y_lim / 6, y_lim / 10),  # Circle 2
                (5 * x_lim / 6, 3 * y_lim / 4, y_lim / 10),
            ]
            # Apply circle masks
            for cx, cy, r in circles:
                mask = (coordinates[0] - cx) ** 2 + (coordinates[1] - cy) ** 2 <= r ** 2
                phase_field[mask] = 0  # Set pixels inside the circle to 1

        case 'laminate':

            phase_field = np.ones(nb_voxels)
            phase_field[coordinates[0] < 0.5] = 0
        case 'laminate2':

            phase_field = np.zeros(nb_voxels)
            # division=1/parameter
            # divisions=np.arange(0, 1, 1 / parameter)
            # divisionss= np.linspace(0, 1, parameter, endpoint = False)

            # divisions2 = np.arange(0, 1, 1 /( parameter-1))
            phases = np.linspace(contrast, 1, parameter)

            # positions = np.arange(0, 1+1 / parameter, 1 / parameter)
            positions = np.linspace(0, 1, parameter + 1)
            for i in np.arange(phases.size - 1):
                # section=divisions[i]
                # phase_field[coordinates[0] >= section] = divisions2[i]
                phase_field[coordinates[0] >= positions[i + 1]] = phases[i + 1]

            # phase_field[coordinates[0] >= divisions[-1]] = positions[-1]
            #print()
        case 'n_laminate':

            phase_field = np.zeros(nb_voxels)
            # division=1/parameter
            # divisions=np.arange(0, 1, 1 / parameter)
            # divisionss= np.linspace(0, 1, parameter, endpoint = False)

            # divisions2 = np.arange(0, 1, 1 /( parameter-1))
            phases = np.linspace(0, 1, parameter)

            # positions = np.arange(0, 1+1 / parameter, 1 / parameter)
            positions = np.linspace(0, 1, parameter + 1)
            for i in np.arange(phases.size - 1):
                # section=divisions[i]
                # phase_field[coordinates[0] >= section] = divisions2[i]
                phase_field[coordinates[0] >= positions[i + 1]] = phases[i + 1]

        case 'right_cluster_x3':

            phase_field = 1 - (1 - coordinates[0]) ** 3
        case 'left_cluster_x3':

            phase_field = (1 - coordinates[0]) ** 3

        case 'laminate_log':

            phase_field = np.zeros(nb_voxels) + np.power(10., contrast)
            # division=1/parameter
            # divisions=np.arange(0, 1, 1 / parameter)
            # divisionss= np.linspace(0, 1, parameter, endpoint = False)

            # divisions2 = np.arange(0, 1, 1 /( parameter-1))
            phases = np.logspace(contrast, 0, parameter)

            # positions = np.arange(0, 1+1 / parameter, 1 / parameter)
            positions = np.linspace(0, 1, parameter + 1)
            for i in np.arange(phases.size - 1):
                # section=divisions[i]
                # phase_field[coordinates[0] >= section] = divisions2[i]
                phase_field[coordinates[0] >= positions[i + 1]] = phases[i + 1]

            # phase_field[coordinates[0] >= divisions[-1]] = positions[-1]
            #print()
        case 'square_inclusion_equal_volfrac':

            phase_field = np.ones(nb_voxels)
            phase_field[np.logical_and(np.logical_and(coordinates[0] < 0.85, coordinates[1] < 0.85),
                                       np.logical_and(coordinates[0] >= 0.15, coordinates[1] >= 0.15))] = 0
        case 'circle_inclusion':
            phase_field = np.ones(nb_voxels)
            if nb_voxels.size == 1:
                phase_field[(np.sqrt(np.power(coordinates[0] - 0.5, 2))) < 0.2] = 0
            elif nb_voxels.size == 2 and nb_voxels[1] == 1:
                phase_field[(np.sqrt(np.power(coordinates[0] - 0.5, 2))) < 0.2] = 0
            elif nb_voxels.size == 2:
                phase_field[(np.sqrt(np.power(coordinates[0] - 0.5, 2) + np.power(coordinates[1] - 0.5, 2))) < 0.2] = 0
            elif nb_voxels.size == 3:
                phase_field[
                    np.power(coordinates[0] - 0.5, 2) +
                    np.power(coordinates[1] - 0.5, 2) +
                    np.power(coordinates[2] - 0.5, 2) < 0.1] = 0

        case 'circle_inclusions':
            nb_circles = kwargs['nb_circles']
            r_0 = kwargs['r_0']
            vol_frac = kwargs['vol_frac']
            r_n = r_0 / nb_circles
            random_density = kwargs['random_density']
            random_centers = kwargs['random_centers']

            inclusion_box_size = 1 / nb_circles
            perturb_of_centers = inclusion_box_size / 2 - r_n

            phase_field = np.zeros(nb_voxels)
            dim = np.size(nb_voxels)
            # number of circles in one direction
            if dim == 2:
                nb_circles_i = np.asarray(nb_circles, dtype=int)

                center = np.linspace(0, 1, nb_circles_i, endpoint=False)
                if nb_circles == 1:
                    center += 1 / 2
                else:
                    center += (center[1] - center[0]) / 2
                centers_x, centers_y = np.meshgrid(center, center)

                # Iterate over the results
                densities = np.random.permutation(np.arange(1, 1 + nb_circles ** dim))
                counter = 0
                for i in range(centers_x.shape[0]):
                    for j in range(centers_y.shape[1]):

                        center = np.array([centers_x[i, j], centers_y[i, j]])
                        if random_centers:
                            center += (perturb_of_centers * 0.99) * np.random.uniform(-1, 1)
                        r_center = np.zeros_like(coordinates)
                        for d in np.arange(dim):
                            r_center[d] = coordinates[d, ...] - center[d]

                        squares = 0
                        squares += sum(r_center[d] ** 2 for d in range(dim))
                        distances = np.sqrt(squares)
                        if random_density:
                            # Create an array from 1 to 10

                            # Shuffle the array randomly

                            # phase_field[distances < r_n] = np.random.random()
                            phase_field[distances < r_n] = densities[counter]
                        else:
                            phase_field[distances < r_n] = 1
                        counter += 1
            elif dim == 3:
                raise (NotImplementedError)

        case 'abs_val':
            phase_field = np.zeros(nb_voxels)
            if nb_voxels.size == 2:
                phase_field = np.abs(coordinates[0] - 0.5) + np.abs(coordinates[1] - 0.5)
            elif nb_voxels.size == 3:
                np.abs(coordinates[0] - 0.5) + np.abs(coordinates[1] - 0.5) + np.abs(coordinates[2] - 0.5)

        case 'sine_wave_rapid':
            phase_field = np.zeros(nb_voxels)
            if nb_voxels.size == 2:
                phase_field = 0.5 + 0.25 * np.cos(30 * 2 * np.pi * coordinates[0]) + 0.25 * np.cos(
                    10 * 2 * np.pi * coordinates[1])
            elif nb_voxels.size == 3:
                phase_field = np.sin(coordinates)
        case 'sine_wave':
            phase_field = np.zeros(nb_voxels)
            if nb_voxels.size == 2:
                phase_field = 0.5 + 0.25 * np.cos(3 * 2 * np.pi * coordinates[0]) + 0.25 * np.cos(
                    3 * 2 * np.pi * coordinates[1])
            elif nb_voxels.size == 3:
                phase_field = 0.5 + 0.25 * np.cos(3 * 2 * np.pi * coordinates[0]) + 0.25 * np.cos(
                    3 * 2 * np.pi * coordinates[1]) + 0.25 * np.cos(
                    3 * 2 * np.pi * coordinates[2])
        case 'sine_wave_':
            phase_field = np.zeros(nb_voxels)
            if nb_voxels.size == 2:
                phase_field = 0.5 + 0.25 * np.cos(
                    2 * np.pi * coordinates[0] - 2 * np.pi * coordinates[1]) + 0.25 * np.cos(
                    2 * np.pi * coordinates[1] + 2 * np.pi * coordinates[0])
            elif nb_voxels.size == 3:
                phase_field = (0.5 + 0.25 * np.cos(
                    2 * np.pi * coordinates[0] - 2 * np.pi * coordinates[1] - 2 * np.pi * coordinates[2]) +
                               0.25 * np.cos(
                            2 * np.pi * coordinates[1] + 2 * np.pi * coordinates[0] + 2 * np.pi * coordinates[2]))

        case 'sine_wave_inv':
            phase_field = np.zeros(nb_voxels)
            if nb_voxels.size == 2:
                phase_field = 0.5 - 0.25 * np.cos(
                    2 * np.pi * coordinates[0] - 2 * np.pi * coordinates[1]) - 0.25 * np.cos(
                    2 * np.pi * coordinates[1] + 2 * np.pi * coordinates[0])
            elif nb_voxels.size == 3:
                phase_field = (0.5 + 0.25 * np.cos(
                    2 * np.pi * coordinates[0] - 2 * np.pi * coordinates[1] - 2 * np.pi * coordinates[2]) +
                               0.25 * np.cos(
                            2 * np.pi * coordinates[1] + 2 * np.pi * coordinates[0] + 2 * np.pi * coordinates[2]))
        case 'tanh':
            phase_field = np.zeros(nb_voxels)
            if nb_voxels.size == 2:
                phase_field = np.tanh((coordinates[0] - 0.5) / 0.3 * (coordinates[1] - 0.5) / 0.3)
            elif nb_voxels.size == 3:
                phase_field = np.sin(coordinates)
        case 'linear':
            phase_field = np.zeros(nb_voxels)
            if nb_voxels.size == 2:
                phase_field = coordinates[0]
            elif nb_voxels.size == 3:
                phase_field = coordinates[0]
        case 'bilinear':
            phase_field = np.zeros(nb_voxels)
            if nb_voxels.size == 2:
                phase_field = coordinates[0] * coordinates[1]
            elif nb_voxels.size == 3:
                phase_field = coordinates[0] * coordinates[1] * coordinates[2]

        case 'symmetric_linear':
            phase_field = np.zeros(nb_voxels)
            if nb_voxels.size == 2:

                phase_field = coordinates[0]
                phase_field[nb_voxels[0] // 2:] = np.flipud(phase_field[:nb_voxels[0] // 2])


            elif nb_voxels.size == 3:
                raise "Not IMPLEMENTED"
        # --- Category I : Material in faces
        case 'geometry_I_1_3D':
            check_dimension(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            check_equal_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            check_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name, min_nb_voxels=19)
            #  Cube Frame
            phase_field = HSCC(*nb_voxels)

        case 'geometry_I_2_3D':
            #  Cube Frame with one diagonal in each face
            check_dimension(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            check_equal_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            check_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name, min_nb_voxels=19)

            phase_field = HFDC(*nb_voxels)

        case 'geometry_I_3_3D':
            # Cube Frame with two diagonals in each face
            check_dimension(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            check_equal_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name)

            phase_field = HFCC(*nb_voxels)

        case 'geometry_I_4_3D':
            # Just two diagonals in each face
            check_dimension(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            check_equal_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name)

            phase_field = HFCC_no_frame(*nb_voxels)

        case 'geometry_I_5_3D':
            #  Hollow Cube  with the Circle removed from each face
            check_dimension(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            check_equal_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name)

            phase_field = Circle_Frame(*nb_voxels)

        # --- Category II : in body geometries
        case 'geometry_II_0_3D':
            # Filled Cube
            check_dimension(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            # here should come your code
            phase_field = Normalcube(*nb_voxels)

        case 'geometry_II_1_3D':
            # Cube with the Body Diagonals
            check_dimension(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            check_equal_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name)

            phase_field = HBCC(*nb_voxels)

        case 'geometry_II_3_3D':
            # Cube with a another isocentric connected with the diagonals of Both Cubes Subtracted
            check_dimension(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            check_equal_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name)

            phase_field = Metamaterial_1(*nb_voxels)

        case 'geometry_II_4_3D':
            #  Filled Cube with a Sphere removed from it
            check_dimension(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            # here should come your code
            phase_field = SphereinCube(*nb_voxels)

        # --- Category III : Metamaterials
        case 'geometry_III_1_3D':
            #  lightweight strong metamaterial
            check_dimension(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            check_equal_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            check_unequal_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name,
                                           ratios=[1.8, 1.8, 1])

            phase_field = Metamaterial_3(*nb_voxels)

        case 'geometry_III_2_3D':
            #  ligtweight strong metamaterial
            check_dimension(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            check_equal_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            check_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name, min_nb_voxels=39)

            phase_field = Metamaterial_2(*nb_voxels)

        case 'geometry_III_3_3D':
            #  ligtweight strong metamaterial
            check_dimension(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            # check_equal_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            # check_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name, min_nb_voxels=20)

            phase_field = Metamaterial_4(*nb_voxels)

        case 'geometry_III_4_3D':
            # Define a  chiral metamaterial.
            check_dimension(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            # check_equal_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            # check_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name, min_nb_voxels=20)
            if parameter is None:
                parameter = {'lengths': [1., 1., 1],
                             'radius': 0.4,
                             'thickness': 0.1,
                             'alpha': 0.2}
            elif "lengths" not in parameter:
                parameter['lengths'] = [1., 1., 1]
            elif "radius" not in parameter:
                parameter['radius'] = 0.4
            elif "thickness" not in parameter:
                parameter['thickness'] = 0.1
            elif "alpha" not in parameter:
                parameter['alpha'] = 0.2

            phase_field = chiral_metamaterial(nb_grid_pts=nb_voxels,
                                                                        lengths=parameter['lengths'],
                                                                        radius=parameter['radius'],
                                                                        thickness=parameter['thickness'],
                                                                        alpha=parameter['alpha'])

        case 'geometry_III_5_3D':
            # Define a (more complex) chiral metamaterial.
            check_dimension(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            # check_equal_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            # check_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name, min_nb_voxels=20)
            if parameter is None:
                parameter = {
                    'lengths': [1.1, 1.1, 1],
                    'radius_out': 0.3,
                    'radius_inn': 0.2,
                    'thickness': 0.1,
                    'alpha': 0.25}

            elif "lengths" not in parameter:
                parameter['lengths'] = [1.1, 1.1, 1]
            elif "radius_out" not in parameter:
                parameter['radius_out'] = 0.3
            elif "radius_inn" not in parameter:
                parameter['radius_inn'] = 0.2
            elif "thickness" not in parameter:
                parameter['thickness'] = 0.1
            elif "alpha" not in parameter:
                parameter['alpha'] = 0.25

            phase_field = chiral_metamaterial_2(nb_grid_pts=nb_voxels,
                                                                          lengths=parameter['lengths'],
                                                                          radius_out=parameter['radius_out'],
                                                                          radius_inn=parameter['radius_inn'],
                                                                          thickness=parameter['thickness'],
                                                                          alpha=parameter['alpha'])

    return phase_field  # size is


def Circle_A(Nx, r):
    if Nx % 2 != 0:
        geom = np.zeros((Nx, Nx))
        for i in range(Nx):
            for j in range(Nx):
                if (i - (Nx) // 2) ** 2 + (j - (Nx) // 2) ** 2 > r ** 2:
                    geom[i, j] = 1
        geom = geom[0:Nx, 0:Nx]
    else:
        geom = np.zeros((Nx, Nx))
        for i in range(Nx):
            for j in range(Nx):
                if (i - (Nx) // 2 + 0.5) ** 2 + (j - (Nx) // 2 + 0.5) ** 2 > r ** 2:
                    geom[i, j] = 1
        geom = geom[0:Nx, 0:Nx]
    return geom


def Circle_Frame(*nb_voxels):
    (Nx, Ny, Nz) = nb_voxels
    # Create cube
    Cube = np.zeros((Nx, Ny, Nz))
    k = int(0.05 * Nx)
    r = int(0.4 * Nx)

    # Create meshgrid
    I, J, K = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1), np.arange(1, Nz + 1))

    G = np.zeros_like(Cube)
    Frame = square_frame2D(Nx, k)
    Circle = Circle_A(Nx, r)
    Overall = np.logical_or(Circle, Frame)

    for i in range(k):
        G[i, :, :] = Overall
        G[-i - 1, :, :] = Overall
        G[:, i, :] = Overall
        G[:, -i - 1, :] = Overall
        G[:, :, i] = Overall
        G[:, :, -i - 1] = Overall

    # Restrict D to Nx x Nx x Nx
    G = G[0:Nx, 0:Nx, 0:Nx]
    return G


def Diagonal2D_A(Nx, k):
    geom = np.zeros((Nx, Nx))
    for i in range(Nx):
        for j in range(Nx):
            if i == j or (i - k <= j <= i + k):
                geom[i, j] = 1
    geom = geom[0:Nx, 0:Nx]
    return geom


def Diagonal2D_B(Nx, k):
    geom = np.zeros((Nx, Nx))
    for i in range(Nx):
        for j in range(Nx):
            if (Nx) - i == j or (Nx) - (i + k + 1) <= j <= (Nx) - (i - k + 1):
                geom[i, j] = 1
    geom = geom[0:Nx, 0:Nx];
    return geom


def Diagonal2D_FACE(Nx, k):
    A = Diagonal2D_A(Nx, k)
    B = Diagonal2D_B(Nx, k)
    C = np.logical_or(A, B).astype(int)
    return C


def HBCC(*nb_voxels):
    (Nx, Ny, Nz) = nb_voxels
    # Create cube
    Cube = np.zeros((Nx, Ny, Nz))
    k = int(0.05 * Nx)
    # Create meshgrid
    I, J, K = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1), np.arange(1, Nz + 1))

    H = np.zeros_like(Cube)
    Frame = square_frame2D(Nx, k)

    # TopandBottom=np.logical_or(Frame,Diagonal1)
    # Sides=np.logical_or(Frame,Diagonal2)
    l = k * np.sqrt(3)
    for i in range(k):
        H[i, :, :] = Frame
        H[-i - 1, :, :] = Frame
        H[:, i, :] = Frame
        H[:, -i - 1, :] = Frame
        H[:, :, i] = Frame
        H[:, :, -i - 1] = Frame

    for i in range(Nx):
        for j in range(Nx):
            for k in range(Nx):
                if i <= j + l and i >= j - l and k <= j + l and k >= j - l:
                    H[i, j, k] = 1
                if i <= Nx - j - 1 + l and i >= Nx - j - 1 - l and k <= j + l and k >= j - l:
                    H[i, j, k] = 1
                if i <= j + l and i >= j - l and k + 1 <= Nx - j + l and k + 1 >= Nx - j - l:
                    H[i, j, k] = 1
                if Nx - i - 1 <= j + l and Nx - i - 1 >= j - l and Nx - k - 1 <= j + l and Nx - k - 1 >= j - l:
                    H[i, j, k] = 1

    # Restrict D to Nx x Nx x Nx
    H = H[0:Nx, 0:Ny, 0:Nz]
    return H


def HFCC(*nb_voxels):
    (Nx, Ny, Nz) = nb_voxels
    # Create cube
    Cube = np.zeros((Nx, Ny, Nz))
    k = int(0.05 * Nx)

    # Create meshgrid
    I, J, K = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1), np.arange(1, Nz + 1))

    # Get Diagonal2D_FACE matrix
    Face = Diagonal2D_FACE(Nx, k)
    Frame = square_frame2D(Nx, k)
    Overall = np.logical_or(Face, Frame)
    # Assign values to D
    D = np.zeros_like(Cube)
    for i in range(k):
        D[i, :, :] = Overall
        D[-i - 1, :, :] = Overall
        D[:, i, :] = Overall
        D[:, -i - 1, :] = Overall
        D[:, :, i] = Overall
        D[:, :, -i - 1] = Overall

    # Restrict D to Nx x Nx x Nx
    D = D[0:Nx, 0:Ny, 0:Nz]
    return D


def HFCC_no_frame(*nb_voxels):
    (Nx, Ny, Nz) = nb_voxels
    # Create cube
    Cube = np.zeros((Nx, Ny, Nz))
    k = int(0.05 * Nx)

    # Create meshgrid
    I, J, K = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1), np.arange(1, Nz + 1))

    # Get Diagonal2D_FACE matrix
    Face = Diagonal2D_FACE(Nx, k)
    Overall = np.logical_or(Face, Face)
    # Assign values to D
    D = np.zeros_like(Cube)
    for i in range(k):
        D[i, :, :] = Overall
        D[-i - 1, :, :] = Overall
        D[:, i, :] = Overall
        D[:, -i - 1, :] = Overall
        D[:, :, i] = Overall
        D[:, :, -i - 1] = Overall

    # Restrict D to Nx x Nx x Nx
    D = D[0:Nx, 0:Ny, 0:Nz]
    return D


def HSCC(*nb_voxels):
    (Nx, Ny, Nz) = nb_voxels
    # Create cube
    Cube = np.zeros((Nx, Ny, Nz))
    k = int(0.05 * Nx)
    Frame = square_frame2D(Nx, k)
    E = np.zeros_like(Cube)
    for i in range(k):
        E[i, :, :] = Frame
        E[-i - 1, :, :] = Frame
        E[:, i, :] = Frame
        E[:, -i - 1, :] = Frame
        E[:, :, i] = Frame
        E[:, :, -i - 1] = Frame

    # Restrict D to Nx x Nx x Nx
    E = E[0:Nx, 0:Ny, 0:Nz]
    return E


def HFDC(*nb_voxels):
    (Nx, Ny, Nz) = nb_voxels
    # Create cube
    Cube = np.zeros((Nx, Ny, Nz))
    k = int(0.05 * Nx)

    # Create meshgrid
    I, J, K = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1), np.arange(1, Nz + 1))

    # Get Diagonal2D_FACE matrix
    Face = Diagonal2D_A(Nx, k)
    Frame = square_frame2D(Nx, k)
    Overall = np.logical_or(Face, Frame)
    # Assign values to D
    D = np.zeros_like(Cube)
    for i in range(k):
        D[i, :, :] = Overall
        D[-i - 1, :, :] = Overall
        D[:, i, :] = Overall
        D[:, -i - 1, :] = Overall
        D[:, :, i] = Overall
        D[:, :, -i - 1] = Overall

    # Restrict D to Nx x Nx x Nx
    D = D[0:Nx, 0:Ny, 0:Nz]
    return D


def Kite(Nx, k):
    # Nx=100
    t = Nx // 2
    # k=5
    geom = np.zeros((Nx + k, Nx + k))

    for k in range(k):
        for i in range(0, Nx):
            for j in range(0, Nx):
                if j == -i + t - 1:
                    geom[i, j] = 1
                    geom[i + k, j] = 1
                    geom[i - k, j] = 1

        for i in range(0, Nx):
            for j in range(0, Nx):
                if i == j + t:
                    geom[i, j] = 1
                    geom[i, j + k] = 1
                    geom[i, j - k] = 1

        for i in range(0, Nx):
            for j in range(0, Nx):
                if j == i + t:
                    geom[i, j] = 1
                    geom[i + k, j] = 1
                    geom[i - k, j] = 1

        for i in range(0, Nx):
            for j in range(0, Nx):
                if j == -i + Nx + t - 1:
                    geom[i, j] = 1
                    geom[i + k, j] = 1
                    geom[i - k, j] = 1

    geom = geom[0:Nx, 0:Nx]
    return geom


def Normalcube(*nb_voxels):
    (Nx, Ny, Nz) = nb_voxels
    # Create cube
    geom = np.ones((Nx, Ny, Nz))
    geom = geom[0:Nx, 0:Ny, 0:Nz]
    return geom


def Sphere(Nx, r):
    geom = np.zeros((Nx, Nx, Nx))
    for i in range(Nx):
        for j in range(Nx):
            for k in range(Nx):
                if Nx % 2 != 0:
                    if np.sqrt((i - Nx // 2) ** 2 + (j - Nx // 2) ** 2 + (k - Nx // 2) ** 2) < r:
                        geom[i, j, k] = 1
                else:
                    if np.sqrt((i - Nx // 2 + 0.5) ** 2 + (j - Nx // 2 + 0.5) ** 2 + (k - Nx // 2 + 0.5) ** 2) < r:
                        geom[i, j, k] = 1
    geom = geom[0:Nx, 0:Nx]
    return geom


def SphereinCube(*nb_voxels):
    (Nx, Ny, Nz) = nb_voxels
    # Create cube
    Cube = np.zeros((Nx, Ny, Nz))
    r = int(0.6 * Nx)

    # Create meshgrid
    I, J, K = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Nx + 1), np.arange(1, Nx + 1))

    N = np.zeros_like(Cube)
    N[:, :, :] = Normalcube(*nb_voxels) - Sphere(Nx, r)
    # Restrict D to Nx x Nx x Nx
    N = N[0:Nx, 0:Nx, 0:Nx]
    return N


def square_frame2D_flexible(Nx, k1, k2):
    geom1 = square_frame2D(Nx, k1)
    geom2 = square_frame2D(Nx, k2)
    geom = np.logical_xor(geom1, geom2).astype(int)
    geom = geom[0:Nx, 0:Nx]
    return geom


def square_frame2D(Nx, k):
    geom = np.zeros((Nx, Nx))
    for i in range(Nx):
        for j in range(Nx):
            if i < k or Nx - i <= k or j < k or Nx - j <= k:
                geom[i, j] = 1
    geom = geom[0:Nx, 0:Nx]
    return geom


def Metamaterial_1(*nb_voxels):
    (Nx, Ny, Nz) = nb_voxels
    # Create cube
    Cube = np.zeros((Nx, Ny, Nz))
    k = int(0.05 * Nx)
    k2 = int(0.35 * Nx)
    k1 = int(0.30 * Nx)
    l = k * np.sqrt(3)
    # Create cube
    Cube = np.zeros((Nx, Ny, Nz), dtype=int)

    # Create meshgrid
    I, J, K = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1), np.arange(1, Nz + 1))
    Frame1 = square_frame2D_flexible(Nx, k1, k2)
    Frame2 = square_frame2D(Nx, k)

    # Assign values to D
    M = np.zeros_like(Cube)
    for i in range(k):
        M[k1 + i, :, :] = Frame1
        M[-k1 - i - 1, :, :] = Frame1
        M[:, k1 + i, :] = Frame1
        M[:, -k1 - i - 1, :] = Frame1
        M[:, :, k1 + i] = Frame1
        M[:, :, -k1 - i - 1] = Frame1

    for i in range(k):
        M[i, :, :] = Frame2
        M[-i - 1, :, :] = Frame2
        M[:, i, :] = Frame2
        M[:, -i - 1, :] = Frame2
        M[:, :, i] = Frame2
        M[:, :, -i - 1] = Frame2

    for i in range(Nx):
        for j in range(Nx):
            for k in range(Nx):
                if i <= j + l and i >= j - l and k <= j + l and k >= j - l:
                    M[i, j, k] = 1
                if i <= Nx - j - 1 + l and i >= Nx - j - 1 - l and k <= j + l and k >= j - l:
                    M[i, j, k] = 1
                if i <= j + l and i >= j - l and k + 1 <= Nx - j + l and k + 1 >= Nx - j - l:
                    M[i, j, k] = 1
                if Nx - i - 1 <= j + l and Nx - i - 1 >= j - l and Nx - k - 1 <= j + l and Nx - k - 1 >= j - l:
                    M[i, j, k] = 1

    for i in range(k2, Nx - k2):
        for j in range(k2, Nx - k2):
            for k in range(k2, Nx - k2):
                M[i, j, k] = 0

    # Restrict D to Nx x Nx x Nx
    M = M[0:Nx, 0:Nx, 0:Nx]
    return M


def Metamaterial_2(*nb_voxels):
    (Nx, Ny, Nz) = nb_voxels
    # Create cube
    Cube = np.zeros((Nx, Ny, Nz))
    k = int(0.05 * Nx)
    # Create meshgrid
    I, J, K = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Nx + 1), np.arange(1, Nx + 1))
    h = int(Nx / 2)
    l = int((k / 2) + 1)
    # Get Diagonal2D_FACE matrix
    Face = Diagonal2D_FACE(Nx, k)
    kite = Kite(Nx, l)

    # Assign values to D
    D = np.zeros_like(Cube)
    for i in range(k):
        D[i, :, :] = Face
        D[-i - 1, :, :] = Face
        D[:, i, :] = Face
        D[:, -i - 1, :] = Face
        D[:, :, i] = Face
        D[:, :, -i - 1] = Face

    for i in range(l):
        D[h + i, :, :] = kite
        D[h - i, :, :] = kite
        D[:, h + i, :] = kite
        D[:, h - i, :] = kite
        D[:, :, h + i] = kite
        D[:, :, h - i] = kite

    # Restrict D to Nx x Nx x Nx
    D = D[0:Nx, 0:Nx, 0:Nx]
    return D


def specialface(Nx):
    k = int(0.05 * Nx)
    t = int(Nx // 2)
    geom = np.zeros((Nx + k, Nx + k))
    for i in range(Nx):
        for j in range(Nx):
            if i < 2 * k or Nx - i <= 2 * k or t - k - 1 < i < t + k:
                geom[i, j] = 1

            if j == -i + t - 1:
                geom[i, j] = 1
                geom[i + k, j] = 1
                geom[i - k, j] = 1
            if i == j + t:
                geom[i, j] = 1
                geom[i, j + k] = 1
                geom[i, j - k] = 1
            if j == i + t:
                geom[i, j] = 1
                geom[i + k, j] = 1
                geom[i - k, j] = 1
            if j == -i + Nx + t - 1:
                geom[i, j] = 1
                geom[i + k, j] = 1
                geom[i - k, j] = 1

    geom[0:2 * k, int(0.45 * Nx):int(0.55 * Nx)] = 0
    geom[Nx - 2 * k - 1:Nx, int(0.45 * Nx):int(0.55 * Nx)] = 0

    geom = geom[0:Nx, 0:Nx]
    return geom


# %%
def specialface1(Nx):
    k = int(0.05 * Nx)
    t = int(Nx // 2)
    geom = np.zeros((Nx, Nx))
    for i in range(Nx):
        for j in range(Nx):
            if i == j or (i - k <= j <= i + k):
                geom[i, j] = 1
    for i in range(Nx):
        for j in range(Nx):
            if (Nx) - i == j or (Nx) - (i + k + 1) <= j <= (Nx) - (i - k + 1):
                geom[i, j] = 1

    for i in range(k):
        geom[t - k:t + k, :] = 1

    geom = geom[0:Nx, 0:Nx]
    return geom


def height(Nx, h, k):
    t = Nx // 2
    geom = np.zeros((Nx, h))
    geom[t - k:t + k, :] = 1
    geom = geom[0:Nx, 0:h]
    return geom


def height1(Nx, h, k):
    geom = np.zeros((Nx, h))
    geom[0:2 * k, :] = 1
    geom[Nx - 2 * k:Nx, :] = 1
    geom = geom[0:Nx, 0:h]
    return geom


def height2(Nx, h, k):
    t = Nx // 2
    geom = np.zeros((Nx, h))
    geom[t - k:t + k, :] = 1
    geom = geom[0:Nx, 0:h]
    return geom


def Metamaterial_3(*nb_voxels):
    (Nx, Ny, Nz) = nb_voxels
    # Create cube
    Cube = np.zeros((Nx, Nx, Nx))
    Nz = int(Nx + 0.80 * Nx)
    k = int(0.05 * Nx)
    t = Nx // 2
    h = (Nz - Nx) // 2
    h1 = h // 2
    # Create mesh grid
    I, J, K = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Nx + 1), np.arange(1, Nz + 1))

    # Get Diagonal2D_FACE matrix
    Face = specialface(Nx)
    Lonee = specialface1(Nx)
    Height = height(Nx, h, k)
    Height1 = height1(Nx, h1, k)
    Height2 = height2(Nx, h1, k)

    D = np.zeros_like(Cube)
    # Assign values to D
    D[t, :, :] = Lonee
    D[:, t, :] = Lonee

    for i in range(k):
        D[t - i, :, :] = Lonee
        D[t + i, :, :] = Lonee
        D[:, t - i, :] = Lonee
        D[:, t + i, :] = Lonee

    for i in range(2 * k):
        D[i, :, :] = Face
        D[-i - 1, :, :] = Face
        D[:, i, :] = Face
        D[:, -i - 1, :] = Face
        # D[:, :, i] = Face
        # D[:, :, -i-1] = Face

    # Restrict D to Nx x Nx x Nx
    D = D[0:Nx, 0:Nx, 0:Nx]

    Cuboid = np.zeros((Nx, Nx, Nz))
    E = np.zeros_like(Cuboid)
    E[:, :, h1:t + h1] = D[:, :, 0:t]
    E[:, :, -t - 1 - h1:-1 - h1] = D[:, :, -t - 1:-1]

    for i in range(2 * k):
        E[i, :, h1 + t:h1 + t + h] = Height
        E[-i - 1, :, h1 + t:h1 + t + h] = Height
        E[:, i, h1 + t:h1 + t + h] = Height
        E[:, -i - 1, h1 + t:h1 + t + h] = Height
        E[i, :, 0:h1] = Height1
        E[-i - 1, :, 0:h1] = Height1
        E[:, i, 0:h1] = Height1
        E[:, -i - 1, 0:h1] = Height1
        E[i, :, -h1 - 1:-1] = Height1
        E[-i - 1, :, -h1 - 1:-1] = Height1
        E[:, i, -h1 - 1:-1] = Height1
        E[:, -i - 1, -h1 - 1:-1] = Height1

    for i in range(k):
        E[t - i, :, 0:h1] = Height2
        E[t + i, :, 0:h1] = Height2
        E[:, t - i, 0:h1] = Height2
        E[:, t + i, 0:h1] = Height2
        E[:, t - i, -(h1 + 1):-1] = Height2
        E[:, t + i, -(h1 + 1):-1] = Height2
        E[t - i, :, -(h1 + 1):-1] = Height2
        E[t + i, :, -(h1 + 1):-1] = Height2

    E = E[0:Nx, 0:Nx, 0:Nz]

    return E


def Structure2D(Nx, k):
    geom = np.zeros((Nx, Nx))
    N = int(0.6 * Nx)
    geom1 = Diagonal2D_FACE(N, k)
    geom[int(0.1 * Nx):int(0.4 * Nx), int(0.2 * Nx):int(0.8 * Nx)] = geom1[0:int(N / 2), 0:N]
    geom[int(0.6 * Nx):int(0.9 * Nx), int(0.2 * Nx):int(0.8 * Nx)] = geom1[int(N / 2):N, 0:N]
    for i in range(k):
        geom[int(0.1 * Nx):int(0.9 * Nx), int(0.2 * Nx) + i] = 1
        geom[int(0.1 * Nx):int(0.9 * Nx), int(0.8 * Nx) - i] = 1
        geom[int(0.1 * Nx):int(0.9 * Nx), int(0.2 * Nx) - i] = 1
        geom[int(0.1 * Nx):int(0.9 * Nx), int(0.8 * Nx) + i] = 1
        geom[0:int(0.4 * Nx), int(0.5 * Nx) + i] = 1
        geom[0:int(0.4 * Nx), int(0.5 * Nx) - i] = 1
        geom[int(0.6 * Nx):Nx, int(0.5 * Nx) + i] = 1
        geom[int(0.6 * Nx):Nx, int(0.5 * Nx) - i] = 1

    geom = geom[0:Nx, 0:Nx]
    return geom


def Structure2D_FACE(NX, k):
    Nx = int(3 * NX / 2)

    geom = np.zeros((Nx, Nx))
    unit = np.transpose(Structure2D(int(Nx / 3), k))
    unit = unit[k + 1:-k, :]
    geom[0 + 1:int(Nx / 3) - 2 * k, 0:int(Nx / 3)] = unit
    geom[int((Nx / 3) - 2 * k - 2 * k) + 1:int((2 * Nx / 3) - 4 * k - 2 * k), 0:int(Nx / 3)] = unit
    geom[int(2 * (Nx / 3 - 2 * k) - 4 * k) + 1:int(3 * (Nx / 3 - 2 * k) - 4 * k), 0:int(Nx / 3)] = unit
    geom[0 + 1:int(Nx / 3) - 2 * k, int(Nx / 3):int(2 * Nx / 3)] = unit
    geom[int((Nx / 3) - 2 * k - 2 * k) + 1:int((2 * Nx / 3) - 4 * k - 2 * k), int(Nx / 3):int(2 * Nx / 3)] = unit
    geom[int(2 * (Nx / 3 - 2 * k) - 4 * k) + 1:int(3 * (Nx / 3 - 2 * k) - 4 * k), int(Nx / 3):int(2 * Nx / 3)] = unit

    geom = geom[0:NX, 0:NX]
    return geom


def Metamaterial_4(*nb_voxels):
    (Nx, Ny, Nz) = nb_voxels
    # Create cube
    Cube = np.zeros((Nx, Ny, Nx))
    k = int(Nx / 20)
    # Create meshgrid
    I, J, K = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Nx + 1), np.arange(1, Nx + 1))

    # Get Diagonal2D_FACE matrix
    Face = Structure2D_FACE(Nx, k)
    # Assign values to D
    D = np.zeros_like(Cube)
    for i in range(Ny):
        D[:, i, :] = Face
    # Restrict D to Nx x Nx x Nx
    D = D[0:Nx, 0:Ny, 0:Nx]

    return D


def visualize_voxels(phase_field_xyz, figure=None, ax=None):
    # -----
    # phase_field_xyz  - indicator field in every voxel
    # plot voxelized geometry of the phase_field
    #
    # -----
    # phase_field_bool = phase_field_xyz.round(decimals=0).astype(int).astype(bool)
    phase_field_bool = np.empty(phase_field_xyz.shape, dtype=bool)
    phase_field_bool[np.abs(phase_field_xyz) / abs(phase_field_xyz).max() < 0.1] = False
    phase_field_bool[np.abs(phase_field_xyz) / abs(phase_field_xyz).max() >= 0.1] = True
    # test_bool = test.astype(int).astype(bool)

    # te=phase_field_xyz[abs(phase_field_xyz) >= 0.1] = 1
    negative_values = phase_field_xyz < 0
    positive_values = phase_field_xyz > 0

    # set the colors of each object
    face_colors = np.zeros(list(phase_field_xyz.shape) + [4], dtype=np.float32)
    alpha = 0.0
    # set possitive to blues
    face_colors[positive_values] = [0, 0, 1, alpha]
    face_colors[negative_values] = [1, 0, 0, alpha]
    # edge_colors = face_colors

    # set transparency --- opacity --- alpha     scale 0-1
    face_colors[..., -1] = abs(phase_field_xyz) / abs(phase_field_xyz).max()
    if figure is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(projection='3d')
    ax.voxels(phase_field_bool, facecolors=face_colors, edgecolor='k', linewidth=0.01)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.voxels(np.full(phase_field_bool.shape, True), facecolors=face_colors, edgecolor='k', linewidth=0.01)
    return fig, ax


def check_equal_number_of_voxels(nb_voxels, microstructure_name):
    if nb_voxels[0] != nb_voxels[1] != nb_voxels[2]:
        raise ValueError(
            'Microstructure_name {} is implemented only in Nx=Ny=Nz grids'.format(microstructure_name))


def check_unequal_number_of_voxels(nb_voxels, microstructure_name, ratios):
    if (np.allclose(nb_voxels[0] * ratios[0], nb_voxels[1] * ratios[1])) and not (
            np.allclose(nb_voxels[0] * ratios[0], nb_voxels[2] * ratios[2])):
        raise ValueError(
            'Microstructure_name {} is implemented only in {} Nx = {} Ny = {} Nz grids'.format(microstructure_name,
                                                                                               ratios[0],
                                                                                               ratios[1], ratios[2]))


def check_number_of_voxels(nb_voxels, microstructure_name, min_nb_voxels):
    if not (nb_voxels[0] > min_nb_voxels and nb_voxels[1] > min_nb_voxels and nb_voxels[2] > min_nb_voxels):
        # and nb_voxels[0] % 5 == 0 and nb_voxels[1] % 5 == 0 and nb_voxels[2] % 5 == 0
        raise ValueError('Microstructure_name {} is implemented only when Size '
                         'of any dimension is more than {} and it is a multiple of 5'.format(
            microstructure_name, min_nb_voxels))


def check_dimension(nb_voxels, microstructure_name):
    if nb_voxels.size != 3:
        raise ValueError('Microstructure_name {} is implemented only in 3D'.format(microstructure_name))


### ----- Chiral metamaterials (contributed by Indre Joedicke) ----- ###

def chiral_metamaterial(nb_grid_pts, lengths, radius, thickness, alpha=0):
    """
    Define a (relatively simple) chiral metamaterial. It consists of two
    rings connected by four beams. Each beam is inclined by an angle alpha.

    Arguments
    ---------
    nb_grid_pts: list of 3 ints
                 Number of grid pts in each direction.
    lengths: list of 3 floats
             Lengths of unit cell in each direction.
    radius: float
            Radius of the cirlces
    thickness: float
               Thickness of the cirles and beams.
    alpha: float
           Angle at wich the connecting beams are inclined.
           Default is 0, meaning the beams are vertical.
    Returns
    -------
    mask: np.ndarray of floats
          Representation of the geometry with 0 corresponding
          to void and 1 corresponding to material.
    """
    # Parameters
    hx = lengths[0] / nb_grid_pts[0]
    hy = lengths[1] / nb_grid_pts[1]
    hz = lengths[2] / nb_grid_pts[2]
    x_axis = nb_grid_pts[0] // 2 * hx # + 0.5 * hx
    y_axis = nb_grid_pts[1] // 2 * hy # + 0.5 * hy
    thickness_x = round(thickness / hx)
    thickness_y = round(thickness / hy)
    thickness_z = round(thickness / hz)

    # Check wether the parameters are meaningful
    if (radius > lengths[0]/2 - hx) or (radius > lengths[1]/2 - hy):
        message = 'Attention: The diameter of the cylinder is larger '
        message += 'then the unit cell. THE PERIODIC BOUNDARIES ARE '
        message += 'NOT BROKEN.'
        print(message)
    if (radius < thickness + hx) or (radius < thickness + hy):
        message = 'Error: The radius is too small.'
        assert radius < thickness + hx, message
        assert radius < thickness + hy, message
    if (hx > thickness) or (hy > thickness) or (hy > thickness):
        message = 'Error: The pixels are larger than the thickness.'
        message += ' Please refine the discretization.'
        assert hx > thickness, message
        assert hy > thickness, message
        assert hz > thickness, message
    if (3*hx > thickness) or (3*hy > thickness):
        message = 'Attention: The thickness is represented by less then 3 pixels.'
        message += ' Please consider refining the discretization.'
        print(message)
    helper = np.arctan((radius - thickness / 2) / (lengths[2] - 2 * thickness))
    if (alpha > helper) or (alpha < - helper):
        message = f'Error: The angle must lie between {-helper} and {helper}'
        message += f'but it is {alpha}.'
        assert alpha > helper, message
        assert alpha < - helper, message

    # Circles at top and bottom
    mask = np.zeros(nb_grid_pts)
    for ind_x in range(nb_grid_pts[0]):
        for ind_y in range(nb_grid_pts[1]):
            x = ind_x * hx + 0.5 * hx
            y = ind_y * hy + 0.5 * hy
            dist = np.sqrt((x - x_axis)**2 + (y - y_axis)**2)
            if (dist < radius) and (dist >= radius - thickness):
                mask[ind_x, ind_y, 0:thickness_z] = 1
                mask[ind_x, ind_y, nb_grid_pts[2]-thickness_z:] = 1

    # Step in x- and y-direction of connecting beams
    step_1 = hz * np.tan(alpha)
    helper = (lengths[2] - 2 * thickness) * np.tan(alpha)
    helper = radius - thickness / 2 -\
        ((radius - thickness / 2)**2 - helper**2) ** 0.5
    step_2 = helper / (lengths[2] - 2 * thickness) * hz

    # Starting points for connecting beams
    start1_x = nb_grid_pts[0] // 2 - thickness_x // 2
    start2_x = round((lengths[0] / 2 + radius - thickness) / hx)
    start3_x = nb_grid_pts[0] // 2 - thickness_x // 2
    start4_x = round((lengths[0] / 2 - radius) / hx)
    start1_y = round((lengths[1] / 2 - radius) / hy)
    start2_y = nb_grid_pts[1] // 2 - thickness_y // 2
    start3_y = round((lengths[1] / 2 + radius - thickness) / hy)
    start4_y = nb_grid_pts[1] // 2 - thickness_y // 2

    # Connecting beams
    for ind_z in range(thickness_z, nb_grid_pts[2]-thickness_z):
        help_x1 = round((ind_z - thickness_z) * step_1 / hx)
        help_y1 = round((ind_z - thickness_z) * step_1 / hy)
        help_x2 = round((ind_z - thickness_z) * step_2 / hx)
        help_y2 = round((ind_z - thickness_z) * step_2 / hy)

        # 1. beam
        start_x = start1_x + help_x1
        start_y = start1_y + help_y2
        mask[start_x:start_x+thickness_x, start_y:start_y+thickness_y, ind_z] = 1
        # 2. beam
        start_x = start2_x - help_x2
        start_y = start2_y + help_y1
        mask[start_x:start_x+thickness_x, start_y:start_y+thickness_y, ind_z] = 1
        # 3. beam
        start_x = start3_x - help_x1
        start_y = start3_y - help_y2
        mask[start_x:start_x+thickness_x, start_y:start_y+thickness_y, ind_z] = 1
        # 4. beam
        start_x = start4_x + help_x2
        start_y = start4_y - help_y1
        mask[start_x:start_x+thickness_x, start_y:start_y+thickness_y, ind_z] = 1

    return mask


def chiral_metamaterial_2(nb_grid_pts, lengths, radius_out, radius_inn,
                          thickness, alpha=0):
    """
    Define a (more complex) chiral metamaterial. It consists of a beam on each
    face of the RVE connected to the edges by four beams. The beams are
    inclined with an angle alpha.

    Arguments
    ---------
    nb_grid_pts: list of 3 ints
                 Number of grid pts in each direction.
    lengths: list of 3 floats
             Lengths of unit cell in each direction. Note that the size of
             the RVE corresponds to lengths[2], so that lengths[0] and
             lengths[1] must be larger than lengths[2] to break the periodicity.
    radius_out: float
                Outer radius of the circles.
    radius_inn: float
                Inner radius of the circles.
    thickness: float
               Thickness of the connecting beams.
    alpha: float
           Angle at wich the connecting beams are inclined.
           Default is 0.
    Returns
    -------
    mask: np.ndarray of floats
          Representation of the geometry with 0 corresponding
          to void and 1 corresponding to material.
    """
    ### ----- Parameters ----- ###
    hx = lengths[0] / nb_grid_pts[0]
    hy = lengths[1] / nb_grid_pts[1]
    hz = lengths[2] / nb_grid_pts[2]
    thickness_x = round(thickness / hx)
    thickness_y = round(thickness / hy)
    thickness_z = round(thickness / hz)
    a = lengths[2]
    b = 1.5 * thickness
    boundary = (lengths[0] - a) / 2

    # Check wether the parameters are meaningful
    if (radius_out > a/2 - hx) or (radius_out > a/2 - hy):
        message = 'ATTENTION: The diameter of the outer circle is larger '
        message += 'then the unit cell.'
        print(message)
    if (radius_inn < thickness + hx) or (radius_inn < thickness + hy):
        message = 'ERROR: The inner radius is too small.'
        assert radius_inn < thickness + hx, message
        assert radius_inn < thickness + hy, message
    if (hx > thickness) or (hy > thickness) or (hy > thickness):
        message = 'ERROR: The pixels are larger than the thickness.'
        message += ' Please refine the discretization.'
        assert hx > thickness, message
        assert hy > thickness, message
        assert hz > thickness, message
    if (3*hx > thickness) or (3*hy > thickness):
        message = 'ATTENTION: The thickness is represented by less then 3 pixels.'
        message += ' Please consider refining the discretization.'
        print(message)
    message = 'lengths[0] is not large enough to break the periodicity.'
    assert lengths[0] > a, message
    message = 'lengths[1] is not large enough to break the periodicity.'
    assert lengths[1] > a, message

    ### ----- Define the four corners ----- ###
    mask = np.zeros(nb_grid_pts)
    bx = round(b / hx)
    by = round(b / hy)
    bz = round(b / hz)
    boundary_x = round(boundary / hx)
    boundary_y = round(boundary / hy)
    mask[boundary_x:boundary_x+bx, boundary_y:boundary_y+by, 0:bz] = 1
    mask[-bx-boundary_x:-boundary_x, boundary_y:boundary_y+by, 0:bz] = 1
    mask[boundary_x:boundary_x+bx, -by-boundary_y:-boundary_y, 0:bz] = 1
    mask[-bx-boundary_x:-boundary_x, -by-boundary_y:-boundary_y, 0:bz] = 1
    mask[boundary_x:boundary_x+bx, boundary_y:boundary_y+by, -bz:] = 1
    mask[-bx-boundary_x:-boundary_x, boundary_y:boundary_y+by, -bz:] = 1
    mask[boundary_x:boundary_x+bx, -by-boundary_y:-boundary_y, -bz:] = 1
    mask[-bx-boundary_x:-boundary_x, -by-boundary_y:-boundary_y, -bz:] = 1

    ### ----- Define the circle at each face ----- ###
    x = np.arange(nb_grid_pts[0]) * hx
    y = np.arange(nb_grid_pts[1]) * hy
    z = np.arange(nb_grid_pts[2]) * hz
    X, Y, Z = np.meshgrid(x, y, z)
    X = X.transpose((1, 0, 2)) + 0.5 * hx
    Y = Y.transpose((1, 0, 2)) + 0.5 * hy
    Z = Z.transpose((1, 0, 2)) + 0.5 * hz

    # Circles in xz-planes
    x0 = lengths[0] / 2
    z0 = lengths[2] / 2
    dist = (X[:, 0, :] - x0) ** 2 + (Z[:, 0, :] - z0) ** 2
    dist = dist ** 0.5
    material = np.logical_and(dist < radius_out, dist > radius_inn)
    material = np.expand_dims(material, axis=1)
    material = np.broadcast_to(material, nb_grid_pts).copy()
    material[:, 0:boundary_y, :] = False
    material[:, -boundary_y:, :] = False
    material[:, boundary_y+thickness_y:-thickness_y-boundary_y, :] = False
    mask[material] = 1

    # Circles in yz-planes
    y0 = lengths[1] / 2
    z0 = lengths[2] / 2
    dist = (Y[0, :, :] - y0) ** 2 + (Z[0, :, :] - z0) ** 2
    dist = dist ** 0.5
    material = np.logical_and(dist < radius_out, dist > radius_inn)
    material = np.expand_dims(material, axis=0)
    material = np.broadcast_to(material, nb_grid_pts).copy()
    material[0:boundary_x, :, :] = False
    material[-boundary_x:, :, :] = False
    material[boundary_x+thickness_x:-thickness_x-boundary_x, :, :] = False
    mask[material] = 1

    # Circles in xy-planes
    x0 = lengths[0] / 2
    y0 = lengths[1] / 2
    dist = (X[:, :, 0] - x0) ** 2 + (Y[:, :, 0] - y0) ** 2
    dist = dist ** 0.5
    material = np.logical_and(dist < radius_out, dist > radius_inn)
    material = np.expand_dims(material, axis=2)
    material = np.broadcast_to(material, nb_grid_pts).copy()
    material[:, :, thickness_z//2:-thickness_z//2] = False
    mask[material] = 1

    ### ----- Define the connecting beams ----- ###
    beta = np.pi / 4 - alpha
    step_exact = hz * np.tan(beta)

    helper_a = 1 + np.tan(beta) ** 2
    helper_b = - 2 * (np.tan(beta) + 1) * (a/2 - b/2)
    helper_c = 2 * (a/2 - b/2) ** 2 - (radius_out/2 + radius_inn/2) ** 2
    helper = helper_b ** 2 - 4 * helper_a * helper_c
    message = 'ERROR: The angle of the material is too large.'
    assert helper > 0, message
    stop = (- helper_b - helper ** 0.5 ) / 2 / helper_a

    # Beams in xz-planes
    start_x = round((boundary + b/2) / hx)
    stop_x = round((boundary + stop + b/2) / hx)
    start_y = boundary_y
    stop_y = boundary_y + thickness_y
    start_z = round(b / 2 / hz)
    stop_z = round((b/2 + stop) / hz)
    t_half_x = round(thickness / 2 / hx)
    t_half_y = round(thickness / 2 / hy)
    t_half_z = round(thickness / 2 / hz)
    for ind_x in range(start_x, stop_x):
        step = round((ind_x - start_x) * step_exact / hz)
        mask[ind_x - t_half_x + 1 : ind_x + t_half_x + 1,
             start_y : stop_y,
             start_z + step - t_half_x : start_z + step + t_half_x] = 1
        mask[ind_x - t_half_x + 1 : ind_x + t_half_x + 1,
             -stop_y : -start_y,
             start_z + step - t_half_z : start_z + step + t_half_z] = 1
        helper = -start_z - step + t_half_z
        if helper > -1:
            mask[-ind_x - t_half_x - 1: -ind_x + t_half_x - 1,
                 start_y : stop_y,
                 -start_z - step - t_half_z : ] = 1
            mask[-ind_x - t_half_x - 1: -ind_x + t_half_x - 1,
                 -stop_y : -start_y,
                 -start_z - step - t_half_z : ] = 1
        else:
            mask[-ind_x -t_half_x - 1: -ind_x + t_half_x - 1,
                 start_y : stop_y,
                 -start_z - step - t_half_z : helper] = 1
            mask[-ind_x - t_half_x - 1: -ind_x + t_half_x - 1,
                 -stop_y : -start_y,
                 -start_z - step - t_half_z : helper] = 1
    for ind_z in range(start_z, stop_z):
        step = round((ind_z - start_z) * step_exact / hx)
        mask[-start_x - step - t_half_x : -start_x - step + t_half_x,
             start_y : stop_y,
             ind_z - t_half_z : ind_z + t_half_z] = 1
        mask[start_x + step - t_half_x : start_x + step + t_half_x,
             start_y : stop_y,
             -ind_z - t_half_z : -ind_z + t_half_z] = 1
        mask[-start_x - step - t_half_x : -start_x - step + t_half_x,
             -stop_y : -start_y,
             ind_z - t_half_z : ind_z + t_half_z] = 1
        mask[start_x + step - t_half_x : start_x + step + t_half_x,
             -stop_y : -start_y,
             -ind_z - t_half_z : -ind_z + t_half_z] = 1

    # Beams in yz-planes
    start_x = boundary_x
    stop_x = boundary_x + thickness_x
    start_y = round((boundary + b/2) / hy)
    stop_y = round((boundary + stop + b/2) / hy)
    for ind_y in range(start_y, stop_y):
        step = round((ind_y - start_y) * step_exact / hz)
        mask[start_x : stop_x,
             ind_y - t_half_y + 1 : ind_y + t_half_y + 1,
             start_z + step - t_half_z : start_z + step + t_half_z] = 1
        mask[-stop_x : -start_x,
             ind_y - t_half_y + 1 : ind_y + t_half_y + 1,
             start_z + step - t_half_z : start_z + step + t_half_z] = 1
        helper = -start_z - step + t_half_z
        if helper > -1:
            mask[start_x : stop_x,
                 -ind_y - t_half_y - 1: -ind_y + t_half_y - 1,
                 -start_z - step - t_half_z : ] = 1
            mask[-stop_x : -start_x,
                 -ind_y - t_half_y - 1: -ind_y + t_half_y - 1,
                 -start_z - step - t_half_z : ] = 1
        else:
            mask[start_x : stop_x,
                 -ind_y - t_half_y - 1: -ind_y + t_half_y - 1,
                 -start_z - step - t_half_z : helper] = 1
            mask[-stop_x : -start_x,
                 -ind_y - t_half_y - 1: -ind_y + t_half_y - 1,
                 -start_z - step - t_half_z : helper] = 1
    for ind_z in range(start_z, stop_z):
        step = round((ind_z - start_z) * step_exact / hy)
        mask[start_x : stop_x,
             -start_y - step - t_half_y : -start_y - step + t_half_y,
             ind_z - t_half_z : ind_z + t_half_z] = 1
        mask[start_x : stop_x,
             start_y + step - t_half_y : start_y + step + t_half_y,
             -ind_z - t_half_z : -ind_z + t_half_z] = 1
        mask[-stop_x : -start_x,
             -start_y - step - t_half_y : -start_y - step + t_half_y,
             ind_z - t_half_z : ind_z + t_half_z] = 1
        mask[-stop_x : -start_x,
             start_y + step - t_half_y : start_y + step + t_half_y,
             -ind_z - t_half_z : -ind_z + t_half_z] = 1

    # Beams in xy-planes
    start_x = round((boundary + b/2) / hx)
    stop_x = round((boundary + stop + b/2) / hx)
    start_y = round((boundary + b/2) / hy)
    stop_y = round((boundary + stop + b/2) / hy)
    stop_z = round(thickness / 2 / hz)
    for ind_x in range(start_x, stop_x):
        step = round((ind_x - start_x) * step_exact / hy)
        mask[ind_x - t_half_x + 1 : ind_x + t_half_x + 1,
             start_y + step - t_half_y : start_y + step + t_half_y,
             : stop_z] = 1
        mask[ind_x - t_half_x + 1 : ind_x + t_half_x + 1,
             start_y + step - t_half_y : start_y + step + t_half_y,
             -stop_z :] = 1
        helper = -start_y - step + t_half_y
        if helper > -1:
            mask[-ind_x - t_half_x - 1: -ind_x + t_half_x - 1,
                 -start_y - step - t_half_y : ,
                 : stop_z] = 1
            mask[-ind_x - t_half_x - 1: -ind_x + t_half_x - 1,
                 -start_y - step - t_half_y : ,
                 -stop_z :] = 1
        else:
            mask[-ind_x - t_half_x - 1: -ind_x + t_half_x - 1,
                 -start_y- step - t_half_y : helper,
                 : stop_z] = 1
            mask[-ind_x - t_half_x - 1: -ind_x + t_half_x - 1,
                 -start_y- step - t_half_y : helper,
                 -stop_z :] = 1
    for ind_y in range(start_y, stop_y):
        step = round((ind_y - start_y) * step_exact / hx)
        mask[-start_x - step - t_half_x : -start_x - step + t_half_x,
             ind_y - t_half_y : ind_y + t_half_y,
             : stop_z] = 1
        mask[start_x + step - t_half_x : start_x + step + t_half_x,
             -ind_y - t_half_y : -ind_y + t_half_y,
             : stop_z] = 1
        mask[-start_x - step - t_half_x : -start_x - step + t_half_x,
             ind_y - t_half_y : ind_y + t_half_y,
             -stop_z :] = 1
        mask[start_x + step - t_half_x : start_x + step + t_half_x,
             -ind_y - t_half_y : -ind_y + t_half_y,
             -stop_z :] = 1

    return mask


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    # Display the image
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

    # plot  geometry
    geometry_ID = 'geometry_III_5_3D'
    N = 60
    nb_of_pixels = np.asarray(3 * (N,), dtype=int)
    # nb_of_pixels = np.asarray( (N,N,1.8*N), dtype=int)

    phase_field = get_geometry(nb_voxels=nb_of_pixels,
                               microstructure_name=geometry_ID)

    fig, ax = visualize_voxels(phase_field_xyz=phase_field)
    ax.set_title(geometry_ID)
    save_plot = True
    if save_plot:
        src = '/home/martin/Programming/muFFTTO/experiments/figures/'  # source folder\
        fig_data_name = f'muFFTTO_{geometry_ID}_N{N}'

        fname = src + fig_data_name + '_geometry{}'.format('.pdf')
        print(('create figure: {}'.format(fname)))  # axes[1, 0].legend(loc='upper right')
        plt.savefig(fname, dpi=1000, pad_inches=0.02, bbox_inches='tight',
                    facecolor='auto', edgecolor='auto')
        print('END plot ')

    plt.show()
