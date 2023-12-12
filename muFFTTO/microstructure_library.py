import warnings


import numpy as np


def get_geometry(nb_voxels,
                 microstructure_name='random_distribution',
                 parameter=None):
    if not microstructure_name in ['random_distribution', 'geometry_1_3D']:
        raise ValueError('Unrecognised microstructure_name {}'.format(microstructure_name))

    match microstructure_name:
        case 'random_distribution':

            phase_field = np.random.rand(*nb_voxels)

        case 'geometry_1_3D':  # TODO[Bharat] : this is an template for you
            if nb_voxels.size != 3:
                raise ValueError('Microstructure_name {} is implemented only in 3D'.format(microstructure_name))
            # here should come your code
            phase_field = np.random.rand(*nb_voxels)

            # here should come your code
        case 'geometry_2_2D':
            if nb_voxels.size != 2:
                raise ValueError('Microstructure_name {} is implemented only in 3D'.format(microstructure_name))
            if not parameter in ['radius']:
                raise ValueError('I need radius')


    return phase_field  # size is
