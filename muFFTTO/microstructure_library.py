import warnings

import numpy as np


def get_geometry(nb_voxels,
                 microstructure_name='random_distribution'):
    if not microstructure_name in ['random_distribution', 'geometry_1_3D']:
        raise ValueError('Unrecognised microstructure_name {}'.format(microstructure_name))

    match microstructure_name:
        case 'random_distribution':

            phase_field = np.random.rand(*nb_voxels)

        case 'geometry_1_3D':  # TODO : this is an template for Bharat
            if nb_voxels.size != 3:
                raise ValueError('Microstructure_name {} is implemented only in 3D'.format(microstructure_name))
            # here should come your code
            phase_field = np.random.rand(*nb_voxels)

            # here should come your code

    return phase_field
