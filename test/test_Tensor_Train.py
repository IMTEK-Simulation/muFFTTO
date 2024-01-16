import pytest

import numpy as np

from muFFTTO import TT_tools
from muFFTTO import microstructure_library


@pytest.fixture()
def tensor_fixture(nb_voxels, epsilon, microstructure):
    return nb_voxels, epsilon, microstructure


def test_tt_decomposition_rank():
    nb_voxels = [20, 20, 20]
    tensor = microstructure_library.get_geometry(nb_voxels=nb_voxels,
                                                 microstructure_name='random_distribution',
                                                 parameter=None)
    tensor_norm = np.linalg.norm(tensor)

    ranks = (1, 20, 20, 1)
    tt_tensor = TT_tools.tt_decompose_rank(tensor_xyz=tensor,
                                           ranks=ranks)

    tt_reconstructed_tensor = TT_tools.tt_to_full_format(tt_cores=tt_tensor)

    error = np.allclose(tensor, tt_reconstructed_tensor)
    assert error, (
        "TT reconstruction does not work {}".format(error))


@pytest.mark.parametrize('nb_voxels, epsilon , microstructure', [
    ([30, 30, 30], 0.1, 'geometry_I_1_3D'),
    ([20, 30, 35], 0.2, 'random_distribution'),
    ([22, 40, 45], 0.3, 'random_distribution'),
    ([32, 25, 25], 0.4, 'random_distribution'),
    ([30, 36, 35], 0.5, 'random_distribution'),
    ([30, 47, 45], 0.6, 'random_distribution'),
    ([4, 2, 25], 0.7, 'random_distribution'),
    ([40, 39, 35], 0.8, 'random_distribution'),
    ([40, 40, 45], 0.9, 'random_distribution'),
    ([20, 20, 25], 1.0, 'random_distribution'),
    ([17, 20], 1.0, 'random_distribution')])
def test_tt_decomposition_error(tensor_fixture):
    nb_voxels = np.array(tensor_fixture[0], dtype=int)
    epsilon = tensor_fixture[1]
    dim = len(nb_voxels)
    microstructure = tensor_fixture[2]
    A_tensor = microstructure_library.get_geometry(nb_voxels=nb_voxels,
                                                   microstructure_name=microstructure,
                                                   parameter=None)
    A_norm = np.linalg.norm(A_tensor)

    tt_tensor, ranks = TT_tools.tt_decompose_error(tensor_xyz=A_tensor,
                                                   rel_error_norm=epsilon)

    tt_reconstructed_tensor = TT_tools.tt_to_full_format(tt_cores=tt_tensor)

    delta = A_norm*epsilon #/ (np.sqrt(dim - 1))
# TODO [martin] TT decomposition does not  preserve the error
    error_norm_abs = np.linalg.norm(A_tensor - tt_reconstructed_tensor)
    error_norm_rel = (error_norm_abs / A_norm)

    assert error_norm_abs <= delta, (
        "TT reconstruction does not work properly. Abs. error norm = {} >> delta = {}".format(error_norm_abs,
                                                                                              delta))

  # *
    # print(error_norm_rel)
    # print(epsilon)


    assert error_norm_rel <= epsilon, (
        "TT reconstruction does not work properly. Relative error norm = {} >> delta = {}".format(error_norm_rel,
                                                                                                  delta))


def test_tt_summation():
    nb_voxels = [20, 20, 20]
    tensor = microstructure_library.get_geometry(nb_voxels=nb_voxels,
                                                 microstructure_name='random_distribution',
                                                 parameter=None)
    tensor_norm = np.linalg.norm(tensor)

    ranks = (1, 20, 20, 1)
    tt_tensor = TT_tools.tt_decompose_rank(tensor_xyz=tensor,
                                           ranks=ranks)

    tt_reconstructed_tensor = TT_tools.tt_to_full_format(tt_cores=tt_tensor)

    error = np.allclose(tensor, tt_reconstructed_tensor)
    assert error, (
        "TT summation does not work {}".format(error)   )