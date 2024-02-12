import pytest

import numpy as np
import muFFTTO

from muFFTTO import tensor_train_tools
from muFFTTO import microstructure_library
from muFFTTO import domain


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
    tt_tensor = tensor_train_tools.tt_decompose_rank(tensor_xyz=tensor,
                                                     ranks=ranks)

    tt_reconstructed_tensor = tensor_train_tools.tt_to_full_format(tt_cores=tt_tensor)

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

    tt_tensor, ranks = tensor_train_tools.tt_decompose_error(tensor_xyz=A_tensor,
                                                             rel_error_norm=epsilon)

    tt_reconstructed_tensor = tensor_train_tools.tt_to_full_format(tt_cores=tt_tensor)

    delta = A_norm * epsilon  # / (np.sqrt(dim - 1))
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
    tt_tensor = tensor_train_tools.tt_decompose_rank(tensor_xyz=tensor,
                                                     ranks=ranks)

    tt_reconstructed_tensor = tensor_train_tools.tt_to_full_format(tt_cores=tt_tensor)

    error = np.allclose(tensor, tt_reconstructed_tensor)
    assert error, ("TT summation does not work {}".format(error))


def test_tt_partial_derivative_2D():
    # compare  fem gradient with low rank gradient for linear elements
    number_of_pixels = [10, 10]
    domain_size = [3, 4]
    problem_type = 'conductivity'  # 'elasticity'#,'conductivity'
    my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                      problem_type=problem_type)

    discretization_type = 'finite_element'
    element_type = 'linear_triangles'

    discretization = domain.Discretization(cell=my_cell,
                                           number_of_pixels=number_of_pixels,
                                           discretization_type=discretization_type,
                                           element_type=element_type)

    nodal_coordinates = discretization.get_nodal_points_coordinates()
    quad_coordinates = discretization.get_quad_points_coordinates()

    u_fun_4x3y = lambda x, y: 4 * x * y + 3 * y * np.sin(x)
    #            u_fun_4x3y = lambda x, y: np.random.rand() * x + np.random.rand() * y  # np.sin(x)

    temperature = discretization.get_temperature_sized_field()
    temperature_gradient = discretization.get_temperature_gradient_size_field()

    temperature[0, 0, :, :] = u_fun_4x3y(nodal_coordinates[0, 0, :, :],
                                         nodal_coordinates[1, 0, :, :])

    # temperature[0, 0, :, :] = np.random.rand(*number_of_pixels)

    tensor_dx = discretization.apply_gradient_operator(temperature, temperature_gradient)[0, 0, 0]
    tensor_dy = discretization.apply_gradient_operator(temperature, temperature_gradient)[0, 1, 0]

    for rank_i in reversed(np.arange(1, max(number_of_pixels) + 1)):
        # print(rank_i)

        tensor = np.copy(temperature[0, 0, :, :])

        tt_field = tensor_train_tools.tt_decompose_rank(tensor_xyz=tensor,
                                                        ranks=[1, rank_i, 1])

        tt_reconstructed_field = tensor_train_tools.tt_to_full_format(tt_cores=tt_field)

        Nx_basis = tt_field[0]
        Ny_basis = tt_field[1]

        dNx_x = (np.roll(Nx_basis, -1, axis=1) - Nx_basis) / discretization.pixel_size[0]
        dNy_y = (np.roll(Ny_basis, -1, axis=1) - Ny_basis) / discretization.pixel_size[1]

        tt_dNx_x = [dNx_x, Ny_basis]
        tt_dNy_y = [Nx_basis, dNy_y]

        tt_dNx_field = tensor_train_tools.tt_to_full_format(tt_cores=tt_dNx_x)
        tt_dNy_field = tensor_train_tools.tt_to_full_format(tt_cores=tt_dNy_y)

        field_norm = np.linalg.norm(tensor)
        abs_error_norms = (np.linalg.norm(tensor - tt_reconstructed_field))
        rel_error_norms = (abs_error_norms / field_norm)
        #
        # print(abs_error_norms)
        # print(rel_error_norms)
        #
        # print(np.linalg.norm(tt_dNx_field - tensor_dx) / np.linalg.norm(tensor_dx))
        # print(np.linalg.norm(tt_dNy_field - tensor_dy) / np.linalg.norm(tensor_dy))

        error_dx = np.allclose(tt_dNx_field, tensor_dx)
        error_dy = np.allclose(tt_dNy_field, tensor_dy)
        assert error_dx, (
            "TT-full rank gradient dx is not equal to full format gradient  {}".format(error_dx))
        assert error_dy, (
            "TT-full rank gradient dy is not equal to full format gradient  {}".format(error_dy))


def test_tt_partial_derivative_3D():
    # test TT FD compared to finite element gradient
    # compare  fem gradient with low rank gradient for linear elements
    number_of_pixels = [13, 13, 13]
    domain_size = [3, 4, 5]
    problem_type = 'conductivity'  # 'elasticity'#,'conductivity'
    my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                      problem_type=problem_type)

    discretization_type = 'finite_element'
    element_type = 'trilinear_hexahedron'

    discretization = domain.Discretization(cell=my_cell,
                                           number_of_pixels=number_of_pixels,
                                           discretization_type=discretization_type,
                                           element_type=element_type)

    nodal_coordinates = discretization.get_nodal_points_coordinates()
    quad_coordinates = discretization.get_quad_points_coordinates()

    u_fun_4x3y = lambda x, y, z: (4 * x + 3 * y + 5 * z)  # * np.sin(x)
    #            u_fun_4x3y = lambda x, y: np.random.rand() * x + np.random.rand() * y  # np.sin(x)

    temperature = discretization.get_temperature_sized_field()
    temperature_gradient = discretization.get_temperature_gradient_size_field()

    temperature[0, 0, :, :] = u_fun_4x3y(nodal_coordinates[0, 0, :, :],
                                         nodal_coordinates[1, 0, :, :],
                                         nodal_coordinates[2, 0, :, :]
                                         )

    # temperature[0, 0, :, :] = np.random.rand(*number_of_pixels)

    tensor_dx = discretization.apply_gradient_operator(temperature, temperature_gradient).mean(axis=2)[0, 0]
    tensor_dy = discretization.apply_gradient_operator(temperature, temperature_gradient).mean(axis=2)[0, 1]
    tensor_dz = discretization.apply_gradient_operator(temperature, temperature_gradient).mean(axis=2)[0, 2]

    for rank_i in reversed(np.arange(2, max(number_of_pixels) + 1)):
        # print(rank_i)

        tensor = np.copy(temperature[0, 0, :, :])

        tt_cores = tensor_train_tools.tt_decompose_rank(tensor_xyz=tensor,
                                                        ranks=[1, rank_i, rank_i, 1])
        tt_reconstructed_field = tensor_train_tools.tt_to_full_format(tt_cores=tt_cores)

        # #
        tt_cores_grad = tensor_train_tools.get_gradient_finite_difference(tt_cores=tt_cores,
                                                                          voxel_sizes=discretization.pixel_size)
        tt_dNx_field_FD = tensor_train_tools.tt_to_full_format(tt_cores=tt_cores_grad[0])
        tt_dNy_field_FD = tensor_train_tools.tt_to_full_format(tt_cores=tt_cores_grad[1])
        tt_dNz_field_FD = tensor_train_tools.tt_to_full_format(tt_cores=tt_cores_grad[2])

        field_norm = np.linalg.norm(tensor)
        abs_error_norms = (np.linalg.norm(tensor - tt_reconstructed_field))
        rel_error_norms = (abs_error_norms / field_norm)

        # print(abs_error_norms)
        # print(rel_error_norms)
        #
        # print(np.linalg.norm(tt_dNx_field_FD - tensor_dx) / np.linalg.norm(tensor_dx))
        # print(np.linalg.norm(tt_dNy_field_FD - tensor_dy) / np.linalg.norm(tensor_dy))
        # print(np.linalg.norm(tt_dNz_field_FD - tensor_dz) / np.linalg.norm(tensor_dz))

        error_dx = np.allclose(tt_dNx_field_FD, tensor_dx)
        error_dy = np.allclose(tt_dNy_field_FD, tensor_dy)
        error_dz = np.allclose(tt_dNz_field_FD, tensor_dz)

        assert error_dx, (
            "TT-full rank gradient dx is not equal to full format gradient  {}".format(error_dx))
        assert error_dy, (
            "TT-full rank gradient dy is not equal to full format gradient  {}".format(error_dy))
        assert error_dz, (
            "TT-full rank gradient dy is not equal to full format gradient  {}".format(error_dz))


@pytest.mark.parametrize('domain_size , element_type, number_of_pixels', [
    ([2, 3], 'linear_triangles', [2, 2]),
    ([3, 2], 'linear_triangles', [4, 3]),
    ([1, 3], 'linear_triangles', [6, 4]),
    ([3, 4], 'linear_triangles', [3, 5]),
    ([2, 3], 'bilinear_rectangle', [2, 2]),
    ([3, 2], 'bilinear_rectangle', [4, 3]),
    ([1, 3], 'bilinear_rectangle', [6, 4]),
    ([3, 2], 'bilinear_rectangle', [3, 5]),
    ([2, 2, 3], 'trilinear_hexahedron', [2, 2, 2]),
    ([3, 4, 2], 'trilinear_hexahedron', [3, 4, 3]),
    ([2, 6, 3], 'trilinear_hexahedron', [4, 6, 4]),
    ([3, 3, 2], 'trilinear_hexahedron', [5, 3, 5])])
def test_tt_addition(domain_size, element_type, number_of_pixels):
    # test TT FD compared to finite element gradient
    # compare  fem gradient with low rank gradient for linear elements
    # number_of_pixels = [5, 6, 7]
    # domain_size = [3, 4, 5]  #
    problem_type = 'conductivity'  # 'elasticity'#,'conductivity'
    my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                      problem_type=problem_type)

    discretization_type = 'finite_element'
    # element_type = 'trilinear_hexahedron'  # #linear_triangles

    discretization = domain.Discretization(cell=my_cell,
                                           number_of_pixels=number_of_pixels,
                                           discretization_type=discretization_type,
                                           element_type=element_type)

    nodal_coordinates = discretization.get_nodal_points_coordinates()
    quad_coordinates = discretization.get_quad_points_coordinates()

    test_field_1 = discretization.get_temperature_sized_field()
    test_field_2 = discretization.get_temperature_sized_field()

    if number_of_pixels.__len__() == 2:
        u_fun_4x = lambda x, y: (4 * x + 1 * y) * np.sin(x)
        u_fun_3y = lambda x, y: (2 * x + 3 * y) * np.sin(x)
        test_field_1[0, 0, :, :] = u_fun_4x(nodal_coordinates[0, 0, :, :],
                                            nodal_coordinates[1, 0, :, :]
                                            )
        test_field_2[0, 0, :, :] = u_fun_3y(nodal_coordinates[0, 0, :, :],
                                            nodal_coordinates[1, 0, :, :]
                                            )

    if number_of_pixels.__len__() == 3:
        u_fun_4x = lambda x, y, z: (4 * x + 2 * y + 5 * z) * np.sin(x)
        u_fun_3y = lambda x, y, z: (1 * x + 3 * y + 7 * z) * np.sin(x)
        test_field_1[0, 0, :, :] = u_fun_4x(nodal_coordinates[0, 0, :, :],
                                            nodal_coordinates[1, 0, :, :],
                                            nodal_coordinates[2, 0, :, :]
                                            )
        test_field_2[0, 0, :, :] = u_fun_3y(nodal_coordinates[0, 0, :, :],
                                            nodal_coordinates[1, 0, :, :],
                                            nodal_coordinates[2, 0, :, :]
                                            )

    #            u_fun_4x3y = lambda x, y: np.random.rand() * x + np.random.rand() * y  # np.sin(x)

    test_field_1 = test_field_1[0, 0, :, :]
    test_field_2 = test_field_2[0, 0, :, :]

    tt_cores_1, ranks_1 = tensor_train_tools.tt_decompose_error(tensor_xyz=test_field_1,
                                                                rel_error_norm=0.)

    tt_cores_2, ranks_2 = tensor_train_tools.tt_decompose_error(tensor_xyz=test_field_2,
                                                                rel_error_norm=0.)

    tt_plus = tensor_train_tools.tt_addition(tt_cores_1=tt_cores_1,
                                             tt_cores_2=tt_cores_2)

    tt_minus = tensor_train_tools.tt_subtraction(tt_cores_1=tt_cores_1,
                                                 tt_cores_2=tt_cores_2)

    tt_plus_reconstructed = tensor_train_tools.tt_to_full_format(tt_cores=tt_plus)
    tt_minus_reconstructed = tensor_train_tools.tt_to_full_format(tt_cores=tt_minus)

    error_plus = np.allclose(tt_plus_reconstructed, test_field_1 + test_field_2)

    error_minus = np.allclose(tt_minus_reconstructed, test_field_1 - test_field_2)

    assert error_plus, (
        "TT-addition does not return correct numbers {}".format(
            np.linalg.norm(tt_plus_reconstructed - (test_field_1 + test_field_2))))
    assert error_minus, (
        "TT-subtraction does not return correct numbers {}".format(
            np.linalg.norm(tt_minus_reconstructed - (test_field_1 - test_field_2))))


@pytest.mark.parametrize('domain_size , element_type, number_of_pixels', [
    ([2, 3,4], 'trilinear_hexahedron', [2, 3,4]),
    ([3, 2,5], 'trilinear_hexahedron', [4, 3,5]),])
def test_tt_rounding(domain_size, element_type, number_of_pixels):
    # test TT FD compared to finite element gradient
    # compare  fem gradient with low rank gradient for linear elements
    # number_of_pixels = [5, 6, 7]
    # domain_size = [3, 4, 5]  #
    problem_type = 'conductivity'  # 'elasticity'#,'conductivity'
    my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                      problem_type=problem_type)

    discretization_type = 'finite_element'
    # element_type = 'trilinear_hexahedron'  # #linear_triangles

    discretization = domain.Discretization(cell=my_cell,
                                           number_of_pixels=number_of_pixels,
                                           discretization_type=discretization_type,
                                           element_type=element_type)

    nodal_coordinates = discretization.get_nodal_points_coordinates()
    quad_coordinates = discretization.get_quad_points_coordinates()

    test_field_1 = discretization.get_temperature_sized_field()
    test_field_2 = discretization.get_temperature_sized_field()

    if number_of_pixels.__len__() == 2:
        u_fun_4x = lambda x, y: (4 * x + 1 * y) * np.sin(x)
        u_fun_3y = lambda x, y: (2 * x + 3 * y) * np.sin(x)
        test_field_1[0, 0, :, :] = u_fun_4x(nodal_coordinates[0, 0, :, :],
                                            nodal_coordinates[1, 0, :, :]
                                            )
        test_field_2[0, 0, :, :] = u_fun_3y(nodal_coordinates[0, 0, :, :],
                                            nodal_coordinates[1, 0, :, :]
                                            )

    if number_of_pixels.__len__() == 3:
        u_fun_4x = lambda x, y, z: (4 * x + 2 * y + 5 * z) * np.sin(x)
        u_fun_3y = lambda x, y, z: (1 * x + 3 * y + 7 * z) * np.sin(x)
        test_field_1[0, 0, :, :] = u_fun_4x(nodal_coordinates[0, 0, :, :],
                                            nodal_coordinates[1, 0, :, :],
                                            nodal_coordinates[2, 0, :, :]
                                            )
        test_field_2[0, 0, :, :] = u_fun_3y(nodal_coordinates[0, 0, :, :],
                                            nodal_coordinates[1, 0, :, :],
                                            nodal_coordinates[2, 0, :, :]
                                            )

    #            u_fun_4x3y = lambda x, y: np.random.rand() * x + np.random.rand() * y  # np.sin(x)

    test_field_1 = test_field_1[0, 0, :, :]
    test_field_2 = test_field_2[0, 0, :, :]

    tt_cores_1, ranks_1 = tensor_train_tools.tt_decompose_error(tensor_xyz=test_field_1,
                                                                rel_error_norm=0.)

    tt_cores_2, ranks_2 = tensor_train_tools.tt_decompose_error(tensor_xyz=test_field_2,
                                                                rel_error_norm=0.)

    tt_plus = tensor_train_tools.tt_addition(tt_cores_1=tt_cores_1,
                                             tt_cores_2=tt_cores_2)

    tt_minus = tensor_train_tools.tt_subtraction(tt_cores_1=tt_cores_1,
                                                 tt_cores_2=tt_cores_2)

    tt_plus_reconstructed = tensor_train_tools.tt_to_full_format(tt_cores=tt_plus)
    tt_minus_reconstructed = tensor_train_tools.tt_to_full_format(tt_cores=tt_minus)

    rounded = tensor_train_tools.tt_rounding_Martin(cores=tt_plus, epsilon=1.)

    error_plus = np.allclose(tt_plus_reconstructed, test_field_1 + test_field_2)

    error_minus = np.allclose(tt_minus_reconstructed, test_field_1 - test_field_2)

    assert error_plus, (
        "TT-rounding does not return correct numbers {}".format(
            np.linalg.norm(tt_plus_reconstructed - (test_field_1 + test_field_2))))
    assert error_minus, (
        "TT-rounding does not return correct numbers {}".format(
            np.linalg.norm(tt_minus_reconstructed - (test_field_1 - test_field_2))))
