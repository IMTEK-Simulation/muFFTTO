import unittest

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from muFFTTO import domain
from muFFTTO import solvers


class DiscretizationTestCase(unittest.TestCase):
    def test_discretization_initialization(self):
        domain_size = [3, 4]
        problem_type = 'elasticity'
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)

        number_of_pixels = (4, 5)

        discretization_type = 'finite_element'
        element_type = 'bilinear_rectangle'  # 'linear_triangles'

        discretization = domain.Discretization(cell=my_cell,
                                               number_of_pixels=number_of_pixels,
                                               discretization_type=discretization_type,
                                               element_type=element_type)

        self.assertTrue(hasattr(discretization, "cell"), 'Discretization has no "cell" ')
        self.assertTrue(hasattr(discretization, "domain_dimension"), 'Discretization has no "domain_dimension"')
        self.assertTrue(hasattr(discretization, "B_gradient"), 'Discretization has no "B_gradient" matrix')

        self.assertTrue(hasattr(discretization, "quadrature_weights"), 'Discretization has no "quadrature_weights" ')

        self.assertTrue(hasattr(discretization, "nb_quad_points_per_pixel"),
                        'Discretization has no "nb_quad_points_per_pixel" ')

        self.assertTrue(hasattr(discretization, "nb_nodes_per_pixel"), 'Discretization has no "nb_nodes_per_pixel" ')

    def test_2D_gradients_linear_conductivity(self):
        domain_size = [3, 4]
        problem_type = 'conductivity'  # 'elasticity'#,'conductivity'
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)

        number_of_pixels = (4, 5)

        discretization_type = 'finite_element'
        for element_type in ['linear_triangles', 'bilinear_rectangle']:
            discretization = domain.Discretization(cell=my_cell,
                                                   number_of_pixels=number_of_pixels,
                                                   discretization_type=discretization_type,
                                                   element_type=element_type)

            nodal_coordinates = discretization.get_nodal_points_coordinates()
            quad_coordinates = discretization.get_quad_points_coordinates()

            u_fun_4x3y = lambda x, y: 4 * x + 3 * y  # np.sin(x)
            du_fun_4 = lambda x: 4 + 0 * x  # np.cos(x)
            du_fun_3 = lambda y: 3 + 0 * y

            temperature = discretization.get_temperature_sized_field()
            temperature_gradient = discretization.get_temperature_gradient_size_field()
            temperature_gradient_anal = discretization.get_temperature_gradient_size_field()

            temperature[0, 0, :, :] = u_fun_4x3y(nodal_coordinates[0, 0, :, :],
                                                 nodal_coordinates[1, 0, :, :])
            temperature_gradient_anal[0, 0, :, :, :] = du_fun_4(quad_coordinates[0, :, :, :])
            temperature_gradient_anal[0, 1, :, :, :] = du_fun_3(quad_coordinates[1, :, :, :])

            temperature_gradient = discretization.apply_gradient_operator(temperature, temperature_gradient)
            temperature_gradient_rolled = discretization.apply_gradient_operator(temperature,
                                                                                 temperature_gradient)

            # test 1
            average = np.ndarray.sum(temperature_gradient)
            message = "Gradient does not have zero mean !!!! for 2D element {} in {} problem".format(element_type,
                                                                                                     problem_type)
            self.assertLessEqual(average, 1e-14, message)

            # test 2
            # compare values of gradient element wise --- without last-- periodic pixel that differs
            value_1 = np.alltrue(
                temperature_gradient[..., 0:-1, 0:-1] == temperature_gradient_anal[..., 0:-1, 0:-1])
            diff = np.ndarray.sum(
                temperature_gradient[..., 0:-1, 0:-1] - temperature_gradient_anal[..., 0:-1, 0:-1])
            value = np.allclose(temperature_gradient[..., 0:-1, 0:-1], temperature_gradient_anal[..., 0:-1, 0:-1],
                                rtol=1e-16, atol=1e-14)
            self.assertTrue(value,
                            'Gradient is not equal to analytical expression for 2D element {} in {} problem. Difference is {}'.format(
                                element_type, problem_type, diff))

            value_roll = np.alltrue(
                temperature_gradient == temperature_gradient_rolled)
            self.assertTrue(value_roll,
                            'Gradient is not equal to analytical expression for 2D element {} in {} problem. Difference is {}'.format(
                                element_type, problem_type, diff))

    def test_2D_gradients_bilinear_conductivity(self):
        domain_size = [4, 5]
        problem_type = 'conductivity'  # 'elasticity'#,'conductivity'
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)

        number_of_pixels = (4, 5)

        discretization_type = 'finite_element'
        for element_type in ['bilinear_rectangle']:
            discretization = domain.Discretization(cell=my_cell,
                                                   number_of_pixels=number_of_pixels,
                                                   discretization_type=discretization_type,
                                                   element_type=element_type)

            nodal_coordinates = discretization.get_nodal_points_coordinates()
            quad_coordinates = discretization.get_quad_points_coordinates()

            u_fun_4x3y = lambda x, y: x * y  # + 4 * y  # + 3 * y ** 2  # np.sin(x)
            du_fun_4 = lambda y: y  # np.cos(x)
            du_fun_3 = lambda x: x

            temperature = discretization.get_temperature_sized_field()
            temperature_gradient = discretization.get_temperature_gradient_size_field()
            temperature_gradient_rolled = discretization.get_temperature_gradient_size_field()

            temperature_gradient_anal = discretization.get_temperature_gradient_size_field()

            temperature[0, 0, :, :] = u_fun_4x3y(nodal_coordinates[0, 0, :, :],
                                                 nodal_coordinates[1, 0, :, :])
            temperature_gradient_anal[0, 0, :, :, :] = du_fun_4(quad_coordinates[1, :, :, :])
            temperature_gradient_anal[0, 1, :, :, :] = du_fun_3(quad_coordinates[0, :, :, :])

            temperature_gradient = discretization.apply_gradient_operator(temperature, temperature_gradient)

            temperature_gradient_rolled = discretization.apply_gradient_operator(temperature,
                                                                                 temperature_gradient_rolled)

            # test 1
            average = np.ndarray.sum(temperature_gradient)
            message = "Gradient does not have zero mean !!!! for 2D element {} in {} problem".format(element_type,
                                                                                                     problem_type)
            self.assertLessEqual(average, 1e-10, message)  ### weaker condition !!!!!

            # test 2
            # compare values of gradient element wise --- without last-- periodic pixel that differs

            for dir in range(domain_size.__len__()):
                value_1 = np.alltrue(
                    temperature_gradient[0, dir, ..., 0:-1, 0:-1] == temperature_gradient_anal[0, dir, ..., 0:-1, 0:-1])
                diff = np.ndarray.sum(
                    temperature_gradient[0, dir, ..., 0:-1, 0:-1] - temperature_gradient_anal[0, dir, ..., 0:-1, 0:-1])
                value = np.allclose(temperature_gradient[0, dir, ..., 0:-1, 0:-1],
                                    temperature_gradient_anal[0, dir, ..., 0:-1, 0:-1],
                                    rtol=1e-12, atol=1e-14)
                self.assertTrue(value,
                                'Gradient is not equal to analytical expression in direction {} for 2D element {} in {} '
                                ' problem. Difference is {}'.format(
                                    dir,
                                    element_type, problem_type, diff))

                value2 = np.allclose(temperature_gradient[0, dir, ..., 0:-1, 0:-1],
                                     temperature_gradient_rolled[0, dir, ..., 0:-1, 0:-1],
                                     rtol=1e-12, atol=1e-14)

                self.assertTrue(value2,
                                'Rolled gradient is not equal to looped implementation in direction {} for 2D element '
                                '{} in {}'
                                ' problem. Difference is {}'.format(
                                    dir,
                                    element_type, problem_type, diff))

    def test_2D_gradients_linear_elasticity(self):
        domain_size = [3, 4]
        problem_type = 'elasticity'  # 'elasticity'#,'conductivity'
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)

        number_of_pixels = (4, 5)

        discretization_type = 'finite_element'

        for direction in range(domain_size.__len__()):
            for element_type in ['linear_triangles', 'bilinear_rectangle']:
                discretization = domain.Discretization(cell=my_cell,
                                                       number_of_pixels=number_of_pixels,
                                                       discretization_type=discretization_type,
                                                       element_type=element_type)

                nodal_coordinates = discretization.get_nodal_points_coordinates()
                quad_coordinates = discretization.get_quad_points_coordinates()

                u_fun_4x3y = lambda x, y: 4 * x + 3 * y  # np.sin(x)
                du_fun_4 = lambda y: 4  # np.cos(x)
                du_fun_3 = lambda x: 3

                # displacement = discretization.get_unknown_size_field()
                displacement = discretization.get_displacement_sized_field()
                displacement_gradient = discretization.get_displacement_gradient_size_field()
                displacement_gradient_rolled = discretization.get_displacement_gradient_size_field()

                displacement_gradient_anal = discretization.get_displacement_gradient_size_field()

                displacement[direction, 0, :, :] = u_fun_4x3y(nodal_coordinates[0, 0, :, :],
                                                              nodal_coordinates[1, 0, :, :])

                displacement_gradient_anal[direction, 0, :, :, :] = du_fun_4(quad_coordinates[0, 0])
                displacement_gradient_anal[direction, 1, :, :, :] = du_fun_3(quad_coordinates[0, 0])

                displacement_gradient = discretization.apply_gradient_operator(displacement, displacement_gradient)

                displacement_gradient_rolled = discretization.apply_gradient_operator(
                    displacement,
                    displacement_gradient_rolled)

                for dir in range(domain_size.__len__()):
                    # test 1
                    average = np.ndarray.sum(displacement_gradient)
                    message = "Gradient does not have zero mean !!!! for 2D element {} in {} problem".format(
                        element_type,
                        problem_type)
                    self.assertLessEqual(average, 1e-14, message)

                    # test 2
                    # compare values of gradient element wise --- without last-- periodic pixel that differs
                    value_1 = np.alltrue(
                        displacement_gradient[direction, dir, :, 0:-1, 0:-1] == displacement_gradient_anal[direction,
                                                                                dir, :, 0:-1, 0:-1])
                    diff = np.ndarray.sum(
                        displacement_gradient[direction, dir, :, 0:-1, 0:-1] - displacement_gradient_anal[direction,
                                                                               dir, :, 0:-1, 0:-1])
                    value = np.allclose(displacement_gradient[direction, dir, :, 0:-1, 0:-1],
                                        displacement_gradient_anal[direction, dir, :, 0:-1, 0:-1],
                                        rtol=1e-16, atol=1e-14)
                    self.assertTrue(value,
                                    'Gradient is not equal to analytical expression for 2D element {} in {} problem. Difference is {}'.format(
                                        element_type, problem_type, diff))

                    value = np.allclose(displacement_gradient[direction, dir, :, 0:-1, 0:-1],
                                        displacement_gradient_rolled[direction, dir, :, 0:-1, 0:-1],
                                        rtol=1e-16, atol=1e-14)
                    self.assertTrue(value,
                                    'Rolled gradient do not coincide with looped gradient {} in {} problem. Difference is {}'.format(
                                        element_type, problem_type, diff))

    def test_2D_gradients_transposed_linear_conductivity(self):
        domain_size = [3, 4]
        problem_type = 'conductivity'  # 'elasticity'#,'conductivity'
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)

        number_of_pixels = (4, 5)

        discretization_type = 'finite_element'

        for element_type in ['linear_triangles', 'bilinear_rectangle']:
            discretization = domain.Discretization(cell=my_cell,
                                                   number_of_pixels=number_of_pixels,
                                                   discretization_type=discretization_type,
                                                   element_type=element_type)

            nodal_coordinates = discretization.get_nodal_points_coordinates()
            quad_coordinates = discretization.get_quad_points_coordinates()

            u_fun_4x3y = lambda x, y: 3 * x + 4 * y  # np.sin(x)
            du_fun_4 = lambda x: 3 + 0 * x  # np.cos(x)
            du_fun_3 = lambda y: 4 + 0 * y

            temperature = discretization.get_temperature_sized_field()
            temperature_gradient = discretization.get_temperature_gradient_size_field()
            temperature_gradient_rolled = discretization.get_temperature_gradient_size_field()

            temperature_gradient_anal = discretization.get_temperature_gradient_size_field()

            temperature[0, 0, :, :] = u_fun_4x3y(nodal_coordinates[0, 0, :, :],
                                                 nodal_coordinates[1, 0, :, :])
            temperature_gradient_anal[0, 0, :, :, :] = du_fun_4(quad_coordinates[0, :, :, :])
            temperature_gradient_anal[0, 1, :, :, :] = du_fun_3(quad_coordinates[1, :, :, :])

            temperature_gradient = discretization.apply_gradient_operator(temperature, temperature_gradient)
            temperature_gradient_rolled = discretization.apply_gradient_operator(temperature,
                                                                                 temperature_gradient_rolled)

            div_flux = discretization.get_unknown_size_field()
            div_flux_rolled = discretization.get_unknown_size_field()

            div_flux = discretization.apply_gradient_transposed_operator(temperature_gradient, div_flux)
            div_flux_rolled = discretization.apply_gradient_transposed_operator(
                temperature_gradient_rolled,
                div_flux_rolled)

            # test 1
            average = np.ndarray.sum(temperature_gradient)
            message = "Gradient does not have zero mean !!!! for 2D element {} in {} problem".format(element_type,
                                                                                                     problem_type)
            self.assertLessEqual(average, 1e-14, message)
            if element_type in ['linear_triangles']:
                solution = np.array([[-82.0, -32.0, -32.0, -31.999999999999993, 18.0],
                                     [-50.0, 3.552713678800501e-15, -3.552713678800501e-15, 7.105427357601002e-15,
                                      50.0],
                                     [-50.0, -3.552713678800501e-15, 7.105427357601002e-15, 0.0, 50.0],
                                     [-18.0, 32.0, 31.999999999999986, 32.0, 82.0]])
            elif element_type in ['bilinear_rectangle']:  ### we are missing integration weights !!!
                solution = 2 * np.array([[-82.0, -32.0, -32.0, -31.999999999999993, 18.0],
                                         [-50.0, 3.552713678800501e-15, -3.552713678800501e-15, 7.105427357601002e-15,
                                          50.0],
                                         [-50.0, -3.552713678800501e-15, 7.105427357601002e-15, 0.0, 50.0],
                                         [-18.0, 32.0, 31.999999999999986, 32.0, 82.0]])
            # test 2
            # compare values of gradient element wise --- without last-- periodic pixel that differs
            value_1 = np.alltrue(div_flux == solution)
            diff = np.ndarray.sum(div_flux - solution)
            value = np.allclose(div_flux, solution,
                                rtol=1e-16, atol=1e-13)
            self.assertTrue(value,
                            'B_transpose times B does return wrong field: 2D element {} in {} problem. Difference is {}'.format(
                                element_type, problem_type, diff))

            value_roll = np.allclose(solution, div_flux_rolled,
                                     rtol=1e-16, atol=1e-13)
            self.assertTrue(value_roll,
                            'Rolled gradient transposed do not coincide with looped gradient transposed : 2D element {} in {} problem. Difference is {}'.format(
                                element_type, problem_type, diff))

    def test_2D_gradients_transposed_linear_elasticity(self):
        domain_size = [3, 4]
        problem_type = 'elasticity'  # 'elasticity'#,'conductivity'
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)

        number_of_pixels = (4, 5)

        discretization_type = 'finite_element'
        for direction in range(domain_size.__len__()):
            for element_type in ['linear_triangles', 'bilinear_rectangle']:
                discretization = domain.Discretization(cell=my_cell,
                                                       number_of_pixels=number_of_pixels,
                                                       discretization_type=discretization_type,
                                                       element_type=element_type)

                nodal_coordinates = discretization.get_nodal_points_coordinates()
                quad_coordinates = discretization.get_quad_points_coordinates()

                u_fun_4x3y = lambda x, y: 3 * x + 4 * y  # np.sin(x)
                du_fun_4 = lambda x: 3 + 0 * x  # np.cos(x)
                du_fun_3 = lambda y: 4 + 0 * y

                displacement = discretization.get_displacement_sized_field()
                strain = discretization.get_displacement_gradient_size_field()
                strain_anal = discretization.get_displacement_gradient_size_field()

                displacement[direction, 0, :, :] = u_fun_4x3y(nodal_coordinates[0, 0, :, :],
                                                              nodal_coordinates[1, 0, :, :])
                strain_anal[direction, 0, :, :, :] = du_fun_4(quad_coordinates[0, :, :, :])
                strain_anal[direction, 1, :, :, :] = du_fun_3(quad_coordinates[1, :, :, :])

                strain = discretization.apply_gradient_operator(displacement, strain)

                div_flux = discretization.get_displacement_sized_field()
                div_flux = discretization.apply_gradient_transposed_operator(strain, div_flux)
                # test 1
                average = np.ndarray.sum(strain)
                message = "Gradient does not have zero mean !!!! for 2D element {} in {} problem".format(element_type,
                                                                                                         problem_type)
                self.assertLessEqual(average, 1e-14, message)
                if element_type in ['linear_triangles']:
                    solution = np.array([[-82.0, -32.0, -32.0, -31.999999999999993, 18.0],
                                         [-50.0, 3.552713678800501e-15, -3.552713678800501e-15, 7.105427357601002e-15,
                                          50.0],
                                         [-50.0, -3.552713678800501e-15, 7.105427357601002e-15, 0.0, 50.0],
                                         [-18.0, 32.0, 31.999999999999986, 32.0, 82.0]])
                elif element_type in ['bilinear_rectangle']:  ### we are missing integration weights !!!
                    solution = 2 * np.array([[-82.0, -32.0, -32.0, -31.999999999999993, 18.0],
                                             [-50.0, 3.552713678800501e-15, -3.552713678800501e-15,
                                              7.105427357601002e-15,
                                              50.0],
                                             [-50.0, -3.552713678800501e-15, 7.105427357601002e-15, 0.0, 50.0],
                                             [-18.0, 32.0, 31.999999999999986, 32.0, 82.0]])
                # test 2
                # compare values of gradient element wise --- without last-- periodic pixel that differs
                value_1 = np.alltrue(div_flux[direction, 0] == solution)
                diff = np.ndarray.sum(div_flux[direction, 0] - solution)
                value = np.allclose(div_flux[direction, 0], solution,
                                    rtol=1e-16, atol=1e-13)
                self.assertTrue(value,
                                'B_transpose times B does return wrong field: 2D element {} in {} problem. Difference is {}'.format(
                                    element_type, problem_type, diff))

    def test_2D_material_data_multiplication_linear_elasticity(self):
        # test multiplication with material data.
        # stress field have to be identical to strain if Identity tensor is applied

        domain_size = [3, 4]
        problem_type = 'elasticity'  # 'elasticity'#,'conductivity'
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)

        number_of_pixels = (4, 5)
        discretization_type = 'finite_element'
        for element_type in ['linear_triangles', 'bilinear_rectangle']:
            discretization = domain.Discretization(cell=my_cell,
                                                   number_of_pixels=number_of_pixels,
                                                   discretization_type=discretization_type,
                                                   element_type=element_type)

            displacement = discretization.get_displacement_sized_field()
            displacement = np.random.rand(*displacement.shape)

            strain = discretization.get_displacement_gradient_size_field()
            strain = discretization.apply_gradient_operator(displacement, strain)

            material_data = discretization.get_material_data_size_field()

            # identity tensor                                               [single tensor]
            i = np.eye(discretization.domain_dimension)
            # identity tensors                                            [grid of tensors]
            I = np.einsum('ij,xy', i, np.ones(discretization.nb_of_pixels))
            I4 = np.einsum('ijkl,qxy->ijklqxy', np.einsum('il,jk', i, i),
                           np.ones(np.array([discretization.nb_quad_points_per_pixel, *discretization.nb_of_pixels])))
            I4rt = np.einsum('ijkl,qxy->ijklqxy', np.einsum('ik,jl', i, i),
                             np.ones(np.array([discretization.nb_quad_points_per_pixel, *discretization.nb_of_pixels])))
            I4s = (I4 + I4rt) / 2.
            # dyad22 = lambda A2, B2: np.einsum('ijxy  ,klxy  ->ijklxy', A2, B2)
            # II = dyad22(I, I)

            K_1, G_1 = domain.get_bulk_and_shear_modulus(E=1, poison=0.0)
            K_2, G_2 = domain.get_bulk_and_shear_modulus(E=10, poison=0.3)

            mat_1 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension, K=K_1, mu=G_1,
                                                       kind='linear')
            mat_2 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension, K=K_2, mu=G_2,
                                                       kind='linear')

            material_data_field = discretization.get_elasticity_material_data_field()
            #  material_data_field=mat_1[..., np.newaxis, np.newaxis]
            material_data_field = np.einsum('ijkl,qxy->ijklqxy', mat_1,
                                            np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                              *discretization.nb_of_pixels])))
            stress_I4s = discretization.apply_material_data_elasticity(I4s, strain)
            stress_mat1 = discretization.apply_material_data_elasticity(material_data_field, strain)
            stress_I4 = discretization.apply_material_data_elasticity(I4, strain)

            # wq=discretization.apply_quadrature_weights_elasticity(material_data_field)
            # test 2
            # compare values of gradient element wise --- without last-- periodic pixel that differs

            value_1 = np.alltrue(stress_I4 == strain)
            diff = np.ndarray.sum(stress_I4 - strain)
            value = np.allclose(stress_I4, strain,
                                rtol=1e-16, atol=1e-13)

            self.assertTrue(value,
                            'Stress is not equal to strain after apling I4 tensor: 2D element {} in {} problem. Difference is {}'.format(
                                element_type, problem_type, diff))

    def test_2D_system_matrix_symmetricity(self):
        domain_size = [3, 4]
        for problem_type in ['conductivity', 'elasticity']:  # 'elasticity'#,'conductivity'
            my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                              problem_type=problem_type)
            number_of_pixels = (4, 5)
            discretization_type = 'finite_element'

            for element_type in ['linear_triangles', 'bilinear_rectangle']:
                discretization = domain.Discretization(cell=my_cell,
                                                       number_of_pixels=number_of_pixels,
                                                       discretization_type=discretization_type,
                                                       element_type=element_type)

                if problem_type == 'elasticity':
                    K_1, G_1 = domain.get_bulk_and_shear_modulus(E=3, poison=0.2)

                    mat_1 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension, K=K_1, mu=G_1,
                                                               kind='linear')

                    material_data_field = np.einsum('ijkl,qxy->ijklqxy', mat_1,
                                                    np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                                      *discretization.nb_of_pixels])))
                elif problem_type == 'conductivity':
                    mat_1 = np.array([[1, 0], [0, 1]])
                    material_data_field = np.einsum('ij,qxy->ijqxy', mat_1,
                                                    np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                                      *discretization.nb_of_pixels])))

                K = discretization.get_system_matrix(material_data_field)
                # test symmetricity

                self.assertTrue(np.allclose(K, K.T, rtol=1e-15, atol=1e-14),
                                'System matrix is not symmetric: 2D element {} in {} problem.'.format(
                                    element_type, problem_type))
                # test column sum to be 0
                for i in np.arange(K.shape[0]):
                    self.assertTrue(np.allclose(np.sum(K[i, :]), 0, rtol=1e-15, atol=1e-14),
                                    'Sum of  {} -th column of system matrix is not zero: 2D element {} in {} problem.'.format(
                                        i,
                                        element_type, problem_type))

    def test_2D_homogenization_problem_solution(self):
        domain_size = [3, 4]
        for problem_type in ['conductivity']:  # TODO add 'elasticity'
            my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                              problem_type=problem_type)
            number_of_pixels = (4, 5)
            discretization_type = 'finite_element'

            for element_type in ['linear_triangles', 'bilinear_rectangle']:
                discretization = domain.Discretization(cell=my_cell,
                                                       number_of_pixels=number_of_pixels,
                                                       discretization_type=discretization_type,
                                                       element_type=element_type)

                nodal_coordinates = discretization.get_nodal_points_coordinates()
                quad_coordinates = discretization.get_quad_points_coordinates()

                u_fun_4x3y = lambda x: 1 * x  # np.sin(x)

                if problem_type == 'elasticity':
                    K_1, G_1 = domain.get_bulk_and_shear_modulus(E=3, poison=0.2)

                    mat_1 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension, K=K_1, mu=G_1,
                                                               kind='linear')

                    material_data_field = np.einsum('ijkl,qxy->ijklqxy', mat_1,
                                                    np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                                      *discretization.nb_of_pixels])))
                    ref_material_data_field = np.copy(material_data_field)
                    material_data_field[:, :, :, :, :, 1, 1] = 2 * material_data_field[:, :, :, :, :, 1, 1]

                    macro_gradient = np.zeros([discretization.domain_dimension, discretization.domain_dimension])
                    macro_gradient = np.array([[1, 0], [0, 1]])


                elif problem_type == 'conductivity':
                    mat_1 = np.array([[1, 0], [0, 1]])
                    material_data_field = np.einsum('ij,qxy->ijqxy', mat_1,
                                                    np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                                      *discretization.nb_of_pixels])))
                    # TODO do not forget about this magic
                    #  material_data_field = np.einsum('ij,qxy->ijqxy', mat_1,
                    #                                 quad_coordinates[0])
                    ref_material_data_field = np.copy(material_data_field)
                    material_data_field[:, :, :, 1, 1] = 2 * material_data_field[:, :, :, 1, 1]
                    macro_gradient = np.zeros([1, discretization.domain_dimension])
                    macro_gradient[0, :] = np.array([1, 0])

                    if element_type == 'linear_triangles':  # $ element_type in ['linear_triangles', 'bilinear_rectangle']:
                        matlab_solution = np.array([[0.0095, 0.0241, 0.0241, 0.0095, 0.0052],
                                                    [0.0193, 0.0853, 0.0853, 0.0193, 0.0082],
                                                    [-0.0193, -0.0853, -0.0853, -0.0193, -0.0082],
                                                    [-0.0095, -0.0241, -0.0241, -0.0095, -0.0052]])
                        matlab_residials = np.array(
                            [0.6400, 0.047655764323228, 0.008172361420757, 0.001083793705242, 0.000025955687695,
                             0.000000003633173])
                        A_h = np.array([[1.038629693814802, 0], [0, 1.038486005762097]])

                    elif element_type == 'bilinear_rectangle':
                        matlab_solution = np.array(
                            [[0.010881546413312, 0.023360445840561, 0.023360445840561, 0.010881546413312,
                              0.000240833264357, ],
                             [0.001825629490337, 0.100100728911584, 0.100100728911584, 0.001825629490337,
                              0.002321736512470, ],
                             [- 0.001825629490337, -0.100100728911584, -0.100100728911584, -0.001825629490337,
                              -0.002321736512470, ],
                             [-0.010881546413312, -0.023360445840561, -0.023360445840561, -0.010881546413312,
                              -0.000240833264357]])
                        matlab_residials = np.array(
                            [0.640000000000000, 0.023650193538911, 0.000631355256664, 0.000028708702151,
                             0.000000310696540, 8.490181660642319e-10])
                        A_h = np.array([[1.036653236145122, 0], [0, 1.036742177950770]])

                macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)
                rhs = discretization.get_rhs(material_data_field, macro_gradient_field)

                K_fun = lambda x: discretization.apply_system_matrix(material_data_field, x)
                M_fun = lambda x: 1 * x

                solution, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(500), toler=1e-6)
                # test homogenized stress
                homogenized_stress = discretization.get_homogenized_stress(material_data_field,
                                                                           displacement_field=solution,
                                                                           macro_gradient_field=macro_gradient_field)

                self.assertTrue(np.allclose(matlab_solution, solution, rtol=1e-05, atol=1e-04),
                                'Solution is not equal to reference MatLab implementation: 2D element {} in {} problem.'
                                .format(element_type, problem_type))
                self.assertTrue(
                    np.allclose(np.asarray(norms['residual_rr'])[:-1], matlab_residials, rtol=1e-15, atol=1e-14),
                    'Residuals are not equal to reference MatLab implementation: 2D element {} in {} problem.'.format(
                        element_type, problem_type))
                self.assertTrue(np.allclose(homogenized_stress[0, 0], A_h[0, 0], rtol=1e-15, atol=1e-15),
                                'Homogenized stress is not equal to reference MatLab implementation: 2D element {} in {}'
                                ' problem.'.format(
                                    element_type, problem_type))
                # test if the preconditioner does not change the solution
                K = discretization.get_system_matrix(material_data_field)

                preconditioner = discretization.get_preconditioner(
                    reference_material_data_field=ref_material_data_field)

                M_fun = lambda x: discretization.apply_preconditioner(preconditioner, x)
                solution_M, norms_M = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(500), toler=1e-6)
                # test homogenized stress
                homogenized_stress_M = discretization.get_homogenized_stress(material_data_field,
                                                                             displacement_field=solution_M,
                                                                             macro_gradient_field=macro_gradient_field)
                self.assertTrue(np.allclose(matlab_solution, solution_M, rtol=1e-05, atol=1e-04),
                                'Preconditioned solution is not equal to reference MatLab implementation: 2D element {} in {} problem.'.format(
                                    element_type, problem_type))

                self.assertTrue(np.allclose(homogenized_stress_M[0, 0], A_h[0, 0], rtol=1e-15, atol=1e-15),
                                'Preconditioned homogenized stress is not equal to reference MatLab implementation: 2D element {} in {} problem.'.format(
                                    element_type, problem_type))

    def test_plot_2D_mesh(self):
        domain_size = [3, 4]
        problem_type = 'conductivity'
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)

        number_of_pixels = (4, 5)

        discretization_type = 'finite_element'
        for element_type in ['bilinear_rectangle', 'linear_triangles']:

            discretization = domain.Discretization(cell=my_cell,
                                                   number_of_pixels=number_of_pixels,
                                                   discretization_type=discretization_type,
                                                   element_type=element_type)

            nodal_coordinates = discretization.get_nodal_points_coordinates()
            quad_coordinates = discretization.get_quad_points_coordinates()
            plt.scatter(nodal_coordinates[0, 0], nodal_coordinates[1, 0])
            segs1 = np.stack((nodal_coordinates[0, 0], nodal_coordinates[1, 0]), axis=2)
            segs2 = segs1.transpose(1, 0, 2)
            plt.gca().add_collection(LineCollection(segs1))
            plt.gca().add_collection(LineCollection(segs2))
            for q in range(0, discretization.nb_quad_points_per_pixel):
                plt.scatter(quad_coordinates[0, q], quad_coordinates[1, q])

            plt.show()


if __name__ == '__main__':
    unittest.main()
