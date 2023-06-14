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
        for problem_type in ['elasticity', 'conductivity']:  # TODO add 'elasticity'
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

                if problem_type == 'elasticity':
                    K_1, G_1 = domain.get_bulk_and_shear_modulus(E=1, poison=0.0)

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
                if problem_type == 'conductivity':
                    self.assertTrue(np.allclose(matlab_solution, solution, rtol=1e-05, atol=1e-04),
                                    'Solution is not equal to reference MatLab implementation: 2D element {} in {} problem.'
                                    .format(element_type, problem_type))
                    self.assertTrue(
                        np.allclose(np.asarray(norms['residual_rr'])[:-1], matlab_residials, rtol=1e-15, atol=1e-14),
                        'Residuals are not equal to reference MatLab implementation: 2D element {} in {} problem.'.format(
                            element_type, problem_type))
                    self.assertTrue(np.allclose(A_h[0], homogenized_stress[0], rtol=1e-15, atol=1e-15),
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
                self.assertTrue(np.allclose(solution, solution_M, rtol=1e-05, atol=1e-04),
                                'Preconditioned solution is not equal to un preconditioned solution: 2D element {} in {} problem.'.format(
                                    element_type, problem_type))

                self.assertTrue(
                    np.allclose(homogenized_stress_M[0, 0], homogenized_stress[0, 0], rtol=1e-15, atol=1e-8),
                    'Preconditioned homogenized stress is not equal to to un preconditioned solution: 2D element {} in {} problem.'.format(
                        element_type, problem_type))

    def test_2D_integral_linearity(self):
        global material_data_field
        domain_size = [3, 4]
        for problem_type in ['conductivity', 'elasticity']:  # TODO add 'elasticity'
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

                    target_stress = np.array([[1, 0.3], [0.3, 2]])
                elif problem_type == 'conductivity':
                    mat_1 = np.array([[1, 0], [0, 1]])
                    material_data_field = np.einsum('ij,qxy->ijqxy', mat_1,
                                                    np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                                      *discretization.nb_of_pixels])))
                    target_stress = np.array([2, 0.5])
                    target_stress = target_stress[np.newaxis,]

                actual_stress = np.random.rand(*discretization.get_gradient_size_field().shape)
                actual_stress_int = discretization.integrate_over_cell(actual_stress)

                stress_difference = actual_stress - target_stress[(...,) + (np.newaxis,) * (actual_stress.ndim - 2)]
                stress_difference_int = discretization.integrate_over_cell(stress_difference)

                integral_target_stress = discretization.cell.domain_volume * target_stress

                self.assertTrue(
                    np.allclose(stress_difference_int + integral_target_stress, actual_stress_int, rtol=1e-15,
                                atol=1e-15),
                    'Integral linearity violation: 2D element {} in {} problem.'.format(
                        element_type, problem_type))

                # ------------- this is something different ---- implementation for phase field #
                def compute_df_sigma_du(actual_stress, target_stress, material_data_field):
                    # df_sigma_du = int( 2*(Sigma-Sigma_target):C:grad_sym )d_Omega # TODO missing grad_sym operator
                    stress_difference = 2 * actual_stress - target_stress[
                        (...,) + (np.newaxis,) * (actual_stress.ndim - 2)]

                    stress_difference = discretization.apply_material_data(material_data=material_data_field,
                                                                           gradient_field=stress_difference)

                    df_sigma_du = discretization.integrate_over_cell(stress_difference)
                    return df_sigma_du

                df_sigma_du = compute_df_sigma_du(actual_stress, target_stress, material_data_field)

                def compute_df_sigma_drho(actual_stress, target_stress, material_data_field, phase_field): # todo dadasdsadassdasdadasd
                    # df_sigma_drho = int( 2*(Sigma-Sigma_target):dK/drho )d_Omega
                    # dK/drho =

                    stress_difference = 2 * actual_stress - target_stress[
                        (...,) + (np.newaxis,) * (actual_stress.ndim - 2)]

                   # material_data_phase_field=

                    stress_difference = discretization.apply_material_data(material_data=material_data_field,
                                                                           gradient_field=stress_difference)


                    integral = discretization.integrate_over_cell(stress_difference)
                    return integral

                phase_field = np.random.rand(*discretization.get_temperature_sized_field().shape)
                def compute_gradient_of_double_well_potential(phase_field, w=1, eta=1):
                    # Derivative of the double-well potential with respect to phase-field
                    # phase field potential = int ( rho^2(1-rho)^2 )/eta   dx
                    # gradient phase field potential = int ((2 * phase_field( + 2 * phase_field^2  -  3 * phase_field +1 )) )/eta   dx
                    # d/dρ(ρ^2 (1 - ρ)^2) = 2 ρ (2 ρ^2 - 3 ρ + 1)

                    # phase field gradient  =( |grad (rgo)|^2 ) *eta

                    integrant = (2 * phase_field(2 * phase_field * phase_field - 3 * phase_field + 1))
                    # INDRE  derivative = w / eta * 2 * phase * (1 - phase) * (1 - 2 * phase) * lengths[0] * lengths[1] / nb_pixels
                    integral = discretization.integrate_over_cell(integrant)

                    return integral

                phase_field_potential=compute_gradient_of_double_well_potential(phase_field, w=1, eta=1)

                stress_difference_squared_int = discretization.integrate_over_cell(
                    stress_difference * stress_difference)

                integral_difference = np.einsum('fdqxy...->fd', stress_difference)
                integral_actual_stress = np.einsum('fdqxy...->fd', actual_stress)

                integral_target_stress = np.einsum('fdqxy...->fd',
                                                   target_stress[(...,) + (np.newaxis,) * (actual_stress.ndim - 2)])
                integral_actual_stress_W = np.einsum('ijq...,q->ijq...', actual_stress,
                                                     discretization.quadrature_weights)

    def test_2D_preconditioner_is_inverse_of_homogeneous_problem(self):
        global material_data_field
        domain_size = [3, 4]
        for problem_type in ['conductivity', 'elasticity']:  # TODO add 'elasticity'
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
                    if element_type == 'linear_triangles':
                        matlab_M = np.array(
                            [[0, 0.771847250933311, 0.294819415733356, 0.294819415733356, 0.771847250933311],
                             [0.468750000000000, 0.291636466712144, 0.180987606728960, 0.180987606728960,
                              0.291636466712144],
                             [0.234375000000000, 0.179783044222786, 0.130572618508737, 0.130572618508737,
                              0.179783044222786],
                             [0.468750000000000, 0.291636466712144, 0.180987606728960, 0.180987606728960,
                              0.291636466712144]])

                    elif element_type == 'bilinear_rectangle':
                        matlab_M = np.array(
                            [
                                [0, 0.771847250933311, 0.294819415733356, 0.294819415733356, 0.771847250933311],
                                [0.468750000000000, 0.399090648415113, 0.321730395643978, 0.321730395643978,
                                 0.399090648415113],
                                [0.234375000000000, 0.269121075316603, 0.354047706546202, 0.354047706546202,
                                 0.269121075316603],
                                [0.468750000000000, 0.399090648415113, 0.321730395643978, 0.321730395643978,
                                 0.399090648415113]])

                        ref_material_data_field = np.copy(material_data_field)

                        x_0 = np.random.rand(*discretization.get_unknown_size_field().shape)
                        x_0 -= x_0.mean()

                        K_fun = lambda x: discretization.apply_system_matrix(material_data_field, x)

                        preconditioner = discretization.get_preconditioner(
                            reference_material_data_field=ref_material_data_field)
                        # check if M is equal to reference implementation
                        if problem_type == 'conductivity':
                            self.assertTrue(np.allclose(preconditioner, matlab_M, rtol=1e-15, atol=1e-15),
                                            'Preconditioner is not equal to preconditioner from Matlab implementation: 2D element {} in {} problem.'.format(
                                                element_type, problem_type))
                        M_fun = lambda x: discretization.apply_preconditioner(preconditioner, x)

                        f_0 = K_fun(x_0)
                        x_1 = M_fun(f_0)

                        diff = x_0 - x_1
                        print(np.sum(diff))
                        self.assertTrue(np.allclose(x_0, x_1, rtol=1e-15, atol=1e-15),
                                        'Preconditioner is not the inverse of the system matrix with homogeneous data: 2D element {} in {} problem.'.format(
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
