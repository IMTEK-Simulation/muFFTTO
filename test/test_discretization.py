import unittest

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from muFFTTO import domain


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

        self.assertTrue(hasattr(discretization, "nb_quad_points_per_element"),
                        'Discretization has no "nb_quad_points_per_element" ')

        self.assertTrue(hasattr(discretization, "nb_elements_per_pixel"),
                        'Discretization has no "nb_element_per_pixel"')
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

            temperature = discretization.get_unknown_size_field()
            temperature_gradient = discretization.get_gradient_size_field()
            temperature_gradient_anal = discretization.get_gradient_size_field()

            temperature[0, 0, :, :] = u_fun_4x3y(nodal_coordinates[0, 0, :, :],
                                                 nodal_coordinates[1, 0, :, :])
            temperature_gradient_anal[0, :, :, :, :] = du_fun_4(quad_coordinates[0, :, :, :, :])
            temperature_gradient_anal[1, :, :, :, :] = du_fun_3(quad_coordinates[1, :, :, :, :])

            temperature_gradient = discretization.apply_gradient_operator(temperature, temperature_gradient)

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

            temperature = discretization.get_unknown_size_field()
            temperature_gradient = discretization.get_gradient_size_field()
            temperature_gradient_anal = discretization.get_gradient_size_field()

            temperature[0, 0, :, :] = u_fun_4x3y(nodal_coordinates[0, 0, :, :],
                                                 nodal_coordinates[1, 0, :, :])
            temperature_gradient_anal[0, :, :, :, :] = du_fun_4(quad_coordinates[1, :, :, :, :])
            temperature_gradient_anal[1, :, :, :, :] = du_fun_3(quad_coordinates[0, :, :, :, :])

            temperature_gradient = discretization.apply_gradient_operator(temperature, temperature_gradient)

            # test 1
            average = np.ndarray.sum(temperature_gradient)
            message = "Gradient does not have zero mean !!!! for 2D element {} in {} problem".format(element_type,
                                                                                                     problem_type)
            self.assertLessEqual(average, 1e-10, message)  ### weaker condition !!!!!

            # test 2
            # compare values of gradient element wise --- without last-- periodic pixel that differs

            for dir in range(domain_size.__len__()):
                value_1 = np.alltrue(
                    temperature_gradient[dir, ..., 0:-1, 0:-1] == temperature_gradient_anal[dir, ..., 0:-1, 0:-1])
                diff = np.ndarray.sum(
                    temperature_gradient[dir, ..., 0:-1, 0:-1] - temperature_gradient_anal[dir, ..., 0:-1, 0:-1])
                value = np.allclose(temperature_gradient[dir, ..., 0:-1, 0:-1],
                                    temperature_gradient_anal[dir, ..., 0:-1, 0:-1],
                                    rtol=1e-12, atol=1e-14)
                self.assertTrue(value,
                                'Gradient is not equal to analytical expression in direction {} for 2D element {} in {} '
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
                displacement_gradient_anal = discretization.get_displacement_gradient_size_field()

                displacement[direction, 0, :, :] = u_fun_4x3y(nodal_coordinates[0, 0, :, :],
                                                              nodal_coordinates[1, 0, :, :])

                displacement_gradient_anal[direction, 0, :, :, :, :] = du_fun_4(quad_coordinates[0, 0])
                displacement_gradient_anal[direction, 1, :, :, :, :] = du_fun_3(quad_coordinates[0, 0])

                displacement_gradient = discretization.apply_gradient_operator(displacement, displacement_gradient)

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
                        displacement_gradient[direction, dir, :, :, 0:-1, 0:-1] == displacement_gradient_anal[direction,
                                                                                   dir, :, :, 0:-1, 0:-1])
                    diff = np.ndarray.sum(
                        displacement_gradient[direction, dir, :, :, 0:-1, 0:-1] - displacement_gradient_anal[direction,
                                                                                  dir, :, :, 0:-1, 0:-1])
                    value = np.allclose(displacement_gradient[direction, dir, :, :, 0:-1, 0:-1],
                                        displacement_gradient_anal[direction, dir, :, :, 0:-1, 0:-1],
                                        rtol=1e-16, atol=1e-14)
                    self.assertTrue(value,
                                    'Gradient is not equal to analytical expression for 2D element {} in {} problem. Difference is {}'.format(
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

            temperature = discretization.get_unknown_size_field()
            temperature_gradient = discretization.get_gradient_size_field()
            temperature_gradient_anal = discretization.get_gradient_size_field()

            temperature[0, 0, :, :] = u_fun_4x3y(nodal_coordinates[0, 0, :, :],
                                                 nodal_coordinates[1, 0, :, :])
            temperature_gradient_anal[0, :, :, :, :] = du_fun_4(quad_coordinates[0, :, :, :, :])
            temperature_gradient_anal[1, :, :, :, :] = du_fun_3(quad_coordinates[1, :, :, :, :])

            temperature_gradient = discretization.apply_gradient_operator(temperature, temperature_gradient)

            div_flux = discretization.get_unknown_size_field()
            div_flux = discretization.apply_gradient_transposed_operator(temperature_gradient, div_flux)
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
            for e in range(0, discretization.nb_elements_per_pixel):
                for q in range(0, discretization.nb_quad_points_per_element):
                    plt.scatter(quad_coordinates[0, q, e], quad_coordinates[1, q, e])

            plt.show()


if __name__ == '__main__':
    unittest.main()
