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

        self.assertTrue(hasattr(discretization, "nb_element_per_pixel"), 'Discretization has no "nb_element_per_pixel"')
        self.assertTrue(hasattr(discretization, "nb_nodes_per_pixel"), 'Discretization has no "nb_nodes_per_pixel" ')

    def test_plot_2D_mesh_lin_triangles(self):
        domain_size = [3, 4]
        problem_type = 'conductivity'
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)

        number_of_pixels = (4, 5)

        discretization_type = 'finite_element'
        element_type = 'linear_triangles'  # 'linear_triangles'

        discretization = domain.Discretization(cell=my_cell,
                                               number_of_pixels=number_of_pixels,
                                               discretization_type=discretization_type,
                                               element_type=element_type)

        nodal_coordinates= discretization.get_nodal_points_coordinates()
        quad_coordinates= discretization.get_quad_points_coordinates()
        plt.scatter(nodal_coordinates[0,0],nodal_coordinates[1,0])
        segs1 = np.stack((nodal_coordinates[0,0],nodal_coordinates[1,0]), axis=2)
        segs2 = segs1.transpose(1, 0, 2)
        plt.gca().add_collection(LineCollection(segs1))
        plt.gca().add_collection(LineCollection(segs2))
        plt.scatter(quad_coordinates[0, 0, 0], quad_coordinates[1, 0, 0])
        plt.scatter(quad_coordinates[0, 0, 1], quad_coordinates[1, 0, 1])

        plt.show()
        u_fun_sinx = lambda x: np.sin(x)

        temperature = discretization.get_unknown_size_field()
        temperature = u_fun_sinx(nodal_coordinates[0,0,:,:])
#        Du = apply_differential_operator(u, test_mesh_info.B, test_mesh_info);




        temperature_gradient = discretization.get_gradient_size_field()

        conductivity_data = discretization.get_material_data_size_field()
    def test_plot_2D_mesh_bilin_rectangles(self):
        domain_size = [3, 4]
        problem_type = 'conductivity'
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)

        number_of_pixels = (4, 5)

        discretization_type = 'finite_element'
        element_type = 'bilinear_rectangle'  # 'linear_triangles'

        discretization = domain.Discretization(cell=my_cell,
                                               number_of_pixels=number_of_pixels,
                                               discretization_type=discretization_type,
                                               element_type=element_type)

        nodal_coordinates= discretization.get_nodal_points_coordinates()
        quad_coordinates= discretization.get_quad_points_coordinates()
        plt.scatter(nodal_coordinates[0,0],nodal_coordinates[1,0])
        segs1 = np.stack((nodal_coordinates[0,0],nodal_coordinates[1,0]), axis=2)
        segs2 = segs1.transpose(1, 0, 2)
        plt.gca().add_collection(LineCollection(segs1))
        plt.gca().add_collection(LineCollection(segs2))
        for q in range(0,discretization.nb_quad_points_per_element):
            plt.scatter(quad_coordinates[0, q, 0], quad_coordinates[1, q, 0])

        plt.show()
        u_fun_sinx = lambda x: np.sin(x)

        temperature = discretization.get_unknown_size_field()
        temperature = u_fun_sinx(nodal_coordinates[0,0,:,:])
#        Du = apply_differential_operator(u, test_mesh_info.B, test_mesh_info);




        temperature_gradient = discretization.get_gradient_size_field()

        conductivity_data = discretization.get_material_data_size_field()

if __name__ == '__main__':
    unittest.main()
