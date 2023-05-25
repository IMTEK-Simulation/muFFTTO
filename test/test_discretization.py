import unittest

from muFFTTO import domain


class MyTestCase(unittest.TestCase):
    def test_discretization_initialization(self):
        domain_size = [3, 4]
        problem_type = 'elasticity'
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)

        number_of_pixels = (4, 5)

        discretization_type = 'finite_element'
        element_type = 'bilinear_rectangle'  # 'linear_triangles'

        discretization = domain.DiscretizationLibrary(cell=my_cell,
                                                      number_of_pixels=number_of_pixels,
                                                      discretization_type=discretization_type,
                                                      element_type=element_type)

        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
