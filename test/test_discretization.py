import unittest

from muFFTTO import domain


class MyTestCase(unittest.TestCase):
    def test_discretization_initialization(self):
        number_of_pixels = (3, 3)
        # number_of_pixels = [3, 3]

        domain_size = [3,4]
        discretization=domain.DiscretizationLibrary(number_of_pixels=number_of_pixels, domain_size=domain_size)


        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
