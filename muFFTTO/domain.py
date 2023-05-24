import numpy as np


class PeriodicUnitCell:
    def __init__(self, name='', NumberOfPixels=None, CellSize=None, problem_type='conductivity', element_type='linear'):

        self.name = name
        self.N = tuple(np.array(NumberOfPixels, dtype=int))  # number of pixels, non-periodic nodes
        self.dim = self.N.__len__()  # dimension of problem
        self.element_type = element_type

        if problem_type == 'conductivity':
            self.u_shape = ()
            self.du_shape = (self.dim,)
            self.mat_shape = 2 * (self.dim,)

        elif problem_type == 'elastic':
            self.u_shape = (self.dim,)
            self.du_shape = 2 * (self.dim,)
            self.mat_shape = 4 * (self.dim,)


class DiscretizationLibrary:
    # Discretization is a container that store all important information about discretization of unit cell
    # such as physical dimension, number of pixels/voxels, FE type, number of quadrature points,
    # number of nodal points, etc....
    #
    def __init__(self, number_of_pixels=None, domain_size=None, discretization_type='finite_element'):
        self.number_of_pixels = tuple(
            np.array(number_of_pixels, dtype=int))  # number of pixels/voxels, non-periodic nodes
        self.domain_dimension = self.number_of_pixels.__len__()  # dimension of problem
        self.domain_size = tuple(np.array(domain_size, dtype=float))  # physical dimension of domain 1,2, or 3 Dim
        self.discretization_type = discretization_type # could be finite difference, Fourier or finite elements

    def get_discretization_info(element_type='linear_triangles'):


