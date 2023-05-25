import numpy as np


class PeriodicUnitCell:
    def __init__(self, name='my_unit_cell', domain_size=None, problem_type='conductivity'):

        self.name = name
        self.domain_dimension = domain_size.__len__()  # dimension of problem
        self.domain_size = np.asarray(domain_size, dtype=float)  # physical dimension of domain 1,2, or 3 Dim

        if not problem_type in ['conductivity', 'elasticity']:
            raise ValueError(
                'Unrecognised physical problem type {}. Choose from ' \
                ' : conductivity, or elasticity'.format(problem_type))

        if problem_type == 'conductivity':
            self.unknown_shape = np.array([1], dtype=int)  # temperature is a single scalar
            self.gradient_shape = np.array([self.domain_dimension],
                                           dtype=int)  # temp. gradient is a vector of d components
            self.material_data_shape = np.array([self.domain_dimension, self.domain_dimension],
                                                dtype=int)  # mat. data matrix a vector of d components

        elif problem_type == 'elasticity':
            self.unknown_shape = np.array([self.domain_dimension], dtype=int)
            self.gradient_shape = np.array([self.domain_dimension, self.domain_dimension],
                                           dtype=int)  # mat. data matrix a vector of d components

            self.material_data_shape = np.array(
                [self.domain_dimension, self.domain_dimension, self.domain_dimension, self.domain_dimension],
                dtype=int)  # mat. data matrix a vector of d components


class DiscretizationLibrary:
    # Discretization is a container that store all important information about discretization of unit cell
    # such as physical dimension, number of pixels/voxels, FE type, number of quadrature points,
    # number of nodal points, etc....
    #
    def __init__(self, cell, number_of_pixels=None, discretization_type='finite_element',
                 element_type='linear_triangles'):
        self.cell = cell
        self.domain_dimension = cell.domain_dimension
        self.domain_size = cell.domain_size

        # number of pixels/voxels, without periodic nodes
        self.number_of_pixels = np.asarray(number_of_pixels, dtype=int)

        if not discretization_type in ['finite_element', 'finite_difference', 'Fourier']:
            raise ValueError(
                'Unrecognised discretization type {}. Choose from ' \
                ' : finite_element, finite_difference, or Fourier'.format(discretization_type))
        self.discretization_type = discretization_type  # could be finite difference, Fourier or finite elements

        # pixel properties
        self.pixel_size = self.domain_size / self.number_of_pixels
        self.nb_element_per_pixel = None
        self.nb_nodes_per_pixel = None

        if discretization_type == 'finite_element':
            # finite element properties
            self.nb_quad_points_per_element = None
            self.quadrature_weights = None
            self.quad_points_coord = None
            self.get_discretization_info(element_type)

    def get_discretization_info(self, element_type):

        if not element_type in ['linear_triangles', 'bilinear_rectangle']:
            raise ValueError('Unrecognised element_type {}'.format(element_type))

        match element_type:
            case 'linear_triangles':
                if self.domain_dimension != 2:
                    raise ValueError('Element_type {} is implemented only in 2D'.format(element_type))
                """ Geometry for 2 linear triangular elements in pixel
                    x_4_______________x_3
                      |  \            |
                      |     \  e = 2  |
                      |        \      |
                      | e = 1    \    |
                    x_1_______________x_2
                """
                self.nb_quad_points_per_element = 1
                self.nb_element_per_pixel = 2
                self.nb_nodes_per_pixel = 4
                """  Structure of B matrix: 
                     B(:,:,q,e) --> is B matrix evaluate gradient at point q in  element e
                     B(:,:,q,e) has size [dim,nb_of_nodes/basis_functions] 
                                           (usually 4 in 2D and 8 in 3D)
                     B(:,:,q,e) = [ ∂φ_1/∂x_1  ∂φ_2/∂x_1  ∂φ_3/∂x_1 ∂φ_4/∂x_1 ;
                                    ∂φ_1/∂x_2  ∂φ_2/∂x_2  ∂φ_3/∂x_2 ∂φ_4/∂x_2]   at (q)
                """
                self.B_gradient = np.zeros([self.domain_dimension, self.nb_nodes_per_pixel,
                                            self.nb_quad_points_per_element, self.nb_element_per_pixel])
                h_x = self.pixel_size[0]
                h_y = self.pixel_size[1]

                # @formatter:off
                self.B_gradient[:, :, 0, 0] = [[-1 / h_x,   1 / h_x,    0,          0],
                                               [-1 / h_y,         0,    0,    1 / h_y]]
                self.B_gradient[:, :, 0, 1] = [[0,         0, 1 / h_x, - 1 / h_x],
                                               [0, - 1 / h_y, 1 / h_y,         0]]
                # @formatter:on
                self.quadrature_weights = np.zeros([self.nb_quad_points_per_element, self.nb_element_per_pixel])
                self.quadrature_weights[0, 0] = h_x * h_y / 2
                self.quadrature_weights[0, 1] = h_x * h_y / 2

                return
            case 'bilinear_rectangle':

                """ 
                    %   x_4____e=1__x_3
                    %   |            |
                    %   |  q=4   q=3 |
                    %   |            |
                    %   |  q=1   q=2 |
                    %   x_1_________x_2
                    *(-1, 1) | (1, 1)
                    *x       - --------x
                    * | | |
                    * | | |
                    *-- | --------- | ----->  ξ
                    * | | |
                    * | | |
                    *x - --------x
                    *(-1, -1) | (1, -1)
                    *
                    *N₁ = (1 - ξ)(1 - η) / 4
                    *N₂ = (1 + ξ)(1 - η) / 4
                    *N₃ = (1 + ξ)(1 + η) / 4
                    *N₄ = (1 - ξ)(1 + η) / 4
                    * ∂N₁ / ∂ξ = - (1 - η) / 4, ∂N₁ / ∂η = - (1 - ξ) / 4
                    * ∂N₂ / ∂ξ = + (1 - η) / 4, ∂N₂ / ∂η = - (1 + ξ) / 4
                    * ∂N₃ / ∂ξ = + (1 + η) / 4, ∂N₃ / ∂η = + (1 + ξ) / 4
                    * ∂N₄ / ∂ξ = - (1 + η) / 4, ∂N₄ / ∂η = + (1 - ξ) / 4
                """
                self.nb_quad_points_per_element = 4
                self.nb_element_per_pixel = 1
                self.nb_nodes_per_pixel = 4

                #  pixel sizes for better readability
                h_x = self.pixel_size[0]
                h_y = self.pixel_size[1]

                coord_helper = np.zeros(2)
                coord_helper[0] = -1. / (np.sqrt(3))
                coord_helper[1] = +1. / (np.sqrt(3))

                self.quad_points_coord = np.zeros([self.nb_quad_points_per_element, self.domain_dimension])
                self.quad_points_coord[0, :] = [coord_helper[0], coord_helper[0]]
                self.quad_points_coord[1, :] = [coord_helper[1], coord_helper[0]]
                self.quad_points_coord[2, :] = [coord_helper[1], coord_helper[1]]
                self.quad_points_coord[3, :] = [coord_helper[0], coord_helper[1]]

                self.B_gradient = np.zeros([self.domain_dimension, self.nb_nodes_per_pixel,
                                            self.nb_quad_points_per_element, self.nb_element_per_pixel])
                # Jacobian matrix of transformation from iso-element to current size

                det_jacobian = h_x * h_y / 4

                inv_jacobian = np.array([[h_y / 2, 0], [0, h_x / 2]]) / det_jacobian

                # construction of B matrix
                for qp in range(0, self.nb_quad_points_per_element ):
                    x_q = self.quad_points_coord[qp, :]
                    xi = x_q[0]
                    eta = x_q[1]
                    # @formatter:off
                    self.B_gradient[:,:, qp, 0]=np.array( [[(eta - 1) / 4, (-eta + 1) / 4, (eta + 1) / 4, (-eta - 1) / 4],
                                                                         [(xi  - 1) / 4, (-xi  - 1) / 4, (xi  + 1) / 4, (-xi  + 1) / 4]])

                    self.B_gradient[:,:, qp, 0] = np.matmul(inv_jacobian,self.B_gradient[:,:, qp, 0])
                    # @formatter:on

                self.quadrature_weights = np.zeros([self.nb_quad_points_per_element, self.nb_element_per_pixel])
                self.quadrature_weights[0, 0] = h_x * h_y / 4
                self.quadrature_weights[1, 0] = h_x * h_y / 4
                self.quadrature_weights[2, 0] = h_x * h_y / 4
                self.quadrature_weights[3, 0] = h_x * h_y / 4

            case _:
                raise ValueError('Element type {} is not implemented yet'.format(element_type))
