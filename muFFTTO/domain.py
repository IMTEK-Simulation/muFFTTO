import numpy as np
import itertools
import warnings


class PeriodicUnitCell:
    def __init__(self, name='my_unit_cell', domain_size=None, problem_type='conductivity'):

        self.name = name
        self.domain_dimension = domain_size.__len__()  # dimension of problem
        self.domain_size = np.asarray(domain_size, dtype=float)  # physical dimension of domain 1,2, or 3 Dim
        # TODO[Martin] left bottom corner of domain is in [0,0,0] should we change it?
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


class Discretization:
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
        self.nb_of_pixels = np.asarray(number_of_pixels, dtype=np.intp)

        if not discretization_type in ['finite_element', 'finite_difference', 'Fourier']:
            raise ValueError(
                'Unrecognised discretization type {}. Choose from ' \
                ' : finite_element, finite_difference, or Fourier'.format(discretization_type))
        self.discretization_type = discretization_type  # could be finite difference, Fourier or finite elements

        # pixel properties
        self.pixel_size = self.domain_size / self.nb_of_pixels
        self.nb_elements_per_pixel = None
        self.nb_nodes_per_pixel = None
        self.nodal_points_coordinates = None
        self.nb_vertices_per_pixel = 2 ** self.domain_dimension

        if discretization_type == 'finite_element':
            # finite element properties
            self.nb_quad_points_per_element = None
            self.quadrature_weights = None
            self.quad_points_coord = None
            self.get_discretization_info(element_type)
            self.unknown_size = [*self.cell.unknown_shape, self.nb_nodes_per_pixel, *self.nb_of_pixels]
            self.gradient_size = [*self.cell.gradient_shape, self.nb_quad_points_per_element,
                                  self.nb_elements_per_pixel, *self.nb_of_pixels]
            self.material_data_size = [*self.cell.material_data_shape, self.nb_quad_points_per_element,
                                       self.nb_elements_per_pixel, *self.nb_of_pixels]

    def get_nodal_points_coordinates(self):
        # TODO[more then one nodal point] add coords for more than one nodal point
        nodal_points_coordinates = np.zeros([self.domain_dimension, self.nb_nodes_per_pixel, *self.nb_of_pixels])
        nodal_points_coordinates[:, 0] = np.meshgrid(
            *[np.arange(0, self.domain_size[d], self.pixel_size[d]) for d in range(0, self.domain_dimension)],
            indexing='ij')

        return nodal_points_coordinates

    def get_quad_points_coordinates(self):
        quad_points_coordinates = np.zeros([self.domain_dimension, self.nb_quad_points_per_element,
                                            self.nb_elements_per_pixel, *self.nb_of_pixels])

        for e in range(0, self.nb_elements_per_pixel):
            for q in range(0, self.nb_quad_points_per_element):
                print(e, q)
                quad_points_coordinates[:, q, e] = np.meshgrid(
                    *[np.arange(0 + self.quad_points_coord[d, q, e], self.domain_size[d], self.pixel_size[d]) for d in
                      range(0, self.domain_dimension)],
                    indexing='ij')

        return quad_points_coordinates

    def apply_gradient_operator(self, u, Du):
        u_at_pixel = np.zeros([*self.cell.unknown_shape, self.nb_nodes_per_pixel,
                               *self.domain_dimension * (2,)])
        # B_at_pixel = np.zeros([self.domain_dimension, self.nb_nodes_per_pixel,
        #                        *self.domain_dimension * (2,), self.nb_quad_points_per_element,
        #                        self.nb_elements_per_pixel])

        B_at_pixel_dnijkqe = self.B_gradient.reshape(self.domain_dimension,
                                                     self.nb_nodes_per_pixel,
                                                     *self.domain_dimension * (2,),
                                                     self.nb_quad_points_per_element,
                                                     self.nb_elements_per_pixel)
        if self.domain_dimension == 2:
            B_at_pixel_dnijkqe = np.swapaxes(B_at_pixel_dnijkqe, 2, 3)
        elif self.domain_dimension == 3:
            B_at_pixel_dnijkqe = np.swapaxes(B_at_pixel_dnijkqe, 2, 4)
            warnings.warn('Swapaxes for 3D is not tested.')

        if self.nb_nodes_per_pixel > 1:
            warnings.warn('Gradient operator is not tested for multiple nodal points per pixel.')

        for pixel_index in np.ndindex(*self.nb_of_pixels):  # iteration over pixels
            pixel_index = np.asarray(pixel_index)
            # vertices_indices = [slice(pixel_index[d], pixel_index[d] + 2) for d in range(0, self.domain_dimension)]
            vertices_indices2 = [(pixel_index[d], (pixel_index[d] + 1) % self.nb_of_pixels[d]) for d in
                                 range(0, self.domain_dimension)]
            pixel_indices = [(pixel_index[d], (pixel_index[d] + 1) % self.nb_of_pixels[d]) for d in
                             range(0, self.domain_dimension)]
          #  pixel_indices_np = np.asarray(pixel_indices)
          ##  test_i=u[0,0] # u[(..., *np.ix_([0, 1], [0, 1]))]
           # u[(..., *np.ix_(*pixel_indices))]
            #print(pixel_index)
           # print(vertices_indices)
           # print(vertices_indices_2)
            # n1[:, None], n2[None, :]    u[(..., pixel_indices[0],pixel_indices[1])]
            u_at_pixel = u[(..., *np.ix_(*pixel_indices))]# u[(..., *pixel_indices)]
            # aaaaa = np.einsum('dnijqe,fnij-->fdqe', self.B_gradient, u_at_pixel)
            Du[(..., *pixel_index)] = np.einsum('dnijqe,fnij->fdqe', B_at_pixel_dnijkqe, u_at_pixel)  # TODO

            # for n in np.arange(self.nb_nodes_per_pixel):
            #
            #     u_at_pixel[:, corner] = u[(..., *((n,) + tuple(pixel_index + self.offsets[:, corner])))]
            #     # TODO add periodic conditions!!!!!
            #     for e in np.arange(self.nb_elements_per_pixel):  # iteration over elements
            #         B_for_element = self.B_gradient[:, :, :, e]
            #         for q in np.arange(self.nb_quad_points_per_element):  # iteration over quad points
            #             B_for_quad = B_for_element[:, :, q]
            #             Du[:, q, e, pixel_index] = np.matmul(B_for_quad, u_at_pixel.transpose())

        return Du

    def get_unknown_size_field(self):
        # return zero field with the shape of unknown
        return np.zeros(self.unknown_size)

    def get_gradient_size_field(self):
        # return zero field for  the  (discretized)  gradient of temperature/displacement
        return np.zeros(self.gradient_size)

    def get_material_data_size_field(self):
        # return zero field for the (discretized) material data
        return np.zeros(self.material_data_size)

    def get_discretization_info(self, element_type):

        if not element_type in ['linear_triangles', 'bilinear_rectangle']:
            raise ValueError('Unrecognised element_type {}'.format(element_type))

        match element_type:
            case 'linear_triangles':
                if self.domain_dimension != 2:
                    raise ValueError('Element_type {} is implemented only in 2D'.format(element_type))
                """ Geometry for 2 linear triangular elements in pixel
                    x_3_______________x_4
                      |  \     e = 2  |
                      |     \  q = 1  |
                      | e = 1  \      |
                      | q = 1    \    |
                    x_1_______________x_2
                """
                self.nb_quad_points_per_element = 1
                self.nb_elements_per_pixel = 2
                self.nb_nodes_per_pixel = 1  # left bottom corner belong to pixel.
                # nodal points offsets
                self.offsets = np.array([[0, 1, 1, 0],
                                         [0, 0, 1, 1]])
                """  Structure of B matrix: 
                     B(:,:,q,e) --> is B matrix evaluate gradient at point q in  element e
                     B(:,:,q,e) has size [dim,nb_of_nodes/basis_functions] 
                                           (usually 4 in 2D and 8 in 3D)
                     B(:,:,q,e) = [ ∂φ_1/∂x_1  ∂φ_2/∂x_1  ∂φ_3/∂x_1 ∂φ_4/∂x_1 ;
                                    ∂φ_1/∂x_2  ∂φ_2/∂x_2  ∂φ_3/∂x_2 ∂φ_4/∂x_2]   at (q)
                """
                self.B_gradient = np.zeros([self.domain_dimension, 4,
                                            self.nb_quad_points_per_element, self.nb_elements_per_pixel])
                h_x = self.pixel_size[0]
                h_y = self.pixel_size[1]

                self.quad_points_coord = np.zeros(
                    [self.domain_dimension, self.nb_quad_points_per_element, self.nb_elements_per_pixel])
                self.quad_points_coord[:, 0, 0] = [h_x / 3, h_y / 3]
                self.quad_points_coord[:, 0, 1] = [h_x * 2 / 3, h_y * 2 / 3]

                # @formatter:off
                self.B_gradient[:, :, 0, 0] = [[-1 / h_x,   1 / h_x,    0,          0],
                                               [-1 / h_y,         0,    1 / h_y,    0]]
                self.B_gradient[:, :, 0, 1] = [[0,         0, - 1 / h_x, 1 / h_x],
                                               [0, - 1 / h_y,         0, 1 / h_y]]
                # @formatter:on
                self.quadrature_weights = np.zeros([self.nb_quad_points_per_element, self.nb_elements_per_pixel])
                self.quadrature_weights[0, 0] = h_x * h_y / 2
                self.quadrature_weights[0, 1] = h_x * h_y / 2

                return
            case 'bilinear_rectangle':
                if self.domain_dimension != 2:
                    raise ValueError('Element_type {} is implemented only in 2D'.format(element_type))
                """ 
                    %   x_3____e=1__x_4
                    %   |            |
                    %   |  q=4   q=3 |
                    %   |            |
                    %   |  q=1   q=2 |
                    %   x_1_________x_2 - --------x
                    *(-1, 1) | (1, 1)
                    *x       
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
                self.nb_elements_per_pixel = 1
                self.nb_nodes_per_pixel = 1  # x1 is the only pixel assigned node, x2 belongs to pixel +1

                #  pixel sizes for better readability
                h_x = self.pixel_size[0]
                h_y = self.pixel_size[1]
                # nodal points offsets
                self.offsets = np.array([[0, 1, 0, 1],
                                         [0, 0, 1, 1]])
                coord_helper = np.zeros(2)
                coord_helper[0] = -1. / (np.sqrt(3))
                coord_helper[1] = +1. / (np.sqrt(3))

                self.quad_points_coord = np.zeros(
                    [self.domain_dimension, self.nb_quad_points_per_element, self.nb_elements_per_pixel])

                self.quad_points_coord[:, 0, 0] = [coord_helper[0], coord_helper[0]]
                self.quad_points_coord[:, 1, 0] = [coord_helper[1], coord_helper[0]]
                self.quad_points_coord[:, 2, 0] = [coord_helper[0], coord_helper[1]]
                self.quad_points_coord[:, 3, 0] = [coord_helper[1], coord_helper[1]]

                self.B_gradient = np.zeros([self.domain_dimension, 4,
                                            self.nb_quad_points_per_element, self.nb_elements_per_pixel])
                # Jacobian matrix of transformation from iso-element to current size

                det_jacobian = h_x * h_y / 4

                inv_jacobian = np.array([[h_y / 2, 0], [0, h_x / 2]]) / det_jacobian

                # construction of B matrix
                for qp in range(0, self.nb_quad_points_per_element):
                    x_q = self.quad_points_coord[:, qp, 0]
                    xi = x_q[0]
                    eta = x_q[1]
                    # @formatter:off
                    self.B_gradient[:,:, qp, 0]=np.array( [[(eta - 1) / 4, (-eta + 1) / 4, (-eta - 1) / 4, (eta + 1) / 4],
                                                           [(xi  - 1) / 4, (-xi  - 1) / 4, (-xi  + 1) / 4, (xi  + 1) / 4]])

                    self.B_gradient[:,:, qp, 0] = np.matmul(inv_jacobian,self.B_gradient[:,:, qp, 0])
                    # @formatter:on

                self.quadrature_weights = np.zeros([self.nb_quad_points_per_element, self.nb_elements_per_pixel])
                self.quadrature_weights[0, 0] = h_x * h_y / 4
                self.quadrature_weights[1, 0] = h_x * h_y / 4
                self.quadrature_weights[2, 0] = h_x * h_y / 4
                self.quadrature_weights[3, 0] = h_x * h_y / 4
                # we have to recompute positions base on the size of current pixel !!!
                self.quad_points_coord = np.zeros(
                    [self.domain_dimension, self.nb_quad_points_per_element, self.nb_elements_per_pixel])

                self.quad_points_coord[:, 0, 0] = [h_x / 2 + h_x * coord_helper[0] / 2,
                                                   h_y / 2 + h_y * coord_helper[0] / 2]
                self.quad_points_coord[:, 1, 0] = [h_x / 2 + h_x * coord_helper[1] / 2,
                                                   h_y / 2 + h_y * coord_helper[0] / 2]
                self.quad_points_coord[:, 2, 0] = [h_x / 2 + h_x * coord_helper[0] / 2,
                                                   h_y / 2 + h_y * coord_helper[1] / 2]
                self.quad_points_coord[:, 3, 0] = [h_x / 2 + h_x * coord_helper[1] / 2,
                                                   h_y / 2 + h_y * coord_helper[1] / 2]

            case _:
                raise ValueError('Element type {} is not implemented yet'.format(element_type))
