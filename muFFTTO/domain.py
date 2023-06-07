import warnings

import numpy as np


class PeriodicUnitCell:
    def __init__(self, name='my_unit_cell', domain_size=None, problem_type='conductivity'):

        self.name = name
        self.domain_dimension = domain_size.__len__()  # dimension of problem
        self.domain_size = np.asarray(domain_size, dtype=float)  # physical dimension of domain 1,2, or 3 Dim
        self.problem_type = problem_type
        # TODO[Martin] left bottom corner of domain is in [0,0,0] should we change it?

        if not problem_type in ['conductivity', 'elasticity']:
            raise ValueError(
                'Unrecognised physical problem type {}. Choose from ' \
                ' : conductivity, or elasticity'.format(problem_type))

        if problem_type == 'conductivity':
            self.unknown_shape = np.array([1], dtype=int)  # temperature is a single scalar
            self.gradient_shape = np.array([1, self.domain_dimension],
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
        # self.nb_elements_per_pixel = None
        self.nb_nodes_per_pixel = None
        self.nodal_points_coordinates = None
        self.nb_vertices_per_pixel = 2 ** self.domain_dimension

        if discretization_type == 'finite_element':
            # finite element properties
            self.nb_quad_points_per_pixel = None
            self.quadrature_weights = None
            self.quad_points_coord = None
            self.get_discretization_info(element_type)
            self.unknown_size = [*self.cell.unknown_shape, self.nb_nodes_per_pixel, *self.nb_of_pixels]
            self.gradient_size = [*self.cell.gradient_shape, self.nb_quad_points_per_pixel, *self.nb_of_pixels]
            self.material_data_size = [*self.cell.material_data_shape, self.nb_quad_points_per_pixel,
                                       *self.nb_of_pixels]

    def get_nodal_points_coordinates(self):
        # TODO[more then one nodal point] add coords for more than one nodal point
        nodal_points_coordinates = np.zeros([self.domain_dimension, self.nb_nodes_per_pixel, *self.nb_of_pixels])
        nodal_points_coordinates[:, 0] = np.meshgrid(
            *[np.arange(0, self.domain_size[d], self.pixel_size[d]) for d in range(0, self.domain_dimension)],
            indexing='ij')

        return nodal_points_coordinates

    def get_quad_points_coordinates(self):
        quad_points_coordinates = np.zeros([self.domain_dimension, self.nb_quad_points_per_pixel, *self.nb_of_pixels])

        #      for e in range(0, self.nb_elements_per_pixel):
        for q in range(0, self.nb_quad_points_per_pixel):
            quad_points_coordinates[:, q] = np.meshgrid(
                *[np.arange(0 + self.quad_points_coord[d, q], self.domain_size[d], self.pixel_size[d]) for d in
                  range(0, self.domain_dimension)],
                indexing='ij')

        return quad_points_coordinates

    def apply_gradient_operator(self, u, gradient_of_u=None):
        if gradient_of_u is None:
            gradient_of_u = self.get_gradient_size_field()

        if self.nb_nodes_per_pixel > 1:
            warnings.warn('Gradient operator is not tested for multiple nodal points per pixel.')

        gradient_of_u.fill(0)

        if self.domain_dimension == 2:  # TODO find the way how to make it dimensionles .. working for 3 as well
            for pixel_node in np.ndindex(
                    *np.ones([self.domain_dimension], dtype=int) * 2):  # iteration over all voxel corners
                pixel_node = np.asarray(pixel_node)

                gradient_of_u += np.einsum('dqn,fnxy->fdqxy', self.B_grad_at_pixel_dqnijk[(..., *pixel_node)],
                                           np.roll(u, -1 * pixel_node, axis=(2, 3)))

        elif self.domain_dimension == 3:
            for pixel_node in np.ndindex(
                    *np.ones([self.domain_dimension], dtype=int) * 2):  # iteration over all voxel corners
                pixel_node = np.asarray(pixel_node)

                gradient_of_u += np.einsum('dqn,fnxyz->fdqxyz', self.B_grad_at_pixel_dqnijk[(..., *pixel_node)],
                                           np.roll(u, -1 * pixel_node, axis=(2, 3, 4)))

        return gradient_of_u

    def apply_gradient_transposed_operator(self, gradient_of_u_fdqxyz, div_u_fnxyz=None):
        if div_u_fnxyz is None:
            div_u_fnxyz = self.get_unknown_size_field()

        if self.nb_nodes_per_pixel > 1:
            warnings.warn('Gradient operator is not tested for multiple nodal points per pixel.')

        div_u_fnxyz.fill(0)

        if self.domain_dimension == 2:  # TODO find the way how to make it dimensionless.. working for 3 as well

            for pixel_node in np.ndindex(
                    *np.ones([self.domain_dimension], dtype=int) * 2):  # iteration over all voxel corners
                pixel_node = np.asarray(pixel_node)

                div_fnxyz_pixel_node = np.einsum('dqn,fdqxy->fnxy', self.B_grad_at_pixel_dqnijk[(..., *pixel_node)],
                                                 gradient_of_u_fdqxyz)

                div_u_fnxyz += np.roll(div_fnxyz_pixel_node, 1 * pixel_node, axis=(2, 3))

        elif self.domain_dimension == 3:
            for pixel_node in np.ndindex(
                    *np.ones([self.domain_dimension], dtype=int) * 2):  # iteration over all voxel corners
                pixel_node = np.asarray(pixel_node)

                div_fnxyz_pixel_node = np.einsum('dqn,fdqxyz->fnxyz',
                                                 self.B_grad_at_pixel_dqnijk[(..., *pixel_node)],
                                                 gradient_of_u_fdqxyz)

                div_u_fnxyz += np.roll(div_fnxyz_pixel_node, 1 * pixel_node, axis=(2, 3, 4))
                warnings.warn('Gradient transposed is not tested for 3D.')

        return div_u_fnxyz

    # def get_einsum_path(self):
    # TODO make a einsum optimization for repetitive run of Gradeint operator
    #  self.einsum_path = np.einsum_path('dqn,fnxy->fdqxy', self.B_grad_at_pixel_dqnijk[(..., *pixel_node)],
    #                                       np.roll(u, -1 * pixel_node, axis=(2, 3)), optimize='optimal')[0]
    # for iteration in range(500):
    #     _ = np.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a, optimize=path)
    #
    def get_rhs(self, material_data_field, macro_gradient_field):
        # macro_gradient_field    [f,d,q,x,y,z]
        # material_data_field [d,d,d,d,q,x,y,z] - elasticity
        # material_data_field     [d,d,q,x,y,z] - conductivity
        # rhs                       [f,n,x,y,z]
        #  rhs=-Dt*A*E
        material_data_field = self.apply_quadrature_weights(material_data_field)
        stress = self.apply_material_data(material_data_field, macro_gradient_field)
        rhs = self.apply_gradient_transposed_operator(stress)

        return -rhs

    def get_macro_gradient_field(self, macro_gradient):
        # return macro gradient field from single macro gradient vector ---
        # macro_gradient_field [f,d,q,x,y,z]
        # macro_gradient       [f,d]

        macro_gradient_field = self.get_gradient_size_field()
        macro_gradient_field[..., :] = macro_gradient[(...,) + (np.newaxis,) * (macro_gradient_field.ndim - 2)]

        return macro_gradient_field

    def apply_quadrature_weights_elasticity(self, material_data):

        weighted_material_data = np.einsum('ijklq...,q->ijklq...', material_data, self.quadrature_weights)

        return weighted_material_data

    def apply_quadrature_weights_conductivity(self, material_data):

        weighted_material_data = np.einsum('ijq...,q->ijq...', material_data, self.quadrature_weights)

        return weighted_material_data

    def apply_quadrature_weights(self, material_data):
        if self.cell.problem_type == 'conductivity':
            return self.apply_quadrature_weights_conductivity(material_data)
        elif self.cell.problem_type == 'elasticity':
            return self.apply_quadrature_weights_elasticity(material_data)

    def apply_material_data_elasticity(self, material_data, gradient_field):
        # ddot42 = lambda A4, B2: np.einsum('ijklxyz,lkxyz  ->ijxyz  ', A4, B2)
        stress = np.einsum('ijkl...,lk...->ij...', material_data, gradient_field)

        return stress

    def apply_material_data_conductivity(self, material_data, gradient_field):
        # dot21  = lambda A,v: np.einsum('ij...,j...  ->i...',A,v)
        flux = np.einsum('ij...,uj...->ui...', material_data,
                         gradient_field)  # 'u' just to keep the size of array consistent

        return flux

    def apply_material_data(self, material_data, gradient_field):
        if self.cell.problem_type == 'conductivity':
            return self.apply_material_data_conductivity(material_data, gradient_field)
        elif self.cell.problem_type == 'elasticity':
            return self.apply_material_data_elasticity(material_data, gradient_field)

    def get_system_matrix(self, material_data):
        # memory hungry process that returns
        # loop over all possible unit impulses, to get all columns of system matrix
        unit_impulse = self.get_unknown_size_field()
        K_system_matrix = np.zeros([np.prod(unit_impulse.shape), np.prod(unit_impulse.shape)])
        i = 0
        for impuls_position in np.ndindex(unit_impulse.shape):
            unit_impulse.fill(0)
            unit_impulse[impuls_position] = 1
            K_impuls = self.apply_system_matrix(material_data, displacement_field=unit_impulse)

            K_system_matrix[i] = K_impuls.flatten()
            i += 1

        return K_system_matrix

    def apply_system_matrix(self, material_data_field, displacement_field):

        strain = self.apply_gradient_operator(displacement_field)
        material_data_field = self.apply_quadrature_weights(material_data_field)
        stress = self.apply_material_data(material_data_field, strain)
        force = self.apply_gradient_transposed_operator(stress)

        return force

    def get_unknown_size_field(self):
        # return zero field with the shape of unknown
        return np.zeros(self.unknown_size)

    def get_gradient_size_field(self):
        # return zero field for  the  (discretized)  gradient of temperature/displacement
        return np.zeros(self.gradient_size)

    def get_temperature_sized_field(self):
        # return zero field with the shape of discretized temperature field
        if not self.cell.problem_type == 'conductivity':
            warnings.warn(
                'Cell problem type is {}. But temperature sized field  is returned !!!'.format(self.cell.problem_type))

        return np.zeros([1, self.nb_nodes_per_pixel, *self.nb_of_pixels])

    def get_temperature_gradient_size_field(self):
        # return zero field for  the  (discretized)  gradient of temperature
        if not self.cell.problem_type == 'conductivity':
            warnings.warn(
                'Cell problem type is {}. But temperature gradient  sized field  is returned !!!'.format(
                    self.cell.problem_type))

        return np.zeros([1, self.domain_dimension, self.nb_quad_points_per_pixel, *self.nb_of_pixels])

    def get_temperature_material_data_size_field(self):
        # return zero field for  the  (discretized)  gradient of temperature
        if not self.cell.problem_type == 'conductivity':
            warnings.warn(
                'Cell problem type is {}. But temperature material data  sized field  is returned !!!'.format(
                    self.cell.problem_type))

        return np.zeros(
            [self.domain_dimension, self.domain_dimension, self.nb_quad_points_per_pixel, *self.nb_of_pixels])

    def get_displacement_sized_field(self):
        # return zero field with the shape of discretized displacement field
        if not self.cell.problem_type == 'elasticity':
            warnings.warn(
                'Cell problem type is {}. But displacement sized field  is returned !!!'.format(self.cell.problem_type))
        return np.zeros([self.domain_dimension, self.nb_nodes_per_pixel, *self.nb_of_pixels])

    def get_displacement_gradient_size_field(self):
        # return zero field for  the  (discretized)  gradient of  displacement field / strain
        if not self.cell.problem_type == 'elasticity':
            warnings.warn(
                'Cell problem type is {}. But displacement gradient  sized field  is returned !!!'.format(
                    self.cell.problem_type))

        return np.zeros(
            [self.domain_dimension, self.domain_dimension, self.nb_quad_points_per_pixel, *self.nb_of_pixels])

    def get_elasticity_material_data_field(self):
        # return zero field for  the  (discretized)  gradient of  displacement field / strain
        if not self.cell.problem_type == 'elasticity':
            warnings.warn(
                'Cell problem type is {}. But displacement material data  sized field  is returned !!!'.format(
                    self.cell.problem_type))

        return np.zeros([self.domain_dimension, self.domain_dimension, self.domain_dimension,
                         self.domain_dimension, self.nb_quad_points_per_pixel, *self.nb_of_pixels])

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
                      |  \            |
                      |     \  q = 2  |
                      |        \      |
                      | q = 1    \    |
                    x_1_______________x_2
                """
                self.nb_quad_points_per_pixel = 2
                self.nb_nodes_per_pixel = 1  # left bottom corner belong to pixel.
                self.nb_unique_nodes_per_pixel = 1

                # nodal points offsets
                self.offsets = np.array([[0, 0], [1, 0],
                                         [0, 1], [1, 1]])
                """  Structure of B matrix: 
                     B(:,:,q,e) --> is B matrix evaluate gradient at point q in  element e
                     B(:,:,q,e) has size [dim,nb_of_nodes/basis_functions] 
                                           (usually 4 in 2D and 8 in 3D)
                     B(:,:,q,e) = [ ∂φ_1/∂x_1  ∂φ_2/∂x_1  ∂φ_3/∂x_1 ∂φ_4/∂x_1 ;
                                    ∂φ_1/∂x_2  ∂φ_2/∂x_2  ∂φ_3/∂x_2 ∂φ_4/∂x_2]   at (q)
                """
                self.B_gradient = np.zeros([self.domain_dimension, 4,
                                            self.nb_quad_points_per_pixel])
                h_x = self.pixel_size[0]
                h_y = self.pixel_size[1]

                self.quad_points_coord = np.zeros(
                    [self.domain_dimension, self.nb_quad_points_per_pixel])
                self.quad_points_coord[:, 0] = [h_x / 3, h_y / 3]
                self.quad_points_coord[:, 1] = [h_x * 2 / 3, h_y * 2 / 3]

                # @formatter:off   B(dim,number of nodal values,quad point ,element)
                self.B_gradient[:, :,  0] = [[-1 / h_x,   1 / h_x,    0,          0],
                                               [-1 / h_y,         0,    1 / h_y,    0]]
                self.B_gradient[:, :,  1] = [[0,         0, - 1 / h_x, 1 / h_x],
                                               [0, - 1 / h_y,         0, 1 / h_y]]
                # @formatter:on
                self.quadrature_weights = np.zeros([self.nb_quad_points_per_pixel])
                self.quadrature_weights[0] = h_x * h_y / 2
                self.quadrature_weights[1] = h_x * h_y / 2

                B_at_pixel_dnijkq = self.B_gradient.reshape(self.domain_dimension,
                                                            self.nb_nodes_per_pixel,
                                                            *self.domain_dimension * (2,),
                                                            self.nb_quad_points_per_pixel)

                if self.domain_dimension == 2:
                    B_at_pixel_dnijkq = np.swapaxes(B_at_pixel_dnijkq, 2, 3)
                elif self.domain_dimension == 3:
                    B_at_pixel_dnijkq = np.swapaxes(B_at_pixel_dnijkq, 2, 4)
                    warnings.warn('Swapaxes for 3D is not tested.')

                self.B_grad_at_pixel_dqnijk = np.moveaxis(B_at_pixel_dnijkq, [-1], [1])

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
                self.nb_quad_points_per_pixel = 4
                # self.nb_elements_per_pixel = 1
                self.nb_nodes_per_pixel = 1  # x1 is the only pixel assigned node, x2 belongs to pixel +1

                #  pixel sizes for better readability
                h_x = self.pixel_size[0]
                h_y = self.pixel_size[1]
                # nodal points offsets
                self.offsets = np.array([[0, 0], [1, 0],
                                         [0, 1], [1, 1]])
                coord_helper = np.zeros(2)
                coord_helper[0] = -1. / (np.sqrt(3))
                coord_helper[1] = +1. / (np.sqrt(3))

                self.quad_points_coord = np.zeros(
                    [self.domain_dimension, self.nb_quad_points_per_pixel])

                self.quad_points_coord[:, 0] = [coord_helper[0], coord_helper[0]]
                self.quad_points_coord[:, 1] = [coord_helper[1], coord_helper[0]]
                self.quad_points_coord[:, 2] = [coord_helper[0], coord_helper[1]]
                self.quad_points_coord[:, 3] = [coord_helper[1], coord_helper[1]]

                self.B_gradient = np.zeros([self.domain_dimension, 4,
                                            self.nb_quad_points_per_pixel])
                # Jacobian matrix of transformation from iso-element to current size

                det_jacobian = h_x * h_y / 4

                inv_jacobian = np.array([[h_y / 2, 0], [0, h_x / 2]]) / det_jacobian

                # construction of B matrix
                for qp in range(0, self.nb_quad_points_per_pixel):
                    x_q = self.quad_points_coord[:, qp]
                    xi = x_q[0]
                    eta = x_q[1]
                    # @formatter:off
                    self.B_gradient[:,:, qp]=np.array( [[(eta - 1) / 4, (-eta + 1) / 4, (-eta - 1) / 4, (eta + 1) / 4],
                                                           [(xi  - 1) / 4, (-xi  - 1) / 4, (-xi  + 1) / 4, (xi  + 1) / 4]])

                    self.B_gradient[:,:, qp] = np.matmul(inv_jacobian,self.B_gradient[:,:, qp])
                    # @formatter:on

                self.quadrature_weights = np.zeros([self.nb_quad_points_per_pixel])
                self.quadrature_weights[0] = h_x * h_y / 4
                self.quadrature_weights[1] = h_x * h_y / 4
                self.quadrature_weights[2] = h_x * h_y / 4
                self.quadrature_weights[3] = h_x * h_y / 4
                # we have to recompute positions base on the size of current pixel !!!
                self.quad_points_coord = np.zeros(
                    [self.domain_dimension, self.nb_quad_points_per_pixel])

                self.quad_points_coord[:, 0] = [h_x / 2 + h_x * coord_helper[0] / 2,
                                                h_y / 2 + h_y * coord_helper[0] / 2]
                self.quad_points_coord[:, 1] = [h_x / 2 + h_x * coord_helper[1] / 2,
                                                h_y / 2 + h_y * coord_helper[0] / 2]
                self.quad_points_coord[:, 2] = [h_x / 2 + h_x * coord_helper[0] / 2,
                                                h_y / 2 + h_y * coord_helper[1] / 2]
                self.quad_points_coord[:, 3] = [h_x / 2 + h_x * coord_helper[1] / 2,
                                                h_y / 2 + h_y * coord_helper[1] / 2]
                # TODO find proper way of creating B matrices
                B_at_pixel_dnijkq = self.B_gradient.reshape(self.domain_dimension,
                                                            self.nb_nodes_per_pixel,
                                                            *self.domain_dimension * (2,),
                                                            self.nb_quad_points_per_pixel)

                if self.domain_dimension == 2:
                    B_at_pixel_dnijkq = np.swapaxes(B_at_pixel_dnijkq, 2, 3)
                elif self.domain_dimension == 3:
                    B_at_pixel_dnijkq = np.swapaxes(B_at_pixel_dnijkq, 2, 4)
                    warnings.warn('Swapaxes for 3D is not tested.')

                self.B_grad_at_pixel_dqnijk = np.moveaxis(B_at_pixel_dnijkq, [-1], [1])

            case _:
                raise ValueError('Element type {} is not implemented yet'.format(element_type))


def compute_Vight_notation(C):
    # function return Voigt notation of elastic tensor in quad. point
    if len(C) == 2:
        C_voigt = np.zeros([3, 3])
        i_ind = [(0, 0), (1, 1), (0, 1)]
        for i in np.arange(len(C_voigt[0])):
            for j in np.arange(len(C_voigt[1])):
                # print()
                # print([i_ind[i]+quad_point+i_ind[j]+quad_point])
                C_voigt[i, j] = C[i_ind[i] + i_ind[j]]

    elif len(C) == 3:
        C_voigt = np.zeros([6, 6])
        i_ind = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
        for i in np.arange(len(C_voigt[0])):
            for j in np.arange(len(C_voigt[1])):
                C_voigt[i, j] = C[i_ind[i] + i_ind[j]]
                # print()
    return C_voigt


def get_bulk_and_shear_modulus(E, poison):
    K = E / (3 * (1 - 2 * poison))
    G = E / (2 * (1 + poison))
    return K, G


def get_elastic_material_tensor(dim, K=1, mu=0.5, kind='linear'):
    shape = np.array(4 * [dim, ])
    mat = np.zeros(shape)
    kron = lambda a, b: 1 if a == b else 0

    if kind in 'linear':
        for alpha, beta, gamma, delta in np.ndindex(*shape):
            mat[alpha, beta, gamma, delta] = (K * (kron(alpha, beta) * kron(gamma, delta))
                                              + mu * (kron(alpha, gamma) * kron(beta, delta) +
                                                      kron(alpha, delta) * kron(beta, gamma) -
                                                      2 / 3 * kron(alpha, beta) * kron(gamma, delta)))
            # https://en.wikipedia.org/wiki/Linear_elasticity
    return mat
