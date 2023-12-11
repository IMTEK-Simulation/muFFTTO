import warnings

import numpy as np

from muFFTTO import discretization_library


# import pyfftw  TODO ask about FFTW or numpy FFT

class PeriodicUnitCell:
    def __init__(self, name='my_unit_cell', domain_size=None, problem_type='conductivity'):

        self.name = name
        self.domain_dimension = len(domain_size)  # dimension of problem
        self.domain_size = np.asarray(domain_size, dtype=float)  # physical dimension of domain 1,2, or 3 Dim
        self.domain_volume = np.prod(self.domain_size)

        self.problem_type = problem_type
        # TODO[Martin] left bottom corner of domain is in [0,0,0] should we change it?

        if not problem_type in ['conductivity', 'elasticity']:
            raise ValueError(
                'Unrecognised physical problem type {}. Choose from ' \
                ': conductivity, or elasticity'.format(problem_type))

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
    def __init__(self, cell,
                 number_of_pixels=None,
                 discretization_type='finite_element',
                 element_type='linear_triangles'):
        self.cell = cell
        self.domain_dimension = cell.domain_dimension
        self.domain_size = cell.domain_size
        # number of pixels/voxels, without periodic nodes
        self.nb_of_pixels = np.asarray(number_of_pixels, dtype=np.intp)

        if not discretization_type in ['finite_element']:
            raise ValueError(
                'Unrecognised discretization type {}. Choose from ' \
                ' : finite_element, finite_difference, or Fourier'.format(discretization_type))
        self.discretization_type = discretization_type  # only finite elements for now

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
            # displacement              [f,n,x,y,z]
            # rhs                       [f,n,x,y,z]
            # macro_gradient_field    [f,d,q,x,y,z]
            # material_data_field     [d,d,q,x,y,z] - conductivity
            # material_data_field [d,d,d,d,q,x,y,z] - elasticity
            #  rhs=-Dt*A*E

    def get_nodal_points_coordinates(self):
        # TODO[more then one nodal point] add coords for more than one nodal point
        nodal_points_coordinates = np.zeros([self.domain_dimension, self.nb_nodes_per_pixel, *self.nb_of_pixels])
        nodal_points_coordinates[:, 0] = np.meshgrid(
            *[np.arange(0, self.domain_size[d], self.pixel_size[d]) for d in range(0, self.domain_dimension)],
            indexing='ij')
        return nodal_points_coordinates

    # @property
    def get_quad_points_coordinates(self):
        # creates a field with [x,y,z] coordinates of all quadrature points
        quad_points_coordinates = np.zeros([self.domain_dimension, self.nb_quad_points_per_pixel, *self.nb_of_pixels])

        for q in range(0, self.nb_quad_points_per_pixel):
            quad_points_coordinates[:, q] = np.meshgrid(
                *[np.arange(0 + self.quad_points_coord[d, q], self.domain_size[d], self.pixel_size[d]) for d in
                  range(0, self.domain_dimension)],
                indexing='ij')

        return quad_points_coordinates

    def apply_gradient_operator(self, u, gradient_of_u=None):  ## TODO symmetric gradient operator """"""""""""""""""""
        if gradient_of_u is None:  # if gradient_of_u is not specified, determine the size
            #   u_field size :      [f,n,x,y,z]
            #   u_gradient_field    [f,d,q,x,y,z]
            gradient_of_u_shape = list(u.shape)  # [f,n,x,y,z]
            gradient_of_u_shape.insert(1, self.domain_dimension)  # [f,d,n,x,y,z]
            gradient_of_u_shape[2] = self.nb_quad_points_per_pixel  # [f,d,q,x,y,z]
            gradient_of_u = np.zeros(gradient_of_u_shape)  # create gradient field

        if self.nb_nodes_per_pixel > 1:
            warnings.warn('Gradient operator is not tested for multiple nodal points per pixel.')

        gradient_of_u.fill(0)  # To ensure that gradient field is empty/zero

        for pixel_node in np.ndindex(
                *np.ones([self.domain_dimension], dtype=int) * 2):  # iteration over all voxel corners
            pixel_node = np.asarray(pixel_node)
            if self.domain_dimension == 2:  # TODO find the way how to make it dimensionless .. working for 3 as well
                gradient_of_u += np.einsum('dqn,fnxy->fdqxy', self.B_grad_at_pixel_dqnijk[(..., *pixel_node)],
                                           np.roll(u, -1 * pixel_node, axis=(2, 3)))

            elif self.domain_dimension == 3:
                gradient_of_u += np.einsum('dqn,fnxyz->fdqxyz', self.B_grad_at_pixel_dqnijk[(..., *pixel_node)],
                                           np.roll(u, -1 * pixel_node, axis=(2, 3, 4)))

        return gradient_of_u

    def apply_gradient_operator_symmetrized(self, u, gradient_of_u=None):
        # computes symmetrized gradient (small-strain)  \epsilon_{ij} = \frac{1}{2} (u_{i,j} + u_{j,i})
        #   u_field size :      [f,n,x,y,z]
        #   u_gradient_field    [f,d,q,x,y,z]
        # 1. compute gradient
        gradient_of_u_ijqxyz = self.apply_gradient_operator(u)
        # 2. symmetrize it
        gradient_of_u_ijqxyz = (gradient_of_u_ijqxyz + np.swapaxes(gradient_of_u_ijqxyz, 0, 1)) / 2
        return gradient_of_u_ijqxyz

    def apply_gradient_transposed_operator(self, gradient_of_u_fdqxyz, div_u_fnxyz=None):
        if div_u_fnxyz is None:  # if div_u_fnxyz is not specified, determine the size
            #   gradient_of_u_fdqxyz :   [f,d,q,x,y,z]
            #   div_u_fnxyz size :      [f,n,x,y,z]
            div_u_fnxyz_shape = list(gradient_of_u_fdqxyz.shape)  # [f,d,q,x,y,z]
            div_u_fnxyz_shape.pop(1)  # [f,q,x,y,z]
            div_u_fnxyz_shape[1] = self.nb_nodes_per_pixel  # [f,n,x,y,z]
            div_u_fnxyz = np.zeros(div_u_fnxyz_shape)  # create div field

        if self.nb_nodes_per_pixel > 1:
            warnings.warn('Gradient operator is not tested for multiple nodal points per pixel.')

        div_u_fnxyz.fill(0)

        for pixel_node in np.ndindex(
                *np.ones([self.domain_dimension], dtype=int) * 2):  # iteration over all voxel corners
            pixel_node = np.asarray(pixel_node)
            if self.domain_dimension == 2:  # TODO find the way how to make it dimensionless.. working for 3 as well

                div_fnxyz_pixel_node = np.einsum('dqn,fdqxy->fnxy', self.B_grad_at_pixel_dqnijk[(..., *pixel_node)],
                                                 gradient_of_u_fdqxyz)

                div_u_fnxyz += np.roll(div_fnxyz_pixel_node, 1 * pixel_node, axis=(2, 3))

            elif self.domain_dimension == 3:

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
    def get_rhs(self, material_data_field_ijklqxyz, macro_gradient_field_ijqxyz):
        # macro_gradient_field    [f,d,q,x,y,z]
        # material_data_field [d,d,d,d,q,x,y,z] - elasticity
        # material_data_field     [d,d,q,x,y,z] - conductivity
        # rhs                       [f,n,x,y,z]
        #  rhs=-Dt*wA*E
        material_data_field_ijklqxyz = self.apply_quadrature_weights(material_data_field_ijklqxyz)
        stress = self.apply_material_data(material_data_field_ijklqxyz, macro_gradient_field_ijqxyz)
        rhs_fnxyz = self.apply_gradient_transposed_operator(stress)

        return -rhs_fnxyz

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

    def get_homogenized_stress(self, material_data_field_ijklqxyz,
                               displacement_field_fnxyz,
                               macro_gradient_field_ijqxyz,
                               formulation=None):
        # work for macro_grad= column of identity matrix:  eye(mesh_info.dim)[:, i]
        # A_h * macro_grad = int(A * (macro_grad + micro_grad))  dx / | domain |
        if formulation == 'small_strain':
            strain = self.apply_gradient_operator_symmetrized(displacement_field_fnxyz)
        else:
            strain = self.apply_gradient_operator(displacement_field_fnxyz)

        strain = strain + macro_gradient_field_ijqxyz
        material_data_field_ijklqxyz = self.apply_quadrature_weights(material_data_field_ijklqxyz)
        stress = self.apply_material_data(material_data_field_ijklqxyz, strain)
        homogenized_stress = np.sum(stress, axis=tuple(range(2, stress.shape.__len__()))) / self.cell.domain_volume
        return homogenized_stress

    def get_stress_field(self,
                         material_data_field_ijklqxyz,
                         displacement_field_fnxyz,
                         macro_gradient_field_ijqxyz,
                         formulation=None):
        # work for macro_grad= column of identity matrix:  eye(mesh_info.dim)[:, i]
        #  stress = A * (macro_grad + micro_grad)
        if formulation == 'small_strain':
            strain = self.apply_gradient_operator_symmetrized(displacement_field_fnxyz)
        else:
            strain = self.apply_gradient_operator(displacement_field_fnxyz)
        strain = strain + macro_gradient_field_ijqxyz
        stress = self.apply_material_data(material_data_field_ijklqxyz, strain)
        return stress

    def apply_quadrature_weights_conductivity(self, material_data):

        weighted_material_data = np.einsum('ijq...,q->ijq...', material_data, self.quadrature_weights)

        return weighted_material_data

    def apply_quadrature_weights(self, material_data):
        if self.cell.problem_type == 'conductivity':
            return self.apply_quadrature_weights_conductivity(material_data)
        elif self.cell.problem_type == 'elasticity':
            return self.apply_quadrature_weights_elasticity(material_data)

    def apply_quadrature_weights_on_gradient_field(self, grad_field):
        # apply quadrature weights without material data
        grad_field = np.einsum('ijq...,q->ijq...', grad_field, self.quadrature_weights)

        return grad_field

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

    def get_system_matrix(self, material_data_field):
        # memory hungry process that returns
        # loop over all possible unit impulses, to get all columns of system matrix
        unit_impulse = self.get_unknown_size_field()
        K_system_matrix = np.zeros([np.prod(unit_impulse.shape), np.prod(unit_impulse.shape)])
        i = 0
        for impuls_position in np.ndindex(unit_impulse.shape):
            unit_impulse.fill(0)
            unit_impulse[impuls_position] = 1
            K_impulse = self.apply_system_matrix(material_data_field, displacement_field=unit_impulse)

            K_system_matrix[i] = K_impulse.flatten()
            i += 1

        return K_system_matrix

    def get_preconditioner(self, reference_material_data_field_ijklqxyz,
                           formulation=None):
        # return diagonals of preconditioned matrix in Fourier space
        # unit_impulse [f,n,x,y,z]
        # for every type of degree of freedom DOF, there is one diagonal of preconditioner matrix
        # diagonals_in_Fourier_space [f,n,f,n][0,0,0]  # all DOFs in first pixel

        unit_impulse_fnxyz = self.get_unknown_size_field()
        preconditioner_diagonals_fnfnxyz = np.zeros(unit_impulse_fnxyz.shape[:2] + unit_impulse_fnxyz.shape)

        # construct unit impulses responses for all type of DOFs
        for impulse_position in np.ndindex(
                unit_impulse_fnxyz.shape[0:2]):  # loop over all types of degree of freedom [f,n]
            unit_impulse_fnxyz.fill(0)  # empty the unit impulse vector
            unit_impulse_fnxyz[impulse_position + (0,) * (
                    unit_impulse_fnxyz.ndim - 2)] = 1  # set 1 --- the unit impulse --- to a proper positions
            preconditioner_diagonals_fnfnxyz[impulse_position] = self.apply_system_matrix(
                material_data_field=reference_material_data_field_ijklqxyz,
                displacement_field=unit_impulse_fnxyz,
                formulation=formulation)

        # construct diagonals from unit impulses responses using FFT
        preconditioner_diagonals_fnfnxyz = np.fft.fftn(preconditioner_diagonals_fnfnxyz, [*self.nb_of_pixels])

        # compute inverse of diagonals
        for pixel_index in np.ndindex(unit_impulse_fnxyz.shape[2:]):  # TODO find the woy to avoid loops
            if pixel_index == (
                    0,) * self.domain_dimension:  # avoid  inversion of zeros # TODO find better solution for setting this to 0
                continue
            # pick local matrix with size [f*n,f*n]
            local_matrix = np.reshape(preconditioner_diagonals_fnfnxyz[(..., *pixel_index)],
                                      (np.prod(unit_impulse_fnxyz.shape[0:2]),
                                       np.prod(unit_impulse_fnxyz.shape[0:2])))
            # compute inversion
            local_matrix = np.linalg.inv(local_matrix)
            # rearrange local inverse matrix into global field
            preconditioner_diagonals_fnfnxyz[(..., *pixel_index)] = np.reshape(local_matrix, (
                    unit_impulse_fnxyz.shape[0:2] + unit_impulse_fnxyz.shape[0:2]))

        return preconditioner_diagonals_fnfnxyz  # return diagonals of preconditioner in Fourier space

    def apply_preconditioner(self, preconditioner_Fourier_fnfnxyz, nodal_field_fnxyz):
        # apply preconditioner using FFT
        # nodal_field_fnxyz [f,n,x,y,z]
        # preconditioner_Fourier_fnfnxyz [f,n,f,n,x,y,z] # TODO find better indexing notation

        # FFTn of the input field
        nodal_field_fnxyz = np.fft.fftn(nodal_field_fnxyz, [*self.nb_of_pixels])
        # multiplication with a diagonals of preconditioner
        nodal_field_fnxyz = np.einsum('abcd...,cd...->ab...', preconditioner_Fourier_fnfnxyz, nodal_field_fnxyz)
        # iFFTn
        nodal_field_fnxyz = np.real(np.fft.ifftn(nodal_field_fnxyz, [*self.nb_of_pixels]))

        return nodal_field_fnxyz

    def apply_system_matrix(self, material_data_field, displacement_field, formulation=None):

        if formulation == 'small_strain':
            strain = self.apply_gradient_operator_symmetrized(displacement_field)

        else:
            strain = self.apply_gradient_operator(displacement_field)

        material_data_field = self.apply_quadrature_weights(material_data_field)
        stress = self.apply_material_data(material_data_field, strain)
        force = self.apply_gradient_transposed_operator(stress)

        return force

    def integrate_over_cell(self, stress_field):
        # compute integral of stress field over the domain: int sigma d Omega = sum x_q *w_q

        # stress_field = np.einsum('ijq...,q->ijq...', stress_field, self.quadrature_weights)
        #
        # integral = np.einsum('fdqxy...->fd', stress_field)

        return integrate_field(stress_field, self.quadrature_weights)

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

    def get_scalar_sized_field(self):
        # return zero field with the shape of one scalar per nodal point

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
        discretization_library.get_shape_function_gradient_matrix(self, element_type)


def compute_Voigt_notation_4order(C_ijkl):
    # function return Voigt notation of elastic tensor in quad. point
    if len(C_ijkl) == 2:
        C_voigt_kl = np.zeros([3, 3])
        ij_ind = [(0, 0), (1, 1), (0, 1)]
        for k in np.arange(len(C_voigt_kl[0])):
            for l in np.arange(len(C_voigt_kl[1])):
                # print()
                # print([i_ind[i]+quad_point+i_ind[j]+quad_point])
                C_voigt_kl[k, l] = C_ijkl[ij_ind[k] + ij_ind[l]]

    elif len(C_ijkl) == 3:
        C_voigt_kl = np.zeros([6, 6])
        ij_ind = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
        for i in np.arange(len(C_voigt_kl[0])):
            for j in np.arange(len(C_voigt_kl[1])):
                C_voigt_kl[i, j] = C_ijkl[ij_ind[i] + ij_ind[j]]
                # print()
    return C_voigt_kl


def compute_Voigt_notation_2order(sigma_ij):
    # function return Voigt notation of second order tensor in quad. point
    if len(sigma_ij) == 2:
        sigma_voigt_k = np.zeros([3])
        ij_ind = [(0, 0), (1, 1), (0, 1)]

        for k in np.arange(len(sigma_voigt_k)):
            sigma_voigt_k[k] = sigma_ij[ij_ind[k]]

    elif len(sigma_ij) == 3:
        sigma_voigt_k = np.zeros([6])
        ij_ind = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
        for k in np.arange(len(sigma_voigt_k)):
            sigma_voigt_k[k] = sigma_ij[ij_ind[k]]

    return sigma_voigt_k


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


def compute_stress_difference(actual_stress, target_stress):
    stress_difference = actual_stress - target_stress[(...,) + (np.newaxis,) * (actual_stress.ndim - 2)]

    return stress_difference


def integrate_field(stress_field, quadrature_weights):
    stress_field = np.einsum('ijq...,q->ijq...', stress_field, quadrature_weights)

    integral = np.einsum('fdqxy...->fd', stress_field)

    return integral


def integrate_flux_field(flux_field, quadrature_weights):
    stress_field = np.einsum('ijq...,q->ijq...', flux_field, quadrature_weights)

    integral = np.einsum('fdqxy...->fd', stress_field)

    return integral
