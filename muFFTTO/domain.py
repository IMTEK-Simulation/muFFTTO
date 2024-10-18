import warnings

import numpy as np
import scipy as sc

from muFFTTO import discretization_library

from NuMPI.Tools import Reduction

from mpi4py import MPI
from muGrid import Communicator
from muGrid import FileIONetCDF, OpenMode
from muFFT import FFT


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
                                                dtype=int)  # mat. data matrix a  of dxd components

        elif problem_type == 'elasticity':
            self.unknown_shape = np.array([self.domain_dimension], dtype=int)
            # displacement. gradient is a vector of d components
            self.gradient_shape = np.array([self.domain_dimension, self.domain_dimension],
                                           dtype=int)  # gradient  matrix of d x d components
            self.material_data_shape = np.array(
                [self.domain_dimension, self.domain_dimension, self.domain_dimension, self.domain_dimension],
                dtype=int)  # mat. data tensor  of dxdxdxd components


class Discretization:
    # Discretization is a container that store all important information about discretization of unit cell
    # such as physical dimension, number of pixels/voxels, FE type, number of quadrature points,
    # number of nodal points, etc....
    #
    def __init__(self, cell,
                 nb_of_pixels_global=None,
                 discretization_type='finite_element',
                 element_type='linear_triangles',
                 communicator=MPI.COMM_WORLD):

        self.cell = cell
        self.domain_dimension = cell.domain_dimension
        self.domain_size = cell.domain_size
        # number of pixels/voxels, without periodic nodes
        self.nb_of_pixels_global = nb_of_pixels_global

        ## #todo[Lars] what engine?  FFT(nb_grid_pts, engine='mpi', communicator=MPI.COMM_WORLD)
        self.fft = FFT(nb_grid_pts=nb_of_pixels_global, engine='fftwmpi', communicator=communicator)
        self.mpi_reduction = Reduction(MPI.COMM_WORLD)
        self.nb_of_pixels = np.asarray(self.fft.nb_subdomain_grid_pts,
                                       dtype=np.intp)  # self.fft.nb_subdomain_grid_pts  #todo
        if not discretization_type in ['finite_element']:
            raise ValueError(
                'Unrecognised discretization type {}. Choose from ' \
                ' : finite_element, finite_difference, or Fourier'.format(discretization_type))
        self.discretization_type = discretization_type  # only finite elements for now
        # MPI.COMM_WORLD.Barrier()  # Barrier so header is printed first
        # print('rank' f'{MPI.COMM_WORLD.rank:6}')
        # print('nb sub points ' f'{self.fft.nb_subdomain_grid_pts}')

        # pixel properties
        self.pixel_size = self.domain_size / self.nb_of_pixels_global
        # self.nb_elements_per_pixel = None
        self.nb_nodes_per_pixel = None
        self.nodal_points_coordinates = None
        # MPI.COMM_WORLD.Barrier()  # Barrier so header is printed first
        # print('rank' f'{MPI.COMM_WORLD.rank:6}')
        # print('coords ' f'{self.nodal_points_coordinates}')

        self.nb_vertices_per_pixel = 2 ** self.domain_dimension

        if discretization_type == 'finite_element':
            # finite element properties
            self.element_type = element_type
            self.nb_quad_points_per_pixel = None
            self.quadrature_weights = None
            self.quad_points_coord = None
            self.quad_points_coord_parametric = None

            self.get_discretization_info(element_type)
            # MPI.COMM_WORLD.Barrier()  # Barrier so header is printed first
            # print('rank' f'{MPI.COMM_WORLD.rank:6}')
            # print('B matrix ' f'{self.B_grad_at_pixel_dqnijk}')
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
        nodal_points_coordinates = self.domain_size[:, np.newaxis, np.newaxis] * self.fft.coords
        #
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

    def roll(self, fft, u, shift, axis):
        phase = -2 * np.pi * sum(s * fft.fftfreq[a] for s, a in zip(shift, axis))
        # a= fft.ifft(np.exp(1j * phase) * fft.fft(u)) * fft.normalisation
        # a=a.reshape(u.shape)
        return (fft.ifft(np.exp(1j * phase) * fft.fft(u)) * fft.normalisation).reshape(u.shape)

    def apply_gradient_operator(self, u, gradient_of_u=None):
        if gradient_of_u is None:  # if gradient_of_u is not specified, determine the size
            #   u_field size :      [f,n,x,y,z]
            #   u_gradient_field    [f,d,q,x,y,z]
            gradient_of_u_shape = list(u.shape)  # [f,n,x,y,z]
            gradient_of_u_shape.insert(1, self.domain_dimension)  # [f,d,n,x,y,z]
            gradient_of_u_shape[2] = self.nb_quad_points_per_pixel  # [f,d,q,x,y,z]
            gradient_of_u = np.zeros(gradient_of_u_shape)  # create gradient field

        # MPI.COMM_WORLD.Barrier()  # Barrier so header is printed first
        # print('rank' f'{MPI.COMM_WORLD.rank:6} apply_gradient_operator:gradient_of_u_shape ' f'{gradient_of_u_shape}')
        # print('rank' f'{MPI.COMM_WORLD.rank:6} apply_gradient_operator:displacement_field_shape=' f'{u.shape}')

        if self.nb_nodes_per_pixel > 1:
            warnings.warn('Gradient operator is not tested for multiple nodal points per pixel.')

        gradient_of_u.fill(0)  # To ensure that gradient field is empty/zero
        # gradient_of_u_selfroll = np.copy(gradient_of_u)

        # !/usr/bin/env python3

        for pixel_node in np.ndindex(
                *np.ones([self.domain_dimension], dtype=int) * 2):  # iteration over all voxel corners
            # print('rank' f'{MPI.COMM_WORLD.rank:6} apply_gradient_operator: loop over pixel node =' f'{pixel_node}')

            pixel_node = np.asarray(pixel_node)
            if self.domain_dimension == 2:  # TODO find the way how to make it dimensionless .. working for 3 as well
                # gradient_of_u += np.einsum('dqn,fnxy->fdqxy', self.B_grad_at_pixel_dqnijk[(..., *pixel_node)],
                #                             np.roll(u, -1 * pixel_node, axis=(2, 3)))
                # print('rank' f'{MPI.COMM_WORLD.rank:6} apply_gradient_operator: einsum =' f'{pixel_node}')
                gradient_of_u += np.einsum('dqn,fnxy->fdqxy', self.B_grad_at_pixel_dqnijk[(..., *pixel_node)],
                                           self.roll(self.fft, u, -1 * pixel_node, axis=(0, 1)))

            elif self.domain_dimension == 3:
                # gradient_of_u += np.einsum('dqn,fnxyz->fdqxyz', self.B_grad_at_pixel_dqnijk[(..., *pixel_node)],
                #                            np.roll(u, -1 * pixel_node, axis=(2, 3, 4)))

                gradient_of_u += np.einsum('dqn,fnxyz->fdqxyz', self.B_grad_at_pixel_dqnijk[(..., *pixel_node)],
                                           self.roll(self.fft, u, -1 * pixel_node, axis=(0, 1, 2)))
        # TODO roll[u]= ifft (e^iq dx fft (u))
        # print('rank' f'{MPI.COMM_WORLD.rank:6} apply_gradient_operator: finish =' f'{gradient_of_u.shape}')

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

                # div_u_fnxyz += np.roll(div_fnxyz_pixel_node, 1 * pixel_node, axis=(2, 3))
                div_u_fnxyz += self.roll(self.fft, div_fnxyz_pixel_node, 1 * pixel_node, axis=(0, 1))

            elif self.domain_dimension == 3:

                div_fnxyz_pixel_node = np.einsum('dqn,fdqxyz->fnxyz',
                                                 self.B_grad_at_pixel_dqnijk[(..., *pixel_node)],
                                                 gradient_of_u_fdqxyz)

                # div_u_fnxyz += np.roll(div_fnxyz_pixel_node, 1 * pixel_node, axis=(2, 3, 4))
                div_u_fnxyz += self.roll(self.fft, div_fnxyz_pixel_node, 1 * pixel_node, axis=(0, 1, 2))

                warnings.warn('Gradient transposed is not tested for 3D.')

        return div_u_fnxyz

    # def get_einsum_path(self):
    # TODO make a einsum optimization for repetitive run of Gradeint operator
    #  self.einsum_path = np.einsum_path('dqn,fnxy->fdqxy', self.B_grad_at_pixel_dqnijk[(..., *pixel_node)],
    #                                       np.roll(u, -1 * pixel_node, axis=(2, 3)), optimize='optimal')[0]
    # for iteration in range(500):
    #     _ = np.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a, optimize=path)
    #

    def evaluate_field_at_quad_points(self,
                                      nodal_field_fnxyz,
                                      quad_field_fqnxyz=None,
                                      quad_points_coords_dq=None):
        # TODO: Implement evaluation of at quad points
        if quad_points_coords_dq is None:  # if quad_points_coords are not specified, use basic ones from B matrix
            nb_quad_points_per_pixel = self.nb_quad_points_per_pixel
            quad_points_coords_dq = self.quad_points_coord_parametric  # quad_points_coord[:,q]=[x_q,y_q,z_q]

        nb_quad_points_per_pixel = quad_points_coords_dq.shape[-1]
        if quad_field_fqnxyz is None:  # if quad_field_fqxyz is not specified, determine the size
            #   nodal_field_fnxyz : [f,n,x,y,z]
            #   quad_field_fqxyz : [f,q,x,y,z]

            quad_field_shape = list(nodal_field_fnxyz.shape)  # [f,n,x,y,z]
            quad_field_shape = np.insert(quad_field_shape, 1, nb_quad_points_per_pixel)
            quad_field_fqnxyz = np.zeros(quad_field_shape)  # create quad_field field

        if self.nb_nodes_per_pixel > 1:
            warnings.warn('Interpolator operator does not work for multiple nodal points per pixel.')

        quad_field_fqnxyz.fill(0)
        f_size = nodal_field_fnxyz.shape[0]
        n_size = nodal_field_fnxyz.shape[1]  # To ensure that gradient field is empty/zero
        N_at_quad_points_qnijk = np.zeros(
            [nb_quad_points_per_pixel, n_size, *self.domain_dimension * (self.domain_dimension,)])
        for quad_point_idx in range(nb_quad_points_per_pixel):
            quad_point_coords = quad_points_coords_dq[:, quad_point_idx]
            # iteration over all voxel corners
            for pixel_node in np.ndindex(*np.ones([self.domain_dimension], dtype=int) * 2):
                N_at_quad_points_qnijk[(quad_point_idx, 0, *pixel_node)] = self.N_basis_interpolator_array[pixel_node](
                    *quad_point_coords)

        for pixel_node in np.ndindex(
                *np.ones([self.domain_dimension], dtype=int) * 2):  # iteration over all voxel corners
            pixel_node = np.asarray(pixel_node)
            if self.domain_dimension == 2:
                # quad_field_fqnxyz += np.einsum('qn,fnxy->fqnxy', N_at_quad_points_qnijk[(..., *pixel_node)],
                #                                np.roll(nodal_field_fnxyz, -1 * pixel_node, axis=(2, 3)))

                quad_field_fqnxyz += np.einsum('qn,fnxy->fqnxy', N_at_quad_points_qnijk[(..., *pixel_node)],
                                               self.roll(self.fft, nodal_field_fnxyz, -1 * pixel_node, axis=(0, 1)))

            elif self.domain_dimension == 3:  # TODO 3D interpolation is not tested
                # quad_field_fqnxyz += np.einsum('qn,fnxyz->fqnxyz', N_at_quad_points_qnijk[(..., *pixel_node)],
                #                                np.roll(nodal_field_fnxyz, -1 * pixel_node, axis=(2, 3, 4)))

                quad_field_fqnxyz += np.einsum('qn,fnxyz->fqnxyz', N_at_quad_points_qnijk[(..., *pixel_node)],
                                               self.roll(self.fft, nodal_field_fnxyz, -1 * pixel_node,
                                                         axis=(0, 1, 2)))

        return quad_field_fqnxyz, N_at_quad_points_qnijk

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

        # homogenized_stress = np.sum(stress, axis=tuple(range(2, stress.shape.__len__()))) / self.cell.domain_volume
        # print('rank' f'{MPI.COMM_WORLD.rank:6} homogenized_stress =' f'{homogenized_stress}')
        # total_sum = comm.reduce(local_data, op=MPI.SUM, root=0)

        Reductor_numpi = Reduction(MPI.COMM_WORLD)
        homogenized_stress = Reductor_numpi.sum(stress, axis=tuple(range(-self.domain_dimension - 1, 0)))  #
        # print('rank' f'{MPI.COMM_WORLD.rank:6} homogenized_stress =' f'{homogenized_stress}')
        return homogenized_stress / self.cell.domain_volume

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

    def get_preconditioner_DELETE(self, reference_material_data_field_ijklqxyz,
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
        preconditioner_diagonals_fnfnqks = np.fft.fftn(preconditioner_diagonals_fnfnxyz, [*self.nb_of_pixels])
        # compute inverse of diagonals
        for pixel_index in np.ndindex(unit_impulse_fnxyz.shape[2:]):  # TODO find the woy to avoid loops
            if pixel_index == (
                    0,) * self.domain_dimension:  # avoid  inversion of zeros # TODO find better solution for setting this to 0
                continue
            # pick local matrix with size [f*n,f*n]
            local_matrix = np.reshape(preconditioner_diagonals_fnfnqks[(..., *pixel_index)],
                                      (np.prod(unit_impulse_fnxyz.shape[0:2]),
                                       np.prod(unit_impulse_fnxyz.shape[0:2])))
            # compute inversion
            local_matrix = np.linalg.inv(local_matrix)
            # rearrange local inverse matrix into global field
            preconditioner_diagonals_fnfnqks[(..., *pixel_index)] = np.reshape(local_matrix, (
                    unit_impulse_fnxyz.shape[0:2] + unit_impulse_fnxyz.shape[0:2]))

        return preconditioner_diagonals_fnfnqks  # return diagonals of preconditioner in Fourier space

    def get_preconditioner_NEW(self, reference_material_data_field_ijklqxyz,
                               formulation=None):
        # return diagonals of preconditioned matrix in Fourier space
        # unit_impulse [f,n,x,y,z]
        # for every type of degree of freedom DOF, there is one diagonal of preconditioner matrix
        # diagonals_in_Fourier_space [f,n,f,n][0,0,0]  # all DOFs in first pixel

        unit_impulse_fnxyz = self.get_unknown_size_field()
        # print()

        # print('unit_impulse_fnxyz.shape = {}'.format(unit_impulse_fnxyz.shape))

        preconditioner_diagonals_fnfnxyz = np.zeros(unit_impulse_fnxyz.shape[:2] + unit_impulse_fnxyz.shape)
        # print('preconditioner_diagonals_fnfnxyz.shape = {}'.format(preconditioner_diagonals_fnfnxyz.shape))
        # construct unit impulses responses for all type of DOFs
        # print('rank' f'{MPI.COMM_WORLD.rank:6} self.fft.ifftfreq[0]=' f'{self.fft.icoords[0]}')
        # print('rank' f'{MPI.COMM_WORLD.rank:6}  self.fft.ifftfreq[1]=' f'{self.fft.icoords[1]}')

        # print(            'rank' f'{MPI.COMM_WORLD.rank:6} get_preconditioner_NEW zero node=' f'{np.any(np.all(self.fft.icoords == 0, axis=0))}')
        # check if I have zero frequency on the core

        # TODO self.fft.icoords == 0, axis = 0
        # loop over all types of degree of freedom [f,n]
        for impulse_position in np.ndindex(unit_impulse_fnxyz.shape[0:2]):
            # print('rank' f'{MPI.COMM_WORLD.rank:6} get_preconditioner_NEW  impulse_position=' f'{impulse_position}')

            unit_impulse_fnxyz.fill(0)  # empty the unit impulse vector
            if np.any(np.all(self.fft.icoords == 0, axis=0)):
                # print(                   'rank' f'{MPI.COMM_WORLD.rank:6} get_preconditioner_NEW zero node =' f'{np.any(np.all(self.fft.icoords == 0, axis=0))}')
                # set 1 --- the unit impulse --- to a proper positions
                unit_impulse_fnxyz[impulse_position + (0,) * (unit_impulse_fnxyz.ndim - 2)] = 1
            # print('rank' f'{MPI.COMM_WORLD.rank:6} get_preconditioner_NEW unit_impulse_fnxyz=' f'{unit_impulse_fnxyz}')

            preconditioner_diagonals_fnfnxyz[impulse_position] = self.apply_system_matrix(
                material_data_field=reference_material_data_field_ijklqxyz,
                displacement_field=unit_impulse_fnxyz,
                formulation=formulation)

        # MPI.COMM_WORLD.Barrier()  # Barrier so header is printed first

        # print(preconditioner_diagonals_fnfnxyz.reshape([2, 2, *preconditioner_diagonals_fnfnxyz.shape[-2:]]))
        # print('rank' f'{MPI.COMM_WORLD.rank:6} {MPI.COMM_WORLD.size:6}  ')
        # construct diagonals from unit impulses responses using FFT
        preconditioner_diagonals_fnfnqks_NEW = self.fft.fft(preconditioner_diagonals_fnfnxyz)
        if self.cell.problem_type == 'conductivity':  # TODO[LaRs muFFT] FIX THIS
            preconditioner_diagonals_fnfnqks_NEW = np.expand_dims(preconditioner_diagonals_fnfnqks_NEW,
                                                                  axis=(0, 1, 2, 3))

        # THE SIZE OF   DIAGONAL IS [nb_unit_dofs,nb_unit_dofs,nb_unit_dofs,nb_unit_dofs, xyz]
        # compute inverse of diagonals
        # zero_indices = np.all(array == 0, axis=0)
        # print('rank' f'{MPI.COMM_WORLD.rank:6} preconditioner_diagonals_fnfnqks_NEW=' f'{preconditioner_diagonals_fnfnqks_NEW}')

        for pixel_index in np.ndindex(self.fft.ifftfreq[0].shape):  # TODO find the woy to avoid loops

            if np.all(self.fft.ifftfreq[(..., *pixel_index)] == (
                    0,) * self.domain_dimension):  # avoid  inversion of zeros # TODO find better solution for setting this to 0
                continue

            local_matrix_ijkl_NEW = np.reshape(preconditioner_diagonals_fnfnqks_NEW[(..., *pixel_index)],
                                               (np.prod(unit_impulse_fnxyz.shape[0:2]),
                                                np.prod(unit_impulse_fnxyz.shape[0:2])))
            # print('rank' f'{MPI.COMM_WORLD.rank:6}  Local matrix of preconditioner NEW ')
            # print(pixel_index)
            # print(local_matrix_ijkl_NEW)
            # compute inversion
            local_matrix_ijkl = np.linalg.inv(local_matrix_ijkl_NEW)

            # rearrange local inverse matrix into global field
            preconditioner_diagonals_fnfnqks_NEW[(..., *pixel_index)] = np.reshape(local_matrix_ijkl, (
                    unit_impulse_fnxyz.shape[0:2] + unit_impulse_fnxyz.shape[0:2]))

        return preconditioner_diagonals_fnfnqks_NEW

    def get_preconditioner_Jacoby(self, material_data_field_ijklqxyz,
                                  formulation=None):
        # return diagonals of system matrix
        # unit_impulse [f,n,x,y,z]
        # for every type of degree of freedom DOF, there is one diagonal of preconditioner matrix
        # diagonals_in_Fourier_space [f,n,f,n][0,0,0]  # all DOFs in first pixel

        # print('unit_impulse_fnxyz.shape = {}'.format(unit_impulse_fnxyz.shape))
        system_matrix = self.get_system_matrix(material_data_field=material_data_field_ijklqxyz)
        K_diag = np.diag(system_matrix).reshape(self.unknown_size)
        K_diag_inv_sym = K_diag ** (-1 / 2)
        return K_diag_inv_sym

    def get_preconditioner_Jacoby_fast(self, material_data_field_ijklqxyz,
                                       gradient_of_u=None,
                                       formulation=None,
                                       prec_type=None):
        # return diagonals of system matrix
        # unit_impulse [f,n,x,y,z]
        # for every type of degree of freedom DOF, there is one diagonal of preconditioner matrix
        # diagonals_in_Fourier_space [f,n,f,n][0,0,0]  # all DOFs in first pixel
        if gradient_of_u is None:  # if gradient_of_u is not specified, determine the size
            gradient_of_u = np.zeros(self.gradient_size)  # create gradient field

        material_data_field_ijklqxyz = self.apply_quadrature_weights(material_data=material_data_field_ijklqxyz)

        if self.nb_nodes_per_pixel > 1:
            warnings.warn('get_preconditioner_Jacoby_fast is not tested for multiple nodal points per pixel.')

        gradient_of_u.fill(0)  # To ensure that gradient field is empty/zero
        # gradient_of_u_selfroll = np.copy(gradient_of_u)
        # grad_at_points=np.sum(self.B_grad_at_pixel_dqnijk[:])
        diagonal_fnxyz = self.get_unknown_size_field()
        # !/usr/bin/env python3
        if self.cell.problem_type == 'conductivity':

            for pixel_node in np.ndindex(
                    *np.ones([self.domain_dimension], dtype=int) * 2):  # iteration over all voxel corners
                # print('rank' f'{MPI.COMM_WORLD.rank:6} apply_gradient_operator: loop over pixel node =' f'{pixel_node}')

                pixel_node = np.asarray(pixel_node)
                for d in np.ndindex(*self.cell.unknown_shape):
                    d = np.asarray(d)
                    if self.domain_dimension == 2:
                        gradient_of_u_fqnxy = np.einsum('fdqxy,dqn->fqnxy',
                                                        material_data_field_ijklqxyz,
                                                        self.B_grad_at_pixel_dqnijk[(..., *pixel_node)])
                        div_fnxyz_pixel_node = np.einsum('dqn,fdqnxy->nxy',
                                                         self.B_grad_at_pixel_dqnijk[(..., *pixel_node)],
                                                         gradient_of_u_fqnxy)
                        diagonal_fnxyz += self.roll(self.fft, div_fnxyz_pixel_node, 1 * pixel_node, axis=(0, 1))

                    elif self.domain_dimension == 3:
                        gradient_of_u_fqnxy = np.einsum('fdqxyz,dqn->fqnxyz',
                                                        material_data_field_ijklqxyz,
                                                        self.B_grad_at_pixel_dqnijk[(..., *pixel_node)])
                        div_fnxyz_pixel_node = np.einsum('dqn,fdqnxyz->nxyz',
                                                         self.B_grad_at_pixel_dqnijk[(..., *pixel_node)],
                                                         gradient_of_u_fqnxy)
                        diagonal_fnxyz += self.roll(self.fft, div_fnxyz_pixel_node, 1 * pixel_node, axis=(0, 1, 2))
                        print('Jacoby preconditioner  is not tested in 3D ')

        elif self.cell.problem_type == 'elasticity':
            if prec_type == 'full':
                diagonal_matrices_fnfnxyz = np.zeros(diagonal_fnxyz.shape[:2] + diagonal_fnxyz.shape)
            shape_function_gradients_fdnijk = np.zeros([*self.cell.unknown_shape, *self.B_grad_at_pixel_dqnijk.shape])
            for f in np.arange(0, self.cell.unknown_shape):
                shape_function_gradients_fdnijk[f] = np.copy(self.B_grad_at_pixel_dqnijk)
            for pixel_node in np.ndindex(
                    *np.ones([self.domain_dimension], dtype=int) * 2):  # iteration over all voxel corners
                # print('rank' f'{MPI.COMM_WORLD.rank:6} apply_gradient_operator: loop over pixel node =' f'{pixel_node}')

                pixel_node = np.asarray(pixel_node)
                for d in np.ndindex(*self.cell.unknown_shape):
                    d = np.asarray(d)
                    if self.domain_dimension == 2:

                        gradient_of_u_ijqnxy = np.einsum('ijklqxy,lkqn->ijqnxy',
                                                         material_data_field_ijklqxyz[:, :, :, d, ...],
                                                         shape_function_gradients_fdnijk[(d, ..., *pixel_node)])

                        div_at_quad_fnxy = np.einsum('dqn,fdqnxy->fnxy',
                                                     self.B_grad_at_pixel_dqnijk[(..., *pixel_node)],
                                                     gradient_of_u_ijqnxy)

                        diagonal_fnxyz[d, ...] += self.roll(self.fft, div_at_quad_fnxy[d], 1 * pixel_node, axis=(0, 1))
                        if prec_type == 'full':
                            diagonal_matrices_fnfnxyz[d, 0, ...] += self.roll(self.fft, div_at_quad_fnxy,
                                                                              1 * pixel_node,
                                                                              axis=(0, 1))
                    elif self.domain_dimension == 3:
                        gradient_of_u_ijqnxy = np.einsum('ijklqxyz,lkqn->ijqnxyz',
                                                         material_data_field_ijklqxyz[:, :, :, d, ...],
                                                         shape_function_gradients_fdnijk[(d, ..., *pixel_node)])

                        div_at_quad_fnxy = np.einsum('dqn,fdqnxyz->fnxyz',
                                                     self.B_grad_at_pixel_dqnijk[(..., *pixel_node)],
                                                     gradient_of_u_ijqnxy)

                        diagonal_fnxyz[d, ...] += self.roll(self.fft, div_at_quad_fnxy[d], 1 * pixel_node,
                                                            axis=(0, 1, 2))

                        print('Jacoby preconditioner  is not tested in 3D ')
        if prec_type == 'full':
            # compute inverse of diagonals
            for pixel_index in np.ndindex(self.fft.ifftfreq[0].shape):
                if np.all(self.fft.ifftfreq[(..., *pixel_index)] == (
                        0,) * self.domain_dimension):  # avoid  inversion of zeros
                    continue
                # pick local matrix with size [f*n,f*n]
                local_matrix_ij = np.reshape(diagonal_matrices_fnfnxyz[(..., *pixel_index)],
                                             (np.prod(diagonal_matrices_fnfnxyz.shape[0:2]),
                                              np.prod(diagonal_matrices_fnfnxyz.shape[0:2])))
                # compute inversion
                local_matrix_ij = np.linalg.inv(local_matrix_ij)
                local_matrix_ij_ = sc.linalg.fractional_matrix_power(local_matrix_ij, 1 / 2)
                # rearrange local inverse matrix into global field
                diagonal_matrices_fnfnxyz[(..., *pixel_index)] = np.reshape(local_matrix_ij_, (
                        diagonal_matrices_fnfnxyz.shape[0:2] + diagonal_matrices_fnfnxyz.shape[0:2]))
                return diagonal_matrices_fnfnxyz
        # return self.apply_quadrature_weights_elasticity(material_data)
        # K_diag_inv_sym = K_diag ** (-1 / 2)
        diagonal_fnxyz[diagonal_fnxyz < 1e-15] = 0
        diagonal_fnxyz[diagonal_fnxyz != 0] = diagonal_fnxyz[diagonal_fnxyz != 0] ** (-1 / 2)
        #diagonal_fnxyz ** (-1 / 2)
        return diagonal_fnxyz

    def apply_preconditioner_DELETE(self, preconditioner_Fourier_fnfnqks, nodal_field_fnxyz):
        # apply preconditioner using FFT
        # nodal_field_fnxyz [f,n,x,y,z]
        # preconditioner_Fourier_fnfnxyz [f,n,f,n,x,y,z] # TODO find better indexing notation

        # FFTn of the input field
        nodal_field_fnxyz = np.fft.fftn(nodal_field_fnxyz, [*self.nb_of_pixels])
        # print('rank' f'{MPI.COMM_WORLD.rank:6} apply_preconditioner:nodal_field_fnxyz=' f'{nodal_field_fnxyz}')

        # multiplication with a diagonals of preconditioner
        nodal_field_fnxyz = np.einsum('abcd...,cd...->ab...', preconditioner_Fourier_fnfnqks, nodal_field_fnxyz)
        # print('rank' f'{MPI.COMM_WORLD.rank:6} einsum:nodal_field_fnxyz=' f'{nodal_field_fnxyz}')

        # iFFTn
        nodal_field_fnxyz = np.real(np.fft.ifftn(nodal_field_fnxyz, [*self.nb_of_pixels]))

        # print('rank' f'{MPI.COMM_WORLD.rank:6} apply_preconditioner:nodal_field_fnxyz=' f'{nodal_field_fnxyz}')
        return nodal_field_fnxyz

    def apply_preconditioner_NEW(self, preconditioner_Fourier_fnfnqks, nodal_field_fnxyz):
        # apply preconditioner using FFT
        # nodal_field_fnxyz [f,n,x,y,z]
        # preconditioner_Fourier_fnfnxyz [f,n,f,n,x,y,z] # TODO find better indexing notation

        # FFTn of the input field
        # nodal_field_fnxyz_frq = self.fft.fft(nodal_field_fnxyz)
        # Compute Fourier transform
        # ffield_fnqks = self.fft.fourier_space_field('force_field',
        #                                     self.unknown_size[:-4])  # TODO [this is fixed for one node per pixel !!]
        # self.fft.fft(nodal_field_fnxyz, ffield_fnqks)
        # FFTn of input array
        ffield_fnqks = self.fft.fft(nodal_field_fnxyz)
        if self.cell.problem_type == 'conductivity':  # TODO[LaRs muFFT] FIX THIS
            ffield_fnqks = np.expand_dims(ffield_fnqks,
                                          axis=(0, 1))

        # print('rank' f'{MPI.COMM_WORLD.rank:6} apply_preconditioner_NEW:nodal_field_fnxyz=' f'{ffield_fnqks}')

        # multiplication with a diagonals of preconditioner
        ffield_fnqks = np.einsum('abcd...,cd...->ab...', preconditioner_Fourier_fnfnqks, ffield_fnqks)
        # print('rank' f'{MPI.COMM_WORLD.rank:6} apply_preconditioner_NEW:einsum=' f'{ffield_fnqks}')

        # normalization
        ffield_fnqks *= self.fft.normalisation
        # iFFTn
        nodal_field_fnxyz = self.fft.ifft(ffield_fnqks)
        # self.fft.ifft(ffield_fnqks,nodal_field_fnxyz)
        if self.cell.problem_type == 'conductivity':  # TODO[LaRs muFFT] FIX THIS
            nodal_field_fnxyz = np.expand_dims(nodal_field_fnxyz,
                                               axis=(0, 1))
        # print('rank' f'{MPI.COMM_WORLD.rank:6} apply_preconditioner_NEW:nodal_field_fnxyz=' f'{nodal_field_fnxyz}')

        return nodal_field_fnxyz

    def apply_system_matrix(self, material_data_field, displacement_field, formulation=None):
        # print('rank' f'{MPI.COMM_WORLD.rank:6} apply_system_matrix:displacement_field=')  # f'{displacement_field}')

        if formulation == 'small_strain':
            # print('rank' f'{MPI.COMM_WORLD.rank:6} apply_system_matrix:formulation=' f'{formulation}')

            strain = self.apply_gradient_operator_symmetrized(displacement_field)

        else:
            # print('rank' f'{MPI.COMM_WORLD.rank:6} apply_system_matrix:formulation=' f'{formulation}')
            # print('rank' f'{MPI.COMM_WORLD.rank:6} apply_system_matrix:displacement_field=' f'{displacement_field}')
            strain = self.apply_gradient_operator(displacement_field)

        material_data_field = self.apply_quadrature_weights(material_data_field)
        # print('rank' f'{MPI.COMM_WORLD.rank:6} apply_system_matrix:material_data_field=')  # f'{material_data_field}')

        stress = self.apply_material_data(material_data_field, strain)
        # print('rank' f'{MPI.COMM_WORLD.rank:6} apply_system_matrix:stress=')  # f'{stress}')

        force = self.apply_gradient_transposed_operator(stress)
        # print('rank' f'{MPI.COMM_WORLD.rank:6} apply_system_matrix:force=')  # f'{force}')

        return force

    def integrate_over_cell(self, stress_field):
        # compute integral of stress field over the domain: int sigma d Omega = sum x_q *w_q

        stress_field = np.einsum('ijq...,q->ijq...', stress_field, self.quadrature_weights)
        #
        # integral = np.einsum('fdqxy...->fd', stress_field)
        Reductor_numpi = Reduction(MPI.COMM_WORLD)
        # TODO change this to muGRID
        integral = Reductor_numpi.sum(stress_field, axis=tuple(range(-self.domain_dimension - 1, 0)))  #

        return integral

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

    def integrate_field_OLD(self, field_fnxyz,
                            nb_quad_points_per_pixel=None):
        # for scalar field only - evaluate in quadrature points
        if field_fnxyz.shape[0] != 1:
            raise ValueError(
                'integrate_field is implemented only for scalar field {}'.format(field_fnxyz.shape[0]))
        if nb_quad_points_per_pixel is None:
            nb_quad_points_per_pixel = self.nb_quad_points_per_pixel

        quad_points_coord, quad_points_weights = get_gauss_points_and_weights(element_type=self.element_type,
                                                                              nb_quad_points_per_pixel=nb_quad_points_per_pixel)
        Jacobian_matrix = np.diag(self.pixel_size)
        Jacobian_det = np.linalg.det(
            Jacobian_matrix)  # this is product of diagonal term of Jacoby transformation matrix
        quad_points_weights = quad_points_weights * Jacobian_det
        # Evaluate field on the quadrature points
        quad_field_fqnxyz = self.evaluate_field_at_quad_points(
            nodal_field_fnxyz=field_fnxyz,
            quad_field_fqnxyz=None,
            quad_points_coords_dq=quad_points_coord)[0]
        # Multiply with quadrature weights
        quad_field_fqnxyz = np.einsum('fq...,q->fq...', quad_field_fqnxyz, quad_points_weights)

        Reductor_numpi = Reduction(MPI.COMM_WORLD)
        integral_fq = self.mpi_reduction.sum(quad_field_fqnxyz, axis=tuple(range(-self.domain_dimension - 1, 0)))  #
        # TODO: what kidn of sum I need here?
        return np.sum(quad_field_fqnxyz)


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

    integral = np.einsum('fdqxy...->fd...', stress_field)

    return integral


def integrate_flux_field(flux_field, quadrature_weights):
    stress_field = np.einsum('ijq...,q->ijq...', flux_field, quadrature_weights)

    integral = np.einsum('fdqxy...->fd', stress_field)

    return integral


def get_gauss_points_and_weights(element_type, nb_quad_points_per_pixel):
    if element_type != 'linear_triangles':
        raise ValueError('Quadrature weights for Element_type {} is not implemented'.format(element_type))

    if element_type == 'linear_triangles':
        if nb_quad_points_per_pixel == 2:
            quad_points_coord = np.zeros(
                [2, nb_quad_points_per_pixel])
            quad_points_coord[:, 0] = [1 / 3, 1 / 3]
            quad_points_coord[:, 1] = [2 / 3, 2 / 3]
            quad_points_weights = np.zeros(
                [nb_quad_points_per_pixel])
            quad_points_weights[0:2] = 1 / 2

        if nb_quad_points_per_pixel == 6:
            quad_points_coord = np.zeros(
                [2, nb_quad_points_per_pixel])
            quad_points_coord[:, 0] = [1 / 6, 1 / 6]
            quad_points_coord[:, 1] = [2 / 3, 1 / 6]
            quad_points_coord[:, 2] = [1 / 6, 2 / 3]
            quad_points_coord[:, 3] = [1 - 1 / 6, 1 - 1 / 6]
            quad_points_coord[:, 4] = [1 - 2 / 3, 1 - 1 / 6]
            quad_points_coord[:, 5] = [1 - 1 / 6, 1 - 2 / 3]
            quad_points_weights = np.zeros(
                [nb_quad_points_per_pixel])
            quad_points_weights[0:6] = 1 / 6

        elif nb_quad_points_per_pixel == 8:
            # 8 nodes
            quad_points_coord = np.zeros(
                [2, nb_quad_points_per_pixel])
            quad_points_coord[:, 0] = [0.280019915499074, 0.644948974278318]
            quad_points_coord[:, 1] = [0.666390246014701, 0.155051025721682]
            quad_points_coord[:, 2] = [0.075031110222608, 0.644948974278318]
            quad_points_coord[:, 3] = [0.178558728263616, 0.155051025721682]
            quad_points_coord[:, 4] = [0.355051025721682, 0.924968889777392]
            quad_points_coord[:, 5] = [0.844948974278318, 0.821441271736384]
            quad_points_coord[:, 6] = [0.355051025721682, 0.719980084500926]
            quad_points_coord[:, 7] = [0.844948974278318, 0.333609753985299]

            quad_points_weights = np.zeros(
                [nb_quad_points_per_pixel])
            quad_points_weights[0] = 0.090979309128011
            quad_points_weights[1] = 0.159020690871989
            quad_points_weights[2] = 0.090979309128011
            quad_points_weights[3] = 0.159020690871989
            quad_points_weights[4] = 0.090979309128011
            quad_points_weights[5] = 0.159020690871989
            quad_points_weights[6] = 0.090979309128011
            quad_points_weights[7] = 0.159020690871989

        elif nb_quad_points_per_pixel == 18:
            # 18 nodes
            quad_points_coord = np.zeros(
                [2, nb_quad_points_per_pixel])
            quad_points_coord[:, 0] = [0.188409405952072, 0.787659461760847]
            quad_points_coord[:, 1] = [0.523979067720101, 0.409466864440735]
            quad_points_coord[:, 2] = [0.808694385677670, 0.0885879595127039]
            quad_points_coord[:, 3] = [0.106170269119576, 0.787659461760847]
            quad_points_coord[:, 4] = [0.295266567779633, 0.409466864440735]
            quad_points_coord[:, 5] = [0.455706020243648, 0.0885879595127039]
            quad_points_coord[:, 6] = [0.0239311322870805, 0.787659461760847]
            quad_points_coord[:, 7] = [0.0665540678391645, 0.409466864440735]
            quad_points_coord[:, 8] = [0.102717654809626, 0.0885879595127039]

            quad_points_coord[:, 9] = [0.212340538239153, 0.976068867712919]
            quad_points_coord[:, 10] = [0.590533135559265, 0.933445932160836]
            quad_points_coord[:, 11] = [0.911412040487296, 0.897282345190374]
            quad_points_coord[:, 12] = [0.212340538239153, 0.893829730880424]
            quad_points_coord[:, 13] = [0.590533135559265, 0.704733432220367]
            quad_points_coord[:, 14] = [0.911412040487296, 0.544293979756352]
            quad_points_coord[:, 15] = [0.212340538239153, 0.811590594047928]
            quad_points_coord[:, 16] = [0.590533135559265, 0.476020932279899]
            quad_points_coord[:, 17] = [0.911412040487296, 0.191305614322330]

            quad_points_weights = np.zeros([nb_quad_points_per_pixel])
            quad_points_weights[0] = 0.0193963833059595
            quad_points_weights[1] = 0.0636780850998851
            quad_points_weights[2] = 0.0558144204830443
            quad_points_weights[3] = 0.0310342132895352
            quad_points_weights[4] = 0.101884936159816
            quad_points_weights[5] = 0.0893030727728709
            quad_points_weights[6] = 0.0193963833059595
            quad_points_weights[7] = 0.0636780850998851
            quad_points_weights[8] = 0.0558144204830443
            quad_points_weights[9] = 0.0193963833059595
            quad_points_weights[10] = 0.0636780850998851
            quad_points_weights[11] = 0.0558144204830443
            quad_points_weights[12] = 0.0310342132895352
            quad_points_weights[13] = 0.101884936159816
            quad_points_weights[14] = 0.0893030727728709
            quad_points_weights[15] = 0.0193963833059595
            quad_points_weights[16] = 0.0636780850998851
            quad_points_weights[17] = 0.0558144204830443

        return quad_points_coord, quad_points_weights
