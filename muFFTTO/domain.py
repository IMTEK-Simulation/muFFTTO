"""Periodic unit cell and its (FEM / Fourier) discretization.

This module is the core of muFFTTO. It defines

* :class:`PeriodicUnitCell` -- the physical problem: dimension, size of the
  periodic cell and the type of physics ('conductivity' = scalar unknown,
  'elasticity' = vector unknown), which fixes the tensor shapes of the
  unknown, its gradient and the material tangent.
* :class:`Discretization` -- a regular (pixel/voxel) grid on the cell,
  distributed over MPI ranks by a ``muGrid.FFTEngine``. It owns the
  element-level operators (shape-function gradients ``B``, interpolation
  ``N``, Hessian ``H``, Laplacian ``L``; filled in by
  :mod:`muFFTTO.discretization_library`) wrapped as muGrid
  ``GenericLinearOperator`` stencils, and implements the matrix-free
  building blocks of the FFT-accelerated FEM homogenization solver:

  - gradient ``B u`` and divergence ``B^T w sigma`` operators,
  - action of the system matrix ``K u = B^T w C B u`` (also on deformed
    grids and with an explicit, possibly nonlinear, constitutive law),
  - right-hand side ``f = -B^T w C E`` for a prescribed macroscopic
    gradient ``E``,
  - homogenized (volume averaged) stress/flux and energy,
  - Green (reference-material, Fourier-diagonal) and Jacobi preconditioners,
  - factory helpers that allocate correctly shaped muGrid fields.
* A few free functions for integrating quadrature-point fields and a table of
  Gauss quadrature rules for triangles.

Index / array-layout conventions
--------------------------------
Variable names carry their index layout as a suffix, e.g. ``u_inxyz``:

* ``i, j, k, l`` (also ``f, d``) -- tensor components (size ``d`` = spatial
  dimension, or 1 for a scalar unknown),
* ``n`` -- nodal sub-point within a pixel (``nb_nodes_per_pixel``),
* ``q`` -- quadrature sub-point within a pixel (``nb_quad_points_per_pixel``),
* ``x, y, z`` -- pixel (grid) indices in real space,
* ``q, k, s`` *after* the components in Fourier fields (e.g. ``..._fnfnqks``)
  -- wave-vector indices in Fourier space (not quadrature points!).

Fields are muGrid ``Field`` objects. ``field.s`` is the numpy view of the
locally owned part (without ghost layers), ``field.sg`` the view including
the ghost buffers. Before a stencil (convolution) operator reads a field,
its ghost layers must be filled by ``self.fft.communicate_ghosts(field)``
(MPI halo exchange + periodic wrap-around).

The global (MPI-reduced) sums are done with ``NuMPI.Tools.Reduction``.
The quadrature weights ``self.quadrature_weights`` are *physical* weights
(they already contain the pixel area/volume), so
``sum_{q, pixels} w_q f_q`` approximates ``\\int_\\Omega f dx``.
"""
import warnings

import numpy as np
import scipy as sc

from NuMPI.Tools import Reduction
from mpi4py import MPI

import muGrid
from muGrid import GenericLinearOperator  # ConvolutionOperator
from muGrid import Field

from muFFTTO import discretization_library, tensor_operations


class PeriodicUnitCell:
    """Physical description of a periodic unit cell (representative volume).

    Stores the geometry (dimension, size, volume) and the physics type, which
    determines the tensor shapes used by :class:`Discretization`:

    ============== ============== ============== ======================
    problem_type   unknown_shape  gradient_shape material_data_shape
    ============== ============== ============== ======================
    conductivity   [1]            [1, d]         [d, d]
    elasticity     [d]            [d, d]         [d, d, d, d]
    ============== ============== ============== ======================

    Attributes
    ----------
    name : str
    domain_dimension : int
        Spatial dimension ``d`` (= ``len(domain_size)``).
    domain_size : ndarray of float, shape (d,)
        Edge lengths of the (rectangular) cell.
    domain_volume : float
        ``prod(domain_size)``, used to turn integrals into volume averages.
    problem_type : str
    unknown_shape, gradient_shape, material_data_shape : ndarray of int
        Component shapes, see table above.
    displacement_shape, temperature_shape, scalar_shape : ndarray of int
        Convenience shapes ``[d]``, ``[1]``, ``[1]``.
    """

    def __init__(self, name='my_unit_cell', domain_size=None, problem_type='conductivity'):
        """Initialize a periodic unit cell.

        Parameters
        ----------
        name : str, optional
            Name identifier for the unit cell (default ``'my_unit_cell'``).
        domain_size : array-like of float, shape (d,)
            Physical size of the domain in each dimension. Its length defines
            the spatial dimension ``d``. Must be given (``None`` fails).
        problem_type : str, optional
            Type of physics problem: ``'conductivity'`` (scalar unknown,
            e.g. temperature) or ``'elasticity'`` (vector unknown,
            displacement). Default ``'conductivity'``.

        Raises
        ------
        ValueError
            If problem_type is not 'conductivity' or 'elasticity'
        """
        self.name = name
        self.domain_dimension = len(domain_size)
        self.domain_size = np.asarray(domain_size, dtype=float)
        self.domain_volume = np.prod(self.domain_size)

        self.problem_type = problem_type
        if not problem_type in ['conductivity', 'elasticity']:
            raise ValueError(
                'Unrecognised physical problem type {}. Choose from ' \
                ': conductivity, or elasticity'.format(problem_type))

        if problem_type == 'conductivity':
            # temperature is a single scalar
            self.unknown_shape = np.array([1], dtype=int)
            self.gradient_shape = np.array([1, self.domain_dimension],
                                           dtype=int)  # temp. gradient is a vector of d components
            self.material_data_shape = np.array([self.domain_dimension, self.domain_dimension],
                                                dtype=int)  # mat. data matrix a  of dxd components

        elif problem_type == 'elasticity':
            # displacement is a vector of d components
            self.unknown_shape = np.array([self.domain_dimension], dtype=int)
            self.gradient_shape = np.array([self.domain_dimension, self.domain_dimension],
                                           dtype=int)  # gradient  matrix of d x d components
            self.material_data_shape = np.array(
                [self.domain_dimension, self.domain_dimension, self.domain_dimension, self.domain_dimension],
                dtype=int)  # mat. data tensor  of dxdxdxd components

        self.displacement_shape = np.array([self.domain_dimension], dtype=int)
        self.temperature_shape = np.array([1], dtype=int)
        self.scalar_shape = np.array([1], dtype=int)


class Discretization:
    """Container for unit cell discretization information and FEM operators.

    Stores discretization parameters including grid dimensions, element types,
    quadrature points, and provides FEM operators for gradient and interpolation.

    The cell is split into a regular grid of ``nb_of_pixels_global`` pixels
    (voxels in 3D). Every pixel carries the same reference element
    (``element_type``) with ``nb_nodes_per_pixel`` nodes owned by the pixel
    (shared nodes of neighbouring pixels belong to the neighbour, periodicity
    closes the grid) and ``nb_quad_points_per_pixel`` quadrature points.
    Because all pixels are identical, global FEM operators are translation
    invariant stencils (convolutions), which are applied matrix-free through
    muGrid ``GenericLinearOperator`` objects and are diagonalised (per
    wave vector) by the FFT -- the basis of the Green preconditioner.

    Discrete linear homogenization problem (small strain / conductivity)::

        find u (periodic fluctuation):   B^T W C B u = -B^T W C E
        K u = f,   K = B^T W C B,   f = -B^T W C E

    with ``B`` the gradient operator (nodes -> quadrature points), ``W`` the
    diagonal matrix of quadrature weights, ``C`` the material tangent at the
    quadrature points and ``E`` the prescribed macroscopic gradient.

    Main attributes (set in ``__init__`` or by
    :func:`muFFTTO.discretization_library.get_shape_function_gradient_matrix`)
    ----------------------------------------------------------------------
    cell : PeriodicUnitCell
    fft : muGrid.FFTEngine
        Parallel FFT engine; owns the real/Fourier field collections and the
        MPI domain decomposition (with one ghost layer on each side).
    nb_of_pixels_global : tuple of int
        Global grid size.
    nb_of_pixels : ndarray of int
        Grid size of the local MPI subdomain.
    pixel_size : ndarray of float, shape (d,)
    field_collection, ffield_collection
        muGrid real- and Fourier-space field collections with the sub-point
        types ``'quad_points'`` and ``'nodal_points'`` registered.
    quadrature_weights : ndarray, shape (q,)
        Physical quadrature weights (include the pixel measure).
    B_grad_at_pixel_dqnijk : ndarray
        Shape function gradients ``dN_n/dx_d`` at quadrature point ``q`` for
        node ``n`` of the pixel with offset ``(i, j, k)`` -- the gradient
        stencil.
    N_at_quad_points_dqnijk, H_hess_at_pixel_deqnijk, L_laplace_at_pixel_eqnijk
        Interpolation, Hessian and Laplacian stencils (same layout idea).
    gradient_op, interpolation_op, hessian_op, laplacian
        muGrid ``GenericLinearOperator`` wrappers of the stencils above.
    mpi_reduction : NuMPI.Tools.Reduction
        Helper for global (all-rank) sums/min/max.
    """

    def __init__(self, cell,
                 nb_of_pixels_global=None,
                 discretization_type='finite_element',
                 element_type='linear_triangles',
                 communicator=muGrid.Communicator(MPI.COMM_WORLD)):
        """Initialize discretization.

        Parameters
        ----------
        cell : PeriodicUnitCell
            Unit cell definition
        nb_of_pixels_global : tuple of int
            Number of pixels (elements) in each dimension of the global grid.
        discretization_type : str, optional
            'finite_element' (default) or 'Fourier'
        element_type : str, optional
            Element family, e.g. 'linear_triangles' (default). Passed to
            :func:`muFFTTO.discretization_library.get_shape_function_gradient_matrix`,
            which defines quadrature, Jacobian and the operator stencils.
        communicator : muGrid.Communicator, optional
            MPI communicator for parallel computation. Note: the default is
            evaluated once at import time (``MPI.COMM_WORLD``).

        Raises
        ------
        ValueError
            If discretization_type is not 'finite_element' or 'Fourier'

        Notes
        -----
        Side effects: creates the ``muGrid.FFTEngine`` (collective MPI call)
        and the field collections; all fields of this discretization are
        later allocated from these collections by name (requesting a field
        with an already existing name returns the existing field).
        """

        self.cell = cell
        self.domain_dimension = cell.domain_dimension
        self.domain_size = cell.domain_size
        # total number of pixels/voxels, without periodic nodes
        self.nb_of_pixels_global = tuple(map(int, nb_of_pixels_global))
        # pixel properties
        self.pixel_size = self.domain_size / self.nb_of_pixels_global
        self.nb_nodes_per_pixel = None
        self.nodal_points_coordinates = None
        self.nb_vertices_per_pixel = 2 ** self.domain_dimension

        # fills element data (quadrature, Jacobian, B/N/H/L stencils, nb_nodes_per_pixel, ...)
        # into `self`; needed already here because nb_nodes_per_pixel etc. are used below
        self.get_discretization_info(element_type)

        # One ghost layer on each side is enough for stencils that reach only the
        # nearest neighbouring pixel (e.g. linear elements).
        # number of ghost buffers -> # TODO[Martin]: have to be changed base on the stencil
        left_ghosts = [1, ] * self.domain_dimension
        right_ghosts = [1, ] * self.domain_dimension
        self.fft = muGrid.FFTEngine(nb_domain_grid_pts=nb_of_pixels_global,
                                    communicator=communicator,
                                    nb_ghosts_left=left_ghosts,
                                    nb_ghosts_right=right_ghosts,
                                    # nb_sub_pts=self.nb_nodes_per_pixel
                                    )
        self.communicator = communicator
        self.mpi_reduction = Reduction(MPI.COMM_WORLD)
        # Validate the symmetries of material data where it enters a solve
        # (see check_material_symmetry); can be switched off for speed.
        self.check_material_symmetry = True

        # number of pixels/voxels of a subdomain for MPI
        self.nb_of_pixels = np.asarray(self.fft.nb_subdomain_grid_pts,
                                       dtype=np.intp)
        if MPI.COMM_WORLD.size > 1:
            # adjust the number of points
            # NOTE: currently a no-op (the "- 2" correction is commented out)
            self.nb_of_pixels[-1] = self.nb_of_pixels[-1]  # - 2  # TODO this is for buffer of size 1x1
            # adjust subdomain location to not take into account buffers
            sub_dom_locations = np.asarray(self.fft.subdomain_locations)
            sub_dom_locations += left_ghosts
            self.subdomain_locations_no_buffers = tuple(sub_dom_locations)
            # (second shift has no effect on stored attributes: the tuple above is already copied)
            sub_dom_locations += left_ghosts
            # compute max nb_max_subdomain_grid_pts for save npy TODO: THIS IS QUICK FIX considering pencil decompositon
            max_size_of_subdomain = self.mpi_reduction.max(np.asarray(self.fft.nb_subdomain_grid_pts[-1])) - 2
            self.nb_max_subdomain_grid_pts = np.asarray(self.fft.nb_subdomain_grid_pts)
            self.nb_max_subdomain_grid_pts[-1] = max_size_of_subdomain

        else:
            self.subdomain_locations_no_buffers = self.fft.subdomain_locations

        self.nb_of_pixels_with_buffers = np.asarray(self.fft.nb_subdomain_grid_pts,
                                                    dtype=np.intp)
        self.sub_domain_size = np.prod(self.nb_of_pixels)
        if not discretization_type in ['finite_element', 'Fourier']:
            raise ValueError(
                'Unrecognised discretization type {}. Choose from ' \
                ' : finite_element, finite_difference, or Fourier'.format(discretization_type))
        self.discretization_type = discretization_type  # only finite elements for now

        if discretization_type == 'finite_element':
            # finite element properties
            self.element_type = element_type
            self.nb_quad_points_per_pixel = None
            self.quadrature_weights = None
            self.quad_points_coord = None
            self.quad_points_coord_parametric = None

            self.get_discretization_info(element_type)

            # full (local) array shapes:
            #   unknown       [f, n, x, y, z]
            #   gradient      [f, d, q, x, y, z]
            #   material data [d, d, q, x, y, z]  or  [d, d, d, d, q, x, y, z]
            self.unknown_size = [*self.cell.unknown_shape, self.nb_nodes_per_pixel, *self.nb_of_pixels]
            self.gradient_size = [*self.cell.gradient_shape, self.nb_quad_points_per_pixel, *self.nb_of_pixels]
            self.material_data_size = [*self.cell.material_data_shape, self.nb_quad_points_per_pixel,
                                       *self.nb_of_pixels]

            # register the sub-point types: every pixel carries q quadrature points and n nodes
            self.field_collection = self.fft.real_space_collection
            self.ffield_collection = self.fft.fourier_space_collection
            self.field_collection.set_nb_sub_pts('quad_points', self.nb_quad_points_per_pixel)
            self.field_collection.set_nb_sub_pts('nodal_points', self.nb_nodes_per_pixel)
            self.ffield_collection.set_nb_sub_pts('quad_points', self.nb_quad_points_per_pixel)
            self.ffield_collection.set_nb_sub_pts('nodal_points', self.nb_nodes_per_pixel)
            point_of_origin = self.domain_dimension * [0, ]  # TODO This has to be a discretization stencil dependant

            # Wrap the element stencils as muGrid convolution operators. Each
            # operator maps a nodal field [c, n, x, y, z] to a quadrature field
            # [c, o, q, x, y, z] (apply) and back (transpose, with weights):
            #   gradient_op:      o = d (spatial derivative direction)
            #   hessian_op:       o = d*d (flattened pair (j,k))
            #   interpolation_op: o = 1
            #   laplacian:        o = 1
            # Missing stencils (not provided by the element) are skipped with a message.
            try:
                self.gradient_op = GenericLinearOperator(point_of_origin, self.B_grad_at_pixel_dqnijk)
            except:
                print(f'self.gradient_op does not exist ')
            try:
                # Hessian operator ---> due to muGrid set up, we can't have ij outpu shape. So I reshape the Hessian operator
                H_deqnijk = self.H_hess_at_pixel_deqnijk
                H_flat_Dqnijk = np.ascontiguousarray(
                    self.H_hess_at_pixel_deqnijk.reshape(self.domain_dimension * self.domain_dimension,
                                                         *H_deqnijk.shape[2:]))
                self.hessian_op = GenericLinearOperator(point_of_origin, H_flat_Dqnijk)
            except:
                print(f'self.hessian_op does not exist ')
            try:
                self.interpolation_op = GenericLinearOperator(point_of_origin, self.N_at_quad_points_dqnijk)
            except:
                print(f'self.interpolation_op does not exist ')

            try:
                self.laplacian = GenericLinearOperator(point_of_origin, self.L_laplace_at_pixel_eqnijk)
            except:
                print(f'self.interpolation_op does not exist ')

            # displacement              [f,n,x,y,z]
            # rhs                       [f,n,x,y,z]
            # macro_gradient_field    [f,d,q,x,y,z]
            # material_data_field     [d,d,q,x,y,z] - conductivity
            # material_data_field [d,d,d,d,q,x,y,z] - elasticity
            #  rhs=-Dt*A*E
        if discretization_type == 'Fourier':
            # NOTE: in the Fourier branch no field collections or muGrid operators are set up;
            # only the array sizes are defined.
            # finite element properties
            self.element_type = element_type
            self.nb_quad_points_per_pixel = None
            self.quadrature_weights = None
            self.quad_points_coord = None
            self.quad_points_coord_parametric = None

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
        """Calculate spatial coordinates of nodal points scaled to domain size.

        Returns
        -------
        nodal_points_coordinates_inxyz : muGrid Field
            Spatial coordinates of discretization nodes [dim, n, x, y, z]
            (local MPI subdomain only), stored in the field named
            ``"nodal_points_coordinates_inxyz"``.

        Notes
        -----
        The first node of every pixel is its lower-left(-front) corner. For
        sheared pixels (non-diagonal ``jacobian_of_pixel``) the coordinates
        are mapped with the column-normalised Jacobian, see below.
        For ``nb_nodes_per_pixel == 4`` the extra nodes are placed at half
        pixel offsets (mid-edge / centre nodes); for any other number of
        nodes only a warning is issued and the field is returned without
        being filled.
        """

        dim = self.domain_dimension
        # creates a field with coordinates of all nodal points
        nodal_points_coordinates_inxyz = self.field_collection.real_field(
            name="nodal_points_coordinates_inxyz",  # name of the field
            components=(dim,),  # shape of components
            sub_pt='nodal_points'  # sub-point type
        )
        # mugrid give coordinates from [0,1)**dim
        # I transform coordinates from [0,1)**dim to to physical element. Firt by appliing Jacobian of transformation
        # x= x*J^T
        # transformed_coordinates_ixyz = np.einsum('ij,jxy->ixy', self.jacobian_of_pixel, self.fft.coords)

        # fft.coords are fractional coordinates in [0,1) of the local grid points, shape [dim, x, y, z];
        # scale component-wise by the cell size -> physical coordinates of the pixel origins
        nodal_points_coordinates_ixyz = self.domain_size[tuple([slice(None)] + [np.newaxis] * dim)] * self.fft.coords
        # Coordinates above are already in physical units, so each column of the pixel Jacobian
        # is normalised by its own diagonal entry: that yields a unit diagonal regardless of the
        # element's reference domain ([0,1] for triangles, [-1,1] for quads) and expresses the
        # off-diagonal shear per unit physical length instead of per unit parametric length.
        adjusted_jacobian = self.jacobian_of_pixel / np.diag(self.jacobian_of_pixel)[np.newaxis, :]

        # x_j = sum_i J_adj[j, i] * x_i   (apply the adjusted Jacobian to every grid point)
        nodal_points_coordinates_ixyz = np.einsum('i...,ji->j...', nodal_points_coordinates_ixyz, adjusted_jacobian)
        if self.nb_nodes_per_pixel == 1:
            nodal_points_coordinates_inxyz.s[...] = np.expand_dims(nodal_points_coordinates_ixyz, axis=1)  # x, axis = 0
        elif self.nb_nodes_per_pixel == 4:
            half_pixel_size = self.pixel_size / 2
            # first node
            nodal_points_coordinates_inxyz.s[...] = np.expand_dims(nodal_points_coordinates_ixyz, axis=1)  # x, axis = 0

            if dim == 2:
                # second node
                nodal_points_coordinates_inxyz.s[0, 1, ...] += half_pixel_size[0]
                # third node
                nodal_points_coordinates_inxyz.s[1, 2, ...] += half_pixel_size[1]
                # fourth node
                nodal_points_coordinates_inxyz.s[0, 3, ...] += half_pixel_size[0]
                nodal_points_coordinates_inxyz.s[1, 3, ...] += half_pixel_size[1]
            if dim == 3:
                # second node
                nodal_points_coordinates_inxyz.s[0, 5, ...] += half_pixel_size[0]
                # third node
                nodal_points_coordinates_inxyz.s[1, 6, ...] += half_pixel_size[1]
                # fourth node
                nodal_points_coordinates_inxyz.s[0, 7, ...] += half_pixel_size[0]
                nodal_points_coordinates_inxyz.s[1, 7, ...] += half_pixel_size[1]
                # z direction add
                # second node
                nodal_points_coordinates_inxyz.s[2, 5, ...] += half_pixel_size[2]
                # third node
                nodal_points_coordinates_inxyz.s[2, 6, ...] += half_pixel_size[2]
                # fourth node
                nodal_points_coordinates_inxyz.s[2, 7, ...] += half_pixel_size[2]
                nodal_points_coordinates_inxyz.s[2, 7, ...] += half_pixel_size[2]

        else:
            warnings.warn(
                "get_nodal_points_coordinates does not support more than one nodal point"
            )

        return nodal_points_coordinates_inxyz

    def get_nodal_points_coordinates_with_periodic_nodes(self):
        """Calculate coordinates of nodal points including periodic boundary nodes.

        Returns
        -------
        nodal_points_coordinates_inxyz : ndarray
            Coordinates of nodal points including periodic boundary nodes [dim, n, Nx+1, Ny+1, (Nz+1)].

        Notes
        -----
        Works on the *global* grid (not MPI-distributed) and returns plain
        numpy data, useful for plotting. Coordinates are *normalised* to
        ``[0, 1]`` (not scaled by ``domain_size``) and only node ``n = 0`` is
        filled; further nodes per pixel remain zero.
        """
        # if self.nb_nodes_per_pixel != 1:
        #     raise ValueError(
        #         'get_nodal_points_coordinates does not support more than one nodal point')

        # create nd array  with proper shape including periodic nodes
        extended_number_of_nodes = (len(self.nb_of_pixels_global), self.nb_nodes_per_pixel) + tuple(
            x + 1 for x in self.nb_of_pixels_global)
        nodal_points_coordinates_inxyz = np.zeros(extended_number_of_nodes)
        # generate coordinates for each node
        grids = np.meshgrid(*[np.linspace(0, 1, n + 1) for n in self.nb_of_pixels_global], indexing='ij')
        for i, g in enumerate(grids):
            # fill in the coordinates
            nodal_points_coordinates_inxyz[i, 0] = g
            # for multiple nodes per pixel, we need to add a loop over the nodes

        return nodal_points_coordinates_inxyz

    # @property
    def get_quad_points_coordinates(self):
        """Calculate spatial coordinates of quadrature points in physical domain.

        Returns
        -------
        quad_points_coordinates_iqxyz : muGrid Field
            Spatial coordinates of quadrature points [dim, q, x, y, z].

        Notes
        -----
        For each quadrature point ``q`` a regular grid with spacing
        ``pixel_size`` is built, offset by ``self.quad_points_coord[:, q]``
        (the physical offset of the quadrature point within a pixel). The
        upper bound ``domain_size + 0.9*offset`` only guards ``np.arange``
        against floating-point overshoot. The grid is the *global* one, so
        this assumes a single MPI rank (shapes would not match the local
        field otherwise).
        """
        dim = self.domain_dimension
        # creates a field with coordinates of all quadrature points
        quad_points_coordinates_iqxyz = self.field_collection.real_field(
            name="quad_points_coordinates_iqxyz",  # name of the field
            components=(dim,),  # shape of components
            sub_pt='quad_points'  # sub-point type
        )

        for q in range(0, self.nb_quad_points_per_pixel):
            quad_points_coordinates_iqxyz.s[:, q] = np.meshgrid(
                *[np.arange(0 + self.quad_points_coord[d, q], self.domain_size[d] + 0.9 * self.quad_points_coord[d, q],
                            self.pixel_size[d])
                  for d in
                  range(0, self.domain_dimension)],
                indexing='ij')
        return quad_points_coordinates_iqxyz

    def roll(self, fft, u_inxyz, shift, axis):
        """Circular shift field along specified axes using FFT phase shift.

        Parameters
        ----------
        fft : muGrid.FFTEngine
            FFT engine for the discretization.
        u_inxyz : ndarray or muGrid Field
            Input field to be rolled.
        shift : array_like of int
            Shift distance along each specified axis.
        axis : tuple of int
            Axes along which the shift is performed.

        Returns
        -------
        return_field_inxyz : ndarray
            Rolled (shifted) field.

        Notes
        -----
        Uses the Fourier shift theorem: ``u(x - s)`` <-> ``exp(-2 pi i k.s) u_hat(k)``,
        i.e. equivalent to ``np.roll(u, shift, axis)`` for integer shifts but
        MPI-parallel. ``fft.fftfreq[a]`` holds the normalised frequencies
        (cycles per grid point) along axis ``a``. Uses temporary fields named
        ``'f_field_phase_roll_temp'`` / ``'field_phase_roll_temp'``; the
        returned array is a view of the latter (it is overwritten by the
        next call). ``fft.normalisation`` (= 1/N) undoes the unnormalised
        forward+inverse FFT pair.
        """
        # phase(k) = -2 pi sum_a s_a k_a
        phase = -2 * np.pi * sum(s * fft.fftfreq[a] for s, a in zip(shift, axis))
        f_field_inqrs = self.ffield_collection.complex_field(
            name='f_field_phase_roll_temp',  # name of the field
            components=(u_inxyz.shape[0],))
        fft.fft(u_inxyz, f_field_inqrs)
        f_field_inqrs.s[...] *= np.exp(1j * phase)
        return_field_inxyz = self.field_collection.real_field(
            name='field_phase_roll_temp',  # name of the field
            components=(u_inxyz.shape[0],))

        fft.ifft(f_field_inqrs, return_field_inxyz)

        return return_field_inxyz.s * fft.normalisation

    def apply_gradient_operator_mugrid(self, u_inxyz, grad_u_ijqxyz):
        """Compute gradient of field u using muGrid convolution operator.

        Parameters
        ----------
        u_inxyz : muGrid Field
            Nodal field [i, n, x, y, z].
        grad_u_ijqxyz : muGrid Field
            Output gradient field at quadrature points [i, j, q, x, y, z] (modified in-place).

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If ``u_inxyz`` is a plain ndarray (see Notes).

        Notes
        -----
        Computes ``(grad u)_{ij}(x_q) = sum_{n, offsets} u_i^n * dN^n/dx_j (x_q)``
        -- the discrete gradient ``B u``. The stencil
        ``B_grad_at_pixel_dqnijk`` gathers nodal values of the pixel itself
        and of its neighbours (offsets ``ijk``), hence the ghost layers of
        ``u_inxyz`` are refreshed first (MPI halo exchange; collective).
        """

        if self.nb_nodes_per_pixel > 1:
            warnings.warn('Gradient operator is not tested for multiple nodal points per pixel.')

        # if the input is ndArray, create muGrid field out of it
        if isinstance(u_inxyz, np.ndarray):
            raise TypeError("apply_gradient_operator_mugrid does not support ndarray")

        self.fft.communicate_ghosts(field=u_inxyz)
        self.gradient_op.apply(nodal_field=u_inxyz,
                               quadrature_point_field=grad_u_ijqxyz)

    def apply_gradient_operator_symmetrized_mugrid(self, u_inxyz, grad_u_ijqxyz):
        """Compute symmetrized gradient (strain) of vector field u.

        Parameters
        ----------
        u_inxyz : muGrid Field
            Nodal displacement field [i, n, x, y, z].
        grad_u_ijqxyz : muGrid Field
            Output symmetrized gradient field at quadrature points [i, j, q, x, y, z] (modified in-place).

        Returns
        -------
        None

        Notes
        -----
        ``eps_ij = (du_i/dx_j + du_j/dx_i) / 2`` (small-strain tensor).
        """
        # computes symmetrized gradient (small-strain)

        # 1. compute gradient
        self.apply_gradient_operator_mugrid(u_inxyz=u_inxyz,
                                            grad_u_ijqxyz=grad_u_ijqxyz)

        # 2. symmetrize it
        grad_u_ijqxyz.s[...] = (grad_u_ijqxyz.s + np.swapaxes(grad_u_ijqxyz.s, 0, 1)) / 2

    def apply_gradient_transposed_operator_mugrid(self,
                                                  gradient_field_ijqxyz,
                                                  div_u_fnxyz,
                                                  apply_weights=True):
        """Compute divergence (B^T operator) from flux or stress field at quadrature points.

        Parameters
        ----------
        gradient_field_ijqxyz : muGrid Field
            Stress or flux field at quadrature points [i, j, q, x, y, z].
        div_u_fnxyz : muGrid Field
            Output divergence field at nodal points [i, n, x, y, z] (modified in-place).
        apply_weights : bool, optional
            Whether to apply quadrature weights during integration (default is True).

        Returns
        -------
        None

        Notes
        -----
        Computes the discrete (weak) divergence
        ``f_i^n = sum_{q, pixels} w_q * dN^n/dx_j (x_q) * sigma_ij(x_q)``,
        i.e. ``B^T W sigma`` -- the internal force vector for a stress
        field ``sigma``. With ``apply_weights=False`` it is the pure
        transpose ``B^T sigma``. Contributions scattered by the stencil to
        neighbouring pixels are accumulated by muGrid; the ghost layers are
        refreshed before and after (collective MPI communication).
        """
        # if the input is ndArray, create muGrid field out of it

        if isinstance(gradient_field_ijqxyz, np.ndarray):
            raise TypeError("apply_gradient_transposed_operator_mugrid does not support ndarray")

        # if self.nb_nodes_per_pixel > 1:
        #     warnings.warn('Gradient operator is not tested for multiple nodal points per pixel.')
        # clear div array
        # get quadrature weights
        if apply_weights:
            weights = self.quadrature_weights
        else:
            weights = np.ones(self.quadrature_weights.shape)

        #   convolution operator
        self.fft.communicate_ghosts(field=gradient_field_ijqxyz)
        # apply B^transposed via the convolution operator
        self.gradient_op.transpose(quadrature_point_field=gradient_field_ijqxyz,
                                   nodal_field=div_u_fnxyz,
                                   weights=weights)

        self.fft.communicate_ghosts(field=div_u_fnxyz)

    def apply_hessian_operator_to_scalar_field_mugrid(self, u_inxyz, hess_u_ijkqxyz):
        """Compute Hessian (second derivatives) of scalar field u at quadrature points.

        Parameters
        ----------
        u_inxyz : muGrid Field
            Nodal scalar field [1, n, x, y, z].
        hess_u_ijkqxyz : muGrid Field
            Output Hessian field at quadrature points [1, j, k, q, x, y, z] (modified in-place).

        Returns
        -------
        None

        Notes
        -----
        ``H_jk = d^2 u / dx_j dx_k`` evaluated with the Hessian stencil
        ``H_hess_at_pixel_deqnijk``. muGrid operators only support a single
        output component axis, so the operator works on a flattened
        ``[1, d*d, q, ...]`` scratch field (``'Hessian_u_flat'``) which is
        reshaped into ``[1, d, d, q, ...]`` afterwards. Note that the
        Hessian of standard linear elements vanishes inside an element;
        the result depends entirely on the stencil provided by the element.
        """
        if self.nb_nodes_per_pixel > 1:
            warnings.warn('Hessian operator is not tested for multiple nodal points per pixel.')

        # if the input is ndArray, create muGrid field out of it
        if isinstance(u_inxyz, np.ndarray):
            raise TypeError("apply_hessian_operator_mugrid does not support ndarray")

        dim = self.domain_dimension

        # scratch field with the derivative pair flattened: [i,J,q,x,y,z], J = j*dim + k
        hess_u_iJqxyz = self.get_temperature_hessian_size_field_mugrid_compatible(name='Hessian_u_flat')

        self.fft.communicate_ghosts(field=u_inxyz)
        # compute Hessian
        self.hessian_op.apply(nodal_field=u_inxyz,
                              quadrature_point_field=hess_u_iJqxyz)

        # put it back to Hessian_ijkqxyz from Hessina_iJqxyz
        # splitting axis 1 (J) into (j,k) is a pure view, no copy, even though
        # .s is a strided window into the ghosted buffer
        hess_u_ijkqxyz.s[...] = hess_u_iJqxyz.s.reshape(hess_u_iJqxyz.s.shape[0], dim, dim,
                                                        *hess_u_iJqxyz.s.shape[2:])

        self.fft.communicate_ghosts(field=hess_u_ijkqxyz)

    def apply_hessian_operator_to_vector_field_mugrid(self, u_inxyz, hess_u_ijkqxyz):
        """Compute Hessian (second derivatives) of vector field u at quadrature points.

        Parameters
        ----------
        u_inxyz : muGrid Field
            Nodal vector field [i, n, x, y, z].
        hess_u_ijkqxyz : muGrid Field
            Output Hessian field at quadrature points [i, j, k, q, x, y, z] (modified in-place).

        Returns
        -------
        None

        Notes
        -----
        Component-wise version of
        :meth:`apply_hessian_operator_to_scalar_field_mugrid`:
        ``H_ijk = d^2 u_i / dx_j dx_k``, computed via the flattened
        ``[d, d*d, q, ...]`` scratch field ``'Hessian_u_flat'``.
        """
        # if self.nb_nodes_per_pixel > 1:
        #     warnings.warn('Hessian operator is not tested for multiple nodal points per pixel.')

        # if the input is ndArray, create muGrid field out of it
        if isinstance(u_inxyz, np.ndarray):
            raise TypeError("apply_hessian_operator_mugrid does not support ndarray")

        dim = self.domain_dimension

        # scratch field with the derivative pair flattened: [i,J,q,x,y,z], J = j*dim + k
        hess_u_iJqxyz = self.get_displacement_hessian_size_field_mugrid_compatible(name='Hessian_u_flat')
        self.fft.communicate_ghosts(field=u_inxyz)

        # compute Hessian
        self.hessian_op.apply(nodal_field=u_inxyz,
                              quadrature_point_field=hess_u_iJqxyz)

        # put it back to Hessian_ijkqxyz from Hessina_iJqxyz
        # splitting axis 1 (J) into (j,k) is a pure view, no copy, even though
        # .s is a strided window into the ghosted buffer
        hess_u_ijkqxyz.s[...] = hess_u_iJqxyz.s.reshape(hess_u_iJqxyz.s.shape[0], dim, dim,
                                                        *hess_u_iJqxyz.s.shape[2:])

        self.fft.communicate_ghosts(field=hess_u_ijkqxyz)

    def apply_hessian_operator_transposed_to_scalar_field_mugrid(self, hess_u_ijkqxyz, nodal_field_inxyz,
                                                                 apply_weights=True):
        """Apply transposed Hessian operator to scalar field.

        Parameters
        ----------
        hess_u_ijkqxyz : muGrid Field
            Hessian field at quadrature points [i,j,k,q,x,y,z]
        nodal_field_inxyz : muGrid Field
            Output nodal field [i,n,x,y,z]
        apply_weights : bool
            Apply quadrature weights if True

        Returns
        -------
        None
            Modifies nodal_field_inxyz in-place

        Notes
        -----
        Computes ``H^T W h`` (or ``H^T h`` without weights), i.e.
        ``f^n = sum_q w_q d^2N^n/dx_j dx_k (x_q) h_jk(x_q)`` -- the adjoint of
        :meth:`apply_hessian_operator_to_scalar_field_mugrid`, used e.g. in
        gradients of objectives that depend on second derivatives.
        """
        if self.nb_nodes_per_pixel > 1:
            warnings.warn('Hessian operator is not tested for multiple nodal points per pixel.')

        # if the input is ndArray, create muGrid field out of it
        if isinstance(nodal_field_inxyz, np.ndarray):
            raise TypeError("apply_hessian_operator_mugrid does not support ndarray")

        dim = self.domain_dimension

        # flatten the derivative pair into the muGrid-compatible layout:
        # [i,j,k,q,x,y,z] -> [i,J,q,x,y,z],  J = j*dim + k
        hess_u_iJqxyz = self.get_temperature_hessian_size_field_mugrid_compatible(name='Hessian_u_flat')
        hess_u_iJqxyz.s[...] = hess_u_ijkqxyz.s.reshape(hess_u_ijkqxyz.s.shape[0], dim * dim,
                                                        *hess_u_ijkqxyz.s.shape[3:])

        # get quadrature weights
        if apply_weights:
            weights = self.quadrature_weights
        else:
            weights = np.ones(self.quadrature_weights.shape)

        # (the flattening (j,k) -> J was already done above; here we only refresh
        #  the ghost layers of the flat field before the transposed stencil reads it)

        self.fft.communicate_ghosts(field=hess_u_iJqxyz)
        # apply H^transposed via the convolution operator
        self.hessian_op.transpose(quadrature_point_field=hess_u_iJqxyz,
                                  nodal_field=nodal_field_inxyz,
                                  weights=weights)

        self.fft.communicate_ghosts(field=nodal_field_inxyz)

    def apply_hessian_operator_transposed_to_vector_field_mugrid(self, hess_u_ijkqxyz, nodal_field_inxyz,
                                                                 apply_weights=True):
        """Apply transposed Hessian operator to vector field.

        Parameters
        ----------
        hess_u_ijkqxyz : muGrid Field
            Hessian field at quadrature points [i,j,k,q,x,y,z]
        nodal_field_inxyz : muGrid Field
            Output nodal field [i,n,x,y,z]
        apply_weights : bool
            Apply quadrature weights if True

        Returns
        -------
        None
            Modifies nodal_field_inxyz in-place

        Notes
        -----
        Vector version of
        :meth:`apply_hessian_operator_transposed_to_scalar_field_mugrid`:
        ``f_i^n = sum_q w_q d^2N^n/dx_j dx_k (x_q) h_ijk(x_q)``.
        """
        if isinstance(nodal_field_inxyz, np.ndarray):
            raise TypeError("apply_hessian_operator_mugrid does not support ndarray")

        dim = self.domain_dimension

        # flatten the derivative pair into the muGrid-compatible layout:
        # [i,j,k,q,x,y,z] -> [i,J,q,x,y,z],  J = j*dim + k
        hess_u_iJqxyz = self.get_displacement_hessian_size_field_mugrid_compatible(name='Hessian_u_flat')
        hess_u_iJqxyz.s[...] = hess_u_ijkqxyz.s.reshape(hess_u_ijkqxyz.s.shape[0], dim * dim,
                                                        *hess_u_ijkqxyz.s.shape[3:])

        # get quadrature weights
        if apply_weights:
            weights = self.quadrature_weights
        else:
            weights = np.ones(self.quadrature_weights.shape)

        # (the flattening (j,k) -> J was already done above; here we only refresh
        #  the ghost layers of the flat field before the transposed stencil reads it)

        self.fft.communicate_ghosts(field=hess_u_iJqxyz)
        # apply H^transposed via the convolution operator
        self.hessian_op.transpose(quadrature_point_field=hess_u_iJqxyz,
                                  nodal_field=nodal_field_inxyz,
                                  weights=weights)

        self.fft.communicate_ghosts(field=nodal_field_inxyz)

    def evaluate_field_at_quad_points(self,
                                      nodal_field_fnxyz,
                                      quad_field_fqnxyz=None,
                                      quad_points_coords_iq=None):
        """Evaluate nodal field at quadrature points via interpolation operator N.

        Parameters
        ----------
        nodal_field_fnxyz : muGrid Field
            Input field at nodal points [f, n, x, y, z].
        quad_field_fqnxyz : muGrid Field, optional
            Output field at quadrature points [f, q, x, y, z].
        quad_points_coords_iq : ndarray, optional
            Parametric coordinates of quadrature points.

        Returns
        -------
        quad_field_fqnxyz : muGrid Field
            Interpolated field at quadrature points [f, q, x, y, z].
            Same object as the input ``quad_field_fqnxyz`` (filled in-place);
            it must therefore be provided despite the ``None`` default.

        Notes
        -----
        ``u_f(x_q) = sum_n N^n(x_q) u_f^n``. ``quad_points_coords_iq`` is
        currently unused: the quadrature points baked into the element's
        interpolation stencil are always used. Functionally identical to
        :meth:`apply_N_operator_mugrid`.
        """
        # if the input is ndArray, create muGrid field out of it
        if isinstance(nodal_field_fnxyz, np.ndarray):
            raise TypeError("apply_N_operator_mugrid does not support ndarray")

        self.fft.communicate_ghosts(field=nodal_field_fnxyz)
        self.interpolation_op.apply(nodal_field=nodal_field_fnxyz,
                                    quadrature_point_field=quad_field_fqnxyz)
        self.fft.communicate_ghosts(field=quad_field_fqnxyz)

        return quad_field_fqnxyz

    def apply_N_operator_mugrid(self, nodal_field_inxyz, quad_field_ijqnxyz):
        """Interpolate nodal field to quadrature points using convolution operator N.

        Parameters
        ----------
        nodal_field_inxyz : muGrid Field
            Nodal field [i, n, x, y, z].
        quad_field_ijqnxyz : muGrid Field
            Output field at quadrature points [i, q, x, y, z] (modified in-place).

        Returns
        -------
        None

        Notes
        -----
        Applies the interpolation matrix ``N``:
        ``u_i(x_q) = sum_n N^n(x_q) u_i^n`` (including nodes of neighbouring
        pixels, hence the ghost exchange). Typical use: evaluating a nodal
        design/phase field at the quadrature points to build material data.
        """

        if self.nb_nodes_per_pixel > 1:
            warnings.warn('Projection operator does not work for multiple nodal points per pixel.')

        # if the input is ndArray, create muGrid field out of it
        if isinstance(nodal_field_inxyz, np.ndarray):
            raise TypeError("apply_N_operator_mugrid does not support ndarray")

        self.fft.communicate_ghosts(field=nodal_field_inxyz)
        self.interpolation_op.apply(nodal_field=nodal_field_inxyz,
                                    quadrature_point_field=quad_field_ijqnxyz)
        self.fft.communicate_ghosts(field=quad_field_ijqnxyz)

    def apply_N_transposed_operator_mugrid(self,
                                           quad_field_ijqxyz,
                                           nodal_field_inxyz,
                                           apply_weights=True):
        """Apply transposed interpolation operator N^T from quadrature to nodal points.

        Parameters
        ----------
        quad_field_ijqxyz : muGrid Field
            Field at quadrature points [i, q, x, y, z].
        nodal_field_inxyz : muGrid Field
            Output field at nodal points [i, n, x, y, z] (modified in-place).
        apply_weights : bool, optional
            Whether to apply quadrature weights (default is True).

        Returns
        -------
        None

        Notes
        -----
        ``f_i^n = sum_q w_q N^n(x_q) g_i(x_q)`` = ``N^T W g`` -- the adjoint of
        :meth:`apply_N_operator_mugrid`. Used e.g. to pull sensitivities
        computed at quadrature points back to nodal design variables
        (chain rule through the interpolation).
        """

        if isinstance(quad_field_ijqxyz, np.ndarray):
            raise TypeError("apply_N_transposed_operator_mugrid does not support ndarray")

        if self.nb_nodes_per_pixel > 1:
            warnings.warn('apply_N_transposed_operator_mugrid is not tested for multiple nodal points per pixel.')
        # clear div array
        # get quadrature weights
        if apply_weights:
            weights = self.quadrature_weights
        else:
            weights = np.ones(self.quadrature_weights.shape)

        self.fft.communicate_ghosts(field=quad_field_ijqxyz)
        # apply N^transposed via the convolution operator
        self.interpolation_op.transpose(quadrature_point_field=quad_field_ijqxyz,
                                        nodal_field=nodal_field_inxyz,
                                        weights=weights)

        self.fft.communicate_ghosts(field=nodal_field_inxyz)

    def get_rhs_mugrid(self, material_data_field_ijklqxyz,
                       macro_gradient_field_ijqxyz,
                       rhs_inxyz):
        """
        Function that computes right hand side vector of linear elastic homogenization problem
        rhs= - B^t: C: E

        Parameters
        ----------
        material_data_field_ijklqxyz: numpy ndarray of discretized  material data tangent field [i,j,k,l,q,x,y,z]
            - elasticity shape   [i,j,k,l,q,x,y,z] and i,j,k,l = 0,...,d-1.
            - conductivity shape     [i,j,q,x,y,z] and i,j  = 0,...,d-1.
            - q is a quadrature point index

        macro_gradient_field_ijqxyz: numpy ndarray of discretized  macroscopic gradient E - constant part of gradient
            - shape [i,j,q,x,y,z]
            - q is quadrature point index

        rhs_inxyz : muGrid Field [i, n, x, y, z]
            Output; overwritten in-place with the right-hand side.

        Returns
        -------
        None
            (the rhs is written into ``rhs_inxyz``)

        Notes
        -----
        Despite the "numpy ndarray" wording above, both inputs must be muGrid
        fields (``.s`` is accessed). The weak form
        ``int grad(v) : C : (E + grad(u)) dx = 0`` for all periodic ``v``
        gives ``K u = -B^T W C E``. The macro gradient is copied into the
        scratch field ``'stress_temporary_rhs'`` so the input is not modified.
        """

        # macro_gradient_field    [f,d,q,x,y,z]
        # material_data_field [d,d,d,d,q,x,y,z] - elasticity
        # material_data_field     [d,d,q,x,y,z] - conductivity
        # rhs                       [f,n,x,y,z]
        #  rhs=-Dt*wA*E

        self.assert_material_symmetry(material_data_field_ijklqxyz, name='material data in get_rhs')
        gradient_ijqxyz = self.get_gradient_size_field(name='stress_temporary_rhs')
        gradient_ijqxyz.s[...] = macro_gradient_field_ijqxyz.s[...]

        # sigma = C : E   (in place in the scratch field)
        self.apply_material_data_mugrid(material_data_field_ijklqxyz, gradient_ijqxyz)

        # rhs = B^T W sigma
        self.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=gradient_ijqxyz,
                                                       div_u_fnxyz=rhs_inxyz,
                                                       apply_weights=True)

        # rhs = - B^T W C E
        rhs_inxyz.s[...] *= -1

        self.fft.communicate_ghosts(field=rhs_inxyz)

    def get_rhs_mugrid_deformed_grid(self, material_data_field_ijklqxyz,
                                     macro_gradient_field_ijqxyz,
                                     rhs_inxyz,
                                     det_of_deformation_gradient,
                                     inv_of_deformation_gradient):
        """
        Function that computes right hand side vector of linear  homogenization problem
        on deformed grid
        rhs = -B^T W [ det(F_q) * (C : (E F_q^{-1})) F_q^{-T} ]

        Parameters
        ----------
        material_data_field_ijklqxyz: numpy ndarray of discretized  material data tangent field [i,j,k,l,q,x,y,z]
            - elasticity shape   [i,j,k,l,q,x,y,z] and i,j,k,l = 0,...,d-1.
            - conductivity shape     [i,j,q,x,y,z] and i,j  = 0,...,d-1.
            - q is a quadrature point index

        macro_gradient_field_ijqxyz: numpy ndarray of discretized  macroscopic gradient E - constant part of gradient
            - shape [i,j,q,x,y,z]
            - q is quadrature point index

        rhs_inxyz : muGrid Field [i, n, x, y, z]
            Output; overwritten in-place.
        det_of_deformation_gradient : muGrid Field, ``.s`` shape [1, 1, q, x, y, z]
            (scalar quadrature field, e.g. from ``get_quad_field_scalar``);
            ``det(F_q)`` of the map from the regular reference grid to the
            deformed (physical) grid.
        inv_of_deformation_gradient : muGrid Field [d, d, q, x, y, z]
            ``F_q^{-1}``.

        Returns
        -------
        None
            (the rhs is written into ``rhs_inxyz``)

        Notes
        -----
        Pull-back of the weak form from the deformed grid to the regular
        reference grid ``X`` (x = phi(X), F = dphi/dX)::

            grad_x v = grad_X v . F^{-1},   dx = det(F) dX
            rhs = - B^T W [ det(F) (C : (E . F^{-1})) . F^{-T} ]

        Mirrors :meth:`apply_system_matrix_mugrid_deformed_grid`. Note that,
        as implemented, the macroscopic gradient ``E`` is also multiplied by
        ``F^{-1}`` and no symmetrisation is applied.
        """

        # aliasing
        det_F = det_of_deformation_gradient
        inv_F = inv_of_deformation_gradient

        self.assert_material_symmetry(material_data_field_ijklqxyz, name='material data in get_rhs')
        gradient_ijqxyz = self.get_gradient_size_field(name='stress_temporary_rhs')
        gradient_ijqxyz.s[...] = macro_gradient_field_ijqxyz.s[...]

        # Macro gradient in reference domain
        # gradient_ijqxyz.s[...] = np.einsum('ij...,jk...->ik...', gradient_ijqxyz.s[...], inv_F)
        tensor_operations.dot22(gradient_ijqxyz, inv_F, gradient_ijqxyz)

        self.fft.communicate_ghosts(field=gradient_ijqxyz)

        # apply constitutive law
        self.apply_material_data_mugrid(material_data_field_ijklqxyz, gradient_ijqxyz)

        # w_q * div( det(F^q) * σ^q · (F^q)^-T ) // transformed divergence
        gradient_ijqxyz.s[...] = np.einsum('ij...,kj...->ik...', gradient_ijqxyz.s[...], inv_F.s[...]) * det_F.s[None, None, ...]


        self.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=gradient_ijqxyz,
                                                       div_u_fnxyz=rhs_inxyz,
                                                       apply_weights=True)

        rhs_inxyz.s[...] *= -1

        self.fft.communicate_ghosts(field=rhs_inxyz)

    def get_rhs_explicit_stress_mugrid(self, stress_function,
                                       gradient_field_ijqxyz,
                                       rhs_inxyz, **kwargs):
        """
        Function that computes right hand side vector of linear elastic homogenization problem
        rhs= - B^t: C: E+grad_U

        Parameters
        ----------
        stress_function : callable
            ``stress_function(gradient_field, stress_field)`` -- evaluates the
            (possibly nonlinear) constitutive law and writes the stress/flux
            into the second argument (a muGrid field [i, j, q, x, y, z]).
        gradient_field_ijqxyz : muGrid Field [i, j, q, x, y, z]
            Total gradient at which the stress is evaluated, typically
            ``E + grad(u)`` (only ``E`` for the first Newton step).
        rhs_inxyz : muGrid Field [i, n, x, y, z]
            Output; overwritten in-place with ``-B^T W sigma(gradient)``.
        **kwargs
            Ignored.

        Returns
        -------
        None

        Notes
        -----
        This is the negative residual (internal force) used in Newton
        iterations for nonlinear materials.
        """

        # macro_gradient_field    [f,d,q,x,y,z]
        # material_data_field [d,d,d,d,q,x,y,z] - elasticity
        # material_data_field     [d,d,q,x,y,z] - conductivity
        # rhs                       [f,n,x,y,z]
        #  rhs=-Dt*wA*E

        stress = self.get_gradient_size_field(name='stress_temporary')
        # stress.s, _ = stress_function(gradient_field_ijqxyz)
        stress_function(gradient_field_ijqxyz, stress)

        self.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=stress,
                                                       div_u_fnxyz=rhs_inxyz,
                                                       apply_weights=True)
        rhs_inxyz.s[...] *= -1

        self.fft.communicate_ghosts(field=rhs_inxyz)

    def get_macro_gradient_field_mugrid(self,
                                        macro_gradient_ij,
                                        macro_gradient_field_ijqxyz):
        """
        Function that returns macro gradient field E from single macro gradient vector

        Parameters
        ----------
        macro_gradient_ij: numpy ndarray of macro gradient [i,j ]
        macro_gradient_field_ijqxyz : muGrid Field [i, j, q, x, y, z]
            Output field, filled in-place.

        Returns
        -------
        macro_gradient_field_ijqxyz:  quadrature point field of macroscopic gradient [i,j,q, x,y,z]
            The same object as the input field.
        """

        # broadcast E_ij to every quadrature point and pixel: append (ndim-2) singleton axes
        macro_gradient_field_ijqxyz.s[..., :] = macro_gradient_ij[
            (...,) + (np.newaxis,) * (macro_gradient_field_ijqxyz.s.ndim - 2)]
        return macro_gradient_field_ijqxyz

    def get_homogenized_stress_mugrid(self,
                                      material_data_field_ijklqxyz,
                                      displacement_field_inxyz,
                                      macro_gradient_field_ijqxyz,
                                      formulation=None):
        """
         Function that computes homogenized  (averaged) stress field (or flux)

         Parameters
         ----------
         material_data_field_ijklqxyz: numpy ndarray of discretized  material data tangent field [i,j,k,l,q,x,y,z]
            - quadrature point field - q is a quadrature point index
            - elasticity shape   [i,j,k,l,q,x,y,z] and i,j,k,l = 0,...,d-1.
            - conductivity shape     [i,j,q,x,y,z] and i,j  = 0,...,d-1.

         displacement_field_inxyz:
            - nodal point field - displacement or temperature field

         macro_gradient_field_ijqxyz:
            - quadrature point field of macroscopic gradient [i,j,q, x,y,z]

         formulation: small strain or finite strain -'small_strain'

         Returns
         -------
         homogenized_stress_ij: nd array of homogenized stress of flux field
                    - int (C * (macro_grad + micro_grad))  dx / | domain |

         Notes
         -----
         ``<sigma> = 1/|Omega| sum_{q, pixels} w_q C_q : (E + grad u)_q``,
         reduced over all MPI ranks (collective). For a unit macro gradient
         ``E = e_k (x) e_l`` the result is the column ``A_eff[:, :, k, l]``
         of the effective tangent (``sigma_ij = C_ijkl E_kl``). The material data is
         copied into the scratch field ``'weighted_data_field_temporary'``
         and must be a muGrid field.
         """
        self.assert_material_symmetry(material_data_field_ijklqxyz,
                                      minor=(formulation == 'small_strain'),
                                      name='material data in get_homogenized_stress_mugrid')
        self.fft.communicate_ghosts(field=displacement_field_inxyz)

        gradient_field_ijqxyz = self.get_gradient_size_field(name='strain_temp')

        if formulation == 'small_strain':
            self.apply_gradient_operator_symmetrized_mugrid(u_inxyz=displacement_field_inxyz,
                                                            grad_u_ijqxyz=gradient_field_ijqxyz)

        else:
            self.apply_gradient_operator_mugrid(u_inxyz=displacement_field_inxyz,
                                                grad_u_ijqxyz=gradient_field_ijqxyz)

        # compute total strain
        gradient_field_ijqxyz.s[...] = gradient_field_ijqxyz.s + macro_gradient_field_ijqxyz.s

        mat_data_temp = self.get_material_data_size_field_mugrid(name='weighted_data_field_temporary')
        if isinstance(material_data_field_ijklqxyz, np.ndarray):
            # mat_data_temp.sg[...] = material_data_field_ijklqxyz
            raise TypeError("NOT YET  does not support ndarray")
        else:
            mat_data_temp.s[...] = material_data_field_ijklqxyz.s[...]

        self.apply_material_data_mugrid(material_data=mat_data_temp,
                                        gradient_field=gradient_field_ijqxyz)

        self.apply_quadrature_weights_on_gradient_field_mugrid(grad_field=gradient_field_ijqxyz)

        # sum over the last (d + 1) axes = quadrature points q and pixels x, y(, z); global MPI sum
        homogenized_stress_ij = self.mpi_reduction.sum(gradient_field_ijqxyz.s,
                                                       axis=tuple(range(-self.domain_dimension - 1, 0)))  #
        return homogenized_stress_ij / self.cell.domain_volume

    def get_homogenized_energy_mugrid(self,
                                      material_data_field_ijklqxyz,
                                      displacement_field_inxyz,
                                      macro_gradient_field_ijqxyz,
                                      formulation=None):
        """
         Function that computes the homogenized energy  E : A_eff : E

         Unlike get_homogenized_stress_mugrid, which is LINEAR in the displacement,
         this evaluation is QUADRATIC.  The two agree only when the discrete weak
         form holds, i.e. at the exact discrete solution or at a PCG iterate
         obtained with a zero initial guess.  For a nonzero initial guess they
         differ by  -u.r  (note: the referenced helper
         ``get_homogenized_energy_from_stress`` does not exist in this module).

         Parameters
         ----------
         material_data_field_ijklqxyz: numpy ndarray of discretized material data tangent field
            - quadrature point field - q is a quadrature point index
            - elasticity shape   [i,j,k,l,q,x,y,z] and i,j,k,l = 0,...,d-1.
            - conductivity shape     [i,j,q,x,y,z] and i,j  = 0,...,d-1.

         displacement_field_inxyz:
            - nodal point field - displacement or temperature field

         macro_gradient_field_ijqxyz:
            - quadrature point field of macroscopic gradient [i,j,q,x,y,z]

         formulation: small strain or finite strain -'small_strain'

         Returns
         -------
         homogenized_energy: float
                    - int (macro_grad + micro_grad) : C : (macro_grad + micro_grad) dx / | domain |
         """
        self.assert_material_symmetry(material_data_field_ijklqxyz,
                                      minor=(formulation == 'small_strain'),
                                      name='material data in get_homogenized_energy_mugrid')
        self.fft.communicate_ghosts(field=displacement_field_inxyz)

        gradient_field_ijqxyz = self.get_gradient_size_field(name='strain_temp_energy')

        if formulation == 'small_strain':
            self.apply_gradient_operator_symmetrized_mugrid(u_inxyz=displacement_field_inxyz,
                                                            grad_u_ijqxyz=gradient_field_ijqxyz)
        else:
            self.apply_gradient_operator_mugrid(u_inxyz=displacement_field_inxyz,
                                                grad_u_ijqxyz=gradient_field_ijqxyz)

        # total gradient  eps = E + grad u
        gradient_field_ijqxyz.s[...] = gradient_field_ijqxyz.s + macro_gradient_field_ijqxyz.s

        # second copy: apply_material_data_mugrid overwrites in place, but the
        # energy needs the UNWEIGHTED total gradient as the left factor
        flux_field_ijqxyz = self.get_gradient_size_field(name='flux_temp_energy')
        flux_field_ijqxyz.s[...] = gradient_field_ijqxyz.s[...]

        mat_data_temp = self.get_material_data_size_field_mugrid(name='weighted_data_field_temporary')
        if isinstance(material_data_field_ijklqxyz, np.ndarray):
            raise NotImplementedError("NOT YET does not support ndarray")
        else:
            mat_data_temp.s[...] = material_data_field_ijklqxyz.s[...]

        # flux <- C : (E + grad u)
        self.apply_material_data_mugrid(material_data=mat_data_temp,
                                        gradient_field=flux_field_ijqxyz)

        # flux <- w_q * C : (E + grad u)   -- weights on ONE factor only
        self.apply_quadrature_weights_on_gradient_field_mugrid(grad_field=flux_field_ijqxyz)

        # (E + grad u) : w_q C : (E + grad u),  reduced over q and space
        contracted_ij = self.mpi_reduction.sum(gradient_field_ijqxyz.s * flux_field_ijqxyz.s,
                                               axis=tuple(range(-self.domain_dimension - 1, 0)))

        return np.sum(contracted_ij) / self.cell.domain_volume

    def get_homogenized_stress_mugrid_explicit_stress(self,
                                                      constitutive: callable,
                                                      displacement_field_inxyz,
                                                      macro_gradient_field_ijqxyz,
                                                      formulation=None):
        """
         Function that computes homogenized  (averaged) stress field (or flux)

         Parameters
         ----------
         constitutive: callable fucntion strain ---> stress
            Called as ``constitutive(gradient_field, stress_field)``; here
            both arguments are the same field, so it must support in-place
            evaluation (output overwrites input).

         displacement_field_inxyz:
            - nodal point field - displacement or temperature field

         macro_gradient_field_ijqxyz:
            - quadrature point field of macroscopic gradient [i,j,q, x,y,z]

         formulation: small strain or finite strain -'small_strain'

         Returns
         -------
         homogenized_stress_ij: nd array of homogenized stress of flux field
                    - int (C * (macro_grad + micro_grad))  dx / | domain |
         """
        self.fft.communicate_ghosts(field=displacement_field_inxyz)

        gradient_field_ijqxyz = self.get_gradient_size_field(name='strain_temp')

        if formulation == 'small_strain':
            self.apply_gradient_operator_symmetrized_mugrid(u_inxyz=displacement_field_inxyz,
                                                            grad_u_ijqxyz=gradient_field_ijqxyz)

        else:
            self.apply_gradient_operator_mugrid(u_inxyz=displacement_field_inxyz,
                                                grad_u_ijqxyz=gradient_field_ijqxyz)

        # compute total strain
        gradient_field_ijqxyz.s[...] = gradient_field_ijqxyz.s + macro_gradient_field_ijqxyz.s

        # compute stress/flux field
        constitutive(gradient_field_ijqxyz, gradient_field_ijqxyz)

        self.apply_quadrature_weights_on_gradient_field_mugrid(grad_field=gradient_field_ijqxyz)

        homogenized_stress_ij = self.mpi_reduction.sum(gradient_field_ijqxyz.s,
                                                       axis=tuple(range(-self.domain_dimension - 1, 0)))  #
        return homogenized_stress_ij / self.cell.domain_volume

    def get_homogenized_stress_mugrid_deformed_grid(self, material_data_field_ijklqxyz,
                                                    temperature_field_inxyz,
                                                    macro_gradient_field_ijqxyz,
                                                    det_of_deformation_gradient,
                                                    inv_of_deformation_gradient,
                                                    formulation=None):
        '''
        Function computes the homogenized stress (elasticity) or flux
        (conductivity) from the solution on the deformed grid

        Parameters
        ----------
        material_data_field_ijklqxyz : muGrid Field
            Material tangent at quadrature points ([d,d,q,...] for
            conductivity, [d,d,d,d,q,...] for elasticity).
        temperature_field_inxyz : muGrid Field [i, n, x, y, z]
            Solution (fluctuation) on the reference grid; despite the name it
            may also be a displacement field.
        macro_gradient_field_ijqxyz : muGrid Field [i, j, q, x, y, z]
            Macroscopic gradient ``E``.
        det_of_deformation_gradient : muGrid Field
            ``det(F_q)`` per quadrature point (``.s`` shape [1, 1, q, x, y, z]).
        inv_of_deformation_gradient : muGrid Field [d, d, q, x, y, z]
            ``F_q^{-1}``.
        formulation : str, optional
            ``'small_strain'`` symmetrises the transformed gradient.

        Returns
        -------
        ndarray [i, j]
            ``1/|Omega_ref| sum_q w_q det(F_q) C_q : ((E + grad_X u) . F_q^{-1})``,
            MPI-reduced. Note: divided by ``cell.domain_volume`` (the
            reference cell volume).
        '''
        self.assert_material_symmetry(material_data_field_ijklqxyz,
                                      minor=np.all(formulation == 'small_strain'),
                                      name='material data in get_homogenized_stress_mugrid_deformed_grid')

        # aliasing
        det_F = det_of_deformation_gradient
        inv_F = inv_of_deformation_gradient

        gradient_field_ijqxyz = self.get_gradient_size_field(name='grad_temp')
        self.fft.communicate_ghosts(field=temperature_field_inxyz)
        self.apply_gradient_operator_mugrid(u_inxyz=temperature_field_inxyz,
                                            grad_u_ijqxyz=gradient_field_ijqxyz)

        # compute total heat gradient field
        gradient_field_ijqxyz.s[...] = gradient_field_ijqxyz.s + macro_gradient_field_ijqxyz.s

        # apply deformation gradient    (∇ũ)^q · (F^q)^-1    // q-th transformed gradient
        #gradient_field_ijqxyz.s[...] = np.einsum('ij...,jk...->ik...', gradient_field_ijqxyz.s[...], inv_F)
        tensor_operations.dot22(gradient_field_ijqxyz, inv_F, gradient_field_ijqxyz)
        # symmetrization for small-strain elasticity
        if np.all(formulation == 'small_strain'):
            #  symmetrize it
            gradient_field_ijqxyz.s[...] = (gradient_field_ijqxyz.s + np.swapaxes(gradient_field_ijqxyz.s, 0, 1)) / 2

        # compute stress/flux field : σ^q ← C^q : ε^q              // constitutive model
        self.apply_material_data_mugrid(material_data=material_data_field_ijklqxyz,
                                        gradient_field=gradient_field_ijqxyz)

        # w_q *   det(F^q) * σ^q
        self.apply_quadrature_weights_on_gradient_field_mugrid(grad_field=gradient_field_ijqxyz)
        gradient_field_ijqxyz.s[...] = gradient_field_ijqxyz.s[...] * det_F.s[...][None, None, ...]

        homogenized_stress_ij = self.mpi_reduction.sum(gradient_field_ijqxyz.s,
                                                       axis=tuple(range(-self.domain_dimension - 1, 0)))  #
        return homogenized_stress_ij / self.cell.domain_volume

    def get_stress_field_mugrid(self,
                                material_data_field_ijklqxyz,
                                displacement_field_inxyz,
                                macro_gradient_field_ijqxyz,
                                output_stress_field_ijqxyz,
                                formulation=None):
        """
         Function that computes stress field (or flux)
            sigma  = C:(E+grad(u_fluctiation))
         Parameters
         ----------
         material_data_field_ijklqxyz: numpy ndarray of discretized  material data tangent field [i,j,k,l,q,x,y,z]
            - quadrature point field - q is a quadrature point index
            - elasticity shape   [i,j,k,l,q,x,y,z] and i,j,k,l = 0,...,d-1.
            - conductivity shape     [i,j,q,x,y,z] and i,j  = 0,...,d-1.

         displacement_field_inxyz:
            - nodal point field - displacement or temperature field

         macro_gradient_field_ijqxyz:
            - quadrature point field of macroscopic gradient [i,j,q, x,y,z]

         formulation: small strain or finite strain -'small_strain'

         output_stress_field_ijqxyz : muGrid Field [i, j, q, x, y, z]
            Output; first used as scratch for the strain, then overwritten
            with the stress.

         Returns
         -------
         None
            The stress ``sigma_ij = C_ijkl (E + grad u)_kl`` (per
            quadrature point, unweighted) is written into
            ``output_stress_field_ijqxyz``. Assumes elasticity-shaped
            material data.
         """

        if formulation == 'small_strain':
            # output_field_ijqxyz is strain field
            self.apply_gradient_operator_symmetrized_mugrid(u_inxyz=displacement_field_inxyz,
                                                            grad_u_ijqxyz=output_stress_field_ijqxyz)
        else:
            # output_field_ijqxyz is strain field
            self.apply_gradient_operator_mugrid(u_inxyz=displacement_field_inxyz,
                                                grad_u_ijqxyz=output_stress_field_ijqxyz)

        output_stress_field_ijqxyz.s[...] = output_stress_field_ijqxyz.s + macro_gradient_field_ijqxyz.s
        output_stress_field_ijqxyz.s[...] = np.einsum('ijkl...,kl...->ij...', material_data_field_ijklqxyz.s,
                                                      output_stress_field_ijqxyz.s)


    def get_flux_field_mugrid(self,
                              material_data_field_ijqxyz,
                              temperature_field_inxyz,
                              macro_gradient_field_ijqxyz,
                              output_flux_field_ijqxyz):
        """
         Function that computes flux field for given data and heat gradient
            sigma  = C:(E+grad(u_fluctiation))
         Parameters
         ----------
         material_data_field_ijqxyz: numpy ndarray of discretized  material data tangent field [i,j,q,x,y,z]
            - quadrature point field - q is a quadrature point index
            - conductivity shape     [i,j,q,x,y,z] and i,j  = 0,...,d-1.

         temperature_field_inxyz:
            - nodal point field -  temperature field

         macro_gradient_field_ijqxyz:
            - quadrature point field of macroscopic gradient [i,j,q, x,y,z]

         output_flux_field_ijqxyz : muGrid Field [1, d, q, x, y, z]
            Output; overwritten in-place with the flux.

         Returns
         -------
         None
            Result is written into ``output_flux_field_ijqxyz``.

         Notes
         -----
         Computes ``q_i = A_ij (E + grad T)_j`` at every quadrature point,
         the same contraction as
         :meth:`apply_material_data_conductivity_mugrid`. (The sign of
         Fourier's law, ``q = -A grad T``, is not included.)
         """

        # output_field_ijqxyz is strain field
        self.apply_gradient_operator_mugrid(u_inxyz=temperature_field_inxyz,
                                            grad_u_ijqxyz=output_flux_field_ijqxyz)
        #  macro_grad + micro_grad
        output_flux_field_ijqxyz.s[...] = output_flux_field_ijqxyz.s + macro_gradient_field_ijqxyz.s
        # q = C * (macro_grad + micro_grad)
        output_flux_field_ijqxyz.s[...] = np.einsum('ij...,uj...->ui...', material_data_field_ijqxyz.s,
                                                    output_flux_field_ijqxyz.s)



    def apply_quadrature_weights(self, material_data):
        """
        Wrapper for apply_quadrature_weights_conductivity and apply_quadrature_weights_elasticity

        Parameters
        ----------
        material_data : numpy ndarray of discretized  material data tangent field
            - conductivity shape  [i,j,q,x,y,z] and i,j  = 0,...,d-1.
            - elasticity shape   [i,j,k,l,q,x,y,z] and i,j,k,l = 0,...,d-1.
        Returns
        -------
        weighted_material_data : the same shape as input material_data
        """
        if self.cell.problem_type == 'conductivity':
            return self.apply_quadrature_weights_conductivity(material_data_ijqxyz=material_data)
        elif self.cell.problem_type == 'elasticity':
            return self.apply_quadrature_weights_elasticity(material_data_ijklqxyz=material_data)
        else:
            raise ValueError(
                'Unrecognised problem_type type {}. Choose from ' \
                ' : conductivity, elasticity '.format(self.cell.problem_type))

    def apply_quadrature_weights_conductivity(self, material_data_ijqxyz):
        """
        Function that applies quadrature weights to material tangent

        Parameters
        ----------
        material_data_ijqxyz: numpy ndarray of discretized  material data tangent field
            - conductivity shape  [i,j,q,x,y,z] and i,j  = 0,...,d-1.
            - q is a quadrature point index
        Returns
        -------
        weighted_material_data_ijqxyz: ndarray
            ``w_q * A_ij(x_q)``, a new numpy array. Note: uses ``.sg``, i.e.
            the returned array *includes* the ghost layers (unlike the
            elasticity variant, which uses ``.s``). Input must be a muGrid
            field.
        """
        weighted_material_data_ijqxyz = np.einsum('ijq...,q->ijq...', material_data_ijqxyz.sg, self.quadrature_weights)
        return weighted_material_data_ijqxyz

    def apply_quadrature_weights_elasticity(self, material_data_ijklqxyz):
        """
        Function that applies quadrature weights to material tangent

        Parameters
        ----------
        material_data_field_ijklqxyz: numpy ndarray of discretized  material data tangent field [i,j,k,l,q,x,y,z]
            - elasticity shape   [i,j,k,l,q,x,y,z] and i,j,k,l = 0,...,d-1.
            - q is a quadrature point index
        Returns
        -------
        weighted_material_data_ijklqxyz: ndarray
            ``w_q * C_ijkl(x_q)`` as a new numpy array (input not modified).
            Accepts ndarray or muGrid field (then ``.s`` without ghosts).
        """
        if isinstance(material_data_ijklqxyz, np.ndarray):
            weighted_material_data_ijklqxyz = np.einsum('ijklq...,q->ijklq...', material_data_ijklqxyz,
                                                        self.quadrature_weights)
        else:
            weighted_material_data_ijklqxyz = np.einsum('ijklq...,q->ijklq...', material_data_ijklqxyz.s,
                                                        self.quadrature_weights)

        return weighted_material_data_ijklqxyz

    def apply_quadrature_weights_on_gradient_field(self, grad_field):
        """Multiply a gradient-shaped numpy array by the quadrature weights.

        Parameters
        ----------
        grad_field : ndarray [i, j, q, x, y, z]

        Returns
        -------
        ndarray [i, j, q, x, y, z]
            New array ``w_q * grad_field[i, j, q, ...]``.
        """
        # apply quadrature weights without material data
        grad_field = np.einsum('ijq...,q->ijq...', grad_field, self.quadrature_weights)

        return grad_field

    def apply_quadrature_weights_on_gradient_field_mugrid(self, grad_field):
        """Multiply a gradient-shaped muGrid field by the quadrature weights, in place.

        Parameters
        ----------
        grad_field : muGrid Field [i, j, q, x, y, z]
            Overwritten with ``w_q * grad_field``. Summing the result over
            ``q`` and pixels yields the integral over the cell.

        Returns
        -------
        None
        """
        # apply quadrature weights without material data
        grad_field.s[...] = np.einsum('ijq...,q->ijq...', grad_field.s, self.quadrature_weights)

    def apply_material_data(self, material_data, gradient_field):
        """Legacy (numpy-returning) constitutive map, dispatched on problem type.

        Parameters
        ----------
        material_data : ndarray or muGrid Field
            Material tangent, see :meth:`apply_material_data_conductivity` /
            :meth:`apply_material_data_elasticity`.
        gradient_field : muGrid Field
            Gradient at quadrature points [i, j, q, x, y, z].

        Returns
        -------
        ndarray
            Flux / stress (a new array, input unchanged).

        Raises
        ------
        ValueError
            For an unknown ``cell.problem_type``.
        """
        if self.cell.problem_type == 'conductivity':
            return self.apply_material_data_conductivity(material_data, gradient_field)
        elif self.cell.problem_type == 'elasticity':
            return self.apply_material_data_elasticity(material_data, gradient_field)
        else:
            raise ValueError(
                'Unrecognised problem_type type {}. Choose from ' \
                ' : conductivity, elasticity '.format(self.cell.problem_type))

    def evaluate_material_model(self, material_data, gradient_field, **kwargs):
        """Evaluate a simple nonlinear material model.

        Parameters
        ----------
        material_data : muGrid Field [d, d, d, d, q, x, y, z]
            Stiffness-like tensor ``C``.
        gradient_field : muGrid Field [d, d, q, x, y, z]
            Strain ``eps``.
        **kwargs
            Must contain ``mat_model`` (str). Only models whose name contains
            ``'power_law_elasticity'`` are supported.

        Returns
        -------
        ndarray [d, d, q, x, y, z]
            ``sigma_ij = C_ijkl (eps_kl)^n`` with ``n = 0.3`` applied
            element-wise to the strain components.

        Raises
        ------
        ValueError
            For an unknown material model.

        Notes
        -----
        ``np.power`` of negative strain components with ``n = 0.3`` gives NaN.
        """
        if "power_law_elasticity" in kwargs['mat_model']:
            n = 0.3  # strain-hardening exponent
            stress = np.einsum('ijkl...,kl...->ij...', material_data.s, np.power(gradient_field.s, n))


        else:
            raise ValueError('Unknown material model')

        return stress

    def apply_material_data_elasticity(self, material_data, gradient_field):
        """Compute ``sigma_ij = C_ijkl eps_kl`` and return it as a new array.

        Parameters
        ----------
        material_data : ndarray or muGrid Field
            Either a single (reference) tensor [d, d, d, d] applied
            everywhere, or a full field [d, d, d, d, q, x, y, z].
        gradient_field : muGrid Field [d, d, q, x, y, z]

        Returns
        -------
        ndarray [d, d, q, x, y, z]
            Stress (without ghost layers).
        """
        # ddot42 = lambda A4, B2: np.einsum('ijklxyz,lkxyz  ->ijxyz  ', A4, B2)
        if isinstance(material_data, np.ndarray):
            # for the case of ref material, we need only one single material tensor
            if material_data.ndim == 4:
                stress = np.einsum('ijkl,kl...->ij...', material_data, gradient_field.s)
            elif material_data.ndim > 4:
                stress = np.einsum('ijkl...,kl...->ij...', material_data, gradient_field.s)
        else:
            stress = np.einsum('ijkl...,kl...->ij...', material_data.s, gradient_field.s)

        return stress

    def apply_material_data_conductivity(self, material_data, gradient_field):
        """Compute the flux ``q_ui = A_ij g_uj`` and return it as a new array.

        Parameters
        ----------
        material_data : ndarray or muGrid Field
            Single conductivity matrix [d, d] or full field [d, d, q, x, y, z].
        gradient_field : muGrid Field [1, d, q, x, y, z]
            Temperature gradient; the leading size-1 axis ``u`` is kept.

        Returns
        -------
        ndarray [1, d, q, x, y, z]
            Flux, *including* ghost layers (``.sg`` is used here). With
            ndarray full-field material data its shape must therefore match
            the ghosted layout.
        """
        # dot21  = lambda A,v: np.einsum('ij...,j...  ->i...',A,v)
        if isinstance(material_data, np.ndarray):
            # for the case of ref material, we need only one single material tensor
            if material_data.ndim == 2:
                flux_ijnxyz = np.einsum('ij,uj...->ui...', material_data,
                                        gradient_field.sg)  # 'u' just to keep the size of array consistent
            else:
                flux_ijnxyz = np.einsum('ij...,uj...->ui...', material_data,
                                        gradient_field.sg)

                # raise ValueError('The reference material_data for conductivity hase more dimensions than 2')
        else:
            flux_ijnxyz = np.einsum('ij...,uj...->ui...', material_data.sg,
                                    gradient_field.sg)  # 'u' just to keep the size of array consistent

        return flux_ijnxyz

    def apply_material_data_mugrid(self, material_data, gradient_field):
        """Apply the material tangent in place: ``gradient_field <- C : gradient_field``.

        Dispatches to :meth:`apply_material_data_conductivity_mugrid` or
        :meth:`apply_material_data_elasticity_mugrid` according to
        ``cell.problem_type``.

        Parameters
        ----------
        material_data : muGrid Field or ndarray
            Full material field at quadrature points, or a single reference
            tensor ([d, d] / [d, d, d, d]) used in every quadrature point.
        gradient_field : muGrid Field [i, j, q, x, y, z]
            Input gradient; **overwritten** with the flux/stress.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            For an unknown ``cell.problem_type``.
        """
        if self.cell.problem_type == 'conductivity':
            self.apply_material_data_conductivity_mugrid(material_data, gradient_field)
        elif self.cell.problem_type == 'elasticity':
            self.apply_material_data_elasticity_mugrid(material_data, gradient_field)
        else:
            raise ValueError(
                'Unrecognised problem_type type {}. Choose from ' \
                ' : conductivity, elasticity '.format(self.cell.problem_type))

    def apply_material_data_conductivity_mugrid(self, material_data, gradient_field):
        """In-place flux evaluation ``g_ui <- A_ij g_uj`` (``u`` = size-1 axis).

        Parameters
        ----------
        material_data : muGrid Field [d, d, q, x, y, z] or ndarray [d, d]
            Conductivity field, or a single (reference) conductivity matrix.
            A full ndarray field is not supported.
        gradient_field : muGrid Field [1, d, q, x, y, z]
            Temperature gradient; overwritten with the flux.

        Returns
        -------
        None
        """
        # overwrite the array !!!
        # dot21  = lambda A,v: np.einsum('ij...,j...  ->i...',A,v)
        if isinstance(material_data, np.ndarray):
            # for the case of ref material, we need only one single material tensorF
            if material_data.ndim == 2:
                gradient_field.s[...] = np.einsum('ij,uj...->ui...', material_data,
                                                  gradient_field.s)  # 'u' just to keep the size of array consistent
            else:
                raise TypeError("apply_material_data_mugrid does not support global ndarray")

                # raise ValueError('The reference material_data for conductivity hase more dimensions than 2')
        else:
            gradient_field.s[...] = np.einsum('ij...,uj...->ui...', material_data.s,
                                              gradient_field.s)  # 'u' just to keep the size of array consistent

    def apply_material_data_elasticity_mugrid(self, material_data, gradient_field):
        """In-place stress evaluation ``eps_ij <- C_ijkl eps_kl``.

        Parameters
        ----------
        material_data : muGrid Field [d, d, d, d, q, x, y, z] or ndarray [d, d, d, d]
            Stiffness field, or a single (reference) stiffness tensor.
            A full ndarray field is not supported.
        gradient_field : muGrid Field [d, d, q, x, y, z]
            Strain / displacement gradient; overwritten with the stress.

        Returns
        -------
        None

        Notes
        -----
        The contraction is the standard ``C_ijkl eps_kl``. For an ndarray with ndim > 4 a TypeError is raised
        (``TypeError``); an ndarray with ndim < 4 is silently ignored.
        """
        # ddot42 = lambda A4, B2: np.einsum('ijklxyz,lkxyz  ->ijxyz  ', A4, B2)
        if isinstance(material_data, np.ndarray):
            # for the case of ref material, we need only one single material tensor
            if material_data.ndim == 4:
                gradient_field.s[...] = np.einsum('ijkl,kl...->ij...', material_data, gradient_field.s)

            elif material_data.ndim > 4:
                raise TypeError("apply_material_data_elasticity_mugrid does not support global ndarray")
        else:
            # gradient_field.s[...] = np.einsum('ijkl...,kl...->ij...', material_data.s, gradient_field.s)
            tensor_operations.ddot42(material_data, gradient_field, gradient_field)

    def get_system_matrix_mugrid(self, material_data_field, formulation=None):
        """
        Function that assembly global system matrix K
        - memory hungry process that returns
        - loop over all possible unit impulses, to get all columns of system matrix
        Parameters
        ----------
        material_data_field : numpy ndarray of discretized  material data tangent field
                - conductivity shape  [i,j,q,x,y,z] and i,j  = 0,...,d-1.
                - elasticity shape   [i,j,k,l,q,x,y,z] and i,j,k,l = 0,...,d-1.
        Returns
        -------
        K_system_matrix : system matrix with shape [nb_dof,nb_dofs]# TODO look for orderign
            - ordering of dofs as follows:
                - first are all degrees of freedom for displacement in x direction
                - second are all degrees of freedom for displacement in y direction
                - third are all degrees of freedom for displacement in z direction
                K = [ K_00   K_01   K_02,
                      K_10   K_11   K_12,
                      K_20   K_21   K_22]

        formulation : str, optional
            Passed to :meth:`apply_system_matrix_mugrid` (``'small_strain'``
            uses the symmetrised gradient).

        Notes
        -----
        The dof ordering is that of ``np.ndindex`` over the local array
        ``[f, n, x, y, z]`` (C order). Row ``i`` stores ``K e_i``, i.e. the
        ``i``-th column (``K`` is symmetric). Dense ``O(N^2)`` memory; meant
        for small grids, tests and debugging, and only meaningful on a single
        MPI rank.
        """

        unit_impulse = self.get_unknown_size_field(name='unit_impulse')
        K_impulse = self.get_unknown_size_field(name='K_impulse')
        K_system_matrix = np.zeros([np.prod(unit_impulse.s.shape), np.prod(unit_impulse.s.shape)])
        i = 0
        for impuls_position in np.ndindex(unit_impulse.s.shape):
            unit_impulse.s.fill(0)
            unit_impulse.s[impuls_position] = 1
            self.apply_system_matrix_mugrid(material_data_field=material_data_field,
                                            input_field_inxyz=unit_impulse,
                                            output_field_inxyz=K_impulse,
                                            formulation=formulation)

            K_system_matrix[i] = K_impulse.s.flatten()
            i += 1

        return K_system_matrix

    def get_preconditioner_Green_mugrid(self, reference_material_data_ijkl,
                                        formulation=None,
                                        operator=None):
        """Assemble the Fourier-space Green preconditioner ``(K_ref)^{-1}``.

        Parameters
        ----------
        reference_material_data_ijkl : ndarray or muGrid Field
            Homogeneous reference material: a single tensor ([d, d] for
            conductivity, [d, d, d, d] for elasticity) or a constant field.
        formulation : str, optional
            Passed to :meth:`apply_system_matrix_mugrid`
            (``'small_strain'`` -> symmetrised gradient).
        operator : callable, optional
            Custom linear operator ``operator(input_field_inxyz=...,
            output_field_inxyz=...)`` to use instead of the reference
            stiffness ``K_ref``. Must be translation invariant (same stencil
            in every pixel) for the construction to be valid.

        Returns
        -------
        preconditioner_diagonals_ininqks : muGrid complex Field [f, n, f, n, k_x, k_y, (k_z)]
            Field ``'Greens_diagonal_fast'``: for every wave vector ``k`` the
            inverse of the (``f*n`` x ``f*n``) block ``K_ref_hat(k)``.

        Notes
        -----
        For a homogeneous material ``K_ref = B^T W C_ref B`` is a
        block-circulant matrix (the same stencil in every pixel), so the FFT
        block-diagonalises it: ``K_ref_hat(k)`` is a small dense matrix
        coupling the ``f*n`` dofs of one pixel. Its columns are obtained as
        the FFT of the response ``K_ref e_{f,n}`` to a unit impulse placed at
        dof ``(f, n)`` of the pixel at the origin. Inverting every block
        gives the preconditioner ``M^{-1} = F^{-1} [K_ref_hat]^{-1} F``,
        applied by :meth:`apply_preconditioner_mugrid`.

        The zero-frequency block ``k = 0`` is singular (rigid body
        translations / constant temperature are in the null space of a
        periodic problem). For one node per pixel it is left as is (not
        inverted; it is (numerically) zero because the stencil of ``K_ref``
        sums to zero, so the preconditioner annihilates the mean); for
        several nodes per pixel its pseudo-inverse is used.
        ``np.any(np.all(icoords == 0, axis=0))`` checks whether the local MPI
        rank owns the pixel/frequency at the origin -- only that rank places
        the impulse and treats index 0 specially. The impulse response
        computation is collective (ghost exchange, FFT).

        """
        if operator is None:
            self.assert_material_symmetry(reference_material_data_ijkl,
                                          minor=np.all(formulation == 'small_strain'),
                                          name='reference material of the Green preconditioner')
        # return diagonals of preconditioned matrix in Fourier space
        # unit_impulse [f,n,x,y,z]
        # for every type of degree of freedom DOF, there is one diagonal of preconditioner matrix
        # diagonals_in_Fourier_space [f,n,f,n][0,0,0]  # all DOFs in first pixel
        nb_dofs_per_voxel = self.nb_nodes_per_pixel * self.cell.unknown_shape[0]
        if self.nb_nodes_per_pixel == 1:
            # for one node per pixel, we can simplify the algorithm
            # for more nodes per pixel, we can add it later
            unit_impulse_inxyz = self.get_unknown_size_field(name='unit_impulse')
            unit_impulse_response_inxyz = self.get_unknown_size_field(name='unit_impulse_response')

            preconditioner_diagonals_ininqks = self.ffield_collection.complex_field(
                name='Greens_diagonal_fast',  # name of the field
                components=(*self.unknown_size[:2] + self.unknown_size[:1],),  # shape of components
                sub_pt='nodal_points'
            )  #
            unit_impulse_response_inqks = self.ffield_collection.complex_field(
                name='unit_impulse_response_inqks',  # name of the field
                components=(self.unknown_size[0],),  # shape of components
                sub_pt='nodal_points'
            )
            for impulse_position in np.ndindex(unit_impulse_inxyz.s.shape[0:2]):
                unit_impulse_inxyz.s.fill(0)  # empty the unit impulse vector
                if np.any(np.all(self.fft.icoords == 0, axis=0)):
                    # set 1 --- the unit impulse --- to a proper positions
                    unit_impulse_inxyz.s[impulse_position + (0,) * (unit_impulse_inxyz.s.ndim - 2)] = 1
                if operator is None:
                    self.apply_system_matrix_mugrid(
                        material_data_field=reference_material_data_ijkl,
                        input_field_inxyz=unit_impulse_inxyz,
                        output_field_inxyz=unit_impulse_response_inxyz,
                        formulation=formulation)
                else:
                    operator(
                        input_field_inxyz=unit_impulse_inxyz,
                        output_field_inxyz=unit_impulse_response_inxyz)

                self.fft.communicate_ghosts(unit_impulse_response_inxyz)
                # print(f"unit_impulse_response_inxyz {unit_impulse_response_inxyz.s[...]}")

                # FFT of the impulse response = column (f, n) of the block K_hat(k) for all k
                self.fft.fft(unit_impulse_response_inxyz, unit_impulse_response_inqks)
                # print(f"unit_impulse_response_inqks {unit_impulse_response_inqks.s[...]}")

                # store as K_hat[f, n, :, :, k]   (first index pair = impulse dof)
                preconditioner_diagonals_ininqks.s[impulse_position] = np.copy(unit_impulse_response_inqks.s[...])

            # THE SIZE OF DIAGONAL IS [nb_unit_dofs,nb_unit_dofs,nb_unit_dofs,nb_unit_dofs, xyz]
            # compute inverse of diagonals
            original_shape_ininqks = preconditioner_diagonals_ininqks.s.shape
            # n = 1: drop the two node axes -> [f, f, k...]
            prec_diagonals_ijqks = np.squeeze(preconditioner_diagonals_ininqks.s, axis=(1, 3))

            # Reshape the array to (n_u_dofs, n_u_dofs, ndof) for easier processing
            reshaped_matrices = prec_diagonals_ijqks.reshape(nb_dofs_per_voxel, nb_dofs_per_voxel, -1)
            # Transpose to shape (N, d, d) for batch inversion
            G_batch = reshaped_matrices.transpose(2, 0, 1)  # shape: (N, d, d)
            # (transpose returns a view, so the in-place inversion below also updates reshaped_matrices)
            # Invert each matrix using np.linalg.inv (vectorized)
            if np.any(np.all(self.fft.icoords == 0, axis=0)):  # check if the core has zero mode
                G_batch[1:, ...] = np.linalg.inv(G_batch[1:, ...])  # shape: (N, d, d) # do not inverte zero mode
            else:
                G_batch[0:, ...] = np.linalg.inv(G_batch[0:, ...])  # shape: (N, d, d)

            # Reshape the result back to the original shape
            G_diag_ijxy = G_batch.transpose(1, 2, 0).reshape(nb_dofs_per_voxel, nb_dofs_per_voxel,
                                                             *preconditioner_diagonals_ininqks.shape[4:])

            preconditioner_diagonals_ininqks.s[...] = G_diag_ijxy.reshape(original_shape_ininqks)[...]
        else:
            # General case: several nodes per pixel -> blocks of size (f*n) x (f*n).
            # (the two comment lines below are copied from the one-node branch)
            # for one node per pixel, we can simplify the algorithm
            # for more nodes per pixel, we can add it later
            unit_impulse_inxyz = self.get_unknown_size_field(name='unit_impulse')
            unit_impulse_response_inxyz = self.get_unknown_size_field(name='unit_impulse_response')

            preconditioner_diagonals_ininqks = self.ffield_collection.complex_field(
                name='Greens_diagonal_fast',  # name of the field
                components=(*self.unknown_size[:2] + self.unknown_size[:1],),  # shape of components
                sub_pt='nodal_points'  # sub-point type
            )  #
            unit_impulse_response_inqks = self.ffield_collection.complex_field(
                name='unit_impulse_response_inqks',  # name of the field
                components=(self.unknown_size[0],),  # shape of components
                sub_pt='nodal_points'
            )
            for impulse_position in np.ndindex(unit_impulse_inxyz.s.shape[0:2]):
                unit_impulse_inxyz.sg.fill(0)  # empty the unit impulse vector
                if np.any(np.all(self.fft.icoords == 0, axis=0)):
                    # set 1 --- the unit impulse --- to a proper positions
                    unit_impulse_inxyz.s[impulse_position + (0,) * (unit_impulse_inxyz.s.ndim - 2)] = 1
                    # print(f"Unit impulse set at position {impulse_position}")
                    # print(
                    #     f"impulse_position + (0,) * (unit_impulse_inxyz.s.ndim - 2){impulse_position + (0,) * (unit_impulse_inxyz.s.ndim - 2)}")

                unit_impulse_response_inxyz.sg.fill(0)
                if operator is None:
                    self.apply_system_matrix_mugrid(
                        material_data_field=reference_material_data_ijkl,
                        input_field_inxyz=unit_impulse_inxyz,
                        output_field_inxyz=unit_impulse_response_inxyz,
                        formulation=formulation)
                else:
                    operator(
                        input_field_inxyz=unit_impulse_inxyz,
                        output_field_inxyz=unit_impulse_response_inxyz)

                self.fft.communicate_ghosts(unit_impulse_response_inxyz)
                # print(f"unit_impulse_response_inxyz {unit_impulse_response_inxyz.s[...]}")

                unit_impulse_response_inqks.sg.fill(0)
                # self.fft.fft(unit_impulse_response_inxyz, unit_impulse_response_inqks)

                # Forward FFT: real -> Fourier
                # self.multinodal_fft(real_field=unit_impulse_response_inxyz,
                #                     fourier_field=unit_impulse_response_inqks)
                self.fft.fft(unit_impulse_response_inxyz, unit_impulse_response_inqks)
                # print(f"unit_impulse_response_inqks {unit_impulse_response_inqks.s[...]}")
                # Unpack tuple to get normal indexing:
                i, n = impulse_position
                preconditioner_diagonals_ininqks.s[i, n, ...] = np.copy(unit_impulse_response_inqks.s[...])

            # THE SIZE OF DIAGONAL IS [nb_unit_dofs,nb_unit_dofs,nb_unit_dofs,nb_unit_dofs, xyz]
            # compute inverse of diagonals
            original_shape_ininqks = preconditioner_diagonals_ininqks.s.shape

            # prec_diagonals_ijqks = np.squeeze(preconditioner_diagonals_ininqks.s, axis=(0, 2))

            # Reshape the array to (n_u_dofs, n_u_dofs, ndof) for easier processing
            # reshaped_matrices = preconditioner_diagonals_ininqks.s.reshape(nb_dofs_per_voxel, nb_dofs_per_voxel, -1)
            reshaped_matrices = preconditioner_diagonals_ininqks.s.reshape(nb_dofs_per_voxel, nb_dofs_per_voxel, -1)
            # d mean nb_dofs_per_voxel
            # Transpose to shape (N, n_dof, n_dof) for batch inversion
            G_batch = reshaped_matrices.transpose(2, 0, 1)  # shape: (N, d, d)
            # Invert each matrix using np.linalg.inv (vectorized)
            if np.any(np.all(self.fft.icoords == 0, axis=0)):  # check if the core has zero mode
                # with several nodes per pixel K_hat(0) is singular but not zero
                # (relative motion of the sub-nodes is resisted) -> pseudo-inverse
                G_batch[0, ...] = np.linalg.pinv(G_batch[0, ...],
                                                 rcond=1e-8)  # shape: (N, d, d) # do not invert zero mode

                G_batch[1:, ...] = np.linalg.inv(G_batch[1:, ...])  # shape: (N, d, d) # do not invert zero mode
            else:
                G_batch[0:, ...] = np.linalg.inv(G_batch[0:, ...])  # shape: (N, d, d)

            # Reshape the result back to the original shape
            G_diag_ijxy = G_batch.transpose(1, 2, 0).reshape(original_shape_ininqks)

            preconditioner_diagonals_ininqks.s[...] = G_diag_ijxy.reshape(original_shape_ininqks)[...]

        return preconditioner_diagonals_ininqks

    def get_preconditioner_Jacobi_mugrid(self, material_data_field_ijklqxyz: Field = None,
                                         constitutive: callable = None,
                                         formulation=None,
                                         **kwargs):
        """Matrix-free Jacobi preconditioner ``diag(K)^{-1/2}`` via Dirac combs.

        Parameters
        ----------
        material_data_field_ijklqxyz : muGrid Field, optional
            Material tangent; if given, ``K`` is the linear system matrix
            (:meth:`apply_system_matrix_mugrid`).
        constitutive : callable, optional
            Used (2D only) when no material data is given; ``K`` is then the
            operator of :meth:`apply_system_matrix_mugrid_explicit_stress`.
        formulation : str, optional
            Passed to the system-matrix routine.
        **kwargs
            ``zero_threshold`` (float, default 1.0): value stored where the
            diagonal entry is exactly zero (e.g. void pixels).

        Returns
        -------
        diagonal_inxyz : muGrid Field [f, n, x, y, z]
            Field ``'jacobi_diagonal_inxyz'`` containing ``1/sqrt(K_ii)``,
            to be used as a symmetric (split) scaling ``D^{-1/2} K D^{-1/2}``.

        Notes
        -----
        The stencil of ``K`` for nearest-neighbour elements couples a node
        only with nodes at distance <= 1 pixel. Hence a "Dirac comb" with ones
        on every second pixel in each direction (one colour of a
        ``2^d``-colouring) and on one component ``f`` returns, at the comb
        points, exactly the diagonal entries ``K_ii``: all other impulses are
        >= 2 pixels away. ``f * 2^d`` operator applications give the full
        diagonal. Requires even grid sizes for the colouring to be periodic,
        and assumes one node per pixel (only ``n = 0`` is filled). The 3D
        branch always uses ``material_data_field_ijklqxyz`` (``constitutive``
        is ignored).
        """
        # return diagonals of system matrix
        # unit_impulse [f,n,x,y,z]
        # for every type of degree of freedom DOF, there is one diagonal of preconditioner matrix
        # diagonals_in_Fourier_space [f,n,f,n][0,0,0]  # all DOFs in first pixel

        threshold = kwargs.get('zero_threshold', 1.)

        diagonal_inxyz = self.get_unknown_size_field(name='jacobi_diagonal_inxyz')
        dirac_comb_inxyz = self.get_unknown_size_field(name='jacobi_dirac_comb_inxyz_temporary')
        dirac_comb_response_inxyz = self.get_unknown_size_field(name='jacobi_dirac_comb_response_inxyz_temporary')

        if self.domain_dimension == 2:
            for d_i in range(self.cell.unknown_shape[0]):
                for x_i in range(2):
                    for y_i in range(2):
                        dirac_comb_inxyz.s.fill(0)
                        dirac_comb_inxyz.s[d_i, 0, x_i::2, y_i::2] = 1.0
                        # compute response of diract comb
                        if material_data_field_ijklqxyz is not None:
                            self.apply_system_matrix_mugrid(material_data_field=material_data_field_ijklqxyz,
                                                            input_field_inxyz=dirac_comb_inxyz,
                                                            output_field_inxyz=dirac_comb_response_inxyz,
                                                            formulation=formulation
                                                            )
                        elif constitutive is not None:
                            self.apply_system_matrix_mugrid_explicit_stress(constitutive=constitutive,
                                                                            input_field_inxyz=dirac_comb_inxyz,
                                                                            output_field_inxyz=dirac_comb_response_inxyz,
                                                                            formulation=formulation
                                                                            )

                        # at the comb points the response equals K_ii -> store 1/sqrt(K_ii)
                        # (np.where evaluates both branches: zero entries may emit a divide warning)
                        diagonal_inxyz.s[d_i, 0, x_i::2, y_i::2] = np.where(
                            dirac_comb_response_inxyz.s[d_i, 0, x_i::2, y_i::2] != 0.,
                            1 / np.sqrt(dirac_comb_response_inxyz.s[d_i, 0, x_i::2, y_i::2]),
                            threshold
                        )

        elif self.domain_dimension == 3:
            for d_i in range(self.cell.unknown_shape[0]):
                for x_i in range(2):
                    for y_i in range(2):
                        for z_i in range(2):
                            dirac_comb_inxyz.s.fill(0)
                            dirac_comb_inxyz.s[d_i, 0, x_i::2, y_i::2, z_i::2] = 1.0
                            # compute response of diract comb
                            self.apply_system_matrix_mugrid(material_data_field=material_data_field_ijklqxyz,
                                                            input_field_inxyz=dirac_comb_inxyz,
                                                            output_field_inxyz=dirac_comb_response_inxyz,
                                                            formulation=formulation)

                            diagonal_inxyz.s[d_i, 0, x_i::2, y_i::2, z_i::2] = np.where(
                                dirac_comb_response_inxyz.s[d_i, 0, x_i::2, y_i::2, z_i::2] != 0.,
                                1 / np.sqrt(dirac_comb_response_inxyz.s[d_i, 0, x_i::2, y_i::2, z_i::2]),
                                threshold
                            )

        return diagonal_inxyz

    def apply_preconditioner_NEW(self, preconditioner_Fourier_fnfnqks, nodal_field_fnxyz):
        """Apply preconditioner to nodal field using FFT.

        Parameters
        ----------
        preconditioner_Fourier_fnfnqks : ndarray
            Preconditioner diagonals in Fourier space [f,n,f,n,q,k,s]
        nodal_field_fnxyz : ndarray
            Input nodal field [f,n,x,y,z]

        Returns
        -------
        ndarray
            Preconditioned field [f,n,x,y,z]

        Notes
        -----
        ``z = F^{-1}[ G_hat(k) . F[r](k) ]`` with the block product
        ``z_hat_ab = G_hat_abcd r_hat_cd``. Legacy: uses ``.sg`` (ghosted
        arrays) throughout and a Fourier field without the ``nodal_points``
        sub-point; returns a view of the internal scratch field
        ``'temp_output_field_in_apply_preconditioner_fnxyz'``.

        Currently does not run with the present muGrid (``can't set attribute
        'sg'``), and its ``'abcd...,cd...->ab...'`` applies the transposed
        block compared with :meth:`apply_preconditioner_mugrid`
        (``'cdab...'``), which matters for elements with several nodes per
        pixel. Kept for experiments with the Green-Jacobi preconditioner.
        """

        # allocate field
        temp_nodal_field_fnxyz = self.get_unknown_size_field(name='temp_nodal_field_in_apply_preconditioner_fnxyz')
        temp_ouput_field_fnxyz = self.get_unknown_size_field(name='temp_output_field_in_apply_preconditioner_fnxyz')

        ffield_fnqks = self.ffield_collection.complex_field(
            name='temp_F_nodal_field_in_apply_preconditioner_fnxyz',  # name of the field
            components=(*self.cell.unknown_shape,))  # sub-point type

        if isinstance(nodal_field_fnxyz, np.ndarray):
            temp_nodal_field_fnxyz.sg[...] = nodal_field_fnxyz
        else:
            temp_nodal_field_fnxyz.sg[...] = nodal_field_fnxyz.sg

        # FFTn of input array
        self.fft.fft(temp_nodal_field_fnxyz, ffield_fnqks)

        # multiplication with a diagonals of preconditioner
        ffield_fnqks.sg[...] = np.einsum('abcd...,cd...->ab...', preconditioner_Fourier_fnfnqks.sg, ffield_fnqks.sg)

        # normalization
        ffield_fnqks.sg *= self.fft.normalisation
        # iFFTn
        self.fft.ifft(ffield_fnqks, temp_ouput_field_fnxyz)

        return temp_ouput_field_fnxyz.sg

    def apply_preconditioner_mugrid(self, preconditioner_Fourier_fnfnqks,
                                    input_nodal_field_fnxyz,
                                    output_nodal_field_fnxyz):
        """Apply preconditioner to nodal field using FFT (muGrid version).

        Parameters
        ----------
        preconditioner_Fourier_fnfnqks : muGrid Field
            Preconditioner diagonals in Fourier space [f,n,f,n,q,k,s]
        input_nodal_field_fnxyz : muGrid Field
            Input nodal field [f,n,x,y,z]
        output_nodal_field_fnxyz : muGrid Field
            Output nodal field [f,n,x,y,z] (modified in-place)

        Returns
        -------
        None
            Modifies output_nodal_field_fnxyz in-place

        Notes
        -----
        Applies ``M^{-1} r = F^{-1}[ G_hat(k) r_hat(k) ]`` where ``G_hat`` is
        the output of :meth:`get_preconditioner_Green_mugrid`. That routine
        stores the response to impulse dof ``(c, d)`` in ``G[c, d, a, b]``,
        hence the contraction ``'cdab...,cd...->ab...'``
        (``z_hat_ab = sum_cd G_cdab r_hat_cd``). The FFT pair is
        unnormalised, so the result is multiplied by ``fft.normalisation``
        (= 1/N_total). Collective MPI operation (parallel FFT).
        """

        ffield_fnqks = self.ffield_collection.complex_field(
            name='temp_F_nodal_field_in_apply_preconditioner_fnxyz',  # name of the field
            components=(*self.cell.unknown_shape,),  # shape of components
            sub_pt='nodal_points')  # sub-point type

        if isinstance(input_nodal_field_fnxyz, np.ndarray):
            raise TypeError("apply_preconditioner_mugrid does not support ndarray")

        # FFTn of input array

        self.fft.fft(input_nodal_field_fnxyz, ffield_fnqks)

        # multiplication with a diagonals of preconditioner
        ffield_fnqks.s[...] = np.einsum('cdab...,cd...->ab...', preconditioner_Fourier_fnfnqks.s, ffield_fnqks.s)

        # iFFTn
        self.fft.ifft(ffield_fnqks, output_nodal_field_fnxyz)
        # normalization
        output_nodal_field_fnxyz.s[...] *= self.fft.normalisation

    def apply_preconditioner_Green_Jacobi_full(self, green_fnfnqks,
                                               jacobi_half_fnfnxyz,
                                               nodal_field_fnxyz):
        """Apply combined Green and Jacobi preconditioner.

        Parameters
        ----------
        green_fnfnqks : ndarray
            Green preconditioner in Fourier space [f,n,f,n,q,k,s]
        jacobi_half_fnfnxyz : ndarray
            Jacobi preconditioner blocks [f,n,f,n,x,y,z]
        nodal_field_fnxyz : ndarray
            Input nodal field [f,n,x,y,z]

        Returns
        -------
        ndarray
            Preconditioned field [f,n,x,y,z]

        Notes
        -----
        Computes ``J^{1/2} G J^{1/2} r`` with ``J^{1/2}`` the per-pixel block
        Jacobi scaling and ``G`` the Green preconditioner applied by
        :meth:`apply_preconditioner_NEW` (legacy numpy interface). The
        block-Jacobi input used to come from ``get_preconditioner_Jacoby_fast``
        (``prec_type='full'``), which was removed because it no longer ran;
        this method is currently unused.
        """
        # apply Jacobi 1/2 --- multiplication with a left diagonal blocks of preconditioner
        nodal_field_fnxyz = np.einsum('abcd...,cd...->ab...', jacobi_half_fnfnxyz, nodal_field_fnxyz)

        # apply Green preconditioner using FFT
        nodal_field_fnxyz = self.apply_preconditioner_NEW(
            preconditioner_Fourier_fnfnqks=green_fnfnqks,
            nodal_field_fnxyz=nodal_field_fnxyz)

        # apply Jacobi 1/2 --- multiplication with a right diagonal blocks of preconditioner
        nodal_field_fnxyz = np.einsum('abcd...,cd...->ab...', jacobi_half_fnfnxyz, nodal_field_fnxyz)
        return nodal_field_fnxyz

    def apply_system_matrix_mugrid(self,
                                   material_data_field,
                                   input_field_inxyz,
                                   output_field_inxyz,
                                   formulation=None,
                                   **kwargs):
        """Matrix-free action of the (linear) system matrix ``K u = B^T W C B u``.

        This is the operator passed to the (preconditioned) conjugate
        gradient solver.

        Parameters
        ----------
        material_data_field : muGrid Field or ndarray
            Material tangent ``C`` at quadrature points ([d,d,q,...] or
            [d,d,d,d,q,...]) or a single reference tensor ([d,d] /
            [d,d,d,d]), see :meth:`apply_material_data_mugrid`.
        input_field_inxyz : muGrid Field [f, n, x, y, z]
            Vector ``u`` (its ghost layers are refreshed).
        output_field_inxyz : muGrid Field [f, n, x, y, z]
            Result ``K u``, written in-place.
        formulation : str, optional
            ``'small_strain'`` uses the symmetrised gradient
            ``eps = sym(grad u)``; otherwise the full gradient is used.
        **kwargs
            Ignored.

        Returns
        -------
        None

        Notes
        -----
        Steps: ``g = B u`` (gradient at quadrature points) ->
        ``sigma = C : g`` -> ``f = B^T W sigma``. Uses the scratch field
        ``'grad_field_temporary'``. Collective MPI operation (ghost exchange).
        """

        if isinstance(input_field_inxyz, np.ndarray):
            raise TypeError("apply_system_matrix_mugrid does not support ndarray")

        self.fft.communicate_ghosts(input_field_inxyz)
        # allocate temporary fields
        gradient_ijqxyz = self.get_gradient_size_field(name='grad_field_temporary')

        # g = B u   (or sym(B u))
        if np.all(formulation == 'small_strain'):
            self.apply_gradient_operator_symmetrized_mugrid(u_inxyz=input_field_inxyz,
                                                            grad_u_ijqxyz=gradient_ijqxyz)

        else:
            self.apply_gradient_operator_mugrid(u_inxyz=input_field_inxyz,
                                                grad_u_ijqxyz=gradient_ijqxyz)

        # compute stress/flux field
        self.apply_material_data_mugrid(material_data=material_data_field,
                                        gradient_field=gradient_ijqxyz)

        self.fft.communicate_ghosts(gradient_ijqxyz)
        self.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=gradient_ijqxyz,
                                                       div_u_fnxyz=output_field_inxyz,
                                                       apply_weights=True)

    def apply_system_matrix_mugrid_deformed_grid(self,
                                                 material_data_field,
                                                 input_field_inxyz,
                                                 output_field_inxyz,
                                                 det_of_deformation_gradient,
                                                 inv_of_deformation_gradient,
                                                 formulation=None,
                                                 **kwargs):
        '''System-matrix action on a deformed (mapped) grid.

        Parameters
        ----------
        material_data_field : muGrid Field or ndarray
            Material tangent at quadrature points (see
            :meth:`apply_material_data_mugrid`).
        input_field_inxyz : muGrid Field [f, n, x, y, z]
            Vector ``u`` defined on the regular reference grid.
        output_field_inxyz : muGrid Field [f, n, x, y, z]
            Result, written in-place.
        det_of_deformation_gradient : muGrid Field
            ``det(F_q)`` per quadrature point (``.s`` shape [1, 1, q, x, y, z]).
        inv_of_deformation_gradient : muGrid Field [d, d, q, x, y, z]
            ``F_q^{-1}``.
        formulation : str, optional
            ``'small_strain'`` symmetrises the transformed gradient.
        **kwargs
            Ignored.

        Returns
        -------
        None

        Notes
        -----
        The physical grid is the image ``x = phi(X)`` of the regular grid
        ``X``, with deformation gradient ``F = dphi/dX``. Pulling the weak
        form back to ``X`` (``grad_x = grad_X . F^{-1}``, ``dx = det F dX``)::

            K u = B^T W [ det(F) * (C : sym(B u . F^{-1})) . F^{-T} ]

        i.e. the standard regular-grid operators with a modified, spatially
        varying "material" -- this is what enables FFT preconditioning on
        deformed (e.g. boundary-fitted) grids.
        '''
        # aliasing
        det_F = det_of_deformation_gradient
        inv_F = inv_of_deformation_gradient

        if isinstance(input_field_inxyz, np.ndarray):
            raise TypeError("apply_system_matrix_mugrid does not support ndarray")

        self.fft.communicate_ghosts(input_field_inxyz)
        # allocate temporary fields
        gradient_ijqxyz = self.get_gradient_size_field(name='grad_field_temporary')
        self.apply_gradient_operator_mugrid(u_inxyz=input_field_inxyz,
                                            grad_u_ijqxyz=gradient_ijqxyz)

        # apply deformation gradient ε^q ← sym( (∇ũ)^q · (F^q)^-1 )   // q-th transformed gradient
        gradient_ijqxyz.s[...] = np.einsum('ij...,jk...->ik...', gradient_ijqxyz.s[...], inv_F.s[...])
        # symmetrization for small-strain elasticity
        if np.all(formulation == 'small_strain'):
            #  symmetrize it
            gradient_ijqxyz.s[...] = (gradient_ijqxyz.s + np.swapaxes(gradient_ijqxyz.s, 0, 1)) / 2

        # compute stress/flux field : σ^q ← C^q : ε^q              // constitutive model
        self.apply_material_data_mugrid(material_data=material_data_field,
                                        gradient_field=gradient_ijqxyz)

        # w_q * div( det(F^q) * σ^q · (F^q)^-T ) // transformed divergence
        gradient_ijqxyz.s[...] = np.einsum('ij...,kj...->ik...', gradient_ijqxyz.s[...], inv_F.s[...]) * det_F.s[None, None, ...]

        self.fft.communicate_ghosts(gradient_ijqxyz)
        self.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=gradient_ijqxyz,
                                                       div_u_fnxyz=output_field_inxyz,
                                                       apply_weights=True)

    def apply_system_matrix_mugrid_explicit_stress(self,
                                                   constitutive: callable,
                                                   input_field_inxyz: Field,
                                                   output_field_inxyz: Field,
                                                   formulation=None,
                                                   **kwargs):
        """Operator ``u -> B^T W sigma(B u)`` with an explicit constitutive law.

        Parameters
        ----------
        constitutive : callable
            ``constitutive(gradient_field, stress_field)``; called here
            in-place (same field for both arguments). For a nonlinear
            problem this should be the *tangent* applied to an increment,
            so that the operator is linear (e.g. inside Newton/CG).
        input_field_inxyz : muGrid Field [f, n, x, y, z]
        output_field_inxyz : muGrid Field [f, n, x, y, z]
            Result, written in-place.
        formulation : str, optional
            ``'small_strain'`` -> symmetrised gradient.
        **kwargs
            Ignored.

        Returns
        -------
        None
        """
        if isinstance(input_field_inxyz, np.ndarray):
            raise TypeError("apply_system_matrix_mugrid does not support ndarray")

        self.fft.communicate_ghosts(input_field_inxyz)
        # allocate temporary fields
        gradient_ijqxyz = self.get_gradient_size_field(name='grad_field_temporary')

        if np.all(formulation == 'small_strain'):
            self.apply_gradient_operator_symmetrized_mugrid(u_inxyz=input_field_inxyz,
                                                            grad_u_ijqxyz=gradient_ijqxyz)

        else:
            self.apply_gradient_operator_mugrid(u_inxyz=input_field_inxyz,
                                                grad_u_ijqxyz=gradient_ijqxyz)

        # compute stress/flux field
        constitutive(gradient_ijqxyz, gradient_ijqxyz)

        self.fft.communicate_ghosts(gradient_ijqxyz)
        self.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=gradient_ijqxyz,
                                                       div_u_fnxyz=output_field_inxyz,
                                                       apply_weights=True)

    def integrate_over_cell(self, stress_field):
        """Integrate a gradient-shaped numpy array over the (global) cell.

        Parameters
        ----------
        stress_field : ndarray [i, j, q, x, y, z]

        Returns
        -------
        ndarray [i, j]
            ``sum_{q, pixels} w_q sigma_ij`` reduced over all MPI ranks
            (not divided by the cell volume).
        """
        # compute integral of stress field over the domain: int sigma d Omega = sum x_q *w_q

        stress_field = np.einsum('ijq...,q->ijq...', stress_field, self.quadrature_weights)
        # TODO change this to muGRID
        integral = self.mpi_reduction.sum(stress_field, axis=tuple(range(-self.domain_dimension - 1, 0)))  #
        return integral

    def integrate_over_cell_mugrid(self, stress_field):
        """Integrate a gradient-shaped muGrid field over the (global) cell.

        Parameters
        ----------
        stress_field : muGrid Field [i, j, q, x, y, z]
            **Modified in place**: multiplied by the quadrature weights.

        Returns
        -------
        ndarray [i, j]
            ``sum_{q, pixels} w_q sigma_ij`` reduced over all MPI ranks
            (not divided by the cell volume).
        """
        # compute integral of stress field over the domain: int sigma d Omega = sum x_q *w_q

        stress_field.s[...] = np.einsum('ijq...,q->ijq...', stress_field.s, self.quadrature_weights)
        integral = self.mpi_reduction.sum(stress_field.s, axis=tuple(range(-self.domain_dimension - 1, 0)))  #

        return integral

    def get_unknown_size_field(self, name):
        """Create zero field with unknown shape.

        Parameters
        ----------
        name : str
            Field name identifier

        Returns
        -------
        muGrid Field
            Zero field with shape matching problem unknowns [f,n,x,y,z]

        Notes
        -----
        Fields are registered by ``name`` in ``self.field_collection``. The
        code throughout this module relies on repeated requests with the
        same name returning the *same* (already existing) field -- e.g. the
        scratch fields ``'grad_field_temporary'``. Such a field is only zero
        on first creation and keeps its old content afterwards, and two
        callers using the same name share memory. The same holds for all
        ``get_*_field`` factories below.
        """
        u_inxyz = self.field_collection.real_field(
            name=name,  # name of the field
            components=(*self.cell.unknown_shape,),  # shape of components
            sub_pt='nodal_points')  # sub-point type
        return u_inxyz

    def get_custom_size_nodal_field(self, name, shape):
        """Create zero nodal field with custom shape.

        Parameters
        ----------
        name : str
            Field name identifier
        shape : tuple
            Shape of field components

        Returns
        -------
        muGrid Field
            Zero nodal field with specified shape
        """
        u_inxyz = self.field_collection.real_field(
            name=name,  # name of the field
            components=shape,  # shape of components
            sub_pt='nodal_points')  # sub-point type

        return u_inxyz

    def get_custom_size_quad_field(self, name, shape):
        """Create zero quadrature field with custom shape.

        Parameters
        ----------
        name : str
            Field name identifier
        shape : tuple
            Shape of field components

        Returns
        -------
        muGrid Field
            Zero quadrature field with specified shape
        """
        u_inxyz = self.field_collection.real_field(
            name=name,  # name of the field
            components=shape,  # shape of components
            sub_pt='quad_points')  # sub-point type

        return u_inxyz

    def get_gradient_size_field(self, name):
        """Create zero field for gradient of unknowns.

        Parameters
        ----------
        name : str
            Field name identifier

        Returns
        -------
        muGrid Field
            Zero gradient field at quadrature points [i,j,q,x,y,z]
        """
        grad_u_ijqxyz = self.field_collection.real_field(
            name=name,  # name of the field
            components=(*self.cell.gradient_shape,),  # shape of components
            sub_pt='quad_points'  # sub-point type
        )
        return grad_u_ijqxyz

    def get_temperature_sized_field(self, name):
        """Create zero temperature field (conductivity problem).

        Parameters
        ----------
        name : str
            Field name identifier

        Returns
        -------
        muGrid Field
            Zero temperature field [1,n,x,y,z]
        """
        if not self.cell.problem_type == 'conductivity':
            warnings.warn(
                'Cell problem type is {}. But temperature sized field  is returned !!!'.format(self.cell.problem_type))

        u_inxyz = self.field_collection.real_field(
            name=name,  # name of the field
            components=(*self.cell.unknown_shape,),  # shape of components
            sub_pt='nodal_points'  # sub-point type
        )

        return u_inxyz

    def get_scalar_field(self, name):
        """Create zero scalar field.

        Parameters
        ----------
        name : str
            Field name identifier

        Returns
        -------
        muGrid Field
            Zero scalar field [1,n,x,y,z]
        """
        return self.field_collection.real_field(
            name=name,  # name of the field
            components=(1,),  # shape of components
            sub_pt='nodal_points'  # sub-point type
        )

    def get_gradient_of_scalar_field(self, name):
        """Create zero gradient of scalar field.

        Parameters
        ----------
        name : str
            Field name identifier

        Returns
        -------
        muGrid Field
            Zero gradient field [1,d,q,x,y,z]
        """
        return self.field_collection.real_field(
            name=name,  # name of the field
            components=(1, self.cell.domain_dimension,),  # shape of components
            sub_pt='quad_points'  # sub-point type
        )

    def get_quad_field_scalar(self, name):
        """Create zero quadrature scalar field.

        Parameters
        ----------
        name : str
            Field name identifier

        Returns
        -------
        muGrid Field
            Zero scalar field at quadrature points [1,1,q,x,y,z]
        """
        return self.field_collection.real_field(
            name=name,  # name of the field
            components=(1, 1),  # shape of components
            sub_pt='quad_points'  # sub-point type
        )

    def get_temperature_gradient_size_field(self, name):
        """Create zero temperature gradient field.

        Parameters
        ----------
        name : str
            Field name identifier

        Returns
        -------
        muGrid Field
            Zero gradient field [d,q,x,y,z]
        """
        if not self.cell.problem_type == 'conductivity':
            warnings.warn(
                'Cell problem type is {}. But temperature gradient  sized field  is returned !!!'.format(
                    self.cell.problem_type))

        # Get a tensor-field (for example to represent the heat gradient)

        grad_u_ijqxyz = self.field_collection.real_field(
            name=name,  # name of the field
            components=(*self.cell.gradient_shape,),  # shape of components
            sub_pt='quad_points'  # sub-point type
        )
        return grad_u_ijqxyz

    def get_temperature_hessian_size_field(self, name):
        """Create zero temperature Hessian field.

        Parameters
        ----------
        name : str
            Field name identifier

        Returns
        -------
        muGrid Field
            Zero Hessian field [1,d,d,q,x,y,z]
        """
        if not self.cell.problem_type == 'conductivity':
            warnings.warn(
                'Cell problem type is {}. But temperature Hessian  sized field  is returned !!!'.format(
                    self.cell.problem_type))
        shape_of_hessian_of_scalar = np.array([1, self.domain_dimension, self.domain_dimension],
                                              dtype=int)
        hess_u_ijkqxyz = self.field_collection.real_field(
            name=name,  # name of the field
            components=(*shape_of_hessian_of_scalar,),  # shape of components
            sub_pt='quad_points'  # sub-point type
        )
        return hess_u_ijkqxyz

    def get_temperature_hessian_size_field_mugrid_compatible(self, name):
        """Create zero temperature Hessian field (muGrid compatible layout).

        Parameters
        ----------
        name : str
            Field name identifier

        Returns
        -------
        muGrid Field
            Zero Hessian field with flattened indices [1,d*d,q,x,y,z]
        """
        if not self.cell.problem_type == 'conductivity':
            warnings.warn(
                'Cell problem type is {}. But temperature Hessian  sized field  is returned !!!'.format(
                    self.cell.problem_type))
        shape_of_hessian_of_scalar = np.array([1, self.domain_dimension * self.domain_dimension],
                                              dtype=int)
        hess_u_iJqxyz = self.field_collection.real_field(
            name=name,  # name of the field
            components=(*shape_of_hessian_of_scalar,),  # shape of components
            sub_pt='quad_points'  # sub-point type
        )
        # her J is a composition of jk indices. J is flattened jk
        return hess_u_iJqxyz

    def get_displacement_hessian_size_field(self, name):
        """Create zero displacement Hessian field.

        Parameters
        ----------
        name : str
            Field name identifier

        Returns
        -------
        muGrid Field
            Zero Hessian field [d,d,d,q,x,y,z]
        """
        if not self.cell.problem_type == 'elasticity':
            warnings.warn(
                'Cell problem type is {}. But elasticity Hessian  sized field  is returned !!!'.format(
                    self.cell.problem_type))
        shape_of_hessian_of_scalar = np.array([self.domain_dimension, self.domain_dimension, self.domain_dimension],
                                              dtype=int)
        hess_u_ijkqxyz = self.field_collection.real_field(
            name=name,  # name of the field
            components=(*shape_of_hessian_of_scalar,),  # shape of components
            sub_pt='quad_points'  # sub-point type
        )
        return hess_u_ijkqxyz

    def get_displacement_hessian_size_field_mugrid_compatible(self, name):
        """Create zero displacement Hessian field (muGrid compatible layout).

        Parameters
        ----------
        name : str
            Field name identifier

        Returns
        -------
        muGrid Field
            Zero Hessian field with flattened indices [d,d*d,q,x,y,z]
        """
        if not self.cell.problem_type == 'elasticity':
            warnings.warn(
                'Cell problem type is {}. But displacement Hessian  sized field  is returned !!!'.format(
                    self.cell.problem_type))
        shape_of_hessian_of_scalar = np.array([self.domain_dimension, self.domain_dimension * self.domain_dimension],
                                              dtype=int)
        hess_u_iJqxyz = self.field_collection.real_field(
            name=name,  # name of the field
            components=(*shape_of_hessian_of_scalar,),  # shape of components
            sub_pt='quad_points'  # sub-point type
        )
        # her J is a composition of jk indices. J is flattened jk
        return hess_u_iJqxyz

    def get_displacement_laplacian_at_quad_field(self, name):
        """Create zero displacement Laplacian field.

        Parameters
        ----------
        name : str
            Field name identifier

        Returns
        -------
        muGrid Field
            Zero Laplacian field [d,1,q,x,y,z]
        """
        if not self.cell.problem_type == 'elasticity':
            warnings.warn(
                'Cell problem type is {}. But elasticity Laplacian  sized field  is returned !!!'.format(
                    self.cell.problem_type))

        lap_u_ikqxyz = self.field_collection.real_field(
            name=name,  # name of the field
            components=(self.domain_dimension, 1,),  # shape of components # for temperature, it will be just 1,1
            sub_pt='quad_points'  # sub-point type
        )
        return lap_u_ikqxyz

    def get_temperature_material_data_size_field(self):
        """Create zero material data field for conductivity problem.

        Returns
        -------
        ndarray
            Zero field [d,d,q,x,y,z] -- a plain numpy array (not a muGrid
            field), local subdomain size, no ghost layers.
        """
        if not self.cell.problem_type == 'conductivity':
            warnings.warn(
                'Cell problem type is {}. But temperature material data  sized field  is returned !!!'.format(
                    self.cell.problem_type))

        return np.zeros(
            [self.domain_dimension, self.domain_dimension, self.nb_quad_points_per_pixel, *self.nb_of_pixels])

    def get_displacement_sized_field(self, name):
        """Create zero displacement field (elasticity problem).

        Parameters
        ----------
        name : str
            Field name identifier

        Returns
        -------
        muGrid Field
            Zero displacement field [d,n,x,y,z]
        """
        if not self.cell.problem_type == 'elasticity':
            warnings.warn(
                'Cell problem type is {}. But displacement sized field  is returned !!!'.format(self.cell.problem_type))

        u_inxyz = self.field_collection.real_field(
            name=name,  # name of the field
            components=(self.cell.domain_dimension,),  # shape of components
            sub_pt='nodal_points'  # sub-point type
        )

        return u_inxyz

    def get_displacement_gradient_sized_field(self, name):
        """Create zero displacement gradient (strain) field.

        Parameters
        ----------
        name : str
            Field name identifier

        Returns
        -------
        muGrid Field
            Zero strain field [d,d,q,x,y,z]
        """
        if not self.cell.problem_type == 'elasticity':
            warnings.warn(
                'Cell problem type is {}. But displacement gradient  sized field  is returned !!!'.format(
                    self.cell.problem_type))

        # Get a tensor-field (for example to represent the strain)
        grad_u_ijqxyz = self.field_collection.real_field(
            name=name,  # name of the field
            components=(self.cell.domain_dimension, self.cell.domain_dimension,),  # shape of components
            sub_pt='quad_points'  # sub-point type
        )

        return grad_u_ijqxyz

    def get_strain_sized_field(self, name):
        """Create zero strain field.

        Parameters
        ----------
        name : str
            Field name identifier

        Returns
        -------
        muGrid Field
            Zero strain field [d,d,q,x,y,z]
        """
        grad_u_ijqxyz = self.field_collection.real_field(
            name=name,  # name of the field
            components=(self.cell.domain_dimension, self.cell.domain_dimension,),  # shape of components
            sub_pt='quad_points'  # sub-point type
        )

        return grad_u_ijqxyz

    def get_stress_sized_field(self, name):
        """Create zero stress field.

        Parameters
        ----------
        name : str
            Field name identifier

        Returns
        -------
        muGrid Field
            Zero stress field [d,d,q,x,y,z]
        """
        stress_ijqxyz = self.field_collection.real_field(
            name=name,  # name of the field
            components=(self.cell.domain_dimension, self.cell.domain_dimension,),  # shape of components
            sub_pt='quad_points'  # sub-point type
        )

        return stress_ijqxyz

    def get_material_data_size_field_mugrid(self, name):
        """Create zero material data field.

        Parameters
        ----------
        name : str
            Field name identifier

        Returns
        -------
        muGrid Field
            Zero material data field at quadrature points
        """
        material_data_ijqxyz = self.field_collection.real_field(
            name=name,  # name of the field
            components=(*self.cell.material_data_shape,),  # shape of components
            sub_pt='quad_points'  # sub-point type
        )

        return material_data_ijqxyz

    def get_material_data_size_field(self, name):
        """Create zero material data field (alias).

        Parameters
        ----------
        name : str
            Field name identifier

        Returns
        -------
        muGrid Field
            Zero material data field
        """
        return self.get_material_data_size_field_mugrid(name)

    # Convenience aliases for _mugrid methods
    def get_rhs(self, **kwargs):
        """Get RHS vector (alias for get_rhs_mugrid)."""
        return self.get_rhs_mugrid(**kwargs)

    def get_macro_gradient_field(self, **kwargs):
        """Get macro gradient field (alias for get_macro_gradient_field_mugrid)."""
        return self.get_macro_gradient_field_mugrid(**kwargs)

    def get_homogenized_stress(self, **kwargs):
        """Get homogenized stress (alias for get_homogenized_stress_mugrid)."""
        return self.get_homogenized_stress_mugrid(**kwargs)

    def get_rhs_explicit_stress(self, **kwargs):
        """Get RHS with explicit stress (alias for get_rhs_explicit_stress_mugrid)."""
        return self.get_rhs_explicit_stress_mugrid(**kwargs)

    def assert_material_symmetry(self, material_data, minor=False, name='material data',
                                 rtol=1e-10):
        """Check that material data has the symmetries the solvers rely on.

        All contractions use the standard convention ``sigma_ij = C_ijkl eps_kl``
        (conductivity: ``q_i = A_ij g_j``). The methods built on top of them
        need:

        * **major symmetry** ``C_ijkl = C_klij`` (conductivity: ``A_ij = A_ji``):
          the system matrix ``K = B^T W C B`` is then symmetric, which conjugate
          gradients requires, and the adjoint operator ``K^T`` equals ``K``;
        * **minor symmetry** ``C_ijkl = C_jikl`` (small strain only): the stress
          is symmetric, so ``B^T W sigma`` equals the symmetric-gradient form
          ``B_sym^T W sigma`` of the weak form.

        The check runs where material data enters a solve (right-hand side,
        homogenized stress, preconditioner, sensitivities), not inside the CG
        loop. It compares one index-pair slice at a time, so it needs no
        full-size temporary, and costs about one pass over the data. Set
        ``discretization.check_material_symmetry = False`` to skip it.

        Parameters
        ----------
        material_data : muGrid Field or ndarray
            ``[d, d(, d, d), ...]``: a field over quadrature points and pixels
            or a constant tensor.
        minor : bool, optional
            Also check minor symmetry (elasticity, small strain).
        name : str, optional
            Used in the error message.
        rtol : float, optional
            Tolerance relative to ``max |C|``.

        Raises
        ------
        ValueError
            On every MPI rank, if a required symmetry is violated anywhere.
        """
        if not self.check_material_symmetry:
            return
        C = material_data.s if isinstance(material_data, Field) else np.asarray(material_data)
        d = self.domain_dimension
        is_elasticity = self.cell.problem_type == 'elasticity'

        # index pairs (a, b) that must hold equal values: C[a] == C[b]
        pairs = []
        if is_elasticity:
            for i, j, k, l in np.ndindex(d, d, d, d):
                if (i, j) < (k, l):
                    pairs.append(((i, j, k, l), (k, l, i, j), 'major'))  # C_ijkl = C_klij
                if minor and i < j:
                    pairs.append(((i, j, k, l), (j, i, k, l), 'minor'))  # C_ijkl = C_jikl
        else:
            for i, j in np.ndindex(d, d):
                if i < j:
                    pairs.append(((i, j), (j, i), 'major'))  # A_ij = A_ji

        local_scale = float(max(abs(C.max()), abs(C.min()))) if C.size else 0.0
        local_violation = {'major': 0.0, 'minor': 0.0}
        for a, b, kind in pairs:
            if C[a].size:
                local_violation[kind] = max(local_violation[kind],
                                            float(np.max(np.abs(C[a] - C[b]))))

        scale = float(self.mpi_reduction.max(np.asarray(local_scale)))
        tolerance = rtol * scale + np.finfo(float).tiny
        for kind, value in local_violation.items():
            violation = float(self.mpi_reduction.max(np.asarray(value)))
            if violation > tolerance:
                required = {'major': 'C_ijkl = C_klij' if is_elasticity else 'A_ij = A_ji',
                            'minor': 'C_ijkl = C_jikl'}[kind]
                raise ValueError(
                    f'{name} violates {kind} symmetry {required} (max deviation '
                    f'{violation:.3e}, max |C| = {scale:.3e}). Conjugate gradients and the '
                    f'adjoint/weak forms in muFFTTO require it; see '
                    f'Discretization.assert_material_symmetry.')

    def get_discretization_info(self, element_type):
        """Load discretization information for element type.

        Parameters
        ----------
        element_type : str
            Element family identifier

        Notes
        -----
        Side effect only: sets attributes on ``self`` (quadrature points and
        weights, ``nb_nodes_per_pixel``, ``nb_quad_points_per_pixel``,
        ``jacobian_of_pixel`` and the stencils ``B_grad_at_pixel_dqnijk``,
        ``N_at_quad_points_dqnijk``, ``H_hess_at_pixel_deqnijk``,
        ``L_laplace_at_pixel_eqnijk``).
        """
        discretization_library.get_shape_function_gradient_matrix(self, element_type)

    def scale_field_mugrid(self, field, min_val, max_val):
        """Scale field to specified range [min_val, max_val].

        Parameters
        ----------
        field : muGrid Field
            Field to scale (modified in-place)
        min_val : float
            Minimum value after scaling
        max_val : float
            Maximum value after scaling

        Notes
        -----
        Affine map ``f <- min_val + (max_val - min_val) (f - f_min) / (f_max - f_min)``
        with global (MPI-reduced) extrema; divides by zero for a constant field.
        """
        field_min = self.mpi_reduction.min(field.s)
        field_max = self.mpi_reduction.max(field.s)
        field.s[...] = (field.s - field_min) / (field_max - field_min)
        field.s *= (max_val - min_val)
        field.s += min_val


def compute_stress_difference(actual_stress, target_stress):
    """Compute difference between actual and target stress.

    Parameters
    ----------
    actual_stress : ndarray
        Actual stress field [i,j,q,x,y,z]
    target_stress : ndarray
        Target stress [i,j]

    Returns
    -------
    ndarray
        Stress difference [i,j,q,x,y,z]
    """
    stress_difference = actual_stress - target_stress[(...,) + (np.newaxis,) * (actual_stress.ndim - 2)]
    return stress_difference


def integrate_field(stress_field, quadrature_weights):
    """Integrate stress field over domain using quadrature weights.

    Parameters
    ----------
    stress_field : ndarray
        Stress field at quadrature points [i,j,q,x,y,...]
    quadrature_weights : ndarray
        Quadrature weights [q]

    Returns
    -------
    ndarray
        Integrated field [i,j,...]

    Notes
    -----
    Local sum over all quadrature points and pixels (no MPI reduction).
    """
    stress_field = np.einsum('ijq...,q->ijq...', stress_field, quadrature_weights)
    # sum over quadrature points and all pixel axes (x, y[, z])
    integral = stress_field.sum(axis=tuple(range(2, stress_field.ndim)))
    return integral


def integrate_flux_field(flux_field, quadrature_weights):
    """Integrate flux field over domain using quadrature weights.

    Parameters
    ----------
    flux_field : ndarray
        Flux field at quadrature points [i,j,q,x,y,...]
    quadrature_weights : ndarray
        Quadrature weights [q]

    Returns
    -------
    ndarray
        Integrated flux [i,j]

    Notes
    -----
    Local sum over all quadrature points and pixels (no MPI reduction).
    """
    stress_field = np.einsum('ijq...,q->ijq...', flux_field, quadrature_weights)
    # sum over quadrature points and all pixel axes (x, y[, z])
    integral = stress_field.sum(axis=tuple(range(2, stress_field.ndim)))
    return integral


def get_gauss_points_and_weights(element_type, nb_quad_points_per_pixel):
    """Get Gauss quadrature points and weights for element type.

    Parameters
    ----------
    element_type : str
        'linear_triangles' or 'linear_triangles_tilled'
    nb_quad_points_per_pixel : int
        Number of quadrature points per element (2, 6, 8, or 18)

    Returns
    -------
    tuple of ndarray
        (quad_points_coord, quad_points_weights)
        - quad_points_coord: shape [dim, nb_quad_points_per_pixel]
        - quad_points_weights: shape [nb_quad_points_per_pixel]

    Raises
    ------
    ValueError
        If element_type is not supported. (An unsupported
        ``nb_quad_points_per_pixel`` actually ends in an
        ``UnboundLocalError`` at the ``return``.)

    Notes
    -----
    Rules for a unit square pixel split into two triangles; coordinates are
    in reference units ``[0, 1]^2`` and the weights sum to 1 (the reference
    pixel area), i.e. they are *not* scaled to the physical pixel size.

    * 2 points: one centroid per triangle (1/3, 1/3) and (2/3, 2/3).
    * 6 points: 3-point interior rule per triangle.
    * 8 / 18 points: higher-order (collapsed / Duffy-type Gauss) rules,
      4 resp. 9 points per triangle.
    """
    if element_type != 'linear_triangles' and element_type != 'linear_triangles_tilled':
        raise ValueError('Quadrature weights for Element_type {} is not implemented'.format(element_type))

    if element_type == 'linear_triangles' or element_type == 'linear_triangles_tilled':
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
