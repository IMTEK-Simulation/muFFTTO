import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Integration with the existing discretization API
# ---------------------------------------------------------------------------

_ELEMENT_FACTORIES = {
    'linear_1D': lambda domain: Element.linear_1d(domain.pixel_size),
    'bilinear_rectangle': lambda domain: Element.bilinear_quad(domain.pixel_size),
    'trilinear_hexahedron': lambda domain: Element.trilinear_hex(domain.pixel_size),
    'trilinear_hexahedron_1Q': lambda domain: Element.trilinear_hex_1Q(domain.pixel_size),
    'linear_triangles': lambda domain: Element.linear_triangle(domain.pixel_size),
    'linear_triangles_tilled': lambda domain: Element.linear_triangle_tilled(domain.pixel_size),
}


def get_shape_function_gradient_matrix(domain, element_type):
    """
    Compute and attach FEM element data to `domain`.
    Drop-in replacement for discretization_library.get_shape_function_gradient_matrix.
    """
    if element_type not in _ELEMENT_FACTORIES:
        raise ValueError(f'Element type {element_type} is not implemented')

    element = _ELEMENT_FACTORIES[element_type](domain)

    domain.B_grad_at_pixel_dqnijk = element.B_grad_at_pixel_dqnijk
    domain.N_at_quad_points_qnijk = element.N_at_quad_points_qnijk
    domain.quad_points_coord = element.quad_points_coord_physical
    domain.quad_points_coord_parametric = element.quad_points_coord_parametric
    domain.quadrature_weights = element.quadrature_weights
    domain.nb_quad_points_per_pixel = element.quadrature_weights.shape[0]
    domain.nb_nodes_per_pixel = 1
    domain.nb_unique_nodes_per_pixel = 1
    domain.N_basis_interpolator_array = element.N_basis_interpolator_array
    domain.jacobian_of_pixel = element.jacobian_of_pixel


# ---------------------------------------------------------------------------
# Element class
# ---------------------------------------------------------------------------

class Element:
    """
    FEM reference element for a regular grid.

    On a regular grid the Jacobian J = dx/dxi is constant (or piecewise
    constant for triangles), so shape function gradients need to be computed
    only once at construction time.  AD via jax.jacobian differentiates the
    user-supplied shape_functions automatically.

    Instantiate through the factory classmethods, e.g.:
        element = Element.bilinear_quad(pixel_size=[0.1, 0.1])
    """

    def __init__(self,
                 shape_functions,
                 quadrature_points_qd,
                 quadrature_weights_physical_q,
                 jacobian_inv_per_quadrature_point_qij,
                 pixel_size,
                 quadrature_points_physical_qd,
                 jacobian_of_pixel):
        """
        Parameters
        ----------
        shape_functions : callable
            Maps parametric coordinates xi : (dim,) -> N : (n_nodes,).
            Must use jax.numpy so jax.jacobian can trace it.
        quadrature_points : ndarray, shape (n_qp, dim)
            Parametric coordinates of quadrature points on the reference element.
        quadrature_weights_physical_q : ndarray, shape (n_qp,)
            Integration weights in physical space.
        jacobian_inv_per_quadrature_point : ndarray, shape (n_qp, dim, dim)
            Inverse Jacobian J^{-1} = dxi/dx at each quadrature point.
        pixel_size : array-like, shape (dim,)
            Physical size of one pixel/voxel.
        quadrature_points_physical : ndarray, shape (n_qp, dim)
            Physical coordinates of quadrature points within one pixel (from its corner).
            Supplied explicitly by each factory because the reference-to-physical mapping
            differs between element families ([-1,1] for quads, [0,1] for triangles).
        """
        self.pixel_size = np.asarray(pixel_size, dtype=float)
        self.quadrature_weights = quadrature_weights_physical_q

        # Stored as (dim, n_qp) to match the existing API convention
        self.quad_points_coord_parametric = quadrature_points_qd.T
        self.quad_points_coord_physical = quadrature_points_physical_qd.T
        self.jacobian_of_pixel = jacobian_of_pixel

        # Store shape functions for later evaluation at arbitrary points
        self.shape_functions = shape_functions
        self.dim = len(np.asarray(pixel_size))

        # Create N_basis_interpolator_array: callable for each node position
        self.N_basis_interpolator_array = self._make_shape_function_array()

        self._compute_element_matrices(
            shape_functions=shape_functions,
            quadrature_points=quadrature_points_qd,
            jacobian_inv_per_quadrature_point=jacobian_inv_per_quadrature_point_qij,
        )

    def _make_shape_function_array(self):
        """Create an array of callables for each node position."""
        node_layout = tuple([2] * self.dim)
        result = np.empty(node_layout, dtype=object)

        for idx in np.ndindex(node_layout):
            node_position = idx

            # Create a closure that captures the node position and evaluates shape_functions
            def make_evaluator(node_pos):
                def evaluator(*coords):
                    # Convert to numpy array for jax
                    xi = np.array(coords)
                    N = np.array(self.shape_functions(jnp.array(xi)))
                    # Reshape and extract value for this node position
                    N_reshaped = N.reshape(node_layout, order='F')
                    return N_reshaped[node_pos]

                return evaluator

            result[idx] = make_evaluator(node_position)

        return result

    def _compute_element_matrices(self,
                                  shape_functions,
                                  quadrature_points,
                                  jacobian_inv_per_quadrature_point):
        """
        Use AD to compute B_grad and N at all quadrature points.

        Fills
        -----
        self.B_grad_at_pixel_dqnijk : shape (dim, n_qp, 1, *node_layout)
            Physical-space shape function gradients.
            B[d, q, 0, i, j, k] = dN_{ijk} / dx_d  evaluated at quadrature point q.
        self.N_at_quad_points_qnijk : shape (1, 1, n_qp, 1, *node_layout)
            Shape function values at quadrature points.
        """
        n_quadrature_points, dim = quadrature_points.shape

        # AD: shape_functions maps R^dim -> R^n_nodes,
        # jacobian gives dN/dxi with shape (n_nodes, dim)
        dN_dxi_func = jax.jacobian(shape_functions)

        # Evaluate N and dN/dxi at every quadrature point
        N_at_quadrature_points_qn = []  # will become (n_qp, n_nodes)
        dN_dxi_at_quadrature_points_qnj = []  # will become (n_qp, n_nodes, dim)

        for quadrature_point in quadrature_points:
            xi = jnp.array(quadrature_point)
            N_at_quadrature_points_qn.append(np.array(shape_functions(xi)))
            dN_dxi_at_quadrature_points_qnj.append(np.array(dN_dxi_func(xi)))

        # this is in parametric domain
        N_at_quadrature_points_qn = np.array(N_at_quadrature_points_qn)  # (n_qp, n_nodes)
        dN_dxi_at_quadrature_points_qnj = np.array(dN_dxi_at_quadrature_points_qnj)  # (n_qp, n_nodes, dim)

        # Transform parametric gradients to physical gradients:
        #   dN/dx_d = sum_e (dN/dxi_e) * (dxi_e/dx_d),  where J^{-1}[e, d] = dxi_e/dx_d
        # einsum axes: q=quadrature point, n=node, e=parametric dir, d=physical dir
        dN_dx_at_quadrature_points = np.einsum(
            'qne, qed -> qnd',
            dN_dxi_at_quadrature_points_qnj,
            jacobian_inv_per_quadrature_point,
        )  # (n_qp, n_nodes, dim)

        # Each pixel owns 2^dim nodes addressed by multi-index (i,), (i,j), or (i,j,k)
        node_layout = tuple([2] * dim)  # e.g. (2,2) in 2D, (2,2,2) in 3D

        # --- Build B_grad_at_pixel_dqnijk : (dim, n_qp, 1, *node_layout) ---
        # Reshape node axis back into spatial multi-index, then move dim to front
        # Use Fortran order to match old library convention where first index varies fastest
        B = dN_dx_at_quadrature_points.reshape(n_quadrature_points, *node_layout, dim,
                                               order='F')  # (n_qp, *layout, dim)
        B = np.moveaxis(B, source=-1, destination=0)  # (dim, n_qp, *layout)
        # Insert the "unique nodes per pixel" axis (always 1 for regular grids)
        self.B_grad_at_pixel_dqnijk = np.expand_dims(B, axis=2)  # (dim, n_qp, 1, *layout)

        # --- Build N_at_quad_points_qnijk : (1, n_qp, 1, *node_layout) ---
        # Use Fortran order to match old library convention: shape should be (f, q, n, *layout)
        N = N_at_quadrature_points_qn.reshape(n_quadrature_points, *node_layout, order='F')  # (n_qp, *layout)
        N = np.expand_dims(N, axis=1)  # (n_qp, 1, *layout)
        self.N_at_quad_points_qnijk = np.expand_dims(N, axis=0)  # (1, n_qp, 1, *layout)

    # -----------------------------------------------------------------------
    # Factory classmethods — one per element type
    # -----------------------------------------------------------------------

    @classmethod
    def linear_1d(cls, pixel_size):
        """2-node linear element. Reference element: [-1, 1]. One midpoint quadrature point."""
        h = pixel_size[0]

        def shape_functions(xi):
            return jnp.array([(1.0 - xi[0]) / 2.0,
                              (1.0 + xi[0]) / 2.0])

        # One quadrature point at the midpoint of [-1,1]
        quadrature_points = np.array([[0.0]])  # (1, 1) — parametric midpoint
        quadrature_weights = np.array([h])  # exact for linears

        # Reference [-1,1] -> physical [0,h] via x = h/2*(xi+1)
        quadrature_points_physical = np.array([[h / 2.0]])  # (1, 1)

        # J = h/2  =>  J^{-1} = 2/h
        jacobian_of_pixel = np.array([[h / 2.0]])
        jacobian_inv = np.array([[[2.0 / h]]])  # (1, 1, 1)

        return cls(shape_functions, quadrature_points, quadrature_weights,
                   jacobian_inv, pixel_size, quadrature_points_physical,
                   jacobian_of_pixel=jacobian_of_pixel)

    @classmethod
    def bilinear_quad(cls, pixel_size):
        """4-node bilinear quadrilateral. Reference element: [-1,1]^2. 2x2 Gauss quadrature."""
        h_x, h_y = pixel_size

        def shape_functions(xi):
            signs = jnp.array([-1.0, 1.0])
            N_xi = (1.0 + signs * xi[0]) / 2.0  # (2,) — factors along xi
            N_eta = (1.0 + signs * xi[1]) / 2.0  # (2,) — factors along eta
            # Fortran-order ravel to match expected node ordering: (0,0), (1,0), (0,1), (1,1)
            return jnp.outer(N_xi, N_eta).ravel(order='F')  # (4,) in Fortran-order

        # Physical mapping: x = h_x*xi, y = h_y*eta  (NOT the [-1,1] formula)
        # Reference [-1,1]^2 -> physical [0,h_x] x [0,h_y]
        jacobian_of_pixel = np.array([[h_x / 2, 0.],
                                      [0., h_y / 2]])


        gauss_coord = 1.0 / np.sqrt(3)
        gauss_1d = np.array([-gauss_coord, gauss_coord])

        quadrature_points_qi = np.array([[xi, eta]
                                      for eta in gauss_1d
                                      for xi in gauss_1d])  # (4, 2) — parametric
        # Map quadrature points from [-1,1]^2 to physical element.
        # xi in [-1,1] -> (xi+1) in [0,2] -> J*(xi+1) in [0,h_x] x [0,h_y] -> shift by element origin x0
        x0=  np.array([0, 0])
        quadrature_points_physical = x0 + (quadrature_points_qi + 1.0) @ jacobian_of_pixel.T        # J = diag(h_x/2, h_y/2), constant at all quadrature points



        # Scale reference weights by |J| to account for the mapping from parametric to physical space
        quadrature_weights = np.full(4, 1.)
        quadrature_weights_physical_q = quadrature_weights * np.linalg.det(jacobian_of_pixel)

        jacobian_inv_single = np.linalg.inv(jacobian_of_pixel)
        jacobian_inv_qij = np.tile(jacobian_inv_single, (4,1,1 ))  # (4, 2, 2)

        return cls(shape_functions=shape_functions,
                   quadrature_points_qd=quadrature_points_qi,
                   quadrature_weights_physical_q=quadrature_weights_physical_q,
                   jacobian_inv_per_quadrature_point_qij=jacobian_inv_qij,
                   pixel_size=pixel_size,
                   quadrature_points_physical_qd=quadrature_points_physical,
                   jacobian_of_pixel=jacobian_of_pixel ) # this 2 is here because I need it to scale coordinates

    @classmethod
    def trilinear_hex(cls, pixel_size):
        """8-node trilinear hexahedron. Reference element: [-1,1]^3. 2x2x2 Gauss quadrature."""
        h_x, h_y, h_z = pixel_size

        def shape_functions(xi):
            signs = jnp.array([-1.0, 1.0])
            N_xi = (1.0 + signs * xi[0]) / 2.0  # (2,)
            N_eta = (1.0 + signs * xi[1]) / 2.0  # (2,)
            N_zeta = (1.0 + signs * xi[2]) / 2.0  # (2,)
            # Fortran-order ravel to match expected node ordering
            return jnp.einsum('i,j,k->ijk', N_xi, N_eta, N_zeta).ravel(order='F')  # (8,)

        gauss_coord = 1.0 / np.sqrt(3)
        gauss_1d = np.array([-gauss_coord, gauss_coord])

        quadrature_points = np.array([[xi, eta, zeta]
                                      for zeta in gauss_1d
                                      for eta in gauss_1d
                                      for xi in gauss_1d])  # (8, 3) — parametric
        quadrature_weights = np.full(8, h_x * h_y * h_z / 8.0)

        # Reference [-1,1]^3 -> physical [0,h]^3 via x = h/2*(xi+1)
        quadrature_points_physical = (quadrature_points + 1.0) / 2.0 * np.array([h_x, h_y, h_z])  # (8, 3)

        # J = diag(h_x/2, h_y/2, h_z/2), constant across all quadrature points
        jacobian_of_pixel = np.diag([h_x / 2.0, h_y / 2.0, h_z / 2.0])
        jacobian_inv_single = np.diag([2.0 / h_x, 2.0 / h_y, 2.0 / h_z])
        jacobian_inv = np.tile(jacobian_inv_single, (8, 1, 1))  # (8, 3, 3)

        return cls(shape_functions, quadrature_points, quadrature_weights,
                   jacobian_inv, pixel_size, quadrature_points_physical,
                   jacobian_of_pixel=jacobian_of_pixel)

    @classmethod
    def trilinear_hex_1Q(cls, pixel_size):
        """8-node trilinear hexahedron with single Gauss point (1Q) at center.
        Reference element: [-1,1]^3. Quadrature point at (0, 0, 0)."""
        h_x, h_y, h_z = pixel_size

        def shape_functions(xi):
            signs = jnp.array([-1.0, 1.0])
            N_xi = (1.0 + signs * xi[0]) / 2.0  # (2,)
            N_eta = (1.0 + signs * xi[1]) / 2.0  # (2,)
            N_zeta = (1.0 + signs * xi[2]) / 2.0  # (2,)
            return jnp.einsum('i,j,k->ijk', N_xi, N_eta, N_zeta).ravel(order='F')  # (8,)

        # Single quadrature point at center of reference element
        quadrature_points = np.array([[0.0, 0.0, 0.0]])  # (1, 3)
        quadrature_weights = np.array([h_x * h_y * h_z])  # (1,)

        # Physical coordinates of centroid
        quadrature_points_physical = np.array([[h_x / 2.0, h_y / 2.0, h_z / 2.0]])  # (1, 3)

        # J = diag(h_x/2, h_y/2, h_z/2), constant
        jacobian_of_pixel = np.diag([h_x / 2.0, h_y / 2.0, h_z / 2.0])
        jacobian_inv_single = np.diag([2.0 / h_x, 2.0 / h_y, 2.0 / h_z])
        jacobian_inv = jacobian_inv_single[np.newaxis, :, :]  # (1, 3, 3)

        return cls(shape_functions, quadrature_points, quadrature_weights,
                   jacobian_inv, pixel_size, quadrature_points_physical,
                   jacobian_of_pixel=jacobian_of_pixel)

    @classmethod
    def linear_triangle(cls, pixel_size):
        """
        Two linear triangles per pixel (pixel split along the diagonal).

        Reference element: standard simplex xi>=0, eta>=0, xi+eta<=1,
        with the SAME parametric space [0,1]^2 used for the entire pixel.
        The physical mapping is simply x = h_x*xi, y = h_y*eta for both triangles,
        so J = diag(h_x, h_y) and J^{-1} = diag(1/h_x, 1/h_y) at both quadrature points.

        Shape functions are piecewise, switching on the diagonal xi+eta=1.
        jnp.where evaluates both branches but quadrature points are interior
        so the "wrong" branch gradient is discarded safely.

        Node ordering matches pixel corners: (0,0), (1,0), (0,1), (1,1).
        Lower triangle uses nodes (0,0), (1,0), (0,1).
        Upper triangle uses nodes (1,0), (0,1), (1,1).
        """
        h_x, h_y = pixel_size

        def shape_functions(xi):
            N_lower = jnp.array([1.0 - xi[0] - xi[1],  # node (0,0)
                                 xi[0],  # node (1,0)
                                 xi[1],  # node (0,1)
                                 0.0])  # node (1,1) — absent in lower

            N_upper = jnp.array([0.0,  # node (0,0) — absent in upper
                                 1.0 - xi[1],  # node (1,0)
                                 1.0 - xi[0],  # node (0,1)
                                 xi[0] + xi[1] - 1.0])  # node (1,1)

            return jnp.where(xi[1] < 1.0 - xi[0], N_lower, N_upper)

        # Physical mapping: x = h_x*xi, y = h_y*eta  (NOT the [-1,1] formula)
        # Reference [0,1]^2 -> physical [0,h_x] x [0,h_y]
        jacobian_of_pixel = np.array([[h_x, 0.],
                                               [0., h_y]])

        # One quadrature point per triangle (centroid of the reference simplex)
        quadrature_points_qi = np.array([[1.0 / 3.0, 1.0 / 3.0],  # lower triangle centroid
                                         [2.0 / 3.0, 2.0 / 3.0]])  # upper triangle centroid (in shared [0,1]^2 space)
        # Map quadrature points from [0,1]^2 to physical element.
        x0 = np.array([0, 0])
        quadrature_points_physical_qi = x0 + (
            quadrature_points_qi) @ jacobian_of_pixel.T  # J = diag(h_x/2, h_y/2), constant at all quadrature points

        # Scale reference weights by |J| to account for the mapping from parametric to physical space
        quadrature_weights_q = np.array([1/2, 1/2])
        quadrature_weights_physical_q = quadrature_weights_q * np.linalg.det(jacobian_of_pixel)

        # J = diag(h_x, h_y) is the SAME for both triangles since they share parametric space
        # J^{-1} = diag(1/h_x, 1/h_y)
        jacobian_inv_single = np.linalg.inv(jacobian_of_pixel)
        jacobian_inv = np.tile(jacobian_inv_single, (2, 1, 1))  # (2, 2, 2)

        return cls(shape_functions=shape_functions,
                   quadrature_points_qd=quadrature_points_qi,
                   quadrature_weights_physical_q=quadrature_weights_physical_q,
                   jacobian_inv_per_quadrature_point_qij=jacobian_inv,
                   pixel_size=pixel_size,
                   quadrature_points_physical_qd=quadrature_points_physical_qi,
                   jacobian_of_pixel=jacobian_of_pixel)

    @classmethod
    def linear_triangle_tilled(cls, pixel_size):
        """
        Two linear triangles per pixel with a sheared (60-degree) tiling.

        Reference element: standard simplex xi>=0, eta>=0, xi+eta<=1,
        same [0,1]^2 parametric space shared by both triangles.

        Physical mapping includes an x-shear so rows of pixels interlock:
            x = h_x*xi + (h_x/2)*eta
            y = h_y*eta
        Node physical coordinates (pixel corners under the sheared mapping):
            node (0,0) -> (0,         0   )
            node (1,0) -> (h_x,       0   )
            node (0,1) -> (h_x/2,     h_y )
            node (1,1) -> (3*h_x/2,   h_y )

        Shape functions are identical to linear_triangle in parametric space;
        the shear is entirely captured by the Jacobian.
        """
        h_x, h_y = pixel_size
        x_shear = h_x / 2.0  # x-offset per unit eta (half pixel width)

        def shape_functions(xi):
            N_lower = jnp.array([1.0 - xi[0] - xi[1],  # node (0,0)
                                 xi[0],  # node (1,0)
                                 xi[1],  # node (0,1)
                                 0.0])  # node (1,1) — absent in lower triangle

            N_upper = jnp.array([0.0,  # node (0,0) — absent in upper triangle
                                 1.0 - xi[1],  # node (1,0)
                                 1.0 - xi[0],  # node (0,1)
                                 xi[0] + xi[1] - 1.0])  # node (1,1)

            return jnp.where(xi[1] < 1.0 - xi[0], N_lower, N_upper)

        # Physical mapping: x = h_x*xi, y = h_y*eta  (NOT the [-1,1] formula)
        # Reference [0,1]^2 -> physical [0,h_x] x [0,h_y]
        jacobian_of_pixel = np.array([[h_x, x_shear],
                                       [0., h_y]])

        # One quadrature point per triangle (centroid of the reference simplex)
        quadrature_points_qi = np.array([[1.0 / 3.0, 1.0 / 3.0],  # lower triangle centroid
                                         [2.0 / 3.0, 2.0 / 3.0]])  # upper triangle centroid

        # Map quadrature points from [-1,1]^2 to physical element.
        x0 = np.array([0, 0])
        # quadrature_points_physical_qi = x0 + (
        #     quadrature_points_qi) @ jacobian_of_pixel   #   constant at all quadrature points
        quadrature_points_physical_qi = np.einsum('qi,ji->qj',quadrature_points_qi, jacobian_of_pixel )

        # Scale reference weights by |J| to account for the mapping from parametric to physical space
        quadrature_weights_q = np.array([1/2, 1/2])
        quadrature_weights_physical_q = quadrature_weights_q * np.linalg.det(jacobian_of_pixel)


        # Sheared Jacobian J = [[h_x, x_shear], [0, h_y]] — constant for both triangles
        jacobian_inv_pixel = np.linalg.inv(jacobian_of_pixel)
        jacobian_inv = np.tile(jacobian_inv_pixel, (2, 1, 1))  # (2, 2, 2) — same at both quadrature points


        return cls(shape_functions=shape_functions,
                   quadrature_points_qd=quadrature_points_qi,
                   quadrature_weights_physical_q=quadrature_weights_physical_q,
                   jacobian_inv_per_quadrature_point_qij=jacobian_inv,
                   pixel_size=pixel_size,
                   quadrature_points_physical_qd=quadrature_points_physical_qi,
                   jacobian_of_pixel=jacobian_of_pixel)
