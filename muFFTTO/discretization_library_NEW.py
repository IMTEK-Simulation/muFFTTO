import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Shared helper for tensor construction
# ---------------------------------------------------------------------------

def _unflatten_node_axis_to_stencil(values_q_n_and_leading, node_layout, n_leading_dim_axes):
    """
    Transform a flat-node-indexed array into a stencil-layout array.

    This helper replaces the duplicated reshape/moveaxis/expand_dims blocks that appear
    when building B_grad, H_hess, and N tensors from flat per-quadrature-point node vectors.
    The transformation preserves a critical invariant: the trailing `*node_layout` axes are
    not an arbitrary flattening — they are literal per-spatial-dimension multi-indices that
    two independent downstream code paths both depend on:

    1. muGrid.GenericLinearOperator (muFFTTO/domain.py:137,140) treats these axes directly
       as the stencil_shape of an FFT-convolution kernel, where axis k has size node_layout[k]
       and represents a pixel offset of 0..node_layout[k]-1 in direction k.

    2. Hand-written code in domain.py (evaluate_field_at_quad_points, get_preconditioner_Jacoby_fast)
       iterates `for pixel_node in np.ndindex(*node_layout)` and uses the SAME tuple both to
       index these tensors' trailing axes AND as a literal shift vector for FFT rolling.
       A node at multi-index (1,0) must mean "+1 pixel in direction 0, +0 elsewhere".

    Consequence: never reorder these axes, never replace the Fortran flat-index convention
    without also updating every shape_functions implementation, and never treat node_layout
    as "just an arbitrary flattening of n_nodes".

    Parameters
    ----------
    values_q_n_and_leading : ndarray, shape (n_qp, n_nodes, *([dim]*n_leading_dim_axes))
        The input array with axes: quadrature point (q), flat node index (n, following
        Fortran order over node_layout), and zero or more trailing physical-direction axes
        (e.g., d for gradients, d,e for Hessians).

    node_layout : tuple of int, shape (dim,)
        Number of nodes per spatial direction (e.g., (2,2) for Q1-2D, would be (3,3) for Q2-2D).
        The flat node index is related to the multi-index via np.ravel_multi_index(..., order='F').

    n_leading_dim_axes : int, in {0, 1, 2}
        Number of trailing physical-direction axes: 0 for shape-function values (N),
        1 for gradients (B), 2 for Hessians (H).

    Returns
    -------
    ndarray, shape (*([dim]*n_leading_dim_axes), n_qp,  *node_layout)
        Physical-direction axes (if any) moved to the front in their original relative order,
        followed by the quadrature point axis (q), followed by a size  of "unique nodes per pixel"
        axis, followed by the node axis unraveled into node_layout multi-index axes.
    """
    n_qp = values_q_n_and_leading.shape[0]
    n_nodes = values_q_n_and_leading.shape[1]

    # Sanity checks (never trip on correct input, safe to leave in)
    assert n_nodes == np.prod(node_layout), (
        f'Flat node axis size {n_nodes} != prod(node_layout)={np.prod(node_layout)}; '
        f'check that shape_functions returns correct number of nodes.'
    )
    assert values_q_n_and_leading.ndim == 2 + n_leading_dim_axes, (
        f'Input shape {values_q_n_and_leading.shape} incompatible with n_leading_dim_axes={n_leading_dim_axes}'
    )

    # Reshape the flat node axis into node_layout, using Fortran order.
    # This unravels axis 1 into dim new axes, leaving axis 0 (q) and trailing axes untouched.
    reshaped = values_q_n_and_leading.reshape(
        n_qp, *node_layout, *values_q_n_and_leading.shape[2:], order='F'
    )

    # Move trailing physical-direction axes to the front (no-op when n_leading_dim_axes==0).
    if n_leading_dim_axes > 0:
        reshaped = np.moveaxis(
            reshaped,
            source=list(range(-n_leading_dim_axes, 0)),
            destination=list(range(n_leading_dim_axes))
        )

    # Insert a size-1 axis for "nb_unique_nodes_per_pixel" right after the q axis.
    result = np.expand_dims(reshaped, axis=n_leading_dim_axes + 1)

    return result


# ---------------------------------------------------------------------------
# Integration with the existing discretization API
# ---------------------------------------------------------------------------

_ELEMENT_FACTORIES = {
    'linear_1D': lambda domain: Element.linear_1d(domain.pixel_size),
    'quadratic_1D': lambda domain: Element.quadratic_1D(domain.pixel_size),
    'bilinear_rectangle': lambda domain: Element.bilinear_quad(domain.pixel_size),
    'biquadratic_rectangle': lambda domain: Element.biquadratic_quad(domain.pixel_size),
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

    domain.quad_points_coord = element.quad_points_coord_physical
    domain.quad_points_coord_parametric = element.quad_points_coord_parametric
    domain.quadrature_weights = element.quadrature_weights
    domain.nb_quad_points_per_pixel = element.quadrature_weights.shape[0]
    # this does not mean corners of rectangle. This means uniques nodes associated with each pixel.
    # typically, this is 1
    domain.nb_nodes_per_pixel = element.nb_nodes_per_pixel
    # nb_nodes_per_pixel is the size of the 'n' axis in N/B/H tensors (currently 1).
    # Currently unread by any live code, but forward-looking hook for multi-node-per-pixel elements (Q2, etc).
    # domain.N_basis_interpolator_array = element.N_basis_interpolator_array
    domain.jacobian_of_pixel = element.jacobian_of_pixel

    domain.N_at_quad_points_dqnijk = element.N_at_quad_points_dqnijk
    domain.B_grad_at_pixel_dqnijk = element.B_grad_at_pixel_dqnijk
    domain.H_hess_at_pixel_deqnijk = element.H_hess_at_pixel_deqnijk


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
                 jacobian_of_pixel,
                 nb_nodes_per_pixel=1,
                 node_layout=None):
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
        node_layout : tuple of int, optional
            Number of nodes per spatial direction (e.g., (2,2) for Q1-2D, (3,3) for Q2-2D).
            Default None means (2,...,2) — one quad per direction.
        """
        self.pixel_size = np.asarray(pixel_size, dtype=float)
        self.quadrature_weights = quadrature_weights_physical_q

        # Stored as (dim, n_qp) to match the existing API convention
        self.quad_points_coord_parametric = quadrature_points_qd.T
        self.quad_points_coord_physical = quadrature_points_physical_qd.T
        self.jacobian_of_pixel = jacobian_of_pixel
        self.nb_nodes_per_pixel = nb_nodes_per_pixel
        # Store shape functions for later evaluation at arbitrary points
        self.shape_functions = shape_functions
        self.dim = len(np.asarray(pixel_size))

        # Store node_layout: geometry of nodes per pixel per spatial direction
        self.node_layout = tuple(node_layout) if node_layout is not None else tuple([2] * self.dim)

        # Create N_basis_interpolator_array: callable for each node position
        # self.N_basis_interpolator_array = self._make_shape_function_array()

        self._compute_element_matrices(
            shape_functions=shape_functions,
            quadrature_points=quadrature_points_qd,
            jacobian_inv_per_quadrature_point=jacobian_inv_per_quadrature_point_qij,
        )
        self._compute_hessian_matrices(
            shape_functions=shape_functions,
            quadrature_points=quadrature_points_qd,
            jacobian_inv_per_quadrature_point=jacobian_inv_per_quadrature_point_qij,
        )

    def _make_shape_function_array(self):
        """
        Create an array of callables for each node position, indexed by spatial multi-index.

        Each callable re-evaluates shape_functions at given parametric coordinates
        (unavoidable: the external contract requires each entry to be independent),
        but extracts its value by precomputed flat index rather than reshaping the
        entire vector on every call. This is mathematically equivalent to
        N.reshape(node_layout, order='F')[node_position] but more efficient and clearer.
        """
        result = np.empty(self.node_layout, dtype=object)

        for node_position in np.ndindex(self.node_layout):
            # Precompute the flat index for this node position (Fortran order),
            # then use it to extract the value directly instead of reshaping.
            flat_index = np.ravel_multi_index(node_position, self.node_layout, order='F')

            def evaluator(*coords, flat_index=flat_index):
                # Must match the Fortran node-flattening convention shape_functions uses
                xi = np.array(coords)
                N = np.array(self.shape_functions(jnp.array(xi)))
                return N[flat_index]

            result[node_position] = evaluator

        return result

    def _compute_element_matrices(self,
                                  shape_functions,
                                  quadrature_points,
                                  jacobian_inv_per_quadrature_point):
        """
        Use AD to compute B_grad and N at all quadrature points.

        Fills
        -----
        self.B_grad_at_pixel_dqnijk : shape (dim, n_qp, n_np, *node_layout)
            Physical-space shape function gradients.
            B[d, q, n_np, i, j, k] = dN_{ijk} / dx_d  evaluated at quadrature point q.
        self.N_at_quad_points_qnijk : shape (1, 1, n_qp, n_np, *node_layout)
            Shape function values at quadrature points.
        """
        n_quadrature_points, dim = quadrature_points.shape

        # AD: shape_functions maps R^dim -> R^n_nodes,
        # jacobian gives dN/dxi with shape (n_nodes, dim)
        dN_dxi_func = jax.jacobian(shape_functions)

        # Evaluate N and dN/dxi at every quadrature point
        N_at_quadrature_points_qnijk = []  # will become (n_qp, ,IJK)
        dN_dxi_at_quadrature_points_qnijkd = []  # will become (n_qp, n_nodes, IJK, directopm of derivative)

        for quadrature_point in quadrature_points:
            # quadrature points position  (xi, eta, ..)
            xi = jnp.array(quadrature_point)
            # Shape function in quadrature point  #
            N_at_quadrature_points_qnijk.append(np.array(shape_functions(xi)))
            # Gradient of shape functions in quadrature point
            dN_dxi_at_quadrature_points_qnijkd.append(np.array(dN_dxi_func(xi)))

        # this is in parametric domain
        N_at_quadrature_points_qnijk = np.array(
            N_at_quadrature_points_qnijk)  # (n_qp, n,I J K)
        dN_dxi_at_quadrature_points_qnijkd = np.array(dN_dxi_at_quadrature_points_qnijkd)  # (n_qp, n_nodes, dim)

        # Transform parametric gradients to physical gradients:
        # We only consider linear  transformation of the parametric pixel. Not general isoparametric like in standard fem
        #   dN/dx_d = sum_e (dN/dxi_e) * (dxi_e/dx_d),  where J^{-1}[e, d] = dxi_e/dx_d
        # einsum axes: q=quadrature point, n=node, e=parametric dir, d=physical dir
        dN_dx_at_quadrature_points_qnijkd = np.einsum(
            'qni...e, qed -> qni...d',
            dN_dxi_at_quadrature_points_qnijkd,
            jacobian_inv_per_quadrature_point,
        )  # (n_qp, n_nodes, I,J ,K, dir) * J

        # Sanity check: verify that shape_functions returned the right number of nodes
        n_nodes = np.prod(N_at_quadrature_points_qnijk.shape[1:])
        assert n_nodes == np.prod(self.node_layout), (
            f'shape_functions returned {n_nodes} nodes but node_layout={self.node_layout} '
            f'implies {np.prod(self.node_layout)}; a future Q2/Q3 element must pass a matching node_layout.'
        )

        # --- Build B_grad_at_pixel_dqnijk : (dim, n_qp, 1, *node_layout) ---
        # Trailing axes are literal per-direction pixel offsets — see _unflatten_node_axis_to_stencil docstring
        # self.B_grad_at_pixel_dqnijk =dN_dxi_at_quadrature_points_qnijkd

        # dN_dx_at_quadrature_points_dqnijk=np.swapaxes(dN_dx_at_quadrature_points_qnijkd, -1, 0)

        # Move the last axis (d) to the front
        dN_dx_at_quadrature_points_dqnijk = np.moveaxis(dN_dx_at_quadrature_points_qnijkd, -1, 0)
        # Result: (d, q, n, i, j, k)
        self.B_grad_at_pixel_dqnijk = dN_dx_at_quadrature_points_dqnijk  # np.expand_dims(dN_dx_at_quadrature_points_dqnijk, axis=2)

        # unflatten_node_axis_to_stencil(
        #   dN_dx_at_quadrature_points, self.node_layout, n_leading_dim_axes=1)

        # --- Build N_at_quad_points_qnijk : (1, n_qp, 1, *node_layout) ---
        # Helper produces (n_qp, 1, *node_layout) for n_leading_dim_axes=0. The leading size-1 axis
        # (nb_output_components=1) is added explicitly here, not by the helper, because N always has
        # # exactly one output component (unlike B/H), so this axis is not a "moved" physical-direction axis.
        # N = _unflatten_node_axis_to_stencil(
        # add dummy dimension in the beginning.
        self.N_at_quad_points_dqnijk = np.expand_dims(N_at_quadrature_points_qnijk, axis=0)  #

    def _compute_hessian_matrices(self,
                                  shape_functions,
                                  quadrature_points,
                                  jacobian_inv_per_quadrature_point):
        """
        Use AD to compute physical-space shape function Hessians at all quadrature
        points.

        The chain rule for a second derivative carries two terms:

            d2N/dx_d dx_e = sum_{a,b} (d2N/dxi_a dxi_b) (dxi_a/dx_d) (dxi_b/dx_e)
                          + sum_a     (dN/dxi_a) (d2 xi_a / dx_d dx_e)

        The second term vanishes iff the reference-to-physical map is affine.
        Every element in this library has a Jacobian that is constant over the
        pixel (`jacobian_of_pixel` is a single matrix, tiled across quadrature
        points), so the term is exactly zero -- not neglected.  This is asserted
        rather than assumed, so a future curved or non-affine element fails loudly
        instead of returning a silently wrong operator.

        Fills
        -----
        self.H_hess_at_pixel_deqnijk : shape (dim, dim, n_qp, n_un, *node_layout)
            H[d, e, q, 0, i, j, k] = d^2 N_{ijk} / dx_d dx_e at quadrature point q.

        Notes
        -----
        For Q1 elements every PURE second derivative is identically zero
        (d2N/dx_d^2 = 0), so only the dim*(dim-1)/2 mixed pairs carry information:
        1 pair in 2D, 3 in 3D.  For P1 triangles the whole array is zero, since
        linear shape functions have no curvature -- a HuHu regularization built on
        this operator does nothing on `linear_triangles` and
        `linear_triangles_tilled` by construction.
        """
        n_quadrature_points, dim = quadrature_points.shape
        jacobian_inv = np.asarray(jacobian_inv_per_quadrature_point)

        if not np.allclose(jacobian_inv, jacobian_inv[0]):
            raise NotImplementedError(
                'The Jacobian varies between quadrature points, so the '
                'reference-to-physical map is not affine and d2(xi)/dx2 != 0. '
                'The second chain-rule term must then be included; it is omitted '
                'here because every current element has a constant Jacobian.')

        # AD: shape_functions maps R^dim -> R^n_nodes,
        # jax.hessian gives d2N/dxi_a dxi_b with shape (n_nodes, dim, dim)
        d2N_dxi2_func = jax.hessian(shape_functions)

        d2N_dxi2_at_quadrature_points_qnab = []  # -> will become (n_qp, n_nodes, IJK, dim, dim)

        # evaluate Hessian at each quadrature point
        for quadrature_point in quadrature_points:
            # quadrature points position  (xi, eta, ..)
            xi = jnp.array(quadrature_point)
            # Hessian of shape functions in quadrature point
            d2N_dxi2_at_quadrature_points_qnab.append(np.array(d2N_dxi2_func(xi)))

        # cast to numpy array
        d2N_dxi2_at_quadrature_points_qnab = np.array(
            d2N_dxi2_at_quadrature_points_qnab)

        # Transform parametric Hessians to physical Hessians:
        #   d2N/dx_d dx_e = sum_{a,b} (d2N/dxi_a dxi_b) J^{-1}[a,d] J^{-1}[b,e]
        # einsum axes: q=quad point, n=node, a,b=parametric dirs, d,e=physical dirs
        d2N_dx2_at_quadrature_points_qnijkab = np.einsum(
            'qni...ab, qad, qbe -> qni...de',
            d2N_dxi2_at_quadrature_points_qnab,
            jacobian_inv,
            jacobian_inv,
        )  # (n_qp, n_nodes, dim, dim)

        # --- Build H_hess_at_pixel_deqnijk : (dim, dim, n_qp, n_un, *node_layout) ---
        # Trailing axes are literal per-direction pixel offsets — see _unflatten_node_axis_to_stencil docstring
        d2N_dx2_at_quadrature_points_abqnijk = np.moveaxis(
            d2N_dx2_at_quadrature_points_qnijkab,
            (-2, -1),  # source axes: a and b (positions -2 and -1)
            (0, 1)  # destination axes: move to front
        )

        self.H_hess_at_pixel_deqnijk = d2N_dx2_at_quadrature_points_abqnijk

    # -----------------------------------------------------------------------
    # Factory classmethods — one per element type
    # -----------------------------------------------------------------------

    @classmethod
    def linear_1d(cls, pixel_size):
        """2-node linear element. Reference element: [-1, 1]. One midpoint quadrature point."""
        h = pixel_size[0]

        def shape_functions(xi):
            N_xi = jnp.array([(1.0 - xi[0]) / 2.0,
                              (1.0 + xi[0]) / 2.0])

            return jnp.expand_dims(N_xi, axis=0)

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
    def quadratic_1D(cls, pixel_size):
        """2-node quadratic. Reference element: [-1,1]. 3 Gauss quadrature."""
        h_x = pixel_size
        ''' nodes order       n_np, i, j
        |                    |                   
        0,0 ____ 1,0 ___ 0,1 ____ 1,1 ___ 
        '''

        def shape_functions(xi):
            # 1D quadratic Lagrange factors on nodes at xi in {-1, 0, 1}
            # index 0 -> xi = -1, index 1 -> xi = 0, index 2 -> xi = +1
            N_xi = jnp.array([
                xi[0] * (xi[0] - 1.0) / 2.0,
                1.0 - xi[0] ** 2,
                xi[0] * (xi[0] + 1.0) / 2.0,
                xi[0] * 0
            ])
            # reshape to (2,2), 2 nodes and two pixels
            basis_ni = jnp.array([
                [N_xi[0], N_xi[2]],
                [N_xi[1], N_xi[3]]
            ])

            return basis_ni

        # Reference [-1,1] -> physical [0,h_x]
        jacobian_of_pixel = np.array([h_x / 2])

        # 3x3 Gauss-Legendre quadrature in reference [-1,1]^2
        gauss_coord = np.sqrt(3.0 / 5.0)
        gauss_1d_pts = np.array([-gauss_coord, 0.0, gauss_coord])
        gauss_1d_w = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])

        quadrature_points_qi = np.array([[xi] for xi in gauss_1d_pts])  # (9, 2) — parametric

        x0 = np.array([0])
        quadrature_points_physical = x0 + (quadrature_points_qi + 1.0) @ jacobian_of_pixel.T

        # Reference weights for tensor-product 3 Gauss rule
        quadrature_weights_ref = np.array([w_x  for w_x in gauss_1d_w])  # (9,)

        # Scale reference weights by |J| to account for the mapping from parametric to physical space
        quadrature_weights_physical_q = quadrature_weights_ref * np.linalg.det(jacobian_of_pixel)

        jacobian_inv_single = np.linalg.inv(jacobian_of_pixel)
        jacobian_inv_qij = np.tile(jacobian_inv_single, (3, 1, 1))  # (9, 2, 2)

        return cls(shape_functions=shape_functions,
                   quadrature_points_qd=quadrature_points_qi,
                   quadrature_weights_physical_q=quadrature_weights_physical_q,
                   jacobian_inv_per_quadrature_point_qij=jacobian_inv_qij,
                   pixel_size=pixel_size,
                   quadrature_points_physical_qd=quadrature_points_physical,
                   jacobian_of_pixel=jacobian_of_pixel,
                   nb_nodes_per_pixel=2,
                   node_layout=(2,2))

    @classmethod
    def bilinear_quad(cls, pixel_size):
        """4-node bilinear quadrilateral. Reference element: [-1,1]^2. 2x2 Gauss quadrature."""
        h_x, h_y = pixel_size

        def shape_functions(xi):
            signs = jnp.array([-1.0, 1.0])
            N_xi = (1.0 + signs * xi[0]) / 2.0  # (2,) — factors along xi
            N_eta = (1.0 + signs * xi[1]) / 2.0  # (2,) — factors along eta
            # Fortran-order ravel to match expected node ordering: (0,0), (1,0), (0,1), (1,1)
            # return jnp.outer(N_xi, N_eta)  # .ravel(order='F')  # (4,) in Fortran-order

            return jnp.expand_dims(jnp.outer(N_xi, N_eta), axis=0)

        # Reference [-1,1]^2 -> physical [0,h_x] x [0,h_y]
        jacobian_of_pixel = np.array([[h_x / 2, 0.],
                                      [0., h_y / 2]])
        # J = diag(h_x/2, h_y/2), constant at all quadrature points

        # quad coords in reference [-1,1]^2
        gauss_coord = 1.0 / np.sqrt(3)
        gauss_1d = np.array([-gauss_coord, gauss_coord])

        quadrature_points_qi = np.array([[xi, eta]
                                         for eta in gauss_1d
                                         for xi in gauss_1d])  # (4, 2) — parametric
        # Map quadrature points from [-1,1]^2 to physical element.
        # xi in [-1,1] -> (xi+1) in [0,2] -> J*(xi+1) in [0,h_x] x [0,h_y] -> shift by element origin x0
        x0 = np.array([0, 0])
        quadrature_points_physical = x0 + (quadrature_points_qi + 1.0) @ jacobian_of_pixel.T

        # Scale reference weights by |J| to account for the mapping from parametric to physical space
        quadrature_weights = np.full(4, 1.)
        quadrature_weights_physical_q = quadrature_weights * np.linalg.det(jacobian_of_pixel)

        jacobian_inv_single = np.linalg.inv(jacobian_of_pixel)
        jacobian_inv_qij = np.tile(jacobian_inv_single, (4, 1, 1))  # (4, 2, 2)

        return cls(shape_functions=shape_functions,
                   quadrature_points_qd=quadrature_points_qi,
                   quadrature_weights_physical_q=quadrature_weights_physical_q,
                   jacobian_inv_per_quadrature_point_qij=jacobian_inv_qij,
                   pixel_size=pixel_size,
                   quadrature_points_physical_qd=quadrature_points_physical,
                   jacobian_of_pixel=jacobian_of_pixel,
                   node_layout=(1, 2, 2))

    @classmethod
    def biquadratic_quad(cls, pixel_size):
        """9-node biquadratic quadrilateral (Q9). Reference element: [-1,1]^2. 3x3 Gauss quadrature."""
        h_x, h_y = pixel_size
        ''' nodes order       n_np, i, j
        |                    |                       
        |                    |                       
        2,0,1      3,0,1     2,1,1      3,1,1       
        |                    |                       
        |                    |                      
        0,0,1 ____ 1,0,1 ___ 0,1,1 ____ 1,1,1 ___     
        |                    |                    
        |                    |                    
        2,0,0      3,0,0     2,1,0      3,1,0     
        |                    |                   
        |                    |                   
        0,0,0 ____ 1,0,0 ___ 0,1,0 ____ 1,1,0 ___ 
        '''
        # so it may be easier to define basiss with shape (4x4) but the we need to reshape it
        # columns: a, b, n_np, local_i, local_j  (source idx..., target idx...)
        MAPPING = np.array([
            [0, 0, 0, 0, 0], [1, 0, 1, 0, 0], [2, 0, 0, 1, 0], [3, 0, 1, 1, 0],
            [0, 1, 2, 0, 0], [1, 1, 3, 0, 0], [2, 1, 2, 1, 0], [3, 1, 3, 1, 0],
            [0, 2, 0, 0, 1], [1, 2, 1, 0, 1], [2, 2, 0, 1, 1], [3, 2, 1, 1, 1],
            [0, 3, 2, 0, 1], [1, 3, 3, 0, 1], [2, 3, 2, 1, 1], [3, 3, 3, 1, 1],
        ])

        def make_gather(mapping, n_src_dims, target_shape):
            """Build gather-index arrays from an (source..., target...) table.
            One vectorized scatter — no python loop, works for any dim counts."""
            src, tgt = mapping[:, :n_src_dims], mapping[:, n_src_dims:]
            gather = np.zeros(target_shape + (n_src_dims,), dtype=int)
            gather[tuple(tgt.T)] = src
            return [jnp.array(gather[..., d]) for d in range(n_src_dims)]

        GATHER_A, GATHER_B = make_gather(MAPPING, n_src_dims=2, target_shape=(4, 2, 2))

        # while
        def shape_functions(xi):
            # 1D quadratic Lagrange factors on nodes at xi in {-1, 0, 1}
            # index 0 -> xi = -1, index 1 -> xi = 0, index 2 -> xi = +1
            N_xi = jnp.array([
                xi[0] * (xi[0] - 1.0) / 2.0,
                1.0 - xi[0] ** 2,
                xi[0] * (xi[0] + 1.0) / 2.0,
                xi[0] * 0
            ])
            N_eta = jnp.array([
                xi[1] * (xi[1] - 1.0) / 2.0,
                1.0 - xi[1] ** 2,
                xi[1] * (xi[1] + 1.0) / 2.0,
                xi[1] * 0
            ])

            basis_IJ = jnp.outer(N_xi, N_eta)  # (4, 4), axes (a=xi_idx, b=eta_idx)
            # reshaped = basis_IJ.reshape(2, 2, 2, 2)  # (local_i, p, local_j, q)
            # basis_nij = jnp.transpose(reshaped, (3, 1, 0, 2)).reshape(4, 2, 2)  # (n_np, local_i, local_j)
            basis_nij = basis_IJ[GATHER_A, GATHER_B]  # (4, 2, 2), via the mapping table

            return basis_nij

        # Reference [-1,1]^2 -> physical [0,h_x] x [0,h_y]
        jacobian_of_pixel = np.array([[h_x / 2, 0.],
                                      [0., h_y / 2]])

        # 3x3 Gauss-Legendre quadrature in reference [-1,1]^2
        gauss_coord = np.sqrt(3.0 / 5.0)
        gauss_1d_pts = np.array([-gauss_coord, 0.0, gauss_coord])
        gauss_1d_w = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])

        quadrature_points_qi = np.array([[xi, eta]
                                         for eta in gauss_1d_pts
                                         for xi in gauss_1d_pts])  # (9, 2) — parametric

        x0 = np.array([0, 0])
        quadrature_points_physical = x0 + (quadrature_points_qi + 1.0) @ jacobian_of_pixel.T

        # Reference weights for tensor-product 3x3 Gauss rule
        quadrature_weights_ref = np.array([w_x * w_y
                                           for w_y in gauss_1d_w
                                           for w_x in gauss_1d_w])  # (9,)

        # Scale reference weights by |J| to account for the mapping from parametric to physical space
        quadrature_weights_physical_q = quadrature_weights_ref * np.linalg.det(jacobian_of_pixel)

        jacobian_inv_single = np.linalg.inv(jacobian_of_pixel)
        jacobian_inv_qij = np.tile(jacobian_inv_single, (9, 1, 1))  # (9, 2, 2)

        return cls(shape_functions=shape_functions,
                   quadrature_points_qd=quadrature_points_qi,
                   quadrature_weights_physical_q=quadrature_weights_physical_q,
                   jacobian_inv_per_quadrature_point_qij=jacobian_inv_qij,
                   pixel_size=pixel_size,
                   quadrature_points_physical_qd=quadrature_points_physical,
                   jacobian_of_pixel=jacobian_of_pixel,
                   nb_nodes_per_pixel=4,
                   node_layout=(4, 2, 2))

    @classmethod
    def trilinear_hex(cls, pixel_size):
        """8-node trilinear hexahedron. Reference element: [-1,1]^3. 2x2x2 Gauss quadrature."""
        h_x, h_y, h_z = pixel_size

        def shape_functions(xi):
            signs = jnp.array([-1.0, 1.0])
            N_xi = (1.0 + signs * xi[0]) / 2.0  # (2,)
            N_eta = (1.0 + signs * xi[1]) / 2.0  # (2,)
            N_zeta = (1.0 + signs * xi[2]) / 2.0  # (2,)

            return jnp.expand_dims(
                jnp.einsum('i,j,k->ijk', N_xi, N_eta, N_zeta),
                axis=0
            )

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
            return jnp.expand_dims(
                jnp.einsum('i,j,k->ijk', N_xi, N_eta, N_zeta),
                axis=0
            )

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
            N_lower = jnp.array([[1.0 - xi[0] - xi[1], xi[1]],  # nodes (0,0), (0,1)
                                 [xi[0], 0.0]])  # nodes (1,0), (1,1)

            N_upper = jnp.array([[0.0, 1.0 - xi[0]],  # nodes (0,0), (0,1)
                                 [1.0 - xi[1], xi[0] + xi[1] - 1.0]])  # nodes (1,0), (1,1)
            # Expand dimension to fit the  requared shape n_n, I,J,K
            N_lower = jnp.expand_dims(N_lower, axis=0)
            N_upper = jnp.expand_dims(N_upper, axis=0)
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
        quadrature_weights_q = np.array([1 / 2, 1 / 2])
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
            N_lower = jnp.array([[1.0 - xi[0] - xi[1], xi[1]],  # nodes (0,0), (0,1)
                                 [xi[0], 0.0]])  # nodes (1,0), (1,1)

            N_upper = jnp.array([[0.0, 1.0 - xi[0]],  # nodes (0,0), (0,1)
                                 [1.0 - xi[1], xi[0] + xi[1] - 1.0]])  # nodes (1,0), (1,1)
            # Expand dimension to fit the  requared shape n_n, I,J,K
            N_lower = jnp.expand_dims(N_lower, axis=0)
            N_upper = jnp.expand_dims(N_upper, axis=0)
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
        quadrature_points_physical_qi = np.einsum('qi,ji->qj', quadrature_points_qi, jacobian_of_pixel)

        # Scale reference weights by |J| to account for the mapping from parametric to physical space
        quadrature_weights_q = np.array([1 / 2, 1 / 2])
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
