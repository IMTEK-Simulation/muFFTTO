import sys
import numpy as np
import pytest

sys.path.append('..')

from muFFTTO.discretization_library_NEW import Element, _ELEMENT_FACTORIES


class DummyDomain:
    """Mock domain object with pixel_size attribute."""
    def __init__(self, pixel_size):
        self.pixel_size = pixel_size


class TestBiquadraticQuadElement:
    """Unit tests for the 9-node biquadratic quadrilateral (Q9) element."""

    @pytest.fixture
    def non_square_pixel_size(self):
        return [0.7, 1.3]

    @pytest.fixture
    def q9_element(self, non_square_pixel_size):
        return Element.biquadratic_quad(pixel_size=non_square_pixel_size)

    def test_shapes(self, q9_element):
        """Test node_layout and tensor shapes for N, B_grad, and H_hess."""
        elem = q9_element
        assert elem.node_layout == (3, 3)
        assert elem.dim == 2
        # N: (1, n_qp, 1, 3, 3) where n_qp = 9
        assert elem.N_at_quad_points_qnijk.shape == (1, 9, 1, 3, 3)
        # B: (dim, n_qp, 1, 3, 3)
        assert elem.B_grad_at_pixel_dqnijk.shape == (2, 9, 1, 3, 3)
        # H: (dim, dim, n_qp, 1, 3, 3)
        assert elem.H_hess_at_pixel_deqnijk.shape == (2, 2, 9, 1, 3, 3)

    def test_quadrature(self, q9_element, non_square_pixel_size):
        """Test that 3x3 Gauss quadrature points and weights integrate area correctly."""
        elem = q9_element
        hx, hy = non_square_pixel_size

        assert elem.quadrature_weights.shape == (9,)
        assert elem.quad_points_coord_parametric.shape == (2, 9)
        assert elem.quad_points_coord_physical.shape == (2, 9)

        # Sum of physical weights must equal the pixel area hx * hy
        expected_area = hx * hy
        assert np.isclose(np.sum(elem.quadrature_weights), expected_area, rtol=1e-14, atol=1e-14)

        # Physical coordinates of quadrature points must lie within [0, hx] x [0, hy]
        x_qp = elem.quad_points_coord_physical[0]
        y_qp = elem.quad_points_coord_physical[1]
        assert np.all((x_qp >= 0.0) & (x_qp <= hx))
        assert np.all((y_qp >= 0.0) & (y_qp <= hy))

    def test_kronecker_delta_property(self, q9_element):
        """Test shape_functions(xi_node_j)[node_i] == delta_{ij} on reference element."""
        elem = q9_element
        # Reference coordinates for nodes at {-1, 0, 1} x {-1, 0, 1} in Fortran order
        nodes_1d = [-1.0, 0.0, 1.0]
        ref_nodes = np.array([[xi, eta] for eta in nodes_1d for xi in nodes_1d])  # (9, 2)

        eval_matrix = np.zeros((9, 9))
        for j, node_coord in enumerate(ref_nodes):
            eval_matrix[:, j] = np.array(elem.shape_functions(node_coord))

        assert np.allclose(eval_matrix, np.eye(9), atol=1e-15), \
            f"Kronecker delta property failed: max error = {np.max(np.abs(eval_matrix - np.eye(9)))}"

    def test_partition_of_unity(self, q9_element):
        """Test sum of N over nodes is 1, and sum of derivatives over nodes is 0."""
        elem = q9_element

        # N sums to 1 over node axes at all quad points
        sum_N = np.sum(elem.N_at_quad_points_qnijk, axis=(-2, -1))  # (1, 9, 1)
        assert np.allclose(sum_N, 1.0, atol=1e-14), "Partition of unity for N violated"

        # B_grad sums to 0 over node axes (derivative of constant field is 0)
        sum_B = np.sum(elem.B_grad_at_pixel_dqnijk, axis=(-2, -1))  # (2, 9, 1)
        assert np.allclose(sum_B, 0.0, atol=1e-14), "Partition of unity derivative (B_grad) violated"

        # H_hess sums to 0 over node axes (second derivative of constant field is 0)
        sum_H = np.sum(elem.H_hess_at_pixel_deqnijk, axis=(-2, -1))  # (2, 2, 9, 1)
        assert np.allclose(sum_H, 0.0, atol=1e-14), "Partition of unity second derivative (H_hess) violated"

    def test_exact_recovery_of_full_biquadratic_field(self, q9_element, non_square_pixel_size):
        """
        Test that Q9 basis exactly interpolates and differentiates a full biquadratic polynomial:
        u(x, y) = a + b*x + c*y + d*x*y + e*x^2 + f*y^2 + g*x^2*y + h*x*y^2 + k*x^2*y^2
        """
        elem = q9_element
        hx, hy = non_square_pixel_size

        # Polynomial coefficients spanning full tensor-product quadratic space
        a, b, c = 1.2, -2.3, 3.4
        d, e, f = 0.5, -1.1, 2.2
        g, h, k = 0.7, -0.9, 1.5

        def u_poly(x, y):
            return (a + b * x + c * y + d * x * y +
                    e * x**2 + f * y**2 +
                    g * (x**2) * y + h * x * (y**2) +
                    k * (x**2) * (y**2))

        def grad_u_poly(x, y):
            du_dx = (b + d * y + 2.0 * e * x +
                     2.0 * g * x * y + h * y**2 +
                     2.0 * k * x * (y**2))
            du_dy = (c + d * x + 2.0 * f * y +
                     g * x**2 + 2.0 * h * x * y +
                     2.0 * k * (x**2) * y)
            return np.array([du_dx, du_dy])

        def hess_u_poly(x, y):
            d2u_dx2 = 2.0 * e + 2.0 * g * y + 2.0 * k * (y**2)
            d2u_dy2 = 2.0 * f + 2.0 * h * x + 2.0 * k * (x**2)
            d2u_dxdy = d + 2.0 * g * x + 2.0 * h * y + 4.0 * k * x * y
            return np.array([
                [d2u_dx2, d2u_dxdy],
                [d2u_dxdy, d2u_dy2]
            ])

        # Evaluate polynomial at physical node coordinates
        # In multi-index [i, j]: x = i * hx / 2, y = j * hy / 2 for i,j in {0, 1, 2}
        x_nodes = np.array([0.0, hx / 2.0, hx])
        y_nodes = np.array([0.0, hy / 2.0, hy])
        X, Y = np.meshgrid(x_nodes, y_nodes, indexing='ij')  # shape (3, 3)
        u_nodes = u_poly(X, Y)

        # Quadrature points physical coordinates
        xq = elem.quad_points_coord_physical[0]
        yq = elem.quad_points_coord_physical[1]

        # 1. Check field interpolation: N . u_nodes
        u_interp = np.einsum('cqnij,ij->q', elem.N_at_quad_points_qnijk, u_nodes)
        u_exact = u_poly(xq, yq)
        assert np.allclose(u_interp, u_exact, rtol=1e-12, atol=1e-14), \
            f"Field interpolation error: max err = {np.max(np.abs(u_interp - u_exact))}"

        # 2. Check gradient computation: B . u_nodes
        grad_u_interp = np.einsum('dqnij,ij->dq', elem.B_grad_at_pixel_dqnijk, u_nodes)
        grad_u_exact = grad_u_poly(xq, yq)
        assert np.allclose(grad_u_interp, grad_u_exact, rtol=1e-12, atol=1e-14), \
            f"Gradient computation error: max err = {np.max(np.abs(grad_u_interp - grad_u_exact))}"

        # 3. Check Hessian computation: H . u_nodes
        hess_u_interp = np.einsum('deqnij,ij->deq', elem.H_hess_at_pixel_deqnijk, u_nodes)
        hess_u_exact = hess_u_poly(xq, yq)
        assert np.allclose(hess_u_interp, hess_u_exact, rtol=1e-12, atol=1e-14), \
            f"Hessian computation error: max err = {np.max(np.abs(hess_u_interp - hess_u_exact))}"

    def test_node_physical_coordinates_and_interpolator_array(self, q9_element, non_square_pixel_size):
        """Test physical coordinates mapping and N_basis_interpolator_array."""
        elem = q9_element
        hx, hy = non_square_pixel_size

        expected_phys_coords = {
            (0, 0): (0.0, 0.0),
            (1, 0): (hx / 2.0, 0.0),
            (2, 0): (hx, 0.0),
            (0, 1): (0.0, hy / 2.0),
            (1, 1): (hx / 2.0, hy / 2.0),  # center
            (2, 1): (hx, hy / 2.0),
            (0, 2): (0.0, hy),
            (1, 2): (hx / 2.0, hy),
            (2, 2): (hx, hy),
        }

        ref_coords_1d = [-1.0, 0.0, 1.0]

        for (i, j), (expected_x, expected_y) in expected_phys_coords.items():
            xi_ref = np.array([ref_coords_1d[i], ref_coords_1d[j]])
            # Check interpolator callable for this node
            interpolator = elem.N_basis_interpolator_array[i, j]
            # At its own node, it should evaluate to 1.0
            val_self = interpolator(*xi_ref)
            assert np.isclose(val_self, 1.0, atol=1e-15)

            # At other nodes, it should evaluate to 0.0
            for (other_i, other_j) in expected_phys_coords:
                if (other_i, other_j) != (i, j):
                    other_xi = np.array([ref_coords_1d[other_i], ref_coords_1d[other_j]])
                    val_other = interpolator(*other_xi)
                    assert np.isclose(val_other, 0.0, atol=1e-15)

    def test_element_factory_registration(self, non_square_pixel_size):
        """Test that biquadratic_rectangle is registered in _ELEMENT_FACTORIES."""
        assert 'biquadratic_rectangle' in _ELEMENT_FACTORIES

        domain_mock = DummyDomain(pixel_size=non_square_pixel_size)
        element = _ELEMENT_FACTORIES['biquadratic_rectangle'](domain_mock)

        assert isinstance(element, Element)
        assert element.node_layout == (3, 3)
        assert element.B_grad_at_pixel_dqnijk.shape == (2, 9, 1, 3, 3)
