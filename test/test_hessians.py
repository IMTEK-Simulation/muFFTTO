import sys
import warnings

sys.path.append('..')  # Add parent directory to path

import numpy as np
import pytest

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import material_models
from muFFTTO.discretization_library_NEW import Element


class TestHessianOperatorBasics:
    """Tests for H_hess_at_pixel_deqnijk shape function Hessian tensor."""

    def test_hessian_tensor_shapes(self):
        """Test that H_hess_at_pixel_deqnijk has expected shape (dim, dim, n_qp, 1, *node_layout)."""
        pixel_size_1d = [0.2]
        pixel_size_2d = [0.3, 0.4]
        pixel_size_3d = [0.2, 0.3, 0.5]

        configs = [
            ('linear_1d', Element.linear_1d(pixel_size_1d), (1, 1, 1, 1, 2)),
            ('bilinear_quad', Element.bilinear_quad(pixel_size_2d), (2, 2, 4, 1, 2, 2)),
            ('linear_triangle', Element.linear_triangle(pixel_size_2d), (2, 2, 2, 1, 2, 2)),
            ('linear_triangle_tilled', Element.linear_triangle_tilled(pixel_size_2d), (2, 2, 2, 1, 2, 2)),
            ('trilinear_hex', Element.trilinear_hex(pixel_size_3d), (3, 3, 8, 1, 2, 2, 2)),
            ('trilinear_hex_1Q', Element.trilinear_hex_1Q(pixel_size_3d), (3, 3, 1, 1, 2, 2, 2)),
        ]

        for name, elem, expected_shape in configs:
            assert elem.H_hess_at_pixel_deqnijk.shape == expected_shape, \
                f'Shape mismatch for {name}: expected {expected_shape}, got {elem.H_hess_at_pixel_deqnijk.shape}'

    def test_hessian_symmetry(self):
        """Test Schwarz theorem: d^2 N / (dx_d dx_e) == d^2 N / (dx_e dx_d)."""
        pixel_size_2d = [0.3, 0.4]
        pixel_size_3d = [0.2, 0.3, 0.5]

        elements = [
            Element.bilinear_quad(pixel_size_2d),
            Element.linear_triangle(pixel_size_2d),
            Element.linear_triangle_tilled(pixel_size_2d),
            Element.trilinear_hex(pixel_size_3d),
            Element.trilinear_hex_1Q(pixel_size_3d),
        ]

        for elem in elements:
            H = elem.H_hess_at_pixel_deqnijk
            dim = elem.dim
            for d in range(dim):
                for e in range(dim):
                    diff = np.max(np.abs(H[d, e] - H[e, d]))
                    assert diff < 1e-14, f'Hessian not symmetric for {elem}: max diff = {diff}'

    def test_hessian_partition_of_unity(self):
        """Test sum over all nodes of d^2 N_i / (dx_d dx_e) == 0 (since sum(N_i) == 1)."""
        pixel_size_2d = [0.3, 0.4]
        pixel_size_3d = [0.2, 0.3, 0.5]

        elements = [
            Element.linear_1d([0.2]),
            Element.bilinear_quad(pixel_size_2d),
            Element.linear_triangle(pixel_size_2d),
            Element.linear_triangle_tilled(pixel_size_2d),
            Element.trilinear_hex(pixel_size_3d),
            Element.trilinear_hex_1Q(pixel_size_3d),
        ]

        for elem in elements:
            H = elem.H_hess_at_pixel_deqnijk
            # Sum over all node axes (axes 4 to end, plus axis 3 which has size 1)
            node_axes = tuple(range(3, H.ndim))
            sum_over_nodes = np.sum(H, axis=node_axes)
            max_err = np.max(np.abs(sum_over_nodes))
            assert max_err < 1e-14, f'Partition of unity violated for {elem}: sum = {max_err}'

    def test_hessian_linear_elements_identically_zero(self):
        """Test that linear/triangular elements have exactly zero Hessian."""
        pixel_size = [0.3, 0.4]
        elements = [
            Element.linear_1d([0.2]),
            Element.linear_triangle(pixel_size),
            Element.linear_triangle_tilled(pixel_size),
        ]

        for elem in elements:
            H = elem.H_hess_at_pixel_deqnijk
            assert np.allclose(H, 0.0, atol=1e-15), \
                f'Expected identically zero Hessian for linear element, got max={np.max(np.abs(H))}'

    def test_hessian_bilinear_rectangle_analytical(self):
        """Test analytical values for bilinear quadrilateral Hessian."""
        hx, hy = 0.3, 0.7
        elem = Element.bilinear_quad(pixel_size=[hx, hy])
        H = elem.H_hess_at_pixel_deqnijk  # (2, 2, 4, 1, 2, 2)

        # Pure second derivatives must be identically zero
        assert np.allclose(H[0, 0], 0.0, atol=1e-15), "d^2 N / dx^2 must be zero for bilinear element"
        assert np.allclose(H[1, 1], 0.0, atol=1e-15), "d^2 N / dy^2 must be zero for bilinear element"

        # Mixed derivatives:
        # N(x,y) = (1 +- (2x/hx - 1))/2 * (1 +- (2y/hy - 1))/2
        # d^2 N_{ij} / dx dy = s_i * s_j / (hx * hy) where s = [-1, +1]
        # Nodes: (0,0)->+1/(hx*hy), (1,0)->-1/(hx*hy), (0,1)->-1/(hx*hy), (1,1)->+1/(hx*hy)
        expected_d2N_dxdy = np.array([
            [[[1.0 / (hx * hy), -1.0 / (hx * hy)],
              [-1.0 / (hx * hy), 1.0 / (hx * hy)]]]
        ])  # shape (1, 1, 2, 2)

        for q in range(4):
            assert np.allclose(H[0, 1, q], expected_d2N_dxdy, rtol=1e-12, atol=1e-14), \
                f'Mixed derivative d2N/dxdy mismatch at quad point {q}'
            assert np.allclose(H[1, 0, q], expected_d2N_dxdy, rtol=1e-12, atol=1e-14), \
                f'Mixed derivative d2N/dydx mismatch at quad point {q}'

    def test_hessian_trilinear_hexahedron_analytical(self):
        """Test analytical values for trilinear hexahedron Hessian."""
        hx, hy, hz = 0.2, 0.3, 0.4
        elem = Element.trilinear_hex(pixel_size=[hx, hy, hz])
        H = elem.H_hess_at_pixel_deqnijk  # (3, 3, 8, 1, 2, 2, 2)

        # All pure second derivatives must be zero
        for d in range(3):
            assert np.allclose(H[d, d], 0.0, atol=1e-15), \
                f'Pure second derivative d^2 N / dx_{d}^2 must be zero for trilinear hex'

        # Mixed derivatives:
        # d^2 N_{ijk} / dx dy = (s_i s_j / (hx*hy)) * ((1 + s_k zeta_q)/2)
        quad_pts = elem.quad_points_coord_parametric.T  # (8, 3) in [-1, 1]^3
        signs = np.array([-1.0, 1.0])

        for q in range(8):
            xi_q, eta_q, zeta_q = quad_pts[q]
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        # d2N/dx dy
                        expected_xy = (signs[i] * signs[j] / (hx * hy)) * ((1.0 + signs[k] * zeta_q) / 2.0)
                        assert np.isclose(H[0, 1, q, 0, i, j, k], expected_xy, rtol=1e-12, atol=1e-14)

                        # d2N/dx dz
                        expected_xz = (signs[i] * signs[k] / (hx * hz)) * ((1.0 + signs[j] * eta_q) / 2.0)
                        assert np.isclose(H[0, 2, q, 0, i, j, k], expected_xz, rtol=1e-12, atol=1e-14)

                        # d2N/dy dz
                        expected_yz = (signs[j] * signs[k] / (hy * hz)) * ((1.0 + signs[i] * xi_q) / 2.0)
                        assert np.isclose(H[1, 2, q, 0, i, j, k], expected_yz, rtol=1e-12, atol=1e-14)

    def test_hessian_reproduces_field_second_derivatives_2D(self):
        """
        Test contracting H_hess_at_pixel_deqnijk with 2D nodal polynomial values.
        -> this is just Hassian on a single pixel
        """
        hx, hy = 0.25, 0.5
        elem = Element.bilinear_quad(pixel_size=[hx, hy])
        H = elem.H_hess_at_pixel_deqnijk  # (2, 2, 4, 1, 2, 2)

        # Node coordinates for one pixel: (0,0), (hx,0), (0,hy), (hx,hy)
        # In indexing [i, j]: x = i * hx, y = j * hy
        i_coords = np.array([0.0, hx])
        j_coords = np.array([0.0, hy])
        X, Y = np.meshgrid(i_coords, j_coords, indexing='ij')

        # u(x, y) = 3*x^2 (projected to bilinear) + 5*x*y + 2*x - 4*y + 7
        # Note: on a single bilinear quad, nodal values of u(x,y) = 5*x*y + 2*x - 4*y + 7
        # have exact analytical Hessian: d2u/dx2 = 0, d2u/dy2 = 0, d2u/dxdy = 5
        u_nodes = 5.0 * X * Y + 2.0 * X - 4.0 * Y + 7.0

        # Contract H with u_nodes: sum_{i,j} H[d, e, q, n, i, j] * u[i, j]
        d2u = np.einsum('deqnij,ij->deq', H, u_nodes)

        # Check all quad points
        assert np.allclose(d2u[0, 0, :], 0.0, atol=1e-14), "d^2 u / dx^2 should be 0"
        assert np.allclose(d2u[1, 1, :], 0.0, atol=1e-14), "d^2 u / dy^2 should be 0"
        assert np.allclose(d2u[0, 1, :], 5.0, rtol=1e-12, atol=1e-14), "d^2 u / dx dy should be 5.0"
        assert np.allclose(d2u[1, 0, :], 5.0, rtol=1e-12, atol=1e-14), "d^2 u / dy dx should be 5.0"

    def test_hessian_reproduces_field_second_derivatives_3D(self):
        """Test contracting H_hess_at_pixel_deqnijk with 3D nodal polynomial values.
                -> this is just Hassian on a single voxel
        """
        hx, hy, hz = 0.2, 0.3, 0.4
        elem = Element.trilinear_hex(pixel_size=[hx, hy, hz])
        H = elem.H_hess_at_pixel_deqnijk  # (3, 3, 8, 1, 2, 2, 2)

        # Nodal coordinates for one pixel: [i, j, k] -> (i*hx, j*hy, k*hz)
        x_c = np.array([0.0, hx])
        y_c = np.array([0.0, hy])
        z_c = np.array([0.0, hz])
        X, Y, Z = np.meshgrid(x_c, y_c, z_c, indexing='ij')

        # u(x,y,z) = 3*x*y - 2*y*z + 4*x*z + 1.5*x - 2.5*y + 6.0
        # Second derivatives:
        # d2u/dx2 = 0, d2u/dy2 = 0, d2u/dz2 = 0
        # d2u/dxdy = 3, d2u/dydz = -2, d2u/dxdz = 4
        u_nodes = 3.0 * X * Y - 2.0 * Y * Z + 4.0 * X * Z + 1.5 * X - 2.5 * Y + 6.0

        d2u = np.einsum('deqnijk,ijk->deq', H, u_nodes)

        # Check pure derivatives
        for d in range(3):
            assert np.allclose(d2u[d, d, :], 0.0, atol=1e-14), f"d^2 u / dx_{d}^2 should be 0"

        # Check mixed derivatives
        assert np.allclose(d2u[0, 1, :], 3.0, rtol=1e-12, atol=1e-14), "d^2 u / dx dy should be 3.0"
        assert np.allclose(d2u[1, 0, :], 3.0, rtol=1e-12, atol=1e-14), "d^2 u / dy dx should be 3.0"
        assert np.allclose(d2u[1, 2, :], -2.0, rtol=1e-12, atol=1e-14), "d^2 u / dy dz should be -2.0"
        assert np.allclose(d2u[2, 1, :], -2.0, rtol=1e-12, atol=1e-14), "d^2 u / dz dy should be -2.0"
        assert np.allclose(d2u[0, 2, :], 4.0, rtol=1e-12, atol=1e-14), "d^2 u / dx dz should be 4.0"
        assert np.allclose(d2u[2, 0, :], 4.0, rtol=1e-12, atol=1e-14), "d^2 u / dz dx should be 4.0"

    def test_hessian_domain_attachment(self):
        """Test that discretization/domain correctly gets H_hess_at_pixel_deqnijk attached."""
        domain_size = [4, 5]
        problem_type = 'conductivity'
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)
        number_of_pixels = (4, 5)
        discretization_type = 'finite_element'

        for element_type in ['linear_triangles', 'bilinear_rectangle', 'linear_triangles_tilled']:
            discretization = domain.Discretization(cell=my_cell,
                                                   nb_of_pixels_global=number_of_pixels,
                                                   discretization_type=discretization_type,
                                                   element_type=element_type)

            assert hasattr(discretization, 'H_hess_at_pixel_deqnijk'), \
                f'Discretization missing H_hess_at_pixel_deqnijk for {element_type}'
            assert discretization.H_hess_at_pixel_deqnijk is not None

    def test_hessian_reproduces_field_second_derivatives_2D_on_grid(self):
        """
        This test test if hassian operator can compute hessian on global fields
          Test apply_hessian_operator_to_scalar_field_mugrid against 2D nodal polynomial values."""
        domain_size = [4, 5]
        dim = len(domain_size)
        problem_type = 'conductivity'
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)
        number_of_pixels = (2, 3)
        discretization_type = 'finite_element'

        discretization = domain.Discretization(cell=my_cell,
                                               nb_of_pixels_global=number_of_pixels,
                                               discretization_type=discretization_type,
                                               element_type='bilinear_rectangle')

        nodal_coords = discretization.get_nodal_points_coordinates()

        X = nodal_coords.s[0, 0, :, :]
        Y = nodal_coords.s[1, 0, :, :]

        u_inxyz = discretization.get_temperature_sized_field(name='u')
        # u(x, y) = 5*x*y + 2*x - 4*y + 7
        # bilinear elements represent this exactly, so
        #   d2u/dx2 = 0, d2u/dy2 = 0, d2u/dxdy = 5
        u_inxyz.s[0, 0, :, :] = 5.0 * X * Y + 2.0 * X - 4.0 * Y + 7.0

        hess_u_ijkqxyz = discretization.get_temperature_hessian_size_field(name='Hessian_u')

        discretization.fft.communicate_ghosts(field=u_inxyz)
        discretization.apply_hessian_operator_to_scalar_field_mugrid(u_inxyz=u_inxyz,
                                                                     hess_u_ijkqxyz=hess_u_ijkqxyz)

        # symmetry in the derivative pair
        assert np.allclose(hess_u_ijkqxyz.s[:, 0, 1], hess_u_ijkqxyz.s[:, 1, 0])

        # Check all quad points
        assert np.allclose(hess_u_ijkqxyz.s[0, 0, 0, :, :-1, :-1], 0.0,
                           atol=1e-14), "d^2 u / dx^2 should be 0"
        assert np.allclose(hess_u_ijkqxyz.s[0, 1, 1, :, :-1, :-1], 0.0,
                           atol=1e-14), "d^2 u / dy^2 should be 0"
        assert np.allclose(hess_u_ijkqxyz.s[0, 0, 1, :, :-1, :-1], 5.0,
                           rtol=1e-12, atol=1e-14), "d^2 u / dx dy should be 5.0"
        assert np.allclose(hess_u_ijkqxyz.s[0, 1, 0, :, :-1, :-1], 5.0,
                           rtol=1e-12, atol=1e-14), "d^2 u / dy dx should be 5.0"



    def test_hessian_reproduces_field_second_derivatives_3D_on_grid(self):
        """
        This test test if hassian operator can compute hessian on global fields
          Test apply_hessian_operator_to_scalar_field_mugrid against 3D nodal polynomial values."""
        domain_size = [4, 5, 3]
        dim = len(domain_size)
        problem_type = 'conductivity'
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)
        number_of_pixels = (2, 3, 4)
        discretization_type = 'finite_element'

        discretization = domain.Discretization(cell=my_cell,
                                               nb_of_pixels_global=number_of_pixels,
                                               discretization_type=discretization_type,
                                               element_type='trilinear_hexahedron')

        nodal_coords = discretization.get_nodal_points_coordinates()

        X = nodal_coords.s[0, 0, :, :, :]
        Y = nodal_coords.s[1, 0, :, :, :]
        Z = nodal_coords.s[2, 0, :, :, :]

        u_inxyz = discretization.get_temperature_sized_field(name='u')
        # u(x,y,z) = 5*x*y - 3*y*z + 2*x*z + 2*x - 4*y + 6*z + 7
        # trilinear elements represent this exactly, so
        #   d2u/dx2 = d2u/dy2 = d2u/dz2 = 0
        #   d2u/dxdy = 5, d2u/dydz = -3, d2u/dxdz = 2
        u_inxyz.s[0, 0, :, :, :] = (5.0 * X * Y - 3.0 * Y * Z + 2.0 * X * Z
                                    + 2.0 * X - 4.0 * Y + 6.0 * Z + 7.0)

        hess_u_ijkqxyz = discretization.get_temperature_hessian_size_field(name='Hessian_u')

        discretization.fft.communicate_ghosts(field=u_inxyz)
        discretization.apply_hessian_operator_to_scalar_field_mugrid(u_inxyz=u_inxyz,
                                                                     hess_u_ijkqxyz=hess_u_ijkqxyz)

        # symmetry in the derivative pair
        assert np.allclose(hess_u_ijkqxyz.s[:, 0, 1], hess_u_ijkqxyz.s[:, 1, 0])
        assert np.allclose(hess_u_ijkqxyz.s[:, 1, 2], hess_u_ijkqxyz.s[:, 2, 1])
        assert np.allclose(hess_u_ijkqxyz.s[:, 0, 2], hess_u_ijkqxyz.s[:, 2, 0])

        # Check all quad points
        interior = np.s_[:, :-1, :-1, :-1]
        assert np.allclose(hess_u_ijkqxyz.s[(0, 0, 0) + interior], 0.0,
                           atol=1e-14), "d^2 u / dx^2 should be 0"
        assert np.allclose(hess_u_ijkqxyz.s[(0, 1, 1) + interior], 0.0,
                           atol=1e-14), "d^2 u / dy^2 should be 0"
        assert np.allclose(hess_u_ijkqxyz.s[(0, 2, 2) + interior], 0.0,
                           atol=1e-14), "d^2 u / dz^2 should be 0"
        assert np.allclose(hess_u_ijkqxyz.s[(0, 0, 1) + interior], 5.0,
                           rtol=1e-12, atol=1e-14), "d^2 u / dx dy should be 5.0"
        assert np.allclose(hess_u_ijkqxyz.s[(0, 1, 0) + interior], 5.0,
                           rtol=1e-12, atol=1e-14), "d^2 u / dy dx should be 5.0"
        assert np.allclose(hess_u_ijkqxyz.s[(0, 1, 2) + interior], -3.0,
                           rtol=1e-12, atol=1e-14), "d^2 u / dy dz should be -3.0"
        assert np.allclose(hess_u_ijkqxyz.s[(0, 2, 1) + interior], -3.0,
                           rtol=1e-12, atol=1e-14), "d^2 u / dz dy should be -3.0"
        assert np.allclose(hess_u_ijkqxyz.s[(0, 0, 2) + interior], 2.0,
                           rtol=1e-12, atol=1e-14), "d^2 u / dx dz should be 2.0"
        assert np.allclose(hess_u_ijkqxyz.s[(0, 2, 0) + interior], 2.0,
                           rtol=1e-12, atol=1e-14), "d^2 u / dz dx should be 2.0"

    def test_hessian_reproduces_field_second_derivatives_2D_on_grid_elasticity(self):
        """
        This test test if hassian operator can compute hessian on global fields
          Test apply_hessian_operator_to_vector_field_mugrid against 2D nodal polynomial values."""
        domain_size = [4, 5]
        dim = len(domain_size)
        problem_type = 'elasticity'
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)
        number_of_pixels = (2, 2)
        discretization_type = 'finite_element'
        discretization = domain.Discretization(cell=my_cell,
                                               nb_of_pixels_global=number_of_pixels,
                                               discretization_type=discretization_type,
                                               element_type='bilinear_rectangle')
        nodal_coords = discretization.get_nodal_points_coordinates()
        X = nodal_coords.s[0, 0, :, :]
        Y = nodal_coords.s[1, 0, :, :]
        u_inxyz = discretization.get_displacement_sized_field(name='u')
        # u(x, y) = a*x*y + 2*x - 4*y + 7
        # bilinear elements represent this exactly, so
        #   d2u/dx2 = 0, d2u/dy2 = 0, d2u/dxdy = a
        u_inxyz.s[0, 0, :, :] = 5.0 * X * Y + 2.0 * X - 4.0 * Y + 7.0
        u_inxyz.s[1, 0, :, :] = 8.0 * X * Y + 2.0 * X - 4.0 * Y + 7.0
        hess_u_ijkqxyz = discretization.get_displacement_hessian_size_field(name='Hessian_u')
        discretization.fft.communicate_ghosts(field=u_inxyz)
        discretization.apply_hessian_operator_to_vector_field_mugrid(u_inxyz=u_inxyz,
                                                                     hess_u_ijkqxyz=hess_u_ijkqxyz)
        # symmetry in the derivative pair
        assert np.allclose(hess_u_ijkqxyz.s[:, 0, 1], hess_u_ijkqxyz.s[:, 1, 0])
        # Check all quad points, all displacement components
        interior = np.s_[:, :-1, :-1]
        mixed = {0: 5.0,
                 1: 8.0}
        for f, a in mixed.items():
            assert np.allclose(hess_u_ijkqxyz.s[(f, 0, 0) + interior], 0.0,
                               atol=1e-14), f"u_{f}: d^2 u / dx^2 should be 0"
            assert np.allclose(hess_u_ijkqxyz.s[(f, 1, 1) + interior], 0.0,
                               atol=1e-14), f"u_{f}: d^2 u / dy^2 should be 0"
            assert np.allclose(hess_u_ijkqxyz.s[(f, 0, 1) + interior], a,
                               rtol=1e-12, atol=1e-14), f"u_{f}: d^2 u / dx dy should be {a}"
            assert np.allclose(hess_u_ijkqxyz.s[(f, 1, 0) + interior], a,
                               rtol=1e-12, atol=1e-14), f"u_{f}: d^2 u / dy dx should be {a}"

    def test_hessian_reproduces_field_second_derivatives_3D_on_grid_elasticity(self):
        """
        This test test if hassian operator can compute hessian on global fields
          Test apply_hessian_operator_to_vector_field_mugrid against 3D nodal polynomial values."""
        domain_size = [4, 5, 3]
        dim = len(domain_size)
        problem_type = 'elasticity'
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)
        number_of_pixels = (2, 2, 2)
        discretization_type = 'finite_element'
        discretization = domain.Discretization(cell=my_cell,
                                               nb_of_pixels_global=number_of_pixels,
                                               discretization_type=discretization_type,
                                               element_type='trilinear_hexahedron')
        nodal_coords = discretization.get_nodal_points_coordinates()
        X = nodal_coords.s[0, 0, :, :, :]
        Y = nodal_coords.s[1, 0, :, :, :]
        Z = nodal_coords.s[2, 0, :, :, :]
        u_inxyz = discretization.get_displacement_sized_field(name='u')
        # u(x,y,z) = a*x*y + b*y*z + c*x*z + 2*x - 4*y + 6*z + 7
        # trilinear elements represent this exactly, so
        #   d2u/dx2 = d2u/dy2 = d2u/dz2 = 0
        #   d2u/dxdy = a, d2u/dydz = b, d2u/dxdz = c
        u_inxyz.s[0, 0, :, :, :] = (5.0 * X * Y - 3.0 * Y * Z + 2.0 * X * Z
                                    + 2.0 * X - 4.0 * Y + 6.0 * Z + 7.0)
        u_inxyz.s[1, 0, :, :, :] = (8.0 * X * Y + 1.0 * Y * Z - 6.0 * X * Z
                                    + 2.0 * X - 4.0 * Y + 6.0 * Z + 7.0)
        u_inxyz.s[2, 0, :, :, :] = (-2.0 * X * Y + 4.0 * Y * Z + 7.0 * X * Z
                                    + 2.0 * X - 4.0 * Y + 6.0 * Z + 7.0)
        hess_u_ijkqxyz = discretization.get_displacement_hessian_size_field(name='Hessian_u')
        discretization.fft.communicate_ghosts(field=u_inxyz)
        discretization.apply_hessian_operator_to_vector_field_mugrid(u_inxyz=u_inxyz,
                                                                     hess_u_ijkqxyz=hess_u_ijkqxyz)
        # symmetry in the derivative pair
        assert np.allclose(hess_u_ijkqxyz.s[:, 0, 1], hess_u_ijkqxyz.s[:, 1, 0])
        assert np.allclose(hess_u_ijkqxyz.s[:, 1, 2], hess_u_ijkqxyz.s[:, 2, 1])
        assert np.allclose(hess_u_ijkqxyz.s[:, 0, 2], hess_u_ijkqxyz.s[:, 2, 0])
        # Check all quad points, all displacement components
        interior = np.s_[:, :-1, :-1, :-1]
        mixed = {0: (5.0, -3.0, 2.0),
                 1: (8.0, 1.0, -6.0),
                 2: (-2.0, 4.0, 7.0)}
        for f, (a, b, c) in mixed.items():
            assert np.allclose(hess_u_ijkqxyz.s[(f, 0, 0) + interior], 0.0,
                               atol=1e-14), f"u_{f}: d^2 u / dx^2 should be 0"
            assert np.allclose(hess_u_ijkqxyz.s[(f, 1, 1) + interior], 0.0,
                               atol=1e-14), f"u_{f}: d^2 u / dy^2 should be 0"
            assert np.allclose(hess_u_ijkqxyz.s[(f, 2, 2) + interior], 0.0,
                               atol=1e-14), f"u_{f}: d^2 u / dz^2 should be 0"
            assert np.allclose(hess_u_ijkqxyz.s[(f, 0, 1) + interior], a,
                               rtol=1e-12, atol=1e-14), f"u_{f}: d^2 u / dx dy should be {a}"
            assert np.allclose(hess_u_ijkqxyz.s[(f, 1, 0) + interior], a,
                               rtol=1e-12, atol=1e-14), f"u_{f}: d^2 u / dy dx should be {a}"
            assert np.allclose(hess_u_ijkqxyz.s[(f, 1, 2) + interior], b,
                               rtol=1e-12, atol=1e-14), f"u_{f}: d^2 u / dy dz should be {b}"
            assert np.allclose(hess_u_ijkqxyz.s[(f, 2, 1) + interior], b,
                               rtol=1e-12, atol=1e-14), f"u_{f}: d^2 u / dz dy should be {b}"
            assert np.allclose(hess_u_ijkqxyz.s[(f, 0, 2) + interior], c,
                               rtol=1e-12, atol=1e-14), f"u_{f}: d^2 u / dx dz should be {c}"
            assert np.allclose(hess_u_ijkqxyz.s[(f, 2, 0) + interior], c,
                               rtol=1e-12, atol=1e-14), f"u_{f}: d^2 u / dz dx should be {c}"

    def test_hessian_operator_transposed_is_adjoint_2D(self):
        """<H u, V>_W == <u, H^T V>  for random u, V."""
        self._check_hessian_adjointness(domain_size=[4, 5],
                                        number_of_pixels=(4, 5),
                                        element_type='bilinear_rectangle')

    def test_hessian_operator_transposed_is_adjoint_3D(self):
        """<H u, V>_W == <u, H^T V>  for random u, V."""
        self._check_hessian_adjointness(domain_size=[4, 5, 3],
                                        number_of_pixels=(4, 5, 3),
                                        element_type='trilinear_hexahedron')

    def _check_hessian_adjointness(self, domain_size, number_of_pixels, element_type):
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type='elasticity')
        discretization = domain.Discretization(cell=my_cell,
                                               nb_of_pixels_global=number_of_pixels,
                                               discretization_type='finite_element',
                                               element_type=element_type)
        rng = np.random.default_rng(0)

        u = discretization.get_displacement_sized_field(name='u')
        V = discretization.get_displacement_hessian_size_field(name='V')
        Hu = discretization.get_displacement_hessian_size_field(name='Hu')
        HtV = discretization.get_displacement_sized_field(name='HtV')

        u.s[...] = rng.random(u.s.shape)
        V.s[...] = rng.random(V.s.shape)
        V.s[...] = 0.5 * (V.s + np.swapaxes(V.s, 1, 2))  # H^T only sees the (j,k)-symmetric part

        discretization.apply_hessian_operator_to_vector_field_mugrid(
            u_inxyz=u, hess_u_ijkqxyz=Hu)
        discretization.apply_hessian_operator_transposed_to_vector_field_mugrid(
            hess_u_ijkqxyz=V, u_inxyz=HtV)

        lhs = discretization.communicator.sum(
            np.einsum('ijkq...,ijkq...,q->', Hu.s, V.s, discretization.quadrature_weights))
        rhs = discretization.communicator.sum(np.sum(u.s * HtV.s))

        assert np.isclose(lhs, rhs, rtol=1e-12, atol=1e-14), \
            f'adjointness violated: <Hu,V> = {lhs:.12e}, <u,H^T V> = {rhs:.12e}'