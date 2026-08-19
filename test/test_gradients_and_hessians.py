import sys
import warnings

sys.path.append('..')  # Add parent directory to path

import numpy as np
import pytest

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import material_models


class TestGradientOperatorBasics:
    """Test basic properties of gradient operators."""

    def test_gradient_zero_mean(self):
        """Gradient of any periodic field should have zero mean."""
        domain_size = [3, 4]
        problem_type = 'conductivity'
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)
        number_of_pixels = (4, 5)
        discretization_type = 'finite_element'

        for element_type in ['linear_triangles', 'bilinear_rectangle']:
            discretization = domain.Discretization(cell=my_cell,
                                                   nb_of_pixels_global=number_of_pixels,
                                                   discretization_type=discretization_type,
                                                   element_type=element_type)

            nodal_coords = discretization.get_nodal_points_coordinates()
            temperature = discretization.get_temperature_sized_field(name='temperature')
            temperature_gradient = discretization.get_temperature_gradient_size_field(
                name='gradient_of_temp')

            # Set a random periodic field
            temperature.s[0, 0, :, :] = np.sin(2 * np.pi * nodal_coords.s[0, 0, :, :])

            discretization.apply_gradient_operator_mugrid(temperature, temperature_gradient)

            mean = np.mean(temperature_gradient.s)
            assert np.abs(mean) <= 1e-14, \
                f'Gradient mean not zero for {element_type}: mean={mean}'

    def test_gradient_linear_field_exact(self):
        """Gradient of linear field should be exact."""
        domain_size = [5, 3] ### So whe hx of pixel is not one, I get wrong resutls
        problem_type = 'conductivity'
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)
        number_of_pixels = (4, 5)
        discretization_type = 'finite_element'

        for element_type in [ 'linear_triangles_tilled', 'linear_triangles','bilinear_rectangle',]:
            discretization = domain.Discretization(cell=my_cell,
                                                   nb_of_pixels_global=number_of_pixels,
                                                   discretization_type=discretization_type,
                                                   element_type=element_type)

            nodal_coords = discretization.get_nodal_points_coordinates()
            quad_coords = discretization.get_quad_points_coordinates()

            # u(x, y) = 4*x + 3*y
            u = discretization.get_temperature_sized_field(name='u')
            u.s[0, 0, :, :] = 4 * nodal_coords.s[0, 0, :, :] + 3 * nodal_coords.s[1, 0, :, :]

            grad_u = discretization.get_temperature_gradient_size_field(name='grad_u')
            grad_u_analytical = discretization.get_temperature_gradient_size_field(
                name='grad_u_analytical')

            # Analytical gradient for u = 3*y: ∂u/∂x = 0, ∂u/∂y = 3
            grad_u_analytical.s[0, 0, :, :, :] = 4.0
            grad_u_analytical.s[0, 1, :, :, :] = 3.0

            discretization.apply_gradient_operator_mugrid(u, grad_u)

            # Compare excluding periodic boundary
            diff=grad_u.s[..., :-1, :-1] - grad_u_analytical.s[..., :-1, :-1]
            error = np.max(np.abs(diff))
            assert np.allclose(grad_u.s[..., :-1, :-1], grad_u_analytical.s[..., :-1, :-1],
                               rtol=1e-12, atol=1e-14), \
                f'Gradient not exact for linear field ({element_type}): max_error={error}'

    def test_gradient_quadratic_field(self):
        """Test gradient of quadratic field (u = x*y)."""
        domain_size = [4, 5]
        problem_type = 'conductivity'
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)
        number_of_pixels = (4, 5)
        discretization_type = 'finite_element'

        discretization = domain.Discretization(cell=my_cell,
                                               nb_of_pixels_global=number_of_pixels,
                                               discretization_type=discretization_type,
                                               element_type='bilinear_rectangle')

        nodal_coords = discretization.get_nodal_points_coordinates()
        quad_coords = discretization.get_quad_points_coordinates()

        # u(x, y) = x*y
        x = nodal_coords.s[0, 0, :, :]
        y = nodal_coords.s[1, 0, :, :]
        u = discretization.get_temperature_sized_field(name='u')
        u.s[0, 0, :, :] = x * y

        grad_u = discretization.get_temperature_gradient_size_field(name='grad_u')
        grad_u_analytical = discretization.get_temperature_gradient_size_field(
            name='grad_u_analytical')

        # Analytical gradient: ∂u/∂x = y, ∂u/∂y = x
        grad_u_analytical.s[0, 0, :, :, :] = quad_coords.s[1, :, :, :]  # ∂u/∂x = y
        grad_u_analytical.s[0, 1, :, :, :] = quad_coords.s[0, :, :, :]  # ∂u/∂y = x

        discretization.apply_gradient_operator_mugrid(u, grad_u)

        for direction in range(domain_size.__len__()):
            error = np.max(np.abs(grad_u.s[0, direction, ..., :-1, :-1] -
                                  grad_u_analytical.s[0, direction, ..., :-1, :-1]))
            assert np.allclose(grad_u.s[0, direction, ..., :-1, :-1],
                               grad_u_analytical.s[0, direction, ..., :-1, :-1],
                               rtol=1e-12, atol=1e-14), \
                f'Gradient not accurate for x*y field, direction {direction}: max_error={error}'


    def test_gradient_linear_field_tilled_triangles_diagnostic(self):
        """Diagnostic test for linear_triangles_tilled gradient computation."""
        domain_size = [3, 4]
        problem_type = 'conductivity'
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)
        number_of_pixels = (4, 5)
        discretization_type = 'finite_element'
        element_type = 'linear_triangles_tilled'

        discretization = domain.Discretization(cell=my_cell,
                                               nb_of_pixels_global=number_of_pixels,
                                               discretization_type=discretization_type,
                                               element_type=element_type)

        nodal_coords = discretization.get_nodal_points_coordinates()
        quad_coords = discretization.get_quad_points_coordinates()

        # u(x, y) = 4*x + 3*y
        u = discretization.get_temperature_sized_field(name='u')
        u.s[0, 0, :, :] = 4 * nodal_coords.s[0, 0, :, :] + 3 * nodal_coords.s[1, 0, :, :]

        grad_u = discretization.get_temperature_gradient_size_field(name='grad_u')
        grad_u_analytical = discretization.get_temperature_gradient_size_field(
            name='grad_u_analytical')

        # Analytical gradient: ∂u/∂x = 4, ∂u/∂y = 3
        grad_u_analytical.s[0, 0, :, :, :] = 4.0
        grad_u_analytical.s[0, 1, :, :, :] = 3.0

        discretization.apply_gradient_operator_mugrid(u, grad_u)

        # Detailed analysis
        interior = grad_u.s[..., :-1, :-1]
        analytical = grad_u_analytical.s[..., :-1, :-1]

        for dir_idx in range(2):
            comp_grad = interior[0, dir_idx]
            anal_grad = analytical[0, dir_idx]
            diff = comp_grad - anal_grad

            error_max = np.max(np.abs(diff))
            error_min = np.min(np.abs(diff))
            error_mean = np.mean(np.abs(diff))
            error_std = np.std(np.abs(diff))

            expected = 4.0 if dir_idx == 0 else 3.0

            print(f'\nlinear_triangles_tilled - ∂u/∂x_dir{dir_idx}:')
            print(f'  Expected: {expected}')
            print(f'  Error - max: {error_max:.2e}, min: {error_min:.2e}, mean: {error_mean:.2e}, std: {error_std:.2e}')
            print(f'  Computed values range: [{np.min(comp_grad):.6f}, {np.max(comp_grad):.6f}]')

        # Assert correctness
        assert np.allclose(interior, analytical, rtol=1e-12, atol=1e-14), \
            f'Linear field gradient incorrect for linear_triangles_tilled'


class TestGradientTranspose:
    """Test transposed gradient operator (divergence)."""

    def test_gradient_transpose_conductivity(self):
        """Test gradient transpose on scalar field."""
        domain_size = [3, 4]
        problem_type = 'conductivity'
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)
        number_of_pixels = (4, 5)
        discretization_type = 'finite_element'

        for element_type in ['linear_triangles', 'bilinear_rectangle']:
            discretization = domain.Discretization(cell=my_cell,
                                                   nb_of_pixels_global=number_of_pixels,
                                                   discretization_type=discretization_type,
                                                   element_type=element_type)

            nodal_coords = discretization.get_nodal_points_coordinates()

            # Create a nodal field
            u = discretization.get_temperature_sized_field(name='u')
            u.s[0, 0, :, :] = 3 * nodal_coords.s[0, :, :] + 4 * nodal_coords.s[1, :, :]

            # Compute gradient
            grad_u = discretization.get_temperature_gradient_size_field(name='grad_u')
            discretization.apply_gradient_operator_mugrid(u, grad_u)

            # Apply transposed (divergence without weights)
            div_grad_u = discretization.get_temperature_sized_field(name='div_grad_u')
            discretization.apply_gradient_transposed_operator_mugrid(
                gradient_field_ijqxyz=grad_u,
                div_u_fnxyz=div_grad_u,
                apply_weights=False)

            # Result should have zero mean
            assert np.abs(np.mean(div_grad_u.s)) <= 1e-14, \
                f'Div(Grad) mean not zero for {element_type}'

    def test_gradient_transpose_elasticity(self):
        """Test gradient transpose on displacement field."""
        domain_size = [3, 4]
        problem_type = 'elasticity'
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)
        number_of_pixels = (4, 5)
        discretization_type = 'finite_element'

        for element_type in ['linear_triangles', 'bilinear_rectangle']:
            discretization = domain.Discretization(cell=my_cell,
                                                   nb_of_pixels_global=number_of_pixels,
                                                   discretization_type=discretization_type,
                                                   element_type=element_type)

            nodal_coords = discretization.get_nodal_points_coordinates()

            # Create displacement field
            u = discretization.get_displacement_sized_field(name='u')
            for direction in range(domain_size.__len__()):
                u.s[direction, 0, :, :] = 4 * nodal_coords.s[0, :, :] + 3 * nodal_coords.s[1, :, :]

            # Compute gradient
            grad_u = discretization.get_displacement_gradient_sized_field(name='grad_u')
            discretization.apply_gradient_operator_mugrid(u, grad_u)

            # Apply transposed
            div_grad_u = discretization.get_displacement_sized_field(name='div_grad_u')
            discretization.apply_gradient_transposed_operator_mugrid(
                gradient_field_ijqxyz=grad_u,
                div_u_fnxyz=div_grad_u,
                apply_weights=False)

            # Result should have zero mean
            for direction in range(domain_size.__len__()):
                assert np.abs(np.mean(div_grad_u.s[direction, 0])) <= 1e-14, \
                    f'Div(Grad) mean not zero for direction {direction} ({element_type})'


class TestGradientWith3D:
    """Test gradient operator in 3D."""

    def test_3d_gradient_linear_field(self):
        """Gradient of linear field in 3D should be exact."""
        domain_size = [3, 4, 5]
        problem_type = 'conductivity'
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)
        number_of_pixels = (4, 5, 6)
        discretization_type = 'finite_element'

        discretization = domain.Discretization(cell=my_cell,
                                               nb_of_pixels_global=number_of_pixels,
                                               discretization_type=discretization_type,
                                               element_type='trilinear_hexahedron')

        nodal_coords = discretization.get_nodal_points_coordinates()
        quad_coords = discretization.get_quad_points_coordinates()

        # u(x, y, z) = 4*x + 3*y + 5*z
        u = discretization.get_temperature_sized_field(name='u')
        u.s[0, 0, :, :, :] = (4 * nodal_coords.s[0, 0, :, :, :] +
                              3 * nodal_coords.s[1, 0, :, :, :] +
                              5 * nodal_coords.s[2, 0, :, :, :])

        grad_u = discretization.get_temperature_gradient_size_field(name='grad_u')
        grad_u_analytical = discretization.get_temperature_gradient_size_field(
            name='grad_u_analytical')

        # Analytical gradient
        grad_u_analytical.s[0, 0, :, :, :, :] = 4.0
        grad_u_analytical.s[0, 1, :, :, :, :] = 3.0
        grad_u_analytical.s[0, 2, :, :, :, :] = 5.0

        discretization.apply_gradient_operator_mugrid(u, grad_u)

        # Compare excluding periodic boundary
        error = np.max(np.abs(grad_u.s[..., :-1, :-1, :-1] - grad_u_analytical.s[..., :-1, :-1, :-1]))
        assert np.allclose(grad_u.s[..., :-1, :-1, :-1],
                           grad_u_analytical.s[..., :-1, :-1, :-1],
                           rtol=1e-14, atol=1e-14), \
            f'Gradient not exact for linear field in 3D: max_error={error}'

    def test_3d_gradient_zero_mean(self):
        """Gradient of periodic field should have zero mean in 3D."""
        domain_size = [3, 4, 5]
        problem_type = 'conductivity'
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)
        number_of_pixels = (3, 4, 5)
        discretization_type = 'finite_element'

        discretization = domain.Discretization(cell=my_cell,
                                               nb_of_pixels_global=number_of_pixels,
                                               discretization_type=discretization_type,
                                               element_type='trilinear_hexahedron')

        nodal_coords = discretization.get_nodal_points_coordinates()
        u = discretization.get_temperature_sized_field(name='u')
        u.s[0, 0, :, :, :] = np.sin(2 * np.pi * nodal_coords.s[0, 0, :, :, :])

        grad_u = discretization.get_temperature_gradient_size_field(name='grad_u')
        discretization.apply_gradient_operator_mugrid(u, grad_u)

        mean = np.mean(grad_u.s)
        assert np.abs(mean) <= 1e-12, \
            f'Gradient mean not zero in 3D: mean={mean}'


class TestGradientElasticity:
    """Test gradient operator on displacement fields (elasticity)."""

    def test_2d_displacement_gradient_linear(self):
        """Test gradient on displacement field in 2D."""
        domain_size = [3, 4]
        problem_type = 'elasticity'
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)
        number_of_pixels = (4, 5)
        discretization_type = 'finite_element'

        for element_type in ['linear_triangles', 'bilinear_rectangle']:
            discretization = domain.Discretization(cell=my_cell,
                                                   nb_of_pixels_global=number_of_pixels,
                                                   discretization_type=discretization_type,
                                                   element_type=element_type)

            nodal_coords = discretization.get_nodal_points_coordinates()
            quad_coords = discretization.get_quad_points_coordinates()

            u = discretization.get_displacement_sized_field(name='u')
            # u_x = 4*x + 3*y, u_y = 2*x + 5*y
            for direction in range(2):
                coeff = [4, 3] if direction == 0 else [2, 5]
                u.s[direction, 0, :, :] = (coeff[0] * nodal_coords.s[0, :, :] +
                                            coeff[1] * nodal_coords.s[1, :, :])

            grad_u = discretization.get_displacement_gradient_sized_field(name='grad_u')
            discretization.apply_gradient_operator_mugrid(u, grad_u)

            # Verify gradient values
            err_00 = np.max(np.abs(grad_u.s[0, 0, ..., :-1, :-1] - 4.0))
            err_01 = np.max(np.abs(grad_u.s[0, 1, ..., :-1, :-1] - 3.0))
            err_10 = np.max(np.abs(grad_u.s[1, 0, ..., :-1, :-1] - 2.0))
            err_11 = np.max(np.abs(grad_u.s[1, 1, ..., :-1, :-1] - 5.0))

            assert np.allclose(grad_u.s[0, 0, ..., :-1, :-1], 4.0, atol=1e-14), \
                f'∂u_x/∂x incorrect for {element_type}: error={err_00}'
            assert np.allclose(grad_u.s[0, 1, ..., :-1, :-1], 3.0, atol=1e-14), \
                f'∂u_x/∂y incorrect for {element_type}: error={err_01}'
            assert np.allclose(grad_u.s[1, 0, ..., :-1, :-1], 2.0, atol=1e-14), \
                f'∂u_y/∂x incorrect for {element_type}: error={err_10}'
            assert np.allclose(grad_u.s[1, 1, ..., :-1, :-1], 5.0, atol=1e-14), \
                f'∂u_y/∂y incorrect for {element_type}: error={err_11}'

    def test_displacement_gradient_zero_mean(self):
        """Gradient of displacement field should have zero mean."""
        domain_size = [3, 4]
        problem_type = 'elasticity'
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)
        number_of_pixels = (4, 5)
        discretization_type = 'finite_element'

        for element_type in ['linear_triangles', 'bilinear_rectangle']:
            discretization = domain.Discretization(cell=my_cell,
                                                   nb_of_pixels_global=number_of_pixels,
                                                   discretization_type=discretization_type,
                                                   element_type=element_type)

            nodal_coords = discretization.get_nodal_points_coordinates()
            u = discretization.get_displacement_sized_field(name='u')

            for direction in range(2):
                u.s[direction, 0, :, :] = np.sin(2 * np.pi * nodal_coords.s[0, :, :])

            grad_u = discretization.get_displacement_gradient_sized_field(name='grad_u')
            discretization.apply_gradient_operator_mugrid(u, grad_u)

            mean = np.mean(grad_u.s)
            assert np.abs(mean) <= 1e-13, \
                f'Displacement gradient mean not zero for {element_type}: mean={mean}'


class TestGradientProperties:
    """Test mathematical properties of gradient operators."""

    def test_gradient_adjoint_property(self):
        """Test adjoint property: <grad(u), v> = <u, -div(v)> for periodic BCs."""
        domain_size = [3, 4]
        problem_type = 'conductivity'
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)
        number_of_pixels = (4, 5)
        discretization_type = 'finite_element'

        discretization = domain.Discretization(cell=my_cell,
                                               nb_of_pixels_global=number_of_pixels,
                                               discretization_type=discretization_type,
                                               element_type='bilinear_rectangle')

        nodal_coords = discretization.get_nodal_points_coordinates()

        # Create random fields
        u = discretization.get_temperature_sized_field(name='u')
        u.s[0, 0, :, :] = np.random.rand(*u.s[0, 0].shape)
        u.s[0, 0, :, :] -= u.s[0, 0].mean()

        # Create random gradient field
        grad_v = discretization.get_temperature_gradient_size_field(name='grad_v')
        grad_v.s[...] = np.random.rand(*grad_v.s.shape)

        # Compute grad(u)
        grad_u = discretization.get_temperature_gradient_size_field(name='grad_u')
        discretization.apply_gradient_operator_mugrid(u, grad_u)

        # Compute div(grad_v)
        div_grad_v = discretization.get_temperature_sized_field(name='div_grad_v')
        discretization.apply_gradient_transposed_operator_mugrid(
            gradient_field_ijqxyz=grad_v,
            div_u_fnxyz=div_grad_v,
            apply_weights=False)

        # Compute inner products
        lhs = np.sum(grad_u.s * grad_v.s)  # <grad(u), v>
        rhs = np.sum(u.s * div_grad_v.s)  # -<u, div(v)>

        # Note: due to discretization, these won't be exactly equal
        error = np.abs(lhs - rhs)
        assert np.allclose(lhs, rhs, rtol=1e-10), \
            f'Adjoint property violated: lhs={lhs}, rhs={rhs}, error={error}'



class TestGradientWeightedSum:
    """Test gradient operations with quadrature weights."""

    def test_weighted_transpose_identity(self):
        """Test that weighted transpose followed by forward is identity for constant field."""
        domain_size = [3, 3]
        problem_type = 'conductivity'
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)
        number_of_pixels = (3, 3)
        discretization_type = 'finite_element'

        for element_type in ['linear_triangles', 'linear_triangles_tilled', 'bilinear_rectangle']:
            discretization = domain.Discretization(cell=my_cell,
                                                   nb_of_pixels_global=number_of_pixels,
                                                   discretization_type=discretization_type,
                                                   element_type=element_type)

            # Create a constant field at nodes
            u = discretization.get_temperature_sized_field(name='u')
            u.s[0, 0, :, :] = 1.0

            # Get it at quad points via interpolation
            u_at_quad = discretization.get_quad_field_scalar(name='u_at_quad')
            discretization.apply_N_operator_mugrid(nodal_field_inxyz=u, quad_field_ijqnxyz=u_at_quad)

            # Apply transpose with weights
            u_back = discretization.get_temperature_sized_field(name='u_back')
            discretization.apply_N_transposed_operator_mugrid(
                quad_field_ijqxyz=u_at_quad,
                nodal_field_inxyz=u_back,
                apply_weights=True)

            # Result should be constant and equal to 1
            assert np.allclose(u_back.s, 1.0, rtol=1e-12, atol=1e-12), \
                f'N^T * W * N * 1 != 1 for {element_type}'



