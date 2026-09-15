import sys
import warnings

sys.path.append('..')  # Add parent directory to path

import numpy as np
import pytest

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import material_models

# Define which elements work in which dimensions # look at element_types in discretization_fixture
element_types_list = ['linear_triangles', 'linear_triangles_tilled',
                      'bilinear_rectangle', 'biquadratic_rectangle',
                      'trilinear_hexahedron', 'trilinear_hexahedron_1Q',
                      'linear_1D', 'quadratic_1D', ]
element_1d = [7,6]  # linear_1D, quadratic_1D, etc.
element_2d = [0, 1, 2, 3]  # linear_triangles, bilinear_rectangle, etc.
element_3d = [4, 5]  # trilinear_hexahedron, etc.

domain_1d = [[1.3], [0.5]]
domain_2d = [[1.3, 3.2], ]
domain_3d = [[1.3, 3.2, 0.7]]

pixels_1d = [ [3], [4]]
pixels_2d = [[2, 3], [2, 4], [3, 2], [3, 4], [4, 2], [4, 3]]
pixels_3d = [[2, 3, 5], [3, 5, 2], [5, 3, 2]]

problem_types = ['conductivity', 'elasticity']

# Generate valid combinations (domain, element, pixels, problem_type)
oned_test_cases = (
    [(d, e, p, pt) for d in domain_1d for e in element_1d for p in pixels_1d for pt in problem_types]
)
debug_test_cases = (
    [(d, e, p, pt) for d in [[5]] for e in [7] for p in [[5]] for pt in ['conductivity']]
)

twod_test_cases = (
    [(d, e, p, pt) for d in domain_2d for e in element_2d for p in pixels_2d for pt in problem_types]
)

threed_test_cases = (
    [(d, e, p, pt) for d in domain_3d for e in element_3d for p in pixels_3d for pt in problem_types]
)

all_test_cases = twod_test_cases + threed_test_cases

# debug_test_cases = (
#     [(d, e, p, pt) for d in [[1, 1]] for e in [3] for p in [[5, 5]] for pt in ['conductivity']]
# )
debug_test_cases = (
    [(d, e, p, pt) for d in [[5]] for e in [7] for p in [[5]] for pt in ['conductivity']]
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def problem_type_fixture(request):
    """Extract problem_type from parametrized test."""
    # request.param is just the problem_type string (tc[3])
    return request.param


@pytest.fixture
def discretization_fixture(request):
    """Create discretization from parametrized test case."""
    domain_size, element_idx, nb_pixels, problem_type = request.param

    my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                      problem_type=problem_type)
    discretization_type = 'finite_element'
    discretization = domain.Discretization(
        cell=my_cell,
        nb_of_pixels_global=nb_pixels,
        discretization_type=discretization_type,
        element_type=element_types_list[element_idx]
    )
    return discretization


@pytest.fixture
def material_data_field_fixture(discretization_fixture, problem_type_fixture):
    """Create material data field based on problem type and dimension."""
    discretization = discretization_fixture
    problem_type = problem_type_fixture

    material_data_field = discretization.get_material_data_size_field_mugrid(
        name='material_dat'
    )

    if problem_type == 'elasticity':
        K_1, G_1 = material_models.get_bulk_and_shear_modulus(E=3, poisson=0.2)
        mat_1 = material_models.get_elastic_material_tensor(
            dim=discretization.domain_dimension,
            K=K_1,
            mu=G_1,
            kind='linear'
        )
        dim = discretization.domain_dimension

        if dim == 1:
            material_data_field.s[...] = mat_1[:, :, :, :,np.newaxis, np.newaxis]
        if dim == 2:
            material_data_field.s[...] = mat_1[:, :, :, :,np.newaxis, np.newaxis, np.newaxis]
        elif dim == 3:
            material_data_field.s[...] = mat_1[:, :, :, :,np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    elif problem_type == 'conductivity':
        dim = discretization.domain_dimension
        mat_1 = np.eye(dim)  # Identity matrix for dimension
        if dim == 1:
            material_data_field.s[...] = mat_1[:, :, np.newaxis, np.newaxis]
        if dim == 2:
            material_data_field.s[...] = mat_1[:, :, np.newaxis, np.newaxis, np.newaxis]
        elif dim == 3:
            material_data_field.s[...] = mat_1[:, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]

    return material_data_field, mat_1


@pytest.fixture
def preconditioner_fixture(discretization_fixture, material_data_field_fixture):
    """Create preconditioner."""
    discretization = discretization_fixture
    material_data_field, ref_material_data = material_data_field_fixture

    preconditioner = discretization.get_preconditioner_Green_mugrid(
        reference_material_data_ijkl=ref_material_data
    )
    return preconditioner, discretization, material_data_field, ref_material_data


@pytest.fixture
def random_field_fixture(discretization_fixture):
    """Create random field x_0."""
    discretization = discretization_fixture
    x_0 = discretization.get_unknown_size_field(name='x_0')
    x_0.s[...] = np.random.rand(*x_0.s.shape)

    # Remove mean from each field component
    for f in range(discretization.cell.unknown_shape[0]):
        x_0.s[f] -= x_0.s[f].mean()

    return x_0


# ============================================================================
# Tests
# ============================================================================
debug_test_cases_fft = (
    [(d, e, p, pt) for d in [[5]] for e in [7] for p in [[5]] for pt in ['conductivity', 'elasticity']]
)
debug_test_cases_quad_2d = (
    [(d, e, p, pt) for d in [[1, 1]] for e in [3] for p in [[5, 5]] for pt in ['conductivity', 'elasticity']]
)


@pytest.mark.parametrize('discretization_fixture,problem_type_fixture',
                         [(tc, tc[3]) for tc in all_test_cases],  # all_test_cases
                         indirect=['discretization_fixture', 'problem_type_fixture'])
def test_fft_on_multiple_nodes(
        discretization_fixture,
        problem_type_fixture,
        random_field_fixture
):
    """
     The goal of this test is to verify that the Fast Fourier Transform (FFT) operation is correctly implemented on multiple nodes.
     I will compare fft on multiple nodes. with multiple fft on single nodes.
    """

    # first compute FFT on multiple nodes
    discretization = discretization_fixture
    if discretization.nb_nodes_per_pixel == 1:
        x_0 = random_field_fixture
        original = x_0.s.copy()

        fx_0 = discretization.ffield_collection.complex_field(
            name='unit_impulse_response_inqks',  # name of the field
            components=(discretization.unknown_size[0],),  # shape of components
            sub_pt='nodal_points'
        )

        # f_0 = discretization.get_unknown_size_field(name='f_0')

        discretization.fft.communicate_ghosts(x_0)
        fx_0.sg.fill(0)
        discretization.fft.fft(x_0, fx_0)

        # Inverse FFT: Fourier -> real
        discretization.fft.ifft(fx_0, x_0)

        # Apply normalization for roundtrip
        x_0.s[:] *= discretization.fft.normalisation

        # Verify roundtrip
        np.testing.assert_allclose(x_0.s, original, atol=1e-14,
                                   err_msg=f"Roundtrip error: {np.max(np.abs(x_0.s - original)):.2e}")

    # do multiple a single node ffts
    if discretization.nb_nodes_per_pixel > 1:
        x_0 = random_field_fixture
        original = x_0.s.copy()

        fx_0_single_node = discretization.ffield_collection.complex_field(
            name='fourier_field_inqks_single',  # name of the field
            components=(discretization.unknown_size[0],),  # shape of components
        )

        x_0_single_node = discretization.field_collection.real_field(
            name='real_field_inqks_single',  # name of the field
            components=(discretization.unknown_size[0],),  # shape of components
        )
        for node in np.arange(discretization.nb_nodes_per_pixel):
            x_0_single_node.s[:, 0, :] = np.copy(x_0.s[:, node, :])

            discretization.fft.communicate_ghosts(x_0_single_node)
            fx_0_single_node.sg.fill(0)

            discretization.fft.fft(x_0_single_node, fx_0_single_node)

            # Inverse FFT: Fourier -> real
            discretization.fft.ifft(fx_0_single_node, x_0_single_node)

            # Apply normalization for roundtrip
            x_0_single_node.s[:] *= discretization.fft.normalisation

            #  fill back the original field
            x_0.s[:, node, ...] = np.copy(x_0_single_node.s[:, 0, ...])

            # Verify roundtrip

            np.testing.assert_allclose(x_0_single_node.s[:, 0, ...], original[:, node, ...], atol=1e-14,
                                       err_msg=f"Roundtrip error: {np.max(np.abs(x_0_single_node.s[:, 0, ...] - original[:, node, ...])):.2e}")
        np.testing.assert_allclose(x_0.s, original, atol=1e-14,
                                   err_msg=f"Roundtrip error: {np.max(np.abs(x_0.s - original)):.2e}")

    print()
    # create quad point field, and compute fft on that=. This may work
    x_0_multi_nodal = random_field_fixture
    discretization.fft.communicate_ghosts(x_0_multi_nodal)
    original = x_0_multi_nodal.s.copy()

    fx_0_multi_nodal = discretization.ffield_collection.complex_field(
        name='fourier_field_inqks_multi_nodal ',  # name of the field
        components=(discretization.unknown_size[0],),  # shape of components
        sub_pt='nodal_points'
    )

    # x_0_multi_nodal = discretization.get_unknown_size_field(name='real_field_inqks_multi_nodal')

    fx_0_multi_nodal.sg.fill(0)
    # Forward FFT: real -> Fourier
    discretization.multinodal_fft(real_field=x_0_multi_nodal,
                                  fourier_field=fx_0_multi_nodal)

    x_0_multi_nodal.sg.fill(0)
    # Inverse FFT: Fourier -> real
    discretization.multinodal_ifft(fourier_field=fx_0_multi_nodal,
                                   real_field=x_0_multi_nodal)
    # Apply normalization for roundtrip
    discretization.multinodal_fft_normalisation(real_field=x_0_multi_nodal)

    np.testing.assert_allclose(x_0_multi_nodal.s[...], original[...], atol=1e-14,
                               err_msg=f"Roundtrip error: {np.max(np.abs(x_0_multi_nodal.s - original)):.2e}")



@pytest.mark.parametrize('discretization_fixture,problem_type_fixture',
                         [(tc, tc[3]) for tc in all_test_cases],  #  oned_test_cases
                         indirect=['discretization_fixture', 'problem_type_fixture'])
def test_green_preconditioner_is_inverse_of_homogeneous_problem(
        discretization_fixture,
        material_data_field_fixture,
        preconditioner_fixture,
        random_field_fixture
):
    """Test that preconditioner is inverse of system matrix for homogeneous data."""
    discretization = discretization_fixture
    material_data_field, _ = material_data_field_fixture
    preconditioner, _, _, _ = preconditioner_fixture
    x_0 = random_field_fixture
    print( f'element {discretization.element_type}')
    print( f'domain_size {discretization.domain_size}')
    print( f'nb_of_pixels {discretization.nb_of_pixels}')

    # Define system matrix application
    def K_fun(x, Ax):
        discretization.apply_system_matrix_mugrid(
            material_data_field=material_data_field,
            input_field_inxyz=x,
            output_field_inxyz=Ax
        )

    # Define preconditioner application
    def M_fun(x, Px):
        discretization.apply_preconditioner_mugrid(
            preconditioner_Fourier_fnfnqks=preconditioner,
            input_nodal_field_fnxyz=x,
            output_nodal_field_fnxyz=Px
        )

    # Apply system matrix: f_0 = K @ x_0
    f_0 = discretization.get_unknown_size_field(name='f_0')
    K_fun(x_0, f_0)
    #K=discretization.get_system_matrix_mugrid( material_data_field=material_data_field)
    # Apply preconditioner: x_1 = M @ f_0 (should recover x_0)
    x_1 = discretization.get_unknown_size_field(name='x_1')
    M_fun(f_0, x_1)

    # Check that preconditioner inverts system matrix
    diff = x_0.s - x_1.s
    assert_condition = np.allclose(x_0.s, x_1.s, rtol=1e-10, atol=1e-10)

    assert assert_condition, (
        f'Preconditioner is not the inverse of the system matrix with '
        f'homogeneous data. Discrepancy = {diff}'
        f'element {discretization.element_type}'
    )


@pytest.mark.parametrize('discretization_fixture', all_test_cases, indirect=True)
def test_discretization_properties(discretization_fixture):
    """Test that discretization has all required properties."""
    assert hasattr(discretization_fixture, "cell")
    assert hasattr(discretization_fixture, "domain_dimension")
    assert hasattr(discretization_fixture, "B_grad_at_pixel_dqnijk")
    assert hasattr(discretization_fixture, "quadrature_weights")
    assert hasattr(discretization_fixture, "nb_quad_points_per_pixel")
    assert hasattr(discretization_fixture, "nb_nodes_per_pixel")
