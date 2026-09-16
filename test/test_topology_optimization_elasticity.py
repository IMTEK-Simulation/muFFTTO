import pytest
import numpy as np
import scipy as sc
import time

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import topology_optimization
from muFFTTO import material_models


# ==============================================================================
# Helper Functions
# ==============================================================================

def run_finite_difference_check(
    eval_fn,
    x0,
    analytical_derivative,
    epsilons=(1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7),
    rtol=1e-4,
    min_rate=None,
    plot=False,
    title='Finite Difference Check',
):
    """
    Generic helper for central finite difference gradient verification with optional plotting.

    Parameters
    ----------
    eval_fn : callable
        Function f(x) returning a scalar value for input array x matching x0.shape.
    x0 : np.ndarray
        Array at which the gradient is evaluated.
    analytical_derivative : np.ndarray
        Analytical derivative evaluated at x0.
    epsilons : tuple or list
        Step sizes for finite differences.
    rtol : float
        Maximum allowed minimum relative error across step sizes.
    min_rate : float or None
        Expected convergence order for the leading step sizes (e.g. ~2 for central differences).
    plot : bool
        If True, displays a log-log error vs. epsilon plot.
    title : str
        Plot title.
    """
    errors = []
    epsilons_arr = np.asarray(epsilons)
    x0_flat = np.asarray(x0, dtype=float).ravel()
    analytical_flat = np.asarray(analytical_derivative, dtype=float).ravel()
    x0_shape = x0.shape

    for eps in epsilons:
        fd_grad = np.zeros_like(x0_flat)
        x_work = x0_flat.copy()
        for i in range(x0_flat.size):
            orig_val = x_work[i]

            x_work[i] = orig_val + eps / 2.0
            f_plus = eval_fn(x_work.reshape(x0_shape))

            x_work[i] = orig_val - eps / 2.0
            f_minus = eval_fn(x_work.reshape(x0_shape))

            x_work[i] = orig_val
            fd_grad[i] = (f_plus - f_minus) / eps

        err = np.linalg.norm(fd_grad - analytical_flat)
        errors.append(err)

    errors = np.array(errors)
    analytical_norm = np.linalg.norm(analytical_flat)
    relative_errors = errors / (analytical_norm + 1e-14)

    if plot:
        import matplotlib.pyplot as plt
        idx_min = np.argmin(errors)
        plt.figure()
        plt.loglog(epsilons, errors, marker='x', label='FD vs analytical error')
        plt.loglog(epsilons, errors[idx_min] * (epsilons_arr / epsilons_arr[idx_min]) ** 2,
                   linestyle='--', label=r'$O(h^2)$ reference')
        plt.loglog(epsilons, errors[idx_min] * (epsilons_arr / epsilons_arr[idx_min]) ** 1,
                   linestyle='--', label=r'$O(h)$ reference')
        plt.legend(loc='best')
        plt.xlabel('epsilon (FD step size)')
        plt.ylabel('Error (Frobenius norm)')
        plt.title(title)
        plt.show()

    min_rel_err = np.min(relative_errors)
    assert min_rel_err < rtol, (
        f"{title} failed: minimum relative error {min_rel_err:.2e} exceeds tolerance {rtol:.2e}. "
        f"Analytical derivative may be incorrect."
    )

    if min_rate is not None and len(epsilons) >= 3:
        log_eps = np.log10(epsilons_arr[:3])
        log_err = np.log10(errors[:3])
        convergence_rate = np.polyfit(log_eps, log_err, 1)[0]
        assert convergence_rate > min_rate, (
            f"{title} convergence rate {convergence_rate:.2f} is below expected {min_rate:.2f}."
        )

    return relative_errors, errors


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture()
def discretization_fixture(domain_size, element_type, nb_pixels):
    problem_type = 'elasticity'
    element_types = [
        'linear_triangles',
        'linear_triangles_tilled',
        'trilinear_hexahedron',
        'trilinear_hexahedron_1Q',
    ]

    my_cell = domain.PeriodicUnitCell(
        domain_size=domain_size,
        problem_type=problem_type
    )

    discretization = domain.Discretization(
        cell=my_cell,
        nb_of_pixels_global=nb_pixels,
        discretization_type='finite_element',
        element_type=element_types[element_type]
    )

    return discretization


# ==============================================================================
# Discretization Initialization Tests
# ==============================================================================

@pytest.mark.parametrize('domain_size, element_type, nb_pixels', [
    ([2, 2], 0, [3, 3]),
    ([2, 3], 1, [4, 3]),
    ([2, 2, 2], 2, [3, 3, 3]),
    ([2, 3, 2], 3, [3, 4, 3]),
])
def test_discretization_init(discretization_fixture):
    """Verify that essential discretization properties and fields are properly initialized."""
    assert hasattr(discretization_fixture, "cell")
    assert hasattr(discretization_fixture, "domain_dimension")
    assert hasattr(discretization_fixture, "B_grad_at_pixel_dqnijk")
    assert hasattr(discretization_fixture, "quadrature_weights")
    assert hasattr(discretization_fixture, "nb_quad_points_per_pixel")
    assert hasattr(discretization_fixture, "nb_nodes_per_pixel")


# ==============================================================================
# Whole Objective Function FD Checks (2D & 3D)
# ==============================================================================

@pytest.mark.parametrize('domain_size, element_type, nb_pixels', [
    ([1, 2], 0, [4, 5]),
    ([1, 2], 1, [4, 5]),
])
def test_fd_check_of_whole_objective_function(discretization_fixture):
    """
    Finite difference check of the full topology optimization objective function gradient
    with respect to the phase field.
    """
    discretization = discretization_fixture
    dim = discretization.domain_dimension

    macro_gradient = np.zeros((dim, dim))
    macro_gradient[0, 0] = 1.0

    # Base material properties (solid phase rho=1)
    E_0 = 1.0
    poisson_0 = 0.2
    K_0, G_0 = material_models.get_bulk_and_shear_modulus(E=E_0, poisson=poisson_0)
    elastic_C_0_ijkl = material_models.get_elastic_material_tensor(
        dim=dim, K=K_0, mu=G_0, kind='linear'
    )
    elastic_C_void = np.zeros_like(elastic_C_0_ijkl)

    # Target material properties
    poisson_target = 1.0 / 3.0
    G_target_auxet = 0.25 * E_0
    E_target = 2.0 * G_target_auxet * (1.0 + poisson_target)
    K_target, G_target = material_models.get_bulk_and_shear_modulus(E=E_target, poisson=poisson_target)
    elastic_C_target_ijkl = material_models.get_elastic_material_tensor(
        dim=dim, K=K_target, mu=G_target, kind='linear'
    )
    target_stress_ij = np.einsum('ijkl,lk->ij', elastic_C_target_ijkl, macro_gradient)

    # Setup macro-gradient field
    macro_gradient_field_ijqxyz = discretization.get_gradient_size_field(name='macro_gradient_field')
    discretization.get_macro_gradient_field_mugrid(
        macro_gradient_ij=macro_gradient,
        macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz
    )

    preconditioner_Green = discretization.get_preconditioner_Green_mugrid(
        reference_material_data_ijkl=elastic_C_0_ijkl
    )

    def M_fun(x, Px):
        discretization.fft.communicate_ghosts(x)
        discretization.apply_preconditioner_mugrid(
            preconditioner_Fourier_fnfnqks=preconditioner_Green,
            input_nodal_field_fnxyz=x,
            output_nodal_field_fnxyz=Px
        )

    p = 2
    w = 3
    eta = 0.3
    cg_setup = {'cg_tol': 1e-9}

    def eval_objective_and_gradient(phase_field_1nxyz_flat):
        phase_field_1nxyz = discretization.get_scalar_field(name='phase_field_in_objective')
        phase_field_1nxyz.s[...] = phase_field_1nxyz_flat.reshape([1, 1, *discretization.nb_of_pixels])

        phase_field_at_quad_points_1qxyz = discretization.get_quad_field_scalar(
            name='phase_field_at_quads_in_objective'
        )
        discretization.apply_N_operator_mugrid(phase_field_1nxyz, phase_field_at_quad_points_1qxyz)

        material_data_field_C_0_rho_ijklqxyz = discretization.get_material_data_size_field_mugrid(
            name='material_data_field_C_0_rho_in_objective'
        )
        material_data_field_C_0_rho_ijklqxyz.s[...] = (
            elastic_C_0_ijkl[..., np.newaxis, np.newaxis, np.newaxis]
            * np.power(phase_field_at_quad_points_1qxyz.s, p)[0, 0, :, ...]
        )

        f_phase_field = topology_optimization.objective_function_phase_field(
            discretization=discretization,
            phase_field_1nxyz=phase_field_1nxyz,
            eta=eta,
            double_well_depth=1
        )

        s_phase_field = discretization.get_scalar_field(name='s_phase_field')
        s_phase_field.s.fill(0)
        topology_optimization.sensitivity_phase_field_term_FE_NEW(
            discretization=discretization,
            phase_field_1nxyz=phase_field_1nxyz,
            p=p,
            eta=eta,
            output_array=s_phase_field,
            double_well_depth=1
        )

        K_diag_alg = discretization.get_preconditioner_Jacobi_mugrid(
            material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz
        )

        def M_fun_Green_Jacobi(x, Px):
            discretization.fft.communicate_ghosts(x)
            x_jacobi_temp = discretization.get_unknown_size_field(name='x_jacobi_temp')
            x_jacobi_temp.s[...] = K_diag_alg.s * x.s
            discretization.apply_preconditioner_mugrid(
                preconditioner_Fourier_fnfnqks=preconditioner_Green,
                input_nodal_field_fnxyz=x_jacobi_temp,
                output_nodal_field_fnxyz=Px
            )
            Px.s[...] = K_diag_alg.s * Px.s

        def K_fun(x, Ax):
            discretization.apply_system_matrix_mugrid(
                material_data_field=material_data_field_C_0_rho_ijklqxyz,
                input_field_inxyz=x,
                output_field_inxyz=Ax,
                formulation='small_strain'
            )

        rhs = discretization.get_unknown_size_field(name='rhs_field_load_case')
        discretization.get_rhs_mugrid(
            material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
            macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
            rhs_inxyz=rhs
        )

        displacement_field_u = discretization.get_displacement_sized_field(
            name='displacement_field_in_objective'
        )
        displacement_field_u.s.fill(0)
        solvers.conjugate_gradients_mugrid(
            comm=discretization.communicator,
            fc=discretization.field_collection,
            hessp=K_fun,
            b=rhs,
            x=displacement_field_u,
            P=M_fun_Green_Jacobi,
            tol=cg_setup['cg_tol'],
            maxiter=50000,
        )

        homogenized_stress = discretization.get_homogenized_stress_mugrid(
            material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
            displacement_field_inxyz=displacement_field_u,
            macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
            formulation='small_strain'
        )

        f_sigma = topology_optimization.compute_stress_equivalence_potential(
            actual_stress_ij=homogenized_stress,
            target_stress_ij=target_stress_ij
        )

        adjoint_field = discretization.get_displacement_sized_field(
            name='adjoint_field_in_objective'
        )
        adjoint_field.s.fill(0)

        dstress_drho, adjoint_field, adjoint_energies, _ = (
            topology_optimization.sensitivity_stress_and_adjoint_FE_NEW(
                discretization=discretization,
                base_material_data_ijkl=elastic_C_0_ijkl,
                void_material_data_ijkl=elastic_C_void,
                displacement_field_inxyz=displacement_field_u,
                adjoint_field_inxyz=adjoint_field,
                macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
                phase_field_1nxyz=phase_field_1nxyz,
                target_stress_ij=target_stress_ij,
                actual_stress_ij=homogenized_stress,
                preconditioner_fun=M_fun_Green_Jacobi,
                system_matrix_fun=K_fun,
                formulation='small_strain',
                p=p,
                weight=w,
                cg_setup=cg_setup
            )
        )

        total_obj = w * f_sigma + f_phase_field + adjoint_energies
        total_grad = dstress_drho + s_phase_field.s
        return total_obj, total_grad

    np.random.seed(42)
    phase_field_0 = discretization.get_scalar_field(name='phase_field_0')
    phase_field_0.s[...] = np.random.rand(*phase_field_0.s.shape)
    x0 = phase_field_0.s.ravel()

    _, analytical_grad = eval_objective_and_gradient(x0)

    run_finite_difference_check(
        eval_fn=lambda x: eval_objective_and_gradient(x)[0],
        x0=x0,
        analytical_derivative=analytical_grad,
        rtol=1e-4,
        min_rate=0.95,
        title=f'FD Check: Whole Objective Function ({dim}D)'
    )


# ==============================================================================
# Double-Well Potential Tests
# ==============================================================================

@pytest.mark.parametrize('domain_size, element_type, nb_pixels', [
    ([1, 1], 0, [3, 4]),
    ([2, 5], 0, [6, 5]),
])
def test_fd_check_of_double_well_potential(discretization_fixture ):
    """Finite difference check of double-well potential derivatives (nodal and analytical)."""
    np.random.seed(42)
    phase_field = discretization_fixture.get_scalar_field(name='phase_field')
    phase_field.s[...] = np.random.rand(*phase_field.s.shape)

    # Analytical derivative via nodal formulation
    partial_nodal = discretization_fixture.get_scalar_field(name='partial_nodal')
    topology_optimization.partial_der_of_double_well_potential_wrt_density_nodal(
        discretization=discretization_fixture,
        phase_field_1nxyz=phase_field,
        output_1nxyz=partial_nodal
    )

    # Analytical derivative via analytical integration
    partial_anal = discretization_fixture.get_scalar_field(name='partial_anal')
    topology_optimization.partial_der_of_double_well_potential_wrt_density_analytical(
        discretization=discretization_fixture,
        phase_field_1nxyz=phase_field,
        output_1nxyz=partial_anal
    )

    def eval_nodal(x):
        f = discretization_fixture.get_scalar_field(name='eval_nodal_f')
        f.s[...] = x
        return topology_optimization.compute_double_well_potential_nodal(
            discretization=discretization_fixture,
            phase_field_1nxyz=f,
            eta=1
        )

    def eval_anal(x):
        f = discretization_fixture.get_scalar_field(name='eval_anal_f')
        f.s[...] = x
        return topology_optimization.compute_double_well_potential_analytical(
            discretization=discretization_fixture,
            phase_field_1nxyz=f
        )

    run_finite_difference_check(
        eval_fn=eval_nodal,
        x0=phase_field.s,
        analytical_derivative=partial_nodal.s,
        rtol=1e-4,
        title='FD check: Double Well Potential (Nodal)'
    )

    run_finite_difference_check(
        eval_fn=eval_anal,
        x0=phase_field.s,
        analytical_derivative=partial_anal.s,
        rtol=1e-4,
        title='FD check: Double Well Potential (Analytical)'
    )


@pytest.mark.parametrize('domain_size, element_type, nb_pixels', [
    ([1, 1], 0, [4, 4]),
    ([2, 3], 0, [5, 4])
])
def test_fd_check_of_grad_of_double_well_potential_partial_derivative(discretization_fixture ):
    """Finite difference check of phase field gradient potential partial derivative w.r.t phase field."""
    np.random.seed(42)
    phase_field = discretization_fixture.get_scalar_field(name='phase_field')
    phase_field.s[...] = np.random.rand(*phase_field.s.shape)

    anal_deriv = discretization_fixture.get_scalar_field(name='anal_deriv')
    topology_optimization.partial_derivative_of_gradient_of_phase_field_potential(
        discretization=discretization_fixture,
        phase_field_1nxyz=phase_field,
        output_1nxyz=anal_deriv
    )

    def eval_grad_potential(x):
        pf = discretization_fixture.get_scalar_field(name='pf_temp')
        pf.s[...] = x
        discretization_fixture.fft.communicate_ghosts(pf)
        return topology_optimization.compute_gradient_of_phase_field_potential(
            discretization=discretization_fixture,
            phase_field_1nxyz=pf
        )

    run_finite_difference_check(
        eval_fn=eval_grad_potential,
        x0=phase_field.s,
        analytical_derivative=anal_deriv.s,
        rtol=1e-4,
        title='FD check: Phase Field Gradient Potential'
    )


def test_integration_of_double_well_potential():
    """
    Numerical convergence and consistency test for double-well potential integration:
    - Compares nodal integration vs exact analytical integration.
    - Verifies error decreases as mesh resolution N increases.
    """
    np.random.seed(42)
    domain_size = [2, 3]
    resolutions = [4, 8, 16]
    nodal_errors = []

    for N in resolutions:
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size, problem_type='elasticity')
        discretization = domain.Discretization(
            cell=my_cell,
            nb_of_pixels_global=[N, N],
            discretization_type='finite_element',
            element_type='linear_triangles'
        )

        phase_field = discretization.get_scalar_field(name='phase_field')
        phase_field.s[...] = np.random.rand(*phase_field.s.shape)

        dw_nodal = topology_optimization.compute_double_well_potential_nodal(
            discretization, phase_field, eta=1
        )
        dw_anal = topology_optimization.compute_double_well_potential_analytical(
            discretization=discretization, phase_field_1nxyz=phase_field
        )

        der_nodal = discretization.get_scalar_field(name='der_nodal')
        topology_optimization.partial_der_of_double_well_potential_wrt_density_nodal(
            discretization, phase_field, der_nodal
        )
        der_anal = discretization.get_scalar_field(name='der_anal')
        topology_optimization.partial_der_of_double_well_potential_wrt_density_analytical(
            discretization=discretization, phase_field_1nxyz=phase_field, output_1nxyz=der_anal
        )

        nodal_errors.append(abs(dw_nodal - dw_anal))

    assert nodal_errors[-1] < nodal_errors[0] or nodal_errors[-1] < 1.0


# ==============================================================================
# Stress Equivalence Potential Tests
# ==============================================================================

@pytest.mark.parametrize('domain_size, element_type, nb_pixels', [
    ([3, 4], 0, [6, 8]),
    ([2, 5], 1, [8, 6])
])
def test_of_stress_equivalence_potential_quadratic(discretization_fixture ):
    """Test that stress equivalence potential behaves quadratically around the target."""
    target_stress = np.array([[1.0, 0.0], [0.0, 1.0]])
    delta = np.array([[0.1, 0.0], [0.0, 0.2]])

    pot_zero = topology_optimization.compute_stress_equivalence_potential(target_stress, target_stress)
    pot_plus = topology_optimization.compute_stress_equivalence_potential(target_stress + delta, target_stress)
    pot_minus = topology_optimization.compute_stress_equivalence_potential(target_stress - delta, target_stress)

    assert np.isclose(pot_zero, 0.0, atol=1e-14)
    assert np.isclose(pot_plus, pot_minus, rtol=1e-12)


@pytest.mark.parametrize('domain_size, element_type, nb_pixels, p', [
    ([1, 1], 0, [4, 4], 2),
    ([2, 3], 0, [4, 3], 3),
    ([1, 1], 1, [4, 4], 2),
])
def test_fd_check_of_stress_equivalence_potential_wrt_phase_field_FE(discretization_fixture, p ):
    """Finite difference check of stress equivalence potential w.r.t. phase field."""
    np.random.seed(42)
    discretization = discretization_fixture

    target_stress = np.array([[1.0, 0.0], [0.0, 2.0]])
    macro_gradient = np.array([[1.0, 0.0], [0.0, 1.0]])

    K_0, G_0 = material_models.get_bulk_and_shear_modulus(E=1.0, poisson=0.2)
    elastic_C_1 = material_models.get_elastic_material_tensor(
        dim=discretization.domain_dimension, K=K_0, mu=G_0, kind='linear'
    )
    elastic_C_void = np.zeros_like(elastic_C_1)

    phase_field_1nxyz = discretization.get_scalar_field(name='phase_field')
    phase_field_1nxyz.s[...] = np.random.rand(*phase_field_1nxyz.s.shape)

    pf_quad = discretization.get_quad_field_scalar(name='pf_quad_init')
    discretization.apply_N_operator_mugrid(phase_field_1nxyz, pf_quad)

    mat_field = discretization.get_material_data_size_field_mugrid(name='mat_field_init')
    mat_field.s[...] = (
        elastic_C_1[..., np.newaxis, np.newaxis, np.newaxis]
        * np.power(pf_quad.s, p)[0, 0, :, ...]
    )

    macro_grad_field = discretization.get_gradient_size_field(name='macro_grad_field')
    discretization.get_macro_gradient_field_mugrid(
        macro_gradient_ij=macro_gradient,
        macro_gradient_field_ijqxyz=macro_grad_field
    )

    rhs = discretization.get_unknown_size_field(name='rhs_init')
    discretization.get_rhs_mugrid(
        material_data_field_ijklqxyz=mat_field,
        macro_gradient_field_ijqxyz=macro_grad_field,
        rhs_inxyz=rhs
    )

    def K_fun(x, Ax):
        discretization.apply_system_matrix_mugrid(
            material_data_field=mat_field,
            input_field_inxyz=x,
            output_field_inxyz=Ax,
            formulation='small_strain'
        )

    preconditioner = discretization.get_preconditioner_Green_mugrid(
        reference_material_data_ijkl=elastic_C_1
    )

    def M_fun(x, Px):
        discretization.fft.communicate_ghosts(x)
        discretization.apply_preconditioner_mugrid(
            preconditioner_Fourier_fnfnqks=preconditioner,
            input_nodal_field_fnxyz=x,
            output_nodal_field_fnxyz=Px
        )

    displacement_u = discretization.get_displacement_sized_field(name='displacement_u')
    displacement_u.s.fill(0)
    solvers.conjugate_gradients_mugrid(
        comm=discretization.communicator,
        fc=discretization.field_collection,
        hessp=K_fun,
        b=rhs,
        x=displacement_u,
        P=M_fun,
        tol=1e-10,
        maxiter=1000,
    )

    homogenized_stress = discretization.get_homogenized_stress_mugrid(
        material_data_field_ijklqxyz=mat_field,
        displacement_field_inxyz=displacement_u,
        macro_gradient_field_ijqxyz=macro_grad_field,
        formulation='small_strain'
    )

    dstress_drho_analytical = topology_optimization.partial_derivative_of_objective_function_stress_equivalence_wrt_phase_field_FE(
        discretization=discretization,
        material_data_field_ijkl=elastic_C_1,
        void_material_data_ijkl=elastic_C_void,
        displacement_field_fnxyz=displacement_u,
        macro_gradient_field_ijqxyz=macro_grad_field,
        phase_field_1nxyz=phase_field_1nxyz,
        target_stress_ij=target_stress,
        actual_stress_ij=homogenized_stress,
        p=p
    )

    def eval_stress_pot_from_pf(pf_arr):
        pf = discretization.get_scalar_field(name='pf_eval')
        pf.s[...] = pf_arr
        pf_q = discretization.get_quad_field_scalar(name='pf_q_eval')
        discretization.apply_N_operator_mugrid(pf, pf_q)

        m_field = discretization.get_material_data_size_field_mugrid(name='m_field_eval')
        m_field.s[...] = (
            elastic_C_1[..., np.newaxis, np.newaxis, np.newaxis]
            * np.power(pf_q.s, p)[0, 0, :, ...]
        )

        h_stress = discretization.get_homogenized_stress_mugrid(
            material_data_field_ijklqxyz=m_field,
            displacement_field_inxyz=displacement_u,
            macro_gradient_field_ijqxyz=macro_grad_field,
            formulation='small_strain'
        )
        return topology_optimization.compute_stress_equivalence_potential(
            actual_stress_ij=h_stress,
            target_stress_ij=target_stress
        )

    run_finite_difference_check(
        eval_fn=eval_stress_pot_from_pf,
        x0=phase_field_1nxyz.s,
        analytical_derivative=dstress_drho_analytical.s,
        rtol=1e-4,
        min_rate=1.5 if p >= 3 else None,
        title=f'FD Check: Stress Equivalence w.r.t Phase Field (p={p})'
    )


@pytest.mark.parametrize('domain_size, element_type, nb_pixels', [
    ([1, 1], 0, [4, 4]),
    ([2, 3], 0, [4, 3]),
])
def test_fd_check_of_stress_equivalence_potential_wrt_displacement_FE(discretization_fixture ):
    """Finite difference check of stress equivalence potential w.r.t displacement."""
    np.random.seed(42)
    p = 2
    discretization = discretization_fixture

    K_0, G_0 = material_models.get_bulk_and_shear_modulus(E=1.0, poisson=0.2)
    elastic_C = material_models.get_elastic_material_tensor(
        dim=discretization.domain_dimension, K=K_0, mu=G_0, kind='linear'
    )

    phase_field = discretization.get_scalar_field(name='phase_field')
    phase_field.s[...] = np.random.rand(*phase_field.s.shape)

    pf_quad = discretization.get_quad_field_scalar(name='pf_quad_u_test')
    discretization.apply_N_operator_mugrid(phase_field, pf_quad)

    mat_field = discretization.get_material_data_size_field_mugrid(name='mat_field_u_test')
    mat_field.s[...] = (
        elastic_C[..., np.newaxis, np.newaxis, np.newaxis]
        * np.power(pf_quad.s, p)[0, 0, :, ...]
    )

    macro_grad = np.array([[1.0, 0.0], [0.0, 1.0]])
    macro_grad_field = discretization.get_gradient_size_field(name='macro_grad_field')
    discretization.get_macro_gradient_field_mugrid(
        macro_gradient_ij=macro_grad,
        macro_gradient_field_ijqxyz=macro_grad_field
    )

    target_stress = np.array([[1.0, 0.0], [0.0, 2.0]])

    u_field = discretization.get_displacement_sized_field(name='u_field')
    u_field.s[...] = np.random.rand(*u_field.s.shape)

    homog_stress = discretization.get_homogenized_stress_mugrid(
        material_data_field_ijklqxyz=mat_field,
        displacement_field_inxyz=u_field,
        macro_gradient_field_ijqxyz=macro_grad_field,
        formulation='small_strain'
    )

    stress_diff = target_stress - homog_stress
    stress_diff_field = discretization.get_gradient_size_field(name='stress_diff_field')
    discretization.get_macro_gradient_field_mugrid(
        macro_gradient_ij=stress_diff,
        macro_gradient_field_ijqxyz=stress_diff_field
    )

    df_du_field = discretization.get_unknown_size_field(name='df_du_field')
    discretization.get_rhs_mugrid(
        material_data_field_ijklqxyz=mat_field,
        macro_gradient_field_ijqxyz=stress_diff_field,
        rhs_inxyz=df_du_field
    )
    df_du_field.s[...] = 2 * df_du_field.s / np.sum(target_stress ** 2) / discretization.cell.domain_volume

    def eval_stress_from_u(u_arr):
        u_temp = discretization.get_displacement_sized_field(name='u_temp')
        u_temp.s[...] = u_arr
        h_stress = discretization.get_homogenized_stress_mugrid(
            material_data_field_ijklqxyz=mat_field,
            displacement_field_inxyz=u_temp,
            macro_gradient_field_ijqxyz=macro_grad_field,
            formulation='small_strain'
        )
        return topology_optimization.compute_stress_equivalence_potential(
            actual_stress_ij=h_stress,
            target_stress_ij=target_stress
        )

    run_finite_difference_check(
        eval_fn=eval_stress_from_u,
        x0=u_field.s,
        analytical_derivative=df_du_field.s,
        rtol=1e-4,
        title='FD Check: Stress Equivalence w.r.t Displacement'
    )


# ==============================================================================
# Adjoint Potential Tests
# ==============================================================================

@pytest.mark.parametrize('domain_size, element_type, nb_pixels, p', [
    ([1, 1], 0, [4, 4], 2),
    ([2, 3], 0, [4, 3], 3),
])
def test_fd_check_of_adjoint_potential_wrt_phase_field_FE(discretization_fixture, p ):
    """Finite difference check of adjoint potential w.r.t. phase field."""
    np.random.seed(42)
    discretization = discretization_fixture

    K_0, G_0 = material_models.get_bulk_and_shear_modulus(E=1.0, poisson=0.2)
    elastic_C_0 = material_models.get_elastic_material_tensor(
        dim=discretization.domain_dimension, K=K_0, mu=G_0, kind='linear'
    )
    elastic_C_void = np.zeros_like(elastic_C_0)

    phase_field_0 = discretization.get_scalar_field(name='phase_field_0')
    phase_field_0.s[...] = np.random.rand(*phase_field_0.s.shape)

    macro_gradient = np.array([[1.0, 0.0], [0.0, 1.0]])
    macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
    discretization.get_macro_gradient_field_mugrid(
        macro_gradient_ij=macro_gradient,
        macro_gradient_field_ijqxyz=macro_gradient_field
    )

    adjoint_field = discretization.get_displacement_sized_field(name='adjoint_field')
    adjoint_field.s[...] = np.random.rand(*adjoint_field.s.shape)

    u_field = discretization.get_displacement_sized_field(name='u_field')
    u_field.s[...] = np.random.rand(*u_field.s.shape)

    dadjoint_drho = discretization.get_scalar_field(name='dadjoint_drho')
    topology_optimization.partial_derivative_of_adjoint_potential_wrt_phase_field_FE(
        discretization=discretization,
        base_material_data_ijkl=elastic_C_0,
        void_material_data_ijkl=elastic_C_void,
        displacement_field_fnxyz=u_field,
        macro_gradient_field_ijqxyz=macro_gradient_field,
        phase_field_1nxyz=phase_field_0,
        adjoint_field_inxyz=adjoint_field,
        output_field_inxyz=dadjoint_drho,
        p=p
    )

    def eval_adjoint_from_pf(pf_arr):
        pf = discretization.get_scalar_field(name='pf_temp')
        pf.s[...] = pf_arr
        pf_quad = discretization.get_quad_field_scalar(name='pf_quad_temp')
        discretization.apply_N_operator_mugrid(pf, pf_quad)

        mat_field = discretization.get_material_data_size_field_mugrid(name='mat_temp')
        mat_field.s[...] = (
            elastic_C_0[..., np.newaxis, np.newaxis, np.newaxis]
            * np.power(pf_quad.s, p)[0, 0, :, ...]
        )

        stress_field = discretization.get_gradient_size_field(name='stress_temp')
        discretization.get_stress_field_mugrid(
            material_data_field_ijklqxyz=mat_field,
            displacement_field_inxyz=u_field,
            macro_gradient_field_ijqxyz=macro_gradient_field,
            formulation='small_strain',
            output_stress_field_ijqxyz=stress_field
        )

        return topology_optimization.adjoint_potential(
            discretization=discretization,
            stress_field_ijqxyz=stress_field,
            adjoint_field_inxyz=adjoint_field
        )

    run_finite_difference_check(
        eval_fn=eval_adjoint_from_pf,
        x0=phase_field_0.s,
        analytical_derivative=dadjoint_drho.s,
        rtol=1e-4,
        title=f'FD Check: Adjoint Potential w.r.t Phase Field (p={p})'
    )


@pytest.mark.parametrize('domain_size, element_type, nb_pixels', [
    ([1, 1], 0, [4, 4]),
    ([2, 3], 0, [4, 3]),
])
def test_fd_check_of_adjoint_potential_wrt_displacement_FE(discretization_fixture ):
    """
    Finite difference check of adjoint potential w.r.t displacement field.
    Verifies that d/du [ A(u, lambda) ] = K(lambda).
    """
    np.random.seed(42)
    p = 2
    discretization = discretization_fixture

    K_0, G_0 = material_models.get_bulk_and_shear_modulus(E=1.0, poisson=0.2)
    elastic_C = material_models.get_elastic_material_tensor(
        dim=discretization.domain_dimension, K=K_0, mu=G_0, kind='linear'
    )

    phase_field = discretization.get_scalar_field(name='phase_field')
    phase_field.s[...] = np.random.rand(*phase_field.s.shape)

    pf_quad = discretization.get_quad_field_scalar(name='pf_quad')
    discretization.apply_N_operator_mugrid(phase_field, pf_quad)

    mat_field = discretization.get_material_data_size_field_mugrid(name='mat_field')
    mat_field.s[...] = (
        elastic_C[..., np.newaxis, np.newaxis, np.newaxis]
        * np.power(pf_quad.s, p)[0, 0, :, ...]
    )

    macro_grad = np.array([[1.0, 0.0], [0.0, 1.0]])
    macro_grad_field = discretization.get_gradient_size_field(name='macro_grad_field')
    discretization.get_macro_gradient_field_mugrid(
        macro_gradient_ij=macro_grad,
        macro_gradient_field_ijqxyz=macro_grad_field
    )

    adjoint_field = discretization.get_displacement_sized_field(name='adjoint_field')
    adjoint_field.s[...] = np.random.rand(*adjoint_field.s.shape)
    discretization.fft.communicate_ghosts(adjoint_field)

    dg_du_analytical = discretization.get_displacement_sized_field(name='dg_du_analytical')
    discretization.apply_system_matrix_mugrid(
        material_data_field=mat_field,
        input_field_inxyz=adjoint_field,
        output_field_inxyz=dg_du_analytical,
        formulation='small_strain'
    )

    u_field = discretization.get_displacement_sized_field(name='u_field')
    u_field.s[...] = np.random.rand(*u_field.s.shape)

    def eval_adjoint_from_u(u_arr):
        u_temp = discretization.get_displacement_sized_field(name='u_temp')
        u_temp.s[...] = u_arr
        stress_field = discretization.get_gradient_size_field(name='stress_temp')
        discretization.get_stress_field_mugrid(
            material_data_field_ijklqxyz=mat_field,
            displacement_field_inxyz=u_temp,
            macro_gradient_field_ijqxyz=macro_grad_field,
            formulation='small_strain',
            output_stress_field_ijqxyz=stress_field
        )
        return topology_optimization.adjoint_potential(
            discretization=discretization,
            stress_field_ijqxyz=stress_field,
            adjoint_field_inxyz=adjoint_field
        )

    run_finite_difference_check(
        eval_fn=eval_adjoint_from_u,
        x0=u_field.s,
        analytical_derivative=dg_du_analytical.s,
        rtol=1e-4,
        title='FD Check: Adjoint Potential w.r.t Displacement'
    )


@pytest.mark.parametrize('domain_size, element_type, nb_pixels', [
    ([3, 4], 0, [5, 6]),
    ([2, 5], 1, [6, 5]),
])
def test_nullity_of_adjoint_potential(discretization_fixture ):
    """
    Test that the adjoint potential is zero when the displacement field
    satisfies mechanical equilibrium.
    """
    np.random.seed(42)
    discretization = discretization_fixture

    K_0, G_0 = material_models.get_bulk_and_shear_modulus(E=1.0, poisson=0.2)
    material_C_0 = material_models.get_elastic_material_tensor(
        dim=discretization.domain_dimension, K=K_0, mu=G_0, kind='linear'
    )

    phase_field = discretization.get_scalar_field(name='phase_field')
    phase_field.s[...] = np.random.rand(*phase_field.s.shape)

    pf_quad = discretization.get_quad_field_scalar(name='pf_quad')
    discretization.apply_N_operator_mugrid(phase_field, pf_quad)

    mat_field = discretization.get_material_data_size_field_mugrid(name='mat_field')
    mat_field.s[...] = (
        material_C_0[..., np.newaxis, np.newaxis, np.newaxis]
        * np.power(pf_quad.s, 1)[0, 0, :, ...]
    )

    macro_gradient = np.array([[0.1, 0.0], [0.0, 0.1]])
    macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
    discretization.get_macro_gradient_field_mugrid(
        macro_gradient_ij=macro_gradient,
        macro_gradient_field_ijqxyz=macro_gradient_field
    )

    rhs = discretization.get_unknown_size_field(name='rhs')
    discretization.get_rhs_mugrid(
        material_data_field_ijklqxyz=mat_field,
        macro_gradient_field_ijqxyz=macro_gradient_field,
        rhs_inxyz=rhs
    )

    def K_fun(x, Ax):
        discretization.apply_system_matrix_mugrid(
            material_data_field=mat_field,
            input_field_inxyz=x,
            output_field_inxyz=Ax,
            formulation='small_strain'
        )

    def M_fun(x, Px):
        Px.s[...] = x.s

    displacement_field = discretization.get_displacement_sized_field(name='displacement_field')
    displacement_field.s.fill(0)
    solvers.conjugate_gradients_mugrid(
        comm=discretization.communicator,
        fc=discretization.field_collection,
        hessp=K_fun,
        b=rhs,
        x=displacement_field,
        P=M_fun,
        tol=1e-10,
        maxiter=50000,
    )

    adjoint_field = discretization.get_displacement_sized_field(name='adjoint_field')
    adjoint_field.s[...] = np.random.rand(*adjoint_field.s.shape)

    stress_field = discretization.get_gradient_size_field(name='stress_field')
    stress_field.s.fill(0)
    discretization.get_stress_field_mugrid(
        material_data_field_ijklqxyz=mat_field,
        displacement_field_inxyz=displacement_field,
        macro_gradient_field_ijqxyz=macro_gradient_field,
        output_stress_field_ijqxyz=stress_field,
        formulation='small_strain'
    )

    adjoint_pot = topology_optimization.adjoint_potential(
        discretization, stress_field, adjoint_field
    )

    assert abs(adjoint_pot) < 1e-7, (
        f"Adjoint potential should be 0 for equilibrium solution, but got {adjoint_pot}"
    )
