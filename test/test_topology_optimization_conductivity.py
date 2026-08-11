import pytest

import numpy as np
import scipy as sc
import time

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import topology_optimization_conductivity as topology_optimization
from muFFTTO import material_models


@pytest.fixture()
def discretization_fixture(domain_size, element_type, nb_pixels):
    problem_type = 'conductivity'
    element_types = ['linear_triangles', 'linear_triangles_tilled'
        , 'trilinear_hexahedron', 'trilinear_hexahedron_1Q']

    my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                      problem_type=problem_type)

    discretization_type = 'finite_element'

    discretization = domain.Discretization(cell=my_cell,
                                           nb_of_pixels_global=nb_pixels,
                                           discretization_type=discretization_type,
                                           element_type=element_types[element_type])

    return discretization


@pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
    ([2, 2], 0, [2, 2]),
    ([2, 3], 0, [2, 3]),
    ([2, 4], 0, [2, 4]),
    ([3, 2], 0, [3, 2]),
    ([3, 3], 0, [3, 3]),
    ([3, 4], 0, [3, 4]),
    ([4, 2], 0, [4, 2]),
    ([4, 3], 0, [4, 3]),
    ([4, 4], 0, [4, 4]),
    ([2, 2], 1, [2, 2]),
    ([2, 3], 1, [2, 3]),
    ([2, 4], 1, [2, 4]),
    ([3, 2], 1, [3, 2]),
    ([3, 3], 1, [3, 3]),
    ([3, 4], 1, [3, 4]),
    ([4, 2], 1, [4, 2]),
    ([4, 3], 1, [4, 3]),
    ([4, 4], 1, [4, 4])])
def test_discretization_init(discretization_fixture):
    print(discretization_fixture.domain_size)
    assert hasattr(discretization_fixture, "cell")
    assert hasattr(discretization_fixture, "domain_dimension")
    assert hasattr(discretization_fixture, "B_gradient")
    assert hasattr(discretization_fixture, "quadrature_weights")
    assert hasattr(discretization_fixture, "nb_quad_points_per_pixel")
    assert hasattr(discretization_fixture, "nb_nodes_per_pixel")


@pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
    ([1, 2], 0, [4, 5]),
    ([1, 2], 1, [4, 5]),
    ([3.1, 6.4], 0, [7, 6])])
def test_fd_check_of_whole_objective_function_2D_conductivity(discretization_fixture, plot=True):
    """
    Finite difference check of the whole objective function gradient
    with respect to the phase field.
    """
    preconditioner_type = 'Green_Jacobi'

    discretization = discretization_fixture

    macro_gradient = np.array([1.0, .0])
    print('macro_gradient = \n {}'.format(macro_gradient))

    # create material data of solid phase rho=1
    # Base Material Properties
    conductivity_C_0 = np.array([[1., 0], [0, 1.0]])
    #
    soft_phase = 0
    conductivity_C_void = conductivity_C_0 * soft_phase


    # create target material data
    conductivity_C_target = np.array([[0.2, 0], [0, 1.0]])

    target_flux_ij = np.einsum('ij,j->i', conductivity_C_target, macro_gradient)
    print('target_flux = \n {}'.format(target_flux_ij))
    # Set up the equilibrium system

    macro_gradient_field_ijqxyz = discretization.get_gradient_size_field(name='macro_gradient_field')
    discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient,
                                                   macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz
                                                   )

    preconditioner_Green = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=conductivity_C_0)

    def M_fun(x, Px):
        """
        Function to compute the product of the Preconditioner matrix with a vector.
        The Preconditioner is represented by the convolution operator.
        """
        discretization.fft.communicate_ghosts(x)
        discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner_Green,
                                                   input_nodal_field_fnxyz=x,
                                                   output_nodal_field_fnxyz=Px)

    p = 2
    w = 3
    eta = .3
    cg_setup = {'cg_tol': 1e-9}

    def my_objective_function(phase_field_1nxyz_flat):
        # reshape the field
        phase_field_1nxyz = discretization.get_scalar_field(name='phase_field_in_objective')
        phase_field_1nxyz.s[...] = phase_field_1nxyz_flat.reshape([1, 1, *discretization.nb_of_pixels])

        # Phase field  in quadrature points
        phase_field_at_quad_poits_1qxyz = discretization.get_quad_field_scalar(
            name='phase_field_at_quads_in_objective_function_multiple_load_cases')
        discretization.apply_N_operator_mugrid(phase_field_1nxyz, phase_field_at_quad_poits_1qxyz)

        # Material data in quadrature points
        material_data_field_C_0_rho_ijklqxyz = discretization.get_material_data_size_field_mugrid(
            name='material_data_field_C_0_rho_ijklqxyz_in_objective')
        material_data_field_C_0_rho_ijklqxyz.s[...] = conductivity_C_0[..., np.newaxis, np.newaxis, np.newaxis] * \
                                                      np.power(phase_field_at_quad_poits_1qxyz.s, p)[0, 0, :, ...]

        f_phase_field = topology_optimization.objective_function_phase_field(discretization=discretization,
                                                                             phase_field_1nxyz=phase_field_1nxyz,
                                                                             eta=eta,
                                                                             double_well_depth=1)
        #  sensitivity phase field terms
        s_phase_field = discretization.get_scalar_field(name='s_phase_field')
        s_phase_field.s.fill(0)

        topology_optimization.sensitivity_phase_field_term_FE_NEW(discretization=discretization,
                                                                  phase_field_1nxyz=phase_field_1nxyz,
                                                                  p=p,
                                                                  eta=eta,
                                                                  output_array=s_phase_field,
                                                                  double_well_depth=1)

        if preconditioner_type == 'Green_Jacobi':
            K_diag_alg = discretization.get_preconditioner_Jacobi_mugrid(
                material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz)

            def M_fun_Green_Jacobi(x, Px):
                discretization.fft.communicate_ghosts(x)
                x_jacobi_temp = discretization.get_unknown_size_field(name='x_jacobi_temp')

                x_jacobi_temp.s[...] = K_diag_alg.s * x.s
                discretization.apply_preconditioner_mugrid(
                    preconditioner_Fourier_fnfnqks=preconditioner_Green,
                    input_nodal_field_fnxyz=x_jacobi_temp,
                    output_nodal_field_fnxyz=Px)

                Px.s[...] = K_diag_alg.s * Px.s
                discretization.fft.communicate_ghosts(Px)

            M_fun = M_fun_Green_Jacobi

        # Solve mechanical equilibrium constrain
        def K_fun(x, Ax):
            discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0_rho_ijklqxyz,
                                                      input_field_inxyz=x,
                                                      output_field_inxyz=Ax)

        # mechanical equilibrium rhs
        rhs_inxyz = discretization.get_unknown_size_field(name='rhs_field')
        discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                                      macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
                                      rhs_inxyz=rhs_inxyz)

        temperature_field = discretization.get_unknown_size_field(name='temperature_field_')
        temperature_field.s.fill(0)

        solvers.conjugate_gradients_mugrid(
            comm=discretization.communicator,
            fc=discretization.field_collection,
            hessp=K_fun,  # linear operator
            b=rhs_inxyz,
            x=temperature_field,
            P=M_fun,
            tol=cg_setup['cg_tol'],
            maxiter=10000,
        )
        # compute homogenized stress field corresponding t
        homogenized_flux = discretization.get_homogenized_stress(
            material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
            displacement_field_inxyz=temperature_field,
            macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz )

        f_sigma = topology_optimization.compute_flux_equivalence_potential(
            actual_flux_ij=homogenized_flux,
            target_flux_ij=target_flux_ij)

        adjoint_field = discretization.get_unknown_size_field(name='adjoint_field')
        adjoint_field.s.fill(0)

        sensitivity_analytical = discretization.get_scalar_field(
            name='sensitivity_analytical')
        sensitivity_analytical.s.fill(0)

        sensitivity_analytical.s[
            0, 0], adjoint_field, adjoint_energies, info_adjoint_current = topology_optimization.sensitivity_flux_and_adjoint(
            discretization=discretization,
            base_material_data_ijkl=conductivity_C_0,
            void_material_data_ijkl=conductivity_C_void,
            displacement_field_inxyz=temperature_field,
            adjoint_field_inxyz=adjoint_field,
            macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
            phase_field_1nxyz=phase_field_1nxyz,
            target_flux_ij=target_flux_ij,
            actual_flux_ij=homogenized_flux,
            preconditioner_fun=M_fun,
            system_matrix_fun=K_fun,
            p=p,
            weight=w,
            disp=True,
            **cg_setup)

        sensitivity_analytical.s[...] += s_phase_field.s

        objective_function = w * f_sigma + f_phase_field
        #objective_function += adjoint_energies
        print(f'ob')
        return objective_function, f_sigma, f_phase_field, sensitivity_analytical,

    np.random.seed(1)
    phase_field = discretization.get_scalar_field(name='phase_field_0')
    phase_field.s[...] = np.random.rand(*phase_field.s.shape) ** 1
    # Save a copy of the original phase field
    phase_field_0_fixed = discretization.get_scalar_field(name='phase_field_0_fixed')
    phase_field_0_fixed.s[...] = np.copy(phase_field.s)

    _, _, _, analytical_sensitivity = my_objective_function(phase_field.s.ravel())

    # Phase field lives in [0,1] — large epsilon drives it outside the linearization regime.
    # O(h^2) convergence is visible only for epsilon << 1.
    epsilons = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]

    fd_sensitivity = discretization_fixture.get_scalar_field(name='fd_sensitivity')
    fd_sensitivity_drho_dro = discretization_fixture.get_scalar_field(name='fd_sensitivity_drho_dro')
    fd_sensitivity_dsigma_dro = discretization_fixture.get_scalar_field(name='fd_sensitivity_dsigma_dro')

    error_fd_vs_analytical = []
    error_fd_vs_analytical_max = []
    norm_fd_sensitivity_dsigma_dro = []
    norm_fd_sensitivity_df_dro = []
    norm_fd_sensitivity = []
    fd_scheme = 2.
    for epsilon in epsilons:
        # loop over every single element of phase field
        for x in np.arange(discretization_fixture.nb_of_pixels[0]):
            for y in np.arange(discretization_fixture.nb_of_pixels[1]):
                # set phase_field to ones
                phase_field.s[...] = np.copy(phase_field_0_fixed.s)
                #
                phase_field.s[0, 0, x, y] = phase_field.s[0, 0, x, y] + epsilon / fd_scheme

                of_plus_eps, f_sigma_plus_eps, f_rho_plus_eps, _ = my_objective_function(phase_field.s.ravel())

                phase_field.s[0, 0, x, y] = phase_field.s[0, 0, x, y] - epsilon
                # phase_field_0 = phase_field.reshape(-1)

                of_minu_eps, f_sigma_minu_eps, f_rho_minu_eps, _ = my_objective_function(phase_field.s.ravel())

                fd_sensitivity.s[0, 0, x, y] = (of_plus_eps - of_minu_eps) / (epsilon)
                fd_sensitivity_drho_dro.s[0, 0, x, y] = (f_rho_plus_eps - f_rho_minu_eps) / (epsilon)
                fd_sensitivity_dsigma_dro.s[0, 0, x, y] = (f_sigma_plus_eps - f_sigma_minu_eps) / (epsilon)

        error_fd_vs_analytical.append(
            np.linalg.norm((fd_sensitivity.s - analytical_sensitivity.s)[0, 0], 'fro'))
        error_fd_vs_analytical_max.append(
            np.max((fd_sensitivity.s - analytical_sensitivity.s)[0, 0]))
        norm_fd_sensitivity.append(
            np.linalg.norm(fd_sensitivity.s[0, 0], 'fro'))
        norm_fd_sensitivity_df_dro.append(
            np.linalg.norm(fd_sensitivity_drho_dro.s[0, 0], 'fro'))
        norm_fd_sensitivity_dsigma_dro.append(
            np.linalg.norm(fd_sensitivity_dsigma_dro.s[0, 0], 'fro'))
    print()
    print(error_fd_vs_analytical)
    print(norm_fd_sensitivity)
    print(norm_fd_sensitivity_df_dro)
    print(norm_fd_sensitivity_dsigma_dro)
    print(error_fd_vs_analytical)
    errors = np.array(error_fd_vs_analytical)
    epsilons_arr = np.array(epsilons)
    analytical_norm = np.linalg.norm(analytical_sensitivity.s[0, 0], 'fro')
    relative_errors = errors / analytical_norm

    if plot:
        import matplotlib.pyplot as plt
        idx_min = np.argmax(errors)
        plt.figure()
        plt.loglog(epsilons, errors, marker='x', label='FD vs analytical error')
        plt.loglog(epsilons, errors[idx_min] * (epsilons_arr / epsilons_arr[idx_min]) ** 2,
                   linestyle='--', label=r'$O(h^2)$ reference')
        plt.loglog(epsilons, errors[idx_min] * (epsilons_arr / epsilons_arr[idx_min]) ** 1,
                   linestyle='--', label=r'$O(h)$ reference')
        plt.legend(loc='best')
        plt.xlabel('epsilon (FD step size)')
        plt.ylabel('Error (Frobenius norm)')
        plt.title('FD check: whole objective function')
        plt.show()

    # Always check: minimum relative error should be small
    assert np.min(relative_errors) < 1e-4, (
        f"FD check failed: minimum relative error {np.min(relative_errors):.2e} exceeds 1e-4. "
        f"Analytical derivative may be wrong.")

    # Convergence rate check
    log_eps = np.log10(epsilons_arr[:3])
    log_err = np.log10(errors[:3])
    convergence_rate = np.polyfit(log_eps, log_err, 1)[0]
    assert convergence_rate > 0.95, (
        f"FD convergence rate {convergence_rate:.2f} too low "
        f"(expected ~2 for central differences). Analytical derivative may be wrong.")
