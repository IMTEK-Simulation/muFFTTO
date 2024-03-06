import pytest

import numpy as np
import scipy as sc

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import topology_optimization
from muFFTTO import microstructure_library


@pytest.fixture()
def discretization_fixture(domain_size, element_type, nb_pixels):
    problem_type = 'elasticity'
    element_types = ['linear_triangles', 'bilinear_rectangle']

    my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                      problem_type=problem_type)

    discretization_type = 'finite_element'

    discretization = domain.Discretization(cell=my_cell,
                                           number_of_pixels=nb_pixels,
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
    ([3, 4], 0, [15, 15])])
def test_finite_difference_check_of_whole_objective_function(discretization_fixture):
    problem_type = 'elasticity'
    discretization_type = 'finite_element'
    element_type = discretization_fixture.element_type  # 'bilinear_rectangle'##'linear_triangles' #
    formulation = 'small_strain'
    domain_size = [1, 1]
    number_of_pixels = discretization_fixture.nb_of_pixels

    my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                      problem_type=problem_type)

    discretization = domain.Discretization(cell=my_cell,
                                           number_of_pixels=number_of_pixels,
                                           discretization_type=discretization_type,
                                           element_type=element_type)

    macro_gradient = np.array([[0.1, 0.0],
                               [0.0, 0.0]])
    print('macro_gradient = \n {}'.format(macro_gradient))

    # Set up the equilibrium system
    macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)

    # create material data of solid phase rho=1
    E_0 = 100
    poison_0 = 0.

    K_0, G_0 = domain.get_bulk_and_shear_modulus(E=E_0, poison=poison_0)

    elastic_C_0 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                     K=K_0,
                                                     mu=G_0,
                                                     kind='linear')

    material_data_field_C_0 = np.einsum('ijkl,qxy->ijklqxy', elastic_C_0,
                                        np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                          *discretization.nb_of_pixels])))

    stress = np.einsum('ijkl,lk->ij', elastic_C_0, macro_gradient)

    # create target material data
    print('init_stress = \n {}'.format(stress))
    # validation metamaterials

    poison_target = 1 / 3  # lambda = -10
    G_target_auxet = (1 / 4) * E_0  # 23   25
    E_target = 2 * G_target_auxet * (1 + poison_target)

    K_targer, G_target = domain.get_bulk_and_shear_modulus(E=E_target, poison=poison_target)

    elastic_C_target = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                          K=K_targer,
                                                          mu=G_target,
                                                          kind='linear')
    # target_stress = np.array([[0.0, 0.05],
    #                           [0.05, 0.0]])
    target_stress = np.einsum('ijkl,lk->ij', elastic_C_target, macro_gradient)
    print('target_stress = \n {}'.format(target_stress))

    p = 1
    w = 1  # * E_0  # 1 / 10  # 1e-4 Young modulus of solid
    eta = 1

    def my_objective_function(phase_field_1nxyz):
        # print('Objective function:')
        # reshape the field
        phase_field_1nxyz = phase_field_1nxyz.reshape([1, 1, *number_of_pixels])

        material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * np.power(phase_field_1nxyz,
                                                                                    p)

        # Solve mechanical equilibrium constrain
        rhs = discretization.get_rhs(material_data_field_C_0_rho, macro_gradient_field)

        K_fun = lambda x: discretization.apply_system_matrix(material_data_field_C_0_rho,
                                                             x,
                                                             formulation='small_strain')
        preconditioner = discretization.get_preconditioner(
            reference_material_data_field_ijklqxyz=material_data_field_C_0)
        M_fun = lambda x: discretization.apply_preconditioner(preconditioner_Fourier_fnfnxyz=preconditioner,
                                                              nodal_field_fnxyz=x)
        displacement_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(500), toler=1e-8)

        # compute homogenized stress field corresponding t
        homogenized_stress = discretization.get_homogenized_stress(
            material_data_field_ijklqxyz=material_data_field_C_0_rho,
            displacement_field_fnxyz=displacement_field,
            macro_gradient_field_ijqxyz=macro_gradient_field,
            formulation='small_strain')
        # print('homogenized stress = \n'          ' {} '.format(homogenized_stress))

        objective_function = topology_optimization.objective_function_small_strain(
            discretization=discretization,
            actual_stress_ij=homogenized_stress,
            target_stress_ij=target_stress,
            phase_field_1nxyz=phase_field_1nxyz,
            eta=eta,
            w=w)
        # print('objective_function= \n'' {} '.format(objective_function))

        return objective_function

    def my_sensitivity(phase_field_1nxyz):
        # print('Sensitivity calculation')

        phase_field_1nxyz = phase_field_1nxyz.reshape([1, 1, *number_of_pixels])

        # Compute homogenized stress field for current phase field
        material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * np.power(phase_field_1nxyz,
                                                                                    p)
        # Solve mechanical equilibrium constrain for hom
        rhs = discretization.get_rhs(material_data_field_C_0_rho, macro_gradient_field)

        K_fun = lambda x: discretization.apply_system_matrix(material_data_field_C_0_rho, x,
                                                             formulation='small_strain')
        # M_fun = lambda x: 1 * x
        preconditioner = discretization.get_preconditioner(
            reference_material_data_field_ijklqxyz=material_data_field_C_0)
        M_fun = lambda x: discretization.apply_preconditioner(preconditioner_Fourier_fnfnxyz=preconditioner,
                                                              nodal_field_fnxyz=x)

        displacement_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(500), toler=1e-8)

        # compute homogenized stress field corresponding t
        homogenized_stress = discretization.get_homogenized_stress(
            material_data_field_ijklqxyz=material_data_field_C_0_rho,
            displacement_field_fnxyz=displacement_field,
            macro_gradient_field_ijqxyz=macro_gradient_field,
            formulation='small_strain')

        # print('Sensitivity_analytical')
        sensitivity_analytical = topology_optimization.sensitivity_with_adjoint_problem(
            discretization=discretization,
            material_data_field_ijklqxyz=material_data_field_C_0,
            displacement_field_fnxyz=displacement_field,
            macro_gradient_field_ijqxyz=macro_gradient_field,
            phase_field_1nxyz=phase_field_1nxyz,
            target_stress_ij=target_stress,
            actual_stress_ij=homogenized_stress,
            formulation='small_strain',
            p=p,
            eta=eta,
            weight=w)

        return sensitivity_analytical.reshape(-1)

    phase_field_0 = np.random.rand(*discretization.get_scalar_sized_field().shape) ** 1
    phase_field_00 = np.copy(phase_field_0)

    phase_field_0 = phase_field_0.reshape(-1)
    analitical_sensitivity = my_sensitivity(phase_field_0)
    analitical_sensitivity = analitical_sensitivity.reshape([1, 1, *number_of_pixels])

    epsilons = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    fd_sensitivity = discretization_fixture.get_scalar_sized_field()
    error_fd_vs_analytical = []

    for epsilon in epsilons:
        # loop over every single element of phase field
        for x in np.arange(discretization_fixture.nb_of_pixels[0]):
            for y in np.arange(discretization_fixture.nb_of_pixels[1]):
                # set phase_field to ones
                phase_field = np.copy(phase_field_00)
                #
                phase_field[0, 0, x, y] = phase_field[0, 0, x, y] + epsilon / 2
                phase_field_0 = phase_field.reshape(-1)
                of_plus_eps = my_objective_function(phase_field_0)

                phase_field[0, 0, x, y] = phase_field[0, 0, x, y] - epsilon
                phase_field_0 = phase_field.reshape(-1)

                of_minu_eps = my_objective_function(phase_field_0)

                fd_sensitivity[0, 0, x, y] = (of_plus_eps - of_minu_eps) / epsilon

        error_fd_vs_analytical.append(
            np.linalg.norm((fd_sensitivity - analitical_sensitivity)[0, 0], 'fro'))

    print(error_fd_vs_analytical)
# @pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
#     ([3, 4], 0, [5, 8]),
#     ([4, 5], 1, [7, 6])])
# def test_adjoint_sensitivity_(discretization_fixture):

@pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
    ([2, 2], 0, [10, 10]),
    ([2, 2], 0, [2, 2])])
def test_finite_difference_check_of_double_well_potential(discretization_fixture):
    epsilons = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    # epsilons = [1e-4]
    fd_derivative = discretization_fixture.get_scalar_sized_field()
    fd_derivative_NEW = discretization_fixture.get_scalar_sized_field()
    fd_derivative_Gauss_quad = discretization_fixture.get_scalar_sized_field()
    # compute double-well potential without perturbations
    phase_field = discretization_fixture.get_scalar_sized_field() + 1  # Phase field has  one  value per pixel
    phase_field = np.random.rand(*discretization_fixture.get_scalar_sized_field().shape) ** 0

    linfunc = lambda x: 1 * x
    phase_field[0, 0] = linfunc(discretization_fixture.get_nodal_points_coordinates()[0, 0])
    phase_field_0 = np.copy(phase_field)
    # phase_field[0, 0, 2:5, 2:4] = phase_field[0, 0, 2:5, 2:4] / 3  # can be random in this case

    # get analytical partial derivative of the double-well potential with respect to phase-field
    partial_der_of_double_well_potential = topology_optimization.partial_der_of_double_well_potential_wrt_density(
        discretization_fixture,
        phase_field)

    partial_der_of_double_well_potential_NEW = (
        topology_optimization.partial_der_of_double_well_potential_wrt_density_NEW(
            discretization_fixture,
            phase_field))

    error_fd_vs_analytical = []
    error_fd_vs_analytical_NEW = []
    error_fd_NEW_vs_analytical_NEW = []

    error_fd_NEW_vs_analytical_NEW = []
    error_fd_Gaus_vs_analytical_Gaus = []
    for epsilon in epsilons:
        # shifted: compute gradient in the middle point between phase_field and phase_field+epsilon
        # phase_field_shift = np.copy(phase_field_0) + epsilon
        # # double_well_potential for a phase field without perturbation
        # double_well_potential_plus_eps = topology_optimization.compute_double_well_potential(discretization_fixture,
        #                                                                                      phase_field, eta=1)
        # double_well_potential_NEW = topology_optimization.compute_double_well_potential_NEW(discretization_fixture,
        #                                                                                     phase_field, eta=1)
        #
        #

        # loop over every single element of phase field
        for x in np.arange(discretization_fixture.nb_of_pixels[0]):
            for y in np.arange(discretization_fixture.nb_of_pixels[1]):
                # set phase_field to ones
                # phase_field = discretization_fixture.get_scalar_sized_field() + 1  # Phase field has  one  value per pixel
                phase_field = np.copy(phase_field_0)

                # phase_field[0, 0, 2:5, 2:4] = phase_field[0, 0, 2:5, 2:4] / 3  # can be random in this case
                #
                phase_field[0, 0, x, y] = phase_field[0, 0, x, y] + epsilon / 2

                double_well_potential_plus_eps = topology_optimization.compute_double_well_potential(
                    discretization_fixture,
                    phase_field, eta=1)
                double_well_potential_plus_eps_NEW = topology_optimization.compute_double_well_potential_NEW(
                    discretization_fixture,
                    phase_field, eta=1)

                double_well_potential_plus_eps_Gauss_quad = topology_optimization.compute_double_well_potential_Gauss_quad(
                    discretization_fixture,
                    phase_field, eta=1)

                phase_field[0, 0, x, y] = phase_field[0, 0, x, y] - epsilon
                # assert error_fd_vs_analytical[-1] < epsilon * 1e2, (
                double_well_potential_minus_eps = topology_optimization.compute_double_well_potential(
                    discretization_fixture,
                    phase_field, eta=1)
                double_well_potential_minus_eps_NEW = topology_optimization.compute_double_well_potential_NEW(
                    discretization_fixture,
                    phase_field, eta=1)

                double_well_potential_minus_eps_Gauss_quad = topology_optimization.compute_double_well_potential_Gauss_quad(
                    discretization_fixture,
                    phase_field, eta=1)

                fd_derivative[0, 0, x, y] = (double_well_potential_plus_eps - double_well_potential_minus_eps) / epsilon

                # print(fd_derivative[0, 0])
                fd_derivative_NEW[0, 0, x, y] = (double_well_potential_plus_eps_NEW -
                                                 double_well_potential_minus_eps_NEW) / epsilon
                fd_derivative_Gauss_quad[0, 0, x, y] = (double_well_potential_plus_eps_Gauss_quad -
                                                        double_well_potential_minus_eps_Gauss_quad) / epsilon

        # print('error_fd_NEW_vs_analytical_NEW: {}'.format(fd_derivative_NEW))
        error_fd_vs_analytical.append(
            np.linalg.norm((fd_derivative - partial_der_of_double_well_potential)[0, 0], 'fro'))
        error_fd_vs_analytical_NEW.append(
            np.linalg.norm((fd_derivative - partial_der_of_double_well_potential_NEW)[0, 0], 'fro'))
        error_fd_NEW_vs_analytical_NEW.append(
            np.linalg.norm((fd_derivative_NEW - partial_der_of_double_well_potential_NEW)[0, 0], 'fro'))
        error_fd_Gaus_vs_analytical_Gaus.append(
            np.linalg.norm((fd_derivative_Gauss_quad - partial_der_of_double_well_potential_NEW)[0, 0], 'fro'))
    print('error_fd_vs_analytical: {}'.format(error_fd_vs_analytical))
    print('error_fd_vs_analytical_NEW: {}'.format(error_fd_vs_analytical_NEW))
    print('error_fd_NEW_vs_analytical_NEW: {}'.format(error_fd_NEW_vs_analytical_NEW))
    print('error_fd_Gaus_vs_analytical_Gaus: {}'.format(error_fd_Gaus_vs_analytical_Gaus))

    # assert error_fd_vs_analytical[-1] < epsilon * 1e2, (
    #   "Finite difference derivative do not corresponds to the analytical expression "
    #   "for partial derivative of double well potential ")


@pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
    ([3, 4], 0, [6, 8]),
    ([2, 5], 0, [12, 7]),
    ([3, 4], 1, [6, 8]),
    ([2, 5], 1, [12, 7])])
def test_finite_difference_check_of_gradient_of_phase_field_potential(discretization_fixture):
    epsilons = [1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    # epsilons = [1e-5]
    fd_derivative = discretization_fixture.get_scalar_sized_field()
    # Compute phase field gradient potential without perturbations
    phase_field = discretization_fixture.get_scalar_sized_field() + 1  # Phase field has  one  value per pixel
    phase_field[0, 0, 2:5, 2:4] = phase_field[0, 0, 2:5, 2:4] / 3  # can be random in this case

    # get analytical partial derivative of phase field gradient potential for a phase field with respect to phase-field
    partial_der_of_f_rho_grad_potential = topology_optimization.partial_derivative_of_gradient_of_phase_field_potential(
        discretization_fixture,
        phase_field_1nxyz=phase_field)

    error_fd_vs_analytical = []
    for epsilon in epsilons:
        # loop over every single element of phase field
        for x in np.arange(discretization_fixture.nb_of_pixels[0]):
            for y in np.arange(discretization_fixture.nb_of_pixels[1]):
                # set phase_field to ones
                phase_field = discretization_fixture.get_scalar_sized_field() + 1  # Phase field has  one  value per pixel
                phase_field[0, 0, 2:5, 2:4] = phase_field[0, 0, 2:5, 2:4] / 3  # can be random in this case
                #
                # phase_field[0, 0, :, :] = u_fun_3(nodal_coordinates[0, 0, :, :],
                #                                   nodal_coordinates[1, 0, :, :])
                phase_field[0, 0, x, y] = phase_field[0, 0, x, y] + epsilon / 2

                f_rho_grad_potential_perturbed = topology_optimization.compute_gradient_of_phase_field_potential(
                    discretization_fixture,
                    phase_field_1nxyz=phase_field,
                    eta=1)

                # phase field gradient potential for a phase field without perturbation
                phase_field[0, 0, x, y] = phase_field[0, 0, x, y] - epsilon
                f_rho_grad_potential = topology_optimization.compute_gradient_of_phase_field_potential(
                    discretization_fixture,
                    phase_field_1nxyz=phase_field,
                    eta=1)

                fd_derivative[0, 0, x, y] = (f_rho_grad_potential_perturbed - f_rho_grad_potential) / epsilon

        error_fd_vs_analytical.append(
            np.linalg.norm((fd_derivative - partial_der_of_f_rho_grad_potential)[0, 0], 'fro'))

        print(error_fd_vs_analytical)

        assert error_fd_vs_analytical[-1] < epsilon * 1e2, (
            "Finite difference derivative do not corresponds to the analytical expression "
            "for partial derivative of gradient of phase-field potential ")


@pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
    ([3, 4], 0, [6, 8]),
    ([2, 5], 0, [12, 7]),
    ([3, 4], 1, [6, 8]),
    ([2, 5], 1, [12, 7])])
def test_of_stress_equivalence_potential_quadratic(discretization_fixture, plot=False):
    # this test shows how the stress equivalence potential evolves with respect to perturbation of stress
    epsilons = np.arange(-6, 8.1, 0.5)
    # epsilons = [1e-4]

    target_stress = np.array([[1, 0.3], [0.3, 2]])

    stress_diffrence_potential = []
    for epsilon in epsilons:
        actual_stress = np.array([[1, 0.3], [0.3, 2]]) * epsilon

        f_sigma_diff_potential_perturbed = topology_optimization.objective_function_stress_equivalence(
            discretization_fixture,
            actual_stress,
            target_stress)
        stress_diffrence_potential.append(f_sigma_diff_potential_perturbed)
        # fd_derivative[0, 0, x, y] = (f_sigma_diff_potential_perturbed - f_sigma_diff_potential) / epsilon

    # error_fd_vs_analytical.append(np.linalg.norm((fd_derivative[0, 0] - f_rho_grad_potential), 'fro'))
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(epsilons, stress_diffrence_potential)
        plt.show()

    for i in np.arange(epsilons.__len__() // 2):
        assert stress_diffrence_potential[i] == pytest.approx(stress_diffrence_potential[-(i + 1)], 1e-9), (
            "stress_equivalence_potential is not symmetric for  {} ".format(i))


@pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
    ([3, 4], 0, [6, 8])
    # ,
    # ([2, 5], 0, [12, 7]),
    #  ([3, 4], 1, [6, 8]),
    # ([2, 5], 1, [12, 7])
])
def test_finite_difference_check_of_stress_equivalence_potential(discretization_fixture):
    epsilons = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    # epsilons = [1e-4]
    p = 1
    fd_derivative = discretization_fixture.get_scalar_sized_field()
    fd_derivative_wo = discretization_fixture.get_scalar_sized_field()

    target_stress = np.array([[1, 0.3], [0.3, 2]])
    macro_gradient = np.array([[0.01, 0], [0, 0.02]])

    # create material data field
    K_0, G_0 = domain.get_bulk_and_shear_modulus(E=1, poison=0.2)

    elastic_C_1 = domain.get_elastic_material_tensor(dim=discretization_fixture.domain_dimension,
                                                     K=K_0,
                                                     mu=G_0,
                                                     kind='linear')

    material_data_field_C_0 = np.einsum('ijkl,qxy->ijklqxy', elastic_C_1,
                                        np.ones(np.array([discretization_fixture.nb_quad_points_per_pixel,
                                                          *discretization_fixture.nb_of_pixels])))

    # phase_field = discretization_fixture.get_scalar_sized_field() + 1  #

    phase_field = np.random.rand(*discretization_fixture.get_scalar_sized_field().shape)  # set random distribution
    # Phase field has  one  value per pixel
    # phase_field[0, 0, 2:5, 2:4] = phase_field[0, 0, 2:5, 2:4] / 3  #
    #
    # apply material distribution
    material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * np.power(phase_field[0, 0],
                                                                                p)

    # Set up the equilibrium system
    macro_gradient_field = discretization_fixture.get_macro_gradient_field(macro_gradient)
    rhs = discretization_fixture.get_rhs(material_data_field_C_0_rho, macro_gradient_field)

    K_fun = lambda x: discretization_fixture.apply_system_matrix(material_data_field_C_0_rho, x,
                                                                 formulation='small_strain')
    M_fun = lambda x: 1 * x

    displacement_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(500), toler=1e-10)

    homogenized_stress = discretization_fixture.get_homogenized_stress(
        material_data_field_ijklqxyz=material_data_field_C_0_rho,
        displacement_field_fnxyz=displacement_field,
        macro_gradient_field_ijqxyz=macro_gradient_field,
        formulation='small_strain')

    objective_function_stress_part = topology_optimization.objective_function_stress_equivalence(
        discretization=discretization_fixture,
        actual_stress_ij=homogenized_stress,
        target_stress_ij=target_stress)

    # objective_function = topology_optimization.objective_function_small_strain(
    #     discretization=discretization_fixture,
    #     actual_stress_ij=homogenized_stress,
    #     target_stress_ij=target_stress,
    #     phase_field_1nxyz=phase_field,
    #     eta=1,
    #     w=1)

    # phase field gradient potential for a phase field without perturbation
    df_drho_analytical = (
        topology_optimization.partial_derivative_of_objective_function_stress_equivalence_wrt_phase_field(
            discretization_fixture,
            phase_field_1nxyz=phase_field,
            target_stress_ij=target_stress,
            actual_stress_ij=homogenized_stress,
            material_data_field_ijklqxyz=material_data_field_C_0,
            displacement_field_fnxyz=displacement_field,
            macro_gradient_field_ijqxyz=macro_gradient_field,
            p=p))

    # get analytical partial derivative of phase field gradient potential for a phase field with respect to phase-field

    error_fd_vs_analytical = []
    for epsilon in epsilons:
        # loop over every single element of phase field
        for x in np.arange(discretization_fixture.nb_of_pixels[0]):
            for y in np.arange(discretization_fixture.nb_of_pixels[1]):
                # set phase_field to ones
                phase_field_perturbed = phase_field.copy()  # Phase field has  one  value per pixel
                phase_field_perturbed[0, 0, x, y] = phase_field_perturbed[0, 0, x, y] + epsilon / 2

                # apply material distribution
                material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * np.power(phase_field_perturbed[0, 0],
                                                                                            p)

                # rhs = discretization_fixture.get_rhs(material_data_field_C_0_rho, macro_gradient_field)
                #
                # K_fun = lambda x: discretization_fixture.apply_system_matrix(material_data_field_C_0_rho, x,
                #                                                              formulation='small_strain')
                # M_fun = lambda x: 1 * x
                #
                # displacement_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(500), toler=1e-10)

                homogenized_stress_plus = discretization_fixture.get_homogenized_stress(
                    material_data_field_ijklqxyz=material_data_field_C_0_rho,
                    displacement_field_fnxyz=displacement_field,
                    macro_gradient_field_ijqxyz=macro_gradient_field,
                    formulation='small_strain')

                objective_function_stress_part_perturbed_plus = topology_optimization.objective_function_stress_equivalence(
                    discretization=discretization_fixture,
                    actual_stress_ij=homogenized_stress_plus,
                    target_stress_ij=target_stress)

                phase_field_perturbed[0, 0, x, y] = phase_field_perturbed[0, 0, x, y] - epsilon

                # apply material distribution
                material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * np.power(phase_field_perturbed[0, 0],
                                                                                            p)

                homogenized_stress_minus = discretization_fixture.get_homogenized_stress(
                    material_data_field_ijklqxyz=material_data_field_C_0_rho,
                    displacement_field_fnxyz=displacement_field,
                    macro_gradient_field_ijqxyz=macro_gradient_field,
                    formulation='small_strain')

                objective_function_stress_part_perturbed_minus = topology_optimization.objective_function_stress_equivalence(
                    discretization=discretization_fixture,
                    actual_stress_ij=homogenized_stress_minus,
                    target_stress_ij=target_stress)

                # objective_function_perturbed = topology_optimization.objective_function_small_strain(
                #     discretization=discretization_fixture,
                #     actual_stress_ij=homogenized_stress,
                #     target_stress_ij=target_stress,
                #     phase_field_1nxyz=phase_field_perturbed,
                #     eta=1,
                #     w=1)

                fd_derivative[0, 0, x, y] = (objective_function_stress_part_perturbed_plus
                                             -
                                             objective_function_stress_part_perturbed_minus) / epsilon
                # fd_derivative_wo[0, 0, x, y] = (
                #                                 objective_function_perturbed - objective_function) / epsilon

        # print(df_drho_analytical[0, 0])
        error_fd_vs_analytical.append(
            np.linalg.norm((fd_derivative[0, 0] - df_drho_analytical), 'fro'))
        np.prod(discretization_fixture.pixel_size)
        # print(fd_derivative[0, 0, 0, 0])
        # print(f_rho_grad_potential_analytical)

        print(error_fd_vs_analytical)

        assert error_fd_vs_analytical[-1] < epsilon * 10, (
            "Finite difference derivative do not corresponds to the analytical expression "
            "for partial derivative of gradient of phase-field potential ")


@pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
    ([3, 4], 0, [6, 8]),
    ([2, 5], 0, [12, 7]),
    ([3, 4], 1, [6, 8]),
    ([2, 5], 1, [12, 7])])
def test_nullity_of_adjoint_potential(discretization_fixture, plot=False):
    # create material data field
    K_0, G_0 = domain.get_bulk_and_shear_modulus(E=1, poison=0.2)

    material_C_0 = domain.get_elastic_material_tensor(dim=discretization_fixture.domain_dimension,
                                                      K=K_0,
                                                      mu=G_0,
                                                      kind='linear')

    material_data_field_C_0 = np.einsum('ijkl,qxy->ijklqxy', material_C_0,
                                        np.ones(np.array([discretization_fixture.nb_quad_points_per_pixel,
                                                          *discretization_fixture.nb_of_pixels])))

    phase_field = np.random.rand(*discretization_fixture.get_scalar_sized_field().shape)  # set random distribution
    # apply material distribution
    material_data_field_C_0 = material_data_field_C_0[..., :, :] * phase_field[0, 0]

    macro_gradient = np.array([[0.1, 0], [0, 0.1]])
    macro_gradient_field = discretization_fixture.get_macro_gradient_field(macro_gradient)
    #
    rhs = discretization_fixture.get_rhs(material_data_field_C_0, macro_gradient_field)

    K_fun = lambda x: discretization_fixture.apply_system_matrix(material_data_field_C_0, x,
                                                                 formulation='small_strain')
    M_fun = lambda x: 1 * x

    displacement_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(500), toler=1e-12)

    #
    adjoint_field = np.random.rand(
        *discretization_fixture.get_displacement_sized_field().shape)  # set random adjoint field

    # compute stress field corresponding to equilibrated displacement
    stress_field = discretization_fixture.get_stress_field(material_data_field_C_0, displacement_field,
                                                           macro_gradient_field)

    adjoint_potential = topology_optimization.adjoint_potential(discretization_fixture, stress_field, adjoint_field)

    assert adjoint_potential < 1e-12, (
        "Adjoint potential should be 0 for every solution of equilibrium constrain"
        "but adjoint_potential = {}".format(adjoint_potential))  # this number depends on toler of CG solver !!!


@pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
    ([3, 4], 0, [6, 8]),
    ([2, 5], 0, [12, 7]),
    ([3, 4], 1, [6, 8]),
    ([2, 5], 1, [12, 7])])
def test_finite_difference_check_of_adjoint_potential_wrt_displacement(discretization_fixture, plot=False):
    epsilons = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    # epsilons = [1e-4]
    fd_derivative = np.zeros([*discretization_fixture.get_displacement_sized_field().shape])

    macro_gradient = np.array([[0.01, 0], [0, 0.01]])

    # create material data field
    K_0, G_0 = domain.get_bulk_and_shear_modulus(E=1, poison=0.2)

    material_C_0 = domain.get_elastic_material_tensor(dim=discretization_fixture.domain_dimension,
                                                      K=K_0,
                                                      mu=G_0,
                                                      kind='linear')

    material_data_field_C_0 = np.einsum('ijkl,qxy->ijklqxy', material_C_0,
                                        np.ones(np.array([discretization_fixture.nb_quad_points_per_pixel,
                                                          *discretization_fixture.nb_of_pixels])))

    phase_field = np.random.rand(*discretization_fixture.get_scalar_sized_field().shape)  # set random distribution
    # apply material distribution
    material_data_field_C_0 = material_data_field_C_0[..., :, :] * phase_field[0, 0]

    macro_gradient_field = discretization_fixture.get_macro_gradient_field(macro_gradient)
    #
    rhs = discretization_fixture.get_rhs(material_data_field_C_0, macro_gradient_field)

    K_fun = lambda x: discretization_fixture.apply_system_matrix(material_data_field_C_0, x,
                                                                 formulation='small_strain')
    M_fun = lambda x: 1 * x

    displacement_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(500), toler=1e-6)

    # compute stress field corresponding to equilibrated displacement
    stress_field = discretization_fixture.get_stress_field(material_data_field_C_0, displacement_field,
                                                           macro_gradient_field)

    # create random adjoint field
    adjoint_field = np.random.rand(
        *discretization_fixture.get_displacement_sized_field().shape)  # set random adjoint field

    # compute adjoint_potential
    adjoint_potential = topology_optimization.adjoint_potential(discretization_fixture, stress_field, adjoint_field)

    dg_du_analytical = topology_optimization.partial_derivative_of_adjoint_potential_wrt_displacement(
        discretization_fixture,
        material_data_field_C_0,
        adjoint_field)

    error_fd_vs_analytical = []
    for epsilon in epsilons:
        fd_norms = np.zeros(
            [discretization_fixture.cell.unknown_shape[0], discretization_fixture.nb_unique_nodes_per_pixel])

        # loop over every single element of displacement field
        for f in np.arange(discretization_fixture.cell.unknown_shape[0]):
            for n in np.arange(discretization_fixture.nb_unique_nodes_per_pixel):

                for x in np.arange(discretization_fixture.nb_of_pixels[0]):
                    for y in np.arange(discretization_fixture.nb_of_pixels[1]):
                        displacement_field_fnxyz = np.copy(displacement_field)
                        displacement_field_fnxyz[f, n, x, y] = displacement_field_fnxyz[f, n, x, y] + epsilon
                        # compute stress field corresponding to equilibrated displacement
                        stress_field = discretization_fixture.get_stress_field(material_data_field_C_0,
                                                                               displacement_field_fnxyz,
                                                                               macro_gradient_field)

                        adjoint_potential_perturbed = topology_optimization.adjoint_potential(discretization_fixture,
                                                                                              stress_field,
                                                                                              adjoint_field)

                        fd_derivative[f, n, x, y] = (adjoint_potential_perturbed - adjoint_potential) / epsilon
            fd_norms[f, n] = np.sum(np.linalg.norm((fd_derivative[f, n] - dg_du_analytical[f, n]), 'fro'))

            # print('finite difference norm {0}{1} = {2}'.format(f, n, np.linalg.norm(fd_derivative[f, n], 'fro')))
            # print('analytical derivative {0}{1} = {2}'.format(f, n, np.linalg.norm(dg_du_analytical[f, n], 'fro')))
        # (error_fd_vs_analytical)

        error_fd_vs_analytical.append(np.sum(fd_norms))
        assert error_fd_vs_analytical[-1] < 1e-6, (
            "Finite difference derivative  do not corresponds to the analytical expression "
            "for partial derivative of adjoint potential  w.r.t. displacement ")


@pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
    ([3, 4], 0, [6, 8]),
    ([2, 5], 0, [12, 7]),
    ([3, 4], 1, [6, 8]),
    ([2, 5], 1, [12, 7])])
def test_finite_difference_check_of_pd_objective_function_wrt_displacement_small_strain(discretization_fixture):
    # This test compares analytical expression for partial derivative  of objective function w.r.t. displacement
    epsilons = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    # epsilons = [1e-2]
    fd_derivative = np.zeros([*discretization_fixture.get_displacement_sized_field().shape])
    fd_derivative_wo_phase = np.zeros([*discretization_fixture.get_displacement_sized_field().shape])
    # set stress difference to zero
    target_stress = np.array([[1, 0.5], [0.5, 2]])
    macro_gradient = np.array([[1, 0], [0, 1]])

    ## compute objective function without perturbations
    phase_field = discretization_fixture.get_scalar_sized_field() + 1  # Phase field has  one  value per pixel
    phase_field[0, 0, 2:4, 2:4] = phase_field[0, 0, 2:4, 2:4] / 2  # for
    #
    K_1, G_1 = domain.get_bulk_and_shear_modulus(E=3, poison=0.2)

    mat_1 = domain.get_elastic_material_tensor(dim=discretization_fixture.domain_dimension, K=K_1, mu=G_1,
                                               kind='linear')

    material_data_field = np.einsum('ijkl,qxy->ijklqxy', mat_1,
                                    np.ones(np.array([discretization_fixture.nb_quad_points_per_pixel,
                                                      *discretization_fixture.nb_of_pixels])))

    # Update material data based on current Phase-field
    material_data_field_i = (phase_field) * material_data_field

    ##### solve equilibrium constrain
    # set up system
    macro_gradient_field = discretization_fixture.get_macro_gradient_field(macro_gradient)
    rhs = discretization_fixture.get_rhs(material_data_field_i, macro_gradient_field)

    K_fun = lambda x: discretization_fixture.apply_system_matrix(material_data_field_i, x)
    M_fun = lambda x: 1 * x

    displacement_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(500), toler=1e-6)

    # test homogenized stress
    homogenized_stress = discretization_fixture.get_homogenized_stress(
        material_data_field_ijklqxyz=material_data_field_i,
        displacement_field_fnxyz=displacement_field,
        macro_gradient_field_ijqxyz=macro_gradient_field,
        formulation='small_strain')

    actual_stress_field = np.zeros(discretization_fixture.gradient_size)
    actual_stress_field[..., :] = homogenized_stress[(...,) + (np.newaxis,) * (actual_stress_field.ndim - 2)]

    stress_diff = target_stress - homogenized_stress
    # objective function  without phase-field
    f_sigma = np.sum(stress_diff ** 2)
    # objective function
    objective_function = topology_optimization.objective_function_small_strain(
        discretization=discretization_fixture,
        actual_stress_ij=homogenized_stress,
        target_stress_ij=target_stress,
        phase_field_1nxyz=phase_field,
        eta=1,
        w=1)

    d_of_d_u_analytical = topology_optimization.partial_der_of_objective_function_wrt_displacement_small_strain(
        discretization_fixture,
        material_data_field_i,
        stress_diff,
        eta=1,
        w=1)

    error_fd_vs_analytical = []
    for epsilon in epsilons:
        fd_norms = np.zeros(
            [discretization_fixture.cell.unknown_shape[0], discretization_fixture.nb_unique_nodes_per_pixel])
        fd_norms_wo_phase = np.zeros(
            [discretization_fixture.cell.unknown_shape[0], discretization_fixture.nb_unique_nodes_per_pixel])

        # loop over every single element of displacement field
        for f in np.arange(discretization_fixture.cell.unknown_shape[0]):
            for n in np.arange(discretization_fixture.nb_unique_nodes_per_pixel):
                # loop over every single element of phase field
                for x in np.arange(discretization_fixture.nb_of_pixels[0]):
                    for y in np.arange(discretization_fixture.nb_of_pixels[1]):
                        # perturb f,n,x,y,z-component displacement field with epsilon
                        displacement_field_fnxyz = np.copy(displacement_field)
                        displacement_field_fnxyz[f, n, x, y] = displacement_field_fnxyz[f, n, x, y] + epsilon

                        # homogenized stress
                        homogenized_stress = discretization_fixture.get_homogenized_stress(
                            material_data_field_ijklqxyz=material_data_field_i,
                            displacement_field_fnxyz=displacement_field_fnxyz,
                            macro_gradient_field_ijqxyz=macro_gradient_field,
                            formulation='small_strain')

                        stress_diff = target_stress - homogenized_stress
                        # objective function without phase-field
                        f_sigma_perturbed = np.sum(stress_diff ** 2)

                        # objective function
                        objective_function_perturbed = topology_optimization.objective_function_small_strain(
                            discretization=discretization_fixture,
                            actual_stress_ij=homogenized_stress,
                            target_stress_ij=target_stress,
                            phase_field_1nxyz=phase_field,
                            eta=1,
                            w=1)

                        fd_derivative_wo_phase[f, n, x, y] = (f_sigma_perturbed - f_sigma) / epsilon
                        fd_derivative[f, n, x, y] = (objective_function_perturbed - objective_function) / epsilon

                fd_norms[f, n] = np.sum(np.linalg.norm((fd_derivative[f, n] - d_of_d_u_analytical[f, n]), 'fro'))
                fd_norms_wo_phase[f, n] = np.sum(
                    np.linalg.norm((fd_derivative_wo_phase[f, n] - d_of_d_u_analytical[f, n]), 'fro'))
        # print('finite difference norm {0}{1} = {2}'.format(f, n, np.linalg.norm(fd_derivative[f, n], 'fro')))
        # print('analytical derivative {0}{1} = {2}'.format(f, n, np.linalg.norm(d_of_d_u_analytical[f, n], 'fro')))
        # assert np.allclose(fd_norms, fd_norms_wo_phase), (
        #     "Finite difference derivative with phase field do not corresponds to the one without phase field "
        #     "for partial derivative of adjoint potential  w.r.t. displacement "
        #     "epsilon = {}".format(epsilon))

        error_fd_vs_analytical.append(np.sum(fd_norms))

        assert error_fd_vs_analytical[-1] < epsilon * 10, (
            "Finite difference derivative  do not corresponds to the analytical expression "
            "for partial derivative of adjoint potential  w.r.t. displacement ")


# TODO set p as a parameter
@pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
    ([3, 4], 0, [6, 8]),
    ([2, 5], 0, [12, 7]),
    ([3, 4], 1, [6, 8]),
    ([2, 5], 1, [12, 7])])
def test_finite_difference_check_of_adjoint_potential_wrt_phase_field(discretization_fixture):
    epsilons = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, ]
    # epsilons = [1e-4]
    p = 4
    fd_derivative = discretization_fixture.get_scalar_sized_field()

    macro_gradient = np.array([[0.01, 0], [0, 0.01]])

    # create material data field
    K_0, G_0 = domain.get_bulk_and_shear_modulus(E=1, poison=0.2)

    material_C_0 = domain.get_elastic_material_tensor(dim=discretization_fixture.domain_dimension,
                                                      K=K_0,
                                                      mu=G_0,
                                                      kind='linear')

    material_data_field_C_0 = np.einsum('ijkl,qxy->ijklqxy', material_C_0,
                                        np.ones(np.array([discretization_fixture.nb_quad_points_per_pixel,
                                                          *discretization_fixture.nb_of_pixels])))

    phase_field = np.random.rand(
        *discretization_fixture.get_scalar_sized_field().shape)  # * 0 + 1  # set random distribution
    # apply material distribution
    material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * np.power(phase_field[0, 0],
                                                                                p)

    # Set up the equilibrium system
    macro_gradient_field = discretization_fixture.get_macro_gradient_field(macro_gradient)

    # create random displacement field
    displacement_field = np.random.rand(*discretization_fixture.get_scalar_sized_field().shape)

    # ----------------------------------------------------------------------
    # compute stress field corresponding to displacement
    stress_field = discretization_fixture.get_stress_field(material_data_field_C_0_rho,
                                                           displacement_field,
                                                           macro_gradient_field)

    # create random adjoint field
    adjoint_field = np.random.rand(
        *discretization_fixture.get_displacement_sized_field().shape)  # set random adjoint field

    # compute adjoint_potential
    adjoint_potential = topology_optimization.adjoint_potential(discretization_fixture,
                                                                stress_field,
                                                                adjoint_field)

    dg_drho_analytical = topology_optimization.partial_derivative_of_adjoint_potential_wrt_phase_field(
        discretization=discretization_fixture,
        material_data_field_ijklqxyz=material_data_field_C_0,
        displacement_field_fnxyz=displacement_field,
        macro_gradient_field_ijqxyz=macro_gradient_field,
        phase_field_1nxyz=phase_field,
        adjoint_field_fnxyz=adjoint_field,
        p=p)

    error_fd_vs_analytical = []
    for epsilon in epsilons:
        # loop over every single element of displacement field
        for x in np.arange(discretization_fixture.nb_of_pixels[0]):
            for y in np.arange(discretization_fixture.nb_of_pixels[1]):
                phase_field_perturbed = np.copy(phase_field)
                phase_field_perturbed[0, 0, x, y] = phase_field_perturbed[0, 0, x, y] + epsilon
                # compute stress field corresponding to equilibrated displacement

                # apply material distribution
                material_data_field_C_0_per = material_data_field_C_0[..., :, :] * np.power(phase_field_perturbed[0, 0],
                                                                                            p)
                #
                stress_field = discretization_fixture.get_stress_field(
                    material_data_field_ijklqxyz=material_data_field_C_0_per,
                    displacement_field_fnxyz=displacement_field,
                    macro_gradient_field_ijqxyz=macro_gradient_field,
                    formulation='small_strain')

                adjoint_potential_perturbed = topology_optimization.adjoint_potential(
                    discretization=discretization_fixture,
                    stress_field_ijqxyz=stress_field,
                    adjoint_field_fnxyz=adjoint_field)

                fd_derivative[0, 0, x, y] = (adjoint_potential_perturbed
                                             -
                                             adjoint_potential) / epsilon

        fd_norm = np.sum(np.linalg.norm((fd_derivative[0, 0] - dg_drho_analytical), 'fro'))

        # print('finite difference norm {0}{1} = {2}'.format(f, n, np.linalg.norm(fd_derivative[f, n], 'fro')))
        # print('analytical derivative {0}{1} = {2}'.format(f, n, np.linalg.norm(dg_du_analytical[f, n], 'fro')))
        # (error_fd_vs_analytical)

        error_fd_vs_analytical.append(fd_norm)

        assert error_fd_vs_analytical[-1] < epsilon * 1e2, (
            "Finite difference derivative  do not corresponds to the analytical expression "
            "for partial derivative of adjoint potential  w.r.t. displacement "
            "for epsilon = {} and p = {}".format(epsilon, p))


@pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
    ([3, 4], 0, [6, 8]),
    ([2, 5], 0, [12, 7]),
    ([3, 4], 1, [6, 8]),
    ([2, 5], 1, [12, 7])
])
def test_finite_difference_check_of_stress_equivalence_potential222(discretization_fixture):
    # TODO this check works for p=1,2
    epsilons = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    # epsilons = [1e-4]
    p = 2
    fd_derivative = discretization_fixture.get_scalar_sized_field()

    target_stress = np.array([[1, 0.3], [0.3, 2]])
    macro_gradient = np.array([[0.01, 0], [0, 0.02]])

    # create material data field
    K_0, G_0 = domain.get_bulk_and_shear_modulus(E=1, poison=0.2)

    elastic_C_1 = domain.get_elastic_material_tensor(dim=discretization_fixture.domain_dimension,
                                                     K=K_0,
                                                     mu=G_0,
                                                     kind='linear')

    material_data_field_C_0 = np.einsum('ijkl,qxy->ijklqxy', elastic_C_1,
                                        np.ones(np.array([discretization_fixture.nb_quad_points_per_pixel,
                                                          *discretization_fixture.nb_of_pixels])))

    # phase_field = discretization_fixture.get_scalar_sized_field() + 1  #
    phase_field = np.random.rand(*discretization_fixture.get_scalar_sized_field().shape)  # set random distribution

    # apply material distribution
    material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * np.power(phase_field[0, 0],
                                                                                p)

    # Set up the equilibrium system
    macro_gradient_field = discretization_fixture.get_macro_gradient_field(macro_gradient)

    # create random displacement field
    displacement_field = np.random.rand(*discretization_fixture.get_scalar_sized_field().shape)

    # ----------------------------------------------------------------------
    # compute stress field corresponding to displacement
    homogenized_stress = discretization_fixture.get_homogenized_stress(
        material_data_field_ijklqxyz=material_data_field_C_0_rho,
        displacement_field_fnxyz=displacement_field,
        macro_gradient_field_ijqxyz=macro_gradient_field,
        formulation='small_strain')

    objective_function_stress_part = topology_optimization.objective_function_stress_equivalence(
        discretization=discretization_fixture,
        actual_stress_ij=homogenized_stress,
        target_stress_ij=target_stress)

    # phase field gradient potential for a phase field without perturbation
    # objective_function = topology_optimization.objective_function_small_strain(
    #     discretization=discretization_fixture,
    #     actual_stress_ij=homogenized_stress,
    #     target_stress_ij=target_stress,
    #     phase_field_1nxyz=phase_field,
    #     eta=1,
    #     w=1)

    # get analytical partial derivative of stress equivalent potential for a phase field with respect to phase-field
    df_drho_analytical = (
        topology_optimization.partial_derivative_of_objective_function_stress_equivalence_wrt_phase_field(
            discretization=discretization_fixture,
            material_data_field_ijklqxyz=material_data_field_C_0,
            displacement_field_fnxyz=displacement_field,
            macro_gradient_field_ijqxyz=macro_gradient_field,
            phase_field_1nxyz=phase_field,
            target_stress_ij=target_stress,
            actual_stress_ij=homogenized_stress,
            p=p
        ))

    error_fd_vs_analytical = []
    for epsilon in epsilons:
        # loop over every single element of phase field
        for x in np.arange(discretization_fixture.nb_of_pixels[0]):
            for y in np.arange(discretization_fixture.nb_of_pixels[1]):
                # set phase_field to ones
                phase_field_perturbed = np.copy(phase_field)  # Phase field has  one  value per pixel
                # phase_field_perturbed=phase_field_perturbed**p
                phase_field_perturbed[0, 0, x, y] = phase_field_perturbed[0, 0, x, y] + epsilon

                # apply material distribution
                material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * np.power(phase_field_perturbed[0, 0],
                                                                                            p)  # ** p

                homogenized_stress = discretization_fixture.get_homogenized_stress(
                    material_data_field_ijklqxyz=material_data_field_C_0_rho,
                    displacement_field_fnxyz=displacement_field,
                    macro_gradient_field_ijqxyz=macro_gradient_field,
                    formulation='small_strain')

                objective_function_stress_part_perturbed = topology_optimization.objective_function_stress_equivalence(
                    discretization=discretization_fixture,
                    actual_stress_ij=homogenized_stress,
                    target_stress_ij=target_stress)

                # objective_function_perturbed = topology_optimization.objective_function_small_strain(
                #     discretization=discretization_fixture,
                #     actual_stress_ij=homogenized_stress,
                #     target_stress_ij=target_stress,
                #     phase_field_1nxyz=phase_field_perturbed,
                #     eta=1,
                #     w=1)

                fd_derivative[0, 0, x, y] = (objective_function_stress_part_perturbed
                                             -
                                             objective_function_stress_part) / epsilon
                # fd_derivative_wo[0, 0, x, y] = (
                #                                 objective_function_perturbed - objective_function) / epsilon

        # print(df_drho_analytical[0, 0])
        fd_norm = np.sum(np.linalg.norm((fd_derivative[0, 0] - df_drho_analytical), 'fro'))

        # print(f_rho_grad_potential_analytical)
        error_fd_vs_analytical.append(fd_norm)

        # print(error_fd_vs_analytical)

        assert error_fd_vs_analytical[-1] < epsilon * 10, (
            "Finite difference derivative do not corresponds to the analytical expression "
            "for partial derivative of gradient of phase-field potential ")


@pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
    ([3, 4], 0, [6, 8]),
    ([2, 5], 0, [12, 7]),
    ([3, 4], 0, [6, 8]),
    ([2, 5], 0, [12, 7])  # TODO add gaus quadrature for bilinear elements
])
def test_finite_difference_check_of_objective_function_wrt_phase_field(discretization_fixture):
    # f = (flux_h -flux_target)^2 + w*eta* int (  (grad(rho))^2 )dx  +    int ( rho^2(1-rho)^2 ) / eta   dx
    # f =  f_sigma + w*(eta* f_rho_grad  + f_dw/eta)
    # f= objective_function_small_strain
    # f_grad
    #
    epsilons = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    # epsilons = [1e-4]
    p = 1
    fd_derivative = discretization_fixture.get_scalar_sized_field()

    target_stress = np.array([[1, 0.3], [0.3, 2]])
    macro_gradient = np.array([[0.01, 0], [0, 0.02]])

    # create material data field
    K_0, G_0 = domain.get_bulk_and_shear_modulus(E=1, poison=0.2)

    elastic_C_1 = domain.get_elastic_material_tensor(dim=discretization_fixture.domain_dimension,
                                                     K=K_0,
                                                     mu=G_0,
                                                     kind='linear')

    material_data_field_C_0 = np.einsum('ijkl,qxy->ijklqxy', elastic_C_1,
                                        np.ones(np.array([discretization_fixture.nb_quad_points_per_pixel,
                                                          *discretization_fixture.nb_of_pixels])))

    # phase_field = discretization_fixture.get_scalar_sized_field() + 1  #
    phase_field = np.random.rand(*discretization_fixture.get_scalar_sized_field().shape)  # set random distribution

    # apply material distribution
    material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * np.power(phase_field[0, 0],
                                                                                p)

    # Set up the equilibrium system
    macro_gradient_field = discretization_fixture.get_macro_gradient_field(macro_gradient)

    # create random displacement field
    displacement_field = np.random.rand(*discretization_fixture.get_scalar_sized_field().shape)

    # ----------------------------------------------------------------------
    # compute stress field corresponding to displacement
    homogenized_stress = discretization_fixture.get_homogenized_stress(
        material_data_field_ijklqxyz=material_data_field_C_0_rho,
        displacement_field_fnxyz=displacement_field,
        macro_gradient_field_ijqxyz=macro_gradient_field,
        formulation='small_strain')

    # get analytical partial derivative of stress equivalent potential for a phase field with respect to phase-field

    df_drho_analytical = topology_optimization.partial_derivative_of_objective_function_wrt_phase_field(
        discretization=discretization_fixture,
        material_data_field_ijklqxyz=material_data_field_C_0,
        displacement_field_fnxyz=displacement_field,
        macro_gradient_field_ijqxyz=macro_gradient_field,
        phase_field_1nxyz=phase_field,
        target_stress_ij=target_stress,
        actual_stress_ij=homogenized_stress,
        p=p
    )

    error_fd_vs_analytical = []
    for epsilon in epsilons:
        # loop over every single element of phase field
        for x in np.arange(discretization_fixture.nb_of_pixels[0]):
            for y in np.arange(discretization_fixture.nb_of_pixels[1]):
                # set phase_field to ones
                phase_field_perturbed = np.copy(phase_field)  # Phase field has  one  value per pixel
                # phase_field_perturbed=phase_field_perturbed**p
                phase_field_perturbed[0, 0, x, y] = phase_field_perturbed[0, 0, x, y] + epsilon / 2

                # apply material distribution
                material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * np.power(phase_field_perturbed[0, 0],
                                                                                            p)  # ** p
                # plus epsilon
                homogenized_stress_plus = discretization_fixture.get_homogenized_stress(
                    material_data_field_ijklqxyz=material_data_field_C_0_rho,
                    displacement_field_fnxyz=displacement_field,
                    macro_gradient_field_ijqxyz=macro_gradient_field,
                    formulation='small_strain')

                objective_function_perturbed_plus = topology_optimization.objective_function_small_strain(
                    discretization=discretization_fixture,
                    actual_stress_ij=homogenized_stress_plus,
                    target_stress_ij=target_stress,
                    phase_field_1nxyz=phase_field_perturbed,
                    eta=1, w=1)

                phase_field_perturbed[0, 0, x, y] = phase_field_perturbed[0, 0, x, y] - epsilon
                # apply material distribution
                material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * np.power(phase_field_perturbed[0, 0],
                                                                                            p)  # ** p
                homogenized_stress_minus = discretization_fixture.get_homogenized_stress(
                    material_data_field_ijklqxyz=material_data_field_C_0_rho,
                    displacement_field_fnxyz=displacement_field,
                    macro_gradient_field_ijqxyz=macro_gradient_field,
                    formulation='small_strain')

                objective_function_perturbed_minus = topology_optimization.objective_function_small_strain(
                    discretization=discretization_fixture,
                    actual_stress_ij=homogenized_stress_minus,
                    target_stress_ij=target_stress,
                    phase_field_1nxyz=phase_field_perturbed,
                    eta=1, w=1)

                fd_derivative[0, 0, x, y] = (objective_function_perturbed_plus
                                             -
                                             objective_function_perturbed_minus) / epsilon

        # print(df_drho_analytical[0, 0])
        fd_norm = np.sum(np.linalg.norm((fd_derivative[0, 0] - df_drho_analytical[0, 0]), 'fro'))

        # print(f_rho_grad_potential_analytical)
        error_fd_vs_analytical.append(fd_norm)

        print(error_fd_vs_analytical)

        assert error_fd_vs_analytical[-1] < epsilon * 100, (
            "Finite difference derivative do not corresponds to the analytical expression "
            "for partial derivative of gradient of phase-field potential "
            "error_fd_vs_analytical = {}".format(error_fd_vs_analytical))


@pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
    ([3, 4], 0, [6, 8]),
    ([2, 5], 0, [12, 7]),
    ([3, 4], 0, [6, 8]),
    ([2, 5], 0, [12, 7])
])
def test_finite_difference_check_of_objective_function_with_adjoin_potential_wrt_phase_field(discretization_fixture):
    # f = (flux_h -flux_target)^2 + w*eta* int (  (grad(rho))^2 )dx  +    int ( rho^2(1-rho)^2 ) / eta   dx
    # f =  f_sigma + w*(eta* f_rho_grad  + f_dw/eta)
    # f= objective_function_small_strain
    # f_grad
    #
    epsilons = [1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    # epsilons = [1e-4]
    p = 1
    fd_derivative = discretization_fixture.get_scalar_sized_field()

    target_stress = np.array([[1, 0.3], [0.3, 2]])
    macro_gradient = np.array([[0.2, 0], [0, 0.2]])

    # create material data field
    K_0, G_0 = domain.get_bulk_and_shear_modulus(E=1, poison=0.2)

    elastic_C_1 = domain.get_elastic_material_tensor(dim=discretization_fixture.domain_dimension,
                                                     K=K_0,
                                                     mu=G_0,
                                                     kind='linear')

    material_data_field_C_0 = np.einsum('ijkl,qxy->ijklqxy', elastic_C_1,
                                        np.ones(np.array([discretization_fixture.nb_quad_points_per_pixel,
                                                          *discretization_fixture.nb_of_pixels])))

    # phase_field = discretization_fixture.get_scalar_sized_field() + 1  #
    phase_field = np.random.rand(*discretization_fixture.get_scalar_sized_field().shape)  # set random distribution

    # apply material distribution
    material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * np.power(phase_field[0, 0],
                                                                                p)

    # Set up the equilibrium system
    macro_gradient_field = discretization_fixture.get_macro_gradient_field(macro_gradient)

    # Solve mechanical equilibrium constrain
    rhs = discretization_fixture.get_rhs(material_data_field_C_0_rho, macro_gradient_field)

    K_fun = lambda x: discretization_fixture.apply_system_matrix(material_data_field_C_0_rho, x,
                                                                 formulation='small_strain')
    M_fun = lambda x: 1 * x

    displacement_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(500), toler=1e-12)

    # ----------------------------------------------------------------------
    # compute homogenized stress field corresponding to displacement
    homogenized_stress = discretization_fixture.get_homogenized_stress(
        material_data_field_ijklqxyz=material_data_field_C_0_rho,
        displacement_field_fnxyz=displacement_field,
        macro_gradient_field_ijqxyz=macro_gradient_field,
        formulation='small_strain')

    objective_function = topology_optimization.objective_function_small_strain(
        discretization=discretization_fixture,
        actual_stress_ij=homogenized_stress,
        target_stress_ij=target_stress,
        phase_field_1nxyz=phase_field,
        eta=1, w=1)

    # get analytical partial derivative of stress equivalent potential for a phase field with respect to phase-field

    df_drho_analytical = topology_optimization.partial_derivative_of_objective_function_wrt_phase_field(
        discretization=discretization_fixture,
        material_data_field_ijklqxyz=material_data_field_C_0,
        displacement_field_fnxyz=displacement_field,
        macro_gradient_field_ijqxyz=macro_gradient_field,
        phase_field_1nxyz=phase_field,
        target_stress_ij=target_stress,
        actual_stress_ij=homogenized_stress,
        p=p)

    # stress difference potential: actual_stress_ij is homogenized stress
    stress_difference_ij = homogenized_stress - target_stress

    stress_difference_ijqxyz = discretization_fixture.get_gradient_size_field()
    stress_difference_ijqxyz[:, :, ...] = stress_difference_ij[
        (...,) + (np.newaxis,) * (stress_difference_ijqxyz.ndim - 2)]

    adjoint_field = topology_optimization.solve_adjoint_problem(
        discretization=discretization_fixture,
        material_data_field_ijklqxyz=material_data_field_C_0_rho,
        stress_difference_ij=stress_difference_ij,
        formulation='small_strain')

    stress_field = discretization_fixture.get_stress_field(
        material_data_field_ijklqxyz=material_data_field_C_0_rho,
        displacement_field_fnxyz=displacement_field,
        macro_gradient_field_ijqxyz=macro_gradient_field,
        formulation='small_strain')

    adjoint_potential = topology_optimization.adjoint_potential(
        discretization=discretization_fixture,
        stress_field_ijqxyz=stress_field,
        adjoint_field_fnxyz=adjoint_field)
    assert np.abs(adjoint_potential) < 1e-12, (
        "Adjoint potential si not zero for equilibrated stress field"
        " adjoint_potential = {}".format(adjoint_potential))

    dg_drho_analytical = topology_optimization.partial_derivative_of_adjoint_potential_wrt_phase_field(
        discretization=discretization_fixture,
        material_data_field_ijklqxyz=material_data_field_C_0,
        displacement_field_fnxyz=displacement_field,
        macro_gradient_field_ijqxyz=macro_gradient_field,
        phase_field_1nxyz=phase_field,
        adjoint_field_fnxyz=adjoint_field,
        p=p)

    # sensitivity_analytical_old = df_drho_analytical + dg_drho_analytical
    sensitivity_analytical = topology_optimization.sensitivity(
        discretization=discretization_fixture,
        material_data_field_ijklqxyz=material_data_field_C_0,
        displacement_field_fnxyz=displacement_field,
        macro_gradient_field_ijqxyz=macro_gradient_field,
        phase_field_1nxyz=phase_field,
        adjoint_field_fnxyz=adjoint_field,
        target_stress_ij=target_stress,
        actual_stress_ij=homogenized_stress,
        p=p,
        eta=1)

    error_fd_vs_analytical = []
    for epsilon in epsilons:
        # loop over every single element of phase field
        for x in np.arange(discretization_fixture.nb_of_pixels[0]):
            for y in np.arange(discretization_fixture.nb_of_pixels[1]):
                # set phase_field to ones
                phase_field_perturbed = np.copy(phase_field)  # Phase field has  one  value per pixel
                # phase_field_perturbed=phase_field_perturbed**p
                phase_field_perturbed[0, 0, x, y] = phase_field_perturbed[0, 0, x, y] + epsilon

                # apply material distribution
                material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * np.power(phase_field_perturbed[0, 0],
                                                                                            p)  # ** p

                homogenized_stress = discretization_fixture.get_homogenized_stress(
                    material_data_field_ijklqxyz=material_data_field_C_0_rho,
                    displacement_field_fnxyz=displacement_field,
                    macro_gradient_field_ijqxyz=macro_gradient_field,
                    formulation='small_strain')

                objective_function_perturbed = topology_optimization.objective_function_small_strain(
                    discretization=discretization_fixture,
                    actual_stress_ij=homogenized_stress,
                    target_stress_ij=target_stress,
                    phase_field_1nxyz=phase_field_perturbed,
                    eta=1, w=1)

                stress_field_perturbed = discretization_fixture.get_stress_field(
                    material_data_field_ijklqxyz=material_data_field_C_0_rho,
                    displacement_field_fnxyz=displacement_field,
                    macro_gradient_field_ijqxyz=macro_gradient_field,
                    formulation='small_strain')

                adjoint_potential_perturbed = topology_optimization.adjoint_potential(
                    discretization=discretization_fixture,
                    stress_field_ijqxyz=stress_field_perturbed,
                    adjoint_field_fnxyz=adjoint_field)

                fd_derivative[0, 0, x, y] = (objective_function_perturbed + adjoint_potential_perturbed
                                             -
                                             objective_function) / epsilon

        # print(df_drho_analytical[0, 0])
        fd_norm = np.sum(np.linalg.norm((fd_derivative[0, 0] - sensitivity_analytical[0, 0]), 'fro'))

        # print(f_rho_grad_potential_analytical)
        error_fd_vs_analytical.append(fd_norm)

        print(error_fd_vs_analytical)

    assert error_fd_vs_analytical[-1] < epsilon * 200, (
        "Finite difference derivative do not corresponds to the analytical expression "
        "for whole Sensitivity "
        "error_fd_vs_analytical = {}".format(error_fd_vs_analytical))  # 200 is housbumero


def test_phase_field_size_independance(plot=True):
    domain_size = [1, 1]
    eta = 1
    nb_pixels = (20, 20)
    problem_type = 'elasticity'
    element_types = ['linear_triangles']
    element_type = 0

    my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                      problem_type=problem_type)

    discretization_type = 'finite_element'

    discretization = domain.Discretization(cell=my_cell,
                                           number_of_pixels=nb_pixels,
                                           discretization_type=discretization_type,
                                           element_type=element_types[element_type])

    p = 1

    geometry_ID = 'geometry_III_3_2D'
    nodal_coordinates = discretization.get_nodal_points_coordinates()

    # phase_field_0 = discretization.get_scalar_sized_field()  # set random distribution
    # phase_field_0[] = 1  # material distribution
    phase_field_0 = np.random.rand(*discretization.get_scalar_sized_field().shape) ** 1  # set random distribution
    # phase_field_0[0, 0, phase_field_0.shape[2] * 1 // 4:phase_field_0.shape[2] * 3 // 4,
    # phase_field_0.shape[2] * 1 // 4:phase_field_0.shape[2] * 3 // 4] = 0

    # linfunc = lambda x: 1 * x
    # cos_fun = lambda x: 1 * np.abs(np.cos(5*x*np.pi/2))

    # phase_field_0[0, 0] = cos_fun(nodal_coordinates[0, 0])

    # linfunc(nodal_coordinates)
    x_coords = np.linspace(0, discretization.domain_size[0], discretization.nb_of_pixels[0] + 1, endpoint=True)
    y_coords = np.linspace(0, discretization.domain_size[1], discretization.nb_of_pixels[1] + 1, endpoint=True)

    phase_field_0_periodic = np.c_[phase_field_0[0, 0], phase_field_0[0, 0, :, 0]]  # add a column
    phase_field_0_periodic = np.r_[phase_field_0_periodic, [phase_field_0_periodic[0, :]]]  # add a column

    phase_field_interpolator = sc.interpolate.interp2d(x_coords,
                                                       y_coords,
                                                       phase_field_0_periodic,
                                                       kind='linear')
    # f.z.reshape(discretization_fixture.nb_of_pixels)

    f_dw_0_old = topology_optimization.compute_double_well_potential(discretization=discretization,
                                                                     phase_field_1nxyz=phase_field_0,
                                                                     eta=eta)
    f_dw_0 = topology_optimization.compute_double_well_potential_NEW(discretization=discretization,
                                                                     phase_field_1nxyz=phase_field_0,
                                                                     eta=eta)
    f_dw_quad = topology_optimization.compute_double_well_potential_Gauss_quad(discretization=discretization,
                                                                               phase_field_1nxyz=phase_field_0,
                                                                               eta=eta)

    f_dphase_0 = topology_optimization.compute_gradient_of_phase_field_potential(discretization=discretization,
                                                                                 phase_field_1nxyz=phase_field_0,
                                                                                 eta=eta)
    f_ddw_drho_0 = topology_optimization.partial_der_of_double_well_potential_wrt_density_NEW(
        discretization=discretization,
        phase_field_1nxyz=phase_field_0,
        eta=eta)

    homogenized_stress = np.array([[1, 0.0, 0.0],
                                   [0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0]])
    target_stress = np.array([[1, 0.0, 0.0],
                              [0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0]])

    of_0 = topology_optimization.objective_function_small_strain(
        discretization=discretization,
        actual_stress_ij=homogenized_stress,
        target_stress_ij=target_stress,
        phase_field_1nxyz=phase_field_0,
        eta=1,
        w=1)
    print()
    # print(f_dw_0_old)
    # print(f_dw_quad)
    print('f_dw_grid_{}_old_initial =  {} '.format(nb_pixels[0], f_dw_0_old))
    print('f_dw_grid_{}_NEW_initial =  {} '.format(nb_pixels[0], f_dw_0))
    print('f_dw_grid_{}_Gquad_initial =  {} '.format(nb_pixels[0], f_dw_quad))

    # print(f_ddw_drho_0)
    print('objective {} =  {} '.format(nb_pixels[0], of_0))
    print('Interpolated geometries')
    if plot:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        # plt.contourf(nodal_coordinates_k[0, 0], nodal_coordinates_k[1, 0],integrant)
        plt.figure()

        plt.plot(nodal_coordinates[0, 0, :, 0], phase_field_0[0, 0, :, 0], label='phase_field_0')
        plt.figure()
        # plt.show()

    for nb_pixel_x in [20, 40, 100, 500]:  # ,160,320
        nb_pixels = (nb_pixel_x, nb_pixel_x)

        discretization_k = domain.Discretization(cell=my_cell,
                                                 number_of_pixels=nb_pixels,
                                                 discretization_type=discretization_type,
                                                 element_type=element_types[element_type])
        # test if the phase field functional return same value for differente domain sizes and number of pixels

        nodal_coordinates_k = discretization_k.get_nodal_points_coordinates()

        phase_field_k = discretization_k.get_scalar_sized_field()
        phase_field_k[0, 0] = phase_field_interpolator(nodal_coordinates_k[0, 0, :, 0], nodal_coordinates_k[1, 0, 0, :])

        integrant = 16 * (phase_field_k[0, 0] ** 2) * (1 - phase_field_k[0, 0]) ** 2
        grad_integrant_fnxyz = 1 * (2 * phase_field_k[0, 0] * (
                2 * phase_field_k[0, 0] * phase_field_k[0, 0] - 3 * phase_field_k[0, 0] + 1))

        # at_quad_points=discretization.evaluate_at_quad_points(phase_field_k)
        if plot:
            import matplotlib.pyplot as plt
            from matplotlib.collections import LineCollection
            # plt.contourf(nodal_coordinates_k[0, 0], nodal_coordinates_k[1, 0],integrant)

            plt.plot(nodal_coordinates_k[0, 0, :, 0], integrant[:, 0], label='f_rho ')
            # plt.plot(nodal_coordinates_k[0, 0, :, 0], grad_integrant_fnxyz[0, :], label='df_rho/drho')
            # plt.plot(nodal_coordinates_k[0, 0, :, 0], phase_field_k[0, 0, :, 0], label='phase_field_k')

            # segs1 = np.stack((nodal_coordinates[0, 0], nodal_coordinates[1, 0]), axis=2)
            # segs2 = segs1.transpose(1, 0, 2)
        #   ax.legend()
        f_dw_old = topology_optimization.compute_double_well_potential(discretization=discretization_k,
                                                                       phase_field_1nxyz=phase_field_k,
                                                                       eta=eta)

        f_dw = topology_optimization.compute_double_well_potential_NEW(discretization=discretization_k,
                                                                       phase_field_1nxyz=phase_field_k,
                                                                       eta=eta)
        f_dw_Gauss_quad = topology_optimization.compute_double_well_potential_Gauss_quad(
            discretization=discretization_k,
            phase_field_1nxyz=phase_field_k,
            eta=eta)
        of_k = topology_optimization.objective_function_small_strain(
            discretization=discretization,
            actual_stress_ij=homogenized_stress,
            target_stress_ij=target_stress,
            phase_field_1nxyz=phase_field_0,
            eta=1,
            w=1)

        df_dw_drho = topology_optimization.partial_der_of_double_well_potential_wrt_density(
            discretization=discretization_k,
            phase_field_1nxyz=phase_field_k,
            eta=1)

        f_dphase = topology_optimization.compute_gradient_of_phase_field_potential(
            discretization=discretization_k,
            phase_field_1nxyz=phase_field_k,
            eta=1)

        print('f_dw_grid_{}_old =  {} '.format(nb_pixel_x, f_dw_old))
        print('f_dw_grid_{}_NEW =  {} '.format(nb_pixel_x, f_dw))
        print('f_dw_grid_{}_Gquad =  {} '.format(nb_pixel_x, f_dw_Gauss_quad))

        print('objective_{} =  {} '.format(nb_pixel_x, of_k))
        print()
        # print(f_dphase)
        # print(of_k)

    plt.legend()
    plt.show()


def test_d_phase_field_d_rho_integration():
    import matplotlib.pyplot as plt
    l_0 = 0
    l_N = 1
    domain_volume = l_N - l_0
    N = 10
    x_coords = np.linspace(l_0, l_N, N, endpoint=True)
    dx = domain_volume / (N)

    lin_fun = lambda x: 1 * x

    rho_i = lin_fun(x_coords)
    # rho_i=np.random.rand(N)

    rho_interpolator = sc.interpolate.interp1d(x_coords, rho_i)
    ynew = rho_interpolator(x_coords)
    integral_test = (np.sum(rho_interpolator(x_coords)) * dx)

    k = 2
    x_coords_k = np.linspace(l_0, l_N, k * N, endpoint=False)
    dx_k = domain_volume / (k * N)
    integral_test_k = (np.sum(rho_interpolator(x_coords_k)) * dx_k)

    phase_field = lin_fun(x_coords)
    double_well = 16 * (phase_field ** 2) * (1 - phase_field) ** 2

    integral = (np.sum(double_well) / np.prod(double_well.shape)) * domain_volume

    grad_integrant_fnxyz = 1 * (2 * phase_field * (2 * phase_field * phase_field - 3 * phase_field + 1))

    # integral_fnxyz = (integrant_fnxyz / np.prod(integrant_fnxyz.shape)) * discretization.cell.domain_volume

    print()

    print(x_coords)

    print(double_well)
    print(integral)
    print(grad_integrant_fnxyz)

    plt.plot(x_coords, phase_field)
    plt.plot(x_coords, double_well)
    plt.plot(x_coords, grad_integrant_fnxyz)
    plt.show()


def test_phase_field_integration():
    import matplotlib.pyplot as plt
    l_0 = 0
    l_N = 1
    domain_volume = l_N - l_0
    x_coords = np.linspace(l_0, l_N, 50)

    lin_fun = lambda x: 1 * x
    cos_fun = lambda x: 1 * np.cos(x * np.pi / 2)

    phase_field = cos_fun(x_coords)

    double_well = 16 * (phase_field ** 2) * (1 - phase_field) ** 2

    integral = (np.sum(double_well) / np.prod(double_well.shape)) * domain_volume

    grad_integrant_fnxyz = 1 * (2 * phase_field * (2 * phase_field * phase_field - 3 * phase_field + 1))

    # integral_fnxyz = (integrant_fnxyz / np.prod(integrant_fnxyz.shape)) * discretization.cell.domain_volume

    print()

    print(x_coords)

    print(double_well)
    print(integral)
    print(grad_integrant_fnxyz)

    plt.plot(x_coords, phase_field)
    plt.plot(x_coords, double_well)
    plt.plot(x_coords, grad_integrant_fnxyz)
    plt.show()
