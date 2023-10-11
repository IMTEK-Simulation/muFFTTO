import pytest

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import topology_optimization


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
    ([3, 4], 0, [6, 8]),
    ([3, 4], 1, [6, 8])])
def test_finite_difference_check_of_whole_objective_function(discretization_fixture):
    # TODO NOT FINISHED
    # set stress difference to zero
    target_stress = np.array([[1, 0.5], [0.5, 2]])
    macro_gradient = np.array([[1, 0], [0, 1]])
    epsilon = 1e-1
    fd_sensitivity = discretization_fixture.get_scalar_sized_field()

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

    solution, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(500), toler=1e-6)

    # test homogenized stress
    homogenized_stress = discretization_fixture.get_homogenized_stress(material_data_field_i,
                                                                       displacement_field=solution,
                                                                       macro_gradient_field=macro_gradient_field)

    actual_stress_field = np.zeros(discretization_fixture.gradient_size)
    actual_stress_field[..., :] = homogenized_stress[(...,) + (np.newaxis,) * (actual_stress_field.ndim - 2)]

    objective_function = topology_optimization.objective_function_small_strain(discretization_fixture,
                                                                               actual_stress_field,
                                                                               target_stress, phase_field, eta=1, w=1)

    # loop over every single element of phase field
    for x in np.arange(discretization_fixture.nb_of_pixels[0]):
        for y in np.arange(discretization_fixture.nb_of_pixels[1]):
            # set phase_field to ones
            phase_field = discretization_fixture.get_scalar_sized_field() + 1  # Phase field has  one  value per pixel
            phase_field[0, 0, 2:4, 2:4] = phase_field[0, 0, 2:4, 2:4] / 2  # for
            #
            phase_field[0, 0, x, y] = phase_field[0, 0, x, y] + epsilon

            # Update material data based on current Phase-field
            material_data_field_i = (phase_field) * material_data_field

            ##### solve equilibrium constrain
            # set up system
            rhs = discretization_fixture.get_rhs(material_data_field_i, macro_gradient_field)

            K_fun = lambda x: discretization_fixture.apply_system_matrix(material_data_field_i, x)
            M_fun = lambda x: 1 * x

            solution, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(500), toler=1e-6)

            # test homogenized stress
            homogenized_stress = discretization_fixture.get_homogenized_stress(material_data_field_i,
                                                                               displacement_field=solution,
                                                                               macro_gradient_field=macro_gradient_field)

            actual_stress_field = np.zeros(discretization_fixture.gradient_size)
            actual_stress_field[..., :] = homogenized_stress[(...,) + (np.newaxis,) * (actual_stress_field.ndim - 2)]

            objective_function_perturbed = topology_optimization.objective_function_small_strain(discretization_fixture,
                                                                                                 actual_stress_field,
                                                                                                 target_stress,
                                                                                                 phase_field, eta=1,
                                                                                                 w=1)
            fd_sensitivity[0, 0, x, y] = (objective_function_perturbed - objective_function) / epsilon

    # TODO this just computes the sensitivity using finite difference --- will be used for FD check in the future


# @pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
#     ([3, 4], 0, [5, 8]),
#     ([4, 5], 1, [7, 6])])
# def test_adjoint_sensitivity_(discretization_fixture):

@pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
    ([3, 4], 0, [6, 8]),
    ([3, 4], 1, [6, 8])])
def test_finite_difference_check_of_double_well_potential(discretization_fixture):
    # epsilons = [1e0,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10]:
    epsilons = [1e-4]
    fd_derivative = discretization_fixture.get_scalar_sized_field()

    # compute double-well potential without perturbations
    phase_field = discretization_fixture.get_scalar_sized_field() + 1  # Phase field has  one  value per pixel
    phase_field[0, 0, 2:5, 2:4] = phase_field[0, 0, 2:5, 2:4] / 3  # can be random in this case

    # get analytical partial derivative of the double-well potential with respect to phase-field
    partial_der_of_double_well_potential = topology_optimization.partial_der_of_double_well_potential_wrt_density(
        discretization_fixture,
        phase_field)
    # double_well_potential for a phase field without perturbation
    double_well_potential = topology_optimization.compute_double_well_potential(discretization_fixture,
                                                                                phase_field, eta=1)

    error_fd_vs_analytical = []
    for epsilon in epsilons:
        # loop over every single element of phase field
        for x in np.arange(discretization_fixture.nb_of_pixels[0]):
            for y in np.arange(discretization_fixture.nb_of_pixels[1]):
                # set phase_field to ones
                phase_field = discretization_fixture.get_scalar_sized_field() + 1  # Phase field has  one  value per pixel
                phase_field[0, 0, 2:5, 2:4] = phase_field[0, 0, 2:5, 2:4] / 3  # can be random in this case
                #
                phase_field[0, 0, x, y] = phase_field[0, 0, x, y] + epsilon

                double_well_potential_perturbed = topology_optimization.compute_double_well_potential(
                    discretization_fixture,
                    phase_field, eta=1)
                fd_derivative[0, 0, x, y] = (double_well_potential_perturbed - double_well_potential) / epsilon
                # print(fd_derivative[0, 0])

        error_fd_vs_analytical.append(
            np.linalg.norm((fd_derivative - partial_der_of_double_well_potential)[0, 0], 'fro'))
    assert error_fd_vs_analytical[0] < 1e-3, (
        "Finite difference derivative do not corresponds to the analytical expression "
        "for partial derivative of double well potential ")


@pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
    ([3, 4], 0, [6, 8]),
    ([2, 5], 0, [12, 7]),
    ([3, 4], 1, [6, 8]),
    ([2, 5], 1, [12, 7])])
def test_finite_difference_check_of_gradient_of_phase_field_potential(discretization_fixture):
    # epsilons = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    epsilons = [1e-5]
    fd_derivative = discretization_fixture.get_scalar_sized_field()
    nodal_coordinates = discretization_fixture.get_nodal_points_coordinates()

    # Compute phase field gradient potential without perturbations
    phase_field = discretization_fixture.get_scalar_sized_field() + 1  # Phase field has  one  value per pixel
    phase_field[0, 0, 2:5, 2:4] = phase_field[0, 0, 2:5, 2:4] / 3  # can be random in this case

    # create a linear phase field
    # u_fun_3 = lambda x,y: 0*x + 3 * y
    # phase_field[0, 0, :, :] = u_fun_3(nodal_coordinates[0, 0, :, :],
    #                                   nodal_coordinates[1, 0, :, :])

    # phase field gradient potential for a phase field without perturbation
    f_rho_grad_potential = topology_optimization.compute_gradient_of_phase_field_potential(discretization_fixture,
                                                                                           phase_field, eta=1)

    # get analytical partial derivative of phase field gradient potential for a phase field with respect to phase-field
    partial_der_of_f_rho_grad_potential = topology_optimization.partial_derivative_of_gradient_of_phase_field_potential(
        discretization_fixture,
        phase_field)

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
                phase_field[0, 0, x, y] = phase_field[0, 0, x, y] + epsilon

                f_rho_grad_potential_perturbed = topology_optimization.compute_gradient_of_phase_field_potential(
                    discretization_fixture,
                    phase_field, eta=1)
                fd_derivative[0, 0, x, y] = (f_rho_grad_potential_perturbed - f_rho_grad_potential) / epsilon

        error_fd_vs_analytical.append(
            np.linalg.norm((fd_derivative - partial_der_of_f_rho_grad_potential)[0, 0], 'fro'))

    print(error_fd_vs_analytical)

    assert error_fd_vs_analytical[0] < 1e-3, (
        "Finite difference derivative do not corresponds to the analytical expression "
        "for partial derivative of gradient of phase-field potential ")


@pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
    ([3, 4], 0, [6, 8]),
    ([2, 5], 0, [12, 7]),
    ([3, 4], 1, [6, 8]),
    ([2, 5], 1, [12, 7])])
def test_of_stress_equivalence_potential(discretization_fixture, plot=False):
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
    # TODO finite_difference_check_of_stress_equivalence_potential DOES NOT  work!
    epsilons = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    # epsilons = [1e-4]
    fd_derivative = discretization_fixture.get_scalar_sized_field()

    target_stress = np.array([[1, 0.3], [0.3, 2]])
    macro_gradient = np.array([[0.01, 0], [0, 0.01]])

    # Compute phase field gradient potential without perturbations
    phase_field = discretization_fixture.get_scalar_sized_field() + 1  # Phase field has  one  value per pixel
    phase_field[0, 0, 2:5, 2:4] = phase_field[0, 0, 2:5, 2:4] / 3  # can be random in this case
    # phase_field[0, 0, 0, 0] = phase_field[0, 0, 0, 0] / 2  # can be random in this case

    # phase_field = np.random.rand(*phase_field.shape)  # set random distribution
    # create a linear phase field
    # identity tensor                                               [single tensor]
    i = np.eye(discretization_fixture.domain_dimension)
    # identity tensors                                            [grid of tensors]
    I = np.einsum('ij,xy', i, np.ones(discretization_fixture.nb_of_pixels))
    I4 = np.einsum('ijkl,qxy->ijklqxy', np.einsum('il,jk', i, i),
                   np.ones(np.array(
                       [discretization_fixture.nb_quad_points_per_pixel, *discretization_fixture.nb_of_pixels])))
    I4rt = np.einsum('ijkl,qxy->ijklqxy', np.einsum('ik,jl', i, i),
                     np.ones(np.array(
                         [discretization_fixture.nb_quad_points_per_pixel, *discretization_fixture.nb_of_pixels])))
    I4s = (I4 + I4rt) / 2.
    # create material data field
    # K_0, G_0 = domain.get_bulk_and_shear_modulus(E=1, poison=0.2)
    #
    # material_C_0 = domain.get_elastic_material_tensor(dim=discretization_fixture.domain_dimension,
    #                                                   K=K_0,
    #                                                   mu=G_0,
    #                                                   kind='linear')
    #
    # material_data_field_C_0 = np.einsum('ijkl,qxy->ijklqxy', material_C_0,
    #                                     np.ones(np.array([discretization_fixture.nb_quad_points_per_pixel,
    #                                                       *discretization_fixture.nb_of_pixels])))

    material_data_field_C_0 = I4s
    # apply material distribution
    material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * phase_field[0, 0]

    macro_gradient_field = discretization_fixture.get_macro_gradient_field(macro_gradient)
    #
    rhs = discretization_fixture.get_rhs(material_data_field_C_0_rho, macro_gradient_field)

    K_fun = lambda x: discretization_fixture.apply_system_matrix(material_data_field_C_0_rho, x,
                                                                 formulation='small_strain')
    M_fun = lambda x: 1 * x

    displacement_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(500), toler=1e-6)

    actual_stress = discretization_fixture.get_homogenized_stress(material_data_field_C_0_rho, displacement_field,
                                                                  macro_gradient_field, formulation='small_strain')

    f_sigma_diff_potential = topology_optimization.objective_function_stress_equivalence(discretization_fixture,
                                                                                         actual_stress,
                                                                                         target_stress)

    # phase field gradient potential for a phase field without perturbation
    f_rho_grad_potential_analytical = topology_optimization.partial_derivative_of_objective_function_stress_equivalence(
        discretization_fixture,
        phase_field,
        actual_stress,
        target_stress,
        material_data_field_C_0,
        displacement_field,
        macro_gradient_field)

    # get analytical partial derivative of phase field gradient potential for a phase field with respect to phase-field

    error_fd_vs_analytical = []
    for epsilon in epsilons:
        # loop over every single element of phase field
        for x in np.arange(discretization_fixture.nb_of_pixels[0]):
            for y in np.arange(discretization_fixture.nb_of_pixels[1]):
                # set phase_field to ones
                # phase_field_xy = phase_field.copy()  # Phase field has  one  value per pixel
                phase_field_xy = discretization_fixture.get_scalar_sized_field() + 1  # Phase field has  one  value per pixel
                # phase_field_xy[0, 0, 0, 0] = phase_field_xy[0, 0, 0, 0] / 2  # can be random in this case
                phase_field_xy[0, 0, 2:5, 2:4] = phase_field_xy[0, 0, 2:5, 2:4] / 3  # can be random in this case
                phase_field_xy[0, 0, x, y] = phase_field_xy[0, 0, x, y] - epsilon

                # apply material distribution
                material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * phase_field_xy[0, 0]

                rhs = discretization_fixture.get_rhs(material_data_field_C_0_rho, macro_gradient_field)

                K_fun = lambda x: discretization_fixture.apply_system_matrix(material_data_field_C_0_rho, x,
                                                                             formulation='small_strain')
                M_fun = lambda x: 1 * x

                displacement_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(500), toler=1e-6)

                actual_stress = discretization_fixture.get_homogenized_stress(material_data_field_C_0_rho,
                                                                              displacement_field,
                                                                              macro_gradient_field,
                                                                              formulation='small_strain')

                f_sigma_diff_potential_perturbed = topology_optimization.objective_function_stress_equivalence(
                    discretization_fixture,
                    actual_stress,
                    target_stress)

                fd_derivative[0, 0, x, y] = (f_sigma_diff_potential_perturbed - f_sigma_diff_potential) / epsilon

        print(fd_derivative[0, 0, 0, 0])
        print(f_rho_grad_potential_analytical[0, 0])
        error_fd_vs_analytical.append(np.linalg.norm((fd_derivative[0, 0] - f_rho_grad_potential_analytical), 'fro'))

    # print(f_rho_grad_potential_analytical)

    print(error_fd_vs_analytical)

    assert error_fd_vs_analytical[0] < 1e-3, (
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
    # epsilons = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    epsilons = [1e-4]
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

    displacement_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(500), toler=1e-12)

    #
    adjoint_field = np.random.rand(
        *discretization_fixture.get_displacement_sized_field().shape)  # set random adjoint field

    # compute stress field corresponding to equilibrated displacement
    stress_field = discretization_fixture.get_stress_field(material_data_field_C_0, displacement_field,
                                                           macro_gradient_field)

    adjoint_potential = topology_optimization.adjoint_potential(discretization_fixture, stress_field, adjoint_field)

    dg_du_analytical = topology_optimization.partial_derivative_of_adjoint_potential_wrt_displacement(
        discretization_fixture,
        material_data_field_C_0,
        adjoint_field)

    error_fd_vs_analytical = []
    for epsilon in epsilons:
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
                                                                                    stress_field, adjoint_field)

                        fd_derivative[f, n, x, y] = (adjoint_potential_perturbed - adjoint_potential) / epsilon

            error_fd_vs_analytical.append(np.linalg.norm((fd_derivative[f, n] - dg_du_analytical[f, n]), 'fro'))
    # TODO double check this test !!!!!
    #print(fd_derivative[0, 0, 0, 0])
    #print(dg_du_analytical[0, 0])
    print(error_fd_vs_analytical)
