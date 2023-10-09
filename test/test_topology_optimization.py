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

    #epsilons = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
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
        "for partial derivative of double well potential ")
