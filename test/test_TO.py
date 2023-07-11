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
    epsilon = 1e-2
    sensitivity = discretization_fixture.get_scalar_sized_field()
    # loop over every single element of phase field
    for x in np.arange(discretization_fixture.nb_of_pixels[0]):
        for y in np.arange(discretization_fixture.nb_of_pixels[1]):
            # set phase_field to ones
            phase_field = discretization_fixture.get_scalar_sized_field() + 1  # Phase field has  one  value per pixel
            phase_field[0, 0, 2:4, 2:4] = phase_field[0, 0, 2:4, 2:4] / 2
            phase_field[0, 0, x, y] = phase_field[0, 0, x, y] + epsilon

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

            objective_f = topology_optimization.objective_function_small_strain(discretization_fixture,
                                                                                actual_stress_field,
                                                                                target_stress, phase_field, eta=1, w=1)
            sensitivity[0,0,x,y]=objective_f
    #TODO this just computes the sensitivity using finite difference --- will be used for FD check in the future




@pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
    ([3, 4], 0, [5, 8]),
    ([4, 5], 1, [7, 6])])
def test_adjoint_sensitivity_(discretization_fixture):