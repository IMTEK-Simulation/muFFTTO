import pytest

import numpy as np
import scipy as sc
import time

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import topology_optimization
from muFFTTO import microstructure_library


# TODO implement test for bilinear elements and 3D
@pytest.fixture()
def discretization_fixture(domain_size, element_type, nb_pixels):
    problem_type = 'elasticity'
    element_types = ['linear_triangles', ]  # ,'linear_triangles_tilled',  'bilinear_rectangle'

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
    ([1, 2], 0, [4, 5])])
# ,([3.1, 6.4], 0, [7, 6])])
def test_fd_check_of_whole_objective_function(discretization_fixture, plot=True):
    problem_type = 'elasticity'
    discretization_type = 'finite_element'
    element_type = 'linear_triangles'  # discretization_fixture.element_type  # 'bilinear_rectangle'##'linear_triangles' #
    formulation = 'small_strain'
    domain_size = [1, 1]
    number_of_pixels = discretization_fixture.nb_of_pixels

    my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                      problem_type=problem_type)

    discretization = domain.Discretization(cell=my_cell,
                                           nb_of_pixels_global=tuple(number_of_pixels),
                                           discretization_type=discretization_type,
                                           element_type=element_type)

    macro_gradient = np.array([[1.0, .0],
                               [0.0, 0.]])
    print('macro_gradient = \n {}'.format(macro_gradient))

    # create material data of solid phase rho=1
    E_0 = 1
    poison_0 = 0.2

    K_0, G_0 = domain.get_bulk_and_shear_modulus(E=E_0, poison=poison_0)

    elastic_C_0_ijkl = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                          K=K_0,
                                                          mu=G_0,
                                                          kind='linear')

    material_data_field_C_0_ijklqxyz = discretization.get_material_data_size_field_mugrid(name='base_material')

    material_data_field_C_0_ijklqxyz.s = np.einsum('ijkl,qxy->ijklqxy', elastic_C_0_ijkl,
                                                   np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                                     *discretization.nb_of_pixels])))

    stress = np.einsum('ijkl,lk->ij', elastic_C_0_ijkl, macro_gradient)

    # create target material data
    print('init_stress = \n {}'.format(stress))
    # validation metamaterials

    poison_target = 1 / 3  # lambda = -10
    G_target_auxet = (1 / 4) * E_0  # 23   25
    E_target = 2 * G_target_auxet * (1 + poison_target)

    K_targer, G_target = domain.get_bulk_and_shear_modulus(E=E_target, poison=poison_target)

    elastic_C_target_ijkl = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                               K=K_targer,
                                                               mu=G_target,
                                                               kind='linear')
    # # target_stress = np.array([[0.0, 0.05],
    # #                           [0.05, 0.0]])
    target_stress_ij = np.einsum('ijkl,lk->ij', elastic_C_target_ijkl, macro_gradient)
    print('target_stress = \n {}'.format(target_stress_ij))
    # Set up the equilibrium system

    macro_gradient_field_ijqxyz = discretization.get_gradient_size_field(name='macro_gradient_field')
    discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient,
                                                   macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz
                                                   )
    # M_fun = lambda x: 1 * x
    preconditioner_Green = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=elastic_C_0_ijkl)

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
    w = 3  # * E_0  # 1 / 10  # 1e-4 Young modulus of solid
    eta = 0.02

    def my_objective_function(phase_field_1nxyz_flat):
        # print('Objective function:')
        # reshape the field
        phase_field_1nxyz = discretization.get_scalar_field(name='phase_field_in_objective')
        phase_field_1nxyz.s = phase_field_1nxyz_flat.reshape([1, 1, *number_of_pixels])

        # Material data in quadrature points
        phase_field_at_quad_poits_1qxyz = discretization_fixture.get_quad_field_scalar(
            name='phase_field_at_quad_poits_1qxyz')
        discretization_fixture.apply_N_operator_mugrid(phase_field_1nxyz, phase_field_at_quad_poits_1qxyz)
        # Todo: numpy array
        # phase_field_at_quad_points_1qnxyz, N_at_quad_points_qnijk = discretization.evaluate_field_at_quad_points(
        #     nodal_field_fnxyz=phase_field_1nxyz,
        #     quad_field_fqnxyz=None,
        #     quad_points_coords_iq=None)
        #
        material_data_field_C_0_rho_ijklqxyz = discretization.get_material_data_size_field_mugrid(
            name='data_field_in_objective')
        # material_data_field_C_0_rho_ijklqxyz.s = elastic_C_0_ijkl[..., np.newaxis, np.newaxis, np.newaxis] * \
        #                                          np.power(phase_field_at_quad_points_1qnxyz, p)[0, :, 0, ...]
        material_data_field_C_0_rho_ijklqxyz.s = elastic_C_0_ijkl[..., np.newaxis, np.newaxis, np.newaxis] * \
                                    np.power(phase_field_at_quad_poits_1qxyz.s, p)[0, 0, :, ...]

        # Solve mechanical equilibrium constrain
        rhs_inxyz = discretization.get_unknown_size_field(name='rhs_field')
        discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                                      macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
                                      rhs_inxyz=rhs_inxyz)

        def K_fun(x, Ax):
            discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0_rho_ijklqxyz,
                                                      input_field_inxyz=x,
                                                      output_field_inxyz=Ax,
                                                      formulation='small_strain')

        # K_diag_alg = discretization.get_preconditioner_Jacoby_fast(
        #     material_data_field_ijklqxyz=material_data_field_C_0_rho)
        # M_fun = lambda x: K_diag_alg * discretization.apply_preconditioner_NEW(
        #     preconditioner_Fourier_fnfnqks=preconditioner,
        #     nodal_field_fnxyz=K_diag_alg * x)
        displacement_field = discretization.get_unknown_size_field(name='displacement_field_')
        displacement_field.s.fill(0)

        solvers.conjugate_gradients_mugrid(
            comm=discretization.fft.communicator,
            fc=discretization.field_collection,
            hessp=K_fun,  # linear operator
            b=rhs_inxyz,
            x=displacement_field,
            P=M_fun,
            tol=1e-5,
            maxiter=1000,
            # callback=callback,
            # norm_metric=res_norm
        )
        # displacement_field.s, norms = solvers.PCG(Afun=K_fun, B=rhs_inxyz,
        #                                           x0=None, P=M_fun,
        #                                           steps=int(1500),
        #                                           toler=1e-15,
        #                                           norm_type='rr_rel'
        #                                           )

        # compute homogenized stress field corresponding t
        homogenized_stress = discretization.get_homogenized_stress_mugrid(
            material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
            displacement_field_inxyz=displacement_field,
            macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
            formulation='small_strain')
        # print('homogenized stress = \n'          ' {} '.format(homogenized_stress))

        f_sigma, f_rho = topology_optimization.objective_function_small_strain_testing(
            discretization=discretization,
            actual_stress_ij=homogenized_stress,
            target_stress_ij=target_stress_ij,
            phase_field_1nxyz=phase_field_1nxyz,
            eta=eta,
            w=w)

        # print('objective_function= \n'' {} '.format(objective_function))
        objective_function = w * f_sigma + f_rho

        # print('Sensitivity_analytical')
        sensitivity_analytical, sensitivity_parts = topology_optimization.sensitivity_with_adjoint_problem_FE_NEW(
            discretization=discretization,
            base_material_data_ijkl=elastic_C_0_ijkl,
            displacement_field_inxyz=displacement_field,
            macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
            phase_field_1nxyz=phase_field_1nxyz,
            target_stress_ij=target_stress_ij,
            actual_stress_ij=homogenized_stress,
            preconditioner_fun=M_fun,
            system_matrix_fun=K_fun,
            formulation='small_strain',
            p=p,
            eta=eta,
            weight=w)

        objective_function += sensitivity_parts['adjoint_energy']
        # print(f'objective_function= {objective_function}')
        print('adjoint_energy={}'.format(sensitivity_parts))

        return objective_function, f_sigma, f_rho, sensitivity_analytical, sensitivity_parts

    np.random.seed(1)
    phase_field_0 = discretization.get_scalar_field(name='phase_field_0')
    phase_field_0.s = np.random.rand(*phase_field_0.s.shape) ** 1
    phase_field_0.s +=5
    # save a copy of the original phase field
    phase_field_0_fixed = discretization.get_scalar_field(name='phase_field_0_fixed')

    phase_field_0_fixed.s = np.copy(phase_field_0.s)
    # flatten the array --- just nick
    # phase_field_0_flat = phase_field_0.s.ravel()  # TODO: check if this is copy or not
    analytical_sensitivity, sensitivity_parts = my_objective_function(phase_field_0.s.ravel())[-2:]
    # analitical_sensitivity = analitical_sensitivity.reshape([1, 1, *number_of_pixels])
    print(sensitivity_parts)

    epsilons = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]  # 1e6, 1e5, 1e4, 1e3, 1e2, 1e1,

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
                phase_field = np.copy(phase_field_0_fixed.s)
                #
                phase_field[0, 0, x, y] = phase_field[0, 0, x, y] + epsilon / fd_scheme

                of_plus_eps, f_sigma_plus_eps, f_rho_plus_eps, _, _ = my_objective_function(phase_field.ravel())

                phase_field[0, 0, x, y] = phase_field[0, 0, x, y] - epsilon
                # phase_field_0 = phase_field.reshape(-1)

                of_minu_eps, f_sigma_minu_eps, f_rho_minu_eps, _, _ = my_objective_function(phase_field.ravel())

                fd_sensitivity.s[0, 0, x, y] = (of_plus_eps - of_minu_eps) / (epsilon)
                fd_sensitivity_drho_dro.s[0, 0, x, y] = (f_rho_plus_eps - f_rho_minu_eps) / (epsilon)
                fd_sensitivity_dsigma_dro.s[0, 0, x, y] = (f_sigma_plus_eps - f_sigma_minu_eps) / (epsilon)

        error_fd_vs_analytical.append(
            np.linalg.norm((fd_sensitivity.s - analytical_sensitivity)[0, 0], 'fro'))
        error_fd_vs_analytical_max.append(
            np.max((fd_sensitivity.s - analytical_sensitivity)[0, 0]))
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
    quad_fit_of_error = np.multiply(error_fd_vs_analytical[6], np.asarray(epsilons) ** 2)
    lin_fit_of_error = np.multiply(error_fd_vs_analytical[6], np.asarray(epsilons) ** 1)
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.loglog(epsilons, error_fd_vs_analytical,
                   label=r' error_fd_vs_analytical'.format())
        plt.loglog(epsilons, error_fd_vs_analytical_max,
                   label=r' error_fd_vs_analytical_max'.format())
        # plt.loglog(epsilons, norm_fd_sensitivity - np.linalg.norm(analytical_sensitivity[0, 0], 'fro'),
        #            label=r' error_fd_vs_analytical'.format())
        plt.loglog(epsilons, quad_fit_of_error,
                   label=r' quad_fit_of_error'.format())
        plt.loglog(epsilons, lin_fit_of_error,
                   label=r' lin_fit_of_error'.format())
        plt.legend(loc='best')
        plt.ylim([1e-10, 1e6])
        #   ax.legend()
        # assert error_fd_vs_analytical[-1] < epsilon * 1e2, (
        #   "Finite difference derivative do not corresponds to the analytical expression "
        #   "for partial derivative of double well potential ")
        plt.show()


@pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
    ([1, 1], 0, [6, 6]), ([3, 1.6], 0, [12, 13])
])
def test_fd_check_of_whole_objective_function_stress_equivalence(discretization_fixture, plot=True):
    problem_type = 'elasticity'
    discretization_type = 'finite_element'
    element_type = discretization_fixture.element_type  # 'bilinear_rectangle'##'linear_triangles' #
    formulation = 'small_strain'
    domain_size = [1, 1]
    number_of_pixels = discretization_fixture.nb_of_pixels

    my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                      problem_type=problem_type)

    discretization = domain.Discretization(cell=my_cell,
                                           nb_of_pixels_global=number_of_pixels,
                                           discretization_type=discretization_type,
                                           element_type=element_type)

    macro_gradient = np.array([[1.0, 0.0],
                               [0.0, 0.0]])

    print('macro_gradient = \n {}'.format(macro_gradient))

    # create material data of solid phase rho=1
    E_0 = 1
    poison_0 = 0.2

    K_0, G_0 = domain.get_bulk_and_shear_modulus(E=E_0, poison=poison_0)

    elastic_C_0 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                     K=K_0,
                                                     mu=G_0,
                                                     kind='linear')
    material_data_field_C_0 = discretization.get_material_data_size_field_mugrid(name='base_material')

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
    # # target_stress = np.array([[0.0, 0.05],
    # #                           [0.05, 0.0]])
    target_stress = np.einsum('ijkl,lk->ij', elastic_C_target, macro_gradient)
    # target_stress = np.array([[1, 0.], [0., 2]])

    # Set up the equilibrium system
    macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
    discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient,
                                                   macro_gradient_field_ijqxyz=macro_gradient_field)

    # M_fun = lambda x: 1 * x
    preconditioner = discretization_fixture.get_preconditioner_Green_mugrid(reference_material_data_ijkl=elastic_C_0)

    def M_fun(x, Px):
        discretization_fixture.fft.communicate_ghosts(x)
        discretization_fixture.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                                           input_nodal_field_fnxyz=x,
                                                           output_nodal_field_fnxyz=Px)

    p = 2
    w = 2  # * E_0  # 1 / 10  # 1e-4 Young modulus of solid
    eta = 0.01

    def my_objective_function_energy(phase_field_1nxyz_flat):
        # print('Objective function:')
        # reshape the field
        phase_field_1nxyz = discretization.get_scalar_field(name='my_objective_function_energy')
        phase_field_1nxyz.s = phase_field_1nxyz_flat.reshape([1, 1, *number_of_pixels])

        # objective function phase field terms
        f_phase_field = topology_optimization.objective_function_phase_field(discretization=discretization,
                                                                             phase_field_1nxyz=phase_field_1nxyz,
                                                                             eta=eta,
                                                                             double_well_depth=1)
        #  sensitivity phase field terms
        s_phase_field = topology_optimization.sensitivity_phase_field_term_FE_NEW(discretization=discretization,
                                                                                  base_material_data_ijkl=elastic_C_0,
                                                                                  phase_field_1nxyz=phase_field_1nxyz,
                                                                                  p=p,
                                                                                  eta=eta,
                                                                                  double_well_depth=1)

        # Material data in quadrature points
        phase_field_at_quad_poits_1qxyz = discretization_fixture.get_quad_field_scalar(
            name='phase_field_at_quad_poits_1qxyz')
        discretization_fixture.apply_N_operator_mugrid(phase_field_1nxyz, phase_field_at_quad_poits_1qxyz)

        material_data_field_C_0.s = elastic_C_0[..., np.newaxis, np.newaxis, np.newaxis] * \
                                    np.power(phase_field_at_quad_poits_1qxyz.s, p)[0, 0, :, ...]

        # Solve mechanical equilibrium constrain
        rhs = discretization.get_unknown_size_field(name='rhs_field_at_load_case')
        rhs.s.fill(0)
        discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0,
                                      macro_gradient_field_ijqxyz=macro_gradient_field,
                                      rhs_inxyz=rhs)

        def K_fun(x, Ax):
            discretization_fixture.fft.communicate_ghosts(x)
            discretization_fixture.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0,
                                                              input_field_inxyz=x,
                                                              output_field_inxyz=Ax,
                                                              formulation='small_strain')

        # K_diag_alg = discretization.get_preconditioner_Jacoby_fast(
        #     material_data_field_ijklqxyz=material_data_field_C_0_rho)
        # M_fun = lambda x: K_diag_alg * discretization.apply_preconditioner_NEW(
        #     preconditioner_Fourier_fnfnqks=preconditioner,
        #     nodal_field_fnxyz=K_diag_alg * x)
        displacement_field = discretization.get_unknown_size_field(name=f'displacement_field_load_case')
        displacement_field.s.fill(0)
        # displacement_field.s, norms = solvers.PCG(K_fun, rhs.s, x0=None, P=M_fun, steps=int(1500), toler=1e-14)
        solvers.conjugate_gradients_mugrid(
            comm=discretization_fixture.fft.communicator,
            fc=discretization_fixture.field_collection,
            hessp=K_fun,  # linear operator
            b=rhs,
            x=displacement_field,
            P=M_fun,
            tol=1e-10,
            maxiter=50000,
        )
        # compute homogenized stress field corresponding t
        homogenized_stress = discretization.get_homogenized_stress_mugrid(
            material_data_field_ijklqxyz=material_data_field_C_0,
            displacement_field_inxyz=displacement_field,
            macro_gradient_field_ijqxyz=macro_gradient_field,
            formulation='small_strain')
        # print('homogenized stress = \n'          ' {} '.format(homogenized_stress))
        # strain_fluctuation_ijqxyz = discretization.apply_gradient_operator_symmetrized(
        #     displacement_field)
        # actual_strain_ijqxyz = macro_gradient_field + strain_fluctuation_ijqxyz
        f_sigma = topology_optimization.compute_stress_equivalence_potential(
            actual_stress_ij=homogenized_stress,
            target_stress_ij=target_stress)

        print('f_sigma= \n'' {} '.format(f_sigma))

        adjoint_field_inxyz = discretization.get_unknown_size_field(name=f'adjoint_field_load_case')
        adjoint_field_inxyz.s.fill(0)
        sensitivity_analytical, adjoint_field_inxyz, adjoint_energies = topology_optimization.sensitivity_stress_and_adjoint_FE_NEW(
            discretization=discretization,
            base_material_data_ijkl=elastic_C_0,
            displacement_field_inxyz=displacement_field,
            adjoint_field_inxyz=adjoint_field_inxyz,
            macro_gradient_field_ijqxyz=macro_gradient_field,
            phase_field_1nxyz=phase_field_1nxyz,
            target_stress_ij=target_stress,
            actual_stress_ij=homogenized_stress,
            preconditioner_fun=M_fun,
            system_matrix_fun=K_fun,
            formulation='small_strain',
            p=p,
            weight=w,
            disp=False)

        objective_function = w * f_sigma + f_phase_field + adjoint_energies
        print(f'objective_function= {objective_function}')
        # print('adjoint_energy={}'.format(sensitivity_parts['adjoint_energy']) )

        return objective_function, f_sigma, f_phase_field, sensitivity_analytical + s_phase_field

    np.random.seed(1)
    phase_field_0 = discretization.get_scalar_field(name='phase_field_0')
    phase_field_0.s = np.random.rand(*phase_field_0.s.shape) ** 1
    phase_field_0.s[phase_field_0.s < 1e-3] = 1e-3
    # phase_field_0[0,0] =    discretization.fft.icoords[0]
    phase_field_00 = discretization.get_scalar_field(name='phase_field_00')
    phase_field_00.s = np.copy(phase_field_0.s)

    analytical_sensitivity = my_objective_function_energy(phase_field_0.s.ravel())[-1]
    # analitical_sensitivity = analitical_sensitivity.reshape([1, 1, *number_of_pixels])
    # print(sensitivity_parts)

    epsilons = [ 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]  #
    fd_sensitivity = discretization_fixture.get_scalar_field(name='fd_sensitivity')
    fd_sensitivity_drho_dro = discretization_fixture.get_scalar_field(name='fd_sensitivity_drho_dro')
    fd_sensitivity_dsigma_dro = discretization_fixture.get_scalar_field(name='fd_sensitivity_dsigma_dro')

    error_fd_vs_analytical = []
    error_fd_vs_analytical_max = []
    norm_fd_sensitivity_dsigma_dro = []
    norm_fd_sensitivity_df_dro = []
    norm_fd_sensitivity = []

    phase_field_perturbed = discretization_fixture.get_scalar_field(
        name='phase_field_perturbed')
    for epsilon in epsilons:
        # loop over every single element of phase field
        for x in np.arange(discretization_fixture.nb_of_pixels[0]):
            for y in np.arange(discretization_fixture.nb_of_pixels[1]):
                phase_field_perturbed.s.fill(0)
                phase_field_perturbed.s = np.copy(phase_field_00.s)
                # set phase_field to ones
                #
                phase_field_perturbed.s[0, 0, x, y] = phase_field_perturbed.s[0, 0, x, y] + epsilon / 2
                of_plus_eps, f_sigma_plus_eps, f_rho_plus_eps, _ = my_objective_function_energy(
                    phase_field_perturbed.s.reshape(-1))

                phase_field_perturbed.s[0, 0, x, y] = phase_field_perturbed.s[0, 0, x, y] - epsilon

                of_minu_eps, f_sigma_minu_eps, f_rho_minu_eps, _ = my_objective_function_energy(
                    phase_field_perturbed.s.reshape(-1))

                fd_sensitivity.s[0, 0, x, y] = (of_plus_eps - of_minu_eps) / epsilon
                fd_sensitivity_drho_dro.s[0, 0, x, y] = (f_rho_plus_eps - f_rho_minu_eps) / epsilon
                fd_sensitivity_dsigma_dro.s[0, 0, x, y] = (f_sigma_plus_eps - f_sigma_minu_eps) / epsilon

        error_fd_vs_analytical.append(
            np.linalg.norm((fd_sensitivity.s - analytical_sensitivity)[0, 0], 'fro'))
        error_fd_vs_analytical_max.append(
            np.max((fd_sensitivity.s - analytical_sensitivity)[0, 0]))
        norm_fd_sensitivity.append(
            np.linalg.norm(fd_sensitivity.s[0, 0], 'fro'))
        norm_fd_sensitivity_df_dro.append(
            np.linalg.norm(fd_sensitivity_drho_dro.s[0, 0], 'fro'))
        norm_fd_sensitivity_dsigma_dro.append(
            np.linalg.norm(fd_sensitivity_dsigma_dro.s[0, 0], 'fro'))
    print(np.linalg.norm(analytical_sensitivity[0, 0], 'fro'))
    print(error_fd_vs_analytical)
    print(norm_fd_sensitivity)
    print(norm_fd_sensitivity_df_dro)
    print(norm_fd_sensitivity_dsigma_dro)
    # print(error_fd_vs_analytical)
    quad_fit_of_error = np.multiply(error_fd_vs_analytical[6], (np.asarray(epsilons) / 2) ** 2)
    lin_fit_of_error = np.multiply(error_fd_vs_analytical[6], np.asarray(epsilons) ** 1)
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.loglog(epsilons, error_fd_vs_analytical, marker='x',
                   label=r' error_fd_vs_analytical'.format())
        plt.loglog(epsilons, error_fd_vs_analytical_max, marker='>',
                   label=r' error_fd_vs_analytical_max'.format())
        # plt.loglog(epsilons, norm_fd_sensitivity - np.linalg.norm(analytical_sensitivity[0, 0], 'fro'),
        #            label=r' error_fd_vs_analytical'.format())
        plt.loglog(np.asarray(epsilons) / 2, quad_fit_of_error, linestyle='--',
                   label=r' quad_fit_of_error'.format())
        plt.loglog(epsilons, lin_fit_of_error, linestyle='--',
                   label=r' lin_fit_of_error'.format())
        plt.legend(loc='best')
        plt.ylim([1e-10, 1e6])
        plt.xlabel('epsilon - size of Finite Difference step')
        plt.ylabel('Error')
        plt.show()


@pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
    ([1, 1], 0, [6, 6])])
def test_fd_check_of_whole_objective_function_energy_equivalence(discretization_fixture, plot=True):
    problem_type = 'elasticity'
    discretization_type = 'finite_element'
    element_type = discretization_fixture.element_type  # 'bilinear_rectangle'##'linear_triangles' #
    formulation = 'small_strain'
    domain_size = [1, 1]
    number_of_pixels = discretization_fixture.nb_of_pixels

    my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                      problem_type=problem_type)

    discretization = domain.Discretization(cell=my_cell,
                                           nb_of_pixels_global=number_of_pixels,
                                           discretization_type=discretization_type,
                                           element_type=element_type)

    macro_gradient = np.array([[1.0, 0.0],
                               [0.0, 0.0]])
    left_macro_gradient = np.array([[0.0, .0],
                                    [.0, 1.0]])
    print('macro_gradient = \n {}'.format(macro_gradient))

    # create material data of solid phase rho=1
    E_0 = 1
    poison_0 = 0.2

    K_0, G_0 = domain.get_bulk_and_shear_modulus(E=E_0, poison=poison_0)

    elastic_C_0 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                     K=K_0,
                                                     mu=G_0,
                                                     kind='linear')
    material_data_field_C_0 = discretization.get_material_data_size_field(name='base_material')

    # material_data_field_C_0 = np.einsum('ijkl,qxy->ijklqxy', elastic_C_0,
    #                                     np.ones(np.array([discretization.nb_quad_points_per_pixel,
    #                                                       *discretization.nb_of_pixels])))

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
    # # target_stress = np.array([[0.0, 0.05],
    # #                           [0.05, 0.0]])
    target_stress = np.einsum('ijkl,lk->ij', elastic_C_target, macro_gradient)
    # target_stress = np.array([[1, 0.], [0., 2]])

    target_energy = np.einsum('ij,ijkl,lk->', left_macro_gradient, elastic_C_target,
                              macro_gradient)
    print('target_stress = \n {}'.format(target_stress))
    # Set up the equilibrium system
    macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
    macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient_ij=macro_gradient,
                                                                   macro_gradient_field_ijqxyz=macro_gradient_field
                                                                   )

    # M_fun = lambda x: 1 * x
    preconditioner = discretization.get_preconditioner_Green_fast(
        reference_material_data_ijkl=elastic_C_0)
    M_fun = lambda x: discretization.apply_preconditioner_NEW(preconditioner_Fourier_fnfnqks=preconditioner,
                                                              nodal_field_fnxyz=x)

    p = 2
    w = 2  # * E_0  # 1 / 10  # 1e-4 Young modulus of solid
    eta = 0.01

    def my_objective_function_energy(phase_field_1nxyz_flat):
        # print('Objective function:')
        # reshape the field
        phase_field_1nxyz = discretization.get_scalar_field(name='my_objective_function_energy')
        phase_field_1nxyz.s = phase_field_1nxyz_flat.reshape([1, 1, *number_of_pixels])

        # objective function phase field terms
        f_phase_field = topology_optimization.objective_function_phase_field(discretization=discretization,
                                                                             phase_field_1nxyz=phase_field_1nxyz,
                                                                             eta=eta,
                                                                             double_well_depth=1)
        #  sensitivity phase field terms
        s_phase_field = topology_optimization.sensitivity_phase_field_term_FE_NEW(discretization=discretization,
                                                                                  base_material_data_ijkl=elastic_C_0,
                                                                                  phase_field_1nxyz=phase_field_1nxyz,
                                                                                  p=p,
                                                                                  eta=eta,
                                                                                  double_well_depth=1)

        # Material data in quadrature points
        phase_field_at_quad_points_1qnxyz, N_at_quad_points_qnijk = discretization.evaluate_field_at_quad_points(
            nodal_field_fnxyz=phase_field_1nxyz,
            quad_field_fqnxyz=None,
            quad_points_coords_iq=None)

        # material_data_field_C_0_rho = material_data_field_C_0[..., :, :, :] * np.power(
        #     phase_field_at_quad_points_1qnxyz, p)[0, :, 0, ...]

        material_data_field_C_0.s = elastic_C_0[..., np.newaxis, np.newaxis, np.newaxis] * \
                                    np.power(phase_field_at_quad_points_1qnxyz, p)[0, :, 0, ...]

        # Solve mechanical equilibrium constrain
        # rhs = discretization.get_rhs(material_data_field_C_0_rho, macro_gradient_field)

        rhs = discretization.get_unknown_size_field(name='rhs_field_at_load_case')
        rhs.s.fill(0)
        rhs = discretization.get_rhs(
            material_data_field_ijklqxyz=material_data_field_C_0,
            macro_gradient_field_ijqxyz=macro_gradient_field,
            rhs_inxyz=rhs)

        K_fun = lambda x: discretization.apply_system_matrix(material_data_field=material_data_field_C_0,
                                                             displacement_field=x,
                                                             formulation='small_strain')
        # K_diag_alg = discretization.get_preconditioner_Jacoby_fast(
        #     material_data_field_ijklqxyz=material_data_field_C_0_rho)
        # M_fun = lambda x: K_diag_alg * discretization.apply_preconditioner_NEW(
        #     preconditioner_Fourier_fnfnqks=preconditioner,
        #     nodal_field_fnxyz=K_diag_alg * x)
        displacement_field = discretization.get_unknown_size_field(name=f'displacement_field_load_case')
        displacement_field.s.fill(0)
        displacement_field.s, norms = solvers.PCG(K_fun, rhs.s, x0=None, P=M_fun, steps=int(1500), toler=1e-14)

        # compute homogenized stress field corresponding t
        homogenized_stress = discretization.get_homogenized_stress(
            material_data_field_ijklqxyz=material_data_field_C_0,
            displacement_field_inxyz=displacement_field,
            macro_gradient_field_ijqxyz=macro_gradient_field,
            formulation='small_strain')
        # print('homogenized stress = \n'          ' {} '.format(homogenized_stress))
        # strain_fluctuation_ijqxyz = discretization.apply_gradient_operator_symmetrized(
        #     displacement_field)
        # actual_strain_ijqxyz = macro_gradient_field + strain_fluctuation_ijqxyz

        f_sigma = (topology_optimization.compute_elastic_energy_equivalence_potential(
            discretization=discretization,
            actual_stress_ij=homogenized_stress,
            target_stress_ij=target_stress,
            left_macro_gradient_ij=left_macro_gradient,
            target_energy=target_energy))

        print('f_sigma= \n'' {} '.format(f_sigma))

        adjoint_field_inxyz = discretization.get_unknown_size_field(name=f'adjoint_field_load_case')
        adjoint_field_inxyz.s.fill(0)
        sensitivity_analytical, adjoint_field_inxyz, adjoint_energies = topology_optimization.sensitivity_elastic_energy_and_adjoint_FE_NEW(
            discretization=discretization,
            base_material_data_ijkl=elastic_C_0,
            displacement_field_inxyz=displacement_field,
            adjoint_field_inxyz=adjoint_field_inxyz,
            macro_gradient_field_ijqxyz=macro_gradient_field,
            left_macro_gradient_ij=left_macro_gradient,
            phase_field_1nxyz=phase_field_1nxyz,
            target_stress_ij=target_stress,
            actual_stress_ij=homogenized_stress,
            preconditioner_fun=M_fun,
            system_matrix_fun=K_fun,
            formulation='small_strain',
            target_energy=target_energy,
            p=p,
            weight=w,
            disp=False)

        objective_function = w * f_sigma + f_phase_field + adjoint_energies
        print(f'objective_function= {objective_function}')
        # print('adjoint_energy={}'.format(sensitivity_parts['adjoint_energy']) )

        return objective_function, f_sigma, f_phase_field, sensitivity_analytical + s_phase_field

    np.random.seed(1)
    phase_field_0 = discretization.get_scalar_field(name='phase_field_0')
    phase_field_0.s = np.random.rand(*phase_field_0.s.shape) ** 1
    # phase_field_0[0,0] =    discretization.fft.icoords[0]
    phase_field_00 = np.copy(phase_field_0)

    phase_field_0 = phase_field_0.s.ravel()
    analytical_sensitivity = my_objective_function_energy(phase_field_0)[-1]
    # analitical_sensitivity = analitical_sensitivity.reshape([1, 1, *number_of_pixels])
    # print(sensitivity_parts)

    epsilons = [1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]  #
    fd_sensitivity = discretization_fixture.get_scalar_field(name='fd_sensitivity')
    fd_sensitivity_drho_dro = discretization_fixture.get_scalar_field(name='fd_sensitivity_drho_dro')
    fd_sensitivity_dsigma_dro = discretization_fixture.get_scalar_field(name='fd_sensitivity_dsigma_dro')

    error_fd_vs_analytical = []
    error_fd_vs_analytical_max = []
    norm_fd_sensitivity_dsigma_dro = []
    norm_fd_sensitivity_df_dro = []
    norm_fd_sensitivity = []

    phase_field_perturbed = discretization_fixture.get_scalar_field(
        name='phase_field_perturbed')
    for epsilon in epsilons:
        # loop over every single element of phase field
        for x in np.arange(discretization_fixture.nb_of_pixels[0]):
            for y in np.arange(discretization_fixture.nb_of_pixels[1]):
                phase_field_perturbed.s.fill(0)
                phase_field_perturbed.s = np.copy(phase_field_00)
                # set phase_field to ones
                #
                phase_field_perturbed.s[0, 0, x, y] = phase_field_perturbed.s[0, 0, x, y] + epsilon / 2
                phase_field_0 = phase_field_perturbed.s.reshape(-1)
                of_plus_eps, f_sigma_plus_eps, f_rho_plus_eps, _ = my_objective_function_energy(phase_field_0)

                phase_field_perturbed.s[0, 0, x, y] = phase_field_perturbed.s[0, 0, x, y] - epsilon
                phase_field_0 = phase_field_perturbed.s.reshape(-1)

                of_minu_eps, f_sigma_minu_eps, f_rho_minu_eps, _ = my_objective_function_energy(phase_field_0)

                fd_sensitivity.s[0, 0, x, y] = (of_plus_eps - of_minu_eps) / epsilon
                fd_sensitivity_drho_dro.s[0, 0, x, y] = (f_rho_plus_eps - f_rho_minu_eps) / epsilon
                fd_sensitivity_dsigma_dro.s[0, 0, x, y] = (f_sigma_plus_eps - f_sigma_minu_eps) / epsilon

        error_fd_vs_analytical.append(
            np.linalg.norm((fd_sensitivity.s - analytical_sensitivity)[0, 0], 'fro'))
        error_fd_vs_analytical_max.append(
            np.max((fd_sensitivity.s - analytical_sensitivity)[0, 0]))
        norm_fd_sensitivity.append(
            np.linalg.norm(fd_sensitivity.s[0, 0], 'fro'))
        norm_fd_sensitivity_df_dro.append(
            np.linalg.norm(fd_sensitivity_drho_dro.s[0, 0], 'fro'))
        norm_fd_sensitivity_dsigma_dro.append(
            np.linalg.norm(fd_sensitivity_dsigma_dro.s[0, 0], 'fro'))
    print(np.linalg.norm(analytical_sensitivity[0, 0], 'fro'))
    print(error_fd_vs_analytical)
    print(norm_fd_sensitivity)
    print(norm_fd_sensitivity_df_dro)
    print(norm_fd_sensitivity_dsigma_dro)
    # print(error_fd_vs_analytical)
    quad_fit_of_error = np.multiply(error_fd_vs_analytical[6], (np.asarray(epsilons) / 2) ** 2)
    lin_fit_of_error = np.multiply(error_fd_vs_analytical[6], np.asarray(epsilons) ** 1)
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.loglog(epsilons, error_fd_vs_analytical, marker='x',
                   label=r' error_fd_vs_analytical'.format())
        plt.loglog(epsilons, error_fd_vs_analytical_max, marker='>',
                   label=r' error_fd_vs_analytical_max'.format())
        # plt.loglog(epsilons, norm_fd_sensitivity - np.linalg.norm(analytical_sensitivity[0, 0], 'fro'),
        #            label=r' error_fd_vs_analytical'.format())
        plt.loglog(np.asarray(epsilons) / 2, quad_fit_of_error, linestyle='--',
                   label=r' quad_fit_of_error'.format())
        plt.loglog(epsilons, lin_fit_of_error, linestyle='--',
                   label=r' lin_fit_of_error'.format())
        plt.legend(loc='best')
        plt.ylim([1e-10, 1e6])
        plt.xlabel('epsilon - size of Finite Difference step')
        plt.ylabel('Error')
        plt.show()


# @pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
#     ([3, 4], 0, [5, 8]),
#     ([4, 5], 1, [7, 6])])
# def test_adjoint_sensitivity_(discretization_fixture):

@pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
    ([1, 1], 0, [3, 4]),
    ([2, 5], 0, [11, 8]),
    ([4, 3], 0, [22, 7])])
def test_fd_check_of_double_well_potential(discretization_fixture, plot=True):
    epsilons = [1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11,
                1e-12, 1e-13]  #
    # epsilons = [1e-4]
    fd_derivative_nodal = discretization_fixture.get_scalar_field(name='fd_derivative_nodal')
    fd_derivative_anal = discretization_fixture.get_scalar_field(name='fd_derivative_anal')

    # compute double-well potential without perturbations
    phase_field = discretization_fixture.get_scalar_field(
        name='phase_field')
    # Phase field has  one  value per pixel
    phase_field.s = np.random.rand(*phase_field.s.shape) ** 1

    phase_field_0 = discretization_fixture.get_scalar_field(
        name='phase_field_0')
    phase_field_0.s = np.copy(phase_field.s)

    # get analytical partial derivative of the double-well potential with respect to phase-field
    partial_der_of_double_well_potential_nodal = (
        topology_optimization.partial_der_of_double_well_potential_wrt_density_nodal(
            discretization=discretization_fixture,
            phase_field_1nxyz=phase_field))

    partial_der_of_double_well_potential_analytical = discretization_fixture.get_scalar_field(
        name='partial_der_of_double_well_potential_analytical')
    partial_der_of_double_well_potential_analytical = (
        topology_optimization.partial_der_of_double_well_potential_wrt_density_analytical(
            discretization=discretization_fixture,
            phase_field_1nxyz=phase_field,
            output_1nxyz=partial_der_of_double_well_potential_analytical))

    # we have 2 implementations: nodal and analytical formulation
    error_fd_nodal_vs_analytical = []
    error_fd_nodal_vs_nodal = []
    error_fd_anal_vs_nodal = []
    error_fd_anal_vs_analytical = []

    phase_field_perturbed = discretization_fixture.get_scalar_field(
        name='phase_field_perturbed')
    for epsilon in epsilons:
        # shifted: compute gradient in the middle point between phase_field and phase_field+epsilon
        # loop over every single element of phase field
        for x in np.arange(discretization_fixture.nb_of_pixels[0]):
            for y in np.arange(discretization_fixture.nb_of_pixels[1]):
                # set phase_field to ones
                # phase_field = discretization_fixture.get_scalar_sized_field() + 1  # Phase field has  one  value per pixel
                phase_field_perturbed.s.fill(0)
                phase_field_perturbed.s = np.copy(phase_field_0.s)

                # phase_field[0, 0, 2:5, 2:4] = phase_field[0, 0, 2:5, 2:4] / 3  # can be random in this case
                #
                phase_field_perturbed.s[0, 0, x, y] = phase_field_perturbed.s[0, 0, x, y] + epsilon / 2

                double_well_potential_plus_eps_nodal = topology_optimization.compute_double_well_potential_nodal(
                    discretization=discretization_fixture,
                    phase_field_1nxyz=phase_field_perturbed, eta=1)

                double_well_potential_plus_eps_anal = topology_optimization.compute_double_well_potential_analytical(
                    discretization=discretization_fixture,
                    phase_field_1nxyz=phase_field_perturbed)

                phase_field_perturbed.s[0, 0, x, y] = phase_field_perturbed.s[0, 0, x, y] - epsilon
                # assert error_fd_vs_analytical[-1] < epsilon * 1e2, (
                double_well_potential_minus_eps_nodal = topology_optimization.compute_double_well_potential_nodal(
                    discretization=discretization_fixture,
                    phase_field_1nxyz=phase_field_perturbed, eta=1)

                double_well_potential_minus_eps_anal = topology_optimization.compute_double_well_potential_analytical(
                    discretization=discretization_fixture,
                    phase_field_1nxyz=phase_field_perturbed)

                fd_derivative_nodal.s[0, 0, x, y] = (double_well_potential_plus_eps_nodal
                                                     - double_well_potential_minus_eps_nodal) / epsilon

                fd_derivative_anal.s[0, 0, x, y] = (double_well_potential_plus_eps_anal -
                                                    double_well_potential_minus_eps_anal) / epsilon
        # print('error_fd_NEW_vs_analytical_NEW: {}'.format(fd_derivative_NEW))
        error_fd_nodal_vs_analytical.append(
            np.linalg.norm((fd_derivative_nodal.s - partial_der_of_double_well_potential_analytical.s)[0, 0], 'fro'))
        error_fd_nodal_vs_nodal.append(
            np.linalg.norm((fd_derivative_nodal.s - partial_der_of_double_well_potential_nodal)[0, 0], 'fro'))
        error_fd_anal_vs_nodal.append(
            np.linalg.norm((fd_derivative_anal.s - partial_der_of_double_well_potential_nodal)[0, 0], 'fro'))
        error_fd_anal_vs_analytical.append(
            np.linalg.norm((fd_derivative_anal.s - partial_der_of_double_well_potential_analytical.s)[0, 0], 'fro'))
    print('error_fd_nodal_vs_analytical: {}\n'.format(error_fd_nodal_vs_analytical))
    print('error_fd_nodal_vs_nodal: {}\n'.format(error_fd_nodal_vs_nodal))
    print('error_fd_anal_vs_nodal: {}\n'.format(error_fd_anal_vs_nodal))
    print('error_fd_anal_vs_analytical: {}\n'.format(error_fd_anal_vs_analytical))
    if plot:
        import matplotlib.pyplot as plt
        quad_fit_of_error = np.multiply(error_fd_anal_vs_analytical[6], np.asarray(epsilons) ** 2)
        lin_fit_of_error = np.multiply(error_fd_anal_vs_analytical[6], np.asarray(epsilons) ** 1)
        plt.figure()

        plt.loglog(epsilons, error_fd_nodal_vs_analytical,
                   label=r' error_fd_nodal_vs_analytical'.format())
        plt.loglog(epsilons, error_fd_nodal_vs_nodal,
                   label=r' error_fd_nodal_vs_nodal'.format())
        plt.loglog(epsilons, error_fd_anal_vs_nodal, marker='x',
                   label=r' error_fd_anal_vs_nodal'.format())
        plt.loglog(epsilons, error_fd_anal_vs_analytical,
                   label=r' error_fd_anal_vs_analytical'.format())
        plt.loglog(epsilons, quad_fit_of_error, linestyle='--',
                   label=r' quad_fit_of_error'.format())
        plt.loglog(epsilons, lin_fit_of_error, linestyle='--',
                   label=r' lin_fit_of_error'.format())
        plt.legend(loc='best')
        plt.xlabel('epsilon - size of Finite Difference step')
        plt.ylabel('Error')
        plt.legend(loc='best')
        plt.ylim([1e-16, 1e10])
        plt.xlim([1e-9, 1e6])

        #   ax.legend()

        plt.show()
    # assert error_fd_anal_vs_analytical[-1] < epsilon * 1e2, (
    #     "Finite difference derivative do not corresponds to the analytical expression "
    #     "for partial derivative of double well potential ")


@pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
    ([1, 1], 0, [6, 6]),
    ([2, 5], 0, [12, 7])])
def test_fd_check_of_grad_of_double_well_potential_partial_derivative_wrt_phase_field_FE(discretization_fixture,
                                                                                         plot=True):
    phase_field_1nxyz = discretization_fixture.get_scalar_field(
        name='phase_field')
    phase_field_1nxyz.s = np.random.rand(*phase_field_1nxyz.s.shape) ** 1

    #
    dgradrho_drho_analytical = discretization_fixture.get_scalar_field(
        name='dgradrho_drho_analytical')
    topology_optimization.partial_derivative_of_gradient_of_phase_field_potential(
        discretization=discretization_fixture,
        phase_field_1nxyz=phase_field_1nxyz,
        output_1nxyz=dgradrho_drho_analytical)

    # get analytical partial derivative of phase field gradient potential for a phase field with respect to phase-field
    epsilons = [1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, ]  #
    error_fd_vs_analytical = []
    fd_derivative = discretization_fixture.get_scalar_field(name='fd_derivative')
    phase_field_perturbed = discretization_fixture.get_scalar_field(name='phase_field_perturbed')

    for epsilon in epsilons:
        # loop over every single element of phase field
        for x in np.arange(discretization_fixture.nb_of_pixels[0]):
            for y in np.arange(discretization_fixture.nb_of_pixels[1]):
                phase_field_perturbed.s.fill(0)  # Phase field has  one  value per pixel
                # set phase_field to ones
                phase_field_perturbed.s = phase_field_1nxyz.s.copy()

                phase_field_perturbed.s[0, 0, x, y] = phase_field_perturbed.s[0, 0, x, y] + epsilon / 2
                discretization_fixture.fft.communicate_ghosts(phase_field_perturbed)

                f_dw_plus = topology_optimization.compute_gradient_of_phase_field_potential(
                    discretization=discretization_fixture,
                    phase_field_1nxyz=phase_field_perturbed)

                phase_field_perturbed.s[0, 0, x, y] = phase_field_perturbed.s[0, 0, x, y] - epsilon
                discretization_fixture.fft.communicate_ghosts(phase_field_perturbed)

                f_dw_minus = topology_optimization.compute_gradient_of_phase_field_potential(
                    discretization=discretization_fixture,
                    phase_field_1nxyz=phase_field_perturbed)

                fd_derivative.s[0, 0, x, y] = (f_dw_plus - f_dw_minus) / epsilon

        error_fd_vs_analytical.append(
            np.linalg.norm(fd_derivative.s[0, 0] - dgradrho_drho_analytical.s[0, 0], 'fro'))

        # assert error_fd_vs_analytical[-1] < epsilon * 100, (
        #     "Finite difference derivative do not corresponds to the analytical expression "
        #     "for partial derivative of gradient of phase-field potential ")
    print(error_fd_vs_analytical)
    if plot:
        import matplotlib.pyplot as plt
        quad_fit_of_error = np.multiply(error_fd_vs_analytical[6], np.asarray(epsilons) ** 2)
        lin_fit_of_error = np.multiply(error_fd_vs_analytical[6], np.asarray(epsilons) ** 1)

        plt.figure()
        plt.loglog(epsilons, error_fd_vs_analytical, marker='x',
                   label=r' error_fd_vs_analytical'.format())
        plt.loglog(epsilons, quad_fit_of_error, linestyle='--',
                   label=r' quad_fit_of_error'.format())
        plt.loglog(epsilons, lin_fit_of_error, linestyle='--',
                   label=r' lin_fit_of_error'.format())
        plt.legend(loc='best')
        plt.xlabel('epsilon - size of Finite Difference step')
        plt.ylabel('Error')
        plt.legend(loc='best')
        plt.ylim([1e-16, 1e10])
        plt.xlim([epsilons[-1], epsilons[0]])
        plt.show()


def test_integration_of_double_well_potential(plot=True):
    #  to test errors between different implementation of integration of double well potentials
    problem_type = 'elasticity'
    element_types = ['linear_triangles', 'bilinear_rectangle']
    discretization_type = 'finite_element'
    domain_size = [2, 3]
    double_well_potential_nodal_res = []
    double_well_potential_interpolated_res = []
    double_well_potential_plus_eps_Gauss_quad_res = []
    double_well_potential_plus_eps_anal_res = []
    double_well_potential_plus_eps_anal_fast_res = []

    error_nodal_vs_analytical = []
    error_gauss_vs_analytical = []
    error_nodal_vs_gauss = []
    error_fast_vs_analytical = []
    error_fast_vs_fast2 = []

    norm_nodal = []
    norm_gauss = []
    norm_analytical = []

    Ns = [10, 100, 200, 300, 400, 500, 600, 1024]
    for N in Ns:
        print()

        nb_pixels = 2 * (N,)
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)

        discretization = domain.Discretization(cell=my_cell,
                                               nb_of_pixels_global=nb_pixels,
                                               discretization_type=discretization_type,
                                               element_type='linear_triangles')

        # compute double-well potential without perturbations
        phase_field = discretization.get_scalar_field(name='phase_field')
        phase_field.s += 1
        phase_field.s = np.random.rand(*phase_field.s.shape) ** 1
        #

        phase_field_0 = np.copy(phase_field)

        double_well_potential_nodal = topology_optimization.compute_double_well_potential_nodal(
            discretization,
            phase_field, eta=1)

        # double_well_potential_interpolated = topology_optimization.compute_double_well_potential_interpolated(
        #     discretization, phase_field, eta=1)

        start_time = time.time()
        double_well_potential_plus_eps_Gauss_quad = topology_optimization.compute_double_well_potential_Gauss_quad(
            discretization,
            phase_field)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("double_well_potential_plus_eps_Gauss_quad time: ", elapsed_time)

        start_time = time.time()
        double_well_potential_plus_eps_anal = topology_optimization.compute_double_well_potential_analytical(
            discretization=discretization,
            phase_field_1nxyz=phase_field)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("double_well_potential_plus_eps_anal time: ", elapsed_time)

        print()
        double_well_potential_nodal_res.append(double_well_potential_nodal)
        double_well_potential_plus_eps_Gauss_quad_res.append(double_well_potential_plus_eps_Gauss_quad)
        double_well_potential_plus_eps_anal_res.append(double_well_potential_plus_eps_anal)

        start_time = time.time()
        partial_der_of_double_well_potential = topology_optimization.partial_der_of_double_well_potential_wrt_density_nodal(
            discretization,
            phase_field)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("partial_der_of_double_well_potential time: ", elapsed_time)

        start_time = time.time()

        partial_der_of_double_well_potential_gauss = (
            topology_optimization.partial_der_of_double_well_potential_wrt_density_NEW(
                discretization,
                phase_field))
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("partial_der_of_double_well_potential_gauss time: ", elapsed_time)

        start_time = time.time()
        partial_der_of_double_well_potential_analytical = discretization.get_scalar_field(
            name='partial_der_of_double_well_potential_analytical')
        partial_der_of_double_well_potential_analytical = (
            topology_optimization.partial_der_of_double_well_potential_wrt_density_analytical(
                discretization=discretization,
                phase_field_1nxyz=phase_field,
                output_1nxyz=partial_der_of_double_well_potential_analytical))
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("partial_der_of_double_well_potential_analytical time: ", elapsed_time)

        error_nodal_vs_analytical.append(
            np.linalg.norm(
                (partial_der_of_double_well_potential - partial_der_of_double_well_potential_analytical.s)[0, 0],
                'fro'))
        error_gauss_vs_analytical.append(
            np.linalg.norm(
                (partial_der_of_double_well_potential_gauss - partial_der_of_double_well_potential_analytical.s)[0, 0],
                'fro'))
        error_nodal_vs_gauss.append(
            np.linalg.norm(
                (partial_der_of_double_well_potential - partial_der_of_double_well_potential_gauss)[0, 0], 'fro'))

        norm_nodal.append(np.linalg.norm(partial_der_of_double_well_potential[0, 0], 'fro'))
        norm_gauss.append(np.linalg.norm(partial_der_of_double_well_potential_gauss[0, 0], 'fro'))
        norm_analytical.append(np.linalg.norm(partial_der_of_double_well_potential_analytical.s[0, 0], 'fro'))
        print()

    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.loglog(Ns, double_well_potential_nodal_res,
                   label=r' double_well_potential_nodal_res'.format())
        # plt.plot(Ns, double_well_potential_interpolated_res,
        #         label=r' double_well_potential_interpolated_res'.format())
        plt.loglog(Ns, double_well_potential_plus_eps_Gauss_quad_res, marker='x',
                   label=r' double_well_potential_plus_eps_Gauss_quad_res'.format())
        plt.loglog(Ns, double_well_potential_plus_eps_anal_res,
                   label=r' double_well_potential_plus_eps_anal_res'.format())
        # plt.loglog(Ns, double_well_potential_plus_eps_anal_fast_res,
        #            label=r' double_well_potential_plus_eps_anal_fast_res'.format())

        plt.legend(loc='best')

        plt.figure()
        plt.loglog(Ns, error_nodal_vs_analytical, marker='>',
                   label=r' error_nodal_vs_analytical'.format())
        plt.loglog(Ns, error_nodal_vs_gauss, marker='x',
                   label=r' error_nodal_vs_gauss'.format())
        plt.loglog(Ns, error_gauss_vs_analytical, marker='|',
                   label=r' error_gauss_vs_analytical'.format())
        # plt.loglog(Ns, error_fast_vs_analytical, marker='|',
        #            label=r' error_fast_vs_analytical'.format())
        # plt.loglog(Ns, error_fast_vs_fast2, marker='x',
        #            label=r' error_fast_vs_fast2'.format())
        plt.legend(loc='best')

        plt.figure()
        plt.loglog(Ns, norm_nodal, marker='>',
                   label=r' norm_nodal'.format())
        plt.loglog(Ns, norm_gauss, marker='x',
                   label=r' norm_gauss'.format())
        plt.loglog(Ns, norm_analytical, marker='|',
                   label=r' norm_analytical'.format())
        plt.legend(loc='best')

        #   ax.legend()
        # assert error_fd_vs_analytical[-1] < epsilon * 1e2, (
        #   "Finite difference derivative do not corresponds to the analytical expression "
        #   "for partial derivative of double well potential ")
        plt.show()


@pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
    ([3, 4], 0, [25, 28]),
    ([2, 5], 0, [12, 7]),
    #   ([3, 4], 1, [6, 8]),
    #    ([2, 5], 1, [12, 7])
])
def test_fd_check_of_gradient_of_phase_field_potential(discretization_fixture, plot=True):
    # partial derivative of  phase field gradient potential = 2/eta int (  (grad(rho))^2 )    dx
    # partial derivative = grad (rho). grad I / eta
    epsilons = [1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]

    fd_derivative = discretization_fixture.get_scalar_field(name='fd_derivative')
    # Compute phase field gradient potential without perturbations
    phase_field = discretization_fixture.get_scalar_field(
        name='phase_field')
    # Phase field has  one  value per pixel
    phase_field.s = np.random.rand(*phase_field.s.shape) ** 1

    phase_field_0 = discretization_fixture.get_scalar_field(
        name='phase_field_0')
    phase_field_0.s = np.copy(phase_field.s)

    # get analytical partial derivative of phase field gradient potential for a phase field with respect to phase-field
    anal_derivative = discretization_fixture.get_scalar_field(
        name='anal_derivative')
    topology_optimization.partial_derivative_of_gradient_of_phase_field_potential(
        discretization=discretization_fixture,
        phase_field_1nxyz=phase_field,
        output_1nxyz=anal_derivative)

    error_fd_vs_analytical = []

    phase_field_epsilon = discretization_fixture.get_scalar_field(
        name='phase_field_epsilon')

    for epsilon in epsilons:
        # loop over every single element of phase field
        for x in np.arange(discretization_fixture.nb_of_pixels[0]):
            for y in np.arange(discretization_fixture.nb_of_pixels[1]):
                # set phase_field to ones
                # phase_field = discretization_fixture.get_scalar_sized_field() + 1  # Phase field has  one  value per pixel
                phase_field_epsilon.s.fill(0)
                phase_field_epsilon.s = np.copy(phase_field_0.s)

                phase_field_epsilon.s[0, 0, x, y] = phase_field_epsilon.s[0, 0, x, y] + epsilon / 2

                f_rho_grad_potential_perturbed = topology_optimization.compute_gradient_of_phase_field_potential(
                    discretization=discretization_fixture,
                    phase_field_1nxyz=phase_field_epsilon)

                # phase field gradient potential for a phase field without perturbation
                phase_field_epsilon.s[0, 0, x, y] = phase_field_epsilon.s[0, 0, x, y] - epsilon
                f_rho_grad_potential = topology_optimization.compute_gradient_of_phase_field_potential(
                    discretization=discretization_fixture,
                    phase_field_1nxyz=phase_field_epsilon)

                fd_derivative.s[0, 0, x, y] = (f_rho_grad_potential_perturbed - f_rho_grad_potential) / epsilon

        error_fd_vs_analytical.append(
            np.linalg.norm((fd_derivative.s - anal_derivative.s)[0, 0], 'fro'))

        # print(error_fd_vs_analytical)

        assert error_fd_vs_analytical[-1] < epsilon * 1e100, (
            "Finite difference derivative do not corresponds to the analytical expression "
            "for partial derivative of gradient of phase-field potential ")
    if plot:
        import matplotlib.pyplot as plt
        quad_fit_of_error = np.multiply(error_fd_vs_analytical[6], np.asarray(epsilons) ** 2)
        lin_fit_of_error = np.multiply(error_fd_vs_analytical[6], np.asarray(epsilons) ** 1)

        plt.figure()
        plt.loglog(epsilons, error_fd_vs_analytical, marker='x',
                   label=r' error_fd_vs_analytical'.format())
        plt.loglog(epsilons, quad_fit_of_error, linestyle='--',
                   label=r' quad_fit_of_error'.format())
        plt.loglog(epsilons, lin_fit_of_error, linestyle='--',
                   label=r' lin_fit_of_error'.format())
        plt.legend(loc='best')
        plt.xlabel('epsilon - size of Finite Difference step')
        plt.ylabel('Error')
        plt.legend(loc='best')
        plt.ylim([1e-16, 1e10])
        plt.xlim([epsilons[-1], epsilons[0]])

        #   ax.legend()
        # assert error_fd_vs_analytical[-1] < epsilon * 1e2, (
        #   "Finite difference derivative do not corresponds to the analytical expression "
        #   "for partial derivative of double well potential ")
        plt.show()


@pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
    ([3, 4], 0, [6, 8]),
    ([2, 5], 0, [12, 7])])
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
    ([1, 1], 0, [6, 6]),
    ([2, 5], 0, [12, 7]),
    # ([3, 4], 1, [6, 8]),
    # ([2, 5], 1, [12, 7])
])
def test_fd_check_of_stress_equivalence_potential_wrt_phase_field_FE(discretization_fixture, plot=True):
    p = 2
    fd_derivative = discretization_fixture.get_scalar_field(name='fd_derivative')

    target_stress = np.array([[1, 0.], [0., 2]])
    macro_gradient = np.array([[1., 0], [0, 1.]])

    # create material data field
    K_0, G_0 = domain.get_bulk_and_shear_modulus(E=1, poison=0.2)

    elastic_C_1 = domain.get_elastic_material_tensor(dim=discretization_fixture.domain_dimension,
                                                     K=K_0,
                                                     mu=G_0,
                                                     kind='linear')

    material_data_field_C_0_rho_ijklqxyz = discretization_fixture.get_material_data_size_field_mugrid(
        name='test_DATA_FIELD')

    # phase_field = discretization_fixture.get_scalar_sized_field() + 1  #
    phase_field_1nxyz = discretization_fixture.get_scalar_field(
        name='phase_field')
    phase_field_1nxyz.s = np.random.rand(*phase_field_1nxyz.s.shape) ** 1
    #
    phase_field_at_quad_poits_1qnxyz, N_at_quad_points_qnijk = discretization_fixture.evaluate_field_at_quad_points(
        nodal_field_fnxyz=phase_field_1nxyz,
        quad_field_fqnxyz=None,
        quad_points_coords_iq=None)
    # apply material distribution

    material_data_field_C_0_rho_ijklqxyz.s = elastic_C_1[..., np.newaxis, np.newaxis, np.newaxis] * \
                                             np.power(phase_field_at_quad_poits_1qnxyz, p)[0, :, 0, ...]

    # Set up the equilibrium system
    macro_gradient_field_ijqxyz = discretization_fixture.get_gradient_size_field(name='macro_gradient_field')
    discretization_fixture.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient,
                                                           macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz)

    # macro_gradient_field = discretization_fixture.get_macro_gradient_field(macro_gradient)
    rhs_inxyz = discretization_fixture.get_unknown_size_field(name='rhs_field')
    discretization_fixture.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                                          macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
                                          rhs_inxyz=rhs_inxyz)

    def K_fun(x, Ax):
        discretization_fixture.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0_rho_ijklqxyz,
                                                          input_field_inxyz=x,
                                                          output_field_inxyz=Ax,
                                                          formulation='small_strain')

    preconditioner = discretization_fixture.get_preconditioner_Green_mugrid(reference_material_data_ijkl=elastic_C_1)

    def M_fun(x, Px):
        """
        Function to compute the product of the Preconditioner matrix with a vector.
        The Preconditioner is represented by the convolution operator.
        """
        discretization_fixture.fft.communicate_ghosts(x)
        discretization_fixture.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                                           input_nodal_field_fnxyz=x,
                                                           output_nodal_field_fnxyz=Px)

    # Solve mechanical equilibrium constrain
    displacement_field_u_inxyz = discretization_fixture.get_displacement_sized_field(
        name='test_displacement_field_in_test')
    displacement_field_u_inxyz.s.fill(0)
    # displacement_field_u_inxyz.s, norms = solvers.PCG(Afun=K_fun,
    #                                                   B=rhs_inxyz.s,
    #                                                   x0=None,
    #                                                   P=M_fun,
    #                                                   steps=int(1500),
    #                                                   toler=1e-10)
    solvers.conjugate_gradients_mugrid(
        comm=discretization_fixture.fft.communicator,
        fc=discretization_fixture.field_collection,
        hessp=K_fun,  # linear operator
        b=rhs_inxyz,
        x=displacement_field_u_inxyz,
        P=M_fun,
        tol=1e-10,
        maxiter=1000,
        # norm_metric=res_norm
    )

    homogenized_stress = discretization_fixture.get_homogenized_stress_mugrid(
        material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
        displacement_field_inxyz=displacement_field_u_inxyz,
        macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
        formulation='small_strain')

    # phase field gradient potential for a phase field without perturbation
    dstress_drho_analytical = (
        topology_optimization.partial_derivative_of_objective_function_stress_equivalence_wrt_phase_field_FE(
            discretization=discretization_fixture,
            material_data_field_ijkl=elastic_C_1,
            displacement_field_fnxyz=displacement_field_u_inxyz,
            macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
            phase_field_1nxyz=phase_field_1nxyz,
            target_stress_ij=target_stress,
            actual_stress_ij=homogenized_stress,
            p=p))

    # get analytical partial derivative of phase field gradient potential for a phase field with respect to phase-field
    epsilons = [1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8,
                1e-9]  # 1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4,
    error_fd_vs_analytical = []

    phase_field_perturbed = discretization_fixture.get_scalar_field(
        name='phase_field_perturbed')
    # material_data_field_C_0_rho_ijklqxyz = discretization_fixture.get_material_data_size_field(name='test_DATA_FIELD')

    for epsilon in epsilons:
        # loop over every single element of phase field
        for x in np.arange(discretization_fixture.nb_of_pixels[0]):
            for y in np.arange(discretization_fixture.nb_of_pixels[1]):
                # set phase_field to ones
                phase_field_perturbed.s.fill(0)
                phase_field_perturbed.s = np.copy(phase_field_1nxyz.s)  # Phase field has  one  value per pixel
                phase_field_perturbed.s[0, 0, x, y] = phase_field_perturbed.s[0, 0, x, y] + epsilon / 2

                # apply material distribution
                phase_field_at_quad_points_1qnxyz, N_at_quad_points_qnijk = discretization_fixture.evaluate_field_at_quad_points(
                    nodal_field_fnxyz=phase_field_perturbed,
                    quad_field_fqnxyz=None,
                    quad_points_coords_iq=None)

                material_data_field_C_0_rho_ijklqxyz.s = elastic_C_1[..., np.newaxis, np.newaxis, np.newaxis] * \
                                                         np.power(phase_field_at_quad_points_1qnxyz, p)[0, :, 0, ...]

                homogenized_stress_plus = discretization_fixture.get_homogenized_stress_mugrid(
                    material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                    displacement_field_inxyz=displacement_field_u_inxyz,
                    macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
                    formulation='small_strain')

                objective_function_stress_part_perturbed_plus = topology_optimization.objective_function_stress_equivalence(
                    discretization=discretization_fixture,
                    actual_stress_ij=homogenized_stress_plus,
                    target_stress_ij=target_stress)
                # backward evaluation

                phase_field_perturbed.s[0, 0, x, y] = phase_field_perturbed.s[0, 0, x, y] - epsilon

                # Gradient of material data with respect to phase field
                phase_field_at_quad_points_1qnxyz, N_at_quad_points_qnijk = discretization_fixture.evaluate_field_at_quad_points(
                    nodal_field_fnxyz=phase_field_perturbed,
                    quad_field_fqnxyz=None,
                    quad_points_coords_iq=None)
                # apply material distribution
                material_data_field_C_0_rho_ijklqxyz.s = elastic_C_1[..., np.newaxis, np.newaxis, np.newaxis] * \
                                                         np.power(phase_field_at_quad_points_1qnxyz, p)[0, :, 0, ...]

                homogenized_stress_minus = discretization_fixture.get_homogenized_stress_mugrid(
                    material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                    displacement_field_inxyz=displacement_field_u_inxyz,
                    macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
                    formulation='small_strain')

                objective_function_stress_part_perturbed_minus = topology_optimization.objective_function_stress_equivalence(
                    discretization=discretization_fixture,
                    actual_stress_ij=homogenized_stress_minus,
                    target_stress_ij=target_stress)

                fd_derivative.s[0, 0, x, y] = (objective_function_stress_part_perturbed_plus
                                               -
                                               objective_function_stress_part_perturbed_minus) / epsilon

        # print(df_drho_analytical[0, 0])
        error_fd_vs_analytical.append(
            np.linalg.norm((fd_derivative.s[0, 0] - dstress_drho_analytical[0, 0]), 'fro'))

        np.prod(discretization_fixture.pixel_size)

        # assert error_fd_vs_analytical[-1] < epsilon * 100, (
        #     "Finite difference derivative do not corresponds to the analytical expression "
        #     "for partial derivative of gradient of phase-field potential ")
    quad_fit_of_error = np.multiply(error_fd_vs_analytical[6], np.asarray(epsilons) ** 2)
    lin_fit_of_error = np.multiply(error_fd_vs_analytical[6], np.asarray(epsilons) ** 1)

    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.loglog(epsilons, error_fd_vs_analytical, marker='x',
                   label=r' error_fd_vs_analytical'.format())
        plt.loglog(epsilons, quad_fit_of_error, linestyle='--',
                   label=r' quad_fit_of_error'.format())
        plt.loglog(epsilons, lin_fit_of_error, linestyle='--',
                   label=r' lin_fit_of_error'.format())
        plt.legend(loc='best')
        plt.xlabel('epsilon - size of Finite Difference step')
        plt.ylabel('Error')
        plt.show()


@pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
    ([1, 1], 0, [6, 6]),
    ([2, 5], 0, [12, 7]),
    # ([3, 4], 1, [6, 8]),
    # ([2, 5], 1, [12, 7])
])
def test_fd_check_of_stress_equivalence_potential_wrt_displacement_FE(discretization_fixture, plot=True):
    p = 2

    # create material data field
    K_0, G_0 = domain.get_bulk_and_shear_modulus(E=1, poison=0.2)

    elastic_C_1 = domain.get_elastic_material_tensor(dim=discretization_fixture.domain_dimension,
                                                     K=K_0,
                                                     mu=G_0,
                                                     kind='linear')

    material_data_field_C_0_rho = discretization_fixture.get_material_data_size_field_mugrid(name='test_DATA_FIELD')

    # phase field rho
    phase_field_1nxyz = discretization_fixture.get_scalar_field(
        name='phase_field')
    phase_field_1nxyz.s = np.random.rand(*phase_field_1nxyz.s.shape) ** 1  # set random distribution
    #
    phase_field_at_quad_poits_1qnxyz, N_at_quad_points_qnijk = discretization_fixture.evaluate_field_at_quad_points(
        nodal_field_fnxyz=phase_field_1nxyz,
        quad_field_fqnxyz=None,
        quad_points_coords_iq=None)
    # apply material distribution

    material_data_field_C_0_rho.s = elastic_C_1[..., np.newaxis, np.newaxis, np.newaxis] * \
                                    np.power(phase_field_at_quad_poits_1qnxyz, p)[0, :, 0, ...]

    # Set up the equilibrium system
    macro_gradient_ij = np.array([[1., 0], [0, 1.]])
    # macro_gradient_field = discretization_fixture.get_macro_gradient_field(macro_gradient_ij)
    macro_gradient_field_ijqxyz = discretization_fixture.get_gradient_size_field(name='macro_gradient_field')
    discretization_fixture.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient_ij,
                                                           macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz)

    rhs = discretization_fixture.get_unknown_size_field(name='rhs_field')
    discretization_fixture.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0_rho,
                                          macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
                                          rhs_inxyz=rhs)

    def K_fun(x, Ax):
        discretization_fixture.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0_rho,
                                                          input_field_inxyz=x,
                                                          output_field_inxyz=Ax,
                                                          formulation='small_strain')

    preconditioner = discretization_fixture.get_preconditioner_Green_mugrid(reference_material_data_ijkl=elastic_C_1)

    def M_fun(x, Px):
        discretization_fixture.fft.communicate_ghosts(x)
        discretization_fixture.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                                           input_nodal_field_fnxyz=x,
                                                           output_nodal_field_fnxyz=Px)

    # Solve mechanical equilibrium constrain
    displacement_field_u_inxyz = discretization_fixture.get_displacement_sized_field(
        name='test_displacement_field_in_test')
    displacement_field_u_inxyz.s.fill(0)
    # displacement_field_u_inxyz.s, norms = solvers.PCG(K_fun, rhs.s, x0=None, P=M_fun, steps=int(1500), toler=1e-10)
    solvers.conjugate_gradients_mugrid(
        comm=discretization_fixture.fft.communicator,
        fc=discretization_fixture.field_collection,
        hessp=K_fun,  # linear operator
        b=rhs,
        x=displacement_field_u_inxyz,
        P=M_fun,
        tol=1e-10,
        maxiter=1000,
        # norm_metric=res_norm
    )
    # compute homogenized stress field corresponding to displacement

    actual_stress_ij = discretization_fixture.get_homogenized_stress_mugrid(
        material_data_field_ijklqxyz=material_data_field_C_0_rho,
        displacement_field_inxyz=displacement_field_u_inxyz,
        macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
        formulation='small_strain')

    target_stress_ij = np.array([[1, 0.], [0., 2]])
    stress_difference_ij = (target_stress_ij - actual_stress_ij)

    # phase field gradient potential for a phase field without perturbation
    stress_difference_ijqxyz = discretization_fixture.get_gradient_size_field(
        name='stress_difference_ijqxyz_test_')
    discretization_fixture.get_macro_gradient_field_mugrid(macro_gradient_ij=stress_difference_ij,
                                                           macro_gradient_field_ijqxyz=stress_difference_ijqxyz)

    df_du_field = discretization_fixture.get_unknown_size_field(name='adjoint_problem_rhs_test')
    discretization_fixture.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0_rho,
                                          macro_gradient_field_ijqxyz=stress_difference_ijqxyz,
                                          rhs_inxyz=df_du_field)
    # minus sign is already there
    # Normalization
    df_du_field.s = 2 * df_du_field.s / np.sum(target_stress_ij ** 2) / discretization_fixture.cell.domain_volume
    # get analytical partial derivative of phase field gradient potential for a phase field with respect to phase-field
    # TODO
    # Normalization
    # df_du_field.s = weight * df_du_field.s / np.sum(target_stress_ij ** 2)
    # TODO
    # finite difference detivative
    epsilons = [1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    # allocate fields
    fd_derivative = discretization_fixture.get_displacement_sized_field(name='fd_derivative_in_fd_test')

    error_fd_vs_analytical_x = []
    error_fd_vs_analytical_y = []
    u_inxyz_perturbed = discretization_fixture.get_displacement_sized_field(
        name='test_displacement_field_in_test_perturbed')
    for epsilon in epsilons:
        # loop over every single element of displacement field
        for f in np.arange(discretization_fixture.cell.unknown_shape[0]):
            for n in np.arange(discretization_fixture.nb_nodes_per_pixel):
                for x in np.arange(discretization_fixture.nb_of_pixels[0]):
                    for y in np.arange(discretization_fixture.nb_of_pixels[1]):
                        u_inxyz_perturbed.s.fill(0)
                        u_inxyz_perturbed.s = np.copy(displacement_field_u_inxyz.s)
                        u_inxyz_perturbed.s[f, n, x, y] = u_inxyz_perturbed.s[f, n, x, y] + epsilon / 1
                        # set phase_field to ones
                        # compute homogenized stress field for perturbed displacement

                        actual_stress_ij = discretization_fixture.get_homogenized_stress_mugrid(
                            material_data_field_ijklqxyz=material_data_field_C_0_rho,
                            displacement_field_inxyz=u_inxyz_perturbed,
                            macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
                            formulation='small_strain')

                        f_sigma_plus = topology_optimization.compute_stress_equivalence_potential(
                            actual_stress_ij=actual_stress_ij,
                            target_stress_ij=target_stress_ij)

                        u_inxyz_perturbed.s[f, n, x, y] = u_inxyz_perturbed.s[f, n, x, y] - epsilon
                        actual_stress_ij = discretization_fixture.get_homogenized_stress_mugrid(
                            material_data_field_ijklqxyz=material_data_field_C_0_rho,
                            displacement_field_inxyz=u_inxyz_perturbed,
                            macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
                            formulation='small_strain')

                        f_sigma_minus = topology_optimization.compute_stress_equivalence_potential(
                            actual_stress_ij=actual_stress_ij,
                            target_stress_ij=target_stress_ij)

                        fd_derivative.s[f, n, x, y] = (f_sigma_plus - f_sigma_minus) / epsilon

        # print(df_drho_analytical[0, 0])
        error_fd_vs_analytical_x.append(
            np.linalg.norm(fd_derivative.s[0, 0] - df_du_field.s[0, 0], 'fro'))
        error_fd_vs_analytical_y.append(
            np.linalg.norm(fd_derivative.s[1, 0] - df_du_field.s[1, 0], 'fro'))
        # print(epsilon)

        # assert error_fd_vs_analytical_x[-1] < epsilon * 1000, (
        #     "Finite difference derivative do not corresponds to the analytical expression "
        #     "for partial derivative  stress eq with respect to displacement ")
        # assert error_fd_vs_analytical_y[-1] < epsilon * 1000, (
        #     "Finite difference derivative do not corresponds to the analytical expression "
        #     "for partial derivative  stress eq with respect to displacement ")
    quad_fit_of_error = np.multiply(error_fd_vs_analytical_x[4], np.asarray(epsilons) ** 2)
    lin_fit_of_error = np.multiply(error_fd_vs_analytical_x[4], np.asarray(epsilons) ** 1)
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.loglog(epsilons, error_fd_vs_analytical_x, marker='x',
                   label=r' error_fd_vs_analytical x'.format())
        plt.loglog(epsilons, error_fd_vs_analytical_y, marker='o',
                   label=r' error_fd_vs_analytical y'.format())
        plt.loglog(epsilons, quad_fit_of_error, linestyle='--',
                   label=r' quad_fit_of_error'.format())
        plt.loglog(epsilons, lin_fit_of_error, linestyle='--',
                   label=r' lin_fit_of_error'.format())
        plt.legend(loc='best')
        plt.xlabel('epsilon - size of Finite Difference step')
        plt.ylabel('Error')
        # plt.ylim([1e-12, 1e4])

        plt.show()


@pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
    ([3, 4], 0, [6, 8]),
    ([2, 5], 0, [12, 7])  # ,
    # ([3, 4], 1, [6, 8]),
    # ([2, 5], 1, [12, 7])
])
def test_fd_check_of_stress_equivalence_potential(discretization_fixture, plot=True):
    # TODO this check works for p=1,2
    epsilons = [1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    # epsilons = [1e-4]
    p = 2
    fd_derivative = discretization_fixture.get_scalar_field(
        name='fd_derivative')

    target_stress = np.array([[1, 0.3], [0.3, 2]])
    macro_gradient = np.array([[0.01, 0], [0, 0.02]])

    # create material data field
    K_0, G_0 = domain.get_bulk_and_shear_modulus(E=1, poison=0.2)

    elastic_C_0 = domain.get_elastic_material_tensor(dim=discretization_fixture.domain_dimension,
                                                     K=K_0,
                                                     mu=G_0,
                                                     kind='linear')

    material_data_field_C_0_rho = discretization_fixture.get_material_data_size_field_mugrid(
        name='material_data_field_C_0_test_DATA_FIELD')

    # phase_field = discretization_fixture.get_scalar_sized_field() + 1  #
    phase_field_1nxyz = discretization_fixture.get_scalar_field(
        name='phase_field')
    phase_field_1nxyz.s = np.random.rand(*phase_field_1nxyz.s.shape) ** 1  # set random distribution

    phase_field_at_quad_poits_1qxyz = discretization_fixture.get_quad_field_scalar(
        name='   phase_field_at_quad_poits_1qxyz ')
    discretization_fixture.apply_N_operator_mugrid(phase_field_1nxyz, phase_field_at_quad_poits_1qxyz)

    # apply material distribution
    material_data_field_C_0_rho.s = elastic_C_0[..., np.newaxis, np.newaxis, np.newaxis] * \
                                    np.power(phase_field_at_quad_poits_1qxyz.s, p)[0, 0, :, ...]

    # Set up the equilibrium system
    macro_gradient_field = discretization_fixture.get_gradient_size_field(name='macro_gradient_field')
    discretization_fixture.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient,
                                                           macro_gradient_field_ijqxyz=macro_gradient_field)
    rhs = discretization_fixture.get_unknown_size_field(name='rhs_field')
    discretization_fixture.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0_rho,
                                          macro_gradient_field_ijqxyz=macro_gradient_field,
                                          rhs_inxyz=rhs)

    def K_fun(x, Ax):
        discretization_fixture.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0_rho,
                                                          input_field_inxyz=x,
                                                          output_field_inxyz=Ax,
                                                          formulation='small_strain')

    preconditioner = discretization_fixture.get_preconditioner_Green_mugrid(reference_material_data_ijkl=elastic_C_0)

    def M_fun(x, Px):
        discretization_fixture.fft.communicate_ghosts(x)
        discretization_fixture.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                                           input_nodal_field_fnxyz=x,
                                                           output_nodal_field_fnxyz=Px)

    # Solve mechanical equilibrium constrain
    displacement_field = discretization_fixture.get_displacement_sized_field(
        name='test_displacement_field_in_test')
    displacement_field.s.fill(0)
    # displacement_field.s, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(1500), toler=1e-10)
    solvers.conjugate_gradients_mugrid(
        comm=discretization_fixture.fft.communicator,
        fc=discretization_fixture.field_collection,
        hessp=K_fun,  # linear operator
        b=rhs,
        x=displacement_field,
        P=M_fun,
        tol=1e-10,
        maxiter=50000,
    )
    # ----------------------------------------------------------------------
    # compute stress field corresponding to displacement
    homogenized_stress = discretization_fixture.get_homogenized_stress_mugrid(
        material_data_field_ijklqxyz=material_data_field_C_0_rho,
        displacement_field_inxyz=displacement_field,
        macro_gradient_field_ijqxyz=macro_gradient_field,
        formulation='small_strain')

    # objective_function_stress_part = topology_optimization.objective_function_stress_equivalence(
    #     discretization=discretization_fixture,
    #     actual_stress_ij=homogenized_stress,
    #     target_stress_ij=target_stress)

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
        topology_optimization.partial_derivative_of_objective_function_stress_equivalence_wrt_phase_field_FE(
            discretization=discretization_fixture,
            material_data_field_ijkl=elastic_C_0,
            displacement_field_fnxyz=displacement_field,
            macro_gradient_field_ijqxyz=macro_gradient_field,
            phase_field_1nxyz=phase_field_1nxyz,
            target_stress_ij=target_stress,
            actual_stress_ij=homogenized_stress,
            p=p
        ))

    error_fd_vs_analytical = []
    phase_field_perturbed = discretization_fixture.get_scalar_field(
        name='phase_field_perturbed')

    for epsilon in epsilons:
        # loop over every single element of phase field
        for x in np.arange(discretization_fixture.nb_of_pixels[0]):
            for y in np.arange(discretization_fixture.nb_of_pixels[1]):
                # set phase_field to ones
                phase_field_perturbed.s.fill(0)
                phase_field_perturbed.s = np.copy(phase_field_1nxyz.s)  # Phase field has  one  value per pixel
                # phase_field_perturbed=phase_field_perturbed**p
                phase_field_perturbed.s[0, 0, x, y] = phase_field_perturbed.s[0, 0, x, y] + epsilon / 2

                # apply material distribution
                discretization_fixture.apply_N_operator_mugrid(phase_field_perturbed, phase_field_at_quad_poits_1qxyz)
                material_data_field_C_0_rho.s = elastic_C_0[..., np.newaxis, np.newaxis, np.newaxis] * \
                                                np.power(phase_field_at_quad_poits_1qxyz.s, p)[0, 0, :, ...]

                homogenized_stress = discretization_fixture.get_homogenized_stress_mugrid(
                    material_data_field_ijklqxyz=material_data_field_C_0_rho,
                    displacement_field_inxyz=displacement_field,
                    macro_gradient_field_ijqxyz=macro_gradient_field,
                    formulation='small_strain')

                objective_function_stress_part_perturbed_plus = topology_optimization.objective_function_stress_equivalence(
                    discretization=discretization_fixture,
                    actual_stress_ij=homogenized_stress,
                    target_stress_ij=target_stress)

                # phase_field_perturbed=phase_field_perturbed**p
                phase_field_perturbed.s[0, 0, x, y] = phase_field_perturbed.s[0, 0, x, y] - epsilon
                # phase_field_at_quad_poits_1qnxyz, _ = discretization_fixture.evaluate_field_at_quad_points(
                #     nodal_field_fnxyz=phase_field_perturbed,
                #     quad_field_fqnxyz=None,
                #     quad_points_coords_iq=None)
                # # apply material distribution
                # material_data_field_C_0_rho.s = elastic_C_0[..., np.newaxis, np.newaxis, np.newaxis] * \
                #                                 np.power(phase_field_at_quad_poits_1qnxyz, p)[0, :, 0, ...]
                discretization_fixture.apply_N_operator_mugrid(phase_field_perturbed, phase_field_at_quad_poits_1qxyz)
                material_data_field_C_0_rho.s = elastic_C_0[..., np.newaxis, np.newaxis, np.newaxis] * \
                                                np.power(phase_field_at_quad_poits_1qxyz.s, p)[0, 0, :, ...]

                homogenized_stress = discretization_fixture.get_homogenized_stress_mugrid(
                    material_data_field_ijklqxyz=material_data_field_C_0_rho,
                    displacement_field_inxyz=displacement_field,
                    macro_gradient_field_ijqxyz=macro_gradient_field,
                    formulation='small_strain')

                objective_function_stress_part_perturbed_minus = topology_optimization.objective_function_stress_equivalence(
                    discretization=discretization_fixture,
                    actual_stress_ij=homogenized_stress,
                    target_stress_ij=target_stress)

                fd_derivative.s[0, 0, x, y] = (objective_function_stress_part_perturbed_plus
                                               -
                                               objective_function_stress_part_perturbed_minus) / epsilon
                # fd_derivative_wo[0, 0, x, y] = (
                #                                 objective_function_perturbed - objective_function) / epsilon

        # print(df_drho_analytical[0, 0])
        fd_norm = np.sum(np.linalg.norm((fd_derivative.s[0, 0] - df_drho_analytical[0, 0]), 'fro'))

        # print(f_rho_grad_potential_analytical)
        error_fd_vs_analytical.append(fd_norm)

        # print(error_fd_vs_analytical)

        # assert error_fd_vs_analytical[-1] < epsilon * 10, (
        #     "Finite difference derivative do not corresponds to the analytical expression "
        #     "for partial derivative of gradient of phase-field potential ")
    quad_fit_of_error = np.multiply(error_fd_vs_analytical[6], np.asarray(epsilons) ** 2)
    lin_fit_of_error = np.multiply(error_fd_vs_analytical[6], np.asarray(epsilons) ** 1)
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.loglog(epsilons, error_fd_vs_analytical, marker='x',
                   label=r' error_fd_vs_analytical'.format())
        plt.loglog(epsilons, quad_fit_of_error, linestyle='--',
                   label=r' quad_fit_of_error'.format())
        plt.loglog(epsilons, lin_fit_of_error, linestyle='--',
                   label=r' lin_fit_of_error'.format())
        plt.legend(loc='best')
        plt.xlabel('epsilon - size of Finite Difference step')
        plt.ylabel('Error')
        plt.ylim([1e-12, 1e5])
        plt.show()


@pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
    ([1, 1], 0, [6, 6]),
    ([2, 5], 0, [12, 7]),
    ([3, 4], 0, [8, 13])])
def test_fd_check_of_adjoint_potential_wrt_phase_field_FE(discretization_fixture, plot=True):
    p = 2
    weight = 7

    phase_field_1nxyz = discretization_fixture.get_scalar_field(
        name='phase_field')
    phase_field_1nxyz.s = np.random.rand(*phase_field_1nxyz.s.shape) ** 1  # set random distribution

    phase_field_at_quad_poits_1qxyz = discretization_fixture.get_quad_field_scalar(
        name='phase_field_at_quad_poits_1qxyz')
    discretization_fixture.apply_N_operator_mugrid(phase_field_1nxyz, phase_field_at_quad_poits_1qxyz)

    # just to set up the system
    E_0 = 1
    poison_0 = 0.2

    K_0, G_0 = domain.get_bulk_and_shear_modulus(E=E_0, poison=poison_0)

    elastic_C_0 = domain.get_elastic_material_tensor(dim=discretization_fixture.domain_dimension,
                                                     K=K_0,
                                                     mu=G_0,
                                                     kind='linear')
    material_data_field_C_0_rho_ijklqxyz = discretization_fixture.get_material_data_size_field_mugrid(
        name='test_DATA_FIELD')
    material_data_field_C_0_rho_ijklqxyz.s = elastic_C_0[..., np.newaxis, np.newaxis, np.newaxis] * \
                                             np.power(phase_field_at_quad_poits_1qxyz.s, p)[0, 0, :, ...]

    preconditioner = discretization_fixture.get_preconditioner_Green_mugrid(reference_material_data_ijkl=elastic_C_0)

    def M_fun(x, Px):
        discretization_fixture.fft.communicate_ghosts(x)
        discretization_fixture.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                                           input_nodal_field_fnxyz=x,
                                                           output_nodal_field_fnxyz=Px)

    # Set up the equilibrium system
    macro_gradient_ij = np.array([[1., 0.0],
                                  [0.0, 1.0]])
    macro_gradient_field_ijqxyz = discretization_fixture.get_gradient_size_field(name='macro_gradient_field')
    discretization_fixture.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient_ij,
                                                           macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz)
    # macro_gradient_field_ijqxyz = discretization_fixture.get_macro_gradient_field(macro_gradient_ij)
    rhs_inxyz = discretization_fixture.get_unknown_size_field(name='rhs_field')
    discretization_fixture.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                                          macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
                                          rhs_inxyz=rhs_inxyz)

    def K_fun(x, Ax):
        discretization_fixture.fft.communicate_ghosts(x)
        discretization_fixture.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0_rho_ijklqxyz,
                                                          input_field_inxyz=x,
                                                          output_field_inxyz=Ax,
                                                          formulation='small_strain')

    # Solve mechanical equilibrium constrain
    displacement_field_u_inxyz = discretization_fixture.get_displacement_sized_field(
        name='test_displacement_field_in_test')
    displacement_field_u_inxyz.s.fill(0)
    solvers.conjugate_gradients_mugrid(
        comm=discretization_fixture.fft.communicator,
        fc=discretization_fixture.field_collection,
        hessp=K_fun,  # linear operator
        b=rhs_inxyz,
        x=displacement_field_u_inxyz,
        P=M_fun,
        tol=1e-10,
        maxiter=50000,
    )

    # compute strain field from to displacement and macro gradient
    strain_ijqxyz = discretization_fixture.get_displacement_gradient_sized_field(
        name='strain_ijqxyz_local_at_testy')

    discretization_fixture.apply_gradient_operator_symmetrized_mugrid(u_inxyz=displacement_field_u_inxyz,
                                                                      grad_u_ijqxyz=strain_ijqxyz)

    strain_ijqxyz.s = macro_gradient_field_ijqxyz.s + strain_ijqxyz.s
    # now this part will compute adjoint field K*lambda =
    # compute homogenized stress field corresponding to displacement
    actual_stress_ij = discretization_fixture.get_homogenized_stress_mugrid(
        material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
        displacement_field_inxyz=displacement_field_u_inxyz,
        macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
        formulation='small_strain')

    target_stress_ij = np.array([[0.5, 0.0],
                                 [0.0, 0.5]])
    stress_difference_ij = (target_stress_ij - actual_stress_ij)

    # Solve
    # stress difference potential
    # rhs=-Dt*wA*E  -- we can use it to assemble df_du_field
    stress_difference_ijqxyz = discretization_fixture.get_gradient_size_field(
        name='stress_difference_ijqxyz_test_')
    # stress_difference_ijqxyz = discretization_fixture.get_gradient_size_field()
    #  stress_difference_ijqxyz[:, :, ...] = stress_difference_ij[
    #      (...,) + (np.newaxis,) * (stress_difference_ijqxyz.ndim - 2)]
    discretization_fixture.get_macro_gradient_field_mugrid(macro_gradient_ij=stress_difference_ij,
                                                           macro_gradient_field_ijqxyz=stress_difference_ijqxyz)
    # minus sign is already there
    df_du_field = discretization_fixture.get_unknown_size_field(name='adjoint_problem_rhs_test')
    discretization_fixture.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                                          macro_gradient_field_ijqxyz=stress_difference_ijqxyz,
                                          rhs_inxyz=df_du_field)  # minus sign is already there
    df_du_field.s = -2 * df_du_field.s / discretization_fixture.cell.domain_volume
    # Normalization
    df_du_field.s = weight * df_du_field.s / np.sum(target_stress_ij ** 2)

    # solve adjoint problem
    adjoint_field_inxyz = discretization_fixture.get_unknown_size_field(name='adjoint_field_inxyz_test')
    adjoint_field_inxyz.s.fill(0)

    solvers.conjugate_gradients_mugrid(
        comm=discretization_fixture.fft.communicator,
        fc=discretization_fixture.field_collection,
        hessp=K_fun,  # linear operator
        b=df_du_field,
        x=adjoint_field_inxyz,
        P=M_fun,
        tol=1e-10,
        maxiter=50000,
    )

    #
    dadjoin_drho = discretization_fixture.get_scalar_field(name='dadjoin_drho_test')
    topology_optimization.partial_derivative_of_adjoint_potential_wrt_phase_field_FE(
        discretization=discretization_fixture,
        base_material_data_ijkl=elastic_C_0,
        displacement_field_fnxyz=displacement_field_u_inxyz,
        macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
        phase_field_1nxyz=phase_field_1nxyz,
        adjoint_field_inxyz=adjoint_field_inxyz,
        output_field_inxyz=dadjoin_drho,
        p=p)
    gradient_of_adjoint_field_ijqxyz = discretization_fixture.get_gradient_size_field(
        name='gradient_of_adjoint_field_ijqxyz_test')
    gradient_of_adjoint_field_ijqxyz = discretization_fixture.apply_gradient_operator_symmetrized_mugrid(
        u_inxyz=adjoint_field_inxyz, grad_u_ijqxyz=gradient_of_adjoint_field_ijqxyz)

    # get analytical partial derivative of phase field gradient potential for a phase field with respect to phase-field
    # epsilons = [1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    epsilons = [1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]

    error_fd_vs_analytical = []
    fd_derivative = discretization_fixture.get_scalar_field(name='fd_derivative_in_fd_test')
    # allocate fields
    phase_field_perturbed = discretization_fixture.get_scalar_field(
        name='phase_field_perturbed')
    stress_field_ijqxyz = discretization_fixture.get_gradient_size_field(
        name='stress_field_ijqxyz_in_test_perturbed')
    material_data_field_C_0_rho_ijklqxyz = discretization_fixture.get_material_data_size_field_mugrid(
        name='test_DATA_FIELD')
    for epsilon in epsilons:
        # loop over every single element of phase field
        for x in np.arange(discretization_fixture.nb_of_pixels[0]):
            for y in np.arange(discretization_fixture.nb_of_pixels[1]):
                # set phase_field to ones
                phase_field_perturbed.s.fill(0)

                phase_field_perturbed.s = phase_field_1nxyz.s.copy()  # Phase field has  one  value per pixel
                # f(x+eps/2)
                # perturb phase field
                phase_field_perturbed.s[0, 0, x, y] = phase_field_perturbed.s[0, 0, x, y] + epsilon / 2
                # evaluate to quad points
                # phase_field_at_quad_poits_1qnxyz, _ = discretization_fixture.evaluate_field_at_quad_points(
                #     nodal_field_fnxyz=phase_field_perturbed,
                #     quad_field_fqnxyz=None,
                #     quad_points_coords_iq=None)
                discretization_fixture.apply_N_operator_mugrid(phase_field_perturbed, phase_field_at_quad_poits_1qxyz)
                # material_data_field_C_0_rho_ijklqxyz = material_data_field_C_0_ijklqxyz[..., :, :, :] * (
                #     np.power(
                #         phase_field_at_quad_poits_1qnxyz,
                #         p))[0,
                # :, 0, ...]
                # material_data_field_C_0_rho_ijklqxyz.s = elastic_C_0[..., np.newaxis, np.newaxis, np.newaxis] * \
                #                                          np.power(phase_field_at_quad_poits_1qnxyz, p)[0, :, 0, ...]
                material_data_field_C_0_rho_ijklqxyz.s = elastic_C_0[..., np.newaxis, np.newaxis, np.newaxis] * \
                                                         np.power(phase_field_at_quad_poits_1qxyz.s, p)[0, 0, :, ...]
                stress_field_ijqxyz.s.fill(0)
                discretization_fixture.get_stress_field_mugrid(
                    material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                    displacement_field_inxyz=displacement_field_u_inxyz,
                    macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
                    output_stress_field_ijqxyz=stress_field_ijqxyz,
                    formulation='small_strain')

                adjoint_energy_plus = topology_optimization.adjoint_potential(
                    discretization=discretization_fixture,
                    stress_field_ijqxyz=stress_field_ijqxyz,
                    adjoint_field_inxyz=adjoint_field_inxyz)

                # f(x-eps/2)
                phase_field_perturbed.s[0, 0, x, y] = phase_field_perturbed.s[0, 0, x, y] - epsilon
                # evaluate to quad points
                discretization_fixture.apply_N_operator_mugrid(phase_field_perturbed, phase_field_at_quad_poits_1qxyz)
                material_data_field_C_0_rho_ijklqxyz.s = elastic_C_0[..., np.newaxis, np.newaxis, np.newaxis] * \
                                                         np.power(phase_field_at_quad_poits_1qxyz.s, p)[0, 0, :, ...]

                stress_field_ijqxyz.s.fill(0)
                discretization_fixture.get_stress_field_mugrid(
                    material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                    displacement_field_inxyz=displacement_field_u_inxyz,
                    macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
                    output_stress_field_ijqxyz=stress_field_ijqxyz,
                    formulation='small_strain')

                adjoint_energy_minus = topology_optimization.adjoint_potential(
                    discretization=discretization_fixture,
                    stress_field_ijqxyz=stress_field_ijqxyz,
                    adjoint_field_inxyz=adjoint_field_inxyz)

                fd_derivative.s[0, 0, x, y] = (adjoint_energy_plus - adjoint_energy_minus) / epsilon

        error_fd_vs_analytical.append(
            np.linalg.norm(fd_derivative.s[0, 0] - dadjoin_drho.s[0, 0], 'fro'))

        # assert error_fd_vs_analytical[-1] < epsilon * 10, (
        #     "Finite difference derivative do not corresponds to the analytical expression "
        #     "for partial derivative of gradient of phase-field potential ")
    print(error_fd_vs_analytical)
    quad_fit_of_error = np.multiply(error_fd_vs_analytical[6], np.asarray(epsilons) ** 2)
    lin_fit_of_error = np.multiply(error_fd_vs_analytical[6], np.asarray(epsilons) ** 1)
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.loglog(epsilons, error_fd_vs_analytical, marker='x',
                   label=r' error_fd_vs_analytical'.format())
        plt.loglog(epsilons, quad_fit_of_error, linestyle='--',
                   label=r' quad_fit_of_error'.format())
        plt.loglog(epsilons, lin_fit_of_error, linestyle='--',
                   label=r' lin_fit_of_error'.format())
        plt.legend(loc='best')
        plt.xlabel('epsilon - size of Finite Difference step')
        plt.ylabel('Error')

        plt.show()


@pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
    ([3, 4], 0, [6, 8]),
    ([2, 5], 0, [12, 7])])
def test_fd_check_of_adjoint_potential_wrt_phase_field_FE_2(discretization_fixture, plot=True):
    epsilons = [1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    # epsilonminuss = [1e-4] #
    p = 2
    fd_derivative = discretization_fixture.get_scalar_field(name='fd_derivative')

    macro_gradient = np.array([[0.01, 0], [0, 0.01]])

    # create material data field
    K_0, G_0 = domain.get_bulk_and_shear_modulus(E=1, poison=0.2)

    material_C_0 = domain.get_elastic_material_tensor(dim=discretization_fixture.domain_dimension,
                                                      K=K_0,
                                                      mu=G_0,
                                                      kind='linear')
    material_data_field_C_0_rho = discretization_fixture.get_material_data_size_field_mugrid(name='test_DATA_FIELD')

    phase_field = discretization_fixture.get_scalar_field(
        name='phase_field')
    phase_field.s = np.random.rand(*phase_field.s.shape) ** 1  # set random distribution

    # apply material distribution
    phase_field_at_quad_poits_1qxyz = discretization_fixture.get_quad_field_scalar(
        name='phase_field_at_quad_poits_1qxyz')
    discretization_fixture.apply_N_operator_mugrid(phase_field, phase_field_at_quad_poits_1qxyz)

    material_data_field_C_0_rho.s = material_C_0[..., np.newaxis, np.newaxis, np.newaxis] * \
                                    np.power(phase_field_at_quad_poits_1qxyz.s, p)[0, 0, :, ...]

    # Set up the equilibrium system
    macro_gradient_field = discretization_fixture.get_gradient_size_field(name='macro_gradient_field')
    discretization_fixture.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient,
                                                           macro_gradient_field_ijqxyz=macro_gradient_field)
    # create random displacement field
    displacement_field = discretization_fixture.get_displacement_sized_field(
        name='test_displacement_field_in_test')
    displacement_field.s.fill(0)
    displacement_field.s = np.random.rand(*displacement_field.s.shape)

    # ----------------------------------------------------------------------
    # create random adjoint field
    adjoint_field = discretization_fixture.get_displacement_sized_field(
        name='test_adjoint_field_in_test')
    adjoint_field.s.fill(0)
    adjoint_field.s = np.random.rand(*adjoint_field.s.shape) ** 1

    # compute partial derivative of adjoint_potential
    dg_drho_analytical_inxyz = discretization_fixture.get_scalar_field(name='dg_drho_analytical_inxyz_intests')
    topology_optimization.partial_derivative_of_adjoint_potential_wrt_phase_field_FE(
        discretization=discretization_fixture,
        base_material_data_ijkl=material_C_0,
        displacement_field_fnxyz=displacement_field,
        macro_gradient_field_ijqxyz=macro_gradient_field,
        phase_field_1nxyz=phase_field,
        adjoint_field_inxyz=adjoint_field,
        output_field_inxyz=dg_drho_analytical_inxyz,
        p=p)

    error_fd_vs_analytical = []
    error_fd_vs_analytical_FE = []
    phase_field_perturbed = discretization_fixture.get_scalar_field(
        name='phase_field_perturbed')

    stress_field_per = discretization_fixture.get_gradient_size_field(
        name='stress_field_ijqxyz_in_test_perturbed')

    for epsilon in epsilons:
        # loop over every single element of displacement field
        for x in np.arange(discretization_fixture.nb_of_pixels[0]):
            for y in np.arange(discretization_fixture.nb_of_pixels[1]):
                phase_field_perturbed.s.fill(0)
                phase_field_perturbed.s = np.copy(phase_field.s)
                phase_field_perturbed.s[0, 0, x, y] = phase_field_perturbed.s[0, 0, x, y] + epsilon / 2

                # compute stress field corresponding to equilibrated displacement

                # apply material distribution
                # phase_field_at_quad_poits_1qnxyz = discretization_fixture.evaluate_field_at_quad_points(
                #     nodal_field_fnxyz=phase_field_perturbed,
                #     quad_field_fqnxyz=None,
                #     quad_points_coords_iq=None)[0]
                # #
                # material_data_field_C_0_rho.s = material_C_0[..., np.newaxis, np.newaxis, np.newaxis] * \
                #                                 np.power(phase_field_at_quad_poits_1qnxyz, p)[0, :, 0, ...]
                discretization_fixture.apply_N_operator_mugrid(phase_field_perturbed, phase_field_at_quad_poits_1qxyz)

                material_data_field_C_0_rho.s = material_C_0[..., np.newaxis, np.newaxis, np.newaxis] * \
                                                np.power(phase_field_at_quad_poits_1qxyz.s, p)[0, 0, :, ...]

                #
                discretization_fixture.get_stress_field_mugrid(
                    material_data_field_ijklqxyz=material_data_field_C_0_rho,
                    displacement_field_inxyz=displacement_field,
                    macro_gradient_field_ijqxyz=macro_gradient_field,
                    output_stress_field_ijqxyz=stress_field_per,
                    formulation='small_strain')

                adjoint_potential_perturbed_plus = topology_optimization.adjoint_potential(
                    discretization=discretization_fixture,
                    stress_field_ijqxyz=stress_field_per,
                    adjoint_field_inxyz=adjoint_field)

                # - epsilon 2
                phase_field_perturbed.s[0, 0, x, y] = phase_field_perturbed.s[0, 0, x, y] - epsilon

                # apply material distribution
                discretization_fixture.apply_N_operator_mugrid(phase_field_perturbed, phase_field_at_quad_poits_1qxyz)
                material_data_field_C_0_rho.s = material_C_0[..., np.newaxis, np.newaxis, np.newaxis] * \
                                                np.power(phase_field_at_quad_poits_1qxyz.s, p)[0, 0, :, ...]

                discretization_fixture.get_stress_field_mugrid(
                    material_data_field_ijklqxyz=material_data_field_C_0_rho,
                    displacement_field_inxyz=displacement_field,
                    macro_gradient_field_ijqxyz=macro_gradient_field,
                    output_stress_field_ijqxyz=stress_field_per,
                    formulation='small_strain')

                adjoint_potential_perturbed_minus = topology_optimization.adjoint_potential(
                    discretization=discretization_fixture,
                    stress_field_ijqxyz=stress_field_per,
                    adjoint_field_inxyz=adjoint_field)

                fd_derivative.s[0, 0, x, y] = (adjoint_potential_perturbed_plus
                                               -
                                               adjoint_potential_perturbed_minus) / epsilon

        fd_norm_FE = np.sum(np.linalg.norm((fd_derivative.s[0, 0] - dg_drho_analytical_inxyz.s[0, 0]), 'fro'))

        error_fd_vs_analytical.append(fd_norm_FE)
        # print('fd_norm: ', fd_norm)
        print('error_fd_vs_analytical_FE: ', error_fd_vs_analytical_FE)
        #
        # assert error_fd_vs_analytical_FE[-1] < epsilon * 1e5, (
        #     "Finite difference derivative  do not corresponds to the analytical expsression "
        #     "for partial derivative of adjoint potential  w.r.t. displacement "
        #     "for epsilon = {} and p = {}".format(epsilon, p))
    if plot:
        quad_fit_of_error = np.multiply(error_fd_vs_analytical[6], np.asarray(epsilons) ** 2)
        lin_fit_of_error = np.multiply(error_fd_vs_analytical[6], np.asarray(epsilons) ** 1)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.loglog(epsilons, error_fd_vs_analytical, marker='x',
                   label=r' error_fd_vs_analytical'.format())
        plt.loglog(epsilons, quad_fit_of_error, linestyle='--',
                   label=r' quad_fit_of_error'.format())
        plt.loglog(epsilons, lin_fit_of_error, linestyle='--',
                   label=r' lin_fit_of_error'.format())
        plt.xlabel('epsilon - size of Finite Difference step')
        plt.ylabel('Error')
        plt.legend(loc='best')
        plt.ylim([1e-16, 1e10])
        plt.xlim([epsilons[-1], epsilons[0]])
        plt.show()


@pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
    ([1, 1], 0, [6, 6]),
    ([2, 5], 0, [12, 7]),
    # ([3, 4], 1, [6, 8]),
    # ([2, 5], 1, [12, 7])
])
def test_fd_check_of_adjoint_potential_wrt_displacement_FE(discretization_fixture, plot=True):
    p = 2

    phase_field_1nxyz = discretization_fixture.get_scalar_field(
        name='phase_field')
    phase_field_1nxyz.s = np.random.rand(*phase_field_1nxyz.s.shape) ** 1  # # set random distribution

    phase_field_at_quad_poits_1qxyz = discretization_fixture.get_quad_field_scalar(
        name='phase_field_at_quad_poits_1qxyz')
    discretization_fixture.apply_N_operator_mugrid(phase_field_1nxyz, phase_field_at_quad_poits_1qxyz)

    # just to set up the system
    E_0 = 1
    poison_0 = 0.2

    K_0, G_0 = domain.get_bulk_and_shear_modulus(E=E_0, poison=poison_0)

    elastic_C_0 = domain.get_elastic_material_tensor(dim=discretization_fixture.domain_dimension,
                                                     K=K_0,
                                                     mu=G_0,
                                                     kind='linear')

    material_data_field_C_0_rho_ijklqxyz = discretization_fixture.get_material_data_size_field_mugrid(
        name='test_DATA_FIELD')
    material_data_field_C_0_rho_ijklqxyz.s = elastic_C_0[..., np.newaxis, np.newaxis, np.newaxis] * \
                                             np.power(phase_field_at_quad_poits_1qxyz.s, p)[0, 0, :, ...]

    preconditioner = discretization_fixture.get_preconditioner_Green_mugrid(reference_material_data_ijkl=elastic_C_0)

    def M_fun(x, Px):
        discretization_fixture.fft.communicate_ghosts(x)
        discretization_fixture.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                                           input_nodal_field_fnxyz=x,
                                                           output_nodal_field_fnxyz=Px)

    # Set up the equilibrium system
    macro_gradient_ij = np.array([[1., 0], [0, 1.]])
    # macro_gradient_field = discretization_fixture.get_macro_gradient_field(macro_gradient_ij)
    macro_gradient_field_ijqxyz = discretization_fixture.get_gradient_size_field(name='macro_gradient_field')
    discretization_fixture.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient_ij,
                                                           macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz)
    rhs = discretization_fixture.get_unknown_size_field(name='rhs_field')
    discretization_fixture.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                                          macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
                                          rhs_inxyz=rhs)

    def K_fun(x, Ax):
        discretization_fixture.fft.communicate_ghosts(x)
        discretization_fixture.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0_rho_ijklqxyz,
                                                          input_field_inxyz=x,
                                                          output_field_inxyz=Ax,
                                                          formulation='small_strain')

    # Solve mechanical equilibrium constrain
    displacement_field_u_inxyz = discretization_fixture.get_displacement_sized_field(
        name='test_displacement_field_in_test')
    displacement_field_u_inxyz.s.fill(0)
    # displacement_field_u_inxyz.s, norms = solvers.PCG(Afun=K_fun,
    #                                                   B=rhs.s,
    #                                                   x0=None,
    #                                                   P=M_fun,
    #                                                   steps=int(1500),
    #                                                   toler=1e-6)
    solvers.conjugate_gradients_mugrid(
        comm=discretization_fixture.fft.communicator,
        fc=discretization_fixture.field_collection,
        hessp=K_fun,  # linear operator
        b=rhs,
        x=displacement_field_u_inxyz,
        P=M_fun,
        tol=1e-10,
        maxiter=50000,
    )
    # compute strain field from to displacement and macro gradient
    # strain_ijqxyz = discretization_fixture.apply_gradient_operator_symmetrized(displacement_field_u_inxyz)
    # strain_ijqxyz = macro_gradient_field_ijqxyz + strain_ijqxyz
    # now this part will compute adjoint field K*lambda =
    # compute homogenized stress field corresponding to displacement
    # actual_stress_ij = discretization_fixture.get_homogenized_stress_mugrid(
    #     material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
    #     displacement_field_inxyz=displacement_field_u_inxyz,
    #     macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
    #     formulation='small_strain')
    #
    # target_stress_ij = np.array([[0.5, 0.0],
    #                              [0.0, 0.5]])
    # stress_difference_ij = (target_stress_ij - actual_stress_ij)

    adjoint_field_inxyz = discretization_fixture.get_displacement_sized_field(
        name='adjoint_field_fnxyz_in_test')
    adjoint_field_inxyz.s.fill(0)
    adjoint_field_inxyz.s = np.random.rand(*adjoint_field_inxyz.s.shape)
    discretization_fixture.fft.communicate_ghosts(adjoint_field_inxyz)
    # gradient_of_adjoint_field_ijqxyz = discretization_fixture.apply_gradient_operator_symmetrized(adjoint_field_fnxyz)
    #
    # # minus sign is already there
    # dg_du_field_analytical = -(
    #         discretization_fixture.get_rhs(material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
    #                                        macro_gradient_field_ijqxyz=gradient_of_adjoint_field_ijqxyz))  # minus sign is already there
    dg_du_field_analytical = discretization_fixture.get_displacement_sized_field(
        name='dg_du_field_analytical')
    K_fun(adjoint_field_inxyz, dg_du_field_analytical)

    # partial derivative of phase field gradient potential for a phase field with respect to phase-field
    # finite difference detivative
    # allocate fields
    fd_derivative = discretization_fixture.get_displacement_sized_field(name='fd_derivative_in_fd_test')
    u_inxyz_perturbed = discretization_fixture.get_displacement_sized_field(
        name='test_displacement_field_in_test_perturbed')
    stress_field_ijqxyz = discretization_fixture.get_gradient_size_field(
        name='stress_field_ijqxyz_in_test_perturbed')
    epsilons = [1e6, 1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]  #
    error_fd_vs_analytical_x = []
    error_fd_vs_analytical_y = []
    for epsilon in epsilons:
        # loop over every single element of displacement field
        for f in np.arange(discretization_fixture.cell.unknown_shape[0]):
            for n in np.arange(discretization_fixture.nb_nodes_per_pixel):
                for x in np.arange(discretization_fixture.nb_of_pixels[0]):
                    for y in np.arange(discretization_fixture.nb_of_pixels[1]):
                        u_inxyz_perturbed.s.fill(0)
                        u_inxyz_perturbed.s = np.copy(displacement_field_u_inxyz.s)

                        u_inxyz_perturbed.s[f, n, x, y] = u_inxyz_perturbed.s[f, n, x, y] + epsilon / 2
                        # set phase_field to ones
                        # compute homogenized stress field for perturbed displacement
                        discretization_fixture.get_stress_field_mugrid(
                            material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                            displacement_field_inxyz=u_inxyz_perturbed,
                            macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
                            formulation='small_strain',
                            output_stress_field_ijqxyz=stress_field_ijqxyz)

                        adjoint_energy_plus = topology_optimization.adjoint_potential(
                            discretization=discretization_fixture,
                            stress_field_ijqxyz=stress_field_ijqxyz,
                            adjoint_field_inxyz=adjoint_field_inxyz)

                        u_inxyz_perturbed.s[f, n, x, y] = u_inxyz_perturbed.s[f, n, x, y] - epsilon
                        # set phase_field to ones
                        # compute homogenized stress field for perturbed displacement
                        discretization_fixture.get_stress_field_mugrid(
                            material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                            displacement_field_inxyz=u_inxyz_perturbed,
                            macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
                            formulation='small_strain',
                            output_stress_field_ijqxyz=stress_field_ijqxyz)

                        adjoint_energy_minus = topology_optimization.adjoint_potential(
                            discretization=discretization_fixture,
                            stress_field_ijqxyz=stress_field_ijqxyz,
                            adjoint_field_inxyz=adjoint_field_inxyz)

                        fd_derivative.s[f, n, x, y] = (adjoint_energy_plus - adjoint_energy_minus) / epsilon

        # print(df_drho_analytical[0, 0])
        error_fd_vs_analytical_x.append(
            np.linalg.norm(fd_derivative.s[0, 0] - dg_du_field_analytical.s[0, 0], 'fro'))
        error_fd_vs_analytical_y.append(
            np.linalg.norm(fd_derivative.s[1, 0] - dg_du_field_analytical.s[1, 0], 'fro'))
        # print(epsilon)

        # assert error_fd_vs_analytical_x[-1] < epsilon * 100, (
        #     "Finite difference derivative do not corresponds to the analytical expression "
        #     "for partial derivative adjoint potential eq with respect to displacement ")
        # assert error_fd_vs_analytical_y[-1] < epsilon * 100, (
        #     "Finite difference derivative do not corresponds to the analytical expression "
        #     "for partial derivative  adjoint potential with respect to displacement ")
    quad_fit_of_error = np.multiply(error_fd_vs_analytical_x[4], np.asarray(epsilons) ** 2)
    lin_fit_of_error = np.multiply(error_fd_vs_analytical_x[4], np.asarray(epsilons) ** 1)

    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.loglog(epsilons, error_fd_vs_analytical_x, marker='x',
                   label=r' error_fd_vs_analytical x'.format())
        plt.loglog(epsilons, error_fd_vs_analytical_y, marker='o',
                   label=r' error_fd_vs_analytical y'.format())
        plt.loglog(epsilons, quad_fit_of_error, linestyle='--',
                   label=r' quad_fit_of_error'.format())
        plt.loglog(epsilons, lin_fit_of_error, linestyle='--',
                   label=r' lin_fit_of_error'.format())
        plt.legend(loc='best')
        plt.xlabel('epsilon - size of Finite Difference step')
        plt.ylabel('Error')
        plt.ylim([1e-16, 1e5])

        plt.show()


@pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
    ([3, 4], 0, [6, 8]),
    ([2, 5], 0, [12, 7]),
    # ([3, 4], 1, [6, 8]),
    # ([2, 5], 1, [12, 7])
])
def test_nullity_of_adjoint_potential(discretization_fixture, plot=False):
    # create material data field
    K_0, G_0 = domain.get_bulk_and_shear_modulus(E=1, poison=0.2)

    material_C_0 = domain.get_elastic_material_tensor(dim=discretization_fixture.domain_dimension,
                                                      K=K_0,
                                                      mu=G_0,
                                                      kind='linear')

    material_data_field_C_0_rho_ijklqxyz = discretization_fixture.get_material_data_size_field_mugrid(
        name='test_DATA_FIELD')

    phase_field = discretization_fixture.get_scalar_field(name='phase_field')
    phase_field.s = np.random.rand(*phase_field.s.shape) ** 1

    # evaluate to quad points
    phase_field_at_quad_poits_1qxyz = discretization_fixture.get_quad_field_scalar(
        name='   phase_field_at_quad_poits_1qxyz ')
    discretization_fixture.apply_N_operator_mugrid(phase_field, phase_field_at_quad_poits_1qxyz)
    # apply material distribution

    material_data_field_C_0_rho_ijklqxyz.s = material_C_0[..., np.newaxis, np.newaxis, np.newaxis] * \
                                             np.power(phase_field_at_quad_poits_1qxyz.s, 1)[0, 0, :, ...]

    macro_gradient = np.array([[0.1, 0], [0, 0.1]])
    macro_gradient_field = discretization_fixture.get_gradient_size_field(name='macro_gradient_field')
    discretization_fixture.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient,
                                                           macro_gradient_field_ijqxyz=macro_gradient_field)
    #
    rhs = discretization_fixture.get_unknown_size_field(name='rhs')
    discretization_fixture.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                                          macro_gradient_field_ijqxyz=macro_gradient_field,
                                          rhs_inxyz=rhs)

    def K_fun(x, Ax):
        discretization_fixture.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0_rho_ijklqxyz,
                                                          input_field_inxyz=x,
                                                          output_field_inxyz=Ax,
                                                          formulation='small_strain')

    def M_fun(x, Px):
        Px.s = 1 * x.s

    # Get valid displacement field
    displacement_field = discretization_fixture.get_displacement_sized_field(
        name='displacement_field')
    displacement_field.s.fill(0)
    solvers.conjugate_gradients_mugrid(
        comm=discretization_fixture.fft.communicator,
        fc=discretization_fixture.field_collection,
        hessp=K_fun,  # linear operator
        b=rhs,
        x=displacement_field,
        P=M_fun,
        tol=1e-10,
        maxiter=50000,
    )
    # set random adjoint field
    adjoint_field = discretization_fixture.get_displacement_sized_field(
        name='adjoint_field')
    adjoint_field.s = np.random.rand(*adjoint_field.s.shape)
    # compute stress field corresponding to equilibrated displacement
    stress_field_ijqxyz = discretization_fixture.get_gradient_size_field(
        name='stress_field_ijqxyz_in_test_perturbed')
    stress_field_ijqxyz.s.fill(0)
    discretization_fixture.get_stress_field_mugrid(
        material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
        displacement_field_inxyz=displacement_field,
        macro_gradient_field_ijqxyz=macro_gradient_field,
        output_stress_field_ijqxyz=stress_field_ijqxyz,
        formulation='small_strain'
    )

    adjoint_potential = topology_optimization.adjoint_potential(discretization_fixture, stress_field_ijqxyz,
                                                                adjoint_field)
    print(adjoint_potential)
    assert adjoint_potential < 1e-7, (
        "Adjoint potential should be 0 for every solution of equilibrium constrain"
        "but adjoint_potential = {}".format(adjoint_potential))  # this number depends on toler of CG solver !!!


# @pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
#     ([3, 4], 0, [6, 8]),
#     ([2, 5], 0, [12, 7]),
#     ([3, 4], 1, [6, 8]),
#     ([2, 5], 1, [12, 7])])
# def test_fd_check_of_adjoint_potential_wrt_displacement(discretization_fixture, plot=False):
#     epsilons = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
#     # epsilons = [1e-4]
#     fd_derivative = np.zeros([*discretization_fixture.get_displacement_sized_field().shape])
#
#     macro_gradient = np.array([[0.01, 0], [0, 0.01]])
#
#     # create material data field
#     K_0, G_0 = domain.get_bulk_and_shear_modulus(E=1, poison=0.2)
#
#     material_C_0 = domain.get_elastic_material_tensor(dim=discretization_fixture.domain_dimension,
#                                                       K=K_0,
#                                                       mu=G_0,
#                                                       kind='linear')
#
#     material_data_field_C_0 = np.einsum('ijkl,qxy->ijklqxy', material_C_0,
#                                         np.ones(np.array([discretization_fixture.nb_quad_points_per_pixel,
#                                                           *discretization_fixture.nb_of_pixels])))
#
#     phase_field = np.random.rand(*discretization_fixture.get_scalar_sized_field().shape)  # set random distribution
#     # apply material distribution
#     material_data_field_C_0 = material_data_field_C_0[..., :, :] * phase_field[0, 0]
#
#     macro_gradient_field = discretization_fixture.get_macro_gradient_field(macro_gradient)
#     #
#     rhs = discretization_fixture.get_rhs(material_data_field_C_0, macro_gradient_field)
#
#     K_fun = lambda x: discretization_fixture.apply_system_matrix(material_data_field_C_0, x,
#                                                                  formulation='small_strain')
#     M_fun = lambda x: 1 * x
#
#     displacement_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(500), toler=1e-6)
#
#     # compute stress field corresponding to equilibrated displacement
#     stress_field = discretization_fixture.get_stress_field(material_data_field_C_0, displacement_field,
#                                                            macro_gradient_field)
#
#     # create random adjoint field
#     adjoint_field = np.random.rand(
#         *discretization_fixture.get_displacement_sized_field().shape)  # set random adjoint field
#
#     # compute adjoint_potential
#     adjoint_potential = topology_optimization.adjoint_potential(discretization_fixture, stress_field, adjoint_field)
#
#     dg_du_analytical = topology_optimization.partial_derivative_of_adjoint_potential_wrt_displacement(
#         discretization_fixture,
#         material_data_field_C_0,
#         adjoint_field)
#
#     error_fd_vs_analytical = []
#     for epsilon in epsilons:
#         fd_norms = np.zeros(
#             [discretization_fixture.cell.unknown_shape[0], discretization_fixture.nb_unique_nodes_per_pixel])
#
#         # loop over every single element of displacement field
#         for f in np.arange(discretization_fixture.cell.unknown_shape[0]):
#             for n in np.arange(discretization_fixture.nb_unique_nodes_per_pixel):
#
#                 for x in np.arange(discretization_fixture.nb_of_pixels[0]):
#                     for y in np.arange(discretization_fixture.nb_of_pixels[1]):
#                         displacement_field_fnxyz = np.copy(displacement_field)
#                         displacement_field_fnxyz[f, n, x, y] = displacement_field_fnxyz[f, n, x, y] + epsilon
#                         # compute stress field corresponding to equilibrated displacement
#                         stress_field = discretization_fixture.get_stress_field(material_data_field_C_0,
#                                                                                displacement_field_fnxyz,
#                                                                                macro_gradient_field)
#
#                         adjoint_potential_perturbed = topology_optimization.adjoint_potential(discretization_fixture,
#                                                                                               stress_field,
#                                                                                               adjoint_field)
#
#                         fd_derivative[f, n, x, y] = (adjoint_potential_perturbed - adjoint_potential) / epsilon
#             fd_norms[f, n] = np.sum(np.linalg.norm((fd_derivative[f, n] - dg_du_analytical[f, n]), 'fro'))
#
#             # print('finite difference norm {0}{1} = {2}'.format(f, n, np.linalg.norm(fd_derivative[f, n], 'fro')))
#             # print('analytical derivative {0}{1} = {2}'.format(f, n, np.linalg.norm(dg_du_analytical[f, n], 'fro')))
#         # (error_fd_vs_analytical)
#
#         error_fd_vs_analytical.append(np.sum(fd_norms))
#         assert error_fd_vs_analytical[-1] < 1e-6, (
#             "Finite difference derivative  do not corresponds to the analytical expression "
#             "for partial derivative of adjoint potential  w.r.t. displacement ")


# @pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
#     ([3, 4], 0, [6, 8]),
#     ([2, 5], 0, [12, 7]),
#     ([3, 4], 0, [6, 8]),
#     ([2, 5], 0, [12, 7])])
# def test_fd_check_of_pd_objective_function_wrt_displacement_small_strain(discretization_fixture):
#     # This test compares analytical expression for partial derivative  of objective function w.r.t. displacement
#     epsilons = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
#     # epsilons = [1e-2]
#     fd_derivative = np.zeros([*discretization_fixture.get_displacement_sized_field().shape])
#     fd_derivative_wo_phase = np.zeros([*discretization_fixture.get_displacement_sized_field().shape])
#     # set stress difference to zero
#     target_stress = np.array([[1, 0.5], [0.5, 2]])
#     macro_gradient = np.array([[1, 0], [0, 1]])
#
#     ## compute objective function without perturbations
#     phase_field = discretization_fixture.get_scalar_sized_field() + 1  # Phase field has  one  value per pixel
#     phase_field[0, 0, 2:4, 2:4] = phase_field[0, 0, 2:4, 2:4] / 2  # for
#     #
#     K_1, G_1 = domain.get_bulk_and_shear_modulus(E=3, poison=0.2)
#
#     mat_1 = domain.get_elastic_material_tensor(dim=discretization_fixture.domain_dimension, K=K_1, mu=G_1,
#                                                kind='linear')
#
#     material_data_field = np.einsum('ijkl,qxy->ijklqxy', mat_1,
#                                     np.ones(np.array([discretization_fixture.nb_quad_points_per_pixel,
#                                                       *discretization_fixture.nb_of_pixels])))
#
#     # Update material data based on current Phase-field
#     material_data_field_i = (phase_field) * material_data_field
#
#     ##### solve equilibrium constrain
#     # set up system
#     macro_gradient_field = discretization_fixture.get_macro_gradient_field(macro_gradient)
#     rhs = discretization_fixture.get_rhs(material_data_field_i, macro_gradient_field)
#
#     K_fun = lambda x: discretization_fixture.apply_system_matrix(material_data_field_i, x)
#     M_fun = lambda x: 1 * x
#
#     displacement_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(500), toler=1e-6)
#
#     # test homogenized stress
#     homogenized_stress = discretization_fixture.get_homogenized_stress(
#         material_data_field_ijklqxyz=material_data_field_i,
#         displacement_field_inxyz=displacement_field,
#         macro_gradient_field_ijqxyz=macro_gradient_field,
#         formulation='small_strain')
#
#     actual_stress_field = np.zeros(discretization_fixture.gradient_size)
#     actual_stress_field[..., :] = homogenized_stress[(...,) + (np.newaxis,) * (actual_stress_field.ndim - 2)]
#
#     stress_diff = target_stress - homogenized_stress
#     # objective function  without phase-field
#     f_sigma = np.sum(stress_diff ** 2)
#     # objective function
#     objective_function = topology_optimization.objective_function_small_strain(
#         discretization=discretization_fixture,
#         actual_stress_ij=homogenized_stress,
#         target_stress_ij=target_stress,
#         phase_field_1nxyz=phase_field,
#         eta=1,
#         w=1)
#
#     d_of_d_u_analytical = topology_optimization.partial_der_of_objective_function_wrt_displacement_small_strain(
#         discretization_fixture,
#         material_data_field_i,
#         stress_diff,
#         eta=1,
#         w=1)
#
#     error_fd_vs_analytical = []
#     for epsilon in epsilons:
#         fd_norms = np.zeros(
#             [discretization_fixture.cell.unknown_shape[0], discretization_fixture.nb_unique_nodes_per_pixel])
#         fd_norms_wo_phase = np.zeros(
#             [discretization_fixture.cell.unknown_shape[0], discretization_fixture.nb_unique_nodes_per_pixel])
#
#         # loop over every single element of displacement field
#         for f in np.arange(discretization_fixture.cell.unknown_shape[0]):
#             for n in np.arange(discretization_fixture.nb_unique_nodes_per_pixel):
#                 # loop over every single element of phase field
#                 for x in np.arange(discretization_fixture.nb_of_pixels[0]):
#                     for y in np.arange(discretization_fixture.nb_of_pixels[1]):
#                         # perturb f,n,x,y,z-component displacement field with epsilon
#                         displacement_field_fnxyz = np.copy(displacement_field)
#                         displacement_field_fnxyz[f, n, x, y] = displacement_field_fnxyz[f, n, x, y] + epsilon
#
#                         # homogenized stress
#                         homogenized_stress = discretization_fixture.get_homogenized_stress(
#                             material_data_field_ijklqxyz=material_data_field_i,
#                             displacement_field_inxyz=displacement_field_fnxyz,
#                             macro_gradient_field_ijqxyz=macro_gradient_field,
#                             formulation='small_strain')
#
#                         stress_diff = target_stress - homogenized_stress
#                         # objective function without phase-field
#                         f_sigma_perturbed = np.sum(stress_diff ** 2)
#
#                         # objective function
#                         objective_function_perturbed = topology_optimization.objective_function_small_strain(
#                             discretization=discretization_fixture,
#                             actual_stress_ij=homogenized_stress,
#                             target_stress_ij=target_stress,
#                             phase_field_1nxyz=phase_field,
#                             eta=1,
#                             w=1)
#
#                         fd_derivative_wo_phase[f, n, x, y] = (f_sigma_perturbed - f_sigma) / epsilon
#                         fd_derivative[f, n, x, y] = (objective_function_perturbed - objective_function) / epsilon
#
#                 fd_norms[f, n] = np.sum(np.linalg.norm((fd_derivative[f, n] - d_of_d_u_analytical[f, n]), 'fro'))
#                 fd_norms_wo_phase[f, n] = np.sum(
#                     np.linalg.norm((fd_derivative_wo_phase[f, n] - d_of_d_u_analytical[f, n]), 'fro'))
#         # print('finite difference norm {0}{1} = {2}'.format(f, n, np.linalg.norm(fd_derivative[f, n], 'fro')))
#         # print('analytical derivative {0}{1} = {2}'.format(f, n, np.linalg.norm(d_of_d_u_analytical[f, n], 'fro')))
#         # assert np.allclose(fd_norms, fd_norms_wo_phase), (
#         #     "Finite difference derivative with phase field do not corresponds to the one without phase field "
#         #     "for partial derivative of adjoint potential  w.r.t. displacement "
#         #     "epsilon = {}".format(epsilon))
#
#         error_fd_vs_analytical.append(np.sum(fd_norms))
#
#         assert error_fd_vs_analytical[-1] < epsilon * 10, (
#             "Finite difference derivative  do not corresponds to the analytical expression "
#             "for partial derivative of adjoint potential  w.r.t. displacement ")


# # TODO set p as a parameter
# @pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
#     ([3, 4], 0, [6, 8]),
#     ([2, 5], 0, [12, 7]),
#     ([3, 4], 1, [6, 8]),
#     ([2, 5], 1, [12, 7])])
# def test_fd_check_of_adjoint_potential_wrt_phase_field_pixel(discretization_fixture):
#     epsilons = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
#     # epsilons = [1e-4]
#     p = 1
#     fd_derivative = discretization_fixture.get_scalar_sized_field()
#
#     macro_gradient = np.array([[0.1, 0], [0, 0.1]])
#
#     # create material data field
#     K_0, G_0 = domain.get_bulk_and_shear_modulus(E=1, poison=0.2)
#
#     material_C_0 = domain.get_elastic_material_tensor(dim=discretization_fixture.domain_dimension,
#                                                       K=K_0,
#                                                       mu=G_0,
#                                                       kind='linear')
#
#     material_data_field_C_0 = np.einsum('ijkl,qxy->ijklqxy', material_C_0,
#                                         np.ones(np.array([discretization_fixture.nb_quad_points_per_pixel,
#                                                           *discretization_fixture.nb_of_pixels])))
#
#     phase_field = np.random.rand(
#         *discretization_fixture.get_scalar_sized_field().shape)  # * 0 + 1  # set random distribution
#     # apply material distribution
#     material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * np.power(phase_field[0, 0],
#                                                                                 p)
#
#     # Set up the equilibrium system
#     macro_gradient_field = discretization_fixture.get_macro_gradient_field(macro_gradient)
#
#     # create random displacement field
#     displacement_field = np.random.rand(*discretization_fixture.get_scalar_sized_field().shape)
#
#     # ----------------------------------------------------------------------
#     # compute stress field corresponding to displacement
#     stress_field = discretization_fixture.get_stress_field(material_data_field_C_0_rho,
#                                                            displacement_field,
#                                                            macro_gradient_field)
#
#     # create random adjoint field
#     adjoint_field = np.random.rand(
#         *discretization_fixture.get_displacement_sized_field().shape)  # set random adjoint field
#
#     # compute adjoint_potential
#     adjoint_potential = topology_optimization.adjoint_potential(discretization_fixture,
#                                                                 stress_field,
#                                                                 adjoint_field)
#
#     dg_drho_analytical = topology_optimization.partial_derivative_of_adjoint_potential_wrt_phase_field(
#         discretization=discretization_fixture,
#         material_data_field_ijklqxyz=material_data_field_C_0,
#         displacement_field_fnxyz=displacement_field,
#         macro_gradient_field_ijqxyz=macro_gradient_field,
#         phase_field_1nxyz=phase_field,
#         adjoint_field_fnxyz=adjoint_field,
#         p=p)
#
#     error_fd_vs_analytical = []
#     for epsilon in epsilons:
#         # loop over every single element of displacement field
#         for x in np.arange(discretization_fixture.nb_of_pixels[0]):
#             for y in np.arange(discretization_fixture.nb_of_pixels[1]):
#                 phase_field_perturbed = np.copy(phase_field)
#                 phase_field_perturbed[0, 0, x, y] = phase_field_perturbed[0, 0, x, y] + epsilon
#                 # compute stress field corresponding to equilibrated displacement
#
#                 # apply material distribution
#                 material_data_field_C_0_per = material_data_field_C_0[..., :, :] * np.power(phase_field_perturbed[0, 0],
#                                                                                             p)
#                 #
#                 stress_field = discretization_fixture.get_stress_field(
#                     material_data_field_ijklqxyz=material_data_field_C_0_per,
#                     displacement_field_inxyz=displacement_field,
#                     macro_gradient_field_ijqxyz=macro_gradient_field,
#                     formulation='small_strain')
#
#                 adjoint_potential_perturbed = topology_optimization.adjoint_potential(
#                     discretization=discretization_fixture,
#                     stress_field_ijqxyz=stress_field,
#                     adjoint_field_fnxyz=adjoint_field)
#
#                 fd_derivative[0, 0, x, y] = (adjoint_potential_perturbed
#                                              -
#                                              adjoint_potential) / epsilon
#
#         fd_norm = np.sum(np.linalg.norm((fd_derivative[0, 0] - dg_drho_analytical), 'fro'))
#
#         # print('finite difference norm {0}{1} = {2}'.format(f, n, np.linalg.norm(fd_derivative[f, n], 'fro')))
#         # print('analytical derivative {0}{1} = {2}'.format(f, n, np.linalg.norm(dg_du_analytical[f, n], 'fro')))
#         # (error_fd_vs_analytical)
#
#         error_fd_vs_analytical.append(fd_norm)
#         print('error_fd_vs_analytical: {}'.format(error_fd_vs_analytical))
#         assert error_fd_vs_analytical[-1] < epsilon * 1e0, (
#             "Finite difference derivative  do not corresponds to the analytical expression "
#             "for partial derivative of adjoint potential  w.r.t. displacement "
#             "for epsilon = {} and p = {}".format(epsilon, p))


@pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
    ([3, 4], 0, [6, 8]),
    ([2, 5], 0, [12, 7]),
    ([3, 4], 0, [6, 8]),
    ([2, 5], 0, [12, 7])  # TODO add gaus quadrature for bilinear elements
])
def NOT_WORKING_test_fd_check_of_objective_function_wrt_phase_field(discretization_fixture, plot=False):
    # f = (flux_h -flux_target)^2 + w*eta* int (  (grad(rho))^2 )dx  +    int ( rho^2(1-rho)^2 ) / eta   dx
    # f =  f_sigma + w*(eta* f_rho_grad  + f_dw/eta)
    # f= objective_function_small_strain
    # f_grad
    #
    epsilons = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
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
    # material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * np.power(phase_field[0, 0],
    #                                                                            p)
    phase_field_at_quad_poits_1qnxyz = discretization_fixture.evaluate_field_at_quad_points(
        nodal_field_fnxyz=phase_field,
        quad_field_fqnxyz=None,
        quad_points_coords_iq=None)[0]

    material_data_field_C_0_rho = material_data_field_C_0[..., :, :, :] * (
        np.power(phase_field_at_quad_poits_1qnxyz[0, :, 0, ...], (p)))

    # Set up the equilibrium system
    macro_gradient_field = discretization_fixture.get_macro_gradient_field(macro_gradient)

    # create random displacement field
    displacement_field = np.random.rand(*discretization_fixture.get_scalar_sized_field().shape)

    # ----------------------------------------------------------------------
    # compute stress field corresponding to displacement
    homogenized_stress = discretization_fixture.get_homogenized_stress(
        material_data_field_ijklqxyz=material_data_field_C_0_rho,
        displacement_field_inxyz=displacement_field,
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
    df_drho_analytical_FE = topology_optimization.partial_derivative_of_objective_function_wrt_phase_field_FE(
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
    error_fd_vs_analytical_FE = []
    for epsilon in epsilons:
        # loop over every single element of phase field
        for x in np.arange(discretization_fixture.nb_of_pixels[0]):
            for y in np.arange(discretization_fixture.nb_of_pixels[1]):
                # set phase_field to ones
                phase_field_perturbed = np.copy(phase_field)  # Phase field has  one  value per pixel
                # phase_field_perturbed=phase_field_perturbed**p
                phase_field_perturbed[0, 0, x, y] = phase_field_perturbed[0, 0, x, y] + epsilon / 2

                # apply material distribution
                # material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * np.power(phase_field_perturbed[0, 0],
                #                                                                           p)  # ** p
                # apply material distribution
                phase_field_at_quad_poits_1qnxyz = discretization_fixture.evaluate_field_at_quad_points(
                    nodal_field_fnxyz=phase_field_perturbed,
                    quad_field_fqnxyz=None,
                    quad_points_coords_iq=None)[0]

                material_data_field_drho_ijklqxyz = material_data_field_C_0[..., :, :, :] * (
                    np.power(phase_field_at_quad_poits_1qnxyz[0, :, 0, ...], (p)))
                # plus epsilon
                homogenized_stress_plus = discretization_fixture.get_homogenized_stress(
                    material_data_field_ijklqxyz=material_data_field_drho_ijklqxyz,
                    displacement_field_inxyz=displacement_field,
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
                phase_field_at_quad_poits_1qnxyz = discretization_fixture.evaluate_field_at_quad_points(
                    nodal_field_fnxyz=phase_field_perturbed,
                    quad_field_fqnxyz=None,
                    quad_points_coords_iq=None)[0]

                material_data_field_drho_ijklqxyz = material_data_field_C_0[..., :, :, :] * (
                    np.power(phase_field_at_quad_poits_1qnxyz[0, :, 0, ...], (p)))

                # material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * np.power(phase_field_perturbed[0, 0],
                #                                                                 p)  # ** p
                homogenized_stress_minus = discretization_fixture.get_homogenized_stress(
                    material_data_field_ijklqxyz=material_data_field_drho_ijklqxyz,
                    displacement_field_inxyz=displacement_field,
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
        fd_norm_FE = np.sum(np.linalg.norm((fd_derivative[0, 0] - df_drho_analytical_FE[0, 0]), 'fro'))

        # print(f_rho_grad_potential_analytical)
        error_fd_vs_analytical.append(fd_norm)
        error_fd_vs_analytical_FE.append(fd_norm_FE)

        print(error_fd_vs_analytical)
        print(error_fd_vs_analytical_FE)
        assert error_fd_vs_analytical_FE[-1] < epsilon * 10, (
            "Finite difference derivative do not corresponds to the analytical expression "
            "for partial derivative of gradient of phase-field potential "
            "error_fd_vs_analytical = {}, \n "
            "epsilon = {},".format(error_fd_vs_analytical_FE, epsilon))
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.loglog(epsilons, error_fd_vs_analytical,
                   label=r' error_fd_vs_analytical'.format())
        plt.loglog(epsilons, error_fd_vs_analytical_FE,
                   label=r' error_fd_vs_analytical_FE'.format())

        plt.legend(loc='best')

        plt.show()


@pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
    ([3, 4], 0, [6, 8]),
    ([2, 5], 0, [12, 7]),
    ([3, 4], 0, [6, 8]),
    ([2, 5], 0, [12, 7])
])
def NOT_WORKING_test_fd_check_of_objective_function_with_adjoin_potential_wrt_phase_field(discretization_fixture):
    # f = (flux_h -flux_target)^2 + w*eta* int (  (grad(rho))^2 )dx  +    int ( rho^2(1-rho)^2 ) / eta   dx
    # f =  f_sigma + w*(eta* f_rho_grad  + f_dw/eta)
    # f= objective_function_small_strain
    # f_grad
    #
    epsilons = [1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    # epsilons = [1e-4]
    p = 1
    fd_derivative = discretization_fixture.get_scalar_field(name='fd_derivative')
    target_stress = np.array([[1, 0.3], [0.3, 2]])
    macro_gradient = np.array([[0.2, 0], [0, 0.2]])

    # create material data field
    K_0, G_0 = domain.get_bulk_and_shear_modulus(E=1, poison=0.2)

    elastic_C_1 = domain.get_elastic_material_tensor(dim=discretization_fixture.domain_dimension,
                                                     K=K_0,
                                                     mu=G_0,
                                                     kind='linear')

    material_data_field_C_0_rho = discretization_fixture.get_material_data_size_field(name='test_DATA_FIELD')

    # phase_field = discretization_fixture.get_scalar_sized_field() + 1  #
    phase_field = discretization_fixture.get_scalar_field(
        name='phase_field')
    phase_field.s = np.random.rand(*phase_field.s.shape) ** 1
    #
    phase_field_at_quad_poits_1qnxyz, N_at_quad_points_qnijk = discretization_fixture.evaluate_field_at_quad_points(
        nodal_field_fnxyz=phase_field,
        quad_field_fqnxyz=None,
        quad_points_coords_iq=None)

    # apply material distribution
    material_data_field_C_0_rho.s = elastic_C_1[..., np.newaxis, np.newaxis, np.newaxis] * \
                                    np.power(phase_field_at_quad_poits_1qnxyz, p)[0, :, 0, ...]

    # Set up the equilibrium system
    macro_gradient_field = discretization_fixture.get_gradient_size_field(name='macro_gradient_field')
    macro_gradient_field = discretization_fixture.get_macro_gradient_field(macro_gradient_ij=macro_gradient,
                                                                           macro_gradient_field_ijqxyz=macro_gradient_field
                                                                           )
    # Solve mechanical equilibrium constrain
    rhs = discretization_fixture.get_unknown_size_field(name='rhs_field')
    rhs = discretization_fixture.get_rhs(material_data_field_ijklqxyz=material_data_field_C_0_rho,
                                         macro_gradient_field_ijqxyz=macro_gradient_field,
                                         rhs_inxyz=rhs)

    K_fun = lambda x: discretization_fixture.apply_system_matrix(material_data_field_C_0_rho, x,
                                                                 formulation='small_strain')
    M_fun = lambda x: 1 * x

    # Solve mechanical equilibrium constrain
    displacement_field = discretization_fixture.get_displacement_sized_field(
        name='test_displacement_field_in_test')
    displacement_field.s.fill(0)
    displacement_field.s, norms = solvers.PCG(K_fun, rhs.s, x0=None, P=M_fun, steps=int(500), toler=1e-12)

    # ----------------------------------------------------------------------
    # compute homogenized stress field corresponding to displacement
    homogenized_stress = discretization_fixture.get_homogenized_stress(
        material_data_field_ijklqxyz=material_data_field_C_0_rho,
        displacement_field_inxyz=displacement_field,
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
        material_data_field_ijklqxyz=material_data_field_C_0_rho,
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

    dg_drho_analytical = topology_optimization.partial_derivative_of_adjoint_potential_wrt_phase_field_pixel(
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
    nb_pixels = (10, 10)
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
    f_dw_0 = topology_optimization.compute_double_well_potential_interpolated(discretization=discretization,
                                                                              phase_field_1nxyz=phase_field_0, eta=eta)
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
        plt.plot(nodal_coordinates[0, 0, :, 0], phase_field_0[0, 0, :, 0], label='phase_field_0')
        # plt.show()

    for nb_pixel_x in [10, 20, 150]:  # ,160,320
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

            plt.plot(nodal_coordinates_k[0, 0, :, 0], integrant[:, 0],
                     label=r' $16 * (\rho^2)(1-\rho)^2, grid$ = {}'.format(nb_pixel_x))
            # plt.plot(nodal_coordinates_k[0, 0, :, 0], grad_integrant_fnxyz[0, :], label='df_rho/drho')
            # plt.plot(nodal_coordinates_k[0, 0, :, 0], phase_field_k[0, 0, :, 0], label='phase_field_k')

            # segs1 = np.stack((nodal_coordinates[0, 0], nodal_coordinates[1, 0]), axis=2)
            # segs2 = segs1.transpose(1, 0, 2)
        #   ax.legend()
        f_dw_old = topology_optimization.compute_double_well_potential(discretization=discretization_k,
                                                                       phase_field_1nxyz=phase_field_k,
                                                                       eta=eta)

        f_dw = topology_optimization.compute_double_well_potential_interpolated(discretization=discretization_k,
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
