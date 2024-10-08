import unittest

import numpy as np

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import topology_optimization


class SensitivityTestCase(unittest.TestCase):
    def NOtest_objective_function_stress_part(self):
        # setup unit cell
        domain_size = [4, 5]
        problem_type = 'elasticity'
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)

        number_of_pixels = np.random.randint(2, 10, size=2)

        discretization_type = 'finite_element'
        for element_type in ['linear_triangles', 'bilinear_rectangle']:
            discretization = domain.Discretization(cell=my_cell,
                                                   number_of_pixels=number_of_pixels,
                                                   discretization_type=discretization_type,
                                                   element_type=element_type)

            # test stress difference
            # set phase_field to zero
            phase_field = discretization.get_scalar_sized_field()  # Phase field has  one  value per pixel
            # phase_field = np.random.rand(
            #     *discretization.get_unknown_size_field().shape)

            target_stress = np.random.rand(2, 2)

            actual_stress = np.random.rand(2, 2)
            actual_stress_field = np.zeros(discretization.gradient_size)
            actual_stress_field[..., :] = actual_stress[(...,) + (np.newaxis,) * (actual_stress_field.ndim - 2)]

            objective_stress = topology_optimization.objective_function_small_strain(discretization,
                                                                                     actual_stress=actual_stress_field,
                                                                                     target_stress=target_stress,
                                                                                     phase_field=phase_field,
                                                                                     eta=1, w=1)
            # objective_stress_analytical = int_omega (actual_stress - target_stress) ^ 2 dx / size Omega
            objective_stress_analytical = discretization.cell.domain_volume * np.sum(
                np.power((target_stress - actual_stress), 2)) / discretization.cell.domain_volume

            self.assertAlmostEqual(objective_stress, objective_stress_analytical, 13,
                                   'Objective_stress is not equal to analytical expression for 2D element {} in {} problem '.format(
                                       element_type, problem_type))

    def NOtest_objective_function_double_well_potential_part(self, plot=False):
        # test double well potential. for constant phase field
        # shape of double well potential
        # number of pixel independence
        # domain size independence is NOT preserved !!!

        # setup unit cell
        domain_size = [15, 100]  # [4, 5]  #  [10, 10]
        problem_type = 'elasticity'
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)
        discretization_type = 'finite_element'
        for element_type in ['linear_triangles', 'bilinear_rectangle']:

            rhos = np.arange(-0.5, 1.5, 0.1)
            of_phase = np.zeros(rhos.shape)
            of_phase_last = np.zeros(rhos.shape)
            for size in np.arange(10):  # for 10 random sizes it has to be equal
                number_of_pixels = np.random.randint(2, 10, size=2)
                # print(number_of_pixels)
                discretization = domain.Discretization(cell=my_cell,
                                                       number_of_pixels=number_of_pixels,
                                                       discretization_type=discretization_type,
                                                       element_type=element_type)

                # test phase_field
                # set stress difference to zero
                target_stress = np.random.rand(2, 2)

                actual_stress = target_stress
                actual_stress_field = np.zeros(discretization.gradient_size)
                actual_stress_field[..., :] = actual_stress[(...,) + (np.newaxis,) * (actual_stress_field.ndim - 2)]

                # set phase_field constant such that gradient is zero

                for index in np.arange(rhos.size):
                    phase_field = rhos[index] + discretization.get_scalar_sized_field()
                    of_phase[index] = topology_optimization.objective_function_small_strain(discretization,
                                                                                            actual_stress_ij=actual_stress_field,
                                                                                            target_stress_ij=target_stress,
                                                                                            phase_field_1nxyz=phase_field,
                                                                                            eta=1, w=1)
                if size != 0:
                    value = np.allclose(of_phase, of_phase_last, rtol=1e-16, atol=1e-14)
                    # print(of_phase)
                    # print(of_phase_last)
                    self.assertTrue(value,
                                    'Objective_stress is not equal to analytical expression for 2D element {} in {} '
                                    'problem, size {} '.format(
                                        element_type, problem_type, domain_size))

                of_phase_last = np.copy(of_phase)

            # double well should be 0 at 0 and 1
            phase_field = 0 + discretization.get_scalar_sized_field()
            of_phase_0 = topology_optimization.objective_function_small_strain(discretization,
                                                                               actual_stress_ij=actual_stress_field,
                                                                               target_stress_ij=target_stress,
                                                                               phase_field_1nxyz=phase_field,
                                                                               eta=1, w=1)

            self.assertAlmostEqual(of_phase_0, 0, 16,
                                   'Objective function: double well is not equal to 0 at 0'
                                   ' for 2D element {} in {}  problem, size {} '.format(
                                       element_type, problem_type, domain_size))
            phase_field = 1 + discretization.get_scalar_sized_field()
            of_phase_1 = topology_optimization.objective_function_small_strain(discretization,
                                                                               actual_stress_ij=actual_stress_field,
                                                                               target_stress_ij=target_stress,
                                                                               phase_field_1nxyz=phase_field,
                                                                               eta=1, w=1)

            self.assertAlmostEqual(of_phase_1, 0, 16,
                                   'Objective function: double well is not equal to 0 at 1'
                                   ' for 2D element {} in {} problem, size {} '.format(
                                       element_type, problem_type, domain_size))

        if plot:
            import matplotlib.pyplot as plt
            plt.plot(rhos, of_phase)
            plt.show()

    def nottest_objective_function_phase_field_gradient_part(self):
        # test hase_field_gradient potential
        # shape of double well potential
        # number of pixel independence
        # domain size independence is NOT preserved !!!
        # setup unit cell
        domain_size = [1, 1]
        problem_type = 'elasticity'
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)
        # TODO Test phase field gradient correctly

        discretization_type = 'finite_element'
        for element_type in ['linear_triangles']:  # , 'bilinear_rectangle'
            for size in np.arange(2, 8):  # for 10 random sizes it has to be equal
                Nx = 2 + (size + 1)
                number_of_pixels = [Nx, 5]
                print('Nb pixels= {}'.format(number_of_pixels))
                print('Nb pixels total = {}'.format(np.prod(number_of_pixels)))
                # print(number_of_pixels)
                discretization = domain.Discretization(cell=my_cell,
                                                       number_of_pixels=number_of_pixels,
                                                       discretization_type=discretization_type,
                                                       element_type=element_type)
                # set stress difference to zero
                target_stress = np.random.rand(2, 2)

                actual_stress = target_stress
                actual_stress_field = np.zeros(discretization.gradient_size)
                actual_stress_field[..., :] = actual_stress[(...,) + (np.newaxis,) * (actual_stress_field.ndim - 2)]

                # set a density phase field
                # compute gradient of density phase field
                nodal_coordinates = discretization.get_nodal_points_coordinates()
                quad_coordinates = discretization.get_quad_points_coordinates()

                u_fun_4x3y = lambda x, y: 1 * x / domain_size[0]  # + 3 * y
                du_fun_4 = lambda y: 1
                du_fun_3 = lambda x: 1

                phase_field = discretization.get_scalar_sized_field()
                phase_field_gradient = discretization.get_temperature_gradient_size_field()

                phase_field[0, 0, :, :] = u_fun_4x3y(nodal_coordinates[0, 0, :, :],
                                                     nodal_coordinates[1, 0, :, :])
                # temperature_gradient_anal[0, 0, :, :, :] = du_fun_4(quad_coordinates[1, :, :, :])
                # temperature_gradient_anal[0, 1, :, :, :] = du_fun_3(quad_coordinates[0, :, :, :])

                phase_field_gradient = discretization.apply_gradient_operator(phase_field, phase_field_gradient)
                # test 1
                average = np.ndarray.sum(phase_field_gradient)
                message = "Gradient of phase field does not have zero mean !!!! for 2D element {} in {} problem".format(
                    element_type,
                    problem_type)

                # print(discretization.integrate_over_cell(phase_field_gradient))
                # print(discretization.integrate_over_cell(phase_field_gradient ** 2))
                #   print(discretization.integrate_over_cell(
                #       (np.prod(discretization.pixel_size) ** 2 * phase_field_gradient) ** 2))
                #   phase_field_gradient_exp = discretization.get_temperature_gradient_size_field()
                #   phase_field_gradient_exp[0, 0] = discretization.pixel_size[0] *discretization.pixel_size[0] * phase_field_gradient[0, 0]
                #   phase_field_gradient_exp[0, 1] = discretization.pixel_size[1] *discretization.pixel_size[1] * phase_field_gradient[0, 1]
                #
                # #  print('Pixel size= {}'.format(discretization.pixel_size))
                #  print('Pixel volume= {}'.format(np.prod(discretization.pixel_size)))
                #   print('Pixel volume **2 = {}'.format(np.prod(discretization.pixel_size)**2))

                # print(phase_field_gradient_exp[0, 0])
                #  print('int (( Pixel_volume**2  grad rho)**2)= {}'.format(discretization.integrate_over_cell(phase_field_gradient_exp ** 2)))

                f_rho_grad = np.sum(discretization.integrate_over_cell(phase_field_gradient ** 2))
                f_rho_grad_scaled = np.sum(discretization.integrate_over_cell(
                    (np.prod(discretization.pixel_size) * phase_field_gradient) ** 2))
                # if size != 0:
                #   value = np.allclose(f_rho_grad, f_rho_grad_last, rtol=1e-16, atol=1e-14)
                # print(phase_field_gradient[0, 0, 0,])
                # print(phase_field_gradient[0, 1, 0])
                #   print('sum (int ((grad rho)**2))= {}'.format(f_rho_grad))
                #   print(f_rho_grad_scaled)
                #  print(f_rho_grad_last)
                # self.assertTrue(value,
                #                 'Phase field gradient is not equal NB pixel independant for 2D element {} in {} '
                #                 'problem, size {} '.format(
                #                     element_type, problem_type, domain_size))

                f_rho_grad_last = np.copy(f_rho_grad)
                phase_field_gradient_sc = discretization.get_temperature_gradient_size_field()
                phase_field_gradient_sc = discretization.apply_gradient_operator(phase_field, phase_field_gradient_sc)
                phase_field_gradient_sc = np.einsum('ijq...,q->ijq...', phase_field_gradient_sc,
                                                    discretization.quadrature_weights)

                DtwDphase_field = discretization.get_scalar_sized_field()
                DtwDphase_field = discretization.apply_gradient_transposed_operator(phase_field_gradient_sc,
                                                                                    DtwDphase_field)

                scal = np.sum(phase_field * DtwDphase_field)

            # print(scal)

            # print('end loop')

    def Notest_df_drho(self):
        # test   df_drho   partial der of objective function with respect to density
        # this has 3 parts
        # df_stress__drho
        # df_dw__drho
        #   gradient phase field potential = int ((2 * phase_field( + 2 * phase_field^2  -  3 * phase_field +1 )) )/eta   dx
        # df_dw__drho

        # setup unit cell
        domain_size = [3, 4]  # [4, 5]  #  [10, 10]
        problem_type = 'elasticity'
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)
        discretization_type = 'finite_element'
        for element_type in ['linear_triangles', 'bilinear_rectangle']:
            number_of_pixels = [4, 5]
            discretization = domain.Discretization(cell=my_cell,
                                                   number_of_pixels=number_of_pixels,
                                                   discretization_type=discretization_type,
                                                   element_type=element_type)
            K_1, G_1 = domain.get_bulk_and_shear_modulus(E=1, poison=0.2)

            mat_0 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension, K=K_1, mu=G_1,
                                                       kind='linear')

            material_data_field_0 = np.einsum('ijkl,qxy->ijklqxy', mat_0,
                                              np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                                *discretization.nb_of_pixels])))
            ## TODO we have to evaluate phase field in quad points
            # 1 Set random phase field
            phase_field = np.random.rand(*discretization.get_temperature_sized_field().shape)

            # material data respecting phase distribution

            material_data_field_rho = material_data_field_0 * phase_field
            # np.einsum('ijkl,kl->ij', mat_0,np.array([[1, 0], [0, 0]]))

            target_stress = np.array([[2, 0], [0, 0.5]])
            target_stress = (target_stress + target_stress.transpose()) / 2

            macro_strain = np.array([[1, 0], [0, 0]])

            ##### solve equilibrium constrain
            # set up system
            macro_strain_field = discretization.get_macro_gradient_field(macro_strain)
            rhs = discretization.get_rhs(material_data_field_rho, macro_strain_field)

            K_fun = lambda x: discretization.apply_system_matrix(material_data_field_rho, x)
            M_fun = lambda x: 1 * x
            # solve the system
            solution_i, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(500), toler=1e-6)

            # get homogenized flux
            homogenized_stress = discretization.get_homogenized_stress(
                material_data_field_ijklqxyz=material_data_field_rho,
                displacement_field_fnxyz=solution_i,
                macro_gradient_field_ijqxyz=macro_strain_field)
            actual_stress = discretization.get_stress_field(material_data_field_ijklqxyz=material_data_field_rho,
                                                            displacement_field_fnxyz=solution_i,
                                                            macro_gradient_field_ijqxyz=macro_strain_field)

            actual_stress = target_stress
            actual_stress_field = np.zeros(discretization.gradient_size)
            actual_stress_field[..., :] = actual_stress[(...,) + (np.newaxis,) * (actual_stress_field.ndim - 2)]

            # set phase_field constant such that gradient is zero

            phase_field = phase_field = np.random.rand(*discretization.get_unknown_size_field().shape)

            of = topology_optimization.objective_function_small_strain(discretization,
                                                                       actual_stress_ij=actual_stress_field,
                                                                       target_stress_ij=target_stress,
                                                                       phase_field_1nxyz=phase_field,
                                                                       eta=1, w=1)

            # dfdw_drho_field = 2 * phase_field * (
            #         2 * phase_field * phase_field - 3 * phase_field + 1)  # TODO check which one is valid
            # int_dfdw_drho = ((dfdw_drho_field) / np.prod(dfdw_drho_field.shape)) * discretization.cell.domain_volume
