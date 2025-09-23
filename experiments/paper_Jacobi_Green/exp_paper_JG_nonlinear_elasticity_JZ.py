import numpy as np
import scipy as sc
import time
import os
from NuMPI.IO import save_npy, load_npy
from mpi4py import MPI

import matplotlib as mpl
from matplotlib import pyplot as plt

from muFFTTO import domain
from muFFTTO import solvers

script_name = 'exp_paper_JG_nonlinear_elasticity_JZ'
folder_name = '../exp_data/'

enforce_mean = True
for preconditioner_type in ['Jacobi_Green', 'Green']:
    for nnn in 2 ** np.array([3, 4, 5, 6, 7, 8, 9]):
        number_of_pixels = (nnn, nnn, 1)  # (128, 128, 1)  # (32, 32, 1) # (64, 64, 1)  # (128, 128, 1) #
        domain_size = [1, 1, 1]

        Nx = number_of_pixels[0]
        Ny = number_of_pixels[1]
        Nz = number_of_pixels[2]

        save_results = True
        _info = {}

        problem_type = 'elasticity'
        discretization_type = 'finite_element'
        element_type = 'trilinear_hexahedron'  # 'trilinear_hexahedron' #'trilinear_hexahedron_1Q'
        formulation = 'small_strain'
        print(f'preconditioer {preconditioner_type}')

        _info['problem_type'] = problem_type
        _info['discretization_type'] = discretization_type
        _info['element_type'] = element_type
        _info['formulation'] = formulation
        _info['preconditioner_type'] = preconditioner_type

        file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
        if not os.path.exists(file_folder_path):
            os.makedirs(file_folder_path)
        data_folder_path = (file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                            + f'_{preconditioner_type}' + '/')
        if not os.path.exists(data_folder_path):
            os.makedirs(data_folder_path)
        figure_folder_path = (file_folder_path + '/figures/' + script_name + '/' f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                              + f'_{preconditioner_type}' + '/')
        if not os.path.exists(figure_folder_path):
            os.makedirs(figure_folder_path)

        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)

        discretization = domain.Discretization(cell=my_cell,
                                               nb_of_pixels_global=number_of_pixels,
                                               discretization_type=discretization_type,
                                               element_type=element_type)

        _info['nb_of_pixels'] = discretization.nb_of_pixels_global
        _info['domain_size'] = domain_size

        start_time = time.time()

        # identity tensor                                               [single tensor]
        i = np.eye(discretization.domain_dimension)
        I = np.einsum('ij,xyz', i, np.ones(number_of_pixels))

        # identity tensors                                            [grid of tensors]
        I4 = np.einsum('il,jk', i, i)
        I4rt = np.einsum('ik,jl', i, i)
        II = np.einsum('ij...  ,kl...  ->ijkl...', i, i)
        I4s = (I4 + I4rt) / 2.
        I4d = (I4s - II / 3.)

        # II_qxyz = np.broadcast_to(II[..., np.newaxis, np.newaxis, np.newaxis, np.newaxis],
        #                           (3, 3, 3, 3, discretization.nb_quad_points_per_pixel, *number_of_pixels))
        #
        # I4d_qxyz = np.broadcast_to(I4d[..., np.newaxis, np.newaxis, np.newaxis, np.newaxis],
        #                            (3, 3, 3, 3, discretization.nb_quad_points_per_pixel, *number_of_pixels))

        model_parameters_non_linear = {'K': 2,
                                       'mu': 1,
                                       'sig0': 0.5,
                                       'eps0': 0.1,
                                       'n': 10.0}

        model_parameters_linear = {'K': 2,
                                   'mu': 1}

        _info['model_parameters_non_linear'] = model_parameters_non_linear
        _info['model_parameters_linear'] = model_parameters_linear


        # linear elasticity
        # -----------------
        def linear_elastic_q_points(strain_ijqxyz, **kwargs):
            # parameters
            K = kwargs['K']
            mu = kwargs['mu']  # mu = 1.  # shear modulus
            # bulk  modulus

            # elastic stiffness tensor, and stress response
            # C4 = K * II_qxyz + 2. * mu * I4d_qxyz
            C4 = K * II + 2. * mu * I4d
            sig = np.einsum('ijkl,lkqxyz  ->ijqxyz  ', C4, strain_ijqxyz)
            # sig = ddot42(C4, strain)

            return sig, C4


        ###
        def nonlinear_elastic_q_points(strain, **kwargs):
            # K = 2.  # bulk modulus
            # sigma = K*trace(small_strain)*I_ij  + sigma_0* (strain_eq/epsilon_0)^n * N_ijkl
            K = kwargs['K']
            sig0 = kwargs['sig0']  # 1e3  # 0.25 #* K  # reference stress # 1e5              # 0.5
            eps0 = kwargs['eps0']  # = 0.03  # 0.2  # reference strain #    # 0.03                  # 0.1
            n = kwargs['n']  # 5.0  # 3.0  # hardening exponent  # # 5.0               # 10.0

            strain_trace_qxyz = np.einsum('ii...', strain) / 3  # todo{2 or 3 in 2D }
            # strain_trace_xyz = np.einsum('ijxyz,ji ->xyz', strain, I) / 3  # todo{2 or 3 in 2D }

            # volumetric strain
            strain_vol_ijqxyz = np.ndarray(shape=strain.shape)
            strain_vol_ijqxyz.fill(0)
            for d in np.arange(discretization.domain_dimension):
                strain_vol_ijqxyz[d, d, ...] = strain_trace_qxyz

            # deviatoric strain
            strain_dev_ijqxyz = strain - strain_vol_ijqxyz

            # equivalent strain
            strain_dev_ddot = np.einsum('ijqxyz,jiqxyz-> qxyz', strain_dev_ijqxyz, strain_dev_ijqxyz)
            strain_eq_qxyz = np.sqrt((2. / 3.) * strain_dev_ddot)

            #
            sig = (3. * K * strain_vol_ijqxyz
                   + 2. / 3. * sig0 / (eps0 ** n) *
                   (strain_eq_qxyz ** (n - 1.)) * strain_dev_ijqxyz)
            #
            # sig = 3. * K * strain_vol_ijqxyz * (strain_eq_qxyz == 0.).astype(float) + sig * (
            #         strain_eq_qxyz != 0.).astype(float)

            # K4_d = discretization.get_material_data_size_field(name='alg_tangent')
            strain_dev_dyad = np.einsum('ijqxyz,klqxyz->ijklqxyz', strain_dev_ijqxyz, strain_dev_ijqxyz)

            K4_d = 2. / 3. * sig0 / (eps0 ** n) * (strain_dev_dyad * 2. / 3. * (n - 1.) * strain_eq_qxyz ** (
                    n - 3.) + strain_eq_qxyz ** (n - 1.) * I4d[..., np.newaxis, np.newaxis, np.newaxis, np.newaxis])

            # threshold = 1e-15
            # mask = (np.abs(strain_eq_qxyz) > threshold).astype(float)

            K4 = K * II[
                ..., np.newaxis, np.newaxis, np.newaxis, np.newaxis] + K4_d  # * mask  # *(strain_equivalent_qxyz != 0.).astype(float)
            #
            # np.broadcast_to(II[..., np.newaxis, np.newaxis, np.newaxis, np.newaxis],
            #                 (3, 3, 3, 3, discretization.nb_quad_points_per_pixel, *number_of_pixels))

            return sig, K4


        def constitutive_q_points(strain_ijqxyz):
            phase_field = np.zeros([*number_of_pixels])
            phase_field[
                1 * number_of_pixels[0] // 4:3 * number_of_pixels[0] // 4, 1 * number_of_pixels[0] // 4:3 *
                                                                                                        number_of_pixels[
                                                                                                            0] // 4, :] = 1.
            # phase_field[:26, :, :] = 1.

            # sig_P1, K4_P1 = nonlinear_elastic_q_points(strain_ijqxyz.s, K=2)
            sig_P2, K4_P2 = nonlinear_elastic_q_points(strain_ijqxyz.s, **model_parameters_non_linear)

            sig_P1, K4_P1 = linear_elastic_q_points(strain_ijqxyz.s, **model_parameters_linear)  # inclusion
            # sig_P2, K4_P2 = linear_elastic_q_points(strain_ijqxyz.s, K=2)

            sig_ijqxyz = phase_field * sig_P1 + (1. - phase_field) * sig_P2
            K4_ijklqxyz = phase_field * K4_P1[
                ..., np.newaxis, np.newaxis, np.newaxis, np.newaxis] + (1. - phase_field) * K4_P2

            # remove border pixels
            # sig_ijqxyz[..., 0:2, :] = 0
            # sig_ijqxyz[..., -2:, :] = 0
            # K4_ijklqxyz[..., 0:2, :] = 0
            # K4_ijklqxyz[..., -2:, :] = 0

            return sig_ijqxyz, K4_ijklqxyz


        def constitutive(strain_ijqxyz):
            sig_ijqxyz, K4_ijklqxyz = constitutive_q_points(strain_ijqxyz)

            return sig_ijqxyz, K4_ijklqxyz


        macro_gradient_inc_field = discretization.get_gradient_size_field(name='macro_gradient_inc_field')

        displacement_fluctuation_field = discretization.get_unknown_size_field(name='displacement_fluctuation_field')
        displacement_increment_field = discretization.get_unknown_size_field(name='displacement_increment_field')

        strain_fluc_field = discretization.get_displacement_gradient_sized_field(name='strain_fluctuation_field')
        total_strain_field = discretization.get_displacement_gradient_sized_field(name='strain_field')
        rhs_field = discretization.get_unknown_size_field(name='rhs_field')

        x = np.linspace(start=0, stop=domain_size[0], num=number_of_pixels[0])
        y = np.linspace(start=0, stop=domain_size[1], num=number_of_pixels[1])
        X, Y = np.meshgrid(x, y, indexing='ij')

        # evaluate material law
        stress, K4_ijklqyz = constitutive(total_strain_field)

        if save_results:
            # save strain fluctuation
            results_name = (f'init_K')
            np.save(data_folder_path + results_name + f'.npy', K4_ijklqyz.mean(axis=4))

        # set macroscopic loading increment
        ninc = 1
        _info['ninc'] = ninc

        macro_gradient_inc = np.zeros(shape=(3, 3))
        # macro_gradient_inc[0, 0] += 0.05 / float(ninc)
        macro_gradient_inc[0, 1] += 0.05 / float(ninc)
        macro_gradient_inc[1, 0] += 0.05 / float(ninc)
        dt = 1. / float(ninc)

        # set macroscopic gradient
        macro_gradient_inc_field = discretization.get_macro_gradient_field(macro_gradient_ij=macro_gradient_inc,
                                                                           macro_gradient_field_ijqxyz=macro_gradient_inc_field)

        # assembly preconditioner
        preconditioner = discretization.get_preconditioner_Green_fast(
            reference_material_data_ijkl=I4s)  # K4_ijklqyz.mean(axis=(4, 5, 6, 7))

        M_fun_Green = lambda x: discretization.apply_preconditioner_NEW(preconditioner_Fourier_fnfnqks=preconditioner,
                                                                        nodal_field_fnxyz=x)

        sum_CG_its = 0
        sum_Newton_its = 0
        start_time = time.time()
        iteration_total = 0

        # incremental loading
        for inc in range(ninc):
            print(f'Increment {inc}')
            print(f'==========================================================================')

            # strain-hardening exponent
            total_strain_field.s[...] += macro_gradient_inc_field.s

            # Solve mechanical equilibrium constrain
            rhs_field = discretization.get_rhs_explicit_stress(stress_function=constitutive,  # constitutive_pixel
                                                               gradient_field_ijqxyz=total_strain_field,
                                                               rhs_inxyz=rhs_field)
            # evaluate material law
            stress, K4_ijklqyz = constitutive(total_strain_field)  #

            En = np.linalg.norm(total_strain_field.s.mean(axis=2))

            rhs_t_norm = np.linalg.norm(rhs_field.s)
            # incremental deformation  newton loop
            iiter = 0
            print('Rhs at new laod step {0:10.2e}'.format(np.linalg.norm(rhs_field.s)))
            # preconditioer = 'Green'

            # iterate as long as the iterative update does not vanish
            while True:
                # Set up preconditioner

                if preconditioner_type == 'Green':
                    M_fun = M_fun_Green
                elif preconditioner_type == 'Jacobi_Green':
                    # K_ref = discretization.get_system_matrix(I4s)

                    K_diag_alg = discretization.get_preconditioner_Jacoby_fast(
                        material_data_field_ijklqxyz=K4_ijklqyz)
                    # GJ_matrix = np.diag(K_diag_alg.flatten()) @ K_ref @ np.diag(K_diag_alg.flatten())

                    M_fun_GJ = lambda x: K_diag_alg * discretization.apply_preconditioner_NEW(
                        preconditioner_Fourier_fnfnqks=preconditioner,
                        nodal_field_fnxyz=K_diag_alg * x)
                    if enforce_mean:
                        M_fun = lambda x: (y := M_fun_GJ(x)) - np.mean(y, axis=(-1, -2, -3), keepdims=True)
                    else:
                        M_fun = lambda x: M_fun_GJ(x)

                        # mat_model_pars = {'mat_model': 'power_law_elasticity'}
                K_fun = lambda x: discretization.apply_system_matrix(
                    material_data_field=K4_ijklqyz,  # constitutive_pixel
                    displacement_field=x,
                    formulation='small_strain')


                def my_callback(x_0):
                    print('mean_x0 {}'.format(x_0.mean()))


                displacement_increment_field.s.fill(0)
                displacement_increment_field.s, norms = solvers.PCG(Afun=K_fun,
                                                                    B=rhs_field.s,
                                                                    x0=displacement_increment_field.s,
                                                                    P=M_fun, steps=int(1000),
                                                                    toler=1e-14,
                                                                    norm_type='rr',
                                                                    # callback=my_callback
                                                                    )
                if save_results:
                    results_name = (f'displacement_increment_field_it{iteration_total}')
                    np.save(data_folder_path + results_name + f'.npy', displacement_increment_field.s)

                nb_it_comb = len(norms['residual_rr'])
                norm_rz = norms['residual_rz'][-1]
                norm_rr = norms['residual_rr'][-1]
                print(f'nb iteration CG = {nb_it_comb}')
                sum_CG_its += nb_it_comb

                _info['norm_rr'] = norms['residual_rr']
                _info['norm_rz'] = norms['residual_rz']
                _info['nb_it_comb'] = nb_it_comb

                # phase_field_sol_FE_MPI = xopt.x.reshape([1, 1, *discretization.nb_of_pixels])

                # compute strain from the displacement increment
                strain_fluc_field.s = discretization.apply_gradient_operator_symmetrized(
                    u_inxyz=displacement_increment_field,
                    grad_u_ijqxyz=strain_fluc_field)

                total_strain_field.s += strain_fluc_field.s
                displacement_fluctuation_field.s += displacement_increment_field.s

                if save_results:
                    # save strain fluctuation
                    results_name = (f'strain_fluc_field' + f'_it{iteration_total}')
                    np.save(data_folder_path + results_name + f'.npy', strain_fluc_field.s.mean(axis=2))
                    # save total  strain
                    results_name = (f'total_strain_field' + f'_it{iteration_total}')
                    np.save(data_folder_path + results_name + f'.npy', total_strain_field.s.mean(axis=2))
                    # save stress
                    results_name = (f'stress' + f'_it{iteration_total}')
                    np.save(data_folder_path + results_name + f'.npy', stress.mean(axis=2))
                    # save K4_ijklqyz
                    results_name = (f'K4_ijklqyz' + f'_it{iteration_total}')
                    np.save(data_folder_path + results_name + f'.npy', K4_ijklqyz.mean(axis=4))

                # evaluate material law
                stress, K4_ijklqyz = constitutive(total_strain_field)  #
                # Recompute right hand side
                rhs_field = discretization.get_rhs_explicit_stress(stress_function=constitutive,  # constitutive_pixel,
                                                                   gradient_field_ijqxyz=total_strain_field,
                                                                   rhs_inxyz=rhs_field)
                # rhs *= -1
                # print('=====================')
                print('np.linalg.norm(strain_fluc_field.s) / En {0:10.2e}'.format(
                    np.linalg.norm(strain_fluc_field.s) / En))
                print('np.linalg.norm(rhs_field.s) / rhs_t_norm  {0:10.2e}'.format(
                    np.linalg.norm(rhs_field.s) / rhs_t_norm))

                print('Rhs {0:10.2e}'.format(np.linalg.norm(rhs_field.s)))
                print('strain_fluc_field {0:10.2e}'.format(np.linalg.norm(strain_fluc_field.s)))
                if np.linalg.norm(rhs_field.s) / rhs_t_norm < 1.e-6 and iiter > 0: break
                if np.linalg.norm(strain_fluc_field.s) / En < 1.e-6 and iiter > 0: break
                _info['norm_strain_fluc_field'] = np.linalg.norm(strain_fluc_field.s)
                _info['norm_En'] = np.linalg.norm(strain_fluc_field.s)

                if np.linalg.norm(rhs_field.s) < 1.e-7 and iiter > 0: break
                _info['norm_rhs_field)'] = np.linalg.norm(rhs_field.s)

                # print('Norm of disp displacement_increment_field {0:10.2e}'.format(
                #     np.linalg.norm(displacement_increment_field.s.mean(axis=1))))
                # print('Norm of disp displacement_increment_field/ EN {0:10.2e}'.format(
                #     np.linalg.norm(displacement_increment_field.s) / En))

                np.savez(data_folder_path + f'info_log_it{iteration_total}.npz', **_info)
                print(data_folder_path + f'info_log_it{iteration_total}.npz')

                # update Newton iteration counter
                iiter += 1
                sum_Newton_its += 1
                iteration_total += 1

                if iiter == 100:
                    break

            # # linear part of displacement(X-domain_size[0]/2)
            disp_linear_x = ((X - domain_size[0] / 2) * macro_gradient_inc[0, 0] * inc +
                             Y * macro_gradient_inc[0, 1] * inc)  # (X - domain_size[0] / 2)
            disp_linear_y = ((X - domain_size[0] / 2) * macro_gradient_inc[1, 0] * inc
                             + Y * macro_gradient_inc[1, 1] * inc)
            # displacement in voids should be zero
            # displacement_fluctuation_field.s[:, 0, :, :5] = 0.0

            x_deformed = X + disp_linear_x + displacement_fluctuation_field.s[0, 0, :, :, 0]
            y_deformed = Y + disp_linear_y + displacement_fluctuation_field.s[1, 0, :, :, 0]

            print("element_type : ", element_type)
            print("number_of_pixels: ", number_of_pixels)
            print(f'preconditioner_type: {preconditioner_type}')

            print(f'Total number of CG {sum_CG_its}')
            print(f'Total number of sum_Newton_its {sum_Newton_its}')
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Elapsed time : ", elapsed_time)
            print("Elapsed time: ", elapsed_time / 60)

            if save_results:
                # save deformed positions
                results_name = (f'x_deformed' + f'_it{iteration_total}')
                np.save(data_folder_path + results_name + f'.npy', x_deformed)
                results_name = (f'y_deformed' + f'_it{iteration_total}')
                np.save(data_folder_path + results_name + f'.npy', y_deformed)

                _info['sum_Newton_its'] = sum_Newton_its
                _info['iteration_total'] = iteration_total
                _info['sum_CG_its'] = sum_CG_its
                _info['elapsed_time'] = elapsed_time

                np.savez(data_folder_path + f'info_log_final.npz', **_info)
                print(data_folder_path + f'info_log_final.npz')

            plot_sol_field = False
            if plot_sol_field:
                fig = plt.figure(figsize=(9, 3.0))
                gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.5, width_ratios=[1, 1],
                                      height_ratios=[1, 1])

                ax_strain = fig.add_subplot(gs[1, 0])
                pcm = ax_strain.pcolormesh(x_deformed, y_deformed, total_strain_field.s.mean(axis=2)[0, 1, ..., 0],
                                           cmap=mpl.cm.cividis,  # vmin=1, vmax=3,
                                           rasterized=True)
                plt.colorbar(pcm, ax=ax_strain)
                plt.title('total_strain_field   ')
                ax_strain.spines['top'].set_visible(False)
                ax_strain.spines['right'].set_visible(False)
                ax_strain.spines['bottom'].set_visible(False)
                ax_strain.spines['left'].set_visible(False)

                ax_strain.get_xaxis().set_visible(False)  # hides x-axis only
                ax_strain.get_yaxis().set_visible(False)  # hides y-axis only

                ax_strain = fig.add_subplot(gs[0, 0])
                pcm = ax_strain.pcolormesh(x_deformed, y_deformed, strain_fluc_field.s.mean(axis=2)[0, 1, ..., 0],
                                           cmap=mpl.cm.cividis,  # vmin=1, vmax=3,
                                           rasterized=True)
                plt.colorbar(pcm, ax=ax_strain)
                ax_strain.spines['top'].set_visible(False)
                ax_strain.spines['right'].set_visible(False)
                ax_strain.spines['bottom'].set_visible(False)
                ax_strain.spines['left'].set_visible(False)

                ax_strain.get_xaxis().set_visible(False)  # hides x-axis only
                ax_strain.get_yaxis().set_visible(False)  # hides y-axis only

                plt.title('strain_fluc_field')
                max_stress = stress.mean(axis=2)[0, 1, ..., 0].max()
                min_stress = stress.mean(axis=2)[0, 1, ..., 0].min()

                print('stress min ={}'.format(min_stress))
                print('stress max ={}'.format(max_stress))
                ax_stress = fig.add_subplot(gs[1, 1])
                pcm = ax_stress.pcolormesh(x_deformed, y_deformed, stress.mean(axis=2)[0, 1, ..., 0],
                                           cmap=mpl.cm.cividis, vmin=min_stress, vmax=max_stress,
                                           rasterized=True)
                ax_stress.spines['top'].set_visible(False)
                ax_stress.spines['right'].set_visible(False)
                ax_stress.spines['bottom'].set_visible(False)
                ax_stress.spines['left'].set_visible(False)

                ax_stress.get_xaxis().set_visible(False)  # hides x-axis only
                ax_stress.get_yaxis().set_visible(False)  # hides y-axis only
                plt.colorbar(pcm, ax=ax_stress)
                plt.title('stress   ')

                # plot constitutive tangent
                ax_tangent = fig.add_subplot(gs[0, 1])
                pcm = ax_tangent.pcolormesh(x_deformed, y_deformed, K4_ijklqyz.mean(axis=4)[0, 1, 0, 0, ..., 0],
                                            cmap=mpl.cm.cividis,  # vmin=0, vmax=1500,
                                            rasterized=True)
                ax_tangent.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                # Hide all four spines individually
                ax_tangent.spines['top'].set_visible(False)
                ax_tangent.spines['right'].set_visible(False)
                ax_tangent.spines['bottom'].set_visible(False)
                ax_tangent.spines['left'].set_visible(False)

                ax_tangent.get_xaxis().set_visible(False)  # hides x-axis only
                ax_tangent.get_yaxis().set_visible(False)  # hides y-axis only
                # x_deformed[:, :], y_deformed[:, :],
                plt.colorbar(pcm, ax=ax_tangent)
                plt.title('material_data_field_C_0')

            plt.show()
