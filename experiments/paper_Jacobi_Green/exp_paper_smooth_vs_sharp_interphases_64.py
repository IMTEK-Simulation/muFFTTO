from cProfile import label

import numpy as np
import scipy as sc
import time
import os
from NuMPI.IO import save_npy, load_npy

import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams["text.usetex"] = True

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library
from mpl_toolkits import mplot3d
import copy

script_name = 'exp_paper_smooth_vs_sharp_interphases_64'
file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
figure_folder_path = file_folder_path + '/figures/' + script_name + '/'

old_data_folder_path = file_folder_path + '/exp_data/' + 'exp_paper_smooth_vs_sharp_interphases' + '/'

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'
src = '../figures/'  # source folder\

# microstructure name
name = 'lbfg_muFFTTO_elasticity_exp_paper_JG_2D_elasticity_TO_N64_E_target_0.15_Poisson_-0.50_Poisson0_0.29_w5.00_eta0.02_mac_1.0_p2_prec=Green_bounds=False_FE_NuMPI6_nb_load_cases_3_e_obj_False_random_True'
iteration = 1200


def scale_field(field, min_val, max_val):
    """Scales a 2D random field to be within [min_val, max_val]."""
    field_min, field_max = np.min(field), np.max(field)
    scaled_field = (field - field_min) / (field_max - field_min)  # Normalize to [0,1]
    return scaled_field * (max_val - min_val) + min_val  # Scale to [min_val, max_val]


def scale_field_log(field, min_val, max_val):
    """Scales a 2D random field to be within [min_val, max_val]."""
    field_log = np.log10(field)
    field_min, field_max = np.min(field_log), np.max(
        field_log)

    scaled_field = (field_log - field_min) / (field_max - field_min)  # Normalize to [0,1]
    return 10 ** (scaled_field * (np.log10(max_val) - np.log10(min_val)) + np.log10(
        min_val))  # Scale to [min_val, max_val]


compute = True
plot = True
enforce_mean = True
save_results = True

if compute:

    domain_size = [1, 1]
    nb_pix_multips = [4]  # ,2,3,3,2,

    ratios = np.array([2, 5, 8])  # 4,6,8 5, 2 5,8

    nb_it = np.zeros((ratios.size, 2))
    nb_it_combi = np.zeros((ratios.size, 2))
    nb_it_Jacobi = np.zeros((ratios.size, 2))
    nb_it_Richardson = np.zeros((ratios.size, 2))
    nb_it_Richardson_combi = np.zeros((ratios.size, 2))

    energy_evols_G = []
    energy_evols_GJ = []

    norm_rr_combi = []
    norm_rz_combi = []
    norm_rr_Jacobi = []

    norm_rz_Jacobi = []
    norm_rr = []
    norm_rz = []

    norm_rMr = []
    norm_rMr_combi = []
    norm_rMr_Jacobi = []

    kontrast = []
    kontrast_2 = []
    eigen_LB = []

    for kk in np.arange(np.size(nb_pix_multips)):
        nb_pix_multip = nb_pix_multips[kk]
        # number_of_pixels = (nb_pix_multip * 32, nb_pix_multip * 32)
        number_of_pixels = (nb_pix_multip * 16, nb_pix_multip * 16)

        # number_of_pixels = (16,16)

        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)

        discretization = domain.Discretization(cell=my_cell,
                                               nb_of_pixels_global=number_of_pixels,
                                               discretization_type=discretization_type,
                                               element_type=element_type)
        start_time = time.time()

        # set macroscopic gradient
        # macro_gradient = np.array([[1.0, 0.5], [0.5, 1.0]])
        macro_gradient = np.array([[1.0, 0.], [0.0, 0.0]])

        # create material data field
        K_0, G_0 = 1, 0.5  # domain.get_bulk_and_shear_modulus(E=1, poison=0.2)

        # identity tensor                                               [single tensor]
        ii = np.eye(2)

        shape = tuple((number_of_pixels[0] for _ in range(2)))


        def expand(arr):
            new_shape = (np.prod(arr.shape), np.prod(shape))
            ret_arr = np.zeros(new_shape)
            ret_arr[:] = arr.reshape(-1)[:, np.newaxis]
            return ret_arr.reshape((*arr.shape, *shape))


        # identity tensors                                            [grid of tensors]
        I = ii
        I4 = np.einsum('il,jk', ii, ii)
        I4rt = np.einsum('ik,jl', ii, ii)
        I4s = (I4 + I4rt) / 2.

        elastic_C_1 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                         K=K_0,
                                                         mu=G_0,
                                                         kind='linear')
        C_1 = domain.compute_Voigt_notation_4order(elastic_C_1)

        material_data_field_C_0 = discretization.get_material_data_size_field(name='mat_Data')
        material_data_field_C_0.s = np.einsum('ijkl,qxy->ijklqxy', elastic_C_1,
                                              np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                                *discretization.nb_of_pixels])))

        #
        # refmaterial_data_field_I4s = np.einsum('ijkl,qxy->ijklqxy', elastic_C_1,
        #                                        np.ones(np.array([discretization.nb_quad_points_per_pixel,
        #                                                          *discretization.nb_of_pixels])))

        print('elastic tangent = \n {}'.format(domain.compute_Voigt_notation_4order(elastic_C_1)))

        # material distribution

        geometry = np.load('../exp_data/' + name + f'_it{iteration}.npy', allow_pickle=True)
        phase_field_origin = np.abs(geometry)

        # phase_field = np.random.rand(*discretization.get_scalar_sized_field().shape)  # set random distribution#

        # phase = 1 * np.ones(number_of_pixels)
        inc_contrast = 0.

        # nb_it=[]
        # nb_it_combi=[]
        # nb_it_Jacobi=[]
        # phase_field_origin =# np.abs(phase_field_smooth - 1)
        # flipped_arr = 1 - phase_field

        macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_inc_field')
        rhs_field = discretization.get_unknown_size_field(name='rhs_field')
        x_init = discretization.get_displacement_sized_field(name='x_init')
        displacement_field = discretization.get_unknown_size_field(name='solution')

        # Set up right hand side
        macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient_ij=macro_gradient,
                                                                       macro_gradient_field_ijqxyz=macro_gradient_field)

        phase_field_min = np.min(phase_field_origin)
        phase_field_max = np.max(phase_field_origin)
        jacobi_counter = 0
        min_idx = np.unravel_index(phase_field_origin.argmin(), phase_field_origin.shape)
        for i in np.arange(ratios.shape[0]):
            ratio = ratios[i]

            counter = 0
            for sharp in [False, True]:
                _info = {}
                #
                # if ratio == 0:
                #     phase_field = scale_field(phase_field, min_val=0, max_val=1.0)
                # else:
                #     phase_field = scale_field(phase_field, min_val=1 / 10 ** ratio, max_val=1.0)

                phase_field = np.copy(phase_field_origin)

                if sharp:
                    # phase_field = scale_field(phase_field_origin, min_val=1 / 10 ** ratio, max_val=1.0)
                    phase_field[phase_field < 0.5] = 1 / 10 ** ratio  # phase_field_min#
                    phase_field[phase_field > 0.49] = phase_field_max  # 1

                print(f'ratio={ratio} ')

                phase_field = scale_field_log(np.copy(phase_field), min_val=1 / (10 ** ratio),
                                              max_val=phase_field_max)
                print(f'min ={np.min(phase_field)} ')
                print(f'max ={np.max(phase_field)} ')

                print(f'min ={np.min(phase_field)} ')
                print(f'max ={np.max(phase_field)} ')

                material_data_field_C_0_rho = np.copy(material_data_field_C_0.s[..., :, :, :]) * np.power(
                    phase_field, 1)

                # plt.figure()
                # plt.semilogy(np.power(
                #     phase_field, 1)[10,:])
                # plt.semilogy(np.power(
                #     phase_field, 2)[10,:])
                #
                #
                # plt.show()

                print(f'min ={np.min(material_data_field_C_0_rho)} ')
                print(f'max ={np.max(material_data_field_C_0_rho)} ')
                # print(np.max(np.power(
                #     phase_field, 2)))
                # material_data_field_C_0_rho_ijklqxyz = material_data_field_C_0[..., :, :, :] * np.power(
                #     material_data_field_C_0_rho, 2)[0, :, 0, ...]

                # apply material distribution

                # perturb=np.random.random(macro_gradient_field.shape)
                # macro_gradient_field += perturb#-np.mean(perturb)

                # Solve mechanical equilibrium constrain
                rhs_field = discretization.get_rhs(material_data_field_ijklqxyz=material_data_field_C_0_rho,
                                                   macro_gradient_field_ijqxyz=macro_gradient_field,
                                                   rhs_inxyz=rhs_field)

                K_fun = lambda x: discretization.apply_system_matrix(material_data_field_C_0_rho, x,
                                                                     formulation='small_strain')

                # plotting eigenvalues

                # omega = 1  # 2 / ( eig[-1]+eig[np.argmax(eig>0)])

                preconditioner = discretization.get_preconditioner_Green_fast(reference_material_data_ijkl=elastic_C_1)

                M_fun = lambda x: discretization.apply_preconditioner_NEW(preconditioner_Fourier_fnfnqks=preconditioner,
                                                                          nodal_field_fnxyz=x)

                # K_mat = discretization.get_system_matrix(material_data_field=material_data_field_C_0_rho)

                K_diag_alg = discretization.get_preconditioner_Jacoby_fast(
                    material_data_field_ijklqxyz=material_data_field_C_0_rho)  #
                jacobi_counter += 1

                M_fun_GJ = lambda x: K_diag_alg * discretization.apply_preconditioner_NEW(
                    preconditioner_Fourier_fnfnqks=preconditioner,
                    nodal_field_fnxyz=K_diag_alg * x)
                if enforce_mean:
                    # M_fun_combi = lambda x: (y := M_fun_GJ(x)) - np.mean(y)
                    M_fun_combi = lambda x: (y := M_fun_GJ(x)) - np.mean(y, axis=(-1, -2, -3), keepdims=True)
                else:
                    M_fun_combi = lambda x: M_fun_GJ(x)

                M_fun_null = lambda x: 1 * x

                M_fun_Jacobi = lambda x: K_diag_alg * K_diag_alg * x

                K_fun_callback = lambda x: discretization.apply_system_matrix(
                    material_data_field=material_data_field_C_0_rho,
                    displacement_field=x,
                    output_field_inxyz=x_init,
                    formulation='small_strain')

                energy_evols_G = []

                cg_iter = 0


                def my_callback_G(x_0, r_0=None):
                    global rhs_field, K_fun_callback, energy_evols_G, cg_iter
                    # if r_0 is not None:
                    #     energy = np.sum(x_0 * (-rhs_field.s + -r_0))
                    #     print('energy 2 {}'.format(energy))
                    # else:
                    # energy = np.sum(x_0 * (2 * rhs_field.s - K_fun_callback(x_0)))
                    # print('energy {}'.format(energy))
                    # energy_evols_G.append(energy)

                    # compute strain from the displacement increment
                    strain_fluc_field_it = discretization.get_displacement_gradient_sized_field(
                        name='strain_fluctuation_field_it_G')
                    # disp_fluctuation_field_it = discretization.get_displacement_sized_field(
                    #     name='disp_fluctuation_field_it')
                    # disp_fluctuation_field_it.s.fill(0)
                    # disp_fluctuation_field_it.s = x_0
                    strain_fluc_field_it.s = discretization.apply_gradient_operator_symmetrized(
                        u_inxyz=x_0,
                        grad_u_ijqxyz=strain_fluc_field_it)
                    strain_fluc_field_it.s = strain_fluc_field_it.s + macro_gradient_field.s
                    strain_fluc_field_it.s = discretization.apply_material_data(
                        material_data=material_data_field_C_0_rho,
                        gradient_field=strain_fluc_field_it)

                    results_name = (f'stress_field_it{cg_iter}' + f'ration{i}_sharp{sharp}')
                    np.save(data_folder_path + results_name + f'_G.npy', strain_fluc_field_it.s)
                    cg_iter += 1
                    print('cg_iter G {}'.format(cg_iter))


                # init solution
                x_init.s.fill(0)
                displacement_field.s.fill(0)
                displacement_field.s, norms = solvers.PCG(K_fun, rhs_field.s, x0=x_init.s, P=M_fun,
                                                          steps=int(10000), toler=1e-10,
                                                          norm_type='rr',
                                                          callback=my_callback_G
                                                          )
                if save_results:
                    results_name = (f'displacement_field_' + f'ration{i}_sharp{sharp}')
                    np.save(data_folder_path + results_name + f'_G.npy', displacement_field.s)

                    # compute strain from the displacement increment
                    strain_fluc_field = discretization.get_displacement_gradient_sized_field(
                        name='strain_fluctuation_field_G')
                    strain_fluc_field.s.fill(0)
                    strain_fluc_field.s = discretization.apply_gradient_operator_symmetrized(
                        u_inxyz=displacement_field,
                        grad_u_ijqxyz=strain_fluc_field)

                    results_name = (f'strain_fluc_field_' + f'ration{i}_sharp{sharp}')
                    np.save(data_folder_path + results_name + f'_G.npy', strain_fluc_field.s)

                    # strain_fluc_field.s.fill(0)
                    strain_fluc_field.s = discretization.apply_material_data(material_data=material_data_field_C_0_rho,
                                                                             gradient_field=strain_fluc_field)
                    results_name = (f'stress_fluc_field_' + f'ration{i}_sharp{sharp}')
                    np.save(data_folder_path + results_name + f'_G.npy', strain_fluc_field.s)

                    results_name = (f'energy_evols_G_' + f'ration{i}_sharp{sharp}')
                    np.save(data_folder_path + results_name + f'_G.npy', energy_evols_G)

                nb_it[i, counter] = (len(norms['residual_rr']))
                norm_rz.append(norms['residual_rz'])
                norm_rr.append(norms['residual_rr'])
                norm_rMr.append(norms['data_scaled_rr'])

                print(f'Ration ={ratio} ')
                print(f'Sharp = {sharp} ')

                print(f'Green its = {nb_it} ')
                #########
                #

                energy_evols_GJ = []

                cg_iter = 0


                def my_callback_GJ(x_0, r_0=None):
                    global rhs_field, K_fun_callback, energy_evols_GJ, cg_iter

                    # if r_0 is not None:
                    #     energy = np.sum(x_0 * (-rhs_field.s + -r_0))
                    #     print('energy 2 {}'.format(energy))
                    # else:
                    # energy = np.sum(x_0 * (2 * rhs_field.s - K_fun_callback(x_0)))
                    # print('energy {}'.format(energy))
                    # energy_evols_GJ.append(energy)

                    # compute strain from the displacement increment
                    strain_fluc_field_it = discretization.get_displacement_gradient_sized_field(
                        name='strain_fluctuation_field_it_GJ')
                    # disp_fluctuation_field_it = discretization.get_displacement_sized_field(
                    #     name='disp_fluctuation_field_it')
                    # disp_fluctuation_field_it.s.fill(0)
                    # disp_fluctuation_field_it.s = x_0
                    strain_fluc_field_it.s = discretization.apply_gradient_operator_symmetrized(
                        u_inxyz=x_0,
                        grad_u_ijqxyz=strain_fluc_field_it)
                    strain_fluc_field_it.s = strain_fluc_field_it.s + macro_gradient_field.s
                    strain_fluc_field_it.s = discretization.apply_material_data(
                        material_data=material_data_field_C_0_rho,
                        gradient_field=strain_fluc_field_it)

                    results_name = (f'stress_field_it{cg_iter}' + f'ration{i}_sharp{sharp}')
                    np.save(data_folder_path + results_name + f'_GJ.npy', strain_fluc_field_it.s)
                    cg_iter += 1
                    print('cg_iter GJ {}'.format(cg_iter))


                x_init.s.fill(0)
                displacement_field_combi, norms_combi = solvers.PCG(K_fun, rhs_field.s, x0=x_init.s,
                                                                    P=M_fun_combi,
                                                                    steps=int(4000),
                                                                    toler=1e-10,
                                                                    norm_type='rr',
                                                                    callback=my_callback_GJ
                                                                    )
                print(norms_combi['residual_rr'])
                if save_results:
                    results_name = (f'displacement_field_' + f'ration{i}_sharp{sharp}')
                    np.save(data_folder_path + results_name + f'_GJ.npy', displacement_field_combi)

                    # compute strain from the displacement increment
                    strain_fluc_field = discretization.get_displacement_gradient_sized_field(
                        name='strain_fluctuation_field_GJ')
                    strain_fluc_field.s.fill(0)
                    strain_fluc_field.s = discretization.apply_gradient_operator_symmetrized(
                        u_inxyz=displacement_field_combi,
                        grad_u_ijqxyz=strain_fluc_field)

                    results_name = (f'strain_fluc_field_' + f'ration{i}_sharp{sharp}')
                    np.save(data_folder_path + results_name + f'_GJ.npy', strain_fluc_field.s)

                    # strain_fluc_field.s.fill(0)
                    strain_fluc_field.s = discretization.apply_material_data(material_data=material_data_field_C_0_rho,
                                                                             gradient_field=strain_fluc_field)
                    results_name = (f'stress_fluc_field_' + f'ration{i}_sharp{sharp}')
                    np.save(data_folder_path + results_name + f'_GJ.npy', strain_fluc_field.s)

                    results_name = (f'energy_evols_GJ_' + f'ration{i}_sharp{sharp}')
                    np.save(data_folder_path + results_name + f'_GJ.npy', energy_evols_GJ)

                nb_it_combi[i, counter] = (len(norms_combi['residual_rr']))
                norm_rz_combi.append(norms_combi['residual_rz'])
                norm_rr_combi.append(norms_combi['residual_rr'])
                norm_rMr_combi.append(norms_combi['data_scaled_rr'])

                print(f'GJ its = {nb_it_combi} ')

                x_init.s.fill(0)
                displacement_field_Jacobi, norms_Jacobi = solvers.PCG(K_fun, rhs_field.s, x0=x_init.s,
                                                                      P=M_fun_Jacobi,
                                                                      steps=int(4000),
                                                                      toler=1e-12,
                                                                      norm_type='rr')
                nb_it_Jacobi[i, counter] = (len(norms_Jacobi['residual_rr']))
                norm_rz_Jacobi.append(norms_Jacobi['residual_rz'])
                norm_rr_Jacobi.append(norms_Jacobi['residual_rr'])
                norm_rMr_Jacobi.append(norms_Jacobi['data_scaled_rr'])

                print(len(norms_Jacobi['residual_rz']))
                print(norms_Jacobi['residual_rr'])
                if save_results:
                    results_name = (f'displacement_field_' + f'ration{i}_sharp{sharp}')
                    np.save(data_folder_path + results_name + f'_J.npy', displacement_field_Jacobi)

                    # compute strain from the displacement increment
                    strain_fluc_field = discretization.get_displacement_gradient_sized_field(
                        name='strain_fluctuation_field_J')
                    strain_fluc_field.s.fill(0)
                    strain_fluc_field.s = discretization.apply_gradient_operator_symmetrized(
                        u_inxyz=displacement_field_Jacobi,
                        grad_u_ijqxyz=strain_fluc_field)

                    results_name = (f'strain_fluc_field_' + f'ration{i}_sharp{sharp}')
                    np.save(data_folder_path + results_name + f'_J.npy', strain_fluc_field.s)

                    # strain_fluc_field.s.fill(0)
                    strain_fluc_field.s = discretization.apply_material_data(material_data=material_data_field_C_0_rho,
                                                                             gradient_field=strain_fluc_field)
                    results_name = (f'stress_fluc_field_' + f'ration{i}_sharp{sharp}')
                    np.save(data_folder_path + results_name + f'_J.npy', strain_fluc_field.s)

                # displacement_field_Richardson, norms_Richardson = solvers.Richardson(K_fun, rhs, x0=None, P=M_fun,
                #                                                                      omega=omega,
                #                                                                      steps=int(1000),
                #                                                                      toler=1e-1)

                counter += 1

                _info['norms_G'] = norms['residual_rr']  # ['data_scaled_rr']
                _info['norms_GJ'] = norms_combi['residual_rr']  # ['data_scaled_rr']
                _info['norms_J'] = norms_Jacobi['residual_rr']  # ['data_scaled_rr']

                results_name = f'N64_{ratio}_sharp_{sharp}'

                np.savez(data_folder_path + results_name + f'_log.npz', **_info)
                print(data_folder_path + results_name + f'_log.npz')

plot_energy_evols = False
if plot_energy_evols:
    ratios = np.array([2, 5, 8])
    for sharp in [False, True]:
        fig_err = plt.figure(figsize=(8.3, 5.0))

        for i in np.arange(3, step=1):
            results_name = (f'energy_evols_G_' + f'ration{i}_sharp{sharp}')
            energy_evols_G = np.load(data_folder_path + results_name + f'_G.npy', allow_pickle=True)
            results_name = (f'energy_evols_GJ_' + f'ration{i}_sharp{sharp}')
            energy_evols_GJ = np.load(data_folder_path + results_name + f'_GJ.npy', allow_pickle=True)

            fig_err.gca().semilogy(abs(np.diff(energy_evols_G)), label=r'$JG \kappa=10^' + f'{{{-ratios[i]}}}$',
                                   color='red', linestyle=':', lw=2)
            fig_err.gca().semilogy(abs(np.diff(energy_evols_GJ)), label=r'$JG \kappa=10^' + f'{{{-ratios[i]}}}$',
                                   color='blue', linestyle=':', lw=2)

            fig_err.gca().semilogy(energy_evols_G[-1] - energy_evols_G, label=r'$JG \kappa=10^' + f'{{{-ratios[i]}}}$',
                                   color='red', linestyle='--', lw=2)
            fig_err.gca().semilogy(energy_evols_GJ[-1] - energy_evols_GJ,
                                   label=r'$JG \kappa=10^' + f'{{{-ratios[i]}}}$',
                                   color='blue', linestyle='--', lw=2)
        plt.show()

plot_strain_evols = True
if plot_strain_evols:
    ratios = np.array([2, 5, 8])
    for sharp in [False, True]:
        fig = plt.figure(figsize=(8.3, 5.0))

        # gs = fig.add_gridspec(1, 3)
        gs_global = fig.add_gridspec(1, 1, width_ratios=[1], wspace=0.2)
        ax_stress = fig.add_subplot(gs_global[0, 0])
        for i in np.arange(3):
            # fig = plt.figure(figsize=(11.5, 6))

            results_name = f'N64_{ratios[i]}_sharp_{sharp}'
            _info = np.load(data_folder_path + results_name + f'_log.npz', allow_pickle=True)

            results_name = (f'energy_evols_G_' + f'ration{i}_sharp{sharp}')
            energy_evols_G = np.load(data_folder_path + results_name + f'_G.npy', allow_pickle=True)
            results_name = (f'energy_evols_GJ_' + f'ration{i}_sharp{sharp}')
            energy_evols_GJ = np.load(data_folder_path + results_name + f'_GJ.npy', allow_pickle=True)

            norm_G = _info['norms_G']
            norm_GJ = _info['norms_GJ']
            norm_J = _info['norms_J']

            error_strain_norm = []
            stress_G_norm = []
            stress_00_G_average = []

            for cg_iter_i in range(len(norm_G)):
                results_name = (f'stress_field_it{cg_iter_i}' + f'ration{i}_sharp{sharp}')
                stress_field_G_it = np.load(data_folder_path + results_name + f'_G.npy', allow_pickle=True)
                stress_G_norm.append(np.linalg.norm(stress_field_G_it))
                stress_00_G_average.append(np.mean(stress_field_G_it[0, 0]))

            stress_GJ_norm = []
            stress_00_GJ_average = []
            for cg_iter_i in range(len(norm_GJ)):
                results_name = (f'stress_field_it{cg_iter_i}' + f'ration{i}_sharp{sharp}')
                stress_field_GJ_it = np.load(data_folder_path + results_name + f'_GJ.npy', allow_pickle=True)
                stress_GJ_norm.append(np.linalg.norm(stress_field_GJ_it))
                stress_00_GJ_average.append(np.mean(stress_field_GJ_it[0, 0]))

            # error = stress_field_G_it - stress_field_GJ_it#
            # error_strain_norm.append(np.linalg.norm(error))#
            # ax_stress.semilogy(abs((stress_00_G_average[:-1] - stress_00_G_average[-1]) / stress_00_G_average[-1]), 'g',
            #                    label='Green')
            # ax_stress.semilogy(abs((stress_00_GJ_average[:-1] - stress_00_GJ_average[-1]) / stress_00_GJ_average[-1]),
            #                    'k', label='Green-Jacobi ')
            ax_stress.plot(stress_00_G_average, 'g', label=rf'Green-$10^{ratios[i]}$')
            ax_stress.plot(stress_00_GJ_average, 'k', label=rf'Green-Jacobi-$10^{ratios[i]}$')
            ax_stress.set_title(f'sharp= {sharp}')
            ax_stress.legend(loc='best')

        plt.show()
        print()

plot_residual = True
if plot_residual:
    plt.rcParams.update({
        "text.usetex": True,  # Use LaTeX
        # "font.family": "helvetica",  # Use a serif font
    })
    plt.rcParams.update({'font.size': 11})
    plt.rcParams["font.family"] = "Arial"

    # plt.rcParams.update({'font.size': 14})
    ratios = np.array([2, 5, 8])

    # fig = plt.figure(figsize=(11.5, 6))
    fig = plt.figure(figsize=(8.3, 5.0))

    # gs = fig.add_gridspec(1, 3)
    gs_global = fig.add_gridspec(1, 3, width_ratios=[1, 2, 1], wspace=0.2)

    gs_error = gs_global[1].subgridspec(2, 1, width_ratios=[1], hspace=0.2)  # 0.1, 1, 4
    gs_geom = gs_global[0].subgridspec(2, 2, width_ratios=[0.1, 1], hspace=0.1, wspace=0.5)  # 0.1, 1, 4

    ax_cbar = fig.add_subplot(gs_geom[:, 0])
    lines = ['-', '-.', '--', ':']
    row = 0
    for sharp in [False, True]:
        ax_error = fig.add_subplot(gs_error[row, 0])
        ax_error.text(-0.12, 1.05, rf'\textbf{{(b.{row + 1}}})  ', transform=ax_error.transAxes)

        ax_geom = fig.add_subplot(gs_geom[row, 1])

        if row == 0:
            ax_geom.text(-0.2, 1.21, rf'\textbf{{(a.{row + 1}) }} ', transform=ax_geom.transAxes)
        elif row == 1:
            ax_geom.text(-0.2, 1.16, rf'\textbf{{(a.{row + 1})}}  ', transform=ax_geom.transAxes)

        divnorm = mpl.colors.Normalize(vmin=1e-8, vmax=1)
        cmap_ = mpl.cm.seismic  # mpl.cm.seismic #mpl.cm.Greys
        geometry = np.load('../exp_data/' + name + f'_it{iteration}.npy', allow_pickle=True)

        phase_field_origin = np.abs(geometry)
        phase_field_max = np.max(phase_field_origin)

        phase_field = scale_field_log(np.copy(phase_field_origin), min_val=1 / (10 ** ratios[-1]),
                                      max_val=phase_field_max)
        if sharp:
            phase_field[phase_field < 0.5] = 1 / 10 ** ratios[-1]  # phase_field_min#
            phase_field[phase_field > 0.49] = phase_field_max  # 1

        # np.unravel_index(phase_field_origin.argmin(), phase_field_origin.shape)
        pcm = ax_geom.pcolormesh(np.tile(phase_field, (1, 1)),
                                 cmap=cmap_, linewidth=0,
                                 rasterized=True, norm=divnorm)

        ax_geom.set_xticks([0, 32, 64])
        ax_geom.set_yticks([0, 32, 64])
        # ax_geom.axis('equal' )
        ax_geom.set_aspect('equal', 'box')
        if sharp:
            ax_geom.set_xlabel('pixel index')
        if sharp:
            ax_geom.set_title(r'$\rho_{\rm sharp}$', wrap=True)  # Density
        else:
            ax_geom.set_title(r'$\rho_{\rm smooth}$', wrap=True)  # $Density

        for i in np.arange(ratios.size, step=1):
            results_name = f'N64_{ratios[i]}_sharp_{sharp}'
            _info = np.load(data_folder_path + results_name + f'_log.npz', allow_pickle=True)

            results_name = (f'energy_evols_G_' + f'ration{i}_sharp{sharp}')
            energy_evols_G = np.load(data_folder_path + results_name + f'_G.npy', allow_pickle=True)
            results_name = (f'energy_evols_GJ_' + f'ration{i}_sharp{sharp}')
            energy_evols_GJ = np.load(data_folder_path + results_name + f'_GJ.npy', allow_pickle=True)

            norm_G = _info['norms_G']
            norm_GJ = _info['norms_GJ']
            norm_J = _info['norms_J']
            kappa = 10 ** ratios[i]

            k = np.arange(max([len(norm_GJ), len(norm_G)]))
            # print(f'k \n {k}')

            convergence = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** k
            convergence = convergence  # *norm_rr[i][0]

            relative_error_G = norm_G  # / norm_G[0]
            relative_error_GJ = norm_GJ  # / norm_GJ
            relative_error_J = norm_J  # / norm_GJ[0]

            ax_error.loglog(relative_error_G, label=fr'$\kappa=10^{{{-ratios[i]}}}$', color='g',
                            linestyle=lines[i], lw=2)
            #  ax_1.semilogy(norm_rMr[2*i+1]/norm_rMr[2*i+1][0], label=f'Green ' +r'$\kappa=10^'+f'{{{ratios[i]}}}$', color='r', linestyle=lines[i])
            ax_error.loglog(relative_error_GJ, label=r'$JG \kappa=10^' + f'{{{-ratios[i]}}}$',
                            color='black', linestyle=lines[i], lw=2)

            ax_error.loglog(relative_error_J, label=r'$J \kappa=10^' + f'{{{-ratios[i]}}}$',
                            color='b', linestyle=lines[i], lw=2)

            # ax_error.loglog(abs(np.diff(energy_evols_G)), label=r'$JG \kappa=10^' + f'{{{-ratios[i]}}}$',
            #                 color='red', linestyle=':', lw=2)
            # ax_error.loglog(abs(np.diff(energy_evols_GJ)), label=r'$JG \kappa=10^' + f'{{{-ratios[i]}}}$',
            #                 color='blue', linestyle=':', lw=2)
            # ax_error.loglog(energy_evols_G[-1] - energy_evols_G, label=r'$JG \kappa=10^' + f'{{{-ratios[i]}}}$',
            #                        color='red', linestyle='--', lw=2)
            # ax_error.loglog(energy_evols_GJ[-1] - energy_evols_GJ,
            #                        label=r'$JG \kappa=10^' + f'{{{-ratios[i]}}}$',
            #                        color='blue', linestyle='--', lw=2)
            if sharp:
                ax_error.set_xlabel(r'PCG iteration - $k$')

            ax_error.set_ylabel('Norm of residual')  # - '  fr'$||r_{{k}}||_{{\mathbdf{{G}} }}   $')#^{-1}
            # ax_error.set_title(r'Relative  norm of residua', wrap=True)

            # plt.legend([r'$\kappa$ upper bound','Green', 'Jacobi', 'Green + Jacobi','Richardson'])
            ax_error.set_ylim([1e-10, 1])  # norm_rz[i][0]]/lb)
            ax_error.set_xlim([1, 1e3])

            ax_error.set_yticks([1e-10, 1e-5, 1])
            ax_error.set_yticklabels([fr'$10^{{{-10}}}$', fr'$10^{{{-5}}}$', fr'$10^{{{0}}}$'])
            # ax_error.set_xscale('linear')
            # ax_error.legend(['Green', 'Green-Jacobi'], loc='best')
            if sharp:

                arrows_G = [5, 10, 15]  # anotation arrows
                text_G = np.array([[1.2, 1e-4],  # anotation Text position
                                   [1.6, 5e-7],
                                   [2.0, 1e-9]])

                arrows_GJ = [30, 70, 30]  # anotation arrows
                text_GJ = np.array([[50, 1e-2],  # anotation Text position
                                    [100, 5e-5],
                                    [200, 1e-7]])
            else:
                arrows_G = [5, 70, 300]
                text_G = np.array([[40, 1e-2],
                                   [120, 1e-4],
                                   [250, 1e-6]])

                arrows_GJ = [3, 30, 60]
                text_GJ = np.array([[1.2, 1e-5],
                                    [1.6, 1e-7],
                                    [2.0, 5e-10]])

            ax_error.annotate(text=f'Green-Jacobi\n' + fr'  $\chi^{{\mathrm{{tot}}}} =10^{{{ratios[i]}}}$',
                              xy=(arrows_GJ[i], relative_error_GJ[arrows_GJ[i]]),
                              xytext=(text_GJ[i, 0], text_GJ[i, 1]),
                              arrowprops=dict(arrowstyle='->',
                                              color='black',
                                              lw=1,
                                              ls=lines[i]),
                              fontsize=9,
                              color='black'
                              )
            ax_error.annotate(text=f'Green\n' + fr'$\chi^{{\mathrm{{tot}}}} = 10^{{{ratios[i]}}}$',
                              xy=(arrows_G[i], relative_error_G[arrows_G[i]]),
                              xytext=(text_G[i, 0], text_G[i, 1]),
                              arrowprops=dict(arrowstyle='->',
                                              color='green',
                                              lw=1,
                                              ls=lines[i]),
                              fontsize=9,
                              color='green'
                              )
            ax_error.yaxis.set_ticks_position('right')  # Set y-axis ticks to the right

        cbar = plt.colorbar(pcm, location='left', cax=ax_cbar)
        cbar.ax.yaxis.tick_left()
        # cbar.set_ticks(ticks=[1e-4,1e-2, 1])
        # cbar.set_ticklabels([f'$10^{{{-4}}}$', f'$10^{{{-2}}}$', 1])
        cbar.set_ticks(ticks=[1e-8, 0.5, 1])
        cbar.set_ticklabels([r'$\frac{1}{\chi^{\rm tot}}$', 0.5, 1])
        row += 1

    ##################### add strain difference
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    gs_diff = gs_global[2].subgridspec(2, 2, width_ratios=[1, 0.1], hspace=0.1, wspace=0.5)  # 0.1, 1, 4
    sharpness = [False, True]
    for row in [0, 1]:
        ax_disp = fig.add_subplot(gs_diff[row, 0])
        ax_disp_cbar = fig.add_subplot(gs_diff[row, 1])

        i = 0

        sharp = sharpness[row]

        # load disp
        results_name = (f'displacement_field_' + f'ration{i}_sharp{sharp}')
        disp_G = np.load(data_folder_path + results_name + f'_G.npy', allow_pickle=True)
        disp_GJ = np.load(data_folder_path + results_name + f'_GJ.npy', allow_pickle=True)

        # load strain
        results_name = (f'strain_fluc_field_' + f'ration{i}_sharp{sharp}')
        strain_G = np.load(data_folder_path + results_name + f'_G.npy', allow_pickle=True)
        strain_GJ = np.load(data_folder_path + results_name + f'_GJ.npy', allow_pickle=True)

        # load stress
        results_name = (f'stress_fluc_field_' + f'ration{i}_sharp{sharp}')
        stress_G = np.load(data_folder_path + results_name + f'_G.npy', allow_pickle=True)
        stress_GJ = np.load(data_folder_path + results_name + f'_GJ.npy', allow_pickle=True)

        ### compute the difference
        ij = (0, 0)

        stress_diff = abs((stress_GJ.mean(axis=2)[ij + (...,)] - stress_G.mean(axis=2)[ij + (...,)]))
        # stress_diff=stress_diff/ abs(stress_G.mean(axis=2)[ij + (...,)])
        field_to_plot_G = disp_G[0, 0]  # strain_G.mean(axis=2)[0, 0]    # disp_G[0, 0] #
        field_to_plot_GJ = disp_GJ[0, 0]  # strain_GJ.mean(axis=2)[0, 0]  # disp_GJ[0, 0] #

        # ax_disp.plot(field_to_plot_G[field_to_plot_G.shape[0] // 2, :], linewidth=1,
        #              color='green',
        #              linestyle='--'
        #              # linestyle=linestyles[geom_ax]
        #              )
        # ax_disp.plot(field_to_plot_GJ[field_to_plot_GJ.shape[0] // 2, :], linewidth=1,
        #              color='black',
        #              linestyle=':'
        #              )
        disp_diff = field_to_plot_GJ - field_to_plot_G
        max_diff = np.max([abs(disp_diff.min()), abs(disp_diff.max())])
                # ax_disp.semilogy(abs(disp_diff[disp_diff.shape[0] // 2, :]), linewidth=1,
        #              color='red',
        #              linestyle='-'
        #              )
        divnorm = mpl.colors.Normalize(vmin=-1e-5, vmax=1e-5)#max_diff

        pcm = ax_disp.pcolormesh(np.tile((disp_diff), (1, 1)),
                                 cmap=cmap_, linewidth=0,
                                 rasterized=True,
                                 norm=divnorm
                                 )

        ax_disp.set_xticks([0, 32, 64])
        ax_disp.set_yticks([0, 32, 64])

        disp_min = np.min(
            [field_to_plot_GJ[field_to_plot_GJ.shape[0] // 2, :].min(),
             field_to_plot_GJ[field_to_plot_GJ.shape[0] // 2, :].min()])
        disp_max = np.max(
            [field_to_plot_GJ[field_to_plot_GJ.shape[0] // 2, :].max(),
             field_to_plot_GJ[field_to_plot_GJ.shape[0] // 2, :].max()])
        #
        # ax_stress.yaxis.tick_right()
        ax_disp.set_xlim([0, 64])

        ax_disp.set_xticks([0, 32, 64])
        ax_disp.yaxis.set_ticks_position('right')
        ax_disp.yaxis.set_label_position('right')

        # ax_disp.set_yticks([-1, 0, 1.5])  # [disp_min, 0, disp_max]
        ax_disp.set_aspect('equal')

        cbar = plt.colorbar(pcm, location='right', cax=ax_disp_cbar)
        cbar.ax.yaxis.set_ticks_position('right')  # move ticks to right
        cbar.ax.yaxis.set_label_position('right')  # move label to right

        #cbar.set_ticks(ticks=[-max_diff, 0, max_diff])

        cbar.ax.yaxis.set_major_formatter(mpl.ticker.LogFormatter())

        # # cbar.set_ticklabels([f'{stress_diff_min:.0f}', f'{stress_diff_max / 2:.0f}', f'{stress_diff_max:.0f}'])
        # Set scientific notation for ticks
        #formatter = mpl.ticker.ScalarFormatter(useMathText=True)
        # cbar.ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True) )
        # cbar.ax.set_yticklabels(['$10^{' + str(int(np.log10(y))) + '}$' for y in cbar.ax.get_yticks()])

        #formatter.set_scientific(True)
        # formatter.set_powerlimits((-2, 2))  # Controls when scientific notation kicks in
        #cbar.ax.yaxis.set_major_formatter(formatter)
        #
        # ax_stress_cbar.set_ylabel(fr'$|\sigma^{{G}}_{{11}}-\sigma^{{GJ}}_{{11}}|  $')
        # Create inset
        # ax_inset = inset_axes(ax_disp, width="30%", height="30%", loc='upper right')
        # # ax_inset.plot(disp_GJ[0, 0][disp_GJ[0, 0].shape[0] // 2, 32:50], linewidth=1,
        # #              color='black',
        # #              linestyle=':'
        # #              )
        # ax_inset.plot(field_to_plot_GJ[field_to_plot_GJ.shape[0] // 2, 34:44], color='black', label='Inset')
        # ax_inset.plot(field_to_plot_G[field_to_plot_G.shape[0] // 2, 34:44], color='green', label='Inset')
        #
        # ax_inset.set_title('Inset', fontsize=10)
        # ax_inset.tick_params(labelsize=8)

    fig.tight_layout()
    fname = f'exp_paper_JG_TO_64_sharp_vs_smoot_data_diff' + '{}'.format('.pdf')
    print(('create figure: {}'.format(fname)))
    plt.savefig(figure_folder_path + fname, bbox_inches='tight')
    plt.show()

plot_old = False  # without stress difference
if plot_old:
    plt.rcParams.update({
        "text.usetex": True,  # Use LaTeX
        # "font.family": "helvetica",  # Use a serif font
    })
    plt.rcParams.update({'font.size': 11})
    plt.rcParams["font.family"] = "Arial"

    # plt.rcParams.update({'font.size': 14})
    ratios = np.array([2, 5, 8])

    # fig = plt.figure(figsize=(11.5, 6))
    fig = plt.figure(figsize=(8.3, 5.0))

    # gs = fig.add_gridspec(1, 3)
    gs_global = fig.add_gridspec(1, 2, width_ratios=[1, 2], wspace=0.2)

    gs_error = gs_global[1].subgridspec(2, 1, width_ratios=[1], hspace=0.2)  # 0.1, 1, 4
    gs_geom = gs_global[0].subgridspec(2, 2, width_ratios=[0.1, 1], hspace=0.1, wspace=0.5)  # 0.1, 1, 4

    ax_cbar = fig.add_subplot(gs_geom[:, 0])
    lines = ['-', '-.', '--', ':']
    row = 0
    for sharp in [False, True]:
        ax_error = fig.add_subplot(gs_error[row, 0])
        ax_error.text(-0.12, 1.05, rf'\textbf{{(b.{row + 1}}})  ', transform=ax_error.transAxes)

        ax_geom = fig.add_subplot(gs_geom[row, 1])

        if row == 0:
            ax_geom.text(-0.2, 1.21, rf'\textbf{{(a.{row + 1}) }} ', transform=ax_geom.transAxes)
        elif row == 1:
            ax_geom.text(-0.2, 1.16, rf'\textbf{{(a.{row + 1})}}  ', transform=ax_geom.transAxes)

        divnorm = mpl.colors.Normalize(vmin=1e-8, vmax=1)
        cmap_ = mpl.cm.seismic  # mpl.cm.seismic #mpl.cm.Greys
        geometry = np.load('../exp_data/' + name + f'_it{iteration}.npy', allow_pickle=True)

        phase_field_origin = np.abs(geometry)
        phase_field_max = np.max(phase_field_origin)

        phase_field = scale_field_log(np.copy(phase_field_origin), min_val=1 / (10 ** ratios[-1]),
                                      max_val=phase_field_max)
        if sharp:
            phase_field[phase_field < 0.5] = 1 / 10 ** ratios[-1]  # phase_field_min#
            phase_field[phase_field > 0.49] = phase_field_max  # 1

        # np.unravel_index(phase_field_origin.argmin(), phase_field_origin.shape)
        pcm = ax_geom.pcolormesh(np.tile(phase_field, (1, 1)),
                                 cmap=cmap_, linewidth=0,
                                 rasterized=True, norm=divnorm)

        ax_geom.set_xticks([0, 32, 64])
        ax_geom.set_yticks([0, 32, 64])
        # ax_geom.axis('equal' )
        ax_geom.set_aspect('equal', 'box')
        if sharp:
            ax_geom.set_xlabel('pixel index')
        if sharp:
            ax_geom.set_title(r'$\rho_{\rm sharp}$', wrap=True)  # Density
        else:
            ax_geom.set_title(r'$\rho_{\rm smooth}$', wrap=True)  # $Density

        for i in np.arange(ratios.size, step=1):
            results_name = f'N64_{ratios[i]}_sharp_{sharp}'
            _info = np.load(data_folder_path + results_name + f'_log.npz', allow_pickle=True)

            norm_G = _info['norms_G']
            norm_GJ = _info['norms_GJ']
            norm_J = _info['norms_J']
            kappa = 10 ** ratios[i]

            k = np.arange(max([len(norm_GJ), len(norm_G)]))
            # print(f'k \n {k}')

            convergence = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** k
            convergence = convergence  # *norm_rr[i][0]

            # ax_geom.text(-0.18, 0.97, '(a)', transform=ax_geom.transAxes)
            # print(f'convergecnce \n {convergence}')
            # ax_1.set_title(f'Smooth', wrap=True)
            # ax_1.semilogy(convergence,  label=f'estim {kappa}', color='k', linestyle=lines[i])
            # ax_geom.set_xticks([])
            # ax_geom.set_xticks([])
            relative_error_G = norm_G  # / norm_G[0]
            relative_error_GJ = norm_GJ  # / norm_GJ[0]
            ax_error.loglog(relative_error_G, label=fr'$\kappa=10^{{{-ratios[i]}}}$', color='g',
                            linestyle=lines[i], lw=2)
            #  ax_1.semilogy(norm_rMr[2*i+1]/norm_rMr[2*i+1][0], label=f'Green ' +r'$\kappa=10^'+f'{{{ratios[i]}}}$', color='r', linestyle=lines[i])
            ax_error.loglog(relative_error_GJ, label=r'$JG \kappa=10^' + f'{{{-ratios[i]}}}$',
                            color='black', linestyle=lines[i], lw=2)

            # ax_1.semilogy(norm_rMr_Jacobi[i]/norm_rMr_Jacobi[i][0], label=f' Jacobi {kappa}', color='b', linestyle=lines[i])
            # ax_1.semilogy(norm_rMr_combi[i]/norm_rMr_combi[i][0], label=f' Jacobi-Green {kappa}', color='r', linestyle=lines[i])

            # x_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
            # ax_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
            if sharp:
                ax_error.set_xlabel(r'PCG iteration - $k$')

            ax_error.set_ylabel('Norm of residual')  # - '  fr'$||r_{{k}}||_{{\mathbdf{{G}} }}   $')#^{-1}
            # ax_error.set_title(r'Relative  norm of residua', wrap=True)

            # plt.legend([r'$\kappa$ upper bound','Green', 'Jacobi', 'Green + Jacobi','Richardson'])
            ax_error.set_ylim([1e-10, 1])  # norm_rz[i][0]]/lb)
            ax_error.set_xlim([1, 1e3])

            ax_error.set_yticks([1e-10, 1e-5, 1])
            ax_error.set_yticklabels([fr'$10^{{{-10}}}$', fr'$10^{{{-5}}}$', fr'$10^{{{0}}}$'])
            # ax_error.set_xscale('linear')
            # ax_error.legend(['Green', 'Green-Jacobi'], loc='best')
            if sharp:

                arrows_G = [5, 10, 15]  # anotation arrows
                text_G = np.array([[1.2, 1e-4],  # anotation Text position
                                   [1.6, 5e-7],
                                   [2.0, 1e-9]])

                arrows_GJ = [30, 70, 30]  # anotation arrows
                text_GJ = np.array([[50, 1e-2],  # anotation Text position
                                    [100, 5e-5],
                                    [200, 1e-7]])
            else:
                arrows_G = [5, 70, 300]
                text_G = np.array([[40, 1e-2],
                                   [120, 1e-4],
                                   [250, 1e-6]])

                arrows_GJ = [3, 30, 60]
                text_GJ = np.array([[1.2, 1e-5],
                                    [1.6, 1e-7],
                                    [2.0, 5e-10]])
            # if sharp:
            #
            #     arrows_G = [5, 10, 15]  # anotation arrows
            #     text_G = np.array([[1.1, 5e-5],  # anotation Text position
            #                        [1.5, 5e-7],
            #                        [2.0, 5e-9]])
            #
            #     arrows_GJ = [30, 75, 100]  # anotation arrows
            #     text_GJ = np.array([[50, 5e-2],  # anotation Text position
            #                         [80, 1e-4],
            #                         [100, 1e-7]])
            # else:
            #     arrows_G = [6, 70, 300]
            #     text_G = np.array([[40, 1e-1],
            #                        [100, 1e-3],
            #                        [200, 1e-5]])
            #
            #     arrows_GJ = [5, 35, 60]
            #     text_GJ = np.array([[1.05, 1e-6],
            #                         [1.25, 3e-8],
            #                         [1.5, 1e-9]])

            ax_error.annotate(text=f'Green-Jacobi\n' + fr'  $\chi^{{\mathrm{{tot}}}} =10^{{{ratios[i]}}}$',
                              xy=(arrows_GJ[i], relative_error_GJ[arrows_GJ[i]]),
                              xytext=(text_GJ[i, 0], text_GJ[i, 1]),
                              arrowprops=dict(arrowstyle='->',
                                              color='black',
                                              lw=1,
                                              ls=lines[i]),
                              fontsize=9,
                              color='black'
                              )
            ax_error.annotate(text=f'Green\n' + fr'$\chi^{{\mathrm{{tot}}}} = 10^{{{ratios[i]}}}$',
                              xy=(arrows_G[i], relative_error_G[arrows_G[i]]),
                              xytext=(text_G[i, 0], text_G[i, 1]),
                              arrowprops=dict(arrowstyle='->',
                                              color='green',
                                              lw=1,
                                              ls=lines[i]),
                              fontsize=9,
                              color='green'
                              )
            ax_error.yaxis.set_ticks_position('right')  # Set y-axis ticks to the right

        cbar = plt.colorbar(pcm, location='left', cax=ax_cbar)
        cbar.ax.yaxis.tick_left()
        # cbar.set_ticks(ticks=[1e-4,1e-2, 1])
        # cbar.set_ticklabels([f'$10^{{{-4}}}$', f'$10^{{{-2}}}$', 1])
        cbar.set_ticks(ticks=[1e-8, 0.5, 1])
        cbar.set_ticklabels([r'$\frac{1}{\chi^{\rm tot}}$', 0.5, 1])
        row += 1
    fig.tight_layout()
    fname = f'exp_paper_JG_TO_64_sharp_vs_smoot' + '{}'.format('.pdf')
    print(('create figure: {}'.format(fname)))
    plt.savefig(figure_folder_path + fname, bbox_inches='tight')
    plt.show()
    ##################### add strain difference

quit()
