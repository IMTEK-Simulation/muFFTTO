import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import os

script_name = 'exp_square_inclusion'
file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
figure_folder_path = file_folder_path + '/figures/' + script_name + '/'

from NuMPI.IO import save_npy, load_npy

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library

from experiments.paper_alg_error_estimates import _labels, _colors, _markers

problem_type = 'conductivity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'  # #'linear_triangles'# linear_triangles_tilled
# formulation = 'small_strain'
geometry_ID = 'square_inclusion'

domain_size = [1, 1]

grids_sizes = [3, 4, 5, 6, ]  # [3, 4, 5, 6, 7, 8, 9]  # [4, 6, 8], 10  7, 8, 9, 10
rhos = [-3, -2, -1, 1, 2, 3]

for anisotropy in [False, True]:  # ,True
    for rho in rhos:
        for n in grids_sizes:
            number_of_pixels = (2 ** n, 2 ** n)

            my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                              problem_type=problem_type)

            discretization = domain.Discretization(cell=my_cell,
                                                   nb_of_pixels_global=number_of_pixels,
                                                   discretization_type=discretization_type,
                                                   element_type=element_type)
            start_time = time.time()

            # set macroscopic gradient
            macro_gradient = np.array([1.0, .0])

            # create material data field
            mat_contrast = 1  # matrix
            mat_contrast_2 = 10 ** rho  # inclusion

            if anisotropy:
                a_ani = 10
            else:
                a_ani = 1
            conductivity_C_1 = mat_contrast * np.array([[a_ani, 0], [0, 1.0]])
            conductivity_C_2 = mat_contrast_2 * np.array([[a_ani, 0], [0, 1.0]])
            conductivity_C_ref = np.array([[1., 0], [0, 1.0]])

            eigen_C1 = sp.linalg.eigh(a=conductivity_C_1, b=conductivity_C_ref, eigvals_only=True)
            eigen_C2 = sp.linalg.eigh(a=conductivity_C_2, b=conductivity_C_ref, eigvals_only=True)

            eigen_LB = np.min([eigen_C1, eigen_C2])
            # seigen_LB *=0.9
            eigen_UB = np.max([eigen_C1, eigen_C2])
            total_phase_contrast = eigen_UB / eigen_LB

            print(f'eigen_LB = {eigen_LB}')
            print(f'eigen_UB = {eigen_UB}')
            # J_eff = mat_contrast_2 * np.sqrt((mat_contrast_2 + 3 * mat_contrast) / (3 * mat_contrast_2 + mat_contrast))
            # print("J_eff : ", J_eff)
            A_eff = mat_contrast * np.sqrt((mat_contrast + 3 * mat_contrast_2) / (3 * mat_contrast + mat_contrast_2))
            print("A_eff : ", A_eff)

            material_data_field_C_0_rho = discretization.get_material_data_size_field_mugrid(
                name='material_data_field_C_0')

            material_data_field_C_1 = discretization.get_material_data_size_field_mugrid(name='material_data_field_C_1')
            material_data_field_C_2 = discretization.get_material_data_size_field_mugrid(name='material_data_field_C_2')
            material_data_field_C_ref = discretization.get_material_data_size_field_mugrid(
                name='material_data_field_C_ref')

            material_data_field_C_1.s = np.einsum('ij,qxy->ijqxy', conductivity_C_1,
                                                  np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                                    *discretization.nb_of_pixels])))

            material_data_field_C_2.s = np.einsum('ij,qxy->ijqxy', conductivity_C_2,
                                                  np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                                    *discretization.nb_of_pixels])))

            material_data_field_C_ref.s = np.einsum('ij,qxy->ijqxy', conductivity_C_ref,
                                                    np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                                      *discretization.nb_of_pixels])))
            # material distribution
            phase_field = discretization.get_scalar_field(name='phase_field')

            phase_field.s[0, 0] = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                                      microstructure_name=geometry_ID,
                                                                      coordinates=discretization.fft.coords)
            matrix_mask = phase_field.s[0, 0] > 0
            inc_mask = phase_field.s[0, 0] == 0

            # apply material distribution
            # material_data_field_C_0_rho = material_data_field_C_1[..., :, :] * np.power(phase_field,                                                                                        1)
            # material_data_field_C_0_rho += material_data_field_C_2[..., :, :] * np.power(1 - phase_field, 2)

            material_data_field_C_0_rho.s[..., matrix_mask] = mat_contrast_2 * material_data_field_C_1.s[
                ..., matrix_mask]
            material_data_field_C_0_rho.s[..., inc_mask] = mat_contrast * material_data_field_C_2.s[..., inc_mask]
            # Set up the equilibrium system
            macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
            discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient,
                                                           macro_gradient_field_ijqxyz=macro_gradient_field)

            # Solve mechanical equilibrium constrain
            rhs_field = discretization.get_unknown_size_field(name='rhs_field')
            discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0_rho,
                                          macro_gradient_field_ijqxyz=macro_gradient_field,
                                          rhs_inxyz=rhs_field)

            #K_fun = lambda x: discretization.apply_system_matrix(material_data_field_C_0_rho, x)


            def K_fun(x, Ax):

                discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0_rho,
                                                          input_field_inxyz=x,
                                                          output_field_inxyz=Ax,
                                                          formulation='small_strain')
                discretization.fft.communicate_ghosts(Ax)


            # M_fun = lambda x: 1 * x
            #
            # preconditioner = discretization.get_preconditioner_NEW(
            #     reference_material_data_field_ijklqxyz=material_data_field_C_ref)
            # # preconditioner_old = discretization.get_preconditioner(reference_material_data_field_ijklqxyz=material_data_field_C_0)
            #
            # M_fun = lambda x: discretization.apply_preconditioner_NEW(preconditioner, x)
            preconditioner = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=conductivity_C_ref)


            def M_fun(x, Px):
                """
                Function to compute the product of the Preconditioner matrix with a vector.
                The Preconditioner is represented by the convolution operator.
                """
                discretization.fft.communicate_ghosts(x)
                discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                                           input_nodal_field_fnxyz=x,
                                                           output_nodal_field_fnxyz=Px)


            temperatute_field_precise, norms_precise = solvers.PCG(Afun=K_fun,
                                                                   B=rhs,
                                                                   x0=None,
                                                                   P=M_fun,
                                                                   steps=int(1000),
                                                                   toler=1e-14,
                                                                   norm_energy_upper_bound=True,
                                                                   lambda_min=eigen_LB)

            # compute homogenized stress field corresponding to displacement
            Aeff_h_precise = discretization.get_homogenized_stress(
                material_data_field_ijklqxyz=material_data_field_C_0_rho,
                displacement_field_inxyz=temperatute_field_precise,
                macro_gradient_field_ijqxyz=macro_gradient_field)[0, 0]

            error_in_Aeff_hk = []
            Aeff_hk = []


            def my_callback(x_k):
                # compute homogenized stress field corresponding to displacement
                homogenized_flux = discretization.get_homogenized_stress(
                    material_data_field_ijklqxyz=material_data_field_C_0_rho,
                    displacement_field_inxyz=x_k,
                    macro_gradient_field_ijqxyz=macro_gradient_field)
                Aeff_hk.append(homogenized_flux[0, 0])
                error_in_Aeff_hk.append(homogenized_flux[0, 0] - A_eff)  # J_eff_computed if J_eff is not available


            parameters_CG = {'exact_solution': temperatute_field_precise,
                             'energy_lower_bound': True,
                             'tau': 0.25}
            temperatute_field, norms = solvers.PCG(Afun=K_fun,
                                                   B=rhs,
                                                   x0=None,
                                                   P=M_fun,
                                                   steps=int(1000), toler=1e-14,
                                                   norm_energy_upper_bound=True,
                                                   lambda_min=eigen_LB,
                                                   callback=my_callback,
                                                   **parameters_CG)

            nb_it = len(norms['residual_rz'])
            print(' nb_ steps CG =' f'{nb_it}')

            true_e_error = np.asarray(norms['energy_iter_error'])
            lower_bound = np.asarray(norms['energy_lower_bound'])
            upper_estim = lower_bound / (1 - parameters_CG['tau'])
            upper_bound = np.asarray(norms['energy_upper_bound'])
            trivial_lower_bound = np.asarray(norms['residual_rz'] / eigen_UB)
            trivial_upper_bound = np.asarray(norms['residual_rz'] / eigen_LB)

            # ----------------------------------------------------------------------
            # compute homogenized stress field corresponding to displacement
            homogenized_flux = discretization.get_homogenized_stress(
                material_data_field_ijklqxyz=material_data_field_C_0_rho,
                displacement_field_inxyz=temperatute_field,
                macro_gradient_field_ijqxyz=macro_gradient_field)

            print(homogenized_flux)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Elapsed time: ", elapsed_time)

            _info = {}

            _info['true_e_error'] = true_e_error
            _info['lower_bound'] = lower_bound
            _info['upper_bound'] = upper_bound
            _info['upper_estim'] = upper_estim
            _info['trivial_lower_bound'] = trivial_lower_bound
            _info['trivial_upper_bound'] = trivial_upper_bound
            _info['total_phase_contrast'] = total_phase_contrast

            _info['homogenized_flux'] = homogenized_flux
            _info['Aeff_h_precise'] = Aeff_h_precise
            _info['A_eff'] = A_eff
            _info['Aeff_hk'] = Aeff_hk
            _info['error_in_Aeff_hk'] = error_in_Aeff_hk

            results_name = f'N{number_of_pixels[0]}_rho_inc{mat_contrast_2}_mat{mat_contrast}_ani{a_ani}'
            save_npy(data_folder_path + results_name + f'.npy', temperatute_field,
                     tuple(discretization.fft.subdomain_locations),
                     tuple(discretization.nb_of_pixels_global))

            np.savez(data_folder_path + results_name + f'_log.npz', **_info)
            print(data_folder_path + results_name + f'.npy')

plot = False
if plot:
    fig = plt.figure(figsize=(7, 4.5))
    gs = fig.add_gridspec(1, 1, hspace=0.1, wspace=0.1, width_ratios=1 * (1,),
                          height_ratios=[1, ])
    ax_norms = fig.add_subplot(gs[0])
    ax_norms.semilogy(true_e_error,
                      label=_labels['true_error'],
                      color=_colors['true_error'],
                      alpha=1.,
                      marker=_markers['true_error'],
                      linewidth=1, markersize=5, markevery=5)

    ax_norms.semilogy(trivial_upper_bound,
                      label=_labels['trivial_upper_bound'],
                      color=_colors['trivial_upper_bound'],
                      alpha=0.5,
                      marker=_markers['trivial_upper_bound'],
                      linewidth=1, markersize=5, markevery=5)
    ax_norms.semilogy(trivial_lower_bound,
                      label=_labels['trivial_lower_bound'],
                      color=_colors['trivial_lower_bound'],
                      alpha=0.5,
                      marker=_markers['trivial_lower_bound'],
                      linewidth=1, markersize=5, markevery=5)

    ax_norms.semilogy(upper_bound,
                      label=_labels['PT_upper_bound'],
                      color=_colors['PT_upper_bound'],
                      alpha=0.5,
                      marker=_markers['PT_upper_bound'],
                      linewidth=1, markersize=5, markevery=5)

    ax_norms.semilogy(lower_bound,
                      label=_labels['PT_lower_bound'],
                      color=_colors['PT_lower_bound'],
                      linestyle='--',
                      linewidth=1,
                      alpha=0.5,
                      marker=_markers['PT_lower_bound'],
                      markersize=5, markevery=5)
    ax_norms.semilogy(upper_estim,
                      label=_labels['PT_upper_estimate'],
                      color=_colors['PT_upper_estimate'],
                      linestyle='--',
                      linewidth=1,
                      alpha=0.5,
                      marker=_markers['PT_upper_estimate'],
                      markersize=5, markevery=5)

    # ax_norms.semilogy(norms['residual_rr'], label='residual_rr', color='Black',
    #                   alpha=0.5, marker='.', linewidth=1, markersize=5, markevery=5)
    # ax_norms.semilogy(error_in_Aeff_00,
    #                   label=r'hom prop $\overline{\varepsilon}^{T} (A_{h,k}^{\mathrm{eff}} -A^{\mathrm{eff}}_{h,\infty})\,\overline{\varepsilon} $',
    #                   color='Black',
    #                   alpha=0.5, marker='x', linewidth=1, markersize=5, markevery=1)

    # plt.title('optimizer {}'.format(optimizer))
    # ax_norms.set_ylabel('Norms')
    ax_norms.set_ylim(1e-14, 1e6)
    # ax_norms.set_yticks([1, 34, 67, 100])
    # ax_norms.set_yticklabels([1, 34, 67, 100])

    ax_norms.set_xlabel(r'PCG iteration - $k$')

    ax_norms.set_xlim([0, norms['residual_rr'].__len__() - 1])
    # ax_norms.set_xticks([1, len(eig_G) // 2, len(eig_G)])
    # ax_norms.set_xticklabels([1, len(eig_G) // 2, len(eig_G)])

    plt.grid(True)

    plt.legend()
    fig_name = f'norm_evolution_kappa{total_phase_contrast}'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')

    # fig.tight_layout()
    fname = figure_folder_path + fig_name + '{}'.format('.pdf')
    print(('create figure: {}'.format(fname)))
    plt.savefig(fname, bbox_inches='tight'
                )
    plt.show()

    fig = plt.figure(figsize=(7, 4.5))
    gs = fig.add_gridspec(1, 1, hspace=0.1, wspace=0.1, width_ratios=1 * (1,),
                          height_ratios=[1, ])
    ax_norms = fig.add_subplot(gs[0])

    tmp = min(len(lower_bound), len(upper_bound))

    ax_norms.semilogy(trivial_upper_bound[0:tmp - 1] / true_e_error[0:tmp - 1],
                      label=_labels['trivial_upper_bound'],
                      color=_colors['trivial_upper_bound'],
                      marker=_markers['trivial_upper_bound'],
                      alpha=0.5,
                      linewidth=1, markersize=5, markevery=5
                      )

    ax_norms.semilogy(trivial_lower_bound[0:tmp - 1] / true_e_error[0:tmp - 1],
                      label=_labels['trivial_lower_bound'],
                      color=_colors['trivial_lower_bound'],
                      alpha=0.5,
                      marker=_markers['trivial_lower_bound'],
                      linewidth=1, markersize=5, markevery=5
                      )

    ax_norms.semilogy(upper_bound[0:tmp - 1] / true_e_error[0:tmp - 1],
                      label=_labels['PT_upper_bound'],
                      color=_colors['PT_upper_bound'],
                      alpha=0.5,
                      marker=_markers['PT_upper_bound'],
                      linewidth=1, markersize=5, markevery=5)

    ax_norms.semilogy(lower_bound[0:tmp - 1] / true_e_error[0:tmp - 1],
                      label=_labels['PT_lower_bound'],
                      color=_colors['PT_lower_bound'],
                      linestyle='--',
                      linewidth=1,
                      alpha=0.5,
                      marker=_markers['PT_lower_bound'],
                      markersize=5, markevery=5)

    ax_norms.semilogy(upper_estim[0:tmp - 1] / true_e_error[0:tmp - 1],
                      label=_labels['PT_upper_estimate'],
                      color=_colors['PT_upper_estimate'],
                      linestyle='--',
                      linewidth=1,
                      alpha=0.5,
                      marker=_markers['PT_upper_estimate'],
                      markersize=5, markevery=5)

    # ax_norms.semilogy((1:tmp,upper_estim_M(1:tmp)./norm_ener_error_M(1:tmp))
    # ax_norms.semilogy((1:tmp,estim_M_UB(1:tmp)./norm_ener_error_M(1:tmp))
    ax_norms.semilogy(np.ones(tmp), 'k-')

    # ax_norms.set_title('effectivity indices')
    ax_norms.set_xlabel(r'PCG iteration - $k$')

    # hezci rozsah os, abychom videli efektivitu u jednicky
    ax_norms.set_ylim(1e-4, 1e4)
    ax_norms.legend(loc='best')
    fig_name = f'norm_efficiency_kappa{total_phase_contrast}'  # '  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')

    fig.tight_layout()
    fname = figure_folder_path + fig_name + '{}'.format('.pdf')
    print(('create figure: {}'.format(fname)))
    plt.savefig(fname, bbox_inches='tight'
                )

    plt.show()

# nc = Dataset('temperatures.nc', 'w', format='NETCDF3_64BIT_OFFSET')
# nc.createDimension('coords', 1)
# nc.createDimension('number_of_dofs_x', number_of_pixels[0])
# nc.createDimension('number_of_dofs_y', number_of_pixels[1])
# nc.createDimension('number_of_dofs_per_pixel', 1)
# nc.createDimension('time', None)  # 'unlimited' dimension
# var = nc.createVariable('temperatures', 'f8',
#                         ('time', 'coords', 'number_of_dofs_per_pixel', 'number_of_dofs_x', 'number_of_dofs_y'))
# var[0, ...] = temperatute_field[0, ...]
#
# print(homogenized_flux)
# # var[0, ..., 0] = x
# # var[0, ..., 1] = y
