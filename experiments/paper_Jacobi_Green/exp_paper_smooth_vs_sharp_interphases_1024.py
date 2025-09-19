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

script_name = 'exp_paper_smooth_vs_sharp_interphases_1024'
file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
figure_folder_path = file_folder_path + '/figures/' + script_name + '/'

if not os.path.exists(data_folder_path):
    os.makedirs(data_folder_path)

if not os.path.exists(figure_folder_path):
    os.makedirs(figure_folder_path)

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'
src = '../figures/'  # source folder\

# microstructure name
# name = 'lbfg_muFFTTO_elasticity_exp_paper_JG_2D_elasticity_TO_N64_E_target_0.15_Poisson_-0.50_Poisson0_0.29_w5.00_eta0.02_mac_1.0_p2_prec=Green_bounds=False_FE_NuMPI6_nb_load_cases_3_e_obj_False_random_True'
# iteration = 1200

name = 'exp_2D_elasticity_TO_indre_3exp_N1024_Et_0.15_Pt_-0.5_P0_0.0_w5.0_eta0.01_p2_mpi90_nlc_3_e_False'
iteration = 8740
phase_field = np.load(os.path.expanduser(data_folder_path + name + f'_it{8740}.npy'), allow_pickle=True)


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
if compute:

    domain_size = [1, 1]
    nb_pix_multips = [1024]  # ,2,3,3,2,

    ratios = np.array([2, 5, 8])  # 4,6,8 5, 2 5,8

    nb_it = np.zeros((ratios.size, 2))
    nb_it_combi = np.zeros((ratios.size, 2))
    nb_it_Jacobi = np.zeros((ratios.size, 2))
    nb_it_Richardson = np.zeros((ratios.size, 2))
    nb_it_Richardson_combi = np.zeros((ratios.size, 2))

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
        number_of_pixels = (1024, 1024)

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

        material_data_field_C_0 = np.einsum('ijkl,qxy->ijklqxy', elastic_C_1,
                                            np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                              *discretization.nb_of_pixels])))

        refmaterial_data_field_I4s = np.einsum('ijkl,qxy->ijklqxy', elastic_C_1,
                                               np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                                 *discretization.nb_of_pixels])))

        print('elastic tangent = \n {}'.format(domain.compute_Voigt_notation_4order(elastic_C_1)))

        # material distribution

        geometry = np.load(os.path.expanduser(data_folder_path + name + f'_it{8740}.npy'), allow_pickle=True)
        phase_field_origin = np.abs(geometry)

        # phase_field = np.random.rand(*discretization.get_scalar_sized_field().shape)  # set random distribution#

        # phase = 1 * np.ones(number_of_pixels)
        inc_contrast = 0.

        # nb_it=[]
        # nb_it_combi=[]
        # nb_it_Jacobi=[]
        # phase_field_origin =# np.abs(phase_field_smooth - 1)
        # flipped_arr = 1 - phase_field
        phase_field_min = np.min(phase_field_origin)
        phase_field_max = np.max(phase_field_origin)

        min_idx = np.unravel_index(phase_field_origin.argmin(), phase_field_origin.shape)
        for i in np.arange(ratios.shape[0]):
            ratio = ratios[i]

            counter = 0
            for sharp in [False, True]:

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

                material_data_field_C_0_rho = np.copy(material_data_field_C_0[..., :, :, :]) * np.power(
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

                # Set up right hand side
                macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_inc_field')

                macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient_ij=macro_gradient,
                                                                               macro_gradient_field_ijqxyz=macro_gradient_field)
                # perturb=np.random.random(macro_gradient_field.shape)
                # macro_gradient_field += perturb#-np.mean(perturb)

                # Solve mechanical equilibrium constrain
                rhs_field = discretization.get_unknown_size_field(name='rhs_field')

                # rhs = discretization.get_rhs(material_data_field_C_0_rho, macro_gradient_field)
                rhs_field = discretization.get_rhs(material_data_field_ijklqxyz=material_data_field_C_0_rho,
                                                   # constitutive_pixel
                                                   macro_gradient_field_ijqxyz=macro_gradient_field,
                                                   rhs_inxyz=rhs_field)

                K_fun = lambda x: discretization.apply_system_matrix(material_data_field_C_0_rho, x,
                                                                     formulation='small_strain')

                # plotting eigenvalues

                omega = 1  # 2 / ( eig[-1]+eig[np.argmax(eig>0)])

                preconditioner = discretization.get_preconditioner_NEW(
                    reference_material_data_ijkl=I4s)

                M_fun = lambda x: discretization.apply_preconditioner_NEW(preconditioner_Fourier_fnfnqks=preconditioner,
                                                                          nodal_field_fnxyz=x)

                K_diag_alg = discretization.get_preconditioner_Jacoby_fast(
                    material_data_field_ijklqxyz=material_data_field_C_0_rho)

                M_fun_combi = lambda x: K_diag_alg * discretization.apply_preconditioner_NEW(
                    preconditioner_Fourier_fnfnqks=preconditioner,
                    nodal_field_fnxyz=K_diag_alg * x)
                # #
                M_fun_Jacobi = lambda x: K_diag_alg * K_diag_alg * x
                x_init = discretization.get_displacement_sized_field(name='x_init')

                # x_init=np.random.random(discretization.get_displacement_sized_field().shape)

                displacement_field, norms = solvers.PCG(K_fun, rhs_field.s, x0=x_init, P=M_fun,
                                                        steps=int(10000), toler=1e-12,
                                                        norm_type='data_scaled_rr',
                                                        norm_metric=M_fun)
                nb_it[i, counter] = (len(norms['residual_rz']))
                norm_rz.append(norms['residual_rz'])
                norm_rr.append(norms['residual_rr'])
                norm_rMr.append(norms['data_scaled_rr'])
                # homogenized_stresses = discretization.get_homogenized_stress(
                #     material_data_field_ijklqxyz=material_data_field_C_0_rho,
                #     displacement_field_inxyz=displacement_field,
                #     macro_gradient_field_ijqxyz=macro_gradient_field,
                #     formulation='small_strain')
                # print('Homogenized stress G  = \n {} \n'
                #       ' sharp={}'.format(homogenized_stresses,sharp))

                print(f'i={i} ')
                print(f'counter    ={counter} ')

                print(nb_it)
                #########
                displacement_field_combi, norms_combi = solvers.PCG(K_fun, rhs, x0=x_init, P=M_fun_combi,
                                                                    steps=int(4000),
                                                                    toler=1e-12,
                                                                    norm_type='data_scaled_rr',
                                                                    norm_metric=M_fun)
                nb_it_combi[i, counter] = (len(norms_combi['residual_rz']))
                norm_rz_combi.append(norms_combi['residual_rz'])
                norm_rr_combi.append(norms_combi['residual_rr'])
                norm_rMr_combi.append(norms_combi['data_scaled_rr'])
                # homogenized_stresses = discretization.get_homogenized_stress(
                #     material_data_field_ijklqxyz=material_data_field_C_0_rho,
                #     displacement_field_inxyz=displacement_field_combi,
                #     macro_gradient_field_ijqxyz=macro_gradient_field,
                #     formulation='small_strain')
                # print('Homogenized stress GJ  = \n {} \n'
                #       ' sharp={}'.format(homogenized_stresses, sharp))
                #
                displacement_field_Jacobi, norms_Jacobi = solvers.PCG(K_fun, rhs, x0=None, P=M_fun_Jacobi, steps=int(4),
                                                                      toler=1e-12,
                                                                      norm_type='data_scaled_rr',
                                                                      norm_metric=M_fun)
                nb_it_Jacobi[i, counter] = (len(norms_Jacobi['residual_rz']))
                norm_rz_Jacobi.append(norms_Jacobi['residual_rz'])
                norm_rr_Jacobi.append(norms_Jacobi['residual_rr'])
                norm_rMr_Jacobi.append(norms_Jacobi['data_scaled_rr'])
                print(len(norms_Jacobi['residual_rz']))

                # displacement_field_Richardson, norms_Richardson = solvers.Richardson(K_fun, rhs, x0=None, P=M_fun,
                #                                                                      omega=omega,
                #                                                                      steps=int(1000),
                #                                                                      toler=1e-1)

                counter += 1

                _info = {}
                _info['norms_G'] = norms['data_scaled_rr']
                _info['norms_GJ'] = norms_combi['data_scaled_rr']
                _info['norms_J'] = norms_Jacobi['data_scaled_rr']

                results_name = f'N1024_{ratio}_sharp_{sharp}'

                np.savez(data_folder_path + results_name + f'_log.npz', **_info)
                print(data_folder_path + results_name + f'_log.npz')

plot = False
if plot:
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

                arrows_GJ = [30, 70, 100]  # anotation arrows
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
quit()
if plot:
    ratios = np.array([2, 5, 8])

    sharp = False

    fig = plt.figure(figsize=(11.5, 4))

    # gs = fig.add_gridspec(1, 3)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.1, 1, 4])

    ax_error = fig.add_subplot(gs[0, 2])
    ax_cbar = fig.add_subplot(gs[0, 0])
    ax_geom = fig.add_subplot(gs[0, 1])
    lines = ['-', '-.', '--', ':']

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
    ax_geom.set_xlabel('pixel index')
    if sharp:
        ax_geom.set_title(r'Density $\rho_{\rm sharp}$', wrap=True)
    else:
        ax_geom.set_title(r'Density $\rho_{\rm smooth}$', wrap=True)

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
        relative_error_G = norm_G / norm_G[0]
        relative_error_GJ = norm_GJ / norm_GJ[0]
        ax_error.loglog(relative_error_G, label=fr'$\kappa=10^{{{-ratios[i]}}}$', color='g',
                        linestyle=lines[i], lw=2)
        #  ax_1.semilogy(norm_rMr[2*i+1]/norm_rMr[2*i+1][0], label=f'Green ' +r'$\kappa=10^'+f'{{{ratios[i]}}}$', color='r', linestyle=lines[i])
        ax_error.loglog(relative_error_GJ, label=r'$JG \kappa=10^' + f'{{{-ratios[i]}}}$',
                        color='black', linestyle=lines[i], lw=2)

        # ax_1.semilogy(norm_rMr_Jacobi[i]/norm_rMr_Jacobi[i][0], label=f' Jacobi {kappa}', color='b', linestyle=lines[i])
        # ax_1.semilogy(norm_rMr_combi[i]/norm_rMr_combi[i][0], label=f' Jacobi-Green {kappa}', color='r', linestyle=lines[i])

        # x_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
        # ax_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
        ax_error.set_xlabel(r'PCG iteration - $k$')
        # ax_1.set_ylabel('Relative error')
        ax_error.set_title(r'Relative norm of residua', wrap=True)

        # plt.legend([r'$\kappa$ upper bound','Green', 'Jacobi', 'Green + Jacobi','Richardson'])
        ax_error.set_ylim([1e-10, 1])  # norm_rz[i][0]]/lb)
        ax_error.set_xlim([1, 1e3])
        # ax_error.set_xscale('linear')
        # ax_error.legend(['Green', 'Green-Jacobi'], loc='best')
        if sharp:

            arrows_G = [5, 10, 15]  # anotation arrows
            text_G = np.array([[1.1, 1e-4],  # anotation Text position
                               [1.5, 1e-6],
                               [2.0, 1e-8]])

            arrows_GJ = [50, 75, 100]  # anotation arrows
            text_GJ = np.array([[50, 1e-1],  # anotation Text position
                                [100, 1e-3],
                                [150, 1e-5]])
        else:
            arrows_G = [7, 80, 300]
            text_G = np.array([[50, 1e-1],
                               [100, 1e-3],
                               [250, 1e-5]])

            arrows_GJ = [5, 35, 75]
            text_GJ = np.array([[1.1, 1e-4],
                                [1.5, 1e-6],
                                [2.0, 1e-8]])

        ax_error.annotate(text=fr'Green-Jacobi: $\chi =10^{{{-ratios[i]}}}$',
                          xy=(arrows_GJ[i], relative_error_GJ[arrows_GJ[i]]),
                          xytext=(text_GJ[i, 0], text_GJ[i, 1]),
                          arrowprops=dict(arrowstyle='->',
                                          color='black',
                                          lw=1,
                                          ls=lines[i]),
                          fontsize=12,
                          color='black'
                          )
        ax_error.annotate(text=fr'Green: $\chi = 10^{{{-ratios[i]}}}$',
                          xy=(arrows_G[i], relative_error_G[arrows_G[i]]),
                          xytext=(text_G[i, 0], text_G[i, 1]),
                          arrowprops=dict(arrowstyle='->',
                                          color='green',
                                          lw=1,
                                          ls=lines[i]),
                          fontsize=12,
                          color='green'
                          )

        cbar = plt.colorbar(pcm, location='left', cax=ax_cbar)
        cbar.ax.yaxis.tick_left()
        # cbar.set_ticks(ticks=[1e-4,1e-2, 1])
        # cbar.set_ticklabels([f'$10^{{{-4}}}$', f'$10^{{{-2}}}$', 1])
        cbar.set_ticks(ticks=[1e-8, 0.5, 1])
        cbar.set_ticklabels([r'$\chi$', 0.5, 1])

    fig.tight_layout()
    fname = f'exp_paper_JG_2D_elasticity_TO_64_{sharp}_vert_1' + '{}'.format('.pdf')
    print(('create figure: {}'.format(fname)))
    plt.savefig(figure_folder_path + fname, bbox_inches='tight')
    plt.show()
quit()
fig = plt.figure(figsize=(11.5, 4.))
gs = fig.add_gridspec(1, 3)
ax_1 = fig.add_subplot(gs[0, 2])
ax_cross = fig.add_subplot(gs[0, 1])
ax_geom = fig.add_subplot(gs[0, 0])

lines = ['-', '-.', '--', ':']
for i in np.arange(ratios.size, step=1):
    kappa = 10 ** ratios[i]

    k = np.arange(max(map(len, norm_rr)))
    # print(f'k \n {k}')

    convergence = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** k
    convergence = convergence  # *norm_rr[i][0]
    divnorm = mpl.colors.Normalize(vmin=1e-8, vmax=1)
    cmap_ = mpl.cm.seismic  # mpl.cm.seismic #mpl.cm.Greys
    geometry = np.load('../exp_data/' + name + f'_it{iteration}.npy', allow_pickle=True)
    phase_field_origin = np.abs(geometry)

    phase_field[phase_field < 0.5] = 1 / 10 ** ratios[i]  # phase_field_min#
    phase_field[phase_field > 0.49] = phase_field_max  # 1

    phase_field = scale_field_log(np.copy(phase_field), min_val=1 / (10 ** ratios[i]), max_val=phase_field_max)

    pcm = ax_geom.pcolormesh(np.tile(phase_field, (1, 1)),
                             cmap=cmap_, linewidth=0,
                             rasterized=True, norm=divnorm)
    ax_geom.axhline(y=min_idx[0], color='k', linestyle='-.')

    ax_cross.semilogy(phase_field[min_idx[0], :], label=r'$\kappa=10^' + f'{{{-ratios[i]}}}$', color='k',
                      linestyle=lines[i])
    # ax_geom.set_title(r'Initial ', wrap=True) # :,15
    # ax_cross.set_ylabel(r'Density $\rho$')
    # ax_cross.set_title(r'Density $\rho$', wrap=True)
    ax_cross.set_title(r'Cross section', wrap=True)
    ax_cross.set_xlabel('pixel index')
    ax_cross.set_ylim([1 / (10 ** ratios[i]), 1.1])
    yticks = [1 / (10 ** ratios[d]) for d in range(ratios.size)]

    ax_cross.set_yticks([1, *yticks])
    ax_cross.set_xlim([0, 64])
    ax_cross.set_xticks([0, 32, 64])
    ax_cross.axhline(y=1 / (10 ** ratios[i]), color='k', linestyle=lines[i], linewidth=0.5)

    # ax_cross.set_yscale('linear')

    ax_geom.set_xticks([0, 32, 64])
    ax_geom.set_yticks([0, 32, 64])
    ax_geom.set_aspect('equal', 'box')
    ax_geom.set_xlabel('pixel index')
    ax_geom.set_title(r'Density $\rho$', wrap=True)
    # print(f'convergecnce \n {convergence}')
    # ax_1.set_title(f'{i}', wrap=True)
    # ax_1.semilogy(convergence,  label=f'estim {kappa}', color='k', linestyle=lines[i])

    # ax_1.semilogy(norm_rMr[2*i]/norm_rMr[2*i][0], label=f'Green ' +r'$\kappa=10^'+f'{{{ratios[i]}}}$', color='g', linestyle=lines[i])
    ax_1.loglog(norm_rMr[2 * i + 1] / norm_rMr[2 * i + 1][0], label=r'$\kappa=10^' + f'{{{-ratios[i]}}}$', color='g',
                linestyle=lines[i])
    ax_1.loglog(norm_rMr_combi[2 * i + 1] / norm_rMr_combi[2 * i + 1][0],
                label=r'$JG \kappa=10^' + f'{{{-ratios[i]}}}$', color='r', linestyle=lines[i])

    # ax_1.semilogy(norm_rMr_Jacobi[i]/norm_rMr_Jacobi[i][0], label=f' Jacobi {kappa}', color='b', linestyle=lines[i])
    # ax_1.semilogy(norm_rMr_combi[i]/norm_rMr_combi[i][0], label=f' Jacobi-Green {kappa}', color='r', linestyle=lines[i])

    # x_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
    # ax_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
    ax_1.set_xlabel('PCG iteration number ')
    # ax_1.set_ylabel('Relative error')
    ax_1.set_title(r'Relative error', wrap=True)

    # plt.legend([r'$\kappa$ upper bound','Green', 'Jacobi', 'Green + Jacobi','Richardson'])
    ax_1.set_ylim([1e-10, 1])  # norm_rz[i][0]]/lb)
    print(max(map(len, norm_rr)))
    ax_1.set_xlim([0, max(map(len, norm_rr))])

    # ax_cross.set_yscale('linear')
    # ax_cross.set_yscale('linear')
    ax_geom.set_xticks([0, 32, 64])
    ax_geom.set_yticks([0, 32, 64])
    ax_1.legend(['Green', 'Green-Jacobi'], loc='best')
    # ax_1.legend(loc='best',ncol=2)

    fig.tight_layout()

    fname = src + 'exp_paper_JG_2D_elasticity_TO_64_sharp_vert_1' + '{}'.format('.pdf')
    print(('create figure: {}'.format(fname)))
    plt.savefig(fname, bbox_inches='tight')

plt.show()
quit()
fig = plt.figure(figsize=(11.5, 4))
gs = fig.add_gridspec(1, 3)
ax_1 = fig.add_subplot(gs[0, 2])
ax_cross = fig.add_subplot(gs[0, 1])
ax_geom = fig.add_subplot(gs[0, 0])
lines = ['-', '-.', '--', ':']
for i in np.arange(ratios.size, step=1):
    kappa = 10 ** ratios[i]

    k = np.arange(max(map(len, norm_rr)))
    # print(f'k \n {k}')

    convergence = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** k
    convergence = convergence  # *norm_rr[i][0]

    divnorm = mpl.colors.Normalize(vmin=1e-8, vmax=1)
    cmap_ = mpl.cm.seismic  # mpl.cm.seismic #mpl.cm.Greys
    geometry = np.load('../exp_data/' + name + f'_it{iteration}.npy', allow_pickle=True)
    phase_field_origin = np.abs(geometry)
    phase_field = scale_field_log(np.copy(phase_field_origin), min_val=1 / (10 ** ratios[i]), max_val=phase_field_max)
    # np.unravel_index(phase_field_origin.argmin(), phase_field_origin.shape)
    pcm = ax_geom.pcolormesh(np.tile(phase_field, (1, 1)),
                             cmap=cmap_, linewidth=0,
                             rasterized=True, norm=divnorm)
    ax_geom.axhline(y=min_idx[0], color='k', linestyle='-.')
    # ax_geom.set_aspect('equal', 'box')

    # min_idx
    ax_cross.semilogy(phase_field[min_idx[0], :], label=r'$\kappa=10^' + f'{{{-ratios[i]}}}$', color='k',
                      linestyle=lines[i])
    # ax_geom.set_title(r'Initial ', wrap=True)
    # ax_cross.set_ylabel(r'Density $\rho$')
    ax_cross.set_title(r'Cross section', wrap=True)
    ax_cross.set_xlabel('pixel index')

    ax_cross.set_xlim([0, 64])
    ax_cross.set_xticks([0, 32, 64])
    ax_cross.set_ylim([1 / (10 ** ratios[i]), 1.1])
    yticks = [1 / (10 ** ratios[d]) for d in range(ratios.size)]

    ax_cross.set_yticks([1, *yticks])
    ax_cross.set_xlim([0, 64])
    ax_cross.set_xticks([0, 32, 64])
    ax_cross.axhline(y=1 / (10 ** ratios[i]), color='k', linestyle=lines[i], linewidth=0.5)
    # ax_cross.set_yscale('linear')

    ax_geom.set_xticks([0, 32, 64])
    ax_geom.set_yticks([0, 32, 64])
    # ax_geom.axis('equal' )
    ax_geom.set_aspect('equal', 'box')
    ax_geom.set_xlabel('pixel index')
    ax_geom.set_title(r'Density $\rho$', wrap=True)

    # ax_geom.text(-0.18, 0.97, '(a)', transform=ax_geom.transAxes)
    # print(f'convergecnce \n {convergence}')
    # ax_1.set_title(f'Smooth', wrap=True)
    # ax_1.semilogy(convergence,  label=f'estim {kappa}', color='k', linestyle=lines[i])
    # ax_geom.set_xticks([])
    # ax_geom.set_xticks([])
    ax_1.loglog(norm_rMr[2 * i] / norm_rMr[2 * i][0], label=r'$\kappa=10^' + f'{{{-ratios[i]}}}$', color='g',
                linestyle=lines[i])
    #  ax_1.semilogy(norm_rMr[2*i+1]/norm_rMr[2*i+1][0], label=f'Green ' +r'$\kappa=10^'+f'{{{ratios[i]}}}$', color='r', linestyle=lines[i])
    ax_1.loglog(norm_rMr_combi[2 * i] / norm_rMr_combi[2 * i][0], label=r'$JG \kappa=10^' + f'{{{-ratios[i]}}}$',
                color='r', linestyle=lines[i])

    # ax_1.semilogy(norm_rMr_Jacobi[i]/norm_rMr_Jacobi[i][0], label=f' Jacobi {kappa}', color='b', linestyle=lines[i])
    # ax_1.semilogy(norm_rMr_combi[i]/norm_rMr_combi[i][0], label=f' Jacobi-Green {kappa}', color='r', linestyle=lines[i])

    # x_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
    # ax_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
    ax_1.set_xlabel('PCG iteration number')
    # ax_1.set_ylabel('Relative error')
    ax_1.set_title(r'Relative error', wrap=True)

    # plt.legend([r'$\kappa$ upper bound','Green', 'Jacobi', 'Green + Jacobi','Richardson'])
    ax_1.set_ylim([1e-10, 1])  # norm_rz[i][0]]/lb)
    print(max(map(len, norm_rr)))
    ax_1.set_xlim([0, max(map(len, norm_rr))])
    ax_1.legend(['Green', 'Green-Jacobi'], loc='best')
    fig.tight_layout()

    fname = src + 'exp_paper_JG_2D_elasticity_TO_64_smooth_vert' + '{}'.format('.pdf')
    print(('create figure: {}'.format(fname)))
    plt.savefig(fname, bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(11.5, 4.))
gs = fig.add_gridspec(1, 3)
ax_1 = fig.add_subplot(gs[0, 2])
ax_cross = fig.add_subplot(gs[0, 1])
ax_geom = fig.add_subplot(gs[0, 0])

lines = ['-', '-.', '--', ':']
for i in np.arange(ratios.size, step=1):
    kappa = 10 ** ratios[i]

    k = np.arange(max(map(len, norm_rr)))
    # print(f'k \n {k}')

    convergence = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** k
    convergence = convergence  # *norm_rr[i][0]
    divnorm = mpl.colors.Normalize(vmin=1e-8, vmax=1)
    cmap_ = mpl.cm.seismic  # mpl.cm.seismic #mpl.cm.Greys
    geometry = np.load('../exp_data/' + name + f'_it{iteration}.npy', allow_pickle=True)
    phase_field_origin = np.abs(geometry)

    phase_field[phase_field < 0.5] = 1 / 10 ** ratios[i]  # phase_field_min#
    phase_field[phase_field > 0.49] = phase_field_max  # 1

    phase_field = scale_field_log(np.copy(phase_field), min_val=1 / (10 ** ratios[i]), max_val=phase_field_max)

    pcm = ax_geom.pcolormesh(np.tile(phase_field, (1, 1)),
                             cmap=cmap_, linewidth=0,
                             rasterized=True, norm=divnorm)
    ax_geom.axhline(y=min_idx[0], color='k', linestyle='-.')

    ax_cross.semilogy(phase_field[min_idx[0], :], label=r'$\kappa=10^' + f'{{{-ratios[i]}}}$', color='k',
                      linestyle=lines[i])
    # ax_geom.set_title(r'Initial ', wrap=True) # :,15
    # ax_cross.set_ylabel(r'Density $\rho$')
    # ax_cross.set_title(r'Density $\rho$', wrap=True)
    ax_cross.set_title(r'Cross section', wrap=True)
    ax_cross.set_xlabel('pixel index')
    ax_cross.set_ylim([1 / (10 ** ratios[i]), 1.1])
    yticks = [1 / (10 ** ratios[d]) for d in range(ratios.size)]

    ax_cross.set_yticks([1, *yticks])
    ax_cross.set_xlim([0, 64])
    ax_cross.set_xticks([0, 32, 64])
    ax_cross.axhline(y=1 / (10 ** ratios[i]), color='k', linestyle=lines[i], linewidth=0.5)

    # ax_cross.set_yscale('linear')

    ax_geom.set_xticks([0, 32, 64])
    ax_geom.set_yticks([0, 32, 64])
    ax_geom.set_aspect('equal', 'box')
    ax_geom.set_xlabel('pixel index')
    ax_geom.set_title(r'Density $\rho$', wrap=True)
    # print(f'convergecnce \n {convergence}')
    # ax_1.set_title(f'{i}', wrap=True)
    # ax_1.semilogy(convergence,  label=f'estim {kappa}', color='k', linestyle=lines[i])

    # ax_1.semilogy(norm_rMr[2*i]/norm_rMr[2*i][0], label=f'Green ' +r'$\kappa=10^'+f'{{{ratios[i]}}}$', color='g', linestyle=lines[i])
    ax_1.loglog(norm_rMr[2 * i + 1] / norm_rMr[2 * i + 1][0], label=r'$\kappa=10^' + f'{{{-ratios[i]}}}$', color='g',
                linestyle=lines[i])
    ax_1.loglog(norm_rMr_combi[2 * i + 1] / norm_rMr_combi[2 * i + 1][0],
                label=r'$JG \kappa=10^' + f'{{{-ratios[i]}}}$', color='r', linestyle=lines[i])

    # ax_1.semilogy(norm_rMr_Jacobi[i]/norm_rMr_Jacobi[i][0], label=f' Jacobi {kappa}', color='b', linestyle=lines[i])
    # ax_1.semilogy(norm_rMr_combi[i]/norm_rMr_combi[i][0], label=f' Jacobi-Green {kappa}', color='r', linestyle=lines[i])

    # x_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
    # ax_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
    ax_1.set_xlabel('PCG iteration number ')
    # ax_1.set_ylabel('Relative error')
    ax_1.set_title(r'Relative error', wrap=True)

    # plt.legend([r'$\kappa$ upper bound','Green', 'Jacobi', 'Green + Jacobi','Richardson'])
    ax_1.set_ylim([1e-10, 1])  # norm_rz[i][0]]/lb)
    print(max(map(len, norm_rr)))
    ax_1.set_xlim([0, max(map(len, norm_rr))])

    # ax_cross.set_yscale('linear')
    # ax_cross.set_yscale('linear')
    ax_geom.set_xticks([0, 32, 64])
    ax_geom.set_yticks([0, 32, 64])
    ax_1.legend(['Green', 'Green-Jacobi'], loc='best')
    # ax_1.legend(loc='best',ncol=2)

    fig.tight_layout()

    fname = src + 'exp_paper_JG_2D_elasticity_TO_64_sharp_vert' + '{}'.format('.pdf')
    print(('create figure: {}'.format(fname)))
    plt.savefig(fname, bbox_inches='tight')

plt.show()

quit()

fig = plt.figure()
gs = fig.add_gridspec(1, 1)
ax_1 = fig.add_subplot(gs[0, 0])
# ax_1.semilogy(norm_rr[0], label='PCG: Green', color='blue', linewidth=0)
for kk in np.arange(np.size(nb_pix_multips)):
    ax_1.plot(ratios[0:], nb_it[kk], 'g', marker='|', label=' Green', linewidth=1)
    # axs[1].plot(xopt2.f.num_iteration_.transpose()[1:3*i+1:3],"r", label='DGO ',linewidth=1)
    # axs[1].plot(xopt2.f.num_iteration_.transpose()[2:3*i+2:3],"r", label='DGO ',linewidth=1)

    ax_1.plot(ratios[0:], nb_it_Jacobi[kk], "b", marker='o', label='PCG Jacobi', linewidth=1)  # [0, 0:]
    ax_1.plot(ratios[0:], nb_it_combi[kk], "k", marker='x', label='PCG Green + Jacobi', linewidth=1)
#  ax2.semilogy(ratios[0:i + 1], nb_it_Richardson[0, 0:i + 1], "g", label=' Richardson Green ', linewidth=1)
#  ax2.semilogy(ratios[0:i + 1], nb_it_Richardson_combi[0, 0:i + 1], "y",  label=' Richardson Green + Jacobi ', linewidth=1)

# axs[1].legend()
ax_1.set_ylim(bottom=0)
ax_1.legend(['Green', 'Jacobi', 'Green + Jacobi'])
plt.show()
# quit()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each line with a different z offset
for i in np.arange(len(nb_pix_multips)):
    ax.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it[i], label='PCG: Green', color='blue')
    ax.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Jacobi[i], label='PCG: Jacobi', color='black')
    ax.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_combi[i], label='PCG: Green + Jacobi', color='red')
    ax.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
    ax.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
ax.set_xlabel('nb of filter aplications')
ax.set_ylabel('size')
ax.set_zlabel('# CG iterations')
plt.legend(['DGO', 'Jacobi', 'DGO + Jacobi', 'Richardson'])
plt.show()

for i in np.arange(ratios.size, step=1):
    kappa = 10 ** kontrast[i]

    k = np.arange(max(map(len, norm_rr)))
    print(f'k \n {k}')
    lb = eigen_LB[i]
    print(f'lb \n {lb}')

    convergence = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** k
    convergence = convergence * norm_rr[i][0]

    print(f'convergecnce \n {convergence}')
    fig = plt.figure()
    gs = fig.add_gridspec(1, 1)
    ax_1 = fig.add_subplot(gs[0, 0])
    ax_1.set_title(f'{i}', wrap=True)
    ax_1.semilogy(convergence, '--', label='estim', color='k')

    ax_1.semilogy(norm_rr[i], label=' Green', color='g')
    ax_1.semilogy(norm_rr_Jacobi[i], label=' Jacobi', color='b')
    ax_1.semilogy(norm_rr_combi[i], label=' Jacobi-Green', color='r')

    # x_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
    # ax_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
    ax_1.set_xlabel('CG iterations')
    ax_1.set_ylabel('Norm of residua')
    plt.legend([r'$\kappa$ upper bound', 'Green', 'Jacobi', 'Green + Jacobi', 'Richardson'])
    ax_1.set_ylim([1e-10, norm_rr[i][0]])  # norm_rz[i][0]]/lb)
    print(max(map(len, norm_rr)))
    ax_1.set_xlim([0, max(map(len, norm_rr))])

    plt.show()
