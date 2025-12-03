import time
import os
import sys

sys.path.append("/home/martin/Programming/muFFTTO_paralellFFT_test/muFFTTO")
sys.path.append('../..')

import numpy as np
from mpi4py import MPI
from NuMPI.IO import save_npy, load_npy

from muFFTTO import domain
from muFFTTO import solvers

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


def scale_field_mugrid(field, min_val, max_val):
    """Scales a 2D random field to be within [min_val, max_val]."""
    field_min = discretization.mpi_reduction.min(field.s)
    field_max = discretization.mpi_reduction.max(field.s)
    field.s = (field.s - field_min) / (field_max - field_min)  # Normalize to [0,1]
    field.s *= (max_val - min_val)
    field.s += min_val


def scale_field_log(field, min_val, max_val):
    """Scales a 2D random field to be within [min_val, max_val]."""
    field_log = np.log10(field)
    field_min, field_max = np.min(field_log), np.max(
        field_log)

    scaled_field = (field_log - field_min) / (field_max - field_min)  # Normalize to [0,1]
    return 10 ** (scaled_field * (np.log10(max_val) - np.log10(min_val)) + np.log10(
        min_val))  # Scale to [min_val, max_val]


plot = False
compute = True
enforce_mean = False
if compute:

    domain_size = [1, 1]

    ratios = np.array([15])  # 5, 8, 12, 15, 12,15

    number_of_pixels = (1024, 1024)

    my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                      problem_type=problem_type)

    discretization = domain.Discretization(cell=my_cell,
                                           nb_of_pixels_global=number_of_pixels,
                                           discretization_type=discretization_type,
                                           element_type=element_type)
    start_time = time.time()

    # set macroscopic gradient
    # macro_gradient = np.array([[1.0, 0.5], [0.5, 1.0]])
    macro_gradient = np.array([[1.0, 0.5], [0.5, 1.0]])

    # create material data field
    K_0, G_0 = 1, 0.5  # domain.get_bulk_and_shear_modulus(E=1, poison=0.2)

    elastic_C_1 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                     K=K_0,
                                                     mu=G_0,
                                                     kind='linear')
    C_1 = domain.compute_Voigt_notation_4order(elastic_C_1)

    material_data_field_C_0 = discretization.get_material_data_size_field_mugrid(name='mat_Data')
    material_data_field_C_0.s = np.einsum('ijkl,qxy->ijklqxy', elastic_C_1,
                                          np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                            *discretization.nb_of_pixels])))

    # print('elastic tangent = \n {}'.format(domain.compute_Voigt_notation_4order(elastic_C_1)))

    macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_inc_field')
    rhs_field = discretization.get_unknown_size_field(name='rhs_field')

    # Set up right hand side
    discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient,
                                                   macro_gradient_field_ijqxyz=macro_gradient_field)

    for i in np.arange(ratios.shape[0]):
        ratio = ratios[i]

        for sharp in [False, True]:

            # material distribution
            name = 'microstructure_1024'
            # geometries_data_folder_path = ''
            geometries_data_folder_path = '//work/classic/fr_ml1145-martin_workspace_01/muFFTTO/experiments/paper_Jacobi_Green/'
            phase_field = discretization.get_scalar_field(name='phase_field')

            phase_field.s[0, 0] = load_npy(os.path.expanduser(geometries_data_folder_path + name + f'.npy'),
                                           subdomain_locations=tuple(discretization.subdomain_locations_no_buffers),
                                           nb_subdomain_grid_pts=tuple(discretization.nb_of_pixels),
                                           comm=MPI.COMM_WORLD)

            phase_field.s[0, 0] = phase_field.s[0, 0] ** 2

            phase_field_min = discretization.mpi_reduction.min(phase_field.s)
            phase_field_max = discretization.mpi_reduction.max(phase_field.s)

            if sharp:
                phase_field.s[phase_field.s < 0.5] = 1 / 10 ** ratio  #
                phase_field.s[phase_field.s > 0.49] = phase_field_max  # 1

            scale_field_mugrid(phase_field,
                               min_val=1 / (10 ** ratio),
                               max_val=phase_field_max)
            if discretization.fft.communicator.rank == 0:
                print(f'ratio={ratio} ')

            results_name = (f'phase_field' + f'N1024_{ratio}_sharp_{sharp}')
            save_npy(data_folder_path + results_name + f'.npy', phase_field.s[0].mean(axis=0),
                     tuple(discretization.subdomain_locations_no_buffers),
                     tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD)

            material_data_field_C_0_rho = discretization.get_material_data_size_field_mugrid(name='mat_data_rho')
            material_data_field_C_0_rho.s = np.copy(material_data_field_C_0.s[..., :, :, :]) * phase_field.s

            # Set up RHS
            discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0_rho,
                                          macro_gradient_field_ijqxyz=macro_gradient_field,
                                          rhs_inxyz=rhs_field)


            # Set up Hessian
            def K_fun(x, Ax):
                discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0_rho,
                                                          input_field_inxyz=x,
                                                          output_field_inxyz=Ax,
                                                          formulation='small_strain')


            preconditioner = discretization.get_preconditioner_Green_mugrid(
                reference_material_data_ijkl=elastic_C_1)

            K_diag_alg = discretization.get_preconditioner_Jacobi_mugrid(
                material_data_field_ijklqxyz=material_data_field_C_0_rho,
                formulation=formulation)


            def M_fun_green(x, Px):
                """
                Function to compute the product of the Preconditioner matrix with a vector.
                The Preconditioner is represented by the convolution operator.
                """
                # discretization.fft.communicate_ghosts(x)
                discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                                           input_nodal_field_fnxyz=x,
                                                           output_nodal_field_fnxyz=Px)


            def M_fun_Green_Jacobi(x, Px):
                # discretization.fft.communicate_ghosts(x)
                x_jacobi_temp = discretization.get_unknown_size_field(name='x_jacobi_temp')

                x_jacobi_temp.s = K_diag_alg.s * x.s
                discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                                           input_nodal_field_fnxyz=x_jacobi_temp,
                                                           output_nodal_field_fnxyz=Px)

                Px.s = K_diag_alg.s * Px.s
                # discretization.fft.communicate_ghosts(Px)


            def M_fun_Jacobi(x, Px):
                Px.s = K_diag_alg.s * K_diag_alg.s * x.s
                # discretization.fft.communicate_ghosts(Px)


            # M_fun_GJ = lambda x: K_diag_alg * discretization.apply_preconditioner_NEW(
            #     preconditioner_Fourier_fnfnqks=preconditioner,
            #     nodal_field_fnxyz=K_diag_alg * x)
            # if enforce_mean:
            #     # M_fun_combi = lambda x: (y := M_fun_GJ(x)) - np.mean(y)
            #     M_fun_combi = lambda x: (y := M_fun_GJ(x)) - np.mean(y, axis=(-1, -2, -3), keepdims=True)
            # else:
            #     M_fun_combi = lambda x: M_fun_GJ(x)

            # M_fun_combi = lambda x: 1 * x
            # #
            # M_fun_Jacobi = lambda x: K_diag_alg * K_diag_alg * x
            # M_fun_Jacobi = lambda x: 1 * x

            norms_G = dict()
            norms_G['residual_rr'] = []
            norms_G['residual_rz'] = []
            norms_G['residual_rGr'] = []


            def callback_G(it, x, r, p, z, stop_crit_norm):
                global norms_G

                """
                Callback function to print the current solution, residual, and search direction.
                """
                norm_of_rr = discretization.fft.communicator.sum(np.dot(r.ravel(), r.ravel()))
                norm_of_rz = discretization.fft.communicator.sum(np.dot(r.ravel(), z.ravel()))
                norms_G['residual_rr'].append(norm_of_rr)
                norms_G['residual_rz'].append(norm_of_rz)
                norms_G['residual_rGr'].append(stop_crit_norm)

                if discretization.fft.communicator.rank == 0:
                    print(len(norms_G['residual_rr']))
                    print(norm_of_rr)


            # init solution
            solution_field = discretization.get_unknown_size_field(name='solution')
            solution_field.s.fill(0)
            solvers.conjugate_gradients_mugrid(
                comm=discretization.fft.communicator,
                fc=discretization.field_collection,
                hessp=K_fun,  # linear operator
                b=rhs_field,
                x=solution_field,
                P=M_fun_green,
                tol=1e-5,
                maxiter=20000,
                callback=callback_G,
                norm_metric=M_fun_green
            )
            #########

            norms_J = dict()
            norms_J['residual_rr'] = []
            norms_J['residual_rz'] = []
            norms_J['residual_rGr'] = []


            def callback_J(it, x, r, p, z, stop_crit_norm):
                global norms_J

                """
                Callback function to print the current solution, residual, and search direction.
                """
                norm_of_rr = discretization.fft.communicator.sum(np.dot(r.ravel(), r.ravel()))
                norm_of_rz = discretization.fft.communicator.sum(np.dot(r.ravel(), z.ravel()))
                norms_J['residual_rr'].append(norm_of_rr)
                norms_J['residual_rz'].append(norm_of_rz)
                norms_J['residual_rGr'].append(stop_crit_norm)


            solution_field.s.fill(0)
            solvers.conjugate_gradients_mugrid(
                comm=discretization.fft.communicator,
                fc=discretization.field_collection,
                hessp=K_fun,  # linear operator
                b=rhs_field,
                x=solution_field,
                P=M_fun_Jacobi,
                tol=1e-5,
                maxiter=1,
                callback=callback_J,
                norm_metric=M_fun_green
            )

            norms_GJ = dict()
            norms_GJ['residual_rr'] = []
            norms_GJ['residual_rz'] = []
            norms_GJ['residual_rGr'] = []


            def callback_GJ(it, x, r, p, z, stop_crit_norm):
                global norms_GJ

                """
                Callback function to print the current solution, residual, and search direction.
                """
                norm_of_rr = discretization.fft.communicator.sum(np.dot(r.ravel(), r.ravel()))
                norm_of_rz = discretization.fft.communicator.sum(np.dot(r.ravel(), z.ravel()))
                norms_GJ['residual_rr'].append(norm_of_rr)
                norms_GJ['residual_rz'].append(norm_of_rz)
                norms_GJ['residual_rGr'].append(stop_crit_norm)


            solution_field.s.fill(0)
            solvers.conjugate_gradients_mugrid(
                comm=discretization.fft.communicator,
                fc=discretization.field_collection,
                hessp=K_fun,  # linear operator
                b=rhs_field,
                x=solution_field,
                P=M_fun_Green_Jacobi,
                tol=1e-5,
                maxiter=20000,
                callback=callback_GJ,
                norm_metric=M_fun_green
            )

            # displacement_field_Richardson, norms_Richardson = solvers.Richardson(K_fun, rhs, x0=None, P=M_fun,
            #                                                                      omega=omega,
            #                                                                      steps=int(1000),
            #                                                                      toler=1e-1)

            if discretization.fft.communicator.rank == 0:
                _info = {}
                _info['norms_G_rr'] = norms_G['residual_rr']
                _info['norms_G_rz'] = norms_G['residual_rz']
                _info['norms_G_rGr'] = norms_G['residual_rGr']

                _info['norms_J_rr'] = norms_J['residual_rr']
                _info['norms_J_rz'] = norms_J['residual_rz']
                _info['norms_J_rGr'] = norms_J['residual_rGr']

                _info['norms_GJ_rr'] = norms_GJ['residual_rr']
                _info['norms_GJ_rz'] = norms_GJ['residual_rz']
                _info['norms_GJ_rGr'] = norms_GJ['residual_rGr']
                print(f'Green_{ratio}_sharp_{sharp} iters =' + '{}'.format(len(_info['norms_G_rr'])))
                print(f'Jacobi_{ratio}_sharp_{sharp} iters =' + '{}'.format(len(_info['norms_J_rr'])))
                print(f'Green_Jacobi_{ratio}_sharp_{sharp} iters =' + '{}'.format(len(_info['norms_GJ_rr'])))

                results_name = f'N1024_{ratio}_sharp_{sharp}'
                np.savez(data_folder_path + results_name + f'_log.npz', **_info)
                print(data_folder_path + results_name + f'_log.npz')

if plot:
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    plt.rcParams["text.usetex"] = True
    plt.rcParams.update({
        "text.usetex": True,  # Use LaTeX
        # "font.family": "helvetica",  # Use a serif font
    })
    plt.rcParams.update({'font.size': 11})
    plt.rcParams["font.family"] = "Arial"

    # plt.rcParams.update({'font.size': 14})
    # ratios = np.array([2, 5, 8])
    ratios = np.array([2, 3, 4])  # 5, 8, 12, 15, 15
    # fig = plt.figure(figsize=(11.5, 6))
    fig = plt.figure(figsize=(8.3, 5.0))

    # gs = fig.add_gridspec(1, 3)
    gs_global = fig.add_gridspec(1, 2, width_ratios=[1, 2], wspace=0.2)

    gs_error = gs_global[1].subgridspec(2, 1, width_ratios=[1], hspace=0.2)  # 0.1, 1, 4
    gs_geom = gs_global[0].subgridspec(2, 2, width_ratios=[0.07, 1], hspace=0.2, wspace=0.7)  # 0.1, 1, 4

    ax_cbar = fig.add_subplot(gs_geom[:, 0])
    lines = ['-', '-.', '--', ':', 'dotted']
    row = 0
    for sharp in [False, True]:
        ax_error = fig.add_subplot(gs_error[row, 0])
        ax_error.text(-0.12, 1.05, rf'\textbf{{(b.{row + 1}}})  ', transform=ax_error.transAxes)

        ax_geom = fig.add_subplot(gs_geom[row, 1])

        if row == 0:
            ax_geom.text(-0.2, 1.22, rf'\textbf{{(a.{row + 1}) }} ', transform=ax_geom.transAxes)
        elif row == 1:
            ax_geom.text(-0.2, 1.21, rf'\textbf{{(a.{row + 1})}}  ', transform=ax_geom.transAxes)

        divnorm = mpl.colors.Normalize(vmin=1e-8, vmax=1)
        cmap_ = mpl.cm.seismic  # mpl.cm.seismic #mpl.cm.Greys

        results_name = (f'phase_field' + f'N1024_{ratios[0]}_sharp_{sharp}')
        phase_field = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True)

        phase_field_max = np.max(phase_field)
        if sharp:
            phase_field[phase_field < 0.5] = 1 / 10 ** ratios[-1]  # phase_field_min#
            phase_field[phase_field > 0.49] = phase_field_max  # 1

        # np.unravel_index(phase_field_origin.argmin(), phase_field_origin.shape)
        pcm = ax_geom.pcolormesh(np.tile(phase_field, (1, 1)),
                                 cmap=cmap_, linewidth=0,
                                 rasterized=True, norm=divnorm)

        ax_geom.set_xticks([1, 512, 1024])
        ax_geom.set_yticks([1, 512, 1024])
        # ax_geom.axis('equal' )
        ax_geom.set_aspect('equal', 'box')
        if sharp:
            ax_geom.set_xlabel('pixel index')
        if sharp:
            ax_geom.set_title(r'$\rho_{\rm sharp}$', wrap=True)  # Density
        else:
            ax_geom.set_title(r'$\rho_{\rm smooth}$', wrap=True)  # $Density

        for i in np.arange(ratios.size, step=1):
            results_name = f'N1024_{ratios[i]}_sharp_{sharp}'
            _info = np.load(data_folder_path + results_name + f'_log.npz', allow_pickle=True)

            norm_G = _info['norms_G_rr']
            norm_GJ = _info['norms_GJ_rr']
            norm_J = _info['norms_J_rr']
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
            ax_error.loglog(np.arange(len(norm_G)), relative_error_G, label=fr'$\kappa=10^{{{-ratios[i]}}}$',
                            color='g', linestyle=lines[i], lw=2)
            #  ax_1.semilogy(norm_rMr[2*i+1]/norm_rMr[2*i+1][0], label=f'Green ' +r'$\kappa=10^'+f'{{{ratios[i]}}}$', color='r', linestyle=lines[i])
            ax_error.loglog(np.arange(len(norm_GJ)), relative_error_GJ, label=r'$JG \kappa=10^' + f'{{{-ratios[i]}}}$',
                            color='black', linestyle=lines[i], lw=2)

            # ax_1.semilogy(norm_rMr_Jacobi[i]/norm_rMr_Jacobi[i][0], label=f' Jacobi {kappa}', color='b', linestyle=lines[i])
            # ax_1.semilogy(norm_rMr_combi[i]/norm_rMr_combi[i][0], label=f' Jacobi-Green {kappa}', color='r', linestyle=lines[i])

            # x_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
            # ax_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
            if sharp:
                ax_error.set_xlabel(r'PCG iteration - $k$')

            ax_error.set_ylabel('Relative norm of residual')  # - '  fr'$||r_{{k}}||_{{\mathbdf{{G}} }}   $')#^{-1}
            # ax_error.set_title(r'Relative  norm of residua', wrap=True)

            # plt.legend([r'$\kappa$ upper bound','Green', 'Jacobi', 'Green + Jacobi','Richardson'])
            # ax_error.set_ylim([1e-6, 1e2])  # norm_rz[i][0]]/lb)
            ax_error.set_xlim([1, 1e4])
            # ax_error.set_xscale('linear')
            ax_error.set_yticks([1e-6, 1e-2, 1e2])
            ax_error.set_yticklabels([fr'$10^{{{-6}}}$', fr'$10^{{{-2}}}$', fr'$10^{{{2}}}$'])

            if sharp:

                arrows_G = [5, 10, 15]  # anotation arrows
                text_G = np.array([[1.2, 1e-2],  # anotation Text position
                                   [1.6, 1e-4],
                                   [2.0, 2e-6]])

                arrows_GJ = [300, 600, 1000]  # anotation arrows
                text_GJ = np.array([[30, 1e-2],  # anotation Text position
                                    [40, 1e-4],
                                    [80, 3e-6]])
            else:
                arrows_G = [11, 200, 1000]
                text_G = np.array([[500, 1e-0],
                                   [1000, 1e-2],
                                   [2000, 8e-5]])

                arrows_GJ = [15, 60, 110]
                text_GJ = np.array([[1.2, 1e-2],
                                    [1.6, 2e-4],
                                    [2.0, 3e-6]])
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

            # ax_error.annotate(text=f'Green-Jacobi\n' + fr'  $\chi^{{\mathrm{{tot}}}} =10^{{{ratios[i]}}}$',
            #                   xy=(arrows_GJ[i], relative_error_GJ[arrows_GJ[i]]),
            #                   xytext=(text_GJ[i, 0], text_GJ[i, 1]),
            #                   arrowprops=dict(arrowstyle='->',
            #                                   color='black',
            #                                   lw=1,
            #                                   ls=lines[i]),
            #                   fontsize=9,
            #                   color='black'
            #                   )
            # ax_error.annotate(text=f'Green\n' + fr'$\chi^{{\mathrm{{tot}}}} = 10^{{{ratios[i]}}}$',
            #                   xy=(arrows_G[i], relative_error_G[arrows_G[i]]),
            #                   xytext=(text_G[i, 0], text_G[i, 1]),
            #                   arrowprops=dict(arrowstyle='->',
            #                                   color='green',
            #                                   lw=1,
            #                                   ls=lines[i]),
            #                   fontsize=9,
            #                   color='green'
            #                   )
            # ax_error.yaxis.set_ticks_position('right')  # Set y-axis ticks to the right

        cbar = plt.colorbar(pcm, location='left', cax=ax_cbar)
        cbar.ax.yaxis.tick_left()
        # cbar.set_ticks(ticks=[1e-4,1e-2, 1])
        # cbar.set_ticklabels([f'$10^{{{-4}}}$', f'$10^{{{-2}}}$', 1])
        cbar.set_ticks(ticks=[1e-8, 0.5, 1])
        cbar.set_ticklabels([r'$\frac{1}{\chi^{\rm tot}}$', 0.5, 1])
        row += 1

    fig.tight_layout()
    fname = f'exp_paper_JG_TO_1024_sharp_vs_smoot' + '{}'.format('.pdf')
    print(('create figure: {}'.format(fname)))
    plt.savefig(figure_folder_path + fname, bbox_inches='tight')
    plt.show()
quit()
