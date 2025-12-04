import numpy as np
import scipy as sc
import time
import os
from NuMPI.IO import save_npy, load_npy
from mpi4py import MPI

import matplotlib as mpl
from matplotlib import pyplot as plt

plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX
    # "font.family": "helvetica",  # Use a serif font
})
plt.rcParams.update({'font.size': 11})
plt.rcParams["font.family"] = "Arial"

script_name = 'exp_paper_JG_nonlinear_elasticity_JZ'  # exp_paper_JG_nonlinear_elasticity_JZ
folder_name = '../exp_data/'
file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory

figure_folder_path = file_folder_path + '/figures/' + script_name + '/'

plot_time_vs_dofs = True
plot_stress_field = False
plot_data_vs_CG = True

if plot_time_vs_dofs:
    # print time vs DOFS
    time_G = []
    time_GJ = []
    its_G = []
    its_GJ = []
    Ns = 2 ** np.array([3, 4, 5, 6])  # 4, 5,6,7,8  # numbers of grids points
    for N in Ns:
        Nx = N
        Ny = N
        Nz = N  # N#
        preconditioner_type = 'Green'

        data_folder_path = (file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                            + f'_{preconditioner_type}' + '/')

        _info_final_G = np.load(data_folder_path + f'info_log_final.npz', allow_pickle=True)

        time_G.append(_info_final_G.f.elapsed_time)
        its_G.append(_info_final_G.f.sum_CG_its)

        preconditioner_type = 'Green_Jacobi'

        data_folder_path = (file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                            + f'_{preconditioner_type}' + '/')

        _info_final_GJ = np.load(data_folder_path + f'info_log_final.npz', allow_pickle=True)

        time_GJ.append(_info_final_GJ.f.elapsed_time)
        its_GJ.append(_info_final_GJ.f.sum_CG_its)
    time_G = np.array(time_G)
    time_GJ = np.array(time_GJ)
    its_G = np.array(its_G)
    its_GJ = np.array(its_GJ)

    fig = plt.figure(figsize=(9, 4.50))
    gs = fig.add_gridspec(1, 1, hspace=0.5, wspace=0.5, width_ratios=[1],
                          height_ratios=[1])
    nb_dofs = 3 * Ns ** 2
    line1, = plt.loglog(nb_dofs, time_G, '-x', color='Green', label='Green')
    line2, = plt.loglog(nb_dofs, time_GJ, 'k-', marker='o', markerfacecolor='none', label='Green-Jacobi')
    line3, = plt.loglog(nb_dofs, nb_dofs * np.log(nb_dofs) / (nb_dofs[0] * np.log(nb_dofs[0])) * time_G[0], ':',
                        label=r'Quasilinear - $ \mathcal{O} (N_{\mathrm{N}} \log  N_{\mathrm{N}}$)')
    # plt.loglog(nb_dofs, nb_dofs / (nb_dofs[0]) * time_G[0], '--', label='linear')

    line4, = plt.loglog(nb_dofs, time_G / its_G, 'g--x', label='Green')
    line5, = plt.loglog(nb_dofs, time_GJ / its_GJ, 'k--', marker='o', markerfacecolor='none', label='Green-Jacobi')
    # plt.loglog(nb_dofs,
    #            nb_dofs* np.log(nb_dofs) / (nb_dofs[0]  * np.log(nb_dofs)) * time_G[0] / its_G[0], ':',
    #            label='N log N')
    line6, = plt.loglog(nb_dofs, nb_dofs / (nb_dofs[0]) * time_G[0] / its_G[0], '--',
                        label=r'Linear - $\mathcal{O} (N_{\mathrm{N}})$')
    plt.loglog(np.linspace(1e1, 1e8), 1e-4 * np.linspace(1e1, 1e8), 'k-', linewidth=0.9)

    plt.xlabel(r' $\#$ of degrees of freedom (DOFs) - $d N_{\mathrm{N}}$')
    plt.ylabel('Time (s)')
    plt.gca().set_xlim([nb_dofs[0], nb_dofs[-1]])
    plt.gca().set_xticks([1e3, 1e4, 1e5, 1e6])
    plt.gca().set_ylim([1e-3, 1e4])

    # plt.gca().set_xticks(iterations)

    legend1 = plt.legend(handles=[line1, line2, line3], loc='upper left', title='Wall-clock time')

    plt.gca().add_artist(legend1)  # Add the first legend manually

    # Second legend (bottom right)
    plt.legend(handles=[line4, line5, line6], loc='lower right', title='Wall-clock time / $\#$ of PCG iteration')

    fig.tight_layout()
    fname = f'time_scaling' + '{}'.format('.pdf')
    plt.savefig(figure_folder_path + script_name + fname, bbox_inches='tight')
    print(('create figure: {}'.format(figure_folder_path + script_name + fname)))

    plt.show()

if plot_data_vs_CG:
    # print time vs DOFS
    its_G = []
    its_GJ = []
    stress_diff_norm = []
    strain_fluc_norm = []
    strain_total_norm = []
    diff_rhs_norm = []
    norm_rhs_G = []
    norm_rhs_GJ = []
    rhs_inf_G = []
    rhs_inf_GJ = []
    norm_newrton_stop_G = []
    norm_newrton_stop_GJ = []

    Nx = 2 ** 8
    Ny = Nx
    Nz = Nx
    it_max = 10
    n_exponents = np.array([10])
    iterations = np.arange(it_max)  # numbers of grids points

    its_G = np.zeros([it_max, len(n_exponents)])
    its_GJ = np.zeros([it_max, len(n_exponents)])

    for j in np.arange(len(n_exponents)):
        n_exp = n_exponents[j]
        for iteration_total in iterations:

            preconditioner_type = 'Green'

            data_folder_path = (
                        file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                        + f'_{preconditioner_type}' + '/')
            if iteration_total < it_max:
                if Nx == 256:
                    _info_final_G = np.load(data_folder_path + f'info_log_it{iteration_total}.npz', allow_pickle=True)

                else:
                    _info_final_G = np.load(data_folder_path + f'info_log_exp_{n_exp}_it{iteration_total}.npz',
                                            allow_pickle=True)

            with open(data_folder_path + f'stress' + f'_it{iteration_total}' + f'.npy', 'rb') as f:
                magic = f.read(6)
                print(f"Magic number: {magic}")

            # strain_fluc_G = np.load(data_folder_path + f'strain_fluc_field' + f'_it{iteration_total}' + f'.npy',
            #                         allow_pickle=True)
            # strain_total_G = np.load(data_folder_path + f'total_strain_field' + f'_it{iteration_total}' + f'.npy',
            #                          allow_pickle=True)
            # stress_G = np.load(data_folder_path + f'stress' + f'_it{iteration_total}' + f'.npy',
            #                    allow_pickle=True)  # , allow_pickle=True

            # rhs_field_G = np.load(data_folder_path + f'rhs_field' + f'_it{iteration_total}' + f'.npy', allow_pickle=True)

            its_G[iteration_total, j] = _info_final_G.f.nb_it_comb
            norm_rhs_G.append(_info_final_G.f.norm_rhs_field)
            norm_newrton_stop_G.append(_info_final_G.f.newton_stop_crit)
            info_log_final_G = np.load(data_folder_path + f'info_log_final.npz', allow_pickle=True)

            preconditioner_type = 'Green_Jacobi'

            data_folder_path = (
                        file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                        + f'_{preconditioner_type}' + '/')
            if iteration_total < it_max:
                if Nx == 256:
                    _info_final_GJ = np.load(data_folder_path + f'info_log_it{iteration_total}.npz',
                                             allow_pickle=True)

                else:
                    _info_final_GJ = np.load(data_folder_path + f'info_log_exp_{n_exp}_it{iteration_total}.npz',
                                             allow_pickle=True)
            # stress_GJ = np.load(data_folder_path + f'stress' + f'_it{iteration_total}' + f'.npy', allow_pickle=True)
            # strain_fluc_GJ = np.load(data_folder_path + f'strain_fluc_field' + f'_it{iteration_total}' + f'.npy',
            #                          allow_pickle=True)
            # strain_total_GJ = np.load(data_folder_path + f'total_strain_field' + f'_it{iteration_total}' + f'.npy',
            #                           allow_pickle=True)
            # rhs_field_GJ = np.load(data_folder_path + f'rhs_field' + f'_it{iteration_total}' + f'.npy', allow_pickle=True)

            its_GJ[iteration_total, j] = _info_final_GJ.f.nb_it_comb
            norm_rhs_GJ.append(_info_final_GJ.f.norm_rhs_field)
            norm_newrton_stop_GJ.append(_info_final_GJ.f.newton_stop_crit)
            # diff_stress = stress_G - stress_GJ
            # stress_diff_norm.append(
            #      np.linalg.norm(diff_stress.ravel(), ord=np.inf))  # / np.linalg.norm(stress_G))
            print(_info_final_G.f.norm_rr[0])

            # diff_strain_fluc = strain_fluc_G - strain_fluc_GJ
            # strain_fluc_norm.append(
            #     np.linalg.norm(diff_strain_fluc.ravel(), ord=np.inf))  # / _info_final_GJ.f.norm_En)
            print(_info_final_GJ.f.norm_rr[0])
            # diff_strain_total = strain_total_G - strain_total_GJ
            # strain_total_norm.append(
            #    np.linalg.norm(diff_strain_total.ravel(), ord=np.inf))  # / _info_final_GJ.f.norm_En)

            # diff_rhs = rhs_field_G - rhs_field_GJ
            # diff_rhs_norm.append(
            #    np.linalg.norm(diff_rhs.ravel()))
            # rhs_inf_G.append(
            #     np.linalg.norm(rhs_field_G.ravel(), ord=np.inf))
            # rhs_inf_GJ.append(
            #     np.linalg.norm(rhs_field_GJ.ravel(), ord=np.inf))

            info_log_final_GJ = np.load(data_folder_path + f'info_log_final.npz', allow_pickle=True)

    # del strain_fluc_G, strain_total_G, stress_G, rhs_field_G
    # del strain_fluc_GJ, strain_total_GJ, stress_GJ, rhs_field_GJ
    # del diff_stress

    # data = np.load('large_3d_array.npy', mmap_mode='r')
    K = 2  # _info_final_G.f.model_parameters_linear
    its_G = np.array(its_G)
    its_GJ = np.array(its_GJ)
    strain_0_norm = np.array(np.atleast_1d(_info_final_GJ.f.norm_En))
    # stress_diff_norm = np.array(stress_diff_norm)
    # strain_fluc_norm = np.array(strain_fluc_norm)
    # strain_total_norm = np.array(strain_total_norm)
    # diff_rhs_norm = np.array(diff_rhs_norm)

    norm_rhs_t_G = np.concatenate((np.array(np.atleast_1d(_info_final_G.f.rhs_t_norm)), np.array(norm_rhs_G)))
    norm_rhs_t_GJ = np.concatenate((np.array(np.atleast_1d(_info_final_GJ.f.rhs_t_norm)), np.array(norm_rhs_GJ)))

    fig = plt.figure(figsize=(8.3, 5.0))
    gs = fig.add_gridspec(2, 5, hspace=0.1, wspace=0.1, width_ratios=[0.05, 1, 1, 1, 1],
                          height_ratios=[1, 1.5])

    gs10 = mpl.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, :], wspace=0.04)

    gs_global = fig.add_subplot(gs10[0, 0])

    gs_global.plot(iterations, its_G[:, 0], 'g-', marker='x', label='Green')
    gs_global.plot(iterations, its_GJ[:, 0], 'k-', marker='o', markerfacecolor='none', label='Green-Jacobi')

    # gs_global.plot(iterations, its_G[:, -2], 'g-', marker='x', label='Green')
    # gs_global.plot(iterations, its_GJ[:, -2], 'k-', marker='o', markerfacecolor='none', label='Green-Jacobi')
    #
    # gs_global.plot(iterations, its_G[:, -1], 'g-', marker='x', label='Green')
    # gs_global.plot(iterations, its_GJ[:, -1], 'k-', marker='o', markerfacecolor='none', label='Green-Jacobi')

    gs_global.set_xlabel(r'Newton iteration -  $i$')
    gs_global.set_ylabel(r'$\#$ of PCG iterations')
    # gs_global.legend(loc='best')
    gs_global.set_xlim(-0.05, iterations[-1] + .05)
    gs_global.set_ylim(0., 300)
    gs_global.set_xticks(iterations)

    gs_global.annotate(text=f'Green-Jacobi',  # \n contrast = 100
                       xy=(iterations[2], its_GJ[2]),
                       xytext=(0.5, 100.),
                       arrowprops=dict(arrowstyle='->',
                                       color='Black',
                                       lw=1,
                                       ls='-'),
                       fontsize=11,
                       color='Black',
                       )
    gs_global.annotate(text=f'Green',  # \n contrast = 100
                       xy=(iterations[2], its_G[2]),
                       xytext=(3, 200.),
                       arrowprops=dict(arrowstyle='->',
                                       color='k',
                                       lw=1,
                                       ls='-'),
                       fontsize=11,
                       color='k',
                       )
    gs_global.text(0.02, 0.93, rf'\textbf{{(b.1)}}', transform=gs_global.transAxes)

    # gs_global.set_ylim(0, 800)

    # Right y-axis
    ax2 = fig.add_subplot(gs10[0, 1])

    ax2.semilogy(np.arange(len(norm_newrton_stop_GJ)), norm_newrton_stop_GJ, 'g:', marker='o', markerfacecolor='none',
                 label=r'$norm_newrton_stop_GJ$')
    ax2.semilogy(np.arange(len(norm_newrton_stop_G)), norm_newrton_stop_G, 'g:', marker='o', markerfacecolor='none',
                 label=r'$norm_newrton_stop_GJ$')

    # ax2.semilogy(np.arange(len(stress_diff_norm[1:])) + 1, stress_diff_norm[1:], 'b->',
    #              label=r'$||(\mathsf{\bf{\sigma}}_{\rm{G}}^{(i)}- \mathsf{\bf{\sigma}}_{\rm{GJ}}^{(i)})||_{\infty}$')
    # ax2.annotate(text=r'Stress - $(\mathsf{\bf{\sigma}}_{\rm{G}}^{(i)}- \mathsf{\bf{\sigma}}_{\rm{GJ}}^{(i)})$',
    #              # \n contrast = 100
    #              xy=(2, stress_diff_norm[2]),
    #              xytext=(2, 1e-2),
    #              arrowprops=dict(arrowstyle='->',
    #                              color='k',
    #                              lw=1,
    #                              ls='-'),
    #              fontsize=11,
    #              color='k',
    #              )

    # ax2.semilogy(np.arange(len(strain_fluc_norm[1:])) + 1, strain_fluc_norm[1:], 'r-|',
    #              label=r'$||  \nabla \mathsf{\bf{\tilde{u}}}_{\rm{G}} ^{(i)}- \nabla \mathsf{\bf{\tilde{u}}}_{\rm{GJ}} ^{(i)}||_{\infty}$')
    # ax2.annotate(
    #     text=r'Strain - $(\nabla \mathsf{\bf{\tilde{u}}}_{\rm{G}} ^{(i)}- \nabla \mathsf{\bf{\tilde{u}}}_{\rm{GJ}} ^{(i)})$',
    #     # \n contrast = 100
    #     xy=(4, strain_fluc_norm[1]),
    #     xytext=(3.7, 5e-4),
    #     arrowprops=dict(arrowstyle='->',
    #                     color='k',
    #                     lw=1,
    #                     ls='-'),
    #     fontsize=11,
    #     color='k',
    # )
    #
    # ax2.semilogy(np.arange(len(diff_rhs_norm[1:])) + 1, diff_rhs_norm[1:], '-x', color='brown',
    #              label=r'$|| \mathsf{\bf{f}}_{\rm{G}}^{(i)} - \mathsf{\bf{f}}_{\rm{GJ}}^{(i)} ||_{\infty}$')
    # ax2.annotate(text=r'Force - $(\mathsf{\bf{f}}_{\rm{G}}^{(i)} - \mathsf{\bf{f}}_{\rm{GJ}}^{(i)})$',
    #              # \n contrast = 100
    #              xy=(4, diff_rhs_norm[1]),
    #              xytext=(0.2, 3e-8),
    #              arrowprops=dict(arrowstyle='->',
    #                              color='k',
    #                              lw=1,
    #                              ls='-'),
    #              fontsize=11,
    #              color='k',
    #              )
    #
    # ax2.semilogy(np.arange(len(norm_rhs_GJ)), norm_rhs_GJ, 'g:', marker='o', markerfacecolor='none',
    #              label=r'$|| \mathsf{\bf{f}}_{\rm{GJ}}^{(i)}||_{\infty}$')
    # ax2.annotate(text=r'Force - $\mathsf{\bf{f}}_{\rm{GJ}}^{(i)}$',  # \n contrast = 100
    #              xy=(4, rhs_inf_GJ[1]),
    #              xytext=(0.2, 5e-7),
    #              arrowprops=dict(arrowstyle='->',
    #                              color='k',
    #                              lw=1,
    #                              ls='-'),
    #              fontsize=11,
    #              color='k',
    #              )

    # ax2.semilogy(np.arange(len(norm_rhs_G)), norm_rhs_G, 'k:x',
    #              label=r'$|| \mathsf{\bf{f}}_{\rm{G}}^{(i)}||_{\infty}$')
    # ax2.annotate(text=r'Force - $\mathsf{\bf{f}}_{\rm{G}}^{(i)}$',  # \n contrast = 100
    #              xy=(5, norm_rhs_G[1]),
    #              xytext=(6, 3e-7),
    #              arrowprops=dict(arrowstyle='->',
    #                              color='k',
    #                              lw=1,
    #                              ls='-'),
    #              fontsize=11,
    #              color='k',
    #              )

    # ax2.semilogy(np.arange(len(norm_rhs_G)), abs(norm_rhs_G-norm_rhs_GJ), 'g:x',
    #              label=r'Green - $|| \mathsf{\bf{f}}_{\rm{G}}^{(i)}-\mathsf{\bf{f}}_{\rm{GJ}}^{(i)}||$')
    # ax2.semilogy(np.arange(len(norm_rhs_G)), abs(norm_rhs_G - norm_rhs_GJ), 'g:x', label='error ')

    # ax2.set_ylabel('Norm of error', color='r')

    ax2.set_ylim([1e-10, 1e-1])
    ax2.set_yticks([1e-10, 1e-7, 1e-4, 1e-1])
    ax2.set_yticklabels([r'$10^{-10}$', r'$10^{-7}$', r'$10^{-4}$', r'$10^{-1}$'])
    ax2.yaxis.set_ticks_position('right')  # move ticks to right
    ax2.yaxis.set_label_position('right')  # move label to right
    ax2.set_xlabel(r'Newton iteration - $i$')
    ax2.set_ylabel(r'$||{X} ||_{\infty}$')
    # ax2.legend(loc='best')
    ax2.set_xlim(-0.05, iterations[-1] + .05)
    ax2.set_xticks(iterations)
    ax2.text(0.02, 0.92, rf'\textbf{{(b.2)}}', transform=ax2.transAxes)

    # plot mat data 0 Newton iteration
    ijkl = (0, 0, 0, 0)
    # results_name = (f'init_K')
    # K4_init = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True, mmap_mode='r')
    # del K4_init
    # # plot mat data  in  Newton iterations

    preconditioner_type = 'Green'
    data_folder_path = (file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                        + f'_{preconditioner_type}' + '/')

    # first iteration
    iteration_total = 1
    i = 0
    results_name = (f'K4_ijklqyz' + f'_it{iteration_total}')

    K4_xyz_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True, mmap_mode='r')

    # ax_geom_0 = fig.add_axes([0.3, 0.75, 0.2, 0.2])
    ax_geom_0 = fig.add_subplot(gs[0, 2])

    # max_K = K4_ijklqyz_G[ijkl + (..., 0)].max() / K
    # min_K = K4_ijklqyz_G[ijkl + (..., 0)].min() / K
    cut_to_plot = Nz // 2 - 1
    K4_to_plot_G = K4_xyz_G[..., cut_to_plot]  # K4_xyz_G[i,0,0,0, ..., cut_to_plot]
    max_K = K4_to_plot_G.max() / K
    min_K = K4_to_plot_G.min() / K

    mid_K = K4_to_plot_G.mean() / K  # K4_init[ (Nx // 2, Ny // 2,  Nz // 2)] ijkl +
    norm = mpl.colors.TwoSlopeNorm(vmin=min_K, vcenter=mid_K, vmax=max_K)
    cmap_ = mpl.cm.cividis  # mpl.cm.seismic

    pcm = ax_geom_0.pcolormesh(np.tile(K4_to_plot_G / K, (1, 1)),
                               cmap=cmap_, norm=norm,
                               linewidth=0,
                               rasterized=True)
    ax_geom_0.set_aspect('equal')
    ax_geom_0.set_title(fr'$i={iteration_total}$')
    #ax_geom_0.text(0.32, 0.5, r'$N_{z}$=128', transform=ax_geom_0.transAxes)
    ax_geom_0.text(-0., 1.1, rf'\textbf{{(a.2)}}', transform=ax_geom_0.transAxes)
    ax_geom_0.set_aspect('equal')

    ax_geom_0.set_xticks([])
    ax_geom_0.set_xticklabels([])
    ax_geom_0.set_yticks([])
    ax_geom_0.set_yticklabels([])
    # ax_geom_0.set_xlim([0, Nz])
    # ax_geom_0.set_ylim([0, Nz])
    ax_geom_0.set_box_aspect(1)

    # ----------------
    # colobar is based on the first iteration
    # ax_cbar = fig.add_axes([0.8, 0.22, 0.02, 0.2])
    ax_cbar = fig.add_subplot(gs[0, 0])

    cbar = plt.colorbar(pcm, location='left', cax=ax_cbar)
    cbar.set_ticks(ticks=[min_K, mid_K, max_K])
    cbar.set_ticklabels([f'{min_K:.0f}', f'{mid_K:.0f}', f'{max_K:.0f}'])
    ax_cbar.tick_params(right=True, top=False, labelright=False, labeltop=False, labelrotation=0)
    cbar.ax.yaxis.set_ticks_position('left')  # move ticks to right
    cbar.ax.yaxis.set_label_position('left')  # move label to right
    ax_cbar.set_ylabel(r'$\mathrm{C}_{11}/\mathrm{K}$')
    # ----------------
    iteration_total = 0
    results_name = (f'K4_ijklqyz' + f'_it{iteration_total}')
    del K4_xyz_G
    K4_xyz_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True, mmap_mode='r')
    K4_to_plot_G = K4_xyz_G[..., cut_to_plot]  # K4_xyz_G[i,0,0,0, ..., cut_to_plot]
    ax_geom_0 = fig.add_subplot(gs[0, 1])
    # ax_geom_0 = fig.add_axes([0.1, 0.75, 0.2, 0.2])
    pcm = ax_geom_0.pcolormesh(np.tile(K4_to_plot_G / K, (1, 1)),
                               cmap=cmap_, norm=norm,
                               linewidth=0,
                               rasterized=True)
    ax_geom_0.set_aspect('equal')
    ax_geom_0.set_title(fr'$i={iteration_total}$')
    ax_geom_0.text(0.32, 0.5, r'$N_{z}$=128', transform=ax_geom_0.transAxes)
    ax_geom_0.text(-0., 1.1, rf'\textbf{{(a.1)}}', transform=ax_geom_0.transAxes)
    ax_geom_0.set_aspect('equal')

    ax_geom_0.set_xticks([])
    ax_geom_0.set_xticklabels([])
    ax_geom_0.set_yticks([])
    ax_geom_0.set_yticklabels([])
    # ax_geom_0.set_xlim([0, Nz])
    # ax_geom_0.set_ylim([0, Nz])
    ax_geom_0.set_box_aspect(1)
    # ----------------

    # for iteration_total in 6:
    iteration_total = 2
    # iteration_total = 2
    del K4_xyz_G
    results_name = (f'K4_ijklqyz' + f'_it{iteration_total}')
    K4_xyz_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True, mmap_mode='r')
    K4_to_plot_G = K4_xyz_G[..., cut_to_plot]  # K4_xyz_G[i,0,0,0, ..., cut_to_plot]

    ax_geom_0 = fig.add_subplot(gs[0, 3])
    # ax_geom_0 = fig.add_axes([0.5, 0.75, 0.2, 0.2])
    pcm = ax_geom_0.pcolormesh(np.tile(K4_to_plot_G / K, (1, 1)),
                               cmap=cmap_, norm=norm,
                               linewidth=0,
                               rasterized=True)
    ax_geom_0.set_aspect('equal')
    ax_geom_0.set_title(fr'$i={iteration_total}$')
    #ax_geom_0.text(0.30, 0.5, r'$N_{z}=128$', transform=ax_geom_0.transAxes)

    ax_geom_0.text(-0., 1.1, rf'\textbf{{(a.{3})}}', transform=ax_geom_0.transAxes)
    ax_geom_0.set_aspect('equal')

    ax_geom_0.set_xticks([])
    ax_geom_0.set_xticklabels([])
    ax_geom_0.set_yticks([])
    ax_geom_0.set_yticklabels([])
    # ax_geom_0.set_xlim([0, Nz])
    # ax_geom_0.set_ylim([0, Nz])
    ax_geom_0.set_box_aspect(1)
    # ----------------
    # for iteration_total in 6:
    iteration_total = 3
    # iteration_total = 2
    results_name = (f'K4_ijklqyz' + f'_it{iteration_total}')
    del K4_xyz_G
    K4_xyz_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True, mmap_mode='r')
    K4_to_plot_G = K4_xyz_G[..., cut_to_plot]  # K4_xyz_G[i,0,0,0, ..., cut_to_plot]
    # ax_geom_0 = fig.add_axes([0.7, 0.75, 0.2, 0.2])
    ax_geom_0 = fig.add_subplot(gs[0, 4])

    pcm = ax_geom_0.pcolormesh(np.tile(K4_to_plot_G / K, (1, 1)),
                               cmap=cmap_, norm=norm,
                               linewidth=0,
                               rasterized=True)
    ax_geom_0.set_aspect('equal')
    ax_geom_0.set_title(fr'$i={iteration_total}$')
    ax_geom_0.text(0.32, -0.5, r'$3 \cdot 256^{3}$ DOFs', transform=ax_geom_0.transAxes)
    ax_geom_0.text(0.2, -0.3, r'$ \approx  50 \times 10^{6}DOFs $', transform=ax_geom_0.transAxes)

    ax_geom_0.text(-0., 1.1, rf'\textbf{{(a.{4})}}', transform=ax_geom_0.transAxes)
    ax_geom_0.set_aspect('equal')
    ax_geom_0.set_xticks([])
    ax_geom_0.set_ylabel('Pixel index')
    # ax_geom_0.set_xticks([0, Nx//2, Nx])

    ax_geom_0.set_yticks([0, Nz // 2, Nz])
    #    ax_geom_0.set_xlim([0, Nz])
    #  ax_geom_0.set_ylim([0, Nz])
    ax_geom_0.set_box_aspect(1)  # Maintain square aspect ratio
    ax_geom_0.yaxis.set_ticks_position('right')
    ax_geom_0.yaxis.set_label_position('right')

    # axis for cross sections
    add_stress_plot = False
    if add_stress_plot:
        #### STRESSES
        ax_cross = fig.add_axes([0.65, 0.2, 0.2, 0.2])
        iteration_total = 6
        preconditioner_type = 'Green'

        data_folder_path = (file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                            + f'_{preconditioner_type}' + '/')

        stress_G = np.load(data_folder_path + f'stress' + f'_it{iteration_total}' + f'.npy', allow_pickle=True)
        strain_fluc_G = np.load(data_folder_path + f'strain_fluc_field' + f'_it{iteration_total}' + f'.npy',
                                allow_pickle=True)

        preconditioner_type = 'Jacobi_Green'

        data_folder_path = (file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                            + f'_{preconditioner_type}' + '/')
        stress_GJ = np.load(data_folder_path + f'stress' + f'_it{iteration_total}' + f'.npy', allow_pickle=True)
        strain_fluc_GJ = np.load(data_folder_path + f'strain_fluc_field' + f'_it{iteration_total}' + f'.npy',
                                 allow_pickle=True)

        ### compute the difference
        ij = (0, 0)

        stress_diff = abs((stress_GJ[ij + (..., 0)] - stress_G[ij + (..., 0)]))  # / stress_G[ij + (..., 0)])

        max_K = stress_diff.max()
        min_K = 0  # stress_diff.min()

        pcm = ax_cross.pcolormesh(np.tile(stress_diff, (1, 1)),
                                  cmap=cmap_,
                                  # norm=norm,
                                  linewidth=0,
                                  rasterized=True)
        ax_cross.set_aspect('equal')

        ax_cbar2 = fig.add_axes([0.9, 0.2, 0.01, 0.3])
        cbar = plt.colorbar(pcm, location='left', cax=ax_cbar2)
        cbar.set_ticks(ticks=[0, max_K])  # 0,min_K,
        # cbar.set_ticklabels([f'{min_K:.0f}', f'{0:.0f}', f'{max_K:.0f}'])
        cbar.set_ticklabels(
            ['0',
             f'$10^{{{int(np.log10(max_K))}}}$'])  # f'$10^{{{-int(np.log10(abs(min_K)))}}}$', f'$10^{{{int(np.log10(1))}}}$',
        ax_cbar2.tick_params(right=True, top=False, labelright=False, labeltop=False, labelrotation=0)
        cbar.ax.yaxis.set_ticks_position('right')  # move ticks to right
        cbar.ax.yaxis.set_label_position('right')  # move label to right
        ax_cbar2.set_ylabel(fr'$(\sigma^{{G}}_{{11}}-\sigma^{{GJ}}_{{11}})  $')

    fig.tight_layout()
    fname = f'fig_2' + '{}'.format('.pdf')
    plt.savefig(figure_folder_path + script_name + fname, bbox_inches='tight')
    print(('create figure: {}'.format(figure_folder_path + script_name + fname)))

    plt.show()
##############################################################################################
if plot_stress_field:

    def compute_eq_strain(strain):
        strain_trace_xyz = np.einsum('ii...', strain) / 3  # todo{2 or 3 in 2D }

        # volumetric strain
        strain_vol_ijxyz = np.ndarray(shape=strain.shape)
        strain_vol_ijxyz.fill(0)
        for d in np.arange(3):
            strain_vol_ijxyz[d, d, ...] = strain_trace_xyz

        # deviatoric strain
        strain_dev_ijxyz = strain - strain_vol_ijxyz

        # equivalent strain
        strain_dev_ddot = np.einsum('ijxyz,jixyz-> xyz', strain_dev_ijxyz, strain_dev_ijxyz)
        strain_eq_xyz = np.sqrt((2. / 3.) * strain_dev_ddot)

        return strain_eq_xyz


    # print time vs DOFS
    iteration_total = 6
    Nx = 3
    Ny = 256
    Nz = 256
    i = 0
    j = 1

    preconditioner_type = 'Green'
    data_folder_path = (file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                        + f'_{preconditioner_type}' + '/')

    _info_final_G = np.load(data_folder_path + f'info_log_final.npz', allow_pickle=True)

    results_name = (f'stress_{i, j}' + f'_it{iteration_total}')
    stress_field_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True)
    results_name = (f'total_strain_field_{i, j}' + f'_it{iteration_total}')
    total_strain_field_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True)
    results_name = (f'K4_ijklqyz_{i, 0}' + f'_it{iteration_total}')
    K4_ijklqyz_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True)

    results_name = (f'init_K_{0, 0}')
    K4_init = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True)

    preconditioner_type = 'Jacobi_Green'

    data_folder_path = (file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                        + f'_{preconditioner_type}' + '/')

    _info_final_GJ = np.load(data_folder_path + f'info_log_final.npz', allow_pickle=True)

    results_name = (f'stress_{i, j}' + f'_it{iteration_total}')
    stress_field_GJ = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True)

    results_name = (f'total_strain_field_{i, j}' + f'_it{iteration_total}')
    total_strain_field_GJ = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True)

    results_name = (f'K4_ijklqyz_{i, 0}' + f'_it{iteration_total}')
    K4_ijklqyz_GJ = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True)

    fig = plt.figure(figsize=(9, 9.0))
    gs = fig.add_gridspec(3, 3, hspace=0.2, wspace=0.2, width_ratios=[1, 1, 0.1],
                          height_ratios=[1, 1, 1])

    gs_stress_G = fig.add_subplot(gs[0, 0])
    gs_stress_GJ = fig.add_subplot(gs[0, 1])
    ax_cbar_stress = fig.add_subplot(gs[0, 2])

    # compute max min stress
    ij = (0, 1)
    max_stress = stress_field_G[..., 0].max()  # ij + (..., 0)
    min_stress = stress_field_G[..., 0].min()
    divnorm = mpl.colors.Normalize(vmin=min_stress, vmax=max_stress)
    cmap_ = mpl.cm.cividis

    pcm = gs_stress_G.pcolormesh(np.tile(stress_field_G[..., 0], (1, 1)),
                                 cmap=cmap_, linewidth=0,
                                 rasterized=True,
                                 # norm=divnorm
                                 )
    pcm = gs_stress_GJ.pcolormesh(np.tile(stress_field_GJ[..., 0], (1, 1)),
                                  cmap=cmap_, linewidth=0,
                                  rasterized=True,
                                  # norm=divnorm
                                  )
    cbar = plt.colorbar(pcm, location='left', cax=ax_cbar_stress)

    ##### plot strains
    gs_tot_strain_G = fig.add_subplot(gs[1, 0])
    gs_tot_strain_GJ = fig.add_subplot(gs[1, 1])
    ax_cbar_tot_strain = fig.add_subplot(gs[1, 2])

    eq_strain_G = total_strain_field_G  # compute_eq_strain(total_strain_field_G)
    eq_strain_GJ = total_strain_field_GJ  # compute_eq_strain(total_strain_field_GJ)

    #
    max_strain = eq_strain_G[..., 0].max()
    min_strain = eq_strain_G[..., 0].min()
    divnorm = mpl.colors.Normalize(vmin=min_strain, vmax=max_strain)
    cmap_ = mpl.cm.cividis

    pcm = gs_tot_strain_G.pcolormesh(np.tile(eq_strain_G[..., 0], (1, 1)),
                                     cmap=cmap_, linewidth=0,
                                     rasterized=True,
                                     # norm=divnorm
                                     )
    pcm = gs_tot_strain_GJ.pcolormesh(np.tile(eq_strain_GJ[..., 0], (1, 1)),
                                      cmap=cmap_, linewidth=0,
                                      rasterized=True,
                                      # norm=divnorm
                                      )
    cbar = plt.colorbar(pcm, location='left', cax=ax_cbar_tot_strain)

    ##### plot material data
    gs_K_G = fig.add_subplot(gs[2, 0])
    gs_K_GJ = fig.add_subplot(gs[2, 1])
    ax_cbar_K = fig.add_subplot(gs[2, 2])

    # compute max min stress
    ijkl = (0, 0, 0, 0)
    max_K = K4_ijklqyz_G[..., 0].max()
    min_K = K4_ijklqyz_G[..., 0].min()
    divnorm = mpl.colors.Normalize(vmin=min_K, vmax=max_K)
    cmap_ = mpl.cm.cividis

    pcm = gs_K_G.pcolormesh(np.tile(K4_ijklqyz_G[..., 0], (1, 1)),
                            cmap=cmap_, linewidth=0,
                            rasterized=True,
                            # norm=divnorm
                            )
    pcm = gs_K_GJ.pcolormesh(np.tile(K4_ijklqyz_GJ[..., 0], (1, 1)),
                             cmap=cmap_, linewidth=0,
                             rasterized=True,
                             # norm=divnorm
                             )
    cbar = plt.colorbar(pcm, location='left', cax=ax_cbar_K)

    plt.show()

quit()

print(np.array(time_G))
print(np.array(time_GJ))

# phase_field = np.load('../exp_data/' + file_data_name + f'.npy', allow_pickle=True)

number_of_pixels = (32, 32, 1)  # (128, 128, 1)  # (32, 32, 1) # (64, 64, 1)  # (128, 128, 1) #
domain_size = [1, 1, 1]
Nx = number_of_pixels[0]
Ny = number_of_pixels[1]
Nz = number_of_pixels[2]

_info_final = np.load(data_folder_path + f'info_log_final.npz', allow_pickle=True)

iteration_total = 0
_info_at_it = np.load(data_folder_path + f'info_log_it{iteration_total}.npz', allow_pickle=True)

print(_info_at_it)
# file_data_name = (
#         f'{script_name_save}_gID{geometry_ID}_T{number_of_pixels[0]}_F{filer_id}_kappa{contrast}.npy')
