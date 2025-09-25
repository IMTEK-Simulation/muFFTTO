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

script_name = 'exp_paper_JG_nonlinear_elasticity_JZ'
folder_name = '../exp_data/'
file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory

plot_time_vs_dofs = False
plot_stress_field = False
plot_data_vs_CG = True

if plot_time_vs_dofs:
    # print time vs DOFS
    time_G = []
    time_GJ = []
    its_G = []
    its_GJ = []
    Ns = 2 ** np.array([3, 4, 5, 6, 7, 8, 9])  # numbers of grids points
    for N in Ns:
        Nx = Ny = N
        Nz = 1  # N#
        preconditioner_type = 'Green'

        data_folder_path = (file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                            + f'_{preconditioner_type}' + '/')

        _info_final_G = np.load(data_folder_path + f'info_log_final.npz', allow_pickle=True)

        time_G.append(_info_final_G.f.elapsed_time)
        its_G.append(_info_final_G.f.sum_CG_its)

        preconditioner_type = 'Jacobi_Green'

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
    plt.loglog(nb_dofs, time_GJ, 'k-', label='GJ')
    plt.loglog(nb_dofs, time_G, '-', color='Green', label='G')
    plt.loglog(nb_dofs, nb_dofs * np.log(nb_dofs) / (nb_dofs[0] * np.log(nb_dofs[0])) * time_G[0], ':',
               label='N log N')
    # plt.loglog(nb_dofs, nb_dofs / (nb_dofs[0]) * time_G[0], '--', label='linear')

    plt.loglog(nb_dofs, time_GJ / its_GJ, 'k.-', label='time_GJ/its_GJ')
    plt.loglog(nb_dofs, time_G / its_G, 'g.-', label='time_G/its_G')
    # plt.loglog(nb_dofs,
    #            nb_dofs* np.log(nb_dofs) / (nb_dofs[0]  * np.log(nb_dofs)) * time_G[0] / its_G[0], ':',
    #            label='N log N')
    plt.loglog(nb_dofs, nb_dofs / (nb_dofs[0]) * time_G[0] / its_G[0], '--', label='linear')
    plt.xlabel('N Dofs')
    plt.ylabel('Time (s)')
    # plt.yscale('linear')
    plt.legend(loc='best')
    plt.show()

    fig = plt.figure(figsize=(9, 3.0))
    gs = fig.add_gridspec(1, 1, hspace=0.5, wspace=0.5, width_ratios=[1],
                          height_ratios=[1])

    plt.loglog(3 * Ns ** 2, time_GJ / its_GJ, 'k-', label='GJ')
    plt.loglog(3 * Ns ** 2, time_G / its_G, 'g-', label='G')
    plt.loglog(3 * Ns ** 2, 3 * Ns ** 2 / 2e5, '--', label='linear')
    plt.loglog(3 * Ns ** 2, 3 * Ns ** 2 * np.log(3 * Ns ** 2) / 1e6, ':', label='N log N')
    plt.xlabel('N Dofs')
    plt.ylabel('Time (s)/ nb CG iterations')
    # plt.yscale('linear')

    plt.legend(loc='best')
    plt.show()

if plot_data_vs_CG:
    # print time vs DOFS
    its_G = []
    its_GJ = []
    stress_diff_norm = []
    strain_fluc_norm = []
    strain_total_norm = []

    norm_rhs_G = []
    norm_rhs_GJ = []
    Nx = Ny =  128
    Nz = 1  # Nx
    iterations = np.arange(9)  # numbers of grids points
    for iteration_total in iterations:
        preconditioner_type = 'Green'

        data_folder_path = (file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                            + f'_{preconditioner_type}' + '/')
        if iteration_total < 8:
            _info_final_G = np.load(data_folder_path + f'info_log_it{iteration_total}.npz', allow_pickle=True)

        stress_G = np.load(data_folder_path + f'stress' + f'_it{iteration_total}' + f'.npy', allow_pickle=True)
        strain_fluc_G = np.load(data_folder_path + f'strain_fluc_field' + f'_it{iteration_total}' + f'.npy',
                                allow_pickle=True)
        strain_total_G = np.load(data_folder_path + f'total_strain_field' + f'_it{iteration_total}' + f'.npy',
                                 allow_pickle=True)
        its_G.append(_info_final_G.f.nb_it_comb)
        norm_rhs_G.append(_info_final_G.f.norm_rhs_field)

        preconditioner_type = 'Jacobi_Green'

        data_folder_path = (file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                            + f'_{preconditioner_type}' + '/')
        if iteration_total < 8:

            _info_final_GJ = np.load(data_folder_path + f'info_log_it{iteration_total}.npz', allow_pickle=True)
        stress_GJ = np.load(data_folder_path + f'stress' + f'_it{iteration_total}' + f'.npy', allow_pickle=True)
        strain_fluc_GJ = np.load(data_folder_path + f'strain_fluc_field' + f'_it{iteration_total}' + f'.npy',
                                 allow_pickle=True)
        strain_total_GJ = np.load(data_folder_path + f'total_strain_field' + f'_it{iteration_total}' + f'.npy',
                                  allow_pickle=True)
        norm_rhs_GJ.append(_info_final_GJ.f.norm_rhs_field)

        stress_diff_norm.append(
            np.linalg.norm(stress_G - stress_GJ) )# / np.linalg.norm(stress_G))
        print(_info_final_G.f.norm_rr[0])
        strain_fluc_norm.append(
            np.linalg.norm(strain_fluc_G - strain_fluc_GJ) )# / _info_final_GJ.f.norm_En)
        print(_info_final_GJ.f.norm_rr[0])
        strain_total_norm.append(
            np.linalg.norm(strain_total_G - strain_total_GJ))#/ _info_final_GJ.f.norm_En)

        its_GJ.append(_info_final_GJ.f.nb_it_comb)

    K = 2  # _info_final_G.f.model_parameters_linear
    its_G = np.array(its_G)
    its_GJ = np.array(its_GJ)
    strain_0_norm = np.array(np.atleast_1d(_info_final_GJ.f.norm_En))
    stress_diff_norm = np.array(stress_diff_norm)
    strain_fluc_norm = np.array(strain_fluc_norm)
    strain_total_norm = np.array(strain_total_norm)

    norm_rhs_G = np.concatenate((np.array(np.atleast_1d(_info_final_G.f.rhs_t_norm)), np.array(norm_rhs_G)))
    norm_rhs_GJ = np.concatenate((np.array(np.atleast_1d(_info_final_GJ.f.rhs_t_norm)), np.array(norm_rhs_GJ)))

    fig = plt.figure(figsize=(8.3, 5.0))
    gs = fig.add_gridspec(2, 5, hspace=0.1, wspace=0.1, width_ratios=[0.05, 1, 1, 1, 1],
                          height_ratios=[1, 1.5])

    gs10 = mpl.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, :], wspace=0.1)

    gs_global = fig.add_subplot(gs10[0, 0])

    gs_global.plot(iterations, its_G, 'g-', marker='x', label='Green')
    gs_global.plot(iterations, its_GJ, 'k-', marker='o', markerfacecolor='none', label='Green-Jacobi')
    gs_global.set_xlabel(r'Newton iteration -  $i$')
    gs_global.set_ylabel(r'$\#$ of PCG iterations')
    gs_global.legend(loc='best')
    gs_global.set_xlim(-0.05, iterations[-1] + .05)
    gs_global.set_xticks(iterations)

    # gs_global.set_ylim(0, 800)

    # Right y-axis
    ax2 = fig.add_subplot(gs10[0, 1])
    ax2.semilogy(np.arange(len(stress_diff_norm[1:])) + 1, stress_diff_norm[1:], 'b--',
                 label=r'$|| \mathsf{\bf{\sigma}}_{\rm{G}}^{(i)}- \mathsf{\bf{\sigma}}_{\rm{GJ}}^{(i)}||$')

    ax2.semilogy(np.arange(len(strain_fluc_norm[1:])) + 1, strain_fluc_norm[1:], 'r--',
                 label=r'$|| \nabla \mathsf{\bf{\tilde{u}}}_{\rm{G}} ^{(i)}- \nabla \mathsf{\bf{\tilde{u}}}_{\rm{GJ}} ^{(i)}||$')
    ax2.semilogy(np.arange(len(strain_total_norm[1:])) + 1, strain_total_norm[1:], 'y.-',
                 label=r'$|| \nabla \mathsf{\bf{{u}}}_{\rm{G}} ^{(i)}- \nabla \mathsf{\bf{{u}}}_{\rm{GJ}} ^{(i)}||$')


    ax2.semilogy(np.arange(len(norm_rhs_GJ)), norm_rhs_GJ, 'k:', marker='o', markerfacecolor='none',
                 label=r'$|| \mathsf{\bf{f}}_{\rm{GJ}}^{(i)}||$')
    ax2.semilogy(np.arange(len(norm_rhs_G)), norm_rhs_G, 'g:x',
                 label=r'$|| \mathsf{\bf{f}}_{\rm{G}}^{(i)}||$')
    # ax2.semilogy(np.arange(len(norm_rhs_G)), abs(norm_rhs_G-norm_rhs_GJ), 'g:x',
    #              label=r'Green - $|| \mathsf{\bf{f}}_{\rm{G}}^{(i)}-\mathsf{\bf{f}}_{\rm{GJ}}^{(i)}||$')
    # ax2.semilogy(np.arange(len(norm_rhs_G)), abs(norm_rhs_G - norm_rhs_GJ), 'g:x', label='error ')

    # ax2.set_ylabel('Norm of error', color='r')

    ax2.set_ylim([1e-7, 1e-1])
    ax2.set_yticks([1e-7, 1e-4, 1e-1])
    ax2.set_yticklabels([r'$10^{-7}$', r'$10^{-4}$', r'$10^{-1}$'])
    ax2.yaxis.set_ticks_position('right')  # move ticks to right
    ax2.yaxis.set_label_position('right')  # move label to right
    ax2.set_xlabel(r'Newton iteration -  $i$')
    ax2.legend(loc='best')
    ax2.set_xlim(-0.05, iterations[-1] + .05)
    ax2.set_xticks(iterations)

    # plot mat data 0 Newton iteration
    ijkl = (0, 0, 0, 0)
    results_name = (f'init_K')
    K4_init = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True)

    # plot mat data  in  Newton iterations

    preconditioner_type = 'Green'
    data_folder_path = (file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                        + f'_{preconditioner_type}' + '/')

    # first iteration
    iteration_total = 1
    results_name = (f'K4_ijklqyz' + f'_it{iteration_total}')
    K4_ijklqyz_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True)

    # ax_geom_0 = fig.add_axes([0.3, 0.75, 0.2, 0.2])
    ax_geom_0 = fig.add_subplot(gs[0, 2])

    max_K = K4_ijklqyz_G[ijkl + (..., 0)].max() / K
    min_K = K4_ijklqyz_G[ijkl + (..., 0)].min() / K
    mid_K = K4_init[ijkl + (Nx // 2, Nx // 2, 0)] / K
    norm = mpl.colors.TwoSlopeNorm(vmin=min_K, vcenter=mid_K, vmax=max_K)
    cmap_ = mpl.cm.cividis  # mpl.cm.seismic

    pcm = ax_geom_0.pcolormesh(np.tile(K4_ijklqyz_G[ijkl + (..., 0)] / K, (1, 1)),
                               cmap=cmap_, norm=norm,
                               linewidth=0,
                               rasterized=True)
    ax_geom_0.set_aspect('equal')
    ax_geom_0.set_title(fr'$i={iteration_total}$')
    ax_geom_0.text(-0.1, 1.1, rf'\textbf{{(a.2)}}', transform=ax_geom_0.transAxes)
    ax_geom_0.set_aspect('equal')

    ax_geom_0.set_xticks([])
    ax_geom_0.set_xticklabels([])
    ax_geom_0.set_yticks([])
    ax_geom_0.set_yticklabels([])
    ax_geom_0.set_xlim([0, Nx - 1])
    ax_geom_0.set_ylim([0, Nx - 1])
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
    ax_cbar.set_ylabel(r'${C}_{11}/K$')
    # ----------------
    iteration_total = 0
    results_name = (f'K4_ijklqyz' + f'_it{iteration_total}')
    K4_ijklqyz_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True)

    ax_geom_0 = fig.add_subplot(gs[0, 1])
    # ax_geom_0 = fig.add_axes([0.1, 0.75, 0.2, 0.2])
    pcm = ax_geom_0.pcolormesh(np.tile(K4_ijklqyz_G[ijkl + (..., 0)] / K, (1, 1)),
                               cmap=cmap_, norm=norm,
                               linewidth=0,
                               rasterized=True)
    ax_geom_0.set_aspect('equal')
    ax_geom_0.set_title(fr'$i={iteration_total}$')
    ax_geom_0.text(-0.1, 1.1, rf'\textbf{{(a.1)}}', transform=ax_geom_0.transAxes)
    ax_geom_0.set_aspect('equal')

    ax_geom_0.set_xticks([])
    ax_geom_0.set_xticklabels([])
    ax_geom_0.set_yticks([])
    ax_geom_0.set_yticklabels([])
    ax_geom_0.set_xlim([0, Nx - 1])
    ax_geom_0.set_ylim([0, Nx - 1])
    ax_geom_0.set_box_aspect(1)
    # ----------------

    # for iteration_total in 6:
    iteration_total = 2
    # iteration_total = 2
    results_name = (f'K4_ijklqyz' + f'_it{iteration_total}')
    K4_ijklqyz_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True)

    ax_geom_0 = fig.add_subplot(gs[0, 3])
    # ax_geom_0 = fig.add_axes([0.5, 0.75, 0.2, 0.2])
    pcm = ax_geom_0.pcolormesh(np.tile(K4_ijklqyz_G[ijkl + (..., 0)] / K, (1, 1)),
                               cmap=cmap_, norm=norm,
                               linewidth=0,
                               rasterized=True)
    ax_geom_0.set_aspect('equal')
    ax_geom_0.set_title(fr'$i={iteration_total}$')
    ax_geom_0.text(-0.1, 1.1, rf'\textbf{{(a.{3})}}', transform=ax_geom_0.transAxes)
    ax_geom_0.set_aspect('equal')

    ax_geom_0.set_xticks([])
    ax_geom_0.set_xticklabels([])
    ax_geom_0.set_yticks([])
    ax_geom_0.set_yticklabels([])
    ax_geom_0.set_xlim([0, Nx - 1])
    ax_geom_0.set_ylim([0, Nx - 1])
    ax_geom_0.set_box_aspect(1)
    # ----------------
    # for iteration_total in 6:
    iteration_total = 7
    # iteration_total = 2
    results_name = (f'K4_ijklqyz' + f'_it{iteration_total}')
    K4_ijklqyz_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True)

    # ax_geom_0 = fig.add_axes([0.7, 0.75, 0.2, 0.2])
    ax_geom_0 = fig.add_subplot(gs[0, 4])

    pcm = ax_geom_0.pcolormesh(np.tile(K4_ijklqyz_G[ijkl + (..., 0)] / K, (1, 1)),
                               cmap=cmap_, norm=norm,
                               linewidth=0,
                               rasterized=True)
    ax_geom_0.set_aspect('equal')
    ax_geom_0.set_title(fr'$i={iteration_total}$')
    ax_geom_0.text(-0.1, 1.1, rf'\textbf{{(a.{4})}}', transform=ax_geom_0.transAxes)
    ax_geom_0.set_aspect('equal')
    #ax_geom_0.set_xticks([])
    #ax_geom_0.set_xticklabels([])
    ax_geom_0.set_xticks([0, Nx//2, Nx])
    ax_geom_0.set_yticks([0, Nx//2, Nx])
    ax_geom_0.set_xlim([0, Nx - 1])
    ax_geom_0.set_ylim([0, Nx - 1])
    ax_geom_0.set_box_aspect(1)  # Maintain square aspect ratio


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
    Nx = Ny = 32
    Nz = 1

    preconditioner_type = 'Green'
    data_folder_path = (file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                        + f'_{preconditioner_type}' + '/')

    _info_final_G = np.load(data_folder_path + f'info_log_final.npz', allow_pickle=True)

    results_name = (f'stress' + f'_it{iteration_total}')
    stress_field_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True)
    results_name = (f'total_strain_field' + f'_it{iteration_total}')
    total_strain_field_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True)
    results_name = (f'K4_ijklqyz' + f'_it{iteration_total}')
    K4_ijklqyz_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True)

    K4_init = np.load(data_folder_path + 'init_K' + f'.npy', allow_pickle=True)

    preconditioner_type = 'Jacobi_Green'

    data_folder_path = (file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                        + f'_{preconditioner_type}' + '/')

    _info_final_GJ = np.load(data_folder_path + f'info_log_final.npz', allow_pickle=True)

    results_name = (f'stress' + f'_it{iteration_total}')
    stress_field_GJ = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True)

    results_name = (f'total_strain_field' + f'_it{iteration_total}')
    total_strain_field_GJ = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True)

    results_name = (f'K4_ijklqyz' + f'_it{iteration_total}')
    K4_ijklqyz_GJ = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True)

    fig = plt.figure(figsize=(9, 9.0))
    gs = fig.add_gridspec(3, 3, hspace=0.2, wspace=0.2, width_ratios=[1, 1, 0.1],
                          height_ratios=[1, 1, 1])

    gs_stress_G = fig.add_subplot(gs[0, 0])
    gs_stress_GJ = fig.add_subplot(gs[0, 1])
    ax_cbar_stress = fig.add_subplot(gs[0, 2])

    # compute max min stress
    ij = (0, 1)
    max_stress = stress_field_G[ij + (..., 0)].max()
    min_stress = stress_field_G[ij + (..., 0)].min()
    divnorm = mpl.colors.Normalize(vmin=min_stress, vmax=max_stress)
    cmap_ = mpl.cm.cividis

    pcm = gs_stress_G.pcolormesh(np.tile(stress_field_G[ij + (..., 0)], (1, 1)),
                                 cmap=cmap_, linewidth=0,
                                 rasterized=True,
                                 # norm=divnorm
                                 )
    pcm = gs_stress_GJ.pcolormesh(np.tile(stress_field_GJ[ij + (..., 0)], (1, 1)),
                                  cmap=cmap_, linewidth=0,
                                  rasterized=True,
                                  # norm=divnorm
                                  )
    cbar = plt.colorbar(pcm, location='left', cax=ax_cbar_stress)

    ##### plot strains
    gs_tot_strain_G = fig.add_subplot(gs[1, 0])
    gs_tot_strain_GJ = fig.add_subplot(gs[1, 1])
    ax_cbar_tot_strain = fig.add_subplot(gs[1, 2])

    eq_strain_G = compute_eq_strain(total_strain_field_G)
    eq_strain_GJ = compute_eq_strain(total_strain_field_GJ)

    #
    max_strain = eq_strain_G[(..., 0)].max()
    min_strain = eq_strain_G[(..., 0)].min()
    divnorm = mpl.colors.Normalize(vmin=min_strain, vmax=max_strain)
    cmap_ = mpl.cm.cividis

    pcm = gs_tot_strain_G.pcolormesh(np.tile(eq_strain_G[(..., 0)], (1, 1)),
                                     cmap=cmap_, linewidth=0,
                                     rasterized=True,
                                     # norm=divnorm
                                     )
    pcm = gs_tot_strain_GJ.pcolormesh(np.tile(eq_strain_GJ[(..., 0)], (1, 1)),
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
    max_K = K4_ijklqyz_G[ijkl + (..., 0)].max()
    min_K = K4_ijklqyz_G[ijkl + (..., 0)].min()
    divnorm = mpl.colors.Normalize(vmin=min_K, vmax=max_K)
    cmap_ = mpl.cm.cividis

    pcm = gs_K_G.pcolormesh(np.tile(K4_ijklqyz_G[ijkl + (..., 0)], (1, 1)),
                            cmap=cmap_, linewidth=0,
                            rasterized=True,
                            # norm=divnorm
                            )
    pcm = gs_K_GJ.pcolormesh(np.tile(K4_ijklqyz_GJ[ijkl + (..., 0)], (1, 1)),
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
