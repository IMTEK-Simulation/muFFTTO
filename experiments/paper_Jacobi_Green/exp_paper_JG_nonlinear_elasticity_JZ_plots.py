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

plot_time_vs_dofs = True
plot_stress_field = False
plot_data_vs_CG = True

if plot_time_vs_dofs:
    # print time vs DOFS
    time_G = []
    time_GJ = []
    its_G = []
    its_GJ = []
    Ns = 2 ** np.array([4, 5, 6, 7, 8, 9])  # numbers of grids points
    for N in Ns:
        Nx = Ny = N
        Nz = 1
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

    plt.loglog(3 * Ns ** 2, time_GJ / 60, 'k-', label='GJ')
    plt.loglog(3 * Ns ** 2, time_G / 60, '-',color='Green' , label='G')
    plt.loglog(3 * Ns ** 2, Ns ** 2 / 2e4, '--', label='linear')
    plt.loglog(3 * Ns ** 2, Ns ** 2 * np.log(Ns ** 2) / 1e5, ':', label='N log N')
    plt.xlabel('N Dofs')
    plt.ylabel('Time (s)')
    plt.ylabel('Time per iteration')

    plt.legend(loc='best')
    plt.show()

    fig = plt.figure(figsize=(9, 3.0))
    gs = fig.add_gridspec(1, 1, hspace=0.5, wspace=0.5, width_ratios=[1],
                          height_ratios=[1])

    plt.loglog(3 * Ns ** 2, time_GJ/its_GJ, 'k-', label='GJ')
    plt.loglog(3 * Ns ** 2, time_G/its_G, 'g-', label='G')
    plt.loglog(3 * Ns ** 2, Ns ** 2 / 1e3, '--', label='linear')
    plt.loglog(3 * Ns ** 2, Ns ** 2 * np.log(Ns ** 2) / 1e3, ':', label='N log N')
    plt.xlabel('N Dofs')
    plt.ylabel('Time (s)/ nb CG iterations')
    plt.legend(loc='best')
    plt.show()

if plot_data_vs_CG:
    # print time vs DOFS
    its_G = []
    its_GJ = []
    Nx = Ny = 256
    Nz = 1
    iterations = np.arange(7)  # numbers of grids points
    for iteration_total in iterations:
        preconditioner_type = 'Green'

        data_folder_path = (file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                            + f'_{preconditioner_type}' + '/')

        _info_final_G = np.load(data_folder_path + f'info_log_it{iteration_total}.npz', allow_pickle=True)

        its_G.append(_info_final_G.f.nb_it_comb)

        preconditioner_type = 'Jacobi_Green'

        data_folder_path = (file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                            + f'_{preconditioner_type}' + '/')

        _info_final_GJ = np.load(data_folder_path + f'info_log_it{iteration_total}.npz', allow_pickle=True)

        its_GJ.append(_info_final_GJ.f.nb_it_comb)

    its_G = np.array(its_G)
    its_GJ = np.array(its_GJ)

    fig = plt.figure(figsize=(8.3, 5.0))
    gs = fig.add_gridspec(2, 5, hspace=0.1, wspace=0.1, width_ratios=[1, 1, 1, 1, 0.05],
                          height_ratios=[1, 2])
    gs_global = fig.add_subplot(gs[1, :])
    gs_global.plot(iterations, its_G, 'b-', label='Green')
    gs_global.plot(iterations, its_GJ, 'r-', label='Green-Jacobi')
    gs_global.set_xlabel('Newton iterations i')
    gs_global.set_ylabel(r'$\#$ of PCG iterations')
    gs_global.legend(loc='best')
    gs_global.set_xlim(-0.05, 6.05)
    gs_global.set_ylim(0, 800)
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
    ax_geom_0 = fig.add_subplot(gs[0, 1])

    max_K = K4_ijklqyz_G[ijkl + (..., 0)].max()
    min_K = K4_ijklqyz_G[ijkl + (..., 0)].min()
    mid_K = K4_init[ijkl + (Nx // 2, Nx // 2, 0)]
    norm = mpl.colors.TwoSlopeNorm(vmin=min_K, vcenter=mid_K, vmax=max_K)
    cmap_ = mpl.cm.cividis  # mpl.cm.seismic

    pcm = ax_geom_0.pcolormesh(np.tile(K4_ijklqyz_G[ijkl + (..., 0)], (1, 1)),
                               cmap=cmap_, norm=norm,
                               linewidth=0,
                               rasterized=True)
    ax_geom_0.set_aspect('equal')
    ax_geom_0.set_title(f' i={iteration_total}')
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
    ax_cbar = fig.add_subplot(gs[0, 4])

    cbar = plt.colorbar(pcm, location='left', cax=ax_cbar)
    cbar.set_ticks(ticks=[min_K, mid_K, max_K])
    cbar.set_ticklabels([f'{min_K:.0f}', f'{mid_K:.0f}', f'{max_K:.0f}'])
    ax_cbar.tick_params(right=True, top=False, labelright=False, labeltop=False, labelrotation=0)
    cbar.ax.yaxis.set_ticks_position('right')  # move ticks to right
    cbar.ax.yaxis.set_label_position('right')  # move label to right
    ax_cbar.set_ylabel(r'${C}_{11}$')
    # ----------------
    iteration_total = 0
    results_name = (f'K4_ijklqyz' + f'_it{iteration_total}')
    K4_ijklqyz_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True)

    ax_geom_0 = fig.add_subplot(gs[0, 0])
    # ax_geom_0 = fig.add_axes([0.1, 0.75, 0.2, 0.2])
    pcm = ax_geom_0.pcolormesh(np.tile(K4_ijklqyz_G[ijkl + (..., 0)], (1, 1)),
                               cmap=cmap_, norm=norm,
                               linewidth=0,
                               rasterized=True)
    ax_geom_0.set_aspect('equal')
    ax_geom_0.set_title(f' i={iteration_total}')
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

    ax_geom_0 = fig.add_subplot(gs[0, 2])
    # ax_geom_0 = fig.add_axes([0.5, 0.75, 0.2, 0.2])
    pcm = ax_geom_0.pcolormesh(np.tile(K4_ijklqyz_G[ijkl + (..., 0)], (1, 1)),
                               cmap=cmap_, norm=norm,
                               linewidth=0,
                               rasterized=True)
    ax_geom_0.set_aspect('equal')
    ax_geom_0.set_title(f' i={iteration_total}')
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
    iteration_total = 6
    # iteration_total = 2
    results_name = (f'K4_ijklqyz' + f'_it{iteration_total}')
    K4_ijklqyz_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True)

    # ax_geom_0 = fig.add_axes([0.7, 0.75, 0.2, 0.2])
    ax_geom_0 = fig.add_subplot(gs[0, 3])

    pcm = ax_geom_0.pcolormesh(np.tile(K4_ijklqyz_G[ijkl + (..., 0)], (1, 1)),
                               cmap=cmap_, norm=norm,
                               linewidth=0,
                               rasterized=True)
    ax_geom_0.set_aspect('equal')
    ax_geom_0.set_title(f' i={iteration_total}')
    ax_geom_0.text(-0.1, 1.1, rf'\textbf{{(a.{4})}}', transform=ax_geom_0.transAxes)
    ax_geom_0.set_aspect('equal')
    ax_geom_0.set_xticks([])
    ax_geom_0.set_xticklabels([])
    ax_geom_0.set_yticks([])
    ax_geom_0.set_yticklabels([])
    ax_geom_0.set_xlim([0, Nx - 1])
    ax_geom_0.set_ylim([0, Nx - 1])
    ax_geom_0.set_box_aspect(1)  # Maintain square aspect ratio
    # # print initial geometry
    # ax_geom_0 = fig.add_axes([0.00, 0.75, 0.2, 0.2])
    # pcm = ax_geom_0.pcolormesh(np.tile(K4_init[ijkl + (..., 0)], (1, 1)),
    #                            cmap=cmap_,norm=norm,
    #                            linewidth=0,
    #                            rasterized=True)
    # ax_geom_0.set_title(f' i={-1}')
    # ax_geom_0.text(-0.1, 1.1, rf'\textbf{{(a.1)}}', transform=ax_geom_0.transAxes)
    # ax_geom_0.set_aspect('equal')

    plt.show()

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
