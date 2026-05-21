import numpy as np
import pylab as pl
import scipy as sc
import time
import os
from NuMPI.IO import save_npy, load_npy
from mpi4py import MPI

import matplotlib as mpl
from matplotlib import pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX
    # "font.family": "helvetica",  # Use a serif font
})
plt.rcParams.update({'font.size': 13})
plt.rcParams["font.family"] = "Arial"

script_name = 'exp_paper_JG_nonlinear_elasticity_JZ_cube_2D'  # exp_paper_JG_nonlinear_elasticity_JZ
folder_name = '../exp_data/'
file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory

figure_folder_path = file_folder_path + '/figures/' + script_name + '/'

plot_iterations_vs_grids_size = False

plot_iterations_vs_grids_size_data_anal = True
plot_convergence = False

plot_C12_graph=True
plot_data_convergence=False
if plot_data_convergence:

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

    it_max = 9
    n_exponents = np.array([2, 3, 4])  # , 3, 4, 5
    iterations = np.arange(it_max)  # numbers of grids points

    grid_sizes = np.array(
        [16, 32, 64,128,256])  # , 128, 256
    # grid_sizes= np.array( [ 50, 100, 150 ,200])#,200,128,200

    its_G = np.zeros([len(grid_sizes), it_max, len(n_exponents)])
    its_GJ = np.zeros([len(grid_sizes), it_max, len(n_exponents)])
    its_newton_G = np.zeros([len(grid_sizes), len(n_exponents)])
    its_newton_GJ = np.zeros([len(grid_sizes), len(n_exponents)])

    K4_xyz_GJ_mean = np.zeros([len(grid_sizes), it_max, len(n_exponents)])
    K4_xyz_G_mean = np.zeros([len(grid_sizes), it_max, len(n_exponents)])

    unique_components = np.zeros([len(grid_sizes), it_max, len(n_exponents)])
    total_contrast = np.zeros([len(grid_sizes), it_max, len(n_exponents)])

    for i, n in enumerate(grid_sizes):
        print(i, n)

        Nx = n
        Ny = Nx
        Nz = Nx

        for j in np.arange(len(n_exponents)):
            n_exp = n_exponents[j]

            for iteration_total in iterations:

                preconditioner_type = 'Green'

                data_folder_path = (
                        file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}'
                        + f'_{preconditioner_type}' + '/')
                if iteration_total < it_max:
                    # if Nx == 256:
                    #     _info_final_G = np.load(data_folder_path + f'info_log_it{iteration_total}.npz', allow_pickle=True)
                    #
                    #try:
                        # _info_final_G_final = np.load(data_folder_path + f'info_log_final_exp_{n_exp}.npz',
                        #                               allow_pickle=True)
                        #
                        # its_newton_G[i, j] = _info_final_G_final.f.sum_Newton_its
                        #
                        # _info_final_G = np.load(data_folder_path + f'info_log_exp_{n_exp}_it{iteration_total}.npz',
                        #                         allow_pickle=True)

                    results_name = (f'stress' + f'_exp_{n_exp}_it{iteration_total}')

                    K4_xyz_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True,
                                       mmap_mode='r')
                    K4_xyz_G_mean[i, iteration_total, j] = np.mean(K4_xyz_G)

                   # except:

                     #   K4_xyz_G_mean[i, iteration_total, j] = 0

                preconditioner_type = 'Green_Jacobi'

                data_folder_path = (
                        file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}'
                        + f'_{preconditioner_type}' + '/')
                if iteration_total < it_max:
                    # if Nx == 256:
                    #     _info_final_GJ = np.load(data_folder_path + f'info_log_it{iteration_total}.npz',
                    #                              allow_pickle=True)
                    #
                    # else:
                    try:
                        # _info_final_GJ_final = np.load(data_folder_path + f'info_log_final_exp_{n_exp}.npz',
                        #                                allow_pickle=True)
                        # its_newton_GJ[i, j] = _info_final_GJ_final.f.sum_Newton_its
                        #
                        # _info_final_GJ = np.load(data_folder_path + f'info_log_exp_{n_exp}_it{iteration_total}.npz',
                        #                          allow_pickle=True)
                        results_name = (f'stress' + f'_exp_{n_exp}_it{iteration_total}')

                        K4_xyz_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True,
                                           mmap_mode='r')
                        K4_xyz_GJ_mean[i, iteration_total, j] = np.mean(K4_xyz_G)
                    except:
                        K4_xyz_GJ_mean[i, iteration_total, j] = 0

    for exp_ind, exp in enumerate(n_exponents ):
        last_values = np.array([row[row != 0][-1] if np.any(row != 0) else np.nan
                                for row in K4_xyz_GJ_mean[:, :, exp_ind]])

        data = K4_xyz_GJ_mean[:, :, exp_ind]

        # Last non-zero per row
        # last_values_2 = np.array([row[row != 0][-1] for row in data])

        # Relative error
        reference = last_values[-1]
        relative_error = (last_values - reference) / reference

        # Mask out zeros
       # relative_error[data == 0] = np.nanlast_values
        print(last_values)

        print(relative_error)
    print('K4_xyz_G_mean', K4_xyz_G_mean.shape)

if plot_iterations_vs_grids_size:

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

    it_max = 10
    n_exponents = np.array([ 3,4,5 ])  # , 3, 4, 5
    iterations = np.arange(it_max)  # numbers of grids points

    grid_sizes = np.array(
        [16, 32, 64, 128, 256])  # , 32, 64, 128, 256
    # grid_sizes= np.array( [ 50, 100, 150 ,200])#,200,128,200

    its_G = np.zeros([len(grid_sizes), it_max, len(n_exponents)])
    its_GJ = np.zeros([len(grid_sizes), it_max, len(n_exponents)])
    its_newton_G = np.zeros([len(grid_sizes), len(n_exponents)])
    its_newton_GJ = np.zeros([len(grid_sizes), len(n_exponents)])

    norm_newton_stop_G = np.zeros([len(grid_sizes), it_max, len(n_exponents)])
    norm_newton_stop_GJ = np.zeros([len(grid_sizes), it_max, len(n_exponents)])

    unique_components = np.zeros([len(grid_sizes), it_max, len(n_exponents)])
    total_contrast = np.zeros([len(grid_sizes), it_max, len(n_exponents)])

    for i, n in enumerate(grid_sizes):
        print(i, n)

        Nx = n
        Ny = Nx
        Nz = Nx

        for j in np.arange(len(n_exponents)):
            n_exp = n_exponents[j]

            for iteration_total in iterations:

                preconditioner_type = 'Green'

                data_folder_path = (
                        file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}'
                        + f'_{preconditioner_type}' + '/')
                if iteration_total < it_max:
                    # if Nx == 256:
                    #     _info_final_G = np.load(data_folder_path + f'info_log_it{iteration_total}.npz', allow_pickle=True)
                    #
                    try:
                        _info_final_G_final = np.load(data_folder_path + f'info_log_final_exp_{n_exp}.npz',
                                                      allow_pickle=True)

                        its_newton_G[i, j] = _info_final_G_final.f.sum_Newton_its

                        _info_final_G = np.load(data_folder_path + f'info_log_exp_{n_exp}_it{iteration_total}.npz',
                                                allow_pickle=True)
                        its_G[i, iteration_total, j] = _info_final_G.f.nb_it_comb
                        norm_rhs_G.append(_info_final_G.f.norm_rhs_field)
                        norm_newrton_stop_G.append(_info_final_G.f.newton_stop_crit)
                        norm_newton_stop_G[i, iteration_total, j] = _info_final_G.f.newton_stop_crit
                    except:

                        its_G[i, iteration_total, j] = 0
                        norm_rhs_G.append(0)
                        norm_newrton_stop_G.append(0)
                        norm_newton_stop_G[i, iteration_total, j] = 0

                results_name = (f'K4_ijklqyz' + f'_exp_{n_exp}_it{iteration_total}')

                preconditioner_type = 'Green_Jacobi'

                data_folder_path = (
                        file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}'
                        + f'_{preconditioner_type}' + '/')
                if iteration_total < it_max:
                    # if Nx == 256:
                    #     _info_final_GJ = np.load(data_folder_path + f'info_log_it{iteration_total}.npz',
                    #                              allow_pickle=True)
                    #
                    # else:
                    try:
                        _info_final_GJ_final = np.load(data_folder_path + f'info_log_final_exp_{n_exp}.npz',
                                                       allow_pickle=True)
                        its_newton_GJ[i, j] = _info_final_GJ_final.f.sum_Newton_its

                        _info_final_GJ = np.load(data_folder_path + f'info_log_exp_{n_exp}_it{iteration_total}.npz',
                                                 allow_pickle=True)
                        its_GJ[i, iteration_total, j] = _info_final_GJ.f.nb_it_comb
                        norm_rhs_GJ.append(_info_final_GJ.f.norm_rhs_field)
                        norm_newrton_stop_GJ.append(_info_final_GJ.f.newton_stop_crit)
                        norm_newton_stop_GJ[i, iteration_total, j] = _info_final_GJ.f.newton_stop_crit
                    except:
                        its_GJ[i, iteration_total, j] = 0
                        norm_rhs_GJ.append(0)
                        norm_newrton_stop_GJ.append(0)
                        norm_newton_stop_GJ[i, iteration_total, j] = 0

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'orange', 'purple']

    lines = ['-', '--', '-.', ':', 'dotted', '--', ':', ]

    fig = plt.figure(figsize=(8.3, 4.0))
    gs = fig.add_gridspec(1, 2, hspace=0.5, wspace=0.1, width_ratios=[1, 1],
                          height_ratios=[1])

    gs_iter_newton = fig.add_subplot(gs[0, 0])
    gs_iter_newton.text(-0.00, 1.03, r'$\textbf{(a)}$', transform=gs_iter_newton.transAxes)

    # p_exp_index=2
    grid_size = -1
    nodes_G = [3, 3, 3]  #

    arrows_G = iterations[nodes_G]

    text_G = np.array([[0.3, 300],
                       [1.0, 390],
                       [2.0, 600]])
    # possition of arrows for Green-Jacobi
    # nodes_GJ = [13, 13, 13]
    nodes_GJ = [4, 6, 8]

    arrows_GJ = iterations[nodes_GJ]
    text_GJ = np.array([[1.5, 50],
                        [3.5, 60],
                        [6.5, 60]
                        ])
    # for i, n in enumerate(grid_sizes[2:]):
    for p_exp_index, exp in enumerate(n_exponents):
        # if p_exp_index == 1:
        #     continue
        text_ = fr'$\omega$ = $ {{{exp}}}$'
        line_weight = 2
        if p_exp_index == 1:
            text_ = f'Green\n' + fr'$\omega$ = $ {{{exp}}}$'
            line_weight = 3

        mask = ~(its_G[grid_size, :, p_exp_index] == 0)
        gs_iter_newton.plot(iterations[mask], its_G[grid_size, :, p_exp_index][mask], ls=lines[p_exp_index], marker='x',
                            color='Green',
                            label=f'Green\n' + fr'$\omega$ = $ {{{exp}}}$', lw=line_weight)

        gs_iter_newton.annotate(text=text_,
                                xy=(arrows_G[p_exp_index], its_G[grid_size, :, p_exp_index][nodes_G[p_exp_index]]),
                                xytext=(text_G[p_exp_index, 0], text_G[p_exp_index, 1]),
                                arrowprops=dict(arrowstyle='->',
                                                color='green',
                                                lw=1,
                                                ls=lines[p_exp_index]),
                                fontsize=13,
                                color='green'
                                )

        mask = ~(its_GJ[grid_size, :, p_exp_index] == 0)
        gs_iter_newton.plot(iterations[mask], its_GJ[grid_size, :, p_exp_index][mask], ls=lines[p_exp_index],
                            marker='o', color='k',
                            markerfacecolor='none',
                            label=f'Green-Jacobi \n' + fr'$\omega$ = $ {{{exp}}}$', lw=line_weight)
        text_ = fr'$\omega$ = $ {{{exp}}}$'
        if p_exp_index == 1:
            text_ = f'Green-Jacobi\n' + fr'$\omega$ = $ {{{exp}}}$'
        gs_iter_newton.annotate(text=text_,
                                xy=(arrows_GJ[p_exp_index], its_GJ[grid_size, :, p_exp_index][nodes_GJ[p_exp_index]]),
                                xytext=(text_GJ[p_exp_index, 0], text_GJ[p_exp_index, 1]),
                                arrowprops=dict(arrowstyle='->',
                                                color='black',
                                                lw=1,
                                                ls=lines[p_exp_index]),
                                fontsize=13,
                                color='black'
                                )

        gs_iter_newton.set_xlabel(r'Newton iteration -  $i$')
        gs_iter_newton.set_ylabel(r'Number of PCG iterations')
        # gs_global.legend(loc='best')
        gs_iter_newton.set_xlim(-0.0, iterations[-1])
        gs_iter_newton.set_ylim(30, 1000)
        gs_iter_newton.set_yscale('log')
    # gs_iter_newton.text(0.20, 0.95, r'$ \approx  50 \times 10^{6}$ DOFs ',
    #                transform=gs_iter_newton.transAxes)
    gs_iter_newton.text(0.05, 0.92, r'$ N_{\mathrm{N}}  =256^3$',
                        transform=gs_iter_newton.transAxes,
                        fontsize=13,
                        color='black'
                        )

    gs_iter_grid = fig.add_subplot(gs[0, 1])
    gs_iter_grid.text(-0.00, 1.03, r'$\textbf{(b)}$', transform=gs_iter_grid.transAxes)

    total_it_per_grid_cg = {
        'Green': [],
        'Green_Jacobi': [],
    }
    avarage = False
    # for i, n in enumerate(n_exponents):
    if avarage:

        total_it_per_grid_cg['Green'].append(np.sum(its_G, axis=1) / its_newton_G)
        total_it_per_grid_cg['Green_Jacobi'].append(np.sum(its_GJ, axis=1) / its_newton_GJ)
    else:
        total_it_per_grid_cg['Green'].append(np.sum(its_G, axis=1))
        total_it_per_grid_cg['Green_Jacobi'].append(np.sum(its_GJ, axis=1))

    nb_nodes = grid_sizes ** 3
    # plot set up

    if avarage:
        # possition of arrows for Green
        nodes_G = [3, 2, 3]
        arrows_G = nb_nodes[nodes_G]

        text_G = np.array([[70 ** 3, 3],
                           [20 ** 3, 30],
                           [100 ** 3, 150]])
        # possition of arrows for Green-Jacobi
        nodes_GJ = [3, 2, 3]

        arrows_GJ = nb_nodes[nodes_GJ]
        text_GJ = np.array([[110 ** 3, 1.5],
                            [20 ** 3, 3],
                            [45 ** 3, 70]])
    else:
        nodes_G = [2, 2, 2]

        arrows_G = nb_nodes[nodes_G]

        text_G = np.array([[18 ** 3, 1400],
                           [30 ** 3, 2000],
                           [50 ** 3, 3400]])
        # possition of arrows for Green-Jacobi
        nodes_GJ = [2, 2, 2]

        arrows_GJ = nb_nodes[nodes_GJ]
        text_GJ = np.array([[32 ** 3, 110],
                            [64 ** 3, 130],
                            [128 ** 3, 270]])
    for e, exp in enumerate(n_exponents):
        text_ = fr'$\omega$ = $ {{{exp}}}$'
        line_weight = 2
        if e == 1:
            text_ = f'Green\n' + fr'$\omega$ = $ {{{exp}}}$'
            line_weight = 3

        gs_iter_grid.plot(nb_nodes, total_it_per_grid_cg['Green'][0][:, e], linestyle=lines[e],
                          color='Green', marker='x', label=f'Green\n' + fr'$\omega$ = $ {{{exp}}}$', lw=line_weight
                          )

        gs_iter_grid.annotate(text=text_,
                              xy=(arrows_G[e], total_it_per_grid_cg['Green'][0][:, e][nodes_G[e]]),
                              xytext=(text_G[e, 0], text_G[e, 1]),
                              arrowprops=dict(arrowstyle='->',
                                              color='green',
                                              lw=1,
                                              ls=lines[e]),
                              fontsize=13,
                              color='green'
                              )

        gs_iter_grid.plot(nb_nodes, total_it_per_grid_cg['Green_Jacobi'][0][:, e], linestyle=lines[e],
                          color='k', marker='o', markerfacecolor='none',
                          label=f'Green-Jacobi\n' + fr'$\omega$ = $ {{{exp}}}$', lw=line_weight)
        text_ = fr'$\omega$ = $ {{{exp}}}$'
        if e == 1:
            text_ = f'Green-Jacobi\n' + fr'$\omega$ = $ {{{exp}}}$'
        gs_iter_grid.annotate(text=text_,
                              xy=(arrows_GJ[e], total_it_per_grid_cg['Green_Jacobi'][0][:, e][nodes_GJ[e]]),
                              xytext=(text_GJ[e, 0], text_GJ[e, 1]),
                              arrowprops=dict(arrowstyle='->',
                                              color='black',
                                              lw=1,
                                              ls=lines[e]),
                              fontsize=13,
                              color='black'
                              )

    # gs_iter_grid.legend(
    #     loc='center left',
    #     bbox_to_anchor=(1.02, 0.5)
    # )
    # gs_iter_vs_grid.set_title('total_it_per_grid_cg vs grid_sizes')
    gs_iter_grid.set_xlabel(r'Number of nodes - $N_{\mathrm{N}}$')
    if avarage:
        gs_iter_grid.set_ylabel('Average number of PCG iterations')
    else:
        gs_iter_grid.set_ylabel('Total number of PCG iterations')

    # #
    if avarage:
        # gs_iter_grid.set_title('Avarage number of PCG iterations')

        gs_iter_grid.set_ylim([1e0, 1e2])  # norm_rz[i][0]]/lb)
        gs_iter_grid.set_yscale('log')
    else:
        # gs_iter_grid.set_title('Total number of PCG iterations')
        # gs_iter_vs_grid.set_ylim([1e1, 1e3])
        # gs_iter_vs_grid.set_yscale('log')
        gs_iter_grid.set_ylim([1e2, 5e3])
        gs_iter_grid.set_yscale('log')
    gs_iter_grid.set_xlim([nb_nodes[0], nb_nodes[-1]])

    gs_iter_grid.set_xscale('log')
    gs_iter_grid.set_xticks(nb_nodes)
    gs_iter_grid.set_xticklabels([fr'${n}^{{3}}$' for n in grid_sizes])

    gs_iter_grid.yaxis.set_ticks_position('right')  # Set y-axis ticks to the right
    gs_iter_grid.yaxis.set_label_position('right')

    fig.tight_layout()
    fname = f'nbsteps' + f'ex{n_exp}' + f'_av_{avarage}' + '{}'.format('.pdf')
    plt.savefig(figure_folder_path + fname, bbox_inches='tight')
    print(('create figure: {}'.format(figure_folder_path + fname)))
    plt.show()

    print()

if plot_iterations_vs_grids_size_data_anal:

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

    it_max = 80
    n_exponents = np.array([    3 ])  # 3,4,5
    iterations = np.arange(it_max)  # numbers of grids points
    grid_sizes = np.array([16, 32,64, 128, 256 ])  # 64, 128, 256     ,128, 64, 128, 256]
    # grid_sizes= np.array( [ 50, 100, 150 ,200])#,200,128,200

    its_G = np.zeros([len(grid_sizes), it_max, len(n_exponents)])
    its_GJ = np.zeros([len(grid_sizes), it_max, len(n_exponents)])

    norm_rr_G = np.zeros([len(grid_sizes), it_max, len(n_exponents)])
    norm_rr_GJ = np.zeros([len(grid_sizes), it_max, len(n_exponents)])

    norm_newton_stop_G = np.zeros([len(grid_sizes), it_max, len(n_exponents)])
    norm_newton_stop_GJ = np.zeros([len(grid_sizes), it_max, len(n_exponents)])

    norm_strain_fluc_field_G = np.zeros([len(grid_sizes), it_max, len(n_exponents)])
    norm_strain_fluc_field_GJ = np.zeros([len(grid_sizes), it_max, len(n_exponents)])

    unique_components = np.zeros([len(grid_sizes), it_max, len(n_exponents)])
    total_contrast = np.zeros([len(grid_sizes), it_max, len(n_exponents)])

    for i, n in enumerate(grid_sizes):
        print(i, n)

        Nx = n
        Ny = Nx
        Nz = Nx

        for j in np.arange(len(n_exponents)):
            n_exp = n_exponents[j]
            for iteration_total in iterations:

                preconditioner_type = 'Green'

                data_folder_path = (
                        file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}'
                        + f'_{preconditioner_type}' + '/')
                if iteration_total < it_max:
                    # if Nx == 256:
                    #     _info_final_G = np.load(data_folder_path + f'info_log_it{iteration_total}.npz', allow_pickle=True)
                    #
                    # else:
                    try:
                        _info_final_G = np.load(data_folder_path + f'info_log_exp_{n_exp}_it{iteration_total}.npz',
                                                allow_pickle=True)
                        its_G[i, iteration_total, j] = _info_final_G.f.nb_it_comb
                        # norm_rr_G[i, iteration_total, j] =_info_final_G.f.norm_rr
                        if plot_convergence:
                            plt.figure()
                            plt.loglog(np.arange(1, _info_final_G.f.norm_rr.shape[0] + 1),
                                       _info_final_G.f.norm_rr / _info_final_G.f.norm_rr[0], 'green')  #
                            plt.loglog(np.arange(1, _info_final_G.f.norm_rz.shape[0] + 1),
                                       _info_final_G.f.norm_rz / _info_final_G.f.norm_rz[0], 'green', linestyle=':')  #

                            plt.ylim([1e-16, 1e1])
                            plt.xlim([1, 1000])
                            plt.title(f'  grid- {n}, iteration {iteration_total}, p={n_exp}')
                        # plt.show()
                        norm_rhs_G.append(_info_final_G.f.norm_rhs_field)
                        norm_newrton_stop_G.append(_info_final_G.f.newton_stop_crit)
                        norm_newton_stop_G[i, iteration_total, j] = _info_final_G.f.newton_stop_crit
                        norm_strain_fluc_field_G[i, iteration_total, j] = _info_final_G.f.norm_strain_fluc_field / (
                            _info_final_G.f.norm_En)

                    except:
                        its_G[i, iteration_total, j] = 0
                        norm_rhs_G.append(0)
                        norm_newrton_stop_G.append(0)
                        norm_newton_stop_G[i, iteration_total, j] = 0
                        norm_strain_fluc_field_G[i, iteration_total, j] = 0

                results_name = (f'K4_ijklqyz' + f'_exp_{n_exp}_it{iteration_total}')
                try:
                    K4_xyz_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True, mmap_mode='r')

                    unique_components[i, iteration_total, j] = np.unique(np.round(K4_xyz_G, decimals=6)).size
                    total_contrast[i, iteration_total, j] = np.max(K4_xyz_G) / np.min(K4_xyz_G)
                except:
                    unique_components[i, iteration_total, j] = 0
                    total_contrast[i, iteration_total, j] = 0
                #  info_log_final_G = np.load(data_folder_path + f'info_log_final_exp_{n_exp}.npz', allow_pickle=True)

                preconditioner_type = 'Green_Jacobi'

                data_folder_path = (
                        file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}'
                        + f'_{preconditioner_type}' + '/')
                if iteration_total < it_max:

                    try:
                        _info_final_GJ = np.load(data_folder_path + f'info_log_exp_{n_exp}_it{iteration_total}.npz',
                                                 allow_pickle=True)
                        its_GJ[i, iteration_total, j] = _info_final_GJ.f.nb_it_comb
                        # norm_rr_GJ[i, iteration_total, j] =_info_final_GJ.f.norm_rr
                        if plot_convergence:
                            #  pl.figure()
                            plt.loglog(np.arange(1, _info_final_GJ.f.norm_rr.shape[0] + 1),
                                       _info_final_GJ.f.norm_rr / _info_final_GJ.f.norm_rr[0], 'k')  #
                            plt.loglog(np.arange(1, _info_final_GJ.f.norm_rz.shape[0] + 1),
                                       _info_final_GJ.f.norm_rz / _info_final_GJ.f.norm_rz[0], 'k', linestyle=':')  #
                            #
                            # #plt.ylim([1e-16, 10])
                            plt.show()
                        # plt.title(f'grid- {n}, iteration {iteration_total}, p={n_exp}')
                        norm_rhs_GJ.append(_info_final_GJ.f.norm_rhs_field)
                        norm_newrton_stop_GJ.append(_info_final_GJ.f.newton_stop_crit)
                        norm_newton_stop_GJ[i, iteration_total, j] = _info_final_GJ.f.newton_stop_crit
                        norm_strain_fluc_field_GJ[i, iteration_total, j] = _info_final_GJ.f.norm_strain_fluc_field / (
                            _info_final_GJ.f.norm_En)

                    except:
                        its_GJ[i, iteration_total, j] = 0
                        norm_rhs_GJ.append(0)
                        norm_newrton_stop_GJ.append(0)
                        norm_newton_stop_GJ[i, iteration_total, j] = 0
                        norm_strain_fluc_field_GJ[i, iteration_total, j] = 0

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'orange', 'purple']

    for p_exp_index, p_exp in enumerate(n_exponents):
        fig = plt.figure(figsize=(8.3, 18.0))
        gs = fig.add_gridspec(5, 1, hspace=0.4, wspace=0.1, width_ratios=[1],
                              height_ratios=[1, 1, 1, 1, 1])
        gs_fnorm_vs_iteration = fig.add_subplot(gs[0, 0])
        plt.title(f' exponent = {p_exp}')
        for i, n in enumerate(grid_sizes):
            gs_fnorm_vs_iteration.semilogy(iterations, norm_newton_stop_G[i, :, p_exp_index]
                                           / norm_newton_stop_G[i, 0, p_exp_index]
                                           , '-', color=colors[i],
                                           marker='x', label=f'Green - {n}')  #
            gs_fnorm_vs_iteration.semilogy(iterations,
                                           norm_newton_stop_GJ[i, :, p_exp_index]
                                           / norm_newton_stop_GJ[i, 0, p_exp_index]
                                           , '--', color=colors[i],
                                           marker='o', markerfacecolor='none', label=f'Green-Jacobi - {n}')  #
        gs_fnorm_vs_iteration.legend(loc='upper right')
        gs_fnorm_vs_iteration.set_xlabel('Newton iteration')
        gs_fnorm_vs_iteration.set_ylabel('Relative norm of rhs')

        gs_fnorm_vs_iteration.set_ylim([1e-10, 1e1])

        gs_strain_vs_iteration = fig.add_subplot(gs[1, 0])
        plt.title(f' exponent = {p_exp}')
        for i, n in enumerate(grid_sizes):
            gs_strain_vs_iteration.semilogy(iterations, norm_strain_fluc_field_G[i, :, p_exp_index]
                                            , '-', color=colors[i],
                                            marker='x', label=f'Green - {n}')  #
            gs_strain_vs_iteration.semilogy(iterations,
                                            norm_strain_fluc_field_GJ[i, :, p_exp_index]
                                            , '--', color=colors[i],
                                            marker='o', markerfacecolor='none', label=f'Green-Jacobi - {n}')  #
        gs_strain_vs_iteration.legend(loc='upper right')
        gs_strain_vs_iteration.set_xlabel('Newton iteration')
        gs_strain_vs_iteration.set_ylabel('Relative norm of Strain fluctuation')

        gs_strain_vs_iteration.set_ylim([1e-7, 1e1])

        gs_iter_vs_mesh_size = fig.add_subplot(gs[2, 0])
        for i, n in enumerate(grid_sizes):
            gs_iter_vs_mesh_size.plot(iterations, its_G[i, :, p_exp_index], '-', marker='x', color=colors[i],
                                      label=f'Green - {n}')
            gs_iter_vs_mesh_size.plot(iterations, its_GJ[i, :, p_exp_index], '--', marker='o', color=colors[i],
                                      markerfacecolor='none',
                                      label=f'Green-Jacobi - {n}')
        gs_iter_vs_mesh_size.legend()
        gs_iter_vs_mesh_size.set_ylabel('CG iterations  count')
        #  gs_iter_vs_mesh_size.set_yscale('log')
        gs_iter_vs_mesh_size.set_ylim([1, 100])

        # gs_iter_vs_mesh_size.set_title('Iterations vs Mesh Size')

        gs_iter_vs_unique_ = fig.add_subplot(gs[3, 0])
        for i, n in enumerate(grid_sizes):
            gs_iter_vs_unique_.semilogy(iterations, unique_components[i, :, p_exp_index], '-', marker='x',
                                        color=colors[i],
                                        label=f' {n}')
        gs_iter_vs_unique_.legend()
        # gs_iter_vs_unique_.set_title('Unique vsIterations')
        gs_iter_vs_unique_.set_xlabel('Newton iteration')
        gs_iter_vs_unique_.set_ylabel('Unique material data')

        gs_iter_vs_contrast = fig.add_subplot(gs[4, 0])
        for i, n in enumerate(grid_sizes):
            gs_iter_vs_contrast.semilogy(iterations, total_contrast[i, :, p_exp_index], '-', marker='x', label=f' {n}')
        gs_iter_vs_contrast.legend()
        gs_iter_vs_contrast.set_xlabel('Newton iteration')
        gs_iter_vs_contrast.set_ylabel('K_11 contrast')
        # gs_iter_vs_contrast.set_ylim([1, 1e3])

        fig.tight_layout()
        fname = f'fig_temp' + f'exp_{p_exp_index}' + '{}'.format('.pdf')
        plt.savefig(figure_folder_path + script_name + fname, bbox_inches='tight')
        print(('create figure: {}'.format(figure_folder_path + script_name + fname)))
        plt.show()

    fig = plt.figure(figsize=(4.5, 4.5))
    gs = fig.add_gridspec(1, 1, hspace=0.4, wspace=0.1, width_ratios=[1],
                          height_ratios=[1])
    gs_fnorm_vs_iteration = fig.add_subplot(gs[0, 0])
    # plt.title(f' exponent = {p_exp}')
    for p_exp_index, exp in enumerate(n_exponents):
        gs_fnorm_vs_iteration.semilogy(iterations, norm_newton_stop_G[-1, :, p_exp_index] ** 2 / norm_newton_stop_G[
            -1, 0, p_exp_index] ** 2, '-', color=colors[p_exp_index],
                                       marker='x', label=f'Green - {exp}')  #
        gs_fnorm_vs_iteration.semilogy(iterations,
                                       norm_newton_stop_GJ[-1, :, p_exp_index] ** 2
                                       / norm_newton_stop_GJ[-1, 0, p_exp_index] ** 2
                                       , '--', color=colors[p_exp_index],
                                       marker='o', markerfacecolor='none', label=f'Green-Jacobi - {exp}')  #
    gs_fnorm_vs_iteration.legend(loc='upper right')
    gs_fnorm_vs_iteration.set_xlabel('Newton iteration')
    gs_fnorm_vs_iteration.set_ylabel('Relative norm of residua')

    gs_fnorm_vs_iteration.set_ylim([1e-14, 1e0])
    fig.tight_layout()
    fname = f'fig_newtonconver' + f'exp_{p_exp_index}' + '{}'.format('.pdf')
    plt.savefig(figure_folder_path + script_name + fname, bbox_inches='tight')
    print(('create figure: {}'.format(figure_folder_path + script_name + fname)))
    plt.show()
    print()



    ###### supporting
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'orange', 'purple']

    lines = ['-', '--', '-.', ':', 'dotted', '--', ':', ]

    fig = plt.figure(figsize=(8.3, 4.0))
    gs = fig.add_gridspec(1, 2, hspace=0.5, wspace=0.1, width_ratios=[1, 1],
                          height_ratios=[1])

    gs_iter_newton = fig.add_subplot(gs[0, 0])
    gs_iter_newton.text(-0.00, 1.03, r'$\textbf{(a)}$', transform=gs_iter_newton.transAxes)

    # p_exp_index=2
    grid_size = -1
    nodes_G = [3, 3, 3]  #

    arrows_G = iterations[nodes_G]

    text_G = np.array([[0.3, 300],
                       [1.0, 390],
                       [2.0, 600]])
    # possition of arrows for Green-Jacobi
    # nodes_GJ = [13, 13, 13]
    nodes_GJ = [4, 6, 8]

    arrows_GJ = iterations[nodes_GJ]
    text_GJ = np.array([[2, 60],
                        [3.5, 60],
                        [6.5, 60]
                        ])
    # for i, n in enumerate(grid_sizes[2:]):
    for p_exp_index, exp in enumerate(n_exponents):
        # if p_exp_index == 1:
        #     continue
        text_ = fr'$\omega$ = $ {{{exp}}}$'
        line_weight = 2
        if p_exp_index == 1:
            text_ = f'Green\n' + fr'$\omega$ = $ {{{exp}}}$'
            line_weight = 3

        mask = ~(its_G[grid_size, :, p_exp_index] == 0)
        gs_iter_newton.plot(iterations[mask], total_contrast[grid_size, :, p_exp_index][mask], ls=lines[p_exp_index], marker='x',
                            color='red',
                            label=f'Green\n' + fr'$\omega$ = $ {{{exp}}}$', lw=line_weight)

        gs_iter_newton.annotate(text=text_,
                                xy=(arrows_G[p_exp_index], total_contrast[grid_size, :, p_exp_index][nodes_G[p_exp_index]]),
                                xytext=(text_G[p_exp_index, 0], text_G[p_exp_index, 1]),
                                arrowprops=dict(arrowstyle='->',
                                                color='green',
                                                lw=1,
                                                ls=lines[p_exp_index]),
                                fontsize=13,
                                color='green'
                                )
        gs_iter_newton.plot(iterations[mask], unique_components[grid_size, :, p_exp_index][mask], ls=lines[p_exp_index],
                            marker='o',
                            color='blue',
                            label=f'Green\n' + fr'$\omega$ = $ {{{exp}}}$', lw=line_weight)

        gs_iter_newton.set_xlabel(r'Newton iteration -  $i$')
        gs_iter_newton.set_ylabel(r'Total phase contrast')
        # gs_global.legend(loc='best')
        gs_iter_newton.set_xlim(-0.0, iterations[-1])
       # gs_iter_newton.set_ylim(-0.5, 1000)
        gs_iter_newton.set_yscale('log')
    # gs_iter_newton.text(0.20, 0.95, r'$ \approx  50 \times 10^{6}$ DOFs ',
    #                transform=gs_iter_newton.transAxes)
    gs_iter_newton.text(0.05, 0.92, r'$ N_{\mathrm{N}}  =256^3$',
                        transform=gs_iter_newton.transAxes,
                        fontsize=13,
                        color='black'
                        )

    gs_iter_grid = fig.add_subplot(gs[0, 1])
    gs_iter_grid.text(-0.00, 1.03, r'$\textbf{(b)}$', transform=gs_iter_grid.transAxes)

    total_it_per_grid_cg = {
        'Green': [],
        'Green_Jacobi': [],
    }
    avarage = False
    # for i, n in enumerate(n_exponents):
    if avarage:

        total_it_per_grid_cg['Green'].append(np.sum(its_G, axis=1) / its_newton_G)
        total_it_per_grid_cg['Green_Jacobi'].append(np.sum(its_GJ, axis=1) / its_newton_GJ)
    else:
        total_it_per_grid_cg['Green'].append(np.sum(its_G, axis=1))
        total_it_per_grid_cg['Green_Jacobi'].append(np.sum(its_GJ, axis=1))

    nb_nodes = grid_sizes ** 3
    # plot set up

    if avarage:
        # possition of arrows for Green
        nodes_G = [3, 2, 3]
        arrows_G = nb_nodes[nodes_G]

        text_G = np.array([[70 ** 3, 3],
                           [20 ** 3, 30],
                           [100 ** 3, 150]])
        # possition of arrows for Green-Jacobi
        nodes_GJ = [3, 2, 3]

        arrows_GJ = nb_nodes[nodes_GJ]
        text_GJ = np.array([[110 ** 3, 1.5],
                            [20 ** 3, 3],
                            [45 ** 3, 70]])
    else:
        nodes_G = [2, 2, 2]

        arrows_G = nb_nodes[nodes_G]

        text_G = np.array([[18 ** 3, 1400],
                           [30 ** 3, 2000],
                           [50 ** 3, 3400]])
        # possition of arrows for Green-Jacobi
        nodes_GJ = [2, 2, 2]

        arrows_GJ = nb_nodes[nodes_GJ]
        text_GJ = np.array([[32 ** 3, 100],
                            [64 ** 3, 130],
                            [128 ** 3, 270]])
    for e, exp in enumerate(n_exponents):
        text_ = fr'$\omega$ = $ {{{exp}}}$'
        line_weight = 2
        if e == 1:
            text_ = f'Green\n' + fr'$\omega$ = $ {{{exp}}}$'
            line_weight = 3
        gs_iter_grid.plot(nb_nodes, total_contrast[..., e].mean(axis=1), linestyle=lines[e],
                          color='red', marker='x', label=f'Green\n' + fr'$\omega$ = $ {{{exp}}}$', lw=line_weight
                          )

        # gs_iter_grid.plot(nb_nodes, unique_components[..., e].mean(axis=1), linestyle=lines[e],
        #                   color='blue', marker='x', label=f'Green\n' + fr'$\omega$ = $ {{{exp}}}$', lw=line_weight
        #                   )

    gs_iter_grid.set_xlabel(r'Number of nodes - $N_{\mathrm{N}}$')
    if avarage:
        gs_iter_grid.set_ylabel('Average number of PCG iterations')
    else:
        gs_iter_grid.set_ylabel('Total number of PCG iterations')

    # #
    if avarage:
        # gs_iter_grid.set_title('Avarage number of PCG iterations')

        gs_iter_grid.set_ylim([1e0, 1e2])  # norm_rz[i][0]]/lb)
        gs_iter_grid.set_yscale('log')
    else:
        # gs_iter_grid.set_title('Total number of PCG iterations')
        # gs_iter_vs_grid.set_ylim([1e1, 1e3])
        # gs_iter_vs_grid.set_yscale('log')
        #  gs_iter_grid.set_ylim([1, 650])
        gs_iter_grid.set_yscale('log')
    gs_iter_grid.set_xlim([nb_nodes[0], nb_nodes[-1]])

    gs_iter_grid.set_xscale('log')
    gs_iter_grid.set_xticks(nb_nodes)
    gs_iter_grid.set_xticklabels([fr'${n}^{{3}}$' for n in grid_sizes])

    gs_iter_grid.yaxis.set_ticks_position('right')  # Set y-axis ticks to the right
    gs_iter_grid.yaxis.set_label_position('right')

    fig.tight_layout()
    fname = f'supporting' + f'ex{n_exp}' + f'_av_{avarage}' + '{}'.format('.pdf')
    plt.savefig(figure_folder_path + fname, bbox_inches='tight')
    print(('create figure: {}'.format(figure_folder_path + fname)))
    plt.show()

    print()


def plot_voxels_colormap_2D(fig, gs_position, values, iteration,
                            cmap='cividis_r', z_cut=0, x_def=None, y_def=None, norm=None,
                            label='(b.2)', cbar_gs_position=None,
                            cbar_label=r'$\mathrm{C}_{11}/\mathrm{K}$',
                            keep_ticks=False):
    ax = fig.add_subplot(gs_position)  # , projection='3d'

    # Get dimensions
    nx, ny = values.shape

    ax.dist = 2  # Lower value = closer/larger plot (default is ~10)

    # Remove padding around axes
    ax.margins(0)

    # Tighten the axis bounds
    ax.autoscale_view(tight=True)

    cmap_obj = mpl.colormaps.get_cmap(cmap) if isinstance(cmap, str) else cmap

    # Create mask for cutaway (True = show, False = hide)
    #  mask_cut = np.ones(values.shape, dtype=bool)

    # Convert values to colors using colormap
    # colors = cmap_obj(norm(values))  # Returns RGBA array [Nx, Ny, Nz, 4]

    # Plot voxels with colormap colors
    # ax.imshow(values[:,:, z_cut])
    pcm = ax.pcolormesh(x_def, y_def, masked, norm=norm, cmap=cmap)

    # Axis limits
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    # ax.set_zlim(0, nz)

    # Ticks: First, Middle, Last
    if keep_ticks:
        ax.set_xticks([0, nx // 2, nx])
        ax.set_yticks([0, ny // 2, ny])

        ax.set_xticklabels(['1', str(nx // 2), str(nx)])
        ax.set_yticklabels(['1', str(ny // 2), str(ny)])

        # Axis labels
        ax.set_xlabel(r'$x_1$', labelpad=3)
        ax.set_ylabel(r'$x_2$', labelpad=3)

    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # Title and label
    ax.set_title(fr'$i={iteration}$')
    if label:
        ax.text(0.0, 1.05, rf'\textbf{{{label}}}', transform=ax.transAxes)

    # Add colorbar if position is provided
    cbar = None
    if cbar_gs_position is not None:
        ax_cbar = fig.add_subplot(cbar_gs_position)
        sm = mpl.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])

        cbar = plt.colorbar(sm, location='left', cax=ax_cbar)
        cbar.set_ticks([min_K, mid_K, max_K])
        # cbar.set_ticklabels([f'{min_K:.1f}', f'{mid_K:.1f}', f'{max_K:.1f}'])
        ax_cbar.tick_params(right=True, top=False, labelright=False, labeltop=False, labelrotation=0)
        cbar.ax.yaxis.set_ticks_position('right')
        cbar.ax.yaxis.set_label_position('right')
        ax_cbar.set_ylabel(cbar_label)
        # from matplotlib.ticker import ScalarFormatter

        # fmt = ScalarFormatter()
        # # fmt.set_powerlimits((-3, 3))  # force scientific notation outside this range
        #
        # cbar.ax.yaxis.set_major_formatter(fmt)
        # cbar.set_ticks([min_K, mid_K, max_K])
        # cbar.set_ticklabels([
        #     fr'$10^{{{int(np.log10(min_K))}}}$',
        #     fr'$10^{{{int(np.log10(mid_K))}}}$',
        #     fr'$10^{{{int(np.log10(max_K))}}}$'
        # ])

    return ax, cbar, cmap

if plot_C12_graph:
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

    n_exponents = [3]

    it_max = 10
    iterations = np.arange(it_max)  # numbers of grids points
    grid_sizes = np.array([64])  # 64, 128, 256     ,128
    # grid_sizes= np.array( [ 50, 100, 150 ,200])#,200,128,200

    its_G = np.zeros([len(grid_sizes), it_max, len(n_exponents)])
    its_GJ = np.zeros([len(grid_sizes), it_max, len(n_exponents)])
    norm_strain_fluc_field_G = np.zeros([len(grid_sizes), it_max, len(n_exponents)])
    norm_strain_fluc_field_GJ = np.zeros([len(grid_sizes), it_max, len(n_exponents)])

    unique_components = np.zeros([len(grid_sizes), it_max, len(n_exponents)])
    total_contrast = np.zeros([len(grid_sizes), it_max, len(n_exponents)])

    for i, n in enumerate(grid_sizes):
        print(i, n)

        Nx = n
        Ny = Nx
        Nz = Nx

        for j in np.arange(len(n_exponents)):
            n_exp = n_exponents[j]
            for iteration_total in iterations:

                preconditioner_type = 'Green'

                data_folder_path = (
                        file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}'
                        + f'_{preconditioner_type}' + '/')
                if iteration_total < it_max:
                    # if Nx == 256:
                    #     _info_final_G = np.load(data_folder_path + f'info_log_it{iteration_total}.npz', allow_pickle=True)
                    #
                    # else:
                    try:
                        _info_final_G = np.load(data_folder_path + f'info_log_exp_{n_exp}_it{iteration_total}.npz',
                                                allow_pickle=True)
                        its_G[i, iteration_total, j] = _info_final_G.f.nb_it_comb
                        # norm_rr_G[i, iteration_total, j] =_info_final_G.f.norm_rr
                        if plot_convergence:
                            plt.figure()
                            plt.loglog(np.arange(1, _info_final_G.f.norm_rr.shape[0] + 1),
                                       _info_final_G.f.norm_rr / _info_final_G.f.norm_rr[0], 'green')  #
                            plt.loglog(np.arange(1, _info_final_G.f.norm_rz.shape[0] + 1),
                                       _info_final_G.f.norm_rz / _info_final_G.f.norm_rz[0], 'green', linestyle=':')  #

                            plt.ylim([1e-16, 1e1])
                            plt.xlim([1, 1000])
                            plt.title(f'  grid- {n}, iteration {iteration_total}, p={n_exp}')
                        # plt.show()

                        norm_strain_fluc_field_G[i, iteration_total, j] = _info_final_G.f.norm_strain_fluc_field / (
                            _info_final_G.f.norm_En)

                    except:
                        its_G[i, iteration_total, j] = 0
                        norm_strain_fluc_field_G[i, iteration_total, j] = 0

                preconditioner_type = 'Green_Jacobi'

                data_folder_path = (
                        file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}'
                        + f'_{preconditioner_type}' + '/')
                if iteration_total < it_max:

                    try:
                        _info_final_GJ = np.load(data_folder_path + f'info_log_exp_{n_exp}_it{iteration_total}.npz',
                                                 allow_pickle=True)
                        its_GJ[i, iteration_total, j] = _info_final_GJ.f.nb_it_comb

                        norm_strain_fluc_field_GJ[i, iteration_total, j] = _info_final_GJ.f.norm_strain_fluc_field / (
                            _info_final_GJ.f.norm_En)

                    except:
                        its_GJ[i, iteration_total, j] = 0
                        norm_strain_fluc_field_GJ[i, iteration_total, j] = 0
        # Set up global plot canvas
        fig = plt.figure(figsize=(8.3, 4.))
        gs = fig.add_gridspec(2, 6, hspace=0.2, wspace=0.2, width_ratios=[1., 0.2, 1, 1, 1, 0.05],
                              height_ratios=[1, 1.])
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'orange', 'purple']

        lines = ['-', '--', '-.', ':', 'dotted', '--', ':', ]
        # add convergence of the strain norm
        gs_strain_vs_iteration = fig.add_subplot(gs[:, 0])
        # plt.title(f'p = {n_exponents[0]}')
        for i, n in enumerate(grid_sizes):
            mask = ~(norm_strain_fluc_field_G[i, :, 0] == 0)
            gs_strain_vs_iteration.semilogy(iterations[mask], norm_strain_fluc_field_G[i, :, 0][mask]
                                            , '-', color='g', linestyle=lines[i], markersize=10,
                                            marker='x', label=f'Green')  #
            mask = ~(norm_strain_fluc_field_GJ[i, :, 0] == 0)

            gs_strain_vs_iteration.semilogy(iterations[mask],
                                            norm_strain_fluc_field_GJ[i, :, 0][mask]
                                            , '--', color='k', linestyle=lines[i],
                                            marker='o', markerfacecolor='none', label=f'Green-Jacobi')  #
        gs_strain_vs_iteration.legend(loc='lower left', fontsize=10)
        gs_strain_vs_iteration.set_xlabel(r'Newton iteration - $i$')
        #gs_strain_vs_iteration.set_ylabel('Relative norm of \n the strain fluctuation increment')
        gs_strain_vs_iteration.set_ylabel(r'$\|\mathsf{B} \delta\tilde{\mathsf{u}}^{(i+1)}\| / \|\mathsf{E}\|$')

        gs_strain_vs_iteration.set_ylim([1e-5, 1e0])
        gs_strain_vs_iteration.set_xlim([0, 7])
        gs_strain_vs_iteration.set_xticks([0, 2, 4, 6])

        gs_strain_vs_iteration.text(0.3, 0.93, r'$N_{\mathrm{N}}=256^3$', transform=gs_strain_vs_iteration.transAxes)
        gs_strain_vs_iteration.text(0.4, 0.85, fr'$\omega$ ={{{n_exponents[0]}}}', transform=gs_strain_vs_iteration.transAxes)

        gs_strain_vs_iteration.text(0.0, 1.03, rf'$\textbf{{(a)}}$', transform=gs_strain_vs_iteration.transAxes)

    # right part of the plot
    # numbers of grids points
    from muFFTTO import domain

    Nx = 64  # 3200 # 2 ** 7# 8
    Ny = Nx
    Nz = Nx
    it_max = 6
    domain_size = [1, 1 ]
    number_of_pixels = [Nx, Ny ]
    problem_type = 'elasticity'
    discretization_type = 'finite_element'
    element_type = 'linear_triangles'
    formulation = 'small_strain'
    my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                      problem_type=problem_type)

    discretization = domain.Discretization(cell=my_cell,
                                           nb_of_pixels_global=number_of_pixels,
                                           discretization_type=discretization_type,
                                           element_type=element_type)
    eq_strain_field = discretization.get_scalar_field(name='eq_strain_field')
    grad_eq_strain_field = discretization.get_gradient_of_scalar_field(name='grad_eq_strain_field')

    preconditioner_type = 'Green'
    data_folder_path = (file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}'
                        + f'_{preconditioner_type}' + '/')

    # first iteration
    plot_positions_ = [[0, 2], [0, 3], [0, 4],
                       [1, 2], [1, 3], [1, 4]]
    keep_ticks_x = [0, 0, 0, 1, 1, 1]
    keep_ticks_y = [1, 0, 0, 1, 0, 0]
    label_ = ['(b.1)', '(b.2)', '(b.3)', '(c.1)', '(c.2)', '(c.3)']
    plot_colorbar_ = [0, 0, 1, 0, 0, 0]
    iterations = np.array([0, 1, 2, 0, 1, 2])
    for index, iteration in enumerate(iterations):

        if index < 3:
            field_name_to_plot = f'strain_eq'
            results_name = (field_name_to_plot + f'_exp_{n_exp}_it{iteration}')
        else:
            field_name_to_plot = f'K4_ijklqyz'
            results_name = (field_name_to_plot + f'_exp_{n_exp}_it{iteration}')

        # results_name = (f'strain_eq' + f'_exp_{n_exp}_it{iteration_total}')
        values = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True, mmap_mode='r')
        # eq_strain_field.s[...] = values[...]
        # discretization.fft.communicate_ghosts(eq_strain_field)
        # discretization.apply_gradient_operator_mugrid(u_inxyz=eq_strain_field,
        #                                               grad_u_ijqxyz=grad_eq_strain_field)
        # inclusion_mask = np.ones([Nx, Ny, Nz], dtype=bool)
        # inclusion_mask[Nx // 4:3 * Nx // 4, Ny // 4:3 * Ny // 4, Nz // 4:3 * Nz // 4] = 0
        # averaged_over_pixel_field = grad_eq_strain_field.s[0].mean(axis=1)
        # grad_norm = np.sqrt(
        #     discretization.fft.communicator.sum(np.dot(averaged_over_pixel_field[:, inclusion_mask].ravel(),
        #                                                averaged_over_pixel_field[:, inclusion_mask].ravel())))
        # grad_max = np.sqrt(
        #     discretization.mpi_reduction.max(
        #         averaged_over_pixel_field[:, inclusion_mask] ** 2 + averaged_over_pixel_field[:, inclusion_mask] ** 2))
        # grad_max_inf = discretization.mpi_reduction.max(averaged_over_pixel_field[:, inclusion_mask])
        #
        # print(f'grad_norm={grad_norm}')
        # print(f'grad_max={grad_max}')
        # print(f'grad_max_inf={grad_max_inf}')

        displacement_name = (f'displacement_fluctuation_field' + f'_exp_{n_exp}_it{iteration}')
        displacement_fluctuation_field = np.load(data_folder_path + displacement_name + f'.npy', allow_pickle=True,
                                                 mmap_mode='r')
        results_name = (f'cube_' + f'dof={Nx}')  # prism_  bubbles_
        geom_folder_path = file_folder_path + '/exp_data/' + 'exp_paper_JG_nonlinear_elasticity_JZ_bubles_generate_geom/'
        geometry = np.load(geom_folder_path + results_name + f'.npy', allow_pickle=True, mmap_mode='r')

      #  z_cut = Nz // 2
        max_val = np.max(values)
        # max_val=1800

        if field_name_to_plot == 'strain_eq':
            arr_masked = np.where(np.isclose(values[... ], 0.), np.inf, values[... ])
            min_val = np.min(arr_masked)
            cbar_label_ = r'$\mathrm{\varepsilon}_{eq}$'
            title_ = r'$\mathrm{\varepsilon}_{eq}$'

        elif field_name_to_plot == 'K4_ijklqyz':
            arr_masked = np.where(np.isclose(values[...], 500.), -np.inf, values[...])
            m = np.max(arr_masked)
            min_val = np.min(values[... ])
            cbar_label_ = r'$\mathrm{C}_{66}/\mathrm{\mu^0}$'
            title_ = r'$\mathrm{C}_{66}/\mathrm{\mu^0}$'

        if field_name_to_plot == 'strain_eq':
            slice_ = values[... ]
        elif field_name_to_plot == 'K4_ijklqyz':
            slice_ = values[... ] / 0.1
        # norm = mpl.colors.LogNorm(vmin=min_val, vmax=max_val)

        print(f'max vals tot {np.max(values[...])}  max cut   {max_val}')
        print(f'min vals tot {np.min(values[...])}   min cut {min_val}')
        if field_name_to_plot == 'strain_eq':
            min_val = 0.00
            max_val = 0.01
        elif field_name_to_plot == 'K4_ijklqyz':
            min_val = 1.
            max_val = 300.

        norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)

        # Mask zeros
        masked = np.ma.masked_where(np.isclose(slice_, 0.0), slice_)

        # Choose colormap and set masked color to white
        cmap = mpl.cm.viridis.copy()
        cmap.set_bad(color='white')

        ax = fig.add_subplot(gs[plot_positions_[index][0], plot_positions_[index][1]])
        if index == 0:
            ax.text(0.2, 0.8, rf'$x_3=l_3/2$', transform=ax.transAxes)

        # Ny, Nx = displacement_fluctuation_field[0,..., z_cut].shape

        x, y = discretization.fft.coords[... ]
        # x_li = np.linspace(0, 1, Nx+1)
        # y_li = np.linspace(0, 1, Ny+1)
        #
        # x, y = np.meshgrid(x_li, y_li)

        gamma = 0.025  # e.g. 0.03
        # extended_displacement=np.zeros([2,Nx+1, Ny+1])

        # interior
        # extended_displacement[:, :-1, :-1] = displacement_fluctuation_field[:2, :, :, z_cut]
        # # last column = first column
        # extended_displacement[:, :-1, -1] = displacement_fluctuation_field[:2, :, 0, z_cut]
        # # last row = first row
        # extended_displacement[:, -1, :-1] = displacement_fluctuation_field[:2, 0, :, z_cut]
        # # bottom-right corner = (0,0)
        # extended_displacement[:, -1, -1] = displacement_fluctuation_field[:2, 0, 0, z_cut]

        # x_def = x +  (gamma * y + extended_displacement[0]) * 10 #
        # y_def = y +  (gamma * x + extended_displacement[1]) * 10#
        x_def = x + (gamma * y + displacement_fluctuation_field[0, ...  ]) * 5  #
        y_def = y + (gamma * x + displacement_fluctuation_field[1, ... ]) * 5  #
        pcm = ax.pcolormesh(x_def, y_def, masked, norm=norm, cmap=cmap,  # shading='flat',
                            rasterized=True)

        if keep_ticks_x[index]:
            ax.set_xticks([0, 0.5, 1])
            ax.set_xticklabels(['1', str(0.5), str(1)])
            # Axis labels
            ax.set_xlabel(r'$x_1/l_1$', labelpad=3)
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])

        if keep_ticks_y[index]:
            ax.yaxis.tick_left()
            ax.yaxis.set_label_position("left")  # optional, for the ylabel

            ax.set_yticks([0, 0.5, 1])
            ax.set_yticklabels(['1', str(0.5), str(1)])
            ax.set_ylabel(r'$x_2/l_2$', labelpad=3)
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])

            # Title and label
        ax.text(0.0, 1.05, rf'\textbf{{{label_[index]}}}', transform=ax.transAxes)
        ax.text(0.5, 1.05, fr'$i={iteration}$', transform=ax.transAxes)

        ax.yaxis.set_label_position("left")
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(False)
        # keep bottom spine (x-axis)
        ax.spines['bottom'].set_visible(True)

        if field_name_to_plot == 'strain_eq':
            ax_cbar = fig.add_subplot(gs[0, -1])

            cbar = plt.colorbar(pcm, location='left', cax=ax_cbar)
            ax_cbar.tick_params(right=True, top=False, labelright=False, labeltop=False, labelrotation=0)
            cbar.ax.yaxis.set_ticks_position('right')
            cbar.ax.yaxis.set_label_position('right')
            ax_cbar.set_ylabel(cbar_label_)

            cbar.set_ticks([min_val, (max_val + min_val) / 2, max_val])
            cbar.set_ticklabels([f'{min_val:.2f}', f'{(max_val + min_val) / 2:.2f}', f'{max_val:.2f}'])

        elif field_name_to_plot == 'K4_ijklqyz':
            ax_cbar = fig.add_subplot(gs[1, -1])

            cbar = plt.colorbar(pcm, location='left', cax=ax_cbar)
            ax_cbar.tick_params(right=True, top=False, labelright=False, labeltop=False, labelrotation=0)
            cbar.ax.yaxis.set_ticks_position('right')
            cbar.ax.yaxis.set_label_position('right')
            ax_cbar.set_ylabel(cbar_label_)

            cbar.set_ticks([min_val, (max_val + min_val) / 2, max_val])
            cbar.set_ticklabels([f'{min_val:.0f}', f'{(max_val + min_val) / 2:.0f}', f'{max_val:.0f}'])
    add_3D_plot = False
    if add_3D_plot:
        # Plot
        ax = fig.add_subplot(gs[1, 0], projection='3d')
        # ax.set_position([0.0, 0.0, 1.0, 1.0])
        #  pos = ax.get_position()
        #  ax.set_position([pos.x0 - 0.05, pos.y0 - 0.05,
        #                   pos.width + 0.10, pos.height + 0.10])

        # fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

        ax.view_init(elev=30, azim=-50)
        white = [1.0, 1.0, 1.0, 1.0]
        gray = [0.6, 0.6, 0.6, 1.0]
        dark_blue = [0.0, 0.0, 0.75, 1.0]

        # Assume geometry.shape == (32, 32, 32)
        Nx, Ny, Nz = geometry.shape

        # Boolean voxel mask
        voxels = np.zeros_like(geometry, dtype=bool)

        # 1) Bottom half filled (z = 0..15)
        voxels[:, :, :Nz // 2] = True  # 32 x 32 x 16

        # 2) Blue cube 16x16x16 starting at (8, 8, 8)
        x0, x1 = 8, 24  # 24 - 8 = 16
        y0, y1 = 8, 24
        z0, z1 = 8, 24
        voxels[x0:x1, y0:y1, z0:z1] = True

        # Colors (RGBA)
        colors = np.zeros(voxels.shape + (4,), dtype=float)

        # Bottom half: red
        colors[:, :, :Nz // 2] = dark_blue

        # Blue cube overrides inside that region
        colors[x0:x1, y0:y1, z0:z1] = gray

        # Plot
        step = 4  # or 8 for even cleaner visualization

        vox_vis = voxels[::step, ::step, ::step]
        col_vis = colors[::step, ::step, ::step]
        ax.voxels(
            vox_vis,
            facecolors=col_vis,
            edgecolor='k',
            linewidth=0.4
        )

        # ax.voxels(voxels, facecolors=colors, edgecolor='k', linewidth=0.3,shade=False)
        ax.set_xlim(0, Nx / step)
        ax.set_ylim(0, Ny / step)
        ax.set_zlim(0, Nz / step)
        # ax.set_box_aspect((Nx, Ny, Nz))  # keep cubes looking like cubes
        ax.set_box_aspect((Nx / step, Ny / step, Nz / step))

        add_labels_to_3D_plot = False
        if add_labels_to_3D_plot:
            ax.set_xlabel(r'$x_1/l_1$')
            ax.set_ylabel(r'$x_2/l_2$')
            ax.set_zlabel(r'$x_3/l_3$')

            # X ticks
            ax.set_xticks([0, Nx / step / 2, Nx / step - 1])
            ax.set_xticklabels(['0', '0.5', '1'])

            # Y ticks
            ax.set_yticks([0, Ny / step / 2, Ny / step - 1])
            ax.set_yticklabels(['0', '0.5', '1'])

            # Z ticks
            ax.set_zticks([0, Nz / step / 2, Nz / step - 1])
            ax.set_zticklabels(['0', '0.5', '1'])
            ax.zaxis._axinfo['tick']['inward_factor'] = 0
            ax.zaxis._axinfo['tick']['outward_factor'] = 1.
            ax.zaxis._axinfo['juggled'] = (1, 2, 0)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_zlabel('')

        # Remove background panes
        ax.xaxis.pane.set_visible(False)
        ax.yaxis.pane.set_visible(False)
        ax.zaxis.pane.set_visible(False)

        # Remove grid lines
        ax.grid(False)
        import matplotlib.patches as mpatches

        stiff_patch = mpatches.Patch(color=dark_blue, label='Stiff – nonlinear')
        compliant_patch = mpatches.Patch(color=gray, label='Compliant – linear')

        ax.legend(
            handles=[stiff_patch, compliant_patch],
            bbox_to_anchor=(0.0, 1.5),  # (x, y) in figure coordinates
            loc='upper left',
            frameon=False
        )
    fig.tight_layout()
    fname = f'fig_2Dstain_{Nx}_exp_{n_exp}'
    plt.savefig(figure_folder_path + script_name + fname + '.pdf', dpi=1200, bbox_inches='tight')
    plt.savefig(figure_folder_path + script_name + fname + '.png', dpi=300, bbox_inches='tight')

    # plt.savefig(figure_folder_path + script_name + fname, format='pdf', dpi=1200, bbox_inches='tight', pad_inches=0.1)
    # plt.savefig(figure_folder_path + script_name + fname, format='png', dpi=1200, bbox_inches='tight', pad_inches=0.1)

    print(('create figure: {}'.format(figure_folder_path + script_name + fname)))
    plt.show()

import numpy as np
import pyvista as pv

# -------------------------------------------------------
# Example voxel geometry (replace with your own 3D array)
# -------------------------------------------------------
nx, ny, nz = 100, 100, 100
voxels = np.zeros((nx, ny, nz), dtype=np.uint8)

# inner cube
voxels[30:70, 30:70, 30:70] = 1

# -------------------------------------------------------
# Convert to PyVista UniformGrid
# -------------------------------------------------------
grid = pv.UniformGrid()
grid.dimensions = np.array(voxels.shape) + 1
grid.origin = (0, 0, 0)
grid.spacing = (1, 1, 1)
grid.cell_data["phase"] = voxels.flatten(order="F")

# -------------------------------------------------------
# Create a plotter
# -------------------------------------------------------
p = pv.Plotter()

# Phase 0 (matrix)
phase0 = grid.threshold(value=0.5, invert=True)
p.add_mesh(
    phase0,
    color="lightgray",
    opacity=0.15,
    show_edges=False,
)

# Phase 1 (inclusion)
phase1 = grid.threshold(value=0.5)
p.add_mesh(
    phase1,
    color="royalblue",
    opacity=0.8,
    show_edges=False,
    smooth_shading=True,
)

# -------------------------------------------------------
# Final touches
# -------------------------------------------------------
p.add_axes()
p.show_grid()
p.camera_position = "iso"

p.show()

if plot_data_vs_CG_3D:
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

    Nx = 128  # 3200 # 2 ** 7# 8
    Ny = Nx
    Nz = Nx
    it_max = 6
    n_exponents = np.array([5])
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
                _info_final_G = np.load(data_folder_path + f'info_log_exp_{n_exp}_it{iteration_total}.npz',
                                        allow_pickle=True)

            its_G[iteration_total, j] = _info_final_G.f.nb_it_comb
            norm_rhs_G.append(_info_final_G.f.norm_rhs_field)
            norm_newrton_stop_G.append(_info_final_G.f.newton_stop_crit)
            info_log_final_G = np.load(data_folder_path + f'info_log_final_exp_{n_exp}.npz', allow_pickle=True)

            preconditioner_type = 'Green_Jacobi'

            data_folder_path = (
                    file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                    + f'_{preconditioner_type}' + '/')
            if iteration_total < it_max:
                _info_final_GJ = np.load(data_folder_path + f'info_log_exp_{n_exp}_it{iteration_total}.npz',
                                         allow_pickle=True)

            its_GJ[iteration_total, j] = _info_final_GJ.f.nb_it_comb
            norm_rhs_GJ.append(_info_final_GJ.f.norm_rhs_field)
            norm_newrton_stop_GJ.append(_info_final_GJ.f.newton_stop_crit)
            print(_info_final_G.f.norm_rr[0])

            print(_info_final_GJ.f.norm_rr[0])

            info_log_final_GJ = np.load(data_folder_path + f'info_log_final_exp_{n_exp}.npz', allow_pickle=True)

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

    fig = plt.figure(figsize=(8.3, 4.0))
    gs = fig.add_gridspec(2, 5, hspace=0.2, wspace=0.1, width_ratios=[1, 1, 1, 1, 0.05],
                          height_ratios=[1, 1.])

    # this was a plot of data
    # gs_global.plot(iterations, its_G[:, 0], 'g-', marker='x', label='Green')
    # gs_global.plot(iterations, its_GJ[:, 0], 'k-', marker='o', markerfacecolor='none', label='Green-Jacobi')
    #
    # gs_global.set_xlabel(r'Newton iteration -  $i$')
    # gs_global.set_ylabel(r'$\#$ of PCG iterations')
    # # gs_global.legend(loc='best')
    # gs_global.set_xlim(-0.05, iterations[-1] + .05)
    # # gs_global.set_ylim(0., 150)
    # gs_global.set_xticks(iterations)
    #
    # gs_global.annotate(text=f'Green-Jacobi',  # \n contrast = 100
    #                    xy=(iterations[1], its_GJ[1]),
    #                    xytext=(3., 200.),
    #                    arrowprops=dict(arrowstyle='->',
    #                                    color='Black',
    #                                    lw=1,
    #                                    ls='-'),
    #                    fontsize=11,
    #                    color='Black',
    #                    )
    # gs_global.annotate(text=f'Green',  # \n contrast = 100
    #                    xy=(iterations[2], its_G[2]),
    #                    xytext=(3, 400.),
    #                    arrowprops=dict(arrowstyle='->',
    #                                    color='green',
    #                                    lw=1,
    #                                    ls='-'),
    #                    fontsize=11,
    #                    color='green',
    #                    )
    # gs_global.text(0.20, 0.95, r'$ \approx 24 \times 10^{6}$ DOFs ',
    #                transform=gs_global.transAxes)

    # plot mat data 0 Newton iteration
    ijkl = (0, 0, 0, 0)
    cut_to_plot = Nz // 2 - 1

    preconditioner_type = 'Green'
    data_folder_path = (file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                        + f'_{preconditioner_type}' + '/')

    # first iteration
    iteration_total = 1
    i = 0
    # field_name_to_plot = f'K4_ijklqyz'
    field_name_to_plot = f'strain_eq'

    results_name = (field_name_to_plot + f'_exp_{n_exp}_it{iteration_total}')
    # results_name = (f'strain_eq' + f'_exp_{n_exp}_it{iteration_total}')
    K4_xyz_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True, mmap_mode='r')

    results_name = (f'cube_' + f'dof={Nx}')  # prism_  bubbles_
    geom_folder_path = file_folder_path + '/exp_data/' + 'exp_paper_JG_nonlinear_elasticity_JZ_bubles_generate_geom/'
    geometry = np.load(geom_folder_path + results_name + f'.npy', allow_pickle=True, mmap_mode='r')

    # Normalize values
    # idx = np.unravel_index(np.argmax(values), values.shape)
    values = np.copy(K4_xyz_G) / K
    # Calculate normalization parameter

    flat_idx = np.argmax(values)
    # i, j, z_cut = np.unravel_index(flat_idx, values.shape)
    z_cut = Nz // 2
    max_val = np.max(values[..., z_cut])

    if field_name_to_plot == 'strain_eq':
        arr_masked = np.where(np.isclose(values[..., z_cut], 0.), np.inf, values[..., z_cut])
        min_val = np.min(arr_masked)
        cbar_label_ = r'$\mathrm{E}_{eq}$'
    elif field_name_to_plot == 'K4_ijklqyz':
        arr_masked = np.where(np.isclose(values[..., z_cut], 500.), -np.inf, values[..., z_cut])
        m = np.max(arr_masked)
        min_val = np.min(values[..., z_cut])
        cbar_label_ = r'$\mathrm{C}_{11}/\mathrm{K}$'

    # min_val = np.min(values[...,z_cut])
    max_K = max_val  # values.max()/10#10  #
    min_K = min_val
    mid_K = (max_val + min_val) / 2  # values.mean() #1.66  # max_K/2#

    # Set up colormap and normalization
    norm = mpl.colors.Normalize(vmin=min_K, vmax=max_K)


    # norm = mpl.colors.TwoSlopeNorm(vmin=min_K, vcenter=mid_K, vmax=max_K)
    # norm = mpl.colors.LogNorm(vmin=min_K , vmax=max_K)

    def plot_voxels_colormap_2D(fig, gs_position, K4_xyz_G, K, Nz, iteration_total,
                                cmap='cividis_r', z_cut=0,
                                label='(b.2)', cbar_gs_position=None, norm=norm,
                                cbar_label=r'$\mathrm{C}_{11}/\mathrm{K}$',
                                keep_ticks=False):
        """
        Plot 3D voxels with colormap based on continuous values.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure object
        gs_position : tuple or GridSpec index
            Position in gridspec, e.g., gs[0, 2]
        K4_xyz_G : ndarray, shape [Nx, Ny, Nz]
            3D array of values to plot
        K : float
            Normalization constant
        Nz : int
            Size in z direction
        iteration_total : int
            Iteration number for title
        cmap : str or Colormap
            Colormap to use (default: 'cividis')
        cutaway_type : str
            Type of cutaway: 'half_z', 'quarter', 'none'
        label : str
            Subplot label, e.g., '(b.2)'
        cbar_gs_position : GridSpec index or None
            Position for colorbar, e.g., gs[0, 4]. If None, no colorbar is added.
        cbar_label : str
            Label for colorbar

        Returns
        -------
        ax : Axes3D
            The 3D axes object
        cbar : Colorbar or None
            The colorbar object (if cbar_gs_position is provided)
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import numpy as np

        ax = fig.add_subplot(gs_position)  # , projection='3d'

        # Normalize values
        values = np.copy(K4_xyz_G[:, :, z_cut]) / K  # transpose()

        # Get dimensions
        nx, ny = values.shape

        ax.dist = 2  # Lower value = closer/larger plot (default is ~10)

        # Remove padding around axes
        ax.margins(0)

        # Tighten the axis bounds
        ax.autoscale_view(tight=True)

        cmap_obj = mpl.colormaps.get_cmap(cmap) if isinstance(cmap, str) else cmap

        # Create mask for cutaway (True = show, False = hide)
        mask_cut = np.ones(values.shape, dtype=bool)

        # Convert values to colors using colormap
        colors = cmap_obj(norm(values))  # Returns RGBA array [Nx, Ny, Nz, 4]

        # Plot voxels with colormap colors
        # ax.imshow(values[:,:, z_cut])
        ax.pcolormesh(values, cmap=cmap_obj, norm=norm)

        # Axis limits
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        # ax.set_zlim(0, nz)

        # Ticks: First, Middle, Last
        if keep_ticks:
            ax.set_xticks([0, nx // 2, nx])
            ax.set_yticks([0, ny // 2, ny])

            ax.set_xticklabels(['1', str(nx // 2), str(nx)])
            ax.set_yticklabels(['1', str(ny // 2), str(ny)])

            # Axis labels
            ax.set_xlabel(r'$x_1$', labelpad=3)
            ax.set_ylabel(r'$x_2$', labelpad=3)

        else:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        # Title and label
        ax.set_title(fr'$i={iteration_total}$')
        if label:
            ax.text(0.0, 1.05, rf'\textbf{{{label}}}', transform=ax.transAxes)

        # Add colorbar if position is provided
        cbar = None
        if cbar_gs_position is not None:
            ax_cbar = fig.add_subplot(cbar_gs_position)
            sm = mpl.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
            sm.set_array([])

            cbar = plt.colorbar(sm, location='left', cax=ax_cbar)
            cbar.set_ticks([min_K, mid_K, max_K])
            # cbar.set_ticklabels([f'{min_K:.1f}', f'{mid_K:.1f}', f'{max_K:.1f}'])
            ax_cbar.tick_params(right=True, top=False, labelright=False, labeltop=False, labelrotation=0)
            cbar.ax.yaxis.set_ticks_position('right')
            cbar.ax.yaxis.set_label_position('right')
            ax_cbar.set_ylabel(cbar_label)
            # from matplotlib.ticker import ScalarFormatter

            # fmt = ScalarFormatter()
            # # fmt.set_powerlimits((-3, 3))  # force scientific notation outside this range
            #
            # cbar.ax.yaxis.set_major_formatter(fmt)
            # cbar.set_ticks([min_K, mid_K, max_K])
            # cbar.set_ticklabels([
            #     fr'$10^{{{int(np.log10(min_K))}}}$',
            #     fr'$10^{{{int(np.log10(mid_K))}}}$',
            #     fr'$10^{{{int(np.log10(max_K))}}}$'
            # ])

        return ax, cbar, cmap


    # gs_global = fig.add_subplot(gs[:, 0])
    values = geometry[..., z_cut]
    #  PLot geometry
    ax_geom_0 = plot_voxels_colormap_2D(
        fig=fig,
        gs_position=gs[0, 0],
        K4_xyz_G=geometry,
        cmap='Grays',
        K=K,
        Nz=Nz,
        iteration_total=iteration_total,
        z_cut=z_cut,
        label='(b.1)',
        norm=mpl.colors.Normalize(vmin=0, vmax=1),  # norm,
        # cbar_gs_position=gs[0, 4],
        # cbar_label=r'$\mathrm{C}_{11}/\mathrm{K}$'
    )[0]
    # ===== USAGE EXAMPLE =====

    # Call the function
    ax_geom_0, cbar, cmap_ = plot_voxels_colormap_2D(
        fig=fig,
        gs_position=gs[0, 2],
        K4_xyz_G=K4_xyz_G,
        K=K,
        Nz=Nz,
        iteration_total=iteration_total,
        z_cut=z_cut,
        label='(b.2)',
        norm=norm,
        cbar_gs_position=gs[0, 4],
        cbar_label=cbar_label_)

    # gs[0,0].text(-2.3, 1.05, rf'\textbf{{(a)}}', transform=ax_geom_0.transAxes)
    # ax.text2D(0.0, 1.05, rf'\textbf{{{label}}}', transform=ax.transAxes)
    # Add additional text if needed
    # gs_global.text(-2.3, 1.05, rf'\textbf{{(a)}}', transform=ax_geom_0.transAxes)

    # ----------------
    iteration_total = 0
    # results_name = (f'K4_ijklqyz' + f'_exp_{n_exp}_it{iteration_total}')
    results_name = (field_name_to_plot + f'_exp_{n_exp}_it{iteration_total}')

    del K4_xyz_G
    K4_xyz_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True, mmap_mode='r')

    #  values = K4_xyz_G[..., cut_to_plot] / K
    # values = geometry[..., z_cut]
    # Call the function
    ax_geom_0 = plot_voxels_colormap_2D(
        fig=fig,
        gs_position=gs[0, 1],
        K4_xyz_G=K4_xyz_G,
        K=K,
        Nz=Nz,
        iteration_total=iteration_total,
        z_cut=z_cut,
        label='(b.1)',
        # norm=norm,#mpl.colors.Normalize(vmin=0, vmax=1),  #
        # cbar_gs_position=gs[0, 4],
        # cbar_label=r'$\mathrm{C}_{11}/\mathrm{K}$'
    )[0]
    # ----------------

    # for iteration_total in 6:
    iteration_total = 2
    # iteration_total = 2
    del K4_xyz_G
    # results_name = (f'K4_ijklqyz' + f'_exp_{n_exp}_it{iteration_total}')
    results_name = (field_name_to_plot + f'_exp_{n_exp}_it{iteration_total}')

    K4_xyz_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True, mmap_mode='r')
    # K4_to_plot_G = K4_xyz_G[..., cut_to_plot]  # K4_xyz_G[i,0,0,0, ..., cut_to_plot]

    # ax_geom_0 = fig.add_subplot(gs[0, 3])
    # Call the function
    ax_geom_0 = plot_voxels_colormap_2D(
        fig=fig,
        gs_position=gs[0, 3],
        K4_xyz_G=K4_xyz_G,
        K=K,
        Nz=Nz,
        iteration_total=iteration_total,
        z_cut=z_cut,
        label='(b.3)',
        #  norm=norm,
        # cbar_gs_position=gs[0, 4],
        # cbar_label=r'$\mathrm{C}_{11}/\mathrm{K}$'
    )[0]
    # ----------------
    # for iteration_total in 6:
    iteration_total = 3
    # iteration_total = 2
    # results_name = (f'K4_ijklqyz' + f'_exp_{n_exp}_it{iteration_total}')
    results_name = (field_name_to_plot + f'_exp_{n_exp}_it{iteration_total}')

    del K4_xyz_G
    K4_xyz_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True, mmap_mode='r')
    # K4_to_plot_G = K4_xyz_G[..., cut_to_plot]  # K4_xyz_G[i,0,0,0, ..., cut_to_plot]
    # ax_geom_0 = fig.add_axes([0.7, 0.75, 0.2, 0.2])
    ax_geom_0 = plot_voxels_colormap_2D(
        fig=fig,
        gs_position=gs[1, 1],
        K4_xyz_G=K4_xyz_G,
        K=K,
        Nz=Nz,
        iteration_total=iteration_total,
        z_cut=z_cut,
        label=f'(b.{iteration_total})',
        norm=norm,
        # cbar_gs_position=gs[0, 4],
        # cbar_label=r'$\mathrm{C}_{11}/\mathrm{K}$'
    )[0]

    iteration_total = 4
    # iteration_total = 2
    # results_name = (f'K4_ijklqyz' + f'_exp_{n_exp}_it{iteration_total}')
    results_name = (field_name_to_plot + f'_exp_{n_exp}_it{iteration_total}')

    del K4_xyz_G
    K4_xyz_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True, mmap_mode='r')

    ax_geom_0 = plot_voxels_colormap_2D(
        fig=fig,
        gs_position=gs[1, 2],
        K4_xyz_G=K4_xyz_G,
        K=K,
        Nz=Nz,
        iteration_total=iteration_total,
        z_cut=z_cut,
        label=f'(b.{5})',
        norm=norm,
        # cbar_gs_position=gs[0, 4],
        # cbar_label=r'$\mathrm{C}_{11}/\mathrm{K}$'
    )[0]

    iteration_total = 5
    # iteration_total = 2
    #    results_name = (f'K4_ijklqyz' + f'_exp_{n_exp}_it{iteration_total}')
    results_name = (field_name_to_plot + f'_exp_{n_exp}_it{iteration_total}')

    del K4_xyz_G
    K4_xyz_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True, mmap_mode='r')
    ax_geom_0 = plot_voxels_colormap_2D(
        fig=fig,
        gs_position=gs[1, 3],
        K4_xyz_G=K4_xyz_G,
        K=K,
        Nz=Nz,
        iteration_total=iteration_total,
        z_cut=z_cut,
        label=f'(b.{6})',
        norm=norm,
        keep_ticks=True,
        # cbar_gs_position=gs[0, 4],
        # cbar_label=r'$\mathrm{C}_{11}/\mathrm{K}$'
    )[0]
    # ax_geom_0.yaxis.set_ticks_position('right')
    # ax_geom_0.yaxis.set_label_position('right')

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
    fname = f'fig_3D_{Nx}_exp_{n_exponents[0]}' + '{}'.format('.pdf')
    plt.savefig(figure_folder_path + script_name + fname, bbox_inches='tight')
    print(('create figure: {}'.format(figure_folder_path + script_name + fname)))
    plt.savefig('fig_3D.pdf', format='pdf', dpi=1200, bbox_inches='tight', pad_inches=0.1)
    plt.savefig('fig_3D.png', format='png', dpi=1200, bbox_inches='tight', pad_inches=0.1)
    plt.show()
