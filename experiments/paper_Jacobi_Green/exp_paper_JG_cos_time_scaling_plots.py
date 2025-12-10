import time
import os
import sys
import argparse
import sys

sys.path.append("/home/martin/Programming/muFFTTO_paralellFFT_test/muFFTTO")
sys.path.append('../..')  # Add parent directory to path

from mpi4py import MPI
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library

# Enable LaTeX rendering
plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX
    # "font.family": "helvetica",  # Use a serif font
})
plt.rcParams.update({'font.size': 11})
plt.rcParams["font.family"] = "Arial"
plot_time_single_line = True
plot_time_table = True

if plot_time_table:
    script_name = 'exp_paper_JG_cos'

    file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
    data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
    figure_folder_path = file_folder_path + '/figures/' + script_name + '/'

    nb_pix_multips = 10
    norm_ = 'norm_rr'

    total_phase_contrast = 4
    # 'Jacobi'  # 'Green'  # 'Green_Jacobi'
    nb_it_Green_linear_1 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])
    nb_it_Green_linear_4 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])
    nb_it_Jacobi_linear_1 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])
    nb_it_Jacobi_linear_4 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])
    nb_it_Green_Jacobi_linear_1 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])
    nb_it_Green_Jacobi_linear_4 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])

    time_G_1 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])
    time_J_1 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])
    time_GJ_1 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])

    time_CG_G_1 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])
    time_CG_J_1 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])
    time_CG_GJ_1 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])

    for j in np.arange(2, nb_pix_multips + 1):
        number_of_pixels = 2 ** j
        # print('j=', j)
        for i in np.arange(2, j + 1):
            # print('i=', i)
            nb_laminates = 2 ** i

            preconditioner_type = 'Green'
            results_name = (
                    f'nb_nodes_{number_of_pixels}_' + f'nb_pixels_{nb_laminates}_' + f'contrast_{total_phase_contrast}_' + f'prec_{preconditioner_type}' + f'{norm_}')
            info = np.load(data_folder_path + results_name + f'.npz', allow_pickle=True)
            nb_it_Green_linear_1[i - 2, j - 2] = info.f.nb_steps
            try:
                time_G_1[i - 2, j - 2] = info.f.elapsed_time
                time_CG_G_1[i - 2, j - 2] = info.f.elapsed_time_CG

            except AttributeError:
                time_G_1[i - 2, j - 2] = 0
                time_CG_G_1[i - 2, j - 2] = 0

            preconditioner_type = 'Jacobi'
            results_name = (
                    f'nb_nodes_{number_of_pixels}_' + f'nb_pixels_{nb_laminates}_' + f'contrast_{total_phase_contrast}_' + f'prec_{preconditioner_type}' + f'{norm_}')
            info = np.load(data_folder_path + results_name + f'.npz', allow_pickle=True)
            nb_it_Jacobi_linear_1[i - 2, j - 2] = info.f.nb_steps
            try:
                time_J_1[i - 2, j - 2] = info.f.elapsed_time
                time_CG_J_1[i - 2, j - 2] = info.f.elapsed_time_CG

            except AttributeError:
                time_J_1[i - 2, j - 2] = 0
                time_CG_J_1[i - 2, j - 2] = 0

            # nb_it_Jacobi_linear_1[i - 2, j - 2] = info.f.nb_steps

            preconditioner_type = 'Green_Jacobi'
            results_name = (
                    f'nb_nodes_{number_of_pixels}_' + f'nb_pixels_{nb_laminates}_' + f'contrast_{total_phase_contrast}_' + f'prec_{preconditioner_type}' + f'{norm_}')
            info = np.load(data_folder_path + results_name + f'.npz', allow_pickle=True)

            nb_it_Green_Jacobi_linear_1[i - 2, j - 2] = info.f.nb_steps
            try:
                time_GJ_1[i - 2, j - 2] = info.f.elapsed_time
                time_CG_GJ_1[i - 2, j - 2] = info.f.elapsed_time_CG

            except AttributeError:
                time_GJ_1[i - 2, j - 2] = 0
                time_CG_GJ_1[i - 2, j - 2] = 0

    nb_it_Green_linear_1 = nb_it_Green_linear_1
    nb_it_Green_linear_4 = nb_it_Green_linear_4
    nb_it_Jacobi_linear_1 = nb_it_Jacobi_linear_1
    nb_it_Jacobi_linear_4 = nb_it_Jacobi_linear_4
    nb_it_Green_Jacobi_linear_1 = nb_it_Green_Jacobi_linear_1
    nb_it_Green_Jacobi_linear_4 = nb_it_Green_Jacobi_linear_4

    time_G_1 = time_G_1
    time_CG_G_1 = time_CG_G_1
    time_J_1 = time_J_1
    time_CG_J_1 = time_CG_J_1

    time_GJ_1 = time_GJ_1
    time_CG_GJ_1 = time_CG_GJ_1

    nb_pixels = 3
    Ns = 2 ** np.arange(nb_pixels, 11)

    fig = plt.figure(figsize=(4.0, 4.0))
    gs = fig.add_gridspec(1, 1, hspace=0.0, wspace=0.1, width_ratios=[1],
                          height_ratios=[1])
    ax_time_per_it = fig.add_subplot(gs[:, :])
    ax_time_per_it.text(-0.22, 0.99, rf'\textbf{{(a) }}', transform=ax_time_per_it.transAxes)

    nb_dofs = Ns ** 2

    # scaled_time_G = time_G_1 / time_G_1[1]
    # scaled_time_GJ = time_GJ_1 / time_G_1[1]

    # scaled_time_CG_G = time_CG_G_1 / time_G_1[1]
    # scaled_time_CG_GJ = time_CG_GJ_1 / time_G_1[1]

    time_per_iteration_G_1 = time_G_1 / nb_it_Green_linear_1
    time_per_iteration_J_1 = time_J_1 / nb_it_Jacobi_linear_1
    time_per_iteration_GJ_1 = time_GJ_1 / nb_it_Green_Jacobi_linear_1

    time_per_iteration_CG_G_1 = time_CG_G_1 / nb_it_Green_linear_1
    time_per_iteration_CG_J_1 = time_CG_J_1 / nb_it_Jacobi_linear_1
    time_per_iteration_CG_GJ_1 = time_CG_GJ_1 / nb_it_Green_Jacobi_linear_1

    relative_tpi_J = time_per_iteration_J_1 / time_per_iteration_G_1
    relative_tpi_GJ = time_per_iteration_GJ_1 / time_per_iteration_G_1

    relative_tpi_CG_J = time_per_iteration_CG_J_1 / time_per_iteration_CG_G_1
    relative_tpi_CG_GJ = time_per_iteration_CG_GJ_1 / time_per_iteration_CG_G_1

    nb_pix_multips = [2, 3, 4, 5, 6, 7, 8, 9, 10]  # , 8, 9, 10
    Nx = (np.asarray(nb_pix_multips))
    X, Y = np.meshgrid(Nx, Nx, indexing='ij')
    #
    #   nb_pix_multips = [2, 4, 5, 6, 7, 8]
    # material distribution
    geometry_ID = 'cos'  # linear  # 'abs_val' sine_wave_   ,laminate_log  geometry_ID = 'right_cluster_x3'  # laminate2       # 'abs_val' sine_wave_   ,laminate_log
    # rhs = 'sin_wave'
    linestyles = ['-', '--', ':', '-.', '--', ':', '-.']
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'orange', 'purple']

    fig = plt.figure(figsize=(6.5, 5.5))  # 11, 7.0
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.24, width_ratios=[1.2, 1.2, 0.03],
                          height_ratios=[1, 1])
    row = 0
    for phase_contrast in [4, 4]:  # 2, 4 for sine wave
        ratio = phase_contrast

        # divnorm = mpl.colors.Normalize(vmin=0, vmax=100)
        divnorm = mpl.colors.TwoSlopeNorm(vmin=-30, vcenter=1, vmax=30)
        white_lim = 500

        # jacobi  graph
        gs1 = gs[row, 0].subgridspec(1, 1, wspace=0.1, width_ratios=[5])
        ax = fig.add_subplot(gs1[0, 0])
        #    ax.set_aspect('equal')
        if row == 0:
            nb_iterations = np.transpose(relative_tpi_J)
        else:
            nb_iterations = np.transpose(relative_tpi_CG_J)
        # nb_iterations = np.nan_to_num(nb_iterations, nan=1.0)

        nb_iterations = nb_iterations * 100 - 100

        for i in range(nb_iterations.shape[0]):
            for j in range(nb_iterations.shape[1]):
                if nb_iterations[i, j] == 1:
                    pass
                elif nb_iterations[i, j] < white_lim:
                    ax.text(i + Nx[0], j + Nx[0], f'{nb_iterations[i, j]:.0f}', size=8,
                            ha='center', va='center', color='black')
                elif nb_iterations[i, j] > 999:
                    ax.text(i + Nx[0], j + Nx[0], f'{999:.0f}', size=8,
                            ha='center', va='center', color='white')
                else:
                    ax.text(i + Nx[0], j + Nx[0], f'{nb_iterations[i, j]:.0f}', size=8,
                            ha='center', va='center', color='white')

        pcm = ax.pcolormesh(X, Y, nb_iterations, label='PCG: Green + Jacobi', cmap='RdYlGn_r', norm=divnorm)

        # ax.text(0.05, 0.92, f'Total phase contrast $\chi=10^{phase_contrast}$', transform=ax.transAxes)

        if row == 0:
            ax.text(0.05, 0.82, f'Total time\n per iteration', transform=ax.transAxes)

            ax.set_title('Jacobi/Green' ,pad=10)

        # ax.set_zlim(1 ,100)
        # ax.set_ylabel('# data/geometry sampling points (x direction)')

        # ax.yaxis.set_label_position('right')
        # ax.yaxis.tick_right()
        ax.set_ylabel(r'\#  material pixels - $p$')
        if row == 1:
            ax.text(0.05, 0.82, f'PCG time\n per iteration', transform=ax.transAxes)

        ax.set_xlabel(r'\#  nodal points - $n$')  # ~of~$\mathcal{T}_n$

        ax.set_xticks(Nx)
        ax.set_xticklabels([f'$2^{{{i}}}$' for i in Nx])
        # ax2 = ax.twinx()
        ax.set_yticks(Nx)
        ax.set_yticklabels([f'$2^{{{i}}}$' for i in Nx])
        ax.tick_params(right=True, top=False, labelright=False, labeltop=False, labelrotation=0)
        #    ax.set_aspect('equal')

        # ax.set_zlabel('# CG iterations')
        if row == 0:
            ax.text(-0.20, 1.15, rf'\textbf{{(a.{row + 1}) }}', transform=ax.transAxes)
        elif row == 1:
            ax.text(-0.20, 1.1, rf'\textbf{{(a.{row + 1}) }}', transform=ax.transAxes)
        #ax.set_aspect('equal')




        # plot Green Jacobi
        gs2 = gs[row, 1].subgridspec(1, 1, wspace=0.1, width_ratios=[5])
        ax = fig.add_subplot(gs2[0, 0])

        if row == 0:
            nb_iterations = np.transpose(relative_tpi_GJ)
        else:
            nb_iterations = np.transpose(relative_tpi_CG_GJ)
        # nb_iterations = np.nan_to_num(nb_iterations, nan=1.0)

        nb_iterations = nb_iterations * 100 - 100

        for i in range(nb_iterations.shape[0]):
            for j in range(nb_iterations.shape[1]):
                if nb_iterations[i, j] == 1:
                    pass
                elif nb_iterations[i, j] < white_lim:
                    ax.text(i + Nx[0], j + Nx[0], f'{nb_iterations[i, j]:.0f}', size=8,
                            ha='center', va='center', color='black')
                else:
                    ax.text(i + Nx[0], j + Nx[0], f'{nb_iterations[i, j]:.0f}', size=8,
                            ha='center', va='center', color='white')
        # Replace NaN values with zero

        pcm = ax.pcolormesh(X, Y, nb_iterations, label='PCG: Green + Jacobi', cmap='RdYlGn_r', norm=divnorm)



        if row == 0:
            ax.text(0.05, 0.82, f'Total time\n per iteration', transform=ax.transAxes)
            ax.set_title(' Green-Jacobi/Green' ,pad=10)
        # ax.set_zlim(1 ,100)
        # ax.set_ylabel('# of material phases')

        # ax.yaxis.set_label_position('right')
        # ax.yaxis.tick_right()
        if row == 1:
            ax.text(0.05, 0.82, f'PCG time\n per iteration', transform=ax.transAxes)
        ax.set_xlabel(r'\#  nodal points - $n$')  # ~of~$\mathcal{T}_n$

        ax.set_xticks(Nx)
        ax.set_xticklabels([f'$2^{{{i}}}$' for i in Nx])
        # ax2 = ax.twinx()
        ax.set_yticks(Nx)
        ax.set_yticklabels([f'$2^{{{i}}}$' for i in Nx])
        ax.tick_params(right=True, top=False, labelright=False, labeltop=False, labelrotation=0)
        #   ax.set_aspect('equal')

        if row == 0:
            ax.text(-0.14, 1.15, rf'\textbf{{(b.{row + 1}) }}', transform=ax.transAxes)
        elif row == 1:
            ax.text(-0.14, 1.1, rf'\textbf{{(b.{row + 1}) }}', transform=ax.transAxes)
        # Adding a color bar with custom ticks and labels
        cbar_ax = fig.add_subplot(gs[row, 2])
        cbar = plt.colorbar(pcm, location='left', cax=cbar_ax, ticklocation='right')  # Specify the ticks
        cbar.set_label(r'Percentage $(\%)$', rotation=90, labelpad=20)
        #ax.set_aspect('equal')

        row += 1

    fname = figure_folder_path + 'JG_exp4_GRID_DEP_eom_{}_rho_{}_norm_{}{}'.format('nlaminate', phase_contrast,
                                                                                   norm_, '.pdf')
    print(('create figure: {}'.format(fname)))
    plt.savefig(fname, bbox_inches='tight')

plt.show()

plt.rcParams.update({'font.size': 13})
plt.rcParams["font.family"] = "Arial"
if plot_time_single_line:
    script_name = 'exp_paper_JG_cos'

    file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
    data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
    figure_folder_path = file_folder_path + '/figures/' + script_name + '/'

    nb_pix_multips = 10

    total_phase_contrast = 4
    # 'Jacobi'  # 'Green'  # 'Green_Jacobi'
    nb_it_Green_linear_1 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])
    nb_it_Green_linear_4 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])
    nb_it_Jacobi_linear_1 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])
    nb_it_Jacobi_linear_4 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])
    nb_it_Green_Jacobi_linear_1 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])
    nb_it_Green_Jacobi_linear_4 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])

    time_G_1 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])
    time_J_1 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])
    time_GJ_1 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])

    time_CG_G_1 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])
    time_CG_J_1 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])
    time_CG_GJ_1 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])

    for j in np.arange(3, nb_pix_multips + 1):
        number_of_pixels = 2 ** j
        # print('j=', j)
        i = 3
        # print('i=', i)
        nb_laminates = 2 ** i

        preconditioner_type = 'Green'
        results_name = (
                f'nb_nodes_{number_of_pixels}_' + f'nb_pixels_{nb_laminates}_' + f'contrast_{total_phase_contrast}_' + f'prec_{preconditioner_type}')
        info = np.load(data_folder_path + results_name + f'.npz', allow_pickle=True)
        nb_it_Green_linear_1[i - 2, j - 2] = info.f.nb_steps
        try:
            time_G_1[i - 2, j - 2] = info.f.elapsed_time
            time_CG_G_1[i - 2, j - 2] = info.f.elapsed_time_CG

        except AttributeError:
            time_G_1[i - 2, j - 2] = 0
            time_CG_G_1[i - 2, j - 2] = 0

        preconditioner_type = 'Jacobi'
        results_name = (
                f'nb_nodes_{number_of_pixels}_' + f'nb_pixels_{nb_laminates}_' + f'contrast_{total_phase_contrast}_' + f'prec_{preconditioner_type}')
        info = np.load(data_folder_path + results_name + f'.npz', allow_pickle=True)
        try:
            time_J_1[i - 2, j - 2] = info.f.elapsed_time
            time_CG_J_1[i - 2, j - 2] = info.f.elapsed_time_CG

        except AttributeError:
            time_J_1[i - 2, j - 2] = 0
            time_CG_J_1[i - 2, j - 2] = 0

        nb_it_Jacobi_linear_1[i - 2, j - 2] = info.f.nb_steps

        preconditioner_type = 'Green_Jacobi'
        results_name = (
                f'nb_nodes_{number_of_pixels}_' + f'nb_pixels_{nb_laminates}_' + f'contrast_{total_phase_contrast}_' + f'prec_{preconditioner_type}')
        info = np.load(data_folder_path + results_name + f'.npz', allow_pickle=True)

        nb_it_Green_Jacobi_linear_1[i - 2, j - 2] = info.f.nb_steps
        try:
            time_GJ_1[i - 2, j - 2] = info.f.elapsed_time
            time_CG_GJ_1[i - 2, j - 2] = info.f.elapsed_time_CG

        except AttributeError:
            time_GJ_1[i - 2, j - 2] = 0
            time_CG_GJ_1[i - 2, j - 2] = 0

    nb_it_Green_linear_1 = nb_it_Green_linear_1[1]
    nb_it_Green_linear_4 = nb_it_Green_linear_4[1]
    nb_it_Jacobi_linear_1 = nb_it_Jacobi_linear_1[1]
    nb_it_Jacobi_linear_4 = nb_it_Jacobi_linear_4[1]
    nb_it_Green_Jacobi_linear_1 = nb_it_Green_Jacobi_linear_1[1]
    nb_it_Green_Jacobi_linear_4 = nb_it_Green_Jacobi_linear_4[1]

    time_G_1 = time_G_1[1] / 10
    time_CG_G_1 = time_CG_G_1[1] / 10
    time_J_1 = time_J_1[1] / 10
    time_CG_J_1 = time_CG_J_1[1] / 10

    time_GJ_1 = time_GJ_1[1] / 10
    time_CG_GJ_1 = time_CG_GJ_1[1] / 10

    plot_time = True
    if plot_time:
        nb_pixels = 3
        Ns = 2 ** np.arange(nb_pixels, 11)

        fig = plt.figure(figsize=(4.0, 4.0))
        gs = fig.add_gridspec(1, 1, hspace=0.0, wspace=0.1, width_ratios=[1],
                              height_ratios=[1])
        ax_time_per_it = fig.add_subplot(gs[:, :])
        ax_time_per_it.text(-0.22, 0.99, rf'\textbf{{(a) }}', transform=ax_time_per_it.transAxes)

        nb_dofs = Ns ** 2

        scaled_time_G = time_G_1 / time_G_1[1]
        scaled_time_GJ = time_GJ_1 / time_G_1[1]

        scaled_time_CG_G = time_CG_G_1 / time_G_1[1]
        scaled_time_CG_GJ = time_CG_GJ_1 / time_G_1[1]

        time_per_iteration_G_1 = time_G_1 / nb_it_Green_linear_1
        time_per_iteration_GJ_1 = time_GJ_1 / nb_it_Green_Jacobi_linear_1
        time_per_iteration_CG_G_1 = time_CG_G_1 / nb_it_Green_linear_1
        time_per_iteration_CG_GJ_1 = time_CG_GJ_1 / nb_it_Green_Jacobi_linear_1

        # scaling
        # line1, = ax_time_per_it.loglog(nb_dofs, nb_dofs * np.log(nb_dofs) / (nb_dofs[0] * np.log(nb_dofs[0])), ':',
        #                                color='Grey',
        #                                # /                        time_G_1[2:, 0][0]
        #                                label=r'Quasilinear - $ \mathcal{O} (N_{\mathrm{N}} \log  N_{\mathrm{N}}$)')

        plt.loglog(np.linspace(1e1, 1e8), 1e-3 * np.linspace(5e1, 1e8), 'k-', linewidth=0.9)

        line2, = plt.loglog(nb_dofs, scaled_time_G[1:], '-x', color='Green', label='Green')
        line3, = plt.loglog(nb_dofs, scaled_time_GJ[1:], 'k-', marker='o', markerfacecolor='none',
                            label='Green-Jacobi')
        legend1 = plt.legend(handles=[line2, line3], loc='upper left', title='Wall-clock time')  # line1,

        plt.gca().add_artist(legend1)  # Add the first legend manually

        #
        #  line4, = ax_time_per_it.loglog(nb_dofs, (scaled_time_G / nb_it_Green_linear_1)[1:], 'g-.x',
        #                                 label='Green')  # / time_per_iteration_G_1[2, 0]
        #  line5, = ax_time_per_it.loglog(nb_dofs, (scaled_time_GJ / nb_it_Green_Jacobi_linear_1)[1:], 'k-.',
        #                                 marker='o', markerfacecolor='none', label='Green-Jacobi')
        # #  Second legend (bottom right)
        #  plt.legend(handles=[line4, line5], loc='lower right',
        #             title=fr'$\frac{{\mbox{{Wall-clock time}}}}{{\rule{{0pt}}{{2.0ex}}\mbox{{\# of PCG iterations}}}}$')

        line7, = ax_time_per_it.loglog(nb_dofs, (scaled_time_CG_G / nb_it_Green_linear_1)[1:], 'g--x',  #
                                       label='Green')  # / time_per_iteration_G_1[2, 0]
        line8, = ax_time_per_it.loglog(nb_dofs, (scaled_time_CG_GJ / nb_it_Green_Jacobi_linear_1)[1:], 'k--',  #
                                       marker='o', markerfacecolor='none', label='Green-Jacobi')

        plt.legend(handles=[line7, line8], loc='lower right',
                   title=fr'$\frac{{\mbox{{PCG time}}}}{{\rule{{0pt}}{{2.0ex}}\mbox{{\# of PCG iterations}}}}$')

        # line7, = ax_time_per_it.loglog(nb_dofs, (scaled_time_G / (nb_it_Green_linear_1))[1:], 'g--^',
        #                                label='Green')  # / time_per_iteration_G_1[2, 0]
        # line8, = ax_time_per_it.loglog(nb_dofs, (scaled_time_GJ / (nb_it_Green_Jacobi_linear_1+8))[1:], 'k--^',
        #                                marker='o', markerfacecolor='none', label='Green-Jacobi')

        ax_time_per_it.set_xlabel(r'Discretization')
        ax_time_per_it.set_ylabel(f'Time / Time of Green at $\mathcal{{T}}_{{{8}}}$')
        ax_time_per_it.set_xlim([Ns[0] ** 2, Ns[-1] ** 2])
        ax_time_per_it.set_xticks(Ns ** 2)
        # ax_time_per_it.set_xticklabels([fr'${{2}}^{int(x)}$' for x in np.arange(nb_pixels, 11)])
        # ax_time_per_it.set_xticklabels([fr'$2^{{{int(x)}}}$' for x in np.arange(nb_pixels, 11)])
        ax_time_per_it.set_xticklabels([f'$\mathcal{{T}}_{{{int(x)}}}$' for x in Ns])

        # ax_time_per_it.set_xticks([1e3, 1e4, 1e5, 1e6, 1e6])
        ax_time_per_it.set_ylim([1e-2, 1e5])

        # plt.gca().set_xticks(iterations)

        #
        # ax_relative_time = fig.add_subplot(gs[0, :])
        # # line5, = ax_relative_time.loglog(nb_dofs, time_per_iteration_G_1[2:, 0] , 'g-.|', label='Green')# / time_per_iteration_G_1[2, 0]
        # line1, = ax_relative_time.semilogx(nb_dofs, nb_it_Green_linear_1[1:], 'g-s',
        #                                    marker='x', markerfacecolor='none', label='Green ')
        # line2, = ax_relative_time.semilogx(nb_dofs, 1 * nb_it_Green_Jacobi_linear_1[1:] ,
        #                                    'k-', marker='o', markerfacecolor='none', label='Green-Jacobi')
        # line3, = ax_relative_time.semilogx(nb_dofs, 1 * nb_it_Green_Jacobi_linear_1[1:]+8,
        #                                    'k--', marker='o', markerfacecolor='none', label='Green-Jacobi')
        # ax_relative_time.set_xlim([nb_dofs[0], nb_dofs[-1]])
        # ax_relative_time.set_xticks( [])
        # ax_relative_time.set_xticklabels([])
        #
        # ax_relative_time.set_ylim([0, nb_it_Green_Jacobi_linear_1[-1] ])
        # ax_relative_time.set_yticks( [0,20,40,60] )

        # ax_relative_time.set_ylim([80, 160])
        # Add horizontal line at y=5
        # ax_relative_time.axhline(y=1, color='grey', linestyle='--')

        fig.tight_layout()
        fname = f'time_scaling' + '{}'.format('.pdf')
        plt.savefig(figure_folder_path + script_name + fname, bbox_inches='tight')
        print(('create figure: {}'.format(figure_folder_path + script_name + fname)))

        plt.show()

        nb_pixels = 3
        Ns = 2 ** np.arange(nb_pixels, 11)

        fig = plt.figure(figsize=(4.0, 4.0))
        gs = fig.add_gridspec(1, 1, hspace=0.0, wspace=0.1, width_ratios=[1],
                              height_ratios=[1])
        ax_time_per_it = fig.add_subplot(gs[:, :])
        ax_time_per_it.text(-0.22, 0.99, rf'\textbf{{(a) }}', transform=ax_time_per_it.transAxes)

        nb_dofs = Ns ** 2

        scaled_time_G = time_G_1 / time_G_1[1]
        scaled_time_GJ = time_GJ_1 / time_G_1[1]

        scaled_time_CG_G = time_CG_G_1 / time_G_1[1]
        scaled_time_CG_GJ = time_CG_GJ_1 / time_G_1[1]

        time_per_iteration_G_1 = time_G_1 / nb_it_Green_linear_1
        time_per_iteration_GJ_1 = time_GJ_1 / nb_it_Green_Jacobi_linear_1
        time_per_iteration_CG_G_1 = time_CG_G_1 / nb_it_Green_linear_1
        time_per_iteration_CG_GJ_1 = time_CG_GJ_1 / nb_it_Green_Jacobi_linear_1

        # scaling

        # plt.loglog(np.linspace(1e1, 1e8), 1e-3 * np.linspace(5e1, 1e8), 'k-', linewidth=0.9)

        # line2, = plt.loglog(nb_dofs, scaled_time_G[1:], '-x', color='Green', label='Green')
        # line3, = plt.loglog(nb_dofs, scaled_time_GJ[1:], 'k-', marker='o', markerfacecolor='none',
        #                     label='Green-Jacobi')
        # legend1 = plt.legend(handles=[line2, line3], loc='upper left', title='Wall-clock time')  # line1,

        # plt.gca().add_artist(legend1)  # Add the first legend manually

        line7, = ax_time_per_it.semilogx(nb_dofs, (time_per_iteration_GJ_1 / time_per_iteration_G_1)[1:], 'g--x',  #
                                         label='Total time')

        line7, = ax_time_per_it.semilogx(nb_dofs, (time_per_iteration_CG_GJ_1 / time_per_iteration_CG_G_1)[1:], 'g--x',
                                         #
                                         label='PCG time')
        # / time_per_iteration_G_1[2, 0]
        # line8, = ax_time_per_it.loglog(nb_dofs, (scaled_time_CG_GJ  )[1:], 'k--',  #
        #                                marker='o', markerfacecolor='none', label='Green-Jacobi')

        plt.legend(handles=[line7, line8], loc='lower right',
                   title=fr'$\frac{{\mbox{{PCG time}}}}{{\rule{{0pt}}{{2.0ex}}\mbox{{\# of PCG iterations}}}}$')

        ax_time_per_it.set_xlabel(r'Discretization')
        ax_time_per_it.set_ylabel(f'Time / Time of Green at $\mathcal{{T}}_{{{8}}}$')
        ax_time_per_it.set_xlim([Ns[0] ** 2, Ns[-1] ** 2])
        ax_time_per_it.set_xticks(Ns ** 2)
        # ax_time_per_it.set_xticklabels([fr'${{2}}^{int(x)}$' for x in np.arange(nb_pixels, 11)])
        # ax_time_per_it.set_xticklabels([fr'$2^{{{int(x)}}}$' for x in np.arange(nb_pixels, 11)])
        ax_time_per_it.set_xticklabels([f'$\mathcal{{T}}_{{{int(x)}}}$' for x in Ns])

        # ax_time_per_it.set_xticks([1e3, 1e4, 1e5, 1e6, 1e6])
        # ax_time_per_it.set_ylim([1e-2, 1e5])
        ax_time_per_it.set_yscale('linear')

        fig.tight_layout()
        fname = f'time_scaling' + '{}'.format('.pdf')
        plt.savefig(figure_folder_path + script_name + fname, bbox_inches='tight')
        print(('create figure: {}'.format(figure_folder_path + script_name + fname)))

        plt.show()
