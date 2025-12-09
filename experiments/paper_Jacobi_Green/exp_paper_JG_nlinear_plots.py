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

script_name = 'exp_paper_JG_nlinear'

file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
figure_folder_path = file_folder_path + '/figures/' + script_name + '/'

# nb_pix_multips = 10
#
# # 'Jacobi'  # 'Green'  # 'Green_Jacobi'
# nb_it_Green_linear_1 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])
# nb_it_Green_linear_4 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])
# nb_it_Jacobi_linear_1 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])
# nb_it_Jacobi_linear_4 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])
# nb_it_Green_Jacobi_linear_1 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])
# nb_it_Green_Jacobi_linear_4 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])
#
# time_G_1 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])
# time_J_1 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])
# time_GJ_1 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])
#
# time_CG_G_1 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])
# time_CG_J_1 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])
# time_CG_GJ_1 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])
#
# for j in np.arange(2, nb_pix_multips + 1):
#     number_of_pixels = 2 ** j
#     # print('j=', j)
#     for i in np.arange(2, j + 1):
#         # print('i=', i)
#         nb_laminates = 2 ** i
#
#         total_phase_contrast = 1
#         preconditioner_type = 'Green'
#         results_name = (
#                 f'nb_nodes_{number_of_pixels}_' + f'nb_pixels_{nb_laminates}_' + f'contrast_{total_phase_contrast}_' + f'prec_{preconditioner_type}')
#         info = np.load(data_folder_path + results_name + f'.npz', allow_pickle=True)
#         nb_it_Green_linear_1[i - 2, j - 2] = info.f.nb_steps
#         try:
#             time_G_1[i - 2, j - 2] = info.f.elapsed_time
#             time_CG_G_1[i - 2, j - 2] = info.f.elapsed_time_CG
#
#         except AttributeError:
#             time_G_1[i - 2, j - 2] = 0
#             time_CG_G_1[i - 2, j - 2] = 0
#
#         total_phase_contrast = 1
#         preconditioner_type = 'Jacobi'
#         results_name = (
#                 f'nb_nodes_{number_of_pixels}_' + f'nb_pixels_{nb_laminates}_' + f'contrast_{total_phase_contrast}_' + f'prec_{preconditioner_type}')
#         info = np.load(data_folder_path + results_name + f'.npz', allow_pickle=True)
#         try:
#             time_J_1[i - 2, j - 2] = info.f.elapsed_time
#             time_CG_J_1[i - 2, j - 2] = info.f.elapsed_time_CG
#
#         except AttributeError:
#             time_J_1[i - 2, j - 2] = 0
#             time_CG_J_1[i - 2, j - 2] = 0
#
#         # nb_it_Jacobi_linear_1[i - 2, j - 2] = info.f.nb_steps
#
#         total_phase_contrast = 1
#         preconditioner_type = 'Green_Jacobi'
#         results_name = (
#                 f'nb_nodes_{number_of_pixels}_' + f'nb_pixels_{nb_laminates}_' + f'contrast_{total_phase_contrast}_' + f'prec_{preconditioner_type}')
#         info = np.load(data_folder_path + results_name + f'.npz', allow_pickle=True)
#
#         nb_it_Green_Jacobi_linear_1[i - 2, j - 2] = info.f.nb_steps
#         try:
#             time_GJ_1[i - 2, j - 2] = info.f.elapsed_time
#             time_CG_GJ_1[i - 2, j - 2] = info.f.elapsed_time_CG
#
#         except AttributeError:
#             time_GJ_1[i - 2, j - 2] = 0
#             time_CG_GJ_1[i - 2, j - 2] = 0
#
# nb_it_Green_linear_1 = np.transpose(nb_it_Green_linear_1)
# nb_it_Green_linear_4 = np.transpose(nb_it_Green_linear_4)
# nb_it_Jacobi_linear_1 = np.transpose(nb_it_Jacobi_linear_1)
# nb_it_Jacobi_linear_4 = np.transpose(nb_it_Jacobi_linear_4)
# nb_it_Green_Jacobi_linear_1 = np.transpose(nb_it_Green_Jacobi_linear_1)
# nb_it_Green_Jacobi_linear_4 = np.transpose(nb_it_Green_Jacobi_linear_4)
# plot_time = True
# if plot_time:
#     nb_pixels = 3
#     Ns = 2 ** np.arange(nb_pixels, 8)
#     time_G_1 = np.transpose(time_G_1)
#     time_CG_G_1 = np.transpose(time_CG_G_1)
#     time_GJ_1 = np.transpose(time_GJ_1)
#     time_CG_GJ_1 = np.transpose(time_CG_GJ_1)
#
#     fig = plt.figure(figsize=(4.0, 4.0))
#     gs = fig.add_gridspec(2, 1, hspace=0.5, wspace=0.5, width_ratios=[1],
#                           height_ratios=[1,1])
#     ax_time_per_it = fig.add_subplot(gs[0, :])
#     nb_dofs = 2 * Ns ** 2
#
#     time_per_iteration_G_1 = time_G_1 / nb_it_Green_linear_1
#     time_per_iteration_GJ_1 = time_GJ_1 / nb_it_Green_Jacobi_linear_1
#     time_per_iteration_CG_G_1 = time_CG_G_1 / nb_it_Green_linear_1
#     time_per_iteration_CG_GJ_1 = time_CG_GJ_1 / nb_it_Green_Jacobi_linear_1
#
#     # scaling
#     line1, = ax_time_per_it.loglog(nb_dofs, nb_dofs * np.log(nb_dofs) / (nb_dofs[0] * np.log(nb_dofs[0])) * time_G_1[0, 0] , ':',#/                        time_G_1[2:, 0][0]
#                         label=r'Quasilinear - $ \mathcal{O} (N_{\mathrm{N}} \log  N_{\mathrm{N}}$)')
#     line2, = ax_time_per_it.loglog(nb_dofs, nb_dofs / 5e6, '--',
#                         label=r'Linear - $\mathcal{O} (N_{\mathrm{N}})$')
#     #plt.loglog(np.linspace(1e1, 1e8), 1e-4 * np.linspace(1e1, 1e8), 'k-', linewidth=0.9)
#
#
#     # line3, = plt.loglog(nb_dofs, time_G_1[2:, 0] / time_G_1[2:, 0][0], '-x', color='Green', label='Green')
#     # line4, = plt.loglog(nb_dofs, time_GJ_1[2:, 0] / time_G_1[2:, 0][0], 'k-', marker='o', markerfacecolor='none',
#     #                     label='Green-Jacobi')
#
#     # plt.loglog(nb_dofs, nb_dofs / (nb_dofs[0]) * time_G[0], '--', label='linear')
#
#     line5, = ax_time_per_it.loglog(nb_dofs, time_per_iteration_G_1[2:, 0] , 'g-.|', label='Green')# / time_per_iteration_G_1[2, 0]
#     line6, = ax_time_per_it.loglog(nb_dofs, time_per_iteration_GJ_1[2:, 0]    , 'k-.',
#                          marker='o', markerfacecolor='none', label='Green-Jacobi')
#
#     line7, = ax_time_per_it.loglog(nb_dofs, time_per_iteration_CG_G_1[2:, 0], 'g--^', label='Green')  # / time_per_iteration_G_1[2, 0]
#     line8, = ax_time_per_it.loglog(nb_dofs, time_per_iteration_CG_GJ_1[2:, 0], 'k--^',
#                         marker='o', markerfacecolor='none', label='Green-Jacobi')
#
#     # line1, = plt.loglog(nb_dofs, time_per_iteration_CG_G_1[2:, 0]/time_per_iteration_CG_G_1[2,0]  , 'r-.x', label='Green')#/ time_G_1[2:, 0][0]
#     # line2, = plt.loglog(nb_dofs,time_per_iteration_CG_GJ_1[2:, 0]/time_per_iteration_CG_G_1[2,0]  , 'r--',#/ time_G_1[2:, 0][0]
#     #                     marker='o', markerfacecolor='none', label='Green-Jacobi')
#     # plt.loglog(nb_dofs,
#     #            nb_dofs* np.log(nb_dofs) / (nb_dofs[0]  * np.log(nb_dofs)) * time_G[0] / its_G[0], ':',
#     #            label='N log N')
#
#
#     ax_time_per_it.set_xlabel(r' $\#$ of degrees of freedom (DOFs) - $d N_{\mathrm{N}}$')
#     ax_time_per_it.set_ylabel('Time (s)')
#     ax_time_per_it.set_xlim([nb_dofs[0], nb_dofs[-1]])
#     ax_time_per_it.set_xticks([1e3, 1e4, 1e5, 1e6])
#     ax_time_per_it.set_ylim([1e-4, 1e1])
#
#     # plt.gca().set_xticks(iterations)
#
#   #   legend1 = plt.legend(handles=[line1, line2, line3], loc='upper left', title='Wall-clock time')
#   #
#   #   plt.gca().add_artist(legend1)  # Add the first legend manually
#   #
#   # #  Second legend (bottom right)
#   #   plt.legend(handles=[line4, line5, line6], loc='lower right', title='Wall-clock time / $\#$ of PCG iterations')
#
#
#     ax_relative_time = fig.add_subplot(gs[1, :])
#     #line5, = ax_relative_time.loglog(nb_dofs, time_per_iteration_G_1[2:, 0] , 'g-.|', label='Green')# / time_per_iteration_G_1[2, 0]
#     line1, = ax_relative_time.semilogx(nb_dofs, 100*time_per_iteration_GJ_1[2:, 0]/time_per_iteration_G_1[2:, 0]    , 'k-.',
#                          marker='o', markerfacecolor='none', label='Green-Jacobi')
#     line2, = ax_relative_time.semilogx(nb_dofs, 100*time_per_iteration_CG_GJ_1[2:, 0]/time_per_iteration_CG_G_1[2:, 0]    , 'g-.',
#                          marker='o', markerfacecolor='none', label='Green-Jacobi')
#
#     ax_relative_time.set_xlim([nb_dofs[0], nb_dofs[-1]])
#     ax_relative_time.set_xticks([1e3, 1e4, 1e5, 1e6])
#     ax_relative_time.set_ylim([80, 160])
#
#
#
#
#
#
#
#
#     fig.tight_layout()
#     fname = f'time_scaling' + '{}'.format('.pdf')
#     plt.savefig(figure_folder_path + script_name + fname, bbox_inches='tight')
#     print(('create figure: {}'.format(figure_folder_path + script_name + fname)))
#
#
#
#     plt.show()
#     quit()

nb_pix_multips = 10
norm_='norm_rr'

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

for j in np.arange(2, nb_pix_multips + 1):
    number_of_pixels = 2 ** j
    # print('j=', j)
    for i in np.arange(2, j + 1):
        # print('i=', i)
        nb_laminates = 2 ** i

        total_phase_contrast = 1
        preconditioner_type = 'Green'
        results_name = (
                f'nb_nodes_{number_of_pixels}_' + f'nb_pixels_{nb_laminates}_' + f'contrast_{total_phase_contrast}_' + f'prec_{preconditioner_type}')#+ f'{norm_}'
        info = np.load(data_folder_path + results_name + f'.npz', allow_pickle=True)
        nb_it_Green_linear_1[i - 2, j - 2] = info.f.nb_steps
        try:
            time_G_1[i - 2, j - 2] = info.f.elapsed_time
        except AttributeError:
            time_G_1[i - 2, j - 2] = 0

        total_phase_contrast = 4
        results_name = (
                f'nb_nodes_{number_of_pixels}_' + f'nb_pixels_{nb_laminates}_' + f'contrast_{total_phase_contrast}_' + f'prec_{preconditioner_type}'+ f'{norm_}')
        info = np.load(data_folder_path + results_name + f'.npz', allow_pickle=True)
        nb_it_Green_linear_4[i - 2, j - 2] = info.f.nb_steps

        total_phase_contrast = 1
        preconditioner_type = 'Jacobi'
        results_name = (
                f'nb_nodes_{number_of_pixels}_' + f'nb_pixels_{nb_laminates}_' + f'contrast_{total_phase_contrast}_' + f'prec_{preconditioner_type}') #f'{norm_}'
        info = np.load(data_folder_path + results_name + f'.npz', allow_pickle=True)
        try:
            time_J_1[i - 2, j - 2] = info.f.elapsed_time
        except AttributeError:
            time_J_1[i - 2, j - 2] = 0

        nb_it_Jacobi_linear_1[i - 2, j - 2] = info.f.nb_steps
        total_phase_contrast = 4
        results_name = (
                f'nb_nodes_{number_of_pixels}_' + f'nb_pixels_{nb_laminates}_' + f'contrast_{total_phase_contrast}_' + f'prec_{preconditioner_type}'+ f'{norm_}')
        info = np.load(data_folder_path + results_name + f'.npz', allow_pickle=True)
        nb_it_Jacobi_linear_4[i - 2, j - 2] = info.f.nb_steps

        total_phase_contrast = 1
        preconditioner_type = 'Green_Jacobi'
        results_name = (
                f'nb_nodes_{number_of_pixels}_' + f'nb_pixels_{nb_laminates}_' + f'contrast_{total_phase_contrast}_' + f'prec_{preconditioner_type}')#+ f'{norm_}'
        info = np.load(data_folder_path + results_name + f'.npz', allow_pickle=True)

        nb_it_Green_Jacobi_linear_1[i - 2, j - 2] = info.f.nb_steps
        try:
            time_GJ_1[i - 2, j - 2] = info.f.elapsed_time
        except AttributeError:
            time_GJ_1[i - 2, j - 2] = 0

        total_phase_contrast = 4
        results_name = (
                f'nb_nodes_{number_of_pixels}_' + f'nb_pixels_{nb_laminates}_' + f'contrast_{total_phase_contrast}_' + f'prec_{preconditioner_type}'+ f'{norm_}')
        info = np.load(data_folder_path + results_name + f'.npz', allow_pickle=True)
        nb_it_Green_Jacobi_linear_4[i - 2, j - 2] = info.f.nb_steps

nb_it_Green_linear_1 = np.transpose(nb_it_Green_linear_1)
nb_it_Green_linear_4 = np.transpose(nb_it_Green_linear_4)
nb_it_Jacobi_linear_1 = np.transpose(nb_it_Jacobi_linear_1)
nb_it_Jacobi_linear_4 = np.transpose(nb_it_Jacobi_linear_4)
nb_it_Green_Jacobi_linear_1 = np.transpose(nb_it_Green_Jacobi_linear_1)
nb_it_Green_Jacobi_linear_4 = np.transpose(nb_it_Green_Jacobi_linear_4)
plot_this = True
if plot_this:
    nb_pix_multips = [2, 3, 4, 5, 6, 7, 8, 9, 10]  # , 8, 9, 10
    Nx = (np.asarray(nb_pix_multips))
    X, Y = np.meshgrid(Nx, Nx, indexing='ij')
    #
    #   nb_pix_multips = [2, 4, 5, 6, 7, 8]
    # material distribution
    geometry_ID = 'linear'  # linear  # 'abs_val' sine_wave_   ,laminate_log  geometry_ID = 'right_cluster_x3'  # laminate2       # 'abs_val' sine_wave_   ,laminate_log
    # rhs = 'sin_wave'
    rhs = False
    linestyles = ['-', '--', ':', '-.', '--', ':', '-.']
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'orange', 'purple']
    precc = 6
    fig = plt.figure(figsize=(8.5, 5.5))  # 11, 7.0
    gs = fig.add_gridspec(2, 4, hspace=0.22, wspace=0.25, width_ratios=[1.2, 1.2, 1.2, 0.03],
                          height_ratios=[1, 1])
    row = 0
    for phase_contrast in [1, 4]:  # 2, 4 for sine wave
        ratio = phase_contrast
        if geometry_ID == 'linear':
            divnorm = mpl.colors.Normalize(vmin=0, vmax=100)
            white_lim = 50
        elif geometry_ID == 'sine_wave_':
            divnorm = mpl.colors.Normalize(vmin=0, vmax=100)
            white_lim = 50
        # Green graph
        gs0 = gs[row, 0].subgridspec(1, 1, wspace=0.1, width_ratios=[1])
        ax = fig.add_subplot(gs0[0, 0])
        # ax.set_aspect('equal')
        if phase_contrast == 1:
            nb_iterations = nb_it_Green_linear_1
        elif phase_contrast == 4:
            nb_iterations = nb_it_Green_linear_4

        nb_iterations = np.nan_to_num(nb_iterations, nan=1.0)
        for i in range(nb_iterations.shape[0]):
            for j in range(nb_iterations.shape[1]):
                if nb_iterations[i, j] == 0:
                    pass
                elif nb_iterations[i, j] < white_lim:
                    ax.text(i + Nx[0], j + Nx[0], f'{nb_iterations[i, j]:.0f}', size=8,
                            ha='center', va='center', color='black')
                elif nb_iterations[i, j] > 999:
                    ax.text(i + Nx[0], j + Nx[0], f'{999:.0f}', size=8,
                            ha='center', va='center', color='black')
                else:
                    ax.text(i + Nx[0], j + Nx[0], f'{nb_iterations[i, j]:.0f}', size=8,
                            ha='center', va='center', color='white')

        pcm = ax.pcolormesh(X, Y, nb_iterations, label='PCG: Green + Jacobi', cmap='Reds', norm=divnorm)

        # ax.text(0.05, 0.92, f'Total phase contrast $\chi=10^{phase_contrast}$', transform=ax.transAxes)
        if geometry_ID == 'sine_wave_' and phase_contrast == 2:
            ax.text(0.05, 0.82, f'Total phase contrast \n' + r'$\chi^{\rm tot}=\infty$', transform=ax.transAxes)
        elif geometry_ID == 'sine_wave_':
            ax.text(0.05, 0.82, f'Total phase contrast \n' + fr' $\chi^{{\rm tot}}=10^{{{phase_contrast}}}$',
                    transform=ax.transAxes)
        else:
            ax.text(0.05, 0.82, f'Total phase contrast \n' + fr' $\chi^{{\rm tot}}=10^{{{phase_contrast}}}$',
                    transform=ax.transAxes)

        if row == 0:
            ax.set_title('Number of iterations \n Green ')
        # ax.set_zlim(1 ,100)
        # ax.set_ylabel('# data/geometry sampling points (x direction)')

        # ax.yaxis.set_label_position('right')
        # ax.yaxis.tick_right()
        ax.set_ylabel(r'\#  material pixels - $p$')  # $p$~of~$\mathcal{G}_p$
        if row == 1:
            ax.set_xlabel(r'\#  nodal points - $n$')  # ~of~$\mathcal{T}_n$
        ax.set_xticks(Nx)
        ax.set_xticklabels([f'$2^{{{i}}}$' for i in Nx])
        # ax2 = ax.twinx()
        ax.set_yticks(Nx)
        ax.set_yticklabels([f'$2^{{{i}}}$' for i in Nx])
        ax.tick_params(right=True, top=False, labelright=False, labeltop=False, labelrotation=0)
        #    ax.set_aspect('equal')
        if row == 0:
            ax.text(-0.20, 1.15, rf'\textbf{{(a.{row + 1}) }}', transform=ax.transAxes)

        elif row == 1:
            ax.text(-0.20, 1.05, rf'\textbf{{(a.{row + 1}) }}', transform=ax.transAxes)

        # jacobi  graph
        gs1 = gs[row, 1].subgridspec(1, 1, wspace=0.1, width_ratios=[5])
        ax = fig.add_subplot(gs1[0, 0])
        #    ax.set_aspect('equal')
        if phase_contrast == 1:
            nb_iterations = nb_it_Jacobi_linear_1
        elif phase_contrast == 4:
            nb_iterations = nb_it_Jacobi_linear_4

        nb_iterations = np.nan_to_num(nb_iterations, nan=1.0)
        for i in range(nb_iterations.shape[0]):
            for j in range(nb_iterations.shape[1]):
                if nb_iterations[i, j] == 0:
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

        pcm = ax.pcolormesh(X, Y, nb_iterations, label='PCG: Green + Jacobi', cmap='Reds', norm=divnorm)

        # ax.text(0.05, 0.92, f'Total phase contrast $\chi=10^{phase_contrast}$', transform=ax.transAxes)
        if geometry_ID == 'sine_wave_' and phase_contrast == 2:
            ax.text(0.05, 0.82, f'Total phase contrast \n' + r'$\chi^{\rm tot}=\infty$', transform=ax.transAxes)
        elif geometry_ID == 'sine_wave_':
            ax.text(0.05, 0.82, f'Total phase contrast \n' + fr' $\chi^{{\rm tot}}=10^{{{phase_contrast}}}$',
                    transform=ax.transAxes)
        else:
            ax.text(0.05, 0.82, f'Total phase contrast \n' + fr' $\chi^{{\rm tot}}=10^{{{phase_contrast}}}$',
                    transform=ax.transAxes)

        if row == 0:
            ax.set_title('Number of iterations \n Jacobi ')
        # ax.set_zlim(1 ,100)
        # ax.set_ylabel('# data/geometry sampling points (x direction)')

        # ax.yaxis.set_label_position('right')
        # ax.yaxis.tick_right()
        if row == 1:
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
            ax.text(-0.20, 1.15, rf'\textbf{{(b.{row + 1}) }}', transform=ax.transAxes)
        elif row == 1:
            ax.text(-0.20, 1.05, rf'\textbf{{(b.{row + 1}) }}', transform=ax.transAxes)
        # plot Green Jacobi
        gs2 = gs[row, 2].subgridspec(1, 1, wspace=0.1, width_ratios=[5])
        ax = fig.add_subplot(gs2[0, 0])
        if phase_contrast == 1:
            nb_iterations = nb_it_Green_Jacobi_linear_1
        elif phase_contrast == 4:
            nb_iterations = nb_it_Green_Jacobi_linear_4

        nb_iterations = np.nan_to_num(nb_iterations, nan=1.0)
        for i in range(nb_iterations.shape[0]):
            for j in range(nb_iterations.shape[1]):
                if nb_iterations[i, j] == 0:
                    pass
                elif nb_iterations[i, j] < white_lim:
                    ax.text(i + Nx[0], j + Nx[0], f'{nb_iterations[i, j]:.0f}', size=8,
                            ha='center', va='center', color='black')
                else:
                    ax.text(i + Nx[0], j + Nx[0], f'{nb_iterations[i, j]:.0f}', size=8,
                            ha='center', va='center', color='white')
        # Replace NaN values with zero

        pcm = ax.pcolormesh(X, Y, nb_iterations, label='PCG: Green + Jacobi', cmap='Reds', norm=divnorm)

        if geometry_ID == 'sine_wave_' and phase_contrast == 2:
            ax.text(0.05, 0.82, f'Total phase contrast \n' + r'$\chi^{\rm tot}=\infty$', transform=ax.transAxes)
        elif geometry_ID == 'sine_wave_':
            ax.text(0.05, 0.82, f'Total phase contrast \n ' + fr' $\chi^{{\rm tot}}=10^{{{phase_contrast}}}$',
                    transform=ax.transAxes)
        else:
            ax.text(0.05, 0.82, f'Total phase contrast \n' + fr' $\chi^{{\rm tot}}=10^{{{phase_contrast}}}$',
                    transform=ax.transAxes)

        if row == 0:
            ax.set_title('Number of iterations \n Green-Jacobi')
        # ax.set_zlim(1 ,100)
        # ax.set_ylabel('# of material phases')

        # ax.yaxis.set_label_position('right')
        # ax.yaxis.tick_right()
        if row == 1:
            ax.set_xlabel(r'\#  nodal points - $n$')  # ~of~$\mathcal{T}_n$

        ax.set_xticks(Nx)
        ax.set_xticklabels([f'$2^{{{i}}}$' for i in Nx])
        # ax2 = ax.twinx()
        ax.set_yticks(Nx)
        ax.set_yticklabels([f'$2^{{{i}}}$' for i in Nx])
        ax.tick_params(right=True, top=False, labelright=False, labeltop=False, labelrotation=0)
        #   ax.set_aspect('equal')

        if row == 0:
            ax.text(-0.20, 1.15, rf'\textbf{{(c.{row + 1}) }}', transform=ax.transAxes)
        elif row == 1:
            ax.text(-0.20, 1.05, rf'\textbf{{(c.{row + 1}) }}', transform=ax.transAxes)
        # Adding a color bar with custom ticks and labels
        cbar_ax = fig.add_subplot(gs[row, 3])
        cbar = plt.colorbar(pcm, location='left', cax=cbar_ax, ticklocation='right')  # Specify the ticks
        # cbar.ax.invert_yaxis()
        # # cbar.set_ticks(ticks=[  0, 1,10])
        # cbar.set_ticks([10, 5, 2, 1, 1 / 2, 1 / 5, 1 / 10])
        # cbar.ax.set_yticklabels(
        #     ['Jacobi-Green \n needs less', '5 times', '2 times', 'Equal', '2 times', '5 times',
        #      'Jacobi-Green \n needs more'])

        #

        row += 1

    fname = figure_folder_path + 'JG_exp4_GRID_DEP_nb_its_geom_{}_rho_{}_norm_{}{}'.format('nlaminate', phase_contrast, norm_,'.pdf')
    print(('create figure: {}'.format(fname)))
    plt.savefig(fname, bbox_inches='tight')

plt.show()

script_name = 'exp_paper_JG_cos'
file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
figure_folder_path = file_folder_path + '/figures/' + script_name + '/'

nb_pix_multips = 10

# 'Jacobi'  # 'Green'  # 'Green_Jacobi'
nb_it_Green_linear_1 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])
nb_it_Green_linear_4 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])
nb_it_Jacobi_linear_1 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])
nb_it_Jacobi_linear_4 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])
nb_it_Green_Jacobi_linear_1 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])
nb_it_Green_Jacobi_linear_4 = np.zeros([nb_pix_multips - 1, nb_pix_multips - 1])

for j in np.arange(2, nb_pix_multips + 1):
    number_of_pixels = 2 ** j
    # print('j=', j)
    for i in np.arange(2, j + 1):
        # print('i=', i)
        nb_laminates = 2 ** i

        total_phase_contrast = 0
        preconditioner_type = 'Green'
        results_name = (
                f'nb_nodes_{number_of_pixels}_' + f'nb_pixels_{nb_laminates}_' + f'contrast_{total_phase_contrast}_' + f'prec_{preconditioner_type}'+ f'{norm_}')
        info = np.load(data_folder_path + results_name + f'.npz', allow_pickle=True)
        nb_it_Green_linear_1[i - 2, j - 2] = info.f.nb_steps

        total_phase_contrast = 4
        results_name = (
                f'nb_nodes_{number_of_pixels}_' + f'nb_pixels_{nb_laminates}_' + f'contrast_{total_phase_contrast}_' + f'prec_{preconditioner_type}'+ f'{norm_}')
        info = np.load(data_folder_path + results_name + f'.npz', allow_pickle=True)
        nb_it_Green_linear_4[i - 2, j - 2] = info.f.nb_steps

        total_phase_contrast = 0
        preconditioner_type = 'Jacobi'
        results_name = (
                f'nb_nodes_{number_of_pixels}_' + f'nb_pixels_{nb_laminates}_' + f'contrast_{total_phase_contrast}_' + f'prec_{preconditioner_type}'+ f'{norm_}')
        info = np.load(data_folder_path + results_name + f'.npz', allow_pickle=True)

        nb_it_Jacobi_linear_1[i - 2, j - 2] = info.f.nb_steps
        total_phase_contrast = 4
        results_name = (
                f'nb_nodes_{number_of_pixels}_' + f'nb_pixels_{nb_laminates}_' + f'contrast_{total_phase_contrast}_' + f'prec_{preconditioner_type}'+ f'{norm_}')
        info = np.load(data_folder_path + results_name + f'.npz', allow_pickle=True)
        nb_it_Jacobi_linear_4[i - 2, j - 2] = info.f.nb_steps

        total_phase_contrast = 0
        preconditioner_type = 'Green_Jacobi'
        results_name = (
                f'nb_nodes_{number_of_pixels}_' + f'nb_pixels_{nb_laminates}_' + f'contrast_{total_phase_contrast}_' + f'prec_{preconditioner_type}'+ f'{norm_}')
        info = np.load(data_folder_path + results_name + f'.npz', allow_pickle=True)

        nb_it_Green_Jacobi_linear_1[i - 2, j - 2] = info.f.nb_steps
        total_phase_contrast = 4
        results_name = (
                f'nb_nodes_{number_of_pixels}_' + f'nb_pixels_{nb_laminates}_' + f'contrast_{total_phase_contrast}_' + f'prec_{preconditioner_type}'+ f'{norm_}')
        info = np.load(data_folder_path + results_name + f'.npz', allow_pickle=True)
        nb_it_Green_Jacobi_linear_4[i - 2, j - 2] = info.f.nb_steps

nb_it_Green_linear_1 = np.transpose(nb_it_Green_linear_1)
nb_it_Green_linear_4 = np.transpose(nb_it_Green_linear_4)
nb_it_Jacobi_linear_1 = np.transpose(nb_it_Jacobi_linear_1)
nb_it_Jacobi_linear_4 = np.transpose(nb_it_Jacobi_linear_4)
nb_it_Green_Jacobi_linear_1 = np.transpose(nb_it_Green_Jacobi_linear_1)
nb_it_Green_Jacobi_linear_4 = np.transpose(nb_it_Green_Jacobi_linear_4)

plot_this = True
if plot_this:
    nb_pix_multips = [2, 3, 4, 5, 6, 7, 8, 9, 10]  #
    Nx = (np.asarray(nb_pix_multips))
    X, Y = np.meshgrid(Nx, Nx, indexing='ij')
    #
    #   nb_pix_multips = [2, 4, 5, 6, 7, 8]
    # material distribution
    geometry_ID = 'sine_wave_'  # linear  # 'abs_val' sine_wave_   ,laminate_log  geometry_ID = 'right_cluster_x3'  # laminate2       # 'abs_val' sine_wave_   ,laminate_log
    # rhs = 'sin_wave'
    rhs = False
    linestyles = ['-', '--', ':', '-.', '--', ':', '-.']
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'orange', 'purple']
    fig = plt.figure(figsize=(8.5, 5.5))  # 11, 7.0
    gs = fig.add_gridspec(2, 4, hspace=0.22, wspace=0.25, width_ratios=[1.2, 1.2, 1.2, 0.03],
                          height_ratios=[1, 1])
    row = 0
    for phase_contrast in [0, 4]:  # 2, 4 for sine wave
        ratio = phase_contrast
        if geometry_ID == 'linear':
            divnorm = mpl.colors.Normalize(vmin=0, vmax=100)
            white_lim = 50
        elif geometry_ID == 'sine_wave_':
            divnorm = mpl.colors.Normalize(vmin=0, vmax=100)
            white_lim = 50
        # Green graph
        gs0 = gs[row, 0].subgridspec(1, 1, wspace=0.1, width_ratios=[1])
        ax = fig.add_subplot(gs0[0, 0])
        # ax.set_aspect('equal')
        if phase_contrast == 0:
            nb_iterations = nb_it_Green_linear_1
        elif phase_contrast == 4:
            nb_iterations = nb_it_Green_linear_4

        nb_iterations = np.nan_to_num(nb_iterations, nan=1.0)
        for i in range(nb_iterations.shape[0]):
            for j in range(nb_iterations.shape[1]):
                if nb_iterations[i, j] == 0:
                    pass
                elif nb_iterations[i, j] < white_lim:
                    ax.text(i + Nx[0], j + Nx[0], f'{nb_iterations[i, j]:.0f}', size=8,
                            ha='center', va='center', color='black')
                elif nb_iterations[i, j] > 999:
                    ax.text(i + Nx[0], j + Nx[0], f'{999:.0f}', size=8,
                            ha='center', va='center', color='black')
                else:
                    ax.text(i + Nx[0], j + Nx[0], f'{nb_iterations[i, j]:.0f}', size=8,
                            ha='center', va='center', color='white')

        pcm = ax.pcolormesh(X, Y, nb_iterations, label='PCG: Green + Jacobi', cmap='Reds', norm=divnorm)

        # ax.text(0.05, 0.92, f'Total phase contrast $\chi=10^{phase_contrast}$', transform=ax.transAxes)
        if geometry_ID == 'sine_wave_' and phase_contrast == 0:
            ax.text(0.05, 0.82, f'Total phase contrast \n' + r'$\chi^{\rm tot}=\infty$', transform=ax.transAxes)
        elif geometry_ID == 'sine_wave_':
            ax.text(0.05, 0.82, f'Total phase contrast \n' + fr' $\chi^{{\rm tot}}=10^{{{phase_contrast}}}$',
                    transform=ax.transAxes)
        else:
            ax.text(0.05, 0.82, f'Total phase contrast \n' + fr' $\chi^{{\rm tot}}=10^{{{phase_contrast}}}$',
                    transform=ax.transAxes)

        if row == 0:
            ax.set_title('Number of iterations \n Green ')
        # ax.set_zlim(1 ,100)
        # ax.set_ylabel('# data/geometry sampling points (x direction)')

        # ax.yaxis.set_label_position('right')
        # ax.yaxis.tick_right()
        ax.set_ylabel(r'\#  material pixels - $p$')  # $p$~of~$\mathcal{G}_p$
        if row == 1:
            ax.set_xlabel(r'\#  nodal points - $n$')  # ~of~$\mathcal{T}_n$
        ax.set_xticks(Nx)
        ax.set_xticklabels([f'$2^{{{i}}}$' for i in Nx])
        # ax2 = ax.twinx()
        ax.set_yticks(Nx)
        ax.set_yticklabels([f'$2^{{{i}}}$' for i in Nx])
        ax.tick_params(right=True, top=False, labelright=False, labeltop=False, labelrotation=0)
        #    ax.set_aspect('equal')
        if row == 0:
            ax.text(-0.20, 1.15, rf'\textbf{{(a.{row + 1}) }}', transform=ax.transAxes)

        elif row == 1:
            ax.text(-0.20, 1.05, rf'\textbf{{(a.{row + 1}) }}', transform=ax.transAxes)

        # jacobi  graph
        gs1 = gs[row, 1].subgridspec(1, 1, wspace=0.1, width_ratios=[5])
        ax = fig.add_subplot(gs1[0, 0])
        #    ax.set_aspect('equal')
        if phase_contrast == 0:
            nb_iterations = nb_it_Jacobi_linear_1
        elif phase_contrast == 4:
            nb_iterations = nb_it_Jacobi_linear_4

        nb_iterations = np.nan_to_num(nb_iterations, nan=1.0)
        for i in range(nb_iterations.shape[0]):
            for j in range(nb_iterations.shape[1]):
                if nb_iterations[i, j] == 0:
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

        pcm = ax.pcolormesh(X, Y, nb_iterations, label='PCG: Green + Jacobi', cmap='Reds', norm=divnorm)

        # ax.text(0.05, 0.92, f'Total phase contrast $\chi=10^{phase_contrast}$', transform=ax.transAxes)
        if geometry_ID == 'sine_wave_' and phase_contrast == 0:
            ax.text(0.05, 0.82, f'Total phase contrast \n' + r'$\chi^{\rm tot}=\infty$', transform=ax.transAxes)
        elif geometry_ID == 'sine_wave_':
            ax.text(0.05, 0.82, f'Total phase contrast \n' + fr' $\chi^{{\rm tot}}=10^{{{phase_contrast}}}$',
                    transform=ax.transAxes)
        else:
            ax.text(0.05, 0.82, f'Total phase contrast \n' + fr' $\chi^{{\rm tot}}=10^{{{phase_contrast}}}$',
                    transform=ax.transAxes)

        if row == 0:
            ax.set_title('Number of iterations \n Jacobi ')
        # ax.set_zlim(1 ,100)
        # ax.set_ylabel('# data/geometry sampling points (x direction)')

        # ax.yaxis.set_label_position('right')
        # ax.yaxis.tick_right()
        if row == 1:
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
            ax.text(-0.20, 1.15, rf'\textbf{{(b.{row + 1}) }}', transform=ax.transAxes)
        elif row == 1:
            ax.text(-0.20, 1.05, rf'\textbf{{(b.{row + 1}) }}', transform=ax.transAxes)
        # plot Green Jacobi
        gs2 = gs[row, 2].subgridspec(1, 1, wspace=0.1, width_ratios=[5])
        ax = fig.add_subplot(gs2[0, 0])
        if phase_contrast == 0:
            nb_iterations = nb_it_Green_Jacobi_linear_1
        elif phase_contrast == 4:
            nb_iterations = nb_it_Green_Jacobi_linear_4

        nb_iterations = np.nan_to_num(nb_iterations, nan=1.0)
        for i in range(nb_iterations.shape[0]):
            for j in range(nb_iterations.shape[1]):
                if nb_iterations[i, j] == 0:
                    pass
                elif nb_iterations[i, j] < white_lim:
                    ax.text(i + Nx[0], j + Nx[0], f'{nb_iterations[i, j]:.0f}', size=8,
                            ha='center', va='center', color='black')
                else:
                    ax.text(i + Nx[0], j + Nx[0], f'{nb_iterations[i, j]:.0f}', size=8,
                            ha='center', va='center', color='white')
        # Replace NaN values with zero

        pcm = ax.pcolormesh(X, Y, nb_iterations, label='PCG: Green + Jacobi', cmap='Reds', norm=divnorm)

        if geometry_ID == 'sine_wave_' and phase_contrast == 0:
            ax.text(0.05, 0.82, f'Total phase contrast \n' + r'$\chi^{\rm tot}=\infty$', transform=ax.transAxes)
        elif geometry_ID == 'sine_wave_':
            ax.text(0.05, 0.82, f'Total phase contrast \n ' + fr' $\chi^{{\rm tot}}=10^{{{phase_contrast}}}$',
                    transform=ax.transAxes)
        else:
            ax.text(0.05, 0.82, f'Total phase contrast \n' + fr' $\chi^{{\rm tot}}=10^{{{phase_contrast}}}$',
                    transform=ax.transAxes)

        if row == 0:
            ax.set_title('Number of iterations \n Green-Jacobi')
        # ax.set_zlim(1 ,100)
        # ax.set_ylabel('# of material phases')

        # ax.yaxis.set_label_position('right')
        # ax.yaxis.tick_right()
        if row == 1:
            ax.set_xlabel(r'\#  nodal points - $n$')  # ~of~$\mathcal{T}_n$

        ax.set_xticks(Nx)
        ax.set_xticklabels([f'$2^{{{i}}}$' for i in Nx])
        # ax2 = ax.twinx()
        ax.set_yticks(Nx)
        ax.set_yticklabels([f'$2^{{{i}}}$' for i in Nx])
        ax.tick_params(right=True, top=False, labelright=False, labeltop=False, labelrotation=0)
        #   ax.set_aspect('equal')

        if row == 0:
            ax.text(-0.20, 1.15, rf'\textbf{{(c.{row + 1}) }}', transform=ax.transAxes)
        elif row == 1:
            ax.text(-0.20, 1.05, rf'\textbf{{(c.{row + 1}) }}', transform=ax.transAxes)
        # Adding a color bar with custom ticks and labels
        cbar_ax = fig.add_subplot(gs[row, 3])
        cbar = plt.colorbar(pcm, location='left', cax=cbar_ax, ticklocation='right')  # Specify the ticks
        # cbar.ax.invert_yaxis()
        # # cbar.set_ticks(ticks=[  0, 1,10])
        # cbar.set_ticks([10, 5, 2, 1, 1 / 2, 1 / 5, 1 / 10])
        # cbar.ax.set_yticklabels(
        #     ['Jacobi-Green \n needs less', '5 times', '2 times', 'Equal', '2 times', '5 times',
        #      'Jacobi-Green \n needs more'])

        #

        row += 1

    fname = figure_folder_path + 'JG_exp4_GRID_DEP_nb_its_geom_{}_rho_{}_norm_{}{}'.format('cos', phase_contrast, norm_,'.pdf')
    print(('create figure: {}'.format(fname)))
    plt.savefig(fname, bbox_inches='tight')

plt.show()
