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

script_name = 'exp_paper_JG_cos_time_scaling'

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

    # preconditioner_type = 'Jacobi'
    # results_name = (
    #         f'nb_nodes_{number_of_pixels}_' + f'nb_pixels_{nb_laminates}_' + f'contrast_{total_phase_contrast}_' + f'prec_{preconditioner_type}')
    # info = np.load(data_folder_path + results_name + f'.npz', allow_pickle=True)
    # try:
    #     time_J_1[i - 2, j - 2] = info.f.elapsed_time
    #     time_CG_J_1[i - 2, j - 2] = info.f.elapsed_time_CG
    #
    # except AttributeError:
    #     time_J_1[i - 2, j - 2] = 0
    #     time_CG_J_1[i - 2, j - 2] = 0
    #
    # nb_it_Jacobi_linear_1[i - 2, j - 2] = info.f.nb_steps

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
                          height_ratios=[1 ])
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
    #ax_time_per_it.set_xticklabels([fr'$2^{{{int(x)}}}$' for x in np.arange(nb_pixels, 11)])
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

    #plt.loglog(np.linspace(1e1, 1e8), 1e-3 * np.linspace(5e1, 1e8), 'k-', linewidth=0.9)

    # line2, = plt.loglog(nb_dofs, scaled_time_G[1:], '-x', color='Green', label='Green')
    # line3, = plt.loglog(nb_dofs, scaled_time_GJ[1:], 'k-', marker='o', markerfacecolor='none',
    #                     label='Green-Jacobi')
    # legend1 = plt.legend(handles=[line2, line3], loc='upper left', title='Wall-clock time')  # line1,

   # plt.gca().add_artist(legend1)  # Add the first legend manually

    line7, = ax_time_per_it.semilogx(nb_dofs, (time_per_iteration_GJ_1/time_per_iteration_G_1   )[1:], 'g--x',  #
                                   label='Total time')

    line7, = ax_time_per_it.semilogx(nb_dofs, (time_per_iteration_CG_GJ_1 / time_per_iteration_CG_G_1)[1:], 'g--x',  #
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
    #ax_time_per_it.set_ylim([1e-2, 1e5])
    ax_time_per_it.set_yscale('linear')

    fig.tight_layout()
    fname = f'time_scaling' + '{}'.format('.pdf')
    plt.savefig(figure_folder_path + script_name + fname, bbox_inches='tight')
    print(('create figure: {}'.format(figure_folder_path + script_name + fname)))

    plt.show()