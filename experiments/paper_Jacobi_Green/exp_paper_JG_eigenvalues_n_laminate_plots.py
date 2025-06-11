from cProfile import label

import numpy as np
import os
import scipy as sc
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpi4py import MPI
from NuMPI.Tools import Reduction

from NuMPI.IO import save_npy, load_npy
import matplotlib.pyplot as plt



script_name_save = 'exp_paper_JG_eigenvalues_n_laminate'
file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
data_folder_path = file_folder_path + '/exp_data/' + script_name_save + '/'

src = file_folder_path + '/figures/' + script_name_save + '/'


# Enable LaTeX rendering
plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX
    # "font.family": "helvetica",  # Use a serif font
})
plt.rcParams.update({'font.size': 11})
plt.rcParams["font.family"] = "Arial"

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library
from mpl_toolkits import mplot3d

plt.ion()

colors = ['red', 'blue', 'green', 'orange', 'purple']
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
# markers = ['x', 'o', '|', '>']
markers = ["x", "o", "v", "<", ">", "^", ".", ",", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+",
           "X", "D", "d", "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
           ]
#### plot mesh size independance
plot_this = True
if plot_this:
    geometry_ID = 'linear'  # 'linear'# 'sine_wave_'
    geometry_n = [2, ]
    discretization_n = [3, 4, 5]
    ratio = 2

    # create a figure
    fig = plt.figure(figsize=(8.3, 5.0))

    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.5, width_ratios=[1, 1.3, 1.3],
                          height_ratios=[1, 1, 1])
    ax_1 = fig.add_subplot(gs[:2, 0])
    ax_legend = fig.add_subplot(gs[2, 0])
    ax_legend.set_axis_off()
    ax_1.text(-0.5, 1.1, '(a.1)', transform=ax_1.transAxes)

    for ii in np.arange(np.size(geometry_n)):
        G = geometry_n[ii]
        j2 = [x for x in discretization_n if x >= G]
        counter = 0
        for T in j2:

            print(f'G={G}')
            print(f'T={T}')

            script_name = 'exp_paper_JG_eigenvalues_n_laminate'
            file_data_name = (
                f'{script_name}_gID{geometry_ID}_T{T}_G{G}_kappa{ratio}.npy')
            folder_name = '../exp_data/'

            xopt = np.load('../exp_data/' + file_data_name + f'xopt_log.npz', allow_pickle=True)
            phase_field = np.load('../exp_data/' + file_data_name + f'.npy', allow_pickle=True)

            #
            # ax_1.set_title(f'nb phases {2 ** (nb_starting_phases)}, nb pixels {number_of_pixels[0]}', wrap=True)

            ax_1.semilogy(np.arange(0, len(xopt.f.norm_rMr_G)), xopt.f.norm_rMr_G,
                          label=r'Green - ' + r' $\mathcal{T}$' + f'$_{{{2 ** T}}}$', color='green',
                          linestyle='--', marker=markers[counter])
            ax_1.semilogy(np.arange(0, len(xopt.f.norm_rMr_JG)), xopt.f.norm_rMr_JG,
                          label=r'Green-Jacobi - ' + r' $\mathcal{T}$' + f'$_{{{2 ** T}}}$',
                          color='Black', linestyle='-.', marker=markers[counter])
            ax_1.set_xlabel(r'PCG iteration - $k$')
            ax_1.set_ylabel('Norm of residua - '
                            r'$||r_{k}||_{G^{-1}} $')
            ax_1.set_title(f'Convergence \n ' + r'Geometry - $\mathcal{G}$' + f'$_{{{2 ** G}}}$')
            ax_1.set_ylim([1e-6, 1e4])
            ax_1.set_xlim([0, 9])

            if geometry_ID == 'sine_wave_':
                ax_1.set_xlim([1, 25])
                ax_1.set_xticks([1, 9, 13, 15, 22])
                ax_1.set_xticklabels([1, 9, 13, 15, 22])
            elif geometry_ID == 'linear':
                ax_1.set_xlim([0, 8])
                ax_1.set_xticks([0, 3, 6, 8])
                ax_1.set_xticklabels([0, 3, 6, 8])
            # -------------------------------------# plot eigenvals #-------------------------------------#

            eig_JG = np.real(xopt.f.eigens_JG)
            # Number of bins (fine-grained)
            num_bins = eig_JG.size
            # Define the bin width
            bin_width = 0.05

            # Calculate the bin edges
            min_edge = np.min(sorted(eig_JG)[2:])
            max_edge = np.max(sorted(eig_JG)[2:])
            bins = np.arange(min_edge, max_edge + bin_width, bin_width)
            bins = 100
            # Create the histogram
            # -------------------------------------# plot eigenvals #-------------------------------------#
            ax_eig_G = fig.add_subplot(gs[counter, 1])
            eig_G = xopt.f.eigens_G
            eig_JG = xopt.f.eigens_JG
            # Number of bins (fine-grained)
            num_bins = eig_G.size
            # Define the bin width
            bin_width = 0.05

            # Calculate the bin edges
            min_edge = np.min(sorted(eig_JG)[2:])
            max_edge = np.max(sorted(eig_JG)[2:])
            bins = np.arange(min_edge, max_edge + bin_width, bin_width)
            bins = 100
            ax_eig_G.plot(sorted(eig_G)[2:], color='Green', label=f'Green',
                      alpha=0.5, marker='.', linewidth=0, markersize=5)
            # ax_3.plot(sorted(np.real(eig_JG))[2:], color='Black', label=f'Jacobi-Green',
            #           alpha=0.5, marker='.', linewidth=0, markersize=5)

            ax_eig_G.text(-0.25, 1.05, f'(b.{counter + 1})', transform=ax_eig_G.transAxes)
            if ratio == 0:
                # ax_2.set_ylim([1e-4, max(sorted(eig_G)[2:])])
                ax_eig_G.set_ylim([0, 1e0])
                ax_eig_G.set_yticks([0, 0.5, 1])
                ax_eig_G.set_yticklabels([0, 0.5, 1])
            else:
                # ax_2.set_ylim([1e-4, max(sorted(eig_G)[2:])])
                ax_eig_G.set_ylim([0, 101])
                ax_eig_G.set_yticks([1, 34, 67, 100])
                ax_eig_G.set_yticklabels([1, 34, 67, 100])  # f'$10^{{{2}}}$'
                ax_eig_G.set_xlim([0, len(eig_G)])
                ax_eig_G.set_xticks([1, len(eig_G) // 2, len(eig_G)])
                ax_eig_G.set_xticklabels([1, len(eig_G) // 2, len(eig_G)])
                ax_eig_G.yaxis.set_ticks_position('right')
                # ax_3.set_title(r'Sorted eigenvalues')
            if counter == 0:
                ax_eig_G.set_title('Sorted eigenvalues \n Green')  # + r'Mesh -' + r'$\mathcal{T}$' + f'$_{{{2 ** T}}}$
            # elif counter == 1:
            #     ax_3.set_title(r'Mesh -' + r'$\mathcal{T}$' + f'$_{{{2 ** T}}}$')
            elif counter == 2:
                # ax_3.set_title(
                #     r'Mesh ' + r'$\mathcal{T}$' + f'$_{{{2 ** T}}}$')  # +f' (${{{2**T}}}$ nodes in x direction)'
                ax_eig_G.set_xlabel('eigenvalue index - $i$')
            ax_eig_G.set_ylabel('eigenvalue $\lambda_i$')
            ax_eig_G.text(0.05, 0.8,
                      r'Mesh - ' + r'$\mathcal{T}$' + f'$_{{{2 ** T}}}$',
                      transform=ax_eig_G.transAxes, fontsize=12)
            print(xopt.f.nb_of_pixels)
            print(xopt.f.nb_of_sampling_points)
            # print(xopt.f.norm_rMr_JG)


            # -------------------------------------# plot eigenvals #-------------------------------------#
            ax_weights = fig.add_subplot(gs[counter, 2])
            results_name = f'T{2 ** T}_G{2 ** G}'

            script_name = 'exp_paper_JG_eivals_with_eigenvectors'
            file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
            data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
            figure_folder_path = file_folder_path + '/figures/' + script_name + '/'
            _info = np.load(data_folder_path + results_name + f'_info.npz', allow_pickle=True)

            weight = _info['weights']

            x_values = _info['x_values']
            ritz_values = _info['ritz_values']
            eig_G_no_zero = np.real(_info['eig_G_no_zero'])


            ax_weights.scatter(np.real(weight) / np.real(eig_G_no_zero), np.real(eig_G_no_zero), color='blue',
                               marker='o', label=r"non-zero weights- $w_{i}/ \lambda_{i}$")
            ax_weights.set_xscale('log')

            ax_weights.set_yticks([1, 34, 67, 100])
            ax_weights.set_yticklabels([1, 34, 67, 100])

            ax_weights.set_xlim(1e-10, 1)
            ax_weights.set_xticks([1e-10,1e-5, 1])
            ax_weights.set_xticklabels([fr'$10^{{{-10}}}$',fr'$10^{{{-5}}}$',fr'$10^{{{0}}}$'])

            # ax_weights.semilogy(sorted(eig_JG)[2:], color='Black', label=f'Green-Jacobi',
            #               alpha=0.5, marker='.', linewidth=0, markersize=5)
            # ax_3.plot(sorted(np.real(eig_JG))[2:], color='Black', label=f'Jacobi-Green',
            #           alpha=0.5, marker='.', linewidth=0, markersize=5)

            ax_weights.text(-0.25, 1.05, f'(c.{counter + 1})', transform=ax_weights.transAxes)
            ax_weights.set_yticks([1, 34, 67, 100])
            ax_weights.set_yticklabels([1, 34, 67, 100])

            # ax_2.set_yticks([min_edge,  max_edge])
            # ax_2.set_yticklabels([1, 34, 67, 100])  # f'$10^{{{2}}}$'
            ax_weights.set_xlim(1e-10, 1)
            ax_weights.set_xticks([1e-10, 1e-5, 1])
            ax_weights.set_xticklabels([fr'$10^{{{-10}}}$', fr'$10^{{{-5}}}$', fr'$10^{{{0}}}$'])

            ax_weights.yaxis.set_ticks_position('right')  # Set y-axis ticks to the right

            if counter == 0:
                ax_weights.set_title(
                    'Weights \n Green ')  # \n' + r'Mesh -' + r'$\mathcal{T}$' + f'$_{{{2 ** T}}}$'
            elif counter == 2:

                ax_weights.set_xlabel(r'$w_{i}^{2}/ \lambda_{i}$')
            ax_weights.set_ylabel('eigenvalue $\lambda_i$')
            ax_weights.text(0.05, 0.8,
                      r'Mesh - ' + r'$\mathcal{T}$' + f'$_{{{2 ** T}}}$',
                      transform=ax_weights.transAxes, fontsize=12)



            counter += 1
        # legend = ax_1.legend()
        # legend.get_frame().set_alpha(1.)
        handles, labels = ax_1.get_legend_handles_labels()
        order = [0, 2, 4, 1, 3, 5]
        ax_legend.legend([handles[idx] for idx in order], [labels[idx] for idx in order], ncol=1)

    fname = src + script_name + 'ndep_G_JG_weights' + f'{ratio}_{geometry_ID}' + '{}'.format('.pdf')
    print(('create figure: {}'.format(fname)))
    plt.savefig(fname, bbox_inches='tight')
    plt.show()



    # -------------------------------------# Second plot#-------------------------------------#

#### plot mesh size independance
plot_size_dep = True
if plot_size_dep:
    geometry_ID = 'linear'  # 'linear'# 'sine_wave_' #'linear'
    geometry_n = [2, 3, 5]
    discretization_n = [5]
    ratio = 2
    plt.rcParams.update({'font.size': 11})
    plt.rcParams["font.family"] = "Arial"

    # create a figure
    fig = plt.figure(figsize=(8.3, 5.0))
    # gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.4, width_ratios=3 * (1,),
    #                       height_ratios=[1, 1, 1])
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.5, width_ratios=[1, 1.3, 1.3],
                          height_ratios=[1, 1, 1])
    ax_1 = fig.add_subplot(gs[:2, 0])
    ax_legend = fig.add_subplot(gs[2, 0])
    ax_legend.set_axis_off()

    ax_1.text(-0.5, 1.1, '(a.1)', transform=ax_1.transAxes)
    counter = 0
    for ii in np.arange(np.size(geometry_n)):
        G = geometry_n[ii]
        j2 = [x for x in discretization_n if x >= G]

        for T in j2:

            print(f'G={G}')
            print(f'T={T}')
            script_name = 'exp_paper_JG_eigenvalues_n_laminate'

            file_data_name = (
                f'{script_name}_gID{geometry_ID}_T{T}_G{G}_kappa{ratio}.npy')
            folder_name = '../exp_data/'

            xopt = np.load('../exp_data/' + file_data_name + f'xopt_log.npz', allow_pickle=True)
            phase_field = np.load('../exp_data/' + file_data_name + f'.npy', allow_pickle=True)

            #
            # ax_1.set_title(f'nb phases {2 ** (nb_starting_phases)}, nb pixels {number_of_pixels[0]}', wrap=True)

            ax_1.semilogy(np.arange(0, len(xopt.f.norm_rMr_G)), xopt.f.norm_rMr_G,
                          label=r'  Green - ' + r'$\mathcal{G}$' + f'$_{{{2 ** G}}}$', color='green',
                          linestyle='--', marker=markers[counter], markevery=3)
            ax_1.semilogy(np.arange(0, len(xopt.f.norm_rMr_JG)), xopt.f.norm_rMr_JG,
                          label=r'  Green-Jacobi - ' + r'$\mathcal{G}$' + f'$_{{{2 ** G}}}$',
                          color='Black', linestyle='-.', marker=markers[counter])
            ax_1.set_xlabel(r'PCG iteration - $k$')
            ax_1.set_ylabel('Norm of residua - '
                            r'$||r_{k}||_{G^{-1}} $')
            ax_1.set_title(f'Convergence \n ' + r'Mesh - $\mathcal{T}$' + f'$_{{{2 ** T}}}$')

            if geometry_ID == 'sine_wave_':
                ax_1.set_xlim([1, 60])
                ax_1.set_xticks([1, 8, 15, 18, 22, 30, 53, 60])
                ax_1.set_xticklabels([1, 8, 15, 18, 22, 30, 53, 60])
            elif geometry_ID == 'linear':
                ax_1.set_xlim([0, 24])
                ax_1.set_xticks([0, 3, 7, 24])
                ax_1.set_xticklabels([0, 3, 7, 24])
            ax_1.set_ylim([1e-6, 1e4])

            # -------------------------------------# plot eigenvals #-------------------------------------#
            ax_2 = fig.add_subplot(gs[counter, 1])
            ax_2.text(-0.25, 1.05, f'(b.{counter + 1})', transform=ax_2.transAxes)

            eig_G = xopt.f.eigens_G
            eig_JG = np.real(xopt.f.eigens_JG)
            # Number of bins (fine-grained)
            num_bins = eig_G.size
            # Define the bin width
            bin_width = 0.05

            # Calculate the bin edges
            min_edge = np.min(sorted(eig_JG)[2:])
            max_edge = np.max(sorted(eig_JG)[2:])
            bins = np.arange(min_edge, max_edge + bin_width, bin_width)
            bins = 100
            # Create plots

            ax_2.plot(sorted(eig_G)[2:], color='Green', label=f'Green',
                      alpha=0.5, marker='.', linewidth=0, markersize=5)
            # ax_3.plot(sorted(np.real(eig_JG))[2:], color='Black', label=f'Jacobi-Green',
            #           alpha=0.5, marker='.', linewidth=0, markersize=5)
            if ratio == 0:
                # ax_2.set_ylim([1e-4, max(sorted(eig_G)[2:])])
                ax_2.set_ylim([0, 1e0])
                ax_2.set_yticks([0, 0.5, 1])
                ax_2.set_yticklabels([0, 0.5, 1])
            else:
                ax_2.set_ylim([0, 101])
                ax_2.set_yticks([1, 34, 67, 100])
                ax_2.set_yticklabels([1, 34, 67, 100])  # f'$10^{{{2}}}$'
                ax_2.set_xlim([0, len(eig_G)])
                ax_2.set_xticks([1, len(eig_G) // 2, len(eig_G)])
                ax_2.set_xticklabels([1, len(eig_G) // 2, len(eig_G)])
                ax_2.yaxis.set_ticks_position('right')  # Set y-axis ticks to the right

            if counter == 0:
                ax_2.set_title(
                    'Sorted eigenvalues \n Green ')  # + r'Geometry- ' + r'$\mathcal{G}$' + f'$_{{{2 ** G}}}$')
            # elif counter == 1:
            # ax_2.set_title(r'Geometry -' + r'$\mathcal{G}$' + f'$_{{{2 ** G}}}$')
            elif counter == 2:
                # ax_2.set_title(r'Geometry -' + r'$\mathcal{G}$' + f'$_{{{2 ** G}}}$')
                ax_2.set_xlabel('eigenvalue index - $i$')
            ax_2.set_ylabel('eigenvalue $\lambda_i$')
            ax_2.text(0.05, 0.8,
                      r'Geometry - ' + r'$\mathcal{G}$' + f'$_{{{2 ** G}}}$',
                      transform=ax_2.transAxes, fontsize=12)
            print(xopt.f.nb_of_pixels)
            print(xopt.f.nb_of_sampling_points)

             # -------------------------------------# plot weights #-------------------------------------#
            ax_weights = fig.add_subplot(gs[counter, 2])
            results_name = f'T{2 ** T}_G{2 ** G}'

            script_name = 'exp_paper_JG_eivals_with_eigenvectors'
            file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
            data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
            figure_folder_path = file_folder_path + '/figures/' + script_name + '/'
            _info = np.load(data_folder_path + results_name + f'_info.npz', allow_pickle=True)

            weight = _info['weights']

            x_values = _info['x_values']
            ritz_values = _info['ritz_values']
            eig_G_no_zero = np.real(_info['eig_G_no_zero'])

            ax_weights.scatter(np.real(weight) / np.real(eig_G_no_zero), np.real(eig_G_no_zero), color='blue',
                               marker='o', label=r"non-zero weights- $w_{i}/ \lambda_{i}$")
            ax_weights.set_xscale('log')

            ax_weights.set_yticks([1, 34, 67, 100])
            ax_weights.set_yticklabels([1, 34, 67, 100])

            ax_weights.set_xlim(1e-10, 1)
            ax_weights.set_xticks([1e-10, 1e-5, 1])
            ax_weights.set_xticklabels([fr'$10^{{{-10}}}$', fr'$10^{{{-5}}}$', fr'$10^{{{0}}}$'])

            # ax_weights.semilogy(sorted(eig_JG)[2:], color='Black', label=f'Green-Jacobi',
            #               alpha=0.5, marker='.', linewidth=0, markersize=5)
            # ax_3.plot(sorted(np.real(eig_JG))[2:], color='Black', label=f'Jacobi-Green',
            #           alpha=0.5, marker='.', linewidth=0, markersize=5)

            ax_weights.text(-0.25, 1.05, f'(c.{counter + 1})', transform=ax_weights.transAxes)
            ax_weights.set_yticks([1, 34, 67, 100])
            ax_weights.set_yticklabels([1, 34, 67, 100])

            # ax_2.set_yticks([min_edge,  max_edge])
            # ax_2.set_yticklabels([1, 34, 67, 100])  # f'$10^{{{2}}}$'
            ax_weights.set_xlim(1e-10, 1)
            ax_weights.set_xticks([1e-10, 1e-5, 1])
            ax_weights.set_xticklabels([fr'$10^{{{-10}}}$', fr'$10^{{{-5}}}$', fr'$10^{{{0}}}$'])

            ax_weights.yaxis.set_ticks_position('right')  # Set y-axis ticks to the right

            if counter == 0:
                ax_weights.set_title(
                    'Weights \n Green ')  # \n' + r'Mesh -' + r'$\mathcal{T}$' + f'$_{{{2 ** T}}}$'
            elif counter == 2:

                ax_weights.set_xlabel(r'$w_{i}^{2}/ \lambda_{i}$')
            ax_weights.set_ylabel('eigenvalue $\lambda_i$')

            if counter == 2:
                ax_weights.text(0.05, 0.65,
                                f'Geometry \n - ' + r'$\mathcal{G}$' + f'$_{{{2 ** G}}}$',
                                transform=ax_weights.transAxes, fontsize=12)
            else:
                ax_weights.text(0.05, 0.8,
                      f'Geometry  - ' + r'$\mathcal{G}$' + f'$_{{{2 ** G}}}$',
                      transform=ax_weights.transAxes, fontsize=12)

            print(xopt.f.nb_of_pixels)
            print(xopt.f.nb_of_sampling_points)
            # print(xopt.f.norm_rMr_JG)
            counter += 1

    # ax_1.legend(loc='upper right')
    handles, labels = ax_1.get_legend_handles_labels()
    order = [0, 2, 4, 1, 3, 5]
    ax_legend.legend([handles[idx] for idx in order], [labels[idx] for idx in order], ncol=1)

    fname = src + script_name + 'phasedep_G_JG_weights' + f'{ratio}_{geometry_ID}' + '{}'.format('.pdf')
    print(('create figure: {}'.format(fname)))
    plt.savefig(fname, bbox_inches='tight')
    plt.show()



#### plot mesh size independance
plot_this = True
if plot_this:
    geometry_ID = 'linear'  # 'linear'# 'sine_wave_'
    geometry_n = [2, ]
    discretization_n = [3, 4, 5]

    ratio = 2

    # create a figure
    fig = plt.figure(figsize=(8.3, 5.0))

    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.5, width_ratios=[1, 1.2, 1.2],
                          height_ratios=[1, 1, 1])
    ax_1 = fig.add_subplot(gs[:2, 0])
    ax_legend = fig.add_subplot(gs[2, 0])
    ax_legend.set_axis_off()
    ax_1.text(-0.5, 1.1, '(a.1)', transform=ax_1.transAxes)

    for ii in np.arange(np.size(geometry_n)):
        G = geometry_n[ii]
        j2 = [x for x in discretization_n if x >= G]
        counter = 0
        for T in j2:

            print(f'G={G}')
            print(f'T={T}')
            script_name = 'exp_paper_JG_eigenvalues_n_laminate'

            file_data_name = (
                f'{script_name}_gID{geometry_ID}_T{T}_G{G}_kappa{ratio}.npy')
            folder_name = '../exp_data/'

            xopt = np.load('../exp_data/' + file_data_name + f'xopt_log.npz', allow_pickle=True)
            phase_field = np.load('../exp_data/' + file_data_name + f'.npy', allow_pickle=True)
            print(f'../exp_data/' + file_data_name + f'xopt_log.npz')
            #
            # ax_1.set_title(f'nb phases {2 ** (nb_starting_phases)}, nb pixels {number_of_pixels[0]}', wrap=True)

            ax_1.semilogy(np.arange(0, len(xopt.f.norm_rMr_G)), xopt.f.norm_rMr_G,
                          label=r'  Green - ' + r' $\mathcal{T}$' + f'$_{{{2 ** T}}}$', color='green',
                          linestyle='--', marker=markers[counter])
            ax_1.semilogy(np.arange(0, len(xopt.f.norm_rMr_JG)), xopt.f.norm_rMr_JG,
                          label=r' Green-Jacobi - ' + r' $\mathcal{T}$' + f'$_{{{2 ** T}}}$',
                          color='Black', linestyle='-.', marker=markers[counter])
            ax_1.set_xlabel(r'PCG iteration - $k$')
            ax_1.set_ylabel('Norm of residua - '
                            r'$||r_{k}||_{G^{-1}} $')
            ax_1.set_title(f'Convergence \n ' + r'Geometry - $\mathcal{G}$' + f'$_{{{2 ** G}}}$')
            ax_1.set_ylim([1e-6, 1e4])
            ax_1.set_xlim([0, 9])

            if geometry_ID == 'sine_wave_':
                ax_1.set_xlim([1, 25])
                ax_1.set_xticks([1, 9, 13, 15, 22])
                ax_1.set_xticklabels([1, 9, 13, 15, 22])
            elif geometry_ID == 'linear':
                ax_1.set_xlim([0, 8])
                ax_1.set_xticks([0, 3, 6, 8])
                ax_1.set_xticklabels([0, 3, 6, 8])
            # -------------------------------------# plot eigenvals #-------------------------------------#

            eig_JG = np.real(xopt.f.eigens_JG)
            # Number of bins (fine-grained)
            num_bins = eig_JG.size
            # Define the bin width
            bin_width = 0.05

            # Calculate the bin edges
            min_edge = np.min(sorted(eig_JG)[2:])
            max_edge = np.max(sorted(eig_JG)[2:])
            bins = np.arange(min_edge, max_edge + bin_width, bin_width)
            bins = 100
            # -------------------------------------# plot eigenvals #-------------------------------------#
            ax_2 = fig.add_subplot(gs[counter, 2])
            ax_2.semilogy(sorted(eig_JG)[2:], color='Black', label=f'Green-Jacobi',
                          alpha=0.5, marker='.', linewidth=0, markersize=5)
            # ax_3.plot(sorted(np.real(eig_JG))[2:], color='Black', label=f'Jacobi-Green',
            #           alpha=0.5, marker='.', linewidth=0, markersize=5)

            ax_2.text(-0.25, 1.05, f'(c.{counter + 1})', transform=ax_2.transAxes)
            if ratio == 0:
                # ax_2.set_ylim([1e-4, max(sorted(eig_G)[2:])])
                ax_2.set_ylim([min_edge, max_edge])
                ax_2.set_yticks([0, 0.5, 1])
                ax_2.set_yticklabels([0, 0.5, 1])
            else:
                # ax_2.set_ylim([1e-4, max(sorted(eig_G)[2:])])
                ax_2.set_ylim([min_edge, max_edge])
                ax_2.set_ylim([1e-2, 1e2])

                # ax_2.set_yticks([min_edge,  max_edge])
                # ax_2.set_yticklabels([1, 34, 67, 100])  # f'$10^{{{2}}}$'
                ax_2.set_xlim([0, len(eig_JG)])
                ax_2.set_xticks([1, len(eig_JG) // 2, len(eig_JG)])
                ax_2.set_xticklabels([1, len(eig_JG) // 2, len(eig_JG)])
                ax_2.yaxis.set_ticks_position('right')  # Set y-axis ticks to the right

            if counter == 0:
                ax_2.set_title(
                    'Sorted eigenvalues \n Green-Jacobi ')  # \n' + r'Mesh -' + r'$\mathcal{T}$' + f'$_{{{2 ** T}}}$'
            elif counter == 2:

                ax_2.set_xlabel('eigenvalue index - $i$')
            ax_2.set_ylabel('eigenvalue $\lambda_i$')
            ax_2.text(0.05, 0.8,
                      r'Mesh - ' + r'$\mathcal{T}$' + f'$_{{{2 ** T}}}$',
                      transform=ax_2.transAxes, fontsize=13)

            # Create the histogram

            # -------------------------------------# plot eigenvals #-------------------------------------#
            ax_3 = fig.add_subplot(gs[counter, 1])
            eig_G = xopt.f.eigens_G
            eig_JG = xopt.f.eigens_JG
            # Number of bins (fine-grained)
            num_bins = eig_G.size
            # Define the bin width
            bin_width = 0.05

            # Calculate the bin edges
            min_edge = np.min(sorted(eig_JG)[2:])
            max_edge = np.max(sorted(eig_JG)[2:])
            bins = np.arange(min_edge, max_edge + bin_width, bin_width)
            bins = 100
            ax_3.plot(sorted(eig_G)[2:], color='Green', label=f'Green',
                      alpha=0.5, marker='.', linewidth=0, markersize=5)
            # ax_3.plot(sorted(np.real(eig_JG))[2:], color='Black', label=f'Jacobi-Green',
            #           alpha=0.5, marker='.', linewidth=0, markersize=5)

            ax_3.text(-0.25, 1.05, f'(b.{counter + 1})', transform=ax_3.transAxes)
            if ratio == 0:
                # ax_2.set_ylim([1e-4, max(sorted(eig_G)[2:])])
                ax_3.set_ylim([0, 1e0])
                ax_3.set_yticks([0, 0.5, 1])
                ax_3.set_yticklabels([0, 0.5, 1])
            else:
                # ax_2.set_ylim([1e-4, max(sorted(eig_G)[2:])])
                ax_3.set_ylim([0, 101])
                ax_3.set_yticks([1, 34, 67, 100])
                ax_3.set_yticklabels([1, 34, 67, 100])  # f'$10^{{{2}}}$'
                ax_3.set_xlim([0, len(eig_G)])
                ax_3.set_xticks([1, len(eig_G) // 2, len(eig_G)])
                ax_3.set_xticklabels([1, len(eig_G) // 2, len(eig_G)])
                ax_3.yaxis.set_ticks_position('right')
                # ax_3.set_title(r'Sorted eigenvalues')
            if counter == 0:
                ax_3.set_title('Sorted eigenvalues \n Green')  # + r'Mesh -' + r'$\mathcal{T}$' + f'$_{{{2 ** T}}}$
            # elif counter == 1:
            #     ax_3.set_title(r'Mesh -' + r'$\mathcal{T}$' + f'$_{{{2 ** T}}}$')
            elif counter == 2:
                # ax_3.set_title(
                #     r'Mesh ' + r'$\mathcal{T}$' + f'$_{{{2 ** T}}}$')  # +f' (${{{2**T}}}$ nodes in x direction)'
                ax_3.set_xlabel('eigenvalue index - $i$')
            ax_3.set_ylabel('eigenvalue $\lambda_i$')
            ax_3.text(0.05, 0.8,
                      r'Mesh - ' + r'$\mathcal{T}$' + f'$_{{{2 ** T}}}$',
                      transform=ax_3.transAxes, fontsize=13)
            print(xopt.f.nb_of_pixels)
            print(xopt.f.nb_of_sampling_points)
            # print(xopt.f.norm_rMr_JG)
            counter += 1
        # legend = ax_1.legend()
        # legend.get_frame().set_alpha(1.)
        handles, labels = ax_1.get_legend_handles_labels()
        order = [0, 2, 4, 1, 3, 5]
        ax_legend.legend([handles[idx] for idx in order], [labels[idx] for idx in order], ncol=1)

    fname = src + script_name + 'ndep_G_JG' + f'{ratio}_{geometry_ID}' + '{}'.format('.pdf')
    print(('create figure: {}'.format(fname)))
    plt.savefig(fname, bbox_inches='tight')
    plt.show()
    # -------------------------------------# Second plot#-------------------------------------#

#### plot mesh size independance
plot_size_dep = True
if plot_size_dep:
    geometry_ID = 'linear'  # 'linear'# 'sine_wave_' #'linear'
    geometry_n = [2, 3, 5]
    discretization_n = [5]
    ratio = 2
    plt.rcParams.update({'font.size': 11})
    plt.rcParams["font.family"] = "Arial"

    # create a figure
    fig = plt.figure(figsize=(8.3, 5.0))
    # gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.4, width_ratios=3 * (1,),
    #                       height_ratios=[1, 1, 1])
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.5, width_ratios=[1, 1.2, 1.2],
                          height_ratios=[1, 1, 1])
    ax_1 = fig.add_subplot(gs[:2, 0])
    ax_legend = fig.add_subplot(gs[2, 0])
    ax_legend.set_axis_off()

    ax_1.text(-0.5, 1.1, '(a.1)', transform=ax_1.transAxes)
    counter = 0
    for ii in np.arange(np.size(geometry_n)):
        G = geometry_n[ii]
        j2 = [x for x in discretization_n if x >= G]

        for T in j2:

            print(f'G={G}')
            print(f'T={T}')
            file_data_name = (
                f'{script_name}_gID{geometry_ID}_T{T}_G{G}_kappa{ratio}.npy')
            folder_name = '../exp_data/'

            xopt = np.load('../exp_data/' + file_data_name + f'xopt_log.npz', allow_pickle=True)
            phase_field = np.load('../exp_data/' + file_data_name + f'.npy', allow_pickle=True)

            #
            # ax_1.set_title(f'nb phases {2 ** (nb_starting_phases)}, nb pixels {number_of_pixels[0]}', wrap=True)

            ax_1.semilogy(np.arange(0, len(xopt.f.norm_rMr_G)), xopt.f.norm_rMr_G,
                          label=r'  Green - ' + r'$\mathcal{G}$' + f'$_{{{2 ** G}}}$', color='green',
                          linestyle='--', marker=markers[counter], markevery=3)
            ax_1.semilogy(np.arange(0, len(xopt.f.norm_rMr_JG)), xopt.f.norm_rMr_JG,
                          label=r'  Green-Jacobi - ' + r'$\mathcal{G}$' + f'$_{{{2 ** G}}}$',
                          color='Black', linestyle='-.', marker=markers[counter])
            ax_1.set_xlabel(r'PCG iteration - $k$')
            ax_1.set_ylabel('Norm of residua - '
                            r'$||r_{k}||_{G^{-1}} $')
            ax_1.set_title(f'Convergence \n ' + r'Mesh - $\mathcal{T}$' + f'$_{{{2 ** T}}}$')

            if geometry_ID == 'sine_wave_':
                ax_1.set_xlim([1, 60])
                ax_1.set_xticks([1, 8, 15, 18, 22, 30, 53, 60])
                ax_1.set_xticklabels([1, 8, 15, 18, 22, 30, 53, 60])
            elif geometry_ID == 'linear':
                ax_1.set_xlim([0, 24])
                ax_1.set_xticks([0, 3, 7, 24])
                ax_1.set_xticklabels([0, 3, 7, 24])
            ax_1.set_ylim([1e-6, 1e4])

            # -------------------------------------# plot eigenvals #-------------------------------------#
            ax_2 = fig.add_subplot(gs[counter, 1])
            ax_2.text(-0.25, 1.05, f'(b.{counter + 1})', transform=ax_2.transAxes)

            eig_G = xopt.f.eigens_G
            eig_JG = np.real(xopt.f.eigens_JG)
            # Number of bins (fine-grained)
            num_bins = eig_G.size
            # Define the bin width
            bin_width = 0.05

            # Calculate the bin edges
            min_edge = np.min(sorted(eig_JG)[2:])
            max_edge = np.max(sorted(eig_JG)[2:])
            bins = np.arange(min_edge, max_edge + bin_width, bin_width)
            bins = 100
            # Create plots

            ax_2.plot(sorted(eig_G)[2:], color='Green', label=f'Green',
                      alpha=0.5, marker='.', linewidth=0, markersize=5)
            # ax_3.plot(sorted(np.real(eig_JG))[2:], color='Black', label=f'Jacobi-Green',
            #           alpha=0.5, marker='.', linewidth=0, markersize=5)
            if ratio == 0:
                # ax_2.set_ylim([1e-4, max(sorted(eig_G)[2:])])
                ax_2.set_ylim([0, 1e0])
                ax_2.set_yticks([0, 0.5, 1])
                ax_2.set_yticklabels([0, 0.5, 1])
            else:
                ax_2.set_ylim([0, 101])
                ax_2.set_yticks([1, 34, 67, 100])
                ax_2.set_yticklabels([1, 34, 67, 100])  # f'$10^{{{2}}}$'
                ax_2.set_xlim([0, len(eig_G)])
                ax_2.set_xticks([1, len(eig_G) // 2, len(eig_G)])
                ax_2.set_xticklabels([1, len(eig_G) // 2, len(eig_G)])
                ax_2.yaxis.set_ticks_position('right')  # Set y-axis ticks to the right

            if counter == 0:
                ax_2.set_title(
                    'Sorted eigenvalues \n Green ')  # + r'Geometry- ' + r'$\mathcal{G}$' + f'$_{{{2 ** G}}}$')
            # elif counter == 1:
            # ax_2.set_title(r'Geometry -' + r'$\mathcal{G}$' + f'$_{{{2 ** G}}}$')
            elif counter == 2:
                # ax_2.set_title(r'Geometry -' + r'$\mathcal{G}$' + f'$_{{{2 ** G}}}$')
                ax_2.set_xlabel('eigenvalue index - $i$')
            ax_2.set_ylabel('eigenvalue $\lambda_i$')
            ax_2.text(0.05, 0.8,
                      r'Geometry - ' + r'$\mathcal{G}$' + f'$_{{{2 ** G}}}$',
                      transform=ax_2.transAxes, fontsize=11)
            print(xopt.f.nb_of_pixels)
            print(xopt.f.nb_of_sampling_points)

            # -------------------------------------# plot eigenvals #-------------------------------------#
            ax_3 = fig.add_subplot(gs[counter, 2])
            ax_3.text(-0.25, 1.05, f'(c.{counter + 1})', transform=ax_3.transAxes)

            ax_3.semilogy(sorted(eig_JG)[2:], color='Black', label=f'Green-Jacobi',
                          alpha=0.5, marker='.', linewidth=0, markersize=5)
            # ax_3.plot(sorted(np.real(eig_JG))[2:], color='Black', label=f'Jacobi-Green',
            #           alpha=0.5, marker='.', linewidth=0, markersize=5)

            if ratio == 0:
                # ax_2.set_ylim([1e-4, max(sorted(eig_G)[2:])])
                ax_3.set_ylim([0, 1e0])
                ax_3.set_yticks([0, 0.5, 1])
                ax_3.set_yticklabels([0, 0.5, 1])
            else:
                ax_3.set_ylim([min_edge, max_edge])
                ax_3.set_ylim([1e-2, 1e2])
                # ax_2.set_yticks([min_edge,  max_edge])
                # ax_2.set_yticklabels([1, 34, 67, 100])  # f'$10^{{{2}}}$'
                ax_3.set_xlim([-5, len(eig_JG) + 5])
                ax_3.set_xticks([1, len(eig_JG) // 2, len(eig_JG)])
                ax_3.set_xticklabels([1, len(eig_JG) // 2, len(eig_JG)])

                # ax_3.set_ylim([0, 101])
                # ax_3.set_yticks([1, 34, 67, 100])
                # ax_3.set_yticklabels([1, 34, 67, 100])  # f'$10^{{{2}}}$'
                # ax_3.set_xlim([0, len(eig_JG)])
                # ax_3.set_xticks([1, len(eig_JG) // 2, len(eig_JG)])
                # ax_3.set_xticklabels([1, len(eig_JG) // 2, len(eig_JG)])
                ax_3.yaxis.set_ticks_position('right')  # Set y-axis ticks to the right

            if counter == 0:
                ax_3.set_title(
                    'Sorted eigenvalues \n Green-Jacobi')  # + r'Geometry- ' + r'$\mathcal{G}$' + f'$_{{{2 ** G}}}$
            #   elif counter == 1:
            # ax_3.set_title(r'Geometry -' + r'$\mathcal{G}$' + f'$_{{{2 ** G}}}$')
            elif counter == 2:
                # ax_3.set_title(r'Geometry -' + r'$\mathcal{G}$' + f'$_{{{2 ** G}}}$')
                ax_3.set_xlabel('eigenvalue index - $i$')
            ax_3.set_ylabel('eigenvalue $\lambda_i$')
            ax_3.text(0.05, 0.8,
                      r'Geometry - ' + r'$\mathcal{G}$' + f'$_{{{2 ** G}}}$',
                      transform=ax_3.transAxes, fontsize=13)

            print(xopt.f.nb_of_pixels)
            print(xopt.f.nb_of_sampling_points)
            # print(xopt.f.norm_rMr_JG)
            counter += 1

    # ax_1.legend(loc='upper right')
    handles, labels = ax_1.get_legend_handles_labels()
    order = [0, 2, 4, 1, 3, 5]
    ax_legend.legend([handles[idx] for idx in order], [labels[idx] for idx in order], ncol=1)

    fname = src + script_name + 'phasedep_G_JG' + f'{ratio}_{geometry_ID}' + '{}'.format('.pdf')
    print(('create figure: {}'.format(fname)))
    plt.savefig(fname, bbox_inches='tight')
    plt.show()
    quit()

    ##### 3 column plot
    geometry_ID = 'linear'  # 'linear'# 'sine_wave_'
    geometry_n = [2, ]
    discretization_n = [3, 4, 5]
    ratio = 2

    # create a figure
    fig = plt.figure(figsize=(11, 4.5))
    gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.4, width_ratios=3 * (1,),
                          height_ratios=[1, 1, 1])
    ax_1 = fig.add_subplot(gs[:, 0])
    ax_1.text(-0.25, 1.05, '(a.1)', transform=ax_1.transAxes)

    for ii in np.arange(np.size(geometry_n)):
        G = geometry_n[ii]
        j2 = [x for x in discretization_n if x >= G]
        counter = 0
        for T in j2:

            print(f'G={G}')
            print(f'T={T}')
            file_data_name = (
                f'{script_name}_gID{geometry_ID}_T{T}_G{G}_kappa{ratio}.npy')
            folder_name = '../exp_data/'

            xopt = np.load('../exp_data/' + file_data_name + f'xopt_log.npz', allow_pickle=True)
            phase_field = np.load('../exp_data/' + file_data_name + f'.npy', allow_pickle=True)

            #
            # ax_1.set_title(f'nb phases {2 ** (nb_starting_phases)}, nb pixels {number_of_pixels[0]}', wrap=True)

            ax_1.semilogy(np.arange(0, len(xopt.f.norm_rMr_G)), xopt.f.norm_rMr_G,
                          label=r'  Green ' + r'$\mathcal{T}$' + f'$_{{{2 ** T}}}$', color='green',
                          linestyle='--', marker=markers[counter])
            ax_1.semilogy(np.arange(0, len(xopt.f.norm_rMr_JG)), xopt.f.norm_rMr_JG,
                          label=r' Jacobi-Green  ' + r'$\mathcal{T}$' + f'$_{{{2 ** T}}}$',
                          color='Black', linestyle='-.', marker=markers[counter])
            ax_1.set_xlabel('PCG iteration - k')
            ax_1.set_ylabel('Norm of residua - '
                            r'$||r_{k}||_{G^{-1}} $')
            ax_1.set_title(f'Convergence \n ' + r'Geometry - $\mathcal{G}$' + f'$_{{{2 ** G}}}$')
            ax_1.set_ylim([1e-6, 1e4])
            ax_1.set_xlim([0, 9])

            if geometry_ID == 'sine_wave_':
                ax_1.set_xlim([1, 25])
                ax_1.set_xticks([1, 9, 13, 15, 22])
                ax_1.set_xticklabels([1, 9, 13, 15, 22])
            elif geometry_ID == 'linear':
                ax_1.set_xlim([0, 9])
                ax_1.set_xticks([0, 3, 7, 9])
                ax_1.set_xticklabels([0, 3, 7, 9])
            # -------------------------------------# plot eigenvals #-------------------------------------#
            ax_2 = fig.add_subplot(gs[counter, 1])
            ax_2.text(-0.2, 1.15, f'(b.{counter + 1})', transform=ax_2.transAxes)

            eig_G = xopt.f.eigens_G
            eig_JG = xopt.f.eigens_JG
            # Number of bins (fine-grained)
            num_bins = eig_G.size
            # Define the bin width
            bin_width = 0.05

            # Calculate the bin edges
            min_edge = np.min(sorted(eig_JG)[2:])
            max_edge = np.max(sorted(eig_JG)[2:])
            bins = np.arange(min_edge, max_edge + bin_width, bin_width)
            bins = 100
            # Create the histogram
            ax_2.hist(sorted(eig_G)[2:], bins=bins, color='Green', label=f'Green', edgecolor='Green',
                      alpha=0.99)  # , marker='.', linewidth=0, markersize=5)
            # ax_2.hist(sorted(np.real(eig_JG))[2:], bins=bins, color='Black', label=f'Jacobi-Green', edgecolor='black',
            #          alpha=0.2)  # , marker='.', linewidth=0, markersize=5)
            # ax_3.plot(sorted(eig_G)[2:],np.zeros_like(eig_G[2:]), color='red', label=f'Green',
            #           alpha=0.5, marker='.', linewidth=0, markersize=5)
            # ax_3.plot(sorted(eig_JG)[2:],np.ones_like(eig_JG[2:]), color='b', label=f'Jacobi-Green',
            #           alpha=0.5, marker='.', linewidth=0, markersize=5)

            # ax_3.set_ylim([1e-4, 1e2])  #
            if ratio == 0:
                # ax_2.set_ylim([1e-4, max(sorted(eig_G)[2:])])
                ax_2.set_ylim([0, 1e0])
                ax_2.set_yticks([0, 0.5, 1])
                ax_2.set_yticklabels([0, 0.5, 1])
            else:
                # ax_2.set_ylim([1e-4, max(sorted(eig_G)[2:])])
                ax_2.set_ylim([0, 101])
                ax_2.set_yticks([1, 34, 67, 100])
                ax_2.set_yticklabels([1, 34, 67, 100])  # f'$10^{{{2}}}$'
                ax_2.set_xlim([0, len(eig_G)])
                ax_2.set_xticks([1, len(eig_G) // 2, len(eig_G)])
                ax_2.set_xticklabels([1, len(eig_G) // 2, len(eig_G)])
            if counter == 0:
                ax_2.set_title('Histograms of eigenvalues \n' + r'Mesh -' + r'$\mathcal{T}$' + f'$_{{{2 ** T}}}$')
            elif counter == 1:
                ax_2.set_title(r'Mesh -' + r'$\mathcal{T}$' + f'$_{{{2 ** T}}}$')
            elif counter == 2:
                ax_2.set_title(r'Mesh -' + r'$\mathcal{T}$' + f'$_{{{2 ** T}}}$')
                ax_2.set_xlabel('eigenvalue $\lambda_i$' + f' (in {bins}bins)')
            ax_2.set_ylabel('\# of eigenvalues')

            #
            ax_2.set_ylim([1, 1e3])
            # ax_2.set_xtics([1e-4, 1e0])
            ax_2.set_yscale('log')
            # ax_2.set_xscale('log')
            # -------------------------------------# plot eigenvals #-------------------------------------#
            ax_3 = fig.add_subplot(gs[counter, 2])
            ax_3.plot(sorted(eig_G)[2:], color='Green', label=f'Green',
                      alpha=0.5, marker='.', linewidth=0, markersize=5)
            # ax_3.plot(sorted(np.real(eig_JG))[2:], color='Black', label=f'Jacobi-Green',
            #           alpha=0.5, marker='.', linewidth=0, markersize=5)

            ax_3.text(-0.2, 1.15, f'(c.{counter + 1})', transform=ax_3.transAxes)
            if ratio == 0:
                # ax_2.set_ylim([1e-4, max(sorted(eig_G)[2:])])
                ax_3.set_ylim([0, 1e0])
                ax_3.set_yticks([0, 0.5, 1])
                ax_3.set_yticklabels([0, 0.5, 1])
            else:
                # ax_2.set_ylim([1e-4, max(sorted(eig_G)[2:])])
                ax_3.set_ylim([0, 101])
                ax_3.set_yticks([1, 34, 67, 100])
                ax_3.set_yticklabels([1, 34, 67, 100])  # f'$10^{{{2}}}$'
                ax_3.set_xlim([0, len(eig_G)])
                ax_3.set_xticks([1, len(eig_G) // 2, len(eig_G)])
                ax_3.set_xticklabels([1, len(eig_G) // 2, len(eig_G)])

            # ax_3.set_title(r'Sorted eigenvalues')
            if counter == 0:
                ax_3.set_title('Sorted eigenvalues \n' + r'Mesh -' + r'$\mathcal{T}$' + f'$_{{{2 ** T}}}$')
            elif counter == 1:
                ax_3.set_title(r'Mesh -' + r'$\mathcal{T}$' + f'$_{{{2 ** T}}}$')
            elif counter == 2:
                ax_3.set_title(
                    r'Mesh ' + r'$\mathcal{T}$' + f'$_{{{2 ** T}}}$')  # +f' (${{{2**T}}}$ nodes in x direction)'
                ax_3.set_xlabel('eigenvalue index - $i$')
            ax_3.set_ylabel('eigenvalue $\lambda_i$')
            print(xopt.f.nb_of_pixels)
            print(xopt.f.nb_of_sampling_points)
            # print(xopt.f.norm_rMr_JG)
            counter += 1
        ax_1.legend()
    fname = src + script_name + 'ndep' + f'{ratio}_{geometry_ID}' + '{}'.format('.pdf')
    print(('create figure: {}'.format(fname)))
    plt.savefig(fname, bbox_inches='tight')
    plt.show()

    # -------------------------------------# Second plot#-------------------------------------#
    # -------------------------------------# Second plot#-------------------------------------#
    # old one
    geometry_ID = 'linear'  # 'linear'# 'sine_wave_' #'linear'
    geometry_n = [2, 3, 5]
    discretization_n = [5]
    ratio = 2

    # create a figure
    fig = plt.figure(figsize=(11, 4.5))
    gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.4, width_ratios=3 * (1,),
                          height_ratios=[1, 1, 1])
    ax_1 = fig.add_subplot(gs[:, 0])

    ax_1.text(-0.25, 1.05, '(a.1)', transform=ax_1.transAxes)
    counter = 0
    for ii in np.arange(np.size(geometry_n)):
        G = geometry_n[ii]
        j2 = [x for x in discretization_n if x >= G]

        for T in j2:

            print(f'G={G}')
            print(f'T={T}')
            file_data_name = (
                f'{script_name}_gID{geometry_ID}_T{T}_G{G}_kappa{ratio}.npy')
            folder_name = '../exp_data/'

            xopt = np.load('../exp_data/' + file_data_name + f'xopt_log.npz', allow_pickle=True)
            phase_field = np.load('../exp_data/' + file_data_name + f'.npy', allow_pickle=True)

            #
            # ax_1.set_title(f'nb phases {2 ** (nb_starting_phases)}, nb pixels {number_of_pixels[0]}', wrap=True)

            ax_1.semilogy(np.arange(1, len(xopt.f.norm_rMr_G) + 1), xopt.f.norm_rMr_G,
                          label=r'  Green ' + r'$\mathcal{G}$' + f'$_{{{2 ** G}}}$', color='green',
                          linestyle='--', marker=markers[counter])
            ax_1.semilogy(np.arange(1, len(xopt.f.norm_rMr_JG) + 1), xopt.f.norm_rMr_JG,
                          label=r' Jacobi-Green  ' + r'$\mathcal{G}$' + f'$_{{{2 ** G}}}$',
                          color='Black', linestyle='-.', marker=markers[counter])
            ax_1.set_xlabel('PCG iteration - k')
            ax_1.set_ylabel('Norm of residua - '
                            r'$||r_{k}||_{G^{-1}} $')
            ax_1.set_title(f'Convergence \n ' + r'Mesh - $\mathcal{T}$' + f'$_{{{2 ** T}}}$')

            if geometry_ID == 'sine_wave_':
                ax_1.set_xlim([1, 60])
                ax_1.set_xticks([1, 8, 15, 18, 22, 30, 53, 60])
                ax_1.set_xticklabels([1, 8, 15, 18, 22, 30, 53, 60])
            elif geometry_ID == 'linear':
                ax_1.set_xlim([1, 30])
                ax_1.set_xticks([1, 4, 8, 9, 14, 30])
                ax_1.set_xticklabels([1, 4, 8, 9, 14, 30])
            ax_1.set_ylim([1e-6, 1e4])

            # -------------------------------------# plot eigenvals #-------------------------------------#
            ax_2 = fig.add_subplot(gs[counter, 1])
            ax_2.text(-0.25, 1.15, f'(b.{counter + 1})', transform=ax_2.transAxes)

            eig_G = xopt.f.eigens_G
            eig_JG = xopt.f.eigens_JG
            # Number of bins (fine-grained)
            num_bins = eig_G.size
            # Define the bin width
            bin_width = 0.05

            # Calculate the bin edges
            min_edge = np.min(sorted(eig_JG)[2:])
            max_edge = np.max(sorted(eig_JG)[2:])
            bins = np.arange(min_edge, max_edge + bin_width, bin_width)
            bins = 100
            # Create the histogram
            ax_2.hist(sorted(eig_G)[2:], bins=bins, color='Green', label=f'Green', edgecolor='Green',
                      alpha=0.99)  # , marker='.', linewidth=0, markersize=5)
            # ax_2.hist(sorted(np.real(eig_JG))[2:], bins=bins, color='Black', label=f'Jacobi-Green', edgecolor='black',
            #           alpha=0.2)  # , marker='.', linewidth=0, markersize=5)
            # ax_3.plot(sorted(eig_G)[2:],np.zeros_like(eig_G[2:]), color='red', label=f'Green',
            #           alpha=0.5, marker='.', linewidth=0, markersize=5)
            # ax_3.plot(sorted(eig_JG)[2:],np.ones_like(eig_JG[2:]), color='b', label=f'Jacobi-Green',
            #           alpha=0.5, marker='.', linewidth=0, markersize=5)
            # ax_3.set_ylim([1e-4, 1e2])  #
            if ratio == 0:
                ax_2.set_xlim([0, 1e0])
                ax_2.set_xticks([0, 0.5, 1])
                ax_2.set_xticklabels([0, 0.5, 1])
            else:
                ax_2.set_xlim([0, 101])
                ax_2.set_xticks([1, 34, 67, 100])
                ax_2.set_xticklabels([1, 34, 67, 100])
            # ax_3.set_xticklabels([1e-4, 0.5, 1])
            if counter == 0:
                ax_2.set_title('Histograms of eigenvalues \n' + r'Geometry ' + r'$\mathcal{G}$' + f'$_{{{2 ** G}}}$')
            elif counter == 1:
                ax_2.set_title(r'Geometry ' + r'$\mathcal{G}$' + f'$_{{{2 ** G}}}$')
            elif counter == 2:
                ax_2.set_title(r'Geometry ' + r'$\mathcal{G}$' + f'$_{{{2 ** G}}}$ ')
                ax_2.set_xlabel('eigenvalue $\lambda_i$' + f' (in {bins}bins)')
            ax_2.set_ylabel('\# of eigenvalues')
            #
            ax_2.set_ylim([1, 1e3])
            # ax_2.set_xtics([1e-4, 1e0])
            ax_2.set_yscale('log')
            # ax_2.set_xscale('log')
            # -------------------------------------# plot eigenvals #-------------------------------------#
            ax_3 = fig.add_subplot(gs[counter, 2])
            ax_3.text(-0.25, 1.15, f'(c.{counter + 1})', transform=ax_3.transAxes)

            ax_3.plot(sorted(eig_G)[2:], color='Green', label=f'Green',
                      alpha=0.5, marker='.', linewidth=0, markersize=5)
            # ax_3.plot(sorted(np.real(eig_JG))[2:], color='Black', label=f'Jacobi-Green',
            #           alpha=0.5, marker='.', linewidth=0, markersize=5)

            if ratio == 0:
                # ax_2.set_ylim([1e-4, max(sorted(eig_G)[2:])])
                ax_3.set_ylim([0, 1e0])
                ax_3.set_yticks([0, 0.5, 1])
                ax_3.set_yticklabels([0, 0.5, 1])
            else:
                ax_3.set_ylim([0, 101])
                ax_3.set_yticks([1, 34, 67, 100])
                ax_3.set_yticklabels([1, 34, 67, 100])  # f'$10^{{{2}}}$'
                ax_3.set_xlim([0, len(eig_G)])
                ax_3.set_xticks([1, len(eig_G) // 2, len(eig_G)])
                ax_3.set_xticklabels([1, len(eig_G) // 2, len(eig_G)])
            if counter == 0:
                ax_3.set_title('Sorted eigenvalues \n' + r'Geometry- ' + r'$\mathcal{G}$' + f'$_{{{2 ** G}}}$')
            elif counter == 1:
                ax_3.set_title(r'Geometry -' + r'$\mathcal{G}$' + f'$_{{{2 ** G}}}$')
            elif counter == 2:
                ax_3.set_title(r'Geometry -' + r'$\mathcal{G}$' + f'$_{{{2 ** G}}}$')
                ax_3.set_xlabel('eigenvalue index - $i$')
            ax_3.set_ylabel('eigenvalue $\lambda_i$')
            print(xopt.f.nb_of_pixels)
            print(xopt.f.nb_of_sampling_points)
            # print(xopt.f.norm_rMr_JG)
            counter += 1

        ax_1.legend(loc='upper right')
    fname = src + script_name + 'phasedep' + f'{ratio}_{geometry_ID}' + '{}'.format('.pdf')
    print(('create figure: {}'.format(fname)))
    plt.savefig(fname, bbox_inches='tight')
    plt.show()

#### plot mesh size independance
plot_this = True
if plot_this:
    geometry_ID = 'linear'  # 'linear'# 'sine_wave_'
    geometry_n = [2, ]
    discretization_n = [3, 4, 5]
    ratio = 2

    # create a figure
    fig = plt.figure(figsize=(5, 4.5))
    gs = fig.add_gridspec(3, 2, hspace=0.5, wspace=0.4, width_ratios=2 * (1,),
                          height_ratios=[1, 1, 1])

    for ii in np.arange(np.size(geometry_n)):
        G = geometry_n[ii]
        j2 = [x for x in discretization_n if x >= G]
        counter = 0
        for T in j2:

            print(f'G={G}')
            print(f'T={T}')
            file_data_name = (
                f'{script_name}_gID{geometry_ID}_T{T}_G{G}_kappa{ratio}.npy')
            folder_name = '../exp_data/'

            xopt = np.load('../exp_data/' + file_data_name + f'xopt_log.npz', allow_pickle=True)
            phase_field = np.load('../exp_data/' + file_data_name + f'.npy', allow_pickle=True)

            # -------------------------------------# plot eigenvals #-------------------------------------#
            ax_2 = fig.add_subplot(gs[counter, 1])
            # ax_2.text(-0.2, 1.15, f'(b.{counter+1})', transform=ax_2.transAxes)

            eig_G = xopt.f.eigens_G
            eig_JG = xopt.f.eigens_JG
            # Number of bins (fine-grained)
            num_bins = eig_G.size
            # Define the bin width
            bin_width = 0.05

            # Calculate the bin edges
            min_edge = np.min(sorted(eig_JG)[2:])
            max_edge = np.max(sorted(eig_JG)[2:])
            bins = np.arange(min_edge, max_edge + bin_width, bin_width)
            bins = 100
            # Create the histogram
            ax_2.hist(sorted(eig_G)[2:], bins=bins, color='Green', label=f'Green', edgecolor='Green',
                      alpha=0.99)  # , marker='.', linewidth=0, markersize=5)
            # ax_2.hist(sorted(np.real(eig_JG))[2:], bins=bins, color='Black', label=f'Jacobi-Green', edgecolor='black',
            #          alpha=0.2)  # , marker='.', linewidth=0, markersize=5)
            # ax_3.plot(sorted(eig_G)[2:],np.zeros_like(eig_G[2:]), color='red', label=f'Green',
            #           alpha=0.5, marker='.', linewidth=0, markersize=5)
            # ax_3.plot(sorted(eig_JG)[2:],np.ones_like(eig_JG[2:]), color='b', label=f'Jacobi-Green',
            #           alpha=0.5, marker='.', linewidth=0, markersize=5)

            # ax_3.set_ylim([1e-4, 1e2])  #
            if ratio == 0:
                ax_2.set_xlim([0, 1e0])
                ax_2.set_xticks([0, 0.5, 1])
                ax_2.set_xticklabels([0, 0.5, 1])
            else:
                ax_2.set_xlim([0, 101])
                ax_2.set_xticks([1, 34, 67, 100])
                ax_2.set_xticklabels([1, 34, 67, 100])  # f'$10^{{{2}}}$'f'$10^{{{2}}}$'
            # ax_3.set_xticklabels([1e-4, 0.5, 1])
            if counter == 0:
                ax_2.set_title('Histograms of eigenvalues \n' + r'Mesh -' + r'$\mathcal{T}$' + f'$_{{{2 ** T}}}$')
            elif counter == 1:
                ax_2.set_title(r'Mesh -' + r'$\mathcal{T}$' + f'$_{{{2 ** T}}}$')
            elif counter == 2:
                ax_2.set_title(r'Mesh -' + r'$\mathcal{T}$' + f'$_{{{2 ** T}}}$')
                ax_2.set_xlabel('eigenvalue $\lambda_i$' + f' (in {bins}bins)')
            ax_2.set_ylabel('\# of eigenvalues')

            #
            ax_2.set_ylim([1, 1e3])
            # ax_2.set_xtics([1e-4, 1e0])
            ax_2.set_yscale('log')
            # ax_2.set_xscale('log')
            # -------------------------------------# plot eigenvals #-------------------------------------#
            ax_3 = fig.add_subplot(gs[counter, 0])
            ax_3.plot(sorted(eig_G)[2:], color='Green', label=f'Green',
                      alpha=0.5, marker='.', linewidth=0, markersize=5)
            # ax_3.plot(sorted(np.real(eig_JG))[2:], color='Black', label=f'Jacobi-Green',
            #           alpha=0.5, marker='.', linewidth=0, markersize=5)

            # ax_3.text(-0.2, 1.15, f'(c.{counter+1})', transform=ax_3.transAxes)
            if ratio == 0:
                # ax_2.set_ylim([1e-4, max(sorted(eig_G)[2:])])
                ax_3.set_ylim([0, 1e0])
                ax_3.set_yticks([0, 0.5, 1])
                ax_3.set_yticklabels([0, 0.5, 1])
            else:
                # ax_2.set_ylim([1e-4, max(sorted(eig_G)[2:])])
                ax_3.set_ylim([0, 101])
                ax_3.set_yticks([1, 34, 67, 100])
                ax_3.set_yticklabels([1, 34, 67, 100])  # f'$10^{{{2}}}$'
                ax_3.set_xlim([0, len(eig_G)])
                ax_3.set_xticks([1, len(eig_G) // 2, len(eig_G)])
                ax_3.set_xticklabels([1, len(eig_G) // 2, len(eig_G)])
            # ax_3.set_title(r'Sorted eigenvalues')
            if counter == 0:
                ax_3.set_title('Sorted eigenvalues \n' + r'Mesh -' + r'$\mathcal{T}$' + f'$_{{{2 ** T}}}$')
            elif counter == 1:
                ax_3.set_title(r'Mesh -' + r'$\mathcal{T}$' + f'$_{{{2 ** T}}}$')
            elif counter == 2:
                ax_3.set_title(
                    r'Mesh ' + r'$\mathcal{T}$' + f'$_{{{2 ** T}}}$')  # +f' (${{{2**T}}}$ nodes in x direction)'
                ax_3.set_xlabel('eigenvalue index - $i$')
            ax_3.set_ylabel('eigenvalue $\lambda_i$')
            print(xopt.f.nb_of_pixels)
            print(xopt.f.nb_of_sampling_points)
            # print(xopt.f.norm_rMr_JG)
            counter += 1
    fname = src + script_name + 'eigens' + f'{ratio}_{geometry_ID}' + '{}'.format('.pdf')
    print(('create figure: {}'.format(fname)))
    plt.savefig(fname, bbox_inches='tight')
    plt.show()

#### plot mesh size independance
plot_size_dep = True
if plot_size_dep:
    geometry_ID = 'linear'  # 'linear'# 'sine_wave_' #'linear'
    geometry_n = [2, 3, 5]
    discretization_n = [5]
    ratio = 2

    # create a figure
    # create a figure
    fig = plt.figure(figsize=(5, 4.5))
    gs = fig.add_gridspec(3, 2, hspace=0.5, wspace=0.4, width_ratios=2 * (1,),
                          height_ratios=[1, 1, 1])

    counter = 0
    for ii in np.arange(np.size(geometry_n)):
        G = geometry_n[ii]
        j2 = [x for x in discretization_n if x >= G]

        for T in j2:

            print(f'G={G}')
            print(f'T={T}')
            file_data_name = (
                f'{script_name}_gID{geometry_ID}_T{T}_G{G}_kappa{ratio}.npy')
            folder_name = '../exp_data/'

            xopt = np.load('../exp_data/' + file_data_name + f'xopt_log.npz', allow_pickle=True)
            phase_field = np.load('../exp_data/' + file_data_name + f'.npy', allow_pickle=True)

            #
            # ax_1.set_title(f'nb phases {2 ** (nb_starting_phases)}, nb pixels {number_of_pixels[0]}', wrap=True)

            ax_1.semilogy(np.arange(1, len(xopt.f.norm_rMr_G) + 1), xopt.f.norm_rMr_G,
                          label=r'  Green ' + r'$\mathcal{G}$' + f'$_{{{2 ** G}}}$', color='green',
                          linestyle='--', marker=markers[counter])
            ax_1.semilogy(np.arange(1, len(xopt.f.norm_rMr_JG) + 1), xopt.f.norm_rMr_JG,
                          label=r' Jacobi-Green  ' + r'$\mathcal{G}$' + f'$_{{{2 ** G}}}$',
                          color='Black', linestyle='-.', marker=markers[counter])
            ax_1.set_xlabel('PCG iteration - k')
            ax_1.set_ylabel('Norm of residua - '
                            r'$||r_{k}||_{G^{-1}} $')
            ax_1.set_title(f'Convergence \n ' + r'Mesh - $\mathcal{T}$' + f'$_{{{2 ** T}}}$')

            if geometry_ID == 'sine_wave_':
                ax_1.set_xlim([1, 60])
                ax_1.set_xticks([1, 8, 15, 18, 22, 30, 53, 60])
                ax_1.set_xticklabels([1, 8, 15, 18, 22, 30, 53, 60])
            elif geometry_ID == 'linear':
                ax_1.set_xlim([1, 30])
                ax_1.set_xticks([1, 4, 8, 9, 14, 30])
                ax_1.set_xticklabels([1, 4, 8, 9, 14, 30])
            ax_1.set_ylim([1e-6, 1e4])

            # -------------------------------------# plot eigenvals #-------------------------------------#
            ax_2 = fig.add_subplot(gs[counter, 1])
            # ax_2.text(-0.25, 1.15, f'(b.{counter+1})', transform=ax_2.transAxes)

            eig_G = xopt.f.eigens_G
            eig_JG = xopt.f.eigens_JG
            # Number of bins (fine-grained)
            num_bins = eig_G.size
            # Define the bin width
            bin_width = 0.05

            # Calculate the bin edges
            min_edge = np.min(sorted(eig_JG)[2:])
            max_edge = np.max(sorted(eig_JG)[2:])
            bins = np.arange(min_edge, max_edge + bin_width, bin_width)
            bins = 100
            # Create the histogram
            ax_2.hist(sorted(eig_G)[2:], bins=bins, color='Green', label=f'Green', edgecolor='Green',
                      alpha=0.99)  # , marker='.', linewidth=0, markersize=5)
            # ax_2.hist(sorted(np.real(eig_JG))[2:], bins=bins, color='Black', label=f'Jacobi-Green', edgecolor='black',
            #           alpha=0.2)  # , marker='.', linewidth=0, markersize=5)
            # ax_3.plot(sorted(eig_G)[2:],np.zeros_like(eig_G[2:]), color='red', label=f'Green',
            #           alpha=0.5, marker='.', linewidth=0, markersize=5)
            # ax_3.plot(sorted(eig_JG)[2:],np.ones_like(eig_JG[2:]), color='b', label=f'Jacobi-Green',
            #           alpha=0.5, marker='.', linewidth=0, markersize=5)
            # ax_3.set_ylim([1e-4, 1e2])  #
            if ratio == 0:
                ax_2.set_xlim([0, 1e0])
                ax_2.set_xticks([0, 0.5, 1])
                ax_2.set_xticklabels([0, 0.5, 1])
            else:
                ax_2.set_xlim([0, 101])
                ax_2.set_xticks([1, 34, 67, 100])
                ax_2.set_xticklabels([1, 34, 67, 100])
            # ax_3.set_xticklabels([1e-4, 0.5, 1])
            if counter == 0:
                ax_2.set_title('Histograms of eigenvalues \n' + r'Geometry ' + r'$\mathcal{G}$' + f'$_{{{2 ** G}}}$')
            elif counter == 1:
                ax_2.set_title(r'Geometry ' + r'$\mathcal{G}$' + f'$_{{{2 ** G}}}$')
            elif counter == 2:
                ax_2.set_title(r'Geometry ' + r'$\mathcal{G}$' + f'$_{{{2 ** G}}}$ ')
                ax_2.set_xlabel('eigenvalue $\lambda_i$' + f' (in {bins}bins)')
            ax_2.set_ylabel('\# of eigenvalues')
            #
            ax_2.set_ylim([1, 1e3])
            # ax_2.set_xtics([1e-4, 1e0])
            ax_2.set_yscale('log')
            # ax_2.set_xscale('log')
            # -------------------------------------# plot eigenvals #-------------------------------------#
            ax_3 = fig.add_subplot(gs[counter, 0])
            # ax_3.text(-0.25, 1.15, f'(c.{counter + 1})', transform=ax_3.transAxes)

            ax_3.plot(sorted(eig_G)[2:], color='Green', label=f'Green',
                      alpha=0.5, marker='.', linewidth=0, markersize=5)
            # ax_3.plot(sorted(np.real(eig_JG))[2:], color='Black', label=f'Jacobi-Green',
            #           alpha=0.5, marker='.', linewidth=0, markersize=5)

            if ratio == 0:
                # ax_2.set_ylim([1e-4, max(sorted(eig_G)[2:])])
                ax_3.set_ylim([0, 1e0])
                ax_3.set_yticks([0, 0.5, 1])
                ax_3.set_yticklabels([0, 0.5, 1])
            else:
                ax_3.set_ylim([0, 101])
                ax_3.set_yticks([1, 34, 67, 100])
                ax_3.set_yticklabels([1, 34, 67, 100])  # f'$10^{{{2}}}$'
                ax_3.set_xlim([0, len(eig_G)])
                ax_3.set_xticks([1, len(eig_G) // 2, len(eig_G)])
                ax_3.set_xticklabels([1, len(eig_G) // 2, len(eig_G)])
            if counter == 0:
                ax_3.set_title('Sorted eigenvalues \n' + r'Geometry- ' + r'$\mathcal{G}$' + f'$_{{{2 ** G}}}$')
            elif counter == 1:
                ax_3.set_title(r'Geometry -' + r'$\mathcal{G}$' + f'$_{{{2 ** G}}}$')
            elif counter == 2:
                ax_3.set_title(r'Geometry -' + r'$\mathcal{G}$' + f'$_{{{2 ** G}}}$')
                ax_3.set_xlabel('eigenvalue index - $i$')
            ax_3.set_ylabel('eigenvalue $\lambda_i$')
            print(xopt.f.nb_of_pixels)
            print(xopt.f.nb_of_sampling_points)
            # print(xopt.f.norm_rMr_JG)
            counter += 1

        ax_1.legend(loc='upper right')
    fname = src + script_name + 'ndep' + f'{ratio}_{geometry_ID}' + '{}'.format('.pdf')
    print(('create figure: {}'.format(fname)))
    plt.savefig(fname, bbox_inches='tight')
    plt.show()

quit()
#### plot phase contrast dependanc
plot_phase_contrast = True
if plot_phase_contrast:
    geometry_ID = 'linear'
    geometry_n = [5]
    discretization_n = [5]

    # create a figure
    fig = plt.figure(figsize=(11, 5.5))
    gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.4, width_ratios=3 * (1,),
                          height_ratios=[1, 1, 1])
    ax_1 = fig.add_subplot(gs[:, 0])

    ax_1.text(-0.25, 1.05, '(a.1)', transform=ax_1.transAxes)
    counter = 0
    for ii in np.arange(np.size(geometry_n)):
        G = geometry_n[ii]
        j2 = [x for x in discretization_n if x >= G]

        for T in j2:
            for ratio in [1, 4]:

                print(f'G={G}')
                print(f'T={T}')
                file_data_name = (
                    f'{script_name}_gID{geometry_ID}_T{T}_G{G}_kappa{ratio}.npy')
                folder_name = '../exp_data/'

                xopt = np.load('../exp_data/' + file_data_name + f'xopt_log.npz', allow_pickle=True)
                phase_field = np.load('../exp_data/' + file_data_name + f'.npy', allow_pickle=True)

                #
                # ax_1.set_title(f'nb phases {2 ** (nb_starting_phases)}, nb pixels {number_of_pixels[0]}', wrap=True)

                ax_1.semilogy(np.arange(1, len(xopt.f.norm_rMr_G) + 1), xopt.f.norm_rMr_G,
                              label=r'  Green ' + r'$\mathcal{G}$' + f'$_{G}$', color='green',
                              linestyle='--', marker=markers[counter])
                ax_1.semilogy(np.arange(1, len(xopt.f.norm_rMr_JG) + 1), xopt.f.norm_rMr_JG,
                              label=r' Jacobi-Green  ' + r'$\mathcal{G}$' + f'$_{G}$',
                              color='Black', linestyle='-.', marker=markers[counter])
                ax_1.set_xlabel('PCG iteration - k')
                ax_1.set_ylabel('Norm of residua - '
                                r'$||r_{k}||_{G^{-1}} $')
                ax_1.set_title(f'Convergence \n ' + r'Mesh - $\mathcal{T}_5$')
                ax_1.set_ylim([1e-14, 1e1])
                ax_1.set_xlim([1, 20])
                # ax_1.set_xticks([1,5, 8, 10,15])
                # ax_1.set_xticklabels([1,5, 8, 10,15])
                ax_1.set_xticks([1, 4, 8, 16, 20])  # , 30, 40, 50
                ax_1.set_xticklabels([1, 4, 8, 16, 20])  # 30, 40, 50
                # ax_1.set_xticks([1,5, 10, ])
                # ax_1.set_xticklabels([1,5,10])
                # -------------------------------------# plot eigenvals #-------------------------------------#
                ax_2 = fig.add_subplot(gs[counter, 1])
                ax_2.text(-0.25, 1.15, f'(b.{counter + 1})', transform=ax_2.transAxes)

                eig_G = xopt.f.eigens_G
                eig_JG = xopt.f.eigens_JG
                # Number of bins (fine-grained)
                num_bins = eig_G.size
                # Define the bin width
                bin_width = 0.05

                # Calculate the bin edges
                min_edge = np.min(sorted(eig_JG)[2:])
                max_edge = np.max(sorted(eig_JG)[2:])
                bins = np.arange(min_edge, max_edge + bin_width, bin_width)
                bins = 100
                # Create the histogram
                ax_2.hist(sorted(eig_G)[2:], bins=bins, color='Green', label=f'Green', edgecolor='Green',
                          alpha=0.99)  # , marker='.', linewidth=0, markersize=5)
                ax_2.hist(sorted(np.real(eig_JG))[2:], bins=bins, color='Black', label=f'Jacobi-Green',
                          edgecolor='black',
                          alpha=0.2)  # , marker='.', linewidth=0, markersize=5)
                # ax_3.plot(sorted(eig_G)[2:],np.zeros_like(eig_G[2:]), color='red', label=f'Green',
                #           alpha=0.5, marker='.', linewidth=0, markersize=5)
                # ax_3.plot(sorted(eig_JG)[2:],np.ones_like(eig_JG[2:]), color='b', label=f'Jacobi-Green',
                #           alpha=0.5, marker='.', linewidth=0, markersize=5)
                # ax_3.set_ylim([1e-4, 1e2])  #
                if ratio == 0:
                    ax_2.set_xlim([0, 1e0])
                    ax_2.set_xticks([0, 0.5, 1])
                    ax_2.set_xticklabels([0, 0.5, 1])
                else:
                    ax_2.set_xlim([1 / np.power(10, 4), 1e0])
                    ax_2.set_xticks([1 / np.power(10, 4), 0.5, 1])
                    ax_2.set_xticklabels([f'$10^{{{-4}}}$', 0.5, 1])
                # ax_3.set_xticklabels([1e-4, 0.5, 1])
                if counter == 0:
                    ax_2.set_title('Histograms of eigenvalues \n' + r'Geometry - $\mathcal{G}_3$')
                elif counter == 1:
                    ax_2.set_title(r'Geometry - $\mathcal{G}_4$')
                elif counter == 2:
                    ax_2.set_title(r'Geometry - $\mathcal{G}_5$  ')
                    ax_2.set_xlabel('eigenvalue $\lambda_i$' + f' (in {bins}bins)')
                ax_2.set_ylabel('\# of eigenvalues')
                #
                ax_2.set_ylim([1, 1e3])
                # ax_2.set_xtics([1e-4, 1e0])
                ax_2.set_yscale('log')
                # ax_2.set_xscale('log')
                # -------------------------------------# plot eigenvals #-------------------------------------#
                ax_3 = fig.add_subplot(gs[counter, 2])
                ax_3.text(-0.25, 1.15, f'(c.{counter + 1})', transform=ax_3.transAxes)

                ax_3.plot(sorted(eig_G)[2:], color='Green', label=f'Green',
                          alpha=0.5, marker='.', linewidth=0, markersize=5)
                ax_3.plot(sorted(np.real(eig_JG))[2:], color='Black', label=f'Jacobi-Green',
                          alpha=0.5, marker='.', linewidth=0, markersize=5)

                if ratio == 0:
                    # ax_2.set_ylim([1e-4, max(sorted(eig_G)[2:])])
                    ax_3.set_ylim([0, 1e0])
                    ax_3.set_yticks([0, 0.5, 1])
                    ax_3.set_yticklabels([0, 0.5, 1])
                else:
                    # ax_2.set_ylim([1e-4, max(sorted(eig_G)[2:])])
                    ax_3.set_ylim([1e-4, 1e0])
                    ax_3.set_yticks([1e-4, 0.5, 1])
                    ax_3.set_yticklabels([f'$10^{{{-4}}}$', 0.5, 1])

                if counter == 0:
                    ax_3.set_title('Sorted eigenvalues \n' + r'Geometry - $\mathcal{G}_3$')
                elif counter == 1:
                    ax_3.set_title(r'Geometry - $\mathcal{G}_4$')
                elif counter == 2:
                    ax_3.set_title(r'Geometry - $\mathcal{G}_5$ ')
                    ax_3.set_xlabel('eigenvalue index - $i$')
                ax_3.set_ylabel('eigenvalue $\lambda_i$')
                print(xopt.f.nb_of_pixels)
                print(xopt.f.nb_of_sampling_points)
                # print(xopt.f.norm_rMr_JG)
                counter += 1

        ax_1.legend(loc='upper right')
    fname = src + script_name + 'contrastdep' + f'{ratio}_{geometry_ID}' + '{}'.format('.pdf')
    print(('create figure: {}'.format(fname)))
    plt.savefig(fname, bbox_inches='tight')
    plt.show()
