import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import scipy as sc
import os
from mpi4py import MPI
from NuMPI.Tools import Reduction
from NuMPI.IO import save_npy, load_npy

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO.solvers import PCG
from muFFTTO import microstructure_library

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX
    # "font.family": "helvetica",  # Use a serif font
})
plt.rcParams.update({'font.size': 11})
plt.rcParams["font.family"] = "Arial"

script_name = 'exp_paper_JG_eivals_with_eigenvectors'
file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
figure_folder_path = file_folder_path + '/figures/' + script_name + '/'


def get_cg_polynomial(lambda_val, ritz_values):
    """
    Compute the CG polynomial value at a given lambda value and list of Ritz values.

    Parameters:
    lambda_val (float): The lambda value at which to evaluate the polynomial.
    ritz_values (list of float): List of Ritz values at the j-th CG iteration.

    Returns:
    float: The value of the CG polynomial at the given lambda value.
    """
    j = len(ritz_values)
    numerator = 1.0
    denominator = 1.0

    for theta in ritz_values:
        numerator *= (lambda_val - theta)
        denominator *= theta

    polynomial_value = (-1) ** j * numerator / denominator
    return polynomial_value


# def plot_cg_polynomial_JG_paper(x_values, ritz_values, true_eigenvalues, iteration,ylim=[-2., 2.], weight=None, error_evol=None,
#                                 title=None):
#     i=iteration
#     if i == 0:  # Zero order polynomial is constant
#         polynomial_value = np.ones(len(x_values))
#     else:
#         polynomial_value = get_cg_polynomial(x_values, ritz_values[i - 1, :i])
#
#     fig = plt.figure(figsize=(7.0, 4))
#
#     gs = fig.add_gridspec(2, 1, width_ratios=[1], height_ratios=[2, 1], wspace=0, hspace=0.01)
#     ax_poly = fig.add_subplot(gs[0, 0])
#     ax_weights = fig.add_subplot(gs[1, 0])
#     #            ax_error_true = fig.add_subplot(gs[1, 1])
#     ax_poly.scatter(np.real(true_eigenvalues), [0] * len(true_eigenvalues), color='green', marker='|',
#                     label="Eigenvalues")
#     ax_poly.plot(x_values, polynomial_value, color='red', label=r'$\varphi^{CG}$' + f'$_{{{i}}}$')
#     ax_poly.hlines(xmin=0, xmax=x_values[-1], y=0, linestyles='--', color='gray')
#
#     if i == 0:  # Zero order polynomial is constant
#         ax_poly.scatter(-1, -1, color='red', marker='x',
#                         label=f"Ritz Values\n (Approx Eigenvalues)")
#     else:  # Zero order polynomial is constant
#         ax_poly.scatter(np.real(ritz_values[i - 1, :i]), [0] * len(ritz_values[i - 1, :i]), color='red', marker='x',
#                         label=f"Roots of " + r'$\varphi^{CG}$' + f'$_{{{i}}}$')  # +"(Approx Eigenvalues)"
#     ax_poly.set_xticks([1, 34, 67, 100])
#     ax_poly.set_xticklabels([1, 34, 67, 100])
#     if weight is not None:
#         ax_weights.scatter(np.real(true_eigenvalues), np.real(weight) / np.real(true_eigenvalues), color='blue',
#                            marker='o', label=r"non-zero weights- $w_{i}/ \lambda_{i}$")
#         ax_weights.set_yscale('log')
#         ax_weights.set_ylim(1e-10, 1)
#         ax_weights.set_xlim(-0.1, x_values[-1] + 0.3)
#         # ax_weights.set_ylabel(r"$w_{i}/ \lambda_{i}$")
#         # ax_weights.set_title(f"Weights / Eigens ")
#         ax_weights.set_xlabel('eigenvalue index - $i$ (sorted)')
#         # ax_weights.set_ylabel(r'Weights - $w_{i}/ \lambda_{i}$')
#         ax_weights.set_xticks([1, 34, 67, 100])
#         ax_weights.set_xticklabels([1, 34, 67, 100])
#         ax_weights.legend(ncol=1, loc='lower left')
#
#     # ax_poly.set_xlabel("Eigenvalues --- Approximation")
#     # ax_poly.set_ylabel("CG (Lanczos) Iteration")
#     # ax_poly.set_title(f"CG polynomial (Lanczos Iteration) {{{i}}}")
#     ax_poly.set_ylim(ylim[0], ylim[1])
#     ax_poly.set_xlim(-0.1, 100 + 0.3)
#     ax_poly.set_xticks([])
#     ax_poly.set_xticklabels([])
#
#     ax_poly.legend(ncol=3, loc='upper left')
#
#     plt.show()


##### PLOT FOR MESH REFINEMENT
ratios = [1, 2, 4]
geometry_n = [3, 4, 5]
discretization_n = [5]  # ,5
iteration_to_plot = 7  # 2 ** G - 1

# create a figure
fig = plt.figure(figsize=(8.3, 5.0))
plt.rcParams.update({'font.size': 11})
plt.rcParams["font.family"] = "Arial"
gs = fig.add_gridspec(4, 3, hspace=0.2, wspace=0.1, width_ratios=[1, 1, 1],
                      height_ratios=[0.2, 1, 1, 1])

for ii in np.arange(np.size(geometry_n)):
    G = geometry_n[ii]
    j2 = [x for x in discretization_n if x >= G]
    counter = 0

    for T in j2:
        for ratio in ratios:
            print(f'G={G}')
            print(f'T={T}')

            results_name = f'T{2 ** T}_G{2 ** G}_kappa{ratio}'

            _info = np.load(data_folder_path + results_name + f'_info.npz', allow_pickle=True)

            expanded_data = {key: _info[key] for key in _info.files}

            nb_of_pixels_global = _info['nb_of_pixels']
            phase_fied = _info['nb_of_sampling_points']
            norms = {}
            norms['data_scaled_rr'] = _info['norm_rMr_G']
            norms['energy_upper_bound'] = _info['norm_UB_G']
            norms['residual_rr'] = _info['norm_rr_G']
            eig_G = _info['eigens_G']

            eig_vect_G = _info['eig_vect_G']
            rhs = _info['rhs']
            weight = _info['weights']

            x_values = _info['x_values']
            ritz_values = _info['ritz_values']
            eig_G_no_zero = np.real(_info['eig_G_no_zero'])

            #
            # ax_eigens = fig.add_subplot(gs[ 0,counter])
            #
            # ax_eigens.plot(sorted(eig_G)[2:], color='Green', label=f'Green',
            #                alpha=0.5, marker='.', linewidth=0, markersize=5)

            #
            # plot_cg_polynomial_JG_paper(x_values, ritz_values, true_eigenvalues=np.real(eig_G_no_zero),
            #                             weight=weight,
            #                             error_evol=norms['energy_upper_bound'] / norms['residual_rr'][
            #                                 0], title='Green', iteration=iteration_to_plot)  # energy_lb

            ax_poly = fig.add_subplot(gs[ii + 1, counter])

            i = iteration_to_plot
            if i == 0:  # Zero order polynomial is constant
                polynomial_value = np.ones(len(x_values))
            else:
                polynomial_value = get_cg_polynomial(x_values, ritz_values[i - 1, :i])
            #            ax_error_true = fig.add_subplot(gs[1, 1])

            weights_to_plot = np.real(weight) / np.real(eig_G_no_zero)  # , np.real(eig_G_no_zero)

            opacities = np.abs(np.real(weights_to_plot[::-1]))
            opacities[0:2] = 0
            opacities[opacities > 1e-14] = 1
            opacities[opacities < 1] = 0
            ax_poly.scatter(eig_G_no_zero[::-1], [0] * len(eig_G_no_zero), color='Blue',
                            alpha=opacities, marker='|', label=f"Eigenvalues with non-zero weights")

            # ax_poly.scatter(np.real(weight) / np.real(eig_G_no_zero), [0] * len(weight), color='Blue', marker='|',
            #                 label="Weights")

            # ax_poly.scatter(np.real(eig_G_no_zero), [0] * len(eig_G_no_zero), color='green', marker='|',
            #                 label="Eigenvalues")

            ax_poly.plot(x_values, polynomial_value, color='red', label=r'$\varphi^{CG}$' + f'$_{{{i}}}$')
            ax_poly.hlines(xmin=0, xmax=x_values[-1], y=0, linestyles='-', color='gray')

            if i == 0:  # Zero order polynomial is constant
                ax_poly.scatter(-1, -1, color='red', marker='x',
                                label=f"Ritz Values\n (Approx Eigenvalues)")
            else:  # Zero order polynomial is constant
                ax_poly.scatter(np.real(ritz_values[i - 1, :i]), [0] * len(ritz_values[i - 1, :i]), color='red',
                                marker='x',
                                label=f"Roots of " + r'$\varphi^{CG}$' + f'$_{{{i}}}$')  # +"(Approx Eigenvalues)"
            # ax_poly.set_xticks([1, 34, 67, 100])
            # ax_poly.set_xticklabels([1, 34, 67, 100])
            ax_poly.set_ylim(-1, 1)
            # Specify grid line positions
            if counter == 0:
                ax_poly.set_yticks([-1, -0.5, 0, 0.5, 1])
                ax_poly.set_yticklabels([-1, '', 0, '', 1])


            else:
                ax_poly.set_yticks([-1, -0.5, 0, 0.5, 1])
                ax_poly.set_yticklabels([])

            # if counter == 2:
            ax_poly.text(0.05, 0.09, r'$\mathcal{G}$' + f'$_{{{2 ** G}}}$',
                         transform=ax_poly.transAxes, fontsize=12)  # , rotation=90  #
            ax_poly.text(0.6, 0.09, fr'$\chi^{{\rm  tot}} = 10^{{{ratio}}} $',
                         transform=ax_poly.transAxes, fontsize=12)
            logscale = True
            if logscale:
                ax_poly.set_xlim(1, 10 ** 4 + 0.3)
                ax_poly.set_xscale('log')
            else:
                ax_poly.set_xlim(0, 10 ** ratio + 0.3)

            # ax_poly.set_xticks([])
            # ax_poly.set_xticklabels([])
            #
            # ax_poly.legend(ncol=3, loc='upper left')
            ax_poly.grid(axis='y')
            if ii == 2:
                if logscale:
                    ax_poly.set_xticks([1, 10 ** 4])
                    ax_poly.set_xticklabels([1, fr'$10^{{{4}}}$'])

                else:
                    ax_poly.set_xticks([0, 10 ** ratio])

                    ax_poly.set_xticklabels([0, fr'$10^{{{ratio}}}$'])
                ax_poly.set_xlabel('eigenvalues $\lambda_i$')
            else:
                ax_poly.set_xticks([])

            # if ii == 0:
            # ax_poly.text(0.3, 1.1, fr'$\chi^{{\rm  tot}} = 10^{{{ratio}}} $',
            #            transform=ax_poly.transAxes, fontsize=12)

            # ax_weights = fig.add_subplot(gs[ 1,counter])
            # if weight is not None:
            #     ax_weights.scatter(np.real(weight) / np.real(eig_G_no_zero), np.real(eig_G_no_zero), color='blue',
            #                        marker='o', s=5,label=r"non-zero weights- $w_{i}^{2}/ \lambda_{i}$")
            #     ax_weights.set_xscale('log')
            #
            #     # ax_weights.set_yticks([1, 34, 67, 100])
            #     # ax_weights.set_yticklabels([1, 34, 67, 100])
            #     ax_weights.set_yticks([1, 10 ** ratio])
            #     ax_weights.set_yticklabels([1, 10 ** ratio])
            #
            #     ax_weights.set_xlim(1e-10, 1)
            #     ax_weights.set_xticks([1e-10, 1e-5, 1])
            #     ax_weights.set_xticklabels([fr'$10^{{{-10}}}$', fr'$10^{{{-5}}}$', fr'$10^{{{0}}}$'])
            # ax_weights.set_xlim(-0.1, x_values[-1] + 0.3)
            #
            # ax_weights.set_xlabel('eigenvalue index - $i$ (sorted)')
            #
            # ax_weights.set_xticks([1, 34, 67, 100])
            # ax_weights.set_xticklabels([1, 34, 67, 100])
            # ax_weights.legend(ncol=1, loc='lower left')

            # ax_poly.set_xlabel("Eigenvalues --- Approximation")
            # ax_poly.set_ylabel("CG (Lanczos) Iteration")
            # ax_poly.set_title(f"CG polynomial (Lanczos Iteration) {{{i}}}")

            counter += 1

ax_legend = fig.add_subplot(gs[0, :])
ax_legend.set_axis_off()
# Get current position and modify it
pos = ax_legend.get_position()
ax_legend.set_position([pos.x0, pos.y0 + 0.01, pos.width, pos.height])

# Custom legend handles
eig_label = mpl.lines.Line2D([], [], linestyle=' ', marker='|', color='blue',
                             label=f"Eigenvalues with non-zero weights")
poly_label = mpl.lines.Line2D([], [], linestyle='-', color='red',
                              label=r'$\varphi^{CG}$' + f'$_{{{iteration_to_plot}}}$')
roots_label = mpl.lines.Line2D([], [], linestyle=' ', marker='x', color='red',
                               label=f"Roots of " + r'$\varphi^{CG}$' + f'$_{{{iteration_to_plot}}}$')

# Create legend with custom handles
ax_legend.legend(handles=[eig_label, poly_label, roots_label], ncol=3, loc='upper center')

# ax_arrows = fig.add_subplot(gs[:, :])
# ax_arrows.set_axis_off()
#
# ax_arrows.arrow(x=0, y=0, dx=1, dy=0, **arrow_style)
# ax_arrows.annotate('U',
#              xy=(0, 0),
#              xytext=(10, 0),
#              textcoords='offset points')

# handles, labels = ax_poly.get_legend_handles_labels()
# order = [0, 1, 2]  # [0, 2, 4, 1, 3, 5]
# ax_legend.legend([handles[idx] for idx in order], [labels[idx] for idx in order], ncol=3)

fig.tight_layout()
fname = f'evol_of_CG_contrast_geometry_it{iteration_to_plot}' + '{}'.format('.pdf')
print(('create figure: {}'.format(fname)))
plt.savefig(figure_folder_path + fname, bbox_inches='tight')
plt.show()

quit()
##### PLOT FOR DATA REFINEMENT
ratio = 2

geometry_n = [2, 3, 4]
discretization_n = [4]  #
# create a figure
fig = plt.figure(figsize=(8.3, 5.0))

gs = fig.add_gridspec(3, 3, hspace=0.2, wspace=0.2, width_ratios=[1, 1, 1],
                      height_ratios=[1, 1, 1])

for ii in np.arange(np.size(discretization_n)):
    T = discretization_n[ii]
    j2 = [x for x in geometry_n if x >= T]
    counter = 0
    for jj in np.arange(np.size(geometry_n)):
        G = geometry_n[jj]
        print(f'G={G}')
        print(f'T={T}')

        results_name = f'T{2 ** T}_G{2 ** G}_kappa{ratio}'

        _info = np.load(data_folder_path + results_name + f'_info.npz', allow_pickle=True)

        expanded_data = {key: _info[key] for key in _info.files}

        nb_of_pixels_global = _info['nb_of_pixels']
        phase_fied = _info['nb_of_sampling_points']
        norms = {}
        norms['data_scaled_rr'] = _info['norm_rMr_G']
        norms['energy_upper_bound'] = _info['norm_UB_G']
        norms['residual_rr'] = _info['norm_rr_G']
        eig_G = _info['eigens_G']

        eig_vect_G = _info['eig_vect_G']
        rhs = _info['rhs']
        weight = _info['weights']

        x_values = _info['x_values']
        ritz_values = _info['ritz_values']
        eig_G_no_zero = np.real(_info['eig_G_no_zero'])

        #
        ax_eigens = fig.add_subplot(gs[counter, 0])

        ax_eigens.plot(sorted(eig_G)[2:], color='Green', label=f'Green',
                       alpha=0.5, marker='.', linewidth=0, markersize=5)

        #
        # plot_cg_polynomial_JG_paper(x_values, ritz_values, true_eigenvalues=np.real(eig_G_no_zero),
        #                             weight=weight,
        #                             error_evol=norms['energy_upper_bound'] / norms['residual_rr'][
        #                                 0], title='Green', iteration=iteration_to_plot)  # energy_lb
        ax_poly = fig.add_subplot(gs[counter, 2])
        iteration_to_plot = 2 ** G - 1

        i = iteration_to_plot
        if i == 0:  # Zero order polynomial is constant
            polynomial_value = np.ones(len(x_values))
        else:
            polynomial_value = get_cg_polynomial(x_values, ritz_values[i - 1, :i])

        ax_weights = fig.add_subplot(gs[counter, 1])
        #            ax_error_true = fig.add_subplot(gs[1, 1])
        ax_poly.scatter([0] * len(eig_G_no_zero), np.real(eig_G_no_zero), color='green', marker='|',
                        label="Eigenvalues")
        ax_poly.plot(polynomial_value, x_values, color='red', label=r'$\varphi^{CG}$' + f'$_{{{i}}}$')
        ax_poly.vlines(ymin=0, ymax=x_values[-1], x=0, linestyles='--', color='gray')

        if i == 0:  # Zero order polynomial is constant
            ax_poly.scatter(-1, -1, color='red', marker='x',
                            label=f"Ritz Values\n (Approx Eigenvalues)")
        else:  # Zero order polynomial is constant
            ax_poly.scatter([0] * len(ritz_values[i - 1, :i]), np.real(ritz_values[i - 1, :i]), color='red', marker='x',
                            label=f"Roots of " + r'$\varphi^{CG}$' + f'$_{{{i}}}$')  # +"(Approx Eigenvalues)"
        ax_poly.set_yticks([1, 34, 67, 100])
        ax_poly.set_yticklabels([1, 34, 67, 100])

        if weight is not None:
            ax_weights.scatter(np.real(weight) / np.real(eig_G_no_zero), np.real(eig_G_no_zero), color='blue',
                               marker='o', label=r"non-zero weights- $w_{i}/ \lambda_{i}$")
            ax_weights.set_xscale('log')

            ax_weights.set_yticks([1, 34, 67, 100])
            ax_weights.set_yticklabels([1, 34, 67, 100])

            ax_weights.set_xlim(1e-10, 1)
            ax_weights.set_xticks([1e-10, 1e-5, 1])
            ax_weights.set_xticklabels([fr'$10^{{{-10}}}$', fr'$10^{{{-5}}}$', fr'$10^{{{0}}}$'])
            # ax_weights.set_xlim(-0.1, x_values[-1] + 0.3)
            #
            # ax_weights.set_xlabel('eigenvalue index - $i$ (sorted)')
            #
            # ax_weights.set_xticks([1, 34, 67, 100])
            # ax_weights.set_xticklabels([1, 34, 67, 100])
            # ax_weights.legend(ncol=1, loc='lower left')

        # ax_poly.set_xlabel("Eigenvalues --- Approximation")
        # ax_poly.set_ylabel("CG (Lanczos) Iteration")
        # ax_poly.set_title(f"CG polynomial (Lanczos Iteration) {{{i}}}")
        ax_poly.set_xlim(-2, 2)
        ax_poly.set_ylim(-0.1, 100 + 0.3)
        ax_poly.set_yticks([])
        ax_poly.set_yticklabels([])

        ax_poly.legend(ncol=3, loc='upper left')

        counter += 1

    plt.show()

##### %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##### PLOT FOR MESH REFINEMENT
ratio = 2
geometry_n = [2, 3]
discretization_n = [3, 4]  # ,5
for ii in np.arange(np.size(geometry_n)):
    G = geometry_n[ii]
    j2 = [x for x in discretization_n if x >= G]
    counter = 0

    # create a figure
    fig = plt.figure(figsize=(8.3, 5.0))

    gs = fig.add_gridspec(3, 3, hspace=0.2, wspace=0.2, width_ratios=[1, 1, 1],
                          height_ratios=[1, 1, 1])

    for T in j2:
        print(f'G={G}')
        print(f'T={T}')

        results_name = f'T{2 ** T}_G{2 ** G}_kappa{ratio}'

        _info = np.load(data_folder_path + results_name + f'_info.npz', allow_pickle=True)

        expanded_data = {key: _info[key] for key in _info.files}

        nb_of_pixels_global = _info['nb_of_pixels']
        phase_fied = _info['nb_of_sampling_points']
        norms = {}
        norms['data_scaled_rr'] = _info['norm_rMr_G']
        norms['energy_upper_bound'] = _info['norm_UB_G']
        norms['residual_rr'] = _info['norm_rr_G']
        eig_G = _info['eigens_G']

        eig_vect_G = _info['eig_vect_G']
        rhs = _info['rhs']
        weight = _info['weights']

        x_values = _info['x_values']
        ritz_values = _info['ritz_values']
        eig_G_no_zero = np.real(_info['eig_G_no_zero'])

        #
        ax_eigens = fig.add_subplot(gs[counter, 0])

        ax_eigens.plot(sorted(eig_G)[2:], color='Green', label=f'Green',
                       alpha=0.5, marker='.', linewidth=0, markersize=5)

        #
        # plot_cg_polynomial_JG_paper(x_values, ritz_values, true_eigenvalues=np.real(eig_G_no_zero),
        #                             weight=weight,
        #                             error_evol=norms['energy_upper_bound'] / norms['residual_rr'][
        #                                 0], title='Green', iteration=iteration_to_plot)  # energy_lb
        ax_poly = fig.add_subplot(gs[counter, 2])
        iteration_to_plot = 2 ** G - 1

        i = iteration_to_plot
        if i == 0:  # Zero order polynomial is constant
            polynomial_value = np.ones(len(x_values))
        else:
            polynomial_value = get_cg_polynomial(x_values, ritz_values[i - 1, :i])
        #            ax_error_true = fig.add_subplot(gs[1, 1])
        ax_poly.scatter(np.real(eig_G_no_zero), [0] * len(eig_G_no_zero), color='green', marker='|',
                        label="Eigenvalues")
        ax_poly.plot(x_values, polynomial_value, color='red', label=r'$\varphi^{CG}$' + f'$_{{{i}}}$')
        ax_poly.hlines(xmin=0, xmax=x_values[-1], y=0, linestyles='--', color='gray')

        if i == 0:  # Zero order polynomial is constant
            ax_poly.scatter(-1, -1, color='red', marker='x',
                            label=f"Ritz Values\n (Approx Eigenvalues)")
        else:  # Zero order polynomial is constant
            ax_poly.scatter(np.real(ritz_values[i - 1, :i]), [0] * len(ritz_values[i - 1, :i]), color='red', marker='x',
                            label=f"Roots of " + r'$\varphi^{CG}$' + f'$_{{{i}}}$')  # +"(Approx Eigenvalues)"
        ax_poly.set_xticks([1, 34, 67, 100])
        ax_poly.set_xticklabels([1, 34, 67, 100])

        ax_weights = fig.add_subplot(gs[counter, 1])
        if weight is not None:
            ax_weights.scatter(np.real(weight) / np.real(eig_G_no_zero), np.real(eig_G_no_zero), color='blue',
                               marker='o', label=r"non-zero weights- $w_{i}^{2}/ \lambda_{i}$")
            ax_weights.set_xscale('log')

            ax_weights.set_yticks([1, 34, 67, 100])
            ax_weights.set_yticklabels([1, 34, 67, 100])

            ax_weights.set_xlim(1e-10, 1)
            ax_weights.set_xticks([1e-10, 1e-5, 1])
            ax_weights.set_xticklabels([fr'$10^{{{-10}}}$', fr'$10^{{{-5}}}$', fr'$10^{{{0}}}$'])
            # ax_weights.set_xlim(-0.1, x_values[-1] + 0.3)
            #
            # ax_weights.set_xlabel('eigenvalue index - $i$ (sorted)')
            #
            # ax_weights.set_xticks([1, 34, 67, 100])
            # ax_weights.set_xticklabels([1, 34, 67, 100])
            # ax_weights.legend(ncol=1, loc='lower left')

        # ax_poly.set_xlabel("Eigenvalues --- Approximation")
        # ax_poly.set_ylabel("CG (Lanczos) Iteration")
        # ax_poly.set_title(f"CG polynomial (Lanczos Iteration) {{{i}}}")
        ax_poly.set_ylim(-2, 2)
        ax_poly.set_xlim(-0.1, 100 + 0.3)
        ax_poly.set_xticks([])
        ax_poly.set_xticklabels([])

        ax_poly.legend(ncol=3, loc='upper left')

        counter += 1

    plt.show()

##### PLOT FOR DATA REFINEMENT
ratio = 2

geometry_n = [2, 3, 4]
discretization_n = [5]  #
# create a figure
fig = plt.figure(figsize=(8.3, 5.0))

gs = fig.add_gridspec(3, 3, hspace=0.2, wspace=0.2, width_ratios=[1, 1, 1],
                      height_ratios=[1, 1, 1])

for ii in np.arange(np.size(discretization_n)):
    T = discretization_n[ii]
    j2 = [x for x in geometry_n if x >= T]
    counter = 0
    for jj in np.arange(np.size(geometry_n)):
        G = geometry_n[jj]
        print(f'G={G}')
        print(f'T={T}')

        results_name = f'T{2 ** T}_G{2 ** G}_kappa{ratio}'

        _info = np.load(data_folder_path + results_name + f'_info.npz', allow_pickle=True)

        expanded_data = {key: _info[key] for key in _info.files}

        nb_of_pixels_global = _info['nb_of_pixels']
        phase_fied = _info['nb_of_sampling_points']
        norms = {}
        norms['data_scaled_rr'] = _info['norm_rMr_G']
        norms['energy_upper_bound'] = _info['norm_UB_G']
        norms['residual_rr'] = _info['norm_rr_G']
        eig_G = _info['eigens_G']

        eig_vect_G = _info['eig_vect_G']
        rhs = _info['rhs']
        weight = _info['weights']

        x_values = _info['x_values']
        ritz_values = _info['ritz_values']
        eig_G_no_zero = np.real(_info['eig_G_no_zero'])

        #
        ax_eigens = fig.add_subplot(gs[counter, 0])

        ax_eigens.plot(sorted(eig_G)[2:], color='Green', label=f'Green',
                       alpha=0.5, marker='.', linewidth=0, markersize=5)

        #
        # plot_cg_polynomial_JG_paper(x_values, ritz_values, true_eigenvalues=np.real(eig_G_no_zero),
        #                             weight=weight,
        #                             error_evol=norms['energy_upper_bound'] / norms['residual_rr'][
        #                                 0], title='Green', iteration=iteration_to_plot)  # energy_lb
        ax_poly = fig.add_subplot(gs[counter, 2])
        iteration_to_plot = 2 ** G - 1

        i = iteration_to_plot
        if i == 0:  # Zero order polynomial is constant
            polynomial_value = np.ones(len(x_values))
        else:
            polynomial_value = get_cg_polynomial(x_values, ritz_values[i - 1, :i])

        ax_weights = fig.add_subplot(gs[counter, 1])
        #            ax_error_true = fig.add_subplot(gs[1, 1])
        ax_poly.scatter([0] * len(eig_G_no_zero), np.real(eig_G_no_zero), color='green', marker='|',
                        label="Eigenvalues")
        ax_poly.plot(polynomial_value, x_values, color='red', label=r'$\varphi^{CG}$' + f'$_{{{i}}}$')
        ax_poly.vlines(ymin=0, ymax=x_values[-1], x=0, linestyles='--', color='gray')

        if i == 0:  # Zero order polynomial is constant
            ax_poly.scatter(-1, -1, color='red', marker='x',
                            label=f"Ritz Values\n (Approx Eigenvalues)")
        else:  # Zero order polynomial is constant
            ax_poly.scatter([0] * len(ritz_values[i - 1, :i]), np.real(ritz_values[i - 1, :i]), color='red', marker='x',
                            label=f"Roots of " + r'$\varphi^{CG}$' + f'$_{{{i}}}$')  # +"(Approx Eigenvalues)"
        ax_poly.set_yticks([1, 34, 67, 100])
        ax_poly.set_yticklabels([1, 34, 67, 100])

        if weight is not None:
            ax_weights.scatter(np.real(weight) / np.real(eig_G_no_zero), np.real(eig_G_no_zero), color='blue',
                               marker='o', label=r"non-zero weights- $w_{i}/ \lambda_{i}$")
            ax_weights.set_xscale('log')

            ax_weights.set_yticks([1, 34, 67, 100])
            ax_weights.set_yticklabels([1, 34, 67, 100])

            ax_weights.set_xlim(1e-10, 1)
            ax_weights.set_xticks([1e-10, 1e-5, 1])
            ax_weights.set_xticklabels([fr'$10^{{{-10}}}$', fr'$10^{{{-5}}}$', fr'$10^{{{0}}}$'])
            # ax_weights.set_xlim(-0.1, x_values[-1] + 0.3)
            #
            # ax_weights.set_xlabel('eigenvalue index - $i$ (sorted)')
            #
            # ax_weights.set_xticks([1, 34, 67, 100])
            # ax_weights.set_xticklabels([1, 34, 67, 100])
            # ax_weights.legend(ncol=1, loc='lower left')

        # ax_poly.set_xlabel("Eigenvalues --- Approximation")
        # ax_poly.set_ylabel("CG (Lanczos) Iteration")
        # ax_poly.set_title(f"CG polynomial (Lanczos Iteration) {{{i}}}")
        ax_poly.set_xlim(-2, 2)
        ax_poly.set_ylim(-0.1, 100 + 0.3)
        ax_poly.set_yticks([])
        ax_poly.set_yticklabels([])

        ax_poly.legend(ncol=3, loc='upper left')

        counter += 1

    plt.show()

print()
# phase_field = np.load('../exp_data/' + file_data_name + f'.npy', allow_pickle=True)
# np.savez(data_folder_path + results_name + f'_info.npz',
#          **{key: np.array(value, dtype=object) for key, value in _info.items()})
