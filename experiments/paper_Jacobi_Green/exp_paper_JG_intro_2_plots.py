from cProfile import label

import numpy as np
import scipy as sc
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpi4py import MPI
from NuMPI.Tools import Reduction
import matplotlib.pyplot as plt

from NuMPI.IO import save_npy, load_npy
from IPython.terminal.shortcuts.filters import KEYBINDING_FILTERS
from PySide2.examples.opengl.contextinfo import colors
from matplotlib.animation import FuncAnimation, PillowWriter
from sympy.physics.quantum.sho1d import omega

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library
from mpl_toolkits import mplot3d

src = '../figures/'
# Enable LaTeX rendering
plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX
    "font.family": "helvetica",  # Use a serif font
})

colors = ['red', 'blue', 'green', 'orange', 'purple','olive','brown','purple']
linestyles = [':', '-.', '--', (0, (3, 1, 1, 1))]
# markers = ['x', 'o', '|', '>']
markers = ["x",  "|", ".","v", "<", ">","o",  "^", ".", ",", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+",
           "X", "D", "d", "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
           ]

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'

script_name = 'exp_paper_JG_intro_2'
domain_size = [1, 1]
nb_pix_multip = 2
# material distribution
geometry_ID = 'n_laminate'
number_of_pixels = (nb_pix_multip * 16, nb_pix_multip * 16)

# create a figure
fig = plt.figure(figsize=(5.5, 5.5))
gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.2, width_ratios=[1, 1, 1],
                      height_ratios=[1, 0.05, 1])
cbar_ax = fig.add_subplot(gs[1, :])
ax_cross = fig.add_subplot(gs[2, :])
# plot phases

############################################# plot material phases
x = np.arange(0, number_of_pixels[0])
y = np.arange(0, number_of_pixels[1])
X, Y = np.meshgrid(x, y)
counter = 0
kontrast = 100

T = number_of_pixels[0]
for G in [2, 8, 32]:
    file_data_name = (
        f'{script_name}_gID{geometry_ID}_T{T}_G{G}_kappa{kontrast}.npy')
    folder_name = '../exp_data/'

    xopt = np.load('../exp_data/' + file_data_name + f'xopt_log.npz', allow_pickle=True)
    phase_field = np.load('../exp_data/' + file_data_name + f'.npy', allow_pickle=True)
    if counter < 3:
        ax_geom = fig.add_subplot(gs[0, counter])
    elif counter < 4:
        ax_geom = fig.add_subplot(gs[0, counter - 2])
    pcm = ax_geom.pcolormesh(X, Y, np.transpose(phase_field), cmap=mpl.cm.Greys, vmin=1, vmax=1e2, linewidth=0,
                             rasterized=True)
    ax_geom.set_xticks(np.arange(-.5, number_of_pixels[0], int(number_of_pixels[0] / 4)))
    ax_geom.set_yticks(np.arange(-.5, number_of_pixels[1], int(number_of_pixels[1] / 4)))
    ax_geom.set_xticklabels(np.arange(0, number_of_pixels[0] + 1, int(number_of_pixels[0] / 4)))
    ax_geom.set_yticklabels(np.arange(0, number_of_pixels[1] + 1, int(number_of_pixels[1] / 4)))
    ax_geom.set_title(f'{G} phases')
    ax_geom.hlines(y=10, xmin=-0.5, xmax=number_of_pixels[0] - 0.5, color=colors[counter],
                   linestyle=linestyles[counter], linewidth=1.)
    ax_geom.yaxis.set_label_position('right')
    ax_geom.yaxis.tick_right()
    ax_geom.tick_params(labelbottom=True, labeltop=False, labelleft=False, labelright=False,
                        bottom=True, top=False, left=False, right=False)
    if counter == 0:
        # Set ylabel to the right
        ax_geom.yaxis.set_label_position('left')
        ax_geom.yaxis.tick_left()
        ax_geom.set_ylabel(r'pixel index')
        ax_geom.set_xlabel(r'pixel index')
        ax_geom.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                            bottom=True, top=False, left=True, right=False)
    ax_geom.set_aspect('equal')

    divnorm = mpl.colors.Normalize(vmin=1, vmax=100)

    cbar = plt.colorbar(pcm, location='bottom', cax=cbar_ax, ticklocation='bottom',
                        orientation='horizontal', ticks=[1, 25, 50, 75, 100])  # Specify the ticks

    cbar_ax.set_title(f'Bulk modulus (Pa)')  # , y=-0.7,pad=-22

    extended_x = np.arange(phase_field[:, phase_field.shape[0] // 2].size + 1)
    extended_y = np.append(phase_field[:, phase_field.shape[0] // 2],
                           phase_field[:, phase_field.shape[0] // 2][-1])
    ax_cross.step(extended_x, extended_y
                  , where='post',
                  linewidth=1, color=colors[counter], linestyle=linestyles[counter], marker='|',
                  label=f'{G} phases')
    # ax3.plot(phase_field[:, phase_field.shape[0] // 2], linewidth=1)
    ax_cross.set_ylim([0, 101])
    ax_cross.set_xlim([0, phase_field.shape[0] - 1])
    ax_cross.set_yticks([1, 50, 100])
    ax_cross.set_yticklabels([1, 50, 100])
    ax_cross.set_xticks(np.arange(-.5, number_of_pixels[0], int(number_of_pixels[0] / 4)))
    ax_cross.set_xticklabels(np.arange(0, number_of_pixels[0] + 1, int(number_of_pixels[0] / 4)))

    # ax1.yaxis.set_ticks_position([0.001,0.25,0.5,0.75, 1])
    # ax2.legend(['2 phases', f'{ratio} phases', 'Jacobi', 'Green + Jacobi'])
    ax_cross.legend(loc="upper left")

    ax_cross.set_title(f'Cross sections')
    ax_cross.set_ylabel('Bulk modulus (Pa)')
    ax_cross.set_xlabel('pixel index')
    # ax_cross.step(extended_x, extended_y
    #          , where='post',
    #          linewidth=1, color=colors[counter], linestyle=linestyles[counter], marker='|',
    #          label=f'{G} phases')
    # # ax3.plot(phase_field[:, phase_field.shape[0] // 2], linewidth=1)
    # ax_cross.set_ylim([0, 101])
    # ax_cross.set_xlim([0, phase_field.shape[0]])
    # ax_cross.set_yticks([1,50,100])
    # ax_cross.set_yticklabels([1,50,100])
    # # ax1.yaxis.set_ticks_position([0.001,0.25,0.5,0.75, 1])
    # # ax2.legend(['2 phases', f'{ratio} phases', 'Jacobi', 'Green + Jacobi'])
    # ax_cross.legend(loc="upper left")
    #
    # ax_cross.set_title(f'Cross sections')
    # ax_cross.set_ylabel('Young modulus (Pa)')
    # ax_cross.set_xlabel('x coordinate')

    counter += 1
fname = src + script_name + 'phases' + f'{kontrast}_{geometry_ID}' + '{}'.format('.pdf')
print(('create figure: {}'.format(fname)))
plt.savefig(fname, bbox_inches='tight')
plt.show()
################################################# Plot iterations
# create a figure
fig = plt.figure(figsize=(5.5, 5.5))
gs = fig.add_gridspec(1, 1, hspace=0.3, wspace=0.1, width_ratios=[1],
                      height_ratios=[1])

ax_nb_it = fig.add_subplot(gs[0, 0])

T = number_of_pixels[0]
G = 32
print(f'G={G}')
print(f'T={T}')
counter = 0
linewidths = [1, 1, 3]
for kontrast in [2, 10, 100]:
    file_data_name = (
        f'{script_name}_gID{geometry_ID}_T{T}_G{G}_kappa{kontrast}.npy')
    folder_name = '../exp_data/'

    xopt = np.load('../exp_data/' + file_data_name + f'xopt_log.npz', allow_pickle=True)
    ax_nb_it.plot(np.arange(2, len(xopt.f.nb_it_G[0]) + 2), xopt.f.nb_it_G[0], 'g', linestyle=linestyles[counter],
                  label='Green', linewidth=linewidths[counter], markerfacecolor='white')
    ax_nb_it.plot(np.arange(2, len(xopt.f.nb_it_J[0]) + 2), xopt.f.nb_it_J[0], "b", linestyle=linestyles[counter],
                  label='Jacobi', linewidth=linewidths[counter], markerfacecolor='white')
    ax_nb_it.plot(np.arange(2, len(xopt.f.nb_it_JG[0]) + 2), xopt.f.nb_it_JG[0], "k", linestyle=linestyles[counter],
                  label='Jacobi-Green ', linewidth=linewidths[counter],
                  markerfacecolor='white')
    ax_nb_it.set_ylabel('\# PCG iterations')
    ax_nb_it.set_xlabel('\# material phases (interphases)')

    ax_nb_it.set_ylim([2, 70])
    ax_nb_it.set_yticks([2, 8, 16, 24, 32])
    ax_nb_it.set_yticklabels([2, 8, 16, 24, 32])

    ax_nb_it.set_xlim([2, 32])
    ax_nb_it.set_xticks([2, 8, 16, 24, 32])
    ax_nb_it.set_xticklabels([2, 8, 16, 24, 32])
    ax_nb_it.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=True,
                         bottom=True, top=False, left=True, right=True)

   # ax_nb_it.set_aspect('equal')

    counter += 1
ax_nb_it.annotate(text=f'Green \n contrast = 2',
                  xy=(24, 12.0),
                  xytext=(18., 15.),
                  arrowprops=dict(arrowstyle='->',
                                  color='Green',
                                  lw=1,
                                  ls=linestyles[0]),
                  color='Green'
                  )
ax_nb_it.annotate(text=f'Green \n contrast = 10',
                  xy=(24, 23.0),
                  xytext=(22., 18.),
                  arrowprops=dict(arrowstyle='->',
                                  color='Green',
                                  lw=1,
                                  ls=linestyles[1]),
                  color='Green'
                  )
ax_nb_it.annotate(text=f'Green \n contrast = 100',
                  xy=(26, 26),
                  xytext=(18., 28.),
                  arrowprops=dict(arrowstyle='->',
                                  color='Green',
                                  lw=1,
                                  ls=linestyles[2]),
                  color='Green'
                  )
ax_nb_it.annotate(text=f'Jacobi-Green \n contrast = 2',
                  xy=(13, 7.0),
                  xytext=(6., 3.),
                  arrowprops=dict(arrowstyle='->',
                                  color='Black',
                                  lw=1,
                                  ls=linestyles[0]),
                  color='Black'
                  )
ax_nb_it.annotate(text=f'Jacobi-Green \n contrast = 10',
                  xy=(24, 9.0),
                  xytext=(20., 3.),
                  arrowprops=dict(arrowstyle='->',
                                  color='Black',
                                  lw=1,
                                  ls=linestyles[1]),
                  color='Black'
                  )
ax_nb_it.annotate(text=f'Jacobi-Green \n contrast = 100',
                  xy=(8, 15),
                  xytext=(5., 17.),
                  arrowprops=dict(arrowstyle='->',
                                  color='Black',
                                  lw=1,
                                  ls=linestyles[2]),
                  color='Black'
                  )
ax_nb_it.annotate(text=f'Jacobi contrast \n [2,10,100]',
                  xy=(10, 32.0),
                  xytext=(10., 25.0),
                  arrowprops=dict(arrowstyle='->',
                                  color='Blue',
                                  lw=1,
                                  ls='-'),
                  color='Blue'
                  )

fname = src + script_name + 'intro' + f'{kontrast}_{geometry_ID}' + '{}'.format('.pdf')
print(('create figure: {}'.format(fname)))
plt.savefig(fname, bbox_inches='tight')
plt.show()


# create a figure
fig = plt.figure(figsize=(5.5, 5.5))
gs = fig.add_gridspec(1, 1, hspace=0.3, wspace=0.1, width_ratios=[1],
                      height_ratios=[1])

ax_nb_it = fig.add_subplot(gs[0, 0])

T = number_of_pixels[0]
G = 32
print(f'G={G}')
print(f'T={T}')
counter = 0
linewidths = [3]
for kontrast in [  100]:
    file_data_name = (
        f'{script_name}_gID{geometry_ID}_T{T}_G{G}_kappa{kontrast}.npy')
    folder_name = '../exp_data/'

    xopt = np.load('../exp_data/' + file_data_name + f'xopt_log.npz', allow_pickle=True)
    ax_nb_it.plot(np.arange(2, len(xopt.f.nb_it_G[0]) + 2), xopt.f.nb_it_G[0], 'g', linestyle=linestyles[counter+2],
                  label='Green', linewidth=linewidths[counter], markerfacecolor='white')
    ax_nb_it.plot(np.arange(2, len(xopt.f.nb_it_J[0]) + 2), xopt.f.nb_it_J[0], "b", linestyle=linestyles[counter+2],
                  label='Jacobi', linewidth=linewidths[counter], markerfacecolor='white')
    ax_nb_it.plot(np.arange(2, len(xopt.f.nb_it_JG[0]) + 2), xopt.f.nb_it_JG[0], "k", linestyle=linestyles[counter+2],
                  label='Jacobi-Green ', linewidth=linewidths[counter],
                  markerfacecolor='white')
    ax_nb_it.set_ylabel('\# PCG iterations')
    ax_nb_it.set_xlabel('\# material phases (interphases)')

    ax_nb_it.set_ylim([2,70])
    ax_nb_it.set_yticks([2, 8, 16, 24, 32])
    ax_nb_it.set_yticklabels([2, 8, 16, 24, 32])

    ax_nb_it.set_xlim([2, 32])
    ax_nb_it.set_xticks([2, 8, 16, 24, 32])
    ax_nb_it.set_xticklabels([2, 8, 16, 24, 32])
    ax_nb_it.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=True,
                         bottom=True, top=False, left=True, right=True)

    ax_nb_it.set_aspect('equal')

    counter += 1
ax_nb_it.annotate(text=f'Green \n contrast = 2',
                  xy=(24, 12.0),
                  xytext=(18., 15.),
                  arrowprops=dict(arrowstyle='->',
                                  color='Green',
                                  lw=1,
                                  ls=linestyles[0]),
                  color='Green'
                  )
ax_nb_it.annotate(text=f'Green \n contrast = 10',
                  xy=(24, 23.0),
                  xytext=(22., 18.),
                  arrowprops=dict(arrowstyle='->',
                                  color='Green',
                                  lw=1,
                                  ls=linestyles[1]),
                  color='Green'
                  )
ax_nb_it.annotate(text=f'Green \n contrast = 100',
                  xy=(26, 26),
                  xytext=(18., 28.),
                  arrowprops=dict(arrowstyle='->',
                                  color='Green',
                                  lw=1,
                                  ls=linestyles[2]),
                  color='Green'
                  )

ax_nb_it.annotate(text=f'Jacobi-Green \n contrast = 100',
                  xy=(8, 15),
                  xytext=(5., 17.),
                  arrowprops=dict(arrowstyle='->',
                                  color='Black',
                                  lw=1,
                                  ls=linestyles[2]),
                  color='Black'
                  )
ax_nb_it.annotate(text=f'Jacobi contrast \n [100]',
                  xy=(10, 32.0),
                  xytext=(10., 25.0),
                  arrowprops=dict(arrowstyle='->',
                                  color='Blue',
                                  lw=1,
                                  ls='-'),
                  color='Blue'
                  )

fname = src + script_name + 'intro' + f'{kontrast}_{geometry_ID}_simple' + '{}'.format('.pdf')
print(('create figure: {}'.format(fname)))
plt.savefig(fname, bbox_inches='tight')
plt.show()

# create a figure
fig = plt.figure(figsize=(5.5, 5.5))
gs = fig.add_gridspec(1, 1, hspace=0.3, wspace=0.1, width_ratios=[1],
                      height_ratios=[1])

ax_nb_it = fig.add_subplot(gs[0, 0])

T = number_of_pixels[0]
G = 32
print(f'G={G}')
print(f'T={T}')
counter = 0
linewidths = [1,1,3]
for kontrast in [2,10,  100]:
    file_data_name = (
        f'{script_name}_gID{geometry_ID}_T{T}_G{G}_kappa{kontrast}.npy')
    folder_name = '../exp_data/'

    xopt = np.load('../exp_data/' + file_data_name + f'xopt_log.npz', allow_pickle=True)
    ax_nb_it.plot(np.arange(2, len(xopt.f.nb_it_G[0]) + 2), xopt.f.nb_it_G[0], 'g', linestyle=linestyles[counter],
                  label='Green', linewidth=linewidths[counter], markerfacecolor='white')
    # ax_nb_it.plot(np.arange(2, len(xopt.f.nb_it_J[0]) + 2), xopt.f.nb_it_J[0], "b", linestyle=linestyles[counter],
    #               label='Jacobi', linewidth=linewidths[counter], markerfacecolor='white')
    # ax_nb_it.plot(np.arange(2, len(xopt.f.nb_it_JG[0]) + 2), xopt.f.nb_it_JG[0], "k", linestyle=linestyles[counter],
    #               label='Jacobi-Green ', linewidth=linewidths[counter],
    #               markerfacecolor='white')
    ax_nb_it.set_ylabel('\# PCG iterations')
    ax_nb_it.set_xlabel('\# material phases (interphases)')

    ax_nb_it.set_ylim([2, 70])
    ax_nb_it.set_yticks([2, 8, 16, 24, 32])
    ax_nb_it.set_yticklabels([2, 8, 16, 24, 32])

    ax_nb_it.set_xlim([2, 32])
    ax_nb_it.set_xticks([2, 8, 16, 24, 32])
    ax_nb_it.set_xticklabels([2, 8, 16, 24, 32])
    ax_nb_it.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=True,
                         bottom=True, top=False, left=True, right=True)

    ax_nb_it.set_aspect('equal')

    counter += 1
ax_nb_it.annotate(text=f'Green \n contrast = 2',
                  xy=(24, 12.0),
                  xytext=(18., 15.),
                  arrowprops=dict(arrowstyle='->',
                                  color='Green',
                                  lw=1,
                                  ls=linestyles[0]),
                  color='Green'
                  )
ax_nb_it.annotate(text=f'Green \n contrast = 10',
                  xy=(24, 23.0),
                  xytext=(22., 18.),
                  arrowprops=dict(arrowstyle='->',
                                  color='Green',
                                  lw=1,
                                  ls=linestyles[1]),
                  color='Green'
                  )
ax_nb_it.annotate(text=f'Green \n contrast = 100',
                  xy=(26, 26),
                  xytext=(18., 28.),
                  arrowprops=dict(arrowstyle='->',
                                  color='Green',
                                  lw=1,
                                  ls=linestyles[2]),
                  color='Green'
                  )



fname = src + script_name + 'intro' + f'{kontrast}_{geometry_ID}_Green' + '{}'.format('.pdf')
print(('create figure: {}'.format(fname)))
plt.savefig(fname, bbox_inches='tight')
plt.show()

# create a figure
fig = plt.figure(figsize=(5.5, 5.5))
gs = fig.add_gridspec(1, 1, hspace=0.3, wspace=0.1, width_ratios=[1],
                      height_ratios=[1])

ax_iters = fig.add_subplot(gs[0, 0])
T = number_of_pixels[0]
G = 16
kontrast = 100
counter = 0

print(f'G={G}')
print(f'T={T}')
# for G in [4,16,32]:
for G in [8,16,32]:#, 16, 32
    counter_2 = 0

    for kontrast in [2, 10, 100]:
        file_data_name = (
            f'{script_name}_gID{geometry_ID}_T{T}_G{G}_kappa{kontrast}.npy')
        folder_name = '../exp_data/'

        xopt = np.load('../exp_data/' + file_data_name + f'xopt_log.npz', allow_pickle=True)

        ax_iters.semilogy(np.arange(1, len(xopt.f.norm_rMr_G) + 1), xopt.f.norm_rMr_G,
                          label=r'  Green ' + r'$\mathcal{T}$' + f'$_{{{2 ** T}}}$', color=colors[-counter-1],
                          linestyle=linestyles[counter],marker=markers[counter_2])
        # ax_iters.semilogy(np.arange(1, len(xopt.f.norm_rMr_JG) + 1), xopt.f.norm_rMr_JG,
        #               label=r' Jacobi-Green  ' + r'$\mathcal{T}$' + f'$_{{{2 ** T}}}$',
        #               color='Black', linestyle='-.', marker=markers[0])
        ax_iters.set_xlabel(r'PCG iteration - $k$')
        ax_iters.set_ylabel('Norm of residua - '
                            r'$||r_{k}||_{G^{-1}} $')
        ax_iters.set_title(f'Green preconditioner \n Convergence  ')
        ax_iters.set_ylim([1e-16, 1e4])
        ax_iters.set_yticks([1e-16,1e-14,1e-12,1e-8,1e-4,1e0,1e4])
        #ax_iters.set_yticklabels([1e-14,1e-8,1e-4,1e0,1e4])
        ax_iters.set_yticklabels([f'$10^{{{i}}}$' for i in [-16,-14,-12,-8,-4,0,4]])
        ax_iters.set_xlim([1, 32])
        ax_iters.set_xticks([1, 8, 16, 24, 32])
        ax_iters.set_xticklabels([1, 8, 16, 24, 32])
        ax_iters.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=True,
                             bottom=True, top=False, left=True, right=True)
        ax_iters.set_aspect('equal')
        counter_2 += 1

    counter += 1
    if G == 8:
        ax_iters.annotate(text=r'$\mathcal{G}$' + f'$_{{{8}}}$' + ' \n ' + '$\kappa$ = 100',
                          xy=(7, 1e1),
                          xytext=(10., 1e2),
                          arrowprops=dict(arrowstyle='->',
                                          color=colors[-1],
                                          lw=1,
                                          ls=linestyles[1]),
                          color=colors[-1]
                          )
        ax_iters.annotate(text=r'$\mathcal{G}$' + f'$_{{{8}}}$' + ' \n ' + '$\kappa$ = 10 ',
                          xy=(7, 1e-2),
                          xytext=(8., 1e-6),
                          arrowprops=dict(arrowstyle='->',
                                          color=colors[-1],
                                          lw=1,
                                          ls=linestyles[1]),
                          color=colors[-1]
                          )
        ax_iters.annotate(text=r'$\mathcal{G}$' + f'$_{{{8}}}$' + ' \n ' + '$\kappa$ = 2',
                          xy=(7, 1e-12),
                          xytext=(2., 2e-14),
                          arrowprops=dict(arrowstyle='->',
                                          color=colors[-1],
                                          lw=1,
                                          ls=linestyles[1]),
                          color=colors[-1]
                          )
    elif G == 16:
        ax_iters.annotate(text=r'$\mathcal{G}$' + f'$_{{{16}}}$' + ' \n ' + '$\kappa$ = 100',
                          xy=(14, 1e-2),
                          xytext=(16., 5e-4),
                          arrowprops=dict(arrowstyle='->',
                                          color=colors[-2],
                                          lw=1,
                                          ls=linestyles[2]),
                          color=colors[-2]
                          )
        ax_iters.annotate(text=r'$\mathcal{G}$' + f'$_{{{16}}}$' + ' \n ' + '$\kappa$ = 10 ',
                          xy=(14, 1e-6),
                          xytext=(10.5, 1e-9),
                          arrowprops=dict(arrowstyle='->',
                                          color=colors[-2],
                                          lw=1,
                                          ls=linestyles[2]),
                          color=colors[-2]
                          )
        ax_iters.annotate(text=r'$\mathcal{G}$' + f'$_{{{16}}}$' + ' \n ' + '$\kappa$ = 2',
                          xy=(9.5, 1e-10),
                          xytext=(8., 2e-14),
                          arrowprops=dict(arrowstyle='->',
                                          color=colors[-2],
                                          lw=1,
                                          ls=linestyles[2]),
                          color=colors[-2]
                          )
    elif G == 32:
        ax_iters.annotate(text=r'$\mathcal{G}$' + f'$_{{{32}}}$' + ' \n ' + '$\kappa$ = 100',
                          xy=(22, 1e-4),
                          xytext=(23., 1e-2),
                          arrowprops=dict(arrowstyle='->',
                                          color=colors[-3],
                                          lw=1,
                                          ls=linestyles[3]),
                          color=colors[-3]
                          )
        ax_iters.annotate(text=r'$\mathcal{G}$' + f'$_{{{32}}}$' + ' \n ' + '$\kappa$ = 10 ',
                          xy=(21, 2e-10),
                          xytext=(16., 1e-10),
                          arrowprops=dict(arrowstyle='->',
                                          color=colors[-3],
                                          lw=1,
                                          ls=linestyles[3]),
                          color=colors[-3]
                          )
        ax_iters.annotate(text=r'$\mathcal{G}$' + f'$_{{{32}}}$' + ' \n ' + '$\kappa$ = 2',
                          xy=(10.8, 1e-12),
                          xytext=(12.5, 2e-14),
                          arrowprops=dict(arrowstyle='->',
                                          color=colors[-3],
                                          lw=1,
                                          ls=linestyles[3]),
                          color=colors[-3]
                          )


ax_iters.hlines(y=1e-14, xmin=1, xmax=32, color='black',
               linestyle=':', linewidth=1.)



#r'$\mathcal{G}$'+f'$_{{{2**G}}}$'

fname = src + script_name + 'converg' + f'{kontrast}_{geometry_ID}_G{G}' + '{}'.format('.pdf')
print(('create figure: {}'.format(fname)))
plt.savefig(fname, bbox_inches='tight')
plt.show()












# create a figure
fig = plt.figure(figsize=(5.5, 5.5))
gs = fig.add_gridspec(1, 1, hspace=0.3, wspace=0.1, width_ratios=[1],
                      height_ratios=[1])

ax_iters = fig.add_subplot(gs[0, 0])
T = number_of_pixels[0]
G = 16
kontrast = 100
counter = 0

print(f'G={G}')
print(f'T={T}')
# for G in [4,16,32]:
for G in [8, 16, 32]:
    counter_2 = 0

    for kontrast in [2, 10, 100]:
        file_data_name = (
            f'{script_name}_gID{geometry_ID}_T{T}_G{G}_kappa{kontrast}.npy')
        folder_name = '../exp_data/'

        xopt = np.load('../exp_data/' + file_data_name + f'xopt_log.npz', allow_pickle=True)

        # ax_iters.semilogy(np.arange(1, len(xopt.f.norm_rMr_G) + 1), xopt.f.norm_rMr_G,
        #                   label=r'  Green ' + r'$\mathcal{T}$' + f'$_{{{2 ** T}}}$', color=colors[-counter-1],
        #                   linestyle=linestyles[counter],marker=markers[counter_2])
        ax_iters.semilogy(np.arange(1, len(xopt.f.norm_rMr_JG) + 1), xopt.f.norm_rMr_JG,
                      label=r' Jacobi-Green  ' + r'$\mathcal{G}$' + f'$_{{{G}}}$'
                            + r'$\kappa$' + f'$_{{{kontrast}}}$', color=colors[-counter-1],
                          linestyle=linestyles[counter],marker=markers[counter_2])
        ax_iters.set_xlabel(r'PCG iteration - $k$')
        ax_iters.set_ylabel('Norm of residua - '
                            r'$||r_{k}||_{G^{-1}} $')
        ax_iters.set_title(f'Green preconditioner \n Convergence  ')
        ax_iters.set_ylim([1e-16, 1e4])
        ax_iters.set_yticks([1e-16,1e-14,1e-12,1e-8,1e-4,1e0,1e4])
        #ax_iters.set_yticklabels([1e-14,1e-8,1e-4,1e0,1e4])
        ax_iters.set_yticklabels([f'$10^{{{i}}}$' for i in [-16,-14,-12,-8,-4,0,4]])
        ax_iters.set_xlim([1, 16])
        ax_iters.set_xticks([1, 8, 16 ])
        ax_iters.set_xticklabels([1, 8, 16])#, 24, 32
        ax_iters.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=True,
                             bottom=True, top=False, left=True, right=True)
        ax_iters.set_aspect('equal')
        counter_2 += 1
    plt.legend()
    counter += 1
ax_iters.hlines(y=1e-14, xmin=1, xmax=32, color='black',
               linestyle=':', linewidth=1.)
# ax_iters.annotate(text=r'$\mathcal{G}$'+f'$_{{{32}}}$'+' \n '+'$\kappa$ = 100',
#                   xy=(22, 1e-4),
#                   xytext=(23., 1e-2),
#                   arrowprops=dict(arrowstyle='->',
#                                   color=colors[-3],
#                                   lw=1,
#                                   ls=linestyles[3]),
#                   color=colors[-3]
#                   )
# ax_iters.annotate(text=r'$\mathcal{G}$'+f'$_{{{32}}}$'+' \n '+'$\kappa$ = 10 ',
#                   xy=(21, 2e-10),
#                   xytext=(16., 1e-10),
#                   arrowprops=dict(arrowstyle='->',
#                                   color=colors[-3],
#                                   lw=1,
#                                   ls=linestyles[3]),
#                   color=colors[-3]
#                   )
# ax_iters.annotate(text=r'$\mathcal{G}$'+f'$_{{{32}}}$'+' \n '+'$\kappa$ = 2',
#                   xy=(10.8, 1e-12),
#                   xytext=(12.5, 2e-14),
#                   arrowprops=dict(arrowstyle='->',
#                                   color=colors[-3],
#                                   lw=1,
#                                   ls=linestyles[3]),
#                   color=colors[-3]
#                   )
#
# ax_iters.annotate(text=r'$\mathcal{G}$'+f'$_{{{16}}}$'+' \n '+'$\kappa$ = 100',
#                   xy=(14, 1e-2),
#                   xytext=(16., 5e-4),
#                   arrowprops=dict(arrowstyle='->',
#                                   color=colors[-2],
#                                   lw=1,
#                                   ls=linestyles[2]),
#                   color=colors[-2]
#                   )
# ax_iters.annotate(text=r'$\mathcal{G}$'+f'$_{{{16}}}$'+' \n '+'$\kappa$ = 10 ',
#                   xy=(14, 1e-6),
#                   xytext=(10.5, 1e-9),
#                   arrowprops=dict(arrowstyle='->',
#                                   color=colors[-2],
#                                   lw=1,
#                                   ls=linestyles[2]),
#                   color=colors[-2]
#                   )
# ax_iters.annotate(text=r'$\mathcal{G}$'+f'$_{{{16}}}$'+' \n '+'$\kappa$ = 2',
#                   xy=(9.5, 1e-10),
#                   xytext=(8., 2e-14),
#                   arrowprops=dict(arrowstyle='->',
#                                   color=colors[-2],
#                                   lw=1,
#                                   ls=linestyles[2]),
#                   color=colors[-2]
#                   )
#
# ax_iters.annotate(text=r'$\mathcal{G}$'+f'$_{{{8}}}$'+' \n '+'$\kappa$ = 100',
#                   xy=(7, 1e1),
#                   xytext=(10., 1e2),
#                   arrowprops=dict(arrowstyle='->',
#                                   color=colors[-1],
#                                   lw=1,
#                                   ls=linestyles[1]),
#                   color=colors[-1]
#                   )
# ax_iters.annotate(text=r'$\mathcal{G}$'+f'$_{{{8}}}$'+' \n '+'$\kappa$ = 10 ',
#                   xy=(7, 1e-2),
#                   xytext=(8., 1e-6),
#                   arrowprops=dict(arrowstyle='->',
#                                   color=colors[-1],
#                                   lw=1,
#                                   ls=linestyles[1]),
#                   color=colors[-1]
#                   )
# ax_iters.annotate(text=r'$\mathcal{G}$'+f'$_{{{8}}}$'+' \n '+'$\kappa$ = 2',
#                   xy=(7, 1e-12),
#                   xytext=(2., 2e-14),
#                   arrowprops=dict(arrowstyle='->',
#                                   color=colors[-1],
#                                   lw=1,
#                                   ls=linestyles[1]),
#                   color=colors[-1]
#                   )



#r'$\mathcal{G}$'+f'$_{{{2**G}}}$'

fname = src + script_name + 'convergJG' + f'{kontrast}_{geometry_ID}' + '{}'.format('.pdf')
print(('create figure: {}'.format(fname)))
plt.savefig(fname, bbox_inches='tight')
plt.show()
quit()

# create a figure
fig = plt.figure(figsize=(6.5, 5.5))
gs = fig.add_gridspec(1, 2, hspace=0.5, wspace=0.4, width_ratios=[1, 1],
                      height_ratios=[1])
gs1 = gs[0, 0].subgridspec(3, 2, hspace=0.5, wspace=0.5, width_ratios=[1, 1],
                           height_ratios=[0.2, 1, 1])
cbar_ax = fig.add_subplot(gs1[0, :])
gs2 = gs[0, 1].subgridspec(3, 1, hspace=0.5, wspace=0.2, width_ratios=[1], height_ratios=[1, 1, 1])

ax_nb_it = fig.add_subplot(gs2[:, 0])
# plot phases

############################################# plot material phases
x = np.arange(0, number_of_pixels[0])
y = np.arange(0, number_of_pixels[1])
X, Y = np.meshgrid(x, y)
counter = 0
kontrast = 100

T = number_of_pixels[0]
for G in [4, 8, 32]:
    file_data_name = (
        f'{script_name}_gID{geometry_ID}_T{T}_G{G}_kappa{kontrast}.npy')
    folder_name = '../exp_data/'

    xopt = np.load('../exp_data/' + file_data_name + f'xopt_log.npz', allow_pickle=True)
    phase_field = np.load('../exp_data/' + file_data_name + f'.npy', allow_pickle=True)

    ax_geom = fig.add_subplot(gs1[1 + counter // 2, counter // 2])
    pcm = ax_geom.pcolormesh(X, Y, np.transpose(phase_field), cmap=mpl.cm.Greys, vmin=1, vmax=1e2, linewidth=0,
                             rasterized=True)
    ax_geom.set_xticks(np.arange(-.5, number_of_pixels[0], int(number_of_pixels[0] / 4)))
    ax_geom.set_yticks(np.arange(-.5, number_of_pixels[1], int(number_of_pixels[1] / 4)))
    ax_geom.set_xticklabels(np.arange(0, number_of_pixels[0] + 1, int(number_of_pixels[0] / 4)))
    ax_geom.set_yticklabels(np.arange(0, number_of_pixels[1] + 1, int(number_of_pixels[1] / 4)))
    ax_geom.set_title(f'{G} phases')
    ax_geom.hlines(y=10, xmin=-0.5, xmax=number_of_pixels[0] - 0.5, color=colors[counter],
                   linestyle=linestyles[counter], linewidth=1.)
    if counter == 2:
        ax_geom.set_ylabel('y coordinate')
        ax_geom.set_xlabel('x coordinate')

    divnorm = mpl.colors.Normalize(vmin=1, vmax=100)

    cbar = plt.colorbar(pcm, location='top', cax=cbar_ax, ticklocation='top',
                        orientation='horizontal', ticks=[1, 25, 50, 75, 100])  # Specify the ticks
    ax_geom.set_aspect('equal')

    counter += 1

################################################# Plot iterations


T = number_of_pixels[0]
G = 32
print(f'G={G}')
print(f'T={T}')
for kontrast in [2, 10, 100]:
    file_data_name = (
        f'{script_name}_gID{geometry_ID}_T{T}_G{G}_kappa{kontrast}.npy')
    folder_name = '../exp_data/'

    xopt = np.load('../exp_data/' + file_data_name + f'xopt_log.npz', allow_pickle=True)
    ax_nb_it.plot(np.arange(1, len(xopt.f.nb_it_G[0]) + 1), xopt.f.nb_it_G[0], 'g', marker='o', label=' Green',
                  linewidth=1, markerfacecolor='white')
    ax_nb_it.plot(np.arange(1, len(xopt.f.nb_it_J[0]) + 1), xopt.f.nb_it_J[0], "b", marker='^', label='PCG Jacobi',
                  linewidth=1, markerfacecolor='white')
    ax_nb_it.plot(np.arange(1, len(xopt.f.nb_it_JG[0]) + 1), xopt.f.nb_it_JG[0], "k", marker='x',
                  label='PCG Jacobi-Green ', linewidth=1,
                  markerfacecolor='white')

plt.show()

# create a figure
fig = plt.figure(figsize=(11, 5.5))
gs = fig.add_gridspec(1, 2, hspace=0.5, wspace=0.4, width_ratios=[1, 2],
                      height_ratios=[1])
gs1 = gs[0, 0].subgridspec(4, 1, hspace=0.5, wspace=0.5, width_ratios=[1],
                           height_ratios=[0.2, 1, 1, 1])
cbar_ax = fig.add_subplot(gs1[0, 0])
gs2 = gs[0, 1].subgridspec(3, 1, hspace=0.5, wspace=0.2, width_ratios=[1], height_ratios=[1, 1, 1])

ax_cross = fig.add_subplot(gs2[0, 0])
ax_nb_it = fig.add_subplot(gs2[1, 0])
ax_iters = fig.add_subplot(gs2[2, 0])
ax_iters.text(-0.25, 1.05, '(a.1)', transform=ax_iters.transAxes)
# plot phases

############################################# plot material phases
x = np.arange(0, number_of_pixels[0])
y = np.arange(0, number_of_pixels[1])
X, Y = np.meshgrid(x, y)
counter = 0
kontrast = 100

T = number_of_pixels[0]
for G in [4, 8, 32]:
    file_data_name = (
        f'{script_name}_gID{geometry_ID}_T{T}_G{G}_kappa{kontrast}.npy')
    folder_name = '../exp_data/'

    xopt = np.load('../exp_data/' + file_data_name + f'xopt_log.npz', allow_pickle=True)
    phase_field = np.load('../exp_data/' + file_data_name + f'.npy', allow_pickle=True)

    ax_geom = fig.add_subplot(gs1[counter + 1, 0])
    pcm = ax_geom.pcolormesh(X, Y, np.transpose(phase_field), cmap=mpl.cm.Greys, vmin=1, vmax=1e2, linewidth=0,
                             rasterized=True)
    ax_geom.set_xticks(np.arange(-.5, number_of_pixels[0], int(number_of_pixels[0] / 4)))
    ax_geom.set_yticks(np.arange(-.5, number_of_pixels[1], int(number_of_pixels[1] / 4)))
    ax_geom.set_xticklabels(np.arange(0, number_of_pixels[0] + 1, int(number_of_pixels[0] / 4)))
    ax_geom.set_yticklabels(np.arange(0, number_of_pixels[1] + 1, int(number_of_pixels[1] / 4)))
    ax_geom.set_title(f'{G} phases')
    ax_geom.hlines(y=10, xmin=-0.5, xmax=number_of_pixels[0] - 0.5, color=colors[counter],
                   linestyle=linestyles[counter], linewidth=1.)
    if counter == 2:
        ax_geom.set_ylabel('y coordinate')
        ax_geom.set_xlabel('x coordinate')

    divnorm = mpl.colors.Normalize(vmin=1, vmax=100)

    cbar = plt.colorbar(pcm, location='top', cax=cbar_ax, ticklocation='top',
                        orientation='horizontal', ticks=[1, 25, 50, 75, 100])  # Specify the ticks
    ax_geom.set_aspect('equal')

    extended_x = np.arange(phase_field[:, phase_field.shape[0] // 2].size + 1)
    extended_y = np.append(phase_field[:, phase_field.shape[0] // 2],
                           phase_field[:, phase_field.shape[0] // 2][-1])

    ax_cross.step(extended_x, extended_y
                  , where='post',
                  linewidth=1, color=colors[counter], linestyle=linestyles[counter], marker='|',
                  label=f'{G} phases')
    # ax3.plot(phase_field[:, phase_field.shape[0] // 2], linewidth=1)
    ax_cross.set_ylim([0, 101])
    ax_cross.set_xlim([0, phase_field.shape[0]])
    ax_cross.set_yticks([1, 50, 100])
    ax_cross.set_yticklabels([1, 50, 100])
    # ax1.yaxis.set_ticks_position([0.001,0.25,0.5,0.75, 1])
    # ax2.legend(['2 phases', f'{ratio} phases', 'Jacobi', 'Green + Jacobi'])
    ax_cross.legend(loc="upper left")

    ax_cross.set_title(f'Cross sections')
    ax_cross.set_ylabel('Young modulus (Pa)')
    ax_cross.set_xlabel('x coordinate')

    counter += 1

################################################# Plot iterations


T = number_of_pixels[0]
G = 32
print(f'G={G}')
print(f'T={T}')
for kontrast in [2, 10, 100]:
    file_data_name = (
        f'{script_name}_gID{geometry_ID}_T{T}_G{G}_kappa{kontrast}.npy')
    folder_name = '../exp_data/'

    xopt = np.load('../exp_data/' + file_data_name + f'xopt_log.npz', allow_pickle=True)
    ax_nb_it.plot(np.arange(1, len(xopt.f.nb_it_G[0]) + 1), xopt.f.nb_it_G[0], 'g', marker='o', label=' Green',
                  linewidth=1, markerfacecolor='white')
    ax_nb_it.plot(np.arange(1, len(xopt.f.nb_it_J[0]) + 1), xopt.f.nb_it_J[0], "b", marker='^', label='PCG Jacobi',
                  linewidth=1, markerfacecolor='white')
    ax_nb_it.plot(np.arange(1, len(xopt.f.nb_it_JG[0]) + 1), xopt.f.nb_it_JG[0], "k", marker='x',
                  label='PCG Jacobi-Green ', linewidth=1,
                  markerfacecolor='white')
