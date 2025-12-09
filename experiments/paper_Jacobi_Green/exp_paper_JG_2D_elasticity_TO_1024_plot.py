import numpy as np
import scipy as sc
import time
import os
import sys
import gc
import argparse

sys.path.append("/home/martin/Programming/muFFTTO_paralellFFT_test/muFFTTO")
sys.path.append('../..')

from NuMPI.IO import save_npy, load_npy
from mpi4py import MPI

from muFFTTO import domain
from muFFTTO import solvers

script_name = os.path.splitext('exp_paper_JG_2D_elasticity_TO_1024')[0]
file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
figure_folder_path = file_folder_path + '/figures/' + script_name + '/'

if not os.path.exists(data_folder_path):
    os.makedirs(data_folder_path)

if not os.path.exists(figure_folder_path):
    os.makedirs(figure_folder_path)

parser = argparse.ArgumentParser(
    prog="exp_paper_JG_2D_elasticity_TO_1024_plot.py",
    description="Solve non-linear elasticity example "
                "from J.Zeman et al., Int. J. Numer. Meth. Engng 111, 903–926 (2017)."
)
parser.add_argument("-n", "--nb_pixel", default="1024")
parser.add_argument("-it", "--iteration", default="1")

parser.add_argument(
    "-p", "--preconditioner_type",
    type=str,
    choices=["Green", "Jacobi", "Green_Jacobi"],  # example options
    default="Green",
    help="Type of preconditioner to use"
)
args = parser.parse_args()

n_pix = int(args.nb_pixel)
number_of_pixels = (n_pix, n_pix)  # (1024, 1024)
iteration = args.iteration
preconditioner_type = args.preconditioner_type

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'

import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams["text.usetex"] = True
plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX
    # "font.family": "helvetica",  # Use a serif font
})
plt.rcParams.update({'font.size': 11})
plt.rcParams["font.family"] = "Arial"
nb_iterations_G = []
nb_iterations_J = []
nb_iterations_GJ = []
nb_iterations_G_512 = []
nb_iterations_J_512 = []
nb_iterations_GJ_512 = []
max_phase = []
min_phase = []
contrast = []
max_phase_512  = []
min_phase_512  = []
contrast_512  = []

grad_norm = []
grad_max = []
grad_max_inf = []
grad_norm_512 = []
grad_max_512 = []
grad_max_inf_512 = []

_info = {}
max_it = 500
for iteration in np.arange(1, max_it):
    try:
        info_log_final_G = np.load(data_folder_path + f'_info_N_{number_of_pixels[0]}_Green_it_{iteration}.npz',
                                   allow_pickle=True)
        info_log_final_G_512 = np.load(data_folder_path + f'_info_N_{512}_Green_it_{iteration}.npz',
                                   allow_pickle=True)
    except Exception:
        info_log_final_G = info_log_final_G
        info_log_final_G_512 = info_log_final_G_512

    nb_iterations_G.append(len(info_log_final_G.f.norm_rr))
    nb_iterations_G_512 .append(len(info_log_final_G_512.f.norm_rr))

    try:
        info_log_final_J = np.load(data_folder_path + f'_info_N_{number_of_pixels[0]}_Jacobi_it_{iteration}.npz',
                                   allow_pickle=True)
        info_log_final_J_512 = np.load(data_folder_path + f'_info_N_{512}_Jacobi_it_{iteration}.npz',
                                   allow_pickle=True)
    except Exception:
        info_log_final_J = info_log_final_J
        info_log_final_J_512 = info_log_final_J_512

    nb_iterations_J.append(len(info_log_final_J.f.norm_rr))
    nb_iterations_J_512.append(len(info_log_final_J_512.f.norm_rr))

    try:
        info_log_final_GJ = np.load(data_folder_path + f'_info_N_{number_of_pixels[0]}_Green_Jacobi_it_{iteration}.npz',
                                    allow_pickle=True)
        info_log_final_GJ_512 = np.load(data_folder_path + f'_info_N_{512}_Green_Jacobi_it_{iteration}.npz',
                                    allow_pickle=True)
    except Exception:
        info_log_final_GJ = info_log_final_GJ
        info_log_final_GJ_512 = info_log_final_GJ_512

    nb_iterations_GJ.append(len(info_log_final_GJ.f.norm_rr))
    nb_iterations_GJ_512 .append(len(info_log_final_GJ_512 .f.norm_rr))

    geometries_data_folder_path = '/home/martin/exp_data/'
    name = 'exp_2D_elasticity_TO_indre_3exp_N1024_Et_0.15_Pt_-0.5_P0_0.0_w5.0_eta0.01_p2_mpi90_nlc_3_e_False'

    phase_1024_at_it = load_npy(os.path.expanduser(geometries_data_folder_path + name + f'_it{iteration}.npy'))
    min_phase.append((phase_1024_at_it**2).min())
    max_phase.append((phase_1024_at_it**2).max())
    contrast.append((phase_1024_at_it**2).max()/(phase_1024_at_it**2).min())

    name = 'exp_2D_elasticity_TO_indre_3exp_N512_Et_0.15_Pt_-0.5_P0_0.0_w5.0_eta0.01_p2_mpi80_nlc_3_e_False'
    try:
        phase_512_at_it = load_npy(os.path.expanduser(geometries_data_folder_path + name + f'_it{iteration}.npy'))
        min_phase_512.append((phase_512_at_it**2).min())
        max_phase_512.append((phase_512_at_it**2).max())
        contrast_512.append((phase_512_at_it**2).max()/(phase_512_at_it**2).min())
    except Exception:
        pass
    grad_info_log_ = np.load(data_folder_path + f'grad_info_N_{number_of_pixels[0]}_{preconditioner_type}_it_{iteration}.npz',
                                    allow_pickle=True)
    grad_norm.append(grad_info_log_['grad_norm'])
    grad_max.append(grad_info_log_['grad_max'])
    grad_max_inf.append(grad_info_log_['grad_max_inf'])

    try:
        grad_info_log_ = np.load(
            data_folder_path + f'grad_info_N_{512}_{preconditioner_type}_it_{iteration}.npz',
            allow_pickle=True)
        grad_norm_512.append(grad_info_log_['grad_norm'])
        grad_max_512.append(grad_info_log_['grad_max'])
        grad_max_inf_512.append(grad_info_log_['grad_max_inf'])
    except Exception:
        pass

grad_norm = np.asarray(grad_norm)
grad_max = np.asarray(grad_max)
grad_max_inf= np.asarray(grad_max_inf)
grad_norm_512 = np.asarray(grad_norm_512)
grad_max_512 = np.asarray(grad_max_512)
grad_max_inf_512= np.asarray(grad_max_inf_512)

min_phase = np.array(min_phase)
max_phase = np.array(max_phase)
contrast = np.array(contrast)
min_phase_512 = np.array(min_phase_512)
max_phase_512 = np.array(max_phase_512)
contrast_512 = np.array(contrast_512)

nb_iterations_G = np.array(nb_iterations_G)
nb_iterations_J = np.array(nb_iterations_J)
nb_iterations_GJ = np.array(nb_iterations_GJ)
nb_iterations_G_512 =   np.array(nb_iterations_G_512)
nb_iterations_J_512 =   np.array(nb_iterations_J_512)
nb_iterations_GJ_512 = np.array(nb_iterations_GJ_512)

# fig = plt.figure(figsize=(11.5, 6))
fig = plt.figure(figsize=(8.3, 6.1))

plt.rcParams.update({'font.size': 13})
plt.rcParams["font.family"] = "Arial"

gs = fig.add_gridspec(4, 6, width_ratios=[3, 3, 3,3,3, 0.2]
                      , height_ratios=[1, 1.7,0.7 ,0.7], hspace=0.13)
ax_iterations = fig.add_subplot(gs[1, :])
ax_iterations.text(-0.12, 1.0, rf'\textbf{{(b)}}', transform=ax_iterations.transAxes)
ax_iterations.plot(np.linspace(1, max_it, nb_iterations_G.shape[0]), nb_iterations_G, "g", label='Green N=1024',
                   linewidth=2)
ax_iterations.plot(np.linspace(1, max_it, nb_iterations_J.shape[0]), nb_iterations_J, "b", label='Jacobi N=1024',
                   linewidth=2)
ax_iterations.plot(np.linspace(1, max_it, nb_iterations_GJ.shape[0]), nb_iterations_GJ, "k", label='Green-Jacobi  N=1024',
                   linewidth=2)

ax_iterations.plot(np.linspace(1, max_it, nb_iterations_G_512.shape[0]), nb_iterations_G_512, "g", label='Green N=512',
                   linewidth=0.5, linestyle='-.')
ax_iterations.plot(np.linspace(1, max_it, nb_iterations_J_512.shape[0]), nb_iterations_J_512, "b", label='Jacobi N=512',
                   linewidth=0.5, linestyle='-.')
ax_iterations.plot(np.linspace(1, max_it, nb_iterations_GJ_512.shape[0]), nb_iterations_GJ_512, "k", label='Green-Jacobi  N=512',
                   linewidth=0.5, linestyle='-.')

ax_iterations.set_xlim(0, max_it)
#ax_iterations.set_xticks([1, max_it])
ax_iterations.set_xticks([])

#ax_iterations.set_xticklabels([f'Start', f'Converged'])
#ax_iterations.set_ylim([1, 2600])
#ax_iterations.set_yscale('linear')
#ax_iterations.set_ylim([1 , 10000])
ax_iterations.set_yscale('log')
ax_iterations.set_ylim([4 , 2e4])
# ax_iterations.set_yticks(ticks=[1,1000,2500,5000,7500,10000])
# ax_iterations.set_yticklabels([1,1000,2500,5000,7500,10000])



#ax_iterations.set_xlabel("L-BFGS optimization process")
ax_iterations.set_ylabel(r"$\#$ PCG iterations")
ax_iterations.annotate(text=r'Green-Jacobi - $\mathcal{T}$' + f'$_{{{512}}}$',
                       xy=(120, nb_iterations_GJ_512[120]),
                       xytext=(108., 5.6),
                       arrowprops=dict(arrowstyle='->',
                                       color='Black',
                                       lw=1,
                                       ls='-'),
                       color='Black'
                       )
ax_iterations.annotate(text=r'\textbf{{Green-Jacobi - $\mathcal{T}$' + f'$_{{{1024}}}$}}',
                       xy=(255, nb_iterations_GJ[255]),
                       xytext=(260., 15.6),
                       arrowprops=dict(arrowstyle='->',
                                       color='Black',
                                       lw=1,
                                       ls='-'),
                       color='Black'
                       )


ax_iterations.annotate(text=r'Jacobi - $\mathcal{T}$' + f'$_{{{512}}}$',
                       xy=(60, nb_iterations_J_512[60]),
                       xytext=(5., 9000.0),
                       arrowprops=dict(arrowstyle='->',
                                       color='Blue',
                                       lw=1,
                                       ls='-'),
                       color='Blue'
                       )
ax_iterations.annotate(text=r'\textbf{{Jacobi - $\mathcal{T}$' + f'$_{{{1024}}}$}}',
                       xy=(390, nb_iterations_J[390]),
                       xytext=(400., 120.0),
                       arrowprops=dict(arrowstyle='->',
                                       color='Blue',
                                       lw=1,
                                       ls='-'),
                       color='Blue'
                       )
ax_iterations.annotate(text=r'Green - $\mathcal{T}$' + f'$_{{{512}}}$',
                       xy=(80, nb_iterations_G_512[80]),
                       xytext=(110., 1000.),
                       arrowprops=dict(arrowstyle='->',
                                       color='Green',
                                       lw=1,
                                       ls='-'),
                       color='Green'
                       )
ax_iterations.annotate(text=r'\textbf{{Green - $\mathcal{T}$' + f'$_{{{1024}}}$}}',
                       xy=(300, nb_iterations_G[300]),
                       xytext=(320., 50.),
                       arrowprops=dict(arrowstyle='->',
                                       color='Green',
                                       lw=1,
                                       ls='-'),
                       color='Green'
                       )




# ax_phase_min.plot(np.linspace(1, max_it, min_phase.shape[0]), min_phase, "g", label='min_phase ',
#                    linewidth=1)
# ax_phase_min.plot(np.linspace(1, max_it, max_phase.shape[0]), max_phase, "b", label='max_phase ',
#                    linewidth=1)
# ax_iterations.plot(nb_iterations_J, "b", label='Jacobi N=64', linewidth=1)
ax_phase_min=fig.add_subplot(gs[3 , :])

ax_phase_min.plot(np.linspace(1, max_it, contrast.shape[0]), contrast, "red", label='contrast   ',
                   linewidth=2)
ax_phase_min.plot(np.linspace(1, max_it, contrast_512.shape[0]), contrast_512, "red", label='contrast   ',
                   linewidth=1, linestyle='-.')

ax_phase_min.annotate(text=r'$\mathcal{T}$' + f'$_{{{1024}}}$',
                       xy=(100, contrast[100]),
                       xytext=(120., 1e7),
                       arrowprops=dict(arrowstyle='->',
                                       color='Red',
                                       lw=1,
                                       ls='-'),
                       color='Red'
                       )
ax_phase_min.annotate(text=r'$\mathcal{T}$' + f'$_{{{512}}}$',
                       xy=(40, contrast_512[40]),
                       xytext=(10.,1e10),
                       arrowprops=dict(arrowstyle='->',
                                       color='Red',
                                       lw=1,
                                       ls='-.'),
                       color='Red'
                       )

ax_phase_min.text(-0.12, 1.0, rf'\textbf{{(d)}}', transform=ax_phase_min.transAxes)


init = 1
a2_cut = 50

middle = 200
middle_2 = 250
end = max_it-20

ax_phase_min.set_xlim(0, max_it)
ax_phase_min.set_xticks([0,  a2_cut,middle, middle_2,max_it ])
#ax_phase_min.set_xticklabels([ ])#[f'Start', init,a2_cut,middle,f'Converged']
#ax_phase_min.set_xlabel(r"$k$-th iteration of L-BFGS optimization process")
ax_phase_min.set_xticklabels([ 0, a2_cut,middle,middle_2,max_it])#[f'Start', init,a2_cut,middle,f'Converged']
ax_phase_min.set_xlabel(r"$k$-th iteration of L-BFGS optimization process")

ax_phase_min.set_yscale('log')
ax_phase_min.set_ylim([1,1e15])
# ax_phase_min.set_ylabel(r"$\min(\rho_k)$")
ax_phase_min.set_ylabel(r'$\chi_{k}$')

ax_phase_min.set_yticks(ticks=[1,1e5,1e10,1e15])
ax_phase_min.set_yticklabels([f'$10^{{{0}}}$',f'$10^{{{5}}}$',f'$10^{{{10}}}$',f'$10^{{{15}}}$'])

ax_phase_min.text(380, 10, f'Material contrast', fontsize=14, color='r',)






ax_grad=fig.add_subplot(gs[2, :])
ax_grad.text(-0.12, 1.0, rf'\textbf{{(c)}}', transform=ax_grad.transAxes)

ax_grad.semilogy(np.linspace(1, max_it, grad_max_inf.shape[0]), grad_max_inf, "purple", label='contrast   ',
                   linewidth=2)
ax_grad.semilogy(np.linspace(1, max_it, grad_max_inf_512.shape[0]), grad_max_inf_512, "purple", ls='-.', label='contrast   ',
                   linewidth=1,)
ax_grad.set_xlim(0, max_it)
ax_grad.set_xticks([])

ax_grad.set_yticks(ticks=[1,1e1,1e2])
ax_grad.set_yticklabels([f'$10^{{{0}}}$',f'$10^{{{1}}}$',f'$10^{{{2}}}$' ])
ax_grad.set_ylim([2,512])
ax_grad.text(380, 3, f'Density gradient', fontsize=14,                 color='purple',
)
ax_grad.set_ylabel(r'$\|\nabla \rho_i \|_{\infty}$')


ax_grad.annotate(text=r'$\mathcal{T}$' + f'$_{{{1024}}}$',
                       xy=(100, grad_max_inf[100]),
                       xytext=(120., 4),
                       arrowprops=dict(arrowstyle='->',
                                       color='purple',
                                       lw=1,
                                       ls='-'),
                       color='purple'
                       )
ax_grad.annotate(text=r'$\mathcal{T}$' + f'$_{{{512}}}$',
                       xy=(40, grad_max_inf_512[40]),
                       xytext=(10.,100),
                       arrowprops=dict(arrowstyle='->',
                                       color='purple',
                                       lw=1,
                                       ls='-.'),
                       color='purple'
                       )










# plot upper geometruies
name = 'exp_2D_elasticity_TO_indre_3exp_N1024_Et_0.15_Pt_-0.5_P0_0.0_w5.0_eta0.01_p2_mpi90_nlc_3_e_False'

nb_tiles = 1

#ax_iterations.axvline(x=init, color='grey', linestyle='--', linewidth=1, label='x=init')
ax_iterations.axvline(x=a2_cut, color='grey', linestyle='--', linewidth=1, label='x=a2_cut')
ax_iterations.axvline(x=middle, color='grey', linestyle='--', linewidth=1, label='x=middle')
ax_iterations.axvline(x=middle_2, color='grey', linestyle='--', linewidth=1, label='x=middle')

#ax_iterations.axvline(x=end, color='grey', linestyle='--', linewidth=1, label='x=end')

#ax_grad.axvline(x=init, color='grey', linestyle='--', linewidth=1, label='x=init')
ax_grad.axvline(x=a2_cut, color='grey', linestyle='--', linewidth=1, label='x=a2_cut')
ax_grad.axvline(x=middle, color='grey', linestyle='--', linewidth=1, label='x=middle')
ax_grad.axvline(x=middle_2, color='grey', linestyle='--', linewidth=1, label='x=middle')

ax_phase_min.axvline(x=a2_cut, color='grey', linestyle='--', linewidth=1, label='x=a2_cut')
ax_phase_min.axvline(x=middle, color='grey', linestyle='--', linewidth=1, label='x=middle')
ax_phase_min.axvline(x=middle_2, color='grey', linestyle='--', linewidth=1, label='x=middle')

#ax_phase_min.axvline(x=end, color='grey', linestyle='--', linewidth=1, label='x=end')

geometries_data_folder_path = '/home/martin/exp_data/'
xopt_init = load_npy(os.path.expanduser(geometries_data_folder_path + name + f'_it{init}.npy') )

divnorm = mpl.colors.Normalize(vmin=1e-15, vmax=1)
cmap_ = mpl.cm.seismic  # mpl.cm.seismic #mpl.cm.Greys
ax_init = fig.add_subplot(gs[0, 0])
ax_init.text(-0.2, 1.2, rf'\textbf{{(a.1)}}', transform=ax_init.transAxes)

minn=(xopt_init ** 2).min()
pcm = ax_init.pcolormesh(np.tile(xopt_init ** 2, (nb_tiles, nb_tiles)),
                         cmap=cmap_, linewidth=0,
                         rasterized=True, norm=divnorm)
ax_init.set_title(  r'$\rho$'+f'$_{{{0}}}$' , wrap=True)
# ax_init.set_ylabel(r'Density $\rho$')
ax_init.set_aspect('equal', 'box')
ax_init.set_xlim([0, 1024])
ax_init.set_xticks([])
ax_init.set_ylim([0, 1024])
ax_init.set_yticks([])





ax_a2_cut = fig.add_subplot(gs[0, 1])
ax_a2_cut.text(-0.2, 1.2, rf'\textbf{{(a.2)}}', transform=ax_a2_cut.transAxes)
xopt_init = load_npy(os.path.expanduser(geometries_data_folder_path + name + f'_it{a2_cut}.npy') )
pcm = ax_a2_cut.pcolormesh(np.tile(xopt_init ** 2, (nb_tiles, nb_tiles)),
                         cmap=cmap_, linewidth=0,
                         rasterized=True, norm=divnorm)
ax_a2_cut.set_title(  r'$\rho$'+f'$_{{{a2_cut}}}$' , wrap=True)
# ax_init.set_ylabel(r'Density $\rho$')
ax_a2_cut.set_aspect('equal', 'box')
ax_a2_cut.set_xlim([0, 1024])
ax_a2_cut.set_xticks([])
ax_a2_cut.set_ylim([0, 1024])
ax_a2_cut.set_yticks([])

ax_middle = fig.add_subplot(gs[0, 2])
ax_middle.text(-0.2, 1.2, rf'\textbf{{(a.3)}}', transform=ax_middle.transAxes)

# ax_middle = fig.add_axes([0.5, 0.6, 0.1, 0.2])
xopt_middle = load_npy(os.path.expanduser(geometries_data_folder_path + name + f'_it{middle}.npy') )

minn=(xopt_middle ** 2).min()

ax_middle.pcolormesh(np.tile(xopt_middle ** 2, (nb_tiles, nb_tiles)),
                     cmap=cmap_, norm=divnorm, linewidth=0,
                     rasterized=True)
ax_middle.set_title(   r'$\rho$'+f'$_{{{middle}}}$' , wrap=True)
ax_middle.set_aspect('equal', 'box')
ax_middle.set_xlim([0, 1024])
ax_middle.set_xticks([])
ax_middle.set_ylim([0, 1024])
ax_middle.set_yticks([])

ax_end = fig.add_subplot(gs[0, 3])
ax_end.text(-0.2, 1.2, rf'\textbf{{(a.4)}}', transform=ax_end.transAxes)

# ax_end = fig.add_axes([0.7, 0.3, 0.1, 0.2])
xopt_middle_2= load_npy(os.path.expanduser(geometries_data_folder_path + name + f'_it{middle_2}.npy') )

# xopt_init128t_end = np.load('../exp_data/' + name2_128 + f'_it{end}.npy', allow_pickle=True)

pcm = ax_end.pcolormesh(np.tile(xopt_middle_2 ** 2, (nb_tiles, nb_tiles)),
                        cmap=cmap_, norm=divnorm, linewidth=0,
                        rasterized=True)
minn=(xopt_middle_2 ** 2).min()
ax_end.set_title(   r'$\rho$'+f'$_{{{middle_2}}}$'  , wrap=True)
ax_end.set_aspect('equal', 'box')
ax_end.set_xlim([0, 1024])
ax_end.set_xticks([])
ax_end.set_ylim([0, 1024])
ax_end.set_yticks([])

ax_end = fig.add_subplot(gs[0, 4])
ax_end.text(-0.2, 1.2, rf'\textbf{{(a.5)}}', transform=ax_end.transAxes)

# ax_end = fig.add_axes([0.7, 0.3, 0.1, 0.2])
xopt_end = load_npy(os.path.expanduser(geometries_data_folder_path + name + f'_it{8000}.npy') )

# xopt_init128t_end = np.load('../exp_data/' + name2_128 + f'_it{end}.npy', allow_pickle=True)

pcm = ax_end.pcolormesh(np.tile(xopt_end ** 2, (nb_tiles, nb_tiles)),
                        cmap=cmap_, norm=divnorm, linewidth=0,
                        rasterized=True)
minn=(xopt_end ** 2).min()
ax_end.set_title(  r'$\rho^{{\rm opt}}_{\rm{converged}}$'  , wrap=True)
ax_end.set_aspect('equal', 'box')
ax_end.set_xlim([0, 1024])
ax_end.set_xticks([])
ax_end.set_ylim([0, 1024])
ax_end.set_yticks([])
cbar_ax = fig.add_subplot(gs[0, 5])
cbar = plt.colorbar(pcm, location='left', cax=cbar_ax)
cbar.ax.yaxis.tick_right()
# cbar.set_ticks(ticks=[1e-4,1e-2, 1])
# cbar.set_ticklabels([f'$10^{{{-4}}}$', f'$10^{{{-2}}}$', 1])
cbar.set_ticks(ticks=[1e-15, 0.5, 1])
cbar.set_ticklabels([f'$10^{{{-15}}}$', 0.5, 1])

fname = figure_folder_path + 'exp_paper_JG_2D_elasticity_TO_1024' + '{}'.format('.pdf')
print(('create figure: {}'.format(fname)))
plt.savefig(fname, bbox_inches='tight')
plt.show()
