import numpy as np
import matplotlib as mpl
import os

import matplotlib.pyplot as plt

from mpi4py import MPI
from NuMPI.IO import save_npy, load_npy

from muFFTTO import domain
from muFFTTO import solvers
import sys
import argparse

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

script_name = os.path.splitext(os.path.basename(__file__))[0]
file_folder_path = os.path.dirname(os.path.realpath(__file__))
if rank == 0:
    os.makedirs(file_folder_path, exist_ok=True)
data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
if rank == 0:
    os.makedirs(data_folder_path, exist_ok=True)
figure_folder_path = file_folder_path + '/figures/' + script_name + '/'
if rank == 0:
    os.makedirs(figure_folder_path, exist_ok=True)
comm.Barrier()

parser = argparse.ArgumentParser(
    prog=script_name,
)
parser.add_argument("-n", "--nb_pixel", default=32)
parser.add_argument("-nb_filter", "--nb_filter", default=1)
parser.add_argument(
    "-p", "--preconditioner_type",
    type=str,
    choices=["Green", "Jacobi", "Green_Jacobi"],  # example options
    default="Green",
    help="Type of preconditioner to use"
)
parser.add_argument(
    "-g", "--geometry",
    type=str,
    choices=["circle_inclusion", "cos_wave"],  # example options
    default="circle_inclusion",
    help="Type of geometry to use"
)
args = parser.parse_args()

problem_type = 'conductivity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'

domain_size = [1, 1]
number_of_pixels = (int(args.nb_pixel), int(args.nb_pixel))
print(f'number_of_pixels = {number_of_pixels}')
preconditioner_type = args.preconditioner_type
geometry_ID = args.geometry
nb_of_filters = args.nb_filter

tol_cg = 1e-6
contrast = 1e-4

my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)
# set macroscopic gradient
macro_gradient = np.array([1.0, 0])
# create material data field

conductivity_C_0 = np.array([[1., 0], [0, 1.0]])
refmaterial_data_field_ = np.copy(conductivity_C_0)  # [:, :, np.newaxis, np.newaxis, np.newaxis]

print('Data = \n {}'.format(conductivity_C_0))

_info = {}
_info['homogenized_A_ij'] = []
# for filter_index, filter in enumerate(np.arange(nb_of_filters)):
#     # material distribution
#     # geometry_ID ='circle_inclusion'# 'linear'#,
#
#     # initial_phase_field = discretization.get_scalar_field(name='initial_phase_field')
#     # initial_phase_field.s[...] = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
#     #                                                          microstructure_name=geometry_ID,
#     #                                                          coordinates=discretization.fft.coords)

results_name = (f'phase_field_N{number_of_pixels[0]}_F{nb_of_filters}_kappa{contrast}_ge_{geometry_ID}')

geom_folder_path = data_folder_path
# geom_folder_path = '//work/classic/fr_ml1145-martin_workspace_01/muFFTTO/experiments/paper_Jacobi_Green/exp_data/exp_paper_JG_intro_circle/'

phase_field = discretization.get_scalar_field(name='phase_field')
phase_field.s[0, 0] = load_npy(geom_folder_path + results_name + f'.npy',
                               tuple(discretization.subdomain_locations_no_buffers),
                               tuple(discretization.nb_of_pixels), MPI.COMM_WORLD)
# populate phase field for current step of smoothnes

# fig = plt.figure()
# gs = fig.add_gridspec(2, 2)
# ax1 = fig.add_subplot(gs[0, 0])
# ax1.imshow(phase_field.s[0, 0], cmap='gray')
# ax1.pcolormesh(np.transpose(phase_field.s[0, 0]), cmap=mpl.cm.Greys, vmin=contrast, vmax=1,
#                linewidth=0,
#                rasterized=True)
# ax1.set_aspect('equal')
# ax_cross = fig.add_subplot(gs[0, 1])
# ax_cross.semilogy(phase_field.s[0, 0, :, discretization.nb_of_pixels[0] // 2], linewidth=1,
#                   # color=colors[-geom_ax],
#                   # linestyle=linestyles[geom_ax]
#                   )
# # ax_cross.set_ylabel("Y2-axis (Cos)", color='red')
# ax_cross.tick_params(axis='y', labelcolor='black')
# ax_cross.set_xticks([])
# ax_cross.set_xticklabels([])
# ax_cross.set_title(f'filter_index={nb_of_filters} contrast={contrast}', wrap=True)

material_data_field_C_0 = discretization.get_material_data_size_field_mugrid(name='conductivity_tensor')
material_data_field_C_0.s[...] = conductivity_C_0[..., np.newaxis, np.newaxis, np.newaxis] * phase_field.s[
    np.newaxis, ...]

macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
rhs_field = discretization.get_unknown_size_field(name='rhs_field')
# Set up right hand side
macro_gradient_field.sg.fill(0)
discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient,
                                               macro_gradient_field_ijqxyz=macro_gradient_field)
# macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)
# perturb=np.random.random(macro_gradient_field.shape)
# macro_gradient_field += perturb#-np.mean(perturb)

discretization.fft.communicate_ghosts(field=macro_gradient_field)

# Solve equilibrium
rhs_field.sg.fill(0)
discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0,
                              macro_gradient_field_ijqxyz=macro_gradient_field,
                              rhs_inxyz=rhs_field)


def K_fun(x, Ax):
    """
    Function to compute the product of the Hessian matrix with a vector.
    The Hessian is represented by the convolution operator.
    """

    discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0,
                                              input_field_inxyz=x,
                                              output_field_inxyz=Ax)
    discretization.fft.communicate_ghosts(Ax)


preconditioner = discretization.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=refmaterial_data_field_)


def M_fun(x, Px):
    """
    Function to compute the product of the Preconditioner matrix with a vector.
    The Preconditioner is represented by the convolution operator.
    """
    discretization.fft.communicate_ghosts(x)
    discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                               input_nodal_field_fnxyz=x,
                                               output_nodal_field_fnxyz=Px)


K_diag_alg = discretization.get_preconditioner_Jacobi_mugrid(
    material_data_field_ijklqxyz=material_data_field_C_0,
    formulation=None)


def M_fun_Green_Jacobi(x, Px):
    # discretization.fft.communicate_ghosts(x)
    x_jacobi_temp = discretization.get_unknown_size_field(name='x_jacobi_temp')

    x_jacobi_temp.s[...] = K_diag_alg.s * x.s
    discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                               input_nodal_field_fnxyz=x_jacobi_temp,
                                               output_nodal_field_fnxyz=Px)

    Px.s[...] = K_diag_alg.s * Px.s


def M_fun_Jacobi(x, Px):
    Px.s[...] = K_diag_alg.s * K_diag_alg.s * x.s


_info['norms_G_rr'] = []
_info['norms_G_rz'] = []
_info['norms_G_rGr'] = []


def callback(it, x, r, p, z, stop_crit_norm):
    # global norms_cg_mech
    norm_of_rr = discretization.communicator.sum(np.dot(r.ravel(), r.ravel()))
    norm_of_rz = discretization.communicator.sum(np.dot(r.ravel(), z.ravel()))
    _info['norms_G_rr'].append(norm_of_rr)
    _info['norms_G_rz'].append(norm_of_rz)


solution_temperatute_field = discretization.get_unknown_size_field(name=f'solution_temperatute_field')
solution_temperatute_field.s.fill(0)
# displacement_field, norms = solvers.PCG(K_fun, rhs_field, x0=None, P=M_fun, steps=int(1000), toler=1e-6)
x, norms = solvers.conjugate_gradients_mugrid_experimental(
    comm=discretization.communicator,
    fc=discretization.field_collection,
    hessp=K_fun,  # linear operator
    b=rhs_field,
    x=solution_temperatute_field,
    P=M_fun,
    tol=tol_cg,
    maxiter=10000,
    callback=callback,
    rtol=True,
    # norm_metric=res_norm
)
_info['homogenized_A_ij'].append(discretization.get_homogenized_stress_mugrid(
    material_data_field_ijklqxyz=material_data_field_C_0,
    displacement_field_inxyz=solution_temperatute_field,
    macro_gradient_field_ijqxyz=macro_gradient_field)[0, 0])
# ax_norms = fig.add_subplot(gs[1, 1])
# ax_norms.semilogy(np.array(_info['norms_G_rr']), label='norm of residual', color='k')
# ax_norms.semilogy(np.array(norms['energy_lower_bound']), label='norm of energy_lower_bound', color='r')
#
# ax_norms.set_ylim([1e-8, 1])
# ax_norms.set_xlim([0, 50])
#
# if rank == 0:
#     plt.tight_layout()
#     plt.savefig(figure_folder_path + results_name + '_convergence.png', dpi=150)
#     plt.close()
# nb_it[filter_index - 1 ] = (len(norms_cg['residual_rz']))
# norm_rz.append(norms_cg['residual_rz'])
# norm_rr.append(norms_cg['residual_rr'])
_info['nb_of_pixels'] = discretization.nb_of_pixels_global
# _info['homogenized_A_ij'] = homogenized_A_ij

_info['nb_of_filters_aplication'] = nb_of_filters
print(f'nb_of_filters {nb_of_filters} iters = Green ' + '{}'.format(len(_info['norms_G_rr'])))
print(f"homogenized_A_ij = {{ {_info['homogenized_A_ij']} }}")

# print(f'Jacobi_' + '{}'.format(len(_info['norms_J_rr'])))
# print(f'Green_Jacobi_ iters =' + '{}'.format(len(_info['norms_GJ_rr'])))

results_name = (f'N{number_of_pixels[0]}_F{nb_of_filters}_kappa{contrast}_ge_{geometry_ID}')

if rank == 0:
    np.savez(data_folder_path + results_name + f'_log.npz', **_info)
    print(data_folder_path + results_name + f'_log.npz')
    # print(nb_it)
#########
# displacement_field_combi, norms_combi = solvers.PCG(K_fun, rhs, x0=None, P=M_fun_combi, steps=int(1000),
#                                                     toler=1e-6)
# nb_it_combi[kk - 1, i] = (len(norms_combi['residual_rz']))
# norm_rz_combi.append(norms_combi['residual_rz'])
# norm_rr_combi.append(norms_combi['residual_rr'])
#
# displacement_field_Jacobi, norms_Jacobi = solvers.PCG(K_fun, rhs, x0=None, P=M_fun_Jacobi, steps=int(1000),
#                                                       toler=1e-6)
# nb_it_Jacobi[kk - 1, i] = (len(norms_Jacobi['residual_rz']))
# norm_rz_Jacobi.append(norms_Jacobi['residual_rz'])
# norm_rr_Jacobi.append(norms_Jacobi['residual_rr'])
# displacement_field_Richardson, norms_Richardson = solvers.Richardson(K_fun, rhs, x0=None, P=M_fun,
#                                                               omega=omega,
#                                                               steps=int(1000),
#                                                               toler=1e-6)
# nb_it_Richardson[kk - 1, i] = (len(norms_Richardson['residual_rr']))
# norm_rr_Richardson= norms_Richardson['residual_rr'][-1]
#
# displacement_field_Richardson_combi, norms_Richardson_combi = solvers.Richardson(K_fun, rhs, x0=None, P=M_fun_combi,
#                                                                      omega=omega*0.4,
#                                                                      steps=int(1000),
#                                                                      toler=1e-6)
# nb_it_Richardson_combi[kk - 1, i] = (len(norms_Richardson_combi['residual_rr']))
# norm_rr_Richardson_combi = norms_Richardson_combi['residual_rr'][-1]
# kujacobi=K_fun(displacement_field_combi)-rhs
# plt.figure()
# plt.imshow(kujacobi[0,0])
# plt.title('rez Jacobi Green')
# plt.colorbar()
# plt.show()
#
# kugreen= K_fun(displacement_field) - rhs
# plt.figure()
# plt.imshow(kugreen[0, 0])
# plt.title('rez greens')
#
# plt.colorbar()
# plt.show()
# plt.figure()
# plt.imshow((displacement_field_combi-displacement_field)[0,0])
# plt.colorbar()
# plt.show()
#
# plt.figure()
# plt.imshow(displacement_field_combi[0, 0])
# plt.show()
# print(f'norm = {np.linalg.norm(displacement_field_combi[0, 0] - displacement_field[0, 0])}')
##################
#    print(ratio)
# plt.show()
# for i in np.arange(ratios.size,step=3):
#     kappa_Green=kontrast[i]
#     k=np.arange(max(map(len, norm_rr)))
#     print(f'k \n {k}')
#     lb=eigen_LB[i]
#     print(f'lb \n {lb}')
#     print(f'kappa_Green \n {kappa_Green}')
#
#     convergence_Green=((np.sqrt(kappa_Green)-1)/(np.sqrt(kappa_Green)+1))**k
#     convergence_Green=convergence_Green*norm_rr[i][0]
#
#     kappa_Green_Jacobi=kontrast_2[i]
#     convergence_Green_Jacobi = ((np.sqrt(kappa_Green_Jacobi) - 1) / (np.sqrt(kappa_Green_Jacobi) + 1)) ** k
#     convergence_Green_Jacobi = convergence_Green_Jacobi * norm_rr_combi[i][0]
#     print(f'kappa_Green_Jacobi \n {kappa_Green_Jacobi}')
#
#
#
#
#         #print(f'convergecnce \n {convergence_Green}')
#         fig = plt.figure()
#         gs = fig.add_gridspec(1, 1)
#         ax_1 = fig.add_subplot(gs[0, 0])
#         ax_1.set_title(f'{i}', wrap=True)
#         ax_1.semilogy(convergence_Green, '--',label='estim green', color='r')
#         ax_1.semilogy(convergence_Green_Jacobi, '--',label='estim Green Jacobi', color='b')
#
#         ax_1.semilogy(norm_rr[i], label='PCG: Green', color='r')
#         #ax_1.semilogy(norm_rr_Jacobi[i], label='PCG: Jacobi', color='b')
#         ax_1.semilogy(norm_rr_combi[i], label='PCG: Green Jacobi', color='b')
#
#         #x_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
#         #ax_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
#         ax_1.set_xlabel('CG iterations')
#         ax_1.set_ylabel('Norm of residua')
#         plt.legend([r'$\kappa$ upper bound Green',r'$\kappa$ upper bound Green + Jacobi','Green', 'Green + Jacobi','Richardson'])
#         ax_1.set_ylim([1e-7, norm_rr_combi[i][0]])#norm_rz[i][0]]/lb)
#         print(max(map(len, norm_rr)))
#         ax_1.set_xlim([0, max(map(len, norm_rr))])
#
#         plt.show()
#
#     plt.show()
#     fig = plt.figure()
#     gs = fig.add_gridspec(1, 1)
#     ax_1 = fig.add_subplot(gs[0, 0])
#     ax_1.semilogy(norm_rr[0], label='PCG: Green', color='blue', linewidth=0)
#
#     #ax_1.set_ylim([1e-7, 1e0])
#     ax_1.set_ylim([1e-7, norm_rr[0][0]])  # norm_rz[i][0]]/lb)
#
#     print(max(map(len, norm_rz)))
#     ax_1.set_xlim([0, max(map(len, norm_rr))])
#
#     def convergence_gif_rz(i):
#         kappa=kontrast[i]
#         k=np.arange(max(map(len, norm_rr)))
#         print(f'k \n {k}')
#         lb=eigen_LB[i]
#         print(f'lb \n {lb}')
#
#         convergence=((np.sqrt(kappa)-1)/(np.sqrt(kappa)+1))**k
#         convergence=convergence*norm_rr[i][0]
#
#         kappa_Green_Jacobi=kontrast_2[i]
#         convergence_Green_Jacobi = ((np.sqrt(kappa_Green_Jacobi) - 1) / (np.sqrt(kappa_Green_Jacobi) + 1)) ** k
#         convergence_Green_Jacobi = convergence_Green_Jacobi * norm_rr_combi[i][0]
#
#         ax_1.clear()
#
#         ax_1.set_title(f'{i}', wrap=True)
#         ax_1.semilogy(convergence, '--',label='estim', color='r')
#         ax_1.semilogy(convergence_Green_Jacobi, '--',label='estim Green Jacobi', color='b')
#
#         ax_1.semilogy(norm_rr[i], label='PCG: Green', color='r')
#         ax_1.semilogy(norm_rr_combi[i], label='PCG: Jacobi', color='b')
#
#         # x_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
#         # ax_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
#         ax_1.set_xlabel('CG iterations')
#         ax_1.set_ylabel('Norm of residua')
#         plt.legend([r'$\kappa$ upper bound Green',r'$\kappa$ upper bound Green + Jacobi','Green', 'Green + Jacobi','Richardson'])
#         ax_1.set_ylim([1e-7, norm_rr[i][0]])#norm_rz[i][0]]/lb)
#         print(max(map(len, norm_rr)))
#         ax_1.set_xlim([0, max(map(len, norm_rr))])
#         # axs[1].legend()
#         # plt.legend(['', 'FEM: Green', 'FEM: Jacobi', 'FEM: Green + Jacobi','FEM: Richardson'])
#         plt.legend([r'$\kappa$ upper bound Green',r'$\kappa$ upper bound Green + Jacobi','Green', 'Green + Jacobi','Richardson'])
#
#         plt.legend([r'$\kappa$ upper bound Green',r'$\kappa$ upper bound Green + Jacobi','Green', 'Green + Jacobi', 'Richardson Green + Jacobi'],
#                    loc='center left', bbox_to_anchor=(0.5, 0.5))
#
#     ani = FuncAnimation(fig, convergence_gif_rz, frames=ratios.size, blit=False)
#     # axs[1].legend()middlemiddle
#     # Save as a GIF
#     ani.save(f"./figures/convergence_estimatess2tgif_{number_of_pixels[0]}comparison{ratios[-1]}_RichardsonJacobi{geometry_ID}_circle_inc_to_smooth_semiloplots3.gif",
#              writer=PillowWriter(fps=1))
#
#     plt.show()
#     #-------------------------------------------------------------------------------------------------------
#     # for i in np.arange(ratios.size,step=1):
#     #     kappa=kontrast[i]
#     #     kappa_2=kontrast_2[i]
#     #     k=np.arange(len(norm_rr_Jacobi[i]))
#     #     print(f'k \n {k}')
#     #
#     #     convergence=((np.sqrt(kappa)-1)/(np.sqrt(kappa)+1))**k
#     #     convergence=convergence*norm_rr[i][0]
#     #     convergence2 = ((np.sqrt(kappa_2) - 1) / (np.sqrt(kappa_2) + 1)) ** k
#     #     convergence2 = convergence2 * norm_rr[i][0]
#     #
#     #
#     #     print(f'convergecnce \n {convergence}')
#     #     fig = plt.figure()
#     #     gs = fig.add_gridspec(1, 1)
#     #     ax_1 = fig.add_subplot(gs[0, 0])
#     #     ax_1.set_title(f'{i}', wrap=True)
#     #     ax_1.semilogy(convergence, '-',label='estim', color='green')
#     #     #ax_1.semilogy(convergence2,'.-', label='estim2', color='green')
#     #
#     #     ax_1.semilogy(norm_rr[i], label='PCG: Green', color='blue')
#     #     ax_1.semilogy(norm_rr_Jacobi[i], label='PCG: Jacobi', color='black')
#     #     #ax_1.semilogy(norm_rr_combi[i], label='PCG: Green + Jacobi', color='red')
#     #     #x_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
#     #     #ax_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
#     #     ax_1.set_xlabel('CG iterations')
#     #     ax_1.set_ylabel('Norm of residuals')
#     #     plt.legend([r'$\kappa$ upper bound','Green', 'Jacobi', 'Green + Jacobi','Richardson'])
#     #     ax_1.set_ylim([1e-7, norm_rr[i][0]])
#     #     print(max(map(len, norm_rr)))
#     #     ax_1.set_xlim([0, max(map(len, norm_rr))])
#     #
#     #     plt.show()
#
#     fig = plt.figure()
#     gs = fig.add_gridspec(1, 1)
#     ax_1 = fig.add_subplot(gs[0, 0])
#     ax_1.semilogy(norm_rr[0], label='PCG: Green', color='blue', linewidth=0)
#
#     ax_1.set_ylim([1e-7, 1e0])
#     print(max(map(len, norm_rr)))
#     ax_1.set_xlim([0, max(map(len, norm_rr))])
#
#     def convergence_gif(i):
#         kappa = kontrast[i]
#         k = np.arange( max(map(len, norm_rr)))
#         print(f'k \n {k}')
#
#         convergence = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** k
#         convergence = convergence * norm_rr[i][0]
#         print(f'convergecnce \n {convergence}')
#         ax_1.clear()
#
#         ax_1.set_title(f'{i}', wrap=True)
#         ax_1.semilogy(convergence, '--',label='estim', color='k')
#
#         ax_1.semilogy(norm_rr[i], label='PCG: Green', color='r')
#         ax_1.semilogy(norm_rr_Jacobi[i], label='PCG: Jacobi', color='b')
#         #ax_1.semilogy(norm_rr_combi[i], label='PCG: Green + Jacobi', color='red')
#         # x_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
#         # ax_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
#         ax_1.set_xlabel('CG iterations')
#         ax_1.set_ylabel('Norm of residuals')
#         plt.legend([r'$\kappa$ upper bound','Green', 'Jacobi', 'Green + Jacobi', 'Richardson'])
#         ax_1.set_ylim([1e-7, 1e0])
#         print(max(map(len, norm_rr)))
#         ax_1.set_xlim([0, max(map(len, norm_rr))])
#         # axs[1].legend()
#         # plt.legend(['', 'FEM: Green', 'FEM: Jacobi', 'FEM: Green + Jacobi','FEM: Richardson'])
#         plt.legend([r'$\kappa$ upper bound','Green', 'Jacobi', 'Green + Jacobi', 'Richardson Green', 'Richardson Green + Jacobi'],
#                    loc='center left', bbox_to_anchor=(0.8, 0.5))
#
#
#     ani = FuncAnimation(fig, convergence_gif, frames=ratios.size, blit=False)
#     # axs[1].legend()middlemiddle
#     # Save as a GIF
#     ani.save(f"./figures/convergence_gif_{number_of_pixels[0]}comparison{ratios[-1]}_RichardsonJacobi{geometry_ID}_circle_inc_to_smooth_semiloplots3.gif",
#              writer=PillowWriter(fps=1))
#
#     plt.show()
#
#
#
#     plot_evolion = True
#     if plot_evolion:
#         for nb_tiles in [1, ]:
#             # fig = plt.figure()
#
#             #
#             # fig, axs = plt.subplots(nrows=2, ncols=2,
#             #                         figsize=(6, 6)  )
#             fig = plt.figure()
#             gs = fig.add_gridspec(2, 2)
#             ax1 = fig.add_subplot(gs[0, 0])
#             ax3 = fig.add_subplot(gs[0, 1])
#             ax2 = fig.add_subplot(gs[1, :])
#             # axs[0] = plt.axes(xlim=(0, nb_tiles * N), ylim=(0, nb_tiles * N))
#             ax1.imshow(phase_field, cmap=mpl.cm.Greys, vmin=1e-4, vmax=1)
#             ax3.semilogy(phase_field[:, phase_field.shape[0] // 2], linewidth=0)
#             # ax3.plot(phase_field[:,phase_field.shape[0]//2], linewidth=0)
#             ax3.set_ylim([1e-4, 1])
#             print(ratios)
#
#             print(nb_it)
#             ax2.plot(ratios, nb_it_Jacobi[0], label='nb_it_Laplace', linewidth=0)
#             ax3.set_ylim([1e0, 1e3 ])
#
#             # axs[1].plot(xopt.f.num_iteration_.transpose()[::3], 'w'  , linewidth=0)
#             # axs[1].plot(xopt3.f.num_iteration_.transpose(), "b", label='Jacoby', linewidth=0)
#             # axs[1].plot(xopt.f.num_iteration_.transpose(), "k", label='DGO + Jacoby', linewidth=0)
#             # legend = plt.legend()
#             # Animation function to update the image
#             # ax2.set_xlabel('')
#             ax2.set_ylabel('# PCG iterations')
#
#
#             def update(i):
#                 ratio = ratios[i]
#                 # phase_field = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
#                 #                                                   microstructure_name='circle_inclusion',
#                 #                                                   coordinates=discretization.fft.coords)
#                 phase_field = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
#                                                                           microstructure_name=geometry_ID,
#                                                                           coordinates=discretization.fft.coords)
#
#                 phase_field[phase_field < 0.5] = 0
#                 phase_field[phase_field >= 0.5] = 1
#
#                 #phase_field=np.abs(phase_field-1)
#                 phase_field += 1e-4
#                 for a in np.arange(i):
#                     phase_field = apply_smoother_log10(phase_field)
#                 # min_val = np.min(phase_field)
#                 # max_val = np.max(phase_field)
#                 # phase_field = 1e-4 + (phase_field - min_val) * (1 - 1e-4) / (max_val - min_val)
#                 # phase_field = ratio * phase_field_smooth + (1 - ratio) * phase_field_pwconst
#
#                 ax1.clear()
#                 ax1.imshow(phase_field, cmap=mpl.cm.Greys, vmin=1e-4, vmax=1)
#                 ax1.set_title(r'Density $\rho$', wrap=True)
#                 #: {np.max(phase_field)/np.min(phase_field):.1e}  \n'                          f'  min = {np.min(phase_field):.1e}
#                 ax3.clear()
#                 ax3.semilogy(phase_field[:, phase_field.shape[0] // 2], linewidth=1)
#                 # ax3.plot(phase_field[:, phase_field.shape[0] // 2], linewidth=1)
#                 ax3.set_ylim([5e-5, 2])
#                 ax3.set_title(f'Cross section')
#
#                 ax2.plot(ratios[0:i + 1], nb_it[0, 0:i + 1], 'r', label='PCG  Green', linewidth=1)
#                 # axs[1].plot(xopt2.f.num_iteration_.transpose()[1:3*i+1:3],"r", label='DGO ',linewidth=1)
#                 # axs[1].plot(xopt2.f.num_iteration_.transpose()[2:3*i+2:3],"r", label='DGO ',linewidth=1)
#
#                 ax2.plot(ratios[0:i + 1], nb_it_Jacobi[0, 0:i + 1], "b", label='PCG Jacobi', linewidth=1)
#                 ax2.plot(ratios[0:i + 1], nb_it_combi[0, 0:i + 1], "k", label='PCG Green + Jacobi', linewidth=1)
#               #  ax2.semilogy(ratios[0:i + 1], nb_it_Richardson[0, 0:i + 1], "g", label=' Richardson Green ', linewidth=1)
#               #  ax2.semilogy(ratios[0:i + 1], nb_it_Richardson_combi[0, 0:i + 1], "y",  label=' Richardson Green + Jacobi ', linewidth=1)
#
#                 # axs[1].legend()
#                 ax2.legend([ '','Green', 'Jacobi' ,'Green + Jacobi'])
#                 #plt.legend(['', 'FEM: Green', 'FEM: Jacobi', 'FEM: Green + Jacobi','FEM: Richardson'])
#                # plt.legend(['', ' Green', 'Jacobi', 'Green + Jacobi','Richardson Green','Richardson Green + Jacobi'],loc='best', bbox_to_anchor=(0.7, 0.5))
#     #        ax2.legend([r'$\kappa$ upper bound', 'Green', 'Jacobi', 'Green + Jacobi', 'Richardson'])
#
#                 # plt.legend([ '', 'Green', 'Jacobi'  ])
#
#                 # img.set_array(xopt_it)
#             #ax2.legend([r'$\kappa$ upper bound', 'Green', 'Jacobi', 'Green + Jacobi', 'Richardson'])
#
#             # box = ax2.get_position()
#             # ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#             #
#             # # Put a legend to the right of the current axis
#             # ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#             # Create animation
#             # ani = FuncAnimation(fig, update, frames=xopt.f.norms_f.size - 1, blit=False)
#
#             ani = FuncAnimation(fig, update, frames=ratios.size, blit=False)
#             # axs[1].legend()middlemiddle
#             # Save as a GIF
#             ani.save(f"./figures/movie2222_{number_of_pixels[0]}comparison{ratios[-1]}_RichardsonJacobi{geometry_ID}_circle_inc_to_smooth_semiloplots3.gif",
#                      writer=PillowWriter(fps=4))
#
#         plt.show()
#
#
#         # print(norms)
# # box = ax2.get_position()
# # ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# #
# # # Put a legend to the right of the current axis
# # ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# # plt.show()
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # Plot each line with a different z offset
# for i in np.arange(len(nb_pix_multips)):
#     ax.plot(ratios, nb_pix_multips[i] , zs=nb_it[i], label='PCG: Green', color='green')
#     ax.plot(ratios, nb_pix_multips[i] , zs=nb_it_Jacobi[i], label='PCG: Jacobi', color='black')
#     ax.plot(ratios, nb_pix_multips[i] , zs=nb_it_combi[i], label='PCG: Green + Jacobi', color='red')
#     ax.plot(ratios, nb_pix_multips[i] , zs=nb_it_Richardson[i], linestyle='--',label='Richardson Green', color='green',)
#     ax.plot(ratios, nb_pix_multips[i] , zs=nb_it_Richardson_combi[i], linestyle='--',label='Richardson Green+Jacobi', color='red')
# ax.set_xlabel('nb of filter aplications')
# ax.set_ylabel('size')
# ax.set_zlabel('# CG iterations')
# plt.legend(['DGO', 'Jacobi', 'DGO + Jacobi','Richardson'])
# plt.show()
# #quit()
#
# fig = plt.figure()
# gs = fig.add_gridspec(1, 1)
# ax = fig.add_subplot(gs[0, 0])
# # Plot each line with a different z offset
# for i in np.arange(len(nb_pix_multips)):
#     ax.plot(ratios, nb_it[i], label='PCG: Green', color='green')
#     ax.plot(ratios, nb_it_combi[i], label='PCG: Green + Jacobi', color='red')
# ax.set_xlabel('nb of filter applications')
# ax.set_ylabel('# CG iterations')
# plt.legend(['Green', 'Jacobi + Green'])
# plt.show()
