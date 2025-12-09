import numpy as np
import time
import matplotlib as mpl
from mpi4py import MPI
from NuMPI.IO import save_npy, load_npy
import os

script_name = os.path.splitext(os.path.basename(__file__))[0]

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import sys
import argparse

parser = argparse.ArgumentParser(
    prog=script_name,
)
parser.add_argument("-n", "--nb_pixel", default="32")
parser.add_argument("-start", "--start_ratio", default=1)

parser.add_argument("-stop", "--stop_ratio", default=50)

args = parser.parse_args()

sys.path.append("/home/martin/Programming/muFFTTO_paralellFFT_test/muFFTTO")
sys.path.append('../..')

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library

script_name = 'exp_paper_JG_intro_circle'
folder_name = '../exp_data/'

file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
if not os.path.exists(file_folder_path):
    os.makedirs(file_folder_path)
data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
if not os.path.exists(data_folder_path):
    os.makedirs(data_folder_path)
figure_folder_path = file_folder_path + '/figures/' + script_name + '/'
if not os.path.exists(figure_folder_path):
    os.makedirs(figure_folder_path)

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'

domain_size = [1, 1]
nb_pix = int(args.nb_pixel)
# ,2,3,3,2,
number_of_pixels = (nb_pix, nb_pix)
tol_cg = 1e-3
contrast = 1e-4




my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)
start_time = time.time()

# set macroscopic gradient
macro_gradient = np.array([[1.0, 0.5], [0.5, 1.0]])

# create material data field
K_0, G_0 = 1, 0.5  # domain.get_bulk_and_shear_modulus(E=1, poison=0.2)

# identity tensor                                               [single tensor]
ii = np.eye(2)

shape = tuple((number_of_pixels[0] for _ in range(2)))

# identity tensors                                            [grid of tensors]
I = ii
I4 = np.einsum('il,jk', ii, ii)
I4rt = np.einsum('ik,jl', ii, ii)
I4s = (I4 + I4rt) / 2.

elastic_C_1 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                 K=K_0,
                                                 mu=G_0,
                                                 kind='linear')
C_1 = domain.compute_Voigt_notation_4order(elastic_C_1)

material_data_field_C_0 = discretization.get_material_data_size_field_mugrid(name='material_data_field_C_0')

material_data_field_C_0.s = np.einsum('ijkl,qxy->ijklqxy', elastic_C_1,
                                      np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                        *discretization.nb_of_pixels])))

print('elastic tangent = \n {}'.format(domain.compute_Voigt_notation_4order(elastic_C_1)))

# material distribution
# geometry_ID = 'circle_inclusion'  # 'square_inclusion'#'circle_inclusion'#
# phase_field_smooth = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
#                                                          microstructure_name=geometry_ID,
#                                                          coordinates=discretization.fft.coords)
# phase_field_smooth = np.abs(phase_field_smooth)
#
# phase_field = np.abs(phase_field_smooth - 1)

for i in np.arange(int(args.start_ratio),int(args.stop_ratio)):
    nb_of_filters =int(i)
    _info = {}

    phase_fem = np.zeros([2, *number_of_pixels])
    phase_inxyz = discretization.get_scalar_field(name='phase_field')

    # save

    results_name = (f'phase_field_N{number_of_pixels[0]}_F{nb_of_filters}_kappa{contrast}')

    geom_folder_path = file_folder_path + '/exp_data/' + 'exp_paper_JG_intro_circle/'
    # geom_folder_path = '//work/classic/fr_ml1145-martin_workspace_01/muFFTTO/experiments/paper_Jacobi_Green/exp_data/exp_paper_JG_intro_circle/'

    phase_inxyz.s[0, 0] = load_npy(geom_folder_path + results_name + f'.npy',
                                   tuple(discretization.subdomain_locations_no_buffers),
                                   tuple(discretization.nb_of_pixels), MPI.COMM_WORLD)

    material_data_field_C_0.s = np.einsum('ijkl,qxy->ijklqxy', elastic_C_1,
                                          np.broadcast_to(phase_inxyz.s[0, 0],
                                                          material_data_field_C_0.s[0, 0, 0, 0].shape))
    # material_data_field_C_0_rho=phase_field_at_quad_poits_1qnxyz
    # set macroscopic gradient
    macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
    discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient,
                                                   macro_gradient_field_ijqxyz=macro_gradient_field)

    # Set up right hand side
    rhs_field = discretization.get_unknown_size_field(name='rhs_field')
    discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0,
                                  macro_gradient_field_ijqxyz=macro_gradient_field,
                                  rhs_inxyz=rhs_field)


    # Solve mechanical equilibrium constrain
    def K_fun(x, Ax):
        discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0,
                                                  input_field_inxyz=x,
                                                  output_field_inxyz=Ax,
                                                  formulation='small_strain')


    omega = 1  # 2 / ( eig[-1]+eig[np.argmax(eig>0)])
    # Set up preconditioners
    # Green
    preconditioner = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=elastic_C_1)


    def M_fun_green(x, Px):
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
        formulation=formulation)


    def M_fun_Green_Jacobi(x, Px):
        # discretization.fft.communicate_ghosts(x)
        x_jacobi_temp = discretization.get_unknown_size_field(name='x_jacobi_temp')

        x_jacobi_temp.s = K_diag_alg.s * x.s
        discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                                   input_nodal_field_fnxyz=x_jacobi_temp,
                                                   output_nodal_field_fnxyz=Px)

        Px.s = K_diag_alg.s * Px.s


    def M_fun_Jacobi(x, Px):
        Px.s = K_diag_alg.s * K_diag_alg.s * x.s


    _info['norms_G_rr'] = []
    _info['norms_G_rz'] = []
    _info['norms_G_rGr'] = []


    def callback_G(it, x, r, p, z, stop_crit_norm):
        global _info

        """
        Callback function to print the current solution, residual, and search direction.
        """
        norm_of_rr = discretization.fft.communicator.sum(np.dot(r.ravel(), r.ravel()))
        norm_of_rz = discretization.fft.communicator.sum(np.dot(r.ravel(), z.ravel()))

        _info['norms_G_rr'].append(norm_of_rr)
        _info['norms_G_rz'].append(norm_of_rz)
        _info['norms_G_rGr'].append(stop_crit_norm)

        # if discretization.fft.communicator.rank == 0:
        #     print(len(_info['norms_G_rr']))
        #     print(norm_of_rr)


    solution_field_G = discretization.get_unknown_size_field(name='solution_G')

    solution_field_G.s.fill(0)
    solvers.conjugate_gradients_mugrid(
        comm=discretization.fft.communicator,
        fc=discretization.field_collection,
        hessp=K_fun,  # linear operator
        b=rhs_field,
        x=solution_field_G,
        P=M_fun_green,
        tol=tol_cg,
        maxiter=20000,
        callback=callback_G,
        norm_metric=M_fun_green
    )

    _info['norms_J_rr'] = []
    _info['norms_J_rz'] = []
    _info['norms_J_rGr'] = []


    def callback_J(it, x, r, p, z, stop_crit_norm):
        global _info

        """
        Callback function to print the current solution, residual, and search direction.
        """
        norm_of_rr = discretization.fft.communicator.sum(np.dot(r.ravel(), r.ravel()))
        norm_of_rz = discretization.fft.communicator.sum(np.dot(r.ravel(), z.ravel()))
        _info['norms_J_rr'].append(norm_of_rr)
        _info['norms_J_rz'].append(norm_of_rz)
        _info['norms_J_rGr'].append(stop_crit_norm)


    solution_field_J = discretization.get_unknown_size_field(name='solution_J')
    solution_field_J.s.fill(0)

    solvers.conjugate_gradients_mugrid(
        comm=discretization.fft.communicator,
        fc=discretization.field_collection,
        hessp=K_fun,  # linear operator
        b=rhs_field,
        x=solution_field_J,
        P=M_fun_Jacobi,
        tol=tol_cg,
        maxiter=1,
        callback=callback_J,
        norm_metric=M_fun_green
    )

    _info['norms_GJ_rr'] = []
    _info['norms_GJ_rz'] = []
    _info['norms_GJ_rGr'] = []


    def callback_GJ(it, x, r, p, z, stop_crit_norm):
        global _info

        """
        Callback function to print the current solution, residual, and search direction.
        """
        norm_of_rr = discretization.fft.communicator.sum(np.dot(r.ravel(), r.ravel()))
        norm_of_rz = discretization.fft.communicator.sum(np.dot(r.ravel(), z.ravel()))
        _info['norms_GJ_rr'].append(norm_of_rr)
        _info['norms_GJ_rz'].append(norm_of_rz)
        _info['norms_GJ_rGr'].append(stop_crit_norm)


    solution_field_GJ = discretization.get_unknown_size_field(name='solution__GJ')
    solution_field_GJ.s.fill(0)
    solvers.conjugate_gradients_mugrid(
        comm=discretization.fft.communicator,
        fc=discretization.field_collection,
        hessp=K_fun,  # linear operator
        b=rhs_field,
        x=solution_field_GJ,
        P=M_fun_Green_Jacobi,
        tol=tol_cg,
        maxiter=20000,
        callback=callback_GJ,
        norm_metric=M_fun_green
    )
    #
    #
    # nb_it[kk - 1, i] = (len(norms['residual_rz']))
    # norm_rz.append(norms['residual_rz'])
    # norm_rr.append(norms['residual_rr'])
    # norm_rMr.append(norms['data_scaled_rr'])
    # print(nb_it[kk - 1, i])
    # #########
    # displacement_field_combi, norms_combi = solvers.PCG(K_fun, rhs, x0=None, P=M_fun_combi, steps=int(1000),
    #                                                     toler=1e-6,
    #                                                     norm_type='data_scaled_rr',
    #                                                     norm_metric=M_fun)
    # nb_it_combi[kk - 1, i] = (len(norms_combi['residual_rz']))
    # norm_rz_combi.append(norms_combi['residual_rz'])
    # norm_rr_combi.append(norms_combi['residual_rr'])
    # norm_rMr_combi.append(norms_combi['data_scaled_rr'])
    # print(nb_it_combi[kk - 1, i])
    # #
    #
    # displacement_field_Jacobi, norms_Jacobi = solvers.PCG(K_fun, rhs, x0=None, P=M_fun_Jacobi, steps=int(2),
    #                                                       toler=1e-6,
    #                                                       norm_type='data_scaled_rr',
    #                                                       norm_metric=M_fun)
    # nb_it_Jacobi[kk - 1, i] = (len(norms_Jacobi['residual_rz']))
    # norm_rz_Jacobi.append(norms_Jacobi['residual_rz'])
    # norm_rr_Jacobi.append(norms_Jacobi['residual_rr'])
    # norm_rMr_Jacobi.append(norms_Jacobi['data_scaled_rr'])
    #
    # displacement_field_Richardson, norms_Richardson = solvers.Richardson(K_fun, rhs, x0=None, P=M_fun,
    #                                                                      omega=omega,
    #                                                                      steps=int(2),
    #                                                                      toler=1e-6)

    _info['nb_of_pixels'] = discretization.nb_of_pixels_global
    _info['nb_of_filters_aplication'] = nb_of_filters
    # phase_field_sol_FE_MPI = xopt.x.reshape([1, 1, *discretization.nb_of_pixels])
    grad = discretization.get_gradient_of_scalar_field(name='grad_of_phase_field')
    discretization.apply_gradient_operator_mugrid(u_inxyz=phase_inxyz, grad_u_ijqxyz=grad)
    grad_norm = np.sqrt(
        discretization.fft.communicator.sum(np.dot(grad.s[0].mean(axis=1).ravel(), grad.s[0].mean(axis=1).ravel())))
    grad_max = np.sqrt(
        discretization.mpi_reduction.max(grad.s[0].mean(axis=1)[0] ** 2 + grad.s[0].mean(axis=1)[1] ** 2))
    grad_max_inf = discretization.mpi_reduction.max(grad.s[0].mean(axis=1))
    _info['grad_norm'] = grad_norm
    _info['grad_max'] = grad_max
    _info['grad_max_inf'] = grad_max_inf

    if discretization.fft.communicator.rank == 0:
        print(f'nb_of_filters {nb_of_filters} iters = Green ' + '{}'.format(len(_info['norms_G_rr'])))
        print(f'Jacobi_' + '{}'.format(len(_info['norms_J_rr'])))
        print(f'Green_Jacobi_ iters =' + '{}'.format(len(_info['norms_GJ_rr'])))

        results_name = (f'N{number_of_pixels[0]}_F{nb_of_filters}_kappa{contrast}')

        np.savez(data_folder_path + results_name + f'_log.npz', **_info)
        print(data_folder_path + results_name + f'_log.npz')

##################
quit()
# print(norms)
# box = ax2.get_position()
# ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#
# # Put a legend to the right of the current axis
# ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.show()

fig = plt.figure()
gs = fig.add_gridspec(1, 1)
ax_1 = fig.add_subplot(gs[0, 0])
# ax_1.semilogy(norm_rr[0], label='PCG: Green', color='blue', linewidth=0)
for kk in np.arange(np.size(nb_pix_multips)):
    ax_1.plot(ratios[0:], nb_it[kk], 'g', marker='|', label=' Green', linewidth=1)
    # axs[1].plot(xopt2.f.num_iteration_.transpose()[1:3*i+1:3],"r", label='DGO ',linewidth=1)
    # axs[1].plot(xopt2.f.num_iteration_.transpose()[2:3*i+2:3],"r", label='DGO ',linewidth=1)

    ax_1.plot(ratios[0:], nb_it_Jacobi[kk], "b", marker='o', label='PCG Jacobi', linewidth=1)  # [0, 0:]
    ax_1.plot(ratios[0:], nb_it_combi[kk], "k", marker='x', label='PCG Green + Jacobi', linewidth=1)
#  ax2.semilogy(ratios[0:i + 1], nb_it_Richardson[0, 0:i + 1], "g", label=' Richardson Green ', linewidth=1)
#  ax2.semilogy(ratios[0:i + 1], nb_it_Richardson_combi[0, 0:i + 1], "y",  label=' Richardson Green + Jacobi ', linewidth=1)

# axs[1].legend()
ax_1.set_ylim(bottom=0)
ax_1.legend(['Green', 'Jacobi', 'Green + Jacobi'])
plt.show()
# quit()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each line with a different z offset
for i in np.arange(len(nb_pix_multips)):
    ax.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it[i], label='PCG: Green', color='blue')
    ax.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Jacobi[i], label='PCG: Jacobi', color='black')
    ax.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_combi[i], label='PCG: Green + Jacobi', color='red')
    ax.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
    ax.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
ax.set_xlabel('nb of filter aplications')
ax.set_ylabel('size')
ax.set_zlabel('# CG iterations')
plt.legend(['DGO', 'Jacobi', 'DGO + Jacobi', 'Richardson'])
plt.show()

for i in np.arange(ratios.size, step=1):
    kappa = kontrast[i]
    k = np.arange(max(map(len, norm_rr)))
    # print(f'k \n {k}')
    lb = eigen_LB[i]
    # print(f'lb \n {lb}')

    convergence = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** k
    convergence = convergence * norm_rr[i][0]

    # print(f'convergecnce \n {convergence}')
    fig = plt.figure()
    gs = fig.add_gridspec(1, 1)
    ax_1 = fig.add_subplot(gs[0, 0])
    ax_1.set_title(f'{i}', wrap=True)
    ax_1.semilogy(convergence, '--', label='estim', color='k')

    ax_1.semilogy(norm_rr[i], label='PCG: Green', color='r')
    ax_1.semilogy(norm_rr_Jacobi[i], label='PCG: Jacobi', color='b')

    # x_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
    # ax_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
    ax_1.set_xlabel('CG iterations')
    ax_1.set_ylabel('Norm of residua')
    plt.legend([r'$\kappa$ upper bound', 'Green', 'Jacobi', 'Green + Jacobi', 'Richardson'])
    ax_1.set_ylim([1e-7, norm_rr[i][0]])  # norm_rz[i][0]]/lb)
    print(max(map(len, norm_rr)))
    ax_1.set_xlim([0, max(map(len, norm_rr))])

    plt.show()

plt.show()
fig = plt.figure()
gs = fig.add_gridspec(1, 1)
ax_1 = fig.add_subplot(gs[0, 0])
ax_1.semilogy(norm_rr[0], label='PCG: Green', color='blue', linewidth=0)

# ax_1.set_ylim([1e-7, 1e0])
ax_1.set_ylim([1e-7, norm_rr[0][0]])  # norm_rz[i][0]]/lb)

print(max(map(len, norm_rz)))
ax_1.set_xlim([0, max(map(len, norm_rr))])


def convergence_gif_rz(i):
    kappa = kontrast[i]
    k = np.arange(max(map(len, norm_rr)))
    # print(f'k \n {k}')
    lb = eigen_LB[i]
    # print(f'lb \n {lb}')

    convergence = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** k
    convergence = convergence * norm_rr[i][0]

    ax_1.clear()

    ax_1.set_title(f'{i}', wrap=True)
    ax_1.semilogy(convergence, '--', label='estim', color='g')

    ax_1.semilogy(norm_rr[i], label='PCG: Green', color='g')
    ax_1.semilogy(norm_rr_Jacobi[i], label='PCG: Jacobi', color='b')
    ax_1.semilogy(norm_rr_combi[i], label='PCG: Jacobi', color='k')

    # x_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
    # ax_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
    ax_1.set_xlabel('CG iterations')
    ax_1.set_ylabel('Norm of residua')
    plt.legend([r'$\kappa$ upper bound', 'Green', 'Jacobi', 'DGO + Jacobi', 'Richardson'])
    ax_1.set_ylim([1e-7, norm_rr[i][0]])  # norm_rz[i][0]]/lb)
    print(max(map(len, norm_rr)))
    ax_1.set_xlim([0, max(map(len, norm_rr))])
    # axs[1].legend()
    # plt.legend(['', 'FEM: Green', 'FEM: Jacobi', 'FEM: Green + Jacobi','FEM: Richardson'])
    plt.legend(
        [r'$\kappa$ upper bound', 'Green', 'Jacobi', 'Green + Jacobi', 'Richardson Green', 'Richardson Green + Jacobi'],
        loc='center left', bbox_to_anchor=(0.8, 0.5))


ani = FuncAnimation(fig, convergence_gif_rz, frames=ratios.size, blit=False)
# axs[1].legend()middlemiddle
# Save as a GIF
ani.save(
    figure_folder_path + f"convergence__es2tgif_{number_of_pixels[0]}comparison{ratios[-1]}_RichardsonJacobi{geometry_ID}_circle_inc_to_smooth_semiloplots3.gif",
    writer=PillowWriter(fps=1))

plt.show()
# -------------------------------------------------------------------------------------------------------
# for i in np.arange(ratios.size,step=1):
#     kappa=kontrast[i]
#     kappa_2=kontrast_2[i]
#     k=np.arange(len(norm_rr_Jacobi[i]))
#     print(f'k \n {k}')
#
#     convergence=((np.sqrt(kappa)-1)/(np.sqrt(kappa)+1))**k
#     convergence=convergence*norm_rr[i][0]
#     convergence2 = ((np.sqrt(kappa_2) - 1) / (np.sqrt(kappa_2) + 1)) ** k
#     convergence2 = convergence2 * norm_rr[i][0]
#
#
#     print(f'convergecnce \n {convergence}')
#     fig = plt.figure()
#     gs = fig.add_gridspec(1, 1)
#     ax_1 = fig.add_subplot(gs[0, 0])
#     ax_1.set_title(f'{i}', wrap=True)
#     ax_1.semilogy(convergence, '-',label='estim', color='green')
#     #ax_1.semilogy(convergence2,'.-', label='estim2', color='green')
#
#     ax_1.semilogy(norm_rr[i], label='PCG: Green', color='blue')
#     ax_1.semilogy(norm_rr_Jacobi[i], label='PCG: Jacobi', color='black')
#     #ax_1.semilogy(norm_rr_combi[i], label='PCG: Green + Jacobi', color='red')
#     #x_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
#     #ax_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
#     ax_1.set_xlabel('CG iterations')
#     ax_1.set_ylabel('Norm of residuals')
#     plt.legend([r'$\kappa$ upper bound','Green', 'Jacobi', 'Green + Jacobi','Richardson'])
#     ax_1.set_ylim([1e-7, norm_rr[i][0]])
#     print(max(map(len, norm_rr)))
#     ax_1.set_xlim([0, max(map(len, norm_rr))])
#
#     plt.show()

fig = plt.figure()
gs = fig.add_gridspec(1, 1)
ax_1 = fig.add_subplot(gs[0, 0])
ax_1.semilogy(norm_rr[0], label='PCG: Green', color='blue', linewidth=0)

ax_1.set_ylim([1e-7, 1e0])
print(max(map(len, norm_rr)))
ax_1.set_xlim([0, max(map(len, norm_rr))])


def convergence_gif(i):
    kappa = kontrast[i]
    k = np.arange(max(map(len, norm_rr)))
    print(f'k \n {k}')

    convergence = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** k
    convergence = convergence * norm_rr[i][0]
    print(f'convergecnce \n {convergence}')
    ax_1.clear()

    ax_1.set_title(f'{i}', wrap=True)
    ax_1.semilogy(convergence, '--', label='estim', color='k')

    ax_1.semilogy(norm_rr[i], label='PCG: Green', color='r')
    ax_1.semilogy(norm_rr_Jacobi[i], label='PCG: Jacobi', color='b')
    # ax_1.semilogy(norm_rr_combi[i], label='PCG: Green + Jacobi', color='red')
    # x_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
    # ax_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
    ax_1.set_xlabel('CG iterations')
    ax_1.set_ylabel('Norm of residuals')
    plt.legend([r'$\kappa$ upper bound', 'Green', 'Jacobi', 'Green + Jacobi', 'Richardson'])
    ax_1.set_ylim([1e-7, 1e0])
    print(max(map(len, norm_rr)))
    ax_1.set_xlim([0, max(map(len, norm_rr))])
    # axs[1].legend()
    # plt.legend(['', 'FEM: Green', 'FEM: Jacobi', 'FEM: Green + Jacobi','FEM: Richardson'])
    plt.legend(
        [r'$\kappa$ upper bound', 'Green', 'Jacobi', 'Green + Jacobi', 'Richardson Green', 'Richardson Green + Jacobi'],
        loc='center left', bbox_to_anchor=(0.8, 0.5))


ani = FuncAnimation(fig, convergence_gif, frames=ratios.size, blit=False)
# axs[1].legend()middlemiddle
# Save as a GIF
ani.save(figure_folder_path +
         f" convergence_gif_{number_of_pixels[0]}comparison{ratios[-1]}_RichardsonJacobi{geometry_ID}_circle_inc_to_smooth_semiloplots3.gif",
         writer=PillowWriter(fps=1))

plt.show()

plot_evolion = True
if plot_evolion:
    for nb_tiles in [1, ]:
        # fig = plt.figure()

        #
        # fig, axs = plt.subplots(nrows=2, ncols=2,
        #                         figsize=(6, 6)  )
        fig = plt.figure()
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax3 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, :])
        # axs[0] = plt.axes(xlim=(0, nb_tiles * N), ylim=(0, nb_tiles * N))
        ax1.imshow(phase_field, cmap=mpl.cm.Greys, vmin=1e-4, vmax=1)
        ax3.semilogy(phase_field[:, phase_field.shape[0] // 2], linewidth=0)
        # ax3.plot(phase_field[:,phase_field.shape[0]//2], linewidth=0)
        ax3.set_ylim([1e-4, 1])
        # print(ratios)

        # print(nb_it)
        ax2.plot(ratios, nb_it_Jacobi[0], label='nb_it_Laplace', linewidth=0)
        ax2.set_ylim([1, 100])

        # axs[1].plot(xopt.f.num_iteration_.transpose()[::3], 'w'  , linewidth=0)
        # axs[1].plot(xopt3.f.num_iteration_.transpose(), "b", label='Jacoby', linewidth=0)
        # axs[1].plot(xopt.f.num_iteration_.transpose(), "k", label='DGO + Jacoby', linewidth=0)
        # legend = plt.legend()
        # Animation function to update the image
        # ax2.set_xlabel('')
        ax2.set_ylabel('# PCG iterations')


        def update(i):
            ratio = ratios[i]
            phase_field = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                              microstructure_name=geometry_ID,
                                                              coordinates=discretization.fft.coords)
            phase_field = np.abs(phase_field)  # -1
            phase_field += 1e-4
            for a in np.arange(i):
                phase_field = apply_smoother_log10(phase_field)  # _log10
            # min_val = np.min(phase_field)
            # max_val = np.max(phase_field)
            # phase_field = 1e-4 + (phase_field - min_val) * (1 - 1e-4) / (max_val - min_val)
            # phase_field = ratio * phase_field_smooth + (1 - ratio) * phase_field_pwconst

            ax1.clear()
            ax1.imshow(np.transpose(phase_field), cmap=mpl.cm.Greys, vmin=1e-4, vmax=1)
            ax1.set_title(r'Density $\rho$', wrap=True)
            #: {np.max(phase_field)/np.min(phase_field):.1e}  \n'                          f'  min = {np.min(phase_field):.1e}
            ax3.clear()
            ax3.semilogy(phase_field[:, phase_field.shape[0] // 2], linewidth=1)
            # ax3.plot(phase_field[:, phase_field.shape[0] // 2], linewidth=1)
            ax3.set_ylim([1e-4, 1])
            ax3.set_title(f'Cross section')

            ax2.plot(ratios[0:i + 1], nb_it[0, 0:i + 1], 'g', marker='|', label=' Green', linewidth=1)
            # axs[1].plot(xopt2.f.num_iteration_.transpose()[1:3*i+1:3],"r", label='DGO ',linewidth=1)
            # axs[1].plot(xopt2.f.num_iteration_.transpose()[2:3*i+2:3],"r", label='DGO ',linewidth=1)

            ax2.plot(ratios[0:i + 1], nb_it_Jacobi[0, 0:i + 1], "b", marker='o', label='PCG Jacobi', linewidth=1)
            ax2.plot(ratios[0:i + 1], nb_it_combi[0, 0:i + 1], "k", marker='x', label='PCG Green + Jacobi', linewidth=1)
            #  ax2.semilogy(ratios[0:i + 1], nb_it_Richardson[0, 0:i + 1], "g", label=' Richardson Green ', linewidth=1)
            #  ax2.semilogy(ratios[0:i + 1], nb_it_Richardson_combi[0, 0:i + 1], "y",  label=' Richardson Green + Jacobi ', linewidth=1)
            ax2.set_ylim(bottom=0)
            # axs[1].legend()
            ax2.legend(['', 'Green', 'Jacobi', 'Green + Jacobi'])
            # plt.legend(['', 'FEM: Green', 'FEM: Jacobi', 'FEM: Green + Jacobi','FEM: Richardson'])


        # plt.legend(['', ' Green', 'Jacobi', 'Green + Jacobi','Richardson Green','Richardson Green + Jacobi'],loc='best', bbox_to_anchor=(0.7, 0.5))
        #        ax2.legend([r'$\kappa$ upper bound', 'Green', 'Jacobi', 'Green + Jacobi', 'Richardson'])

        # plt.legend([ '', 'Green', 'Jacobi'  ])

        # img.set_array(xopt_it)
        # ax2.legend([r'$\kappa$ upper bound', 'Green', 'Jacobi', 'Green + Jacobi', 'Richardson'])

        # box = ax2.get_position()
        # ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        #
        # # Put a legend to the right of the current axis
        # ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # Create animation
        # ani = FuncAnimation(fig, update, frames=xopt.f.norms_f.size - 1, blit=False)

        ani = FuncAnimation(fig, update, frames=ratios.size, blit=False)
        # axs[1].legend()middlemiddle
        # Save as a GIF
        ani.save(figure_folder_path +
                 f"movie_exp_paper_JG_intro_{number_of_pixels[0]}comparison{ratios[-1]}_RichardsonJacobi{geometry_ID}_circle_inc_to_smooth_semiloplots3.gif",
                 writer=PillowWriter(fps=4))

    plt.show()
