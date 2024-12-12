from cProfile import label

import numpy as np
import scipy as sc
import time
import matplotlib.pyplot as plt
import matplotlib as mpl

import matplotlib.pyplot as plt
from IPython.terminal.shortcuts.filters import KEYBINDING_FILTERS
from PySide2.examples.opengl.contextinfo import colors
from matplotlib.animation import FuncAnimation, PillowWriter
from sympy.physics.quantum.sho1d import omega

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library
from mpl_toolkits import mplot3d

problem_type = 'conductivity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'

domain_size = [1, 1]
nb_pix_multips = [4]  # 3[2,3,4,5,6,7,8]  # ,3
small = np.arange(0., .1, 0.005)
middle = np.arange(0.1, 0.9, 0.03)

large = np.arange(0.9, 1.0 + 0.005, 0.005)
ratios = np.concatenate((small, middle, large))
ratios = np.arange(0., 1.1, 0.2)
ratios = np.arange(0., 1.1, 0.2)
ratios = np.arange(1)

for aniso in np.arange(-4, 7):
    exponent = np.array(aniso, dtype=float)

    nb_it = np.zeros((len(nb_pix_multips), ratios.size), )
    nb_it_combi = np.zeros((len(nb_pix_multips), ratios.size), )
    nb_it_Jacobi = np.zeros((len(nb_pix_multips), ratios.size), )
    nb_it_Richardson = np.zeros((len(nb_pix_multips), ratios.size), )
    nb_it_Richardson_combi = np.zeros((len(nb_pix_multips), ratios.size), )

    norm_rr_combi = []
    norm_rz_combi = []
    norm_rr_Jacobi = []
    norm_rz_Jacobi = []
    norm_rr = []
    norm_rz = []
    norm_up = []
    norm_up_combi = []
    kontrast = []
    kontrast_2 = []
    eigen_LB = []

    for kk in np.arange(np.size(nb_pix_multips)):
        nb_pix_multip = nb_pix_multips[kk]
        number_of_pixels = (nb_pix_multip * 8, nb_pix_multip * 8)
        # number_of_pixels = (16,16)
        # number_of_pixels = (6,6)
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)

        discretization = domain.Discretization(cell=my_cell,
                                               nb_of_pixels_global=number_of_pixels,
                                               discretization_type=discretization_type,
                                               element_type=element_type)
        start_time = time.time()

        # set macroscopic gradient
        macro_gradient = np.array([1.0, 0.0])
        # create material data field

        # conductivity_C_0 = np.array([[1., 0], [0, 1.0]])
        # mat_contrast_2 = 100
        conductivity_C_0 = np.array([[1e+4, 0], [0, 1.0]])  # matrix
        conductivity_C_1 = np.array([[1 , 0], [0, 1.0]])  # inclusion


        conductivity_C_ref = np.array([[10**exponent   , 0], [0, 1]])
        conductivity_C_ref_green_in_jacobi = np.array([[1 , 0], [0, 1.0]])
        conductivity_C_for_rhs = np.array([[1e+4, 0], [0, 1.0]])

        material_data_field_C_0 = np.einsum('ij,qxy->ijqxy', conductivity_C_0,
                                            np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                              *discretization.nb_of_pixels])))

        material_data_field_C_1 = np.einsum('ij,qxy->ijqxy', conductivity_C_1,
                                            np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                              *discretization.nb_of_pixels])))

        refmaterial_data_field_ = np.einsum('ij,qxy->ijqxy', conductivity_C_ref,
                                            np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                              *discretization.nb_of_pixels])))
        refmaterial_data_field_JG = np.einsum('ij,qxy->ijqxy', conductivity_C_ref_green_in_jacobi,
                                            np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                              *discretization.nb_of_pixels])))

        refmaterial_data_field_for_rhs = np.einsum('ij,qxy->ijqxy', conductivity_C_for_rhs,
                                              np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                                *discretization.nb_of_pixels])))
        print('Data = \n {}'.format(conductivity_C_0))

        # material distribution
        geometry_ID = 'square_inclusion_equal_volfrac'  # 'linear'#, square_inclusion_equal_volfrac   square_inclusion
        initial_phase_field = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                                  microstructure_name=geometry_ID,
                                                                  coordinates=discretization.fft.coords)

        initial_phase_field[initial_phase_field < 0.5] = 0
        initial_phase_field[initial_phase_field >= 0.5] = 1
        fig = plt.figure(aniso)
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[0, 1])
        ax4 = fig.add_subplot(gs[1, 1])

        plt.tight_layout()
        for i in np.arange(0, ratios.size):
            ratio = ratios[i]


            def apply_smoother(phase):
                # Define a 2D smoothing kernel
                kernel = np.array([[0.0625, 0.125, 0.0625],
                                   [0.125, 0.25, 0.125],
                                   [0.0625, 0.125, 0.0625]])

                # Apply convolution for smoothing
                smoothed_arr = sc.signal.convolve2d(phase, kernel, mode='same', boundary='wrap')
                return smoothed_arr


            def apply_smoother_log10(phase):
                # Define a 2D smoothing kernel
                kernel = np.array([[0.0625, 0.125, 0.0625],
                                   [0.125, 0.25, 0.125],
                                   [0.0625, 0.125, 0.0625]])

                # Apply convolution for smoothing
                smoothed_arr = sc.signal.convolve2d(np.log10(phase), kernel, mode='same', boundary='wrap')
                smoothed_arr[number_of_pixels[0] // 2 - 1:number_of_pixels[0] // 2 + 1,
                number_of_pixels[0] // 2 - 1:number_of_pixels[0] // 2 + 1] = -4
                smoothed_arr = 10 ** smoothed_arr

                return smoothed_arr


            if i == 0:
                phase_field = initial_phase_field  # + 1e-4
            if i > 0:
                phase_field = apply_smoother_log10(phase_field)

            # phase_fem = np.zeros([2, *number_of_pixels])
            # phase_fnxyz = discretization.get_scalar_sized_field()
            # phase_fnxyz[0, 0, ...] = phase_field
            #
            # material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * phase_field
            material_data_field_C_0_rho = np.zeros_like(material_data_field_C_0)

            material_data_field_C_0_rho[..., phase_field >= 0.5] = material_data_field_C_0[
                ..., phase_field >= 0.5]
            material_data_field_C_0_rho[..., phase_field <= 0.5] = material_data_field_C_1[
                ..., phase_field <= 0.5]

            # material_data_field_C_0_rho=phase_field_at_quad_poits_1qnxyz

            # plotting eigenvalues
            K = discretization.get_system_matrix(refmaterial_data_field_JG)

            K = discretization.get_system_matrix(material_data_field_C_0_rho)
            # reduced_K = np.copy(K)
            K[:, 0] = 0
            K[0, :] = 0
            K[:, number_of_pixels[0]] = 0
            K[number_of_pixels[0], :] = 0
            K[0, 0] = 1
            K[number_of_pixels[0], number_of_pixels[0]] = 1

            M = discretization.get_system_matrix(refmaterial_data_field_)
            M[:, 0] = 0
            M[0, :] = 0
            M[:, number_of_pixels[0]] = 0
            M[number_of_pixels[0], :] = 0
            M[0, 0] = 1
            M[number_of_pixels[0], number_of_pixels[0]] = 1

            min_val = np.min(phase_field)
            max_val = np.max(phase_field)

            #kontrast.append(max_val / min_val)
            eigen_LB.append(min_val)

            K_diag_half = np.copy(np.diag(K))
            K_diag_half[K_diag_half < 1e-16] = 0
            K_diag_half[K_diag_half != 0] = 1 / np.sqrt(K_diag_half[K_diag_half != 0])
            if i // 1 == 0:
                DKDsym = np.matmul(np.diag(K_diag_half), np.matmul(K, np.diag(K_diag_half)))
                # kontrast_2.append(eig[-3] / eig[np.argmax(eig > 0)])
                # eig = sc.linalg.eigh(a=K, b=None, eigvals_only=True)

                # ax1.semilogy(sorted(eig)[:], label=f'{i}', marker='.', linewidth=0, markersize=5)
                # ax1.set_ylim([1e-5, 1e1])
                # ax1.set_title(f'K : min={sorted(eig)[0]:.4g}, max={sorted(eig)[-1]:.4g}')
                # ax1.set_xlim(left=0)

                # eig = sc.linalg.eigh(a=DKDsym, b=None, eigvals_only=True)
                #
                # print(f'min={sorted(eig)[0]:.4g}, max={sorted(eig)[-1]:.4g}')
                #
                # ax2.semilogy(sorted(eig), label=f'{i}', marker='.', linewidth=0, markersize=5)  # sorted(eig)[1:-2]
                # ax2.set_ylim([1e-5, 1e1])
                # ax2.set_xlim(left=0)
                # ax2.set_title(f'DKD :min={sorted(eig)[1]:.4g}, max={sorted(eig)[-1]:.4g}')

                eig ,eigvals= sc.linalg.eigh(a=K, b=M)#, eigvals_only=True
                omega = 1  # 2 / ( eig[-1]+eig[np.argmax(eig>0)])
                ax3.semilogy(sorted(eig)[:], label=f'{i}', marker='.', linewidth=0, markersize=5)
                ax3.set_ylim([1e-4, 1e4])
                ax3.set_xlim(left=0)
                lambda_min_green = sorted(eig)[0]
                kontrast.append(sorted(eig)[-1]/ sorted(eig)[0])
                ax3.set_title(f'G K : min={sorted(eig)[0]:.4g}, max={sorted(eig)[-1]:.4g}')

            eig = sc.linalg.eigh(a=DKDsym, b=M, eigvals_only=True)
            min_J = sorted(eig)[1]
            max_J = sorted(eig)[-2]
            lambda_min_green_jacobi = sorted(eig)[0]
            kontrast_2.append((max_J / min_J))
            # if i // 1 == 0:
            #     print(f'min={sorted(eig)[0]:.4g}, max={sorted(eig)[-1]:.4g}')
            #
            #     ax4.semilogy(sorted(eig), label=f'{i}', marker='.', linewidth=0, markersize=5)  # sorted(eig)[1:-2]
            #     ax4.set_ylim([1e-5, 1e1])
            #     ax4.set_xlim(left=0)
            #
            #     ax4.set_title(f'G DKD :min={sorted(eig)[0]:.4g}, max={sorted(eig)[-1]:.4g}')

            # # $33333

            # K = discretization.get_system_matrix(material_data_field=material_data_field_C_0_rho)
            # material_data_field_C_0=np.mean(material_data_field_C_0_rho,axis=(4,5,6))
            # mean_material=np.mean(material_data_field_C_0_rho,axis=(4,5,6))
            # material_data_field_C_0_ratio = np.einsum('ijkl,qxy->ijklqxy', mean_material,
            #                                     np.ones(np.array([discretization.nb_quad_points_per_pixel,
            #                                                       *discretization.nb_of_pixels])))
            # Set up right hand side
            macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)
            perturb=np.random.random(macro_gradient_field.shape)
            macro_gradient_field += perturb-np.mean(perturb)

            # Solve mechanical equilibrium constrain
            rhs = discretization.get_rhs(material_data_field_C_0_rho, macro_gradient_field)
            #rhs = discretization.get_rhs(refmaterial_data_field_for_rhs, macro_gradient_field)
            # rhs =np.random.random(rhs.shape)
            # rhs += rhs - np.mean(rhs)
            normed_rhs=rhs/np.linalg.norm(rhs)
            weights=np.abs(np.transpose(eigvals) @ normed_rhs.reshape(-1))
            K_fun = lambda x: discretization.apply_system_matrix(material_data_field_C_0_rho, x)

            preconditioner = discretization.get_preconditioner_NEW(
                reference_material_data_field_ijklqxyz=refmaterial_data_field_)

            M_fun = lambda x: discretization.apply_preconditioner_NEW(preconditioner_Fourier_fnfnqks=preconditioner,
                                                                      nodal_field_fnxyz=x)

            K_diag_alg = discretization.get_preconditioner_Jacoby_fast(
                material_data_field_ijklqxyz=material_data_field_C_0_rho)
            preconditioner_combi = discretization.get_preconditioner_NEW(
                reference_material_data_field_ijklqxyz=refmaterial_data_field_JG)
            # print(K_diag_alg)
            M_fun_combi = lambda x: K_diag_alg * discretization.apply_preconditioner_NEW(
                preconditioner_Fourier_fnfnqks=preconditioner_combi,
                nodal_field_fnxyz=K_diag_alg * x)
            # #
            M_fun_Jacobi = lambda x: K_diag_alg * K_diag_alg * x

            displacement_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(1000), toler=1e-10,
                                                    norm_energy_upper_bound=True, lambda_min=lambda_min_green)
            nb_it[kk - 1, i] = (len(norms['residual_rz']))
            norm_rz.append(norms['residual_rz'])
            norm_rr.append(norms['residual_rr'])
            norm_up.append(norms['energy_lb'])

            # print(nb_it)
            #########
            displacement_field_combi, norms_combi = solvers.PCG(K_fun, rhs, x0=None, P=M_fun_combi, steps=int(1000),
                                                                toler=1e-10,
                                                                norm_energy_upper_bound=True,
                                                                lambda_min=lambda_min_green_jacobi)
            nb_it_combi[kk - 1, i] = (len(norms_combi['residual_rz']))
            norm_rz_combi.append(norms_combi['residual_rz'])
            norm_rr_combi.append(norms_combi['residual_rr'])
            norm_up_combi.append(norms_combi['energy_lb'])

            ax1.semilogy(norm_rr[i], label='PCG: Green', color='r')# / norm_rr[i][0]
            ax1.semilogy(norm_up[i] , label='PCG: Green', color='k')#/ norm_up[i][0]
            ax1.set_ylim([1e-10, 1e10])
            ax1.set_xlim([0, 120])

            #ax1.set_xlim(left=0)

            kappa_Green = 100 # kontrast[i]
            k = np.arange(max(map(len, norm_rr)))
            print(f'k \n {k}')
            lb = eigen_LB[i]
            print(f'lb \n {lb}')
            print(f'kappa_Green \n {kappa_Green}')

            convergence_Green = ((np.sqrt(kappa_Green) - 1) / (np.sqrt(kappa_Green) + 1)) ** k
            #convergence_Green = convergence_Green * norm_rr[i][0]
            convergence_Green = convergence_Green * 1e3

            ax1.semilogy(convergence_Green, '--', label='estim green', color='r')
            ax1.semilogy(np.ones_like(convergence_Green), '--', label='estim green', color='g',linewidth=1)
            ax1.grid()

            # ax_1.semilogy(norm_rr_Jacobi[i], label='PCG: Jacobi', color='b')
            ax2.semilogy(norm_rr_combi[i] , label='PCG: Green Jacobi', color='b')#/ norm_rr_combi[i][0]
            ax2.semilogy(norm_up_combi[i] , label='PCG: Green Jacobi Error', color='k')#/ norm_up_combi[i][0]

            displacement_field_Jacobi, norms_Jacobi = solvers.PCG(K_fun, rhs, x0=None, P=M_fun_Jacobi, steps=int(1000),
                                                                  toler=1e-10)
            nb_it_Jacobi[kk - 1, i] = (len(norms_Jacobi['residual_rz']))
            norm_rz_Jacobi.append(norms_Jacobi['residual_rz'])
            norm_rr_Jacobi.append(norms_Jacobi['residual_rr'])

            ax4.semilogy(weights , '.',label='PCG: Green Jacobi Error', color='k')#/ norm_up_combi[i][0]
            ax4.set_ylim([1e-10, 1e0])

            # displacement_field_Richardson, norms_Richardson = solvers.Richardson(K_fun, rhs, x0=None, P=M_fun,
            #                                                                      omega=omega,
            #                                                                      steps=int(1000),
            #                                                                      toler=1e-6)
            # nb_it_Richardson[kk - 1, i] = (len(norms_Richardson['residual_rr']))
            # norm_rr_Richardson = norms_Richardson['residual_rr'][-1]

            # displacement_field_Richardson_combi, norms_Richardson_combi = solvers.Richardson(K_fun, rhs, x0=None,
            #                                                                                  P=M_fun_combi,
            #                                                                                  omega=omega * 0.4,
            #                                                                                  steps=int(1000),
            #                                                                                  toler=1e-6)
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
            print(ratio)
            plt.show()
quit()
for a in [1]:
    for i in np.arange(ratios.size, step=3):
        kappa_Green = kontrast[i]
        k = np.arange(max(map(len, norm_rr)))
        print(f'k \n {k}')
        lb = eigen_LB[i]
        print(f'lb \n {lb}')
        print(f'kappa_Green \n {kappa_Green}')

        convergence_Green = ((np.sqrt(kappa_Green) - 1) / (np.sqrt(kappa_Green) + 1)) ** k
        convergence_Green = convergence_Green * norm_rr[i][0]

        kappa_Green_Jacobi = kontrast_2[i]
        convergence_Green_Jacobi = ((np.sqrt(kappa_Green_Jacobi) - 1) / (np.sqrt(kappa_Green_Jacobi) + 1)) ** k
        convergence_Green_Jacobi = convergence_Green_Jacobi * norm_rr_combi[i][0]
        print(f'kappa_Green_Jacobi \n {kappa_Green_Jacobi}')

        # print(f'convergecnce \n {convergence_Green}')
        fig = plt.figure()
        gs = fig.add_gridspec(1, 1)
        ax_1 = fig.add_subplot(gs[0, 0])
        ax_1.set_title(f'{i}', wrap=True)
        ax_1.semilogy(convergence_Green, '--', label='estim green', color='r')
        ax_1.semilogy(convergence_Green_Jacobi, '--', label='estim Green Jacobi', color='b')

        ax_1.semilogy(norm_rr[i], label='PCG: Green', color='r')
        # ax_1.semilogy(norm_rr_Jacobi[i], label='PCG: Jacobi', color='b')
        ax_1.semilogy(norm_rr_combi[i], label='PCG: Green Jacobi', color='b')

        # x_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
        # ax_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
        ax_1.set_xlabel('CG iterations')
        ax_1.set_ylabel('Norm of residua')
        plt.legend([r'$\kappa$ upper bound Green', r'$\kappa$ upper bound Green + Jacobi', 'Green', 'Green + Jacobi',
                    'Richardson'])
        ax_1.set_ylim([1e-7, norm_rr_combi[i][0]])  # norm_rz[i][0]]/lb)
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
        print(f'k \n {k}')
        lb = eigen_LB[i]
        print(f'lb \n {lb}')

        convergence = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** k
        convergence = convergence * norm_rr[i][0]

        kappa_Green_Jacobi = kontrast_2[i]
        convergence_Green_Jacobi = ((np.sqrt(kappa_Green_Jacobi) - 1) / (np.sqrt(kappa_Green_Jacobi) + 1)) ** k
        convergence_Green_Jacobi = convergence_Green_Jacobi * norm_rr_combi[i][0]

        ax_1.clear()

        ax_1.set_title(f'{i}', wrap=True)
        ax_1.semilogy(convergence, '--', label='estim', color='r')
        ax_1.semilogy(convergence_Green_Jacobi, '--', label='estim Green Jacobi', color='b')

        ax_1.semilogy(norm_rr[i], label='PCG: Green', color='r')
        ax_1.semilogy(norm_rr_combi[i], label='PCG: Jacobi', color='b')

        # x_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
        # ax_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
        ax_1.set_xlabel('CG iterations')
        ax_1.set_ylabel('Norm of residua')
        plt.legend([r'$\kappa$ upper bound Green', r'$\kappa$ upper bound Green + Jacobi', 'Green', 'Green + Jacobi',
                    'Richardson'])
        ax_1.set_ylim([1e-7, norm_rr[i][0]])  # norm_rz[i][0]]/lb)
        print(max(map(len, norm_rr)))
        ax_1.set_xlim([0, max(map(len, norm_rr))])
        # axs[1].legend()
        # plt.legend(['', 'FEM: Green', 'FEM: Jacobi', 'FEM: Green + Jacobi','FEM: Richardson'])
        plt.legend([r'$\kappa$ upper bound Green', r'$\kappa$ upper bound Green + Jacobi', 'Green', 'Green + Jacobi',
                    'Richardson'])

        plt.legend([r'$\kappa$ upper bound Green', r'$\kappa$ upper bound Green + Jacobi', 'Green', 'Green + Jacobi',
                    'Richardson Green + Jacobi'],
                   loc='center left', bbox_to_anchor=(0.5, 0.5))


    ani = FuncAnimation(fig, convergence_gif_rz, frames=ratios.size, blit=False)
    # axs[1].legend()middlemiddle
    # Save as a GIF
    ani.save(
        f"./figures/convergence_estimatess2tgif_{number_of_pixels[0]}comparison{ratios[-1]}_RichardsonJacobi{geometry_ID}_circle_inc_to_smooth_semiloplots3.gif",
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
        plt.legend([r'$\kappa$ upper bound', 'Green', 'Jacobi', 'Green + Jacobi', 'Richardson Green',
                    'Richardson Green + Jacobi'],
                   loc='center left', bbox_to_anchor=(0.8, 0.5))


    ani = FuncAnimation(fig, convergence_gif, frames=ratios.size, blit=False)
    # axs[1].legend()middlemiddle
    # Save as a GIF
    ani.save(
        f"./figures/convergence_gif_{number_of_pixels[0]}comparison{ratios[-1]}_RichardsonJacobi{geometry_ID}_circle_inc_to_smooth_semiloplots3.gif",
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
            print(ratios)

            print(nb_it)
            ax2.plot(ratios, nb_it_Jacobi[0], label='nb_it_Laplace', linewidth=0)
            ax3.set_ylim([1e0, 1e3])

            # axs[1].plot(xopt.f.num_iteration_.transpose()[::3], 'w'  , linewidth=0)
            # axs[1].plot(xopt3.f.num_iteration_.transpose(), "b", label='Jacoby', linewidth=0)
            # axs[1].plot(xopt.f.num_iteration_.transpose(), "k", label='DGO + Jacoby', linewidth=0)
            # legend = plt.legend()
            # Animation function to update the image
            # ax2.set_xlabel('')
            ax2.set_ylabel('# PCG iterations')


            def update(i):
                ratio = ratios[i]
                # phase_field = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                #                                                   microstructure_name='circle_inclusion',
                #                                                   coordinates=discretization.fft.coords)
                phase_field = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                                  microstructure_name=geometry_ID,
                                                                  coordinates=discretization.fft.coords)

                phase_field[phase_field < 0.5] = 0
                phase_field[phase_field >= 0.5] = 1

                # phase_field=np.abs(phase_field-1)
                phase_field += 1e-4
                for a in np.arange(i):
                    phase_field = apply_smoother_log10(phase_field)
                # min_val = np.min(phase_field)
                # max_val = np.max(phase_field)
                # phase_field = 1e-4 + (phase_field - min_val) * (1 - 1e-4) / (max_val - min_val)
                # phase_field = ratio * phase_field_smooth + (1 - ratio) * phase_field_pwconst

                ax1.clear()
                ax1.imshow(phase_field, cmap=mpl.cm.Greys, vmin=1e-4, vmax=1)
                ax1.set_title(r'Density $\rho$', wrap=True)
                #: {np.max(phase_field)/np.min(phase_field):.1e}  \n'                          f'  min = {np.min(phase_field):.1e}
                ax3.clear()
                ax3.semilogy(phase_field[:, phase_field.shape[0] // 2], linewidth=1)
                # ax3.plot(phase_field[:, phase_field.shape[0] // 2], linewidth=1)
                ax3.set_ylim([5e-5, 2])
                ax3.set_title(f'Cross section')

                ax2.plot(ratios[0:i + 1], nb_it[0, 0:i + 1], 'r', label='PCG  Green', linewidth=1)
                # axs[1].plot(xopt2.f.num_iteration_.transpose()[1:3*i+1:3],"r", label='DGO ',linewidth=1)
                # axs[1].plot(xopt2.f.num_iteration_.transpose()[2:3*i+2:3],"r", label='DGO ',linewidth=1)

                ax2.plot(ratios[0:i + 1], nb_it_Jacobi[0, 0:i + 1], "b", label='PCG Jacobi', linewidth=1)
                ax2.plot(ratios[0:i + 1], nb_it_combi[0, 0:i + 1], "k", label='PCG Green + Jacobi', linewidth=1)
                #  ax2.semilogy(ratios[0:i + 1], nb_it_Richardson[0, 0:i + 1], "g", label=' Richardson Green ', linewidth=1)
                #  ax2.semilogy(ratios[0:i + 1], nb_it_Richardson_combi[0, 0:i + 1], "y",  label=' Richardson Green + Jacobi ', linewidth=1)

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
            ani.save(
                f"./figures/movie2222_{number_of_pixels[0]}comparison{ratios[-1]}_RichardsonJacobi{geometry_ID}_circle_inc_to_smooth_semiloplots3.gif",
                writer=PillowWriter(fps=4))

        plt.show()

        # print(norms)
# box = ax2.get_position()
# ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#
# # Put a legend to the right of the current axis
# ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each line with a different z offset
for i in np.arange(len(nb_pix_multips)):
    ax.plot(ratios, nb_pix_multips[i], zs=nb_it[i], label='PCG: Green', color='green')
    ax.plot(ratios, nb_pix_multips[i], zs=nb_it_Jacobi[i], label='PCG: Jacobi', color='black')
    ax.plot(ratios, nb_pix_multips[i], zs=nb_it_combi[i], label='PCG: Green + Jacobi', color='red')
    ax.plot(ratios, nb_pix_multips[i], zs=nb_it_Richardson[i], linestyle='--', label='Richardson Green',
            color='green', )
    ax.plot(ratios, nb_pix_multips[i], zs=nb_it_Richardson_combi[i], linestyle='--', label='Richardson Green+Jacobi',
            color='red')
ax.set_xlabel('nb of filter aplications')
ax.set_ylabel('size')
ax.set_zlabel('# CG iterations')
plt.legend(['DGO', 'Jacobi', 'DGO + Jacobi', 'Richardson'])
plt.show()
# quit()

fig = plt.figure()
gs = fig.add_gridspec(1, 1)
ax = fig.add_subplot(gs[0, 0])
# Plot each line with a different z offset
for i in np.arange(len(nb_pix_multips)):
    ax.plot(ratios, nb_it[i], label='PCG: Green', color='green')
    ax.plot(ratios, nb_it_combi[i], label='PCG: Green + Jacobi', color='red')
ax.set_xlabel('nb of filter applications')
ax.set_ylabel('# CG iterations')
plt.legend(['Green', 'Jacobi + Green'])
plt.show()
