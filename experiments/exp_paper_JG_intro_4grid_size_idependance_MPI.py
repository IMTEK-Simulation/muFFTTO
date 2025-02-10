from cProfile import label

import numpy as np
import scipy as sc
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpi4py import MPI
from NuMPI.Tools import Reduction

import matplotlib.pyplot as plt
from IPython.terminal.shortcuts.filters import KEYBINDING_FILTERS
from PySide2.examples.opengl.contextinfo import colors
from matplotlib.animation import FuncAnimation, PillowWriter
from sympy.physics.quantum.sho1d import omega

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library
from mpl_toolkits import mplot3d

plt.ion()
if MPI.COMM_WORLD.rank == 0:
    print('  Rank   Size          Domain       Subdomain        Location')
    print('  ----   ----          ------       ---------        --------')
MPI.COMM_WORLD.Barrier()  # Barrier so header is printed first

src = './figures/'

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'

domain_size = [1, 1]
nb_pix_multips = [2, 3, 4, 5, 6, 7, 8, 9, 10]  # ,6,7,8,9,10,]  # ,2,3,3,2,  #,5,6,7,8,9 ,5,6,7,8,9,10,11
small = np.arange(0., .1, 0.005)
middle = np.arange(0.1, 0.9, 0.03)

large = np.arange(0.9, 1.0 + 0.005, 0.005)
ratios = np.concatenate((small, middle, large))
ratios = np.arange(0., 1.1, 0.2)
ratios = np.arange(0., 1.1, 0.2)
ratios = np.array([1, 4])  # np.arange(1,5)  # 17  33

nb_it = np.zeros((len(nb_pix_multips), len(nb_pix_multips), ratios.size), )
nb_it_combi = np.zeros((len(nb_pix_multips), len(nb_pix_multips), ratios.size), )
nb_it_Jacobi = np.zeros((len(nb_pix_multips), len(nb_pix_multips), ratios.size), )
nb_it_Richardson = np.zeros((len(nb_pix_multips), len(nb_pix_multips), ratios.size), )
nb_it_Richardson_combi = np.zeros((len(nb_pix_multips), len(nb_pix_multips), ratios.size), )

norm_rr_combi = []
norm_rz_combi = []
norm_rr_Jacobi = []
norm_rz_Jacobi = []
norm_rr = []
norm_rz = []
norm_energy_lb = []
norm_energy_lb_combi = []
norm_rMr_combi = []
norm_rMr = []
kontrast = []
kontrast_2 = []
eigen_LB = []

for nb_starting_phases in np.arange(np.size(nb_pix_multips)):
    # valid_nb_muiltips=nb_pix_multips[nb_pixels:]
    print(f'nb_starting_phases = {nb_starting_phases}')

    for kk in np.arange(np.size(nb_pix_multips[nb_starting_phases:])):
        nb_pix_multip = nb_pix_multips[nb_starting_phases:][kk]
        print(f'kk = {kk}')
        print(f'nb_pix_multip = {nb_pix_multip}')
        # system set up
        number_of_pixels = (2 ** nb_pix_multip, 2 ** nb_pix_multip)

        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)

        discretization = domain.Discretization(cell=my_cell,
                                               nb_of_pixels_global=number_of_pixels,
                                               discretization_type=discretization_type,
                                               element_type=element_type)
        if MPI.COMM_WORLD.rank == 0:
            print('  Rank   Size          Domain       Subdomain        Location')
            print('  ----   ----          ------       ---------        --------')
        MPI.COMM_WORLD.Barrier()  # Barrier so header is printed first

        print(f'{MPI.COMM_WORLD.rank:6} {MPI.COMM_WORLD.size:6} {str(discretization.fft.nb_domain_grid_pts):>15} '
              f'{str(discretization.fft.nb_subdomain_grid_pts):>15} {str(discretization.fft.subdomain_locations):>15}')
        start_time = time.time()

        # set macroscopic gradient
        macro_gradient = np.array([[1.0, 0], [0, 1.0]])

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

        C_1 = domain.compute_Voigt_notation_4order(elastic_C_1)

        material_data_field_C_0 = np.einsum('ijkl,qxy->ijklqxy', elastic_C_1,
                                            np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                              *discretization.nb_of_pixels])))

        refmaterial_data_field_I4s = np.einsum('ijkl,qxy->ijklqxy', I4s,
                                               np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                                 *discretization.nb_of_pixels])))

        print('elastic tangent = \n {}'.format(domain.compute_Voigt_notation_4order(elastic_C_1)))

        # material distribution
        geometry_ID = 'sine_wave_'  # laminate2 #abs_val 'square_inclusion'#'circle_inclusion'#random_distribution  sine_wave_


        def scale_field(field, min_val, max_val):
            """Scales a 2D random field to be within [min_val, max_val]."""
            field_min, field_max = Reduction(MPI.COMM_WORLD).min(field), Reduction(MPI.COMM_WORLD).max(field)
            scaled_field = (field - field_min) / (field_max - field_min)  # Normalize to [0,1]
            return scaled_field * (max_val - min_val) + min_val  # Scale to [min_val, max_val]


        for i in np.arange(ratios.size):
            ratio = ratios[i]

            if kk == 0:
                phase_fied_small_grid = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                                            microstructure_name=geometry_ID,
                                                                            coordinates=discretization.fft.coords,
                                                                            seed=1,
                                                                            parameter=number_of_pixels[0],
                                                                            contrast=1 / 10 ** ratio)
                phase_fied_small_grid += 1 / 10 ** ratio

                # phase_fied_small_grid=np.copy(phase_field_smooth)
                phase_field_smooth = np.copy(phase_fied_small_grid)
            if kk > 0:
                # phase_field_smooth = sc.ndimage.zoom(phase_fied_small_grid, zoom=nb_pix_multip, order=0)
                phase_field_smooth = np.repeat(phase_fied_small_grid, 2 ** (kk), axis=0)
                phase_field_smooth = np.repeat(phase_field_smooth, 2 ** (kk), axis=1)

            # phase_field_smooth = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
            #                                                             microstructure_name=geometry_ID,
            #                                                             coordinates=discretization.fft.coords,
            #                                                             seed=1)

            # print(i + 2)
            # print(f'parametr = {i + 2}')
            # phase_field_smooth = np.abs(phase_field_smooth)
            # phase_field_smooth_ref = np.copy(phase_field_smooth)

            # phase_field_smooth[phase_field_smooth_ref<=0.6]=2
            # phase_field_smooth[phase_field_smooth_ref <  0.6]=0.1
            # phase_field_smooth[phase_field_smooth_ref >= 0.4] = 1

            # phase_field = np.random.rand(*discretization.get_scalar_sized_field().shape)  # set random distribution#

            # phase = 1 * np.ones(number_of_pixels)
            inc_contrast = 0.

            # nb_it=[]
            # nb_it_combi=[]
            # nb_it_Jacobi=[]
            phase_field = np.abs(phase_field_smooth)
            phase_field = scale_field(phase_field, min_val=1 / 10 ** ratio, max_val=1.0)

            # phase_field[phase_field<=1/10**ratio]= 0

            phase_fem = np.zeros([2, *number_of_pixels])
            phase_fnxyz = discretization.get_scalar_sized_field()
            # phase_fnxyz[0, 0, ...] = phase_field
            #
            # # np.save('geometry_jacobi.npy', np.power(phase_field_l, 2),)
            # # sc.io.savemat('geometry_jacobi.mat', {'data':  np.power(phase_field_l, 2)})
            #
            # phase_field_at_quad_poits_1qnxyz = \
            #     discretization.evaluate_field_at_quad_points(nodal_field_fnxyz=phase_fnxyz,
            #                                                  quad_field_fqnxyz=None,
            #                                                  quad_points_coords_dq=None)[0]
            #
            # phase_field_at_quad_poits_1qnxyz[0, :, 0, ...] = phase_fnxyz
            # apply material distribution
            # material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * np.power(phase_field[0, 0], 1)
            # material_data_field_C_0_rho=material_data_field_C_0[..., :, :] * phase_fem
            # material_data_field_C_0_rho +=100*material_data_field_C_0[..., :, :] * (1-phase_fem)
            # material_data_field_C_0_rho = material_data_field_C_0[..., :, :, :] * np.power(
            #     phase_field, 1)[0, :, 0, ...]
            material_data_field_C_0_rho = material_data_field_C_0[..., :, :, :] * np.power(
                phase_field, 1)
            # material_data_field_C_0_rho=phase_field_at_quad_poits_1qnxyz
            # Set up right hand side
            macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)
            # perturb=np.random.random(macro_gradient_field.shape)
            # macro_gradient_field += perturb#-np.mean(perturb)

            # Solve mechanical equilibrium constrain
            rhs = discretization.get_rhs(material_data_field_C_0_rho, macro_gradient_field)

            K_fun = lambda x: discretization.apply_system_matrix(material_data_field_C_0_rho, x,
                                                                 formulation='small_strain')

            # plotting eigenvalues
            # K = discretization.get_system_matrix(material_data_field_C_0_rho)
            # M = discretization.get_system_matrix(refmaterial_data_field_I4s)
            #
            # eig = sc.linalg.eigh(a=K, b=M, eigvals_only=True)

            min_val = Reduction(MPI.COMM_WORLD).min(phase_field)
            max_val = Reduction(MPI.COMM_WORLD).max(phase_field)

            kontrast.append(max_val / min_val)
            # eigen_LB.append(min_val)
            #
            # # kontrast_2.append(eig[-3] / eig[np.argmax(eig > 0)])
            # kontrast_2.append((max_val / min_val) / 10)

            omega = 1  # 2 / ( eig[-1]+eig[np.argmax(eig>0)])
            # ax1.loglog(sorted(eig)[1:],label=f'{i}',marker='.', linewidth=0, markersize=1)
            # ax1.set_ylim([1e-5, 1e1])
            #
            # K_diag_half = np.copy(np.diag(K))
            # K_diag_half[K_diag_half < 9.99e-16] = 0
            # K_diag_half[K_diag_half != 0] = 1/np.sqrt(K_diag_half[K_diag_half != 0])
            #
            # DKDsym = np.matmul(np.diag(K_diag_half),np.matmul(K,np.diag(K_diag_half)))
            # eig = sc.linalg.eigh(a=DKDsym, b=M, eigvals_only=True)
            #
            # ax2.loglog(sorted(eig)[1:-2], label=f'{i}',marker='.', linewidth=0, markersize=1)
            # ax2.set_ylim([1e-5, 1e1])

            # K = discretization.get_system_matrix(material_data_field=material_data_field_C_0_rho)
            # material_data_field_C_0=np.mean(material_data_field_C_0_rho,axis=(4,5,6))
            # mean_material=np.mean(material_data_field_C_0_rho,axis=(4,5,6))
            # material_data_field_C_0_ratio = np.einsum('ijkl,qxy->ijklqxy', mean_material,
            #                                     np.ones(np.array([discretization.nb_quad_points_per_pixel,
            #                                                       *discretization.nb_of_pixels])))

            preconditioner = discretization.get_preconditioner_NEW(
                reference_material_data_field_ijklqxyz=refmaterial_data_field_I4s)

            M_fun = lambda x: discretization.apply_preconditioner_NEW(preconditioner_Fourier_fnfnqks=preconditioner,
                                                                      nodal_field_fnxyz=x)

            K_diag_alg = discretization.get_preconditioner_Jacoby_fast(
                material_data_field_ijklqxyz=material_data_field_C_0_rho)

            M_fun_combi = lambda x: K_diag_alg * discretization.apply_preconditioner_NEW(
                preconditioner_Fourier_fnfnqks=preconditioner,
                nodal_field_fnxyz=K_diag_alg * x)
            # #
            M_fun_Jacobi = lambda x: K_diag_alg * K_diag_alg * x

            # K = discretization.get_system_matrix(material_data_field_C_0_rho)
            # M = discretization.get_system_matrix(refmaterial_data_field_I4s)
            #
            # eig_G, _ = sc.linalg.eig(a=K, b=M)  # , eigvals_only=True
            # eig_G = np.real(eig_G)
            # K_diag_half = np.copy(np.diag(K))
            # K_diag_half[K_diag_half < 9.99e-16] = 0
            # K_diag_half[K_diag_half != 0] = 1 / np.sqrt(K_diag_half[K_diag_half != 0])
            #
            # DKDsym = np.matmul(np.diag(K_diag_half), np.matmul(K, np.diag(K_diag_half)))
            # eig_JG, _ = sc.linalg.eig(a=DKDsym, b=M)  # , eigvals_only=True
            # eig_JG = np.real(eig_JG)
            # print(f'eig_G.min() = {eig_G[eig_G > 0].min()}')
            displacement_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(1000), toler=1e-6,
                                                    norm_type='rz')
            nb_it[kk + nb_starting_phases, nb_starting_phases, i] = (len(norms['residual_rr']))
            print('nb it  = {} '.format(len(norms['residual_rr'])))

            norm_rz.append(norms['residual_rz'])
            norm_rr.append(norms['residual_rr'])
            # norm_rMr.append(norms['data_scaled_rr'])

            # print(nb_it)
            #########
            displacement_field_combi, norms_combi = solvers.PCG(K_fun, rhs, x0=None, P=M_fun_combi, steps=int(1000),
                                                                toler=1e-6,
                                                                norm_type='data_scaled_rr',
                                                                norm_metric=M_fun
                                                                )
            nb_it_combi[kk + nb_starting_phases, nb_starting_phases, i] = (len(norms_combi['residual_rr']))
            norm_rz_combi.append(norms_combi['residual_rz'])
            norm_rr_combi.append(norms_combi['residual_rr'])
            norm_rMr_combi.append(norms_combi['data_scaled_rr'])

            #
            displacement_field_Jacobi, norms_Jacobi = solvers.PCG(K_fun, rhs, x0=None, P=M_fun_Jacobi, steps=int(1),
                                                                  toler=1e-6, norm_type='rr')
            nb_it_Jacobi[kk + nb_starting_phases, nb_starting_phases, i] = (len(norms_Jacobi['residual_rr']))
            norm_rz_Jacobi.append(norms_Jacobi['residual_rz'])
            norm_rr_Jacobi.append(norms_Jacobi['residual_rr'])
            # displacement_field_Richardson, norms_Richardson = solvers.Richardson(K_fun, rhs, x0=None, P=M_fun,
            #                                                                      omega=omega,
            #                                                                      steps=int(100),
            #                                                                      toler=1e-6)
            # nb_it_Richardson[kk+nb_starting_phases,nb_starting_phases , i] = (len(norms_Richardson['residual_rr']))
            # norm_rr_Richardson= norms_Richardson['residual_rr'][-1]
            #
            # displacement_field_Richardson_combi, norms_Richardson_combi = solvers.Richardson(K_fun, rhs, x0=None, P=M_fun_combi,
            #                                                                      omega=omega*0.4,
            #                                                                      steps=int(100),
            #                                                                      toler=1e-6)
            # nb_it_Richardson_combi[kk+nb_starting_phases ,nb_starting_phases, i] = (len(norms_Richardson_combi['residual_rr']))
            # norm_rr_Richardson_combi = norms_Richardson_combi['residual_rr'][-1]

            # print(ratio)

            # fig = plt.figure()  ######################################### PLOT convergence curves
            # # kappa = kontrast[-1]
            #
            # # print(f'convergecnce \n {convergence}')
            # # fig = plt.figure()
            # gs = fig.add_gridspec(1, 1)
            # ax_1 = fig.add_subplot(gs[0, 0])
            # ax_1.set_title(f'{nb_pix_multips[0]}', wrap=True)
            #
            # ax_1.semilogy(norm_rr[-1], label='rr PCG: Green', color='green')
            # ax_1.semilogy(norm_rr_combi[-1], label='rr PCG: Jacobi-Green', color='b')
            #
            # ax_1.semilogy(norm_rz[-1], label='rz PCG: Green', color='green', linestyle='--')
            # ax_1.semilogy(norm_rz_combi[-1], label='rz PCG: Jacobi-Green', color='b', linestyle='--')
            #
            # ax_1.semilogy(norm_rMr[-1], label='rMr PCG: Green', color='green', linestyle='-.', marker='x')
            # ax_1.semilogy(norm_rMr_combi[-1], label='rMr PCG: Jacobi-Green', color='b', linestyle='-.', marker='x')
            #
            # # x_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
            # # ax_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
            # ax_1.set_xlabel('CG iterations')
            # ax_1.set_ylabel('Norm of residua')
            # # plt.legend([r'$\kappa$ upper bound', 'Green', 'Jacobi', 'Green + Jacobi', 'Richardson'])
            # plt.legend()
            # ax_1.set_ylim([1e-14, 1e2])  # norm_rz[i][0]]/lb)
            # print(max(map(len, norm_rr)))
            # # ax_1.set_xlim([0, max(map(len, norm_rr))])
            #
            # plt.show()

            # fig = plt.figure()

            #
            # fig, axs = plt.subplots(nrows=2, ncols=2,
            #                         figsize=(6, 6)  )
            #         fig = plt.figure()
            #         gs = fig.add_gridspec(2, 3)
            #
            #         ax1 = fig.add_subplot(gs[1, :])
            #         # axs[0] = plt.axes(xlim=(0, nb_tiles * N), ylim=(0, nb_tiles * N))
            #         # ax1.imshow(phase_field, cmap=mpl.cm.Greys, vmin=1e-4, vmax=1)
            #
            #         ax1.step(np.arange(phase_field[:, phase_field.shape[0] // 2].size), phase_field[:, phase_field.shape[0] // 2],
            #                  linewidth=0)
            #         # ax3.plot(phase_field[:,phase_field.shape[0]//2], linewidth=0)
            #         ax1.set_ylim([1e-4, 1])
            #
            #         x = np.arange(0, 1 * number_of_pixels[0])
            #         y = np.arange(0, 1 * number_of_pixels[1])
            #         X, Y = np.meshgrid(x, y)
            #         linestyles = ['-', '--', ':']
            #         colors = ['red', 'blue', 'green', 'orange', 'purple']
            #
            #         counter = 0
            #         for i in np.array([0, ratios.size // 2, ratios.size - 1]):
            #
            #             ratio = ratios[i]
            #
            #             phase_field = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
            #                                                               microstructure_name=geometry_ID,
            #                                                               coordinates=discretization.fft.coords,
            #                                                               seed=1
            #                                                               )
            #
            #             phase_field = scale_field(phase_field, min_val=1 / 10 ** ratio, max_val=1.0)
            #             phase_field[phase_field<=1/10**ratio]= 0
            #
            #             ax0 = fig.add_subplot(gs[0, counter])
            #
            #             ax0.pcolormesh(X, Y, np.transpose(phase_field), cmap=mpl.cm.Greys, vmin=1e-4, vmax=1, linewidth=0,
            #                            rasterized=True)
            #             ax0.set_xticks(np.arange(-.5, number_of_pixels[0], int(number_of_pixels[0] / 4)))
            #             ax0.set_yticks(np.arange(-.5, number_of_pixels[1], int(number_of_pixels[1] / 4)))
            #             ax0.set_xticklabels(np.arange(0, number_of_pixels[0] + 1, int(number_of_pixels[0] / 4)))
            #             ax0.set_yticklabels(np.arange(0, number_of_pixels[1] + 1, int(number_of_pixels[1] / 4)))
            #             # ax0.set_title(f'{ratios[i]} phases')
            #             ax0.hlines(y=number_of_pixels[1] // 2, xmin=-0.5, xmax=number_of_pixels[0] - 0.5, color=colors[counter],
            #                        linestyle=linestyles[counter], linewidth=1.)
            #             if counter == 0:
            #                 ax0.set_ylabel('y coordinate')
            #                 ax0.set_xlabel('x coordinate')
            #             # ax0.hlines(y=1, xmin=0, xmax=number_of_pixels[0], colors='black', linestyles='--', linewidth=1.)
            #             # phase_field = np.abs(phase_field)  # -1
            #             # phase_field += 1e-4
            #             # min_val = np.min(phase_field)
            #             # max_val = np.max(phase_field)
            #             # phase_field = 9.99e-1 + (phase_field - min_val) * (1 - 9.99e-1) / (max_val - min_val)
            #             # phase_field = ratio * phase_field_smooth + (1 - ratio) * phase_field_pwconst
            #
            #             # ax1.clear()
            #             # ax1.imshow(np.transpose( phase_field), cmap=mpl.cm.Greys, vmin=1e-4, vmax=1)
            #             # ax1.set_title(r'Density $\rho$', wrap=True)
            #
            #             # #: {np.max(phase_field)/np.min(phase_field):.1e}  \n'                          f'  min = {np.min(phase_field):.1e}
            #             # ax3.clear()
            #
            #             extended_x = np.arange(phase_field[:, phase_field.shape[0] // 2].size + 1)
            #             extended_y = np.append(phase_field[:, phase_field.shape[0] // 2],
            #                                    phase_field[:, phase_field.shape[0] // 2][-1])
            #             ax1.step(extended_x, extended_y
            #                      , where='post',
            #                      linewidth=1, color=colors[counter], linestyle=linestyles[counter], marker='|',
            #                      label=r'phase contrast -' + f'1e{ratios[i]} ')
            #             # ax3.plot(phase_field[:, phase_field.shape[0] // 2], linewidth=1)
            #             ax1.set_ylim([0.000009, 1.1])
            #             ax1.set_xlim([0, phase_field.shape[0]])
            #             ax1.set_yticks([0.001, 0.25, 0.50, 0.75, 1.0001])
            #             ax1.set_yticklabels([0.001, 0.25, 0.50, 0.75, 1.00])
            #             ax1.set_yscale('log')  # ax1.yaxis.set_ticks_position([0.001,0.25,0.5,0.75, 1])
            #             # ax2.legend(['2 phases', f'{ratio} phases', 'Jacobi', 'Green + Jacobi'])
            #             ax1.legend(loc="lower right")
            #
            #             ax1.set_title(f'Cross sections')
            #             ax1.set_ylabel('Young modulus (Pa)')
            #             ax1.set_xlabel('x coordinate')
            #
            #             # ax2.plot(ratios[0:i + 1], nb_it[0, 0:i + 1], 'g', marker='|', label=' Green', linewidth=1)
            #             # axs[1].plot(xopt2.f.num_iteration_.transpose()[1:3*i+1:3],"r", label='DGO ',linewidth=1)
            #             # axs[1].plot(xopt2.f.num_iteration_.transpose()[2:3*i+2:3],"r", label='DGO ',linewidth=1)
            #
            #             # ax2.plot(ratios[0:i + 1], nb_it_Jacobi[0, 0:i + 1], "b", marker='o', label='PCG Jacobi', linewidth=1)
            #             # ax2.plot(ratios[0:i + 1], nb_it_combi[0, 0:i + 1], "k", marker='x', label='PCG Green + Jacobi', linewidth=1)
            #             #  ax2.semilogy(ratios[0:i + 1], nb_it_Richardson[0, 0:i + 1], "g", label=' Richardson Green ', linewidth=1)
            #             #  ax2.semilogy(ratios[0:i + 1], nb_it_Richardson_combi[0, 0:i + 1], "y",  label=' Richardson Green + Jacobi ', linewidth=1)
            #             # ax2.set_ylim(bottom=0)
            #             # axs[1].legend()
            #             # ax2.legend(['', 'Green', 'Jacobi', 'Green + Jacobi'])
            #             counter += 1
            #         plt.tight_layout()
            #         fname = src + 'introduction_geometry_exp3_sine_{}{}'.format(number_of_pixels[0], '.pdf')
            #         print(('create figure: {}'.format(fname)))
            #         #plt.savefig(fname, bbox_inches='tight')
            #     plt.show()

            print(f'ratios = {ratios}')
            print(f'nb_pix_multips = {nb_pix_multips}')

if MPI.COMM_WORLD.rank == 0:
    print('  Rank   Size          Domain       Subdomain        Location')
    print('  ----   ----          ------       ---------        --------')
    # Barrier so header is printed first

    print(f'ratios = {ratios}')
    print(f'nb_pix_multips = {nb_pix_multips}')
    for i in np.arange(ratios.size):
        ratio = ratios[i]
        print(f'ratio= {ratio}')

        print('greeen')

        print(nb_it[:, :, i])
        # print('jacobi')
        # print(nb_it_Jacobi[:,:,i])
        print('combi')
        print(nb_it_combi[:, :, i])

MPI.COMM_WORLD.Barrier()
quit()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each line with a different z offset
Nx = (2 ** np.asarray(nb_pix_multips)) ** 2
for i in np.arange(len(nb_pix_multips)):
    ax.plot(Nx, Nx[i], zs=nb_it[:, i, 0], label='PCG: Green', color='blue')
    ax.plot(Nx, Nx[i], zs=nb_it_Jacobi[:, i, 0], label='PCG: Jacobi', color='black')
    ax.plot(Nx, Nx[i], zs=nb_it_combi[:, i, 0], label='PCG: Green + Jacobi', color='red')
# ax.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
# ax.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
ax.set_zlim(10, 100)
ax.set_ylabel('nb of phases')
ax.set_xlabel('Nb pixels')
ax.set_zlabel('# CG iterations')
# ax.set_xscale('log')
# ax.set_yscale('log')

plt.legend(['DGO', 'Jacobi', 'DGO + Jacobi'])
fname = src + 'introduction_exp4_GRID_aaa_{}{}'.format(number_of_pixels[0], '.pdf')
print(('create figure: {}'.format(fname)))
plt.savefig(fname, bbox_inches='tight')

plt.show()

X, Y = np.meshgrid(Nx, Nx)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Setting the view angle
ax.view_init(elev=30, azim=-100)  # Adjust these values as needed
# Plotting the surface
ax.plot_wireframe(X, Y, nb_it[:, :, 0], label='PCG: Green', color='green')
ax.plot_wireframe(X, Y, nb_it_Jacobi[:, :, 0], label='PCG: Jacobi', color='black')
ax.plot_wireframe(X, Y, nb_it_combi[:, :, 0], label='PCG: Green + Jacobi', color='red')

ax.set_ylabel('nb of phases')
ax.set_xlabel('Nb pixels')
ax.set_zlabel('# CG iterations')
plt.legend(['DGO', 'Jacobi', 'DGO + Jacobi'])
plt.show()

#
# fig = plt.figure()
# gs = fig.add_gridspec(1, 1)
# ax  = fig.add_subplot(gs[0, 0])
# # Plot each line with a different z offset
# #for i in np.arange(len(nb_pix_multips)):
# Nx=2 ** np.asarray( nb_pix_multips)
# ax.plot( Nx, nb_it[:,nb_starting_phases,0], label='PCG: Green', color='blue')
# ax.plot( Nx, nb_it_Jacobi[:,nb_starting_phases,0], label='PCG: Jacobi', color='black')
# ax.plot( Nx, nb_it_combi[:,nb_starting_phases,0], label='PCG: Green + Jacobi', color='red')
# # ax.plot( np.asarray( nb_pix_multips) * 16, nb_it_Richardson[:,0], label='Richardson Green', color='green')
# # ax.plot( np.asarray( nb_pix_multips) * 16, nb_it_Richardson_combi[:,0], label='Richardson Green+Jacobi')
# #ax.set_xlabel('nb of filter aplications')
# ax.set_xlabel('Grid size -Nx')
# ax.set_ylabel('# PCG iterations')
# plt.legend(['PCG: Green', 'PCG: Jacobi', 'PCG: Green + Jacobi', 'Richardson Green','Richardson Green+Jacobi'])
# plt.show()
#
#
# fname = src + 'introduction_exp4_GRID_DEP_{}{}'.format(number_of_pixels[0], '.pdf')
# print(('create figure: {}'.format(fname)))
# plt.savefig(fname, bbox_inches='tight')
#
#
#
# plt.show()
#
#
#
# for i in np.arange(ratios.size, step=1):
#     kappa = kontrast[i]
#     k = np.arange(max(map(len, norm_rr)))
#     print(f'k \n {k}')
#     lb = eigen_LB[i]
#     print(f'lb \n {lb}')
#
#     convergence = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** k
#     convergence = convergence * norm_rr[i][0]
#
#     #print(f'convergecnce \n {convergence}')
#     fig = plt.figure()
#     gs = fig.add_gridspec(1, 1)
#     ax_1 = fig.add_subplot(gs[0, 0])
#     ax_1.set_title(f'{i}', wrap=True)
#     ax_1.semilogy(convergence, '--', label='estim', color='k')
#
#     ax_1.semilogy(norm_rr[i], label='PCG: Green', color='r')
#     ax_1.semilogy(norm_rr_Jacobi[i], label='PCG: Jacobi', color='b')
#
#     # x_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
#     # ax_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
#     ax_1.set_xlabel('CG iterations')
#     ax_1.set_ylabel('Norm of residua')
#     plt.legend([r'$\kappa$ upper bound', 'Green', 'Jacobi', 'Green + Jacobi', 'Richardson'])
#     ax_1.set_ylim([1e-10, norm_rr[i][0]])  # norm_rz[i][0]]/lb)
#     print(max(map(len, norm_rr)))
#     ax_1.set_xlim([0, max(map(len, norm_rr))])
#
#     plt.show()
#
# plt.show()
# fig = plt.figure()
# gs = fig.add_gridspec(1, 1)
# ax_1 = fig.add_subplot(gs[0, 0])
# ax_1.semilogy(norm_rr[0], label='PCG: Green', color='blue', linewidth=0)
#
# # ax_1.set_ylim([1e-7, 1e0])
# ax_1.set_ylim([1e-7, norm_rr[0][0]])  # norm_rz[i][0]]/lb)
#
# print(max(map(len, norm_rz)))
# ax_1.set_xlim([0, max(map(len, norm_rr))])
#
#
# def convergence_gif_rz(i):
#     kappa = kontrast[i]
#     k = np.arange(max(map(len, norm_rr)))
#     #print(f'k \n {k}')
#     lb = eigen_LB[i]
#     #print(f'lb \n {lb}')
#
#     convergence = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** k
#     convergence = convergence * norm_rr[i][0]
#
#     ax_1.clear()
#
#     ax_1.set_title(f'{i}', wrap=True)
#     ax_1.semilogy(convergence, '--', label='estim', color='g')
#
#     ax_1.semilogy(norm_rr[i], label='PCG: Green', color='g')
#     ax_1.semilogy(norm_rr_Jacobi[i], label='PCG: Jacobi', color='b')
#     ax_1.semilogy(norm_rr_combi[i], label='PCG: Jacobi-Green', color='k')
#
#     # x_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
#     # ax_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
#     ax_1.set_xlabel('CG iterations')
#     ax_1.set_ylabel('Norm of residua')
#     plt.legend([r'$\kappa$ upper bound', 'Green', 'Jacobi', 'DGO + Jacobi', 'Richardson'])
#     ax_1.set_ylim([1e-7, norm_rr[i][0]])  # norm_rz[i][0]]/lb)
#     print(max(map(len, norm_rr)))
#     ax_1.set_xlim([0, max(map(len, norm_rr))])
#     # axs[1].legend()
#     # plt.legend(['', 'FEM: Green', 'FEM: Jacobi', 'FEM: Green + Jacobi','FEM: Richardson'])
#     plt.legend(
#         [r'$\kappa$ upper bound', 'Green', 'Jacobi', 'Green + Jacobi', 'Richardson Green', 'Richardson Green + Jacobi'],
#         loc='center left', bbox_to_anchor=(0.8, 0.5))
#
#
# ani = FuncAnimation(fig, convergence_gif_rz, frames=ratios.size, blit=False)
# # axs[1].legend()middlemiddle
# # Save as a GIF
# ani.save(
#     f"./figures/convergence_exp3_squares_{number_of_pixels[0]}comparison{ratios[-1]}_RichardsonJacobi{geometry_ID}_circle_inc_to_smooth_semiloplots3.gif",
#     writer=PillowWriter(fps=1))
#
# plt.show()
# # -------------------------------------------------------------------------------------------------------
# # for i in np.arange(ratios.size,step=1):
# #     kappa=kontrast[i]
# #     kappa_2=kontrast_2[i]
# #     k=np.arange(len(norm_rr_Jacobi[i]))
# #     print(f'k \n {k}')
# #
# #     convergence=((np.sqrt(kappa)-1)/(np.sqrt(kappa)+1))**k
# #     convergence=convergence*norm_rr[i][0]
# #     convergence2 = ((np.sqrt(kappa_2) - 1) / (np.sqrt(kappa_2) + 1)) ** k
# #     convergence2 = convergence2 * norm_rr[i][0]
# #
# #
# #     print(f'convergecnce \n {convergence}')
# #     fig = plt.figure()
# #     gs = fig.add_gridspec(1, 1)
# #     ax_1 = fig.add_subplot(gs[0, 0])
# #     ax_1.set_title(f'{i}', wrap=True)
# #     ax_1.semilogy(convergence, '-',label='estim', color='green')
# #     #ax_1.semilogy(convergence2,'.-', label='estim2', color='green')
# #
# #     ax_1.semilogy(norm_rr[i], label='PCG: Green', color='blue')
# #     ax_1.semilogy(norm_rr_Jacobi[i], label='PCG: Jacobi', color='black')
# #     #ax_1.semilogy(norm_rr_combi[i], label='PCG: Green + Jacobi', color='red')
# #     #x_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
# #     #ax_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
# #     ax_1.set_xlabel('CG iterations')
# #     ax_1.set_ylabel('Norm of residuals')
# #     plt.legend([r'$\kappa$ upper bound','Green', 'Jacobi', 'Green + Jacobi','Richardson'])
# #     ax_1.set_ylim([1e-7, norm_rr[i][0]])
# #     print(max(map(len, norm_rr)))
# #     ax_1.set_xlim([0, max(map(len, norm_rr))])
# #
# #     plt.show()
#
# fig = plt.figure()
# gs = fig.add_gridspec(1, 1)
# ax_1 = fig.add_subplot(gs[0, 0])
# ax_1.semilogy(norm_rr[0], label='PCG: Green', color='blue', linewidth=0)
#
# ax_1.set_ylim([1e-7, 1e0])
# print(max(map(len, norm_rr)))
# ax_1.set_xlim([0, max(map(len, norm_rr))])
#
#
# def convergence_gif(i):
#     kappa = kontrast[i]
#     k = np.arange(max(map(len, norm_rr)))
#     #print(f'k \n {k}')
#
#     convergence = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** k
#     convergence = convergence * norm_rr[i][0]
#     #print(f'convergecnce \n {convergence}')
#     ax_1.clear()
#
#     ax_1.set_title(f'{i}', wrap=True)
#     ax_1.semilogy(convergence, '--', label='estim', color='k')
#
#     ax_1.semilogy(norm_rr[i], label='PCG: Green', color='r')
#     ax_1.semilogy(norm_rr_Jacobi[i], label='PCG: Jacobi', color='b')
#     # ax_1.semilogy(norm_rr_combi[i], label='PCG: Green + Jacobi', color='red')
#     # x_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
#     # ax_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
#     ax_1.set_xlabel('CG iterations')
#     ax_1.set_ylabel('Norm of residuals')
#     plt.legend([r'$\kappa$ upper bound', 'Green', 'Jacobi', 'Green + Jacobi', 'Richardson'])
#     ax_1.set_ylim([1e-7, 1e0])
#    # print(max(map(len, norm_rr)))
#     ax_1.set_xlim([0, max(map(len, norm_rr))])
#     # axs[1].legend()
#     # plt.legend(['', 'FEM: Green', 'FEM: Jacobi', 'FEM: Green + Jacobi','FEM: Richardson'])
#     plt.legend(
#         [r'$\kappa$ upper bound', 'Green', 'Jacobi', 'Green + Jacobi', 'Richardson Green', 'Richardson Green + Jacobi'],
#         loc='center left', bbox_to_anchor=(0.8, 0.5))
#
#
# ani = FuncAnimation(fig, convergence_gif, frames=ratios.size, blit=False)
# # axs[1].legend()middlemiddle
# # Save as a GIF
# ani.save(
#     f"./figures/convergence_gif_{number_of_pixels[0]}comparison{ratios[-1]}_RichardsonJacobi{geometry_ID}_circle_inc_to_smooth_exp3_squares.gif",
#     writer=PillowWriter(fps=1))
#
# plt.show()
#
# plot_evolion = True
# if plot_evolion:
#     for nb_tiles in [1, ]:
#         fig = plt.figure(figsize=(11, 4.5))
#         gs = fig.add_gridspec(1, 1)
#         # ax1 = fig.add_subplot(gs[0, 0])
#         ax2 = fig.add_subplot(gs[0, 0])
#         # ax1.step(np.arange(phase_field[:, phase_field.shape[0] // 2].size), phase_field[:, phase_field.shape[0] // 2],
#         #         linewidth=0)
#         # ax1.set_ylim([1e-4, 1])
#         ax2.plot(ratios, nb_it_Jacobi[0], label='nb_it_Laplace', linewidth=0)
#         ax2.set_ylabel('# PCG iterations')
#         ax2.set_xlabel('# material phases')
#
#         counter = 0
#         #for i in np.array([0, ratios.size // 4 - 1, ratios.size - 1]):
#         #ratio = ratios[i]
#
#         # phase_field = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
#         #                                                   microstructure_name=geometry_ID,
#         #                                                   coordinates=discretization.fft.coords,
#         #                                                   parameter=ratio,
#         #                                                   contrast=1e-4
#         #                                                   )
#         #
#         # linestyles = ['-', '--', ':']
#         # extended_x = np.arange(phase_field[:, phase_field.shape[0] // 2].size + 1)
#         # extended_y = np.append(phase_field[:, phase_field.shape[0] // 2],
#         #                        phase_field[:, phase_field.shape[0] // 2][-1])
#         # ax1.step(extended_x,extended_y
#         #          ,where='post',
#         #          linewidth=1, color='k', linestyle=linestyles[counter], marker='|',
#         #          label=f'{ratios[i]} phases')
#         # ax1.set_ylim([-0.1, 1.1])
#         # ax1.set_xlim([0, phase_field.shape[0]])
#         # ax1.set_yticks([0.001, 0.25, 0.50, 0.75, 1.0001])
#         # ax1.set_yticklabels([0.001, 0.25, 0.50, 0.75, 1.00])
#         # ax1.legend(loc="upper left")
#         #
#         # ax1.set_title(f'Cross sections')
#         # ax1.set_ylabel('Young modulus (Pa)')
#         # ax1.set_xlabel('Nodal point index')
#
#         ax2.plot(ratios, nb_it[0], 'g', marker='o', label=' Green', linewidth=1, markerfacecolor='white')
#         ax2.plot(ratios, nb_it_Jacobi[0], "b", marker='^', label='PCG Jacobi', linewidth=1, markerfacecolor='white')
#         ax2.plot(ratios, nb_it_combi[0], "k", marker='x', label='PCG Green + Jacobi', linewidth=1, markerfacecolor='white')
#
#         ax2.set_ylim(bottom=1)
#         ax2.set_xlim([2, ratios.size+1  ])
#         ax2.set_xticks(np.concatenate(([2,],np.arange(5, ratios.size+1, 5),[ratios.size+1,])))
#         ax2.set_xticklabels(np.concatenate(([2,],np.arange(5, ratios.size+1, 5),[ratios.size+1,])))
#         ax2.legend(['', 'Green', 'Jacobi', 'Green + Jacobi'])
#
#         counter += 1
#         plt.tight_layout()
#     fname = src + 'introduction_exp3_squares_{}{}'.format(number_of_pixels[0], '.pdf')
#     print(('create figure: {}'.format(fname)))
#     plt.savefig(fname, bbox_inches='tight')
#     plt.show()
#
#     for nb_tiles in [1, ]:
#         # fig = plt.figure()
#
#         #
#         # fig, axs = plt.subplots(nrows=2, ncols=2,
#         #                         figsize=(6, 6)  )
#         fig = plt.figure()
#         gs = fig.add_gridspec(2, 3)
#
#         ax1 = fig.add_subplot(gs[1, :])
#         # axs[0] = plt.axes(xlim=(0, nb_tiles * N), ylim=(0, nb_tiles * N))
#         # ax1.imshow(phase_field, cmap=mpl.cm.Greys, vmin=1e-4, vmax=1)
#
#         ax1.step(np.arange(phase_field[:, phase_field.shape[0] // 2].size), phase_field[:, phase_field.shape[0] // 2],
#                  linewidth=0)
#         # ax3.plot(phase_field[:,phase_field.shape[0]//2], linewidth=0)
#         ax1.set_ylim([1e-4, 1])
#         # print(ratios)
#
#         # print(nb_it)
#         ax2.plot(ratios, nb_it_Jacobi[0], label='nb_it_Laplace', linewidth=0)
#         # ax1.set_ylim([1e0, 1e3 ])
#
#         # axs[1].plot(xopt.f.num_iteration_.transpose()[::3], 'w'  , linewidth=0)
#         # axs[1].plot(xopt3.f.num_iteration_.transpose(), "b", label='Jacoby', linewidth=0)
#         # axs[1].plot(xopt.f.num_iteration_.transpose(), "k", label='DGO + Jacoby', linewidth=0)
#         # legend = plt.legend()
#         # Animation function to update the image
#         # ax2.set_xlabel('')
#         ax2.set_ylabel('# PCG iterations')
#         ax2.set_xlabel('# material phases')
#
#         linestyles = ['-', '--', ':']
#         colors = ['red', 'blue', 'green', 'orange', 'purple']
#
#         counter = 0
#         for k in  np.array([0, nb_pix_multips.size //2 , nb_pix_multips.size - 1]):
#
#             mult = nb_pix_multips[k]
#             for i in np.array([0 ]):
#                 ratio = ratios[0]
#                 number_of_pixels = (mult * 16, mult * 16)
#                 x = np.arange(0, nb_tiles * number_of_pixels[0])
#                 y = np.arange(0, nb_tiles * number_of_pixels[1])
#                 X, Y = np.meshgrid(x, y)
#
#                 #phase_field = sc.ndimage.zoom(phase_fied_small_grid, zoom=mult, order=0)
#                 phase_field = np.repeat(phase_fied_small_grid, mult, axis=0)
#                 phase_field = np.repeat(phase_field, mult, axis=1)
#                 phase_field_ref = np.copy(phase_field)
#                 #phase_field[phase_field_ref < 0.6] = 0.1
#                 #phase_field[phase_field_ref >= 0.4] = 1
#
#                 phase_field = scale_field(phase_field, min_val=1 / 10 ** ratio, max_val=1.0)
#                 ax0 = fig.add_subplot(gs[0, counter])
#
#                 ax0.pcolormesh(X, Y, np.transpose(phase_field), cmap=mpl.cm.Greys, vmin=1e-4, vmax=1, linewidth=0,
#                                rasterized=True)
#                 ax0.set_xticks(np.arange(-.5, number_of_pixels[0], int(number_of_pixels[0] / 4)))
#                 ax0.set_yticks(np.arange(-.5, number_of_pixels[1], int(number_of_pixels[1] / 4)))
#                 ax0.set_xticklabels(np.arange(0, number_of_pixels[0] + 1, int(number_of_pixels[0] / 4)))
#                 ax0.set_yticklabels(np.arange(0, number_of_pixels[1] + 1, int(number_of_pixels[1] / 4)))
#                 #ax0.set_title(f'{ratios[i]} phases')
#                 ax0.hlines(y=number_of_pixels[1] // 2 , xmin=-0.5, xmax=number_of_pixels[0] - 0.5, color=colors[counter],
#                            linestyle=linestyles[counter], linewidth=1.)
#                 if counter == 0:
#                     ax0.set_ylabel('pixel coordinates')
#                     ax0.set_xlabel('pixel coordinates')
#                 # ax0.hlines(y=1, xmin=0, xmax=number_of_pixels[0], colors='black', linestyles='--', linewidth=1.)
#                 # phase_field = np.abs(phase_field)  # -1
#                 # phase_field += 1e-4
#                 # min_val = np.min(phase_field)
#                 # max_val = np.max(phase_field)
#                 # phase_field = 9.99e-1 + (phase_field - min_val) * (1 - 9.99e-1) / (max_val - min_val)
#                 # phase_field = ratio * phase_field_smooth + (1 - ratio) * phase_field_pwconst
#
#                 # ax1.clear()
#                 # ax1.imshow(np.transpose( phase_field), cmap=mpl.cm.Greys, vmin=1e-4, vmax=1)
#                 # ax1.set_title(r'Density $\rho$', wrap=True)
#
#                 # #: {np.max(phase_field)/np.min(phase_field):.1e}  \n'                          f'  min = {np.min(phase_field):.1e}
#                 # ax3.clear()
#
#                 extended_x = np.linspace(0,1,phase_field[:, phase_field.shape[0] // 2].size + 1)
#                 extended_y = np.append(phase_field[:, phase_field.shape[0] // 2],
#                                        phase_field[:, phase_field.shape[0] // 2][-1])
#                 ax1.step(extended_x, extended_y
#                          , where='post',
#                          linewidth=1, color=colors[counter], linestyle=linestyles[counter], marker='|',
#                          label=r'phase contrast -' + f'1e{ratios[i]} ')
#                 # ax3.plot(phase_field[:, phase_field.shape[0] // 2], linewidth=1)
#                 ax1.set_ylim([0.000009, 1.1])
#                 ax1.set_xlim([0, 1])
#                 ax1.set_yticks([0.001, 0.25, 0.50, 0.75, 1.0001])
#                 ax1.set_yticklabels([0.001, 0.25, 0.50, 0.75, 1.00])
#                 ax1.set_yscale('log')            # ax1.yaxis.set_ticks_position([0.001,0.25,0.5,0.75, 1])
#                 # ax2.legend(['2 phases', f'{ratio} phases', 'Jacobi', 'Green + Jacobi'])
#                 ax1.legend(loc="lower center")
#
#                 ax1.set_title(f'Cross sections')
#                 ax1.set_ylabel('Young modulus (Pa)')
#                 ax1.set_xlabel('x coordinate')
#
#                 # ax2.plot(ratios[0:i + 1], nb_it[0, 0:i + 1], 'g', marker='|', label=' Green', linewidth=1)
#                 # axs[1].plot(xopt2.f.num_iteration_.transpose()[1:3*i+1:3],"r", label='DGO ',linewidth=1)
#                 # axs[1].plot(xopt2.f.num_iteration_.transpose()[2:3*i+2:3],"r", label='DGO ',linewidth=1)
#
#                 # ax2.plot(ratios[0:i + 1], nb_it_Jacobi[0, 0:i + 1], "b", marker='o', label='PCG Jacobi', linewidth=1)
#                 # ax2.plot(ratios[0:i + 1], nb_it_combi[0, 0:i + 1], "k", marker='x', label='PCG Green + Jacobi', linewidth=1)
#                 #  ax2.semilogy(ratios[0:i + 1], nb_it_Richardson[0, 0:i + 1], "g", label=' Richardson Green ', linewidth=1)
#                 #  ax2.semilogy(ratios[0:i + 1], nb_it_Richardson_combi[0, 0:i + 1], "y",  label=' Richardson Green + Jacobi ', linewidth=1)
#                 # ax2.set_ylim(bottom=0)
#                 # axs[1].legend()
#                 # ax2.legend(['', 'Green', 'Jacobi', 'Green + Jacobi'])
#                 counter += 1
#             plt.tight_layout()
#         fname = src + 'introduction_geometry_exp3_squares_{}{}'.format(number_of_pixels[0], '.pdf')
#         print(('create figure: {}'.format(fname)))
#         plt.savefig(fname, bbox_inches='tight')
#         plt.show()
#
# fig = plt.figure(figsize=(11, 4.5))
# gs = fig.add_gridspec(1, 1)
# ax_1 = fig.add_subplot(gs[0, 0])
# ax_1.set_title(f'Convergence', wrap=True)
# counter = 0
#
# for i in np.array([0, ratios.size //2 , ratios.size - 1]):
#     # kappa=kontrast[i]
#     # k=np.arange(max(map(len, norm_rr)))
#     # print(f'k \n {k}')
#     # lb=eigen_LB[i]
#     # print(f'lb \n {lb}')
#     #
#     # convergence=((np.sqrt(kappa)-1)/(np.sqrt(kappa)+1))**k
#     # convergence=convergence*norm_rr[i][0]
#
#     # print(f'convergecnce \n {convergence}')
#
#     # ax_1.semilogy(convergence, '--',label='estim', color='k')
#
#     ax_1.semilogy(np.arange(1,np.size(norm_rr[i])+1),norm_rr[i], label=f'Green - contrast -' + f'1e{ratios[i]}' , color='green', linestyle=linestyles[counter],
#                   marker='o',markerfacecolor='white',markevery=3)
#     ax_1.semilogy(np.arange(1,np.size(norm_rr_Jacobi[i])+1),norm_rr_Jacobi[i], label=f'Jacobi - contrast -' + f'1e{ratios[i]}', color='blue', linestyle=linestyles[counter],
#                   marker='^',markerfacecolor='white',markevery=3)
#     ax_1.semilogy(np.arange(1,np.size(norm_rr_combi[i])+1), norm_rr_combi[i], label=f'Jacobi-Green - contrast -' + f'1e{ratios[i]}', color='black', linestyle=linestyles[counter],
#                   marker='x',markerfacecolor='white',markevery=3)
#     # x_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
#     # ax_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
#     ax_1.set_xlabel('PCG iteration')
#     ax_1.set_ylabel('Norm of residua')
#     plt.legend(ncol=3,loc='upper center')
#     # ['Green - 2 phases', 'Jacobi - 2 phases', 'Green + Jacobi - 2 phases',
#     #  f'Green - {ratios[i]} phases', f'  Jacobi - {ratios[i]} phases', f'Green + Jacobi - {ratios.size // 4 - 1} phases',
#     #  f'Green - {ratios.size - 1} phases', f'Jacobi - {ratios[i]} phases', f'Green + Jacobi - {ratios.size - 1} phases']
#     ax.set_yscale('symlog')
#     #ax.set_xscale('symlog')
#     ax_1.set_ylim([1e-20, 1e1])  # norm_rz[i][0]]/lb) norm_rr[i][0]
#     #ax_1.set_ylim([0, 1e1])
#     print(max(map(len, norm_rr)))
#     #ax_1.set_xlim([1, max(map(len, norm_rr))])
#     ax_1.set_xlim([1, 165])
#
#     counter += 1
# fname = src + 'introduction_convergence_exp3_squares_{}{}'.format(number_of_pixels[0], '.pdf')
# print(('create figure: {}'.format(fname)))
# plt.savefig(fname, bbox_inches='tight')
# plt.show()
#
# plot_evolion = True
# if plot_evolion:
#     for nb_tiles in [1, ]:
#         # fig = plt.figure()
#
#         #
#         # fig, axs = plt.subplots(nrows=2, ncols=2,
#         #                         figsize=(6, 6)  )
#         fig = plt.figure()
#         gs = fig.add_gridspec(2, 2)
#         ax1 = fig.add_subplot(gs[0, 0])
#         ax3 = fig.add_subplot(gs[0, 1])
#         ax2 = fig.add_subplot(gs[1, :])
#         # axs[0] = plt.axes(xlim=(0, nb_tiles * N), ylim=(0, nb_tiles * N))
#         ax1.imshow(phase_field, cmap=mpl.cm.Greys, vmin=1e-4, vmax=1)
#         ax3.step(np.arange(phase_field[:, phase_field.shape[0] // 2].size), phase_field[:, phase_field.shape[0] // 2],
#                  linewidth=0)
#         # ax3.plot(phase_field[:,phase_field.shape[0]//2], linewidth=0)
#         ax3.set_ylim([1e-4, 1])
#         # print(ratios)
#
#         # print(nb_it)
#         ax2.plot(ratios, nb_it_Jacobi[0], label='nb_it_Laplace', linewidth=0)
#         ax3.set_ylim([1e0, 1e3])
#
#         # axs[1].plot(xopt.f.num_iteration_.transpose()[::3], 'w'  , linewidth=0)
#         # axs[1].plot(xopt3.f.num_iteration_.transpose(), "b", label='Jacoby', linewidth=0)
#         # axs[1].plot(xopt.f.num_iteration_.transpose(), "k", label='DGO + Jacoby', linewidth=0)
#         # legend = plt.legend()
#         # Animation function to update the image
#         # ax2.set_xlabel('')
#         ax2.set_ylabel('# PCG iterations')
#         ax2.set_xlabel('# material phases')
#
#
#         def update(i):
#             ratio = ratios[i]
#
#             phase_field = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
#                                                               microstructure_name=geometry_ID,
#                                                               coordinates=discretization.fft.coords,
#                                                                  seed=1
#                                                               )
#             phase_field_ref = np.copy(phase_field)
#             phase_field[phase_field_ref < 0.6] = 0.1
#             phase_field[phase_field_ref >= 0.4] = 1
#
#             phase_field = scale_field(phase_field, min_val=1 / 10 ** ratio, max_val=1.0)
#
#             # phase_field = np.abs(phase_field)  # -1
#             # phase_field += 1e-4
#             # min_val = np.min(phase_field)
#             # max_val = np.max(phase_field)
#             # phase_field = 9.99e-1 + (phase_field - min_val) * (1 - 9.99e-1) / (max_val - min_val)
#             # phase_field = ratio * phase_field_smooth + (1 - ratio) * phase_field_pwconst
#
#             ax1.clear()
#             ax1.imshow(np.transpose(phase_field), cmap=mpl.cm.Greys, vmin=1e-4, vmax=1)
#             ax1.set_title(r'Density $\rho$', wrap=True)
#             #: {np.max(phase_field)/np.min(phase_field):.1e}  \n'                          f'  min = {np.min(phase_field):.1e}
#             ax3.clear()
#             ax3.step(np.arange(phase_field[:, phase_field.shape[0] // 2].size),
#                      phase_field[:, phase_field.shape[0] // 2], linewidth=1)
#             # ax3.plot(phase_field[:, phase_field.shape[0] // 2], linewidth=1)
#             ax3.set_ylim([1e-4, 1])
#             ax3.set_title(f'Cross section')
#
#             ax2.plot(ratios[0:i + 1], nb_it[0, 0:i + 1], 'g', marker='|', label=' Green', linewidth=1)
#             # axs[1].plot(xopt2.f.num_iteration_.transpose()[1:3*i+1:3],"r", label='DGO ',linewidth=1)
#             # axs[1].plot(xopt2.f.num_iteration_.transpose()[2:3*i+2:3],"r", label='DGO ',linewidth=1)
#
#             ax2.plot(ratios[0:i + 1], nb_it_Jacobi[0, 0:i + 1], "b", marker='o', label='PCG Jacobi', linewidth=1)
#             ax2.plot(ratios[0:i + 1], nb_it_combi[0, 0:i + 1], "k", marker='x', label='PCG Jacobi-Green', linewidth=1)
#             #  ax2.semilogy(ratios[0:i + 1], nb_it_Richardson[0, 0:i + 1], "g", label=' Richardson Green ', linewidth=1)
#             #  ax2.semilogy(ratios[0:i + 1], nb_it_Richardson_combi[0, 0:i + 1], "y",  label=' Richardson Green + Jacobi ', linewidth=1)
#             ax2.set_ylim(bottom=0)
#             # axs[1].legend()
#             ax2.legend(['', 'Green', 'Jacobi', 'Jacobi-Green'])
#             # plt.legend(['', 'FEM: Green', 'FEM: Jacobi', 'FEM: Green + Jacobi','FEM: Richardson'])
#
#
#         # plt.legend(['', ' Green', 'Jacobi', 'Green + Jacobi','Richardson Green','Richardson Green + Jacobi'],loc='best', bbox_to_anchor=(0.7, 0.5))
#         #        ax2.legend([r'$\kappa$ upper bound', 'Green', 'Jacobi', 'Green + Jacobi', 'Richardson'])
#
#         # plt.legend([ '', 'Green', 'Jacobi'  ])
#
#         # img.set_array(xopt_it)
#         # ax2.legend([r'$\kappa$ upper bound', 'Green', 'Jacobi', 'Green + Jacobi', 'Richardson'])
#
#         # box = ax2.get_position()
#         # ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#         #
#         # # Put a legend to the right of the current axis
#         # ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#         # Create animation
#         # ani = FuncAnimation(fig, update, frames=xopt.f.norms_f.size - 1, blit=False)
#
#         ani = FuncAnimation(fig, update, frames=ratios.size, blit=False)
#         # axs[1].legend()middlemiddle
#         # Save as a GIF
#         ani.save(
#             f"./figures/movie_exp_paper_JG_exp3_squares_{number_of_pixels[0]}comparison{ratios[-1]}_RichardsonJacobi{geometry_ID}_circle_inc_to_smooth_semiloplots3.gif",
#             writer=PillowWriter(fps=4))
#
#     plt.show()
