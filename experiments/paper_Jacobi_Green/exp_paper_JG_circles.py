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

src = '../figures/'

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'

domain_size = [1, 1]

nbs_of_cirles = [1, 2, 3, 4,5 ]  # 1, 2,
nbs_nodal_points = [3,4,5,6,7,8,9]#5, 6, 7, 8,

ratios = np.array([2, 4])  # ,4

nb_it = np.zeros((len(nbs_nodal_points), len(nbs_of_cirles), ratios.size), )
nb_it_combi = np.zeros((len(nbs_nodal_points), len(nbs_of_cirles), ratios.size), )
nb_it_Jacobi = np.zeros((len(nbs_nodal_points), len(nbs_of_cirles), ratios.size), )
nb_it_Richardson = np.zeros((len(nbs_nodal_points), len(nbs_of_cirles), ratios.size), )
nb_it_Richardson_combi = np.zeros((len(nbs_nodal_points), len(nbs_of_cirles), ratios.size), )

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
norm_rMr_Jacobi = []
kontrast = []
kontrast_2 = []
eigen_LB = []

for i_circles in np.arange(np.size(nbs_of_cirles)):
    # valid_nb_muiltips=nb_pix_multips[nb_pixels:]
    nb_of_circles = nbs_of_cirles[i_circles]
    # print(f'nb_of_cirles = {nb_of_cirles}')
    print(f'nb_circles =   {nb_of_circles}')
    for j_bn_nodal_points in np.arange(np.size(nbs_nodal_points)):
        nb_pix_multip = nbs_nodal_points[j_bn_nodal_points]
        # print(f'j_bn_nodal_points = {j_bn_nodal_points}')
        print(f'nb_pix_multip = {nb_pix_multip}')
        # system set up
        number_of_pixels = (2 ** nb_pix_multip, 2 ** nb_pix_multip)

        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)

        discretization = domain.Discretization(cell=my_cell,
                                               nb_of_pixels_global=number_of_pixels,
                                               discretization_type=discretization_type,
                                               element_type=element_type)
        # if MPI.COMM_WORLD.rank == 0:
        #     print('  Rank   Size          Domain       Subdomain        Location')
        #     print('  ----   ----          ------       ---------        --------')
        # MPI.COMM_WORLD.Barrier()  # Barrier so header is printed first
        #
        # print(f'{MPI.COMM_WORLD.rank:6} {MPI.COMM_WORLD.size:6} {str(discretization.fft.nb_domain_grid_pts):>15} '
        #       f'{str(discretization.fft.nb_subdomain_grid_pts):>15} {str(discretization.fft.subdomain_locations):>15}')
        start_time = time.time()

        # set macroscopic gradient
        macro_gradient = np.array([[1.0, 0.5], [0.5, 1.0]])

        # create material data field
        K_0, G_0 = (1, 0.5)  # domain.get_bulk_and_shear_modulus(E=1, poison=0.2)

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

        material_data_field_C_0 = np.einsum('ijkl,qxy->ijklqxy', elastic_C_1,
                                            np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                              *discretization.nb_of_pixels])))

        refmaterial_data_field_I4s = np.einsum('ijkl,qxy->ijklqxy', elastic_C_1,
                                               np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                                 *discretization.nb_of_pixels])))

        # print('elastic tangent = \n {}'.format(domain.compute_Voigt_notation_4order(elastic_C_1)))

        # material distribution
        geometry_ID = 'circle_inclusions'  # left_cluster_x2 sine_wave_ linear symmetric_linear laminate_log #n_laminate  laminate2  sine_wave_ #abs_val 'square_inclusion'#'circle_inclusion'#random_distribution  sine_wave_
        nb_circles = nb_of_circles  # number of circles in single dimention
        vol_frac = 1 / 4
        r_0 = np.sqrt(vol_frac / np.pi)
        _geom_parameters = {'nb_circles': nb_circles,
                            'r_0': r_0,
                            'vol_frac': vol_frac,
                            'random_density': True,
                            'random_centers': True}


        def scale_field(field, min_val, max_val):
            """Scales a 2D random field to be within [min_val, max_val]."""
            field_min, field_max = Reduction(MPI.COMM_WORLD).min(field), Reduction(MPI.COMM_WORLD).max(field)
            scaled_field = (field - field_min) / (field_max - field_min)  # Normalize to [0,1]
            return scaled_field * (max_val - min_val) + min_val  # Scale to [min_val, max_val]


        for i in np.arange(ratios.size):
            ratio = ratios[i]

            if j_bn_nodal_points == 0:
                phase_fied_small_grid = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                                            microstructure_name=geometry_ID,
                                                                            coordinates=discretization.fft.coords,
                                                                            **_geom_parameters)  # 1 / 10 ** ratio

                # if ratio <= 0:
                #     phase_fied_small_grid += 1 / 10 ** ratio
                # if ratio > 0:
                # phase_fied_small_grid *= 10 ** ratio
                # phase_fied_small_grid *= ratio
                # phase_fied_small_grid += phase_fied_small_grid
                # phase_fied_small_grid *=1e14
                # phase_fied_small_grid += np.abs(np.sort(phase_fied_small_grid)[0, -2] - np.sort(phase_fied_small_grid)[0, -1])                # phase_fied_small_grid=np.copy(phase_field_smooth)

                # phase_fied_small_grid[:number_of_pixels[0]//2,:number_of_pixels[0]//2]=100

                phase_field_smooth = np.copy(phase_fied_small_grid)

                print('-------------- volume fraction ----------- = {}'.format(
                    np.sum(phase_field_smooth) / np.size(phase_field_smooth)))

            if j_bn_nodal_points > 0:
                # phase_field_smooth = sc.ndimage.zoom(phase_fied_small_grid, zoom=nb_pix_multip, order=0)
                phase_field_smooth = np.repeat(phase_fied_small_grid, 2 ** (j_bn_nodal_points), axis=0)
                phase_field_smooth = np.repeat(phase_field_smooth, 2 ** (j_bn_nodal_points), axis=1)

            # phase_field_smooth = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
            #                                                             microstructure_name=geometry_ID,
            #                                                             coordinates=discretization.fft.coords,
            #                                                             seed=1)
            # print(phase_field_smooth)
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

            # if ratio <= 0:
            #     phase_field = scale_field(phase_field, min_val=1 / 10 ** ratio, max_val=1.0)#
            # if ratio > 0:
            #     phase_field = scale_field(phase_field, min_val=np.power(10,4-ratio), max_val=10 ** 4)#
            # phase_field = scale_field(phase_field, min_val=1 / 10 ** ratio, max_val=1.0)  #
            if ratio == 0:
                pass
            else:
                phase_field = scale_field(phase_field, min_val=1, max_val=10 ** ratio)

            # phase_field = scale_field(phase_field, min_val=1, max_val=2)
            # phase_field[phase_field<=1/10**ratio]= 0
            plt.imshow(phase_field, cmap="gray")
            plt.colorbar( )
            plt.show()
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
            # np.random.seed(seed=1)
            # perturb_dis=np.random.random(discretization.get_displacement_sized_field().shape)
            # perturb_disx=microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
            #                                     microstructure_name='sine_wave_',
            #                                     coordinates=discretization.fft.coords,
            #                                     seed=1,
            #                                     parameter=number_of_pixels[0],
            #                                     contrast=-ratio)
            # perturb_dis[0, 0] = perturb_disx**2
            # perturb_dis[1, 0] = perturb_disx
            # perturb=discretization.apply_gradient_operator_symmetrized( u=perturb_dis )
            # perturb=scale_field(perturb, -0.4, 0.4)
            # macro_gradient_field += (perturb-Reduction(MPI.COMM_WORLD).mean(perturb))

            # Solve mechanical equilibrium constrain
            rhs = discretization.get_rhs(material_data_field_C_0_rho, macro_gradient_field)
            x_init = discretization.get_displacement_sized_field()

            # x_init=np.random.random(discretization.get_displacement_sized_field().shape)
            # perturb_disx = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
            #                                                    microstructure_name='linear',
            #                                                    coordinates=discretization.fft.coords)
            # x_init[0, 0] = -perturb_disx
            # x_init[1, 0] = -perturb_disx.transpose()
            # perturb = discretization.apply_gradient_operator_symmetrized(u=-x_init)

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
            displacement_field, norms = solvers.PCG(K_fun, rhs, x0=x_init, P=M_fun, steps=int(1000), toler=1e-6,  # 5000
                                                    norm_type='rz')
            nb_it[j_bn_nodal_points, i_circles, i] = (len(norms['residual_rr']))
            # print('nb it  = {} '.format(len(norms['residual_rr'])))

            norm_rz.append(norms['residual_rz'])
            norm_rr.append(norms['residual_rr'])
            # norm_rMr.append(norms['data_scaled_rr'])

            # print(nb_it)
            #########
            displacement_field_combi, norms_combi = solvers.PCG(K_fun, rhs, x0=x_init, P=M_fun_combi, steps=int(1),
                                                                # 1000
                                                                toler=1e-6,
                                                                norm_type='data_scaled_rr',
                                                                norm_metric=M_fun
                                                                )
            nb_it_combi[j_bn_nodal_points, i_circles, i] = (len(norms_combi['residual_rr']))
            norm_rz_combi.append(norms_combi['residual_rz'])
            norm_rr_combi.append(norms_combi['residual_rr'])
            norm_rMr_combi.append(norms_combi['data_scaled_rr'])

            #
            displacement_field_Jacobi, norms_Jacobi = solvers.PCG(K_fun, rhs, x0=x_init, P=M_fun_Jacobi, steps=int(1),
                                                                  toler=1e-6,
                                                                  norm_type='data_scaled_rr',
                                                                  norm_metric=M_fun)
            nb_it_Jacobi[j_bn_nodal_points, i_circles, i] = (len(norms_Jacobi['residual_rr']))
            norm_rz_Jacobi.append(norms_Jacobi['residual_rz'])
            norm_rr_Jacobi.append(norms_Jacobi['residual_rr'])
            norm_rMr_Jacobi.append(norms_combi['data_scaled_rr'])
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

            # print(f'ratios = {ratios}')
            # print(f'nb_pix_multips = {nb_pix_multips}')
    if MPI.COMM_WORLD.rank == 0:
        print('  Rank   Size          Domain       Subdomain        Location')
        print('  ----   ----          ------       ---------        --------')
        # Barrier so header is printed first
        print(f'ratios = {ratios}')
        print(f'nbs_nodal_points = {nbs_nodal_points}')
        for i in np.arange(ratios.size):
            ratio = ratios[i]
            print(f'ratio= {ratio}')

            print('greeen')
            print(nb_it[:, :, i])
            # print('jacobi')
            # print(nb_it_Jacobi[:, :, i])
            print('combi')
            print(nb_it_combi[:, :, i])
            print(geometry_ID)
if MPI.COMM_WORLD.rank == 0:
    print('  Rank   Size          Domain       Subdomain        Location')
    print('  ----   ----          ------       ---------        --------')
    # Barrier so header is printed first

    print(f'ratios = {ratios}')
    print(f'nb_pix_multips = {nbs_nodal_points}')
    for i in np.arange(ratios.size):
        ratio = ratios[i]
        print(f'ratio= {ratio}')

        print('greeen')

        print(nb_it[:, :, i])
        print('jacobi')
        print(nb_it_Jacobi[:, :, i])
        print('combi')
        print(nb_it_combi[:, :, i])
        print(geometry_ID)
MPI.COMM_WORLD.Barrier()
quit()
