from cProfile import label

import numpy as np
import scipy as sc
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpi4py import MPI
from NuMPI.Tools import Reduction

from NuMPI.IO import save_npy, load_npy
import matplotlib.pyplot as plt

# Enable LaTeX rendering
plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX
    "font.family": "helvetica",  # Use a serif font
})

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
geom_n = [2,3,4,5 ]  # ,4,5,6 ,6,7,8,9,10,]  # ,2,3,3,2,  #,5,6,7,8,9 ,5,6,7,8,9,10,11
small = np.arange(0., .1, 0.005)
middle = np.arange(0.1, 0.9, 0.03)

large = np.arange(0.9, 1.0 + 0.005, 0.005)
ratios = np.concatenate((small, middle, large))
ratios = np.arange(0., 1.1, 0.2)
ratios = np.arange(0., 1.1, 0.2)
ratios = np.array([1, 4])  # np.arange(1,5)  # 17  33

nb_it = np.zeros((len(geom_n), len(geom_n), ratios.size), )
nb_it_combi = np.zeros((len(geom_n), len(geom_n), ratios.size), )
nb_it_Jacobi = np.zeros((len(geom_n), len(geom_n), ratios.size), )
nb_it_Richardson = np.zeros((len(geom_n), len(geom_n), ratios.size), )
nb_it_Richardson_combi = np.zeros((len(geom_n), len(geom_n), ratios.size), )
colors = ['red', 'blue', 'green', 'orange', 'purple']
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
# markers = ['x', 'o', '|', '>']
markers = [".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x",
           "X", "D", "d", "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
           ]
norm_rr_combi = []
norm_rz_combi = []
norm_rr_Jacobi = []
norm_rz_Jacobi = []
norm_rr = []
norm_rz = []

norm_rMr = []
norm_rMr_Jacobi = []
norm_rMr_combi = []

norm_energy_lb = []
norm_energy_lb_combi = []

kontrast = []
kontrast_2 = []
eigen_LB = []

for geometry_ID in ['linear']:  # ,'sine_wave_','linear', 'right_cluster_x3', 'left_cluster_x3'
    # material distribution
    # geometry_ID =   # right_cluster_linear  laminate_log laminate2 #abs_val 'square_inclusion'#'circle_inclusion'#random_distribution  sine_wave_
    # right_cluster_x3  left_cluster_x3  linear

    for nb_starting_phases in np.arange(np.size(geom_n)):
        # valid_nb_muiltips=geom_n[nb_pixels:]
        print(f'nb_starting_phases = {nb_starting_phases}')
        #   fig = plt.figure(num=99,figsize=(11, 5.5))
        #   gs = fig.add_gridspec(1, np.size(geom_n), hspace=0.5, wspace=0.4, width_ratios=np.size(geom_n)*(1,),
        #                        height_ratios=[1 ])

        #  ax_0 = fig.add_subplot(gs[0, 0])

        # ax_1 = fig.add_subplot(gs[0, 1:])

        for nb_discretization_index in np.arange(np.size(geom_n[nb_starting_phases:])):
            nb_pix_multip = geom_n[nb_starting_phases:][nb_discretization_index]
            print(f'nb_discretization_index = {nb_discretization_index}')
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
            # C_1_5= domain.compute_Voigt_notation_4order(5*elastic_C_1)
            C_1 = domain.compute_Voigt_notation_4order(elastic_C_1)

            material_data_field_C_0 = np.einsum('ijkl,qxy->ijklqxy', elastic_C_1,
                                                np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                                  *discretization.nb_of_pixels])))

            refmaterial_data_field_I4s = np.einsum('ijkl,qxy->ijklqxy', elastic_C_1,
                                                   np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                                     *discretization.nb_of_pixels])))

            print('elastic tangent = \n {}'.format(domain.compute_Voigt_notation_4order(elastic_C_1)))


            def scale_field(field, min_val, max_val):
                """Scales a 2D random field to be within [min_val, max_val]."""
                field_min, field_max = Reduction(MPI.COMM_WORLD).min(field), Reduction(MPI.COMM_WORLD).max(field)
                scaled_field = (field - field_min) / (field_max - field_min)  # Normalize to [0,1]
                return scaled_field * (max_val - min_val) + min_val  # Scale to [min_val, max_val]


            def scale_field_log(field, min_val, max_val):
                """Scales a 2D random field to be within [min_val, max_val]."""
                field_log = np.log10(field)
                field_min, field_max = Reduction(MPI.COMM_WORLD).min(field_log), Reduction(MPI.COMM_WORLD).max(
                    field_log)

                scaled_field = (field_log - field_min) / (field_max - field_min)  # Normalize to [0,1]
                return 10 ** (scaled_field * (np.log10(max_val) - np.log10(min_val)) + np.log10(
                    min_val))  # Scale to [min_val, max_val]


            for i in np.arange(ratios.size):
                ratio = ratios[i]

                if nb_discretization_index == 0:
                    phase_fied_small_grid = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                                                microstructure_name=geometry_ID,
                                                                                coordinates=discretization.fft.coords,
                                                                                seed=1,
                                                                                parameter=number_of_pixels[0])  # ,
                    #                                                                           contrast=-ratio) # $1 / 10 ** ratio
                    if ratio != 0:
                        phase_fied_small_grid += 1 / 10 ** ratio

                    phase_field_smooth = np.copy(phase_fied_small_grid)

                if nb_discretization_index > 0:
                    # phase_field_smooth = sc.ndimage.zoom(phase_fied_small_grid, zoom=nb_pix_multip, order=0)
                    phase_field_smooth = np.repeat(phase_fied_small_grid, 2 ** (nb_discretization_index), axis=0)
                    phase_field_smooth = np.repeat(phase_field_smooth, 2 ** (nb_discretization_index), axis=1)

                # phase_field_smooth = np.copy(phase_fied_small_grid)

                phase_field = np.abs(phase_field_smooth)
                if ratio == 0:
                    phase_field = scale_field(phase_field, min_val=0, max_val=1.0)
                else:
                    phase_field = scale_field(phase_field, min_val=1 / 10 ** ratio, max_val=1.0)
                # phase_field[phase_field>0.3]=1
                # phase_field[phase_field < 0.51] = 1 / 10 ** ratio
                # phase_field = scale_field(phase_field, min_val=1 , max_val=10**ratio)
                # phase_field_log = scale_field_log(phase_field_smooth, min_val=1 / 10 ** ratio, max_val=1.0)

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
                # np.random.seed(seed=1)
                # perturb_dis = np.random.random(discretization.get_displacement_sized_field().shape)
                #
                # perturb=discretization.apply_gradient_operator_symmetrized( u=perturb_dis )
                # perturb=scale_field(perturb, -0.5, 0.5)
                # macro_gradient_field += (perturb-Reduction(MPI.COMM_WORLD).mean(perturb))
                # Solve mechanical equilibrium constrain
                rhs = discretization.get_rhs(material_data_field_C_0_rho, macro_gradient_field)

                K_fun = lambda x: discretization.apply_system_matrix(material_data_field_C_0_rho, x,
                                                                     formulation='small_strain')

                min_val = Reduction(MPI.COMM_WORLD).min(phase_field)
                max_val = Reduction(MPI.COMM_WORLD).max(phase_field)

                # kontrast.append(max_val / min_val)
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
                get_igens = True
                if get_igens:
                    K = discretization.get_system_matrix(material_data_field_C_0_rho)

                    # fixing zero eigenvalues
                    # reduced_K = np.copy(K)
                    K[:, 0] = 0
                    K[0, :] = 0
                    K[:, np.prod(number_of_pixels)] = 0
                    K[np.prod(number_of_pixels), :] = 0
                    K[0, 0] = 1
                    K[np.prod(number_of_pixels), np.prod(number_of_pixels)] = 1
                    M = discretization.get_system_matrix(refmaterial_data_field_I4s)
                    M[:, 0] = 0
                    M[0, :] = 0
                    M[:, np.prod(number_of_pixels)] = 0
                    M[np.prod(number_of_pixels), :] = 0
                    M[0, 0] = 1
                    M[np.prod(number_of_pixels), np.prod(number_of_pixels)] = 1

                    eig_G, eig_vect_G = sc.linalg.eig(a=K, b=M)  # , eigvals_only=True
                    eig_G = np.real(eig_G)
                    eig_G[eig_G == 1.0] = 0

                    K_diag_half = np.copy(np.diag(K))
                    K_diag_half[K_diag_half < 9.99e-16] = 0
                    K_diag_half[K_diag_half != 0] = 1 / np.sqrt(K_diag_half[K_diag_half != 0])

                    DKDsym = np.matmul(np.diag(K_diag_half), np.matmul(K, np.diag(K_diag_half)))
                    eig_JG, eig_vect_JG = sc.linalg.eig(a=DKDsym, b=M)  # , eigvals_only=True

                displacement_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(1000), toler=1e-14,
                                                        norm_type='data_scaled_rr',
                                                        norm_metric=M_fun
                                                        )
                nb_it[nb_discretization_index + nb_starting_phases, nb_starting_phases, i] = (len(norms['residual_rr']))
                print('nb it  = {} '.format(len(norms['residual_rr'])))

                norm_rz.append(norms['residual_rz'])
                norm_rr.append(norms['residual_rr'])
                # norm_energy_lb.append(norms['energy_lb'])
                norm_rMr.append(norms['data_scaled_rr'])

                # print(nb_it)
                #########
                displacement_field_combi, norms_combi = solvers.PCG(K_fun, rhs, x0=None, P=M_fun_combi, steps=int(1000),
                                                                    toler=1e-14, norm_type='data_scaled_rr',
                                                                    norm_metric=M_fun
                                                                    )
                nb_it_combi[nb_discretization_index + nb_starting_phases, nb_starting_phases, i] = (
                    len(norms_combi['residual_rr']))
                norm_rz_combi.append(norms_combi['residual_rz'])
                norm_rr_combi.append(norms_combi['residual_rr'])
                # norm_energy_lb_combi.append(norms_combi['energy_lb'])
                norm_rMr_combi.append(norms_combi['data_scaled_rr'])

                #
                displacement_field_Jacobi, norms_Jacobi = solvers.PCG(K_fun, rhs, x0=None, P=M_fun_Jacobi, steps=int(1),
                                                                      toler=1e-6, norm_type='data_scaled_rr',
                                                                      norm_metric=M_fun
                                                                      )
                nb_it_Jacobi[nb_discretization_index + nb_starting_phases, nb_starting_phases, i] = (
                    len(norms_Jacobi['residual_rr']))
                norm_rz_Jacobi.append(norms_Jacobi['residual_rz'])
                # norm_rr_Jacobi.append(norms_Jacobi['residual_rr'])
                norm_rMr_Jacobi.append(norms_Jacobi['data_scaled_rr'])
                # displacement_field_Richardson, norms_Richardson = solvers.Richardson(K_fun, rhs, x0=None, P=M_fun,
                #                                                                      omega=omega,
                #                                                                      steps=int(100),
                #                                                                      toler=1e-6)
                # nb_it_Richardson[nb_discretization_index+nb_starting_phases,nb_starting_phases , i] = (len(norms_Richardson['residual_rr']))
                # norm_rr_Richardson= norms_Richardson['residual_rr'][-1]
                #
                # displacement_field_Richardson_combi, norms_Richardson_combi = solvers.Richardson(K_fun, rhs, x0=None, P=M_fun_combi,
                #                                                                      omega=omega*0.4,
                #                                                                      steps=int(100),
                #                                                                      toler=1e-6)
                # nb_it_Richardson_combi[nb_discretization_index+nb_starting_phases ,nb_starting_phases, i] = (len(norms_Richardson_combi['residual_rr']))
                # norm_rr_Richardson_combi = norms_Richardson_combi['residual_rr'][-1]

                _info = {}

                _info['nb_of_pixels'] = discretization.nb_of_pixels_global
                _info['nb_of_sampling_points'] = np.shape(phase_fied_small_grid)
                # phase_field_sol_FE_MPI = xopt.x.reshape([1, 1, *discretization.nb_of_pixels])
                _info['norm_rMr_G'] = norms['data_scaled_rr']
                _info['norm_rMr_J'] = norms_Jacobi['data_scaled_rr']
                _info['norm_rMr_JG'] = norms_combi['data_scaled_rr']
                if get_igens:
                    _info['eigens_G'] = eig_G
                    _info['eigens_JG'] = eig_JG

                script_name = 'exp_paper_JG_linear_conv_2_no_eigens'
                file_data_name = (
                    f'{script_name}_gID{geometry_ID}_T{nb_pix_multip}_G{geom_n[nb_starting_phases]}_kappa{ratio}.npy')
                folder_name = '../exp_data/'

                save_npy(folder_name + file_data_name + f'.npy', phase_field,
                         tuple(discretization.fft.subdomain_locations),
                         tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD)
                print(folder_name + file_data_name + f'.npy')

                if MPI.COMM_WORLD.rank == 0:
                    np.savez(folder_name + file_data_name + f'xopt_log.npz', **_info)
                    print(folder_name + file_data_name + f'.xopt_log.npz')
quit()

#                 # print(ratio)
#
#                 x = np.arange(0, 1 * number_of_pixels[0])
#                 y = np.arange(0, 1 * number_of_pixels[1])
#                 X_, Y_ = np.meshgrid(x, y)
#
#                 print(f'nb_discretization_index = {nb_discretization_index}')
#                 # pcm = ax_0.pcolormesh(X_, Y_, np.transpose(phase_field),
#                 #                cmap=mpl.cm.Greys, vmin=1e-4, vmax=1, linewidth=0,
#                 #                rasterized=True)
#                 extended_x = np.linspace(0, 1, phase_field[:, phase_field.shape[0] // 2].size + 1)
#                 # extended_y = np.append(np.diag(phase_field), np.diag(phase_field)[-1])
#                 extended_y = np.append(phase_field[:, phase_field.shape[0] // 2],
#                                        phase_field[:, phase_field.shape[0] // 2][-1])
#                 ax_0.step(extended_x, extended_y
#                           , where='post',
#                           linewidth=1, color='black', linestyle='-',  # marker='|',
#                           label=r'phase contrast -' + f'1e{geom_n[nb_discretization_index]} ')
#                 ax_0.set_xlabel('x coordinate')
#                 ax_0.set_ylabel(f'Phase' + r' $\rho$')
#                 ax_0.set_title(f'Cross section')
#                 # cbar = plt.colorbar(pcm, location='left', cax=ax_0, ticklocation='right')
#                 if ratio == 0:
#                     ax_0.set_yticks([0, 0.5, 1])
#                     ax_0.set_yticklabels([0, 0.5, 1])
#                 else:
#                     ax_0.set_yticks([1/np.power(10,ratio), 0.5, 1])
#                     ax_0.set_yticklabels([f'$10^{{{-ratio}}}$', 0.5, 1])
#
#                 k = np.arange(150)
#                 print(f'k \n {k}')
#                 kappa_G = 10 ** (ratio)
#
#                 convergence = ((np.sqrt(kappa_G) - 1) / (np.sqrt(kappa_G) + 1)) ** k
#                 convergence_G = convergence * norm_rMr[-1][0]
#
#
#                 ax_1.set_title(f'nb phases {2 ** (nb_starting_phases)}, nb pixels {number_of_pixels[0]}', wrap=True)
#
#                 if nb_discretization_index == np.size(geom_n) - 1:
#                     ax_1.semilogy(convergence_G, ':', label=r'$\kappa$ est. - Green', color='green')
#
#                 ax_0.text(-0.25, 1.1, '(a)', transform=ax_0.transAxes)
#
# #
#                 ax_1.set_title(f'nb phases {2 ** (nb_starting_phases)}, nb pixels {number_of_pixels[0]}', wrap=True)
#
#                 ax_1.semilogy(np.arange(1, len(norm_rMr[-1]) + 1), norm_rMr[-1],
#                               label=r'$||r_{k}||_{G^{-1}} $  - Green '+r'$N_{I}$'+f'{number_of_pixels[0]}' , color='green', linestyle='--',marker=markers[nb_discretization_index])
#                 ax_1.semilogy(np.arange(1, len(norm_rMr_combi[-1]) + 1), norm_rMr_combi[-1],
#                               label=r'$||r_{k}||_{G^{-1}} $ - Jacobi-Green  '+r'$N_{I}$' +f'{number_of_pixels[0]}', color='b', linestyle='-.',marker=markers[nb_discretization_index])
#                 #
#                 # ax_1.semilogy(np.arange(1, len(norm_energy_lb[-1]) + 1), norm_energy_lb[-1],
#                 #               label='Upper bound --- Green', color='green', linestyle='--', marker='v')
#                 # ax_1.semilogy(np.arange(1, len(norm_energy_lb_combi[-1]) + 1), norm_energy_lb_combi[-1],
#                 #               label='Upper bound --- Jacobi-Green', color='b', linestyle='-.', marker='v')
#                 ax_1.text(-0.2, 1.05, '(c)', transform=ax_1.transAxes)
#                 # x_1.plot(ratios, geom_n[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
#                 # ax_1.plot(ratios, geom_n[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
#                 ax_1.set_xlabel('PCG iteration - k')
#                 ax_1.set_ylabel('Norm of residua')
#                 ax_1.set_title(f'Convergence')
#
#                 # plt.legend([r'$\kappa$ upper bound', 'Green', 'Jacobi', 'Green + Jacobi', 'Richardson'])
#
#                 ax_1.set_ylim([1e-14, 1e2])  # norm_rz[i][0]]/lb)
#                 print(max(map(len, norm_rr)))
#                 ax_1.set_xlim([1, 100])
#                 # ax_1.set_xticks([1,5, 8, 10,15])
#                 # ax_1.set_xticklabels([1,5, 8, 10,15])
#                 ax_1.set_xticks([1,10, 20, 30,40,50])
#                 ax_1.set_xticklabels([1,10, 20, 30,40,50])
#                 print(f'ratios = {ratios}')
#                 print(f'geom_n = {geom_n}')
#         ax_1.legend(loc='upper right')
#
#         fname = src + 'exp_paper_JG_linear_conv_2_{}_rho{}{}'.format(geometry_ID, ratio,
#                                                                  '.pdf')
#         print(('create figure: {}'.format(fname)))
#         plt.savefig(fname, bbox_inches='tight')
#         plt.show()

quit()
