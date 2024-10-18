import numpy as np
import scipy as sp
import matplotlib as mpl
import time

import matplotlib.pyplot as plt

from NuMPI import Optimization
from NuMPI.IO import save_npy

from mpi4py import MPI
from muGrid import FileIONetCDF, OpenMode, Communicator

plt.rcParams['text.usetex'] = True

import time

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import topology_optimization
from muFFTTO import microstructure_library

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'  # 'bilinear_rectangle'##'linear_triangles' #
formulation = 'small_strain'

domain_size = [1, 1]
number_of_pixels = (256,256)

my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)
start_time = time.time()
if MPI.COMM_WORLD.rank == 0:
    print('  Rank   Size          Domain       Subdomain        Location')
    print('  ----   ----          ------       ---------        --------')
MPI.COMM_WORLD.Barrier()  # Barrier so header is printed first

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)
print(f'{MPI.COMM_WORLD.rank:6} {MPI.COMM_WORLD.size:6} {str(discretization.fft.nb_domain_grid_pts):>15} '
      f'{str(discretization.fft.nb_subdomain_grid_pts):>15} {str(discretization.fft.subdomain_locations):>15}')

# start_time =  MPI.Wtime()

#
# set macroscopic gradient

#
# macro_gradient = np.array([[0.0, 0.3],
#                             [0.3, 0.0]])
macro_gradient = np.array([[1., 0.0],
                           [0.0, 1.0]])
print('macro_gradient = \n {}'.format(macro_gradient))

# Set up the equilibrium system
macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)

# create material data of solid phase rho=1
E_0 = 1
poison_0 = 0.

K_0, G_0 = domain.get_bulk_and_shear_modulus(E=E_0, poison=poison_0)

elastic_C_0 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                 K=K_0,
                                                 mu=G_0,
                                                 kind='linear')

material_data_field_C_0 = np.einsum('ijkl,qxy->ijklqxy', elastic_C_0,
                                    np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                      *discretization.nb_of_pixels])))

# M_fun = lambda x: 1 * x
preconditioner_fnfnqks = discretization.get_preconditioner_NEW(
    reference_material_data_field_ijklqxyz=material_data_field_C_0)

M_fun = lambda x: discretization.apply_preconditioner_NEW(
    preconditioner_Fourier_fnfnqks=preconditioner_fnfnqks,
    nodal_field_fnxyz=x)

stress = np.einsum('ijkl,lk->ij', elastic_C_0, macro_gradient)

# create target material data
print('init_stress = \n {}'.format(stress))
# validation metamaterials
# poison_target = -0.5
# E_target = E_0 * 0.1

# poison_target = 0.2
poison_target = -0.5  # 1 / 3  # lambda = -10

# G_target_auxet =  (7 / 20) * E_0 #(3 / 10) * E_0  #
G_target_auxet = (1 / 4) * E_0

E_target = 2 * G_target_auxet * (1 + poison_target)
# E_target =0.3
# Auxetic metamaterials
# G_target_auxet = (1 / 4) * E_0  #23   25
# E_target=2*G_target_auxet*(1+poison_target)
# test materials


K_targer, G_target = domain.get_bulk_and_shear_modulus(E=E_target, poison=poison_target)

elastic_C_target = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                      K=K_targer,
                                                      mu=G_target,
                                                      kind='linear')
print('Target elastic tangent = \n {}'.format(domain.compute_Voigt_notation_4order(elastic_C_target)))

# target_stress = np.array([[0.0, 0.05],
#                           [0.05, 0.0]])
target_stress = np.einsum('ijkl,lk->ij', elastic_C_target, macro_gradient)
print('target_stress = \n {}'.format(target_stress))
# circle
# p = 1
# w = 1e-5*E_0# 1 / 10  # 1e-4 Young modulus of solid
# #eta = 0.00915#1430#145#357#3#33#5#25#4#7#250
# eta = 0.0555 #0.02125#08#1231925#1515#1430#145#357#3#33#5#25#4#7#250
# Auxetic metamaterials
p = 2
# for w in np.arange(0.1, 1.1, 0.1):  # np.arange(0.2,0.):
for w in [.01, ]:  # np.arange(0.01, 1.5, 0.05):#
    for eta_mult in [1, ]:  # np.arange(0.01, 0.5, 0.05):#
        # w = 1.#1 * 1e-2  # 1e-2 #/6# * E_0  # 1 / 10  # 1e-4 Young modulus of solid
        # eta = 0.01  # 0.005# domain_size[0] / number_of_pixels[0]  # 0.020.005# 2 *
        # eta =0.005#125#/discretization.pixel_size[0]
        eta = eta_mult * discretization.pixel_size[0]

        print('p =   {}'.format(p))
        print('w  =  {}'.format(w))
        print('eta =  {}'.format(eta))


        # eta = 0.00915#1430#145#357#3#33#5#25#4#7#250
        # TODO eta = 0.025
        # TODO w = 0.1

        def my_objective_function_FE(phase_field_1nxyz):
            # print('Objective function:')
            # reshape the field
            phase_field_1nxyz = phase_field_1nxyz.reshape([1, 1, *discretization.nb_of_pixels])

            # Material data in quadrature points
            # phase_field_1nxyz[0, 0, 0:4, 0:4] = 0
            phase_field_at_quad_poits_1qnxyz = \
                discretization.evaluate_field_at_quad_points(nodal_field_fnxyz=phase_field_1nxyz,
                                                             quad_field_fqnxyz=None,
                                                             quad_points_coords_dq=None)[0]

            material_data_field_C_0_rho_ijklqxyz = material_data_field_C_0[..., :, :, :] * np.power(
                phase_field_at_quad_poits_1qnxyz, p)[0, :, 0, ...]

            # Solve mechanical equilibrium constrain
            rhs = discretization.get_rhs(material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                                         macro_gradient_field_ijqxyz=macro_gradient_field)
            if MPI.COMM_WORLD.size == 1:
                print('rhs Of = {}'.format(np.linalg.norm(rhs)))

            K_fun = lambda x: discretization.apply_system_matrix(
                material_data_field=material_data_field_C_0_rho_ijklqxyz,
                displacement_field=x,
                formulation='small_strain')

            # K_matrix=discretization.get_system_matrix(material_data_field=material_data_field_C_0_rho_ijklqxyz)
            # print('K_matrix norm OF = {}'.format(np.linalg.norm(K_matrix)))

            # K_diag_alg = discretization.get_preconditioner_Jacoby(
            #     material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz)
            K_diag_alg = discretization.get_preconditioner_Jacoby_fast(
                material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz)
            M_fun = lambda x: K_diag_alg * discretization.apply_preconditioner_NEW(
                preconditioner_Fourier_fnfnqks=preconditioner_fnfnqks,
                nodal_field_fnxyz=K_diag_alg * x)

            # K_diag_blocks_alg_fnfnxyz = discretization.get_preconditioner_Jacoby_fast(
            #     material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz, prec_type='full')
            #
            # # multiplication with a diagonals of preconditioner
            # M_fun_block = lambda x: np.einsum('abcd...,cd...->ab...', K_diag_blocks_alg_fnfnxyz, x)
            #
            # M_fun = lambda x: M_fun_block(discretization.apply_preconditioner_NEW(
            #     preconditioner_Fourier_fnfnqks=preconditioner_fnfnqks,
            #     nodal_field_fnxyz=M_fun_block(x)))

            displacement_field, norms = solvers.PCG(Afun=K_fun,
                                                    B=rhs,
                                                    x0=None,
                                                    P=M_fun,
                                                    steps=int(2000),
                                                    toler=1e-8)
            if MPI.COMM_WORLD.rank == 0:
                nb_it_comb = len(norms['residual_rz'])
                norm_rz = norms['residual_rz'][-1]
                norm_rr = norms['residual_rr'][-1]
                print(' nb_ steps CG of =' f'{nb_it_comb}, residual_rz = {norm_rz}, residual_rr = {norm_rr}')
                # compute homogenized stress field corresponding t
            homogenized_stress = discretization.get_homogenized_stress(
                material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                displacement_field_fnxyz=displacement_field,
                macro_gradient_field_ijqxyz=macro_gradient_field,
                formulation='small_strain')
            #print('homogenized stress = \n'          ' {} '.format(homogenized_stress)) # good in MPI

            objective_function = topology_optimization.objective_function_small_strain(
                discretization=discretization,
                actual_stress_ij=homogenized_stress,
                target_stress_ij=target_stress,
                phase_field_1nxyz=phase_field_1nxyz,
                eta=eta,
                w=w)
            #print('objective_function= \n'' {} '.format(objective_function))
            # plt.figure()
            # plt.contourf(phase_field_1nxyz[0, 0], cmap=mpl.cm.Greys)
            # # nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
            # plt.clim(0, 1)
            # plt.colorbar()
            # plt.show()
            # print('Sensitivity_analytical')
            sensitivity_analytical, sensitivity_parts = topology_optimization.sensitivity_with_adjoint_problem_FE_NEW(
                discretization=discretization,
                material_data_field_ijklqxyz=material_data_field_C_0,
                displacement_field_fnxyz=displacement_field,
                macro_gradient_field_ijqxyz=macro_gradient_field,
                phase_field_1nxyz=phase_field_1nxyz,
                target_stress_ij=target_stress,
                actual_stress_ij=homogenized_stress,
                preconditioner_fun=M_fun,
                system_matrix_fun=K_fun,
                formulation='small_strain',
                p=p,
                eta=eta,
                weight=w)
            objective_function += sensitivity_parts['adjoint_energy']
            # print('max_grad = {}'.format(np.max(np.abs(sensitivity_analytical.reshape(-1)))))
            # print('abs_grad = {}'.format(np.linalg.norm(sensitivity_analytical.reshape(-1))))

            return objective_function, sensitivity_analytical.reshape(-1)


        if __name__ == '__main__':
            # fp = 'exp_data/muFFTTO_elasticity_random_init_N16_E_target_0.25_Poisson_-0.5_w0.01_eta0.01_p2_bounds=False_FE_NuMPI2.npy'
            # phase = np.load(fp)

            # material distribution
            phase_field_0 = np.random.rand(*discretization.get_scalar_sized_field().shape) ** 1
            # phase_field_0 = np.random.randint(0, high=2, size=discretization.get_scalar_sized_field().shape) ** 1

            # np.random.seed(MPI.COMM_WORLD.rank)
            # np.random.seed(1)
            np.random.seed(MPI.COMM_WORLD.rank)

            phase = np.random.random(discretization.get_scalar_sized_field().shape)
            # phase[0, 0] = np.load(fp)
            # phase = discretization.fft.icoords[0]
            # phase = np.arange(9).reshape(3,3,order='F')

            # # material distribution
            # phase[0, 0] = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
            #                                                   microstructure_name='square_inclusion',
            #                                                   coordinates=discretization.fft.coords)
            # # phase*=np.random.random(discretization.get_scalar_sized_field().shape)
            # phase[0,0,0, 0] = 0
            # phase[0,0,0, 1] = phase[0,0,1, 0] = phase[0,0,1, 1] = 0
            # phase[0,0,0, 2] = phase[0,0,2, 0] = phase[0,0,1, 2] = phase[0,0,2, 1] = phase[0,0,2, 2] = 0
            MPI.COMM_WORLD.Barrier()  # Barrier so header is printed first
            # print('rank' f'{MPI.COMM_WORLD.rank:6} phase=' f'{phase[0, 0]}')
            # plt.figure()
            # plt.contourf(phase[0, 0], cmap=mpl.cm.Greys)
            # # nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
            # plt.clim(0, 1)
            # plt.colorbar()

            # plt.show()
            if MPI.COMM_WORLD.size == 1:
               phase = np.load(f'experiments/exp_data/init_phase_FE_N{number_of_pixels[0]}_NuMPI6.npy')
            phase_field_0 = phase
            # file_data_name = f'init_phase_FE_N{number_of_pixels[0]}_NuMPI{MPI.COMM_WORLD.size}.npy'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')
            #
            # folder_name = 'experiments/exp_data/'
            # save_npy(folder_name + file_data_name, phase_field_0[0, 0],
            #          tuple(discretization.fft.subdomain_locations),
            #          tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD)
            # print(folder_name + file_data_name)
            # phase_field_0[0, 0,
            # phase_field_0.shape[2] * 1 // 4:phase_field_0.shape[2] * 3 // 4,
            # phase_field_0.shape[2] * 1 // 4:phase_field_0.shape[2] * 3 // 4] = 0
            phase_field_00 = np.copy(phase_field_0)
            # my_sensitivity_pixel(phase_field_0).reshape([1, 1, *number_of_pixels])
            phase_field_0 = phase_field_0.reshape(-1)  # b

            print('Init objective function FE  = {}'.format(my_objective_function_FE(phase_field_00)[0]))
            # print('Init objective function pixel  = {}'.format(my_objective_function_pixel(phase_field_00)))

            # xopt = sp.optimize.minimize(my_objective_function_FE,
            #                             phase_field_0,
            #                             method='bfgs',
            #                             jac=my_sensitivity_FE,
            #                             options={'gtol': 1e-6,
            #                                      'disp': True})
            xopt_FE_MPI = Optimization.l_bfgs(fun=my_objective_function_FE,
                                              x=phase_field_0,
                                              jac=True,
                                              maxcor=20,
                                              gtol=1e-5,
                                              ftol=1e-7,
                                              maxiter=15000,
                                              comm=discretization.fft.communicator,
                                              disp=True)

            bounds = False
            xopt_FE_MPI['nb_of_pixels'] = discretization.nb_of_pixels_global
            # phase_field_sol_FE_MPI = xopt.x.reshape([1, 1, *discretization.nb_of_pixels])

            phase_field_sol_FE_MPI = xopt_FE_MPI.x.reshape([1, 1, *discretization.nb_of_pixels])
            sensitivity_sol_FE_MPI = xopt_FE_MPI.jac.reshape([1, 1, *discretization.nb_of_pixels])

            # print('phase_field_sol_FE_MPI  = {}'.format(phase_field_sol_FE_MPI))

            # plt.figure()
            # plt.contourf(phase_field_sol_FE_MPI[0, 0], cmap=mpl.cm.Greys)
            # # nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
            # plt.clim(0, 1)
            # plt.colorbar()
            #
            # plt.show()

            # I/O example
            #  file = FileIONetCDF('example.nc', open_mode=OpenMode.Overwrite,
            #                      communicator=Communicator(MPI.COMM_WORLD))
            #  #v = file.createVariable('phase_field_sol_FE_MPI', np.int32, 'dim')
            #  #f_glob = discretization.fft.register_real_field('phase_field', 1, 'pixel')
            #  #rfield =  discretization.fft.real_space_field('phase_field')
            # # rfield.p = phase_field_sol_FE_MPI[0,0]
            # # discretization.fft.real_field=phase_field_sol_FE_MPI
            #  file.register_field_collection(discretization.fft.real_field_collection)
            #  file.append_frame().write()
            #  file.close()

            file_data_name = f'1muFFTTO_{problem_type}_random_init_N{number_of_pixels[0]}_E_target_{E_target}_Poisson_{poison_target}_Poisson0_{poison_0}_w{w}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{MPI.COMM_WORLD.size}.npy'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')

            folder_name = 'experiments/exp_data/'
            save_npy(folder_name + file_data_name, phase_field_sol_FE_MPI[0, 0],
                     tuple(discretization.fft.subdomain_locations),
                     tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD)
            print(folder_name + file_data_name)

            save_npy(folder_name + f'sensitivity' + file_data_name, sensitivity_sol_FE_MPI[0, 0],
                     tuple(discretization.fft.subdomain_locations),
                     tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD)
            print(folder_name + file_data_name)
            #  quit()
            of = my_objective_function_FE(phase_field_sol_FE_MPI)[0]
            #  # plotting the solution
            #  nodal_coordinates = discretization.get_nodal_points_coordinates()
            #
            #  plt.figure()
            #  plt.contourf(phase_field_00[0, 0])
            #
            #  # plt.clim(0, 1)
            #  plt.colorbar()
            #
            #  #plt.show()
            #
            ######## Postprocess for FE linear solver with NuMPI ########
            # material_data_field_C_0_rho_pixel = material_data_field_C_0[..., :, :] * np.power(phase_field_sol,
            #                                                                             p)
            phase_field_at_quad_poits_1qnxyz = \
                discretization.evaluate_field_at_quad_points(nodal_field_fnxyz=phase_field_sol_FE_MPI,
                                                             quad_field_fqnxyz=None,
                                                             quad_points_coords_dq=None)[0]
            material_data_field_C_0_rho_quad = material_data_field_C_0[..., :, :, :] * np.power(
                phase_field_at_quad_poits_1qnxyz, p)[0, :, 0, ...]

            # Set up the equilibrium system
            macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)

            # Solve mechanical equilibrium constrain
            rhs = discretization.get_rhs(material_data_field_C_0_rho_quad, macro_gradient_field)

            K_fun = lambda x: discretization.apply_system_matrix(material_data_field_C_0_rho_quad, x,
                                                                 formulation='small_strain')

            displacement_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(500), toler=1e-8)

            # compute homogenized stress field corresponding t
            homogenized_stress = discretization.get_homogenized_stress(
                material_data_field_ijklqxyz=material_data_field_C_0_rho_quad,
                displacement_field_fnxyz=displacement_field,
                macro_gradient_field_ijqxyz=macro_gradient_field,
                formulation='small_strain')
            xopt_FE_MPI['target_stress'] = target_stress
            xopt_FE_MPI['homogenized_stress'] = homogenized_stress

            print('init_stress = \n {}'.format(stress))
            print('Target_stress = \n {}'.format(target_stress))
            print('Optimized stress = \n {}'.format(homogenized_stress))

            print('Stress diff = \n {}'.format(target_stress - homogenized_stress))
            objective_function = topology_optimization.objective_function_small_strain(
                discretization=discretization,
                actual_stress_ij=homogenized_stress,
                target_stress_ij=target_stress,
                phase_field_1nxyz=phase_field_sol_FE_MPI,
                eta=eta, w=w)

            print(of)

            # start_time = time.time()
            dim = discretization.domain_dimension
            homogenized_C_ijkl = np.zeros(np.array(4 * [dim, ]))
            # compute whole homogenized elastic tangent
            for i in range(dim):
                for j in range(dim):
                    # set macroscopic gradient
                    macro_gradient_ij = np.zeros([dim, dim])
                    macro_gradient_ij[i, j] = 1
                    # Set up right hand side
                    macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient_ij)

                    # Solve mechanical equilibrium constrain
                    rhs_ij = discretization.get_rhs(material_data_field_C_0_rho_quad, macro_gradient_field)

                    displacement_field_ij, norms = solvers.PCG(K_fun, rhs_ij, x0=None, P=M_fun, steps=int(500),
                                                               toler=1e-8)

                    # ----------------------------------------------------------------------
                    # compute homogenized stress field corresponding
                    homogenized_C_ijkl[i, j] = discretization.get_homogenized_stress(
                        material_data_field_ijklqxyz=material_data_field_C_0_rho_quad,
                        displacement_field_fnxyz=displacement_field_ij,
                        macro_gradient_field_ijqxyz=macro_gradient_field,
                        formulation='small_strain')

            print('Optimized elastic tangent = \n {}'.format(domain.compute_Voigt_notation_4order(homogenized_C_ijkl)))

            MPI.COMM_WORLD.Barrier()  # Barrier so header is printed first
            # print('rank' f'{MPI.COMM_WORLD.rank:6} apply_gradient_operator: finish =' f'{gradient_of_u.shape}')

            end_time = time.time()
            # end_time = MPI.Wtime()

            elapsed_time = end_time - start_time
            print("Elapsed time: ", elapsed_time / 60)
            xopt_FE_MPI['elapsed_time'] = elapsed_time
            # np.save(folder_name + file_data_name+f'xopt_log.npz', xopt_FE_MPI)
            np.savez(folder_name + file_data_name + f'xopt_log.npz', **xopt_FE_MPI)

        #
        #
        #
        #  plt.figure()
        #  plt.contourf(phase_field_sol_FE_MPI[0, 0], cmap=mpl.cm.Greys)
        #  # nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
        #  # plt.clim(0, 1)
        #  plt.colorbar()
        #  plt.title(r" linear FE NuMPI " "\n"
        #            r" Target stress $[{} , {}],[ {}, {}] $" "\n"
        #            r" Stress  $[{} ,{},][ {}, {} ]$" "\n"
        #            r" nb_iter={},  p={}".format(target_stress[0, 0], target_stress[0, 1],
        #                                         target_stress[1, 0], target_stress[1, 1],
        #                                         homogenized_stress[0, 0], homogenized_stress[0, 1],
        #                                         homogenized_stress[1, 0], homogenized_stress[1, 1],
        #                                         xopt_FE_MPI.nit, p), wrap=True)
        #  #plt.show()
        #
        #  src = './figures/'  # source folder\
        #  fig_data_name = f'muFFTTO_{problem_type}_random_init_N{number_of_pixels[0]}_Poisson_{poison_target}_w{w}_eta{eta}_p{p}_bounds={bounds}_FE_NuMPI{MPI.COMM_WORLD.rank}'   #    print('rank' f'{MPI.COMM_WORLD.rank:6} ')
        #
        #
        #  fname = src + fig_data_name + '{}'.format('.png')
        #  print(('create figure: {}'.format(fname)))  # axes[1, 0].legend(loc='upper right')
        #  plt.savefig(fname, bbox_inches='tight')
        #  print('END plot ')
        #
        #  print('Target elastic FE tangent = \n {}'.format(domain.compute_Voigt_notation_4order(elastic_C_target)))
        #
        #  print('Initial elastic tangent = \n {}'.format(domain.compute_Voigt_notation_4order(elastic_C_0)))
        #  # TODO TO FINISH
        #  print('p =   {}'.format(p))
        #  print('w  =  {}'.format(w))
        #  print('eta =  {}'.format(eta))
        #  print(5)
