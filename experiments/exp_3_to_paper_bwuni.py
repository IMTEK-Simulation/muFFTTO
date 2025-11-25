import numpy as np
import scipy as sp
import matplotlib as mpl
import time
import os
import matplotlib.pyplot as plt

from NuMPI import Optimization
from NuMPI.IO import save_npy, load_npy

from mpi4py import MPI

# from muGrid import FileIONetCDF, OpenMode, Communicator

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
number_of_pixels = (32, 32)
dim = np.size(number_of_pixels)
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

# create material data of solid phase rho=1
E_0 = 1
poison_0 = 0.2
K_0, G_0 = domain.get_bulk_and_shear_modulus(E=E_0, poison=poison_0)

elastic_C_0 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                 K=K_0,
                                                 mu=G_0,
                                                 kind='linear')

material_data_field_C_0 = np.einsum('ijkl,qxy->ijklqxy', elastic_C_0,
                                    np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                      *discretization.nb_of_pixels])))

# Set up preconditioner
preconditioner_fnfnqks = discretization.get_preconditioner_NEW(
    reference_material_data_field_ijklqxyz=material_data_field_C_0)

M_fun = lambda x: discretization.apply_preconditioner_NEW(
    preconditioner_Fourier_fnfnqks=preconditioner_fnfnqks,
    nodal_field_fnxyz=x)

# set up load cases
nb_load_cases = 4
macro_gradients = np.zeros([nb_load_cases, dim, dim])
macro_gradients[0] = np.array([[1.0, .0],
                               [.0, .00]])
macro_gradients[1] = np.array([[1.0, .0],
                               [.0, .0]])
macro_gradients[2] = np.array([[.0, .0],
                               [.0, 1.0]])
macro_gradients[3] = np.array([[.0, .0],
                               [.0, 1.0]])
# macro_gradients[2] = np.array([[.0, 0.5],
#                                [0.5, .0]])

left_macro_gradients = np.zeros([nb_load_cases, dim, dim])
left_macro_gradients[0] = np.array([[.0, .0],
                                    [.0, 1.0]])
left_macro_gradients[1] = np.array([[1.0, .0],
                                    [.0, .0]])
left_macro_gradients[2] = np.array([[1.0, .0],
                                    [.0, 0.0]])
left_macro_gradients[3] = np.array([[.0, .0],
                                    [.0, 1.0]])
# left_macro_gradients[0] = np.array([[0., .5],
#                                [.5, .00]])
# left_macro_gradients[1] = np.array([[.0, .5],
#                                [.5,  .0]])

print('macro_gradients = \n {}'.format(macro_gradients))

# Set up  macroscopic gradients
macro_gradient_fields = np.zeros([nb_load_cases, *discretization.get_gradient_size_field().shape])
for load_case in np.arange(nb_load_cases):
    macro_gradient_fields[load_case] = discretization.get_macro_gradient_field(macro_gradients[load_case])
    stress = np.einsum('ijkl,lk->ij', elastic_C_0, macro_gradients[load_case])
    print('init_stress for load case {} = \n {}'.format(load_case, stress))

##### create target material data
# validation metamaterials
# poison_target = -0.5
# E_target = E_0 * 0.1
# poison_target = 0.2
poison_target = -0.5
G_target_auxet = (3 / 20) * E_0  # (3 / 10) * E_0  #
# G_target_auxet = (1 / 4) * E_0
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
##### create target stresses
target_stresses = np.zeros([nb_load_cases, dim, dim])
target_energy = np.zeros([nb_load_cases])

for load_case in np.arange(nb_load_cases):
    target_stresses[load_case] = np.einsum('ijkl,lk->ij', elastic_C_target, macro_gradients[load_case])
    target_energy[load_case] = np.einsum('ij,ijkl,lk->', left_macro_gradients[load_case], elastic_C_target,
                                         macro_gradients[load_case])
    print('target_stress for load case {} = \n {}'.format(load_case, target_stresses[load_case]))
    print('target stress norm for load case {} = \n {}'.format(load_case, np.sum(target_stresses[load_case] ** 2)))
    print('target_energy for load case {} = \n {}'.format(load_case, target_energy[load_case]))

displacement_field_load_case = np.zeros([nb_load_cases, *discretization.get_displacement_sized_field().shape])
adjoint_field_load_case = np.zeros([nb_load_cases, *discretization.get_displacement_sized_field().shape])

# Auxetic metamaterials
p = 2
double_well_depth_test = 1
energy_objective = True
norms_sigma = []
norms_pf = []
# for w in np.arange(0.1, 1.1, 0.1):  # np.arange(0.2,0.):
for w_mult in [4.0, ]:  # np.arange(0.1, 1., 0.1):#[1, ]:  # np.arange(1, 2, 1):  # [2, ]:  #
    for eta_mult in [0.01, ]:
        # np.arange(0.05, 0.5, 0.05):#[0.1, ]:  # np.arange(0.001, 0.01, 0.002):#[0.005, ]:  # np.arange(0.01, 0.5, 0.05):#
        # w = 1.#1 * 1e-2  # 1e-2 #/6# * E_0  # 1 / 10  # 1e-4 Young modulus of solid
        # eta = 0.01  # 0.005# domain_size[0] / number_of_pixels[0]  # 0.020.005# 2 *
        # eta =0.005#125#/discretization.pixel_size[0]
        pixel_diameter = np.sqrt(np.sum(discretization.pixel_size ** 2))
        # w = w_mult / nb_load_cases  # / discretization.pixel_size[0]
        w = w_mult / nb_load_cases  # / discretization.pixel_size[0]
        eta = eta_mult  # * discretization.pixel_size[0]  # pixel_diameter#
        # *eta_mult#pixel_diameter / eta_mult  # * discretization.pixel_size[0]

        print('p =   {}'.format(p))
        print('w  =  {}'.format(w))
        print('eta =  {}'.format(eta))


        def objective_function_multiple_load_cases(phase_field_1nxyz):
            print('Objective function:')
            # reshape the field
            phase_field_1nxyz = phase_field_1nxyz.reshape([1, 1, *discretization.nb_of_pixels])

            # objective function phase field terms
            f_phase_field = topology_optimization.objective_function_phase_field(discretization=discretization,
                                                                                 phase_field_1nxyz=phase_field_1nxyz,
                                                                                 eta=eta,
                                                                                 double_well_depth=double_well_depth_test)
            #  sensitivity phase field terms
            s_phase_field = topology_optimization.sensitivity_phase_field_term_FE_NEW(discretization=discretization,
                                                                                      material_data_field_ijklqxyz=material_data_field_C_0,
                                                                                      phase_field_1nxyz=phase_field_1nxyz,
                                                                                      p=p,
                                                                                      eta=eta,
                                                                                      double_well_depth=1)
            objective_function = f_phase_field

            norms_pf.append(objective_function)
            # Material data in quadrature points
            phase_field_at_quad_poits_1qnxyz = \
                discretization.evaluate_field_at_quad_points(nodal_field_fnxyz=phase_field_1nxyz,
                                                             quad_field_fqnxyz=None,
                                                             quad_points_coords_iq=None)[0]
            print('Phase field at quad points :')

            material_data_field_C_0_rho_ijklqxyz = material_data_field_C_0[..., :, :, :] * np.power(
                phase_field_at_quad_poits_1qnxyz, p)[0, :, 0, ...]

            K_fun = lambda x: discretization.apply_system_matrix(
                material_data_field=material_data_field_C_0_rho_ijklqxyz,
                displacement_field=x,
                formulation='small_strain')
            # Solve mechanical equilibrium constrain
            homogenized_stresses = np.zeros([nb_load_cases, dim, dim])

            f_sigmas = np.zeros([nb_load_cases, 1])
            f_sigmas_energy = np.zeros([nb_load_cases, 1])
            adjoint_energies = np.zeros([nb_load_cases, 1])
            s_stress_and_adjoint_load_cases = np.zeros([nb_load_cases, *s_phase_field.shape])
            s_energy_and_adjoint_load_cases = np.zeros([nb_load_cases, *s_phase_field.shape])
            for load_case in np.arange(nb_load_cases):
                rhs_load_case = discretization.get_rhs(
                    material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                    macro_gradient_field_ijqxyz=macro_gradient_fields[load_case])
                if MPI.COMM_WORLD.size == 1:
                    print('rhs Of = {}'.format(np.linalg.norm(rhs_load_case)))

                K_diag_alg = discretization.get_preconditioner_Jacoby_fast(
                    material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz)
                M_fun = lambda x: K_diag_alg * discretization.apply_preconditioner_NEW(
                    preconditioner_Fourier_fnfnqks=preconditioner_fnfnqks,
                    nodal_field_fnxyz=K_diag_alg * x)

                displacement_field_load_case[load_case], norms = solvers.PCG(Afun=K_fun,
                                                                             B=rhs_load_case,
                                                                             x0=displacement_field_load_case[load_case],
                                                                             P=M_fun,
                                                                             steps=int(10000),
                                                                             toler=1e-10)

                print('displacement_field_load_case:')
                if MPI.COMM_WORLD.rank == 0:
                    nb_it_comb = len(norms['residual_rz'])
                    norm_rz = norms['residual_rz'][-1]
                    norm_rr = norms['residual_rr'][-1]
                    # print(
                    #     'load case ' f'{load_case},  nb_ steps CG of =' f'{nb_it_comb}, residual_rz = {norm_rz}, residual_rr = {norm_rr}')
                    # compute homogenized stress field corresponding t
                homogenized_stresses[load_case] = discretization.get_homogenized_stress(
                    material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                    displacement_field_fnxyz=displacement_field_load_case[load_case],
                    macro_gradient_field_ijqxyz=macro_gradient_fields[load_case],
                    formulation='small_strain')
                print('homogenized stress = \n'          ' {} '.format(homogenized_stresses[load_case]))  # good in MPI

                # stress difference potential: actual_stress_ij is homogenized stress
                # f_sigmas[load_case] = w * topology_optimization.compute_stress_equivalence_potential(
                #     actual_stress_ij=homogenized_stresses[load_case],
                #     target_stress_ij=target_stresses[load_case])
                if energy_objective:
                    # strain_fluctuation_ijqxyz = discretization.apply_gradient_operator_symmetrized(
                    #    displacement_field_load_case[load_case])
                    # actual_strain_ijqxyz = macro_gradient_fields[load_case] + strain_fluctuation_ijqxyz
                    print('f sigma energy')
                    f_sigmas_energy[load_case] = (
                                w * topology_optimization.compute_elastic_energy_equivalence_potential(
                            discretization=discretization,
                            actual_stress_ij=homogenized_stresses[load_case],
                            target_stress_ij=target_stresses[load_case],
                            left_macro_gradient_ij=left_macro_gradients[load_case],
                            target_energy=target_energy[load_case]))
                    print('f sigma energy2 ')

                    s_energy_and_adjoint_load_cases[
                        load_case], adjoint_energies[
                        load_case] = topology_optimization.sensitivity_elastic_energy_and_adjoint_FE_NEW(
                        discretization=discretization,
                        material_data_field_ijklqxyz=material_data_field_C_0,
                        displacement_field_fnxyz=displacement_field_load_case[load_case],
                        macro_gradient_field_ijqxyz=macro_gradient_fields[load_case],
                        left_macro_gradient_ij=left_macro_gradients[load_case],
                        phase_field_1nxyz=phase_field_1nxyz,
                        target_stress_ij=target_stresses[load_case],
                        actual_stress_ij=homogenized_stresses[load_case],
                        preconditioner_fun=M_fun,
                        system_matrix_fun=K_fun,
                        formulation='small_strain',
                        target_energy=target_energy[load_case],
                        p=p,
                        weight=w)
                    print('f sigma energy3')

                    s_phase_field += s_energy_and_adjoint_load_cases[load_case]

                    f_sigmas_energy[load_case] += adjoint_energies[load_case]

                    objective_function += f_sigmas_energy[load_case]
                else:
                    f_sigmas[load_case] = w * (topology_optimization.compute_stress_equivalence_potential(
                        actual_stress_ij=homogenized_stresses[load_case],
                        target_stress_ij=target_stresses[load_case]))
                    # if MPI.COMM_WORLD.rank == 0:
                    #     print('w*f_sigmas  = '          ' {} '.format(f_sigmas[load_case]))  # good in MPI
                    #     print('sum of w*f_sigmas  = '          ' {} '.format(np.sum(f_sigmas)))
                    s_stress_and_adjoint_load_cases[load_case], adjoint_field_load_case[
                        load_case], adjoint_energies[
                        load_case] = topology_optimization.sensitivity_stress_and_adjoint_FE_NEW(
                        discretization=discretization,
                        material_data_field_ijklqxyz=material_data_field_C_0,
                        displacement_field_fnxyz=displacement_field_load_case[load_case],
                        adjoint_field_last_step_fnxyz=adjoint_field_load_case[load_case],
                        macro_gradient_field_ijqxyz=macro_gradient_fields[load_case],
                        phase_field_1nxyz=phase_field_1nxyz,
                        target_stress_ij=target_stresses[load_case],
                        actual_stress_ij=homogenized_stresses[load_case],
                        preconditioner_fun=M_fun,
                        system_matrix_fun=K_fun,
                        formulation='small_strain',
                        target_energy=target_energy[load_case],
                        p=p,
                        weight=w)
                    s_phase_field += s_stress_and_adjoint_load_cases[load_case]

                    f_sigmas[load_case] += adjoint_energies[load_case]

                    objective_function += f_sigmas[load_case]

                # if MPI.COMM_WORLD.rank == 0:
                #     print(
                #         'load case ' f'{load_case},  f_sigmas =' f'{f_sigmas[load_case]}')
                #     print(
                #         'load case ' f'{load_case},  objective_function =' f'{objective_function}')

            norms_sigma.append(objective_function)
            return objective_function[0], s_phase_field.reshape(-1)


        if __name__ == '__main__':
            script_name = 'exp_2D_elasticity_TO_indre_3exp'

            run_adam = False
            run_lbfg = True
            random_initial_geometry = True
            bounds = False

            # fp = 'exp_data/muFFTTO_elasticity_random_init_N16_E_target_0.25_Poisson_-0.5_w0.01_eta0.01_p2_bounds=False_FE_NuMPI2.npy'
            # phase = np.load(fp)
            # # material distribution
            np.random.seed(MPI.COMM_WORLD.rank)
            # phase_field_0 = np.random.rand(*discretization.get_scalar_sized_field().shape) ** 1
            phase_field_0 = np.random.randint(0, high=2, size=discretization.get_scalar_sized_field().shape) ** 1
            # phase_field_0 = np.random.choice([0, 1], size=discretization.get_scalar_sized_field().shape,
            #                                  p=[0.5, 0.5])  # equal probability for 0 and 1

            if not random_initial_geometry:
                phase_field_0[0, 0] = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                                          microstructure_name='square_inclusion',
                                                                          coordinates=discretization.fft.coords)


            def apply_filter(phase):
                f_field = discretization.fft.fft(phase)
                # f_field[0, 0, np.logical_and(np.abs(discretization.fft.fftfreq[0]) > 0.25,
                #                              np.abs(discretization.fft.fftfreq[1]) > 0.25)] = 0
                f_field[0, 0, np.logical_and(np.abs(discretization.fft.ifftfreq[0]) > 8,
                                             np.abs(discretization.fft.ifftfreq[1]) > 8)] = 0
                # f_field[0, 0, 12:, 24:] = 0
                phase = discretization.fft.ifft(f_field) * discretization.fft.normalisation
                phase[phase > 1] = 1
                phase[phase < 0] = 0
                # min_ = discretization.mpi_reduction.min(phase)
                # max_ = discretization.mpi_reduction.max(phase)
                # phase = (phase + np.abs(min_)) / (max_ + np.abs(min_))
                return phase


            # phase = np.random.random(discretization.get_scalar_sized_field().shape)
            phase_field_0 = apply_filter(phase_field_0)

            folder_name = '/muFFTTO_test/experiments/exp_data/'  # s'exp_data/'
            # file_data_name = (
            #     f'1muFFTTO_elasticity_random_init_N{number_of_pixels[0]}_E_target_{E_target}_Poisson_{poison_target}_Poisson0_{poison_0}_w{w}_eta{1}_p{p}_bounds=False_FE_NuMPI{6}.npy')
            file_data_name = (
                f'1muFFTTO_elasticity_{script_name}_N{64}_E_target_{E_target}_Poisson_{poison_target}_Poisson0_{poison_0}_w{w}_eta{2}_p{p}_bounds=False_FE_NuMPI{6}.npy')

            load_init_from_same_grid = False
            if load_init_from_same_grid:
                # file_data_name = f'eta_1muFFTTO_{problem_type}_random_init_N{number_of_pixels[0]}_E_target_{E_target}_Poisson_{poison_target}_Poisson0_{poison_0}_w{w_mult}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{MPI.COMM_WORLD.size}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}.npy'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')
                file_data_name = f'eta_1muFFTTO_{problem_type}_{script_name}_N{number_of_pixels[0]}_E_target_{E_target}_Poisson_{poison_target}_Poisson0_{poison_0}_w{w_mult}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{MPI.COMM_WORLD.size}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}.npy'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')

                if MPI.COMM_WORLD.size == 1 or None:
                    # phase = np.load(f'experiments/exp_data/init_phase_FE_N{number_of_pixels[0]}_NuMPI6.npy')
                    # phase= np.load(f'experiments/exp_data/'  + file_data_name)
                    phase = np.load(f'experiments/exp_data/' + file_data_name)
                else:

                    # file_data_name = (
                    #    f'1muFFTTO_elasticity_random_init_N{number_of_pixels[0]}_E_target_{E_target}_Poisson_{poison_target}_Poisson0_{poison_0}_w{w}_eta{eta_mult}_p{p}_bounds=False_FE_NuMPI{8}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}.npy')

                    phase = load_npy(folder_name + file_data_name,
                                     tuple(discretization.fft.subdomain_locations),
                                     tuple(discretization.nb_of_pixels), MPI.COMM_WORLD)

                phase_field_0[0, 0] = phase  # [discretization.fft.subdomain_slices]

            if MPI.COMM_WORLD.size == 1:
                print('rank' f'{MPI.COMM_WORLD.rank:6} phase=' f'')
                plt.figure()
                plt.contourf(phase_field_0[0, 0], cmap=mpl.cm.Greys)
                # nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
                plt.clim(0, 1)
                plt.colorbar()

                plt.show()

            phase_field_00 = np.copy(phase_field_0)
            # my_sensitivity_pixel(phase_field_0).reshape([1, 1, *number_of_pixels])
            phase_field_0 = phase_field_0.reshape(-1)  # b

            print('Init objective function FE  = {}'.format(objective_function_multiple_load_cases(phase_field_00)[0]))
            # print('Init objective function pixel  = {}'.format(my_objective_function_pixel(phase_field_00)))

            if run_adam:

                norms_f_adam = []
                norms_delta_f_adam = []
                norms_max_grad_f_adam = []
                norms_norm_grad_f_adam = []


                def adam_callback(result_norms):
                    iteration = result_norms[-1]
                    norms_f_adam.append(result_norms[0])
                    norms_delta_f_adam.append(result_norms[1])
                    norms_max_grad_f_adam.append(result_norms[2])
                    norms_norm_grad_f_adam.append(result_norms[3])
                    file_data_name_it = f'adam_muFFTTO_{problem_type}_{script_name}_N{number_of_pixels[0]}_E_target_{E_target}_Poisson_{poison_target}_Poisson0_{poison_0}_w{w_mult}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{MPI.COMM_WORLD.size}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}_it{iteration}'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')

                    save_npy(os.getcwd() + folder_name + file_data_name_it + f'.npy',
                             result_norms[4].reshape([*discretization.nb_of_pixels]),
                             tuple(discretization.fft.subdomain_locations),
                             tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD)
                    if MPI.COMM_WORLD.size == 1:
                        print(folder_name + file_data_name_it + f'.npy')


                f_for_adam = lambda x: objective_function_multiple_load_cases(x)[0]
                df_for_adam = lambda x: objective_function_multiple_load_cases(x)[1]
                # Perform the gradient descent search with Adam
                [best, score, t] = solvers.adam(f=f_for_adam,
                                                df=df_for_adam,
                                                x0=phase_field_0,
                                                n_iter=4000,
                                                alpha=0.05,
                                                beta1=0.9,
                                                beta2=0.999,
                                                callback=adam_callback,
                                                gtol=1e-5,
                                                ftol=1e-14,
                                                )
                print('?---------------------- >%d = %.5f' % (t, score))
                solution_phase = best.reshape([1, 1, *discretization.nb_of_pixels])

                if MPI.COMM_WORLD.size == 1:
                    print('rank' f'{MPI.COMM_WORLD.rank:6} phase=' f' ')
                    plt.figure()
                    plt.contourf(solution_phase[0, 0], cmap=mpl.cm.Greys)
                    # nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
                    plt.clim(0, 1)
                    plt.title('>%d = %.5f' % (t, score))
                    plt.colorbar()

                    plt.show()
                print('Done!')
                print('fx = %f' % (score))
                _info = {}
                _info['nb_of_pixels'] = discretization.nb_of_pixels_global
                # phase_field_sol_FE_MPI = xopt.x.reshape([1, 1, *discretization.nb_of_pixels])
                _info['norms_f'] = norms_f_adam
                _info['norms_delta_f'] = norms_delta_f_adam
                _info['norms_max_grad_f'] = norms_max_grad_f_adam
                _info['norms_norm_grad_f'] = norms_norm_grad_f_adam

                file_data_name = f'adam_muFFTTO_{problem_type}_{script_name}_N{number_of_pixels[0]}_E_target_{E_target}_Poisson_{poison_target}_Poisson0_{poison_0}_w{w_mult}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{MPI.COMM_WORLD.size}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')

                save_npy(os.getcwd() + folder_name + file_data_name + f'.npy', solution_phase[0, 0],
                         tuple(discretization.fft.subdomain_locations),
                         tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD)
                print(folder_name + file_data_name + f'.npy')

                # np.save(folder_name + file_data_name+f'xopt_log.npz', xopt_FE_MPI)
                if MPI.COMM_WORLD.rank == 0:
                    np.savez(os.getcwd() + folder_name + file_data_name + f'xopt_log.npz', **_info)

            if run_lbfg:

                norms_f = []
                norms_delta_f = []
                norms_max_grad_f = []
                norms_norm_grad_f = []
                norms_max_delta_x = []
                norms_norm_delta_x = []


                def my_callback(result_norms):
                    print('my_callback')
                    print(' MPI.COMM_WORLD.size{}'.format(MPI.COMM_WORLD.size))
                    iteration = result_norms[-1]
                    norms_f.append(result_norms[0])
                    norms_delta_f.append(result_norms[1])
                    norms_max_grad_f.append(result_norms[2])
                    norms_norm_grad_f.append(result_norms[3])
                    norms_max_delta_x.append(result_norms[4])
                    norms_norm_delta_x.append(result_norms[5])
                    # file_data_name_it = (f'lbfg_muFFTTO_{problem_type}_{script_name}_N{number_of_pixels[0]}_E_target_{E_target}_Poisson_{poison_target}_Poisson0_{poison_0}_w{w_mult}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{MPI.COMM_WORLD.size}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}_it{iteration}')  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')
                    # file_data_name_it = f'_it{iteration}'
                    file_data_name_it = (
                        f'{script_name}_N{number_of_pixels[0]}_Et_{E_target}_Pt_{poison_target}_P0_{poison_0}_w{w_mult}_eta{eta_mult}_p{p}_mpi{MPI.COMM_WORLD.size}_nlc_{nb_load_cases}_e_{energy_objective}_it{iteration}')

                    print(os.getcwd() + folder_name + file_data_name_it + '.npy')

                    save_npy(os.getcwd() + folder_name + file_data_name_it + '.npy',
                             result_norms[6].reshape([*discretization.nb_of_pixels]),
                             tuple(discretization.fft.subdomain_locations),
                             tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD)
                    if MPI.COMM_WORLD.size == 1:
                        print(folder_name + file_data_name_it + f'.npy')


                xopt_FE_MPI = Optimization.l_bfgs(fun=objective_function_multiple_load_cases,
                                                  x=phase_field_0,
                                                  jac=True,
                                                  maxcor=20,
                                                  gtol=1e-5,
                                                  ftol=1e-12,
                                                  maxiter=5000,
                                                  comm=discretization.fft.communicator,
                                                  disp=True,
                                                  callback=my_callback
                                                  )

                solution_phase = xopt_FE_MPI.x.reshape([1, 1, *discretization.nb_of_pixels])
                sensitivity_sol_FE_MPI = xopt_FE_MPI.jac.reshape([1, 1, *discretization.nb_of_pixels])

                _info = {}

                _info['nb_of_pixels'] = discretization.nb_of_pixels_global
                # phase_field_sol_FE_MPI = xopt.x.reshape([1, 1, *discretization.nb_of_pixels])
                _info['norms_f'] = norms_f
                _info['norms_delta_f'] = norms_delta_f
                _info['norms_max_grad_f'] = norms_max_grad_f
                _info['norms_norm_grad_f'] = norms_norm_grad_f
                _info['norms_max_delta_x'] = norms_max_delta_x
                _info['norms_norm_delta_x'] = norms_norm_delta_x
                _info['norms_sigma'] = norms_sigma
                _info['norms_pf'] = norms_pf

                _info['nb_of_pixels'] = discretization.nb_of_pixels_global
                # phase_field_sol_FE_MPI = xopt.x.reshape([1, 1, *discretization.nb_of_pixels])

                # file_data_name = f'lbfg_muFFTTO_{problem_type}_{script_name}_N{number_of_pixels[0]}_E_target_{E_target}_Poisson_{poison_target}_Poisson0_{poison_0}_w{w_mult}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{MPI.COMM_WORLD.size}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')

                # file_data_name = (f'lbfg_{problem_type}_{script_name}_N{number_of_pixels[0]}_Et_{E_target}_Pt_{poison_target}_P0_{poison_0}_w{w_mult}_eta{eta_mult}_p{p}_NuMPI{MPI.COMM_WORLD.size}_nb_load_cases_{nb_load_cases}_ener_obj_{energy_objective}')
                file_data_name = (
                    f'{script_name}_N{number_of_pixels[0]}_Et_{E_target}_Pt_{poison_target}_P0_{poison_0}_w{w_mult}_eta{eta_mult}_p{p}_mpi{MPI.COMM_WORLD.size}_nlc_{nb_load_cases}_e_{energy_objective}')

                save_npy(os.getcwd() + folder_name + file_data_name + '.npy', solution_phase[0, 0],
                         tuple(discretization.fft.subdomain_locations),
                         tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD)
                print(os.getcwd() + folder_name + file_data_name + f'.npy')
            ######## Postprocess for FE linfile_data_name_it = (f'{script_name}_N{number_of_pixels[0]}_Et_{E_target}_Pt_{poison_target}_P0_{poison_0}_w{w_mult}_eta{eta_mult}_p{p}_mpi{MPI.COMM_WORLD.size}_nlc_{nb_load_cases}_e_{energy_objective}_it{iteration}')
            # ear solver with NuMPI ########
            # material_data_field_C_0_rho_pixel = material_data_field_C_0[..., :, :] * np.power(phase_field_sol,
            #                                                                             p)
            phase_field_at_quad_poits_1qnxyz = \
                discretization.evaluate_field_at_quad_points(nodal_field_fnxyz=solution_phase,
                                                             quad_field_fqnxyz=None,
                                                             quad_points_coords_iq=None)[0]
            material_data_field_C_0_rho_quad = material_data_field_C_0[..., :, :, :] * np.power(
                phase_field_at_quad_poits_1qnxyz, p)[0, :, 0, ...]
            homogenized_stresses = np.zeros([nb_load_cases, dim, dim])

            for load_case in np.arange(nb_load_cases):
                # Set up the equilibrium system
                macro_gradient_field = discretization.get_macro_gradient_field(macro_gradients[load_case])

                # Solve mechanical equilibrium constrain
                rhs = discretization.get_rhs(material_data_field_C_0_rho_quad, macro_gradient_field)

                K_fun = lambda x: discretization.apply_system_matrix(material_data_field_C_0_rho_quad, x,
                                                                     formulation='small_strain')

                displacement_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(500), toler=1e-8)

                # compute homogenized stress field corresponding t
                homogenized_stresses[load_case] = discretization.get_homogenized_stress(
                    material_data_field_ijklqxyz=material_data_field_C_0_rho_quad,
                    displacement_field_fnxyz=displacement_field,
                    macro_gradient_field_ijqxyz=macro_gradient_field,
                    formulation='small_strain')
                _info['target_stress' + f'{load_case}'] = target_stresses[load_case]
                _info['homogenized_stresses' + f'{load_case}'] = homogenized_stresses[load_case]
                stress = np.einsum('ijkl,lk->ij', elastic_C_0, macro_gradients[load_case])

            dim = discretization.domain_dimension
            homogenized_C_ijkl = np.zeros(np.array(4 * [dim, ]))
            # compute whole homogenized elastic tangent
            for i in range(dim):
                for j in range(dim):
                    # set macroscopic gradient
                    macro_gradient_ij = np.zeros([dim, dim])
                    macro_gradient_ij[i, j] = 1
                    # Set up right hand side
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
            if MPI.COMM_WORLD.rank == 0:
                print('Optimized elastic tangent = \n {}'.format(
                    domain.compute_Voigt_notation_4order(homogenized_C_ijkl)))

            _info['homogenized_C_ijkl'] = domain.compute_Voigt_notation_4order(homogenized_C_ijkl)

            # np.save(folder_name + file_data_name+f'xopt_log.npz', xopt_FE_MPI)
            if MPI.COMM_WORLD.rank == 0:
                np.savez(os.getcwd() + folder_name + file_data_name + 'xopt_log.npz', **_info)