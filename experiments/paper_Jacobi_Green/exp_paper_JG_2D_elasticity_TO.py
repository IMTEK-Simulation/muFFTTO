import numpy as np
import scipy as sp
import matplotlib as mpl
import time

import matplotlib.pyplot as plt

from NuMPI import Optimization
from NuMPI.IO import save_npy, load_npy

from mpi4py import MPI
from muGrid import FileIONetCDF, OpenMode, Communicator

plt.rcParams['text.usetex'] = True

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import topology_optimization
from muFFTTO import microstructure_library

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'  # 'bilinear_rectangle'##'linear_triangles' # # linear_triangles_tilled
formulation = 'small_strain'

domain_size = [1, 1]  #
number_of_pixels = (32, 32)
dim = np.size(number_of_pixels)
my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)
start_time = time.time()
print('number_of_pixels = \n {} core {}'.format(number_of_pixels, MPI.COMM_WORLD.rank))
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
K_0, G_0 = 1, 0.5
E_0 = 9 * K_0 * G_0 / (3 * K_0 + G_0)
poison_0 = (3 * K_0 - 2 * G_0) / (2 * (3 * K_0 + G_0))
# poison_0 = 0.2
# G_0 = E_0 / (2 * (1 + poison_0))
K_0, G_0 = domain.get_bulk_and_shear_modulus(E=E_0, poison=poison_0)
K_0, G_0 = 1, 0.5
print('1 = \n   core {}'.format(MPI.COMM_WORLD.rank))

elastic_C_0 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                 K=K_0,
                                                 mu=G_0,
                                                 kind='linear')
print('2 = \n   core {}'.format(MPI.COMM_WORLD.rank))
# material_data_field_C_0 = np.einsum('ijkl,qxy->ijklqxy', elastic_C_0,
#                                     np.ones(np.array([discretization.nb_quad_points_per_pixel,
#                                                       *discretization.nb_of_pixels])))
print('3 = \n   core {}'.format(MPI.COMM_WORLD.rank))
# Set up preconditioner
preconditioner_fnfnqks = discretization.get_preconditioner_Green_fast(
    reference_material_data_ijkl=elastic_C_0)
M_fun_Green = lambda x: discretization.apply_preconditioner_NEW(
    preconditioner_Fourier_fnfnqks=preconditioner_fnfnqks,
    nodal_field_fnxyz=x)
# set up load cases
nb_load_cases = 3
macro_gradients = np.zeros([nb_load_cases, dim, dim])
macro_multip = 1.
macro_gradients[0] = np.array([[1.0, 0.0],
                               [0., .0]]) * macro_multip
macro_gradients[1] = np.array([[.0, .0],
                               [.0, 1.0]]) * macro_multip
macro_gradients[2] = np.array([[.0, 0.5],
                               [0.5, .0]]) * macro_multip
# macro_gradients[3] = np.array([[.0, .0],
#                                [.0, 1.0]])
# macro_gradients[2] = np.array([[.0, 0.5],
#                                [0.5, .0]])

left_macro_gradients = np.zeros([nb_load_cases, dim, dim])
left_macro_gradients[0] = np.array([[.0, .0],
                                    [.0, 1.0]]) * macro_multip
left_macro_gradients[1] = np.array([[1.0, .0],
                                    [.0, .0]]) * macro_multip
left_macro_gradients[2] = np.array([[.0, .5],
                                    [0.5, 0.0]]) * macro_multip
# left_macro_gradients[3] =  np.array([[.0, .0],
#                                     [.0, 1.0]])
# left_macro_gradients[5] = np.array([[0., .5],
#                                [.5, .00]])
# left_macro_gradients[2] = np.array([[.0, .5],
#                                [.5,  .0]])

print('macro_gradients = \n {}'.format(macro_gradients))

# Set up  macroscopic gradients
macro_gradient_fields = []
for load_case in np.arange(nb_load_cases):
    macro_gradient_field_ijqxyz = discretization.get_gradient_size_field(name=f'macro_gradient_field_{load_case}')

    macro_gradient_fields.append(discretization.get_macro_gradient_field(macro_gradient_ij=macro_gradients[load_case],
                                                                         macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz))

    stress = np.einsum('ijkl,lk->ij', elastic_C_0, macro_gradients[load_case])
    print('init_stress for load case {} = \n {}'.format(load_case, stress))

##### create target material data
# validation metamaterials
# poison_target = -0.5
# E_target = E_0 * 0.1
# poison_target = 0.2
for ration in [-0.5]:
    poison_target = ration
    G_target_auxet = (3 / 20) * E_0  # (3 / 10) * E_0  #
    # G_target_auxet = (1 / 4) * E_0
    E_target = 2 * G_target_auxet * (1 + poison_target)
    # E_target = 0.15
    # Auxetic metamaterials
    # G_target_auxet = (1 / 4) * E_0  #23   25
    # E_target=2*G_target_auxet*(1+poison_target)
    # test materials
    #
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

    displacement_field_load_case = []
    adjoint_field_load_case = []
    for load_case in np.arange(nb_load_cases):
        displacement_field_load_case.append(
            discretization.get_unknown_size_field(name=f'displacement_field_load_case{load_case}'))
        adjoint_field_load_case.append(
            discretization.get_unknown_size_field(name=f'adjoint_field_load_case_{load_case}'))

    # Auxetic metamaterials
    p = 2
    double_well_depth_test = 1
    energy_objective = False
    norms_sigma = []
    norms_pf = []
    num_iteration_ = []
    preconditioer = 'Jacobi_Green'  # Jacobi'  # ['Green','Jacobi','Jacobi_Green']
    # np.concatenate([np.arange(0.1, 1., 0.2),np.arange(1, 10, 2),np.arange(10, 110, 10)])
    # for w in np.arange(0.1, 1.1, 0.1):  # np.arange(0.2,0.):
    # weights = np.concatenate(
    #     [np.arange(0.1, 2., 0.1), np.arange(2, 3, 1), np.arange(3, 10, 2), np.arange(10, 110, 20)])
    weights = [3.9]  # np.concatenate([np.arange(0.1, 2., 1)])
    for w_mult in weights:  # ,10.,20.,30.,40.0 np.arange(0.1, 1., 0.1):#[1, ]:  # np.arange(1, 2, 1):  # [2, ]:  #
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


            def objective_function_multiple_load_cases(phase_field_1nxyz_flat):
                # print('Objective function:')
                # reshape the field
                # zero_small_phases = False
                # if zero_small_phases:
                #     phase_field_1nxyz[phase_field_1nxyz < 1e-5] = 0

                phase_field_1nxyz = discretization.get_scalar_field(name='phase_field_in_objective')
                phase_field_1nxyz.s = phase_field_1nxyz_flat.reshape([1, 1, *number_of_pixels])

                # objective function phase field terms
                # f = eta * f_rho_grad + f_dw / eta   !! no stress equivalence yet !!
                f_phase_field = topology_optimization.objective_function_phase_field(discretization=discretization,
                                                                                     phase_field_1nxyz=phase_field_1nxyz,
                                                                                     eta=eta,
                                                                                     double_well_depth=double_well_depth_test)
                #  sensitivity phase field terms
                s_phase_field = topology_optimization.sensitivity_phase_field_term_FE_NEW(discretization=discretization,
                                                                                          base_material_data_ijkl=elastic_C_0,
                                                                                          phase_field_1nxyz=phase_field_1nxyz,
                                                                                          p=p,
                                                                                          eta=eta,
                                                                                          double_well_depth=1)
                # ??????
                objective_function = f_phase_field

                norms_pf.append(objective_function)
                # Material data in quadrature points
                phase_field_at_quad_poits_1qnxyz = \
                    discretization.evaluate_field_at_quad_points(nodal_field_fnxyz=phase_field_1nxyz.s,
                                                                 quad_field_fqnxyz=None,
                                                                 quad_points_coords_iq=None)[0]

                material_data_field_C_0_rho_ijklqxyz = discretization.get_material_data_size_field(
                    name='material_data_field_C_0_rho_ijklqxyz_in_objective')

                # material_data_field_C_0_rho_ijklqxyz = material_data_field_C_0[..., :, :, :] * np.power(
                #     phase_field_at_quad_poits_1qnxyz, p)[0, :, 0, ...]

                material_data_field_C_0_rho_ijklqxyz.s = elastic_C_0[..., np.newaxis, np.newaxis, np.newaxis] * \
                                                         np.power(phase_field_at_quad_poits_1qnxyz, p)[0, :, 0, ...]

                if preconditioer == 'Green':
                    M_fun = M_fun_Green
                elif preconditioer == 'Jacobi':
                    K_diag_alg = discretization.get_preconditioner_Jacoby_fast(
                        material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz)
                    M_fun = lambda x: K_diag_alg * K_diag_alg * x
                elif preconditioer == 'Jacobi_Green':
                    K_diag_alg = discretization.get_preconditioner_Jacoby_fast(
                        material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz)
                    M_fun = lambda x: K_diag_alg * discretization.apply_preconditioner_NEW(
                        preconditioner_Fourier_fnfnqks=preconditioner_fnfnqks,
                        nodal_field_fnxyz=K_diag_alg * x)

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
                    rhs_load_case_inxyz = discretization.get_unknown_size_field(name='rhs_field_at_load_case')
                    rhs_load_case_inxyz.s.fill(0)
                    rhs_load_case_inxyz = discretization.get_rhs(
                        material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                        macro_gradient_field_ijqxyz=macro_gradient_fields[load_case],
                        rhs_inxyz=rhs_load_case_inxyz)
                    # if MPI.COMM_WORLD.size == 1:
                    #     print('rhs Of = {}'.format(np.linalg.norm(rhs_load_case)))
                    if preconditioer == 'Green':
                        displacement_field_load_case[load_case].s, norms = solvers.PCG(Afun=K_fun,
                                                                                       B=rhs_load_case_inxyz.s,
                                                                                       x0=displacement_field_load_case[
                                                                                           load_case].s,
                                                                                       P=M_fun,
                                                                                       steps=int(10000),
                                                                                       toler=1e-6,
                                                                                       norm_type='rr_rel',
                                                                                       # norm_metric=M_fun_Green
                                                                                       )
                    elif preconditioer == 'Jacobi':
                        displacement_field_load_case[load_case], norms = solvers.PCG(Afun=K_fun,
                                                                                     B=rhs_load_case_inxyz,
                                                                                     x0=None,
                                                                                     P=M_fun,
                                                                                     steps=int(10000),
                                                                                     toler=1e-6,
                                                                                     norm_type='rr',
                                                                                     # norm_metric=M_fun_Green
                                                                                     )
                        # displacement_field_load_case[                            load_case],
                    elif preconditioer == 'Jacobi_Green':
                        displacement_field_load_case[load_case].s, norms = solvers.PCG(Afun=K_fun,
                                                                                       B=rhs_load_case_inxyz.s,
                                                                                       x0=displacement_field_load_case[
                                                                                           load_case].s,
                                                                                       P=M_fun,
                                                                                       steps=int(10000),
                                                                                       toler=1e-14,
                                                                                       norm_type='rr_rel',
                                                                                       # norm_metric=M_fun_Green
                                                                                       )

                    if MPI.COMM_WORLD.rank == 0:
                        nb_it_comb = len(norms['residual_rz'])
                        norm_rz = norms['residual_rz'][-1]
                        norm_rr = norms['residual_rr'][-1]
                        num_iteration_.append(nb_it_comb)

                        print(
                            'load case ' f'{load_case},  nb_ steps CG of =' f'{nb_it_comb}, residual_rz = {norm_rz}, residual_rr = {norm_rr}')
                        # compute homogenized stress field corresponding t
                    homogenized_stresses[load_case] = discretization.get_homogenized_stress(
                        material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                        displacement_field_inxyz=displacement_field_load_case[load_case],
                        macro_gradient_field_ijqxyz=macro_gradient_fields[load_case],
                        formulation='small_strain')
                    # print('homogenized stress = \n'          ' {} '.format(homogenized_stresses[load_case] )) # good in MPI

                    # stress difference potential: actual_stress_ij is homogenized stress
                    # f_sigmas[load_case] = w * topology_optimization.compute_stress_equivalence_potential(
                    #     actual_stress_ij=homogenized_stresses[load_case],
                    #     target_stress_ij=target_stresses[load_case])
                    if energy_objective:
                        # strain_fluctuation_ijqxyz = discretization.apply_gradient_operator_symmetrized(
                        #    displacement_field_load_case[load_case])
                        # actual_strain_ijqxyz = macro_gradient_fields[load_case] + strain_fluctuation_ijqxyz
                        f_sigmas_energy[load_case] = (
                                w * topology_optimization.compute_elastic_energy_equivalence_potential(
                            discretization=discretization,
                            actual_stress_ij=homogenized_stresses[load_case],
                            target_stress_ij=target_stresses[load_case],
                            left_macro_gradient_ij=left_macro_gradients[load_case],
                            target_energy=target_energy[load_case]))

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
                        s_phase_field += s_energy_and_adjoint_load_cases[load_case]

                        f_sigmas_energy[load_case] += adjoint_energies[load_case]

                        objective_function += f_sigmas_energy[load_case]
                    else:
                        f_sigmas[load_case] = w * (topology_optimization.compute_stress_equivalence_potential(
                            actual_stress_ij=homogenized_stresses[load_case],
                            target_stress_ij=target_stresses[load_case]))
                        if MPI.COMM_WORLD.rank == 0:
                            print('w*f_sigmas  = '          ' {} '.format(f_sigmas[load_case]))  # good in MPI
                            print('sum of w*f_sigmas  = '          ' {} '.format(np.sum(f_sigmas)))

                        s_stress_and_adjoint_load_cases[load_case], adjoint_field_load_case[
                            load_case], adjoint_energies[
                            load_case] = topology_optimization.sensitivity_stress_and_adjoint_FE_NEW(
                            discretization=discretization,
                            base_material_data_ijkl=elastic_C_0,
                            displacement_field_fnxyz=displacement_field_load_case[load_case],
                            adjoint_field_last_step_fnxyz=adjoint_field_load_case[load_case],
                            macro_gradient_field_ijqxyz=macro_gradient_fields[load_case],
                            phase_field_1nxyz=phase_field_1nxyz,
                            target_stress_ij=target_stresses[load_case],
                            actual_stress_ij=homogenized_stresses[load_case],
                            preconditioner_fun=M_fun,
                            system_matrix_fun=K_fun,
                            formulation='small_strain',
                            p=p,
                            weight=w)
                        s_phase_field += s_stress_and_adjoint_load_cases[load_case]

                        f_sigmas[load_case] += adjoint_energies[load_case]

                        objective_function += f_sigmas[load_case]

                    if MPI.COMM_WORLD.rank == 0:
                        print(
                            'load case ' f'{load_case},  f_sigmas =' f'{f_sigmas[load_case]}')
                        print(
                            'load case ' f'{load_case},  objective_function =' f'{objective_function}')

                norms_sigma.append(objective_function)
                return objective_function[0], s_phase_field.reshape(-1)

if __name__ == '__main__':
    import os

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
    data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
    figure_folder_path = file_folder_path + '/figures/' + script_name + '/'

    if not os.path.exists(data_folder_path):
        os.makedirs(data_folder_path)

    if not os.path.exists(figure_folder_path):
        os.makedirs(figure_folder_path)

    run_adam = False
    run_lbfg = True
    random_initial_geometry = False
    bounds = False

    # fp = 'exp_data/muFFTTO_elasticity_random_init_N16_E_target_0.25_Poisson_-0.5_w0.01_eta0.01_p2_bounds=False_FE_NuMPI2.npy'
    # phase = np.load(fp)
    # # material distribution
    np.random.seed(MPI.COMM_WORLD.rank)
    phase_field_0 = discretization.get_scalar_field(name='phase_field_in_initial_')
    phase_field_0.s = np.random.rand(*phase_field_0.s.shape) ** 1
    # phase_field_0 = np.random.randint(0, high=2, size=discretization.get_scalar_sized_field().shape) ** 1
    # phase_field_0 = np.random.choice([0, 1], size=discretization.get_scalar_sized_field().shape,
    #                                  p=[0.5, 0.5])  # equal probability for 0 and 1

    if not random_initial_geometry:
        phase_field_0.s[0, 0] = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                                    microstructure_name='square_inclusion',
                                                                    coordinates=discretization.fft.coords)


    def apply_filter(phase):
        f_field = discretization.fft.fft(phase)
        # f_field[0, 0, np.logical_and(np.abs(discretization.fft.fftfreq[0]) > 0.25,
        #                              np.abs(discretization.fft.fftfreq[1]) > 0.25)] = 0
        f_field[0, 0, np.logical_or(discretization.fft.ifftfreq[0] > 8,
                                    discretization.fft.ifftfreq[1] > 8)] = 0
        # f_field[0, 0, 12:, 24:] = 0
        phase.s = discretization.fft.ifft(f_field) * discretization.fft.normalisation
        phase.s[phase.s > 1] = 1
        phase.s[phase.s < 0] = 0
        # min_ = discretization.mpi_reduction.min(phase)
        # max_ = discretization.mpi_reduction.max(phase)
        # phase = (phase + np.abs(min_)) / (max_ + np.abs(min_))
        return phase


    phase_field_0.s += 0.7 * np.random.rand(*phase_field_0.s.shape) ** 1
    phase_field_0 = apply_filter(phase_field_0)
    phase_field_00 = np.copy(phase_field_0)
    # my_sensitivity_pixel(phase_field_0).reshape([1, 1, *number_of_pixels])

    # print('Init objective function pixel  = {}'.format(my_objective_function_pixel(phase_field_00)))

    # name = 'exp_2D_elasticity_TO_indre_3exp_N1024_Et_0.15_Pt_-0.5_P0_0.0_w5.0_eta0.01_p2_mpi90_nlc_3_e_False'
    # phase_field = np.load(os.path.expanduser('~/exp_data/' + name + f'_it{8740}.npy'), allow_pickle=True)

    load_init_from_same_grid = False
    if load_init_from_same_grid:
        # file_data_name = f'eta_1muFFTTO_{problem_type}_random_init_N{number_of_pixels[0]}_E_target_{E_target}_Poisson_{poison_target}_Poisson0_{poison_0}_w{w_mult}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{MPI.COMM_WORLD.size}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}.npy'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')
        file_data_name = f'iteration_final'
        if MPI.COMM_WORLD.size == 1 or None:
            # phase = np.load(f'experiments/exp_data/init_phase_FE_N{number_of_pixels[0]}_NuMPI6.npy')
            # phase= np.load(f'experiments/exp_data/'  + file_data_name)
            phase = np.load(os.path.expanduser(data_folder_path + file_data_name + f'.npy'), allow_pickle=True)

            # phase = np.load(f'experiments/exp_data/' + file_data_name)
        else:

            # file_data_name = (
            #    f'1muFFTTO_elasticity_random_init_N{number_of_pixels[0]}_E_target_{E_target}_Poisson_{poison_target}_Poisson0_{poison_0}_w{w}_eta{eta_mult}_p{p}_bounds=False_FE_NuMPI{8}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}.npy')

            phase = load_npy(data_folder_path + file_data_name,
                             tuple(discretization.fft.subdomain_locations),
                             tuple(discretization.nb_of_pixels), MPI.COMM_WORLD)

        phase_field_0.s[0, 0] = phase  # [discretization.fft.subdomain_slices]

  # b


    if MPI.COMM_WORLD.size == 1:
        print('rank' f'{MPI.COMM_WORLD.rank:6} phase=' f'')
        plt.figure()
        plt.contourf(phase_field_0.s[0, 0], cmap=mpl.cm.Greys)
        # nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
        plt.clim(0, 1)
        plt.colorbar()

        plt.show()
    phase_field_0 = phase_field_0.s.ravel()
    print('Init objective function FE  = {}'.format(
        objective_function_multiple_load_cases(phase_field_00)[0]))



    if run_lbfg:

        norms_f = []
        norms_delta_f = []
        norms_max_grad_f = []
        norms_norm_grad_f = []
        norms_max_delta_x = []
        norms_norm_delta_x = []

        iterat = 0


        def my_callback(result_norms):
            global iterat
            iteration = result_norms[-1]
            norms_f.append(result_norms[0])
            norms_delta_f.append(result_norms[1])
            norms_max_grad_f.append(result_norms[2])
            norms_norm_grad_f.append(result_norms[3])
            norms_max_delta_x.append(result_norms[4])
            norms_norm_delta_x.append(result_norms[5])
            file_data_name_it = f'iteration{iterat}'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')

            save_npy(data_folder_path + file_data_name_it + f'.npy',
                     result_norms.reshape([*discretization.nb_of_pixels]),
                     tuple(discretization.fft.subdomain_locations),
                     tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD)
            iterat += 1
            if MPI.COMM_WORLD.size == 1:
                print(data_folder_path + file_data_name_it + f'.npy')


        xopt_FE_MPI = Optimization.l_bfgs(fun=objective_function_multiple_load_cases,
                                          x=phase_field_0,
                                          jac=True,
                                          maxcor=50,
                                          gtol=1e-6,
                                          ftol=1e-8,
                                          maxiter=1000,
                                          comm=MPI.COMM_WORLD,
                                          disp=True,
                                          # callback=my_callback
                                          )

        solution_phase = xopt_FE_MPI.x.reshape([1, 1, *discretization.nb_of_pixels])
        sensitivity_sol_FE_MPI = xopt_FE_MPI.jac.reshape([1, 1, *discretization.nb_of_pixels])

        if MPI.COMM_WORLD.size == 1:
            print('rank' f'{MPI.COMM_WORLD.rank:6} phase=' f' ')
            plt.figure()
            plt.contourf(solution_phase[0, 0], cmap=mpl.cm.Greys)
            # nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
            plt.clim(0, 1)
            plt.colorbar()
            plt.show()
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
        _info['num_iteration_'] = num_iteration_

        file_data_name = f'iteration_final'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')

        save_npy(data_folder_path + file_data_name + f'.npy', solution_phase[0, 0],
                 tuple(discretization.fft.subdomain_locations),
                 tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD)
        print(data_folder_path + file_data_name + f'.npy')
        ######## Postprocess for FE linear solver with NuMPI ########
        # material_data_field_C_0_rho_pixel = material_data_field_C_0[..., :, :] * np.power(phase_field_sol,
        #
        #                                                                             q
        #
        #                                                                             p)

        phase_field_at_quad_poits_1qnxyz = \
            discretization.evaluate_field_at_quad_points(nodal_field_fnxyz=solution_phase,
                                                         quad_field_fqnxyz=None,
                                                         quad_points_coords_iq=None)[0]
    # material_data_field_C_0_rho_quad = material_data_field_C_0[..., :, :, :] * np.power(
    #     phase_field_at_quad_poits_1qnxyz, p)[0, :, 0, ...]
    material_data_field_C_0_rho_quad = elastic_C_0[..., np.newaxis, np.newaxis, np.newaxis] * \
                                       np.power(phase_field_at_quad_poits_1qnxyz, p)[0, :, 0, ...]

    homogenized_stresses = np.zeros([nb_load_cases, dim, dim])

    for load_case in np.arange(nb_load_cases):
        # Set up the equilibrium system

        # Solve mechanical equilibrium constrain
        rhs_field_final = discretization.get_unknown_size_field(name='rhs_field_final')
        rhs_field_final.s.fill(0)

        rhs_field_final = discretization.get_rhs(
            material_data_field_ijklqxyz=material_data_field_C_0_rho_quad,
            macro_gradient_field_ijqxyz=macro_gradient_fields[load_case],
            rhs_inxyz=rhs_field_final)

        K_fun = lambda x: discretization.apply_system_matrix(material_data_field_C_0_rho_quad, x,
                                                             formulation='small_strain')
        M_fun = lambda x: discretization.apply_preconditioner_NEW(
            preconditioner_Fourier_fnfnqks=preconditioner_fnfnqks,
            nodal_field_fnxyz=x)
        displacement_field, norms = solvers.PCG(K_fun, rhs_field_final.s, x0=None, P=M_fun, steps=int(500),
                                                toler=1e-8)

        # compute homogenized stress field corresponding t
        homogenized_stresses[load_case] = discretization.get_homogenized_stress(
            material_data_field_ijklqxyz=material_data_field_C_0_rho_quad,
            displacement_field_inxyz=displacement_field,
            macro_gradient_field_ijqxyz=macro_gradient_fields[load_case],
            formulation='small_strain')

        _info['target_stress' + f'{load_case}'] = target_stresses[load_case]
        _info['homogenized_stresses' + f'{load_case}'] = homogenized_stresses[load_case]
        stress = np.einsum('ijkl,lk->ij', elastic_C_0, macro_gradients[load_case])
        print(f'target_stresses[load_case] = {target_stresses[load_case]}')
        print(f'homogenized_stresses[load_case]= {homogenized_stresses[load_case]}')
    quit()
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
                displacement_field_inxyz=displacement_field_ij,
                macro_gradient_field_ijqxyz=macro_gradient_field,
                formulation='small_strain')
    if MPI.COMM_WORLD.rank == 0:
        print('Optimized elastic tangent = \n {}'.format(
            domain.compute_Voigt_notation_4order(homogenized_C_ijkl)))

    _info['homogenized_C_ijkl'] = domain.compute_Voigt_notation_4order(homogenized_C_ijkl)
    _info['target_C_ijkl'] = domain.compute_Voigt_notation_4order(elastic_C_target)

    # np.save(folder_name + file_data_name+f'xopt_log.npz', xopt_FE_MPI)
    if MPI.COMM_WORLD.rank == 0:
        np.savez(folder_name + file_data_name + f'xopt_log.npz', **_info)
