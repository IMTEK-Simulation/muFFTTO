import numpy as np
import time
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

from NuMPI import Optimization
from NuMPI.IO import save_npy, load_npy

from mpi4py import MPI


plt.rcParams['text.usetex'] = True

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import topology_optimization

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'

domain_size = [1, 1]  # np.sqrt(3)/2
number_of_pixels = (32, 32)
dim = np.size(number_of_pixels)
pixel_size=np.asarray(domain_size)/np.asarray(number_of_pixels)

soft_phase_exponent = int(5)
preconditioner_type = "Green_Jacobi"#'Green'
eta =max(1*pixel_size)
weight =10.
cg_setup = {'cg_tol': 10 ** (-6)}


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

#base material
K_0, G_0 = 1, 0.5
E_0 = 9 * K_0 * G_0 / (3 * K_0 + G_0)

elastic_C_0 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                 K=K_0,
                                                 mu=G_0,
                                                 kind='linear')
if soft_phase_exponent == 0:
    soft_phase = 0
else:
    soft_phase = 10 ** (-soft_phase_exponent)
    # soft_phase = 0
elastic_C_void = elastic_C_0 * soft_phase

# Set up preconditioner
preconditioner_fnfnqks = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=elastic_C_0)

def M_fun_Green(x, Px):
    """
    Function to compute the product of the Preconditioner matrix with a vector.
    The Preconditioner is represented by the convolution operator.
    """
    discretization.fft.communicate_ghosts(x)
    discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner_fnfnqks,
                                               input_nodal_field_fnxyz=x,
                                               output_nodal_field_fnxyz=Px)
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

left_macro_gradients = np.zeros([nb_load_cases, dim, dim])
left_macro_gradients[0] = np.array([[.0, .0],
                                    [.0, 1.0]]) * macro_multip
left_macro_gradients[1] = np.array([[1.0, .0],
                                    [.0, .0]]) * macro_multip
left_macro_gradients[2] = np.array([[.0, .5],
                                    [0.5, 0.0]]) * macro_multip
if MPI.COMM_WORLD.rank == 0:
    print('macro_gradients = \n {}'.format(macro_gradients))

# Set up  macroscopic gradients
macro_gradient_field_ijqxyz = discretization.get_gradient_size_field(name=f'macro_gradient_field_')

for load_case in np.arange(nb_load_cases):
    stress = np.einsum('ijkl,lk->ij', elastic_C_0, macro_gradients[load_case])
    if MPI.COMM_WORLD.rank == 0:
        print('init_stress for load case {} = \n {}'.format(load_case, stress))

##### create target material data
poison_target = -0.5
G_target_auxet = (3 / 20) * E_0
E_target = 2 * G_target_auxet * (1 + poison_target)

K_targer, G_target = domain.get_bulk_and_shear_modulus(E=E_target,
                                                       poison=poison_target)

elastic_C_target = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                      K=K_targer,
                                                      mu=G_target,
                                                      kind='linear')
if MPI.COMM_WORLD.rank == 0:
    print('Target elastic tangent = \n {}'.format(domain.compute_Voigt_notation_4order(elastic_C_target)))
##### create target stresses
target_stresses = np.zeros([nb_load_cases, dim, dim])
target_energy = np.zeros([nb_load_cases])

for load_case in np.arange(nb_load_cases):
    target_stresses[load_case] = np.einsum('ijkl,lk->ij', elastic_C_target, macro_gradients[load_case])
    target_energy[load_case] = np.einsum('ij,ijkl,lk->', left_macro_gradients[load_case], elastic_C_target,
                                         macro_gradients[load_case])
    if MPI.COMM_WORLD.rank == 0:
        print('target_stress for load case {} = \n {}'.format(load_case, target_stresses[load_case]))
        print('target stress norm for load case {} = \n {}'.format(load_case,
                                                                   np.sum(target_stresses[load_case] ** 2)))
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
norms_adjoint_energy = []

info_mech = {}
info_mech['num_iteration_adjoint'] = []
info_mech['residual_rz'] = []

info_adjoint = {}
info_adjoint['num_iteration_adjoint'] = []
info_adjoint['residual_rz'] = []

phase_field_1nxyz = discretization.get_scalar_field(name='phase_field_in_objective')
phase_field_at_quad_poits_1qxyz = discretization.get_quad_field_scalar(
    name='phase_field_at_quads_in_objective_function_multiple_load_cases')
material_data_field_C_0_rho_ijklqxyz = discretization.get_material_data_size_field_mugrid(
    name='material_data_field_C_0_rho_ijklqxyz_in_objective')
s_sensitivity_field = discretization.get_scalar_field(name='s_phase_field')
rhs_load_case_inxyz = discretization.get_unknown_size_field(name='rhs_field_at_load_case')
s_stress_and_adjoint_load_case = discretization.get_scalar_field(
    name='s_stress_and_adjoint_load_case')

pixel_diameter = np.sqrt(np.sum(discretization.pixel_size ** 2))
w = weight / nb_load_cases

if MPI.COMM_WORLD.rank == 0:
    print('p =   {}'.format(p))
    print('w  =  {}'.format(w))
    print('eta =  {}'.format(eta))

def objective_function_multiple_load_cases(phase_field_1nxyz_flat):
    # print('Objective function:')
    # reshape the field
    # zero_small_phases = False
    # if zero_small_phases:
    #     phase_field_1nxyz[phase_field_1nxyz < 1e-5] = 0
    disp = False
    phase_field_1nxyz.s[...] = phase_field_1nxyz_flat.reshape([1, 1, *discretization.nb_of_pixels])

    # Phase field  in quadrature points
    discretization.apply_N_operator_mugrid(phase_field_1nxyz, phase_field_at_quad_poits_1qxyz)

    # Material data in quadrature points
    material_data_field_C_0_rho_ijklqxyz.s[...] = (elastic_C_0 - elastic_C_void)[
                                                 ..., np.newaxis, np.newaxis, np.newaxis] * \
                                             np.power(phase_field_at_quad_poits_1qxyz.s, p)[0, 0, :, ...] + \
                                             elastic_C_void[..., np.newaxis, np.newaxis, np.newaxis]

    # objective function phase field terms
    f_phase_field = topology_optimization.objective_function_phase_field(discretization=discretization,
                                                                         phase_field_1nxyz=phase_field_1nxyz,
                                                                         eta=eta,
                                                                         double_well_depth=double_well_depth_test)
    #  sensitivity phase field terms
    s_sensitivity_field.s.fill(0)

    topology_optimization.sensitivity_phase_field_term_FE_NEW(
        discretization=discretization,
        base_material_data_ijkl=elastic_C_0,
        void_material_data_ijkl=elastic_C_void,
        phase_field_1nxyz=phase_field_1nxyz,
        p=p,
        eta=eta,
        output_array=s_sensitivity_field,
        double_well_depth=1)
    # ??????
    objective_function = f_phase_field

    norms_pf.append(objective_function)

    if preconditioner_type == 'Green':
        M_fun = M_fun_Green
    elif preconditioner_type == 'Jacobi':
        K_diag_alg = discretization.get_preconditioner_Jacobi_mugrid(
            material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz)

        def M_fun_Jacobi(x, Px):
            Px.s[...] = K_diag_alg.s * K_diag_alg.s * x.s
            discretization.fft.communicate_ghosts(Px)

        M_fun = M_fun_Jacobi

    elif preconditioner_type == 'Green_Jacobi':
        K_diag_alg = discretization.get_preconditioner_Jacobi_mugrid(
            material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz)

        def M_fun_Green_Jacobi(x, Px):
            discretization.fft.communicate_ghosts(x)
            x_jacobi_temp = discretization.get_unknown_size_field(name='x_jacobi_temp')

            x_jacobi_temp.s[...] = K_diag_alg.s * x.s
            discretization.apply_preconditioner_mugrid(
                preconditioner_Fourier_fnfnqks=preconditioner_fnfnqks,
                input_nodal_field_fnxyz=x_jacobi_temp,
                output_nodal_field_fnxyz=Px)

            Px.s[...] = K_diag_alg.s * Px.s
            discretization.fft.communicate_ghosts(Px)

        M_fun = M_fun_Green_Jacobi

    def K_fun(x, Ax):
        discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0_rho_ijklqxyz,
                                                  input_field_inxyz=x,
                                                  output_field_inxyz=Ax,
                                                  formulation='small_strain')

    # Solve mechanical equilibrium constrain
    homogenized_stresses = np.zeros([nb_load_cases, dim, dim])

    f_sigmas = np.zeros([nb_load_cases, 1])
    adjoint_energies = np.zeros([nb_load_cases, 1])
    norm_sigma_step = 0
    adjoint_energies_step = 0
    for load_case in np.arange(nb_load_cases):

        discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradients[load_case],
                                                       macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz)

        rhs_load_case_inxyz.s.fill(0)
        discretization.get_rhs_mugrid(
            material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
            macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
            rhs_inxyz=rhs_load_case_inxyz)

        if MPI.COMM_WORLD.rank == 0:
            norms_cg_mech = dict()
            norms_cg_mech['residual_rr'] = []
            norms_cg_mech['residual_rz'] = []

        def callback(it, x, r, p, z, stop_crit_norm):
            # global norms_cg_mech
            norm_of_rr = discretization.communicator.sum(np.dot(r.ravel(), r.ravel()))
            norm_of_rz = discretization.communicator.sum(np.dot(r.ravel(), z.ravel()))
            if MPI.COMM_WORLD.rank == 0:
                norms_cg_mech['residual_rr'].append(norm_of_rr)
                norms_cg_mech['residual_rz'].append(norm_of_rz)

        solvers.conjugate_gradients_mugrid(
            comm=discretization.communicator,
            fc=discretization.field_collection,
            hessp=K_fun,  # linear operator
            b=rhs_load_case_inxyz,
            x=displacement_field_load_case[load_case],
            P=M_fun,
            tol=cg_setup['cg_tol'],
            maxiter=10000,
            callback=callback,
            # norm_metric=res_norm
        )

        if MPI.COMM_WORLD.rank == 0:
            nb_it = len(norms_cg_mech['residual_rr'])
            try:
                norm_rz = norms_cg_mech['residual_rz'][-1]
                norm_rr = norms_cg_mech['residual_rr'][-1]
            except:
                norm_rz = 0
                norm_rr = 0
            info_mech['num_iteration_adjoint'].append(nb_it)
            # info_mech['residual_rz'].append(norms_cg_mech['residual_rz'])

            print(
                'load case ' f'{load_case},  nb_ steps CG mech    =' f'{nb_it}, residual_rz = {norm_rz}, residual_rr = {norm_rr}')
            del norms_cg_mech
            #gc.collect()  # forces garbage collection
            # compute homogenized stress field corresponding t
        homogenized_stresses[load_case] = discretization.get_homogenized_stress_mugrid(
            material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
            displacement_field_inxyz=displacement_field_load_case[load_case],
            macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
            formulation='small_strain')

        f_sigmas[load_case] = topology_optimization.compute_stress_equivalence_potential(
            actual_stress_ij=homogenized_stresses[load_case],
            target_stress_ij=target_stresses[load_case])

        s_stress_and_adjoint_load_case.s[0, 0], adjoint_field_load_case[
            load_case], adjoint_energies[
            load_case], info_adjoint_current = topology_optimization.sensitivity_stress_and_adjoint_FE_NEW(
            discretization=discretization,
            base_material_data_ijkl=elastic_C_0,
            void_material_data_ijkl=elastic_C_void,
            displacement_field_inxyz=displacement_field_load_case[load_case],
            adjoint_field_inxyz=adjoint_field_load_case[load_case],
            macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
            phase_field_1nxyz=phase_field_1nxyz,
            target_stress_ij=target_stresses[load_case],
            actual_stress_ij=homogenized_stresses[load_case],
            preconditioner_fun=M_fun,
            system_matrix_fun=K_fun,
            formulation='small_strain',
            p=p,
            weight=w,
            disp=disp,
            **cg_setup)

        s_sensitivity_field.s[0, 0] += s_stress_and_adjoint_load_case.s[0, 0]

        objective_function += w * f_sigmas[load_case]
        objective_function += adjoint_energies[load_case]
        if MPI.COMM_WORLD.rank == 0:
            info_adjoint['num_iteration_adjoint'].append(info_adjoint_current['num_iteration_adjoint'])
            nb_it_adjoint = info_adjoint_current['num_iteration_adjoint']
            print(
                'load case ' f'{load_case},  nb_steps CG adjoint  =' f'{nb_it_adjoint} ')
            if disp:
                print(
                    'load case ' f'{load_case},  f_sigmas =' f'{f_sigmas[load_case]}')
                print(
                    'load case ' f'{load_case},  objective_function =' f'{objective_function}')
        norm_sigma_step += f_sigmas[load_case]
        adjoint_energies_step += adjoint_energies[load_case]
    discretization.fft.communicate_ghosts(s_sensitivity_field)
    norms_sigma.append(norm_sigma_step)
    norms_adjoint_energy.append(adjoint_energies_step)

    return objective_function[0], s_sensitivity_field.s[0, 0].reshape(-1)


if __name__ == '__main__':

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
    data_folder_path = file_folder_path + '/data/' + script_name + '/'
    figure_folder_path = file_folder_path + '/figures/' + script_name + '/'

    random_init=True

    if MPI.COMM_WORLD.rank == 0:
        if not os.path.exists(data_folder_path):
            os.makedirs(data_folder_path)

        if not os.path.exists(figure_folder_path):
            os.makedirs(figure_folder_path)


    def apply_filter(phase):
        f_field = discretization.ffield_collection.complex_field(
            name='f_field_phase_for_filter',  # name of the field
            components=(1,))
        discretization.fft.fft(phase, f_field)
        f_field.s[0, 0, np.logical_or(discretization.fft.ifftfreq[0] > 8,
                                      discretization.fft.ifftfreq[1] > 8)] = 0
        discretization.fft.ifft(f_field, phase)
        phase.s[...] *= discretization.fft.normalisation
        phase.s[phase.s > 1] = 1
        phase.s[phase.s < 0] = 0


    phase_field_0 = discretization.get_scalar_field(name='phase_field_in_initial_')


    if random_init:
        #np.random.seed(MPI.COMM_WORLD.rank)
        phase_field_0.s[...] += 1.0 * np.random.rand(*phase_field_0.s.shape) ** 1
    else:
        phase_field_0.s[0, 0] = (np.sin(discretization.fft.coords[0] * 4 * np.pi) + np.sin(
            discretization.fft.coords[1] * 4 * np.pi) + 2) / 4
        phase_field_0.s[...] += 0.5 * np.random.rand(*phase_field_0.s.shape) ** 1

    #apply_filter(phase_field_0)

    norms_f = []
    norms_delta_f = []
    norms_max_grad_f = []
    norms_norm_grad_f = []
    norms_max_delta_x = []
    norms_norm_delta_x = []


    _info_iter = {}
    iterat=0

    def my_callback(result_norms):
        global iterat


        iterat += 1
        plt.figure()
        plt.pcolormesh(discretization.fft.coords[0],
                       discretization.fft.coords[1],
            result_norms.reshape([*discretization.nb_of_pixels]), cmap=mpl.cm.Greys)
        # nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
        plt.clim(0, 1)
        plt.title(f'Iteration {iterat}')
        plt.colorbar()
        plt.show()


    xopt_FE_MPI = Optimization.l_bfgs(fun=objective_function_multiple_load_cases,
                                      x=phase_field_0.s.ravel(),
                                      jac=True,
                                      maxcor=20,
                                      gtol=1e-3,
                                      ftol=1e-5,
                                      maxiter=1000,
                                      comm=MPI.COMM_WORLD,
                                      disp=True,
                                      callback=my_callback
                                      )
    solution_phase = discretization.get_scalar_field(name='phase_field_in_initial_')

    solution_phase.s[...] = xopt_FE_MPI.x.reshape([1, 1, *discretization.nb_of_pixels])
    sensitivity_sol_FE_MPI = xopt_FE_MPI.jac.reshape([1, 1, *discretization.nb_of_pixels])

    _info = {}
    if MPI.COMM_WORLD.rank == 0:
        _info["num_iteration_mech"] = np.array(info_mech["num_iteration_adjoint"], dtype=object)
        _info["residual_rz_mech"] = np.array(info_mech["residual_rz"], dtype=object)

        _info["num_iteration_adjoint"] = np.array(info_adjoint["num_iteration_adjoint"], dtype=object)
        _info["residual_rz"] = np.array(info_adjoint["residual_rz"], dtype=object)

    _info['nb_of_pixels'] = discretization.nb_of_pixels_global
    # phase_field_sol_FE_MPI = xopt.x.reshape([1, 1, *discretization.nb_of_pixels])
    # _info['norms_f'] = norms_f
    # _info['norms_delta_f'] = norms_delta_f
    # _info['norms_max_grad_f'] = norms_max_grad_f
    # _info['norms_norm_grad_f'] = norms_norm_grad_f
    # _info['norms_max_delta_x'] = norms_max_delta_x
    # _info['norms_norm_delta_x'] = norms_norm_delta_x
    _info['norms_sigma'] = norms_sigma
    _info['norms_pf'] = norms_pf
    _info['norms_adjoint_energy'] = norms_adjoint_energy

    _info['nb_iterations'] = iterat

    file_data_name = f'_eta_{eta}' + f'_w_{weight}' + f'_final'
    # print('rank' f'{MPI.COMM_WORLD.rank:6} ')
    save_npy(data_folder_path + f'{preconditioner_type}' + file_data_name + f'.npy',
             solution_phase.s[0].mean(axis=0),
             tuple(discretization.subdomain_locations_no_buffers),
             tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD)
    if MPI.COMM_WORLD.rank == 0:
        print(data_folder_path + file_data_name + f'.npy')
    ######## Postprocess for FE linear solver with NuMPI ########
    # material_data_field_C_0_rho_pixel = material_data_field_C_0[..., :, :] * np.power(phase_field_sol,
    #
    #                                                                             q
    #
    #                                                                             p)
    solution_phase_at_quad_poits_1qxyz = discretization.get_quad_field_scalar(
        name='solution_phase_at_quad_poits_1qxyz')
    discretization.apply_N_operator_mugrid(solution_phase, solution_phase_at_quad_poits_1qxyz)

    #     phase_field_at_quad_poits_1qnxyz, p)[0, :, 0, ...]
    material_data_field_C_0_rho_quad = discretization.get_material_data_size_field_mugrid(
        name='material_data_field_C_0_rho_quad')
    material_data_field_C_0_rho_quad.s[...] = (elastic_C_0 - elastic_C_void)[..., np.newaxis, np.newaxis, np.newaxis] * \
                                              np.power(solution_phase_at_quad_poits_1qxyz.s, p)[0, 0, :, ...] + \
                                              elastic_C_void[..., np.newaxis, np.newaxis, np.newaxis]

    homogenized_stresses = np.zeros([nb_load_cases, dim, dim])

    for load_case in np.arange(nb_load_cases):
        # Set up the equilibrium system

        # Solve mechanical equilibrium constrain
        rhs_field_final = discretization.get_unknown_size_field(name='rhs_field_final')
        rhs_field_final.s.fill(0)

        discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradients[load_case],
                                                       macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz)

        discretization.get_rhs_mugrid(
            material_data_field_ijklqxyz=material_data_field_C_0_rho_quad,
            macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
            rhs_inxyz=rhs_field_final)


        def K_fun(x, Ax):
            discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0_rho_quad,
                                                      input_field_inxyz=x,
                                                      output_field_inxyz=Ax,
                                                      formulation='small_strain')


        M_fun = M_fun_Green

        displacement_field = discretization.get_unknown_size_field(name='displacement_field_postprocess')
        # displacement_field.s, norms = solvers.PCG_numpy(K_fun, rhs_field_final.s, x0=None, P=M_fun, steps=int(500),
        #                                                 toler=1e-8)
        solvers.conjugate_gradients_mugrid(
            comm=discretization.communicator,
            fc=discretization.field_collection,
            hessp=K_fun,  # linear operator
            b=rhs_field_final,
            x=displacement_field,
            P=M_fun,
            tol=1e-5,
            maxiter=10000,
        )
        # compute homogenized stress field corresponding t
        homogenized_stresses[load_case] = discretization.get_homogenized_stress_mugrid(
            material_data_field_ijklqxyz=material_data_field_C_0_rho_quad,
            displacement_field_inxyz=displacement_field,
            macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
            formulation='small_strain')

        _info['target_stress' + f'{load_case}'] = target_stresses[load_case]
        _info['homogenized_stresses' + f'{load_case}'] = homogenized_stresses[load_case]
        stress = np.einsum('ijkl,lk->ij', elastic_C_0, macro_gradients[load_case])
        if MPI.COMM_WORLD.rank == 0:
            print(f'target_stresses[load_case] = {target_stresses[load_case]}')
            print(f'homogenized_stresses[load_case]= {homogenized_stresses[load_case]}')

    homogenized_C_ijkl = np.zeros(np.array(4 * [dim, ]))
    macro_gradient_field_ijqxyz = discretization.get_gradient_size_field(name=f'macro_gradient_field_ij')
    rhs_field_final = discretization.get_unknown_size_field(name='rhs_field_final')

    # compute whole homogenized elastic tangent
    for i in range(2):
        for j in range(2):
            # set macroscopic gradient
            macro_gradient_ij = np.zeros([dim, dim])
            macro_gradient_ij[i, j] = 1
            # Set up right hand side
            # macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient_ij)
            discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient_ij,
                                                           macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz)
            # Solve mechanical equilibrium constrain
            # rhs_ij = discretization.get_rhs(material_data_field_C_0_rho_quad, macro_gradient_field)
            discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0_rho_quad,
                                          macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
                                          rhs_inxyz=rhs_field_final)

            solvers.conjugate_gradients_mugrid(
                comm=discretization.communicator,
                fc=discretization.field_collection,
                hessp=K_fun,  # linear operator
                b=rhs_field_final,
                x=displacement_field,
                P=M_fun,
                tol=1e-5,
                maxiter=10000,
            )
            # ----------------------------------------------------------------------
            # compute homogenized stress field corresponding
            homogenized_C_ijkl[i, j] = discretization.get_homogenized_stress_mugrid(
                material_data_field_ijklqxyz=material_data_field_C_0_rho_quad,
                displacement_field_inxyz=displacement_field,
                macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
                formulation='small_strain')
    if MPI.COMM_WORLD.rank == 0:
        print('Optimized elastic tangent = \n {}'.format(
            domain.compute_Voigt_notation_4order(homogenized_C_ijkl)))

    _info['homogenized_C_ijkl'] = domain.compute_Voigt_notation_4order(homogenized_C_ijkl)
    _info['target_C_ijkl'] = domain.compute_Voigt_notation_4order(elastic_C_target)

    # np.save(folder_name + file_data_name+f'xopt_log.npz', xopt_FE_MPI)
    if MPI.COMM_WORLD.rank == 0:
        np.savez(data_folder_path + f'{preconditioner_type}' + f'_eta_{eta}' + f'_w_{weight}' + f'_log.npz',
                 **_info)  # + f'_its_{start}_{start + iterat}'
