import numpy as np
import scipy as sp
import matplotlib as mpl
import time
import sys
import argparse
import gc

sys.path.append("/home/martin/Programming/muFFTTO_paralellFFT_test/muFFTTO")
sys.path.append('../..')  # Add parent directory to path

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

parser = argparse.ArgumentParser(
    prog="exp_paper_JG_2D_elasticity_TO.py", description="Solve topology optimization of negative poison ratio"
)
parser.add_argument("-n", "--nb_pixels", default="32")

# Preconditioner type (string, choose from a set)
parser.add_argument(
    "-p", "--preconditioner_type",
    type=str,
    choices=["Green", "Jacobi", "Green_Jacobi"],  # example options
    default="Green_Jacobi",
    help="Type of preconditioner to use"
)
parser.add_argument(
    "-s", "--save_phases",
    action="store_true",
    help="Enable saving phases"
)
# Total phase contrast (integer)

args = parser.parse_args()
nb_pixels = int(args.nb_pixels)
preconditioner_type = args.preconditioner_type
save_data = args.save_phases

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'  # 'bilinear_rectangle'##'linear_triangles' # # linear_triangles_tilled
formulation = 'small_strain'

domain_size = [1, 1]  #
number_of_pixels = (nb_pixels, nb_pixels)
print(number_of_pixels)
dim = np.size(number_of_pixels)
my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)
start_time = time.time()

if MPI.COMM_WORLD.rank == 0:
    print('number_of_pixels = \n {} core {}'.format(number_of_pixels, MPI.COMM_WORLD.rank))
    print('  Rank   Size          Domain       Subdomain        Location')
    print('  ----   ----          ------       ---------        --------')
MPI.COMM_WORLD.Barrier()  # Barrier so header is printed first

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)
if MPI.COMM_WORLD.rank == 0:
    print(f'{MPI.COMM_WORLD.rank:6} {MPI.COMM_WORLD.size:6} {str(discretization.fft.nb_domain_grid_pts):>15} '
          f'{str(discretization.fft.nb_subdomain_grid_pts):>15} {str(discretization.fft.subdomain_locations):>15}')

# start_time =  MPI.Wtime()

# create material data of solid phase rho=1
K_0, G_0 = 1, 0.5
E_0 = 9 * K_0 * G_0 / (3 * K_0 + G_0)
poison_0 = (3 * K_0 - 2 * G_0) / (2 * (3 * K_0 + G_0))
# poison_0 = 0.2
# G_0 = E_0 / (2 * (1 + poison_0))
# K_0, G_0 = domain.get_bulk_and_shear_modulus(E=E_0, poison=poison_0)
# K_0, G_0 = 1, 0.5
# print('1 = \n   core {}'.format(MPI.COMM_WORLD.rank))

elastic_C_0 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                 K=K_0,
                                                 mu=G_0,
                                                 kind='linear')
# if MPI.COMM_WORLD.rank == 0:
# print('2 = \n   core {}'.format(MPI.COMM_WORLD.rank))
# # material_data_field_C_0 = np.einsum('ijkl,qxy->ijklqxy', elastic_C_0,
# #                                     np.ones(np.array([discretization.nb_quad_points_per_pixel,
# #                                                       *discretization.nb_of_pixels])))
# print('3 = \n   core {}'.format(MPI.COMM_WORLD.rank))
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
macro_multip = 0.01
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
#macro_gradient_fields = []
macro_gradient_field_ijqxyz = discretization.get_gradient_size_field(name=f'macro_gradient_field_')

for load_case in np.arange(nb_load_cases):
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
info_mech = {}
info_mech['num_iteration_adjoint'] = []
info_mech['residual_rz'] = []

info_adjoint = {}
info_adjoint['num_iteration_adjoint'] = []
info_adjoint['residual_rz'] = []
weights = [5]  # np.concatenate([np.arange(0.1, 2., 1)])

w_mult = 5
eta_mult = 0.01
pixel_diameter = np.sqrt(np.sum(discretization.pixel_size ** 2))
w = w_mult / nb_load_cases
eta = eta_mult
if MPI.COMM_WORLD.rank == 0:
    print('p =   {}'.format(p))
    print('w  =  {}'.format(w))
    print('eta =  {}'.format(eta))
phase_field_1nxyz = discretization.get_scalar_field(name='phase_field_in_objective')
phase_field_at_quad_poits_1qxyz = discretization.get_quad_field_scalar(
    name='phase_field_at_quads_in_objective_function_multiple_load_cases')
material_data_field_C_0_rho_ijklqxyz = discretization.get_material_data_size_field_mugrid(
    name='material_data_field_C_0_rho_ijklqxyz_in_objective')
s_sensitivity_field = discretization.get_scalar_field(name='s_phase_field')
rhs_load_case_inxyz = discretization.get_unknown_size_field(name='rhs_field_at_load_case')
s_stress_and_adjoint_load_case = discretization.get_scalar_field(
    name='s_stress_and_adjoint_load_case')
cg_setup = {'cg_tol': 1e-8}


def objective_function_multiple_load_cases(phase_field_1nxyz_flat):
    # print('Objective function:')
    # reshape the field
    # zero_small_phases = False
    # if zero_small_phases:
    #     phase_field_1nxyz[phase_field_1nxyz < 1e-5] = 0
    disp = False
    phase_field_1nxyz.s = phase_field_1nxyz_flat.reshape([1, 1, *discretization.nb_of_pixels])

    # Phase field  in quadrature points
    discretization.apply_N_operator_mugrid(phase_field_1nxyz, phase_field_at_quad_poits_1qxyz)

    # Material data in quadrature points
    material_data_field_C_0_rho_ijklqxyz.s = elastic_C_0[..., np.newaxis, np.newaxis, np.newaxis] * \
                                             np.power(phase_field_at_quad_poits_1qxyz.s, p)[0, 0, :, ...]

    # objective function phase field terms
    f_phase_field = topology_optimization.objective_function_phase_field(discretization=discretization,
                                                                         phase_field_1nxyz=phase_field_1nxyz,
                                                                         eta=eta,
                                                                         double_well_depth=double_well_depth_test)
    #  sensitivity phase field terms
    s_sensitivity_field.s.fill(0)

    s_sensitivity_field.s[0, 0] = topology_optimization.sensitivity_phase_field_term_FE_NEW(
        discretization=discretization,
        base_material_data_ijkl=elastic_C_0,
        phase_field_1nxyz=phase_field_1nxyz,
        p=p,
        eta=eta,
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
            Px.s = K_diag_alg.s * K_diag_alg.s * x.s
            discretization.fft.communicate_ghosts(Px)

        M_fun = M_fun_Jacobi

    elif preconditioner_type == 'Green_Jacobi':
        K_diag_alg = discretization.get_preconditioner_Jacobi_mugrid(
            material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz)

        def M_fun_Green_Jacobi(x, Px):
            discretization.fft.communicate_ghosts(x)
            x_jacobi_temp = discretization.get_unknown_size_field(name='x_jacobi_temp')

            x_jacobi_temp.s = K_diag_alg.s * x.s
            discretization.apply_preconditioner_mugrid(
                preconditioner_Fourier_fnfnqks=preconditioner_fnfnqks,
                input_nodal_field_fnxyz=x_jacobi_temp,
                output_nodal_field_fnxyz=Px)

            Px.s = K_diag_alg.s * Px.s
            discretization.fft.communicate_ghosts(Px)

        M_fun = M_fun_Green_Jacobi

        # M_fun = lambda x: K_diag_alg * discretization.apply_preconditioner_NEW(
        #     preconditioner_Fourier_fnfnqks=preconditioner_fnfnqks,
        #     nodal_field_fnxyz=K_diag_alg * x)

    # K_fun = lambda x: discretization.apply_system_matrix(
    #     material_data_field=material_data_field_C_0_rho_ijklqxyz,
    #     displacement_field=x,
    #     formulation='small_strain')

    def K_fun(x, Ax):
        discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0_rho_ijklqxyz,
                                                  input_field_inxyz=x,
                                                  output_field_inxyz=Ax,
                                                  formulation='small_strain')

    # Solve mechanical equilibrium constrain
    homogenized_stresses = np.zeros([nb_load_cases, dim, dim])

    f_sigmas = np.zeros([nb_load_cases, 1])
    # f_sigmas_energy = np.zeros([nb_load_cases, 1])
    adjoint_energies = np.zeros([nb_load_cases, 1])
    # s_stress_and_adjoint_load_cases = np.zeros([nb_load_cases, *s_phase_field.shape])
    # s_energy_and_adjoint_load_cases = np.zeros([nb_load_cases, *s_phase_field.shape])

    for load_case in np.arange(nb_load_cases):

        discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradients[load_case],
                                                       macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz)


        rhs_load_case_inxyz.s.fill(0)
        discretization.get_rhs_mugrid(
            material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
            macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
            rhs_inxyz=rhs_load_case_inxyz)

        norms_cg_mech = dict()
        norms_cg_mech['residual_rr'] = []
        norms_cg_mech['residual_rz'] = []

        def callback(it, x, r, p, z, stop_crit_norm):
            # global norms_cg_mech
            norm_of_rr = discretization.fft.communicator.sum(np.dot(r.ravel(), r.ravel()))
            norm_of_rz = discretization.fft.communicator.sum(np.dot(r.ravel(), z.ravel()))
            norms_cg_mech['residual_rr'].append(norm_of_rr)
            norms_cg_mech['residual_rz'].append(norm_of_rz)

        solvers.conjugate_gradients_mugrid(
            comm=discretization.fft.communicator,
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
                'load case ' f'{load_case},  nb_ steps CG of =' f'{nb_it}, residual_rz = {norm_rz}, residual_rr = {norm_rr}')
        del norms_cg_mech
        gc.collect()  # forces garbage collection
            # compute homogenized stress field corresponding t
        homogenized_stresses[load_case] = discretization.get_homogenized_stress_mugrid(
            material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
            displacement_field_inxyz=displacement_field_load_case[load_case],
            macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
            formulation='small_strain')
        # print('homogenized stress = \n'          ' {} '.format(homogenized_stresses[load_case] )) # good in MPI

        # stress difference potential: actual_stress_ij is homogenized stress
        # f_sigmas[load_case] = w * topology_optimization.compute_stress_equivalence_potential(
        #     actual_stress_ij=homogenized_stresses[load_case],
        #     target_stress_ij=target_stresses[load_case])

        f_sigmas[load_case] = topology_optimization.compute_stress_equivalence_potential(
            actual_stress_ij=homogenized_stresses[load_case],
            target_stress_ij=target_stresses[load_case])
        # if MPI.COMM_WORLD.rank == 0:
        # print('w*f_sigmas  = '          ' {} '.format(f_sigmas[load_case]))  # good in MPI
        # print('sum of w*f_sigmas  = '          ' {} '.format(np.sum(f_sigmas)))

        s_stress_and_adjoint_load_case.s[0, 0], adjoint_field_load_case[
            load_case], adjoint_energies[
            load_case], info_adjoint_current = topology_optimization.sensitivity_stress_and_adjoint_FE_NEW(
            discretization=discretization,
            base_material_data_ijkl=elastic_C_0,
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

        info_adjoint['num_iteration_adjoint'].append(info_adjoint_current['num_iteration_adjoint'])
        # info_adjoint['residual_rz'].append(info_adjoint_current['residual_rz'])

        if disp:
            if MPI.COMM_WORLD.rank == 0:
                print(
                    'load case ' f'{load_case},  f_sigmas =' f'{f_sigmas[load_case]}')
                print(
                    'load case ' f'{load_case},  objective_function =' f'{objective_function}')

    discretization.fft.communicate_ghosts(s_sensitivity_field)
    norms_sigma.append(objective_function)
    return objective_function[0], s_sensitivity_field.s[0, 0].reshape(-1)


if __name__ == '__main__':
    import os

    script_name = os.path.splitext(os.path.basename(__file__))[0] + f'_N_{number_of_pixels[0]}'
    file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
    data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
    figure_folder_path = file_folder_path + '/figures/' + script_name + '/'

    if not os.path.exists(data_folder_path):
        os.makedirs(data_folder_path)

    if not os.path.exists(figure_folder_path):
        os.makedirs(figure_folder_path)

    # log_name = 'final_log'
    # _info=np.load(data_folder_path + log_name + f'info_log.npz',  allow_pickle=True)

    run_adam = False
    run_lbfg = True
    random_initial_geometry = False
    bounds = False

    # fp = 'exp_data/muFFTTO_elasticity_random_init_N16_E_target_0.25_Poisson_-0.5_w0.01_eta0.01_p2_bounds=False_FE_NuMPI2.npy'
    # phase = np.load(fp)
    np.random.seed(MPI.COMM_WORLD.rank)
    phase_field_0 = discretization.get_scalar_field(name='phase_field_in_initial_')
    if not random_initial_geometry:
        geometry_ID = 'circle_inclusions'
        nb_circles = 3  # number of circles in single dimention
        vol_frac = 1 / 4
        r_0 = np.sqrt(vol_frac / np.pi)
        _geom_parameters = {'nb_circles': nb_circles,
                            'r_0': r_0,
                            'vol_frac': vol_frac,
                            'random_density': True,
                            'random_centers': True}
        phase_field_0.s[0, 0] = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                                    microstructure_name=geometry_ID,
                                                                    coordinates=discretization.fft.coords,
                                                                    **_geom_parameters)

        phase_field_0.s[phase_field_0.s > 1] = 1
        phase_field_0.s[phase_field_0.s < 0] = 0
        phase_field_0.s[0, 0] = abs(phase_field_0.s[0, 0] - 1)


    # # material distribution
    def apply_filter(phase):
        f_field = discretization.fft.fourier_space_field(
            unique_name='f_field_phase_for_filter',  # name of the field
            shape=(1,))
        discretization.fft.fft(phase, f_field)
        # f_field[0, 0, np.logical_and(np.abs(discretization.fft.fftfreq[0]) > 0.25,
        #                              np.abs(discretization.fft.fftfreq[1]) > 0.25)] = 0
        f_field.s[0, 0, np.logical_or(discretization.fft.ifftfreq[0] > 8,
                                      discretization.fft.ifftfreq[1] > 8)] = 0
        # f_field[0, 0, 12:, 24:] = 0
        discretization.fft.ifft(f_field, phase)
        phase.s *= discretization.fft.normalisation
        phase.s[phase.s > 1] = 1
        phase.s[phase.s < 0] = 0
        # min_ = discretization.mpi_reduction.min(phase)
        # max_ = discretization.mpi_reduction.max(phase)
        # phase = (phase + np.abs(min_)) / (max_ + np.abs(min_))


    phase_field_0.s[0, 0] = (np.sin(discretization.fft.coords[0] * 4 * np.pi) + np.sin(
        discretization.fft.coords[1] * 4 * np.pi) + 2) / 4

    # phase_field_0.s += 0.7 * np.random.rand(*phase_field_0.s.shape) ** 1
    apply_filter(phase_field_0)
    phase_field_00 = np.copy(phase_field_0)
    # my_sensitivity_pixel(phase_field_0).reshape([1, 1, *number_of_pixels])

    load_init_from_same_grid = False
    if load_init_from_same_grid:
        # file_data_name = f'eta_1muFFTTO_{problem_type}_random_init_N{number_of_pixels[0]}_E_target_{E_target}_Poisson_{poison_target}_Poisson0_{poison_0}_w{w_mult}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{MPI.COMM_WORLD.size}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}.npy'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')
        file_data_name = f'iteration_final'
        if MPI.COMM_WORLD.size == 1 or None:
            # phase = np.load(f'experiments/exp_data/init_phase_FE_N{number_of_pixels[0]}_NuMPI6.npy')
            # phase= np.load(f'experiments/exp_data/'  + file_data_name)
            phase = np.load(os.path.expanduser(data_folder_path + file_data_name + f'.npy'), allow_pickle=True)
        else:
            phase = load_npy(data_folder_path + file_data_name,
                             tuple(discretization.fft.subdomain_locations),
                             tuple(discretization.nb_of_pixels), MPI.COMM_WORLD)

        phase_field_0.s[0, 0] = phase  # [discretization.fft.subdomain_slices]

    # b

    #    if MPI.COMM_WORLD.size == 1:
    # print('rank' f'{MPI.COMM_WORLD.rank:6} phase=' f'')
    # plt.figure()
    # plt.contourf(phase_field_0.s[0, 0], cmap=mpl.cm.Greys)
    # # nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
    # plt.clim(0, 1)
    # plt.colorbar()
    #
    # plt.show()
    # phase_field_0 = phase_field_0.s.ravel()

    norms_f = []
    norms_delta_f = []
    norms_max_grad_f = []
    norms_norm_grad_f = []
    norms_max_delta_x = []
    norms_norm_delta_x = []

    iterat = 0


    def my_callback(result_norms):
        global iterat
        # iteration = result_norms[-1]
        # norms_f.append(result_norms[0])
        # norms_delta_f.append(result_norms[1])
        # norms_max_grad_f.append(result_norms[2])
        # norms_norm_grad_f.append(result_norms[3])
        # norms_max_delta_x.append(result_norms[4])
        # norms_norm_delta_x.append(result_norms[5])
        file_data_name_it = f'iteration{iterat}'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')

        if save_data:
            save_npy(data_folder_path + f'{preconditioner_type}' + file_data_name_it + f'.npy',
                     result_norms.reshape([*discretization.nb_of_pixels]),
                     tuple(discretization.subdomain_locations_no_buffers),
                     tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD)
            if MPI.COMM_WORLD.size == 1:
                print(data_folder_path + file_data_name_it + f'.npy')

        iterat += 1
        # plt.figure()
        # plt.contourf(result_norms.reshape([*discretization.nb_of_pixels]), cmap=mpl.cm.Greys)
        # # nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
        # plt.clim(0, 1)
        # plt.title(f'Iteration {iterat}')
        # plt.colorbar()
        # plt.show()


    xopt_FE_MPI = Optimization.l_bfgs(fun=objective_function_multiple_load_cases,
                                      x=phase_field_0.s.ravel(),
                                      jac=True,
                                      maxcor=20,
                                      gtol=1e-4,
                                      ftol=1e-4,
                                      maxiter=10,
                                      comm=MPI.COMM_WORLD,
                                      disp=True,
                                      callback=my_callback
                                      )
    solution_phase = discretization.get_scalar_field(name='phase_field_in_initial_')

    solution_phase.s = xopt_FE_MPI.x.reshape([1, 1, *discretization.nb_of_pixels])
    sensitivity_sol_FE_MPI = xopt_FE_MPI.jac.reshape([1, 1, *discretization.nb_of_pixels])

    print("=== REAL SPACE FIELDS ===")
    sum_buffer = 0
    for i, name in enumerate(discretization.field_collection.field_names):
        field = discretization.field_collection.get_field(name)
        print(
            f"{i + 1:3}: {name:30} {field.buffer_size:30} {field.buffer_size * 8 / 1024 / 1024:30} MiB {field.shape} ")
        sum_buffer += field.buffer_size
    print("=== FOURIER SPACE FIELDS ===")
    for i, name in enumerate(discretization.ffield_collection.field_names):
        field = discretization.ffield_collection.get_field(name)
        print(f"{i + 1:3}: {name:30} {field.buffer_size:30} {field.buffer_size:30} MiB {field.shape} ")
        sum_buffer += field.buffer_size
    print(f"Total memory: {sum_buffer * 8 / 1024 / 1024} MiB")

    _info = {}
    # if MPI.COMM_WORLD.size == 1:
    print('rank' f'{MPI.COMM_WORLD.rank:6} phase=' f' ')
    plt.figure()
    plt.contourf(solution_phase.s[0, 0], cmap=mpl.cm.Greys)
    # nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
    plt.clim(0, 1)
    plt.colorbar()
    plt.show()

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
    _info['nb_iterations'] = iterat

    file_data_name = f'iteration_final'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')

    save_npy(data_folder_path + file_data_name + f'.npy', solution_phase.s[0].mean(axis=0),
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
    material_data_field_C_0_rho_quad.s = elastic_C_0[..., np.newaxis, np.newaxis, np.newaxis] * \
                                         np.power(solution_phase_at_quad_poits_1qxyz.s, p)[0, 0, :, ...]

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
            comm=discretization.fft.communicator,
            fc=discretization.field_collection,
            hessp=K_fun,  # linear operator
            b=rhs_field_final,
            x=displacement_field,
            P=M_fun,
            tol=1e-5,
            maxiter=1000,
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
                comm=discretization.fft.communicator,
                fc=discretization.field_collection,
                hessp=K_fun,  # linear operator
                b=rhs_field_final,
                x=displacement_field,
                P=M_fun,
                tol=1e-5,
                maxiter=1000,
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
        np.savez(data_folder_path + f'{preconditioner_type}' + f'_log.npz', **_info)
