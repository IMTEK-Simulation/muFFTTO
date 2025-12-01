import numpy as np
import scipy as sc
import time
import os
import sys
import gc
import argparse

sys.path.append("/home/martin/Programming/muFFTTO_paralellFFT_test/muFFTTO")
sys.path.append('../..')

from NuMPI.IO import save_npy, load_npy
from mpi4py import MPI

from muFFTTO import domain
from muFFTTO import solvers

script_name = os.path.splitext(os.path.basename(__file__))[0]
file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
figure_folder_path = file_folder_path + '/figures/' + script_name + '/'

if not os.path.exists(data_folder_path):
    os.makedirs(data_folder_path)

if not os.path.exists(figure_folder_path):
    os.makedirs(figure_folder_path)

parser = argparse.ArgumentParser(
    prog="exp_paper_JG_2D_elasticity_TO_1024.py",
    description="Solve non-linear elasticity example "
                "from J.Zeman et al., Int. J. Numer. Meth. Engng 111, 903–926 (2017)."
)
parser.add_argument("-n", "--nb_pixel", default="512")
parser.add_argument("-it", "--iteration", default="1")

parser.add_argument(
    "-p", "--preconditioner_type",
    type=str,
    choices=["Green", "Jacobi", "Green_Jacobi"],  # example options
    default="Green",
    help="Type of preconditioner to use"
)
args = parser.parse_args()

n_pix = int(args.nb_pixel)
number_of_pixels = (n_pix, n_pix)  # (1024, 1024)
iteration = args.iteration
preconditioner_type = args.preconditioner_type

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'

# microstructure name
# name = 'lbfg_muFFTTO_elasticity_exp_paper_JG_2D_elasticity_TO_N64_E_target_0.15_Poisson_-0.50_Poisson0_0.29_w5.00_eta0.02_mac_1.0_p2_prec=Green_bounds=False_FE_NuMPI6_nb_load_cases_3_e_obj_False_random_True'
# iteration = 1200
plot_results = False
compute_results = True

if compute_results:

    # geometries_data_folder_path = '/home/martin/exp_data/'
    geometries_data_folder_path = '//work/classic/fr_ml1145-martin_workspace_01/exp_data/'

    domain_size = [1, 1]
    # preconditioner_type = 'Jacobi'  # "Green", "Jacobi", "Green_Jacobi"
    # number_of_pixels = (16,16)

    my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                      problem_type=problem_type)

    discretization = domain.Discretization(cell=my_cell,
                                           nb_of_pixels_global=number_of_pixels,
                                           discretization_type=discretization_type,
                                           element_type=element_type)
    start_time = time.time()

    K_0, G_0 = 1, 0.5
    E_0 = 9 * K_0 * G_0 / (3 * K_0 + G_0)
    poison_0 = (3 * K_0 - 2 * G_0) / (2 * (3 * K_0 + G_0))
    elastic_C_0 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                     K=K_0,
                                                     mu=G_0,
                                                     kind='linear')

    # print('Target elastic tangent = \n {}'.format(domain.compute_Voigt_notation_4order(elastic_C_0)))

    phase_field = discretization.get_scalar_field(name='material_phase_field')
    if number_of_pixels[0] == 64:
        name = 'exp_2D_elasticity_TO_indre_3exp_N64_Et_0.15_Pt_-0.5_P0_0.2_w40.0_eta0.01_p2_mpi6_nlc_4_e_True'
    elif number_of_pixels[0] == 128:
        name = 'exp_2D_elasticity_TO_indre_3exp_N128_Et_0.15_Pt_-0.5_P0_0.35_w40.0_eta0.01_p2_mpi10_nlc_4_e_True'

    elif number_of_pixels[0] == 512:
        name = 'exp_2D_elasticity_TO_indre_3exp_N512_Et_0.15_Pt_-0.5_P0_0.0_w5.0_eta0.01_p2_mpi80_nlc_3_e_False'

    elif number_of_pixels[0] == 1024:
        name = 'exp_2D_elasticity_TO_indre_3exp_N1024_Et_0.15_Pt_-0.5_P0_0.0_w5.0_eta0.01_p2_mpi90_nlc_3_e_False'

    macro_gradient_inc_field = discretization.get_gradient_size_field(name='macro_gradient_inc_field')
    material_data_field = discretization.get_material_data_size_field_mugrid(name='mat_Data')

    rhs_field = discretization.get_unknown_size_field(name='rhs_field')

    # for iteration in np.arange(1, 1200):

    _info = {}
    # phase_field.s[0, 0] = np.load_npy(os.path.expanduser(data_folder_path + name + f'_it{2229}.npy'), allow_pickle=True)
    phase_field.s[0, 0] = load_npy(os.path.expanduser(geometries_data_folder_path + name + f'_it{iteration}.npy'),
                                   subdomain_locations=tuple(discretization.subdomain_locations_no_buffers),
                                   nb_subdomain_grid_pts=tuple(discretization.nb_of_pixels),
                                   comm=MPI.COMM_WORLD)

    phase_field.s[0, 0] = phase_field.s[0, 0] ** 2

    #                                   np.ones(np.array([discretization.nb_quad_points_per_pixel,
    #                                                     *discretization.nb_of_pixels])))
    material_data_field.s = np.einsum('ijkl,qxy->ijklqxy', elastic_C_0,
                                      np.broadcast_to(phase_field.s[0, 0],
                                                      material_data_field.s[0, 0, 0, 0].shape))

    # set macroscopic gradient
    macro_gradient = np.array([[1.0, 0.5], [0.5, 1.0]])
    discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient,
                                                   macro_gradient_field_ijqxyz=macro_gradient_inc_field)

    # Set up right hand side

    discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field,
                                  macro_gradient_field_ijqxyz=macro_gradient_inc_field,
                                  rhs_inxyz=rhs_field)


    # Set up  system matrix function
    def hessian_fun(x, Ax):
        discretization.apply_system_matrix_mugrid(material_data_field=material_data_field,
                                                  input_field_inxyz=x,
                                                  output_field_inxyz=Ax,
                                                  formulation='small_strain')


    # assembly preconditioners
    # Green
    preconditioner = discretization.get_preconditioner_Green_mugrid(
        reference_material_data_ijkl=elastic_C_0)


    def preconditioner_fun_green(x, Px):
        discretization.fft.communicate_ghosts(x)
        discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                                   input_nodal_field_fnxyz=x,
                                                   output_nodal_field_fnxyz=Px)


    # Jacobi
    K_diag_alg = discretization.get_preconditioner_Jacobi_mugrid(
        material_data_field_ijklqxyz=material_data_field,
        formulation=formulation)


    def preconditioner_fun_jacobi(x, Px):
        Px.s = K_diag_alg.s * K_diag_alg.s * x.s
        discretization.fft.communicate_ghosts(Px)


    def preconditioner_fun_green_jacobi(x, Px):
        discretization.fft.communicate_ghosts(x)
        x_jacobi_temp = discretization.get_unknown_size_field(name='x_jacobi_temp')

        x_jacobi_temp.s = K_diag_alg.s * x.s
        discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                                   input_nodal_field_fnxyz=x_jacobi_temp,
                                                   output_nodal_field_fnxyz=Px)

        Px.s = K_diag_alg.s * Px.s
        discretization.fft.communicate_ghosts(Px)


    if preconditioner_type == 'Green':
        prec_fun = preconditioner_fun_green
    elif preconditioner_type == 'Green_Jacobi':
        prec_fun = preconditioner_fun_green_jacobi
    elif preconditioner_type == 'Jacobi':
        prec_fun = preconditioner_fun_jacobi

    norms = dict()
    norms['residual_rr'] = []
    norms['residual_rz'] = []
    norms['residual_rGr'] = []


    def callback(it, x, r, p, z, stop_crit_norm):
        global norms

        """
        Callback function to print the current solution, residual, and search direction.
        """
        norm_of_rr = discretization.fft.communicator.sum(np.dot(r.ravel(), r.ravel()))
        norm_of_rz = discretization.fft.communicator.sum(np.dot(r.ravel(), z.ravel()))
        norms['residual_rr'].append(norm_of_rr)
        norms['residual_rz'].append(norm_of_rz)
        norms['residual_rGr'].append(stop_crit_norm)

        # if discretization.fft.communicator.rank == 0:
        # print(f"{it:5} norm of rr = {norm_of_rr:.5}")
        # print(f"{it:5} norm of rz = {norm_of_rz:.5}")
        # print(f"{it:5} stop_crit_norm = {stop_crit_norm:.5}")


    solution_field = discretization.get_unknown_size_field(name='solution')
    solvers.conjugate_gradients_mugrid(
        comm=discretization.fft.communicator,
        fc=discretization.field_collection,
        hessp=hessian_fun,  # linear operator
        b=rhs_field,
        x=solution_field,
        P=prec_fun,
        tol=1e-5,
        maxiter=20000,
        callback=callback,
        norm_metric=preconditioner_fun_green
    )
    if discretization.fft.communicator.rank == 0:
        nb_steps = len(norms['residual_rr'])
        print(f'nb steps = {nb_steps} ')
        _info['norm_rr'] = norms['residual_rr']
        _info['norm_rz'] = norms['residual_rz']
        _info['residual_rGr'] = norms['residual_rGr']

        np.savez(data_folder_path + f'_info_N_{number_of_pixels[0]}_{preconditioner_type}_it_{iteration}.npz',
                 **_info)
        print(data_folder_path + f'_info_N_{number_of_pixels[0]}_{preconditioner_type}_it_{iteration}.npz')
    # gc.collect()
    # if iteration == 10 :
    #     print("=== REAL SPACE FIELDS ===")
    #     sum_buffer = 0
    #     for i, name in enumerate(discretization.field_collection.field_names):
    #         field = discretization.field_collection.get_field(name)
    #         print(
    #             f"{i + 1:3}: {name:30} {field.buffer_size:30} {field.buffer_size * 8 / 1024 / 1024:30} MiB {field.shape} ")
    #         sum_buffer += field.buffer_size
    #     print("=== FOURIER SPACE FIELDS ===")
    #     for i, name in enumerate(discretization.ffield_collection.field_names):
    #         field = discretization.ffield_collection.get_field(name)
    #         print(f"{i + 1:3}: {name:30} {field.buffer_size:30} {field.buffer_size:30} MiB {field.shape} ")
    #         sum_buffer += field.buffer_size
    #     print(f"Total memory: {sum_buffer * 8 / 1024 / 1024} MiB")

if plot_results:
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    plt.rcParams["text.usetex"] = True
    plt.rcParams.update({
        "text.usetex": True,  # Use LaTeX
        # "font.family": "helvetica",  # Use a serif font
    })
    plt.rcParams.update({'font.size': 11})
    plt.rcParams["font.family"] = "Arial"
    nb_iterations_G = []
    nb_iterations_J = []
    nb_iterations_GJ = []
    for iteration in np.arange(1, 10):
        info_log_final_G = np.load(data_folder_path + f'_info_N_{number_of_pixels[0]}_Green_it_{iteration}.npz',
                                   allow_pickle=True)
        nb_iterations_G.append(len(info_log_final_G.f.norm_rr))

        info_log_final_J = np.load(data_folder_path + f'_info_N_{number_of_pixels[0]}_Jacobi_it_{iteration}.npz',
                                   allow_pickle=True)
        nb_iterations_J.append(len(info_log_final_J.f.norm_rr))

        info_log_final_GJ = np.load(data_folder_path + f'_info_N_{number_of_pixels[0]}_Green_Jacobi_it_{iteration}.npz',
                                    allow_pickle=True)
        nb_iterations_GJ.append(len(info_log_final_GJ.f.norm_rr))

        print()
    nb_iterations_G = np.array(nb_iterations_G)
    nb_iterations_J = np.array(nb_iterations_J)
    nb_iterations_GJ = np.array(nb_iterations_GJ)

    # fig = plt.figure(figsize=(11.5, 6))
    fig = plt.figure(figsize=(8.3, 6.1))

    plt.rcParams.update({'font.size': 11})
    plt.rcParams["font.family"] = "Arial"

    gs = fig.add_gridspec(2, 4, width_ratios=[3, 3, 3, 0.2]
                          , height_ratios=[1, 1.7], hspace=0.07)
    ax_iterations = fig.add_subplot(gs[1:, :])
    ax_iterations.text(-0.1, 1.0, rf'\textbf{{(b)}}', transform=ax_iterations.transAxes)

    ax_iterations.plot(np.linspace(1, 1000, nb_iterations_G.shape[0]), nb_iterations_G, "g", label='Green N=64',
                       linewidth=1)
    ax_iterations.plot(np.linspace(1, 1000, nb_iterations_J.shape[0]), nb_iterations_J, "b", label='Jacobi N=64',
                       linewidth=1)
    # ax_iterations.plot(nb_iterations_J, "b", label='Jacobi N=64', linewidth=1)

    ax_iterations.plot(np.linspace(1, 1000, nb_iterations_GJ.shape[0]), nb_iterations_GJ, "k",
                       label='Green-Jacobi  N=64', linewidth=2)
    #
    # ax_iterations.plot(np.linspace(1, 1000, dgo_32.shape[0]), dgo_32, "g", label='Green N=32', linewidth=1,
    #                    linestyle=':')
    # ax_iterations.plot(np.linspace(1, 1000, jacoby_32.shape[0]), jacoby_32, "b", label='Jacobi N=32', linewidth=1,
    #                    linestyle=':')
    # ax_iterations.plot(np.linspace(1, 1000, combi_32.shape[0]), combi_32, "k", label='Jacobi - Green N=32', linewidth=2,
    # linestyle=':')
    plt.show()
