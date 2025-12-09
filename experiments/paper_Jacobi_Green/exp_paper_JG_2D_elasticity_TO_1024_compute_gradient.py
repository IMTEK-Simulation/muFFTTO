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

script_name = 'exp_paper_JG_2D_elasticity_TO_1024'
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
parser.add_argument("-n", "--nb_pixel", default="1024")
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

    geometries_data_folder_path = '/home/martin/exp_data/'
    # geometries_data_folder_path = '//work/classic/fr_ml1145-martin_workspace_01/exp_data/'

    domain_size = [1, 1]
    # preconditioner_type = 'Jacobi'  # "Green", "Jacobi", "Green_Jacobi"
    # number_of_pixels = (16,16)

    my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                      problem_type=problem_type)

    discretization = domain.Discretization(cell=my_cell,
                                           nb_of_pixels_global=number_of_pixels,
                                           discretization_type=discretization_type,
                                           element_type=element_type)

    phase_field = discretization.get_scalar_field(name='material_phase_field')
    if number_of_pixels[0] == 64:
        name = 'exp_2D_elasticity_TO_indre_3exp_N64_Et_0.15_Pt_-0.5_P0_0.2_w40.0_eta0.01_p2_mpi6_nlc_4_e_True'
    elif number_of_pixels[0] == 128:
        name = 'exp_2D_elasticity_TO_indre_3exp_N128_Et_0.15_Pt_-0.5_P0_0.35_w40.0_eta0.01_p2_mpi10_nlc_4_e_True'

    elif number_of_pixels[0] == 512:
        name = 'exp_2D_elasticity_TO_indre_3exp_N512_Et_0.15_Pt_-0.5_P0_0.0_w5.0_eta0.01_p2_mpi80_nlc_3_e_False'

    elif number_of_pixels[0] == 1024:
        name = 'exp_2D_elasticity_TO_indre_3exp_N1024_Et_0.15_Pt_-0.5_P0_0.0_w5.0_eta0.01_p2_mpi90_nlc_3_e_False'

    phase_field.s[0, 0] = load_npy(os.path.expanduser(geometries_data_folder_path + name + f'_it{iteration}.npy'),
                                   subdomain_locations=tuple(discretization.subdomain_locations_no_buffers),
                                   nb_subdomain_grid_pts=tuple(discretization.nb_of_pixels),
                                   comm=MPI.COMM_WORLD)

    grad = discretization.get_gradient_of_scalar_field(name='grad_of_phase_field')
    discretization.apply_gradient_operator_mugrid(u_inxyz=phase_field, grad_u_ijqxyz=grad)
    grad_norm = np.sqrt(
        discretization.fft.communicator.sum(np.dot(grad.s[0].mean(axis=1).ravel(), grad.s[0].mean(axis=1).ravel())))
    grad_max = np.sqrt(
        discretization.mpi_reduction.max(grad.s[0].mean(axis=1)[0] ** 2 + grad.s[0].mean(axis=1)[1] ** 2))
    grad_max_inf = discretization.mpi_reduction.max(grad.s[0].mean(axis=1))

    _info = {}
    _info['grad_norm'] = grad_norm
    _info['grad_max'] = grad_max
    _info['grad_max_inf'] = grad_max_inf

    if discretization.fft.communicator.rank == 0:
        np.savez(data_folder_path + f'grad_info_N_{number_of_pixels[0]}_{preconditioner_type}_it_{iteration}.npz',
                 **_info)
        print(data_folder_path + f'grad_info_N_{number_of_pixels[0]}_{preconditioner_type}_it_{iteration}.npz')
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
