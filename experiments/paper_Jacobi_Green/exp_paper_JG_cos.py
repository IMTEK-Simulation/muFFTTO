import time
import os
import sys
import argparse
import sys

sys.path.append("/home/martin/Programming/muFFTTO_paralellFFT_test/muFFTTO")
sys.path.append('../..')  # Add parent directory to path

from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library

parser = argparse.ArgumentParser(
    prog="exp_paper_JG_cos.py", description="Solve homogenization problem with cosines data"
)
parser.add_argument("-n", "--nb_pix_multips", default="4")

# Preconditioner type (string, choose from a set)
parser.add_argument(
    "-p", "--preconditioner_type",
    type=str,
    choices=["Green", "Jacobi", "Green_Jacobi"],  # example options
    default="Green",
    help="Type of preconditioner to use"
)

# Total phase contrast (integer)
parser.add_argument(
    "-rho", "--contrast",
    type=int,
    default=4,
    help="Total phase contras"
)



args = parser.parse_args()
nb_pix_multips = int(args.nb_pix_multips)
total_phase_contrast = args.contrast
preconditioner_type = args.preconditioner_type  # 'Jacobi'  # 'Green'  # 'Green_Jacobi'

if MPI.COMM_WORLD.rank == 0:
    print('  Rank   Size          Domain       Subdomain        Location')
    print('  ----   ----          ------       ---------        --------')
MPI.COMM_WORLD.Barrier()  # Barrier so header is printed first
script_name = os.path.splitext(os.path.basename(__file__))[0]

file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
if not os.path.exists(file_folder_path):
    os.makedirs(file_folder_path)
data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
if not os.path.exists(data_folder_path):
    os.makedirs(data_folder_path)
figure_folder_path = file_folder_path + '/figures/' + script_name + '/'
if not os.path.exists(figure_folder_path):
    os.makedirs(figure_folder_path)
src = '../figures/'

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'

domain_size = [1, 1]
# Variables to be set up
for nb_laminates_power in np.arange(2, nb_pix_multips + 1):
    nb_laminates = 2 ** nb_laminates_power

    #
    number_of_pixels = (2 ** nb_pix_multips, 2 ** nb_pix_multips)

    physical_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                            problem_type=problem_type)

    discretization = domain.Discretization(cell=physical_cell,
                                           nb_of_pixels_global=number_of_pixels,
                                           discretization_type=discretization_type,
                                           element_type=element_type)

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

    # create material data field
    K_0, G_0 = 1, 0.5  # domain.get_bulk_and_shear_modulus(E=1, poison=0.2)

    # identity tensor                                               [single tensor]
    ii = np.eye(2)
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
    # print('elastic tangent = \n {}'.format(C_1))

    # populate global field
    material_data_field_C_0 = discretization.get_material_data_size_field_mugrid(name='elastic_tensor')

    # material distribution
    material_distribution = discretization.get_scalar_field(name='material_distribution')

    x_coors = discretization.fft.coords
    material_distribution.s[0, 0] = 0.5 + 0.25 * np.cos(
        2 * np.pi * x_coors[0] - 2 * np.pi * x_coors[1]) + 0.25 * np.cos(
        2 * np.pi * x_coors[1] + 2 * np.pi * x_coors[0])

    phase_contrast = 10 ** total_phase_contrast
    phases_single = phase_contrast
    if total_phase_contrast == 0:
        pass
    else:
        material_distribution.s[0, 0,] += 1 / 10 ** total_phase_contrast

    material_data_field_C_0.s = np.einsum('ijkl,qxy->ijklqxy', elastic_C_1,
                                          np.broadcast_to(material_distribution.s[0, 0],
                                                          material_data_field_C_0.s[0, 0, 0, 0].shape))

    # set macroscopic gradient
    macro_gradient = np.array([[1.0, 0], [0, 1.0]])
    macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
    discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient,
                                                   macro_gradient_field_ijqxyz=macro_gradient_field)
    # Set up right hand side
    rhs_field = discretization.get_unknown_size_field(name='rhs_field')
    discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0,
                                  macro_gradient_field_ijqxyz=macro_gradient_field,
                                  rhs_inxyz=rhs_field)


    # Solve mechanical equilibrium constrain
    def K_fun(x, Ax):
        discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0,
                                                  input_field_inxyz=x,
                                                  output_field_inxyz=Ax,
                                                  formulation='small_strain')
        # discretization.fft.communicate_ghosts(Ax)


    # Set up preconditioners
    # Green
    preconditioner = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=elastic_C_1)

    K_diag_alg = discretization.get_preconditioner_Jacobi_mugrid(
        material_data_field_ijklqxyz=material_data_field_C_0,
        formulation=formulation)


    def M_fun_green(x, Px):
        """
        Function to compute the product of the Preconditioner matrix with a vector.
        The Preconditioner is represented by the convolution operator.
        """
        discretization.fft.communicate_ghosts(x)
        discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                                   input_nodal_field_fnxyz=x,
                                                   output_nodal_field_fnxyz=Px)


    def M_fun_Green_Jacobi(x, Px):
        discretization.fft.communicate_ghosts(x)
        x_jacobi_temp = discretization.get_unknown_size_field(name='x_jacobi_temp')

        x_jacobi_temp.s = K_diag_alg.s * x.s
        discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                                   input_nodal_field_fnxyz=x_jacobi_temp,
                                                   output_nodal_field_fnxyz=Px)

        Px.s = K_diag_alg.s * Px.s
        discretization.fft.communicate_ghosts(Px)


    def M_fun_Jacobi(x, Px):
        Px.s = K_diag_alg.s * K_diag_alg.s * x.s
        discretization.fft.communicate_ghosts(Px)


    if preconditioner_type == 'Green':
        M_fun = M_fun_green
    elif preconditioner_type == 'Green_Jacobi':
        M_fun = M_fun_Green_Jacobi
    elif preconditioner_type == 'Jacobi':
        M_fun = M_fun_Jacobi

    norms = dict()
    norms['residual_rr'] = []
    norms['residual_rz'] = []


    def callback(it, x, r, p, z, stop_crit_norm):
        global norms

        """
        Callback function to print the current solution, residual, and search direction.
        """
        norm_of_rr = discretization.fft.communicator.sum(np.dot(r.ravel(), r.ravel()))
        norm_of_rz = discretization.fft.communicator.sum(np.dot(r.ravel(), z.ravel()))
        norms['residual_rr'].append(norm_of_rr)
        norms['residual_rz'].append(norm_of_rz)

        # if discretization.fft.communicator.rank == 0:
        # print(f"{it:5} norm of rr = {norm_of_rr:.5}")
        # print(f"{it:5} norm of rz = {norm_of_rz:.5}")
        # print(f"{it:5} stop_crit_norm = {stop_crit_norm:.5}")


    solution_field = discretization.get_unknown_size_field(name='solution')
    solvers.conjugate_gradients_mugrid(
        comm=discretization.fft.communicator,
        fc=discretization.field_collection,
        hessp=K_fun,  # linear operator
        b=rhs_field,
        x=solution_field,
        P=M_fun,
        tol=1e-6,
        maxiter=2000,
        callback=callback,
        norm_metric=M_fun_green
    )
    if discretization.fft.communicator.rank == 0:
        nb_steps = len(norms['residual_rr'])
        print(f'nb steps = {nb_steps} ')
        _info = {}
        _info['nb_steps'] = nb_steps
        results_name = (
                f'nb_nodes_{number_of_pixels[0]}_' + f'nb_pixels_{nb_laminates}_' + f'contrast_{total_phase_contrast}_' + f'prec_{preconditioner_type}')

        np.savez(data_folder_path + results_name + f'.npz', **_info)
        print(data_folder_path + results_name + f'.npz')  #
        # infoaa= np.load(data_folder_path + results_name + f'.npz', allow_pickle=True)
