from cProfile import label

import numpy as np
import scipy as sc
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpi4py import MPI
from NuMPI.Tools import Reduction
import matplotlib.pyplot as plt

from NuMPI.IO import save_npy, load_npy
from IPython.terminal.shortcuts.filters import KEYBINDING_FILTERS
from PySide2.examples.opengl.contextinfo import colors
from matplotlib.animation import FuncAnimation, PillowWriter
from sympy.physics.quantum.sho1d import omega

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library
from mpl_toolkits import mplot3d

src = '../figures/'

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'

import numpy as np
import random


def generate_array_with_squares(N, k):
    array = np.zeros((N, N), dtype=bool)
    squares_added = 0

    while squares_added < k:
        # Randomly determine the size of the square
        square_size = random.randint(1, N // 3)  # Example: squares are at most N/3
        # Randomly determine the top-left corner of the square
        row = random.randint(0, N - square_size)
        col = random.randint(0, N - square_size)

        # Check if the area is already occupied
        if np.any(array[row:row + square_size, col:col + square_size]):
            continue  # Skip if it overlaps

        # Place the square
        array[row:row + square_size, col:col + square_size] = True
        squares_added += 1

    return array



def generate_striped_array(N, k):
    """
    Generate a 2D boolean array of NxN with k equal-sized stripes of 1 and k stripes of 0.

    Args:
    N (int): Size of the square array (NxN).
    k (int): Number of stripes of 1 and stripes of 0.

    Returns:
    np.ndarray: A 2D boolean array with alternating equal-sized stripes.
    """
    # Initialize the array
    array = np.zeros((N, N), dtype=bool)

    stripe_width = N // (2 * k)  # Calculate the width of each stripe (2*k stripes total)

    for i in range(2 * k):  # Loop for all stripes (k stripes of 1 and k stripes of 0)
        start = i * stripe_width
        end = start + stripe_width
        if i % 2 == 0:  # Even stripes (1s)
            array[start:end, :] = True

    return array




domain_size = [1, 1]
nb_pix_multips = [2]  # ,2,3,3,2,
nb_squares = np.arange(1,4)  # 65 17  33

nb_it = np.zeros((len(nb_pix_multips), nb_squares.size), )
nb_it_combi = np.zeros((len(nb_pix_multips), nb_squares.size), )
nb_it_Jacobi = np.zeros((len(nb_pix_multips), nb_squares.size), )
nb_it_Richardson = np.zeros((len(nb_pix_multips), nb_squares.size), )
nb_it_Richardson_combi = np.zeros((len(nb_pix_multips), nb_squares.size), )

norm_rr_combi = []
norm_rz_combi = []
norm_rr_Jacobi = []
norm_rz_Jacobi = []
norm_rr = []
norm_rz = []
norm_rMr_combi = []
norm_rMr = []
norm_rMr_Jacobi = []

# kontrast = []
# kontrast_2 = []
eigen_LB = []
kontrast=10
for kk in np.arange(np.size(nb_pix_multips)):
    nb_pix_multip = nb_pix_multips[kk]
    # number_of_pixels = (nb_pix_multip * 32, nb_pix_multip * 32)
    number_of_pixels = (nb_pix_multip * 16, nb_pix_multip * 16)

    # number_of_pixels = (16,16)

    my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                      problem_type=problem_type)

    discretization = domain.Discretization(cell=my_cell,
                                           nb_of_pixels_global=number_of_pixels,
                                           discretization_type=discretization_type,
                                           element_type=element_type)
    start_time = time.time()

    # set macroscopic gradient
    macro_gradient = np.array([[1.0,0.5], [0.5, 1.0]])

    # create material data field
    K_0, G_0 = 1, 0.5  # domain.get_bulk_and_shear_modulus(E=1, poison=0.2)

    # identity tensor                                               [single tensor]
    ii = np.eye(2)

    shape = tuple((number_of_pixels[0] for _ in range(2)))


    def expand(arr):
        new_shape = (np.prod(arr.shape), np.prod(shape))
        ret_arr = np.zeros(new_shape)
        ret_arr[:] = arr.reshape(-1)[:, np.newaxis]
        return ret_arr.reshape((*arr.shape, *shape))


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

    print('elastic tangent = \n {}'.format(domain.compute_Voigt_notation_4order(elastic_C_1)))

    # material distribution
    geometry_ID = 'n_squares'  # 'square_inclusion'#'circle_inclusion'#

    def scale_field(field, min_val, max_val):
        """Scales a 2D random field to be within [min_val, max_val]."""
        field_min, field_max = Reduction(MPI.COMM_WORLD).min(field), Reduction(MPI.COMM_WORLD).max(field)
        scaled_field = (field - field_min) / (field_max - field_min)  # Normalize to [0,1]
        return scaled_field * (max_val - min_val) + min_val  # Scale to [min_val, max_val]


    for i in np.arange(nb_squares.size):
        nb_square = nb_squares[i]

        phase_field_indicator = generate_striped_array(number_of_pixels[0], nb_square)


       # phase_field_indicator = generate_array_with_squares(number_of_pixels[0], nb_square)

        phase_field =  np.zeros(number_of_pixels)
        phase_field[phase_field_indicator==True]=1
        phase_field[phase_field_indicator==False]=100
        print(phase_field)

        # phase_field[phase_field<=0.001]= phase_field + 1e-4

        phase_fem = np.zeros([2, *number_of_pixels])
        phase_fnxyz = discretization.get_scalar_sized_field()
        phase_fnxyz[0, 0, ...] = phase_field

        # np.save('geometry_jacobi.npy', np.power(phase_field_l, 2),)
        # sc.io.savemat('geometry_jacobi.mat', {'data':  np.power(phase_field_l, 2)})

        phase_field_at_quad_poits_1qnxyz = \
            discretization.evaluate_field_at_quad_points(nodal_field_fnxyz=phase_fnxyz,
                                                         quad_field_fqnxyz=None,
                                                         quad_points_coords_dq=None)[0]

        phase_field_at_quad_poits_1qnxyz[0, :, 0, ...] = phase_fnxyz
        # apply material distribution
        # material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * np.power(phase_field[0, 0], 1)
        # material_data_field_C_0_rho=material_data_field_C_0[..., :, :] * phase_fem
        # material_data_field_C_0_rho +=100*material_data_field_C_0[..., :, :] * (1-phase_fem)
        material_data_field_C_0_rho = material_data_field_C_0[..., :, :, :] * np.power(
            phase_field_at_quad_poits_1qnxyz, 1)[0, :, 0, ...]
        # material_data_field_C_0_rho=phase_field_at_quad_poits_1qnxyz
        # Set up right hand side
        macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)
        # perturb=np.random.random(macro_gradient_field.shape)
        # macro_gradient_field += perturb#-np.mean(perturb)

        # Solve mechanical equilibrium constrain
        rhs = discretization.get_rhs(material_data_field_C_0_rho, macro_gradient_field)

        K_fun = lambda x: discretization.apply_system_matrix(material_data_field_C_0_rho, x,
                                                             formulation='small_strain')

        # plotting eigenvalues
        ##  K = discretization.get_system_matrix(material_data_field_C_0_rho)
        ## M = discretization.get_system_matrix(refmaterial_data_field_I4s)

        ## eig = sc.linalg.eigh(a=K, b=M, eigvals_only=True)

        min_val = np.min(phase_field)
        max_val = np.max(phase_field)

        eigen_LB.append(min_val)
        omega = 1  # 2 / ( eig[-1]+eig[np.argmax(eig>0)])


        K = discretization.get_system_matrix(material_data_field=material_data_field_C_0_rho)

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

        x0 = np.random.random(discretization.get_displacement_sized_field().shape)
        #x0 = np.zeros(discretization.get_displacement_sized_field().shape)

        displacement_field, norms = solvers.PCG(K_fun, rhs,
                                                x0=x0,
                                                P=M_fun, steps=int(1000), toler=1e-14,
                                                norm_type='data_scaled_rr',
                                                norm_metric=M_fun)
        nb_it[kk - 1, i] = (len(norms['residual_rz']))
        norm_rz.append(norms['residual_rz'])
        norm_rr.append(norms['residual_rr'])
        norm_rMr.append(norms['data_scaled_rr'])

        print(nb_it)
        #########
        displacement_field_combi, norms_combi = solvers.PCG(K_fun, rhs, x0=x0, P=M_fun_combi, steps=int(1000),
                                                            toler=1e-14,
                                                            norm_type='data_scaled_rr',
                                                            norm_metric=M_fun)
        nb_it_combi[kk - 1, i] = (len(norms_combi['residual_rz']))
        norm_rz_combi.append(norms_combi['residual_rz'])
        norm_rr_combi.append(norms_combi['residual_rr'])
        norm_rMr_combi.append(norms_combi['data_scaled_rr'])

        #
        displacement_field_Jacobi, norms_Jacobi = solvers.PCG(K_fun, rhs, x0=x0, P=M_fun_Jacobi, steps=int(1000),
                                                              toler=1e-14,
                                                            norm_type='data_scaled_rr',
                                                            norm_metric=M_fun)
        nb_it_Jacobi[kk - 1, i] = (len(norms_Jacobi['residual_rz']))
        norm_rz_Jacobi.append(norms_Jacobi['residual_rz'])
        norm_rr_Jacobi.append(norms_Jacobi['residual_rr'])
        norm_rMr_Jacobi.append(norms_Jacobi['data_scaled_rr'])

        displacement_field_Richardson, norms_Richardson = solvers.Richardson(K_fun, rhs, x0=x0, P=M_fun,
                                                                             omega=omega,
                                                                             steps=int(1000),
                                                                             toler=1e-14)

#         _info = {}
#
#         _info['nb_of_pixels'] = discretization.nb_of_pixels_global
#         _info['nb_of_sampling_points'] = ratio
#         # phase_field_sol_FE_MPI = xopt.x.reshape([1, 1, *discretization.nb_of_pixels])
#         _info['norm_rMr_G'] = norms['data_scaled_rr']
#         _info['norm_rMr_J'] = norms_Jacobi['data_scaled_rr']
#         _info['norm_rMr_JG'] = norms_combi['data_scaled_rr']
#         _info['nb_it_G'] = nb_it
#         _info['nb_it_J'] = nb_it_Jacobi
#         _info['nb_it_JG'] = nb_it_combi
#         script_name = 'exp_paper_JG_intro_2'
#         file_data_name = (
#             f'{script_name}_gID{geometry_ID}_T{number_of_pixels[0]}_G{ratio}_kappa{kontrast}.npy')
#         folder_name = '../exp_data/'
#         save_npy(folder_name + file_data_name + f'.npy', phase_field,
#                  tuple(discretization.fft.subdomain_locations),
#                  tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD)
#         print(folder_name + file_data_name + f'.npy')
#
#         if MPI.COMM_WORLD.rank == 0:
#             np.savez(folder_name + file_data_name + f'xopt_log.npz', **_info)
#             print(folder_name + file_data_name + f'.xopt_log.npz')
# ##################




quit()

