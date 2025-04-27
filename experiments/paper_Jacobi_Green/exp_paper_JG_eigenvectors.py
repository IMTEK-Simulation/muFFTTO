import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import scipy as sc

from mpi4py import MPI
from NuMPI.Tools import Reduction
from NuMPI.IO import save_npy, load_npy


from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO.solvers import PCG
from muFFTTO import microstructure_library

from experiments.experiments_with_CG_convergence.simple_CG import  get_ritz_values, plot_ritz_values, get_cg_polynomial, plot_cg_polynomial, plot_eigenvectors, \
    plot_eigendisplacement, plot_rhs, plot_eigenvector_filling

import numpy as np
import matplotlib.pyplot as plt


def get_participation_ration(displacemets_flat, grid_shape):
    # reshape to the grid size

    # eik_eik = np.linalg.norm(atoms.get_array('Displacement'), axis=1)
    # Pk = 1 / (np.sum(eik_eik ** 4) * N_atoms)
    dim = len(grid_shape)
    nb_modes = len(displacemets_flat)
    # particitaion_ratios=np.zeros(nb_modes)
    # for k in np.arange(0, nb_modes):

    # displacemets_fnxyz = displacemets_flat[:, k].reshape(grid_shape)
    #
    # size_of_disp_xyz = np.linalg.norm(displacemets_fnxyz[:, 0, ...], axis=0)

    squared_sum = np.sum(displacemets_flat ** 2, axis=0)  # Sum of squared components per mode
    fourth_sum = np.sum(displacemets_flat ** 4, axis=0)  # Sum of fourth power components per mode
    # N = eigenvectors.shape[1]  # Number of degrees of freedom

    # eigenvector_x = displacemets_fnxyz[:, k].reshape(grid_shape)[0, 0].transpose()
    # eigenvector_y = displacemets_fnxyz[:, k].reshape(grid_shape)[1, 0].transpose()

    particitaion_ratios = (squared_sum ** 2) / (fourth_sum * nb_modes)
    return particitaion_ratios


def plot_participation_ratios(displacemets_flat, grid_shape, eigenvals=None):
    # get participation ratios
    participation_ratios = get_participation_ration(displacemets_flat=displacemets_flat,
                                                    grid_shape=grid_shape)
    # plot participation ratios
    fig = plt.figure(figsize=(5, 5))
    gs = fig.add_gridspec(1, 1, width_ratios=[1])
    ax_ratios = fig.add_subplot(gs[0, 0])
    ax_ratios.scatter(eigenvals, participation_ratios)
    ax_ratios.set_xlabel('Eigenvalue')
    ax_ratios.set_ylabel('Participation ratio of eigenvector')
    ax_ratios.set_xlim([eigenvals[-1], eigenvals[0]])
    ax_ratios.set_xticks([eigenvals[-1], eigenvals[0]])
    plt.show()


def plot_matrix(matrix):
    # plot participation ratios
    fig = plt.figure(figsize=(5, 5))
    gs = fig.add_gridspec(1, 1, width_ratios=[1])
    ax_matrix = fig.add_subplot(gs[0, 0])
    ax_matrix.matshow(matrix)
    plt.show()


def matrix_sqrt_eig(A):
    """
    Compute the square root of a symmetric positive definite matrix using eigendecomposition.

    Parameters:
        A (ndarray): Symmetric positive definite matrix.

    Returns:
        A_sqrt (ndarray): Square root of the matrix.
    """
    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(A)

    # Compute square root of eigenvalues
    sqrt_eigvals = np.sqrt(eigvals)

    # Reconstruct A^(1/2)
    A_sqrt = eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T
    return A_sqrt


def run_simple_CG_Green(initial, RHS, kappa):
    problem_type = 'elasticity'
    discretization_type = 'finite_element'
    element_type = 'linear_triangles'
    formulation = 'small_strain'

    domain_size = [1, 1]
    geom_n = [ 3, 4]  # ,4,5,6 ,6,7,8,9,10,]  # ,2,3,3,2,  #,5,6,7,8,9 ,5,6,7,8,9,10,11

    ratios = np.array([2])  # np.arange(1,5)  # 17  33

    nb_it = np.zeros((len(geom_n), len(geom_n), ratios.size), )
    nb_it_combi = np.zeros((len(geom_n), len(geom_n), ratios.size), )
    nb_it_Jacobi = np.zeros((len(geom_n), len(geom_n), ratios.size), )
    nb_it_Richardson = np.zeros((len(geom_n), len(geom_n), ratios.size), )
    nb_it_Richardson_combi = np.zeros((len(geom_n), len(geom_n), ratios.size), )

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

    for geometry_ID in [
        'n_laminate']:  # n_laminate ,'sine_wave_','linear', 'right_cluster_x3', 'left_cluster_x3' square_inclusion
        for nb_starting_phases in np.arange(np.size(geom_n)):
            print(f'geometry_ID = {geometry_ID}')
            print(f'nb_starting_phases = {nb_starting_phases}')
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

                # set macroscopic gradient
                macro_gradient = np.array([[1.0, 0.5], [0.5, 1.0]])

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

                material_data_field_C_0 = np.einsum('ijkl,qxy->ijklqxy', elastic_C_1,
                                                    np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                                      *discretization.nb_of_pixels])))

                ref_elastic_C_1 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                                     K=K_0,
                                                                     mu=G_0,
                                                                     kind='linear')

                refmaterial_data_field_C1 = np.einsum('ijkl,qxy->ijklqxy', ref_elastic_C_1,
                                                      np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                                        *discretization.nb_of_pixels])))

                print('elastic tangent = \n {}'.format(domain.compute_Voigt_notation_4order(elastic_C_1)))

                def scale_field(field, min_val, max_val):
                    """Scales a 2D random field to be within [min_val, max_val]."""
                    field_min, field_max = Reduction(MPI.COMM_WORLD).min(field), Reduction(MPI.COMM_WORLD).max(field)
                    scaled_field = (field - field_min) / (field_max - field_min)  # Normalize to [0,1]
                    return scaled_field * (max_val - min_val) + min_val  # Scale to [min_val, max_val]

                for i in np.arange(ratios.size):
                    ratio = ratios[i]

                    if nb_discretization_index == 0:
                        phase_fied_small_grid = microstructure_library.get_geometry(
                            nb_voxels=discretization.nb_of_pixels,
                            microstructure_name=geometry_ID,
                            coordinates=discretization.fft.coords,
                            seed=1,
                            parameter=4)  # ,
                        #                                                                           contrast=-ratio) # $1 / 10 ** ratio
                        if ratio != 0:
                            phase_fied_small_grid += 1 / 10 ** ratio

                        phase_field_smooth = np.copy(phase_fied_small_grid)

                    if nb_discretization_index > 0:
                        # phase_field_smooth = sc.ndimage.zoom(phase_fied_small_grid, zoom=nb_pix_multip, order=0)
                        phase_field_smooth = np.repeat(phase_fied_small_grid, 2 ** (nb_discretization_index), axis=0)
                        phase_field_smooth = np.repeat(phase_field_smooth, 2 ** (nb_discretization_index), axis=1)

                    phase_field = np.abs(phase_field_smooth)
                    phase_field = scale_field(phase_field, min_val=1, max_val=10 ** ratio)

                    material_data_field_C_0_rho = material_data_field_C_0[..., :, :, :] * np.power(
                        phase_field, 1)

                    # Set up right hand side
                    macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)
                    # np.random.seed(seed=1)

                    # Solve mechanical equilibrium constrain
                    rhs = discretization.get_rhs(material_data_field_C_0_rho, macro_gradient_field)

                    K_fun = lambda x: discretization.apply_system_matrix(material_data_field_C_0_rho, x,
                                                                         formulation='small_strain')

                    min_val = Reduction(MPI.COMM_WORLD).min(phase_field)
                    max_val = Reduction(MPI.COMM_WORLD).max(phase_field)

                    preconditioner = discretization.get_preconditioner_NEW(
                        reference_material_data_field_ijklqxyz=refmaterial_data_field_C1)

                    M_fun = lambda x: discretization.apply_preconditioner_NEW(
                        preconditioner_Fourier_fnfnqks=preconditioner,
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
                        reduced_K = np.copy(K)
                        K[:, 0] = 0
                        K[0, :] = 0
                        K[:, np.prod(number_of_pixels)] = 0
                        K[np.prod(number_of_pixels), :] = 0
                        K[0, 0] = 1
                        K[np.prod(number_of_pixels), np.prod(number_of_pixels)] = 1

                        eig_K, eig_vect_K = sc.linalg.eig(a=K, b=None)  # , eigvals_only=True
                        eig_K = np.real(eig_K)
                        eig_K[eig_K == 1.0] = 1
                        # Sort in descending order (largest eigenvalues first)
                        idx_K = np.argsort(eig_K)[::-1]  # Get indices of sorted eigenvalues

                        # Reorder eigenvalues and eigenvectors
                        sorte_eig_K = eig_K[idx_K]
                        sorted_eig_vect_K = eig_vect_K[:, idx_K]

                        # Greeen precond
                        M = discretization.get_system_matrix(refmaterial_data_field_C1)
                        M_non_reduced=np.copy(M)
                        # fixing zero eigenvalues
                        M[:, 0] = 0
                        M[0, :] = 0
                        M[:, np.prod(number_of_pixels)] = 0
                        M[np.prod(number_of_pixels), :] = 0
                        M[0, 0] = 1
                        M[np.prod(number_of_pixels), np.prod(number_of_pixels)] = 1
                        #
                        eig_G, eig_vect_G = sc.linalg.eig(a=K, b=M)  # , eigvals_only=True
                        eig_G = np.real(eig_G)
                        eig_G[eig_G == 1.0] = 1

                        M_sym = matrix_sqrt_eig(M)
                        Green_sqrt = matrix_sqrt_eig(np.linalg.inv(M))
                        GKGsym = Green_sqrt @ K @ Green_sqrt
                        eig_G_sym, eig_vect_G_sym = sc.linalg.eig(a=GKGsym)  # , eigvals_only=True
                        #
                        # eig_GK, eig_vect_GK = sc.linalg.eig(a=np.linalg.inv(M) @ K )  # , eigvals_only=True
                        # Sort in descending order (largest eigenvalues first)
                        idx_G = np.argsort(eig_G)[::-1]  # Get indices of sorted eigenvalues

                        # Reorder eigenvalues and eigenvectors
                        sorte_eig_G = eig_G[idx_G]
                        sorted_eig_vect_G = eig_vect_G[:, idx_G]

                        # plot_eigenvectors(eigenvectors_1=np.real(sorted_eig_vect_G), eigenvectors_2=np.real(sorted_eig_vect_G),
                        #                   grid_shape=rhs.shape, dim=2, eigenvals=sorte_eig_G)
                        fig = plt.figure(figsize=(5, 5))
                        gs = fig.add_gridspec(1, 1)
                        ax_matrix = fig.add_subplot(gs[0, 0])
                        # ax_matrix =  fig.add_subplot(gs[0, 0], projection='3d')
                        MiK = np.linalg.inv(M) @ K

                        Nx = np.arange(MiK.shape[0])
                        X, Y = np.meshgrid(Nx, Nx, indexing='ij')
                        # divnorm = mpl.colors.TwoSlopeNorm(vmin=np.min(MiK), vcenter=0., vmax=np.max(MiK))
                        divnorm = mpl.colors.LogNorm(vmin=0.00000001, vmax=np.max(np.abs(MiK)))

                        pcm = ax_matrix.pcolormesh(X, Y, np.abs(MiK).transpose(), label='PCG: Green + Jacobi',
                                                   cmap='Greys', norm=divnorm)
                        # ax_matrix.plot_surface(X, Y, MiK.transpose(), cmap='viridis')
                        ax_matrix.invert_yaxis()
                        plt.show()
                        fig = plt.figure(figsize=(5, 5))
                        gs = fig.add_gridspec(1, 1)
                        ax_matrix = fig.add_subplot(gs[0, 0])
                        pcm = ax_matrix.pcolormesh(X, Y, np.abs(K).transpose(), label='PCG: Green + Jacobi',
                                                   cmap='Greys', norm=divnorm)
                        # ax_matrix.plot_surface(X, Y, MiK.transpose(), cmap='viridis')
                        ax_matrix.invert_yaxis()
                        plt.show()
                        plot_participation_ratios(displacemets_flat=np.real(sorted_eig_vect_G),
                                                  grid_shape=rhs.shape, eigenvals=sorte_eig_G)
                        participation_ratios = get_participation_ration(displacemets_flat=np.real(sorted_eig_vect_G),
                                                                        grid_shape=rhs.shape)
                        # plot_eigendisplacement(eigenvectors_1=np.real(sorted_eig_vect_G),
                        #                        grid_shape=rhs.shape, dim=2, eigenvals=sorte_eig_G,
                        #                        participation_ratios=participation_ratios)



                        # norms = np.linalg.norm(MiK, axis=0)
                        # normalized_matrix = MiK / norms
                        # plot_eigendisplacement(eigenvectors_1=normalized_matrix,
                        #                        grid_shape=rhs.shape, dim=2, eigenvals=eig_G[idx_G],
                        #                        weight=eig_G[idx_G], participation_ratios=participation_ratios)
                        #


                        #### jacobi preconditioner
                        Jacobi_sym = np.copy(np.diag(K))
                        Jacobi_sym[Jacobi_sym < 9.99e-16] = 0
                        Jacobi_sym[Jacobi_sym != 0] = 1 / np.sqrt(Jacobi_sym[Jacobi_sym != 0])

                        K_diag_inv = np.copy(np.diag(K))
                        K_diag_inv[K_diag_inv < 9.99e-16] = 0
                        K_diag_inv[K_diag_inv != 0] = 1 / K_diag_inv[K_diag_inv != 0]

                        K_diag_sym = matrix_sqrt_eig(np.diag(np.diag(K)))

                        JKJsym = np.matmul(np.diag(Jacobi_sym), np.matmul(K, np.diag(Jacobi_sym)))
                        eig_J, eig_vect_J = sc.linalg.eig(a=JKJsym)  # , eigvals_only=True


                        eig_JG, eig_vect_JG = sc.linalg.eig(a=JKJsym, b=M)  # , eigvals_only=True
                        idx_JG = np.argsort(eig_JG)[::-1]
                        # plot_eigendisplacement(eigenvectors_1=eig_vect_JG[:, idx_JG],
                        #                        grid_shape=rhs.shape, dim=2, eigenvals=eig_JG[idx_JG],
                        #                        weight=eig_JG[idx_JG], participation_ratios=participation_ratios)
                    rhs.flatten()[0] = 0
                    rhs.flatten()[np.prod(number_of_pixels)] = 0


                    ######### INITIAL SOLUTION
                    #x0 = np.random.random(discretization.get_displacement_sized_field().shape)
                    x0 = np.zeros(discretization.get_displacement_sized_field().shape)

                    M_null = lambda x: 1 * x
                    displacement_field, norms_origin = solvers.PCG(K_fun, rhs, x0=x0, P=M_null,
                                                                   steps=int(1000), toler=1e-14,
                                                                   norm_energy_upper_bound=True,
                                                                   lambda_min=np.real(sorted(eig_J)[0])
                                                                   )
                    ########################### Greeen  PRE CONDITIONED VERSION ########################################################
                    M_null = lambda x: 1 * x
                    K_fun_G = lambda x: GKGsym @ x

                    rhs_G = Green_sqrt @ rhs.flatten()

                    r0 = rhs_G.flatten() - K_fun_G(M_sym @ x0.flatten())
                    r0_norm = np.linalg.norm(r0.flatten())  # order='F

                    Green_sqrt_eig_vect_J = M_sym @ eig_vect_G
                    normed_eigenvectors = np.zeros_like(Green_sqrt_eig_vect_J)
                    for k in np.arange(Green_sqrt_eig_vect_J[:, 0].shape[0]):
                        normed_eigenvectors[:, k] = Green_sqrt_eig_vect_J[:, k] / np.linalg.norm(
                            Green_sqrt_eig_vect_J[:, k])
                    w_i = (np.dot(np.transpose(normed_eigenvectors), r0.flatten() / r0_norm)) ** 2  # order='F'
                    w_i_for_un_K = (np.dot(np.transpose(eig_vect_K),
                                           rhs.flatten() / np.linalg.norm(rhs.flatten()))) ** 2  ### ONLY FOR ZERO RHS

                    # plot_eigenvectors(eigenvectors_1=normed_eigenvectors, eigenvectors_2=eig_vect_K,
                    #                   grid_shape=rhs.shape, dim=2)

                    x_values = np.linspace(0, np.real(sorted(eig_G)[-1] + 1), 100)
                    # test Ritz values
                    ritz_values = get_ritz_values(A=GKGsym, k_max=32, v0=r0.flatten(),
                                                  M_inv=None)
                    plot_ritz_values(ritz_values=ritz_values, true_eigenvalues=eig_G)

                    displacement_field, norms = solvers.PCG(K_fun_G, rhs_G.flatten(), x0=M_sym @ x0.flatten(), P=M_null,
                                                            steps=int(1000), toler=1e-14,
                                                            norm_energy_upper_bound=True,
                                                            lambda_min=np.real(sorted(eig_G)[0])
                                                            )

                    print(norms)
                    # plot_cg_polynomial(x_values, ritz_values, true_eigenvalues=eig_G[idx_G], weight=w_i[idx_G],
                    #                    error_evol=norms['energy_lb'] / norms['residual_rr'][
                    #                        0], title='Green')  # energy_lb
              #      quit()
                    plot_rhs(rhs=np.real(rhs),
                                           grid_shape=rhs.shape )
                    plot_rhs(rhs=np.real(w_i.reshape(rhs.shape)),
                             grid_shape=rhs.shape)

                    plot_eigenvector_filling(np.real(sorted_eig_vect_G), grid_shape=rhs.shape)

                    quit()
                    plot_eigendisplacement(eigenvectors_1=np.real(eig_vect_K),
                                           grid_shape=rhs.shape, dim=2, eigenvals=eig_K[idx_K], weight=w_i[idx_K],
                                           participation_ratios=participation_ratios)
                    plot_eigendisplacement(eigenvectors_1=np.real(MiK),
                                           grid_shape=rhs.shape, dim=2, eigenvals=eig_G[idx_G], weight=w_i[idx_G],
                                           participation_ratios=participation_ratios)

                    plot_eigendisplacement(eigenvectors_1=np.real(sorted_eig_vect_G),
                                           grid_shape=rhs.shape, dim=2, eigenvals=eig_G[idx_G], weight=w_i[idx_G],
                                           participation_ratios=participation_ratios)

                    quit()
                    ############################ UNPRECONDITIONED VERSION ########################################################
                    M_null = lambda x: 1 * x
                    K_fun_ = lambda x: K @ x

                    r0 = rhs.flatten() - K_fun_(x0.flatten())
                    r0_norm = np.linalg.norm(r0.flatten())  # order='F'
                    w_i = (np.dot(np.transpose(eig_vect_K), r0.flatten() / r0_norm)) ** 2  # order='F'

                    x_values = np.linspace(0, np.real(sorted(eig_K)[-1] + 1), 1000)

                    ritz_values = get_ritz_values(A=K, k_max=100, v0=r0.flatten(),
                                                  M_inv=None)  # r0.flatten(order='F')

                    plot_ritz_values(ritz_values=ritz_values, true_eigenvalues=eig_K)

                    displacement_field, norms = solvers.PCG(K_fun_, rhs.flatten(), x0=x0.flatten(), P=M_null,
                                                            steps=int(1000), toler=1e-14,
                                                            norm_energy_upper_bound=True,
                                                            lambda_min=np.real(sorted(eig_K)[0])
                                                            )
                    plot_cg_polynomial(x_values, ritz_values, true_eigenvalues=eig_K, weight=w_i,
                                       error_evol=norms['energy_lb'] / norms['residual_rr'][0],
                                       title='Unpreconditioned')
                    ########################### JACOBI  PRE CONDITIONED VERSION ########################################################
                    M_null = lambda x: 1 * x
                    K_fun_J = lambda x: JKJsym @ x

                    rhs_J = np.matmul(np.diag(Jacobi_sym), rhs.flatten())

                    r0 = rhs_J.flatten() - K_fun_J(K_diag_sym @ x0.flatten())
                    r0_norm = np.linalg.norm(r0.flatten())  # order='F'

                    K_diag_sym_eig_vect_J = K_diag_sym @ eig_vect_J  # .transpose()  # ?????
                    normed_eigenvectors = np.zeros_like(K_diag_sym_eig_vect_J)
                    for k in np.arange(K_diag_sym_eig_vect_J[:, 0].shape[0]):
                        normed_eigenvectors[:, k] = K_diag_sym_eig_vect_J[:, k] / np.linalg.norm(
                            K_diag_sym_eig_vect_J[:, k])

                    w_i = (np.dot(np.transpose(normed_eigenvectors), r0.flatten() / r0_norm)) ** 2  # order='F'

                    x_values = np.linspace(0, np.real(sorted(eig_J)[-1] + 1), 100)

                    ritz_values = get_ritz_values(A=JKJsym, k_max=80, v0=r0.flatten(),
                                                  M_inv=None)  # r0.flatten(order='F')

                    plot_ritz_values(ritz_values=ritz_values, true_eigenvalues=eig_J)

                    displacement_field, norms = solvers.PCG(K_fun_J, rhs_J.flatten(), x0=K_diag_sym @ x0.flatten(),
                                                            P=M_null,
                                                            steps=int(1000), toler=1e-14,
                                                            norm_energy_upper_bound=True,
                                                            lambda_min=np.real(sorted(eig_J)[0])
                                                            )
                    print(norms)
                    plot_cg_polynomial(x_values, ritz_values, true_eigenvalues=eig_J, weight=w_i,
                                       error_evol=norms['energy_lb'] / norms['residual_rr'][0],
                                       title='Jacobi')  # energy_lb

                    ############################ UNPRECONDITIONED VERSION ########################################################
                    r0 = rhs - K_fun(x0)
                    r0_norm = np.linalg.norm(r0.flatten())  # order='F'
                    w_i = (np.dot(np.transpose(eig_vect_K), r0.flatten() / r0_norm)) ** 2  # order='F'

                    x_values = np.linspace(0, np.real(sorted(eig_K)[-1] + 1), 1000)

                    ritz_values = get_ritz_values(A=K, k_max=3, v0=r0.flatten(),
                                                  M_inv=None)  # r0.flatten(order='F')

                    plot_ritz_values(ritz_values=ritz_values, true_eigenvalues=eig_K)

                    M_null = lambda x: 1 * x
                    K_fun_ = lambda x: K @ x
                    displacement_field, norms = solvers.PCG(K_fun_, rhs.flatten(), x0=x0.flatten(), P=M_null,
                                                            steps=int(1000), toler=1e-14,
                                                            norm_energy_upper_bound=True,
                                                            lambda_min=np.real(sorted(eig_K)[0])
                                                            )
                    plot_cg_polynomial(x_values, ritz_values, true_eigenvalues=eig_K, weight=w_i,
                                       error_evol=norms['energy_lb'] / norms['residual_rr'][0])

                    quit()
                    # SOLVER GREEN
                    displacement_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(1000), toler=1e-14,
                                                            norm_type='data_scaled_rr',
                                                            norm_metric=M_fun
                                                            )
                    nb_it[nb_discretization_index + nb_starting_phases, nb_starting_phases, i] = (
                        len(norms['residual_rr']))
                    print('nb it  = {} '.format(len(norms['residual_rr'])))

                    norm_rz.append(norms['residual_rz'])
                    norm_rr.append(norms['residual_rr'])
                    # norm_energy_lb.append(norms['energy_lb'])
                    norm_rMr.append(norms['data_scaled_rr'])

                    #########
                    displacement_field_combi, norms_combi = solvers.PCG(K_fun, rhs, x0=None, P=M_fun_combi,
                                                                        steps=int(1000),
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
                    displacement_field_Jacobi, norms_Jacobi = solvers.PCG(K_fun, rhs, x0=None, P=M_fun_Jacobi,
                                                                          steps=int(1),
                                                                          toler=1e-6, norm_type='data_scaled_rr',
                                                                          norm_metric=M_fun
                                                                          )
                    nb_it_Jacobi[nb_discretization_index + nb_starting_phases, nb_starting_phases, i] = (
                        len(norms_Jacobi['residual_rr']))
                    norm_rz_Jacobi.append(norms_Jacobi['residual_rz'])
                    # norm_rr_Jacobi.append(norms_Jacobi['residual_rr'])
                    norm_rMr_Jacobi.append(norms_Jacobi['data_scaled_rr'])
    ########################################   Uniform distribution   ###############################################
    print('Uniform distributio  ')

    N = 5
    r = 1

    A = np.zeros([r * N, r * N])
    M = np.zeros([r * N, r * N])

    # Create the diagonal matrix
    A = np.diag(np.arange(1, r * N + 1))
    A[-1, -1] = kappa

    M = np.diag(np.ones(r * N))

    if RHS == 'random':
        rhs = np.random.rand(r * N)
    elif RHS == 'linear':
        rhs = np.copy(np.diag(A))

    rhs = np.diag(A)  # $rhs=np.random.rand(N)
    if initial == 'random':
        x0 = np.random.rand(r * N)
    elif initial == 'zeros':
        x0 = np.zeros(r * N)  #

    A_fun = lambda x: A @ x
    M_fun = lambda x: M @ x

    eig_A, Q_A = sc.linalg.eig(a=A, b=M)  # , eigvals_only=True
    r0 = rhs - A_fun(x0)
    r0_norm = np.linalg.norm(r0)
    w_i = (np.dot(np.transpose(Q_A), r0 / r0_norm)) ** 2

    print(w_i)
    x_values = np.linspace(0, kappa, 100)
    #  Ritz values during iterations
    ritz_values = get_ritz_values(A=A, k_max=r * N, v0=r0)

    plot_ritz_values(ritz_values=ritz_values, true_eigenvalues=eig_A)

    x, norms_N = PCG(Afun=A_fun, B=rhs, x0=x0, P=M_fun, steps=int(5000), toler=1e-14, norm_energy_upper_bound=True,
                     lambda_min=np.real(eig_A[0]))

    plot_cg_polynomial(x_values, ritz_values, true_eigenvalues=eig_A, weight=w_i,
                       error_evol=norms_N['energy_lb'] / norms_N['residual_rr'][0])
    # print(x)
    # print(norms_N)

    fig = plt.figure(figsize=(4.5, 4.5))
    gs = fig.add_gridspec(1, 1, width_ratios=[1])
    ax_iterations = fig.add_subplot(gs[0, 0])
    ax_iterations.semilogy(iterations, convergence_G, "Grey", linestyle=':', label='Condition number  estimate',
                           linewidth=1)
    ax_iterations.semilogy(np.arange(1, norms_N['residual_rr'].__len__() + 1)[:],
                           norms_N['residual_rr'] / norms_N['residual_rr'][0], "g",
                           linestyle='-', marker='x', label='1,2,3,4,5', linewidth=1)
    ax_iterations.semilogy(np.arange(1, norms_N['energy_lb'].__len__() + 1)[:],
                           norms_N['energy_lb'] / norms_N['residual_rr'][0], "g",
                           linestyle='-', marker='x', label='Enorm/r0', linewidth=1)
    ax_iterations.set_xlim(1, x_lim_max)
    ax_iterations.set_xticks([1, 5, 10, 15, 20])
    ax_iterations.set_yscale('log')
    ax_iterations.set_ylim([1e-16, 1e1])
    ax_iterations.set_xlabel("PCG iterations")
    ax_iterations.set_ylabel(r"relative Norm of residua")
    ax_iterations.legend(loc='upper right')
    ax_iterations.set_title(f"x0={initial}, rhs={RHS}")

    src = '../figures/'  # source folder\
    fname = src + 'CG_conver_exampl1_original' + '{}'.format('.pdf')
    plt.savefig(fname, bbox_inches='tight')
    plt.show()
    ########################################   Uniform distribution --- repeated    ###############################################
    # Specify the size of the original matrix and the repetition factor
    print('Uniform distribution --- repeated ')
    r = 3  # repetition factor

    # Create the original diagonal matrix
    original_diag = np.arange(1, N + 1)
    original_diag[-1] = kappa
    # Repeat the diagonal elements 'r' times
    repeated_diag = np.tile(original_diag, r)
    A = np.diag(repeated_diag)
    # Create the new diagonal matrix
    A = np.diag(repeated_diag)

    # Create the new precondition matrix
    M = np.diag(np.ones(r * N))

    if RHS == 'random':
        rhs = np.random.rand(r * N)
    elif RHS == 'linear':
        rhs = np.copy(np.diag(A))

    if initial == 'random':
        x0 = np.random.rand(r * N)
    elif initial == 'zeros':
        x0 = np.zeros(r * N)  #

    A_fun = lambda x: A @ x
    M_fun = lambda x: M @ x

    eig_A, Q_A = sc.linalg.eig(a=A, b=M)  # , eigvals_only=True
    r0 = rhs - A_fun(x0)
    r0_norm = np.linalg.norm(r0)
    w_i = (np.dot(np.transpose(Q_A), r0 / r0_norm)) ** 2
    print(w_i)
    x_values = np.linspace(0, kappa, 100)
    # plot_cg_polynomial(x_values, ritz_values ,true_eigenvalues=eig_A,weight=w_i)
    #  Ritz values during iterations
    ritz_values = get_ritz_values(A=A, k_max=r * N, v0=r0)
    plot_ritz_values(ritz_values=ritz_values, true_eigenvalues=eig_A)

    x, norms_rep = PCG(Afun=A_fun, B=rhs, x0=x0, P=M_fun, steps=int(5000), toler=1e-14, norm_energy_upper_bound=True,
                       lambda_min=np.real(eig_A[0]))

    plot_cg_polynomial(x_values, ritz_values, true_eigenvalues=eig_A, weight=w_i,
                       error_evol=norms_rep['energy_lb'] / norms_rep['residual_rr'][0])

    fig = plt.figure(figsize=(4.5, 4.5))
    gs = fig.add_gridspec(1, 1, width_ratios=[1])
    ax_iterations = fig.add_subplot(gs[0, 0])
    ax_iterations.semilogy(iterations, convergence_G, "Grey", linestyle=':', label='Condition number  estimate',
                           linewidth=1)

    ax_iterations.semilogy(np.arange(1, norms_N['residual_rr'].__len__() + 1)[:],
                           norms_N['residual_rr'] / norms_N['residual_rr'][0], "g",
                           linestyle='-', marker='x', label='1,2,3,4,5', linewidth=1)

    ax_iterations.semilogy(np.arange(1, norms_rep['residual_rr'].__len__() + 1)[:],
                           norms_rep['residual_rr'] / norms_rep['residual_rr'][0], "b",
                           linestyle=':', marker='^', label=f'Repeated {r} times: N={r * N}', linewidth=1)

    ax_iterations.set_xlim(1, x_lim_max)
    ax_iterations.set_xticks([1, 5, 10, 15, 20])
    # ax_iterations.set_ylim([1, 2600])
    # ax_iterations.set_yscale('linear')
    ax_iterations.set_yscale('log')
    ax_iterations.set_ylim([1e-16, 1e1])
    ax_iterations.set_xlabel("PCG iterations")
    ax_iterations.set_ylabel(r"relative Norm of residua")
    ax_iterations.legend(loc='upper right')
    ax_iterations.set_title(f"x0={initial}, rhs={RHS}")

    src = '../figures/'  # source folder\
    fname = src + 'CG_conver_exampl1_repeat' + '{}'.format('.pdf')
    plt.savefig(fname, bbox_inches='tight')

    ########################################   Linear distribution --- Small  ###############################################
    print('Linear distribution --- Small ')

    A = np.zeros([r * N, r * N])
    M = np.zeros([r * N, r * N])

    # Create the diagonal matrix
    A = np.diag(np.linspace(1, kappa, r * N))
    M = np.diag(np.diag(np.ones_like(A)))

    if RHS == 'random':
        rhs = np.random.rand(r * N)
    elif RHS == 'linear':
        rhs = np.copy(np.diag(A))

    if initial == 'random':
        x0 = np.random.rand(r * N)
    elif initial == 'zeros':
        x0 = np.zeros(r * N)  #

    A_fun = lambda x: A @ x
    M_fun = lambda x: M @ x

    eig_A, Q_A = sc.linalg.eig(a=A, b=M)  # , eigvals_only=True
    r0 = rhs - A_fun(x0)
    r0_norm = np.linalg.norm(r0)
    w_i = (np.dot(np.transpose(Q_A), r0 / r0_norm)) ** 2
    print(w_i)
    #  Ritz values during iterations
    ritz_values = get_ritz_values(A=A, k_max=r * N, v0=r0)
    x_values = np.linspace(0, kappa, 100)
    # plot_cg_polynomial(x_values, ritz_values ,true_eigenvalues=eig_A,weight=w_i)
    plot_ritz_values(ritz_values=ritz_values, true_eigenvalues=eig_A)

    x, norms_linearN = PCG(Afun=A_fun, B=rhs, x0=x0, P=M_fun, steps=int(5000), toler=1e-14,
                           norm_energy_upper_bound=True, lambda_min=np.real(eig_A[0]))

    plot_cg_polynomial(x_values, ritz_values, true_eigenvalues=eig_A, weight=w_i,
                       error_evol=norms_linearN['energy_lb'] / norms_linearN['residual_rr'][0])

    print(x)
    print(norms_linearN)

    fig = plt.figure(figsize=(9, 9))
    gs = fig.add_gridspec(1, 1, width_ratios=[1])
    ax_iterations = fig.add_subplot(gs[0, 0])

    ax_iterations.semilogy(iterations, convergence_G, "Grey", linestyle=':', label='Condition number  estimate',
                           linewidth=1)

    ax_iterations.semilogy(np.arange(1, norms_linearN['residual_rr'].__len__() + 1)[:],
                           norms_linearN['residual_rr'] / norms_linearN['residual_rr'][0], "black",
                           linestyle='-.', marker='>', label=f'linear distribution: N={N}', linewidth=1)

    ax_iterations.set_xlim(1, x_lim_max)
    ax_iterations.set_xticks([1, 5, 10, 15, 20])
    # ax_iterations.set_ylim([1, 2600])
    # ax_iterations.set_yscale('linear')
    ax_iterations.set_yscale('log')
    ax_iterations.set_ylim([1e-16, 1e1])
    ax_iterations.set_xlabel("PCG iterations")
    ax_iterations.set_ylabel(r"relative Norm of residua")
    ax_iterations.legend(loc='upper right')
    src = '../figures/'  # source folder\
    fname = src + f'CG_conver_exampl1_lin{N}' + '{}'.format('.pdf')

    plt.savefig(fname, bbox_inches='tight')

    ########################################   Linear distribution --- Large    ###############################################
    print('Linear distribution --- Large ')
    N_large = 50
    # kappa=5
    A = np.zeros([N_large, N_large])
    M = np.zeros([N_large, N_large])

    # Create the diagonal matrix
    A = np.diag(np.linspace(1, kappa, N_large))

    M = np.diag(np.diag(np.ones_like(A)))

    if RHS == 'random':
        rhs = np.random.rand(N_large)
    elif RHS == 'linear':
        rhs = np.copy(np.diag(A))

    if initial == 'random':
        x0 = np.random.rand(N_large)
    elif initial == 'zeros':
        x0 = np.zeros(N_large)

    # rhs=np.diag(A) #$rhs=np.random.rand(N)
    # x0=np.random.rand(N_large)
    eig_A, Q_A = sc.linalg.eig(a=A, b=M)  # , eigvals_only=True
    r0 = rhs - A_fun(x0)
    r0_norm = np.linalg.norm(r0)
    w_i = (np.dot(np.transpose(Q_A), r0 / r0_norm)) ** 2
    print(w_i)
    x_values = np.linspace(0, kappa, 100)

    #  Ritz values during iterations
    ritz_values = get_ritz_values(A=A, k_max=N_large, v0=r0)
    # plot_cg_polynomial(x_values, ritz_values ,true_eigenvalues=eig_A,weight=w_i)
    plot_ritz_values(ritz_values=ritz_values, true_eigenvalues=eig_A)

    A_fun = lambda x: A @ x
    M_fun = lambda x: M @ x

    x, norms_linearN_large = PCG(Afun=A_fun, B=rhs, x0=x0, P=M_fun, steps=int(5000), toler=1e-14,
                                 norm_energy_upper_bound=True, lambda_min=np.real(eig_A[0]))

    plot_cg_polynomial(x_values, ritz_values, true_eigenvalues=eig_A, weight=w_i,
                       error_evol=norms_linearN_large['energy_lb'] / norms_linearN_large['residual_rr'][0])

    ########################################   Linear distribution --- Sparse RHS    ###############################################

    # Create the diagonal matrix
    A = np.diag(np.linspace(1, kappa, N_large))

    M = np.diag(np.diag(np.ones_like(A)))
    # M=np.diag(np.linspace(1,kappa, N_large)**-1)

    if RHS == 'random':
        rhs = np.random.rand(N_large)
    elif RHS == 'linear':
        rhs = np.copy(np.diag(A))

    if initial == 'random':
        x0 = np.random.rand(N_large)
    elif initial == 'zeros':
        x0 = np.zeros(N_large)

    rhs[4:N_large - 4] = 0

    eig_A, Q_A = sc.linalg.eig(a=A, b=M)  # , eigvals_only=True
    r0 = rhs - A_fun(x0)
    r0_norm = np.linalg.norm(r0)
    w_i = (np.dot(np.transpose(Q_A), r0 / r0_norm)) ** 2
    print(w_i)
    x_values = np.linspace(0, kappa, 100)
    #  Ritz values during iterations
    ritz_values = get_ritz_values(A=A, k_max=N_large, v0=r0)

    # plot_cg_polynomial(x_values, ritz_values ,true_eigenvalues=eig_A,weight=w_i)

    A_fun = lambda x: A @ x
    M_fun = lambda x: M @ x
    x, norms_linear_sparse_rhs = PCG(Afun=A_fun, B=rhs, x0=x0, P=M_fun, steps=int(5000), toler=1e-14,
                                     norm_energy_upper_bound=True, lambda_min=np.real(eig_A[0]))

    plot_cg_polynomial(x_values, ritz_values, true_eigenvalues=eig_A, weight=w_i,
                       error_evol=norms_linear_sparse_rhs['energy_lb'] / norms_linear_sparse_rhs['residual_rr'][0])
    print(norms_linear_sparse_rhs)
    #########################################################################################################################

    #########################################################################################################################

    fig = plt.figure(figsize=(4.5, 4.5))
    gs = fig.add_gridspec(1, 1, width_ratios=[1])
    ax_iterations = fig.add_subplot(gs[0, 0])
    ax_iterations.semilogy(iterations, convergence_G, "Grey", linestyle=':', label='Condition number  estimate',
                           linewidth=1)

    ax_iterations.semilogy(np.arange(1, norms_N['residual_rr'].__len__() + 1)[:],
                           norms_N['residual_rr'] / norms_N['residual_rr'][0], "g", linestyle='-', marker='x',
                           label='1,2,3,4,5', linewidth=1)

    ax_iterations.semilogy(np.arange(1, norms_rep['residual_rr'].__len__() + 1)[:],
                           norms_rep['residual_rr'] / norms_rep['residual_rr'][0], "b",
                           linestyle=':', marker='^', label=f'Repeated {r} times: N={r * N}', linewidth=1)
    ax_iterations.semilogy(np.arange(1, norms_linearN['residual_rr'].__len__() + 1)[:],
                           norms_linearN['residual_rr'] / norms_linearN['residual_rr'][0], "black",
                           linestyle='-.', marker='>', label=f'np.linspace(1,{kappa},N={r * N})', linewidth=1)

    ax_iterations.set_xlim(1, x_lim_max)
    ax_iterations.set_xticks([1, 5, 10, 15, 20])
    # ax_iterations.set_ylim([1, 2600])
    # ax_iterations.set_yscale('linear')
    ax_iterations.set_yscale('log')
    ax_iterations.set_ylim([1e-16, 1e3])
    ax_iterations.set_xlabel("PCG iterations")
    ax_iterations.set_ylabel(r" Relative norm of residua")
    ax_iterations.legend(loc='upper right')
    ax_iterations.set_title(f"x0={initial}, rhs={RHS}")

    src = '../figures/'  # source folder\
    fname = src + f'CG_conver_exampl1_linear{N}' + '{}'.format('.pdf')
    plt.savefig(fname, bbox_inches='tight')

    #########################################################################################################################
    fig = plt.figure(figsize=(4.5, 4.5))
    gs = fig.add_gridspec(1, 1, width_ratios=[1])
    ax_iterations = fig.add_subplot(gs[0, 0])
    ax_iterations.semilogy(iterations, convergence_G, "Grey", linestyle=':', label='Condition number  estimate',
                           linewidth=1)

    ax_iterations.semilogy(np.arange(1, norms_N['residual_rr'].__len__() + 1)[:],
                           norms_N['residual_rr'] / norms_N['residual_rr'][0], "g", linestyle='-', marker='x',
                           label='1,2,3,4,5', linewidth=1)

    ax_iterations.semilogy(np.arange(1, norms_rep['residual_rr'].__len__() + 1)[:],
                           norms_rep['residual_rr'] / norms_rep['residual_rr'][0], "b",
                           linestyle=':', marker='^', label=f'Repeated {r} times: N={r * N}', linewidth=1)
    ax_iterations.semilogy(np.arange(1, norms_linearN['residual_rr'].__len__() + 1)[:],
                           norms_linearN['residual_rr'] / norms_linearN['residual_rr'][0], "black",
                           linestyle='-.', marker='>', label=f'np.linspace(1,{kappa},N={r * N}) ', linewidth=1)

    ax_iterations.semilogy(np.arange(1, norms_linearN_large['residual_rr'].__len__() + 1)[:],
                           norms_linearN_large['residual_rr'] / norms_linearN_large['residual_rr'][0], "black",
                           linestyle='-.', marker='x', label=f'np.linspace(1,{kappa}, N={N_large})', linewidth=1)

    ax_iterations.set_xlim(1, x_lim_max)
    ax_iterations.set_xticks([1, 5, 10, 15, 20])
    # ax_iterations.set_ylim([1, 2600])
    # ax_iterations.set_yscale('linear')
    ax_iterations.set_yscale('log')
    ax_iterations.set_ylim([1e-16, 1e3])
    ax_iterations.set_xlabel("PCG iterations")
    ax_iterations.set_ylabel(r" Relative norm of residua")
    ax_iterations.legend(loc='upper right')
    src = '../figures/'  # source folder\
    fname = src + f'CG_conver_exampl1_linear_2{N_large}' + '{}'.format('.pdf')
    plt.savefig(fname, bbox_inches='tight')

    #########################################################################################################################

    fig = plt.figure(figsize=(4.5, 4.5))
    gs = fig.add_gridspec(1, 1, width_ratios=[1])
    ax_iterations = fig.add_subplot(gs[0, 0])
    ax_iterations.semilogy(iterations, convergence_G, "Grey", linestyle=':', label='Condition number  estimate',
                           linewidth=1)

    ax_iterations.semilogy(np.arange(1, norms_N['residual_rr'].__len__() + 1)[:],
                           norms_N['residual_rr'] / norms_N['residual_rr'][0], "g",
                           linestyle='-', marker='x', label='1,2,3,4,5', linewidth=1)

    ax_iterations.semilogy(np.arange(1, norms_rep['residual_rr'].__len__() + 1)[:],
                           norms_rep['residual_rr'] / norms_rep['residual_rr'][0], "b",
                           linestyle=':', marker='^', label=f'Repeated {r} times: N={r * N}', linewidth=1)
    ax_iterations.semilogy(np.arange(1, norms_linearN['residual_rr'].__len__() + 1)[:],
                           norms_linearN['residual_rr'] / norms_linearN['residual_rr'][0], "black",
                           linestyle='-.', marker='>', label=f'np.linspace(1,{kappa},N={r * N}) ', linewidth=1)
    ax_iterations.semilogy(np.arange(1, norms_linearN_large['residual_rr'].__len__() + 1)[:],
                           norms_linearN_large['residual_rr'] / norms_linearN_large['residual_rr'][0], "black",
                           linestyle='-.', marker='x', label=f'np.linspace(1,{kappa}, N={N_large})', linewidth=1)

    ax_iterations.semilogy(np.arange(1, norms_linear_sparse_rhs['residual_rr'].__len__() + 1)[:],
                           norms_linear_sparse_rhs['residual_rr'] / norms_linear_sparse_rhs['residual_rr'][0], "black",
                           linestyle='-.', marker='o', label=f'np.linspace(1,{kappa}, N={N_large}),\n sparse RHS ',
                           linewidth=1)

    ax_iterations.set_xlim(1, x_lim_max)
    ax_iterations.set_xticks([1, 5, 10, 15, x_lim_max])
    # ax_iterations.set_ylim([1, 2600])
    # ax_iterations.set_yscale('linear')
    ax_iterations.set_yscale('log')
    ax_iterations.set_ylim([1e-16, 1e3])
    ax_iterations.set_xlabel("PCG iterations")
    ax_iterations.set_ylabel(r" Relative norm of residua")
    ax_iterations.legend(loc='upper right')
    ax_iterations.set_title(f"x0={initial}, rhs={RHS}")

    src = '../figures/'  # source folder\
    fname = src + f'CG_conver_exampl1_linear_zerosrhsx0={initial}rhs={RHS}' + '{}'.format('.pdf')
    plt.savefig(fname, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    for initial in ['zeros']:  # , 'random'
        for RHS in ['random']:  # , 'random'
            kappa = 10

            run_simple_CG_Green(initial=initial, RHS=RHS, kappa=kappa)
