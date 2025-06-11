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

from experiments.experiments_with_CG_convergence.simple_CG import get_ritz_values, plot_ritz_values, get_cg_polynomial, \
    plot_cg_polynomial, plot_eigenvectors, \
    plot_eigendisplacement, plot_rhs, plot_eigenvector_filling, plot_cg_polynomial_JG_paper

from experiments.paper_Jacobi_Green.exp_paper_JG_geometry_plots import get_triangle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


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


def matrix_sqrt_eig(A, nb_zero_eigens=2):
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
    sqrt_eigvals = np.copy(eigvals)
    sqrt_eigvals[nb_zero_eigens:] = np.sqrt(eigvals[nb_zero_eigens:])

    # Reconstruct A^(1/2)
    A_sqrt = eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T
    return A_sqrt


def run_simple_CG_Green(initial, RHS, kappa):
    problem_type = 'elasticity'
    discretization_type = 'finite_element'
    element_type = 'linear_triangles'
    formulation = 'small_strain'
    src = './figures/'  # source folder\
    # Enable LaTeX rendering
    plt.rcParams.update({
        "text.usetex": True,  # Use LaTeX
        # "font.family": "helvetica",  # Use a serif font
    })
    plt.rcParams.update({'font.size': 11})
    plt.rcParams["font.family"] = "Arial"

    domain_size = [1, 1]
    geom_n = [4]  # 3,,4,5,6 ,6,7,8,9,10,]  # ,2,3,3,2,  #,5,6,7,8,9 ,5,6,7,8,9,10,11

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
                    nb_laminates = 4
                    if nb_discretization_index == 0:
                        phase_fied_small_grid = microstructure_library.get_geometry(
                            nb_voxels=discretization.nb_of_pixels,
                            microstructure_name=geometry_ID,
                            coordinates=discretization.fft.coords,
                            seed=1,
                            parameter=nb_laminates)  # ,
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
                        adjust_system = True

                        K = discretization.get_system_matrix(material_data_field_C_0_rho)
                        # fixing zero eigenvalues
                        reduced_K = np.copy(K)
                        K[:, 0] = 0
                        K[0, :] = 0
                        K[:, np.prod(number_of_pixels)] = 0
                        K[np.prod(number_of_pixels), :] = 0
                        K[0, 0] = 50.5
                        K[np.prod(number_of_pixels), np.prod(number_of_pixels)] = 50.5

                        eig_K, eig_vect_K = sc.linalg.eig(a=K, b=None)  # , eigvals_only=True
                        eig_K = np.real(eig_K)
                        eig_K[eig_K == 50.5] = 0
                        # Sort in descending order (largest eigenvalues first)
                        idx_K = np.argsort(eig_K)[::-1]  # Get indices of sorted eigenvalues
                        # Reorder eigenvalues and eigenvectors
                        sorte_eig_K = eig_K[idx_K]
                        sorted_eig_vect_K = eig_vect_K[:, idx_K]

                        # Greeen precond
                        M = discretization.get_system_matrix(refmaterial_data_field_C1)
                        # M2 = discretization.get_preconditioner_NEW(refmaterial_data_field_C1)
                        # RM2=discretization.fft.ifft(M2[:,0,:,0])
                        # M_inv_fromfft=np.zeros_like(M)
                        # np.fill_diagonal(M_inv_fromfft[:256,:256], RM2[0,0].flatten())
                        M_non_reduced = np.copy(M)
                        # fixing zero eigenvalues
                        M[:, 0] = 0
                        M[0, :] = 0
                        M[:, np.prod(number_of_pixels)] = 0
                        M[np.prod(number_of_pixels), :] = 0
                        M[0, 0] = 1
                        M[np.prod(number_of_pixels), np.prod(number_of_pixels)] = 1
                        ##### Left preconditioned
                        MiK = np.linalg.pinv(M) @ K

                        ####
                        eig_G, eig_vect_G = sc.linalg.eig(a=K, b=M)  # , eigvals_only=True
                        eig_G_no_zero = np.copy(eig_G)
                        eig_G = np.real(eig_G)
                        eig_G[eig_G == 50.5] = 0

                        ### symmetrized precondioner
                        M_sym = matrix_sqrt_eig(A=M, nb_zero_eigens=0)
                        Green_sqrt = matrix_sqrt_eig(A=np.linalg.pinv(M), nb_zero_eigens=0)
                        GKGsym = Green_sqrt @ K @ Green_sqrt
                        eig_G_sym, eig_vect_G_sym = sc.linalg.eig(a=GKGsym)  # , eigvals_only=True
                        #
                        # eig_GK, eig_vect_GK = sc.linalg.eig(a=np.linalg.inv(M) @ K )  # , eigvals_only=True
                        # Sort in descending order (largest eigenvalues first)
                        idx_G = np.argsort(eig_G)[::-1]  # Get indices of sorted eigenvalues

                        # Reorder eigenvalues and eigenvectors
                        sorte_eig_G = eig_G[idx_G]
                        sorted_eig_vect_G = eig_vect_G[:, idx_G]

                        # #### jacobi preconditioner
                        # Jacobi_sym = np.copy(np.diag(K))
                        # Jacobi_sym[Jacobi_sym < 9.99e-16] = 0
                        # Jacobi_sym[Jacobi_sym != 0] = 1 / np.sqrt(Jacobi_sym[Jacobi_sym != 0])
                        #
                        # K_diag_inv = np.copy(np.diag(K))
                        # K_diag_inv[K_diag_inv < 9.99e-16] = 0
                        # K_diag_inv[K_diag_inv != 0] = 1 / K_diag_inv[K_diag_inv != 0]
                        #
                        # K_diag_sym = matrix_sqrt_eig(np.diag(np.diag(K)))
                        #
                        # JKJsym = np.matmul(np.diag(Jacobi_sym), np.matmul(K, np.diag(Jacobi_sym)))
                        # eig_J, eig_vect_J = sc.linalg.eig(a=JKJsym)  # , eigvals_only=True
                        #
                        # eig_JG, eig_vect_JG = sc.linalg.eig(a=JKJsym, b=M)  # , eigvals_only=True
                        # idx_JG = np.argsort(eig_JG)[::-1]
                        # plot_eigendisplacement(eigenvectors_1=eig_vect_JG[:, idx_JG],
                        #                        grid_shape=rhs.shape, dim=2, eigenvals=eig_JG[idx_JG],
                        #                        weight=eig_JG[idx_JG], participation_ratios=participation_ratios)
                    rhs.flatten()[0] = 0
                    rhs.flatten()[np.prod(number_of_pixels)] = 0

                    ######### INITIAL SOLUTION
                    # x0 = np.random.random(discretization.get_displacement_sized_field().shape)
                    x0 = np.zeros(discretization.get_displacement_sized_field().shape)

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
                    # plot_ritz_values(ritz_values=ritz_values, true_eigenvalues=eig_G)

                    displacement_field, norms = solvers.PCG(K_fun_G, rhs_G.flatten(), x0=M_sym @ x0.flatten(), P=M_null,
                                                            steps=int(1000), toler=1e-14,
                                                            norm_energy_upper_bound=True,
                                                            lambda_min=np.real(sorted(eig_G)[-2])
                                                            )



                    # displacement_field, norms = solvers.PCG(K_fun, rhs, x0=x0, P=M_fun,
                    #                                         steps=int(1000), toler=1e-14,
                    #                                         norm_energy_upper_bound=True,
                    #                                         lambda_min=np.real(sorted(eig_G)[0])
                    #                                         )
                    plot_cg_polynomial_JG_paper(x_values, ritz_values, true_eigenvalues=eig_G_no_zero[idx_G], weight=w_i[idx_G],
                                       error_evol=norms['energy_upper_bound'] / norms['residual_rr'][
                                           0], title='Green')  # energy_lb
                    plot_cg_polynomial(x_values, ritz_values, true_eigenvalues=eig_G_no_zero[idx_G],
                                                weight=w_i[idx_G],
                                                error_evol=norms['energy_upper_bound'] / norms['residual_rr'][
                                                    0], title='Green')  # energy_lb

                    # PLOTS
                    # vector_to_plot=np.abs(MiK)#.transpose()
                    vector_to_plot = eig_vect_G[:, idx_G]
                    rhs_to_plot = rhs  # +1
                    weights_to_plot = w_i[idx_G]
                    eigens_to_plots = eig_G[idx_G]

                    # Plot the convergence of Ritz values
                    grid_shape = rhs.shape
                    x = np.arange(0, number_of_pixels[0])
                    y = np.arange(0, number_of_pixels[1])
                    x, y = np.meshgrid(x, y)

                    # fig = plt.figure(figsize=(11, 5.5))
                    fig = plt.figure(figsize=(8.3, 4.5))

                    gs = fig.add_gridspec(1, 1)
                    ax_global = fig.add_subplot(gs[:, :])

                    ax_global.plot(np.arange(1, 513), eigens_to_plots[::-1], color='Green', label=f'Green',
                                   alpha=0.5, marker='.', linewidth=0, markersize=5)

                    opacities = np.abs(np.real(weights_to_plot[::-1]))
                    opacities[0:2] = 0
                    opacities[opacities > 1e-14] = 1
                    opacities[opacities < 1] = 0
                    ax_global.scatter(np.arange(1, 513), eigens_to_plots[::-1], color='Blue', label=f'Green',
                                      alpha=opacities)
                    # plt.gca().invert_xaxis()  # Invert X-axis
                    ax_global.plot(1, color='Red',
                                   alpha=1.0, marker='.', linewidth=0, markersize=4)
                    # weights_to_plot

                    ax_global.set_xlim([1, 512])
                    ax_global.set_xticks([1, 256, 512])
                    ax_global.set_xticklabels([1, 256, 512])
                    ax_global.set_ylim([-1, 103])
                    ax_global.set_yticks([1, 34, 67, 100])
                    ax_global.set_yticklabels([1, 34, 67, 100])
                    #
                    ax_global.set_title(f'\n Sparsity patterns in eigenvectors')
                    ax_global.set_xlabel('eigenvalue index - $i$ (sorted)')
                    ax_global.set_ylabel(r'Eigenvalue - $\lambda_{i}$')
                    ax_global.text(0.02, 0.65,
                                   r'Geometry - $\mathcal{G}_' + f'{nb_laminates}$\n' + r'Mesh - $\mathcal{T}_{16}$',
                                   transform=ax_global.transAxes, fontsize=13)

                    # Create a custom legend symbol
                    custom_symbol = [
                        mpl.lines.Line2D([], [], color='Green', marker='.', linestyle='None', markersize=10,
                                         label='Eigenvalues'),
                        mpl.lines.Line2D([], [], color='Blue', marker='.', linestyle='None',
                                         markersize=10,
                                         label='Non-zero weights'),
                        mpl.lines.Line2D([], [], color='Red', marker='.', linestyle='None',
                                         markersize=10,
                                         label='Non-zero element of vector')]
                    # Add legend with the custom symbol
                    plt.legend(handles=custom_symbol, loc='upper left')
                    # ax_global.legend([f'Eigenvalues ', f'Non-zero weights', f'Non-zero element of eigenvector/ first residual '], loc='upper left')
                    plot_eigenvectors=False
                    if plot_eigenvectors:
                        x_offset = -0.5
                        y_offset = 1.1
                        for upper_ax in np.arange(6):
                            # weight = np.array([0.2, 1, 10, 30, 100])[upper_ax]
                            if upper_ax == 0:
                                # ax1 = fig.add_subplot(gs[0, upper_ax])
                                ax1 = fig.add_axes([0.15, 0.14, 0.1, 0.20])
                                ax1.set_title(r'$\phi_{i}\, (\lambda_{i}=1) $ ')
                                roll_x = 5
                                roll_y = -20
                                # ax_global.annotate('',
                                #              xy=(weight, f_sigmas[np.where(weights == weight)[0][0]]),
                                #              xytext=(0.23, 0.1),
                                #              arrowprops=dict(arrowstyle='->',
                                #                              color='black',
                                #                              lw=1,
                                #                              ls='-')
                                #              )
                                ax_global.text(x_offset + 0.4, y_offset + 0.3, '(a.1)', transform=ax1.transAxes)  #
                                i = 20
                            if upper_ax == 1:
                                ax1 = fig.add_axes([0.34, 0.38, 0.1, 0.20])
                                ax1.set_title(r'$\phi_{i}\, (\lambda_{i}=34) $ ')

                                ax_global.text(x_offset, y_offset, '(a.2)', transform=ax1.transAxes)
                                i = 150

                            if upper_ax == 2:
                                ax1 = fig.add_axes([0.58, 0.62, 0.1, 0.20])
                                ax1.set_title(r'$\phi_{i}\, (\lambda_{i}=67) $ ')
                                ax_global.text(x_offset, y_offset, '(a.3)', transform=ax1.transAxes)

                                i = 350
                            if upper_ax == 3:
                                ax1 = fig.add_axes([0.776, 0.64, 0.1, 0.20])
                                #ax1.set_title(r'$\phi_{i}\, (\lambda_{i}=100) $ ')

                                ax_global.text(x_offset + 0.01, y_offset - 1.4,
                                               r'(a.4) $\phi_{i}\, (\lambda_{i}=100) $',
                                               transform=ax1.transAxes, fontsize=13)
                                i = 500
                            if upper_ax == 4:
                                i = 115
                                # ax1 = fig.add_axes([0.17, 0.6, 0.1, 0.20])

                                ax1 = fig.add_axes([0.60, 0.23, 0.1, 0.20])
                                ax_global.text(x_offset + 0.25, y_offset + 0., '(b.1)', transform=ax1.transAxes)

                                ax_global.annotate('',
                                                   xy=(i + 23, sorted(eig_G)[i]),
                                                   xytext=(350, 20),
                                                   arrowprops=dict(arrowstyle='<-',
                                                                   color='black',
                                                                   lw=0.5,
                                                                   ls='-')
                                                   )
                                # Add an ellipse
                                ellipse = Ellipse(xy=(i, sorted(eig_G)[i]), width=50.0, height=15.0, angle=0,
                                                  edgecolor='black',
                                                  facecolor='none', linewidth=0.5)
                                ax_global.add_patch(ellipse)

                                i = 256
                                ax_global.annotate('',
                                                   xy=(i + 25, sorted(eig_G)[i] - 7),
                                                   xytext=(350, 25),
                                                   arrowprops=dict(arrowstyle='<-',
                                                                   color='black',
                                                                   lw=0.5,
                                                                   ls='-')
                                                   )
                                # Add an ellipse
                                ellipse = Ellipse(xy=(i, sorted(eig_G)[i]), width=71.0, height=31.0, angle=10,
                                                  edgecolor='black',
                                                  facecolor='none', linewidth=0.5)
                                ax_global.add_patch(ellipse)
                                i = 400

                                ax_global.annotate('',
                                                   xy=(i, sorted(eig_G)[i] - 7),
                                                   xytext=(370, 31),
                                                   arrowprops=dict(arrowstyle='<-',
                                                                   color='black',
                                                                   lw=0.5,
                                                                   ls='-')
                                                   )
                                # Add an ellipse
                                ellipse = Ellipse(xy=(i, sorted(eig_G)[i]), width=50.0, height=15.0, angle=0,
                                                  edgecolor='black',
                                                  facecolor='none', linewidth=0.5)
                                ax_global.add_patch(ellipse)

                            if upper_ax == 5:
                                i = -1
                                ax1 = fig.add_axes([0.76, 0.12, 0.1, 0.20])
                                ax1.set_title(f'Initial residual')
                                ax_global.text(x_offset + 0.1, y_offset + 0.3, '(c.1)', transform=ax1.transAxes)
                            if upper_ax == 6:
                                i = -2
                                ax1 = fig.add_axes([0.15, 0.5, 0.1, 0.20])
                                ax1.set_title(r'Weights')
                                ax_global.text(x_offset, y_offset + 0.2, '(d.1)', transform=ax1.transAxes)

                            if i > 0:
                                eigenvector_x = np.real(vector_to_plot[:, ::-1])[:, i].reshape(grid_shape)[
                                    0, 0].transpose()
                                eigenvector_y = np.real(vector_to_plot[:, ::-1])[:, i].reshape(grid_shape)[
                                    1, 0].transpose()
                            elif i == -1:
                                eigenvector_x = np.real(rhs_to_plot).reshape(grid_shape)[
                                    0, 0].transpose()
                                eigenvector_y = np.real(rhs_to_plot).reshape(grid_shape)[
                                    1, 0].transpose()
                            elif i == -2:
                                eigenvector_x = np.real(w_i).reshape(grid_shape)[
                                    0, 0].transpose()
                                eigenvector_y = np.real(w_i).reshape(grid_shape)[
                                    1, 0].transpose()
                            amplitude = np.sqrt(eigenvector_x ** 2 + eigenvector_y ** 2)

                            divnorm = mpl.colors.Normalize(vmin=0, vmax=2)
                            # Define facecolors: Use 'none' for empty elements (zeros) and color for others
                            facecolors = ['none' if value == 0 else 'red' for value in amplitude.flatten()]
                            # Plot circles with empty ones for zero values
                            # plt.scatter(x_coords_flat, y_coords_flat, s=A_flat * 100, facecolors=facecolors, edgecolors='blue', alpha=0.7)
                            sizes = np.copy(amplitude)  # .flatten()
                            sizes[sizes > 1e-10] = 1

                            circles_sizes = 20 * np.ones_like(amplitude)
                            circles_sizes[-1, :] = 0
                            circles_sizes[:, -1] = 0
                            # ax1.scatter(x, y, s=circles_sizes.flatten(), c='white', cmap='Reds', alpha=1.0,
                            #                       norm=divnorm, edgecolors='black', linewidths=0.1),

                            triangles, X, Y = get_triangle(nx=number_of_pixels[0], ny=number_of_pixels[1]
                                                           , lx=number_of_pixels[0], ly=number_of_pixels[1])
                            # Create the triangulation object
                            triangulation = tri.Triangulation(X, Y, triangles)
                            ax1.triplot(triangulation, 'k-', lw=0.1)
                            # ax1.axis('equal')   \

                            x_ = np.arange(0 + 0.5, 1 * number_of_pixels[0] + 0.5)
                            y_ = np.arange(0 + 0.5, 1 * number_of_pixels[1] + 0.5)
                            X_, Y_ = np.meshgrid(x_, y_)
                            ax1.pcolormesh(X_, Y_, np.transpose(phase_field),
                                           cmap=mpl.cm.Greys, vmin=1, vmax=100, linewidth=0,
                                           rasterized=True, alpha=0.6)
                            # Remove the frame (axes spines)
                            # for spine in ax1.spines.values():
                            #     spine.set_visible(False)

                            # Optionally, hide the axes ticks and labels
                            ax1.set_xticks([])
                            ax1.set_yticks([])

                            ax1.scatter(x, y, s=sizes * 10, c=facecolors, cmap='Reds', alpha=1.0, norm=divnorm,
                                        edgecolors='Red', linewidths=0)
                            ax1.set_aspect('equal', 'box')

                            # ax1.set_xlim(0,   1)
                            # ax1.set_ylim(0,  1)

                    fname = src + 'exp1_sparsity_of_eigenvectors{}{}'.format(nb_laminates, '.pdf')
                    print(('create figure: {}'.format(fname)))
                    plt.savefig(fname, bbox_inches='tight')
                    plt.show()
                    quit()
                    # Plot the convergence of Ritz values
                    fig = plt.figure(figsize=(4.0, 4))
                    gs = fig.add_gridspec(2, 1, width_ratios=[1])
                    ax_poly = fig.add_subplot(gs[0, 0])
                    ax_weights = fig.add_subplot(gs[1, 0])
                    #            ax_error_true = fig.add_subplot(gs[1, 1])

                    # ax_poly.plot(x_values, polynomial_value, color='red', label=r'$\varphi^{CG}$' + f'$_{{{i}}}$')
                    # ax_poly.hlines(xmin=0, xmax=x_values[-1], y=0, linestyles='--', color='gray')
                    ax_poly.scatter(np.real(eigens_to_plots), [0] * len(eigens_to_plots), color='blue',
                                    marker='|',
                                    label="True Eigenvalues")
                    # ax_poly.scatter(np.real(ritz_values[5]), [0] * len(ritz_values[5]), color='red', marker='x',
                    #                 label=f"Ritz Values\n (Approx Eigenvalues)")
                    ax_weights.scatter(np.real(eigens_to_plots), np.real(weights_to_plot) / np.real(eigens_to_plots),
                                       color='red',
                                       marker='o', label=r"\frac{w_{i}}{\lamnda_{i}}")
                    ax_weights.set_yscale('log')
                    ax_weights.set_ylim(1e-10, 1)
                    ax_weights.set_xlim(-0.1, x_values[-1] + 0.3)
                    ax_weights.set_ylabel(r"$w_{i}/ \lambda_{i}$")
                    ax_weights.set_title(f"Weights / Eigens ")

                    # ax_poly.set_xlabel("Eigenvalues --- Approximation")
                    # ax_poly.set_ylabel("CG (Lanczos) Iteration")
                    ax_poly.set_title(f"CG polynomial (Lanczos Iteration) {{{i}}}")
                    # ax_poly.set_ylim(ylim[0], ylim[1])
                    ax_poly.set_xlim(-0.1, x_values[-1] + 0.3)
                    ax_poly.legend()

                    # Automatically adjust subplot parameters to avoid overlapping
                    plt.tight_layout()

                    src = '../figures/'  # source folder\
                    fname = src + f'weights_sparsity' + '{}'.format('.pdf')
                    plt.savefig(fname, bbox_inches='tight')
                    plt.show()
                    plot_cg_polynomial(x_values, ritz_values, true_eigenvalues=eigens_to_plots, weight=weights_to_plot,
                                       error_evol=norms['energy_upper_bound'] / norms['residual_rr'][
                                           0], title='Green')  # energy_lb

                    quit()

                    print(norms)
                    plot_cg_polynomial(x_values, ritz_values, true_eigenvalues=eig_G[idx_G], weight=w_i[idx_G],
                                       error_evol=norms['energy_upper_bound'] / norms['residual_rr'][
                                           0], title='Green')  # energy_lb
                    #      quit()
                    plot_rhs(rhs=np.real(rhs),
                             grid_shape=rhs.shape)
                    plot_rhs(rhs=np.real(w_i.reshape(rhs.shape)),
                             grid_shape=rhs.shape)
                    quit()
                    plot_eigenvector_filling(np.real(sorted_eig_vect_G), grid_shape=rhs.shape)

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


if __name__ == '__main__':

    for initial in ['zeros']:  # , 'random'
        for RHS in ['random']:  # , 'random'
            kappa = 10

            run_simple_CG_Green(initial=initial, RHS=RHS, kappa=kappa)
