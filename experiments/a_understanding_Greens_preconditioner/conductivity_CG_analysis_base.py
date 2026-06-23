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

from trivial_CG_experiments_plot import get_ritz_values, plot_ritz_values, get_cg_polynomial, plot_cg_polynomial, \
    plot_eigenvectors,compute_all_ritz_values, \
    plot_eigenvectors_scalar


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
    problem_type = 'conductivity'
    discretization_type = 'finite_element'
    element_type = 'linear_triangles'
    # formulation = 'small_strain'

    domain_size = [1, 1]
    geom_n = 2 # , 4, 5
    ratio=2
    nb_pixels = 16    # np.arange(1,5)  # 17  33

    geometry_ID = 'n_laminate'  # # n_laminate ,'sine_wave_','linear', 'right_cluster_x3', 'left_cluster_x3' square_inclusion

    # system set up
    number_of_pixels = (nb_pixels,nb_pixels)

    my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                      problem_type=problem_type)

    discretization = domain.Discretization(cell=my_cell,
                                           nb_of_pixels_global=number_of_pixels,
                                           discretization_type=discretization_type,
                                           element_type=element_type)

    # set macroscopic gradient
    macro_gradient = np.array([1.0, 1.0])
    # create material data field
    conductivity_C_0 = np.array([[1., 0], [0, 1.0]])
    refmaterial_data_field_ = np.copy(conductivity_C_0)  # [:, :, np.newaxis, np.newaxis, np.newaxis]

    def scale_field(field, min_val, max_val):
        """Scales a 2D random field to be within [min_val, max_val]."""
        field_min, field_max = Reduction(MPI.COMM_WORLD).min(field), Reduction(MPI.COMM_WORLD).max(field)
        scaled_field = (field - field_min) / (field_max - field_min)  # Normalize to [0,1]
        return scaled_field * (max_val - min_val) + min_val  # Scale to [min_val, max_val]

    # if nb_discretization_index == 0:
    #     phase_fied_small_grid = microstructure_library.get_geometry(
    #         nb_voxels=discretization.nb_of_pixels,
    #         microstructure_name=geometry_ID,
    #         coordinates=discretization.fft.coords,
    #         seed=1,
    #         parameter=6)  # ,
    #     #                                                                           contrast=-ratio) # $1 / 10 ** ratio
    #     if ratio != 0:
    #         phase_fied_small_grid += 1 / 10 ** ratio
    #
    #     phase_field_smooth = np.copy(phase_fied_small_grid)
    #
    # if nb_discretization_index > 0:
    #     # phase_field_smooth = sc.ndimage.zoom(phase_fied_small_grid, zoom=nb_pix_multip, order=0)
    #     phase_field_smooth = np.repeat(phase_fied_small_grid, 2 ** (nb_discretization_index), axis=0)
    #     phase_field_smooth = np.repeat(phase_field_smooth, 2 ** (nb_discretization_index), axis=1)
    phase_field_smooth = microstructure_library.get_geometry(
        nb_voxels=discretization.nb_of_pixels,
        microstructure_name=geometry_ID,
        coordinates=discretization.fft.coords,
        seed=1,
        parameter=geom_n)
    phase_field = np.abs(phase_field_smooth)
    phase_field = scale_field(phase_field, min_val=1, max_val=10 ** ratio)

    material_data_field_C_0 = discretization.get_material_data_size_field_mugrid(
        name='conductivity_tensor')
    material_data_field_C_0.s[...] = conductivity_C_0[..., np.newaxis, np.newaxis, np.newaxis] * \
                                     phase_field[np.newaxis, ...]

    # Set up right hand side
    # macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)
    macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
    macro_gradient_field.sg.fill(0)
    discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient,
                                                   macro_gradient_field_ijqxyz=macro_gradient_field)
    discretization.fft.communicate_ghosts(field=macro_gradient_field)
    # np.random.seed(seed=1)

    # Set up right hand side
    rhs_field = discretization.get_unknown_size_field(name='rhs_field')
    rhs_field.sg.fill(0)
    discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0,
                                  macro_gradient_field_ijqxyz=macro_gradient_field,
                                  rhs_inxyz=rhs_field)


    get_igens = True
    if get_igens:
        K = discretization.get_system_matrix_mugrid(material_data_field_C_0)
        # fixing zero eigenvalues
        reduced_K = np.copy(K)
        replacement_value = 10 ** ratio // 2
        K[:, 0] = 0
        K[0, :] = 0
        K[0, 0] =  replacement_value

        eig_K, eig_vect_K = sc.linalg.eig(a=K, b=None)  # , eigvals_only=True
        eig_K = np.real(eig_K)
        eig_K[eig_K == replacement_value] = 0
        # Sort in descending order (largest eigenvalues first)
        idx_K = np.argsort(eig_K)[::-1]  # Get indices of sorted eigenvalues

        # Reorder eigenvalues and eigenvectors
        sorte_eig_K = eig_K[idx_K]
        sorted_eig_vect_K = eig_vect_K[:, idx_K]

        # Greeen precond
        M = discretization.get_system_matrix_mugrid(conductivity_C_0)
        # fixing zero eigenvalues
        M[:, 0] = 0
        M[0, :] = 0
        M[0, 0] = 1
        eig_G, eig_vect_G = sc.linalg.eig(a=K, b=M)  # , eigvals_only=True
        eig_G = np.real(eig_G)
        eig_G_no_zero = np.copy(eig_G)
      #  eig_G[eig_G == 10 ** ratio // 2] = 0
        #eig_G[eig_G == 1.0] = 1

        M_sym = matrix_sqrt_eig(M)
        Green_sqrt = matrix_sqrt_eig(np.linalg.inv(M))
        GKGsym = Green_sqrt @ K @ Green_sqrt
      #  eig_G_sym, eig_vect_G_sym = sc.linalg.eig(a=GKGsym)  # , eigvals_only=True
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

        plot_participation_ratios(displacemets_flat=np.real(sorted_eig_vect_G),
                                  grid_shape=rhs_field.s.shape, eigenvals=sorte_eig_G)
        participation_ratios = get_participation_ration(displacemets_flat=np.real(sorted_eig_vect_G),
                                                        grid_shape=rhs_field.s.shape)
        # plot_eigendisplacement(eigenvectors_1=np.real(sorted_eig_vect_G),
        #                        grid_shape=rhs.shape, dim=2, eigenvals=sorte_eig_G,
        #                        participation_ratios=participation_ratios)

        norms = np.linalg.norm(MiK, axis=0)
        normalized_matrix = MiK / norms
        # THIS PLOTS EIGENVECTORS
        # plot_eigenvectors_scalar(eigenvectors_1=normalized_matrix,
        #                        grid_shape=rhs_field.s.shape, dim=2, eigenvals=eig_G[idx_G],
        #                        weight=eig_G[idx_G], participation_ratios=participation_ratios)

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
    rhs_field.s.flatten()[0] = 0
    # rhs_field.s.flatten()[np.prod(number_of_pixels)] = 0

    ######### INITIAL SOLUTION
    # x0 = np.random.random(discretization.get_displacement_sized_field().shape)
    x0 = discretization.get_unknown_size_field(name='x0')
    x0.s.fill(0)  # = discretization.get_unknown_size_field(name='x0')

    # x0.s[...]= np.random.random( x0.s.shape)
    # x0 = np.zeros(rhs_field.s.shape)
    K_fun = lambda x: K @ x

    M_null = lambda x: 1 * x

    # def M_null(x, Px):
    #     Px.s[...] = 1 * x.s[...]
    displacement_field, norms_origin = solvers.PCG(K_fun, rhs_field.s.flatten(), x0=x0.s.flatten(),
                                                   P=M_null,
                                                   steps=int(1000), toler=1e-14,
                                                   norm_energy_upper_bound=True,
                                                   lambda_min=np.real(sorted(eig_J)[0])
                                                   )

    ########################### Greeen  PRE CONDITIONED VERSION ########################################################
    M_null = lambda x: 1 * x
    K_fun_G = lambda x: GKGsym @ x

    x0.s.fill(0)
    rhs_G = Green_sqrt @ rhs_field.s.flatten()

    r0 = rhs_G.flatten() - K_fun_G(M_sym @ x0.s.flatten())
    r0_norm = np.linalg.norm(r0.flatten())  # order='F

    Green_sqrt_eig_vect_J = M_sym @ eig_vect_G
    normed_eigenvectors = np.zeros_like(Green_sqrt_eig_vect_J)
    for k in np.arange(Green_sqrt_eig_vect_J[:, 0].shape[0]):
        normed_eigenvectors[:, k] = Green_sqrt_eig_vect_J[:, k] / np.linalg.norm(
            Green_sqrt_eig_vect_J[:, k])
    w_i = (np.dot(np.transpose(normed_eigenvectors), r0.flatten() / r0_norm)) ** 2  # order='F'
    w_i_for_un_K = (np.dot(np.transpose(eig_vect_K),
                           rhs_field.s.flatten() / np.linalg.norm(
                               rhs_field.s.flatten()))) ** 2  ### ONLY FOR ZERO RHS

    # plot_eigenvectors(eigenvectors_1=normed_eigenvectors, eigenvectors_2=eig_vect_K,
    #                   grid_shape=rhs.shape, dim=2)

    x_values = np.linspace(0, np.real(sorted(eig_G)[-1] + 1), 100)
    # test Ritz values
    #ritz_values = compute_all_ritz_values(A=GKGsym,  v0=r0.flatten(),m=5 )

    ritz_values = get_ritz_values(A=GKGsym, k_max=5, v0=r0.flatten(),
                                  M_inv=None)


    plot_ritz_values(ritz_values=ritz_values, true_eigenvalues=eig_G)

    # precompute precise solution
    displacement_field, norms = solvers.PCG(K_fun_G, rhs_G.flatten(), x0=M_sym @ x0.s.flatten(),
                                            P=M_null,
                                            steps=int(1000), toler=1e-14,
                                            norm_energy_upper_bound=True,
                                            lambda_min=np.real(sorted(eig_G)[0]))
    setting_CG = {'energy_lower_bound': True,
                  'exact_solution': displacement_field}
    displacement_field, norms = solvers.PCG(K_fun_G, rhs_G.flatten(), x0=M_sym @ x0.s.flatten(),
                                            P=M_null,
                                            steps=int(1000), toler=1e-14,
                                            norm_energy_upper_bound=True,
                                            lambda_min=np.real(sorted(eig_G)[0]),
                                            **setting_CG)

    plot_cg_polynomial(x_values, ritz_values, true_eigenvalues=eig_G, weight=w_i,
                       error_evol=norms['energy_iter_error'] / norms['residual_rr'][0],
                       title='Green PCG', init_res=norms['residual_rr'][0])
    print(norms)
    # quit()
    # # plot_cg_polynomial(x_values, ritz_values, true_eigenvalues=eig_G[idx_G], weight=w_i[idx_G],
    #                    error_evol=norms['energy_lb'] / norms['residual_rr'][
    #                        0], title='Green')  # energy_lb
    # plot_eigendisplacement(eigenvectors_1=np.real(MiK),
    #                        grid_shape=rhs_field.s.shape, dim=2, eigenvals=eig_G[idx_G], weight=w_i[idx_G],
    #                        participation_ratios=participation_ratios)
    #
    # plot_eigendisplacement(eigenvectors_1=np.real(sorted_eig_vect_G),
    #                        grid_shape=rhs_field.s.shape, dim=2, eigenvals=eig_G[idx_G], weight=w_i[idx_G],
    #                        participation_ratios=participation_ratios)

    # quit()
    ############################ UNPRECONDITIONED VERSION ########################################################
    M_null = lambda x: 1 * x
    K_fun_ = lambda x: K @ x
    x0.s.fill(0)
    r0 = rhs_field.s.flatten() - K_fun_(x0.s.flatten())
    r0_norm = np.linalg.norm(r0.flatten())  # order='F'
    w_i = (np.dot(np.transpose(eig_vect_K), r0.flatten() / r0_norm)) ** 2  # order='F'

    x_values = np.linspace(0, np.real(sorted(eig_K)[-1] + 1), 1000)

    ritz_values = get_ritz_values(A=K, k_max=100, v0=r0.flatten(),
                                  M_inv=None)  # r0.flatten(order='F')

    # plot_ritz_values(ritz_values=ritz_values, true_eigenvalues=eig_K)

    displacement_field, norms = solvers.PCG(K_fun_, rhs_field.s.flatten(), x0=x0.s.flatten(), P=M_null,
                                            steps=int(1000), toler=1e-14,
                                            norm_energy_upper_bound=True,
                                            lambda_min=np.real(sorted(eig_K)[0]))
    setting_CG = {'energy_lower_bound': True,
                  'exact_solution': displacement_field}
    displacement_field, norms = solvers.PCG(K_fun_, rhs_field.s.flatten(), x0=x0.s.flatten(), P=M_null,
                                            steps=int(1000), toler=1e-14,
                                            norm_energy_upper_bound=True,
                                            lambda_min=np.real(sorted(eig_K)[0]),
                                            **setting_CG)
    plot_cg_polynomial(x_values, ritz_values, true_eigenvalues=eig_K, weight=w_i,
                       error_evol=norms['energy_iter_error'] / norms['residual_rr'][0],
                       title='Plain CG', init_res=norms['residual_rr'][0])


if __name__ == '__main__':

    for initial in ['zeros']:  # , 'random'
        for RHS in ['random']:  # , 'random'
            kappa = 10

            run_simple_CG_Green(initial=initial, RHS=RHS, kappa=kappa)
