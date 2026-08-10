import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import scipy as sc
import os
from mpi4py import MPI
from NuMPI.Tools import Reduction
from NuMPI.IO import save_npy, load_npy

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO.solvers import PCG
from muFFTTO import microstructure_library

from experiments.a_understanding_Greens_preconditioner.trivial_CG_experiments_plot import get_ritz_values, \
    plot_ritz_values, plot_cg_polynomial_shapr_smooth, \
    plot_cg_polynomial, plot_eigenvectors, \
    plot_eigendisplacement, plot_rhs, plot_eigenvector_filling, plot_cg_polynomial_JG_paper, get_ritz_values_nd_array

from experiments.paper_Jacobi_Green.exp_paper_JG_geometry_plots import get_triangle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Enable LaTeX rendering
plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX
    # "font.family": "helvetica",  # Use a serif font
})
plt.rcParams.update({'font.size': 12})
plt.rcParams["font.family"] = "Arial"
script_name = 'sharp_vs_smooth_eivals_with_eigenvectors'
file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
if not os.path.exists(file_folder_path):
    os.makedirs(file_folder_path)
data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
if not os.path.exists(data_folder_path):
    os.makedirs(data_folder_path)
figure_folder_path = file_folder_path + '/figures/' + script_name + '/'
if not os.path.exists(figure_folder_path):
    os.makedirs(figure_folder_path)

import numpy as np


def get_ritz_values_sequence(A, k_max, v0, Preconditioner=None, rcond=1e-10):
    """
    Ritz values of M^+ A at EVERY Lanczos/PCG iteration.

    Returns
    -------
    ritz_seq : list of 1D arrays; ritz_seq[k] are the (k+1) Ritz values
               after iteration k+1 (sorted ascending).
    """
    A_mv = A if callable(A) else (lambda x: A @ x)

    if Preconditioner is None:
        M_mv = lambda x: x
        nullspace = None
    else:
        w, V = np.linalg.eigh(Preconditioner)
        cutoff = rcond * w.max()
        nonzero = w > cutoff
        w_inv = np.zeros_like(w)
        w_inv[nonzero] = 1.0 / w[nonzero]
        M_mv = lambda x: V @ (w_inv * (V.T @ x))
        nullspace = V[:, ~nonzero]

    r = np.asarray(v0, dtype=float).ravel().copy()
    if nullspace is not None and nullspace.shape[1] > 0:
        r -= nullspace @ (nullspace.T @ r)

    z = M_mv(r)
    rz = r @ z
    p = z.copy()

    alphas, betas = [], []
    diag, off = [], []  # entries of the tridiagonal T
    ritz_seq = []

    for k in range(k_max):
        Ap = A_mv(p)
        pAp = p @ Ap
        if pAp <= 0:
            break
        alpha = rz / pAp

        # --- extend T by one row/column ---
        if k == 0:
            diag.append(1.0 / alpha)
        else:
            diag.append(1.0 / alpha + betas[-1] / alphas[-1])
            off.append(np.sqrt(betas[-1]) / alphas[-1])
        alphas.append(alpha)

        # --- Ritz values after k+1 steps: eigvals of leading block ---
        T_k = np.diag(diag)
        if off:
            T_k += np.diag(off, 1) + np.diag(off, -1)
        ritz_seq.append(np.sort(np.linalg.eigvalsh(T_k)))

        # --- CG update ---
        r = r - alpha * Ap
        if nullspace is not None and nullspace.shape[1] > 0:
            r -= nullspace @ (nullspace.T @ r)
        z = M_mv(r)
        rz_new = r @ z
        if rz_new < 1e-300 * rz:
            break
        beta = rz_new / rz
        betas.append(beta)
        rz = rz_new
        p = z + beta * p

    return ritz_seq


def plot_matrix(matrix):
    # plot participation ratios
    fig = plt.figure(figsize=(5, 5))
    gs = fig.add_gridspec(1, 1, width_ratios=[1])
    ax_matrix = fig.add_subplot(gs[0, 0])
    ax_matrix.matshow(matrix)
    plt.show()


# def matrix_sqrt_eig(A, nb_zero_eigens=2):
#     """
#     Compute the square root of a symmetric positive definite matrix using eigendecomposition.
#
#     Parameters:
#         A (ndarray): Symmetric positive definite matrix.
#
#     Returns:
#         A_sqrt (ndarray): Square root of the matrix.
#     """
#     # Eigendecomposition
#     eigvals, eigvecs = np.linalg.eigh(A)
#
#     # Compute square root of eigenvalues
#     sqrt_eigvals = np.copy(eigvals)
#     sqrt_eigvals[nb_zero_eigens:] = np.sqrt(eigvals[nb_zero_eigens:])
#
#     # Reconstruct A^(1/2)
#     A_sqrt = eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T
#     return A_sqrt
def matrix_sqrt_eig(A, nb_zero_eigens=0, symmetrize=True):
    """
    Symmetric matrix square root via eigendecomposition,
    with explicit handling of a known number of zero eigenvalues.

    A              : (n,n) symmetric PSD ndarray
    nb_zero_eigens : number of (numerically) zero eigenvalues; the
                     nb_zero_eigens smallest eigenvalues are set
                     exactly to zero before taking the square root.
    symmetrize     : enforce symmetry of A before eigh (cheap safeguard
                     against round-off asymmetry, e.g. after pinv).

    Returns A^{1/2} as (n,n) ndarray, exactly symmetric, with the same
    nullspace as A.
    """
    A = np.asarray(A, dtype=float)
    if symmetrize:
        A = 0.5 * (A + A.T)

    w, V = np.linalg.eigh(A)  # ascending eigenvalues

    if nb_zero_eigens > 0:
        w[:nb_zero_eigens] = 0.0  # kill the known zero modes exactly

    if np.any(w < 0):
        if w.min() < -1e-10 * max(w.max(), 1.0):
            raise ValueError(
                f"A has significant negative eigenvalue {w.min():.3e}; "
                "not PSD (or increase nb_zero_eigens)."
            )
        w = np.clip(w, 0.0, None)  # tiny negative round-off -> 0

    sqrt_w = np.sqrt(w)
    A_sqrt = (V * sqrt_w) @ V.T  # V diag(sqrt_w) V^T
    return 0.5 * (A_sqrt + A_sqrt.T)  # exact symmetry


src = './figures/'
compute_ = False

plot_ = True
if compute_:

    problem_type = 'elasticity'
    discretization_type = 'finite_element'
    element_type = 'linear_triangles'
    formulation = 'small_strain'
    # source folder\

    domain_size = [1, 1]
    discretization_n = [4]  # ,4,5
    ratios = np.array([1,])  # 2, 4, 8

    counter = 0
    for T in discretization_n:  # loop over number of discretization points

        nb_pix_multip = T
        # print(f'nb_discretization_index = {nb_discretization_index}')
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
        elastic_C_1 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                         K=K_0,
                                                         mu=G_0,
                                                         kind='linear')

        refmaterial_data_field_ = np.copy(elastic_C_1)  #

        print('elastic tangent = \n {}'.format(domain.compute_Voigt_notation_4order(elastic_C_1)))


        def scale_field(field, min_val, max_val):
            """Scales a 2D random field to be within [min_val, max_val]."""
            field_min, field_max = Reduction(MPI.COMM_WORLD).min(field), Reduction(MPI.COMM_WORLD).max(field)
            scaled_field = (field - field_min) / (field_max - field_min)  # Normalize to [0,1]
            return scaled_field * (max_val - min_val) + min_val  # Scale to [min_val, max_val]


        def scale_field_log(field, min_val, max_val):
            """Scales a 2D random field to be within [min_val, max_val]."""
            field_log = np.log10(field)
            field_min, field_max = np.min(field_log), np.max(
                field_log)

            scaled_field = (field_log - field_min) / (field_max - field_min)  # Normalize to [0,1]
            return 10 ** (scaled_field * (np.log10(max_val) - np.log10(min_val)) + np.log10(
                min_val))  # Scale to [min_val, max_val]


        for i in np.arange(ratios.size):
            ratio = ratios[i]
            size_geom = np.asarray(2 ** T, dtype=int)
            name = f'microstructure_{size_geom}'
            phase_fied_small_grid = np.load('../../' + name + f'.npy', allow_pickle=True)
            phase_field_ = np.abs(phase_fied_small_grid)
            # phase_field_[0, :, :] = 1
            # phase_field_[0, :size_geom // 2, :size_geom // 2] = 0.1
            # phase_field_ = scale_field(phase_field_, min_val=1, max_val=10 ** ratio)
            phase_field = discretization.get_scalar_field(name='phase_field')

            counter = 0
            for sharp in [False, True]:
                _info = {}
                phase_field.s[0, 0, ...] = phase_field_
                #
                # if ratio == 0:
                #     phase_field = scale_field(phase_field, min_val=0, max_val=1.0)
                # else:
                #     phase_field = scale_field(phase_field, min_val=1 / 10 ** ratio, max_val=1.0)

                # phase_field.s[...] = np.copy(phase_field_origin.s)

                if sharp:
                    # phase_field = scale_field(phase_field_origin, min_val=1 / 10 ** ratio, max_val=1.0)
                    phase_field.s[0, 0][phase_field.s[0, 0] < 0.5] = 1 / 10 ** ratio  # phase_field_min#
                    phase_field.s[0, 0][phase_field.s[0, 0] > 0.49] = 1

                print(f'ratio={ratio} ')

                phase_field.s[0, 0] = scale_field_log(np.copy(phase_field.s[0, 0]),
                                                      min_val=1 / (10 ** ratio),
                                                      max_val=1)
                # phase_field.s[0, 0] = scale_field(np.copy(phase_field.s[0, 0]),
                #                                   min_val=1/ (10 ** ratio) ,
                #                                   max_val=1)

                print(f'min ={np.min(phase_field.s)} ')
                print(f'max ={np.max(phase_field.s)} ')

                print(f'min ={np.min(phase_field.s)} ')
                print(f'max ={np.max(phase_field.s)} ')

                material_data_field_C_0 = discretization.get_material_data_size_field_mugrid(
                    name='algortihmic_tangent')
                material_data_field_C_0.s[...] = elastic_C_1[..., np.newaxis, np.newaxis, np.newaxis] * \
                                                 phase_field.s[
                                                     np.newaxis, ...]

                # Set up macro gradient field
                macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
                macro_gradient_field.sg.fill(0)
                discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient,
                                                               macro_gradient_field_ijqxyz=macro_gradient_field)
                discretization.fft.communicate_ghosts(field=macro_gradient_field)

                # Set up right hand side
                rhs_field = discretization.get_unknown_size_field(name='rhs_field')
                rhs_field.sg.fill(0)
                discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0,
                                              macro_gradient_field_ijqxyz=macro_gradient_field,
                                              rhs_inxyz=rhs_field)


                def K_fun(x, Ax):
                    discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0,
                                                              input_field_inxyz=x,
                                                              output_field_inxyz=Ax,
                                                              formulation='small_strain')
                    discretization.fft.communicate_ghosts(Ax)


                preconditioner = discretization.get_preconditioner_Green_mugrid(
                    reference_material_data_ijkl=elastic_C_1)


                def M_fun(x, Px):
                    """
                    Function to compute the product of the Preconditioner matrix with a vector.
                    The Preconditioner is represented by the convolution operator.
                    """
                    discretization.fft.communicate_ghosts(x)
                    discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                                               input_nodal_field_fnxyz=x,
                                                               output_nodal_field_fnxyz=Px)


                print('solvers')
                displacement_field_test = discretization.get_unknown_size_field(name='displacement_field_test')

                solvers.conjugate_gradients_mugrid(
                    comm=discretization.communicator,
                    fc=discretization.field_collection,
                    hessp=K_fun,  # linear operator
                    b=rhs_field,
                    x=displacement_field_test,
                    P=M_fun,
                    tol=1e-5,
                    maxiter=5000,
                )
                stress_field = discretization.get_gradient_size_field(name='stress_field')
                discretization.apply_gradient_operator_symmetrized_mugrid(displacement_field_test, stress_field)
                df_du_field = discretization.get_unknown_size_field(
                    name='adjoint_problem_rhs_in_sensitivity_stress_and_adjoint_FE_NEW')
                discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0,
                                              macro_gradient_field_ijqxyz=stress_field,
                                              rhs_inxyz=df_du_field)
                # minus sign is already there
                df_du_field.s[...] = -2 * df_du_field.s / discretization.cell.domain_volume
                # Normalization
                df_du_field.s[...] = 1 * df_du_field.s  # / np.sum(target_stress_ij ** 2)

                get_igens = True
                if get_igens:
                    adjust_system = True

                    K = discretization.get_system_matrix_mugrid(material_data_field_C_0)
                    # fixing zero eigenvalues
                    # K[:, 0] = 0
                    # K[0, :] = 0
                    # K[:, np.prod(number_of_pixels)] = 0
                    # K[np.prod(number_of_pixels), :] = 0
                    # K[0, 0] = 2 / 10 ** ratio  # 1/10 ** ratio // 2  # 50.5 #
                    # K[np.prod(number_of_pixels), np.prod(
                    #     number_of_pixels)] = 2 / 10 ** ratio  # 10 ** ratio // 2  # 50.5

                    eig_K, eig_vect_K = sc.linalg.eig(a=K, b=None)  # , eigvals_only=True
                    eig_K = np.real(eig_K)
                    # eig_K[eig_K == 50.5] = 0
                    # Sort in descending order (largest eigenvalues first)
                    idx_K = np.argsort(eig_K)[::-1]  # Get indices of sorted eigenvalues
                    # Reorder eigenvalues and eigenvectors
                    sorte_eig_K = eig_K[idx_K]
                    sorted_eig_vect_K = eig_vect_K[:, idx_K]

                    # Greeen precond
                    M = discretization.get_system_matrix_mugrid(refmaterial_data_field_)

                    # fixing zero eigenvalues
                    # M[:, 0] = 0
                    # M[0, :] = 0
                    # M[:, np.prod(number_of_pixels)] = 0
                    # M[np.prod(number_of_pixels), :] = 0
                    # M[0, 0] = 1
                    # M[np.prod(number_of_pixels), np.prod(number_of_pixels)] = 1
                    ##### Left preconditioned
                    # MiK = np.linalg.pinv(M) @ K

                    ####
                    eig_G, eig_vect_G = sc.linalg.eig(a=K, b=M)  # , eigvals_only=True
                    eig_G_no_zero = np.copy(eig_G)

                    eig_G = np.real(eig_G)
                    #       eig_G[eig_G == 10 ** ratio // 2] = 0
                    # Sort in descending order (largest eigenvalues first)
                    idx_G = np.argsort(eig_G)[::-1]  # Get indices of sorted eigenvalues

                    # Reorder eigenvalues and eigenvectors
                    sorte_eig_G = eig_G[idx_G]
                    sorted_eig_vect_G = eig_vect_G[:, idx_G]

                    ### symmetrized precondioner
                    M_sym = matrix_sqrt_eig(A=M, nb_zero_eigens=2)
                    Green_sqrt = matrix_sqrt_eig(A=np.linalg.pinv(M), nb_zero_eigens=2)
                    GKGsym = Green_sqrt @ K @ Green_sqrt

                    # --- 1. Build rigid body modes (2D: translations in x and y) ---
                    ndof = K.shape[0]
                    N = np.prod(number_of_pixels)

                    r1 = np.zeros(ndof)
                    r2 = np.zeros(ndof)
                    r1[:N] = 1.0  # x-translation
                    r2[N:] = 1.0  # y-translation

                    R = np.vstack([r1, r2]).T  # shape (ndof, 2)
                    #
                    # # --- 2. Orthonormalize rigid modes ---
                    # Q, _ = np.linalg.qr(R)  # Q: (ndof, 2)
                    #
                    # # --- 3. Projector onto orthogonal complement ---
                    # P = np.eye(ndof) - Q @ Q.T
                    #
                    # # --- 4. Project K and M ---
                    # Kp = P @ K @ P
                    # Mp = P @ M @ P

                    # --- 5. Solve reduced eigenproblem ---
                    # eigvals, eigvecs = sc.linalg.eig(Kp, Mp)

                    # Z: orthonormal basis of the complement of the rigid modes, shape (ndof, ndof-2)
                    Q_full, _ = np.linalg.qr(np.hstack([R, np.random.randn(ndof, ndof - 2)]))
                    Z = Q_full[:, 2:]

                    Kr = Z.T @ K @ Z  # (ndof-2, ndof-2), SPD
                    Mr = Z.T @ M @ Z  # (ndof-2, ndof-2), SPD
                    eigvals, eigvecs_r = sc.linalg.eigh(Kr, Mr)  # all finite, real, M-orthonormal
                    eigvecs = Z @ eigvecs_r  # back to full space

                    # --- 6. Eigenvectors already orthogonal to rigid modes ---
                    # (Optional) Normalize or sort:
                    idx = np.argsort(np.real(eigvals))
                    eigvals_projected = eigvals[idx]
                    eigvecs_projected = eigvecs[:, idx]

                    print("Physical eigenvalues:")
                #  print(np.real(eigvals))
                # rhs_field.s.flatten()[0] = 0
                # rhs_field.s.flatten()[np.prod(number_of_pixels)] = 0
                for random_init in [True, False]:
                    ######### INITIAL SOLUTION
                    if random_init:
                        # x0 = np.random.random(rhs_field.s.shape)
                       # x0 = np.copy(df_du_field.s[...])
                        x0 = np.zeros(rhs_field.s.shape)
                        rhs_field.s[...] = df_du_field.s[...]
                        # x0[0] -= x0[0].mean()
                    # x0[1] -= x0[1].mean()
                    else:
                        x0 = np.zeros(rhs_field.s.shape)

                    ########################### Greeen  PRE CONDITIONED VERSION ########################################################
                    M_null = lambda x: 1 * x
                    K_fun_G = lambda x: GKGsym @ x

                    rhs_G = Green_sqrt @ rhs_field.s.flatten()

                    r0 = rhs_G.flatten() - K_fun_G(M_sym @ x0.flatten())
                    r0_norm = np.linalg.norm(r0.flatten())  # order='F

                    Green_sqrt_eig_vect_J = M_sym @ eig_vect_G
                    normed_eigenvectors = np.zeros_like(Green_sqrt_eig_vect_J)
                    for k in np.arange(Green_sqrt_eig_vect_J[:, 0].shape[0]):
                        normed_eigenvectors[:, k] = Green_sqrt_eig_vect_J[:, k] / np.linalg.norm(
                            Green_sqrt_eig_vect_J[:, k])
                    w_i = (np.dot(np.transpose(normed_eigenvectors), r0.flatten() / r0_norm)) ** 2  # order='F'

                    # reduced rhs / initial residual
                    rhs_r = Z.T @ rhs_field.s.flatten()
                    y0 = Z.T @ x0.flatten()  # reduced initial guess
                    r0_r = rhs_r - Kr @ y0  # = rhs_r if x0 = 0

                    # generalized eigenpairs of the reduced pencil (Mr-orthonormal vectors)
                    eigvals_r, eigvecs_r = sc.linalg.eigh(Kr, Mr)

                    # w_i decomposition
                    denom = r0_r @ np.linalg.solve(Mr, r0_r)  # ||M^{-1/2} r0||^2
                    w_i_r = (eigvecs_r.T @ r0_r) ** 2 / denom

                    print("sum(w_i) =", w_i.sum())  # must be ~1.0
                    print("sum(w_i_r) =", w_i_r.sum())  # must be ~1.0

                    # w_i = np.nan_to_num(w_i, nan=0.0)
                    # normed_eigenvectors = np.nan_to_num(w_i, nan=0.0)
                    # eig_G_no_zero = np.nan_to_num(eig_G_no_zero, nan=0.0)
                    # w_i_for_un_K = (np.dot(np.transpose(eig_vect_K),
                    #                        rhs_field.s.flatten() / np.linalg.norm(
                    #                            rhs_field.s.flatten()))) ** 2  ### ONLY FOR ZERO RHS

                    # plot_eigenvectors(eigenvectors_1=normed_eigenvectors, eigenvectors_2=eig_vect_K,
                    #                   grid_shape=rhs.shape, dim=2)

                    x_values = np.linspace(1 / 10 ** ratio, 1, 100)
                    # test Ritz values
                    # ritz_values = get_ritz_values(A=GKGsym, k_max=32, v0=r0.flatten(),   M_inv=None)
                    rhs_r = Z.T @ rhs_field.s.flatten()  # (ndof-2,) restriction to the complement
                    ritz_values = get_ritz_values_sequence(Kr, k_max=15, v0=r0_r, Preconditioner=Mr, rcond=1e-13)

                    #
                    # ritz_values_nd = get_ritz_values_nd_array(A=GKGsym, k_max=32, v0=r0.flatten(),
                    #                                           M_inv=None)
                    # plot_ritz_values(ritz_values=ritz_values, true_eigenvalues=eig_G)
                    displacement_field, norms = solvers.PCG(K_fun_G, rhs_G.flatten(), x0=M_sym @ x0.flatten(),
                                                            P=M_null,
                                                            steps=int(1000), toler=1e-14,
                                                            norm_energy_upper_bound=True,
                                                            lambda_min=np.real(sorted(eig_G)[0]))

                    setting_CG = {'energy_lower_bound': True,
                                  'exact_solution': displacement_field}
                    displacement_field, norms = solvers.PCG(K_fun_G, rhs_G.flatten(), x0=M_sym @ x0.flatten(), P=M_null,
                                                            steps=int(1000), toler=1e-14,
                                                            norm_energy_upper_bound=True,
                                                            lambda_min=np.real(sorted(eig_G)[-2]),
                                                            rtol=True,
                                                            **setting_CG
                                                            )
                    if sharp:
                        name_of_plot_ = f'nb_{number_of_pixels[0]}_sharp_{sharp}_kappa{ratio}_random_init_{random_init}'
                    else:
                        name_of_plot_ = f'nb_{number_of_pixels[0]}_sharp_{sharp}_kappa{ratio}_random_init_{random_init}'

                    #  eig_G = np.nan_to_num(eig_G, nan=0.0, posinf=np.sort(eig_G)[0], neginf=0.0)

                    plot_cg_polynomial_shapr_smooth(x_values, ritz_values, true_eigenvalues=eigvals_r, weight=w_i_r,
                                                    error_evol=norms['energy_iter_error'] / norms['residual_rr'][0],
                                                    title=name_of_plot_, init_res=norms['residual_rr'][0])

                    # normalized rigid modes (these ARE the eigenvectors for lambda = 0)
                    q1 = r1 / np.linalg.norm(r1)
                    q2 = r2 / np.linalg.norm(r2)

                    eigvecs_reconstructed = np.column_stack([q1, q2, Z @ eigvecs_r])  # (512, 512)

                    eigvals_reconstructured = np.concatenate(([0.0, 0.0], eigvals_r))
                    w_i_reconstructured = np.concatenate(([0.0, 0.0], w_i_r))

                    eigens = M_sym @ eigvecs_reconstructed
                    _info = {}

                    _info['nb_of_pixels'] = discretization.nb_of_pixels_global
                    _info['nb_of_sampling_points'] = np.shape(phase_fied_small_grid)
                    # phase_field_sol_FE_MPI = xopt.x.reshape([1, 1, *discretization.nb_of_pixels])
                    #   _info['norm_rMr_G'] = norms['data_scaled_rr']
                    _info['norm_rr_G'] = norms['residual_rr']
                    _info['energy_iter_error'] = norms['energy_iter_error']
                    _info['norm_UB_G'] = norms['energy_upper_bound']
                    _info['eigens_G'] = eig_G[idx_G]

                    _info['eig_vect_G'] = eig_vect_G[:, idx_G]
                    _info['rhs'] = rhs_field.s  # rhs_r#
                    _info['weights'] = w_i_reconstructured  # w_i[idx_G]
                    _info['r0'] = rhs_G

                    # _info['x_values'] = x_values
                    # _info['ritz_values'] = ritz_values_nd
                    _info['eig_G_no_zero'] = eig_G_no_zero[idx_G]

                    _info['phase_field'] = phase_field.s[0, 0]
                    _info['eigvals_projected'] = eigvals_reconstructured  # eigvals[idx]
                    _info['eigvecs_projected'] = eigvecs_reconstructed  # eigvecs[:, idx]

                    results_name = f'T{discretization.nb_of_pixels_global[0]}_sharp_{sharp}_kappa{ratio}_random_init_{random_init}'

                    np.savez(data_folder_path + results_name + f'_info.npz', **_info)
                    # np.savez(data_folder_path + results_name + f'_info.npz',
                    #          **{key: np.array(value, dtype=object) for key, value in _info.items()})

                    print(data_folder_path + results_name + f'_log.npz')

if plot_:
    counter = 0

    for sharp in [False, True]:  # , False]
        for random_init in [True, False]:

            number_of_pixels = 32
            plt.figure(figsize=(4.5, 5.))

            # plt.title(f'Sharpness: {sharp}, Random init: {random_init}, Number of pixels: {number_of_pixels} ')
            if sharp:
                name_of_plot_ = r'$\rho_{\mathrm{sharp}}$'
            else:
                name_of_plot_ = r'$\rho_{\mathrm{smooth}}$'
            plt.title(name_of_plot_)
            for i, ratio in enumerate([2, 4, 8]):  # np.array([2])
                results_name = f'T{number_of_pixels}_sharp_{sharp}_kappa{ratio}_random_init_{random_init}'

                _info = np.load(data_folder_path + results_name + f'_info.npz', allow_pickle=True)
                phase_field = _info['phase_field']

                # expanded_data = {key: _info[key] for key in _info.files}

                nb_of_pixels_global = _info['nb_of_pixels']
                phase_fied = _info['nb_of_sampling_points']
                norms = {}
                #            norms['data_scaled_rr'] = _info['norm_rMr_G']
                norms['energy_upper_bound'] = _info['norm_UB_G']
                norms['residual_rr'] = _info['norm_rr_G']
                norms['energy_iter_error'] = _info['energy_iter_error']

                # add ploting of eigenvalues
                eigenvals=_info['eigvals_projected']
                w_i_reconstructured=_info['weights']
                mask = w_i_reconstructured > 1e-10 # mask for eigenvalues with zero weiths
               # eigenvals[mask]
                k = np.arange(1e3)
                 # w_i[idx_G]

                linestyles = ['-', '--', ':', '-.', '--', ':', '-.']
                colors = ['red', 'blue', 'green', 'orange', 'purple', 'orange', 'purple']

                kappa = np.real(max(eigenvals[mask]) / min(eigenvals[mask]))
                r0_for_kappa_bound=norms['energy_iter_error'][0] / norms['residual_rr'][0]
                convergence = (r0_for_kappa_bound) * (2 * (((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** k)) ** 2

                kappa_energy=  norms['energy_iter_error'][0]  * (2 * (((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** k)) ** 2

                # plt.loglog(np.arange(len(norms['residual_rr'])) + 1, norms['residual_rr'] / norms['residual_rr'][0],
                #            '--', label=f'{ratio}')
                plt.loglog(np.arange(len(convergence)) + 1,
                           convergence,
                           label=r"$\kappa-$ bound " + fr'$10^{{{ratio}}}$', color=colors[i])
                plt.loglog(np.arange(len(norms['energy_iter_error'])) + 1,
                           norms['energy_iter_error'] / norms['energy_iter_error'][0],
                           label=r"$\kappa^{\mathrm{tot}}=$" + fr'$10^{{{ratio}}}$', color=colors[i], linestyle='--')
                plt.legend()
                plt.xscale('linear')
                plt.ylabel(rf'$\|u - u_{{k}}\|^{2}_{{K}} / \|u - u_{{0}}\|^{2}_{{K}} $')
                plt.xlabel(r" CG iteration $k$")
                plt.xlim([1, 1e3])
                plt.ylim([1e-10, 1e2])
            fname = src + 'errors_' + results_name + '_{}{}'.format(sharp, '.pdf')
            print(('create figure: {}'.format(fname)))
            plt.savefig(fname, bbox_inches='tight')
            plt.show()
            fig = plt.figure(figsize=(4.5, 5.0))
            gs = fig.add_gridspec(3, 1, hspace=0.3, wspace=0.1)

            for i, ratio in enumerate([2, 4, 8]):  # np.array([2])
                results_name = f'T{number_of_pixels}_sharp_{sharp}_kappa{ratio}_random_init_{random_init}'

                _info = np.load(data_folder_path + results_name + f'_info.npz', allow_pickle=True)
                phase_field = _info['phase_field']
                weight = _info['weights']

                true_eigenvalues = _info['eigvals_projected']
                ax_weights = fig.add_subplot(gs[i, :])
                ax_weights.scatter(np.real(true_eigenvalues), np.real(weight) / np.real(true_eigenvalues), color='blue',
                                   marker='o', s=1, label=r"non-zero weights - $w_{i}/ \lambda_{i}$")
                if sharp:
                    ax_weights.annotate(r"$\kappa^{\mathrm{tot}}=$" + fr'$10^{{{ratio}}}$', xy=(0.8, 0.1),
                                        xycoords='axes fraction', )
                else:
                    ax_weights.annotate(r"$\kappa^{\mathrm{tot}}=$" + fr'$10^{{{ratio}}}$', xy=(0.8, 0.85),
                                        xycoords='axes fraction', )

                ax_weights.set_yscale('log')
                ax_weights.set_ylim(1e-10, 1)

                set_log_scale = True
                if set_log_scale:
                    ax_weights.set_xscale('log')
                    #    ax_weights.set_xlim(1/10**ratio, 1)
                    ax_weights.set_xlim(1 / 10 ** 8, 1)

                else:
                    ax_weights.set_xlim(0, 1)

                # ax_weights.set_xlim(0, 1)
                # ax_weights.set_xscale('log')
                # ax_weights.set_xlim(1/10**ratio, 1)  # 1e-210

                # ax_weights.set_ylabel(r"$w_{i}/ \lambda_{i}$")
                # ax_weights.set_title(f"Weights / Eigens ")
                # ax_weights.set_xlabel('eigenvalue index - $i$ (sorted)')
                if sharp:
                    name_of_plot_ = r'$\rho_{\mathrm{sharp}}$'
                else:
                    name_of_plot_ = r'$\rho_{\mathrm{smooth}}$'

                if i == 0:
                    ax_weights.set_title(
                        name_of_plot_)  # f"Weights - "+ # at Iteration {i}#                     ax_weights.annotate(r"$\kappa^{\mathrm{tot}}=$"+fr'$10^{{{ratio}}}$', xy=(0.8, 0.8), xycoords='axes fraction',)
                # ax_weights.set_xticks([])
                # if i == 1:
                # ax_weights.set_xticks([])
                if i == 2:
                    ax_weights.set_xlabel('eigenvalue $\lambda_{i}$')

                # ax_weights.set_ylabel(r'Weights - $w_{i}/ \lambda_{i}$')
                # ax_weights.set_xticks([1, 34, 67, 100])
                # ax_weights.set_xticklabels([1, 34, 67, 100])
                ax_weights.legend(ncol=1, loc='lower left')
            fname = src + 'wieghts_' + results_name + f'{set_log_scale}' + '_{}_{}{}'.format(sharp, i, '.pdf')
            print(('create figure: {}'.format(fname)))
            plt.savefig(fname, bbox_inches='tight')
            plt.show()

if plot_:
    counter = 0
    for sharp in [False, True]:  # , False]
        random_init = True
        number_of_pixels = 32
        ratio = 4  # np.array([2])
        results_name = f'T{number_of_pixels}_sharp_{sharp}_kappa{ratio}_random_init_{random_init}'

        _info = np.load(data_folder_path + results_name + f'_info.npz', allow_pickle=True)
        phase_field = _info['phase_field']

        # expanded_data = {key: _info[key] for key in _info.files}

        nb_of_pixels_global = _info['nb_of_pixels']
        phase_fied = _info['nb_of_sampling_points']
        norms = {}
        #            norms['data_scaled_rr'] = _info['norm_rMr_G']
        norms['energy_upper_bound'] = _info['norm_UB_G']
        norms['residual_rr'] = _info['norm_rr_G']
        plt.figure(figsize=(11, 5.5))
        plt.title(
            f'Sharpness: {sharp}, Random init: {random_init}, Number of pixels: {number_of_pixels}, Ratio: {ratio}')
        plt.loglog(np.arange(len(norms['residual_rr'])) + 1, norms['residual_rr'] / norms['residual_rr'][0],
                   label='Residual')
        plt.xlim([1, 1e3])
        plt.ylim([1e-10, 1e0])

        plt.show()
        # eig_G = _info['eigens_G']

        # eig_vect_G = _info['eig_vect_G']
        rhs = _info['rhs']
        weight = _info['weights']
        r0 = _info['r0']

        eigvals_projected = _info['eigvals_projected']
        # eigvals_projected = np.concatenate(([0.0, 0.0], eigvals_projected))

        eigvecs_projected = _info['eigvecs_projected']
        #        x_values = _info['x_values']
        #  ritz_values = _info['ritz_values']
        eig_G_no_zero = np.real(_info['eig_G_no_zero'])

        # vector_to_plot=np.abs(MiK)#.transpose()
        vector_to_plot = eigvecs_projected  # [:, idx_G]
        if random_init:
            rhs_to_plot = r0  # rhs_field.s[...]  # +1
        else:
            rhs_to_plot = rhs

        weights_to_plot = weight
        eigens_to_plots = eigvals_projected  # eig_G
        eigens_to_plots = np.nan_to_num(eigens_to_plots, nan=0.0)
        # Plot the convergence of Ritz values
        grid_shape = rhs.shape  # .s.shape
        x = np.arange(0, number_of_pixels)
        y = np.arange(0, number_of_pixels)
        x, y = np.meshgrid(x, y)

        size_of_vector = eigens_to_plots[::-1].size + 1
        # # fig = plt.figure(figsize=(11, 5.5))
        triangles, X, Y = get_triangle(nx=number_of_pixels, ny=number_of_pixels
                                       , lx=number_of_pixels, ly=number_of_pixels)
        # Create the triangulation object
        triangulation = tri.Triangulation(X, Y, triangles)
        x_ = np.arange(0 + 0.5, 1 * number_of_pixels + 0.5)
        y_ = np.arange(0 + 0.5, 1 * number_of_pixels + 0.5)
        X_, Y_ = np.meshgrid(x_, y_)

        # or i in range(size_of_vector):
        plot_eigenvector_separately = False
        if plot_eigenvector_separately:
            for i in np.arange(size_of_vector - 2, 0, -100):
                fig, ax1 = plt.subplots(figsize=(4.5, 4.5))
                ax1.set_title(f'Projected Eigenvalue {i} - $\lambda_{i}$')
                eigvec_x_proj = np.real(eigvecs_projected[:, ::-1])[:, i].reshape(grid_shape)[
                    0, 0].transpose()
                eigvec_y_proj = np.real(eigvecs_projected[:, ::-1])[:, i].reshape(grid_shape)[
                    1, 0].transpose()
                ax1.quiver(x, y, eigvec_x_proj, eigvec_y_proj, color='red', angles='xy', scale_units='xy', scale=0.01)

                plt.show()

                eigenvector_x = np.real(vector_to_plot[:, ::-1])[:, i].reshape(grid_shape)[
                    0, 0].transpose()

                # - np.real(vector_to_plot[:, ::-1])[:, i].reshape(grid_shape)[0, 0].transpose()[0, 12]
                # add correction of non-zero mean

                eigenvector_y = np.real(vector_to_plot[:, ::-1])[:, i].reshape(grid_shape)[
                    1, 0].transpose()

                fig, ax1 = plt.subplots(figsize=(4.5, 4.5))
                ax1.set_title(f'Eigenvalue {i} - $\lambda_{i}$')

                # - np.real(vector_to_plot[:, ::-1])[:, i].reshape(grid_shape)[1, 0].transpose()[0, 12]
                amplitude = np.sqrt(eigenvector_x ** 2 + eigenvector_y ** 2)

                divnorm = mpl.colors.Normalize(vmin=1 / 10 ** ratio, vmax=1)
                # Define facecolors: Use 'none' for empty elements (zeros) and color for others
                facecolors = ['none' if value == 0 else 'red' for value in amplitude.flatten()]
                # Plot circles with empty ones for zero values
                # plt.scatter(x_coords_flat, y_coords_flat, s=A_flat * 100, facecolors=facecolors, edgecolors='blue', alpha=0.7)
                sizes = np.copy(amplitude)  # .flatten()
                sizes[sizes > 1e-13] = 1

                circles_sizes = 20 * np.ones_like(amplitude)
                circles_sizes[-1, :] = 0
                circles_sizes[:, -1] = 0

                ax1.triplot(triangulation, 'k-', lw=0.1)

                ax1.pcolormesh(X_, Y_, np.transpose(phase_field),
                               cmap=mpl.cm.Greys, vmin=1 / 10 ** ratio, vmax=1, linewidth=0,
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
                fname = src + 'eigenvecs_{}_{}{}'.format(sharp, i, '.png')
                print(('create figure: {}'.format(fname)))
                plt.savefig(fname, bbox_inches='tight')

                fig, ax1 = plt.subplots(figsize=(4.5, 4.5))
                ax1.set_title(f'Eigenvalue {i} - $\lambda_{i}$')
                ax1.quiver(x, y, eigenvector_x, eigenvector_y, color='red', angles='xy', scale_units='xy', scale=0.01)

                # eigenvector_x, eigenvector_y: shape (Ny, Nx)

                dx = 1.0  # or your actual spacing in x
                dy = 1.0  # or your actual spacing in y

                dv_dx = np.gradient(eigenvector_y, dx, axis=1)  # ∂v/∂x
                dv_dy = np.gradient(eigenvector_y, dy, axis=0)
                du_dx = np.gradient(eigenvector_x, dx, axis=1)  # ∂u/∂y
                du_dy = np.gradient(eigenvector_x, dy, axis=0)  # ∂u/∂y

                curl = dv_dx - du_dy  # shape (Ny, Nx)
                fig, ax = plt.subplots(figsize=(4.5, 4.5))
                norm = mpl.colors.TwoSlopeNorm(vcenter=0)
                im = ax.imshow(curl, origin='lower', cmap='seismic', norm=norm)
                fig.colorbar(im, ax=ax)
                ax.set_title(f'Curl of eigenvector {i}')

                fig, axes = plt.subplots(2, 2, figsize=(9, 4.5))

                norm = mpl.colors.TwoSlopeNorm(vcenter=0)

                # --- Plot 1: dv/dx ---
                im1 = axes[0, 0].imshow(dv_dx, origin='lower', cmap='seismic', norm=norm)
                axes[0, 0].set_title(f'dv/dx for eigenvector {i}')
                fig.colorbar(im1, ax=axes[0, 0])
                # --- Plot 2: du/dy ---
                im2 = axes[0, 1].imshow(dv_dy, origin='lower', cmap='seismic', norm=norm)
                axes[0, 1].set_title(f'dv/dy for eigenvector {i}')
                fig.colorbar(im2, ax=axes[0, 1])
                plt.tight_layout()

                # --- Plot 1: du/dx ---
                im1 = axes[1, 0].imshow(du_dx, origin='lower', cmap='seismic', norm=norm)
                axes[1, 0].set_title(f'du/dx for eigenvector {i}')
                fig.colorbar(im1, ax=axes[1, 0])
                # --- Plot 2: du/dy ---
                im2 = axes[1, 1].imshow(du_dy, origin='lower', cmap='seismic', norm=norm)
                axes[1, 1].set_title(f'du/dy for eigenvector {i}')
                fig.colorbar(im2, ax=axes[1, 1])

                plt.show()

        fig = plt.figure(figsize=(4.5, 4.5))

        gs = fig.add_gridspec(1, 1)
        ax_global = fig.add_subplot(gs[:, :])

        ax_global.plot(np.arange(1, size_of_vector), eigens_to_plots, color='Green', label=f'Green',
                       alpha=0.5, marker='.', linewidth=0, markersize=5)

        opacities = np.abs(np.real(weights_to_plot))
        opacities[0:2] = 0
        opacities[opacities > 1e-12] = 1
        opacities[opacities < 1] = 0
        ax_global.scatter(np.arange(1, size_of_vector), eigens_to_plots, color='Blue', label=f'Green',
                          alpha=opacities)
        # plt.gca().invert_xaxis()  # Invert X-axis
        ax_global.plot(1, color='Red',
                       alpha=1.0, marker='.', linewidth=0, markersize=4)
        # weights_to_plot
        ax_global.set_xlim([1, size_of_vector])
        ax_global.set_xticks([1, size_of_vector // 2, size_of_vector])
        ax_global.set_xticklabels([1, size_of_vector // 2, size_of_vector])
        ax_global.set_ylim([1 / 10 ** (ratio + 0.1), 1 + 0.01])
        # ax_global.set_yticks([1, 34, 67, 100])
        #  ax_global.set_yticklabels([1, 34, 67, 100])
        # ax_global.set_yscale('log')

        #
        if sharp:
            name_of_plot_ = r'$\rho_{\mathrm{sharp}}$'
        else:
            name_of_plot_ = r'$\rho_{\mathrm{smooth}}$'

        ax_global.set_title(f'\n Sparsity patterns$^*$ - ' + name_of_plot_)  # in eigenvectors
        ax_global.set_xlabel('eigenvalue index - $i$ (sorted)')
        ax_global.set_ylabel(r'Eigenvalue - $\lambda_{i}$')
        # ax_global.text(0.02, 0.65,
        #                r'Geometry - $\mathcal{G}_' + f'sharp_{sharp}$\n' + r'Mesh - $\mathcal{T}_{16}$',
        #                transform=ax_global.transAxes, fontsize=13)

        # Create a custom legend symbol
        custom_symbol = [
            mpl.lines.Line2D([], [], color='Green', marker='.', linestyle='None', markersize=5,
                             label='Eigenvalues'),
            mpl.lines.Line2D([], [], color='Blue', marker='.', linestyle='None',
                             markersize=5, label='Non-zero weights'),
            mpl.lines.Line2D([], [], color='Red', marker='.', linestyle='None',
                             markersize=5, label='Non-zero element of vector')]
        # # Add legend with the custom symbol
        plt.legend(handles=custom_symbol, loc='upper left')
        # ax_global.legend([f'Eigenvalues ', f'Non-zero weights', f'Non-zero element of eigenvector/ first residual '],
        #                  loc='upper left')
        #

        plot_eigenvectors = True
        if plot_eigenvectors:
            x_offset = -0.5
            y_offset = 1.1
            for upper_ax in np.arange(4):
                # weight = np.array([0.2, 1, 10, 30, 100])[upper_ax]
                if upper_ax == 0:
                    # ax1 = fig.add_subplot(gs[0, upper_ax])
                    ax1 = fig.add_axes([0.33, 0.13, 0.2, 0.20])
                    ax1.set_title(r'$\phi_{i}\, (\lambda_{i}=$' + fr'$10^{{-{ratio}}}) $ ')
                    roll_x = 5
                    roll_y = -20
                    ax_global.annotate('',
                                       xy=(500, 0.1),
                                       xytext=(2, 0.01),
                                       arrowprops=dict(arrowstyle='->',
                                                       color='black',
                                                       lw=1,
                                                       ls='-')
                                       )
                    # ax_global.text(x_offset + 0.4, y_offset + 0.3, '(a.1)', transform=ax1.transAxes)  #
                    i = 3
                    eigenvector_x = np.real(vector_to_plot)[:, i].reshape(grid_shape)[0, 0] - \
                                    np.real(vector_to_plot)[:, i].reshape(grid_shape)[0, 0][0, 12]
                    # add correction of non-zero mean

                    eigenvector_y = np.real(vector_to_plot)[:, i].reshape(grid_shape)[1, 0] - \
                                    np.real(vector_to_plot)[:, i].reshape(grid_shape)[1, 0][0, 12]
                if upper_ax == 1:
                    ax1 = fig.add_axes([0.43, 0.40, 0.2, 0.20])
                    ax1.set_title(r'$\phi_{i}\, (\lambda_{i}=0.6) $ ')

                    # ax_global.text(x_offset, y_offset, '(a.2)', transform=ax1.transAxes)
                    i = 250
                    idx = np.argmin(np.abs(eigens_to_plots - 0.6))
                    value = eigens_to_plots[::-1][idx]  # the actual closest value, to verify
                    ax_global.annotate('',
                                       xy=(850, 0.5),
                                       xytext=(idx, 0.6),
                                       arrowprops=dict(arrowstyle='->',
                                                       color='black',
                                                       lw=1,
                                                       ls='-')
                                       )

                    # I have to remove linear displacement from the eigenvalues
                    # first do the eigenvectors in x displacement
                    #
                    # grad_x_x = np.real(vector_to_plot[:, ::-1])[:, i].reshape(grid_shape)[
                    #                0, 0, 0, 0] - np.real(vector_to_plot[:, ::-1])[:, i].reshape(grid_shape)[
                    #                0, 0, 0, 1]
                    # lin_disp = grad_x_x * np.arange(number_of_pixels)
                    # lin_disps = np.tile(lin_disp, (number_of_pixels, 1))
                    # mean_part = np.real(vector_to_plot[:, ::-1])[:, i].reshape(grid_shape)[
                    #     0, 0, 0, 0]

                    eigenvector_x = np.real(vector_to_plot)[:, i].reshape(grid_shape)[0, 0]

                    dv_dx = np.gradient(eigenvector_x, 1., axis=1)
                    mask = (np.abs(dv_dx) > 1e-12).astype(int)

                    # y DIRECTION

                    eigenvector_y = np.real(vector_to_plot)[:, i].reshape(grid_shape)[1, 0]

                    # masking parts with zero gradient
                    eigenvector_x[...] = eigenvector_x * mask
                    eigenvector_y[...] = eigenvector_y * mask
                    # -
                # if upper_ax == 2:
                #     ax1 = fig.add_axes([0.58, 0.62, 0.1, 0.20])
                #     ax1.set_title(r'$\phi_{i}\, (\lambda_{i}=67) $ ')
                #     ax_global.text(x_offset, y_offset, '(a.3)', transform=ax1.transAxes)
                #
                #     i = 2000
                #
                #     eigenvector_x = np.real(vector_to_plot[:, ::-1])[:, i].reshape(grid_shape)[0, 0]
                #
                #     dv_dx = np.gradient(eigenvector_x, 1., axis=1)
                #     dv_dxx = np.gradient(dv_dx, 1., axis=1)
                #
                #
                #     mask = (np.abs(dv_dxx) > 1e-12).astype(int)
                #
                #     # y DIRECTION
                #
                #     eigenvector_y = np.real(vector_to_plot[:, ::-1])[:, i].reshape(grid_shape)[1, 0]
                #
                #     # masking parts with zero gradient
                #     eigenvector_x[...] = eigenvector_x * mask
                #     eigenvector_y[...] = eigenvector_y * mask

                # if upper_ax == 3:
                #     ax1 = fig.add_axes([0.776, 0.64, 0.1, 0.20])
                #     # ax1.set_title(r'$\phi_{i}\, (\lambda_{i}=100) $ ')
                #
                #     ax_global.text(x_offset + 0.01, y_offset - 1.4,
                #                    r'(a.4) $\phi_{i}\, (\lambda_{i}=100) $',
                #                    transform=ax1.transAxes, fontsize=13)
                #     i = 500
                if upper_ax == 2:
                    i = 2000
                    # ax1 = fig.add_axes([0.17, 0.6, 0.1, 0.20])

                    ax1 = fig.add_axes([0.68, 0.45, 0.2, 0.20])
                    # ax_global.text(x_offset + 0.25, y_offset + 0., '(b.1)', transform=ax1.transAxes)
                    ax1.set_title(r'$\phi_{i}\, (\lambda_{i}=1) $ ')
                    # ax_global.text(x_offset, y_offset, '(a.3)', transform=ax1.transAxes)

                    ax_global.annotate('',
                                       xy=(i + 23, sorted(eigvals_projected)[i]),
                                       xytext=(2000, 0.76),
                                       arrowprops=dict(arrowstyle='<-',
                                                       color='black',
                                                       lw=0.5,
                                                       ls='-')
                                       )
                    # Add an ellipse
                    # ellipse = Ellipse(xy=(i, sorted(eig_G)[i]), width=50.0, height=15.0, angle=0,
                    #                   edgecolor='black',
                    #                   facecolor='none', linewidth=0.5)
                    # ax_global.add_patch(ellipse)

                    # i = 256
                    #   plt.show()
                    # ax_global.annotate('',
                    #                    xy=(i + 25, sorted(eig_G)[i] - 1 / 7),
                    #                    xytext=(350, 1 / 25),
                    #                    arrowprops=dict(arrowstyle='<-',
                    #                                    color='black',
                    #                                    lw=0.5,
                    #                                    ls='-')
                    #                    )
                    # Add an ellipse
                    # ellipse = Ellipse(xy=(i, sorted(eig_G)[i] / 2), width=71.0, height=31.0, angle=10,
                    #                   edgecolor='black',
                    #                   facecolor='none', linewidth=0.5)
                    # ax_global.add_patch(ellipse)
                    # i = 400
                    #
                    # ax_global.annotate('',
                    #                    xy=(i, sorted(eig_G)[i] - 1 / 7),
                    #                    xytext=(370, 1 / 31),
                    #                    arrowprops=dict(arrowstyle='<-',
                    #                                    color='black',
                    #                                    lw=0.5,
                    #                                    ls='-')
                    #                    )
                    # Add an ellipse
                    # ellipse = Ellipse(xy=(i, sorted(eig_G)[i]), width=50.0, height=15.0, angle=0,
                    #                   edgecolor='black',
                    #                   facecolor='none', linewidth=0.5)
                    # ax_global.add_patch(ellipse)
                    eigenvector_x = np.real(vector_to_plot)[:, i].reshape(grid_shape)[0, 0]
                    # y DIRECTION
                    eigenvector_y = np.real(vector_to_plot)[:, i].reshape(grid_shape)[1, 0]

                    dv_dx = np.gradient(eigenvector_y, 1, axis=1)  # ∂v/∂x
                    dv_dy = np.gradient(eigenvector_y, 1, axis=0)
                    du_dx = np.gradient(eigenvector_x, 1, axis=1)  # ∂u/∂y
                    du_dy = np.gradient(eigenvector_x, 1, axis=0)  # ∂u/∂y

                    curl = dv_dx - du_dy
                    mask = (np.abs(dv_dx) > 1e-12).astype(int) * (np.abs(dv_dy) > 1e-12).astype(int) * (
                            np.abs(du_dx) > 1e-12).astype(int) * (np.abs(du_dy) > 1e-12).astype(int)
                    # masking parts with zero gradient
                    eigenvector_x[...] = eigenvector_x * mask
                    eigenvector_y[...] = eigenvector_y * mask

                if upper_ax == 3:
                    i = -1
                    ax1 = fig.add_axes([0.65, 0.12, 0.2, 0.20])
                    ax1.set_title(f'Initial residual')
                    # ax_global.text(x_offset + 0.1, y_offset + 0.3, '(c.1)', transform=ax1.transAxes)
                if upper_ax == 6:
                    i = -2
                    ax1 = fig.add_axes([0.15, 0.5, 0.1, 0.20])
                    ax1.set_title(r'Weights')
                    ax_global.text(x_offset, y_offset + 0.2, '(d.1)', transform=ax1.transAxes)

                if i > 0:
                    pass

                elif i == -1:  # r0
                    eigenvector_x = np.real(rhs_to_plot).reshape(grid_shape)[
                        0, 0]
                    eigenvector_y = np.real(rhs_to_plot).reshape(grid_shape)[
                        1, 0]
                elif i == -2:
                    eigenvector_x = np.real(w_i).reshape(grid_shape)[
                        0, 0]
                    eigenvector_y = np.real(w_i).reshape(grid_shape)[
                        1, 0]

                amplitude = np.sqrt(eigenvector_x ** 2 + eigenvector_y ** 2)

                divnorm = mpl.colors.Normalize(vmin=1 / 10 ** ratio, vmax=1)
                # Define facecolors: Use 'none' for empty elements (zeros) and color for others
                facecolors = ['none' if value == 0 else 'red' for value in amplitude.flatten()]
                # Plot circles with empty ones for zero values
                # plt.scatter(x_coords_flat, y_coords_flat, s=A_flat * 100, facecolors=facecolors, edgecolors='blue', alpha=0.7)
                sizes = np.copy(amplitude)  # .flatten()
                sizes[sizes > 1e-10] = 1

                circles_sizes = 1 * np.ones_like(amplitude)
                circles_sizes[-1, :] = 0
                circles_sizes[:, -1] = 0
                # ax1.scatter(x, y, s=circles_sizes.flatten(), c='white', cmap='Reds', alpha=1.0,
                #                       norm=divnorm, edgecolors='black', linewidths=0.1),

                triangles, X, Y = get_triangle(nx=number_of_pixels, ny=number_of_pixels
                                               , lx=number_of_pixels, ly=number_of_pixels)
                # Create the triangulation object
                triangulation = tri.Triangulation(X, Y, triangles)
                ax1.triplot(triangulation, 'k-', lw=0.1)
                # ax1.axis('equal')   \

                x_ = np.arange(0 + 0.5, 1 * number_of_pixels + 0.5)
                y_ = np.arange(0 + 0.5, 1 * number_of_pixels + 0.5)
                X_, Y_ = np.meshgrid(x_, y_)
                ax1.pcolormesh(X_, Y_, phase_field,
                               cmap=mpl.cm.Greys, vmin=1 / 10 ** ratio, vmax=1, linewidth=0,
                               rasterized=True, alpha=0.6)
                # Remove the frame (axes spines)
                # for spine in ax1.spines.values():
                #     spine.set_visible(False)

                # Optionally, hide the axes ticks and labels
                ax1.set_xticks([])
                ax1.set_yticks([])

                ax1.scatter(x, y, s=sizes * 1, c=facecolors, cmap='Reds', alpha=1.0, norm=divnorm,
                            edgecolors='Red', linewidths=0)
                ax1.set_aspect('equal', 'box')

                # ax1.set_xlim(0,   1)
                # ax1.set_ylim(0,  1)

        fname = src + 'exp1_sparsity_of_eigenvectors_sharp_{}_random_init_{}_ratio_{}{}'.format(sharp, random_init,ratio, '.pdf')
        print(('create figure: {}'.format(fname)))
        plt.savefig(fname, dpi=900, bbox_inches='tight')

        # plt.figure('nasad')
        # plt.semilogy(weights_to_plot)
        # plt.ylabel('weights')
        plt.show()

    counter += 1

# quit()
# # Plot the convergence of Ritz values
# fig = plt.figure(figsize=(4.0, 4))
# gs = fig.add_gridspec(2, 1, width_ratios=[1])
# ax_poly = fig.add_subplot(gs[0, 0])
# ax_weights = fig.add_subplot(gs[1, 0])
# #            ax_error_true = fig.add_subplot(gs[1, 1])
#
# # ax_poly.plot(x_values, polynomial_value, color='red', label=r'$\varphi^{CG}$' + f'$_{{{i}}}$')
# # ax_poly.hlines(xmin=0, xmax=x_values[-1], y=0, linestyles='--', color='gray')
# ax_poly.scatter(np.real(eigens_to_plots), [0] * len(eigens_to_plots), color='blue',
#                 marker='|',
#                 label="True Eigenvalues")
# # ax_poly.scatter(np.real(ritz_values[5]), [0] * len(ritz_values[5]), color='red', marker='x',
# #                 label=f"Ritz Values\n (Approx Eigenvalues)")
# ax_weights.scatter(np.real(eigens_to_plots), np.real(weights_to_plot) / np.real(eigens_to_plots),
#                    color='red',
#                    marker='o', label=r"\frac{w_{i}}{\lamnda_{i}}")
# ax_weights.set_yscale('log')
# ax_weights.set_ylim(1e-10, 1)
# ax_weights.set_xlim(-0.1, x_values[-1] + 0.3)
# ax_weights.set_ylabel(r"$w_{i}/ \lambda_{i}$")
# ax_weights.set_title(f"Weights / Eigens ")
#
# # ax_poly.set_xlabel("Eigenvalues --- Approximation")
# # ax_poly.set_ylabel("CG (Lanczos) Iteration")
# ax_poly.set_title(f"CG polynomial (Lanczos Iteration) {{{i}}}")
# # ax_poly.set_ylim(ylim[0], ylim[1])
# ax_poly.set_xlim(-0.1, x_values[-1] + 0.3)
# ax_poly.legend()
#
# # Automatically adjust subplot parameters to avoid overlapping
# plt.tight_layout()
#
# src = '../figures/'  # source folder\
# fname = src + f'weights_sparsity' + '{}'.format('.pdf')
# plt.savefig(fname, bbox_inches='tight')
# plt.show()
# plot_cg_polynomial(x_values, ritz_values, true_eigenvalues=eigens_to_plots, weight=weights_to_plot,
#                    error_evol=norms['energy_upper_bound'] / norms['residual_rr'][
#                        0], title='Green')  # energy_lb
#
# quit()
#
# print(norms)
# plot_cg_polynomial(x_values, ritz_values, true_eigenvalues=eig_G[idx_G], weight=w_i[idx_G],
#                    error_evol=norms['energy_upper_bound'] / norms['residual_rr'][
#                        0], title='Green')  # energy_lb
# #      quit()
# plot_rhs(rhs=np.real(rhs),
#          grid_shape=rhs.shape)
# plot_rhs(rhs=np.real(w_i.reshape(rhs.shape)),
#          grid_shape=rhs.shape)
# quit()
# plot_eigenvector_filling(np.real(sorted_eig_vect_G), grid_shape=rhs.shape)
#
# plot_eigendisplacement(eigenvectors_1=np.real(eig_vect_K),
#                        grid_shape=rhs.shape, dim=2, eigenvals=eig_K[idx_K], weight=w_i[idx_K],
#                        participation_ratios=participation_ratios)
# plot_eigendisplacement(eigenvectors_1=np.real(MiK),
#                        grid_shape=rhs.shape, dim=2, eigenvals=eig_G[idx_G], weight=w_i[idx_G],
#                        participation_ratios=participation_ratios)
#
# plot_eigendisplacement(eigenvectors_1=np.real(sorted_eig_vect_G),
#                        grid_shape=rhs.shape, dim=2, eigenvals=eig_G[idx_G], weight=w_i[idx_G],
#                        participation_ratios=participation_ratios)
#
# quit()
# ############################ UNPRECONDITIONED VERSION ########################################################
# M_null = lambda x: 1 * x
# K_fun_ = lambda x: K @ x
#
# r0 = rhs.flatten() - K_fun_(x0.flatten())
# r0_norm = np.linalg.norm(r0.flatten())  # order='F'
# w_i = (np.dot(np.transpose(eig_vect_K), r0.flatten() / r0_norm)) ** 2  # order='F'
#
# x_values = np.linspace(0, np.real(sorted(eig_K)[-1] + 1), 1000)
#
# ritz_values = get_ritz_values(A=K, k_max=100, v0=r0.flatten(),
#                               M_inv=None)  # r0.flatten(order='F')
#
# plot_ritz_values(ritz_values=ritz_values, true_eigenvalues=eig_K)
#
# displacement_field, norms = solvers.PCG(K_fun_, rhs.flatten(), x0=x0.flatten(), P=M_null,
#                                         steps=int(1000), toler=1e-14,
#                                         norm_energy_upper_bound=True,
#                                         lambda_min=np.real(sorted(eig_K)[0])
#                                         )
# plot_cg_polynomial(x_values, ritz_values, true_eigenvalues=eig_K, weight=w_i,
#                    error_evol=norms['energy_lb'] / norms['residual_rr'][0],
#                    title='Unpreconditioned')
# ########################### JACOBI  PRE CONDITIONED VERSION ########################################################
# M_null = lambda x: 1 * x
# K_fun_J = lambda x: JKJsym @ x
#
# rhs_J = np.matmul(np.diag(Jacobi_sym), rhs.flatten())
#
# r0 = rhs_J.flatten() - K_fun_J(K_diag_sym @ x0.flatten())
# r0_norm = np.linalg.norm(r0.flatten())  # order='F'
#
# K_diag_sym_eig_vect_J = K_diag_sym @ eig_vect_J  # .transpose()  # ?????
# normed_eigenvectors = np.zeros_like(K_diag_sym_eig_vect_J)
# for k in np.arange(K_diag_sym_eig_vect_J[:, 0].shape[0]):
#     normed_eigenvectors[:, k] = K_diag_sym_eig_vect_J[:, k] / np.linalg.norm(
#         K_diag_sym_eig_vect_J[:, k])
#
# w_i = (np.dot(np.transpose(normed_eigenvectors), r0.flatten() / r0_norm)) ** 2  # order='F'
#
# x_values = np.linspace(0, np.real(sorted(eig_J)[-1] + 1), 100)
#
# ritz_values = get_ritz_values(A=JKJsym, k_max=80, v0=r0.flatten(),
#                               M_inv=None)  # r0.flatten(order='F')
#
# plot_ritz_values(ritz_values=ritz_values, true_eigenvalues=eig_J)
#
# displacement_field, norms = solvers.PCG(K_fun_J, rhs_J.flatten(), x0=K_diag_sym @ x0.flatten(),
#                                         P=M_null,
#                                         steps=int(1000), toler=1e-14,
#                                         norm_energy_upper_bound=True,
#                                         lambda_min=np.real(sorted(eig_J)[0])
#                                         )
# print(norms)
# plot_cg_polynomial(x_values, ritz_values, true_eigenvalues=eig_J, weight=w_i,
#                    error_evol=norms['energy_lb'] / norms['residual_rr'][0],
#                    title='Jacobi')  # energy_lb
#
# ############################ UNPRECONDITIONED VERSION ########################################################
# r0 = rhs - K_fun(x0)
# r0_norm = np.linalg.norm(r0.flatten())  # order='F'
# w_i = (np.dot(np.transpose(eig_vect_K), r0.flatten() / r0_norm)) ** 2  # order='F'
#
# x_values = np.linspace(0, np.real(sorted(eig_K)[-1] + 1), 1000)
#
# ritz_values = get_ritz_values(A=K, k_max=3, v0=r0.flatten(),
#                               M_inv=None)  # r0.flatten(order='F')
#
# plot_ritz_values(ritz_values=ritz_values, true_eigenvalues=eig_K)
#
# M_null = lambda x: 1 * x
# K_fun_ = lambda x: K @ x
# displacement_field, norms = solvers.PCG(K_fun_, rhs.flatten(), x0=x0.flatten(), P=M_null,
#                                         steps=int(1000), toler=1e-14,
#                                         norm_energy_upper_bound=True,
#                                         lambda_min=np.real(sorted(eig_K)[0])
#                                         )
# plot_cg_polynomial(x_values, ritz_values, true_eigenvalues=eig_K, weight=w_i,
#                    error_evol=norms['energy_lb'] / norms['residual_rr'][0])
#
# quit()
# # SOLVER GREEN
# displacement_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(1000), toler=1e-14,
#                                         norm_type='data_scaled_rr',
#                                         norm_metric=M_fun
#                                         )
# nb_it[nb_discretization_index + nb_starting_phases, nb_starting_phases, i] = (
#     len(norms['residual_rr']))
# print('nb it  = {} '.format(len(norms['residual_rr'])))
#
# norm_rz.append(norms['residual_rz'])
# norm_rr.append(norms['residual_rr'])
# # norm_energy_lb.append(norms['energy_lb'])
# norm_rMr.append(norms['data_scaled_rr'])
#
# #########
# displacement_field_combi, norms_combi = solvers.PCG(K_fun, rhs, x0=None, P=M_fun_combi,
#                                                     steps=int(1000),
#                                                     toler=1e-14, norm_type='data_scaled_rr',
#                                                     norm_metric=M_fun
#                                                     )
# nb_it_combi[nb_discretization_index + nb_starting_phases, nb_starting_phases, i] = (
#     len(norms_combi['residual_rr']))
# norm_rz_combi.append(norms_combi['residual_rz'])
# norm_rr_combi.append(norms_combi['residual_rr'])
# # norm_energy_lb_combi.append(norms_combi['energy_lb'])
# norm_rMr_combi.append(norms_combi['data_scaled_rr'])
#
# #
# displacement_field_Jacobi, norms_Jacobi = solvers.PCG(K_fun, rhs, x0=None, P=M_fun_Jacobi,
#                                                       steps=int(1),
#                                                       toler=1e-6, norm_type='data_scaled_rr',
#                                                       norm_metric=M_fun
#                                                       )
# nb_it_Jacobi[nb_discretization_index + nb_starting_phases, nb_starting_phases, i] = (
#     len(norms_Jacobi['residual_rr']))
# norm_rz_Jacobi.append(norms_Jacobi['residual_rz'])
# # norm_rr_Jacobi.append(norms_Jacobi['residual_rr'])
# norm_rMr_Jacobi.append(norms_Jacobi['data_scaled_rr'])
#
