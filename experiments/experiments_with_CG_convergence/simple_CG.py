import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

from muFFTTO.solvers import PCG
import numpy as np
import matplotlib.pyplot as plt


def lanczos_generalized(A, k, q, B, tol=1e-10):
    """
    Lanczos algorithm for the generalized eigenvalue problem A v = λ B v.

    Parameters:
        A (ndarray): Symmetric matrix (n x n)
        B (ndarray): Symmetric positive-definite matrix (n x n)
        k (int): Number of Lanczos iterations (must be ≤ n)
        tol (float): Convergence tolerance for reorthogonalization

    Returns:
        T (ndarray): Tridiagonal matrix (k x k)
        Q (ndarray): Orthonormal Lanczos vectors (n x k)
    """
    n = A.shape[0]
    Q = np.zeros((n, k))
    T = np.zeros((k, k))

    alpha = np.zeros(k)
    beta = np.zeros(k)

    # Initial vector, B-orthonormalized
    Q[:, 0] = q / np.sqrt(q.T @ B @ q)

    r = A @ q - (q.T @ A @ q) * B @ q
    beta[0] = np.sqrt(r.T @ B @ r)

    for i in range(1, k):
        q = r / beta[i - 1]
        Q[:, i] = q

        Aq = A @ q
        Bq = B @ q

        alpha[i] = q.T @ Aq
        r = Aq - alpha[i] * q - beta[i - 1] * Q[:, i - 1]

        beta[i] = np.sqrt(r.T @ B @ r)

    T = np.diag(alpha) + np.diag(beta[1:], 1) + np.diag(beta[1:], -1)

    return T, Q


def get_ritz_values_precon(A, k_max, v0, M=None):
    ritz_values = []
    for k in range(1, k_max + 1):
        T, Q = lanczos_generalized(A, k=k, q=v0, B=M)

        # Compute Ritz values (eigenvalues of T)
        ritz = np.linalg.eigvalsh(T)
        ritz_values.append(ritz)

    return ritz_values


def lanczos_iteration(A, k, v0, M_inv=None):
    """Performs k steps of the Lanczos iteration for symmetric matrix A."""

    n = A.shape[0]
    Q = np.zeros((n, k + 1))
    alpha = np.zeros(k)
    beta = np.zeros(k)

    Q[:, 0] = v0 / np.linalg.norm(v0)

    for j in range(k):
        w = A @ Q[:, j]
        if M_inv is not None:
            w = M_inv @ w  # Apply preconditioner
        alpha[j] = np.dot(Q[:, j], w)
        if j > 0:
            w -= beta[j - 1] * Q[:, j - 1]
        w -= alpha[j] * Q[:, j]
        beta[j] = np.linalg.norm(w)

        if beta[j] == 0:
            break  # Converged early

        Q[:, j + 1] = w / beta[j]

    return alpha, beta[:-1]  # Return tridiagonal representation


def get_ritz_values(A, k_max, v0, M_inv=None):
    ritz_values = []
    for k in range(1, k_max + 1):
        alpha, beta = lanczos_iteration(A, k, v0, M_inv=M_inv)

        # Construct the tridiagonal matrix
        T = np.diag(alpha) + np.diag(beta, k=-1) + np.diag(beta, k=1)

        # Compute Ritz values (eigenvalues of T)
        ritz = np.linalg.eigvalsh(T)
        ritz_values.append(ritz)
    return ritz_values


# # Example: Diagonal matrix with eigenvalues 1, 2, ..., 10
# n = 10
# A = np.diag(np.arange(1, n + 1, dtype=float))
#
# # Initial random vector
# v0 = np.random.rand(n)
#
# # Perform Lanczos iteration for k steps
# k_max = n  # We run full Lanczos to capture all eigenvalues
# true_eigenvalues = np.sort(np.diag(A))  # Exact eigenvalues
#
# # Store Ritz values during iterations
# ritz_values =get_ritz_values(A, k_max, v0)


def plot_ritz_values(ritz_values, true_eigenvalues):
    # Plot the convergence of Ritz values
    plt.figure(figsize=(8, 5))
    k_max = len(ritz_values)
    # for i in range(len(true_eigenvalues)):
    # plt.axhline(true_eigenvalues[i], color='gray', linestyle='--', label="True Eigenvalues" if i == 0 else "")
    plt.scatter(np.real(true_eigenvalues), [0] * len(true_eigenvalues), color='red', marker='x',
                label="True Eigenvalues")

    for i in range(k_max):
        # plt.scatter( ritz_values[i],[i + 1] * len(ritz_values[i]), color='blue', label="Ritz Values" if i == 0 else "")
        plt.scatter(np.real(ritz_values[i]), [i + 1] * len(ritz_values[i]), color='blue',
                    label="Ritz Values" if i == 0 else "")
    plt.xlabel("Eigenvalues --- Approximation")
    plt.ylabel("CG (Lanczos) Iteration")
    plt.title("Convergence of Ritz Values (Lanczos Iteration)")
    plt.legend()
    plt.xticks(np.real(true_eigenvalues))
    plt.grid(True)
    plt.show()


def get_cg_polynomial(lambda_val, ritz_values):
    """
    Compute the CG polynomial value at a given lambda value and list of Ritz values.

    Parameters:
    lambda_val (float): The lambda value at which to evaluate the polynomial.
    ritz_values (list of float): List of Ritz values at the j-th CG iteration.

    Returns:
    float: The value of the CG polynomial at the given lambda value.
    """
    j = len(ritz_values)
    numerator = 1.0
    denominator = 1.0

    for theta in ritz_values:
        numerator *= (lambda_val - theta)
        denominator *= theta

    polynomial_value = (-1) ** j * numerator / denominator
    return polynomial_value


def plot_cg_polynomial(x_values, ritz_values, true_eigenvalues, ylim=[-2.5, 2.5], weight=None, error_evol=None,
                       title=None):
    # Plot the convergence of Ritz values
    errors = np.zeros(len(ritz_values))
    for i in np.arange(0, len(ritz_values)):
        polynomial_at_eigens = get_cg_polynomial(np.real(true_eigenvalues), ritz_values[i])

        w_div_lambda = np.real(weight) / np.real(true_eigenvalues)

        errors[i] = np.dot(np.real(polynomial_at_eigens) ** 2, w_div_lambda)

    for i in np.arange(0, len(ritz_values)):
        polynomial_value = get_cg_polynomial(x_values, ritz_values[i])
        fig = plt.figure(figsize=(10.0, 6))

        if weight is None:
            gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
            ax_poly = fig.add_subplot(gs[0, 0])
            ax_error = fig.add_subplot(gs[0, 1])
            ax_error_true = fig.add_subplot(gs[1, 1])
        else:
            gs = fig.add_gridspec(2, 2, width_ratios=[1, 1])
            ax_poly = fig.add_subplot(gs[0, 0])
            ax_error = fig.add_subplot(gs[:, 1])
            ax_weights = fig.add_subplot(gs[1, 0])
        #            ax_error_true = fig.add_subplot(gs[1, 1])

        ax_poly.plot(x_values, polynomial_value, color='red', label=r'$\varphi^{CG}$' + f'$_{{{i}}}$')
        ax_poly.hlines(xmin=0, xmax=x_values[-1], y=0, linestyles='--', color='gray')
        ax_poly.scatter(np.real(true_eigenvalues), [0] * len(true_eigenvalues), color='blue', marker='|',
                        label="True Eigenvalues")
        ax_poly.scatter(np.real(ritz_values[i]), [0] * len(ritz_values[i]), color='red', marker='x',
                        label=f"Ritz Values\n (Approx Eigenvalues)")
        if weight is not None:
            ax_weights.scatter(np.real(true_eigenvalues), np.real(weight) / np.real(true_eigenvalues), color='red',
                               marker='o', label=r"\frac{w_{i}}{\lamnda_{i}}")
            ax_weights.set_yscale('log')
            ax_weights.set_ylim(1e-10, 1)
            ax_weights.set_xlim(-0.1, x_values[-1] + 0.3)
            ax_weights.set_ylabel(r"$w_{i}/ \lambda_{i}$")
            ax_weights.set_title(f"Weights / Eigens ")

        # ax_poly.set_xlabel("Eigenvalues --- Approximation")
        # ax_poly.set_ylabel("CG (Lanczos) Iteration")
        ax_poly.set_title(f"CG polynomial (Lanczos Iteration) {{{i}}}")
        ax_poly.set_ylim(ylim[0], ylim[1])
        ax_poly.set_xlim(-0.1, x_values[-1] + 0.3)
        ax_poly.legend()

        ax_error.scatter(np.arange(1, len(ritz_values[i]) + 1), errors[:i + 1], color='red', marker='x',
                         label=r'$ \sum_{l=1}^{N}\frac{\omega_{l}}{\lambda_{l}}   \left(\varphi_{' + f'{{{i}}}' + r'}^{CG}(\lambda_{l}) \right)^{2}$')
        ax_error.set_yscale('log')
        ax_error.set_ylim(1e-16, 1)
        #ax_error.set_xlim(0,50 )#len(ritz_values[-1]) + 1

        ax_error.semilogy(np.arange(0, error_evol.__len__())[:],
                          error_evol, "g",
                          linestyle='-', marker='x', label='Enorm/r0', linewidth=1)
        ax_error.set_yscale('log')
        ax_error.set_ylim(1e-16, 1)
        ax_error.set_xlim(0, 50) #len(error_evol) + 1
        # ax_error.set_xlabel(r" energy norm /$|| r_{0}||^{2} $ ")
        ax_error.legend()

        # ax_error.set_title(title)
        # Automatically adjust subplot parameters to avoid overlapping
        plt.tight_layout()

        src = '../figures/'  # source folder\
        fname = src + title + f'error_ev_it{i}' + '{}'.format('.pdf')
        plt.savefig(fname, bbox_inches='tight')
        plt.show()


def plot_eigenvectors(eigenvectors_1, eigenvectors_2, grid_shape, dim=2, eigenvals=None):
    # Plot the convergence of Ritz values
    x = np.linspace(0, 1, grid_shape[-2])
    y = np.linspace(0, 1, grid_shape[-1])
    x, y = np.meshgrid(x, y)

    # for d in np.arange(dim):
    # d=0
    divnorm1 = mpl.colors.TwoSlopeNorm(vmin=np.min(np.real(eigenvectors_1)), vcenter=0, vmax=np.max(np.real(eigenvectors_1)))
    divnorm2 = mpl.colors.TwoSlopeNorm(vmin=np.min(np.real(eigenvectors_2)), vcenter=0, vmax=np.max(np.real(eigenvectors_2)))
    for k in np.arange(0, len(eigenvectors_1)):
        fig = plt.figure(figsize=(12.0, 5))
        gs = fig.add_gridspec(1, 4, width_ratios=[0.05, 1, 1, 0.05])
        ax_eig_vecs_1 = fig.add_subplot(gs[0, 1])
        ax_eig_vecs_2 = fig.add_subplot(gs[0, 2])
        cbar_ax1 = fig.add_subplot(gs[0, 0])

        cbar_ax2 = fig.add_subplot(gs[0, 3])
        eigenvector_1 = eigenvectors_1[:, k].reshape(grid_shape)[0, 0].transpose()
        values, counts = np.unique(eigenvector_1, return_counts=True)
        if np.argmax(counts) > 5:
            most_frequent_value = values[np.argmax(counts)]
            eigenvector_1 -= most_frequent_value

        eigenvector_2 = eigenvectors_2[:, k].reshape(grid_shape)[1, 0].transpose()
        values, counts = np.unique(eigenvector_2, return_counts=True)
        if np.argmax(counts) > 5:
            most_frequent_value = values[np.argmax(counts)]
            eigenvector_2 -= most_frequent_value

        # divnorm1 = mpl.colors.TwoSlopeNorm(vmin=np.min(eigenvector_1), vcenter=np.mean(eigenvector_1), vmax=np.max(eigenvector_1))
        # divnorm2 = mpl.colors.TwoSlopeNorm(vmin=np.min(eigenvector_2), vcenter=np.mean(eigenvector_2), vmax=np.max(eigenvector_2))

        # Make them zero mean
        # eigenvector_1=eigenvector_1-np.mean(eigenvectors_1)
        # eigenvector_2=eigenvector_2-np.mean(eigenvectors_2)

        # divnorm = mpl.colors.Normalize(vmin=-1, vmax=1) #LogNorm #$Normalize
        # divnorm = mpl.colors.TwoSlopeNorm(vmin=np.min(eigenvector_1), vcenter=0, vmax=np.max(eigenvector_1))
        # Replace NaN values with zero
        levels = np.linspace(0.0, 1.0, 9)
        pcm = ax_eig_vecs_1.pcolormesh(np.real(eigenvector_1), label='PCG: Green  ', cmap=plt.cm.coolwarm,
                                       norm=divnorm1)
        cbar = plt.colorbar(pcm, location='left', cax=cbar_ax1, ticklocation='right')  # Specify the ticks

        ax_eig_vecs_1.set_title(f"Eigenvector 1 {k}")
        if eigenvals is not None:
            ax_eig_vecs_1.set_title(f"Eigenvector 1 {k},d={0}, eigval = {eigenvals[k]:.2f}")
        pcm = ax_eig_vecs_2.pcolormesh(np.real(eigenvector_2), label='PCG: Green  ', cmap=plt.cm.coolwarm,
                                       norm=divnorm2)
        ax_eig_vecs_1.axis('equal')
        ax_eig_vecs_1.set_xlim([0, grid_shape[-2]])
        ax_eig_vecs_1.set_ylim([0, grid_shape[-1]])
        # ax_eig_vecs_2.plot(np.real(eigenvector_1.flatten()), label='PCG: Green  ', )
        # ax_eig_vecs_2.set_ylim(-1, 1)
        ax_eig_vecs_2.set_title(f"Eigenvector 2 {k}")
        if eigenvals is not None:
            ax_eig_vecs_2.set_title(f"Eigenvector 2  {k},d={1}, eigval = {eigenvals[k]:.2f}")
        cbar = plt.colorbar(pcm, location='left', cax=cbar_ax2,
                            ticklocation='right')  # Specify the ticksplt.tight_layout()
        ax_eig_vecs_2.axis('equal')
        ax_eig_vecs_2.set_xlim([0, grid_shape[-2]])
        ax_eig_vecs_2.set_ylim([0, grid_shape[-1]])
        plt.tight_layout()
        plt.show()
        # plt.xlabel("Eigenvalues --- Approximation")
        # plt.ylabel("CG (Lanczos) Iteration")
        # plt.title("Convergence of Ritz Values (Lanczos Iteration)")
        # plt.legend()
        # plt.xticks(np.real(true_eigenvalues))
        # plt.grid(True)

def plot_eigendisplacement(eigenvectors_1,   grid_shape, dim=2, eigenvals=None,weight=None, participation_ratios=None):
    # Plot the convergence of Ritz values
    x = np.linspace(0, 1, grid_shape[-2])
    y = np.linspace(0, 1, grid_shape[-1])
    x, y = np.meshgrid(x, y)


    # for d in np.arange(dim):
    # d=0
    divnorm1 = mpl.colors.TwoSlopeNorm(vmin=np.min(np.real(eigenvectors_1)), vcenter=0, vmax=np.max(np.real(eigenvectors_1)))
    for k in np.arange(0, len(eigenvectors_1)):
        fig = plt.figure(figsize=(7,5))
        gs = fig.add_gridspec(2,  2, width_ratios=[ 1,2])
        ax_eig_vecs_1 = fig.add_subplot(gs[:, 1])
        ax_weights = fig.add_subplot(gs[1, 0])
        ax_ratios = fig.add_subplot(gs[0, 0])

        eigenvector_x = eigenvectors_1[:, k].reshape(grid_shape)[0, 0].transpose()
        eigenvector_y = eigenvectors_1[:, k].reshape(grid_shape)[1, 0].transpose()


        levels = np.linspace(0.0, 1.0, 9)
        ax_eig_vecs_1.quiver(x, y, eigenvector_x, eigenvector_y, scale=100.)
        ax_eig_vecs_1.set_title(f"Eigenvector  {k},d={1}, eigval = {eigenvals[k]:.2f}")



        if weight is not None:
            ax_weights.scatter(np.real(eigenvals), np.real(weight) / np.real(eigenvals), color='blue',
                               marker='o', label=r"\frac{w_{i}}{\lamnda_{i}}")
            ax_weights.set_yscale('log')
            ax_weights.set_ylim(1e-10, 1)
            ax_weights.set_xlim(-0.1, eigenvals[0] + 0.3)
            ax_weights.set_ylabel(r"$w_{i}/ \lambda_{i}$")
            ax_weights.set_title(f"Weights / Eigens ")
            ax_weights.axvline(eigenvals[k], color='red', linewidth=1)
        ax_ratios.scatter(eigenvals, participation_ratios)
        ax_ratios.axvline(eigenvals[k], color='red', linewidth=1)  # X-axis

        ax_ratios.set_xlabel('Eigenvalue')
        ax_ratios.set_ylabel('Participation ratio of eigenvector')
        ax_ratios.set_xlim([eigenvals[-1], eigenvals[0]])
        ax_ratios.set_xticks([eigenvals[-1], eigenvals[0]])
        # ax_eig_vecs_1.set_title(f"Eigenvector 1 {k}")
        # if eigenvals is not None:
        #     ax_eig_vecs_1.set_title(f"Eigenvector 1 {k},d={0}, eigval = {eigenvals[k]:.2f}")
        #
        # ax_eig_vecs_1.axis('equal')
        # ax_eig_vecs_1.set_xlim([0, grid_shape[-2]])
        # ax_eig_vecs_1.set_ylim([0, grid_shape[-1]])
        # # ax_eig_vecs_2.plot(np.real(eigenvector_1.flatten()), label='PCG: Green  ', )
        # # ax_eig_vecs_2.set_ylim(-1, 1)
        # ax_eig_vecs_2.set_title(f"Eigenvector 2 {k}")
        # if eigenvals is not None:
        #     ax_eig_vecs_2.set_title(f"Eigenvector 2  {k},d={1}, eigval = {eigenvals[k]:.2f}")
        # cbar = plt.colorbar(pcm, location='left', cax=cbar_ax2,
        #                     ticklocation='right')  # Specify the ticksplt.tight_layout()
        # ax_eig_vecs_2.axis('equal')
        # ax_eig_vecs_2.set_xlim([0, grid_shape[-2]])
        # ax_eig_vecs_2.set_ylim([0, grid_shape[-1]])
        plt.tight_layout()
        plt.show()
        # plt.xlabel("Eigenvalues --- Approximation")
        # plt.ylabel("CG (Lanczos) Iteration")
        # plt.title("Convergence of Ritz Values (Lanczos Iteration)")
        # plt.legend()
        # plt.xticks(np.real(true_eigenvalues))
        # plt.grid(True)
# Example usage:


def run_simple_CG(initial, RHS, kappa):
    np.random.seed(seed=1)
    x_lim_max = 50  # 1e3 #
    x_lim_min =0
    k = np.arange(1e3)
    convergence = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** k
    convergence_G = convergence  # * norms_linearN['residual_rr'][0]
    convergence_1 = 2 * ((np.sqrt(2) - 1) / (np.sqrt(2) + 1)) ** k
    convergence_100 = 2 * ((np.sqrt(100) - 1) / (np.sqrt(100) + 1)) ** k
    convergence_10000 = 2 * ((np.sqrt(1e4) - 1) / (np.sqrt(1e4) + 1)) ** k
    # bound1 = 2 * (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1) ** k
    # bound = 2 * np.exp(-2 * k / np.sqrt(kappa))

    iterations = np.arange(0, 1e3)

    #  fig = plt.figure(figsize=(4.5, 4.5))
    #  gs = fig.add_gridspec(1, 1, width_ratios=[1])
    #  ax_iterations = fig.add_subplot(gs[0, 0])
    #  ax_iterations.semilogy(iterations,convergence_1/2, "Grey",  linestyle='-', label=r'Estimate: $\kappa$=2', linewidth=1)
    #  ax_iterations.semilogy(iterations,convergence_100/2, "red",  linestyle=':', label=r'Estimate: $\kappa$=100', linewidth=1)
    #  ax_iterations.semilogy(iterations,convergence_10000/2, "blue",  linestyle=':', label=r'Estimate: $\kappa$=10000', linewidth=1)
    #
    #  ax_iterations.set_xlim(1, x_lim_max)
    #  #ax_iterations.set_xticks([1,5,10,15,20])
    # # ax_iterations.set_ylim([1e-14, 1])
    #  ax_iterations.set_yscale('linear')
    #  #ax_iterations.set_yscale('log')
    #  #ax_iterations.set_xscale('log')
    #  ax_iterations.set_title('x-linear, y-linear ')
    #  ax_iterations.set_ylim([1e-16, 1e0])
    #  ax_iterations.set_xlabel("PCG iterations")
    #  ax_iterations.set_ylabel(r"relative Norm of residua")
    #  ax_iterations.legend(loc='upper right' )
    #
    #  src = '../figures/'  # source folder\
    #  fname = src + f'CG_conver_{kappa}' + '{}'.format('.pdf')
    #  plt.savefig(fname, bbox_inches='tight')
    #  plt.show()
    #  quit()
    ########################################   Uniform distribution   ###############################################

    markers = [
        '.',  # Point
        ',',  # Pixel
        'o',  # Circle
        'v',  # Triangle Down
        '^',  # Triangle Up
        '<',  # Triangle Left
        '>',  # Triangle Right
        '1',  # Tri Down
        '2',  # Tri Up
        '3',  # Tri Left
        '4',  # Tri Right
        's',  # Square
        'p',  # Pentagon
        '*',  # Star
        'h',  # Hexagon1
        'H',  # Hexagon2
        '+',  # Plus
        'x',  # Cross
        'D',  # Diamond
        'd',  # Thin Diamond
        '|',  # Vertical Line
        '_',  # Horizontal Line
    ]
    colors = [
        'b',  # Blue
        'g',  # Green
        'r',  # Red
        'c',  # Cyan
        'm',  # Magenta
        'y',  # Yellow
        'k',  # Black
        'w',  # White (not useful for plots)
    ]
    Names = [r'$\kappa$ bound', r'$K^{1}$', r'$K^{2}$', r'$K^{3}$', r'$K^{4}$', r'$K^{5}$', r'$K^{6}$', r'$K^{7}$',
             r'$K^{8}$', r'$K^{9}$']
    title = r'$x_{0}$' + f'={initial}, rhs={RHS}'
    y_axis_label = r" energy norm /$|| r_{0}||^{2} $ "

    print('Uniform distributio  ')
    name_ = f'CG_conver_exampl1_x0={initial}_rhs={RHS}_kappa{kappa}_1'

    N = 5
    r = 1

    A = np.zeros([r * N, r * N])
    M = np.zeros([r * N, r * N])

    # Create the diagonal matrix
    A = np.diag(np.arange(1, r * N + 1))
    A[-1, -1] = kappa

    M = np.diag(np.random.random(r * N))
    M = np.diag(np.ones(r * N))

    # M = np.copy(A)
    # M[3,3]=1
    # M[4, 4] = 1

    if RHS == 'random':
        rhs = np.random.rand(r * N)
    elif RHS == 'linear':
        rhs = np.copy(np.diag(A))

    # rhs = np.diag(A)  # $rhs=np.random.rand(N)
    if initial == 'random':
        x0 = np.random.rand(r * N)
    elif initial == 'zeros':
        x0 = np.zeros(r * N)  #

    A_fun = lambda x: A @ x

    M_inv = np.linalg.inv(M)
    M_fun = lambda x: M_inv @ x

    eig_A, Q_A = sc.linalg.eig(a=A, b=M)  # , eigvals_only=True
    r0 = rhs - A_fun(x0)
    r0_norm = np.linalg.norm(r0)
    w_i = (np.dot(np.transpose(Q_A), r0 / r0_norm)) ** 2

    print(w_i)
    x_values = np.linspace(0, kappa, 100)
    #  Ritz values during iterations
    ritz_values = get_ritz_values(A=A, k_max=r * N, v0=r0, M_inv=M_inv)
    ritz_values_1 = get_ritz_values_precon(A=A, k_max=r * N, v0=r0, M=M)
    if plot_ritz == True:
        plot_ritz_values(ritz_values=ritz_values, true_eigenvalues=eig_A)

    x, norms_N = PCG(Afun=A_fun, B=rhs, x0=x0, P=M_fun, steps=int(5000), toler=1e-14, norm_energy_upper_bound=True,
                     lambda_min=np.real(eig_A[0]))
    if plot_CGpoly == True:
        plot_cg_polynomial(x_values, ritz_values, true_eigenvalues=eig_A, weight=w_i,
                           error_evol=norms_N['energy_lb'] / norms_N['residual_rr'][0], title=name_)
    # print(x)
    # print(norms_N)

    fig = plt.figure(figsize=(4.5, 4.5))
    gs = fig.add_gridspec(1, 1, width_ratios=[1])
    ax_iterations = fig.add_subplot(gs[0, 0])
    ax_iterations.semilogy(iterations, convergence_G, "Grey", linestyle='-', label=Names[0],
                           linewidth=1)
    # ax_iterations.semilogy(np.arange(1, norms_N['residual_rr'].__len__() + 1)[:],
    #                        norms_N['residual_rr'] / norms_N['residual_rr'][0], "g",
    #                        linestyle='-', marker='x', label='1,2,3,4,5', linewidth=1)
    ax_iterations.semilogy(np.arange(0, norms_N['energy_lb'].__len__())[:],
                           norms_N['energy_lb'] / norms_N['residual_rr'][0], color=colors[0], marker=markers[0],
                           linestyle='-', label=Names[1], linewidth=1)
    ax_iterations.set_xlim(x_lim_min, x_lim_max)
    ax_iterations.set_xticks([1, 5, 8, 15, 20, x_lim_max])
    ax_iterations.set_yscale('log')
    ax_iterations.set_ylim([1e-16, 1e1])
    ax_iterations.set_xlabel("PCG iterations")
    ax_iterations.set_ylabel(y_axis_label)
    ax_iterations.legend(loc='upper right')
    ax_iterations.set_title(title)

    src = '../figures/'  # source folder\
    fname = src + name_ + '{}'.format('.pdf')

    plt.savefig(fname, bbox_inches='tight')
    plt.show()

    ########################################  Dense matrix ---    ###############################################

    # # Step 1: Generate a random orthogonal matrix Q
    # Q, _ = np.linalg.qr(np.random.randn(r*N,r*N))  # QR decomposition to get Q
    #
    #  # Step 3: Construct the SPD matrix
    # A_ = Q @ A @ Q.T  # A = Q * Lambda * Q^T
    #
    # # Ensure symmetry
    # A_ = (A_ + A_.T) / 2
    #
    #
    # A_fun = lambda x: A_ @ x
    #
    #
    # eig_A, Q_A = sc.linalg.eig(a=A_, b=M)  # , eigvals_only=True
    # r0 = rhs - A_fun(x0)
    # r0_norm = np.linalg.norm(r0)
    # w_i = (np.dot(np.transpose(Q_A), r0 / r0_norm)) ** 2
    #
    # print(w_i)
    # x_values = np.linspace(0, kappa, 100)
    # #  Ritz values during iterations
    # ritz_values = get_ritz_values(A=A_, k_max=r * N, v0=r0, M_inv=M_inv)
    # ritz_values_1 = get_ritz_values_precon(A=A_, k_max=r * N, v0=r0, M =M )
    # plot_ritz_values(ritz_values=ritz_values, true_eigenvalues=eig_A)
    #
    # x, norms_N_dense = PCG(Afun=A_fun, B=rhs, x0=x0, P=M_fun, steps=int(5000), toler=1e-14, norm_energy_upper_bound=True,
    #                  lambda_min=np.real(eig_A[0]))
    #
    # plot_cg_polynomial(x_values, ritz_values, true_eigenvalues=eig_A, weight=w_i,
    #                    error_evol=norms_N_dense['energy_lb'] / norms_N_dense['residual_rr'][0])

    ########################################   Uniform distribution --- repeated    ###############################################
    # Specify the size of the original matrix and the repetition factor
    print('Uniform distribution --- repeated ')
    name_ = f'CG_conver_exampl1_x0={initial}_rhs={RHS}_kappa{kappa}_2'

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
    if plot_ritz == True:
        plot_ritz_values(ritz_values=ritz_values, true_eigenvalues=eig_A)

    x, norms_rep = PCG(Afun=A_fun, B=rhs, x0=x0, P=M_fun, steps=int(5000), toler=1e-14, norm_energy_upper_bound=True,
                       lambda_min=np.real(eig_A[0]))
    if plot_CGpoly == True:
        plot_cg_polynomial(x_values, ritz_values, true_eigenvalues=eig_A, weight=w_i,
                           error_evol=norms_rep['energy_lb'] / norms_rep['residual_rr'][0], title=name_)

    fig = plt.figure(figsize=(4.5, 4.5))
    gs = fig.add_gridspec(1, 1, width_ratios=[1])
    ax_iterations = fig.add_subplot(gs[0, 0])
    ax_iterations.semilogy(iterations, convergence_G, "Grey", linestyle=':', label=Names[0],
                           linewidth=1)

    ax_iterations.semilogy(np.arange(0, norms_N['residual_rr'].__len__() )[:],
                           norms_N['residual_rr'] / norms_N['residual_rr'][0], color=colors[0], marker=markers[0],
                           linestyle='-', label=Names[1], linewidth=1)

    ax_iterations.semilogy(np.arange(0, norms_rep['residual_rr'].__len__() )[:],
                           norms_rep['residual_rr'] / norms_rep['residual_rr'][0],
                           linestyle=':', color=colors[1], marker=markers[1], label=Names[2], linewidth=1)

    ax_iterations.set_xlim(x_lim_min, x_lim_max)
    ax_iterations.set_xticks([1, 5, 8, 15, 20, x_lim_max])
    # ax_iterations.set_ylim([1, 2600])
    # ax_iterations.set_yscale('linear')
    ax_iterations.set_yscale('log')
    ax_iterations.set_ylim([1e-16, 1e1])
    ax_iterations.set_xlabel("PCG iterations")
    ax_iterations.set_ylabel(y_axis_label)
    ax_iterations.legend(loc='upper right')
    ax_iterations.set_title(title)

    src = '../figures/'  # source folder\
    fname = src + name_ + '{}'.format('.pdf')
    plt.savefig(fname, bbox_inches='tight')

    ########################################   Linear distribution --- Small  ###############################################
    print('Linear distribution --- Small ')
    name_ = f'CG_conver_exampl1_x0={initial}_rhs={RHS}_kappa{kappa}_3'

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
    M_inv = np.linalg.inv(M)
    M_fun = lambda x: M_inv @ x

    eig_A, Q_A = sc.linalg.eig(a=A, b=M)  # , eigvals_only=True
    r0 = rhs - A_fun(x0)
    r0_norm = np.linalg.norm(r0)
    w_i = (np.dot(np.transpose(Q_A), r0 / r0_norm)) ** 2
    print(w_i)
    #  Ritz values during iterations
    ritz_values = get_ritz_values(A=A, k_max=r * N, v0=r0)
    x_values = np.linspace(0, kappa, 100)
    # plot_cg_polynomial(x_values, ritz_values ,true_eigenvalues=eig_A,weight=w_i)
    if plot_ritz == True:
        plot_ritz_values(ritz_values=ritz_values, true_eigenvalues=eig_A)

    x, norms_linearN = PCG(Afun=A_fun, B=rhs, x0=x0, P=M_fun, steps=int(5000), toler=1e-14,
                           norm_energy_upper_bound=True, lambda_min=np.real(eig_A[0]))
    if plot_CGpoly == True:
        plot_cg_polynomial(x_values, ritz_values, true_eigenvalues=eig_A, weight=w_i,
                           error_evol=norms_linearN['energy_lb'] / norms_linearN['residual_rr'][0], title=name_)

    print(x)
    print(norms_linearN)
    #########################################################################################################################

    fig = plt.figure(figsize=(4.5, 4.5))
    gs = fig.add_gridspec(1, 1, width_ratios=[1])
    ax_iterations = fig.add_subplot(gs[0, 0])
    ax_iterations.semilogy(iterations, convergence_G, "Grey", linestyle=':', label=Names[0],
                           linewidth=1)

    ax_iterations.semilogy(np.arange(0, norms_N['residual_rr'].__len__() )[:],
                           norms_N['residual_rr'] / norms_N['residual_rr'][0], linestyle='-', color=colors[0],
                           marker=markers[0],
                           label=Names[1], linewidth=1)

    ax_iterations.semilogy(np.arange(0, norms_rep['residual_rr'].__len__() )[:],
                           norms_rep['residual_rr'] / norms_rep['residual_rr'][0], color=colors[1], marker=markers[1],
                           linestyle=':', label=Names[2], linewidth=1)
    ax_iterations.semilogy(np.arange(0, norms_linearN['residual_rr'].__len__() )[:],
                           norms_linearN['residual_rr'] / norms_linearN['residual_rr'][0], color=colors[2],
                           marker=markers[2],
                           linestyle='-.', label=Names[3], linewidth=1)

    ax_iterations.set_xlim(x_lim_min, x_lim_max)
    ax_iterations.set_xticks([1, 5, 8, 15, 20, x_lim_max])
    # ax_iterations.set_ylim([1, 2600])
    # ax_iterations.set_yscale('linear')
    ax_iterations.set_yscale('log')
    ax_iterations.set_ylim([1e-16, 1e1])
    ax_iterations.set_xlabel("PCG iterations")
    ax_iterations.set_ylabel(y_axis_label)
    ax_iterations.legend(loc='upper right')
    ax_iterations.set_title(title)

    src = '../figures/'  # source folder\
    fname = src + name_ + '{}'.format('.pdf')
    plt.savefig(fname, bbox_inches='tight')

    ########################################   Linear distribution --- Large    ###############################################
    print('Linear distribution --- Large ')
    name_ = f'CG_conver_exampl1_x0={initial}_rhs={RHS}_kappa{kappa}_4'

    N_large = 100
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
    if plot_ritz == True:
        plot_ritz_values(ritz_values=ritz_values, true_eigenvalues=eig_A)

    A_fun = lambda x: A @ x
    M_fun = lambda x: M @ x

    x, norms_linearN_large = PCG(Afun=A_fun, B=rhs, x0=x0, P=M_fun, steps=int(5000), toler=1e-14,
                                 norm_energy_upper_bound=True, lambda_min=np.real(eig_A[0]))
    if plot_CGpoly == True:
        plot_cg_polynomial(x_values, ritz_values, true_eigenvalues=eig_A, weight=w_i,
                           error_evol=norms_linearN_large['energy_lb'] / norms_linearN_large['residual_rr'][0],
                           title=name_)

    #########################################################################################################################
    fig = plt.figure(figsize=(4.5, 4.5))
    gs = fig.add_gridspec(1, 1, width_ratios=[1])
    ax_iterations = fig.add_subplot(gs[0, 0])
    ax_iterations.semilogy(iterations, convergence_G, "Grey", linestyle=':', label=Names[0],
                           linewidth=1)

    ax_iterations.semilogy(np.arange(0, norms_N['residual_rr'].__len__() )[:],
                           norms_N['residual_rr'] / norms_N['residual_rr'][0], color=colors[0], marker=markers[0],
                           linestyle='-', label=Names[1], linewidth=1)

    ax_iterations.semilogy(np.arange(0, norms_rep['residual_rr'].__len__() )[:],
                           norms_rep['residual_rr'] / norms_rep['residual_rr'][0],
                           linestyle=':', color=colors[1], marker=markers[1], label=Names[2], linewidth=1)

    ax_iterations.semilogy(np.arange(0, norms_linearN['residual_rr'].__len__() )[:],
                           norms_linearN['residual_rr'] / norms_linearN['residual_rr'][0], color=colors[2],
                           marker=markers[2],
                           linestyle='-.', label=Names[3], linewidth=1)

    ax_iterations.semilogy(np.arange(0, norms_linearN_large['residual_rr'].__len__() )[:],
                           norms_linearN_large['residual_rr'] / norms_linearN_large['residual_rr'][0], color=colors[3],
                           marker=markers[3],
                           linestyle='-.', label=Names[4], linewidth=1)

    ax_iterations.set_xlim(x_lim_min, x_lim_max)
    ax_iterations.set_xticks([1, 5, 8, 15, 20, x_lim_max])
    # ax_iterations.set_ylim([1, 2600])
    # ax_iterations.set_yscale('linear')
    ax_iterations.set_yscale('log')
    ax_iterations.set_ylim([1e-16, 1e1])
    ax_iterations.set_xlabel("PCG iterations")
    ax_iterations.set_ylabel(y_axis_label)
    ax_iterations.legend(loc='upper right')
    src = '../figures/'  # source folder\
    fname = src + name_ + '{}'.format('.pdf')
    plt.savefig(fname, bbox_inches='tight')

    ########################################   Linear distribution --- Sparse RHS    ###############################################
    name_ = f'CG_conver_exampl1_x0={initial}_rhs={RHS}_kappa{kappa}_5'

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
    if plot_ritz == True:
        ritz_values = get_ritz_values(A=A, k_max=N_large, v0=r0)

    # plot_cg_polynomial(x_values, ritz_values ,true_eigenvalues=eig_A,weight=w_i)

    A_fun = lambda x: A @ x
    M_fun = lambda x: M @ x
    x, norms_linear_sparse_rhs = PCG(Afun=A_fun, B=rhs, x0=x0, P=M_fun, steps=int(5000), toler=1e-14,
                                     norm_energy_upper_bound=True, lambda_min=np.real(eig_A[0]))
    if plot_CGpoly == True:
        plot_cg_polynomial(x_values, ritz_values, true_eigenvalues=eig_A, weight=w_i,
                           error_evol=norms_linear_sparse_rhs['energy_lb'] / norms_linear_sparse_rhs['residual_rr'][0]
                           , title=name_)
    print(norms_linear_sparse_rhs)
    #########################################################################################################################

    fig = plt.figure(figsize=(4.5, 4.5))
    gs = fig.add_gridspec(1, 1, width_ratios=[1])
    ax_iterations = fig.add_subplot(gs[0, 0])
    ax_iterations.semilogy(iterations, convergence_G, "Grey", linestyle=':', label=Names[0],
                           linewidth=1)

    # ax_iterations.semilogy(np.arange(0, norms_N['residual_rr'].__len__() )[:],
    #                        norms_N['residual_rr'] / norms_N['residual_rr'][0], color=colors[0], marker=markers[0],
    #                        linestyle='-', label=Names[1], linewidth=1)
    #
    # ax_iterations.semilogy(np.arange(0, norms_rep['residual_rr'].__len__() )[:],
    #                        norms_rep['residual_rr'] / norms_rep['residual_rr'][0],
    #                        linestyle=':', color=colors[1], marker=markers[1], label=Names[2], linewidth=1)
    #
    # ax_iterations.semilogy(np.arange(0, norms_linearN['residual_rr'].__len__() )[:],
    #                        norms_linearN['residual_rr'] / norms_linearN['residual_rr'][0], color=colors[2],
    #                        marker=markers[2],
    #                        linestyle='-.', label=Names[3], linewidth=1)

    ax_iterations.semilogy(np.arange(0, norms_linearN_large['residual_rr'].__len__() )[:],
                           norms_linearN_large['residual_rr'] / norms_linearN_large['residual_rr'][0], color=colors[3],
                           marker=markers[3],
                           linestyle='-.', label=Names[4], linewidth=1)

    ax_iterations.semilogy(np.arange(0, norms_linear_sparse_rhs['residual_rr'].__len__() )[:],
                           norms_linear_sparse_rhs['residual_rr'] / norms_linear_sparse_rhs['residual_rr'][0],
                           color=colors[4], marker=markers[4],
                           linestyle='-.', label=Names[5],
                           linewidth=1)

    ax_iterations.set_xlim(x_lim_min, x_lim_max)
    ax_iterations.set_xticks([1, 5, 8, 15, 20, x_lim_max])
    # ax_iterations.set_ylim([1, 2600])
    # ax_iterations.set_yscale('linear')
    ax_iterations.set_yscale('log')
    ax_iterations.set_ylim([1e-16, 1e1])
    ax_iterations.set_xlabel("PCG iterations")
    ax_iterations.set_ylabel(y_axis_label)
    ax_iterations.legend(loc='upper right')
    ax_iterations.set_title(title)

    src = '../figures/'  # source folder\

    fname = src + name_ + '{}'.format('.pdf')
    plt.savefig(fname, bbox_inches='tight')
    plt.show()

    ########################### DENSE MATRIX   ######################################################

    name_ = f'CG_conver_exampl1_x0={initial}_rhs={RHS}_kappa{kappa}_6'
    # Step 1: Generate a random orthogonal matrix Q
    Q, _ = np.linalg.qr(np.random.randn(N_large, N_large))  # QR decomposition to get Q

    # Step 3: Construct the SPD matrix
    A_ = Q @ A @ Q.T  # A = Q * Lambda * Q^T

    # Ensure symmetry
    A_ = (A_ + A_.T) / 2

    A_fun = lambda x: A_ @ x

    eig_A, Q_A = sc.linalg.eig(a=A_, b=M)  # , eigvals_only=True
    r0 = rhs - A_fun(x0)
    r0_norm = np.linalg.norm(r0)
    w_i = (np.dot(np.transpose(Q_A), r0 / r0_norm)) ** 2
    print(w_i)
    x_values = np.linspace(0, kappa, 100)
    #  Ritz values during iterations
    ritz_values = get_ritz_values(A=A_, k_max=N_large, v0=r0)

    # plot_cg_polynomial(x_values, ritz_values ,true_eigenvalues=eig_A,weight=w_i)

    x, norms_linear_sparse_rhs_dense = PCG(Afun=A_fun, B=rhs, x0=x0, P=M_fun, steps=int(5000), toler=1e-14,
                                           norm_energy_upper_bound=True, lambda_min=np.real(eig_A[0]))
    if plot_CGpoly == True:
        plot_cg_polynomial(x_values, ritz_values, true_eigenvalues=eig_A, weight=w_i,
                           error_evol=norms_linear_sparse_rhs_dense['energy_lb'] /
                                      norms_linear_sparse_rhs_dense['residual_rr'][0], title=name_)
    print(norms_linear_sparse_rhs_dense)

    fig = plt.figure(figsize=(4.5, 4.5))
    gs = fig.add_gridspec(1, 1, width_ratios=[1])
    ax_iterations = fig.add_subplot(gs[0, 0])
    ax_iterations.semilogy(iterations, convergence_G, "Grey", linestyle=':', label=Names[0],
                           linewidth=1)

    # ax_iterations.semilogy(np.arange(0, norms_N['residual_rr'].__len__() )[:],
    #                        norms_N['residual_rr'] / norms_N['residual_rr'][0], color=colors[0], marker=markers[0],
    #                        linestyle='-', label=Names[1], linewidth=1)
    #
    # ax_iterations.semilogy(np.arange(0, norms_rep['residual_rr'].__len__() )[:],
    #                        norms_rep['residual_rr'] / norms_rep['residual_rr'][0],
    #                        linestyle=':', color=colors[1], marker=markers[1], label=Names[2], linewidth=1)
    #
    # ax_iterations.semilogy(np.arange(0, norms_linearN['residual_rr'].__len__() )[:],
    #                        norms_linearN['residual_rr'] / norms_linearN['residual_rr'][0], color=colors[2],
    #                        marker=markers[2],
    #                        linestyle='-.', label=Names[3], linewidth=1)

    ax_iterations.semilogy(np.arange(0, norms_linearN_large['residual_rr'].__len__() )[:],
                           norms_linearN_large['residual_rr'] / norms_linearN_large['residual_rr'][0], color=colors[3],
                           marker=markers[3],
                           linestyle='-.', label=Names[4], linewidth=1)

    ax_iterations.semilogy(np.arange(0, norms_linear_sparse_rhs['residual_rr'].__len__() )[:],
                           norms_linear_sparse_rhs['residual_rr'] / norms_linear_sparse_rhs['residual_rr'][0],
                           color=colors[4], marker=markers[4],
                           linestyle='-.', label=Names[5],
                           linewidth=1)



    ax_iterations.semilogy(np.arange(0, norms_linear_sparse_rhs_dense['residual_rr'].__len__() )[:],
                           norms_linear_sparse_rhs_dense['residual_rr'] / norms_linear_sparse_rhs_dense['residual_rr'][
                               0],   color=colors[5], marker=markers[5],
                           linestyle='-.', label=Names[6],
                           linewidth=1)

    ax_iterations.set_xlim(x_lim_min, x_lim_max)
    ax_iterations.set_xticks([1, 5, 8, 15, 20, x_lim_max])
    # ax_iterations.set_ylim([1, 2600])
    # ax_iterations.set_yscale('linear')
    ax_iterations.set_yscale('log')
    ax_iterations.set_ylim([1e-16, 1e1])
    ax_iterations.set_xlabel("PCG iterations")
    ax_iterations.set_ylabel(y_axis_label)
    ax_iterations.legend(loc='upper right')
    ax_iterations.set_title(title)

    src = '../figures/'  # source folder\
    fname = src + name_ + '{}'.format('.pdf')
    plt.savefig(fname, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    for initial in ['zeros']:  # , 'random'  'zeros'
        for RHS in ['random']:  # , 'random'
            kappa = 5
            plot_ritz = False
            plot_CGpoly = True

            run_simple_CG(initial=initial, RHS=RHS, kappa=kappa)
