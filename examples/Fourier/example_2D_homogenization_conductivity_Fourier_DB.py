import numpy as np
import scipy as sc
import scipy.sparse.linalg as sp
import itertools
import matplotlib.pyplot as plt


# smoother of the phase-field
def apply_smoother_log10(phase):
    # Define a 2D smoothing kernel
    kernel = np.array([[0.0625, 0.125, 0.0625],
                       [0.125, 0.25, 0.125],
                       [0.0625, 0.125, 0.0625]])
    # Apply convolution for smoothing
    smoothed_arr = sc.signal.convolve2d(np.log10(phase[0]), kernel, mode='same', boundary='wrap')
    # Fix bouarders
    smoothed_arr[0, :] = 0  # First row
    smoothed_arr[-1, :] = 0  # Last row
    smoothed_arr[:, 0] = 0  # First column
    smoothed_arr[:, -1] = 0  # Last column

    # Fix center point
    smoothed_arr[phase.shape[1] // 2 - 1:phase.shape[1] // 2 + 1,
    phase.shape[2] // 2 - 1:phase.shape[2] // 2 + 1] = -4

    smoothed_arr = 10 ** smoothed_arr

    return smoothed_arr


# ----------------------------------- GRID ------------------------------------
def solve_sparse(A, b, M=None):
    num_iters = 0

    def callback(xk):
        nonlocal num_iters
        num_iters += 1

    x, status = sp.cg(A, b, M=M, atol=1e-12, maxiter=5000, callback=callback)
    return x, status, num_iters


# PARAMETERS ##############################################################
ndim = 2  # number of dimensions (works for 2D and 3D)
N_x = N_y = 9  # number of voxels (assumed equal for all directions)
N = (N_x, N_y)  # number of voxels
delta_x, delta_y = 1, 1  # pixel size / grid spacing

# Quadrature points and weights
nb_quad_points_per_pixel = 1
quadrature_weights = np.zeros([nb_quad_points_per_pixel])
quadrature_weights[:] = delta_x * delta_y

# auxiliary values
prodN = np.prod(np.array(N))  # number of grid points
ndof = 1 * prodN  # number of degrees-of-freedom
flux_shape = (ndim, nb_quad_points_per_pixel) + N  # shape of the vector for storing DOFs

# PROBLEM DEFINITION ######################################################
# material data --- thermal_conductivity
mat_contrast = 1.
inc_contrast = 1e-4
# Material distribution: Square inclusion with: Obnosov solution
phase = np.ones([nb_quad_points_per_pixel, N_x, N_y])
phase[:, phase.shape[1] * 1 // 4:phase.shape[1] * 3 // 4,
phase.shape[2] * 1 // 4:phase.shape[2] * 3 // 4] *= inc_contrast

nb_of_filters = 100
nb_it_wrt_filter = []
for aplication in np.arange(nb_of_filters):
    phase[0] = apply_smoother_log10(phase)

    plot_cross = False  # bool( aplication %  20 == 0)
    if plot_cross:
        plt.figure()
        ax_cross = plt.gca()
        ax_cross.semilogy(phase[0, :, phase.shape[1] // 2], linewidth=1)
        # ax_cross.set_ylabel("Y2-axis (Cos)", color='red')

    A2_0 = mat_contrast * np.eye(ndim)
    # A2_1 = inc_contrast * np.eye(ndim)

    # Material data matrix --- conductivity matrix A_ij per quad point           [grid of tensors]
    mat_data_ijqxy = np.einsum('ij,qxy', A2_0, phase)
    # mat_data_ijqxy += np.einsum('ij,qxy', A2_1, 1 - phase)

    # apply quadrature weights
    mat_data_ijqxy = np.einsum('ijq...,q->ijq...', mat_data_ijqxy, quadrature_weights)

    # A = np.einsum('ij,...->ij...', np.eye(ndim), phase)  # material coefficients

    # set macroscopic loading
    # Macroscopic gradient ---  loading
    macro_grad_j = np.array([1, 0])
    E_jqxy = np.einsum('j,qxy', macro_grad_j,
                       np.ones([nb_quad_points_per_pixel, N_x, N_y]))  # set macroscopic gradient loading

    # PROJECTION IN FOURIER SPACE #############################################
    Ghat = np.zeros((ndim, ndim, nb_quad_points_per_pixel) + N)  # zero initialize
    freq = [np.arange(-(N[ii] - 1) / 2., +(N[ii] + 1) / 2.) for ii in range(ndim)]
    for i, j in itertools.product(range(ndim), repeat=2):
        for ind in itertools.product(*[range(n) for n in N]):
            q = np.empty(ndim)
            for ii in range(ndim):
                q[ii] = freq[ii][ind[ii]]  # frequency vector
            if not q.dot(q) == 0:  # zero freq. -> mean
                Ghat[i, j, 0][ind] = -(q[i] * q[j]) / (q.dot(q))


    def B(u_inxy, grad_u_ijqxy=None):
        # apply gradient operator
        if grad_u_ijqxy is None:
            grad_u_ijqxy = np.zeros([1, ndim, nb_quad_points_per_pixel, *N])

        temp_inxyz_F = fft(u_inxy)
        for dd in range(ndim):
            grad_u_ijqxy[0, dd, 0] = temp_inxyz_F[0, 0] * freq[dd]

        grad_u_ijqxy = ifft(grad_u_ijqxy)

        return grad_u_ijqxy


    # OPERATORS ###############################################################
    # dot21 = lambda A, v: np.einsum('ij...,j...  ->i...', A, v)
    dot21 = lambda A, v: np.einsum('ij...,j...  ->i...', A, v)  # dot product between data and gradient

    fft = lambda V: np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(V), N))
    ifft = lambda V: np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(V), N))

    grad_ = lambda V: fft(V)

    G_fun = lambda V: np.real(ifft(dot21(Ghat, fft(V)))).reshape(-1)
    A_fun = lambda v: dot21(mat_data_ijqxy, v.reshape(flux_shape))

    GA_fun = lambda v: G_fun(A_fun(v))

    # CONJUGATE GRADIENT SOLVER ###############################################
    temp_inxyz = np.zeros([1, 1, N_x, N_y])
    temp_inxyz[0,0]=np.meshgrid(np.arange(0,N_x),np.arange(0,N_x))[0]
    temp_grad_ijqxyz = np.zeros([ndim, nb_quad_points_per_pixel, N_x, N_y])

    temp_grad_ijqxyz = B(u_inxy=temp_inxyz)
    b = -GA_fun(E_jqxy)  # right-hand side
    # e, _ = sp.cg(A=sp.LinearOperator(shape=(ndof, ndof), matvec=GA_fun, dtype='float'), b=b)
    e, status, num_iters = solve_sparse(A=sp.LinearOperator(shape=(ndof, ndof), matvec=GA_fun, dtype='float'), b=b)
    print('Number of steps = {}'.format(num_iters))
    nb_it_wrt_filter.append(num_iters)

    aux = e + E_jqxy.reshape(-1)
    A_eff_11 = np.inner(A_fun(aux).reshape(-1), aux) / prodN

    print('homogenised properties A11 = {}'.format(A_eff_11))
    print('END')

    J_eff = mat_contrast * np.sqrt((mat_contrast + 3 * inc_contrast) / (3 * mat_contrast + inc_contrast))
    print('Analytical effective properties A11 = {}'.format(J_eff))
    print('Error A11 = {}, contrast = {}, N = {}'.format(A_eff_11 - J_eff, inc_contrast / mat_contrast, N_x))

    if plot_cross:
        plt.title(f'Fourier,  nb_filter = {aplication}')
        ax_cross.tick_params(axis='y', labelcolor='black')
        ax_cross.set_xticks([])
        ax_cross.set_xticklabels([])
        plt.show()

print(nb_it_wrt_filter)
plt.figure()
plt.plot(nb_it_wrt_filter)
plt.title(f'Fourier, Nx= {N_x}')
plt.xlabel('Number of filters')
plt.ylabel('Number of iterations')
plt.show()
