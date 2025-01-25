import numpy as np
import scipy.sparse.linalg as sp


def solve_sparse(A, b, M=None):
    num_iters = 0

    def callback(xk):
        nonlocal num_iters
        num_iters += 1

    x, status = sp.cg(A, b, M=M, atol=1e-12, maxiter=1000, callback=callback)
    return x, status, num_iters


###

nb_quad_points_per_pixel = 2
# PARAMETERS ##############################################################
ndim = 2  # number of dimensions (works for 2D and 3D)
N_x = N_y = 48  # number of voxels (assumed equal for all directions)
N = (N_x, N_y)  # number of voxels

delta_x, delta_y = 1, 1  # pixel size / grid spacing
pixel_size = (delta_x, delta_y)
domain_vol = (delta_x * N_x) * (delta_y * N_y)  # domain volume

# auxiliary values
n_u_dofs = 1  # number_of_unique_dofs  1 for heat/ ndim for elasticity

prodN = np.prod(np.array(N))  # number of grid points
ndof = prodN
vec_shape = (ndim,) + N  # shape of the vector for storing DOFs

temp_shape = (1,) + N  # shape of the vector for storing DOFs, (number of degrees-of-freedom)
grad_shape = (1, ndim, nb_quad_points_per_pixel) + N  # shape of the gradient vector, DOFs

# OPERATORS #
dot21 = lambda A, v: np.einsum('ij...,fj...  ->fi...', A, v)  # dot product between data and gradient

# (inverse) Fourier transform (for each tensor component in each direction)
fft = lambda x: np.fft.fftn(x, [*N])
ifft = lambda x: np.fft.ifftn(x, [*N])

##############################################################
# Shape function gradients
B_gradient_dqc = np.zeros([ndim, nb_quad_points_per_pixel, 4])

# @formatter:off   B(dim,number of nodal values,quad point ,element)
B_gradient_dqc[:, 0, :] = [[-1 / delta_x,       0, 1 / delta_x,          0],
                           [-1 / delta_y,        1 / delta_y, 0,        0]] # first quad point
B_gradient_dqc[:, 1, :] = [[0,          - 1 / delta_x, 0, 1 / delta_x],
                            [0, 0, - 1 / delta_y,          1 / delta_y]] # second quad point

B_direct_dqij = B_gradient_dqc.reshape(ndim,
                                       nb_quad_points_per_pixel,
                                       *ndim * (2,))

# TODO do nice/clear explanation with transforms and jacobians
# @formatter:on
quadrature_weights = np.zeros([nb_quad_points_per_pixel])
quadrature_weights[:] = delta_x * delta_y / 2


def B(u_ixy, grad_u_ijqxy=None):
    # apply gradient operator
    if grad_u_ijqxy is None:
        grad_u_ijqxy = np.zeros([1, ndim, nb_quad_points_per_pixel, *N])

    for pixel_node in np.ndindex(*np.ones([ndim], dtype=int) * 2):
        # iteration over all voxel corners
        pixel_node = np.asarray(pixel_node)
        grad_u_ijqxy += np.einsum('jq,ixy->ijqxy',
                                  B_direct_dqij[(..., *pixel_node)],
                                  np.roll(u_ixy, -1 * pixel_node, axis=tuple(range(1, ndim + 1))))
    return grad_u_ijqxy


def B_t(flux_ijqxyz, div_flux_ixy=None):
    if div_flux_ixy is None:  # if div_u_fnxyz is not specified, determine the size
        div_flux_ixy = np.zeros([1, *N])

    for pixel_node in np.ndindex(*np.ones([ndim], dtype=int) * 2):
        # iteration over all voxel corners
        pixel_node = np.asarray(pixel_node)
        div_fnxyz_pixel_node = np.einsum('jq,ijqxy->ixy',
                                         B_direct_dqij[(..., *pixel_node)],
                                         flux_ijqxyz)

        div_flux_ixy += np.roll(div_fnxyz_pixel_node, 1 * pixel_node, axis=tuple(range(1, ndim + 1)))

    return div_flux_ixy


# PROBLEM DEFINITION ######################################################
# Material distribution: Square inclusion with: Obnosov solution
phase = np.ones([nb_quad_points_per_pixel, N_x, N_y])
phase[:, phase.shape[1] * 1 // 4:phase.shape[1] * 3 // 4,
phase.shape[2] * 1 // 4:phase.shape[2] * 3 // 4] *= 0

# material data --- thermal_conductivity
mat_contrast = 1.
inc_contrast =10

A2_0 = mat_contrast * np.eye(ndim)
A2_1 = inc_contrast * np.eye(ndim)

# Material data matrix --- conductivity matrix A_ij per quad point           [grid of tensors]
mat_data_ijqxy = np.einsum('ij,qxy', A2_0, phase)
mat_data_ijqxy += np.einsum('ij,qxy', A2_1, 1 - phase)

# apply quadrature weights
mat_data_ijqxy = np.einsum('ijq...,q->ijq...', mat_data_ijqxy, quadrature_weights)

# Macroscopic gradient ---  loading
macro_grad_j = np.array([1, 0])
E_jqxy = np.einsum('j,qxy', macro_grad_j,
                   np.ones([nb_quad_points_per_pixel, N_x, N_y]))  # set macroscopic gradient loading
E_ijqxy = E_jqxy[np.newaxis, ...]  # f for elasticity

# System matrix function
K_fun_I = lambda x: B_t(
    dot21(mat_data_ijqxy,
          B(u_ixy=x.reshape(temp_shape)))).reshape(-1)
# right hand side vector
b_I = -B_t(dot21(mat_data_ijqxy, E_ijqxy)).reshape(-1)  # right-hand side

# Preconditioner IN FOURIER SPACE #############################################
ref_mat_data_ij = np.array([[1, 0], [0, 1]])
# apply quadrature weights
ref_mat_data_ij = ref_mat_data_ij * quadrature_weights[0]
M_diag_ixy = np.zeros([n_u_dofs, N_x, N_y])
for d in range(n_u_dofs):
    unit_impuls_ixy = np.zeros(temp_shape)
    unit_impuls_ixy[d, 0, 0] = 1
    # M_diag_ixy[:, d, 0, 0] = 1
    # response of the system to unit impulses
    M_diag_ixy[d, ...] = B_t(
        dot21(A=ref_mat_data_ij, v=B(u_ixy=unit_impuls_ixy)))  # TODO {change back!!!!§}

# Unit impulses in Fourier space --- diagonal block of size [n_u_dofs,n_u_dofs]
M_diag_ixy = (fft(x=M_diag_ixy))  # imaginary part is zero
# Compute the inverse of preconditioner
M_diag_ixy[M_diag_ixy != 0] = 1 / M_diag_ixy[M_diag_ixy != 0]
# M_diag_ixy= np.linalg.pinv(M_diag_ixy)

# Preconditioner function
dot11 = lambda A, v: np.einsum('i...,i...  ->i...', A, v)  # dot product between precon and
M_fun_I = lambda x: np.real((ifft(dot11(M_diag_ixy, fft(x=x.reshape(temp_shape))))).reshape(-1))

###### Solver ######
u_sol_vec, status, num_iters = solve_sparse(
    A=sp.LinearOperator(shape=(ndof, ndof), matvec=K_fun_I, dtype='float'),
    b=b_I,
    M=sp.LinearOperator(shape=(ndof, ndof), matvec=M_fun_I, dtype='float'))

print('Number of steps  PCG = {}'.format(num_iters))

du_sol_ijqxy = B(u_ixy=u_sol_vec.reshape(temp_shape))
aux_ijqxy = du_sol_ijqxy + E_ijqxy
A_eff = np.inner(dot21(mat_data_ijqxy, aux_ijqxy).reshape(-1), aux_ijqxy.reshape(-1)) / domain_vol
print('homogenised properties preconditioned A11 = {}'.format(A_eff))
print('END PCG')

# Reference solution without preconditioner
u_sol_plain_I, status, num_iters = solve_sparse(
    A=sp.LinearOperator(shape=(ndof, ndof), matvec=K_fun_I, dtype='float'),
    b=b_I,
    M=None)
print('Number of steps plain= {}'.format(num_iters))

du_sol_plain_ijqxy = B(u_ixy=u_sol_plain_I.reshape(temp_shape))

aux_plain_ijqxy = du_sol_plain_ijqxy + E_ijqxy
print('homogenised properties plain A11 = {}'.format(
    np.inner(dot21(mat_data_ijqxy, aux_plain_ijqxy).reshape(-1), aux_plain_ijqxy.reshape(-1)) / domain_vol))
print('END CG')

J_eff = mat_contrast * np.sqrt((mat_contrast + 3 * inc_contrast) / (3 * mat_contrast + inc_contrast))
print('Analytical effective properties A11 = {}'.format(J_eff))
print('Error A11 = {}, contrast = {}, N = {}'.format(A_eff - J_eff , inc_contrast/mat_contrast, N_x))
