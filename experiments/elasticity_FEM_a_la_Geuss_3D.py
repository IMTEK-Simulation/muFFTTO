import numpy as np
import scipy.sparse.linalg as sp
from scipy.linalg import inv


def solve_sparse(A, b, M=None):
    num_iters = 0

    def callback(xk):
        nonlocal num_iters
        num_iters += 1

    x, status = sp.cg(A, b, M=M, tol=1e-6, callback=callback)
    return x, status, num_iters


###
nb_quad_points_per_pixel = 8

# PARAMETERS ##############################################################
ndim = 3  # number of dimensions (works for 2D and 3D)
N_x = N_y = N_z = 27  # number of voxels (assumed equal for all directions)
N = (N_x, N_y, N_z)  # number of voxels

del_x, del_y, del_z = 1, 1, 1  # pixel size / grid spacing
pixel_size = (del_x, del_y, del_z)
domain_vol = (del_x * N_x) * (del_y * N_y) * (del_z * N_z)  # domain volume

# auxiliary values
n_u_dofs = ndim  # number_of_unique_dofs  1 for heat/ ndim for elasticity

ndof = ndim * np.prod(np.array(N))  # number of dofs
displacement_shape = (ndim,) + N  # shape of the vector for storing DOFs, (number of degrees-of-freedom)
grad_shape = (ndim, ndim, nb_quad_points_per_pixel) + N  # shape of the gradient vector, DOFs

# OPERATORS #
dot21 = lambda A, v: np.einsum('ij...,j...  ->i...', A, v)
ddot42 = lambda A, B: np.einsum('ijkl...,lk... ->ij...  ', A, B)  # dot product between data and gradient
# (inverse) Fourier transform (for each tensor component in each direction)
fft = lambda x: np.fft.fftn(x, [*N])
ifft = lambda x: np.fft.ifftn(x, [*N])

##############################################################
# Shape function gradients


"""  Structure of B matrix: 
    B(d,q,n,i,j,k) --> is B matrix evaluate gradient at point q in 
     B(d,q,n,i,j,k)  is ∂φ_nijk/∂x_d   at (q)


    B(:,:,q) --> is B matrix evaluate gradient at point q in  element e
    B(:,:,q) has size [dim,nb_of_nodes/basis_functions] 
                          (usually 4 in 2D and 8 in 3D)
     % B(:,:,q) = [ ∂φ_1/∂x_1  ∂φ_2/∂x_1  ∂φ_3/∂x_1 ... .... ∂φ_8/∂x_1 ;
%                   ∂φ_1/∂x_2  ∂φ_2/∂x_2  ∂φ_3/∂x_2 ... ...  ∂φ_8/∂x_2 ;
%                   ∂φ_1/∂x_3  ∂φ_2/∂x_3  ∂φ_3/∂x_3 ... ...  ∂φ_8/∂x_3 ]   at (q)
"""

# quadrature points : coordinates
quad_points_coord = np.zeros([ndim, nb_quad_points_per_pixel])

coord_helper = np.zeros(2)
coord_helper[0] = -1. / (np.sqrt(3))
coord_helper[1] = +1. / (np.sqrt(3))

# quadrature points    # TODO This hold for prototypical element      !!!
# TODO MAKE clear how to generate B matrices
quad_points_coord[:, 0] = [coord_helper[0], coord_helper[0], coord_helper[0]]
quad_points_coord[:, 1] = [coord_helper[1], coord_helper[0], coord_helper[0]]
quad_points_coord[:, 2] = [coord_helper[0], coord_helper[1], coord_helper[0]]
quad_points_coord[:, 3] = [coord_helper[1], coord_helper[1], coord_helper[0]]
quad_points_coord[:, 4] = [coord_helper[0], coord_helper[0], coord_helper[1]]
quad_points_coord[:, 5] = [coord_helper[1], coord_helper[0], coord_helper[1]]
quad_points_coord[:, 6] = [coord_helper[0], coord_helper[1], coord_helper[1]]
quad_points_coord[:, 7] = [coord_helper[1], coord_helper[1], coord_helper[1]]

# quadrature points : weights

quadrature_weights = np.zeros([nb_quad_points_per_pixel])
quadrature_weights[:] = del_x * del_y * del_z / 8

# Jabobian
jacoby_matrix = np.array([[(del_x / 2), 0, 0],
                          [0, (del_y / 2), 0],
                          [0, 0, (del_z / 2)]])

det_jacobian = np.linalg.det(jacoby_matrix)
inv_jacobian = np.linalg.inv(jacoby_matrix)

# construction of B matrix
B_dqijk = np.zeros(
    [ndim, nb_quad_points_per_pixel, 2, 2, 2])
for quad_point in range(0, nb_quad_points_per_pixel):
    x_q = quad_points_coord[:, quad_point]
    xi, eta, zeta = x_q[0], x_q[1], x_q[2]

    # this part have to be hard coded
    # @formatter:off

    B_dqijk[:, quad_point,  0, 0, 0] = np.array([- (1 - eta) * (1 - zeta) / 8,
                                                    - (1 -  xi) * (1 - zeta) / 8,
                                                    - (1 -  xi) * (1 -  eta) / 8])

    B_dqijk[:, quad_point,  1, 0, 0] = np.array([+ (1 - eta) * (1 - zeta) / 8,
                                                    - (1 +  xi) * (1 - zeta) / 8,
                                                    - (1 +  xi) * (1 -  eta) / 8])

    B_dqijk[:, quad_point, 0, 1, 0] = np.array([- (1 + eta) * (1 - zeta) / 8,
                                                    + (1 -  xi) * (1 - zeta) / 8,
                                                    - (1 -  xi) * (1 +  eta) / 8])

    B_dqijk[:, quad_point,  1, 1, 0] = np.array([+ (1 + eta) * (1 - zeta) / 8,
                                                    + (1 +  xi) * (1 - zeta) / 8,
                                                    - (1 +  xi) * (1 +  eta) / 8])

    B_dqijk[:, quad_point,  0, 0, 1] = np.array([- (1 - eta) * (1 + zeta) / 8,
                                                    - (1 -  xi) * (1 + zeta) / 8,
                                                    + (1 -  xi) * (1 -  eta) / 8])

    B_dqijk[:, quad_point,  1, 0, 1] = np.array([+ (1 - eta) * (1 + zeta) / 8,
                                                    - (1 +  xi) * (1 + zeta) / 8,
                                                    + (1 +  xi) * (1 -  eta) / 8])

    B_dqijk[:, quad_point, 0, 1, 1] = np.array([- (1 + eta) * (1 + zeta) / 8,
                                                    + (1 -  xi) * (1 + zeta) / 8,
                                                    + (1 -  xi) * (1 +  eta) / 8])

    B_dqijk[:, quad_point,  1, 1, 1] = np.array([+ (1 + eta) * (1 + zeta) / 8,
                                                    + (1 +  xi) * (1 + zeta) / 8,
                                                    + (1 +  xi) * (1 +  eta) / 8])


    # @formatter:on
# multiplication with inverse of jacobian
B_dqijk = np.einsum('dt,tqijk->dqijk', inv_jacobian, B_dqijk)

B_direct_dqij = B_dqijk


def get_gradient(u_ixy, grad_u_ijqxy=None):
    # apply gradient operator
    if grad_u_ijqxy is None:
        grad_u_ijqxy = np.zeros([ndim, ndim, nb_quad_points_per_pixel, *N])

    for pixel_node in np.ndindex(*np.ones([ndim], dtype=int) * 2):
        # iteration over all voxel corners
        pixel_node = np.asarray(pixel_node)
        grad_u_ijqxy += np.einsum('jq,ixy...->ijqxy...',
                                  B_direct_dqij[(..., *pixel_node)],
                                  np.roll(u_ixy, -1 * pixel_node, axis=tuple(range(1, ndim + 1))))
    return grad_u_ijqxy


def get_gradient_transposed(flux_ijqxyz, div_flux_ixy=None):
    if div_flux_ixy is None:  # if div_u_fnxyz is not specified, determine the size
        div_flux_ixy = np.zeros([ndim, *N])

    for pixel_node in np.ndindex(*np.ones([ndim], dtype=int) * 2):
        # iteration over all voxel corners
        pixel_node = np.asarray(pixel_node)
        div_ixyz_pixel_node = np.einsum('jq,ijqxy...->ixy...',
                                        B_direct_dqij[(..., *pixel_node)],
                                        flux_ijqxyz)

        div_flux_ixy += np.roll(div_ixyz_pixel_node, 1 * pixel_node, axis=tuple(range(1, ndim + 1)))

    return div_flux_ixy


# PROBLEM DEFINITION ######################################################
# Square inclusion with: Obnosov solution
phase = np.ones([nb_quad_points_per_pixel, *N])
phase[:, phase.shape[1] * 1 // 4:phase.shape[1] * 3 // 4,
phase.shape[2] * 1 // 4:phase.shape[2] * 3 // 4] *= 0

# identity tensors                                                      [single tensors]
i = np.eye(ndim)
I4 = np.einsum('il,jk', i, i)
I4rt = np.einsum('ik,jl', i, i)
I4s = (I4 + I4rt) / 2.
II = np.einsum('ij,kl ', i, i)

#                                                                       [local tensors]
K_0, K_1 = 1., 100.  # bulk  modulus
mu_0, mu_1 = 0.5, 5.  # shear modulus
# stiffness tensor
K4_0 = K_0 * II + 2. * mu_0 * (I4s - 1. / 3. * II)
K4_1 = K_1 * II + 2. * mu_1 * (I4s - 1. / 3. * II)

# Material data matrix --- stiffness tensor K_ij per quad point q          [grid of tensors]
mat_data_ijklqxy = np.einsum('ijkl,qxy...->ijklqxy...', K4_0, phase)
mat_data_ijklqxy += np.einsum('ijkl,qxy...->ijklqxy...', K4_1, 1 - phase)

# apply quadrature weights
mat_data_ijklqxy = np.einsum('ijklq...,q->ijklq...', mat_data_ijklqxy, quadrature_weights)

# Macroscopic gradient ---  loading
macro_grad_ij = np.zeros([ndim, ndim])  # set macroscopic gradient loading
macro_grad_ij[0, 0] = 1
E_ijqxy = np.einsum('ij,qxy...->ijqxy...', macro_grad_ij,
                    np.ones([nb_quad_points_per_pixel, *N]))

# System matrix function
K_fun_I = lambda x: get_gradient_transposed(
    ddot42(mat_data_ijklqxy,
           get_gradient(u_ixy=x.reshape(displacement_shape)))).reshape(-1)
# right hand side vector
b_I = -get_gradient_transposed(ddot42(A=mat_data_ijklqxy, B=E_ijqxy)).reshape(-1)  # right-hand side

# Preconditioner IN FOURIER SPACE #############################################
ref_mat_data_ijkl = I4s
M_diag_ijxy = np.zeros([n_u_dofs, n_u_dofs, *N])
for d in range(n_u_dofs):
    unit_impuls_ixy = np.zeros(displacement_shape)
    unit_impuls_ixy[d, 0, 0, 0] = 1
    # response of the system to unit impulses
    M_diag_ijxy[:, d, ...] = get_gradient_transposed(ddot42(A=ref_mat_data_ijkl, B=get_gradient(u_ixy=unit_impuls_ixy)))

# Unit impulses in Fourier space --- diagonal block of size [n_u_dofs,n_u_dofs]
M_diag_ijxy = np.real(fft(x=M_diag_ijxy))  # imaginary part is zero

# Compute the inverse of preconditioner
# Reshape the array to (n_u_dofs, n_u_dofs, ndof) for easier processing
reshaped_matrices = M_diag_ijxy.reshape(n_u_dofs, n_u_dofs, -1)
# Compute the inverse of each 2x2 matrix
for i in range(reshaped_matrices.shape[-1]):
    reshaped_matrices[:, :, i] = np.linalg.pinv(reshaped_matrices[:, :, i])
# Reshape the result back to the original (n_u_dofs, n_u_dofs, *N) shape
M_diag_ijxy = reshaped_matrices.reshape(n_u_dofs, n_u_dofs, *N)
# Preconditioner function
M_fun_I = lambda x: ifft(dot21(M_diag_ijxy, fft(x=x.reshape(displacement_shape)))).reshape(-1)

###### Solver ######
u_sol_I, status, num_iters = solve_sparse(
    A=sp.LinearOperator(shape=(ndof, ndof), matvec=K_fun_I, dtype='float'),
    b=b_I,
    M=sp.LinearOperator(shape=(ndof, ndof), matvec=M_fun_I, dtype='float'))

print('Number of steps = {}'.format(num_iters))

du_sol_ijqxy = get_gradient(u_ixy=u_sol_I.reshape(displacement_shape))
aux_ijqxy = du_sol_ijqxy + E_ijqxy
print('Homogenised properties PCG C_11 = {}'.format(
    np.inner(ddot42(mat_data_ijklqxy, aux_ijqxy).reshape(-1), aux_ijqxy.reshape(-1)) / domain_vol))
print('END PCG')

# Reference solution without preconditioner
u_sol_plain_I, status, num_iters = solve_sparse(
    A=sp.LinearOperator(shape=(ndof, ndof), matvec=K_fun_I, dtype='float'),
    b=b_I,
    M=None)
print('Number of steps = {}'.format(num_iters))

du_sol_plain_ijqxy = get_gradient(u_ixy=u_sol_plain_I.reshape(displacement_shape))

aux_plain_ijqxy = du_sol_plain_ijqxy + E_ijqxy
print('Homogenised properties CG C_11 = {}'.format(
    np.inner(ddot42(mat_data_ijklqxy, aux_plain_ijqxy).reshape(-1), aux_plain_ijqxy.reshape(-1)) / domain_vol))
print('END CG')
