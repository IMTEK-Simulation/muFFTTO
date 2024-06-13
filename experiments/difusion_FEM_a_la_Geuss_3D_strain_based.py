import numpy as np
import scipy.sparse.linalg as sp
# Import the time library
import time



def solve_sparse(A, b, M=None):
    num_iters = 0

    def callback(xk):
        nonlocal num_iters
        num_iters += 1

    x, status = sp.cg(A, b, M=M, tol=1e-6, maxiter=1000, callback=callback)
    return x, status, num_iters


###

nb_quad_points_per_pixel = 8
# PARAMETERS ##############################################################
ndim = 3  # number of dimensions (works for 2D and 3D)
N_x = N_y = N_z = 32  # number of voxels (assumed equal for all directions)
N = (N_x, N_y, N_z)  # number of voxels

del_x, del_y, del_z = 1, 1, 1  # pixel size / grid spacing
pixel_size = (del_x, del_y, del_z)
domain_vol = (del_x * N_x) * (del_y * N_y) * (del_z * N_z)  # domain volume

# Voxel discretization setting
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


# recompute quad points coordinates
# @formatter:off



# B_gradient_dqc = np.zeros([ndim, nb_quad_points_per_pixel, 4])
#
# # @formatter:off   B(dim,number of nodal values,quad point ,element)
# B_gradient_dqc[:, 0, :] = [[-1 / del_x,       0, 1 / del_x,          0],
#                            [-1 / del_y,        1 / del_y, 0,        0]] # first quad point
# B_gradient_dqc[:, 1, :] = [[0,          - 1 / del_x, 0, 1 / del_x],
#                             [0, 0, - 1 / del_y,          1 / del_y]] # second quad point
#
# B_direct_dqij = B_gradient_dqc.reshape(ndim,
#                                        nb_quad_points_per_pixel,
#                                        *ndim * (2,))

# TODO do nice/clear explanation with transforms and jacobians
# @formatter:on
# quadrature_weights = np.zeros([nb_quad_points_per_pixel])
# quadrature_weights[0] = del_x * del_y / 2
# quadrature_weights[1] = del_x * del_y / 2


def D(u_ixy, grad_u_ijqxy=None):
    # apply gradient operator
    if grad_u_ijqxy is None:
        grad_u_ijqxy = np.zeros([1, ndim, nb_quad_points_per_pixel, *N])

    for pixel_node in np.ndindex(*np.ones([ndim], dtype=int) * 2):
        # iteration over all voxel corners
        pixel_node = np.asarray(pixel_node)
        grad_u_ijqxy += np.einsum('jq,ixy...->ijqxy...',
                                  B_direct_dqij[(..., *pixel_node)],
                                  np.roll(u_ixy, -1 * pixel_node, axis=tuple(range(1, ndim + 1))))
    return grad_u_ijqxy


def D_t(flux_ijqxyz, div_flux_ixy=None):
    if div_flux_ixy is None:  # if div_u_fnxyz is not specified, determine the size
        div_flux_ixy = np.zeros([1, *N])

    for pixel_node in np.ndindex(*np.ones([ndim], dtype=int) * 2):
        # iteration over all voxel corners
        pixel_node = np.asarray(pixel_node)
        div_fnxyz_pixel_node = np.einsum('jq,ijqxy...->ixy...',
                                         B_direct_dqij[(..., *pixel_node)],
                                         flux_ijqxyz)

        div_flux_ixy += np.roll(div_fnxyz_pixel_node, 1 * pixel_node, axis=tuple(range(1, ndim + 1)))

    return div_flux_ixy


# PROBLEM DEFINITION ######################################################
# Material distribution: Square inclusion with: Obnosov solution
phase = np.ones([nb_quad_points_per_pixel, *N])
phase[:, phase.shape[1] * 1 // 4:phase.shape[1] * 3 // 4,
phase.shape[2] * 1 // 4:phase.shape[2] * 3 // 4,
:] *= 0

# material data --- thermal_conductivity
mat_contrast = 1.
inc_contrast = 20.

A2_0 = mat_contrast * np.eye(ndim)
A2_1 = inc_contrast * np.eye(ndim)

# Material data matrix --- conductivity matrix A_ij per quad point           [grid of tensors]
mat_data_ijqxy = np.einsum('ij,qxy...->ijqxy...', A2_0, phase)
mat_data_ijqxy += np.einsum('ij,qxy...->ijqxy...', A2_1, 1 - phase)

# apply quadrature weights
mat_data_ijqxy = np.einsum('ijq...,q->ijq...', mat_data_ijqxy, quadrature_weights)

# Macroscopic gradient ---  loading
macro_grad_j = np.zeros(ndim)
macro_grad_j[0] = 1
E_jqxy = np.einsum('j,qxy...->jqxy...', macro_grad_j,
                   np.ones([nb_quad_points_per_pixel, *N]))  # set macroscopic gradient loading
E_ijqxy = E_jqxy[np.newaxis, ...]  # f for elasticity

# right-hand side

# Preconditioner IN FOURIER SPACE #############################################
ref_mat_data_ij = np.eye(ndim)
# apply quadrature weights
ref_mat_data_ij = ref_mat_data_ij * quadrature_weights[0]

M_diag_ixy = np.zeros([n_u_dofs, n_u_dofs, *N])
for d in range(n_u_dofs):
    unit_impuls_ixy = np.zeros(temp_shape)
    unit_impuls_ixy[d, 0, 0, 0] = 1
    # M_diag_ixy[:, d, 0, 0, 0] = 1
    # response of the system to unit impulses
    M_diag_ixy[:, d, ...] = D_t(dot21(A=ref_mat_data_ij, v=D(u_ixy=unit_impuls_ixy)))

# Unit impulses in Fourier space --- diagonal block of size [n_u_dofs,n_u_dofs]
M_diag_ixy = np.real(fft(x=M_diag_ixy))  # imaginary part is zero
# Compute the inverse of preconditioner
M_diag_ixy[M_diag_ixy != 0] = 1 / M_diag_ixy[M_diag_ixy != 0]
# Preconditioner function
M_fun_I = lambda x: ifft(M_diag_ixy * fft(x=x.reshape(temp_shape))).real # .reshape(-1)
# Projections
# System matrix function
# K_fun_I = lambda x: dot21(mat_data_ijqxy, x.reshape(grad_shape)).reshape(-1)
# System matrix function
K_fun_I = lambda x: D_t(dot21(mat_data_ijqxy, D(u_ixy=x.reshape(temp_shape)))) # .reshape(-1)

gamma_0 = lambda x: D(M_fun_I(D_t(dot21(mat_data_ijqxy, x.reshape(grad_shape))))[0]).reshape(-1)

# right hand side vectors
b_grad_I = -gamma_0(E_ijqxy)
b_disp_I = -D_t(dot21(mat_data_ijqxy, E_ijqxy)).reshape(-1)  # right-hand side

# Calculate the start time
start = time.time()
###### Solver ######
u_sol_vec, status, num_iters = solve_sparse(
    A=sp.LinearOperator(shape=(ndof, ndof), matvec=K_fun_I, dtype='float'),
    b=b_disp_I,
    M=sp.LinearOperator(shape=(ndof, ndof), matvec=M_fun_I, dtype='float'))

print('Number of steps = {}'.format(num_iters))
# Calculate the end time and time taken
end = time.time()
length = end - start
# Show the results : this can be altered however you like
print("It took", length, "seconds!")


du_sol_ijqxy = D(u_ixy=u_sol_vec.reshape(temp_shape))
aux_ijqxy = du_sol_ijqxy + E_ijqxy
print('homogenised properties preconditioned A11 = {}'.format(
    np.inner(dot21(mat_data_ijqxy, aux_ijqxy).reshape(-1), aux_ijqxy.reshape(-1)) / domain_vol))
print('END PCG')

# Calculate the start time
start = time.time()
# Reference solution without preconditioner
du_sol_plain_I, status, num_iters = solve_sparse(
    A=sp.LinearOperator(shape=(np.prod(grad_shape), np.prod(grad_shape)), matvec=gamma_0, dtype='float'),
    b=b_grad_I,
    M=None)
print('Number of steps = {}'.format(num_iters))
# Calculate the end time and time taken
end = time.time()
length = end - start
# Show the results : this can be altered however you like
print("It took", length, "seconds!")
#
# du_sol_plain_ijqxy = D(u_ixy=u_sol_plain_I.reshape(temp_shape))

aux_plain_ijqxy = du_sol_plain_I.reshape(grad_shape) + E_ijqxy
print('homogenised properties Strain based A11 = {}'.format(
    np.inner(dot21(mat_data_ijqxy, aux_plain_ijqxy).reshape(-1), aux_plain_ijqxy.reshape(-1)) / domain_vol))
print('END CG')

J_eff = mat_contrast * np.sqrt((mat_contrast + 3 * inc_contrast) / (3 * mat_contrast + inc_contrast))
print('Analytical effective properties A11 = {}'.format(J_eff))
