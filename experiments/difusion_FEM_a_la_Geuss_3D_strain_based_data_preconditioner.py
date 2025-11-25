import numpy as np
import scipy as sc

import scipy.sparse.linalg as sp

import muFFTTO.solvers as solvers
# Import the time library
import time


def solve_sparse(A, b, x0=None, M=None):
    num_iters = 0

    def callback(xk):
        nonlocal num_iters
        num_iters += 1

    x, status = sp.cg(A, b, x0=x0, M=M, rtol=1e-10, atol=1e-5, maxiter=1000, callback=callback)
    return x, status, num_iters


###

nb_quad_points_per_pixel = 8
# PARAMETERS ##############################################################
ndim = 3  # number of dimensions (works for 2D and 3D)
N_x = N_y = N_z = 16  # number of voxels (assumed equal for all directions)
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

# quadrature points
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


####### Gradient operator #############
def B(u_ixy, grad_u_ijqxy=None):
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


####### Gradient transposed operator  (Divergence) #############
def B_t(flux_ijqxyz, div_flux_ixy=None):
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
phase.shape[2] * 1 // 4:phase.shape[2] * 3 // 4, :] *= 0
phase[:, phase.shape[1] * 1 // 4:phase.shape[1] * 3 // 4,
phase.shape[2] * 1 // 4:phase.shape[2] * 3 // 4, :] = 1e-4


# phase=np.random.rand(*phase.shape)
def apply_smoother_log10(phase):
    # Define a 2D smoothing kernel
    kernel = np.array([[0.0625, 0.125, 0.0625],
                       [0.125, 0.25, 0.125],
                       [0.0625, 0.125, 0.0625]])
    # kernel = np.array([[0.0, 0.25, 0.0],
    #                    [0.25, 0., 0.25],
    #                    [0.0, 0.25, 0.0]])
    # Apply convolution for smoothing
    smoothed_arr = np.zeros_like(phase)
    for q in range(8):

        for z in range(phase.shape[-1]):
            smoothed_arr[q, :, :, z] = sc.signal.convolve2d(np.log10(phase[q, :, :, z]), kernel, mode='same',
                                                            boundary='wrap')
    # Fix center point
    # smoothed_arr[number_of_pixels[0] // 2 - 1:number_of_pixels[0] // 2 + 1,
    # number_of_pixels[0] // 2 - 1:number_of_pixels[0] // 2 + 1] = -4
    # Fix boarders for laminate
    # smoothed_arr[ :, 0, :, :] = -4
    # smoothed_arr[3 *  phase.shape[0] // 4 - 1: 3 *  phase.shape[1] // 4 + 1, :] = 0

    smoothed_arr = 10 ** smoothed_arr

    return smoothed_arr


# apply smoothening x times
for s in range(5):
    phase = apply_smoother_log10(phase)

# material data --- thermal_conductivity
mat_contrast = 1.
inc_contrast = 1e-4

A2_0 = np.eye(ndim)
A2_0[1, 1] = 5
A2_0[2, 2] = 10
# A2_1 = inc_contrast * np.eye(ndim)

# Material data matrix --- conductivity matrix A_ij per quad point           [grid of tensors]
mat_data_pure_ijqxy = np.einsum('ij,qxy...->ijqxy...', A2_0, phase)
# mat_data_ijqxy += np.einsum('ij,qxy...->ijqxy...', A2_1, 1 - phase)

# apply quadrature weights
mat_data_ijqxy = np.einsum('ijq...,q->ijq...', mat_data_pure_ijqxy, quadrature_weights)

mat_data_pure_inverse_ijqxy = np.zeros_like(mat_data_pure_ijqxy)
mat_data_inverse_ijqxy = np.zeros_like(mat_data_pure_ijqxy)

for q in range(phase.shape[0]):
    for x in range(phase.shape[1]):
        for y in range(phase.shape[2]):
            for z in range(phase.shape[3]):
                mat_data_pure_inverse_ijqxy[:, :, q, x, y, z] = np.linalg.inv(mat_data_pure_ijqxy[:, :, q, x, y, z])
                mat_data_inverse_ijqxy[:, :, q, x, y, z] = np.linalg.inv(mat_data_ijqxy[:, :, q, x, y, z])

# Macroscopic gradient ---  loading
macro_grad_j = np.zeros(ndim)
macro_grad_j[0] = 1
# macro_grad_j[:] = 1
E_jqxy = np.einsum('j,qxy...->jqxy...', macro_grad_j,
                   np.ones([nb_quad_points_per_pixel, *N]))  # set macroscopic gradient loading
E_ijqxy = E_jqxy[np.newaxis, ...]  # f for elasticity

# Preconditioner IN FOURIER SPACE #############################################
ref_mat_data_ij = np.eye(ndim)
# ref_mat_data_ij[1, 1] = 5
# ref_mat_data_ij[2, 2] = 10
ref_mat_data_ij = np.copy(A2_0)

# apply quadrature weights
ref_mat_data_ij = ref_mat_data_ij * quadrature_weights[0]
ref_mat_data_ijqxy = np.einsum('ij,qxy...->ijqxy...', ref_mat_data_ij, phase ** 0)
ref_mat_I_data_ijqxy = np.einsum('ij,qxy...->ijqxy...', np.eye(ndim) * quadrature_weights[0], phase ** 0)
ref_mat_inv_data_ijqxy = np.einsum('ij,qxy...->ijqxy...', np.linalg.inv(ref_mat_data_ij), phase ** 0)

# Assembly preconditioned with reference material I
M_diag_I_ixy = np.zeros([n_u_dofs, n_u_dofs, *N])
for d in range(n_u_dofs):
    unit_impuls_ixy = np.zeros(temp_shape)
    unit_impuls_ixy[d, 0, 0, 0] = 1
    # M_diag_ixy[:, d, 0, 0, 0] = 1
    # response of the system to unit impulses
    M_diag_I_ixy[:, d, ...] = B_t(dot21(A=ref_mat_I_data_ijqxy, v=B(u_ixy=unit_impuls_ixy)))
# Unit impulses in Fourier space --- diagonal block of size [n_u_dofs,n_u_dofs]
M_diag_I_ixy = np.real(fft(x=M_diag_I_ixy))  # imaginary part is zero
# Compute the inverse of preconditioner
M_diag_I_ixy[M_diag_I_ixy != 0] = 1 / M_diag_I_ixy[M_diag_I_ixy != 0]
M_diag_I_ixy[0, 0, 0, 0, 0] = 0
# Preconditioner function
M_fun_I = lambda x: ifft(M_diag_I_ixy * fft(x=x.reshape(temp_shape))).real  # .reshape(-1)

# Assembly preconditioned with reference material C_
M_diag_Cr_ixy = np.zeros([n_u_dofs, n_u_dofs, *N])
for d in range(n_u_dofs):
    unit_impuls_ixy = np.zeros(temp_shape)
    unit_impuls_ixy[d, 0, 0, 0] = 1
    # M_diag_ixy[:, d, 0, 0, 0] = 1
    # response of the system to unit impulses
    M_diag_Cr_ixy[:, d, ...] = B_t(dot21(A=ref_mat_data_ij, v=B(u_ixy=unit_impuls_ixy)))
# Unit impulses in Fourier space --- diagonal block of size [n_u_dofs,n_u_dofs]
M_diag_Cr_ixy = np.real(fft(x=M_diag_Cr_ixy))  # imaginary part is zero
# Compute the inverse of preconditioner
M_diag_Cr_ixy[M_diag_Cr_ixy != 0] = 1 / M_diag_Cr_ixy[M_diag_Cr_ixy != 0]
M_diag_Cr_ixy[0, 0, 0, 0, 0] = 0
# Preconditioner function
M_fun_C = lambda x: ifft(M_diag_Cr_ixy * fft(x=x.reshape(temp_shape))).real  # .reshape(-1)

# Compute the Jacobi
J_diag_ixy = np.zeros([n_u_dofs, n_u_dofs, *N])
for d in range(n_u_dofs):
    for x in range(phase.shape[1]):
        for y in range(phase.shape[2]):
            for z in range(phase.shape[3]):
                unit_impuls_ixy = np.zeros(temp_shape)
                unit_impuls_ixy[d, x, y, z] = 1
                J_diag_ixy[:, d, x, y, z] = 1 / np.sqrt(
                    (B_t(dot21(A=mat_data_ijqxy, v=B(u_ixy=unit_impuls_ixy)))[d, x, y, z]))

# Compute the Jacobi
J_diag_comb_ixy = np.zeros([  n_u_dofs, n_u_dofs, *N])
# TODO add dimension for elasticity
for d in range(1):
    for x_i in range(2):
        for y_i in range(2):
            for z_i in range(2):
                comb_impuls_ixy = np.zeros(temp_shape)
                comb_impuls_ixy[d,x_i::2, y_i::2, z_i::2] = 1.0# set impulses at regular intervals
                # Create 2D grid
                comb_impuls_ixy=B_t(dot21(A=mat_data_ijqxy, v=B(u_ixy=comb_impuls_ixy)))

                J_diag_comb_ixy[:,d,x_i::2, y_i::2, z_i::2]  = np.where(comb_impuls_ixy[d,x_i::2, y_i::2, z_i::2] != 0.,1/ np.sqrt(comb_impuls_ixy[d,x_i::2, y_i::2, z_i::2]), 0.)



# Jacobi   Preconditioner function
# J_fun_half = lambda x:  J_diag_ixy * x.reshape(temp_shape)  # .reshape(-1)

# Jacobi - Green  Preconditioner function
JG_fun_I = lambda x: (J_diag_comb_ixy * M_fun_I(J_diag_comb_ixy * x.reshape(temp_shape)))  # .reshape(-1)
JG_fun_C = lambda x: (J_diag_comb_ixy * M_fun_C(J_diag_comb_ixy * x.reshape(temp_shape)))  # .reshape(-1)

# material data operator:
apply_data = lambda x: dot21(mat_data_ijqxy, x.reshape(grad_shape)).reshape(-1)
apply_pure_data = lambda x: dot21(mat_data_pure_ijqxy, x.reshape(grad_shape)).reshape(-1)

# apply inverse of data
apply_data_inverse = lambda x: dot21(mat_data_inverse_ijqxy, x.reshape(grad_shape))
apply_pure_data_inverse = lambda x: dot21(mat_data_pure_inverse_ijqxy, x.reshape(grad_shape))

# reference material data operator:
apply_ref_data = lambda x: dot21(ref_mat_data_ijqxy, x.reshape(grad_shape))
apply_ref_data_inverse = lambda x: dot21(ref_mat_inv_data_ijqxy, x.reshape(grad_shape)).reshape(-1)
#
apply_I_data = lambda x: dot21(ref_mat_I_data_ijqxy, x.reshape(grad_shape))

# System matrix function
K_fun_I = lambda x: B_t(dot21(mat_data_ijqxy, B(u_ixy=x.reshape(temp_shape))))  # .reshape(-1)

# G_c operator  : G_c  -    D:M:Dt:C_ref
G_I = lambda x: B(M_fun_I(B_t(apply_I_data(x)))[0]).reshape(-1)
G_c = lambda x: B(M_fun_C(B_t(apply_ref_data(x)))[0]).reshape(-1)

# G_c_T operator  : G_c transpose   C_ref:D:M:Dt
G_I_T = lambda x: apply_I_data(B(M_fun_I(B_t(x.reshape(grad_shape)))[0])).reshape(-1)
G_c_T = lambda x: apply_ref_data(B(M_fun_C(B_t(x.reshape(grad_shape)))[0])).reshape(-1)

# Projection operator with material data inverse
G_I_C_inv = lambda x: G_I_T(apply_data_inverse(x.reshape(grad_shape))[0]).reshape(-1)
G_c_C_inv = lambda x: G_c_T(apply_data_inverse(x.reshape(grad_shape))[0]).reshape(-1)

# Projection operator with material data
G_I_T_C = lambda x: G_I_T(apply_data(x.reshape(grad_shape))).reshape(-1)
G_c_T_C = lambda x: G_c_T(apply_data(x.reshape(grad_shape))).reshape(-1)

# Symmetric projection operator with material data
G_I_T_C_G_I = lambda x: G_I_T(apply_data(G_I(x).reshape(grad_shape))).reshape(-1)
G_c_T_C_G_c = lambda x: G_c_T(apply_data(G_c(x).reshape(grad_shape))).reshape(-1)

# Gamma operator  : Gamma_0
gamma_0 = lambda x: B(M_fun_I(B_t(x.reshape(grad_shape)))[0]).reshape(-1)
gamma_0_ref = lambda x: B(M_fun_C(B_t(x.reshape(grad_shape)))[0]).reshape(-1)

# Projection operator with material data evaluation: Gamma_0 *A(gradient)
gamma_0_C = lambda x: gamma_0(dot21(mat_data_ijqxy, x.reshape(grad_shape))[0]).reshape(-1)
gamma_0_ref_C = lambda x: gamma_0_C(dot21(mat_data_ijqxy, x.reshape(grad_shape))[0]).reshape(-1)
# Projection operator with material data evaluation: Gamma_0 *A(gradient)
gamma_0_C_gamma_0 = lambda x: gamma_0(dot21(mat_data_ijqxy, gamma_0(x).reshape(grad_shape))[0]).reshape(-1)

do_nothing = lambda x: 1 * x

# inverse of material data operator: Datacobi
# jacobi_data = np.eye(ndim)
# jacobi_data[1, 1] = 5
# jacobi_data[2, 2] = 10
# mat_data_I_ijqxy = np.einsum('ij,qxy...->ijqxy...', jacobi_data, phase ** 0)

# Apply jacobi preconditioner

# right hand side vectors
b_grad_I = -gamma_0_C(E_ijqxy)
b_disp_I = -B_t(dot21(mat_data_ijqxy, E_ijqxy)).reshape(-1)  # right-hand side

# Calculate the start time ############################################################################################
######################################################################################################################

print(' 1 ................ STRAIN BASED PCG  .....      :   G_I : C    : e ...........  ')
print(' 1 ................ STRAIN BASED PCG  .....  M   :       K      : e ...........  ')
start = time.time()
random_x0_disp = np.random.random(b_disp_I.shape)
random_x0_disp -= random_x0_disp.mean()
random_x0_grad = B(u_ixy=random_x0_disp.reshape(temp_shape))

du_sol_vec_muFFTTO_Jacobi, norms = solvers.PCG(Afun=G_I_T_C,
                                               x0=random_x0_grad.reshape(b_grad_I.shape),
                                               B=-G_I_T_C(E_ijqxy),  # .reshape(grad_shape),
                                               P=do_nothing,
                                               steps=int(5000),
                                               toler=1e-5,
                                               norm_type='rr')

# print('Number of steps  C Jacobi= {}'.format(num_iters))
print('Number of steps  PCG muFFTTO solvers             : G_c  : C :    e = {}'.format(norms['residual_rr'].__len__()))
print('   rr = {}'.format(norms['residual_rr'][-1]))
print('   rz = {}'.format(norms['residual_rz'][-1]))
# Calculate the end time and time taken
end = time.time()
length = end - start
# Show the results : this can be altered however you like
print("It took", length, "seconds!")
#
# du_sol_plain_ijqxy = D(u_ixy=u_sol_plain_I.reshape(temp_shape))

aux_plain_ijqxy = du_sol_vec_muFFTTO_Jacobi.reshape(grad_shape) + E_ijqxy
print('homogenised properties Strain-Based A11 C Jacobi = {}'.format(
    np.inner(dot21(mat_data_ijqxy, aux_plain_ijqxy).reshape(-1), aux_plain_ijqxy.reshape(-1)) / domain_vol))
print('END STRAIN BASED PCG    \n .............................................')

print(' 2 ................ STRAIN BASED PCG  .....  G_I 1/C : G_I  : C :  G_I   : e...........  ')
print(' 2 ................ STRAIN BASED PCG  .....  M       :        K          : e...........  ')
start = time.time()

du_sol_vec_muFFTTO_Jacobi, norms = solvers.PCG(Afun=G_I_T_C ,
                                               x0=random_x0_grad.reshape(b_grad_I.shape),
                                               B=-G_I_T_C(E_ijqxy),  # .reshape(grad_shape),
                                               P=G_I_C_inv,
                                               steps=int(1000),
                                               toler=1e-5,
                                               norm_type='rr',
                                               norm_metric=apply_data)

# print('Number of steps  C Jacobi= {}'.format(num_iters))
print('2 Number of steps  PCG muFFTTO solvers         G_I 1/C : G_I  : C :  G_I   : e = {}'.format(
    norms['residual_rr'].__len__()))
print('   rr = {}'.format(norms['residual_rr'][-1]))
print('   rz = {}'.format(norms['residual_rz'][-1]))
#print('   rCz = {}'.format(norms['data_scaled_rz'][-1]))

# Calculate the end time and time taken
end = time.time()
length = end - start
# Show the results : this can be altered however you like
print("It took", length, "seconds!")
#
# du_sol_plain_ijqxy = D(u_ixy=u_sol_plain_I.reshape(temp_shape))

aux_plain_ijqxy = du_sol_vec_muFFTTO_Jacobi.reshape(grad_shape) + E_ijqxy
print('homogenised properties Strain-Based A11 C Jacobi = {}'.format(
    np.inner(dot21(mat_data_ijqxy, aux_plain_ijqxy).reshape(-1), aux_plain_ijqxy.reshape(-1)) / domain_vol))
print('END STRAIN BASED PCG    \n .............................................')

print(' 2.1 ................ DISPLACEMENT BASED PCG  .....  J G J  :   D_t : C : D : e...........  ')
print(' 2.1 ................ DISPLACEMENT BASED PCG  .....    M    :         K     : e...........  ')

u_sol_vec_muFFTTO, norms = solvers.PCG(Afun=K_fun_I,
                                       x0=np.zeros_like(b_disp_I).reshape(temp_shape),
                                       B=b_disp_I.reshape(temp_shape),
                                       P=JG_fun_I,
                                       steps=int(1000),
                                       toler=1e-5,
                                       norm_type='data_scaled_rr',
                                      norm_metric=M_fun_I)
print(' 2.1  Number of steps  PCG muFFTTO solvers  ...  J G J     :   D_t : C : D  : e = {}'.format(
    norms['residual_rz'].__len__()))
print('   rr = {}'.format(norms['residual_rr'][-1]))
print('   rz = {}'.format(norms['residual_rz'][-1]))
print('   rCr = {}'.format(norms['data_scaled_rr'][-1]))

# Calculate the end time and time taken
end = time.time()
length = end - start
# Show the results : this can be altered however you like
print("It took", length, "seconds!")

du_sol_ijqxy = B(u_ixy=u_sol_vec_muFFTTO.reshape(temp_shape))
aux_ijqxy = du_sol_ijqxy + E_ijqxy
print('homogenised properties JACOBY GREEN  Displacement-Based A11 = {}'.format(
    np.inner(dot21(mat_data_ijqxy, aux_ijqxy).reshape(-1), aux_ijqxy.reshape(-1)) / domain_vol))
print('END DISPLACEMENT BASED PCG JACOBY GREEN  muFFTTO solvers   \n ...............')

print(' 3................. STRAIN BASED PCG  .....       : G_I  : C :  G_I  : e...........  ')
print(' 3 ................ STRAIN BASED PCG  .....  M   :        K      : e...........  ')
start = time.time()

du_sol_vec_muFFTTO_Jacobi, norms = solvers.PCG(Afun=G_I_T_C_G_I,
                                               x0=random_x0_grad.reshape(b_grad_I.shape),
                                               B=-G_I_T_C(E_ijqxy),  # .reshape(grad_shape),
                                               P=do_nothing,
                                               steps=int(1000),
                                               toler=1e-5,
                                               norm_type='rr')

# du_sol_vec_muFFTTO_Jacobi=apply_data(du_sol_vec_muFFTTO_Jacobi)
# print('Number of steps  C Jacobi= {}'.format(num_iters))
print('Number of steps  PCG muFFTTO solvers              : G_I  : C :  G_I : e = {}'.format(
    norms['residual_rr'].__len__()))
print('   rr = {}'.format(norms['residual_rr'][-1]))
print('   rz = {}'.format(norms['residual_rz'][-1]))
# Calculate the end time and time taken
end = time.time()
length = end - start
# Show the results : this can be altered however you like
print("It took", length, "seconds!")
#
# du_sol_plain_ijqxy = D(u_ixy=u_sol_plain_I.reshape(temp_shape))

aux_plain_ijqxy = du_sol_vec_muFFTTO_Jacobi.reshape(grad_shape) + E_ijqxy
print('homogenised properties Strain-Based A11  = {}'.format(
    np.inner(dot21(mat_data_ijqxy, aux_plain_ijqxy).reshape(-1), aux_plain_ijqxy.reshape(-1)) / domain_vol))

###### Solver  Strain - based  plus material preconditioner ######
print(' 4 ................ STRAIN BASED PCG  .....  1 :   Γ : C :    : e...........  ')
print(' 4 ................ STRAIN BASED PCG  .....  M :       K      : e...........  ')
start = time.time()

du_sol_vec_muFFTTO_Jacobi, norms = solvers.PCG(Afun=gamma_0_C,
                                               x0=random_x0_grad.reshape(b_grad_I.shape),  # .reshape(grad_shape),
                                               B=-gamma_0_C(E_ijqxy),  # .reshape(grad_shape),
                                               P=do_nothing,
                                               steps=int(1500),
                                               toler=1e-5,
                                               norm_type='rr')

# print('Number of steps  C Jacobi= {}'.format(num_iters))
print('Number of steps  PCG muFFTTO solvers        1 : Γ : C :    : e = {}'.format(norms['residual_rr'].__len__()))
print('   rr = {}'.format(norms['residual_rr'][-1]))
print('   rz = {}'.format(norms['residual_rz'][-1]))
# Calculate the end time and time taken
end = time.time()
length = end - start
# Show the results : this can be altered however you like
print("It took", length, "seconds!")
#
# du_sol_plain_ijqxy = D(u_ixy=u_sol_plain_I.reshape(temp_shape))

aux_plain_ijqxy = du_sol_vec_muFFTTO_Jacobi.reshape(grad_shape) + E_ijqxy
print('homogenised properties Strain-Based A11 C Jacobi = {}'.format(
    np.inner(dot21(mat_data_ijqxy, aux_plain_ijqxy).reshape(-1), aux_plain_ijqxy.reshape(-1)) / domain_vol))
print('END STRAIN BASED CG    \n .............................................')

###### Solver  Strain - based  plus material preconditioner ######
print(' 5 ................ STRAIN BASED PCG  ..... Γ_c : C : e...........  ')
print(' 5 ................ STRAIN BASED PCG  .....  M  : K : e...........  ')
start = time.time()

du_sol_vec_muFFTTO_Jacobi, norms = solvers.PCG(Afun=apply_pure_data,
                                               x0=np.zeros_like(b_grad_I),  # .reshape(grad_shape),
                                               B=-gamma_0_C(E_ijqxy),  # .reshape(grad_shape),
                                               P=gamma_0_ref,
                                               steps=int(1000),
                                               toler=1e-5,
                                               norm_type='rz')

# print('Number of steps  C Jacobi= {}'.format(num_iters))
print(' 5 Number of steps  PCG muFFTTO solvers  C Jacobi = {}'.format(norms['residual_rz'].__len__()))
print('   rr = {}'.format(norms['residual_rr'][-1]))
print('   rz = {}'.format(norms['residual_rz'][-1]))
# Calculate the end time and time taken
end = time.time()
length = end - start
# Show the results : this can be altered however you like
print("It took", length, "seconds!")
#
# du_sol_plain_ijqxy = D(u_ixy=u_sol_plain_I.reshape(temp_shape))

aux_plain_ijqxy = du_sol_vec_muFFTTO_Jacobi.reshape(grad_shape) + E_ijqxy
print('homogenised properties Strain-Based A11 C Jacobi = {}'.format(
    np.inner(dot21(mat_data_ijqxy, aux_plain_ijqxy).reshape(-1), aux_plain_ijqxy.reshape(-1)) / domain_vol))
print('END STRAIN BASED PCG  Γ G C e  \n .............................................')
print(' 6 ................ DISPLACEMENT BASED PCG  .....  Mc : K : e...........  ')
print(' 6 ................ DISPLACEMENT BASED PCG  .....  M  : K : e...........  ')
print('DISPLACEMENT BASED PCG muFFTTO solvers  ')
u_sol_vec_muFFTTO, norms = solvers.PCG(Afun=K_fun_I,
                                       x0=np.zeros_like(b_disp_I).reshape(temp_shape),
                                       B=b_disp_I.reshape(temp_shape),
                                       P=M_fun_C,
                                       steps=int(1000),
                                       toler=1e-5,
                                       norm_type='rz'
                                       )
print('Number of steps  PCG muFFTTO solvers = {}'.format(norms['residual_rz'].__len__()))
print('   rr = {}'.format(norms['residual_rr'][-1]))
print('   rz = {}'.format(norms['residual_rz'][-1]))
# Calculate the end time and time taken
end = time.time()
length = end - start
# Show the results : this can be altered however you like
print("It took", length, "seconds!")

du_sol_ijqxy = B(u_ixy=u_sol_vec_muFFTTO.reshape(temp_shape))
aux_ijqxy = du_sol_ijqxy + E_ijqxy
print('homogenised properties Displacement-Based A11 = {}'.format(
    np.inner(dot21(mat_data_ijqxy, aux_ijqxy).reshape(-1), aux_ijqxy.reshape(-1)) / domain_vol))
print('END DISPLACEMENT BASED PCG muFFTTO solvers   \n ...............')










print('................  DISPLACEMENT BASED PCG  Scipy solvers  ................ ')

start = time.time()
###### Solver  Displacement based ######
u_sol_vec, status, num_iters = solve_sparse(
    A=sp.LinearOperator(shape=(ndof, ndof), matvec=K_fun_I, dtype='float'),
    b=b_disp_I,
    x0=np.zeros_like(b_disp_I),
    M=sp.LinearOperator(shape=(ndof, ndof), matvec=M_fun_I, dtype='float'))

print('Number of steps  Displacement-Based = {}'.format(num_iters))
# Calculate the end time and time taken
end = time.time()
length = end - start
# Show the results : this can be altered however you like
print("It took", length, "seconds!")

du_sol_ijqxy = B(u_ixy=u_sol_vec.reshape(temp_shape))
aux_ijqxy = du_sol_ijqxy + E_ijqxy
print('homogenised properties Displacement-Based A11 = {}'.format(
    np.inner(dot21(mat_data_ijqxy, aux_ijqxy).reshape(-1), aux_ijqxy.reshape(-1)) / domain_vol))
print('END DISPLACEMENT BASED PCG   \n ...............')

print('DISPLACEMENT BASED PCG muFFTTO solvers  ')
u_sol_vec_muFFTTO, norms = solvers.PCG(Afun=K_fun_I,
                                       x0=np.zeros_like(b_disp_I).reshape(temp_shape),
                                       B=b_disp_I.reshape(temp_shape),
                                       P=M_fun_I,
                                       steps=int(1000),
                                       toler=1e-5,
                                       norm_type='rz'
                                       )
print('Number of steps  PCG muFFTTO solvers = {}'.format(norms['residual_rz'].__len__()))
print('   rr = {}'.format(norms['residual_rr'][-1]))
print('   rz = {}'.format(norms['residual_rz'][-1]))
# Calculate the end time and time taken
end = time.time()
length = end - start
# Show the results : this can be altered however you like
print("It took", length, "seconds!")

du_sol_ijqxy = B(u_ixy=u_sol_vec_muFFTTO.reshape(temp_shape))
aux_ijqxy = du_sol_ijqxy + E_ijqxy
print('homogenised properties Displacement-Based A11 = {}'.format(
    np.inner(dot21(mat_data_ijqxy, aux_ijqxy).reshape(-1), aux_ijqxy.reshape(-1)) / domain_vol))
print('END DISPLACEMENT BASED PCG muFFTTO solvers   \n ...............')

# Calculate the start time
print('........solve_sparse........  STRAIN BASED CG ........      :  Γ_c : C    : e...........   ............. ')
print('........solve_sparse........  STRAIN BASED CG ........  M   :      K      : e...........   ............. ')

start = time.time()
###### Solver  Strain - based ######
du_sol_plain_I, status, num_iters = solve_sparse(
    A=sp.LinearOperator(shape=(np.prod(grad_shape), np.prod(grad_shape)), matvec=gamma_0_C, dtype='float'),
    b=b_grad_I,
    M=None)
print('Number of steps  Strain-Based  ZR= {}'.format(num_iters))
# Calculate the end time and time taken
end = time.time()
length = end - start
# Show the results : this can be altered however you like
print("It took", length, "seconds!")
#
# du_sol_plain_ijqxy = D(u_ixy=u_sol_plain_I.reshape(temp_shape))

aux_plain_ijqxy = du_sol_plain_I.reshape(grad_shape) + E_ijqxy
print('homogenised properties Strain-Based A11 = {}'.format(
    np.inner(dot21(mat_data_ijqxy, aux_plain_ijqxy).reshape(-1), aux_plain_ijqxy.reshape(-1)) / domain_vol))
print('END STRAIN BASED CG   \n ...............')

print('................ STRAIN BASED CG  --- muFFTTOO solver.....      :   Γ_c : C    : e...........  ')
print('................ STRAIN BASED CG  --- muFFTTOO solver.....  M   :       K      : e...........  ')

start = time.time()

du_sol_vec_muFFTTO, norms = solvers.PCG(Afun=gamma_0_C,
                                        x0=np.zeros_like(b_grad_I),  # .reshape(grad_shape),
                                        B=b_grad_I,  # .reshape(grad_shape),
                                        P=do_nothing,
                                        steps=int(1000),
                                        toler=1e-5,
                                        norm_type='rr'
                                        )
print('Number of steps  CG muFFTTO solvers  rz = {}'.format(norms['residual_rz'].__len__()))
print('   rr = {}'.format(norms['residual_rr'][-1]))
print('   rz = {}'.format(norms['residual_rz'][-1]))

# Calculate the end time and time taken
end = time.time()
length = end - start
# Show the results : this can be altered however you like
print("It took", length, "seconds!")
#
aux_plain_ijqxy = du_sol_vec_muFFTTO.reshape(grad_shape) + E_ijqxy
print('homogenised properties Strain-Based A11 = {}'.format(
    np.inner(dot21(mat_data_ijqxy, aux_plain_ijqxy).reshape(-1), aux_plain_ijqxy.reshape(-1)) / domain_vol))

print('END STRAIN BASED CG  --- muFFTTOO solver  \n ...............')

J_eff = mat_contrast * np.sqrt((mat_contrast + 3 * inc_contrast) / (3 * mat_contrast + inc_contrast))
print('WRONG !!!! Analytical effective properties A11 = {}'.format(J_eff))
