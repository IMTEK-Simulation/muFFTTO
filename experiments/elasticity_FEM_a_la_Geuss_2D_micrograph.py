import numpy as np
import scipy.sparse.linalg as sp
from scipy.linalg import inv
from muFFTTO import solvers


def solve_sparse(A, b, x0=None, M=None):
    num_iters = 0

    def callback(xk):
        nonlocal num_iters
        num_iters += 1

    x, status = sp.cg(A, b, x0=x0, M=M, atol=1e-8, maxiter=10000, callback=callback)
    return x, status, num_iters


def solve_sparse_ML(A, b, x0=None, M=None):

    x, norms = solvers.PCG(Afun=A, B=b,
                           x0=x0,
                           P=M,
                           steps=int(100000),
                           toler=1e-10)
    num_iters = norms['residual_rz'].__len__()
    return x,  num_iters


plot = True
compute = True
jacobi_prec = True
###
nb_quad_points_per_pixel = 2
# PARAMETERS ##############################################################
ndim = 2  # number of dimensions (works for 2D and 3D)
N_x = N_y = 128  # number of voxels (assumed equal for all directions)
N = (N_x, N_y)  # number of voxels

phase = np.load('image.npz')['bw'].astype(float)[:N_x, :N_y]

if compute:
    delta_x, delta_y = 1, 1  # pixel size / grid spacing
    domain_vol = (delta_x * N_x) * (delta_y * N_y)  # domain volume

    # auxiliary values
    n_u_dofs = ndim  # number_of_unique_dofs  1 for heat/ ndim for elasticity

    ndof = ndim * np.prod(np.array(N))  # number of dofs
    displacement_shape = (ndim,) + N  # shape of the vector for storing DOFs, (number of degrees-of-freedom)
    grad_shape = (ndim, ndim, nb_quad_points_per_pixel) + N  # shape of the gradient vector, DOFs

    # OPERATORS #
    dot21 = lambda A, v: np.einsum('ij...,j...  ->i...', A, v)
    ddot42 = lambda A, B: np.einsum('ijkl...,lk... ->ij...  ', A, B)  # dot product between data and gradient

    trans2 = lambda A2: np.einsum('ij...          ->ji...  ', A2)
    ddot22 = lambda A2, B2: np.einsum('ij...  ,ji...  ->...    ', A2, B2)
    ddot44 = lambda A4, B4: np.einsum('ijkl...,lkmn...->ijmn...', A4, B4)
    dot11 = lambda A1, B1: np.einsum('i...   ,i...   ->...    ', A1, B1)
    dot22 = lambda A2, B2: np.einsum('ij...  ,jk...  ->ik...  ', A2, B2)
    dot24 = lambda A2, B4: np.einsum('ij...  ,jkmn...->ikmn...', A2, B4)
    dot42 = lambda A4, B2: np.einsum('ijkl...,lm...  ->ijkm...', A4, B2)
    dyad22 = lambda A2, B2: np.einsum('ij...  ,kl...  ->ijkl...', A2, B2)

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
    quadrature_weights[0] = delta_x * delta_y / 2
    quadrature_weights[1] = delta_x * delta_y / 2


    def get_gradient(u_ixy, grad_u_ijqxy=None):
        # apply gradient operator
        if grad_u_ijqxy is None:
            grad_u_ijqxy = np.zeros([ndim, ndim, nb_quad_points_per_pixel, N_x, N_y])

        for pixel_node in np.ndindex(*np.ones([ndim], dtype=int) * 2):
            # iteration over all voxel corners
            pixel_node = np.asarray(pixel_node)
            grad_u_ijqxy += np.einsum('jq,ixy->ijqxy',
                                      B_direct_dqij[(..., *pixel_node)],
                                      np.roll(u_ixy, -1 * pixel_node, axis=(1, 2)))
        return grad_u_ijqxy


    def get_gradient_transposed(flux_ijqxyz, div_flux_ixy=None):
        if div_flux_ixy is None:  # if div_u_fnxyz is not specified, determine the size
            div_flux_ixy = np.zeros([ndim, N_x, N_x])

        for pixel_node in np.ndindex(*np.ones([ndim], dtype=int) * 2):
            # iteration over all voxel corners
            pixel_node = np.asarray(pixel_node)
            div_ixyz_pixel_node = np.einsum('jq,ijqxy->ixy',
                                            B_direct_dqij[(..., *pixel_node)],
                                            flux_ijqxyz)

            div_flux_ixy += np.roll(div_ixyz_pixel_node, 1 * pixel_node, axis=(1, 2))

        return div_flux_ixy


    # PROBLEM DEFINITION ######################################################
    # Square inclusion with: Obnosov solution
    # phase = np.ones([nb_quad_points_per_pixel, N_x, N_y])sig_2
    # phase[:, phase.shape[1] * 1 // 4:phase.shape[1] * 3 // 4,
    # phase.shape[2] * 1 // 4:phase.shape[2] * 3 // 4] *= 0

    # identity tensor                                               [single tensor]
    i = np.eye(ndim).astype(np.float64)
    # identity tensors                                            [grid of tensors]
    I = np.einsum('ij,qxy', i, np.ones([nb_quad_points_per_pixel, N_x, N_y])).astype(np.float64)
    I4 = np.einsum('ijkl,qxy->ijklqxy', np.einsum('il,jk', i, i), np.ones([nb_quad_points_per_pixel, N_x, N_y])).astype(
        np.float64)
    I4rt = np.einsum('ijkl,qxy->ijklqxy', np.einsum('ik,jl', i, i),
                     np.ones([nb_quad_points_per_pixel, N_x, N_y])).astype(np.float64)
    II = dyad22(I, I)
    I4s = (I4 + I4rt) / 2.
    I4d = (I4s - II / 3.)

    # function to convert material parameters to grid of scalars
    param = lambda soft, hard: soft * np.ones([nb_quad_points_per_pixel, N_x, N_y], dtype='float64') * (
            1. - phase[:N_x, :N_y]) + \
                               hard * np.ones([nb_quad_points_per_pixel, N_x, N_y], dtype='float64') * phase[:N_x, :N_y]

    # material parameters
    # TODO CHANGE BACK
    K = param(0.833, 100 * 0.833)  # bulk  modulus
    mu = param(0.386, 0.386)  # shear modulus
    sigy0 = param(0.005, 0.005 * 2.)  # initial yield stress
    H = param(0.005, 0.005 * 2.)  # hardening modulus
    n = param(0.2, 0.2)  # hardening exponent


    # yield function: return yield stress and incremental hardening modulus
    # NB: all integration points are independent, but treated at the same time
    def yield_function(ep):
        # - distinguish very low plastic strains -> linear hardening for "ep<=h"
        h = 0.0001
        low = ep <= h
        ep_hgh = np.array(ep, copy=True)
        ep_hgh[low] = h
        # - normal non-linear hardening
        Sy_hgh = sigy0 + H * ep_hgh ** n
        dH_hgh = n * H * ep_hgh ** (n - 1.)
        # - linearized hardening for "ep<=h": ensure continuity at "ep==h"
        dH_low = n * H * h ** (n - 1.)
        Sy_low = (sigy0 + H * h ** n - dH_low * h) + dH_low * ep
        # - combine initial linear hardening with non-linear hardening
        low = low.astype(np.float64)
        sigy = (1. - low) * Sy_hgh + low * Sy_low
        dH = (1. - low) * dH_hgh + low * dH_low
        # - return yield stress and linearized hardening modulus
        return sigy, dH


    def constitutive(eps, eps_t, epse_t, ep_t):
        # elastic stiffness tensor
        C4e = K * II + 2. * mu * I4d

        # trial state
        epse_s = epse_t + (eps - eps_t)
        sig_s = ddot42(C4e, epse_s)
        sigm_s = ddot22(sig_s, I) / 3.
        sigd_s = sig_s - sigm_s * I
        sigeq_s = np.sqrt(3. / 2. * ddot22(sigd_s, sigd_s))
        # avoid zero division below ("phi_s" is corrected below)
        Z = sigeq_s == 0.
        sigeq_s[Z] = 1.

        # evaluate yield surface, set to zero if elastic (or stress-free)
        sigy, dH = yield_function(ep_t)
        phi_s = sigeq_s - sigy
        phi_s = 1. / 2. * (phi_s + np.abs(phi_s))
        phi_s[Z] = 0.
        el = phi_s <= 0.

        # plastic multiplier, based on non-linear hardening
        # - initialize
        dgamma = np.zeros(ep_t.shape, dtype='float64')
        res = np.array(phi_s, copy=True)
        # - incrementally solve scalar non-linear return-map equation
        while np.max(np.abs(res) / sigy0) > 1.e-6:
            dgamma -= res / (-3. * mu - dH)
            sigy, dH = yield_function(ep_t + dgamma)
            res = sigeq_s - 3. * mu * dgamma - sigy
            res[el] = 0.
        # - enforce elastic quadrature points to stay elastic
        dgamma[el] = 0.
        dH[el] = 0.

        # return map
        N = 3. / 2. * sigd_s / sigeq_s
        ep = ep_t + dgamma
        sig = sig_s - dgamma * N * 2. * mu
        epse = epse_s - dgamma * N

        # plastic tangent stiffness
        C4ep = C4e - \
               6. * (mu ** 2.) * dgamma / sigeq_s * I4d + \
               4. * (mu ** 2.) * (dgamma / sigeq_s - 1. / (3. * mu + dH)) * dyad22(N, N)
        # consistent tangent operator: elastic/plastic switch
        el = el.astype(np.float64)
        K4 = C4e * el + C4ep * (1. - el)

        # return 3-D stress, 2-D stress/tangent, and history
        return sig, sig[:2, :2, :, :], K4[:2, :2, :2, :2, :, :], epse, ep


    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Preconditioner IN FOURIER SPACE #############################################
    ref_mat_data_ijkl = I4s[:2, :2, :2, :2]
    M_diag_ijxy = np.zeros([n_u_dofs, n_u_dofs, N_x, N_y])
    for d in range(n_u_dofs):
        unit_impuls_ixy = np.zeros(displacement_shape)
        unit_impuls_ixy[d, 0, 0] = 1
        # response of the system to unit impulses
        M_diag_ijxy[:, d, ...] = get_gradient_transposed(
            ddot42(A=ref_mat_data_ijkl, B=get_gradient(u_ixy=unit_impuls_ixy)))

    # Unit impulses in Fourier space --- diagonal block of size [n_u_dofs,n_u_dofs]
    M_diag_ijxy = np.real(fft(x=M_diag_ijxy))  # imaginary part is zero

    # Compute the inverse of preconditioner
    # Reshape the array to (n_u_dofs, n_u_dofs, ndof) for easier processing
    reshaped_matrices = M_diag_ijxy.reshape(n_u_dofs, n_u_dofs, -1)
    # Compute the inverse of each 2x2 matrix
    for i in range(reshaped_matrices.shape[-1]):
        reshaped_matrices[:, :, i] = np.linalg.pinv(reshaped_matrices[:, :, i])
    # Reshape the result back to the original (n_u_dofs, n_u_dofs, N_x, N_y) shape
    M_diag_ijxy = reshaped_matrices.reshape(n_u_dofs, n_u_dofs, N_x, N_y)
    # Preconditioner function
    M_fun_I = lambda x: np.real(ifft(dot21(M_diag_ijxy, fft(x=x.reshape(displacement_shape))))).reshape(-1)
    preconditioner_function = M_fun_I
    # ----------------------------- NEWTON ITERATIONS -----------------------------
    # initialize: stress and strain tensor, and history
    eps = np.zeros([2, 2, nb_quad_points_per_pixel, N_x, N_y], dtype='float64')
    eps_t = np.zeros([2, 2, nb_quad_points_per_pixel, N_x, N_y], dtype='float64')
    epse_t = np.zeros([2, 2, nb_quad_points_per_pixel, N_x, N_y], dtype='float64')
    ep_t = np.zeros([nb_quad_points_per_pixel, N_x, N_y], dtype='float64')

    # initial tangent operator: the elastic tangent
    K4_2 = (K * II + 2. * mu * I4d)[:2, :2, :2, :2]

    # apply quadrature weights
    K4_2_ijklqxy = np.einsum('ijklq...,q->ijklq...', K4_2, quadrature_weights)

    # define incremental macroscopic strain
    ninc = 200
    epsbar = 0.1
    # Macroscopic gradient ---  loading
    macro_grad_ij = np.zeros([2, 2])  # set macroscopic gradient loading
    macro_grad_ij[0, 0] = +np.sqrt(3.) / 2. * epsbar / float(ninc)
    macro_grad_ij[1, 1] = -np.sqrt(3.) / 2. * epsbar / float(ninc)

    nb_cg_it = []
    nb_new_it = []
    del_u_sol_I = np.zeros(displacement_shape).reshape(-1)
    del_u_sol_I_ML    = np.zeros(displacement_shape).reshape(-1)

    # incremental deformation
    for inc in range(1, ninc + 1):
        print('=============================')
        print('inc: {0:d}'.format(inc))

        E_ijqxy = np.einsum('ij,qxy', macro_grad_ij,
                            np.ones([nb_quad_points_per_pixel, N_x, N_y]))
        #  right hand size
        b_I = -get_gradient_transposed(ddot42(A=K4_2_ijklqxy, B=E_ijqxy)).reshape(-1)  # right-hand side

        eps += E_ijqxy
        # right hand side vector

        # compute DOF-normalization, set Newton iteration counter
        En = np.linalg.norm(eps)
        iiter = 0

        # iterate as long as the iterative does not vanish
        while True:

            # System matrix function
            K_fun_I = lambda x: get_gradient_transposed(
                ddot42(K4_2_ijklqxy,
                       get_gradient(u_ixy=x.reshape(displacement_shape)))).reshape(-1)

            if jacobi_prec:
                Jacobi_diag_ijxy = np.zeros([n_u_dofs, N_x, N_y])
                for nx in range(N_x):
                    for ny in range(N_y):
                        for d in range(n_u_dofs):
                            unit_impuls_ixy = np.zeros(displacement_shape)
                            unit_impuls_ixy[d, nx, ny] = 1
                            # response of the system to unit impulses
                            response = K_fun_I(unit_impuls_ixy).reshape(n_u_dofs, N_x, N_y)
                            if response[d, nx, ny] < 1e-15:
                                response[d, nx, ny] = 1
                            Jacobi_diag_ijxy[d, nx, ny] = 1 / np.sqrt(response[d, nx, ny])
                #
                M_fun = lambda x: Jacobi_diag_ijxy * (
                    M_fun_I(Jacobi_diag_ijxy * x.reshape(displacement_shape))).reshape(displacement_shape)
                preconditioner_function =M_fun
            ###### Solver ######
            # del_u_sol_I, status, num_iters = solve_sparse(
            #     A=sp.LinearOperator(shape=(ndof, ndof), matvec=K_fun_I, dtype='float'),
            #     b=b_I,
            #     x0=del_u_sol_I,
            #     M=sp.LinearOperator(shape=(ndof, ndof), matvec=M_fun_I, dtype='float'))

            del_u_sol_I, num_iters = solve_sparse_ML(
                A=sp.LinearOperator(shape=(ndof, ndof), matvec=K_fun_I, dtype='float'),
                b=b_I,
                x0=del_u_sol_I ,
                M=sp.LinearOperator(shape=(ndof, ndof), matvec=preconditioner_function, dtype='float'))

            nb_cg_it.append(num_iters)
            print('Number of PCG steps = {}'.format(num_iters))
            #print('Number of PCG steps _ML= {}'.format(num_iters_ML))

            du_sol_ijqxy = get_gradient(u_ixy=del_u_sol_I.reshape(displacement_shape))
            eps[:2, :2] += du_sol_ijqxy

            # check for convergence
            print('Strain / E0 =  {0:10.2e}'.format(np.linalg.norm(du_sol_ijqxy) / En))

            # add gradeint solution of linear system to DOFs

            # update material data
            sig, sig_2, K4_2_ijklqxy, epse, ep = constitutive(eps, eps_t, epse_t, ep_t)
            #  right hand size
            b_I = -get_gradient_transposed(sig_2).reshape(-1)

            # check for convergence
            # print('Div Stress =  {0:10.2e}'.format(np.linalg.norm(du_sol_ijqxy) / En))
            print('Div Stress = {0:10.2e}'.format(np.linalg.norm(b_I)))
            print('En= {0:10.2e}'.format(En))
            print('np.linalg.norm(del_u_sol_I) = {0:10.2e}'.format(np.linalg.norm(del_u_sol_I) ))
            #if np.linalg.norm(del_u_sol_I) / En < 1.e-5 and iiter > 0: break
            if np.linalg.norm(b_I)  < 1.e-5 and iiter > 0: break
            # update Newton iteration counter
            iiter += 1


        nb_new_it.append(iiter)
        print('Newton steps = {}'.format(iiter))
        # end-of-increment: update history
        ep_t = np.array(ep, copy=True)
        epse_t = np.array(epse, copy=True)
        eps_t = np.array(eps, copy=True)

    np.savez(f'Geuss_FEM_jacobi={jacobi_prec}_{N_x}.npz', ep_t=ep_t, epse_t=epse_t,
             eps_t=eps_t, eps=eps, nb_cg_it=nb_cg_it, nb_new_it=nb_new_it)

if plot:
    import matplotlib.pyplot as plt

    for jacobi_prec in [True,False]:#
        results = np.load(f'Geuss_FEM_jacobi={jacobi_prec}_{N_x}.npz', allow_pickle=True)
        plt.imshow(results.f.eps[0, 0, 0], cmap='YlOrRd')
        plt.show()

    fig = plt.figure(figsize=(11, 4.5))
    gs = fig.add_gridspec(2, 3)
    for jacobi_prec in [True, False]:#True,
        results = np.load(f'Geuss_FEM_jacobi={jacobi_prec}_{N_x}.npz', allow_pickle=True)
        if jacobi_prec:
            i = 1
        else:
            i = 0
        # Compute the sum of chunks
        chunk_sums = [np.sum(results.f.nb_cg_it[start:start + size]) for start, size in
                      zip(np.cumsum([0] + list(results.f.nb_new_it[:-1])), results.f.nb_new_it)]

        ax0 = fig.add_subplot(gs[i, 0])
        ax0.plot(results.f.nb_cg_it, label='nb_cg_it')
        ax0.set_xlim(0, 1500)
        ax0.set_ylim(0, 4e2)
        ax1 = fig.add_subplot(gs[i, 1])
        ax1.plot(results.f.nb_new_it, label='nb_new_it')
        ax2 = fig.add_subplot(gs[i, 2])
        ax2.plot(chunk_sums, label='NB CG per Load case')
        ax2.set_xlim(0, 200)
        ax2.set_ylim(0, 4e3)
    plt.show()
    # aux_ijqxy = du_sol_ijqxy + macro_grad_ij[:, :, np.newaxis, np.newaxis, np.newaxis]
    # print('Homogenised properties displacement based PCG C_11 = {}'.format(
    # np.inner(ddot42(K4_2_ijklqxy, aux_ijqxy).reshape(-1), aux_ijqxy.reshape(-1)) / domain_vol))
    # print('END PCG')
    #
    # # # Reference solution without preconditioner
    # # u_sol_plain_I, status, num_iters = solve_sparse(
    # #     A=sp.LinearOperator(shape=(ndof, ndof), matvec=K_fun_I, dtype='float'),
    # #     b=b_I,
    # #     M=None)
    # # print('Number of steps = {}'.format(num_iters))
    # #
    # # du_sol_plain_ijqxy = get_gradient(u_ixy=u_sol_plain_I.reshape(displacement_shape))
    # #
    # # aux_plain_ijqxy = du_sol_plain_ijqxy + E_ijqxy
    # # print('Homogenised properties displacement based CG C_11 = {}'.format(
    # #     np.inner(ddot42(K4_2_ijklqxy, aux_plain_ijqxy).reshape(-1), aux_plain_ijqxy.reshape(-1)) / domain_vol))
    # # print('END CG')
