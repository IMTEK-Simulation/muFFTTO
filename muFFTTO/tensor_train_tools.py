import numpy as np
import scipy as sc
import ttpy as tt
def tt_decompose_rank(tensor_xyz, ranks):
    # function computes TT decomposition of the tensor_xyz with specified ranks
    #
    dim = tensor_xyz.ndim
    shape = tensor_xyz.shape
    size = tensor_xyz.size
    temp = tensor_xyz
    #
    tt_cores = []
    for d in np.arange(0, dim - 1):
        # shape of unfolding in d-step
        d_unfold_shape = np.array([ranks[d] * shape[d], temp.size / (ranks[d] * shape[d])], dtype=int)
        # unfolding --- reshape
        temp = np.reshape(temp, d_unfold_shape)

        # SVD decomposition
        [U, S, Vh] = np.linalg.svd(temp, full_matrices=False)
        # singular_values = S;
        # U[:, 0:ranks[d + 1]]
        # S[0:ranks[d + 1], 0:ranks[d + 1]]
        # Vh[ 0:ranks[d + 1], :]

        # A_truncated = np.dot(U[:, 0:ranks[d + 1]] * S[0:ranks[d + 1]], Vh[0:ranks[d + 1], :])
        # assign to the d-th core
        d_core_shape = np.array([ranks[d], shape[d], ranks[d + 1]], dtype=int)
        tt_cores.append(np.reshape(U[:, 0:ranks[d + 1]], d_core_shape))

        #
        temp = np.matmul(np.diag(S[0:ranks[d + 1]]), Vh[0:ranks[d + 1], :])

    tt_cores.append(temp)
    # formally add new dimension at the end of the last core
    tt_cores[-1] = np.expand_dims(tt_cores[-1], axis=-1)
    return tt_cores


def tt_decompose_error(tensor_xyz, rel_error_norm):
    # function computes TT decomposition of the tensor_xyz with specified error - epsilon
    #
    dim = tensor_xyz.ndim
    shape = tensor_xyz.shape
    size = tensor_xyz.size
    temp = tensor_xyz

    #
    tt_cores = []
    delta = (rel_error_norm * np.linalg.norm(tensor_xyz)) / (np.sqrt(dim - 1))

    ranks = np.ones(dim + 1, dtype=int)
    ranks[0] = 1
    for d in np.arange(0, dim - 1):
        # shape of unfolding in d-step
        d_unfold_shape = np.array([ranks[d] * shape[d], temp.size / (ranks[d] * shape[d])], dtype=int)
        # unfolding --- reshape
        temp = np.reshape(temp, d_unfold_shape)

        # SVD decomposition
        [U, S, Vh] = np.linalg.svd(temp, full_matrices=False)  # TODO [martin]: svd is not normalized
        # singular_values = S;
        # U[:, 0:ranks[d + 1]]
        # S[0:ranks[d + 1], 0:ranks[d + 1]]
        # Vh[ 0:ranks[d + 1], :]
        # Compute the cumulative energy of singular values
        # np.cumsum(np.square(S) / np.sum(np.square(S)))
        accuracies = np.cumsum(np.square(S)[::-1])[::-1]

        # accuracies(accuracies <= delta ^ 2);
        ranks[d + 1] = np.asarray(np.size(S) - np.size(accuracies[accuracies <= delta ** 2]), dtype=int)

        # A_truncated = np.dot(U[:, 0:ranks[d + 1]] * S[0:ranks[d + 1]], Vh[0:ranks[d + 1], :])
        # assign to the d-th core
        d_core_shape = np.array([ranks[d], shape[d], ranks[d + 1]], dtype=int)
        tt_cores.append(np.reshape(U[:, 0:ranks[d + 1]], d_core_shape))

        #
        temp = np.matmul(np.diag(S[0:ranks[d + 1]]), Vh[0:ranks[d + 1], :])

    tt_cores.append(temp)
    # formally add new dimension at the end of the last core
    tt_cores[-1] = np.expand_dims(tt_cores[-1], axis=-1)
    return tt_cores, ranks


def tt_to_full_format(tt_cores):
    # This function reconstructs the full data tensor by using the given tensor core
    # We first multiplies the first two cores in mode-3 unfolding

    full_format = np.tensordot(tt_cores[0], tt_cores[1], axes=1)
    # The loop performs the product on the full_format_tensor with the next core and
    # iterates over until the last core to obtain the complete tensor
    #
    for d in np.arange(2, len(tt_cores)):
        full_format = np.tensordot(full_format, tt_cores[d], axes=1)

    return np.squeeze(full_format, axis=(0, -1))


def get_gradient_finite_difference(tt_cores, voxel_sizes):
    dim = len(tt_cores)
    dtt_cores = []

    for d in np.arange(dim):
        dtt_cores.append(tt_cores.copy())
        dtt_cores[d][d] = (np.roll(tt_cores[d], -1, axis=1) - tt_cores[d]) / voxel_sizes[d]

    return dtt_cores


def tt_addition(tt_cores_1, tt_cores_2):
    if len(tt_cores_1) != len(tt_cores_2):
        raise ValueError('Number of dimensions of TT cores 1 = {} is not equal to '
                         ' Number of dimensions of TT cores 2 = {}'.format(len(tt_cores_1), len(tt_cores_2)))
    #
    dim = len(tt_cores_1)
    # check if spatial dimension are correct for both tt_tensors
    for d in np.arange(0, dim):
        if tt_cores_1[d].shape[1] != tt_cores_2[d].shape[1]:
            raise ValueError('Spatial dimension of TT cores {} = {} is not equal to '
                             'Spatial dimension of TT cores {} = {}'.format(len(tt_cores_1),
                                                                            tt_cores_1[d].shape[1],
                                                                            len(tt_cores_2),
                                                                            tt_cores_2[d].shape[1]))
    # first core
    sum_tt_cores = []
    sum_tt_cores.append(np.concatenate((tt_cores_1[0], tt_cores_2[0]), axis=2))

    for d in np.arange(1, dim - 1):
        print(dim)
        # dtt_cores[d][d] = (np.roll(tt_cores[d], -1, axis=1) - tt_cores[d]) / voxel_sizes[d]
        # aaa=np.concatenate((tt_cores_1[1], tt_cores_2[1]), axis=1,2)
        r1_left = tt_cores_1[d].shape[0]
        r2_left = tt_cores_2[d].shape[0]

        r1_right = tt_cores_1[d].shape[2]
        r2_right = tt_cores_2[d].shape[2]
        nd = tt_cores_1[d].shape[1]
        temp = np.zeros([r1_left + r2_left, nd, r1_right + r2_right])

        temp[:r1_left, :, :r1_right] = tt_cores_1[d]
        temp[r1_left:r1_left + r2_left, :, r1_right:r1_right + r2_right] = tt_cores_2[d]

        sum_tt_cores.append(temp.copy())

    # last core
    sum_tt_cores.append(np.concatenate((tt_cores_1[-1], tt_cores_2[-1]), axis=0))

    return sum_tt_cores


def tt_subtraction(tt_cores_1, tt_cores_2):
    # dim = len(tt_cores_2)
    tt_cores_2[0] = np.copy(-1.0 * (tt_cores_2[0]))
    # for d in np.arange(0, dim):
    #     tt_cores_2[d] =np.copy( -1.0 * (tt_cores_2[d]))

    return tt_addition(tt_cores_1, tt_cores_2)



def tt_rounding_Bharat(cores, epsilon):
    d = len(cores)
    norm_A = [np.linalg.norm(c) for c in cores]
    delta = np.sqrt(epsilon / (d - 1)) * norm_A

    # Right-to-left orthogonalization
    for k in range(d - 1, 0, -1):
        C = cores[k].reshape(-1, cores[k].shape[-1])
        Q, R = np.linalg.qr(C)
        cores[k] = Q.reshape(cores[k].shape[0], cores[k].shape[1], -1)
        cores[k - 1] = np.tensordot(cores[k - 1], R, axes=([2], [1]))

    # Compression
    for k in range(d - 1):
        C = cores[k].reshape(-1, cores[k].shape[-1])
        U, S, Vh = np.linalg.svd(C, full_matrices=False)
        rk = np.sum(S > delta)
        cores[k] = U[:, :rk].reshape(cores[k].shape[0], cores[k].shape[1], rk)
        cores[k + 1] = np.tensordot(np.dot(np.diag(S[:rk]), Vh[:rk, :]), cores[k + 1], axes=([1], [0]))

    return cores


def tt_rounding_Martin(cores, epsilon):
    d = len(cores)
    norm_A=np.linalg.norm(tt_to_full_format(tt_cores=cores))
    #norm_A = np.sum[np.linalg.norm(c) for c in cores]
    delta = np.sqrt(epsilon / (d - 1)) * norm_A
    # Right-to-left orthogonalization
    for k in range(d - 1, 0, -1):
        C_unfolded = cores[k].reshape(cores[k].shape[0], np.prod(cores[k].shape[1:]))
        R, Q = sc.linalg.rq(C_unfolded, mode='economic')

        #G_k_new=Q.reshape(cores[k].shape)
        #cores[k - 1].reshape(np.prod(cores[k - 1].shape[:2]), cores[k - 1].shape[-1])
        aaa=np.dot(cores[k - 1].reshape(np.prod(cores[k - 1].shape[:2]), cores[k - 1].shape[-1]),R)
        cores[k-1]=aaa
        #reshape(cr[i - 1], (r[i - 1] * n[i - 1], r[i])
        #G_kminus=np.tensordot(G_k_new, R, axes=2)
        cores[k] = Q.reshape(cores[k].shape)

    return cores