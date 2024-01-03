import numpy as np


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

    return np.squeeze(full_format, axis=0)
