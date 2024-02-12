import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

from muFFTTO import microstructure_library
from muFFTTO import tensor_train_tools

from muFFTTO.tensor_train_tools import tt_decompose_error
from muFFTTO.tensor_train_tools import tt_to_full_format
from muFFTTO.tensor_train_tools import tt_decompose_rank
#from muFFTTO import tensor_train_tools
#from muFFTTO import microstructure_library
#from muFFTTO import domain

#from TT_tools import tt_decompose_error
#from TT_tools import tt_to_full_format
#from TT_tools import tt_decompose_rank

def tt_svd(A, epsilon):
    d = len(A.shape)  # Tensor dimension
    norm_A = np.linalg.norm(A, 'fro')
    delta = (epsilon /np.sqrt (d - 1)) * norm_A
    ranks = [1]
    cores = []

    C = A.copy()
    for k in range(d - 1):
        C = C.reshape(ranks[-1] * A.shape[k], -1)
        U, S, Vh = la.svd(C, full_matrices=False)
        rk = np.sum(S > delta)
        ranks.append(rk)

        U = U[:, :rk]
        C = np.dot(np.diag(S[:rk]), Vh[:rk, :])
        cores.append(U.reshape(ranks[-2], A.shape[k], rk))

    cores.append(C.reshape(ranks[-1], A.shape[-1], 1))
    return cores


def sum_tt_cores(tt_cores_list):
    """
    Sum corresponding cores from a list of tensors in TT-format.
    Assumes all tensors have the same shape and TT-rank.

    :param tt_cores_list: List of tensors in TT-format (list of lists of cores)
    :return: New set of TT-cores representing the element-wise sum of corresponding cores
    """
    # Check if all tensors have the same number of cores and compatible core shapes
    num_cores = len(tt_cores_list[0])
    for tt_cores in tt_cores_list:
        if len(tt_cores) != num_cores or any(tt_cores[i].shape != tt_cores_list[0][i].shape for i in range(num_cores)):
            raise ValueError("All tensors must have the same number of cores and compatible core shapes")

    # Sum corresponding cores
    summed_cores = []
    for core_idx in range(num_cores):
        sum_core = sum(tt_cores[core_idx] for tt_cores in tt_cores_list)
        summed_cores.append(sum_core)

    return summed_cores


# Example usage:
# Assuming tt_cores_tensor1, tt_cores_tensor2, ... are the TT cores of your tensors
# list_of_tt_cores = [tt_cores_tensor1, tt_cores_tensor2, ...]
# summed_cores = sum_tt_cores(list_of_tt_cores)


def tt_rounding(cores, epsilon):
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
        U, S, Vh = la.svd(C, full_matrices=False)
        rk = np.sum(S > delta)
        cores[k] = U[:, :rk].reshape(cores[k].shape[0], cores[k].shape[1], rk)
        cores[k + 1] = np.tensordot(np.dot(np.diag(S[:rk]), Vh[:rk, :]), cores[k + 1], axes=([1], [0]))

    return cores

def mat_by_vec_prod(tensor, matrix, k):
    shape = list(tensor.shape)
    shape[k] = matrix.shape[0]
    new_tensor = np.zeros(shape)
    for i in range(shape[k]):
        new_tensor[tuple(slice(None) if j != k else i for j in range(tensor.ndim))] = sum(
            tensor[tuple(slice(None) if j != k else m for j in range(tensor.ndim))] * matrix[i, m]
            for m in range(tensor.shape[k]))
    return new_tensor

def canonical_to_tt(U_list):
    d = len(U_list)
    cores = []
    for k in range(d):
        if k == 0:
            core = U_list[k][:, np.newaxis, :]
        elif k == d - 1:
            core = U_list[k][np.newaxis, :, :]
        else:
            core = np.kron(U_list[k], np.eye(U_list[k].shape[1]))
            core = core.reshape(U_list[k].shape[0], U_list[k].shape[1], -1)
        cores.append(core)
    return cores

def tensor_dot_product(tensor1, tensor2):
    """
    Perform dot product between two tensors.

    :param tensor1: First tensor.
    :param tensor2: Second tensor.
    :return: Resultant tensor after dot product.
    """
    # Check if the dimensions are suitable for dot product
    if tensor1.shape[-1] != tensor2.shape[-2]:
        raise ValueError("Shapes of tensors are not aligned for dot product.")

    return np.tensordot(tensor1, tensor2, axes=([-1], [-2]))


# Specify the dimensions of the matrix
l = 3
m = 3
n = 3

# Generate a random 2D matrix with values between 1 and 10
matrix = np.random.randint(1,10, size=(l,m,n))

# Print the generated matrix
print(matrix)

# Cores = tt_svd(matrix, 0.01)
# print("The result of TT_SVD is : ")
# print(Cores)

# Matrix = tt_to_full_format(Cores)
# print(Matrix)

Core = tt_decompose_error(matrix, 0.01)
print("The result of TT_decompose_error is : ")
print(Core)

# Matrix1 = tt_to_full_format(Core)
# print(Matrix1)

# cores1 = tt_rounding(Cores,0.01)
# print(cores1)

Core1 = tt_decompose_rank(matrix, [1,2,2,1])
print("The result of TT_decompose_rank is : ")
print(Core1)

# Matrix2 = tt_to_full_format(Core1)
# print(Matrix2)

# cores2 = tt_rounding(Core1,0.01)
# # print(cores2)
# microstructure_name='geometry_I_5_3D'
# nb_voxels = np.array([50,50,50])
# tensor = microstructure_library.get_geometry(nb_voxels=nb_voxels,
#                                              microstructure_name
#                                              =microstructure_name,
#                                              parameter=None)
#
# A=tt_svd(tensor, 0.01)
# tt_reconstructed_tensor = TT_tools.tt_to_full_format(A)
#
# tensor_norm = np.linalg.norm(tensor)
# rank = 50
# epsilon = 0.01
# memory=[]
# error=[]
# for i in range(1,rank):
#     ranks = (1, i, i, 1)
#     tt_tensor = TT_tools.tt_decompose_rank(tensor_xyz=tensor,
#                                            ranks=ranks)
#     # tt_tensor = tt_svd(tensor_xyz=tensor,ranks=ranks)
#     tt_reconstructed_tensor = TT_tools.tt_to_full_format(tt_cores=tt_tensor)
#     result=tensor - tt_reconstructed_tensor
#     Error = np.linalg.norm(tensor-tt_reconstructed_tensor)/np.linalg.norm(tensor)
#     error.append(Error)
#
#     Memory= (tt_tensor[0].size+tt_tensor[1].size+tt_tensor[2].size)/tensor.size
#     memory.append(Memory)
#
# # Create a figure
# fig = plt.figure()
#
# # Create the first subplot (Error vs Rank)
# plt.subplot(2, 1, 1)
# plt.plot(error)
# plt.xlabel('Rank')
# plt.ylabel('Reconstruction Error')
# plt.title('Error vs Rank in TT Decomposition of {}'.format(microstructure_name))
#
# # Create the second subplot (Memory vs Rank)
# plt.subplot(2, 1, 2)
# plt.plot(memory)
# plt.xlabel('Rank')
# plt.ylabel('Memory Consumed')
# plt.title('Memory vs Rank in TT Decomposition of {}'.format(microstructure_name))
#
# # Adjust layout to prevent overlap
# plt.tight_layout()
#
# # Show the plot
# plt.show()

microstructure_name1='geometry_I_5_3D'
microstructure_name2='geometry_I_4_3D'
nb_voxels = np.array([50,50,50])
tensor1 = microstructure_library.get_geometry(nb_voxels=nb_voxels,
                                             microstructure_name
                                             =microstructure_name1,
                                             parameter=None)
tensor2 = microstructure_library.get_geometry(nb_voxels=nb_voxels,
                                             microstructure_name
                                             =microstructure_name2,
                                             parameter=None)

# Step 1: Create example tensors
# tensor1 = np.random.rand(4, 4, 4)
# tensor2 = np.random.rand(4, 4, 4)

# Step 2: TT-rank Decomposition
tt_cores_tensor1 = tt_decompose_rank(tensor1, [1,30,30,1])
tt_cores_tensor2 = tt_decompose_rank(tensor2, [1,30,30,1])

# Step 3: Perform TT-rounding (Optional)
# rounded_cores_tensor1 = tt_rounding(tt_cores_tensor1, epsilon=0.1)
# rounded_cores_tensor2 = tt_rounding(tt_cores_tensor2, epsilon=0.1)

# Step 4: Sum the Tensor Cores
summed_cores = sum_tt_cores([tt_cores_tensor1, tt_cores_tensor2])

C=tt_rounding(summed_cores,0.01)

# Step 5: Convert Summed Cores Back to a Full Tensor
summed_tensor = tt_to_full_format(summed_cores)

print("Summed Tensor:\n", summed_tensor)
