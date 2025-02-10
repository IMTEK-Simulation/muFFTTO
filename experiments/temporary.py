import numpy as np
import scipy as sc
dim=3
knot_vectors_=np.array([[0,1,2,3],[0,1,2,3]])
knot_vector_v=np.array([0,1,2,3])

#knot_vectors=np.zeros([dim,Nx,Ny,Nz])


knot_matrix=np.zeros([dim,*knot_vectors_.shape])
knot_matrix_=np.asarray(np.meshgrid(knot_vector_u,knot_vector_v))

nb_control_points=[5,8]
control_points=np.random.random([dim,*nb_control_points])

nb_eval_points=[10,10]
eval_poits=np.zeros([dim,*nb_eval_points])

def eval_basis_at_single_point(eval_point,knot_matrix, p=1,q=2):
    # eval_point is one point in which we evaluate all basis
    basis_at_eval_point = np.zeros([1,*knot_matrix.shape[1:]])

    # write function

    return basis_at_eval_point

S=np.zeros([dim,*nb_eval_points])

for this_point in eval_poits.size:
    basis_at_point=eval_basis_at_single_point(eval_point=eval_poits[1,1],
                      knot_matrix=knot_matrix_, p=1,q=2) # this_point

    S[:,this_point] = basis_at_point*control_points

#surface_points=Basis*nb_eval_points



K_loc=np.array([[4.33376667, 1.16678333], [1.16678333, 4.33376667]])




# Eigendecomposition of M
eigenvalues, eigenvectors = sc.linalg.eigh(K_loc)

# Compute M^(-1/2)
K_loc_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T


# Symmetrically precondition A and b
A_tilde = K_loc_sqrt @ K_loc @ K_loc_sqrt


print(A_tilde)