import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import time
from mpi4py import MPI

from muFFTTO import domain
from muFFTTO import microstructure_library
from muFFTTO import material_models
from muFFTTO import solvers

# Homogenization of 2D small-strain linear elasticity with a choice of the CG preconditioner:
#
#   'Green'         P = G,                       G = K(C_ref)^-1, reference C_ref = stiffness of the matrix
#   'Green_Jacobi'  P = D^-1/2 G_2muI D^-1/2,    D = diag(K), reference 2 mu I
#   'dual'          P = G K(C_ref C^-1 C_ref) G, the system matrix with the material data C_ref C^-1 C_ref between
#                                                two Green operators (C^-1 on the symmetric strains);
#                                                lambda_min(PK) >= 1, lambda_max(PK) <= contrast
#
# CG stops when the estimate of the energy error has dropped to 1e-6 of the initial one (delay 10).
#
#   python example_2D_homogenization_elasticity_preconditioners.py [Green | Green_Jacobi | dual]

preconditioner_type = sys.argv[1] if len(sys.argv) > 1 else 'dual'

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'
geometry_ID = 'square_inclusion'

domain_size = [1, 1]
number_of_pixels = (128, 128)

my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)
start_time = time.time()
dim = discretization.domain_dimension

# material: isotropic stiffness C_matrix in the matrix, mat_contrast * C_matrix in the inclusion
mat_contrast = 1e-3
K_0, G_0 = material_models.get_bulk_and_shear_modulus(E=1, poisson=0.2)
C_matrix = material_models.get_elastic_material_tensor(dim=dim, K=K_0, mu=G_0, kind='linear')

phase_field = discretization.get_scalar_field(name='phase_field')
phase_field.s[0, 0] = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                          microstructure_name=geometry_ID,
                                                          coordinates=discretization.fft.coords)
matrix_mask = phase_field.s[0, 0] > 0
inc_mask = phase_field.s[0, 0] == 0

material_data_field_C = discretization.get_material_data_size_field_mugrid(name='elastic_tensor')
material_data_field_C.s[..., matrix_mask] = C_matrix[..., np.newaxis, np.newaxis]
material_data_field_C.s[..., inc_mask] = mat_contrast * C_matrix[..., np.newaxis, np.newaxis]

# reference material of the Green operator of 'Green' and 'dual': the matrix phase
C_ref = C_matrix

# Mandel notation of the symmetric strains: (xx, yy, sqrt(2) xy); C^-1 is the inverse on this space
mandel_pairs = [(0, 0), (1, 1), (0, 1)]
mandel_weights = np.array([1.0, 1.0, np.sqrt(2.0)])


def to_mandel(C):
    """[2, 2, 2, 2, ...] -> [3, 3, ...]"""
    return np.array([[mandel_weights[a] * mandel_weights[b] * C[i, j, k, l] for b, (k, l) in enumerate(mandel_pairs)]
                     for a, (i, j) in enumerate(mandel_pairs)])


def from_mandel(M):
    """[3, 3, ...] -> [2, 2, 2, 2, ...] with both minor symmetries"""
    C = np.zeros((2, 2, 2, 2) + M.shape[2:])
    for a, (i, j) in enumerate(mandel_pairs):
        for b, (k, l) in enumerate(mandel_pairs):
            for ii, jj in {(i, j), (j, i)}:
                for kk, ll in {(k, l), (l, k)}:
                    C[ii, jj, kk, ll] = M[a, b] / (mandel_weights[a] * mandel_weights[b])
    return C


# material data of the dual preconditioner: C_ref C^-1 C_ref at every quadrature point
M_ref = to_mandel(C_ref)
M_C_inv = np.linalg.inv(np.moveaxis(to_mandel(material_data_field_C.s), (0, 1), (-2, -1)))
material_data_field_dual = discretization.get_material_data_size_field_mugrid(name='elastic_tensor_dual')
material_data_field_dual.s[...] = from_mandel(np.moveaxis(M_ref @ M_C_inv @ M_ref, (-2, -1), (0, 1)))


def K_fun(x, Ax):
    discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_C,
                                              input_field_inxyz=x,
                                              output_field_inxyz=Ax,
                                              formulation=formulation)
    discretization.fft.communicate_ghosts(Ax)


# Green operators; every Green operator on one discretization needs its own field name
green_ref = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=C_ref,
                                                           name='green_reference')
green_2muI = discretization.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=2 * G_0 * np.einsum('ik,jl->ijkl', np.eye(dim), np.eye(dim)),
    name='green_2muI')
jacobi = discretization.get_preconditioner_Jacobi_mugrid(material_data_field_ijklqxyz=material_data_field_C)
temp_1 = discretization.get_unknown_size_field(name='preconditioner_temp_1')
temp_2 = discretization.get_unknown_size_field(name='preconditioner_temp_2')


def apply_green(green, x, Px):
    discretization.fft.communicate_ghosts(x)
    discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green,
                                               input_nodal_field_fnxyz=x,
                                               output_nodal_field_fnxyz=Px)


def M_Green(x, Px):
    apply_green(green_ref, x, Px)


def M_Green_Jacobi(x, Px):
    temp_1.s[...] = jacobi.s * x.s
    apply_green(green_2muI, temp_1, Px)
    Px.s[...] *= jacobi.s


def M_dual(x, Px):
    apply_green(green_ref, x, temp_1)
    discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_dual,
                                              input_field_inxyz=temp_1,
                                              output_field_inxyz=temp_2,
                                              formulation=formulation)
    discretization.fft.communicate_ghosts(temp_2)
    apply_green(green_ref, temp_2, Px)


M_fun = {'Green': M_Green, 'Green_Jacobi': M_Green_Jacobi, 'dual': M_dual}[preconditioner_type]

# Allocate fields
macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
rhs_field = discretization.get_unknown_size_field(name='rhs_field')
solution_field = discretization.get_unknown_size_field(name='solution')

homogenized_C_ijkl = np.zeros(np.array(4 * [dim, ]))
# three symmetric load cases: E_xx, E_yy and the shear (E_xy = E_yx = 1/2), so that C : E = C[:, :, i, j]
for i, j in [(0, 0), (1, 1), (0, 1)]:
    macro_gradient_ij = np.zeros([dim, dim])
    macro_gradient_ij[i, j] = macro_gradient_ij[j, i] = 1 if i == j else 0.5
    discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient_ij,
                                                   macro_gradient_field_ijqxyz=macro_gradient_field)
    discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C,
                                  macro_gradient_field_ijqxyz=macro_gradient_field,
                                  rhs_inxyz=rhs_field)
    solution_field.s.fill(0)
    _, norms = solvers.conjugate_gradients_mugrid_experimental(
        comm=discretization.communicator,
        fc=discretization.field_collection,
        hessp=K_fun,
        b=rhs_field,
        x=solution_field,
        P=M_fun,
        tol=1e-6,
        maxiter=10000,
        stop_crit_norm='energy_lower_estim',  # estimate of the energy error ...
        energy_rtol=True,  # ... relative to the initial one
        d_min=10)
    if MPI.COMM_WORLD.rank == 0:
        print(f'load case E_{i}{j}: {preconditioner_type} CG iterations = '
              f'{norms["stop_iteration"]["energy_lower_estim"]}')
    homogenized_C_ijkl[:, :, i, j] = homogenized_C_ijkl[:, :, j, i] = discretization.get_homogenized_stress_mugrid(
        material_data_field_ijklqxyz=material_data_field_C,
        displacement_field_inxyz=solution_field,
        macro_gradient_field_ijqxyz=macro_gradient_field,
        formulation=formulation)

if MPI.COMM_WORLD.rank == 0:
    print('Homogenized elastic tangent (Voigt) =\n' +
          np.array2string(material_models.compute_Voigt_notation_4order(homogenized_C_ijkl),
                          formatter={'float_kind': lambda x: f'{x:0.8f}'}))
    print('Elapsed time: ', time.time() - start_time)
