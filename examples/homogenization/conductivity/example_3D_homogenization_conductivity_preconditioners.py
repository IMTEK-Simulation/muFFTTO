import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from mpi4py import MPI
import numpy as np
import time

from muFFTTO import domain
from muFFTTO import microstructure_library
from muFFTTO import solvers

# Homogenization of 3D heat conductivity with a choice of the CG preconditioner:
#
#   'Green'         P = G,                       G = K(A_ref)^-1, reference A_ref = conductivity of the matrix
#   'Green_Jacobi'  P = D^-1/2 G_I D^-1/2,       D = diag(K), reference I
#   'dual'          P = G K(A_ref A^-1 A_ref) G, the system matrix with the material data A_ref A^-1 A_ref between
#                                                two Green operators; lambda_min(PK) >= 1, lambda_max(PK) <= contrast
#
# CG stops when the estimate of the energy error has dropped to 1e-6 of the initial one (delay 10).
#
#   python example_3D_homogenization_conductivity_preconditioners.py [Green | Green_Jacobi | dual]

preconditioner_type = sys.argv[1] if len(sys.argv) > 1 else 'dual'

problem_type = 'conductivity'
discretization_type = 'finite_element'
element_type = 'trilinear_hexahedron'
geometry_ID = 'square_inclusion'

domain_size = [1, 1, 1]
number_of_pixels = 3 * (32,)

my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)
start_time = time.time()
dim = discretization.domain_dimension

# material: conductivity A_matrix in the matrix, mat_contrast * A_matrix in the inclusion
mat_contrast = 1e-3
conductivity_matrix = np.eye(3)

phase_field = discretization.get_scalar_field(name='phase_field')
phase_field.s[0, 0] = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                          microstructure_name=geometry_ID,
                                                          coordinates=discretization.fft.coords)
matrix_mask = phase_field.s[0, 0] > 0
inc_mask = phase_field.s[0, 0] == 0

material_data_field_A = discretization.get_material_data_size_field_mugrid(name='conductivity_tensor')
material_data_field_A.s[..., matrix_mask] = conductivity_matrix[:, :, np.newaxis, np.newaxis]
material_data_field_A.s[..., inc_mask] = mat_contrast * conductivity_matrix[:, :, np.newaxis, np.newaxis]

# reference material of the Green operator of 'Green' and 'dual': the matrix phase
conductivity_ref = conductivity_matrix

# material data of the dual preconditioner: A_ref A^-1 A_ref at every quadrature point
material_data_field_dual = discretization.get_material_data_size_field_mugrid(name='conductivity_tensor_dual')
A_inv = np.linalg.inv(np.moveaxis(material_data_field_A.s, (0, 1), (-2, -1)))
material_data_field_dual.s[...] = np.moveaxis(conductivity_ref @ A_inv @ conductivity_ref, (-2, -1), (0, 1))


def K_fun(x, Ax):
    discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_A,
                                              input_field_inxyz=x,
                                              output_field_inxyz=Ax)
    discretization.fft.communicate_ghosts(Ax)


# Green operators; every Green operator on one discretization needs its own field name
green_ref = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=conductivity_ref,
                                                           name='green_reference')
green_identity = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=np.eye(dim),
                                                                name='green_identity')
jacobi = discretization.get_preconditioner_Jacobi_mugrid(material_data_field_ijklqxyz=material_data_field_A)
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
    apply_green(green_identity, temp_1, Px)
    Px.s[...] *= jacobi.s


def M_dual(x, Px):
    apply_green(green_ref, x, temp_1)
    discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_dual,
                                              input_field_inxyz=temp_1,
                                              output_field_inxyz=temp_2)
    discretization.fft.communicate_ghosts(temp_2)
    apply_green(green_ref, temp_2, Px)


M_fun = {'Green': M_Green, 'Green_Jacobi': M_Green_Jacobi, 'dual': M_dual}[preconditioner_type]

# Allocate fields
macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
rhs_field = discretization.get_unknown_size_field(name='rhs_field')
solution_field = discretization.get_unknown_size_field(name='solution')

homogenized_A_ij = np.zeros([dim, dim])
for i in range(dim):
    macro_gradient = np.zeros(dim)
    macro_gradient[i] = 1
    discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient,
                                                   macro_gradient_field_ijqxyz=macro_gradient_field)
    discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_A,
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
        print(f'load case {i}: {preconditioner_type} CG iterations = '
              f'{norms["stop_iteration"]["energy_lower_estim"]}')
    homogenized_A_ij[:, i] = discretization.get_homogenized_stress_mugrid(
        material_data_field_ijklqxyz=material_data_field_A,
        displacement_field_inxyz=solution_field,
        macro_gradient_field_ijqxyz=macro_gradient_field)

if MPI.COMM_WORLD.rank == 0:
    print('Homogenized conductivity tensor =\n' +
          np.array2string(homogenized_A_ij, formatter={'float_kind': lambda x: f'{x:0.8f}'}))
    print('Elapsed time: ', time.time() - start_time)
