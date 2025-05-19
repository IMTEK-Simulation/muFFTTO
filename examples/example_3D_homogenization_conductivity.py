import numpy as np
import time
# from netCDF4 import Dataset

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library

problem_type = 'conductivity'
discretization_type = 'finite_element'
element_type = 'trilinear_hexahedron'

domain_size = [1, 1, 1]
number_of_pixels = 3 * (32,)

geometry_ID = 'circle_inclusion'

# set up the system
my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)
start_time = time.time()

# create material data field
conductivity_C_1 = 100 * np.array([[1, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 1]])

material_data_field_C_0 = np.einsum('ij,qxyz->ijqxyz', conductivity_C_1,
                                    np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                      *discretization.nb_of_pixels])))

# material distribution
phase_field = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                  microstructure_name=geometry_ID,
                                                  coordinates=discretization.fft.coords)

microstructure_library.visualize_voxels(phase_field_xyz=phase_field)

# apply material distribution
material_data_field_C_0_rho = material_data_field_C_0[..., :, :, :] * np.power(phase_field, 1)
material_data_field_C_0_rho += 5 * material_data_field_C_0[..., :, :, :] * np.power(1 - phase_field, 1)

# linear system
K_fun = lambda x: discretization.apply_system_matrix(material_data_field=material_data_field_C_0_rho,
                                                     displacement_field=x)
preconditioner = discretization.get_preconditioner_NEW(reference_material_data_field_ijklqxyz=material_data_field_C_0)

# preconditioner
M_fun = lambda x: discretization.apply_preconditioner_NEW(preconditioner_Fourier_fnfnqks=preconditioner,
                                                          nodal_field_fnxyz=x)
# M_fun = lambda x: 1 * x

dim = discretization.domain_dimension
homogenized_A_ij = np.zeros(np.array(2 * [dim, ]))
# compute whole homogenized elastic tangent
for i in range(dim):
    # set macroscopic gradient
    macro_gradient = np.zeros([dim])
    macro_gradient[i] = 1
    # Set up right hand side
    macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient_ij=macro_gradient)

    # Solve equilibrium constrain
    rhs = discretization.get_rhs(material_data_field_ijklqxyz=material_data_field_C_0_rho,
                                 macro_gradient_field_ijqxyz=macro_gradient_field)
    # solver
    temperature_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(500), toler=1e-6)
    print('Number of CG steps = {}'.format(np.size(norms['residual_rz'])))

    # ----------------------------------------------------------------------
    # compute homogenized stress field corresponding to displacement
    homogenized_A_ij[i, :] = discretization.get_homogenized_stress(
        material_data_field_ijklqxyz=material_data_field_C_0_rho,
        displacement_field_inxyz=temperature_field,
        macro_gradient_field_ijqxyz=macro_gradient_field)

print('homogenized elastic tangent = \n {}'.format(homogenized_A_ij))
end_time = time.time()

microstructure_library.visualize_voxels(phase_field_xyz=temperature_field[0, 0])

grad_field = discretization.apply_gradient_operator(temperature_field)

microstructure_library.visualize_voxels(phase_field_xyz=grad_field.mean(axis=2)[0, 0])
microstructure_library.visualize_voxels(phase_field_xyz=grad_field.mean(axis=2)[0, 1])
microstructure_library.visualize_voxels(phase_field_xyz=grad_field.mean(axis=2)[0, 2])
