import numpy as np
import time

from muFFTTO import domain
from muFFTTO import solvers

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'

domain_size = [4, 3]
number_of_pixels = (24, 24)

my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       number_of_pixels=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)
start_time = time.time()

# set macroscopic gradient
macro_gradient = np.array([[0.2, 0], [0, 0.2]])

# create material data field
K_0, G_0 = domain.get_bulk_and_shear_modulus(E=1, poison=0.2)

elastic_C_1 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                 K=K_0,
                                                 mu=G_0,
                                                 kind='linear')

material_data_field_C_0 = np.einsum('ijkl,qxy->ijklqxy', elastic_C_1,
                                    np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                      *discretization.nb_of_pixels])))

print('elastic tangent = \n {}'.format(domain.compute_Voigt_notation_4order(elastic_C_1)))

# material distribution

phase_field = np.random.rand(*discretization.get_scalar_sized_field().shape)  # set random distribution

# apply material distribution
material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * np.power(phase_field[0, 0],
                                                                            1)
# Set up right hand side
macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)

# Solve mechanical equilibrium constrain
rhs = discretization.get_rhs(material_data_field_C_0_rho, macro_gradient_field)

K_fun = lambda x: discretization.apply_system_matrix(material_data_field_C_0_rho, x,
                                                     formulation='small_strain')
# M_fun = lambda x: 1 * x

preconditioner = discretization.get_preconditioner(
    reference_material_data_field_ijklqxyz=material_data_field_C_0)

M_fun = lambda x: discretization.apply_preconditioner(preconditioner_Fourier_fnfnxyz=preconditioner,
                                                      nodal_field_fnxyz=x)

displacement_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(500), toler=1e-12)

# ----------------------------------------------------------------------
# compute homogenized stress field corresponding to displacement
homogenized_stress = discretization.get_homogenized_stress(
    material_data_field_ijklqxyz=material_data_field_C_0_rho,
    displacement_field_fnxyz=displacement_field,
    macro_gradient_field_ijqxyz=macro_gradient_field,
    formulation='small_strain')

print('homogenized stress = \n {}'.format(homogenized_stress))
print('homogenized stress in Voigt notation = \n {}'.format(domain.compute_Voigt_notation_2order(homogenized_stress)))

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)

start_time = time.time()
dim = discretization.domain_dimension
homogenized_C_ijkl = np.zeros(np.array(4 * [dim, ]))
# compute whole homogenized elastic tangent
for i in range(dim):
    for j in range(dim):
        # set macroscopic gradient
        macro_gradient = np.zeros([dim, dim])
        macro_gradient[i, j] = 1
        # Set up right hand side
        macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)

        # Solve mechanical equilibrium constrain
        rhs_ij = discretization.get_rhs(material_data_field_C_0_rho, macro_gradient_field)

        displacement_field_ij, norms = solvers.PCG(K_fun, rhs_ij, x0=None, P=M_fun, steps=int(500), toler=1e-12)

        # ----------------------------------------------------------------------
        # compute homogenized stress field corresponding
        homogenized_C_ijkl[i, j] = discretization.get_homogenized_stress(
            material_data_field_ijklqxyz=material_data_field_C_0_rho,
            displacement_field_fnxyz=displacement_field_ij,
            macro_gradient_field_ijqxyz=macro_gradient_field,
            formulation='small_strain')

print('homogenized elastic tangent = \n {}'.format(domain.compute_Voigt_notation_4order(homogenized_C_ijkl)))
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)
