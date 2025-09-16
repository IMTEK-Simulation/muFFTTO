import numpy as np
import time

from muGrid import ConvolutionOperator
from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'trilinear_hexahedron'
formulation = 'small_strain'

domain_size = [4, 3, 5]
number_of_pixels = (16, 16, 16)
geometry_ID = 'circle_inclusion' #'sine_wave_' #

my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)
start_time = time.time()

# set macroscopic gradient
macro_gradient = np.array([[0.0, 0, 0], [0., 0.0, 0], [0, 0, 1.0]])

# create material data field
K_0, G_0 = domain.get_bulk_and_shear_modulus(E=1, poison=0.2)

elastic_C_1 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                 K=K_0,
                                                 mu=G_0,
                                                 kind='linear')
print(domain.compute_Voigt_notation_4order(elastic_C_1))
material_data_field_C_0 = discretization.get_material_data_size_field(name='elastic_tensor')

material_data_field_C_0.s = np.einsum('ijkl,qxyz->ijklqxyz', elastic_C_1,
                                      np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                        *discretization.nb_of_pixels])))

# material distribution
phase_field = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                  microstructure_name=geometry_ID,
                                                  coordinates=discretization.fft.coords)
# apply material distribution
# material_data_field_C_0_rho = material_data_field_C_0[..., :, :, :] * np.power(phase_field+1, 1)
material_data_field_C_0.s = material_data_field_C_0.s[..., :, :, :] * np.power(phase_field, 1)
material_data_field_C_0.s += 5 * material_data_field_C_0.s[..., :, :, :] * np.power(1 - phase_field, 1)

# Set up right hand side
macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient_ij=macro_gradient,
                                                               macro_gradient_field_ijqxyz=macro_gradient_field)
# macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)

# Solve mechanical equilibrium constrain
rhs_field = discretization.get_unknown_size_field(name='rhs_field')
rhs = discretization.get_rhs(material_data_field_ijklqxyz=material_data_field_C_0,
                             macro_gradient_field_ijqxyz=macro_gradient_field,
                             rhs_inxyz=rhs_field)
# rhs = discretization.get_rhs(material_data_field_C_0_rho, macro_gradient_field)

K_fun = lambda x: discretization.apply_system_matrix(material_data_field_C_0, x,
                                                     formulation='small_strain')
# M_fun = lambda x: 1 * x

preconditioner = discretization.get_preconditioner_NEW(reference_material_data_ijkl=elastic_C_1)

M_fun = lambda x: discretization.apply_preconditioner_NEW(preconditioner_Fourier_fnfnqks=preconditioner,
                                                          nodal_field_fnxyz=x)

solution_field = discretization.get_unknown_size_field(name='solution')

solution_field.s, norms = solvers.PCG(K_fun, rhs.s, x0=None, P=M_fun, steps=int(500), toler=1e-6)
print('Number of CG steps = {}'.format(np.size(norms['residual_rz'])))
# ----------------------------------------------------------------------
# compute homogenized stress field corresponding to displacement
homogenized_stress = discretization.get_homogenized_stress(
    material_data_field_ijklqxyz=material_data_field_C_0,
    displacement_field_inxyz=solution_field,
    macro_gradient_field_ijqxyz=macro_gradient_field,
    formulation='small_strain')
print('homogenized_stress= ')

print(homogenized_stress)
print(domain.compute_Voigt_notation_2order(homogenized_stress))

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
        # macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)
        macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient_ij=macro_gradient,
                                                                       macro_gradient_field_ijqxyz=macro_gradient_field)

        # Solve mechanical equilibrium constrain
        # rhs_ij = discretization.get_rhs(material_data_field_C_0_rho, macro_gradient_field)
        rhs_field = discretization.get_rhs(material_data_field_ijklqxyz=material_data_field_C_0,
                                           macro_gradient_field_ijqxyz=macro_gradient_field,
                                           rhs_inxyz=rhs_field)

        solution_field.s, norms = solvers.PCG(K_fun, rhs_field.s, x0=None, P=M_fun, steps=int(500), toler=1e-12)
        print('Number of CG steps = {}'.format(np.size(norms['residual_rz'])))
        # ----------------------------------------------------------------------
        # compute homogenized stress field corresponding
        homogenized_C_ijkl[i, j] = discretization.get_homogenized_stress(
            material_data_field_ijklqxyz=material_data_field_C_0,
            displacement_field_inxyz=solution_field,
            macro_gradient_field_ijqxyz=macro_gradient_field,
            formulation='small_strain')

print('homogenized elastic tangent = \n {}'.format(domain.compute_Voigt_notation_4order(homogenized_C_ijkl)))
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)
