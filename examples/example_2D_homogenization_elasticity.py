import numpy as np
import scipy as sc
import time
from mpi4py import MPI
from NuMPI.IO import save_npy, load_npy

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'

domain_size = [1, 1]
number_of_pixels = (64, 64)

my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)
start_time = time.time()
print(f'{MPI.COMM_WORLD.rank:6} {MPI.COMM_WORLD.size:6} {str(discretization.fft.nb_domain_grid_pts):>15} '
      f'{str(discretization.fft.nb_subdomain_grid_pts):>15} {str(discretization.fft.subdomain_locations):>15}')
# set macroscopic gradient
macro_gradient = np.array([[1.0, 0], [0, 0.0]])

# create material data field
# K_0, G_0 = 1, 0.5 #domain.get_bulk_and_shear_modulus(E=1, poison=0.2)
K_0, G_0 = domain.get_bulk_and_shear_modulus(E=1, poison=0.0)

elastic_C_1 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                 K=K_0,
                                                 mu=G_0,
                                                 kind='linear')

material_data_field_C_0 = discretization.get_material_data_size_field(name='elastic_tensor')

material_data_field_C_0.s = np.einsum('ijkl,qxy->ijklqxy', elastic_C_1,
                                      np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                        *discretization.nb_of_pixels])))

print('elastic tangent = \n {}'.format(domain.compute_Voigt_notation_4order(elastic_C_1)))

# material distribution
geometry_ID = 'square_inclusion'
phase_field_smooth = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                         microstructure_name=geometry_ID,
                                                         coordinates=discretization.fft.coords)
# folder_name = 'experiments/exp_data/'  # s'exp_data/'
#
# #phase_field = np.random.rand(*discretization.get_scalar_sized_field().shape)  # set random distribution#
# #phase_field_l = np.load('../experiments/exp_data/lbfg_muFFTTO_elasticity_exp_2D_elasticity_TO_indre_3exp_N32_E_target_0.15_Poisson_-0.5_Poisson0_0.0_w4.0_eta0.0203_p2_bounds=False_FE_NuMPI6_nb_load_cases_3_energy_objective_False_random_True_it20.npy', allow_pickle=True)
# #phase_field_l = np.load('../experiments/exp_data/exp_2D_elasticity_TO_indre_3exp_N1024_Et_0.15_Pt_-0.5_P0_0.0_w5.0_eta0.01_p2_mpi90_nlc_3_e_False_it6398.npy', allow_pickle=True)
# phase_field_l = load_npy('experiments/exp_data/exp_2D_elasticity_TO_indre_3exp_N1024_Et_0.15_Pt_-0.5_P0_0.0_w5.0_eta0.01_p2_mpi90_nlc_3_e_False_it6398.npy',
#                      tuple(discretization.fft.subdomain_locations),
#                      tuple(discretization.nb_of_pixels), MPI.COMM_WORLD)

# phase = 1 * np.ones(number_of_pixels)
inc_contrast = 0.
# phase_field =  0.5*np.random.rand(*number_of_pixels)
# phase[10:30, 10:30] = phase[10:30, 10:30] * inc_contrast
# Square inclusion with: Obsonov solution
# phase[phase.shape[0] * 1 // 4:phase.shape[0] * 3 // 4,
# phase.shape[1] * 1 // 4:phase.shape[1] * 3 // 4] *= inc_contrast

# phase_fem = np.zeros([2, *number_of_pixels])
# phase_fem[:] = phase_field_l

phase_field = discretization.get_scalar_field(name='phase_field')
# phase_field[0,0]= phase_field_l
phase_field.s[0, 0] = phase_field_smooth

# phase_field[0,0]=phase_field[0,0]/np.min(phase_field[0,0])

# np.save('geometry_jacobi.npy', np.power(phase_field_l, 2),)
# sc.io.savemat('geometry_jacobi.mat', {'data':  np.power(phase_field_l, 2)})

phase_field_at_quad_poits_1qnxyz = \
    discretization.evaluate_field_at_quad_points(nodal_field_fnxyz=phase_field,
                                                 quad_field_fqnxyz=None,
                                                 quad_points_coords_iq=None)[0]

# apply material distribution
# material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * np.power(phase_field[0, 0], 1)
# material_data_field_C_0_rho=material_data_field_C_0[..., :, :] * phase_fem
# material_data_field_C_0_rho +=100*material_data_field_C_0[..., :, :] * (1-phase_fem)
# material_data_field_C_0.s = material_data_field_C_0.s[..., :, :, :] * np.power(
#     phase_field_at_quad_poits_1qnxyz, 1)[0, :, 0, ...]
material_data_field_C_0.s = material_data_field_C_0.s[..., :, :, :] * phase_field.s[0, 0]

# Set up right hand side
macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient_ij=macro_gradient,
                                                               macro_gradient_field_ijqxyz=macro_gradient_field)

# Solve mechanical equilibrium constrain
rhs_field = discretization.get_unknown_size_field(name='rhs_field')
rhs = discretization.get_rhs(material_data_field_ijklqxyz=material_data_field_C_0,
                             macro_gradient_field_ijqxyz=macro_gradient_field,
                             rhs_inxyz=rhs_field)
# rhs = discretization.get_rhs(material_data_field_C_0_rho, macro_gradient_field)

K_fun = lambda x: discretization.apply_system_matrix(material_data_field_C_0, x,
                                                     formulation='small_strain')
# K= discretization.get_system_matrix(material_data_field=material_data_field_C_0_rho)

preconditioner = discretization.get_preconditioner_Green_fast(reference_material_data_ijkl=elastic_C_1)

M_fun = lambda x: discretization.apply_preconditioner_NEW(preconditioner_Fourier_fnfnqks=preconditioner,
                                                          nodal_field_fnxyz=x)

# K_mat = discretization.get_system_matrix(material_data_field=material_data_field_C_0)

K_diag_alg = discretization.get_preconditioner_Jacoby_fast(
    material_data_field_ijklqxyz=material_data_field_C_0)

M_fun = lambda x: K_diag_alg * discretization.apply_preconditioner_NEW(
    preconditioner_Fourier_fnfnqks=preconditioner,
    nodal_field_fnxyz=K_diag_alg * x)
#
M_fun = lambda x: 1 * x


solution_field = discretization.get_unknown_size_field(name='solution')

solution_field.s, norms = solvers.PCG(K_fun, rhs.s, x0=None, P=M_fun, steps=int(1000), toler=1e-6)
nb_it_comb = len(norms['residual_rz'])
norm_rz = norms['residual_rz'][-1]
norm_rr = norms['residual_rr'][-1]
print(
    '   nb_ steps CG of =' f'{nb_it_comb}, residual_rz = {norm_rz}, residual_rr = {norm_rr}')
print(' residual_rr = {} \n '.format(norms['residual_rr']))
print(' residual_rz = {} \n '.format(norms['residual_rz']))
# print(norms)
# ----------------------------------------------------------------------
# compute homogenized stress field corresponding to displacement
homogenized_stress = discretization.get_homogenized_stress(
    material_data_field_ijklqxyz=material_data_field_C_0,
    displacement_field_inxyz=solution_field,
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
        #macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)
        macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient_ij=macro_gradient,
                                                                       macro_gradient_field_ijqxyz=macro_gradient_field)
        # Solve mechanical equilibrium constrain
        rhs_field = discretization.get_rhs(material_data_field_ijklqxyz=material_data_field_C_0,
                                     macro_gradient_field_ijqxyz=macro_gradient_field,
                                     rhs_inxyz=rhs_field)
        #rhs_ij = discretization.get_rhs(material_data_field_C_0_rh, macro_gradient_field)

        solution_field.s, norms = solvers.PCG(K_fun, rhs_field.s, x0=None, P=M_fun, steps=int(500), toler=1e-8)

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
