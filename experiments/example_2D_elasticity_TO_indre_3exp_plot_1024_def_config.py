import os

import numpy as np
import scipy as sc
import time
from mpi4py import MPI
from NuMPI.IO import save_npy, load_npy

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library

# load the geometry:
for w_mult in [5.00]:  # np.arange(0.1, 1., 0.1):# [1]:
    for eta_mult in [0.01]:  # np.arange(0.05, 0.5, 0.05):#[0.1 ]:
        energy_objective = False
        print(w_mult, eta_mult)
        pixel_size = 0.0078125
        eta = 0.03125  # eta_mult * pixel_size
        N = 1024  # 512
        cores = 90  # 40
        p = 2
        nb_load_cases = 3
        random_initial_geometry = True
        bounds = False
        optimizer = 'lbfg'  # adam
        script_name = 'exp_2D_elasticity_TO_indre_3exp'
        E_target = 0.15
        poison_target = -0.5
        poison_0 = 0.0
        # name = (    f'{optimizer}_muFFTTO_elasticity_{script_name}_N{N}_E_target_0.15_Poisson_-0.5_Poisson0_0.2_w{w_mult}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}')
        name = (
            f'{script_name}_N{N}_Et_{E_target}_Pt_{poison_target}_P0_{poison_0}_w{w_mult}_eta{eta_mult}_p{p}_mpi{cores}_nlc_{nb_load_cases}_e_{energy_objective}')
        # xopt_it = np.load(os.path.expanduser('~/exp_data/' + name + f'_it{1 + 1}.npy'), allow_pickle=True)
        phase_field = np.load(os.path.expanduser('~/exp_data/' + name + f'_it{8740}.npy'), allow_pickle=True)


problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'


domain_size = [1, 1]
number_of_pixels = (1024, 1024)

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
#K_0, G_0 = 1, 0.5 #domain.get_bulk_and_shear_modulus(E=1, poison=0.2)
E_0 = 1
poison_0 = 0.0
G_0 = E_0 / (2 * (1 + poison_0))
K_0, G_0 = domain.get_bulk_and_shear_modulus(E=E_0, poison=poison_0)

elastic_C_0 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                 K=K_0,
                                                 mu=G_0,
                                                 kind='linear')

material_data_field_C_0 = np.einsum('ijkl,qxy->ijklqxy', elastic_C_0,
                                    np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                      *discretization.nb_of_pixels])))

print('elastic tangent = \n {}'.format(domain.compute_Voigt_notation_4order(elastic_C_0)))

# material distribution
geometry_ID = 'square_inclusion'
phase_field_smooth = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                  microstructure_name=geometry_ID,
                                                  coordinates=discretization.fft.coords)



phase_field=discretization.get_scalar_sized_field()
#phase_field[0,0]= phase_field_l
phase_field[0,0]= phase_field_smooth

#phase_field[0,0]=phase_field[0,0]/np.min(phase_field[0,0])

#np.save('geometry_jacobi.npy', np.power(phase_field_l, 2),)
#sc.io.savemat('geometry_jacobi.mat', {'data':  np.power(phase_field_l, 2)})

phase_field_at_quad_poits_1qnxyz = \
                    discretization.evaluate_field_at_quad_points(nodal_field_fnxyz=phase_field,
                                                                 quad_field_fqnxyz=None,
                                                                 quad_points_coords_iq=None)[0]

# apply material distribution
#material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * np.power(phase_field[0, 0], 1)
#material_data_field_C_0_rho=material_data_field_C_0[..., :, :] * phase_fem
#material_data_field_C_0_rho +=100*material_data_field_C_0[..., :, :] * (1-phase_fem)
material_data_field_C_0_rho = material_data_field_C_0[..., :, :, :] * np.power(
                    phase_field_at_quad_poits_1qnxyz, 2)[0, :, 0, ...]

# Set up right hand side
macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)



# Solve mechanical equilibrium constrain
rhs = discretization.get_rhs(material_data_field_C_0_rho, macro_gradient_field)

K_fun = lambda x: discretization.apply_system_matrix(material_data_field_C_0_rho, x,
                                                     formulation='small_strain')
# M_fun = lambda x: 1 * x
#K= discretization.get_system_matrix(material_data_field=material_data_field_C_0_rho)

preconditioner = discretization.get_preconditioner_NEW(
    reference_material_data_field_ijklqxyz=material_data_field_C_0)

M_fun = lambda x: discretization.apply_preconditioner_NEW(preconditioner_Fourier_fnfnqks=preconditioner,
                                                      nodal_field_fnxyz=x)


K_diag_alg = discretization.get_preconditioner_Jacoby_fast(
    material_data_field_ijklqxyz=material_data_field_C_0_rho)

M_fun = lambda x: K_diag_alg * discretization.apply_preconditioner_NEW(
                       preconditioner_Fourier_fnfnqks=preconditioner,
                        nodal_field_fnxyz=K_diag_alg * x)
# #
#M_fun = lambda x: K_diag_alg *  K_diag_alg * x

displacement_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(1000), toler=1e-6)
nb_it_comb = len(norms['residual_rz'])
norm_rz = norms['residual_rz'][-1]
norm_rr = norms['residual_rr'][-1]


print(
    '   nb_ steps CG of =' f'{nb_it_comb}, residual_rz = {norm_rz}, residual_rr = {norm_rr}')
#print(norms)
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

        displacement_field_ij, norms = solvers.PCG(K_fun, rhs_ij, x0=None, P=M_fun, steps=int(500), toler=1e-8)

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
