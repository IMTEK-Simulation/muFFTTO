#import pytest

import sys

sys.path.append("/home/martin/Programming/muFFTTO_paralellFFT_test/muFFTTO")
sys.path.append('../..')  # Add parent directory to path
import numpy as np
np.set_printoptions(precision=4)  # 3 decimal places

import scipy as sc
import time
from mpi4py import MPI

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import topology_optimization
from muFFTTO import microstructure_library


# # TODO implement test for bilinear elements and 3D
# @pytest.fixture()
# def discretization(domain_size, element_type, nb_pixels):
#     problem_type = 'elasticity'
#     element_types = ['linear_triangles', ]  # ,'linear_triangles_tilled',  'bilinear_rectangle'
#
#     my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
#                                       problem_type=problem_type)
#
#     discretization_type = 'finite_element'
#
#     discretization = domain.Discretization(cell=my_cell,
#                                            nb_of_pixels_global=nb_pixels,
#                                            discretization_type=discretization_type,
#                                            element_type=element_types[element_type])
#
#     return discretization
#
#
# @pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
#     ([2, 2], 0, [2, 2]),
#     ([2, 3], 0, [2, 3]),
#     ([2, 4], 0, [2, 4]),
#     ([3, 2], 0, [3, 2]),
#     ([3, 3], 0, [3, 3]),
#     ([3, 4], 0, [3, 4]),
#     ([4, 2], 0, [4, 2]),
#     ([4, 3], 0, [4, 3]),
#     ([4, 4], 0, [4, 4])
#     # ,([2, 2], 1, [2, 2]),
#     # ([2, 3], 1, [2, 3]),
#     # ([2, 4], 1, [2, 4]),
#     # ([3, 2], 1, [3, 2]),
#     # ([3, 3], 1, [3, 3]),
#     # ([3, 4], 1, [3, 4]),
#     # ([4, 2], 1, [4, 2]),
#     # ([4, 3], 1, [4, 3]),
#     # ([4, 4], 1, [4, 4])
# ])
# def test_discretization_init(discretization):
#     print(discretization.domain_size)
#     assert hasattr(discretization, "cell")
#     assert hasattr(discretization, "domain_dimension")
#     assert hasattr(discretization, "B_gradient")
#     assert hasattr(discretization, "quadrature_weights")
#     assert hasattr(discretization, "nb_quad_points_per_pixel")
#     assert hasattr(discretization, "nb_nodes_per_pixel")
#
#
# @pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
#     ([1, 2], 0, [4, 5])])
# # ,([3.1, 6.4], 0, [7, 6])])
# def test_fd_check_of_whole_objective_function(discretization, plot=True):
problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'  # discretization.element_type  # 'bilinear_rectangle'##'linear_triangles' #
formulation = 'small_strain'
preconditioner_type = 'Green_Jacobi'

domain_size = [1, 1]
number_of_pixels = (4,5)



my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=tuple(number_of_pixels),
                                       discretization_type=discretization_type,
                                       element_type=element_type)

macro_gradient = np.array([[1.0, .0],
                           [0.0, 0.]])
print('macro_gradient = \n {}'.format(macro_gradient))

# create material data of solid phase rho=1
E_0 = 1
poison_0 = 0.2

K_0, G_0 = domain.get_bulk_and_shear_modulus(E=E_0, poison=poison_0)

elastic_C_0_ijkl = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                      K=K_0,
                                                      mu=G_0,
                                                      kind='linear')
stress = np.einsum('ijkl,lk->ij', elastic_C_0_ijkl, macro_gradient)

# create target material data
print('init_stress = \n {}'.format(stress))
# validation metamaterials

poison_target = 1 / 3  # lambda = -10
G_target_auxet = (1 / 4) * E_0  # 23   25
E_target = 2 * G_target_auxet * (1 + poison_target)
K_targer, G_target = domain.get_bulk_and_shear_modulus(E=E_target, poison=poison_target)
elastic_C_target_ijkl = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                           K=K_targer,
                                                           mu=G_target,
                                                           kind='linear')

target_stress_ij = np.einsum('ijkl,lk->ij', elastic_C_target_ijkl, macro_gradient)
print('target_stress = \n {}'.format(target_stress_ij))
# Set up the equilibrium system

macro_gradient_field_ijqxyz = discretization.get_gradient_size_field(name='macro_gradient_field')
discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient,
                                               macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz
                                               )

# M_fun = lambda x: 1 * x
# preconditioner_Green = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=elastic_C_0_ijkl)
# #
# def M_fun(x, Px):
#     """
#     Function to compute the product of the Preconditioner matrix with a vector.
#     The Preconditioner is represented by the convolution operator.
#     """
#     discretization.fft.communicate_ghosts(x)
#     discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner_Green,
#                                                input_nodal_field_fnxyz=x,
#                                                output_nodal_field_fnxyz=Px)
def M_fun(x, Px):
    Px.s = 1 * x.s

p = 2
w = 3  # * E_0  # 1 / 10  # 1e-4 Young modulus of solid
eta = 0.02
cg_setup = {'cg_tol': 1e-6}

def my_objective_function(phase_field_1nxyz_flat):
    # print('Objective function:')
    # reshape the field
    phase_field_1nxyz = discretization.get_scalar_field(name='phase_field_in_objective')
    phase_field_1nxyz.s = phase_field_1nxyz_flat.reshape([1, 1, *discretization.nb_of_pixels])

    # Phase field  in quadrature points
    phase_field_at_quad_poits_1qxyz = discretization.get_quad_field_scalar(
        name='phase_field_at_quads_in_objective_function_multiple_load_cases')
    discretization.apply_N_operator_mugrid(phase_field_1nxyz, phase_field_at_quad_poits_1qxyz)

    # Material data in quadrature points
    material_data_field_C_0_rho_ijklqxyz = discretization.get_material_data_size_field_mugrid(
        name='material_data_field_C_0_rho_ijklqxyz_in_objective')
    material_data_field_C_0_rho_ijklqxyz.s = elastic_C_0_ijkl[..., np.newaxis, np.newaxis, np.newaxis] * \
                                             np.power(phase_field_at_quad_poits_1qxyz.s, p)[0, 0, :, ...]

    f_phase_field = topology_optimization.objective_function_phase_field(discretization=discretization,
                                                                         phase_field_1nxyz=phase_field_1nxyz,
                                                                         eta=eta,
                                                                         double_well_depth=1)
    print(f'rank = {MPI.COMM_WORLD.rank}' + 'f_phase_field= '          ' {} '.format(f_phase_field))

    #  sensitivity phase field terms
    s_phase_field = discretization.get_scalar_field(name='s_phase_field')
    s_phase_field.s[0,0] = topology_optimization.sensitivity_phase_field_term_FE_NEW(discretization=discretization,
                                                                              base_material_data_ijkl=elastic_C_0_ijkl,
                                                                              phase_field_1nxyz=phase_field_1nxyz,
                                                                              p=p,
                                                                              eta=eta,
                                                                              double_well_depth=1)
    print(f'rank = {MPI.COMM_WORLD.rank}' + 's_phase_field= \n'          ' {} '.format(s_phase_field.s))



    # if preconditioner_type == 'Green_Jacobi':
    #     K_diag_alg = discretization.get_preconditioner_Jacobi_mugrid(
    #         material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz)
    #
    #     def M_fun_Green_Jacobi(x, Px):
    #         discretization.fft.communicate_ghosts(x)
    #         x_jacobi_temp = discretization.get_unknown_size_field(name='x_jacobi_temp')
    #
    #         x_jacobi_temp.s = K_diag_alg.s * x.s
    #         discretization.apply_preconditioner_mugrid(
    #             preconditioner_Fourier_fnfnqks=preconditioner_Green,
    #             input_nodal_field_fnxyz=x_jacobi_temp,
    #             output_nodal_field_fnxyz=Px)
    #
    #         Px.s = K_diag_alg.s * Px.s
    #         discretization.fft.communicate_ghosts(Px)
    #
    #     M_fun = M_fun_Green_Jacobi

    # Solve mechanical equilibrium constrain
    def K_fun(x, Ax):
        discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0_rho_ijklqxyz,
                                                  input_field_inxyz=x,
                                                  output_field_inxyz=Ax,
                                                  formulation='small_strain')

    # mechanical equilibrium rhs
    rhs_inxyz = discretization.get_unknown_size_field(name='rhs_field')
    discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                                  macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
                                  rhs_inxyz=rhs_inxyz)

    displacement_field = discretization.get_unknown_size_field(name='displacement_field_')
    displacement_field.s.fill(0)

    solvers.conjugate_gradients_mugrid(
        comm=discretization.fft.communicator,
        fc=discretization.field_collection,
        hessp=K_fun,  # linear operator
        b=rhs_inxyz,
        x=displacement_field,
        P=M_fun,
        tol=cg_setup['cg_tol'],
        maxiter=10000,
        # callback=callback,
        # norm_metric=res_norm
    )
    # compute homogenized stress field corresponding t
    homogenized_stress = discretization.get_homogenized_stress_mugrid(
        material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
        displacement_field_inxyz=displacement_field,
        macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
        formulation='small_strain')
    # print('homogenized stress = \n'          ' {} '.format(homogenized_stress))
    print(f'rank = {MPI.COMM_WORLD.rank}' + 'homogenized_stress= \n'          ' {} '.format(homogenized_stress))


    f_sigma = topology_optimization.compute_stress_equivalence_potential(
        actual_stress_ij=homogenized_stress,
        target_stress_ij=target_stress_ij)
    # print('objective_function= \n'' {} '.format(objective_function))
    print(f'rank = {MPI.COMM_WORLD.rank}' + 'f_sigma= \n'          ' {} '.format(f_sigma))
    # print('Sensitivity_analytical')
    adjoint_field = discretization.get_unknown_size_field(name='adjoint_field')
    adjoint_field.s.fill(0)

    s_stress_and_adjoint = discretization.get_scalar_field(name='s_stress_and_adjoint')
    s_stress_and_adjoint.s[0,0], adjoint_field, adjoint_energies, info_adjoint_current = topology_optimization.sensitivity_stress_and_adjoint_FE_NEW(
        discretization=discretization,
        base_material_data_ijkl=elastic_C_0_ijkl,
        displacement_field_inxyz=displacement_field,
        adjoint_field_inxyz=adjoint_field,
        macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
        phase_field_1nxyz=phase_field_1nxyz,
        target_stress_ij=target_stress_ij,
        actual_stress_ij=homogenized_stress,
        preconditioner_fun=M_fun,
        system_matrix_fun=K_fun,
        formulation='small_strain',
        p=p,
        weight=w,
        disp=True,
        **cg_setup)

    print(f'rank = {MPI.COMM_WORLD.rank}' + 's_stress_and_adjoint= \n'          ' {} '.format(s_stress_and_adjoint.s))




    sensitivity_analytical = discretization.get_scalar_field(name='sensitivity_analytical')

    sensitivity_analytical.s = s_phase_field.s + s_stress_and_adjoint.s

    objective_function = w * f_sigma + f_phase_field
    objective_function += adjoint_energies
    # print(f'objective_function= {objective_function}')
    # print('adjoint_energy={}'.format(sensitivity_parts))
    print(f'rank = {MPI.COMM_WORLD.rank} ' + 'sensitivity_analytical= \n'          ' {} '.format(sensitivity_analytical.s))

    return objective_function, f_sigma, f_phase_field, sensitivity_analytical,  # sensitivity_parts

np.random.seed(1)
phase_field = discretization.get_scalar_field(name='phase_field_0')

geometry_ID = 'circle_inclusions'
nb_circles = 3  # number of circles in single dimention
vol_frac = 1 / 4
r_0 = np.sqrt(vol_frac / np.pi)
_geom_parameters = {'nb_circles': nb_circles,
                    'r_0': r_0,
                    'vol_frac': vol_frac,
                    'random_density': True,
                    'random_centers': True}
phase_field.s[0, 0] = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                            microstructure_name=geometry_ID,
                                                            coordinates=discretization.fft.coords,
                                                            **_geom_parameters)

phase_field.s[phase_field.s > 1] = 1
phase_field.s[phase_field.s < 0] = 0
phase_field.s[0, 0] = abs(phase_field.s[0, 0] - 1)

print(f'rank = {MPI.COMM_WORLD.rank} ' + 'phase_field_0= '          ' {} '.format(phase_field.s[0,0]))
# phase_field_0.s += 5
# save a copy of the original phase field
phase_field_0_fixed = discretization.get_scalar_field(name='phase_field_0_fixed')
phase_field_0_fixed.s = np.copy(phase_field.s)
# flatten the array --- just nick
# phase_field_0_flat = phase_field_0.s.ravel()  # TODO: check if this is copy or not
_, _, _, analytical_sensitivity = my_objective_function(phase_field.s.ravel())
print(analytical_sensitivity)
# analitical_sensitivity = analitical_sensitivity.reshape([1, 1, *number_of_pixels])
# print(sensitivity_parts)

epsilons = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]  # 1e6, 1e5, 1e4, 1e3, 1e2, 1e1,

fd_sensitivity = discretization.get_scalar_field(name='fd_sensitivity')
fd_sensitivity_drho_dro = discretization.get_scalar_field(name='fd_sensitivity_drho_dro')
fd_sensitivity_dsigma_dro = discretization.get_scalar_field(name='fd_sensitivity_dsigma_dro')

error_fd_vs_analytical = []
error_fd_vs_analytical_max = []
norm_fd_sensitivity_dsigma_dro = []
norm_fd_sensitivity_df_dro = []
norm_fd_sensitivity = []

fd_scheme = 2.
for epsilon in epsilons:
    # loop over every single element of phase field
    for x in np.arange(discretization.nb_of_pixels[0]):
        for y in np.arange(discretization.nb_of_pixels[1]):
            # set phase_field to ones
            phase_field.s = np.copy(phase_field_0_fixed.s)
            #
            phase_field.s[0, 0, x, y] = phase_field.s[0, 0, x, y] + epsilon / fd_scheme

            of_plus_eps, f_sigma_plus_eps, f_rho_plus_eps, _ = my_objective_function(phase_field.s.ravel())

            phase_field.s[0, 0, x, y] = phase_field.s[0, 0, x, y] - epsilon
            # phase_field_0 = phase_field.reshape(-1)

            of_minu_eps, f_sigma_minu_eps, f_rho_minu_eps, _ = my_objective_function(phase_field.s.ravel())

            fd_sensitivity.s[0, 0, x, y] = (of_plus_eps - of_minu_eps) / (epsilon)
            fd_sensitivity_drho_dro.s[0, 0, x, y] = (f_rho_plus_eps - f_rho_minu_eps) / (epsilon)
            fd_sensitivity_dsigma_dro.s[0, 0, x, y] = (f_sigma_plus_eps - f_sigma_minu_eps) / (epsilon)

    error_fd_vs_analytical.append(
        np.linalg.norm((fd_sensitivity.s - analytical_sensitivity.s)[0, 0], 'fro'))
    error_fd_vs_analytical_max.append(
        np.max((fd_sensitivity.s - analytical_sensitivity.s)[0, 0]))
    norm_fd_sensitivity.append(
        np.linalg.norm(fd_sensitivity.s[0, 0], 'fro'))
    norm_fd_sensitivity_df_dro.append(
        np.linalg.norm(fd_sensitivity_drho_dro.s[0, 0], 'fro'))
    norm_fd_sensitivity_dsigma_dro.append(
        np.linalg.norm(fd_sensitivity_dsigma_dro.s[0, 0], 'fro'))
print()
print(error_fd_vs_analytical)
print(norm_fd_sensitivity)
print(norm_fd_sensitivity_df_dro)
print(norm_fd_sensitivity_dsigma_dro)
print(error_fd_vs_analytical)
quad_fit_of_error = np.multiply(error_fd_vs_analytical[0], np.asarray(epsilons) ** 2)
lin_fit_of_error = np.multiply(error_fd_vs_analytical[0], np.asarray(epsilons) ** 1)
plot=True
if plot:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.loglog(epsilons, error_fd_vs_analytical,
               label=r' error_fd_vs_analytical'.format())
    plt.loglog(epsilons, error_fd_vs_analytical_max,
               label=r' error_fd_vs_analytical_max'.format())
    # plt.loglog(epsilons, norm_fd_sensitivity - np.linalg.norm(analytical_sensitivity[0, 0], 'fro'),
    #            label=r' error_fd_vs_analytical'.format())
    plt.loglog(epsilons, quad_fit_of_error,
               label=r' quad_fit_of_error'.format())
    # plt.loglog(epsilons, lin_fit_of_error,
    #            label=r' lin_fit_of_error'.format())
    plt.legend(loc='best')
    plt.title('CG tol = {}'.format(cg_setup['cg_tol']))
    plt.ylim([1e-10, 1e6])
    #   ax.legend()
    # assert error_fd_vs_analytical[-1] < epsilon * 1e2, (
    #   "Finite difference derivative do not corresponds to the analytical expression "
    #   "for partial derivative of double well potential ")
    plt.show()
