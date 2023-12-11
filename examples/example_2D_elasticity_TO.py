import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import time

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import topology_optimization

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'

domain_size = [1, 1]
number_of_pixels = (31, 31)

my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       number_of_pixels=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)

start_time = time.time()

# set macroscopic gradient

# macro_gradient = np.array([[0.2, 0], [0, 0.2]])

# target_stress = np.array([[1, 0.3], [0.3, 2]])


# set random distribution


# # apply material distribution
# p = 2
# material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * np.power(phase_field[0, 0],
#                                                                             p)
#
# # Set up the equilibrium system
# macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)
#
# # Solve mechanical equilibrium constrain
# rhs = discretization.get_rhs(material_data_field_C_0_rho, macro_gradient_field)
#
# K_fun = lambda x: discretization.apply_system_matrix(material_data_field_C_0_rho, x,
#                                                      formulation='small_strain')
# # M_fun = lambda x: 1 * x
#
# preconditioner = discretization.get_preconditioner(reference_material_data_field=material_data_field_C_0)
#
# M_fun = lambda x: discretization.apply_preconditioner(preconditioner, x)
#
# displacement_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(500), toler=1e-12)
#
# # ----------------------------------------------------------------------
# # compute homogenized stress field corresponding t
# homogenized_stress = discretization.get_homogenized_stress(
#     material_data_field_ijklqxyz=material_data_field_C_0_rho,
#     displacement_field_fnxyz=displacement_field,
#     macro_gradient_field_ijqxyz=macro_gradient_field,
#     formulation='small_strain')


macro_gradient = np.array([[0.0, 0.1],
                           [0.1, 0.0]])
print('macro_gradient = \n {}'.format(macro_gradient))

# Set up the equilibrium system
macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)

# create material data of solid phase rho=1
E_0 = 1
poison_0 = 0.1

K_0, G_0 = domain.get_bulk_and_shear_modulus(E=E_0, poison=poison_0)

elastic_C_0 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                 K=K_0,
                                                 mu=G_0,
                                                 kind='linear')

material_data_field_C_0 = np.einsum('ijkl,qxy->ijklqxy', elastic_C_0,
                                    np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                      *discretization.nb_of_pixels])))

stress = np.einsum('ijkl,lk->ij', elastic_C_0, macro_gradient)

# create target material data


K_targer, G_target = domain.get_bulk_and_shear_modulus(E=E_0, poison=poison_0)
G_target = (7 / 20) * E_0

elastic_C_target = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                      K=K_targer,
                                                      mu=G_target,
                                                      kind='linear')
# target_stress = np.array([[0.0, 0.05],
#                           [0.05, 0.0]])
target_stress = np.einsum('ijkl,lk->ij', elastic_C_target, macro_gradient)
print('target_stress = \n {}'.format(target_stress))

p = 2
w = 1 / 10  # 1e-4 Young modulus of solid
eta = 0.0025


# TODO eta = 0.025
# TODO w = 0.1
def my_objective_function(phase_field_1nxyz):
    print('OF')
    # reshape the field
    phase_field_1nxyz = phase_field_1nxyz.reshape([1, 1, *number_of_pixels])

    material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * np.power(phase_field_1nxyz,
                                                                                p)

    # Solve mechanical equilibrium constrain
    rhs = discretization.get_rhs(material_data_field_C_0_rho, macro_gradient_field)

    K_fun = lambda x: discretization.apply_system_matrix(material_data_field_C_0_rho,
                                                         x,
                                                         formulation='small_strain')
    M_fun = lambda x: 1 * x

    displacement_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(500), toler=1e-6)

    # compute homogenized stress field corresponding t
    homogenized_stress = discretization.get_homogenized_stress(
        material_data_field_ijklqxyz=material_data_field_C_0_rho,
        displacement_field_fnxyz=displacement_field,
        macro_gradient_field_ijqxyz=macro_gradient_field,
        formulation='small_strain')
    print('homogenized stress = \n'
          ' {} '.format(homogenized_stress))

    objective_function = topology_optimization.objective_function_small_strain(
        discretization=discretization,
        actual_stress_ij=homogenized_stress,
        target_stress_ij=target_stress,
        phase_field_1nxyz=phase_field_1nxyz,
        eta=eta, w=w)

    return objective_function


def my_sensitivity(phase_field_1nxyz):
    phase_field_1nxyz = phase_field_1nxyz.reshape([1, 1, *number_of_pixels])

    material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * np.power(phase_field_1nxyz,
                                                                                p)
    # Solve mechanical equilibrium constrain
    rhs = discretization.get_rhs(material_data_field_C_0_rho, macro_gradient_field)

    K_fun = lambda x: discretization.apply_system_matrix(material_data_field_C_0_rho, x,
                                                         formulation='small_strain')
    M_fun = lambda x: 1 * x

    displacement_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(500), toler=1e-6)

    # compute homogenized stress field corresponding t
    homogenized_stress = discretization.get_homogenized_stress(
        material_data_field_ijklqxyz=material_data_field_C_0_rho,
        displacement_field_fnxyz=displacement_field,
        macro_gradient_field_ijqxyz=macro_gradient_field,
        formulation='small_strain')

    sensitivity_analytical = topology_optimization.sensitivity_with_adjoint_problem(
        discretization=discretization,
        material_data_field_ijklqxyz=material_data_field_C_0,
        displacement_field_fnxyz=displacement_field,
        macro_gradient_field_ijqxyz=macro_gradient_field,
        phase_field_1nxyz=phase_field_1nxyz,
        target_stress_ij=target_stress,
        actual_stress_ij=homogenized_stress,
        formulation='small_strain',
        p=p,
        eta=eta)

    return sensitivity_analytical.reshape(-1)


if __name__ == '__main__':
    # material distribution
    phase_field_0 = np.random.rand(*discretization.get_scalar_sized_field().shape)
    phase_field_0 = phase_field_0.reshape(-1)  # b

    bounds = sp.optimize.Bounds(lb=0, ub=1, keep_feasible=True)

    xopt = sp.optimize.minimize(my_objective_function,
                                phase_field_0,
                                method='bfgs',
                                jac=my_sensitivity,
                                options={'gtol': 1e-6,
                                         'disp': True})
    print('I finished optimization')
    ###  post process

    # phase_field_1nxyz =phase_field_1nxyz.reshape([1,1,*number_of_pixels])
    phase_field_sol = xopt.x.reshape([1, 1, *number_of_pixels])
    of = my_objective_function(phase_field_sol)
    # plotting the solution
    nodal_coordinates = discretization.get_nodal_points_coordinates()

    plt.figure()
    plt.contourf(nodal_coordinates[0, 0], nodal_coordinates[1, 0], phase_field_sol[0, 0])
    plt.colorbar()
    plt.show()

    material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * np.power(phase_field_sol,
                                                                                p)
    # Set up the equilibrium system
    macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)

    # Solve mechanical equilibrium constrain
    rhs = discretization.get_rhs(material_data_field_C_0_rho, macro_gradient_field)

    K_fun = lambda x: discretization.apply_system_matrix(material_data_field_C_0_rho, x,
                                                         formulation='small_strain')
    M_fun = lambda x: 1 * x

    displacement_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(500), toler=1e-6)

    # compute homogenized stress field corresponding t
    homogenized_stress = discretization.get_homogenized_stress(
        material_data_field_ijklqxyz=material_data_field_C_0_rho,
        displacement_field_fnxyz=displacement_field,
        macro_gradient_field_ijqxyz=macro_gradient_field,
        formulation='small_strain')

    objective_function = topology_optimization.objective_function_small_strain(
        discretization=discretization,
        actual_stress_ij=homogenized_stress,
        target_stress_ij=target_stress,
        phase_field_1nxyz=phase_field_sol,
        eta=eta, w=w)

    print(of)

    # def my_sensitivity(phase_field_1nxyz,
    #                           material_data_field_C_0_ijklqxyz):
    #
    #     sensitivity_analytical = topology_optimization.sensitivity_with_adjoint_problem(
    #         discretization=discretization,
    #         material_data_field_ijklqxyz=material_data_field_C_0,
    #         displacement_field_fnxyz=displacement_field,
    #         macro_gradient_field_ijqxyz=macro_gradient_field,
    #         phase_field_1nxyz=phase_field,
    #         target_stress_ij=target_stress,
    #         actual_stress_ij=homogenized_stress,
    #         formulation='small_strain',
    #         p=p,
    #         eta=1)
    #
    #     return sensitivity_analytical
    #
    # stress_difference_ij = homogenized_stress - target_stress
    #
    # adjoint_field = topology_optimization.solve_adjoint_problem(
    #     discretization=discretization,
    #     material_data_field_ijklqxyz=material_data_field_C_0_rho,
    #     stress_difference_ij=stress_difference_ij,
    #     formulation='small_strain')
    #
    # sensitivity_analytical = topology_optimization.sensitivity(
    #     discretization=discretization,
    #     material_data_field_ijklqxyz=material_data_field_C_0,
    #     displacement_field_fnxyz=displacement_field,
    #     macro_gradient_field_ijqxyz=macro_gradient_field,
    #     phase_field_1nxyz=phase_field,
    #     adjoint_field_fnxyz=adjoint_field,
    #     target_stress_ij=target_stress,
    #     actual_stress_ij=homogenized_stress,
    #     p=p,
    #     eta=1)

    # TODO TO FINISH

    print(5)
