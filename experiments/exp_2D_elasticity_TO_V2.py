import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt


plt.rcParams['text.usetex'] = True

import time

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import topology_optimization

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'  # 'bilinear_rectangle'##'linear_triangles' #
formulation = 'small_strain'

domain_size = [1, 1]
number_of_pixels = (64,64)

my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       number_of_pixels=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)

start_time = time.time()

# set macroscopic gradient


# macro_gradient = np.array([[0.0, 0.01],1
#                            [0.01, 0.0]])
macro_gradient = np.array([[1.0, 0.0],
                           [0.0, 1.0]])
print('macro_gradient = \n {}'.format(macro_gradient))

# Set up the equilibrium system
macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)

# create material data of solid phase rho=1
E_0 = 1
poison_0 = 0.2

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
print('init_stress = \n {}'.format(stress))
# validation metamaterials
#poison_target = -0.5
#E_target = E_0 * 0.1

# poison_target = 0.2
poison_target= -1/ 2  # lambda = -10

#G_target_auxet = (3 / 10) * E_0  # (7 / 20) * E_0
G_target_auxet = (1 / 4) * E_0

E_target = 2 * G_target_auxet * (1 + poison_target)
# Auxetic metamaterials
# G_target_auxet = (1 / 4) * E_0  #23   25
# E_target=2*G_target_auxet*(1+poison_target)
# test materials


K_targer, G_target = domain.get_bulk_and_shear_modulus(E=E_target, poison=poison_target)

elastic_C_target = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                      K=K_targer,
                                                      mu=G_target,
                                                      kind='linear')
# target_stress = np.array([[0.0, 0.05],
#                           [0.05, 0.0]])
target_stress = np.einsum('ijkl,lk->ij', elastic_C_target, macro_gradient)
print('target_stress = \n {}'.format(target_stress))
# circle
# p = 1
# w = 1e-5*E_0# 1 / 10  # 1e-4 Young modulus of solid
# #eta = 0.00915#1430#145#357#3#33#5#25#4#7#250
# eta = 0.0555 #0.02125#08#1231925#1515#1430#145#357#3#33#5#25#4#7#250
# Auxetic metamaterials
p = 2
w = 5*1e-3 #1e-2 #/6# * E_0  # 1 / 10  # 1e-4 Young modulus of solid
eta =  0.01# domain_size[0] / number_of_pixels[0]  # 0.020.005# 2 *

print('p =   {}'.format(p))
print('w  =  {}'.format(w))
print('eta =  {}'.format(eta))


# eta = 0.00915#1430#145#357#3#33#5#25#4#7#250
# TODO eta = 0.025
# TODO w = 0.1
def my_objective_function_pixel(phase_field_1nxyz):
    # print('Objective function:')
    # reshape the field
    phase_field_1nxyz = phase_field_1nxyz.reshape([1, 1, *number_of_pixels])

    material_data_field_C_0_rho_ijklqxyz = material_data_field_C_0[..., :, :] * np.power(phase_field_1nxyz, p)

    # Solve mechanical equilibrium constrain
    rhs = discretization.get_rhs(material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                                 macro_gradient_field_ijqxyz=macro_gradient_field)

    K_fun = lambda x: discretization.apply_system_matrix(material_data_field=material_data_field_C_0_rho_ijklqxyz,
                                                         displacement_field=x,
                                                         formulation='small_strain')
    # M_fun = lambda x: 1 * x
    preconditioner = discretization.get_preconditioner(
        reference_material_data_field_ijklqxyz=material_data_field_C_0)
    M_fun = lambda x: discretization.apply_preconditioner(preconditioner_Fourier_fnfnxyz=preconditioner,
                                                          nodal_field_fnxyz=x)

    displacement_field, norms = solvers.PCG(Afun=K_fun, B=rhs, x0=None, P=M_fun, steps=int(500), toler=1e-6)

    # compute homogenized stress field corresponding t
    homogenized_stress = discretization.get_homogenized_stress(
        material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
        displacement_field_fnxyz=displacement_field,
        macro_gradient_field_ijqxyz=macro_gradient_field,
        formulation='small_strain')
    # print('homogenized stress = \n'          ' {} '.format(homogenized_stress))

    objective_function = topology_optimization.objective_function_small_strain_pixel(
        discretization=discretization,
        actual_stress_ij=homogenized_stress,
        target_stress_ij=target_stress,
        phase_field_1nxyz=phase_field_1nxyz,
        eta=eta,
        w=w)
    # print('objective_function= \n'' {} '.format(objective_function))

    return objective_function


def my_objective_function_FE(phase_field_1nxyz):
    # print('Objective function:')
    # reshape the field
    phase_field_1nxyz = phase_field_1nxyz.reshape([1, 1, *number_of_pixels])

    phase_field_at_quad_poits_1qnxyz = discretization.evaluate_field_at_quad_points(nodal_field_fnxyz=phase_field_1nxyz,
                                                                                    quad_field_fqnxyz=None,
                                                                                    quad_points_coords_iq=None)[0]

    material_data_field_C_0_rho_ijklqxyz = material_data_field_C_0[..., :, :, :] * np.power(
        phase_field_at_quad_poits_1qnxyz, p)[0, :, 0, ...]
    # TODO [Lars] is this proper formulation ? I want to multiply phase field [0,q,0,xyz] * material_data_field_C_0[ijkl,q,xyz]
    # Solve mechanical equilibrium constrain
    rhs = discretization.get_rhs(material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                                 macro_gradient_field_ijqxyz=macro_gradient_field)

    K_fun = lambda x: discretization.apply_system_matrix(material_data_field=material_data_field_C_0_rho_ijklqxyz,
                                                         displacement_field=x,
                                                         formulation='small_strain')
    # M_fun = lambda x: 1 * x
    preconditioner = discretization.get_preconditioner(
        reference_material_data_field_ijklqxyz=material_data_field_C_0)
    M_fun = lambda x: discretization.apply_preconditioner(preconditioner_Fourier_fnfnxyz=preconditioner,
                                                          nodal_field_fnxyz=x)
    displacement_field, norms = solvers.PCG(Afun=K_fun, B=rhs, x0=None, P=M_fun, steps=int(500), toler=1e-6)

    # compute homogenized stress field corresponding t
    homogenized_stress = discretization.get_homogenized_stress(
        material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
        displacement_field_fnxyz=displacement_field,
        macro_gradient_field_ijqxyz=macro_gradient_field,
        formulation='small_strain')
    # print('homogenized stress = \n'          ' {} '.format(homogenized_stress))

    objective_function = topology_optimization.objective_function_small_strain(
        discretization=discretization,
        actual_stress_ij=homogenized_stress,
        target_stress_ij=target_stress,
        phase_field_1nxyz=phase_field_1nxyz,
        eta=eta,
        w=w)
    # print('objective_function= \n'' {} '.format(objective_function))

    return objective_function


def my_sensitivity_pixel(phase_field_1nxyz):
    # print('Sensitivity calculation')

    phase_field_1nxyz = phase_field_1nxyz.reshape([1, 1, *number_of_pixels])

    # Compute homogenized stress field for current phase field
    material_data_field_C_0_rho_ijklqxyz = material_data_field_C_0[..., :, :] * np.power(phase_field_1nxyz, p)

    # Solve mechanical equilibrium constrain for hom
    rhs = discretization.get_rhs(material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                                 macro_gradient_field_ijqxyz=macro_gradient_field)

    K_fun = lambda x: discretization.apply_system_matrix(material_data_field=material_data_field_C_0_rho_ijklqxyz,
                                                         displacement_field=x,
                                                         formulation='small_strain')
    # M_fun = lambda x: 1 * x
    preconditioner = discretization.get_preconditioner(
        reference_material_data_field_ijklqxyz=material_data_field_C_0)
    M_fun = lambda x: discretization.apply_preconditioner(preconditioner_Fourier_fnfnxyz=preconditioner,
                                                          nodal_field_fnxyz=x)

    displacement_field, norms = solvers.PCG(Afun=K_fun, B=rhs, x0=None, P=M_fun, steps=int(500), toler=1e-6)

    # compute homogenized stress field corresponding t
    homogenized_stress = discretization.get_homogenized_stress(
        material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
        displacement_field_fnxyz=displacement_field,
        macro_gradient_field_ijqxyz=macro_gradient_field,
        formulation='small_strain')

    # print('Sensitivity_analytical')
    sensitivity_analytical = topology_optimization.sensitivity_with_adjoint_problem_pixel(
        discretization=discretization,
        material_data_field_ijklqxyz=material_data_field_C_0,
        displacement_field_fnxyz=displacement_field,
        macro_gradient_field_ijqxyz=macro_gradient_field,
        phase_field_1nxyz=phase_field_1nxyz,
        target_stress_ij=target_stress,
        actual_stress_ij=homogenized_stress,
        formulation='small_strain',
        p=p,
        eta=eta,
        weight=w)

    return sensitivity_analytical.reshape(-1)


def my_sensitivity_FE(phase_field_1nxyz):
    # print('Sensitivity calculation')

    phase_field_1nxyz = phase_field_1nxyz.reshape([1, 1, *number_of_pixels])

    # Compute homogenized stress field for current phase field
    # material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * np.power(phase_field_1nxyz, p)

    phase_field_at_quad_poits_1qnxyz = discretization.evaluate_field_at_quad_points(nodal_field_fnxyz=phase_field_1nxyz,
                                                                                    quad_field_fqnxyz=None,
                                                                                    quad_points_coords_iq=None)[0]

    material_data_field_C_0_rho_ijklqxyz = material_data_field_C_0[..., :, :, :] * np.power(
        phase_field_at_quad_poits_1qnxyz, p)[0, :, 0, ...]

    # Solve mechanical equilibrium constrain for hom
    rhs = discretization.get_rhs(material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                                 macro_gradient_field_ijqxyz=macro_gradient_field)

    K_fun = lambda x: discretization.apply_system_matrix(material_data_field=material_data_field_C_0_rho_ijklqxyz,
                                                         displacement_field=x,
                                                         formulation='small_strain')
    # M_fun = lambda x: 1 * x
    preconditioner = discretization.get_preconditioner(
        reference_material_data_field_ijklqxyz=material_data_field_C_0)
    M_fun = lambda x: discretization.apply_preconditioner(preconditioner_Fourier_fnfnxyz=preconditioner,
                                                          nodal_field_fnxyz=x)

    displacement_field, norms = solvers.PCG(Afun=K_fun, B=rhs, x0=None, P=M_fun, steps=int(500), toler=1e-6)

    # compute homogenized stress field corresponding t
    homogenized_stress = discretization.get_homogenized_stress(
        material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
        displacement_field_fnxyz=displacement_field,
        macro_gradient_field_ijqxyz=macro_gradient_field,
        formulation='small_strain')

    # print('Sensitivity_analytical')
    sensitivity_analytical = topology_optimization.sensitivity_with_adjoint_problem_FE(
        discretization=discretization,
        material_data_field_ijklqxyz=material_data_field_C_0,
        displacement_field_fnxyz=displacement_field,
        macro_gradient_field_ijqxyz=macro_gradient_field,
        phase_field_1nxyz=phase_field_1nxyz,
        target_stress_ij=target_stress,
        actual_stress_ij=homogenized_stress,
        formulation='small_strain',
        p=p,
        eta=eta,
        weight=w)

    return sensitivity_analytical.reshape(-1)


if __name__ == '__main__':
    # material distribution
    phase_field_0 = np.random.rand(*discretization.get_scalar_sized_field().shape) ** 1
    # phase_field_0 = np.random.randint(0, high=2, size=discretization.get_scalar_sized_field().shape) ** 1

    # phase_field_0[0, 0,
    # phase_field_0.shape[2] * 1 // 4:phase_field_0.shape[2] * 3 // 4,
    # phase_field_0.shape[2] * 1 // 4:phase_field_0.shape[2] * 3 // 4] = 0
    phase_field_00 = np.copy(phase_field_0)

    phase_field_0 = phase_field_0.reshape(-1)  # b

    print('Init objective function FE  = {}'.format(my_objective_function_FE(phase_field_00)))
    print('Init objective function pixel  = {}'.format(my_objective_function_pixel(phase_field_00)))

    start_time = time.time()
    bounds = False
    if bounds:
        phase_bounds = sp.optimize.Bounds(lb=0, ub=1, keep_feasible=True)
        xopt_pixel = sp.optimize.minimize(fun=my_objective_function_pixel,
                                          x0=phase_field_0,
                                          method='l-bfgs-b',
                                          jac=my_sensitivity_pixel,
                                          bounds=phase_bounds,
                                          options={'gtol': 1e-6,
                                                   'disp': True,
                                                   'maxiter': 3000})
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("xopt_pixel Bounds time: ", elapsed_time)

        xopt_FE = sp.optimize.minimize(fun=my_objective_function_FE,
                                       x0=phase_field_0,
                                       method='l-bfgs-b',
                                       jac=my_sensitivity_FE,
                                       bounds=phase_bounds,
                                       options={'gtol': 1e-6,
                                                'disp': True,
                                                'maxiter': 3000})
        end_time = time.time()
        elapsed_time = end_time - elapsed_time
        print("xopt_FE Bounds time: ", elapsed_time)

    else:
        xopt_pixel = sp.optimize.minimize(fun=my_objective_function_pixel,
                                          x0=phase_field_0,
                                          method='l-bfgs-b',
                                          jac=my_sensitivity_pixel,
                                          options={'gtol': 1e-6,
                                                   'disp': True,
                                                   'maxiter': 500})
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("xopt_pixel No Bounds time: ", elapsed_time)

        xopt_FE = sp.optimize.minimize(fun=my_objective_function_FE,
                                       x0=phase_field_0,
                                       method='l-bfgs-b',
                                       jac=my_sensitivity_FE,
                                       options={'gtol': 1e-6,
                                                'disp': True,
                                                'maxiter': 2000})
        end_time = time.time()
        elapsed_time = end_time - elapsed_time
        print("xopt_FE No Bounds time: ", elapsed_time)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Optimization time: ", elapsed_time)
    # xopt = sp.optimize.minimize(fun=my_objective_function,
    #                             x0=phase_field_0,
    #                             method='bfgs',
    #                             jac=my_sensitivity,
    #                             options={'gtol': 1e-6,
    #                                      'disp': True})

    # xopt = sp.optimize.minimize(fun=my_objective_function,
    #                             x0=phase_field_0,
    #                             options={'gtol': 1e-6,
    #                                      'disp': True})
    print('I finished optimization')
    ###  post process

    # phase_field_1nxyz =phase_field_1nxyz.reshape([1,1,*number_of_pixels])
    phase_field_sol_FE = xopt_FE.x.reshape([1, 1, *number_of_pixels])
    phase_field_sol_pixel = xopt_pixel.x.reshape([1, 1, *number_of_pixels])

    of = my_objective_function_FE(phase_field_sol_FE)
    # plotting the solution
    nodal_coordinates = discretization.get_nodal_points_coordinates()

    plt.figure()
    plt.contourf(nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
                 phase_field_00[0, 0])

    plt.clim(0, 1)
    plt.colorbar()

    plt.show()
    ######## Postprocess for FE linear solver ########
    # material_data_field_C_0_rho_pixel = material_data_field_C_0[..., :, :] * np.power(phase_field_sol,
    #                                                                             p)
    phase_field_at_quad_poits_1qnxyz = \
    discretization.evaluate_field_at_quad_points(nodal_field_fnxyz=phase_field_sol_FE,
                                                 quad_field_fqnxyz=None,
                                                 quad_points_coords_iq=None)[0]
    material_data_field_C_0_rho_quad = material_data_field_C_0[..., :, :, :] * np.power(
        phase_field_at_quad_poits_1qnxyz, p)[0, :, 0, ...]

    # Set up the equilibrium system
    macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)

    # Solve mechanical equilibrium constrain
    rhs = discretization.get_rhs(material_data_field_C_0_rho_quad, macro_gradient_field)

    K_fun = lambda x: discretization.apply_system_matrix(material_data_field_C_0_rho_quad, x,
                                                         formulation='small_strain')
    preconditioner = discretization.get_preconditioner(
        reference_material_data_field_ijklqxyz=material_data_field_C_0)
    M_fun = lambda x: discretization.apply_preconditioner(preconditioner_Fourier_fnfnxyz=preconditioner,
                                                          nodal_field_fnxyz=x)

    displacement_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(500), toler=1e-8)

    # compute homogenized stress field corresponding t
    homogenized_stress = discretization.get_homogenized_stress(
        material_data_field_ijklqxyz=material_data_field_C_0_rho_quad,
        displacement_field_fnxyz=displacement_field,
        macro_gradient_field_ijqxyz=macro_gradient_field,
        formulation='small_strain')
    print('init_stress = \n {}'.format(stress))
    print('Target_stress = \n {}'.format(target_stress))
    print('Optimized stress = \n {}'.format(homogenized_stress))

    print('Stress diff = \n {}'.format(target_stress - homogenized_stress))
    objective_function = topology_optimization.objective_function_small_strain(
        discretization=discretization,
        actual_stress_ij=homogenized_stress,
        target_stress_ij=target_stress,
        phase_field_1nxyz=phase_field_sol_FE,
        eta=eta, w=w)

    print(of)

    start_time = time.time()
    dim = discretization.domain_dimension
    homogenized_C_ijkl = np.zeros(np.array(4 * [dim, ]))
    # compute whole homogenized elastic tangent
    for i in range(dim):
        for j in range(dim):
            # set macroscopic gradient
            macro_gradient_ij = np.zeros([dim, dim])
            macro_gradient_ij[i, j] = 1
            # Set up right hand side
            macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient_ij)

            # Solve mechanical equilibrium constrain
            rhs_ij = discretization.get_rhs(material_data_field_C_0_rho_quad, macro_gradient_field)

            displacement_field_ij, norms = solvers.PCG(K_fun, rhs_ij, x0=None, P=M_fun, steps=int(500), toler=1e-8)

            # ----------------------------------------------------------------------
            # compute homogenized stress field corresponding
            homogenized_C_ijkl[i, j] = discretization.get_homogenized_stress(
                material_data_field_ijklqxyz=material_data_field_C_0_rho_quad,
                displacement_field_fnxyz=displacement_field_ij,
                macro_gradient_field_ijqxyz=macro_gradient_field,
                formulation='small_strain')

    print('Optimized elastic tangent = \n {}'.format(domain.compute_Voigt_notation_4order(homogenized_C_ijkl)))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time)

    plt.figure()
    plt.contourf(nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
                 phase_field_sol_FE[0, 0], cmap=mpl.cm.Greys)

    plt.clim(0, 1)
    plt.colorbar()
    src = './figures/'  # source folder\
    fig_data_name = f'muFFTTO_{problem_type}_random_init_N{number_of_pixels[0]}_w{w}_eta{eta}_p{p}_bounds={bounds}'
    plt.title(r" linear FE " "\n"
        r" Target stress $[{} , {}],[ {}, {}] $" "\n"
              r" Stress  $[{} ,{},][ {}, {} ]$" "\n"
              r" nb_iter={},  p={}".format(target_stress[0, 0], target_stress[0, 1],
                                    target_stress[1, 0], target_stress[1, 1],
                                    homogenized_stress[0, 0], homogenized_stress[0, 1],
                                    homogenized_stress[1, 0], homogenized_stress[1, 1],
                                    xopt_FE.nit , p), wrap=True)
    fname = src + fig_data_name + '{}'.format('.png')
    print(('create figure: {}'.format(fname)))  # axes[1, 0].legend(loc='upper right')
    plt.savefig(fname, bbox_inches='tight')
    print('END plot ')

    print('Target elastic FE tangent = \n {}'.format(domain.compute_Voigt_notation_4order(elastic_C_target)))

    print('Initial elastic tangent = \n {}'.format(domain.compute_Voigt_notation_4order(elastic_C_0)))


    ######## Postprocess for Pixel constant linear solver ########

    material_data_field_C_0_rho_pixel = material_data_field_C_0[..., :, :] * np.power(phase_field_sol_pixel,
                                                                                p)


    # Set up the equilibrium system
    macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)

    # Solve mechanical equilibrium constrain
    rhs = discretization.get_rhs(material_data_field_C_0_rho_pixel, macro_gradient_field)

    K_fun = lambda x: discretization.apply_system_matrix(material_data_field_C_0_rho_pixel, x,
                                                         formulation='small_strain')


    displacement_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(500), toler=1e-8)

    # compute homogenized stress field corresponding t
    homogenized_stress = discretization.get_homogenized_stress(
        material_data_field_ijklqxyz=material_data_field_C_0_rho_pixel,
        displacement_field_fnxyz=displacement_field,
        macro_gradient_field_ijqxyz=macro_gradient_field,
        formulation='small_strain')
    print('init_stress Pixel= \n {}'.format(stress))
    print('Target_stress Pixel= \n {}'.format(target_stress))
    print('Optimized stress Pixel= \n {}'.format(homogenized_stress))

    print('Stress diff Pixel= \n {}'.format(target_stress - homogenized_stress))
    objective_function = topology_optimization.objective_function_small_strain(
        discretization=discretization,
        actual_stress_ij=homogenized_stress,
        target_stress_ij=target_stress,
        phase_field_1nxyz=phase_field_sol_pixel,
        eta=eta, w=w)

    print(of)

    start_time = time.time()
    dim = discretization.domain_dimension
    homogenized_C_ijkl = np.zeros(np.array(4 * [dim, ]))
    # compute whole homogenized elastic tangent
    for i in range(dim):
        for j in range(dim):
            # set macroscopic gradient
            macro_gradient_ij = np.zeros([dim, dim])
            macro_gradient_ij[i, j] = 1
            # Set up right hand side
            macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient_ij)

            # Solve mechanical equilibrium constrain
            rhs_ij = discretization.get_rhs(material_data_field_C_0_rho_pixel, macro_gradient_field)

            displacement_field_ij, norms = solvers.PCG(K_fun, rhs_ij, x0=None, P=M_fun, steps=int(500), toler=1e-8)

            # ----------------------------------------------------------------------
            # compute homogenized stress field corresponding
            homogenized_C_ijkl[i, j] = discretization.get_homogenized_stress(
                material_data_field_ijklqxyz=material_data_field_C_0_rho_pixel,
                displacement_field_fnxyz=displacement_field_ij,
                macro_gradient_field_ijqxyz=macro_gradient_field,
                formulation='small_strain')

    print('Optimized elastic tangent  Pixel= \n {}'.format(domain.compute_Voigt_notation_4order(homogenized_C_ijkl)))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time Pixel: ", elapsed_time)

    plt.figure()
    plt.contourf(nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
                 phase_field_sol_pixel[0, 0], cmap=mpl.cm.Greys)

    plt.clim(0, 1)
    plt.colorbar()
    src = './figures/'  # source folder\
    fig_data_name = f'muFFTTO_{problem_type}_random_init_N{number_of_pixels[0]}_w{w}_eta{eta}_p{p}_bounds={bounds}_pixel'
    plt.title(r" Pixel " "\n"
              r" Target stress $[{} , {}],[ {}, {}] $" "\n"
              r" Stress  $[{} ,{},][ {}, {} ]$" "\n"
              r" nb_iter={}".format(target_stress[0, 0], target_stress[0, 1],
                                    target_stress[1, 0], target_stress[1, 1],
                                    homogenized_stress[0, 0], homogenized_stress[0, 1],
                                    homogenized_stress[1, 0], homogenized_stress[1, 1],
                                    xopt_pixel.nit), wrap=True)
    fname = src + fig_data_name + '{}'.format('.png')
    print(('create figure: {}'.format(fname)))  # axes[1, 0].legend(loc='upper right')
    plt.savefig(fname, bbox_inches='tight')
    print('END plot ')


    # TODO TO FINISH
    print('p =   {}'.format(p))
    print('w  =  {}'.format(w))
    print('eta =  {}'.format(eta))
    print(5)
