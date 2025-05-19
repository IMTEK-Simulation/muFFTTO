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
number_of_pixels = (16, 16)

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


# macro_gradient = np.array([[0.0, 0.01],
#                            [0.01, 0.0]])
macro_gradient = np.array([[1.0, .0],
                           [.0, 1.0]])
print('macro_gradient = \n {}'.format(macro_gradient))

# Set up the equilibrium system
macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)

# create material data of solid phase rho=1
E_0 = 100
poison_0 = 0.

K_0, G_0 = domain.get_bulk_and_shear_modulus(E=E_0, poison=poison_0)

# identity tensor                                               [single tensor]
i      = np.eye(2)
# identity tensors                                            [grid of tensors]
#I = np.einsum('ij,xy', i, np.ones(number_of_pixels))
#I4 = np.einsum('ijkl,xy->ijklxy', np.einsum('il,jk', i, i), np.ones(number_of_pixels))
#I4rt = np.einsum('ijkl,xy->ijklxy', np.einsum('ik,jl', i, i), np.ones(number_of_pixels))
#I4s = (I4 + I4rt) / 2.

I = np.einsum('ij', i)
I4 =  np.einsum('il,jk', i, i)
I4rt = np.einsum('ik,jl', i, i)
I4s = (I4 + I4rt) / 2.

elastic_C_0 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                 K=K_0,
                                                 mu=G_0,
                                                 kind='linear')
#elastic_C_0=2*I4s
material_data_field_C_0 = np.einsum('ijkl,qxy->ijklqxy', elastic_C_0,
                                    np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                      *discretization.nb_of_pixels])))

stress = np.einsum('ijkl,lk->ij', elastic_C_0, macro_gradient)

# create target material data
print('init_stress = \n {}'.format(stress))
# validation metamaterials
poison_target = -0.5
E_target = E_0 * 0.2

# poison_target = 0.2
# G_target_auxet = (3 / 10) * E_0  # (7 / 20) * E_0
# G_target_auxet = (0.2) * E_0
# E_target = 2 * G_target_auxet * (1 + poison_target)
# Auxetic metamaterials
# poison_target= 1/3  # lambda = -10
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
p = 1.1
w = 1  # e0# +* E_0  # 1 / 10  # 1e-4 Young modulus of solid
eta = 1

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
    # TODO [Lars] is this proper formulation ? I wabt to multipluy phase field [0,q,0,xyz] * material_data_field_C_0[ijkl,q,xyz]
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
    print('homogenized stress = \n'          ' {} '.format(homogenized_stress))

    objective_function = topology_optimization.objective_function_small_strain_FE_testing(
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

    # apply rho distribution to displacement
    displacement_field_rho = displacement_field * phase_field_1nxyz
    # compute homogenized stress field corresponding t
    homogenized_stress = discretization.get_homogenized_stress(
        material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
        displacement_field_fnxyz=displacement_field,
        macro_gradient_field_ijqxyz=macro_gradient_field,
        formulation='small_strain')

    # print('Sensitivity_analytical')
    sensitivity_parts = topology_optimization.sensitivity_with_adjoint_problem_FE_testing(
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

    return sensitivity_parts


if __name__ == '__main__':
    # material distribution
    phase_field_0 = np.random.rand(*discretization.get_scalar_sized_field().shape) ** 0
    # phase_field_0 = np.random.randint(0, high=2, size=discretization.get_scalar_sized_field().shape) ** 1
    #linfunc = lambda x: np.abs(1 * x - 0.5) + 0.1
    #phase_field_0[0, 0] = linfunc(discretization.get_nodal_points_coordinates()[0, 0])
    phase_field_0[0, 0,
    phase_field_0.shape[2] * 1 // 4:phase_field_0.shape[2] * 3 // 4,
    phase_field_0.shape[2] * 1 // 4:phase_field_0.shape[2] * 3 // 4] = 0

    phase_field_00 = np.copy(phase_field_0)

    phase_field_0 = phase_field_0.reshape(-1)  # b

    # print('Init objective function FE  = {}'.format(my_objective_function_FE(phase_field_00)))
    # print('Init objective function pixel  = {}'.format(my_objective_function_pixel(phase_field_00)))

    F_FE = my_objective_function_FE(phase_field_00)
    F_pixel = my_objective_function_pixel(phase_field_00)

    S_FE = my_sensitivity_FE(phase_field_00)['sensitivity'].reshape([1, 1, *number_of_pixels])
    S_pixel = my_sensitivity_pixel(phase_field_00).reshape([1, 1, *number_of_pixels])

    nodal_coordinates = discretization.get_nodal_points_coordinates()

    plt.figure()
    plt.contourf(nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
                 S_FE[0, 0], cmap=mpl.cm.Greys)
    plt.title(r"FE sensitivity p = {}, "
              r"OF = {}".format(p, F_FE, wrap=True))
    # plt.clim(0, 1)
    plt.colorbar()

    plt.figure()
    plt.contourf(nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
                 S_pixel[0, 0], cmap=mpl.cm.Greys)
    plt.title(r"Pixel sensitivity p = {},"
              r"OF = {}".format(p, F_pixel, wrap=True))
    # plt.clim(0, 1)
    plt.colorbar()

    S_FE_parts = my_sensitivity_FE(phase_field_00)

    fig, ax = plt.subplots(2, 2,figsize=(10,6))
    fig.suptitle('Components of sensitivity: FE discretization'"\n"
                 r"p = {}, OF = {}".format(p, F_FE, wrap=True))

    im = ax[0, 0].contourf(nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
                           S_FE_parts['dfstress_drho'].reshape([1, 1, *number_of_pixels])[0, 0], cmap=mpl.cm.viridis)
    ax[0, 0].set_title(r"FE dfstress_drho ")
    # plt.clim(0, 1)
    plt.colorbar(im, ax=ax[0, 0])

    im = ax[0, 1].contourf(nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
                           S_FE_parts['dgradrho_drho'].reshape([1, 1, *number_of_pixels])[0, 0], cmap=mpl.cm.viridis)
    ax[0, 1].set_title(r"FE dgradrho_drho ")
    plt.colorbar(im, ax=ax[0, 1])

    im = ax[1, 0].contourf(nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
                           S_FE_parts['ddouble_well_drho_drho'].reshape([1, 1, *number_of_pixels])[0, 0],
                           cmap=mpl.cm.viridis)
    ax[1, 0].set_title(r"FE ddouble_well_drho ")
    plt.colorbar(im, ax=ax[1, 0])

    im = ax[1, 1].contourf(nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
                           S_FE_parts['dg_drho_nxyz'].reshape([1, 1, *number_of_pixels])[0, 0], cmap=mpl.cm.viridis)
    ax[1, 1].set_title(r"FE dg_drho")
    plt.colorbar(im, ax=ax[1, 1])

    src = './figures/'  # source folder\
    fig_data_name = f'muFFTTO_{problem_type}_Sensitivities_N{number_of_pixels[0]}_w{w}_eta{eta}_p{p}_FE'

    fname = src + fig_data_name + '{}'.format('.png')
    print(('create figure: {}'.format(fname)))  # axes[1, 0].legend(loc='upper right')
    plt.savefig(fname, bbox_inches='tight')
    print('END plot ')

    S_FE_parts = my_sensitivity_FE(phase_field_00)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Components of sensitivity: FE discretization'"\n"
                 r"p = {}, OF = {}".format(p, F_FE, wrap=True))

    im = ax[0].contourf(nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
                           S_FE_parts['dfstress_drho'].reshape([1, 1, *number_of_pixels])[0, 0], cmap=mpl.cm.viridis)
    ax[0].set_title(r"FE dfstress_drho ")
    # plt.clim(0, 1)
    plt.colorbar(im, ax=ax[0])

    im = ax[1].contourf(nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
                           S_FE_parts['dg_drho_nxyz'].reshape([1, 1, *number_of_pixels])[0, 0], cmap=mpl.cm.viridis)
    ax[1].set_title(r"FE dg_drho")
    plt.colorbar(im, ax=ax[1])

    src = './figures/'  # source folder\
    fig_data_name = f'muFFTTO_{problem_type}_Sensitivities_material_N{number_of_pixels[0]}_w{w}_eta{eta}_p{p}_FE'

    fname = src + fig_data_name + '{}'.format('.png')
    print(('create figure: {}'.format(fname)))  # axes[1, 0].legend(loc='upper right')
    plt.savefig(fname, bbox_inches='tight')
    print('END plot ')
    plt.show()
    print()
