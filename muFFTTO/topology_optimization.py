import warnings

import numpy as np
import scipy as sc
import time

from muFFTTO import domain
from muFFTTO import solvers

from NuMPI.Tools import Reduction
from mpi4py import MPI


def objective_function_small_strain(discretization,
                                    actual_stress_ij,
                                    target_stress_ij,
                                    phase_field_1nxyz,
                                    eta,
                                    w,
                                    double_well_depth=1,
                                    disp=False):
    # evaluate objective functions
    # f = (flux_h -flux_target)^2 + w*eta* int (  (grad(rho))^2 )dx  +    int ( rho^2(1-rho)^2 ) / eta   dx
    # f =  f_sigma + w*eta* f_rho_grad  + f_dw/eta

    # stress difference potential: actual_stress_ij is homogenized stress
    f_sigma = compute_stress_equivalence_potential(actual_stress_ij=actual_stress_ij,
                                                   target_stress_ij=target_stress_ij)
    if MPI.COMM_WORLD.rank == 0:
        print('f_sigma= \n'          ' {} '.format(f_sigma))  # good in MPI

    # double - well potential
    # integrant = (phase_field_1nxyz ** 2) * (1 - phase_field_1nxyz) ** 2
    # f_dw = (np.sum(integrant) / np.prod(integrant.shape)) * discretization.cell.domain_volume
    f_dw = compute_double_well_potential_analytical(discretization=discretization,
                                                    phase_field_1nxyz=phase_field_1nxyz)
    if disp and MPI.COMM_WORLD.rank == 0:
        print('f_dw= \n'          ' {} '.format(f_dw))  # wrong in MPI

    # phase_field_gradient = discretization.apply_gradient_operator(phase_field_1nxyz)
    # f_rho_grad = np.sum(discretization.integrate_over_cell(phase_field_gradient ** 2))
    f_rho_grad = compute_gradient_of_phase_field_potential(discretization=discretization,
                                                           phase_field_1nxyz=phase_field_1nxyz)
    if disp and MPI.COMM_WORLD.rank == 0:
        print('f_rho_grad= \n'          ' {} '.format(f_rho_grad))  # good in MPI
    # print()
    # gradient_of_phase_field = compute_gradient_of_phase_field(phase_field_gradient)

    f_rho = eta * f_rho_grad + double_well_depth * f_dw / eta
    # if MPI.COMM_WORLD.rank == 0:
    # print('f_sigma linear=  {} '.format(f_sigma))
    # print('f_rho_grad linear=  {} '.format(f_rho_grad))
    # print('f_dw =  linear {} '.format(f_dw))
    #
    # print('f_rho   linear = {} '.format(f_rho))
    # print('w * f_rho linear =  {} '.format(w * f_rho))
    # print('objective_function linear = {} '.format(f_sigma + w * f_rho))

    return w * f_sigma + f_rho  # / discretization.cell.domain_volume


def objective_function_phase_field(discretization,
                                   phase_field_1nxyz,
                                   eta,
                                   double_well_depth=1,
                                   disp=False):
    # evaluate objective functions
    # f =  w*eta* int (  (grad(rho))^2 )dx  +    int ( rho^2(1-rho)^2 ) / eta   dx
    # f =  eta* f_rho_grad  + f_dw/eta

    # double - well potential
    # integrant = (phase_field_1nxyz ** 2) * (1 - phase_field_1nxyz) ** 2
    f_dw = compute_double_well_potential_analytical(discretization=discretization,
                                                    phase_field_1nxyz=phase_field_1nxyz)
    # if discretization.element_type != 'linear_triangles' :
    #     f_dw=compute_double_well_potential_Gauss_quad(discretization=discretization,
    #                                                 phase_field_1nxyz=phase_field_1nxyz)

    if disp and MPI.COMM_WORLD.rank == 0:
        print('f_dw= '          ' {} '.format(f_dw))  # good in MPI
        print('f_dw / eta= '          ' {} '.format(f_dw / eta))  # good in MPI
    # phase_field_gradient = discretization.apply_gradient_operator(phase_field_1nxyz)
    # f_rho_grad = np.sum(discretization.integrate_over_cell(phase_field_gradient ** 2))
    f_rho_grad = compute_gradient_of_phase_field_potential(discretization=discretization,
                                                           phase_field_1nxyz=phase_field_1nxyz)
    if disp and MPI.COMM_WORLD.rank == 0:
        print('f_rho_grad= '          ' {} '.format(f_rho_grad))  # good in MPI
        print('eta*f_rho_grad= '          ' {} '.format(eta * f_rho_grad))
    f_rho = eta * f_rho_grad + double_well_depth * f_dw / eta
    if disp and MPI.COMM_WORLD.rank == 0:
        print('f_rho= '          ' {} '.format(f_rho))  # good in MPI
    return f_rho


def compute_stress_equivalence_potential(actual_stress_ij,
                                         target_stress_ij,
                                         disp=False):
    # evaluate objective functions
    # f_sigma = ( flux_target-flux_h)^2

    # stress difference potential: actual_stress_ij is homogenized stress
    # stress_difference_ij = actual_stress_ij - target_stress_ij
    stress_difference_ij = target_stress_ij - actual_stress_ij

    f_sigma = np.sum(stress_difference_ij ** 2) / np.sum(target_stress_ij ** 2)
    if disp and MPI.COMM_WORLD.rank == 0:
        print('f_sigma = '          ' {} '.format(f_sigma))  # good in MPI
    return f_sigma


def compute_elastic_energy_equivalence_potential(discretization,
                                                 actual_stress_ij,
                                                 target_stress_ij,
                                                 left_macro_gradient_ij,
                                                 target_energy,
                                                 disp=True):
    # evaluate objective functions
    # stress_difference_ij = actual_stress_ij - target_stress_ij
    stress_difference_ij = target_stress_ij - actual_stress_ij
    actual_energy_difference = np.einsum('ij,ij->...',
                                         left_macro_gradient_ij,
                                         stress_difference_ij
                                         )
    if disp and MPI.COMM_WORLD.rank == 0:
        print('actual_energy_difference = '          ' {} '.format(actual_energy_difference))
    # actual_strain_ijqxyz = macro_gradient_field_ijqxyz + actual_strain_fluctuation_ijqxyz
    # apply quadrature weights
    # actual_strain_ijqxyz = discretization.apply_quadrature_weights_on_gradient_field(actual_strain_ijqxyz)
    # macro_gradient_field_ijqxyz = discretization.apply_quadrature_weights_on_gradient_field(macro_gradient_field_ijqxyz)
    #
    # actual_energy_qxyz = np.einsum('ij,ij...->...',
    #                                stress_difference_ij,
    #                                macro_gradient_field_ijqxyz)
    # # actual_energy_qxyz2 = np.einsum('ij,ij...->...',
    # #                                actual_stress_ij,
    # #                                macro_gradient_field_ijqxyz)
    # actual_energy = discretization.mpi_reduction.sum(actual_energy_qxyz)
    # actual_energy2 = discretization.mpi_reduction.sum(actual_energy_qxyz2)
    # (actual_energy / 2 - target_energy / 2) ** 2 / (target_energy / 2) ** 2
    f_sigma = (actual_energy_difference ** 2) / (target_energy ** 2)
    if disp and MPI.COMM_WORLD.rank == 0:
        print('f_sigma = '          ' {} '.format(f_sigma))  # good in MPI
    return f_sigma


def objective_function_small_strain_testing(discretization,
                                            actual_stress_ij,
                                            target_stress_ij,
                                            phase_field_1nxyz,
                                            eta,
                                            w):
    # evaluate objective functions
    # f = (flux_h -flux_target)^2 + w*eta* int (  (grad(rho))^2 )dx  +    int ( rho^2(1-rho)^2 ) / eta   dx
    # f =  f_sigma + w*eta* f_rho_grad  + f_dw/eta

    # stress difference potential: actual_stress_ij is homogenized stress
    f_sigma = compute_stress_equivalence_potential(actual_stress_ij=actual_stress_ij,
                                                   target_stress_ij=target_stress_ij)
    # double - well potential
    # integrant = (phase_field_1nxyz ** 2) * (1 - phase_field_1nxyz) ** 2
    # f_dw = (np.sum(integrant) / np.prod(integrant.shape)) * discretization.cell.domain_volume
    f_dw = compute_double_well_potential_analytical(discretization=discretization,
                                                    phase_field_1nxyz=phase_field_1nxyz)

    # phase_field_gradient = discretization.apply_gradient_operator(phase_field_1nxyz)
    # f_rho_grad = np.sum(discretization.integrate_over_cell(phase_field_gradient ** 2))
    f_rho_grad = compute_gradient_of_phase_field_potential(discretization=discretization,
                                                           phase_field_1nxyz=phase_field_1nxyz)

    f_rho = eta * f_rho_grad + f_dw / eta
    # if MPI.COMM_WORLD.rank == 0:
    # print('f_sigma linear=  {} '.format(f_sigma))
    # print('f_rho_grad linear=  {} '.format(f_rho_grad))
    # print('f_dw =  linear {} '.format(f_dw))
    #
    # print('f_rho   linear = {} '.format(f_rho))
    # print('w * f_rho linear =  {} '.format(w * f_rho))
    # print('objective_function linear = {} '.format(f_sigma + w * f_rho))

    return f_sigma, f_rho


def objective_function_small_strain_weight(discretization,
                                           actual_stress_ij,
                                           target_stress_ij,
                                           phase_field_1nxyz,
                                           eta,
                                           w):
    # evaluate objective functions
    # f = (flux_h -flux_target)^2 + w*eta* int (  (grad(rho))^2 )dx  +    int ( rho^2(1-rho)^2 ) / eta   dx
    # f =  f_sigma + w*eta* f_rho_grad  + f_dw/eta
    # stress difference potential: actual_stress_ij is homogenized stress
    # stress_difference_ij = actual_stress_ij - target_stress_ij
    stress_difference_ij = (actual_stress_ij - target_stress_ij)
    f_sigma = np.sum(stress_difference_ij ** 2) / np.sum(target_stress_ij ** 2)

    # double - well potential
    # integrant = (phase_field_1nxyz ** 2) * (1 - phase_field_1nxyz) ** 2
    # f_dw = (np.sum(integrant) / np.prod(integrant.shape)) * discretization.cell.domain_volume
    f_dw = compute_double_well_potential_analytical(discretization=discretization,
                                                    phase_field_1nxyz=phase_field_1nxyz)

    phase_field_gradient = discretization.apply_gradient_operator(phase_field_1nxyz)
    f_rho_grad = np.sum(discretization.integrate_over_cell(phase_field_gradient ** 2))

    # gradient_of_phase_field = compute_gradient_of_phase_field(phase_field_gradient)

    f_rho = eta * f_rho_grad + f_dw / eta

    # print('f_sigma linear=  {} '.format(f_sigma))
    print('f_rho_grad linear=  {} '.format(f_rho_grad))
    print('f_dw =  linear {} '.format(f_dw))

    print('f_rho   linear = {} '.format(f_rho))
    print('w * f_rho linear =  {} '.format(w * f_rho))
    # print('objective_function linear = {} '.format(f_sigma + w * f_rho))
    print('weight =  linear {} '.format(w))

    return w * f_sigma + f_rho  # / discretization.cell.domain_volume


def objective_function_small_strain_pixel(discretization,
                                          actual_stress_ij,
                                          target_stress_ij,
                                          phase_field_1nxyz,
                                          eta,
                                          w):
    # evaluate objective functions
    # f = (flux_h -flux_target)^2 + w*eta* int (  (grad(rho))^2 )dx  +    int ( rho^2(1-rho)^2 ) / eta   dx
    # f =  f_sigma + w*eta* f_rho_grad  + f_dw/eta

    # stress difference potential: actual_stress_ij is homogenized stress
    # stress_difference_ij = actual_stress_ij - target_stress_ij
    stress_difference_ij = (actual_stress_ij - target_stress_ij)

    f_sigma = np.sum(stress_difference_ij ** 2) / np.sum(target_stress_ij ** 2)

    # double - well potential
    # integrant = (phase_field_1nxyz ** 2) * (1 - phase_field_1nxyz) ** 2
    # f_dw = (np.sum(integrant) / np.prod(integrant.shape)) * discretization.cell.domain_volume
    f_dw = compute_double_well_potential_nodal(discretization=discretization,
                                               phase_field_1nxyz=phase_field_1nxyz,
                                               eta=1)

    phase_field_gradient = discretization.apply_gradient_operator(phase_field_1nxyz)
    f_rho_grad = np.sum(discretization.integrate_over_cell(phase_field_gradient ** 2))

    # gradient_of_phase_field = compute_gradient_of_phase_field(phase_field_gradient)

    f_rho = eta * f_rho_grad + f_dw / eta

    # print('f_sigma =  {} '.format(f_sigma))
    # print('f_rho_grad =  {} '.format(f_rho_grad))
    # print('f_dw =  {} '.format(f_dw))
    #
    # print('f_rho =  {} '.format(f_rho))
    # print('w * f_rho =  {} '.format(w * f_rho))
    # print('objective_function = {} '.format(f_sigma + w * f_rho))

    return f_sigma + w * f_rho  # / discretization.cell.domain_volume


def compute_double_well_potential_interpolated(discretization, phase_field_1nxyz, eta=1):
    # The double-well potential
    # phase field potential = int ( rho^2(1-rho)^2 ) / eta   dx
    # double - well potential
    # with interpolation for more precise integration
    # integrant = (phase_field_1nxyz ** 2) * (1 - phase_field_1nxyz) ** 2
    # integral = (np.sum(integrant) / np.prod(integrant.shape)) * discretization.cell.domain_volume
    # (ρ^2 (1 - ρ)^2) = ρ^2 - 2ρ^3 + ρ^4
    # TODO[Martin]: do this part first ' integration of double well potential
    if discretization.element_type != 'linear_triangles':
        raise ValueError(
            'precise  evaluation works only for linear triangles. You provided {} '.format(discretization.element_type))

    # TODO make this dimension-less
    x_coords = np.linspace(0, discretization.domain_size[0], discretization.nb_of_pixels[0] + 1, endpoint=True)
    y_coords = np.linspace(0, discretization.domain_size[1], discretization.nb_of_pixels[1] + 1, endpoint=True)
    #  add periodic images
    phase_field_1nxyz_periodic = np.c_[phase_field_1nxyz[0, 0], phase_field_1nxyz[0, 0, :, 0]]  # add a column
    phase_field_1nxyz_periodic = np.r_[phase_field_1nxyz_periodic, [phase_field_1nxyz_periodic[0, :]]]  # add a column

    phase_field_interpolator = sc.interpolate.interp2d(x_coords,
                                                       y_coords,
                                                       phase_field_1nxyz_periodic,
                                                       kind='linear')
    k = 3
    x_coords_interpolated = np.linspace(0, discretization.domain_size[0], k * discretization.nb_of_pixels[0] + 1,
                                        endpoint=True)
    y_coords_interpolated = np.linspace(0, discretization.domain_size[1], k * discretization.nb_of_pixels[1] + 1,
                                        endpoint=True)

    phase_field_interpolated_xyz = np.zeros(
        [1, discretization.nb_nodes_per_pixel, *k * discretization.nb_of_pixels + 1])
    phase_field_interpolated_xyz[0, 0] = phase_field_interpolator(x_coords_interpolated,
                                                                  y_coords_interpolated).transpose()

    integrant_precise = (phase_field_interpolated_xyz ** 2) * (1 - phase_field_interpolated_xyz) ** 2
    integral_precise = (np.sum(integrant_precise[0, 0, :-1, :-1]) / np.prod(
        integrant_precise[0, 0, :-1, :-1].shape)) * discretization.cell.domain_volume

    return integral_precise / eta


def compute_double_well_potential_Gauss_quad(discretization, phase_field_1nxyz):
    # The double-well potential
    # phase field potential = int ( rho^2(1-rho)^2 ) / eta   dx
    # double - well potential
    # with interpolation for more precise integration
    # integrant = (phase_field_1nxyz ** 2) * (1 - phase_field_1nxyz) ** 2
    # integral = (np.sum(integrant) / np.prod(integrant.shape)) * discretization.cell.domain_volume
    # (ρ^2 (1 - ρ)^2) = ρ^2 - 2ρ^3 + ρ^4
    if discretization.element_type != 'linear_triangles' and discretization.element_type != 'linear_triangles_tilled':
        raise ValueError(
            'precise  evaluation works only for linear triangles. You provided {} '.format(discretization.element_type))

    nb_quad_points_per_pixel = 18  #
    quad_points_coord, quad_points_weights = domain.get_gauss_points_and_weights(
        element_type=discretization.element_type,
        nb_quad_points_per_pixel=nb_quad_points_per_pixel)

    Jacobian_matrix = np.diag(discretization.pixel_size)
    Jacobian_det = np.linalg.det(
        Jacobian_matrix)  # this is product of diagonal term of Jacoby transformation matrix
    quad_points_weights = quad_points_weights * Jacobian_det
    # Evaluate field on the quadrature points
    quad_field_fqnxyz, N_at_quad_points_qnijk = discretization.evaluate_field_at_quad_points(
        nodal_field_fnxyz=phase_field_1nxyz,
        quad_field_fqnxyz=None,
        quad_points_coords_iq=quad_points_coord)

    quad_field_fqnxyz = (quad_field_fqnxyz ** 2) * (1 - quad_field_fqnxyz) ** 2
    # Multiply with quadrature weights
    quad_field_fqnxyz = np.einsum('fq...,q->fq...', quad_field_fqnxyz, quad_points_weights)

    return discretization.mpi_reduction.sum(quad_field_fqnxyz)


def compute_double_well_potential_analytical(discretization, phase_field_1nxyz):
    # The double-well potential
    # phase field potential = int ( rho^2(1-rho)^2 ) / eta   dx
    # double - well potential
    # with interpolation for more precise integration
    # (ρ^2 (1 - ρ)^2) = ρ^2 - 2ρ^3 + ρ^4
    # Phase field rho is considered as a linear combination of nodal values phase_field_1nxyz and shape FE functions

    if discretization.element_type != 'linear_triangles' and discretization.element_type != 'linear_triangles_tilled':
        raise ValueError(
            'Analytical evaluation works only for linear triangles. You provided {} '.format(
                discretization.element_type))

    Jacobian_matrix = np.diag(discretization.pixel_size)
    Jacobian_det = np.linalg.det(
        Jacobian_matrix)  # this is product of diagonal term of Jacoby transformation matrix

    # # TODO [ask Lars] is the array copied or is it just "view" on array?
    # rho_00=phase_field_1nxyz[0, 0]
    # rho_01=np.roll(phase_field_1nxyz[0, 0], -1, axis=(0))
    # rho_10=np.roll(phase_field_1nxyz[0, 0], -1, axis=(1))
    # rho_11=np.roll(phase_field_1nxyz[0, 0], -1 * np.array([1, 1]))

    rho_squared_pixel = lambda rho0, rho1, rho2, rho3: (1 / 12) * (rho0 ** 2 + rho1 ** 2 + rho2 ** 2) \
                                                       + (2 / 24) * (rho0 * rho1 + rho0 * rho2 + rho2 * rho1) \
                                                       + (1 / 12) * (rho3 ** 2 + rho1 ** 2 + rho2 ** 2) \
                                                       + (2 / 24) * (rho3 * rho1 + rho3 * rho2 + rho2 * rho1)

    # rho_squared_old = np.sum(rho_squared_pixel(phase_field_1nxyz[0, 0],
    #                                        np.roll(phase_field_1nxyz[0, 0], -1, axis=(0)),
    #                                        np.roll(phase_field_1nxyz[0, 0], -1, axis=(1)),
    #                                        np.roll(phase_field_1nxyz[0, 0], -1 * np.array([1, 1]),
    #                                                axis=(0, 1)))) * Jacobian_det

    rho_squared = discretization.mpi_reduction.sum(rho_squared_pixel(phase_field_1nxyz.s[0, 0],
                                                                     discretization.roll(discretization.fft,
                                                                                         phase_field_1nxyz.s[0, 0],
                                                                                         [-1, 0], axis=(0, 1)),
                                                                     # (2, -1)
                                                                     discretization.roll(discretization.fft,
                                                                                         phase_field_1nxyz.s[0, 0],
                                                                                         [0, -1], axis=(0, 1)),
                                                                     discretization.roll(discretization.fft,
                                                                                         phase_field_1nxyz.s[0, 0],
                                                                                         -1 * np.array([1, 1]),
                                                                                         axis=(0, 1)))) * Jacobian_det
    # print('rho_squared= \n'          ' {} '.format(rho_squared)) #

    rho_qubed_pixel = lambda rho0, rho1, rho2, rho3: (1 / 20) * (rho0 ** 3 + rho1 ** 3 + rho2 ** 3) \
                                                     + (3 / 60) * (rho0 ** 2 * rho1 + rho0 ** 2 * rho2 \
                                                                   + rho1 ** 2 * rho0 + rho1 ** 2 * rho2 \
                                                                   + rho2 ** 2 * rho0 + rho2 ** 2 * rho1) \
                                                     + (6 / 120) * (rho0 * rho1 * rho2) \
                                                     + (1 / 20) * (rho3 ** 3 + rho1 ** 3 + rho2 ** 3) \
                                                     + (3 / 60) * (rho3 ** 2 * rho1 + rho3 ** 2 * rho2 \
                                                                   + rho1 ** 2 * rho3 + rho1 ** 2 * rho2 \
                                                                   + rho2 ** 2 * rho3 + rho2 ** 2 * rho1) \
                                                     + (6 / 120) * (rho3 * rho1 * rho2)

    # rho_qubed_old = np.sum(rho_qubed_pixel(phase_field_1nxyz[0, 0],
    #                                    np.roll(phase_field_1nxyz[0, 0], -1, axis=(0)),
    #                                    np.roll(phase_field_1nxyz[0, 0], -1, axis=(1)),
    #                                    np.roll(phase_field_1nxyz[0, 0], -1 * np.array([1, 1]),
    #                                            axis=(0, 1)))) * Jacobian_det

    rho_qubed = discretization.mpi_reduction.sum(rho_qubed_pixel(phase_field_1nxyz.s[0, 0],
                                                                 discretization.roll(discretization.fft,
                                                                                     phase_field_1nxyz.s[0, 0], [-1, 0],
                                                                                     axis=(0, 1)),
                                                                 discretization.roll(discretization.fft,
                                                                                     phase_field_1nxyz.s[0, 0], [0, -1],
                                                                                     axis=(0, 1)),
                                                                 discretization.roll(discretization.fft,
                                                                                     phase_field_1nxyz.s[0, 0],
                                                                                     -1 * np.array([1, 1]),
                                                                                     axis=(0, 1)))) * Jacobian_det
    # print('rho_qubed= \n'          ' {} '.format(rho_qubed))  #
    rho_quartic_pixel = lambda rho0, rho1, rho2, rho3: (1 / 30) * (rho0 ** 4 + rho1 ** 4 + rho2 ** 4) \
                                                       + (4 / 120) * (rho0 ** 3 * rho1 + rho0 ** 3 * rho2 \
                                                                      + rho1 ** 3 * rho0 + rho1 ** 3 * rho2 \
                                                                      + rho2 ** 3 * rho0 + rho2 ** 3 * rho1) \
                                                       + (6 / 180) * (rho0 ** 2 * rho1 ** 2 \
                                                                      + rho0 ** 2 * rho2 ** 2 \
                                                                      + rho1 ** 2 * rho2 ** 2) \
                                                       + (12 / 360) * (rho0 ** 2 * rho1 * rho2 \
                                                                       + rho0 * rho1 ** 2 * rho2 \
                                                                       + rho0 * rho1 * rho2 ** 2) \
                                                       + (1 / 30) * (rho3 ** 4 + rho1 ** 4 + rho2 ** 4) \
                                                       + (4 / 120) * (rho3 ** 3 * rho1 + rho3 ** 3 * rho2 \
                                                                      + rho1 ** 3 * rho3 + rho1 ** 3 * rho2 \
                                                                      + rho2 ** 3 * rho3 + rho2 ** 3 * rho1) \
                                                       + (6 / 180) * (rho3 ** 2 * rho1 ** 2 \
                                                                      + rho3 ** 2 * rho2 ** 2 \
                                                                      + rho1 ** 2 * rho2 ** 2) \
                                                       + (12 / 360) * (rho3 ** 2 * rho1 * rho2 \
                                                                       + rho3 * rho1 ** 2 * rho2 \
                                                                       + rho3 * rho1 * rho2 ** 2)
    # rho_quartic_old = np.sum(rho_quartic_pixel(phase_field_1nxyz[0, 0],
    #                                        np.roll(phase_field_1nxyz[0, 0], -1, axis=(0)),
    #                                        np.roll(phase_field_1nxyz[0, 0], -1, axis=(1)),
    #                                        np.roll(phase_field_1nxyz[0, 0], -1 * np.array([1, 1]),
    #                                                axis=(0, 1)))) * Jacobian_det

    rho_quartic = discretization.mpi_reduction.sum(rho_quartic_pixel(phase_field_1nxyz.s[0, 0],
                                                                     discretization.roll(discretization.fft,
                                                                                         phase_field_1nxyz.s[0, 0],
                                                                                         [-1, 0],
                                                                                         axis=(0, 1)),
                                                                     discretization.roll(discretization.fft,
                                                                                         phase_field_1nxyz.s[0, 0],
                                                                                         [0, -1],
                                                                                         axis=(0, 1)),
                                                                     discretization.roll(discretization.fft,
                                                                                         phase_field_1nxyz.s[0, 0],
                                                                                         -1 * np.array([1, 1]),
                                                                                         axis=(0, 1)))) * Jacobian_det

    # (ρ^2 (1 - ρ)^2) = ρ^2 - 2ρ^3 + ρ^4
    integral = rho_squared - 2 * rho_qubed + rho_quartic

    return integral


# TODO Delete
# def compute_double_well_potential_analytical_fast(discretization, phase_field_1nxyz, eta=1):
#     # The double-well potential
#     # phase field potential = int ( rho^2(1-rho)^2 ) / eta   dx
#     # double - well potential
#     # with interpolation for more precise integration
#     # (ρ^2 (1 - ρ)^2) = ρ^2 - 2ρ^3 + ρ^4
#     # Phase field rho is considered as a linear combination of nodal values phase_field_1nxyz and shape FE functions
#
#     if discretization.element_type != 'linear_triangles' and discretization.element_type != 'linear_triangles_tilled':
#         raise ValueError(
#             'Analytical evaluation works only for linear triangles. You provided {} '.format(
#                 discretization.element_type))
#
#     Jacobian_matrix = np.diag(discretization.pixel_size)
#     Jacobian_det = np.linalg.det(
#         Jacobian_matrix)  # this is product of diagonal term of Jacoby transformation matrix
#
#     # # TODO [ask Lars] is the array copied or is it just "view" on array?
#     rho0 = phase_field_1nxyz[0, 0]
#     rho1 = np.roll(phase_field_1nxyz[0, 0], -1, axis=(0))
#     rho2 = np.roll(phase_field_1nxyz[0, 0], -1, axis=(1))
#     rho3 = np.roll(phase_field_1nxyz[0, 0], -1 * np.array([1, 1]))
#
#     '''
#     rho_squared_pixel = (1 / 12) * np.sum((rho0 ** 2 + rho3 ** 2) + 2 * (rho1 ** 2 + rho2 ** 2)
#                                           + rho1 * (rho0 + rho2 + rho2 + rho3)
#                                           + rho2 * (rho3 + rho0)
#                                           )
#
#     rho_qubed_pixel = (1 / 20) * np.sum((rho0 ** 3 + rho3 ** 3) + 2 * (rho1 ** 3 + rho2 ** 3)
#                                         + (rho3 ** 2 + rho0 ** 2) * (rho1 + rho2)
#                                         + rho1 ** 2 * (rho0 + 2 * rho2 + rho3)
#                                         + rho2 ** 2 * (rho0 + 2 * rho1 + rho3)
#                                         + rho1 * rho2 * (rho0 + rho3)
#                                         )
#
#     rho_quartic_pixel = (1 / 30) * np.sum(rho0 ** 4 + rho3 ** 4+ 2 * (rho1 ** 4 + rho2 ** 4)
#                                           + rho1 ** 3 * (rho0 + 2 * rho2 + rho3)
#                                           + rho2 ** 3 * (rho0 + 2 * rho1 + rho3)
#                                           + rho0 ** 3 * (rho1 + rho2)
#                                           + rho3 ** 3 * (rho1 + rho2)
#                                           + rho1 ** 2 * (rho0 ** 2 + 2 * rho2 ** 2 + rho3 ** 2)
#                                           + (rho2 ** 2 + rho1 * rho2) * (rho0 ** 2 + rho3 ** 2)
#                                           + (rho0 + rho3) * (rho1 ** 2 * rho2 + rho1 * rho2 ** 2)
#                                           )
#     '''
#     rho_pixel = (((1 / 12) * np.sum((rho0 ** 2 + rho3 ** 2) + 2 * (rho1 ** 2 + rho2 ** 2)
#                                     + rho1 * (rho0 + rho2 + rho2 + rho3)
#                                     + rho2 * (rho3 + rho0)
#                                     )
#                   -
#                   (2 / 20) * np.sum((rho0 ** 3 + rho3 ** 3) + 2 * (rho1 ** 3 + rho2 ** 3)
#                                     + (rho3 ** 2 + rho0 ** 2) * (rho1 + rho2)
#                                     + rho1 ** 2 * (rho0 + 2 * rho2 + rho3)
#                                     + rho2 ** 2 * (rho0 + 2 * rho1 + rho3)
#                                     + rho1 * rho2 * (rho0 + rho3)
#                                     ))
#                  +
#                  (1 / 30) * np.sum(rho0 ** 4 + rho3 ** 4 + 2 * (rho1 ** 4 + rho2 ** 4)
#                                    + rho1 ** 3 * (rho0 + 2 * rho2 + rho3)
#                                    + rho2 ** 3 * (rho0 + 2 * rho1 + rho3)
#                                    + rho0 ** 3 * (rho1 + rho2)
#                                    + rho3 ** 3 * (rho1 + rho2)
#                                    + rho1 ** 2 * (rho0 ** 2 + 2 * rho2 ** 2 + rho3 ** 2)
#                                    + (rho2 ** 2 + rho1 * rho2) * (rho0 ** 2 + rho3 ** 2)
#                                    + (rho0 + rho3) * (rho1 ** 2 * rho2 + rho1 * rho2 ** 2)
#                                    ))
#     # (ρ^2 (1 - ρ)^2) = ρ^2 - 2ρ^3 + ρ^4
#     integral = (rho_pixel) * Jacobian_det
#     return integral / eta


def compute_double_well_potential_nodal(discretization,
                                        phase_field_1nxyz,
                                        eta=1):
    # The double-well potential
    # phase field potential = int ( rho^2(1-rho)^2 ) / eta   dx
    # double - well potential
    integrant = (phase_field_1nxyz.s ** 2) * (1 - phase_field_1nxyz.s) ** 2
    integral = discretization.mpi_reduction.sum(integrant)
    integral = (integral / np.prod(integrant.shape)) * discretization.cell.domain_volume
    return integral / eta


def partial_der_of_double_well_potential_wrt_density_NEW(discretization, phase_field_1nxyz, eta=1):
    # Derivative of the double-well potential with respect to phase-field
    # phase field potential = int ( rho^2(1-rho)^2 )/eta   dx
    # gradient phase field potential = int ((2 * phase_field - 6 * phase_field^2  +  4 * phase_field^3 )) )/eta   dx
    # d/dρ(ρ^2 (1 - ρ)^2) = 2ρ -6ρ^2 + 4ρ^3
    # TODO[Martin]: do this part first ' integration of double well potential
    if discretization.element_type != 'linear_triangles' and discretization.element_type != 'linear_triangles_tilled':
        raise ValueError(
            'precise  evaluation works only for linear triangles. You provided {} '.format(discretization.element_type))
    nb_quad_points_per_pixel = 8
    quad_points_coord, quad_points_weights = domain.get_gauss_points_and_weights(
        element_type=discretization.element_type,
        nb_quad_points_per_pixel=nb_quad_points_per_pixel)

    Jacobian_matrix = np.diag(discretization.pixel_size)
    Jacobian_det = np.linalg.det(
        Jacobian_matrix)  # this is product of diagonal term of Jacoby transformation matrix
    quad_points_weights = quad_points_weights * Jacobian_det
    # Evaluate field on the quadrature points
    quad_field_fqnxyz, N_at_quad_points_qnijk = discretization.evaluate_field_at_quad_points(
        nodal_field_fnxyz=phase_field_1nxyz,
        quad_field_fqnxyz=None,
        quad_points_coords_iq=quad_points_coord)
    # quad_field_fqnxyz = np.einsum('fq...,q->fq...', quad_field_fqnxyz, quad_points_weights)

    quad_field_fqnxyz = (2 * (quad_field_fqnxyz ** 1)
                         - 6 * (quad_field_fqnxyz ** 2)
                         + 4 * (quad_field_fqnxyz ** 3))
    # quad_field_fqnxyz = quad_field_fqnxyz ** 3
    quad_field_fqnxyz = np.einsum('fq...,q->fq...', quad_field_fqnxyz, quad_points_weights)
    # np.sum(quad_field_fqnxyz[0, :nb_quad_points_per_pixel//2, 0, 0, 0])
    # test_phase_field_1nxyz=phase_field_1nxyz*0
    # test_phase_field_1nxyz[0,0,1,1]=1
    # test_quad_field_fqnxyz, N_at_quad_points_qnijk  = discretization.evaluate_field_at_quad_points(
    #     nodal_field_fnxyz=test_phase_field_1nxyz,
    #     quad_field_fqnxyz=None,
    #     quad_points_coords_dq=quad_points_coord)
    nodal_field_u_fnxyz = np.zeros(phase_field_1nxyz.shape)
    for pixel_node in np.ndindex(
            *np.ones([discretization.domain_dimension], dtype=int) * 2):  # iteration over all voxel corners
        pixel_node = np.asarray(pixel_node)
        if discretization.domain_dimension == 2:
            # N_at_quad_points_qnijk
            div_fnxyz_pixel_node = np.einsum('qn,fqnxy->fnxy', N_at_quad_points_qnijk[(..., *pixel_node)],
                                             quad_field_fqnxyz)

            nodal_field_u_fnxyz += np.roll(div_fnxyz_pixel_node, 1 * pixel_node, axis=(2, 3))

        elif discretization.domain_dimension == 3:

            div_fnxyz_pixel_node = np.einsum('dqn,fdqxyz->fnxyz',
                                             N_at_quad_points_qnijk[(..., *pixel_node)],
                                             quad_field_fqnxyz)

            nodal_field_u_fnxyz += np.roll(div_fnxyz_pixel_node, 1 * pixel_node, axis=(2, 3, 4))
            warnings.warn('Gradient transposed is not tested for 3D.')

    # test_quad_field_fqn=test_quad_field_fqnxyz[0,:,0,0,0]
    # el_volume = np.prod(discretization.pixel_size) / 2
    #
    # integrant_fnxyz = 6 * (2 * (phase_field_1nxyz ** 1) * el_volume / 3
    #                        - 6 * (phase_field_1nxyz ** 2) * el_volume / 6
    #                        + 4 * (phase_field_1nxyz ** 3) * el_volume / 10)
    # integrant_fnxyz =
    # integrant_fnxyz = (2 * phase_field_1nxyz * (2 * phase_field_1nxyz * phase_field_1nxyz - 3 * phase_field_1nxyz + 1))

    # integral_fnxyz = (integrant_fnxyz / np.prod(integrant_fnxyz.shape)) * discretization.cell.domain_volume
    # there is no sum here
    # inter = np.sum(quad_field_fqnxyz)
    return nodal_field_u_fnxyz / eta


def partial_der_of_double_well_potential_wrt_density_nodal(discretization,
                                                           phase_field_1nxyz, eta=1):
    # Derivative of the double-well potential with respect to phase-field
    # phase field potential = int ( rho^2(1-rho)^2 )/eta   dx
    # gradient phase field potential = int ((2 * phase_field( + 2 * phase_field^2  -  3 * phase_field +1 )) )/eta   dx
    # d/dρ(ρ^2 (1 - ρ)^2) = 2 ρ (2 ρ^2 - 3 ρ + 1)

    integrant_1nxyz = (
            2 * phase_field_1nxyz.s * (2 * phase_field_1nxyz.s * phase_field_1nxyz.s - 3 * phase_field_1nxyz.s + 1))
    # integral=discretization.mpi_reduction.sum(integrant_1nxyz)

    integral_fnxyz = (integrant_1nxyz / np.prod(integrant_1nxyz.shape)) * discretization.cell.domain_volume
    # there is no sum here
    return integral_fnxyz / eta


def partial_der_of_double_well_potential_wrt_density_analytical(discretization,
                                                                phase_field_1nxyz):
    # Derivative of the double-well potential with respect to phase-field
    # phase field potential = int ( rho^2(1-rho)^2 )/eta   dx
    # gradient phase field potential = int ((2 * phase_field( + 2 * phase_field^2  -  3 * phase_field +1 )) )/eta   dx
    # d/dρ(ρ^2 (1 - ρ)^2) = 2 ρ (2 ρ^2 - 3 ρ + 1)
    # d/dρ(ρ^2 (1 - ρ)^2) = 2ρ -6ρ^2 + 4ρ^3
    if discretization.element_type != 'linear_triangles' and discretization.element_type != 'linear_triangles_tilled':
        raise ValueError(
            'Analytical  evaluation works only for linear triangles. You provided {} '.format(
                discretization.element_type))
    Jacobian_matrix = np.diag(discretization.pixel_size)
    Jacobian_det = np.linalg.det(
        Jacobian_matrix)  # this is product of diagonal term of Jacoby transformation matrix

    # NODAL VALUES OF CONNECTED POINTS
    rho_00 = phase_field_1nxyz.s[0, 0]
    # rho_10 = np.roll(phase_field_1nxyz[0, 0], np.array([-1, 0]), axis=(0, 1))
    # rho_m10 = np.roll(phase_field_1nxyz[0, 0], np.array([1, 0]), axis=(0, 1))
    # rho_01 = np.roll(phase_field_1nxyz[0, 0], np.array([0, -1]), axis=(0, 1))
    # rho_0m1 = np.roll(phase_field_1nxyz[0, 0], np.array([0, 1]), axis=(0, 1))
    # rho_m11 = np.roll(phase_field_1nxyz[0, 0], np.array([1, -1]), axis=(0, 1))
    # rho_1m1 = np.roll(phase_field_1nxyz[0, 0], np.array([-1, 1]), axis=(0, 1))

    rho_10 = discretization.roll(discretization.fft, phase_field_1nxyz.s[0, 0], np.array([-1, 0]), axis=(0, 1))
    rho_m10 = discretization.roll(discretization.fft, phase_field_1nxyz.s[0, 0], np.array([1, 0]), axis=(0, 1))
    rho_01 = discretization.roll(discretization.fft, phase_field_1nxyz.s[0, 0], np.array([0, -1]), axis=(0, 1))
    rho_0m1 = discretization.roll(discretization.fft, phase_field_1nxyz.s[0, 0], np.array([0, 1]), axis=(0, 1))
    rho_m11 = discretization.roll(discretization.fft, phase_field_1nxyz.s[0, 0], np.array([1, -1]), axis=(0, 1))
    rho_1m1 = discretization.roll(discretization.fft, phase_field_1nxyz.s[0, 0], np.array([-1, 1]), axis=(0, 1))

    drho_squared = (rho_00 + 1 / 6 * (rho_10 + rho_m10 + rho_01
                                      + rho_0m1 + rho_m11 + rho_1m1)) * Jacobian_det

    drho_cubed = ((9 / 10) * rho_00 ** 2 \
                  + (1 / 10) * (rho_10 ** 2 + rho_m10 ** 2 + rho_01 ** 2 + rho_0m1 ** 2 + rho_m11 ** 2 + rho_1m1 ** 2) \
                  + (2 / 10) * rho_00 * (rho_10 + rho_m10 + rho_01 + rho_0m1 + rho_m11 + rho_1m1) \
                  + (1 / 20) * (rho_10 * rho_01 + rho_01 * rho_m11 + rho_m11 * rho_m10 \
                                + rho_m10 * rho_0m1 + rho_0m1 * rho_1m1 + rho_1m1 * rho_10) \
                  ) * Jacobian_det

    drho_quartic = ((24 / 30) * rho_00 ** 3 \
                    + (2 / 30) * (
                            rho_10 ** 3 + rho_m10 ** 3 + rho_01 ** 3 + rho_0m1 ** 3 + rho_m11 ** 3 + rho_1m1 ** 3) \
                    + (6 / 30) * rho_00 ** 2 * (rho_10 + rho_m10 + rho_01 + rho_0m1 + rho_m11 + rho_1m1) \
                    + (4 / 30) * rho_00 * (
                            rho_10 ** 2 + rho_m10 ** 2 + rho_01 ** 2 + rho_0m1 ** 2 + rho_m11 ** 2 + rho_1m1 ** 2) \
                    + (1 / 30) * (rho_10 ** 2 * rho_01 + rho_01 ** 2 * rho_m11 + rho_m11 ** 2 * rho_m10 \
                                  + rho_m10 ** 2 * rho_0m1 + rho_0m1 ** 2 * rho_1m1 + rho_1m1 ** 2 * rho_10) \
                    + (1 / 30) * (rho_10 * rho_01 ** 2 + rho_01 * rho_m11 ** 2 + rho_m11 * rho_m10 ** 2 \
                                  + rho_m10 * rho_0m1 ** 2 + rho_0m1 * rho_1m1 ** 2 + rho_1m1 * rho_10 ** 2) \
                    + (2 / 30) * rho_00 * (rho_10 * rho_01 + rho_01 * rho_m11 + rho_m11 * rho_m10 \
                                           + rho_m10 * rho_0m1 + rho_0m1 * rho_1m1 + rho_1m1 * rho_10)
                    ) * Jacobian_det

    return (drho_squared - 2 * drho_cubed + drho_quartic)


def partial_der_of_double_well_potential_wrt_density_Gauss_quad(discretization, phase_field_1nxyz):
    raise ValueError(
        'NOT FINISHED{} '.format(discretization.element_type))
    # The double-well potential
    # phase field potential = int ( rho^2(1-rho)^2 ) / eta   dx
    # double - well potential
    # with interpolation for more precise integration
    # integrant = (phase_field_1nxyz ** 2) * (1 - phase_field_1nxyz) ** 2
    # integral = (np.sum(integrant) / np.prod(integrant.shape)) * discretization.cell.domain_volume
    # (ρ^2 (1 - ρ)^2) = ρ^2 - 2ρ^3 + ρ^4
    if discretization.element_type != 'linear_triangles' and discretization.element_type != 'linear_triangles_tilled':
        raise ValueError(
            'precise  evaluation works only for linear triangles. You provided {} '.format(discretization.element_type))

    nb_quad_points_per_pixel = 18  #
    quad_points_coord, quad_points_weights = domain.get_gauss_points_and_weights(
        element_type=discretization.element_type,
        nb_quad_points_per_pixel=nb_quad_points_per_pixel)

    Jacobian_matrix = np.diag(discretization.pixel_size)
    Jacobian_det = np.linalg.det(
        Jacobian_matrix)  # this is product of diagonal term of Jacoby transformation matrix
    quad_points_weights = quad_points_weights * Jacobian_det
    # Evaluate field on the quadrature points
    quad_field_fqnxyz, N_at_quad_points_qnijk = discretization.evaluate_field_at_quad_points(
        nodal_field_fnxyz=phase_field_1nxyz,
        quad_field_fqnxyz=None,
        quad_points_coords_iq=quad_points_coord)

    quad_field_fqnxyz = (quad_field_fqnxyz ** 2) * (1 - quad_field_fqnxyz) ** 2
    # Multiply with quadrature weights
    quad_field_fqnxyz = np.einsum('fq...,q->fq...', quad_field_fqnxyz, quad_points_weights)

    return discretization.mpi_reduction.sum(quad_field_fqnxyz)


def partial_der_of_double_well_potential_wrt_density_analytical_fast(discretization, phase_field_1nxyz, eta=1):
    # Derivative of the double-well potential with respect to phase-field
    # phase field potential = int ( rho^2(1-rho)^2 )/eta   dx
    # gradient phase field potential = int ((2 * phase_field( + 2 * phase_field^2  -  3 * phase_field +1 )) )/eta   dx
    # d/dρ(ρ^2 (1 - ρ)^2) = 2 ρ (2 ρ^2 - 3 ρ + 1)
    # d/dρ(ρ^2 (1 - ρ)^2) = 2ρ -6ρ^2 + 4ρ^3
    if discretization.element_type != 'linear_triangles':
        raise ValueError(
            'Analytical  evaluation works only for linear triangles. You provided {} '.format(
                discretization.element_type))
    Jacobian_matrix = np.diag(discretization.pixel_size)
    Jacobian_det = np.linalg.det(
        Jacobian_matrix)  # this is product of diagonal term of Jacoby transformation matrix
    # allocate empty field for partial derivative
    ddouble_well_drho_1nxyz = np.zeros(phase_field_1nxyz.shape)
    # integral_fnxyz =  (2 * phase_field_1nxyz[0, 0] \
    #                 - 6 * phase_field_1nxyz[0, 0] ** 2 * 6 / 12 \
    #                 + 4 * phase_field_1nxyz[0, 0] ** 3 * 6 / 20) *Jacobian_det
    # NODAL VALUES OF CONNECTED POINTS
    rho0 = phase_field_1nxyz[0, 0]
    rho1 = np.roll(phase_field_1nxyz[0, 0], -1, axis=(0))
    rho2 = np.roll(phase_field_1nxyz[0, 0], -1, axis=(1))
    rho3 = np.roll(phase_field_1nxyz[0, 0], -1 * np.array([1, 1]), axis=(0, 1))

    # partial derivative of a quadratic term
    # d/dρ(ρ^2) =
    ddouble_well_drho_1nxyz[0, 0] += (1 / 6) * (rho0) + (1 / 12) * (rho1 + rho2)
    ddouble_well_drho_1nxyz[0, 0] += np.roll((1 / 3) * (rho1) + (1 / 12) * (rho0 + 2 * rho2 + rho3), 1, axis=(0))
    ddouble_well_drho_1nxyz[0, 0] += np.roll((1 / 3) * (rho2) + (1 / 12) * (rho0 + 2 * rho1 + rho3), 1, axis=(1))
    ddouble_well_drho_1nxyz[0, 0] += np.roll((1 / 6) * (rho3) + (1 / 12) * (rho1 + rho2), np.array([1, 1]), axis=(0, 1))

    # partial derivative of a cibic term
    # d/dρ(ρ^3) =
    ddouble_well_drho_1nxyz[0, 0] += - 2 * ((1 / 20) * (3 * rho0 ** 2 + rho1 ** 2 + rho2 ** 2)
                                            + (1 / 10) * rho0 * (rho1 + rho2)
                                            + (1 / 20) * rho1 * rho2)

    ddouble_well_drho_1nxyz[0, 0] += - 2 * np.roll(
        ((1 / 20) * (rho0 ** 2 + 6 * rho1 ** 2 + 2 * rho2 ** 2 + rho3 ** 2)
         + (1 / 10) * rho1 * (rho0 + 2 * rho2 + rho3)
         + (1 / 20) * rho2 * (rho0 + rho3)), 1, axis=(0))

    ddouble_well_drho_1nxyz[0, 0] += - 2 * np.roll(
        ((1 / 20) * (rho0 ** 2 + 2 * rho1 ** 2 + 6 * rho2 ** 2 + rho3 ** 2)
         + (1 / 10) * rho2 * (rho0 + 2 * rho1 + rho3)
         + (1 / 20) * rho1 * (rho0 + rho3)), 1, axis=(1))

    ddouble_well_drho_1nxyz[0, 0] += - 2 * np.roll((
            (1 / 20) * (3 * rho3 ** 2 + rho1 ** 2 + rho2 ** 2)
            + (1 / 10) * rho3 * (rho1 + rho2)
            + (1 / 20) * rho1 * rho2), np.array([1, 1]), axis=(0, 1))

    # partial derivative of a cibic term
    # d/dρ(ρ^4) =

    ddouble_well_drho_1nxyz[0, 0] += ((1 / 30) * (4 * rho0 ** 3 + rho1 ** 3 + rho2 ** 3)
                                      + (1 / 10) * rho0 ** 2 * (rho1 + rho2)
                                      + (1 / 30) * rho1 ** 2 * (2 * rho0 + rho2)
                                      + (1 / 30) * rho2 ** 2 * (2 * rho0 + rho1)
                                      + (2 / 30) * rho0 * rho1 * rho2)

    ddouble_well_drho_1nxyz[0, 0] += np.roll(
        ((1 / 30) * (rho0 ** 3 + 8 * rho1 ** 3 + 2 * rho2 ** 3 + rho3 ** 3)
         + (1 / 30) * rho0 ** 2 * (2 * rho1 + rho2)
         + (1 / 30) * rho3 ** 2 * (2 * rho1 + rho2)
         + (1 / 10) * rho1 ** 2 * (rho0 + 2 * rho2 + rho3)
         + (1 / 30) * rho2 ** 2 * (rho0 + 4 * rho1 + rho3)
         + (2 / 30) * rho1 * rho2 * (rho0 + rho3)
         ), 1, axis=(0))

    ddouble_well_drho_1nxyz[0, 0] += np.roll(
        ((1 / 30) * (rho0 ** 3 + 2 * rho1 ** 3 + 8 * rho2 ** 3 + rho3 ** 3)
         + (1 / 30) * rho0 ** 2 * (2 * rho2 + rho1)
         + (1 / 30) * rho3 ** 2 * (2 * rho2 + rho1)
         + (1 / 10) * rho2 ** 2 * (rho0 + 2 * rho1 + rho3)
         + (1 / 30) * rho1 ** 2 * (rho0 + 4 * rho2 + rho3)
         + (2 / 30) * rho2 * rho1 * (rho0 + rho3)
         ), 1, axis=(1))

    ddouble_well_drho_1nxyz[0, 0] += np.roll(
        ((1 / 30) * (4 * rho3 ** 3 + rho1 ** 3 + rho2 ** 3)
         + (1 / 10) * rho3 ** 2 * (rho1 + rho2)
         + (1 / 30) * rho1 ** 2 * (2 * rho3 + rho2)
         + (1 / 30) * rho2 ** 2 * (2 * rho3 + rho1)
         + (2 / 30) * rho3 * rho1 * rho2), np.array([1, 1]), axis=(0, 1))

    return ddouble_well_drho_1nxyz * Jacobian_det / eta


def partial_der_of_double_well_potential_wrt_density_analytical_fast2(discretization, phase_field_1nxyz, eta=1):
    # Derivative of the double-well potential with respect to phase-field
    # phase field potential = int ( rho^2(1-rho)^2 )/eta   dx
    # gradient phase field potential = int ((2 * phase_field( + 2 * phase_field^2  -  3 * phase_field +1 )) )/eta   dx
    # d/dρ(ρ^2 (1 - ρ)^2) = 2 ρ (2 ρ^2 - 3 ρ + 1)
    # d/dρ(ρ^2 (1 - ρ)^2) = 2ρ -6ρ^2 + 4ρ^3
    if discretization.element_type != 'linear_triangles':
        raise ValueError(
            'Analytical  evaluation works only for linear triangles. You provided {} '.format(
                discretization.element_type))
    Jacobian_matrix = np.diag(discretization.pixel_size)
    Jacobian_det = np.linalg.det(
        Jacobian_matrix)  # this is product of diagonal term of Jacoby transformation matrix
    # allocate empty field for partial derivative
    ddouble_well_drho_1nxyz = np.zeros(phase_field_1nxyz.shape)
    # integral_fnxyz =  (2 * phase_field_1nxyz[0, 0] \
    #                 - 6 * phase_field_1nxyz[0, 0] ** 2 * 6 / 12 \
    #                 + 4 * phase_field_1nxyz[0, 0] ** 3 * 6 / 20) *Jacobian_det
    # NODAL VALUES OF CONNECTED POINTS
    rho0 = phase_field_1nxyz[0, 0]
    rho1 = np.roll(phase_field_1nxyz[0, 0], -1, axis=(0))
    rho2 = np.roll(phase_field_1nxyz[0, 0], -1, axis=(1))
    rho3 = np.roll(phase_field_1nxyz[0, 0], -1 * np.array([1, 1]), axis=(0, 1))

    # partial derivative of a quadratic term
    # d/dρ(ρ^2) =
    ddouble_well_drho_1nxyz[0, 0] += ((1 / 6) * (rho0) + (1 / 12) * (rho1 + rho2)
                                      - 2 * ((1 / 20) * (3 * rho0 ** 2 + rho1 ** 2 + rho2 ** 2)
                                             + (1 / 10) * rho0 * (rho1 + rho2)
                                             + (1 / 20) * rho1 * rho2)
                                      + ((1 / 30) * (4 * rho0 ** 3 + rho1 ** 3 + rho2 ** 3)
                                         + (1 / 10) * rho0 ** 2 * (rho1 + rho2)
                                         + (1 / 30) * rho1 ** 2 * (2 * rho0 + rho2)
                                         + (1 / 30) * rho2 ** 2 * (2 * rho0 + rho1)
                                         + (2 / 30) * rho0 * rho1 * rho2))

    ddouble_well_drho_1nxyz[0, 0] += np.roll(
        ((1 / 3) * (rho1) + (1 / 12) * (rho0 + 2 * rho2 + rho3)
         - 2 * ((1 / 20) * (rho0 ** 2 + 6 * rho1 ** 2 + 2 * rho2 ** 2 + rho3 ** 2)
                + (1 / 10) * rho1 * (rho0 + 2 * rho2 + rho3)
                + (1 / 20) * rho2 * (rho0 + rho3))
         + ((1 / 30) * (rho0 ** 3 + 8 * rho1 ** 3 + 2 * rho2 ** 3 + rho3 ** 3)
            + (1 / 30) * rho0 ** 2 * (2 * rho1 + rho2)
            + (1 / 30) * rho3 ** 2 * (2 * rho1 + rho2)
            + (1 / 10) * rho1 ** 2 * (rho0 + 2 * rho2 + rho3)
            + (1 / 30) * rho2 ** 2 * (rho0 + 4 * rho1 + rho3)
            + (2 / 30) * rho1 * rho2 * (rho0 + rho3))
         ), 1, axis=(0))

    ddouble_well_drho_1nxyz[0, 0] += np.roll(
        ((1 / 3) * (rho2) + (1 / 12) * (rho0 + 2 * rho1 + rho3)
         - 2 * ((1 / 20) * (rho0 ** 2 + 2 * rho1 ** 2 + 6 * rho2 ** 2 + rho3 ** 2)
                + (1 / 10) * rho2 * (rho0 + 2 * rho1 + rho3)
                + (1 / 20) * rho1 * (rho0 + rho3))
         + ((1 / 30) * (rho0 ** 3 + 2 * rho1 ** 3 + 8 * rho2 ** 3 + rho3 ** 3)
            + (1 / 30) * rho0 ** 2 * (2 * rho2 + rho1)
            + (1 / 30) * rho3 ** 2 * (2 * rho2 + rho1)
            + (1 / 10) * rho2 ** 2 * (rho0 + 2 * rho1 + rho3)
            + (1 / 30) * rho1 ** 2 * (rho0 + 4 * rho2 + rho3)
            + (2 / 30) * rho2 * rho1 * (rho0 + rho3)
            )
         ), 1, axis=(1))

    ddouble_well_drho_1nxyz[0, 0] += np.roll(
        (1 / 6) * (rho3) + (1 / 12) * (rho1 + rho2)
        - 2 * ((1 / 20) * (3 * rho3 ** 2 + rho1 ** 2 + rho2 ** 2)
               + (1 / 10) * rho3 * (rho1 + rho2)
               + (1 / 20) * rho1 * rho2)
        + ((1 / 30) * (4 * rho3 ** 3 + rho1 ** 3 + rho2 ** 3)
           + (1 / 10) * rho3 ** 2 * (rho1 + rho2)
           + (1 / 30) * rho1 ** 2 * (2 * rho3 + rho2)
           + (1 / 30) * rho2 ** 2 * (2 * rho3 + rho1)
           + (2 / 30) * rho3 * rho1 * rho2)
        , np.array([1, 1]), axis=(0, 1))

    return ddouble_well_drho_1nxyz * Jacobian_det / eta


def compute_gradient_of_phase_field_potential_NEW_WIP(discretization, phase_field_1nxyz, eta=1):
    # Input: phase_field [1,n,x,y,z]
    # Output: potential [1]
    # phase field gradient potential = int (  (grad(rho))^2 )    dx
    phase_field_gradient_ijqxyz = discretization.apply_gradient_operator(phase_field_1nxyz)

    phase_field_gradient_squared_qxyz = np.einsum('ij...,ij...->...',
                                                  phase_field_gradient_ijqxyz,
                                                  phase_field_gradient_ijqxyz)

    f_rho_grad = np.sum(discretization.integrate_over_cell(phase_field_gradient_squared_qxyz))
    return eta * f_rho_grad


def compute_gradient_of_phase_field_potential(discretization, phase_field_1nxyz):
    # Input: phase_field [1,n,x,y,z]
    # Output: potential [1]
    # phase field gradient potential = int (  (grad(rho))^2 )    dx
    # (re) allocate field  for gradient
    phase_field_gradient_ijqxyz = discretization.get_gradient_of_scalar_field(
        name='compute_gradient_of_phase_field_potential')
    phase_field_gradient_ijqxyz.s.fill(0)

    phase_field_gradient_ijqxyz = discretization.apply_gradient_operator(u_inxyz=phase_field_1nxyz,
                                                                         grad_u_ijqxyz=phase_field_gradient_ijqxyz)
    f_rho_grad = np.sum(discretization.integrate_over_cell(phase_field_gradient_ijqxyz.s ** 2))
    return f_rho_grad


def partial_derivative_of_gradient_of_phase_field_potential(discretization,
                                                            phase_field_1nxyz,
                                                            output_1nxyz):
    # Input: phase_field [1,n,x,y,z]
    # Output: ∂ potential/ ∂ pha adjoint_potential ase_field [1,n,x,y,z] # Note: one potential per phase field DOF

    # partial derivative of  phase field gradient potential = 2/eta int (  (grad(rho))^2 )    dx
    # Compute       grad (rho). grad I  without using I
    #
    #  (D rho, D I) ==  ( D I, D rho) and thus  == I D_t D rho
    # I try to implement it in the way = 2/eta (int I D_t D rho )
    phase_field_grad_1jqxyz = discretization.get_gradient_of_scalar_field(
        name='temporary_partial_derivative_of_gradient_of_phase_field_potential')

    discretization.conv_op.apply(nodal_field=phase_field_1nxyz,
                                 quadrature_point_field=phase_field_grad_1jqxyz)

    weights = discretization.quadrature_weights

    discretization.conv_op.transpose(quadrature_point_field=phase_field_grad_1jqxyz,
                                     nodal_field=output_1nxyz,
                                     weights=weights)

    # phase_field_gradient = discretization.apply_gradient_operator(phase_field_1nxyz)
    # phase_field_gradient = discretization.apply_quadrature_weights_on_gradient_field(phase_field_gradient)
    # Dt_D_rho = discretization.apply_gradient_transposed_operator(phase_field_gradient)
    output_1nxyz.s *= 2
    # integrated_Dt_D_rho =  #/ np.prod(Dt_D_rho.shape)) * discretization.cell.domain_volume
    # return 2 * Dt_D_rho
    return output_1nxyz


##
def objective_function_stress_equivalence(discretization,
                                          actual_stress_ij,
                                          target_stress_ij):
    # Input: phase_field [1,n,x,y,z]
    # Output: f_sigma  [1]   == stress difference =  (Sigma_target-Sigma_homogenized,Sigma_target-Sigma_homogenized)

    # stress difference potential: actual_stress_ij is homogenized stress
    # stress_difference_ij = actual_stress_ij - target_stress_ij

    # f_sigma = np.sum(stress_difference_ij ** 2)
    # f_sigma = np.einsum('ij,ij->', stress_difference_ij, stress_difference_ij)
    stress_difference_ij = (target_stress_ij - actual_stress_ij)

    f_sigma = np.sum(stress_difference_ij ** 2) / np.sum(target_stress_ij ** 2)
    # can be done np.tensordot(stress_difference, stress_difference,axes=2)
    return f_sigma


def partial_derivative_of_energy_equivalence_wrt_phase_field_FE(discretization,
                                                                base_material_data_ijkl,
                                                                displacement_field_fnxyz,
                                                                macro_gradient_field_ijqxyz,
                                                                phase_field_1nxyz,
                                                                target_stress_ij,
                                                                actual_stress_ij,
                                                                left_macro_gradient_ij,
                                                                target_energy,
                                                                p):
    # Input: phase_field [1,n,x,y,z]
    #        material_data_field [d,d,d,d,q,x,y,z] - elasticity
    #        target_stress_ij [d,d]
    #        actual_stress_ij [d,d]
    # -- -- -- -- -- -- -- -- -- -- --

    # Gradient of material data with respect to phase field
    # % interpolation of rho into quad points
    # -----    stress difference potential ----- #
    # Gradient of material data with respect to phase field
    phase_field_at_quad_poits_1qnxyz, N_at_quad_points_qnijk = discretization.evaluate_field_at_quad_points(
        nodal_field_fnxyz=phase_field_1nxyz.s,
        quad_field_fqnxyz=None,
        quad_points_coords_iq=None)  # TODO[Martin] missing exact integration
    # apply material distribution
    # material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * np.power(phase_field[0, 0], p)

    # dmaterial_data_field_drho_ijklqxyz = material_data_field_ijklqxyz[..., :, :] * (
    #         p * np.power(phase_field_1nxyz[0, 0], (p - 1)))
    dmaterial_data_field_drho_ijklqxyz = discretization.get_material_data_size_field(
        name='data_field_in_partial_derivative_of_energy_equivalence_wrt_phase_field_FE')
    dmaterial_data_field_drho_ijklqxyz.s = base_material_data_ijkl[..., np.newaxis, np.newaxis, np.newaxis] * np.power(
        p * phase_field_at_quad_poits_1qnxyz, p - 1)[0, :, 0, ...]
    # I consider linear interpolation of material  C_ijkl= p*rho**(p-1) C^0_ijkl
    # so  ∂ C_ijkl/ ∂ rho = 1* C^0_ijkl
    # p = 2
    # dmaterial_data_field_ijklqxyz = material_data_field_ijklqxyz[..., :, :] * (
    #        p * np.power(phase_field_1nxyz[0, 0], (p - 1)))

    # int(∂ C/ ∂ rho_i  * (macro_grad + micro_grad)) dx / | domain |

    # compute strain field from to displacement and macro gradient
    strain_ijqxyz = discretization.get_displacement_gradient_sized_field(name='strain_ijqxyz_local_at_pdofsewpf')
    strain_ijqxyz = discretization.apply_gradient_operator_symmetrized(u_inxyz=displacement_field_fnxyz,
                                                                       grad_u_ijqxyz=strain_ijqxyz)
    strain_ijqxyz.s = macro_gradient_field_ijqxyz.s + strain_ijqxyz.s

    # strain_fluctuation_ijqxyz = discretization.apply_gradient_operator_symmetrized(displacement_field_fnxyz)
    # strain_ijqxyz = macro_gradient_field_ijqxyz + strain_fluctuation_ijqxyz

    # compute stress field
    # ddot42 = lambda A4, B2: np.einsum('ijklxyz,lkxyz  ->ijxyz  ', A4, B2)
    # stress_field_ijqxyz_pixel = np.einsum('ijkl...,lk...->ij...', dmaterial_data_field_ijklqxyz, strain_ijqxyz)
    # stress_field_ijqxyz_FE = np.einsum('ijkl...,lk...->ij...', dmaterial_data_field_drho_ijklqxyz_FE, strain_ijqxyz)
    # stress_field_ijqxyz_FE = discretization.apply_material_data(dmaterial_data_field_drho_ijklqxyz_FE, strain_ijqxyz)

    # Get the stress field (in the strain field name)
    strain_ijqxyz.s = discretization.apply_material_data_elasticity(material_data=dmaterial_data_field_drho_ijklqxyz,
                                                                    gradient_field=strain_ijqxyz)
    # apply quadrature weights
    # stress_ijqxyz_pixel = discretization.apply_quadrature_weights_on_gradient_field(stress_field_ijqxyz_pixel)
    # stress_field_ijqxyz_FE = np.ones(stress_field_ijqxyz_FE.shape)
    # TODO not sure if this should be here
    strain_ijqxyz.s = discretization.apply_quadrature_weights_on_gradient_field(grad_field=strain_ijqxyz.s)

    # stress differenc
    double_contraction_stress_qxyz_FE = np.einsum('ij,ijqxy...->qxy...',
                                                  left_macro_gradient_ij,
                                                  strain_ijqxyz.s)

    dfstress_drho_OLD = discretization.get_scalar_field(name='dfstress_drho_OLD_')
    dfstress_drho_OLD.s.fill(0)
    for pixel_node in np.ndindex(
            *np.ones([discretization.domain_dimension], dtype=int) * 2):  # iteration over all voxel corners
        pixel_node = np.asarray(pixel_node)
        if discretization.domain_dimension == 2:
            # N_at_quad_points_qnijk
            dfstress_drho_pixel_node = np.einsum('qn,qxy->nxy',
                                                 N_at_quad_points_qnijk[(..., *pixel_node)],
                                                 double_contraction_stress_qxyz_FE)

            # dg_drho_nxyz += np.roll(div_fnxyz_pixel_node, 1 * pixel_node, axis=(1, 2))
            dfstress_drho_OLD.s += discretization.roll(discretization.fft, dfstress_drho_pixel_node, 1 * pixel_node,
                                                       axis=(0, 1))
        elif discretization.domain_dimension == 3:
            dfstress_drho_pixel_node = np.einsum('dqn,dqxyz->nxyz',
                                                 N_at_quad_points_qnijk[(..., *pixel_node)],
                                                 double_contraction_stress_qxyz_FE)

            # dg_drho_nxyz += np.roll(div_fnxyz_pixel_node, 1 * pixel_node, axis=(1, 2, 3))
            dfstress_drho_OLD.s += discretization.roll(discretization.fft, dfstress_drho_pixel_node, 1 * pixel_node,
                                                       axis=(0, 1, 2))

            warnings.warn('Gradient transposed is not tested for 3D.')

    return -2 * dfstress_drho_OLD.s / discretization.cell.domain_volume / (target_energy ** 2)


def partial_derivative_of_objective_function_stress_equivalence_wrt_phase_field_FE(discretization,
                                                                                   material_data_field_ijkl,
                                                                                   displacement_field_fnxyz,
                                                                                   macro_gradient_field_ijqxyz,
                                                                                   phase_field_1nxyz,
                                                                                   target_stress_ij,
                                                                                   actual_stress_ij,
                                                                                   p):
    # Input: phase_field [1,n,x,y,z]
    #        material_data_field [d,d,d,d,q,x,y,z] - elasticity
    #        target_stress_ij [d,d]
    #        actual_stress_ij [d,d]
    # -- -- -- -- -- -- -- -- -- -- --

    # Gradient of material data with respect to phase field
    # % interpolation of rho into quad points
    # -----    stress difference potential ----- #
    # Gradient of material data with respect to phase field
    phase_field_at_quad_poits_1qnxyz, N_at_quad_points_qnijk = discretization.evaluate_field_at_quad_points(
        nodal_field_fnxyz=phase_field_1nxyz.s,
        quad_field_fqnxyz=None,
        quad_points_coords_iq=None)  # TODO[Martin] missing exact integration
    # apply material distribution
    # material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * np.power(phase_field[0, 0], p)

    # dmaterial_data_field_drho_ijklqxyz = material_data_field_ijklqxyz[..., :, :] * (
    #         p * np.power(phase_field_1nxyz[0, 0], (p - 1)))

    # dmaterial_data_field_drho_ijklqxyz_FE = material_data_field_ijkl[..., :, :, :] * np.power(
    #     p * phase_field_at_quad_poits_1qnxyz, p - 1)[0, :, 0, ...]

    dmaterial_data_field_drho_ijklqxyz_FE = discretization.get_material_data_size_field(
        name='data_field_drho_partial_derivative_of_objective_function_stress_equivalence_wrt_phase_field_FE')
    dmaterial_data_field_drho_ijklqxyz_FE.s = material_data_field_ijkl[..., np.newaxis, np.newaxis, np.newaxis] * \
                                              np.power(
                                                  p * phase_field_at_quad_poits_1qnxyz, p - 1)[0, :, 0, ...]

    # I consider linear interpolation of material  C_ijkl= p*rho**(p-1) C^0_ijkl
    # so  ∂ C_ijkl/ ∂ rho = 1* C^0_ijkl
    # p = 2
    # dmaterial_data_field_ijklqxyz = material_data_field_ijklqxyz[..., :, :] * (
    #        p * np.power(phase_field_1nxyz[0, 0], (p - 1)))

    # int(∂ C/ ∂ rho_i  * (macro_grad + micro_grad)) dx / | domain |

    # compute strain field from to displacement and macro gradient
    strain_ijqxyz = discretization.get_displacement_gradient_sized_field(name='strain_ijqxyz_local_at_pdofsewpf')

    strain_ijqxyz = discretization.apply_gradient_operator_symmetrized(u_inxyz=displacement_field_fnxyz,
                                                                       grad_u_ijqxyz=strain_ijqxyz)
    strain_ijqxyz.s = macro_gradient_field_ijqxyz.s + strain_ijqxyz.s

    # compute stress field
    # ddot42 = lambda A4, B2: np.einsum('ijklxyz,lkxyz  ->ijxyz  ', A4, B2)
    # stress_field_ijqxyz_pixel = np.einsum('ijkl...,lk...->ij...', dmaterial_data_field_ijklqxyz, strain_ijqxyz)
    # stress_field_ijqxyz_FE = np.einsum('ijkl...,lk...->ij...', dmaterial_data_field_drho_ijklqxyz_FE, strain_ijqxyz)
    # stress_field_ijqxyz_FE = discretization.apply_material_data_elasticity(
    #     material_data=dmaterial_data_field_drho_ijklqxyz_FE,
    #     gradient_field=strain_ijqxyz)
    strain_ijqxyz.s = np.einsum('ijkl...,lk...->ij...',
                                dmaterial_data_field_drho_ijklqxyz_FE.s,
                                strain_ijqxyz.s)
    # apply quadrature weights
    # stress_ijqxyz_pixel = discretization.apply_quadrature_weights_on_gradient_field(stress_field_ijqxyz_pixel)
    # stress_field_ijqxyz_FE = np.ones(stress_field_ijqxyz_FE.shape)
    # this is actually stress
    strain_ijqxyz.s = discretization.apply_quadrature_weights_on_gradient_field(strain_ijqxyz.s)

    # stress difference
    stress_difference_ij = target_stress_ij - actual_stress_ij

    # double_contraction_stress_qxyz_pixel = np.einsum('ij,ijqxy...->qxy...',
    #                                            stress_difference_ij,
    #                                            stress_ijqxyz_pixel)

    double_contraction_stress_qxyz_FE = np.einsum('ij,ijqxy...->qxy...',
                                                  stress_difference_ij,
                                                  strain_ijqxyz.s)
    # np.einsum('ijqxyz  ,jiqxyz  ->xyz    ', A2, B2)
    # Average over quad points in pixel !!!
    # partial_derivative = partial_derivative.mean(axis=0)
    # partial_derivative_xyz = double_contraction_stress_qxyz_pixel.sum(axis=0)
    # shape_FE = np.asarray(stress_ijqxyz_FE.shape)  # TODO [MARTIN] fix sizes
    # shape_FE[2] = 1
    # # np.squeeze(A, axis=1)
    # partial_derivative_xyz_FE = np.zeros(shape_FE)
    #
    # # partial_derivative_xyz_mpi = np.zeros(phase_field_1nxyz.shape)
    # for pixel_node in np.ndindex(discretization.nb_unique_nodes_per_pixel,
    #                              *np.ones([discretization.domain_dimension],
    #                                       dtype=int) * 2):  # iteration over all voxel corners
    #     pixel_node = np.asarray(pixel_node)
    #     if discretization.domain_dimension == 2:
    #         # N_at_quad_points_qnijk
    #         # multiply with basis function that corresponds to pixel node
    #         # + sum over all quad points
    #         stress_times_basis_pixel_node_ijxy = np.einsum('q,ijqxy->ijxy',
    #                                                        N_at_quad_points_qnijk[(..., *pixel_node)],
    #                                                        stress_ijqxyz_FE)
    #
    #         partial_derivative_xyz_FE[..., 0, :, :] += discretization.roll(discretization.fft,
    #                                                                        stress_times_basis_pixel_node_ijxy,
    #                                                                        1 * pixel_node[1:], axis=(0, 1))
    #
    #
    #     elif discretization.domain_dimension == 3:
    #
    #         stress_times_basis_pixel_node_ijxy = np.einsum('q,ijqxy->ijxy',
    #                                                        N_at_quad_points_qnijk[(..., *pixel_node)],
    #                                                        stress_ijqxyz_FE)
    #
    #         # partial_derivative_xyz += np.roll(div_fnxyz_pixel_node, 1 * pixel_node, axis=(1, 2, 3))
    #         partial_derivative_xyz_FE[..., 0, :, :, :] += discretization.roll(discretization.fft,
    #                                                                           stress_times_basis_pixel_node_ijxy,
    #                                                                           1 * pixel_node[1:], axis=(0, 1, 2))
    #         warnings.warn('Gradient transposed is not tested for 3D.')
    #
    # partial_derivative_xyz_FE_sum = np.einsum('ij,ijnxy...->nxy...',
    #                                           stress_difference_ij,
    #                                           partial_derivative_xyz_FE)
    # Average over quad points in pixel !!!
    dfstress_drho_OLD = np.zeros(phase_field_1nxyz.shape)
    for pixel_node in np.ndindex(
            *np.ones([discretization.domain_dimension], dtype=int) * 2):  # iteration over all voxel corners
        pixel_node = np.asarray(pixel_node)
        if discretization.domain_dimension == 2:
            # N_at_quad_points_qnijk
            dfstress_drho_pixel_node = np.einsum('qn,qxy->nxy',
                                                 N_at_quad_points_qnijk[(..., *pixel_node)],
                                                 double_contraction_stress_qxyz_FE)

            # dg_drho_nxyz += np.roll(div_fnxyz_pixel_node, 1 * pixel_node, axis=(1, 2))
            dfstress_drho_OLD += discretization.roll(discretization.fft, dfstress_drho_pixel_node, 1 * pixel_node,
                                                     axis=(0, 1))
        elif discretization.domain_dimension == 3:
            dfstress_drho_pixel_node = np.einsum('dqn,dqxyz->nxyz',
                                                 N_at_quad_points_qnijk[(..., *pixel_node)],
                                                 double_contraction_stress_qxyz_FE)

            # dg_drho_nxyz += np.roll(div_fnxyz_pixel_node, 1 * pixel_node, axis=(1, 2, 3))
            dfstress_drho_OLD += discretization.roll(discretization.fft, dfstress_drho_pixel_node, 1 * pixel_node,
                                                     axis=(0, 1, 2))

            warnings.warn('Gradient transposed is not tested for 3D.')

    return -2 * dfstress_drho_OLD / discretization.cell.domain_volume / np.sum(target_stress_ij ** 2)


def partial_derivative_of_objective_function_stress_equivalence_wrt_phase_field_pixel(discretization,
                                                                                      material_data_field_ijklqxyz,
                                                                                      displacement_field_fnxyz,
                                                                                      macro_gradient_field_ijqxyz,
                                                                                      phase_field_1nxyz,
                                                                                      target_stress_ij,
                                                                                      actual_stress_ij,
                                                                                      p):
    # Input: phase_field [1,n,x,y,z]
    #        material_data_field [d,d,d,d,q,x,y,z] - elasticity
    #        target_stress_ij [d,d]
    #        actual_stress_ij [d,d]
    # -- -- -- -- -- -- -- -- -- -- --

    # Gradient of material data with respect to phase field
    # % interpolation of rho into quad points
    # -----    stress difference potential ----- #
    # Gradient of material data with respect to phase field

    # I consider linear interpolation of material  C_ijkl= p*rho**(p-1) C^0_ijkl
    # so  ∂ C_ijkl/ ∂ rho = 1* C^0_ijkl
    # p = 2
    dmaterial_data_field_ijklqxyz = material_data_field_ijklqxyz[..., :, :] * (
            p * np.power(phase_field_1nxyz[0, 0], (p - 1)))

    # int(∂ C/ ∂ rho_i  * (macro_grad + micro_grad)) dx / | domain |

    # compute strain field from to displacement and macro gradient
    strain_ijqxyz = discretization.apply_gradient_operator_symmetrized(displacement_field_fnxyz)
    strain_ijqxyz = macro_gradient_field_ijqxyz + strain_ijqxyz

    # compute stress field
    # ddot42 = lambda A4, B2: np.einsum('ijklxyz,lkxyz  ->ijxyz  ', A4, B2)
    stress_field_ijqxyz_pixel = np.einsum('ijkl...,lk...->ij...', dmaterial_data_field_ijklqxyz, strain_ijqxyz)

    # apply quadrature weights
    stress_ijqxyz_pixel = discretization.apply_quadrature_weights_on_gradient_field(stress_field_ijqxyz_pixel)

    # stress difference
    stress_difference_ij = target_stress_ij - actual_stress_ij

    double_contraction_stress_qxyz_pixel = np.einsum('ij,ijqxy...->qxy...',
                                                     stress_difference_ij,
                                                     stress_ijqxyz_pixel)

    # np.einsum('ijqxyz  ,jiqxyz  ->xyz    ', A2, B2)
    # Average over quad points in pixel !!!
    # partial_derivative = partial_derivative.mean(axis=0)
    partial_derivative_xyz = double_contraction_stress_qxyz_pixel.sum(axis=0)

    return -2 * partial_derivative_xyz / discretization.cell.domain_volume / np.sum(target_stress_ij ** 2)


def partial_derivative_of_objective_function_wrt_phase_field_OLD(discretization,
                                                                 material_data_field_ijklqxyz,
                                                                 displacement_field_fnxyz,
                                                                 macro_gradient_field_ijqxyz,
                                                                 phase_field_1nxyz,
                                                                 target_stress_ij,
                                                                 actual_stress_ij,
                                                                 p):
    df_drho = partial_derivative_of_objective_function_stress_equivalence_wrt_phase_field_pixel(discretization,
                                                                                                material_data_field_ijklqxyz,
                                                                                                displacement_field_fnxyz,
                                                                                                macro_gradient_field_ijqxyz,
                                                                                                phase_field_1nxyz,
                                                                                                target_stress_ij,
                                                                                                actual_stress_ij,
                                                                                                p)

    dgradrho_drho = partial_derivative_of_gradient_of_phase_field_potential(discretization,
                                                                            phase_field_1nxyz)

    ddouble_well_drho_drho = partial_der_of_double_well_potential_wrt_density_nodal(discretization,
                                                                                    phase_field_1nxyz)

    return df_drho + dgradrho_drho + ddouble_well_drho_drho


def partial_derivative_of_objective_function_wrt_phase_field_DELETE(discretization,
                                                                    material_data_field_ijklqxyz,
                                                                    displacement_field_fnxyz,
                                                                    macro_gradient_field_ijqxyz,
                                                                    phase_field_1nxyz,
                                                                    target_stress_ij,
                                                                    actual_stress_ij,
                                                                    p,
                                                                    eta=1):
    # Input:
    #        material_data_field_ijklqxyz [d,d,d,d,q,x,y,z] - elasticity without applied phase field -- C_0
    #        displacement_field_fnxyz [f,n,x,y,z]
    #        phase_field_1nxyz [1,n,x,y,z]
    #        macro_gradient_field_ijqxyz [d,d,q,x,y,z]
    #        target_stress_ij [d,d]
    #        actual_stress_ij [d,d] # homogenized stress
    #        p [1]  # polynomial order of a material interpolation
    #        eta [1] # weight parameter for balancing the phase field terms

    # Output:
    #        df_drho_fnxyz [1,n,x,y,z]
    # -- -- -- -- -- -- -- -- -- -- --

    # -----    stress difference potential ----- #
    # Gradient of material data with respect to phase field
    material_data_field_ijklqxyz = material_data_field_ijklqxyz[..., :, :] * (
            p * np.power(phase_field_1nxyz[0, 0], (p - 1)))

    # compute strain field from to displacement and macro gradient
    strain_ijqxyz = discretization.apply_gradient_operator_symmetrized(displacement_field_fnxyz)
    strain_ijqxyz = macro_gradient_field_ijqxyz + strain_ijqxyz

    # compute stress field
    stress_field_ijqxyz = np.einsum('ijkl...,lk...->ij...', material_data_field_ijklqxyz, strain_ijqxyz)

    # apply quadrature weights
    stress_ijqxyz = discretization.apply_quadrature_weights_on_gradient_field(
        stress_field_ijqxyz)

    # stress difference
    stress_difference_ij = actual_stress_ij - target_stress_ij

    double_contraction_stress_qxyz = np.einsum('ij,ijqxy...->qxy...',
                                               stress_difference_ij,
                                               stress_ijqxyz)
    # Average over quad points in pixel !!!
    partial_derivative_xyz = double_contraction_stress_qxyz.sum(axis=0)

    dfstress_drho = 2 * partial_derivative_xyz / discretization.cell.domain_volume

    # -----    phase field gradient potential ----- #
    # partial derivative of  phase field gradient potential = 2/eta int (  (grad(rho))^2 )    dx
    #  (D rho, D I) ==  ( D I, D rho) and thus  == I D_t D rho
    # I implement it in the way = 2/eta (  I D_t D rho )
    phase_field_gradient = discretization.apply_gradient_operator(phase_field_1nxyz)
    phase_field_gradient = discretization.apply_quadrature_weights_on_gradient_field(phase_field_gradient)
    Dt_D_rho = discretization.apply_gradient_transposed_operator(phase_field_gradient)

    dgradrho_drho = 2 * Dt_D_rho

    # -----    Double well potential ----- #

    # Derivative of the double-well potential with respect to phase-field
    # phase field potential = int ( rho^2(1-rho)^2 )/eta   dx
    # gradient phase field potential = int ((2 * phase_field( + 2 * phase_field^2  -  3 * phase_field +1 )) )/eta   dx
    # d/dρ(ρ^2 (1 - ρ)^2) = 2 ρ (2 ρ^2 - 3 ρ + 1)

    # integrant_fnxyz = (2 * phase_field_1nxyz * (2 * phase_field_1nxyz * phase_field_1nxyz - 3 * phase_field_1nxyz + 1))
    # integral_fnxyz = (integrant_fnxyz / np.prod(integrant_fnxyz.shape)) * discretization.cell.domain_volume
    # there is no sum here
    # ddouble_well_drho_drho = integral_fnxyz

    ddouble_well_drho_drho = partial_der_of_double_well_potential_wrt_density_analytical(discretization=discretization,
                                                                                         phase_field_1nxyz=phase_field_1nxyz)

    return dfstress_drho + dgradrho_drho * eta + ddouble_well_drho_drho / eta


def partial_derivative_of_objective_function_wrt_phase_field_FE(discretization,
                                                                material_data_field_ijklqxyz,
                                                                displacement_field_fnxyz,
                                                                macro_gradient_field_ijqxyz,
                                                                phase_field_1nxyz,
                                                                target_stress_ij,
                                                                actual_stress_ij,
                                                                p,
                                                                eta=1):
    # Input:
    #        material_data_field_ijklqxyz [d,d,d,d,q,x,y,z] - elasticity without applied phase field -- C_0
    #        displacement_field_fnxyz [f,n,x,y,z]
    #        phase_field_1nxyz [1,n,x,y,z]
    #        macro_gradient_field_ijqxyz [d,d,q,x,y,z]
    #        target_stress_ij [d,d]
    #        actual_stress_ij [d,d] # homogenized stress
    #        p [1]  # polynomial order of a material interpolation
    #        eta [1] # weight parameter for balancing the phase field terms

    # Output:
    #        df_drho_fnxyz [1,n,x,y,z]
    # -- -- -- -- -- -- -- -- -- -- --

    # -----    stress difference potential ----- #
    # Gradient of material data with respect to phase field
    phase_field_at_quad_poits_1qnxyz, N_at_quad_points_qnijk = discretization.evaluate_field_at_quad_points(
        nodal_field_fnxyz=phase_field_1nxyz,
        quad_field_fqnxyz=None,
        quad_points_coords_iq=None)

    dmaterial_data_field_drho_ijklqxyz = material_data_field_ijklqxyz[..., :, :, :] * (
            p * np.power(phase_field_at_quad_poits_1qnxyz[0, :, 0, ...], (p - 1)))

    # compute strain field from to displacement and macro gradient
    strain_ijqxyz = discretization.apply_gradient_operator_symmetrized(displacement_field_fnxyz)
    strain_ijqxyz = macro_gradient_field_ijqxyz + strain_ijqxyz

    # compute stress field
    stress_field_ijqxyz = np.einsum('ijkl...,lk...->ij...', dmaterial_data_field_drho_ijklqxyz, strain_ijqxyz)

    # apply quadrature weights
    stress_ijqxyz = discretization.apply_quadrature_weights_on_gradient_field(stress_field_ijqxyz)

    # stress difference
    stress_difference_ij = actual_stress_ij - target_stress_ij

    double_contraction_stress_qxyz = np.einsum('ij,ijqxy...->qxy...',
                                               stress_difference_ij,
                                               stress_ijqxyz)

    nodal_field_u_nxyz = np.zeros(phase_field_1nxyz.shape)
    for pixel_node in np.ndindex(
            *np.ones([discretization.domain_dimension], dtype=int) * 2):  # iteration over all voxel corners
        pixel_node = np.asarray(pixel_node)
        if discretization.domain_dimension == 2:
            # N_at_quad_points_qnijk
            div_fnxyz_pixel_node = np.einsum('qn,qxy->nxy',
                                             N_at_quad_points_qnijk[(..., *pixel_node)],
                                             double_contraction_stress_qxyz)

            nodal_field_u_nxyz += np.roll(div_fnxyz_pixel_node, 1 * pixel_node, axis=(1, 2))

        elif discretization.domain_dimension == 3:

            div_fnxyz_pixel_node = np.einsum('dqn,dqxyz->nxyz',
                                             N_at_quad_points_qnijk[(..., *pixel_node)],
                                             double_contraction_stress_qxyz)

            nodal_field_u_nxyz += np.roll(div_fnxyz_pixel_node, 1 * pixel_node, axis=(1, 2, 3))
            warnings.warn('Gradient transposed is not tested for 3D.')

    # Average over quad points in pixel !!!
    # partial_derivative_xyz = double_contraction_stress_qxyz.sum(axis=0)

    dfstress_drho = 2 * nodal_field_u_nxyz / discretization.cell.domain_volume

    # -----    phase field gradient potential ----- #
    # partial derivative of  phase field gradient potential = 2/eta int (  (grad(rho))^2 )    dx
    #  (D rho, D I) ==  ( D I, D rho) and thus  == I D_t D rho
    # I implement it in the way = 2/eta (  I D_t D rho )
    phase_field_gradient = discretization.apply_gradient_operator(phase_field_1nxyz)
    phase_field_gradient = discretization.apply_quadrature_weights_on_gradient_field(phase_field_gradient)
    Dt_D_rho = discretization.apply_gradient_transposed_operator(phase_field_gradient)

    dgradrho_drho = 2 * Dt_D_rho

    # -----    Double well potential ----- #

    # Derivative of the double-well potential with respect to phase-field
    # phase field potential = int ( rho^2(1-rho)^2 )/eta   dx
    # gradient phase field potential = int ((2 * phase_field( + 2 * phase_field^2  -  3 * phase_field +1 )) )/eta   dx
    # d/dρ(ρ^2 (1 - ρ)^2) = 2 ρ (2 ρ^2 - 3 ρ + 1)

    # integrant_fnxyz = (2 * phase_field_1nxyz * (2 * phase_field_1nxyz * phase_field_1nxyz - 3 * phase_field_1nxyz + 1))
    # integral_fnxyz = (integrant_fnxyz / np.prod(integrant_fnxyz.shape)) * discretization.cell.domain_volume
    # there is no sum here
    # ddouble_well_drho_drho = integral_fnxyz

    ddouble_well_drho_drho = partial_der_of_double_well_potential_wrt_density_analytical(discretization=discretization,
                                                                                         phase_field_1nxyz=phase_field_1nxyz,
                                                                                         eta=1)

    return dfstress_drho + dgradrho_drho * eta + ddouble_well_drho_drho / eta


def partial_der_of_objective_function_wrt_displacement_small_strain(discretization,
                                                                    material_data_field_ijklqxyz,
                                                                    stress_difference_ij,
                                                                    eta=1,
                                                                    w=1):
    # Input: material_data_field [d,d,d,d,q,x,y,z] - elasticity
    #        stress_diff_ij [d,d] # difference between homogenized stress and target stress
    #
    # Output: ∂ f_sigma/ ∂ u  = - (2 / |domain size|) int  grad_transpose : C: sigma_diff dx [f,n,x,y,z]

    stress_field_ijqxyz = discretization.apply_material_data(material_data_field_ijklqxyz, stress_difference_ij)

    stress_field_ijqxyz = discretization.apply_quadrature_weights_on_gradient_field(stress_field_ijqxyz)

    df_sigma_du_fnxyz = discretization.apply_gradient_transposed_operator(stress_field_ijqxyz)

    return -2 * df_sigma_du_fnxyz / discretization.cell.domain_volume


def solve_adjoint_problem(discretization, material_data_field_ijklqxyz,
                          stress_difference_ij,
                          formulation='small_strain'):
    # Solve adjoint problem ∂f/∂u=-∂g/∂u
    # Dt C D lambda = - 2/|omega| Dt: C : sigma_diff

    # stress difference potential
    # rhs=-Dt*wA*E  -- we can use it to assemble df_du_field

    stress_difference_ijqxyz = discretization.get_gradient_size_field()
    stress_difference_ijqxyz[:, :, ...] = stress_difference_ij[
        (...,) + (np.newaxis,) * (stress_difference_ijqxyz.ndim - 2)]
    df_du_field = 2 * discretization.get_rhs(material_data_field_ijklqxyz=material_data_field_ijklqxyz,
                                             macro_gradient_field_ijqxyz=stress_difference_ijqxyz) / discretization.cell.domain_volume  # minus sign is already there
    #
    K_fun = lambda x: discretization.apply_system_matrix(material_data_field=material_data_field_ijklqxyz,
                                                         displacement_field=x,
                                                         formulation=formulation)
    M_fun = lambda x: 1 * x

    # solve the system
    adjoint_field, adjoint_norms = solvers.PCG(K_fun, df_du_field, x0=None, P=M_fun, steps=int(500), toler=1e-6)

    return adjoint_field


def adjoint_potential(discretization,
                      stress_field_ijqxyz,
                      adjoint_field_inxyz):
    # g = (grad lambda, stress)
    # g = (grad lambda, C grad displacement)
    # g = (grad lambda, C grad displacement)  == lambda_transpose grad_transpose C grad u

    # Input: adjoint_field [f,n,x,y,z]
    #        stress_field  [d,d,q,x,y,z]

    # Output: g  [1] == 0
    # -- -- -- -- -- -- -- -- -- -- --
    # apply quadrature weights

    weights = discretization.quadrature_weights
    # apply B^transposed via the convolution operator
    # stress_field_ijqxyz.s = discretization.apply_quadrature_weights_on_gradient_field(stress_field_ijqxyz.s)
    force_field_inxyz = discretization.get_displacement_sized_field(
        name='force_field_inxyz_in_adjoint_potential_temporary')
    # force_field_inxyz = discretization.apply_gradient_transposed_operator(stress_field_ijqxyz)
    discretization.conv_op.transpose(quadrature_point_field=stress_field_ijqxyz,
                                     nodal_field=force_field_inxyz,
                                     weights=weights)

    adjoint_potential_field = np.einsum('i...,i...->...', adjoint_field_inxyz.s, force_field_inxyz.s)

    # Reductor_numpi = discretization.mpi_reduction(MPI.COMM_WORLD)
    integral = discretization.mpi_reduction.sum(adjoint_potential_field)  #

    return integral


def partial_derivative_of_adjoint_potential_wrt_displacement(discretization,
                                                             material_data_field_ijklqxyz,
                                                             adjoint_field_fnxyz):
    # Input: adjoint_field_inxyz [f,n,x,y,z]
    #        material_data_field_C_0_ijklqxyz  [d,d,d,d,q,x,y,z]

    # Output:  ∂ g/ ∂ u [f,n,x,y,z] = ( grad_transpose: C : grad lambda)
    # -- -- -- -- -- -- -- -- -- -- --
    # apply quadrature weights

    force_field_fnxyz = discretization.apply_system_matrix(material_data_field_ijklqxyz, adjoint_field_fnxyz,
                                                           formulation='small_strain')

    return force_field_fnxyz


def partial_derivative_of_adjoint_potential_wrt_phase_field_pixel_OLD(discretization,
                                                                      material_data_field_ijklqxyz,
                                                                      displacement_field_fnxyz,
                                                                      macro_gradient_field_ijqxyz,
                                                                      phase_field_1nxyz,
                                                                      adjoint_field_fnxyz,
                                                                      p=1):
    # Input:
    #        material_data_field_ijklqxyz [d,d,d,d,q,x,y,z] - elasticity without applied phase field -- C_0
    #        displacement_field_fnxyz [f,n,x,y,z]
    #        macro_gradient_field_ijqxyz [d,d,q,x,y,z]
    #        phase_field_1nxyz [1,n,x,y,z]
    #        adjoint_field_fnxyz  [f,n,x,y,z]
    #        p [1]  # polynomial order of a material interpolation

    # Output:
    #        dg_drho_fnxyz [1, n, x, y, z]
    # -- -- -- -- -- -- -- -- -- -- --

    # Gradient of material data with respect to phasse field   % interpolation of rho into quad points
    # I consider linear interpolation of material  C_ijkl= p*rho**(p-1) C^0_ijkl
    # so  ∂ C_ijkl/ ∂ rho = 1* C^0_ijkl
    material_data_field_ijklqxyz = material_data_field_ijklqxyz[..., :, :] * (
            p * np.power(phase_field_1nxyz[0, 0], (p - 1)))

    # compute strain field from to displacement and macro gradient
    strain_ijqxyz = discretization.apply_gradient_operator_symmetrized(displacement_field_fnxyz)
    strain_ijqxyz = macro_gradient_field_ijqxyz + strain_ijqxyz

    # compute stress field
    # ddot42 = lambda A4, B2: np.einsum('ijklxyz,lkxyz  ->ijxyz  ', A4, B2)
    stress_field_ijqxyz = np.einsum('ijkl...,lk...->ij...', material_data_field_ijklqxyz, strain_ijqxyz)

    # local_stress=np.einsum('ijkl,lk->ij',material_data_field_ijklqxyz[...,0,0,0] , strain_ijqxyz[...,0,0,0])
    # apply quadrature weights
    stress_field_ijqxyz = discretization.apply_quadrature_weights_on_gradient_field(stress_field_ijqxyz)

    # gradient of adjoint_field
    adjoint_field_gradient_ijqxyz = discretization.apply_gradient_operator_symmetrized(adjoint_field_fnxyz)

    # ddot22 = lambda A2, B2:  np.einsum('ijqxyz  ,jiqxyz  ->qxyz    ', A2, B2)
    double_contraction_stress_qxyz = np.einsum('ij...,ij...->...',
                                               adjoint_field_gradient_ijqxyz,
                                               stress_field_ijqxyz)

    return double_contraction_stress_qxyz.sum(axis=0)


def partial_derivative_of_adjoint_potential_wrt_phase_field_FE(discretization,
                                                               base_material_data_ijkl,
                                                               displacement_field_fnxyz,
                                                               macro_gradient_field_ijqxyz,
                                                               phase_field_1nxyz,
                                                               adjoint_field_inxyz,
                                                               output_field_inxyz,
                                                               p=1
                                                               ):
    # Input:
    #        material_data_field_ijklqxyz [d,d,d,d,q,x,y,z] - elasticity without applied phase field -- C_0
    #        displacement_field_fnxyz [f,n,x,y,z]
    #        macro_gradient_field_ijqxyz [d,d,q,x,y,z]
    #        phase_field_1nxyz [1,n,x,y,z]
    #        adjoint_field_fnxyz  [f,n,x,y,z]
    #        p [1]  # polynomial order of a material interpolation

    # Output:
    #        dg_drho_fnxyz [1, n, x, y, z]
    # -- -- -- -- -- -- -- -- -- -- --

    # Gradient of material data with respect to phasse field   % interpolation of rho into quad points
    # I consider linear interpolation of material  C_ijkl= p*rho**(p-1) C^0_ijkl
    # so  ∂ C_ijkl/ ∂ rho = 1* C^0_ijkl

    # Gradient of material data with respect to phase field
    phase_field_at_quad_poits_1qnxyz, N_at_quad_points_qnijk = discretization.evaluate_field_at_quad_points(
        nodal_field_fnxyz=phase_field_1nxyz,
        quad_field_fqnxyz=None,
        quad_points_coords_iq=None)

    dmaterial_data_field_drho_ijklqxyz = discretization.get_material_data_size_field(
        name='data_field_drho_partial_derivative_of_adjoint_potential_wrt_phase_field_FE')
    dmaterial_data_field_drho_ijklqxyz.s = base_material_data_ijkl[..., np.newaxis, np.newaxis, np.newaxis] * (
            p * np.power(phase_field_at_quad_poits_1qnxyz[0, :, 0, ...], (p - 1)))

    # compute strain field from to displacement and macro gradient
    strain_ijqxyz = discretization.get_displacement_gradient_sized_field(name='strain_ijqxyz_local_at_pdapwpf')
    strain_ijqxyz = discretization.apply_gradient_operator_symmetrized(u_inxyz=displacement_field_fnxyz,
                                                                       grad_u_ijqxyz=strain_ijqxyz)
    strain_ijqxyz.s = macro_gradient_field_ijqxyz.s + strain_ijqxyz.s

    # compute stress field
    # ddot42 = lambda A4, B2: np.einsum('ijklxyz,lkxyz  ->ijxyz  ', A4, B2)
    strain_ijqxyz.s = np.einsum('ijkl...,lk...->ij...',
                                dmaterial_data_field_drho_ijklqxyz,
                                strain_ijqxyz.s)

    # local_stress=np.einsum('ijkl,lk->ij',material_data_field_ijklqxyz[...,0,0,0] , strain_ijqxyz[...,0,0,0])
    # apply quadrature weights
    strain_ijqxyz.s = discretization.apply_quadrature_weights_on_gradient_field(strain_ijqxyz.s)

    # gradient of adjoint_field
    adjoint_field_gradient_ijqxyz = discretization.get_displacement_gradient_sized_field(
        name='adjoint_field_gradient_ijqxyz__pdapwpf')
    adjoint_field_gradient_ijqxyz = discretization.apply_gradient_operator_symmetrized(u_inxyz=adjoint_field_inxyz,
                                                                                       grad_u_ijqxyz=adjoint_field_gradient_ijqxyz, )
    # TODO: should this be symmetric gradient?
    # ddot22 = lambda A2, B2:  np.einsum('ijqxyz  ,jiqxyz  ->qxyz    ', A2, B2)
    double_contraction_stress_qxyz = np.einsum('ij...,ij...->...',
                                               adjoint_field_gradient_ijqxyz.s,
                                               strain_ijqxyz.s)  # this is stress, It just cupy the same name

    # nodal_field_u_nxyz=discretization.get_scalar_field(name='nodal_field_u_nxyz')
    # nodal_field_u_nxyz = np.zeros(phase_field_1nxyz.shape)
    output_field_inxyz.s.fill(0)
    for pixel_node in np.ndindex(
            *np.ones([discretization.domain_dimension], dtype=int) * 2):  # iteration over all voxel corners
        pixel_node = np.asarray(pixel_node)
        if discretization.domain_dimension == 2:
            # N_at_quad_points_qnijk
            div_fnxyz_pixel_node = np.einsum('qn,qxy->nxy',
                                             N_at_quad_points_qnijk[(..., *pixel_node)],
                                             double_contraction_stress_qxyz)

            output_field_inxyz.s += discretization.roll(discretization.fft, div_fnxyz_pixel_node, 1 * pixel_node,
                                                        axis=(0, 1))

        elif discretization.domain_dimension == 3:

            div_fnxyz_pixel_node = np.einsum('dqn,dqxyz->nxyz',
                                             N_at_quad_points_qnijk[(..., *pixel_node)],
                                             double_contraction_stress_qxyz)

            output_field_inxyz.s += discretization.roll(discretization.fft, div_fnxyz_pixel_node, 1 * pixel_node,
                                                        axis=(0, 1, 2))
            warnings.warn('Gradient transposed is not tested for 3D.')

    return output_field_inxyz


def sensitivity_OLD_pixel(discretization,
                          material_data_field_ijklqxyz,
                          displacement_field_fnxyz,
                          macro_gradient_field_ijqxyz,
                          phase_field_1nxyz,
                          adjoint_field_fnxyz,
                          target_stress_ij,
                          actual_stress_ij,
                          p,
                          eta=1):
    df_drho = partial_derivative_of_objective_function_wrt_phase_field(
        discretization=discretization,
        material_data_field_ijklqxyz=material_data_field_ijklqxyz,
        displacement_field_fnxyz=displacement_field_fnxyz,
        macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
        phase_field_1nxyz=phase_field_1nxyz,
        target_stress_ij=target_stress_ij,
        actual_stress_ij=actual_stress_ij,
        p=p,
        eta=eta)

    dg_drho = partial_derivative_of_adjoint_potential_wrt_phase_field_pixel(
        discretization=discretization,
        material_data_field_ijklqxyz=material_data_field_ijklqxyz,
        displacement_field_fnxyz=displacement_field_fnxyz,
        macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
        phase_field_1nxyz=phase_field_1nxyz,
        adjoint_field_fnxyz=adjoint_field_fnxyz,
        p=p)

    return df_drho + dg_drho


def sensitivity(discretization,
                material_data_field_ijklqxyz,
                displacement_field_fnxyz,
                macro_gradient_field_ijqxyz,
                phase_field_1nxyz,
                adjoint_field_fnxyz,
                target_stress_ij,
                actual_stress_ij,
                p,
                eta=1):
    # Input:
    #        material_data_field_ijklqxyz [d,d,d,d,q,x,y,z] - elasticity without applied phase field -- C_0
    #        displacement_field_fnxyz [f,n,x,y,z]
    #        phase_field_1nxyz [1,n,x,y,z]
    #        macro_gradient_field_ijqxyz [d,d,q,x,y,z]
    #        adjoint_field_fnxyz  [f,n,x,y,z]
    #        target_stress_ij [d,d]
    #        actual_stress_ij [d,d] # homogenized stress
    #        p [1]  # polynomial order of a material interpolation
    #        eta [1] # weight parameter for balancing the phase field terms

    # Output:
    #        df_drho_fnxyz [1,n,x,y,z]
    # -- -- -- -- -- -- -- -- -- -- --

    # -----    stress difference potential ----- #
    # Gradient of material data with respect to phase field
    dmaterial_data_field_ijklqxyz = material_data_field_ijklqxyz[..., :, :] * (
            p * np.power(phase_field_1nxyz[0, 0], (p - 1)))

    # compute strain field from to displacement and macro gradient
    strain_ijqxyz = discretization.apply_gradient_operator_symmetrized(displacement_field_fnxyz)
    strain_ijqxyz = macro_gradient_field_ijqxyz + strain_ijqxyz

    # compute stress field
    stress_field_ijqxyz = np.einsum('ijkl...,lk...->ij...', dmaterial_data_field_ijklqxyz, strain_ijqxyz)

    # apply quadrature weights
    stress_field_ijqxyz = discretization.apply_quadrature_weights_on_gradient_field(stress_field_ijqxyz)

    # ---  part that is unique for  df_drho ---
    # stress difference
    stress_difference_ij = actual_stress_ij - target_stress_ij

    double_contraction_stress_qxyz = np.einsum('ij,ijqxy...->qxy...',
                                               stress_difference_ij,
                                               stress_field_ijqxyz)
    # Average over quad points in pixel !!!
    partial_derivative_xyz = double_contraction_stress_qxyz.sum(axis=0)

    dfstress_drho = 2 * partial_derivative_xyz / discretization.cell.domain_volume

    # -----    phase field gradient potential ----- #
    # partial derivative of  phase field gradient potential = 2/eta int (  (grad(rho))^2 )    dx
    #  (D rho, D I) ==  ( D I, D rho) and thus  == I D_t D rho
    # I implement it in the way = 2/eta (  I D_t D rho )
    phase_field_gradient = discretization.apply_gradient_operator(phase_field_1nxyz)
    phase_field_gradient = discretization.apply_quadrature_weights_on_gradient_field(phase_field_gradient)
    Dt_D_rho = discretization.apply_gradient_transposed_operator(phase_field_gradient)

    dgradrho_drho = 2 * Dt_D_rho

    # -----    Double well potential ----- #

    # Derivative of the double-well potential with respect to phase-field
    # phase field potential = int ( rho^2(1-rho)^2 )/eta   dx
    # gradient phase field potential = int ((2 * phase_field( + 2 * phase_field^2  -  3 * phase_field +1 )) )/eta   dx
    # d/dρ(ρ^2 (1 - ρ)^2) = 2 ρ (2 ρ^2 - 3 ρ + 1)

    # integrant_fnxyz = (2 * phase_field_1nxyz * (2 * phase_field_1nxyz * phase_field_1nxyz - 3 * phase_field_1nxyz + 1))

    # integral_fnxyz = (integrant_fnxyz / np.prod(integrant_fnxyz.shape)) * discretization.cell.domain_volume
    # there is no sum here
    # ddouble_well_drho_drho = integral_fnxyz
    ddouble_well_drho_drho = partial_der_of_double_well_potential_wrt_density_analytical(discretization=discretization,
                                                                                         phase_field_1nxyz=phase_field_1nxyz,
                                                                                         eta=1)

    # sum of all parts of df_drho
    df_drho = dfstress_drho + dgradrho_drho * eta + ddouble_well_drho_drho / eta

    # --------------------------------------

    # gradient of adjoint_field
    adjoint_field_gradient_ijqxyz = discretization.apply_gradient_operator_symmetrized(adjoint_field_fnxyz)

    # ddot22 = lambda A2, B2:  np.einsum('ijqxyz  ,jiqxyz  ->qxyz    ', A2, B2)
    double_contraction_stress_qxyz = np.einsum('ij...,ij...->...',
                                               adjoint_field_gradient_ijqxyz,
                                               stress_field_ijqxyz)

    dg_drho = double_contraction_stress_qxyz.sum(axis=0)

    return df_drho + dg_drho


def sensitivity_with_adjoint_problem_pixel(discretization,
                                           material_data_field_ijklqxyz,
                                           displacement_field_fnxyz,
                                           macro_gradient_field_ijqxyz,
                                           phase_field_1nxyz,
                                           target_stress_ij,
                                           actual_stress_ij,
                                           formulation,
                                           p,
                                           eta,
                                           weight):
    # Input:
    #        material_data_field_ijklqxyz [d,d,d,d,q,x,y,z] - elasticity tensors without applied phase field -- C_0
    #        displacement_field_fnxyz [f,n,x,y,z]
    #        phase_field_1nxyz [1,n,x,y,z]
    #        macro_gradient_field_ijqxyz [d,d,q,x,y,z]
    #        target_stress_ij [d,d]
    #        actual_stress_ij [d,d] # homogenized stress
    #        formulation  - 'finite_strain', 'small_strain'
    #        p [1]  # polynomial order of a material interpolation
    #        eta [1] # weight parameter for balancing the phase field terms

    # Output:
    #        df_drho_fnxyz [1,n,x,y,z]
    # -- -- -- -- -- -- -- -- -- -- --

    # -----    stress difference potential ----- #
    # Gradient of material data with respect to phase field
    dmaterial_data_field_drho_ijklqxyz = material_data_field_ijklqxyz[..., :, :] * (
            p * np.power(phase_field_1nxyz[0, 0], (p - 1)))

    # TODO [Martin] Interpolation of material data is different compared to Indre

    # compute strain field from to displacement and macro gradient
    strain_ijqxyz = discretization.apply_gradient_operator_symmetrized(displacement_field_fnxyz)
    strain_ijqxyz = macro_gradient_field_ijqxyz + strain_ijqxyz

    # compute stress field
    stress_field_ijqxyz = np.einsum('ijkl...,lk...->ij...', dmaterial_data_field_drho_ijklqxyz, strain_ijqxyz)

    # apply quadrature weights
    stress_field_ijqxyz = discretization.apply_quadrature_weights_on_gradient_field(stress_field_ijqxyz)

    # ---  part that is unique for  df_drho ---
    # stress difference
    stress_difference_ij = actual_stress_ij - target_stress_ij

    double_contraction_stress_qxyz = np.einsum('ij,ijqxy...->qxy...',
                                               stress_difference_ij,
                                               stress_field_ijqxyz)
    # Average over quad points in pixel !!!
    partial_derivative_xyz = double_contraction_stress_qxyz.sum(axis=0)

    dfstress_drho = 2 * partial_derivative_xyz / discretization.cell.domain_volume / np.sum(target_stress_ij ** 2)

    # -----    phase field gradient potential ----- #
    # partial derivative of  phase field gradient potential = 2/eta int (  (grad(rho))^2 )    dx
    #  (D rho, D I) ==  ( D I, D rho) and thus  == I D_t D rho
    # I implement it in the way = 2/eta (  I D_t D rho )
    phase_field_gradient = discretization.apply_gradient_operator(phase_field_1nxyz)
    phase_field_gradient = discretization.apply_quadrature_weights_on_gradient_field(phase_field_gradient)
    Dt_D_rho = discretization.apply_gradient_transposed_operator(phase_field_gradient)

    dgradrho_drho = 2 * Dt_D_rho

    # -----    Double well potential ----- #

    # Derivative of the double-well potential with respect to phase-field
    # phase field potential = int ( rho^2(1-rho)^2 )/eta   dx
    # gradient phase field potential = int ((2 * phase_field( + 2 * phase_field^2  -  3 * phase_field +1 )) )/eta   dx
    # d/dρ(ρ^2 (1 - ρ)^2) = 2 ρ (2 ρ^2 - 3 ρ + 1)

    # integrant_fnxyz = (2 * phase_field_1nxyz * (2 * phase_field_1nxyz * phase_field_1nxyz - 3 * phase_field_1nxyz + 1))

    # integral_fnxyz = (integrant_fnxyz / np.prod(integrant_fnxyz.shape)) * discretization.cell.domain_volume
    # there is no sum here
    # ddouble_well_drho_drho = integral_fnxyz
    ddouble_well_drho_drho = partial_der_of_double_well_potential_wrt_density_nodal(discretization=discretization,
                                                                                    phase_field_1nxyz=phase_field_1nxyz,
                                                                                    eta=1)
    # sum of all parts of df_drho
    df_drho = dfstress_drho + weight * (dgradrho_drho * eta + ddouble_well_drho_drho / eta)

    # --------------------------------------
    # Solve adjoint problem ∂f/∂u=-∂g/∂u
    # Dt C D lambda = - 2/|omega| Dt: C : sigma_diff
    # material_data_field_C_0_rho_ijklqxyz = material_data_field_ijklqxyz[..., :, :] * np.power(phase_field_1nxyz,
    #                                                                                          p)
    # TODO delete if phase field at quad points wokrs
    material_data_field_C_0_rho_ijklqxyz = material_data_field_ijklqxyz[..., :, :] * np.power(phase_field_1nxyz[0, 0],
                                                                                              (p))
    # stress difference potential
    # rhs=-Dt*wA*E  -- we can use it to assemble df_du_field

    stress_difference_ijqxyz = discretization.get_gradient_size_field()
    stress_difference_ijqxyz[:, :, ...] = stress_difference_ij[
        (...,) + (np.newaxis,) * (stress_difference_ijqxyz.ndim - 2)]

    df_du_field = 2 * discretization.get_rhs(material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                                             macro_gradient_field_ijqxyz=stress_difference_ijqxyz) / discretization.cell.domain_volume  # minus sign is already there

    # Normalization
    df_du_field = df_du_field / np.sum(target_stress_ij ** 2)
    #
    K_fun = lambda x: discretization.apply_system_matrix(material_data_field=material_data_field_C_0_rho_ijklqxyz,
                                                         displacement_field=x,
                                                         formulation=formulation)
    # M_fun = lambda x: 1 * x
    preconditioner = discretization.get_preconditioner_NEW(
        reference_material_data_field_ijklqxyz=material_data_field_ijklqxyz)
    M_fun = lambda x: discretization.apply_preconditioner_NEW(preconditioner_Fourier_fnfnqks=preconditioner,
                                                              nodal_field_fnxyz=x)

    # solve the system
    adjoint_field_fnxyz, adjoint_norms = solvers.PCG(Afun=K_fun, B=df_du_field, x0=None, P=M_fun,
                                                     steps=int(500),
                                                     toler=1e-6)

    # gradient of adjoint_field
    adjoint_field_gradient_ijqxyz = discretization.apply_gradient_operator_symmetrized(adjoint_field_fnxyz)

    # ddot22 = lambda A2, B2:  np.einsum('ijqxyz  ,jiqxyz  ->qxyz    ', A2, B2)
    double_contraction_stress_qxyz = np.einsum('ij...,ij...->...',
                                               adjoint_field_gradient_ijqxyz,
                                               stress_field_ijqxyz)

    dg_drho = double_contraction_stress_qxyz.sum(axis=0)

    return df_drho + dg_drho


def sensitivity_with_adjoint_problem_FE(discretization,
                                        material_data_field_ijklqxyz,
                                        displacement_field_fnxyz,
                                        macro_gradient_field_ijqxyz,
                                        phase_field_1nxyz,
                                        target_stress_ij,
                                        actual_stress_ij,
                                        preconditioner_fun,
                                        formulation,
                                        p,
                                        eta,
                                        weight):
    # Input:
    #        material_data_field_ijklqxyz [d,d,d,d,q,x,y,z] - elasticity tensors without applied phase field -- C_0
    #        displacement_field_fnxyz [f,n,x,y,z]
    #        phase_field_1nxyz [1,n,x,y,z]
    #        macro_gradient_field_ijqxyz [d,d,q,x,y,z]
    #        target_stress_ij [d,d]
    #        actual_stress_ij [d,d] # homogenized stress
    #        formulation  - 'finite_strain', 'small_strain'
    #        p [1]  # polynomial order of a material interpolation
    #        eta [1] # weight parameter for balancing the phase field terms

    # Output:
    #        df_drho_fnxyz [1,n,x,y,z]
    # -- -- -- -- -- -- -- -- -- -- --

    # -----    stress difference potential ----- #
    # Gradient of material data with respect to phase field
    phase_field_at_quad_poits_1qnxyz, N_at_quad_points_qnijk = discretization.evaluate_field_at_quad_points(
        nodal_field_fnxyz=phase_field_1nxyz,
        quad_field_fqnxyz=None,
        quad_points_coords_iq=None)
    # dmaterial_data_field_drho_ijklqxyz = material_data_field_ijklqxyz[..., :, :] * (
    #         p * np.power(phase_field_1nxyz[0, 0], (p - 1)))

    dmaterial_data_field_drho_ijklqxyz = material_data_field_ijklqxyz[..., :, :, :] * np.power(
        p * phase_field_at_quad_poits_1qnxyz, p - 1)[0, :, 0, ...]
    # compute strain field from to displacement and macro gradient
    strain_ijqxyz = discretization.apply_gradient_operator_symmetrized(displacement_field_fnxyz)
    if p == 100:
        # TODO CHANGE strain_ijqxyz field adjustment
        strain_ijqxyz = strain_ijqxyz * np.where(phase_field_at_quad_poits_1qnxyz > 0.0001, 1, 0)

    strain_ijqxyz = macro_gradient_field_ijqxyz + strain_ijqxyz

    # compute stress field
    # dmaterial_data_field_drho_ijklqxyz_w = discretization.apply_quadrature_weights(dmaterial_data_field_drho_ijklqxyz)
    stress_field_ijqxyz = discretization.apply_material_data(dmaterial_data_field_drho_ijklqxyz, strain_ijqxyz)
    stress_field_ijqxyz = discretization.apply_quadrature_weights_on_gradient_field(stress_field_ijqxyz)

    # stress_field_ijqxyz = np.einsum('ijkl...,lk...->ij...', dmaterial_data_field_drho_ijklqxyz, strain_ijqxyz)
    # dmaterial_data_field_drho_ijklqxyz_w = discretization.apply_quadrature_weights(dmaterial_data_field_drho_ijklqxyz)

    # apply quadrature weights
    # stress_field_ijqxyz = discretization.apply_quadrature_weights_on_gradient_field(stress_field_ijqxyz)

    # ---  part that is unique for  df_drho ---
    # homogenized_stress_NEW = discretization.get_homogenized_stress(
    #     material_data_field_ijklqxyz=dmaterial_data_field_drho_ijklqxyz,
    #     displacement_field_fnxyz=displacement_field_fnxyz,
    #     macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
    #     formulation=formulation)
    # stress difference
    # stress_difference_ij = actual_stress_ij - target_stress_ij
    stress_difference_ij = target_stress_ij - actual_stress_ij

    double_contraction_stress_qxyz = np.einsum('ij,ijqxy...->qxy...',
                                               stress_difference_ij,
                                               stress_field_ijqxyz)

    # # partial_derivative_xyz_mpi = np.zeros(phase_field_1nxyz.shape)
    # shape_FE = np.asarray(stress_field_ijqxyz.shape)  # TODO [MARTIN] fix sizes
    # shape_FE[2] = 1
    # # np.squeeze(A, axis=1)
    # partial_derivative_xyz_FE = np.zeros(shape_FE)
    #
    # # partial_derivative_xyz_mpi = np.zeros(phase_field_1nxyz.shape)
    # for pixel_node in np.ndindex(discretization.nb_unique_nodes_per_pixel,
    #                              *np.ones([discretization.domain_dimension],
    #                                       dtype=int) * 2):  # iteration over all voxel corners
    #     pixel_node = np.asarray(pixel_node)
    #     if discretization.domain_dimension == 2:
    #         # N_at_quad_points_qnijk
    #         # multiply with basis function that corresponds to pixel node
    #         # + sum over all quad points
    #         stress_times_basis_pixel_node_ijxy = np.einsum('q,ijqxy->ijxy',
    #                                                        N_at_quad_points_qnijk[(..., *pixel_node)],
    #                                                        stress_field_ijqxyz)
    #
    #         partial_derivative_xyz_FE[..., 0, :, :] += discretization.roll(discretization.fft,
    #                                                                        stress_times_basis_pixel_node_ijxy,
    #                                                                        1 * pixel_node[1:], axis=(0, 1))
    #     elif discretization.domain_dimension == 3:
    #
    #         stress_times_basis_pixel_node_ijxy = np.einsum('q,ijqxy->ijxy',
    #                                                        N_at_quad_points_qnijk[(..., *pixel_node)],
    #                                                        stress_field_ijqxyz)
    #
    #         # partial_derivative_xyz += np.roll(div_fnxyz_pixel_node, 1 * pixel_node, axis=(1, 2, 3))
    #         partial_derivative_xyz_FE[..., 0, :, :, :] += discretization.roll(discretization.fft,
    #                                                                           stress_times_basis_pixel_node_ijxy,
    #                                                                           1 * pixel_node[1:], axis=(0, 1, 2))
    #         warnings.warn('Gradient transposed is not tested for 3D.')
    #
    # partial_derivative_xyz_FE_sum = np.einsum('ij,ijnxy...->nxy...',
    #                                           stress_difference_ij,
    #                                           partial_derivative_xyz_FE)
    # -----------------------------------------------------------------------------------------------------------
    # Average over quad points in pixel !!!
    dfstress_drho_OLD = np.zeros(phase_field_1nxyz.shape)
    for pixel_node in np.ndindex(
            *np.ones([discretization.domain_dimension], dtype=int) * 2):  # iteration over all voxel corners
        pixel_node = np.asarray(pixel_node)
        if discretization.domain_dimension == 2:
            # N_at_quad_points_qnijk
            dfstress_drho_pixel_node = np.einsum('qn,qxy->nxy',
                                                 N_at_quad_points_qnijk[(..., *pixel_node)],
                                                 double_contraction_stress_qxyz)

            # dg_drho_nxyz += np.roll(div_fnxyz_pixel_node, 1 * pixel_node, axis=(1, 2))
            dfstress_drho_OLD += discretization.roll(discretization.fft, dfstress_drho_pixel_node, 1 * pixel_node,
                                                     axis=(0, 1))
        elif discretization.domain_dimension == 3:
            dfstress_drho_pixel_node = np.einsum('dqn,dqxyz->nxyz',
                                                 N_at_quad_points_qnijk[(..., *pixel_node)],
                                                 double_contraction_stress_qxyz)

            # dg_drho_nxyz += np.roll(div_fnxyz_pixel_node, 1 * pixel_node, axis=(1, 2, 3))
            dfstress_drho_OLD += discretization.roll(discretization.fft, dfstress_drho_pixel_node, 1 * pixel_node,
                                                     axis=(0, 1, 2))

            warnings.warn('Gradient transposed is not tested for 3D.')

    dfstress_drho = -2 * dfstress_drho_OLD / discretization.cell.domain_volume / np.sum(
        target_stress_ij ** 2)

    # -----    phase field gradient potential ----- #
    # partial derivative of  phase field gradient potential = 2/eta int (  (grad(rho))^2 )    dx
    #  (D rho, D I) ==  ( D I, D rho) and thus  == I D_t D rho
    # I implement it in the way = 2/eta (  I D_t D rho )
    # phase_field_gradient = discretization.apply_gradient_operator(phase_field_1nxyz)
    # phase_field_gradient = discretization.apply_quadrature_weights_on_gradient_field(phase_field_gradient)
    # Dt_D_rho = discretization.apply_gradient_transposed_operator(phase_field_gradient)
    #
    # dgradrho_drho = 2 * Dt_D_rho
    dgradrho_drho = partial_derivative_of_gradient_of_phase_field_potential(discretization=discretization,
                                                                            phase_field_1nxyz=phase_field_1nxyz)
    # -----    Double well potential ----- #

    # Derivative of the double-well potential with respect to phase-field
    # phase field potential = int ( rho^2(1-rho)^2 )/eta   dx
    # gradient phase field potential = int ((2 * phase_field( + 2 * phase_field^2  -  3 * phase_field +1 )) )/eta   dx
    # d/dρ(ρ^2 (1 - ρ)^2) = 2 ρ (2 ρ^2 - 3 ρ + 1)

    # integrant_fnxyz = (2 * phase_field_1nxyz * (2 * phase_field_1nxyz * phase_field_1nxyz - 3 * phase_field_1nxyz + 1))

    # integral_fnxyz = (integrant_fnxyz / np.prod(integrant_fnxyz.shape)) * discretization.cell.domain_volume
    # there is no sum here
    # ddouble_well_drho_drho = integral_fnxyz
    ddouble_well_drho_drho = partial_der_of_double_well_potential_wrt_density_analytical(discretization=discretization,
                                                                                         phase_field_1nxyz=phase_field_1nxyz)
    # sum of all parts of df_drho
    df_drho = dfstress_drho + weight * (dgradrho_drho * eta + ddouble_well_drho_drho / eta)

    # --------------------------------------
    # Solve adjoint problem ∂f/∂u=-∂g/∂u
    # Dt C D lambda = - 2/|omega| Dt: C : sigma_diff
    # material_data_field_C_0_rho_ijklqxyz = material_data_field_ijklqxyz[..., :, :] * np.power(phase_field_1nxyz,
    #                                                                                          p)

    material_data_field_C_0_rho_ijklqxyz = material_data_field_ijklqxyz[..., :, :, :] * (
        np.power(
            phase_field_at_quad_poits_1qnxyz,
            p))[0, :, 0, ...]
    # stress difference potential
    # rhs=-Dt*wA*E  -- we can use it to assemble df_du_field

    stress_difference_ijqxyz = discretization.get_gradient_size_field()
    stress_difference_ijqxyz[:, :, ...] = stress_difference_ij[
        (...,) + (np.newaxis,) * (stress_difference_ijqxyz.ndim - 2)]

    df_du_field = discretization.get_rhs(material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                                         macro_gradient_field_ijqxyz=stress_difference_ijqxyz) / discretization.cell.domain_volume  # minus sign is already there

    # df_du_field_TEST   = np.zeros_like(df_du_field)

    # Normalization
    df_du_field = 2 * df_du_field / np.sum(target_stress_ij ** 2)
    # if MPI.COMM_WORLD.size == 1:
    # print('df_du_field = {}'.format(np.linalg.norm(df_du_field)))

    #
    K_fun = lambda x: discretization.apply_system_matrix(material_data_field=material_data_field_C_0_rho_ijklqxyz,
                                                         displacement_field=x,
                                                         formulation=formulation)
    # M_fun = lambda x: 1 * x
    # preconditioner = discretization.get_preconditioner_NEW(
    #     reference_material_data_field_ijklqxyz=material_data_field_ijklqxyz)
    # M_fun = lambda x: discretization.apply_preconditioner_NEW(preconditioner_Fourier_fnfnqks=preconditioner,
    #                                                           nodal_field_fnxyz=x)

    # K_matrix=discretization.get_system_matrix(material_data_field=material_data_field_C_0_rho_ijklqxyz)
    # print('K_matrix norm adjoint = {}'.format(np.linalg.norm(K_matrix)))

    # solve the system

    adjoint_field_fnxyz, adjoint_norms = solvers.PCG(Afun=K_fun,
                                                     B=-df_du_field,
                                                     x0=None,
                                                     P=preconditioner_fun,
                                                     steps=int(1500),
                                                     toler=1e-12)
    if MPI.COMM_WORLD.rank == 0:
        nb_it_comb = len(adjoint_norms['residual_rz'])
        # print(' nb_ steps CG adjoint_norms =' f'{nb_it_comb}')
    # gradient of adjoint_field
    adjoint_field_gradient_ijqxyz = discretization.apply_gradient_operator_symmetrized(adjoint_field_fnxyz)

    # ddot22 = lambda A2, B2:  np.einsum('ijqxyz  ,jiqxyz  ->qxyz    ', A2, B2)
    double_contraction_stress_qxyz = np.einsum('ij...,ij...->...',
                                               adjoint_field_gradient_ijqxyz,
                                               stress_field_ijqxyz)

    # dg_drho_nxyz = np.zeros(phase_field_1nxyz.shape)
    dg_drho_nxyz_mpi = np.zeros(phase_field_1nxyz.shape)

    for pixel_node in np.ndindex(
            *np.ones([discretization.domain_dimension], dtype=int) * 2):  # iteration over all voxel corners
        pixel_node = np.asarray(pixel_node)
        if discretization.domain_dimension == 2:
            # N_at_quad_points_qnijk
            div_fnxyz_pixel_node = np.einsum('qn,qxy->nxy',
                                             N_at_quad_points_qnijk[(..., *pixel_node)],
                                             double_contraction_stress_qxyz)

            # dg_drho_nxyz += np.roll(div_fnxyz_pixel_node, 1 * pixel_node, axis=(1, 2))
            dg_drho_nxyz_mpi += discretization.roll(discretization.fft, div_fnxyz_pixel_node, 1 * pixel_node,
                                                    axis=(0, 1))
        elif discretization.domain_dimension == 3:

            div_fnxyz_pixel_node = np.einsum('dqn,dqxyz->nxyz',
                                             N_at_quad_points_qnijk[(..., *pixel_node)],
                                             double_contraction_stress_qxyz)

            # dg_drho_nxyz += np.roll(div_fnxyz_pixel_node, 1 * pixel_node, axis=(1, 2, 3))
            dg_drho_nxyz_mpi += discretization.roll(discretization.fft, div_fnxyz_pixel_node, 1 * pixel_node,
                                                    axis=(0, 1, 2))

            warnings.warn('Gradient transposed is not tested for 3D.')
    test = True
    if test == True:
        material_data_field_C_0_rho_ijklqxyz = discretization.apply_quadrature_weights(
            material_data_field_C_0_rho_ijklqxyz)
        stress = discretization.apply_material_data(material_data_field_C_0_rho_ijklqxyz, strain_ijqxyz)
        force = discretization.apply_gradient_transposed_operator(stress)
        adjoint_potential_field = np.einsum('i...,i...->...', adjoint_field_fnxyz, force)

        adjoint_energy = np.sum(adjoint_potential_field)
        sensitivity_parts = {'dfstress_drho': np.linalg.norm(dfstress_drho),
                             'dgradrho_drho': np.linalg.norm(dgradrho_drho),
                             'ddouble_well_drho_drho': np.linalg.norm(ddouble_well_drho_drho),
                             'dphase_drho': np.linalg.norm(dgradrho_drho + ddouble_well_drho_drho),
                             'df_drho_': np.linalg.norm(dfstress_drho + dgradrho_drho + ddouble_well_drho_drho),
                             'df_drho': np.linalg.norm(df_drho),
                             'dg_drho_nxyz_mpi': np.linalg.norm(dg_drho_nxyz_mpi),
                             'sensitivity': np.linalg.norm(df_drho + dg_drho_nxyz_mpi),
                             'adjoint_energy': adjoint_energy}
        # print(sensitivity_parts)

        return df_drho + dg_drho_nxyz_mpi, sensitivity_parts

    else:
        return df_drho + dg_drho_nxyz_mpi


def sensitivity_with_adjoint_problem_FE_NEW(discretization,
                                            base_material_data_ijkl,
                                            displacement_field_inxyz,
                                            macro_gradient_field_ijqxyz,
                                            phase_field_1nxyz,
                                            target_stress_ij,
                                            actual_stress_ij,
                                            preconditioner_fun,
                                            system_matrix_fun,
                                            formulation,
                                            p,
                                            eta,
                                            weight,
                                            double_well_depth=1):
    # Input:
    #        material_data_field_ijklqxyz [d,d,d,d,q,x,y,z] - elasticity tensors without applied phase field -- C_0
    #        displacement_field_fnxyz [f,n,x,y,z]
    #        phase_field_1nxyz [1,n,x,y,z]
    #        macro_gradient_field_ijqxyz [d,d,q,x,y,z]
    #        target_stress_ij [d,d]
    #        actual_stress_ij [d,d] # homogenized stress
    #        formulation  - 'finite_strain', 'small_strain'
    #        p [1]  # polynomial order of a material interpolation
    #        eta [1] # weight parameter for balancing the phase field terms

    # Output:
    #        df_drho_fnxyz [1,n,x,y,z]
    # -- -- -- -- -- -- -- -- -- -- --

    # -----    stress difference potential ----- #
    # Gradient of material data with respect to phase field
    phase_field_at_quad_poits_1qnxyz, N_at_quad_points_qnijk = discretization.evaluate_field_at_quad_points(
        nodal_field_fnxyz=phase_field_1nxyz.s,
        quad_field_fqnxyz=None,
        quad_points_coords_iq=None)

    material_data_field_rho_ijklqxyz = discretization.get_material_data_size_field(
        name='data_field_in_sensitivity_with_adjoint_problem_FE_NEW')
    material_data_field_rho_ijklqxyz.s = base_material_data_ijkl[..., np.newaxis, np.newaxis, np.newaxis] * \
                                         np.power(phase_field_at_quad_poits_1qnxyz, p)[0, :, 0, ...]
    # dmaterial_data_field_drho_ijklqxyz = material_data_field_ijklqxyz[..., :, :, :] *
    # np.power(p * phase_field_at_quad_poits_1qnxyz, p - 1)[0, :, 0, ...]

    # d_stress_d_rho phase field gradient potential for a phase field without perturbation
    dstress_drho = partial_derivative_of_objective_function_stress_equivalence_wrt_phase_field_FE(
        discretization=discretization,
        material_data_field_ijkl=base_material_data_ijkl,
        phase_field_1nxyz=phase_field_1nxyz,
        target_stress_ij=target_stress_ij,
        actual_stress_ij=actual_stress_ij,
        displacement_field_fnxyz=displacement_field_inxyz,
        macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
        p=p)

    # -----    Double well potential ----- #

    # Derivative of the double-well potential with respect to phase-field
    # phase field potential = int ( rho^2(1-rho)^2 )/eta   dx
    # gradient phase field potential = int ((2 * phase_field( + 2 * phase_field^2  -  3 * phase_field +1 )) )/eta   dx
    # d/dρ(ρ^2 (1 - ρ)^2) = 2 ρ (2 ρ^2 - 3 ρ + 1)
    # print(f'id(phase_field_1nxyz)={id(phase_field_1nxyz)}')
    ddw_drho = partial_der_of_double_well_potential_wrt_density_analytical(discretization=discretization,
                                                                           phase_field_1nxyz=phase_field_1nxyz)
    # print(f'id(phase_field_1nxyz)={id(phase_field_1nxyz)}')

    # -----    phase field gradient potential ----- #
    # partial derivative of  phase field gradient potential = 2/eta int (  (grad(rho))^2 )    dx
    #  (D rho, D I) ==  ( D I, D rho) and thus  == I D_t D rho
    dgradrho_drho = discretization.get_scalar_field(name='dgradrho_drho')
    dgradrho_drho.s.fill(0)
    dgradrho_drho = partial_derivative_of_gradient_of_phase_field_potential(discretization=discretization,
                                                                            phase_field_1nxyz=phase_field_1nxyz,
                                                                            output_1nxyz=dgradrho_drho)

    # sum of all parts of df_drho
    df_drho = weight * dstress_drho + (dgradrho_drho.s * eta + double_well_depth * ddw_drho / eta)

    # Adjoint problem
    stress_difference_ijqxyz = discretization.get_gradient_size_field(
        name='stress_difference_ijqxyz_in_sensitivity_with_adjoint_problem')
    stress_difference_ij = target_stress_ij - actual_stress_ij
    # stress_difference_ijqxyz[:, :, ...] = stress_difference_ij[
    #     (...,) + (np.newaxis,) * (stress_difference_ijqxyz.s.ndim - 2)]
    stress_difference_ijqxyz = discretization.get_macro_gradient_field(macro_gradient_ij=stress_difference_ij,
                                                                       macro_gradient_field_ijqxyz=stress_difference_ijqxyz
                                                                       )
    # minus sign is already there
    df_du_field = discretization.get_unknown_size_field(name='adjoint_problem_rhs')
    df_du_field = discretization.get_rhs(material_data_field_ijklqxyz=material_data_field_rho_ijklqxyz,
                                         macro_gradient_field_ijqxyz=stress_difference_ijqxyz,
                                         rhs_inxyz=df_du_field)  # minus sign is already there
    df_du_field.s = -2 * df_du_field.s / discretization.cell.domain_volume  # this is now on the right hand side
    # Normalization
    df_du_field.s = weight * df_du_field.s / np.sum(target_stress_ij ** 2)

    # solve adjoint problem
    adjoint_field_inxyz = discretization.get_unknown_size_field(
        name='adjoint_field_inxyz_in_sensitivity_with_adjoint_problem')
    adjoint_field_inxyz.s, adjoint_norms = solvers.PCG(Afun=system_matrix_fun,
                                                       B=df_du_field,
                                                       x0=None,
                                                       P=preconditioner_fun,
                                                       steps=int(2000),
                                                       toler=1e-10,
                                                       norm_type='rr_rel'
                                                       )
    if MPI.COMM_WORLD.rank == 0:
        nb_it_comb = len(adjoint_norms['residual_rz'])
        norm_rz = adjoint_norms['residual_rz'][-1]
        print(' nb_ steps CG adjoint =' f'{nb_it_comb}, residual_rz = {norm_rz}')

    dadjoin_drho = discretization.get_scalar_field(name='dadjoin_drho')
    dadjoin_drho = partial_derivative_of_adjoint_potential_wrt_phase_field_FE(
        discretization=discretization,
        base_material_data_ijkl=base_material_data_ijkl,
        displacement_field_fnxyz=displacement_field_inxyz,
        macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
        phase_field_1nxyz=phase_field_1nxyz,
        adjoint_field_inxyz=adjoint_field_inxyz,
        output_field_inxyz=dadjoin_drho,
        p=p)

    stress_field_ijqxyz = discretization.get_gradient_size_field(name='stress_field_ijqxyz_in_sensitivityFENEW')
    stress_field_ijqxyz = discretization.get_stress_field(
        material_data_field_ijklqxyz=material_data_field_rho_ijklqxyz,
        displacement_field_inxyz=displacement_field_inxyz,
        macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
        output_stress_field_ijqxyz=stress_field_ijqxyz,
        formulation='small_strain')

    adjoint_energy = adjoint_potential(
        discretization=discretization,
        stress_field_ijqxyz=stress_field_ijqxyz,
        adjoint_field_inxyz=adjoint_field_inxyz)

    test = True
    if test == True:

        sensitivity_parts = {'dfstress_drho': np.linalg.norm(dstress_drho),
                             'dgradrho_drho': np.linalg.norm(dgradrho_drho),
                             'ddouble_well_drho_drho': np.linalg.norm(ddw_drho),
                             'dphase_drho': np.linalg.norm(dgradrho_drho + ddw_drho),
                             'df_drho_': np.linalg.norm(dstress_drho + dgradrho_drho + ddw_drho),
                             'df_drho': np.linalg.norm(df_drho),
                             'dg_drho_nxyz_mpi': np.linalg.norm(dadjoin_drho),
                             'sensitivity': np.linalg.norm(df_drho + dadjoin_drho),
                             'adjoint_energy': adjoint_energy}
        # print(sensitivity_parts)

        return df_drho + dadjoin_drho.s, sensitivity_parts

    else:
        return df_drho + dadjoin_drho.s


def sensitivity_stress_and_adjoint_FE_NEW(discretization,
                                          base_material_data_ijkl,
                                          displacement_field_inxyz,
                                          adjoint_field_inxyz,
                                          macro_gradient_field_ijqxyz,
                                          phase_field_1nxyz,
                                          target_stress_ij,
                                          actual_stress_ij,
                                          preconditioner_fun,
                                          system_matrix_fun,
                                          formulation,
                                          p,
                                          weight, disp=True):
    # Input:
    #        material_data_field_ijklqxyz [d,d,d,d,q,x,y,z] - elasticity tensors without applied phase field -- C_0
    #        displacement_field_fnxyz [f,n,x,y,z]
    #        phase_field_1nxyz [1,n,x,y,z]
    #        macro_gradient_field_ijqxyz [d,d,q,x,y,z]
    #        target_stress_ij [d,d]
    #        actual_stress_ij [d,d] # homogenized stress
    #        formulation  - 'finite_strain', 'small_strain'
    #        p [1]  # polynomial order of a material interpolation
    #        eta [1] # weight parameter for balancing the phase field terms

    # Output:
    #        df_drho_fnxyz [1,n,x,y,z]
    # -- -- -- -- -- -- -- -- -- -- --

    # -----    stress difference potential ----- #
    # Gradient of material data with respect to phase field
    phase_field_at_quad_poits_1qnxyz, N_at_quad_points_qnijk = discretization.evaluate_field_at_quad_points(
        nodal_field_fnxyz=phase_field_1nxyz.s,
        quad_field_fqnxyz=None,
        quad_points_coords_iq=None)

    material_data_field_rho_ijklqxyz = discretization.get_material_data_size_field(
        name='data_field_in_sensitivity_stress_and_adjoint_FE_NEW')
    material_data_field_rho_ijklqxyz.s = base_material_data_ijkl[..., np.newaxis, np.newaxis, np.newaxis] * \
                                         np.power(phase_field_at_quad_poits_1qnxyz, p)[0, :, 0, ...]
    # material_data_field_rho_ijklqxyz = material_data_field_ijklqxyz[..., :, :, :] * (
    #     np.power(phase_field_at_quad_poits_1qnxyz, p)[0, :, 0, ...])

    # dmaterial_data_field_drho_ijklqxyz = material_data_field_ijklqxyz[..., :, :, :] * np.power(
    #     p * phase_field_at_quad_poits_1qnxyz, p - 1)[0, :, 0, ...]

    # d_stress_d_rho phase field gradient potential for a phase field without perturbation
    dstress_drho = partial_derivative_of_objective_function_stress_equivalence_wrt_phase_field_FE(
        discretization=discretization,
        phase_field_1nxyz=phase_field_1nxyz,
        target_stress_ij=target_stress_ij,
        actual_stress_ij=actual_stress_ij,
        material_data_field_ijkl=base_material_data_ijkl,
        displacement_field_fnxyz=displacement_field_inxyz,
        macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
        p=p)

    # Adjoint problem
    # compute strain field from to displacement and macro gradient
    # strain_fluctuation_ijqxyz = discretization.apply_gradient_operator_symmetrized(displacement_field_fnxyz)
    stress_difference_ij = target_stress_ij - actual_stress_ij

    stress_difference_ijqxyz = discretization.get_gradient_size_field(
        name='stress_difference_ijqxyz_in_sensitivity_stress_and_adjoint_FE_NEW')
    # stress_difference_ijqxyz = discretization.get_gradient_size_field()
    # stress_difference_ijqxyz[:, :, ...] = stress_difference_ij[
    #     (...,) + (np.newaxis,) * (stress_difference_ijqxyz.ndim - 2)]
    stress_difference_ijqxyz = discretization.get_macro_gradient_field(macro_gradient_ij=stress_difference_ij,
                                                                       macro_gradient_field_ijqxyz=stress_difference_ijqxyz
                                                                       )
    # minus sign is already there
    # df_du_field = (2 * discretization.get_rhs(material_data_field_ijklqxyz=material_data_field_rho_ijklqxyz,
    #                                           macro_gradient_field_ijqxyz=stress_difference_ijqxyz)
    #                / discretization.cell.domain_volume)  # minus sign is already there
    # minus sign is already there
    df_du_field = discretization.get_unknown_size_field(
        name='adjoint_problem_rhs_in_sensitivity_stress_and_adjoint_FE_NEW')
    df_du_field = discretization.get_rhs(material_data_field_ijklqxyz=material_data_field_rho_ijklqxyz,
                                         macro_gradient_field_ijqxyz=stress_difference_ijqxyz,
                                         rhs_inxyz=df_du_field)
    # minus sign is already there
    df_du_field.s = -2 * df_du_field.s / discretization.cell.domain_volume
    # Normalization
    df_du_field.s = weight * df_du_field.s / np.sum(target_stress_ij ** 2)
    # TODO
    # df_du_field = df_du_field / np.sum(target_stress_ij ** 2)

    adjoint_field_inxyz.s, adjoint_norms = solvers.PCG(Afun=system_matrix_fun,
                                                       B=df_du_field.s,
                                                       x0=adjoint_field_inxyz.s,
                                                       P=preconditioner_fun,
                                                       steps=int(10000),
                                                       toler=1e-14,
                                                       norm_type='rr_rel', )
    if disp and MPI.COMM_WORLD.rank == 0:
        nb_it_comb = len(adjoint_norms['residual_rz'])
        norm_rz = adjoint_norms['residual_rz'][-1]
        print(' nb_ steps CG adjoint =' f'{nb_it_comb}, residual_rz = {norm_rz}')
    dadjoin_drho = discretization.get_scalar_field(name='dadjoin_drho_in_sensitivity_stress_and_adjoint_FE_NEW')
    dadjoin_drho = partial_derivative_of_adjoint_potential_wrt_phase_field_FE(
        discretization=discretization,
        base_material_data_ijkl=base_material_data_ijkl,
        displacement_field_fnxyz=displacement_field_inxyz,
        macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
        phase_field_1nxyz=phase_field_1nxyz,
        adjoint_field_inxyz=adjoint_field_inxyz,
        output_field_inxyz=dadjoin_drho,
        p=p)

    stress_field_ijqxyz = discretization.get_gradient_size_field(
        name='stress_field_ijqxyz_in_sensitivity_stress_and_adjoint_FE_NEW')
    stress_field = discretization.get_stress_field(
        material_data_field_ijklqxyz=material_data_field_rho_ijklqxyz,
        displacement_field_inxyz=displacement_field_inxyz,
        macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
        output_stress_field_ijqxyz=stress_field_ijqxyz,
        formulation='small_strain')

    adjoint_energy = adjoint_potential(
        discretization=discretization,
        stress_field_ijqxyz=stress_field,
        adjoint_field_inxyz=adjoint_field_inxyz)

    return weight * dstress_drho + dadjoin_drho, adjoint_field_inxyz, adjoint_energy
    # return dstress_drho + dadjoin_drho, adjoint_field_fnxyz, adjoint_energy


def sensitivity_elastic_energy_and_adjoint_FE_NEW(discretization,
                                                  base_material_data_ijkl,
                                                  displacement_field_inxyz,
                                                  adjoint_field_inxyz,
                                                  macro_gradient_field_ijqxyz,
                                                  left_macro_gradient_ij,
                                                  phase_field_1nxyz,
                                                  target_stress_ij,
                                                  actual_stress_ij,
                                                  preconditioner_fun,
                                                  system_matrix_fun,
                                                  formulation,
                                                  target_energy,
                                                  p,
                                                  weight,
                                                  disp=False):
    # Input:
    #        material_data_field_ijklqxyz [d,d,d,d,q,x,y,z] - elasticity tensors without applied phase field -- C_0
    #        displacement_field_fnxyz [f,n,x,y,z]
    #        phase_field_1nxyz [1,n,x,y,z]
    #        macro_gradient_field_ijqxyz [d,d,q,x,y,z]
    #        target_stress_ij [d,d]
    #        actual_stress_ij [d,d] # homogenized stress
    #        formulation  - 'finite_strain', 'small_strain'
    #        p [1]  # polynomial order of a material interpolation
    #        eta [1] # weight parameter for balancing the phase field terms

    # Output:
    #        df_drho_fnxyz [1,n,x,y,z]
    # -- -- -- -- -- -- -- -- -- -- --

    # -----    stress difference potential ----- #
    # Gradient of material data with respect to phase field
    phase_field_at_quad_poits_1qnxyz, N_at_quad_points_qnijk = discretization.evaluate_field_at_quad_points(
        nodal_field_fnxyz=phase_field_1nxyz.s,
        quad_field_fqnxyz=None,
        quad_points_coords_iq=None)

    material_data_field_rho_ijklqxyz = discretization.get_material_data_size_field(
        name='data_field_in_sensitivity_elastic_energy_and_adjoint_FE_NEW')
    material_data_field_rho_ijklqxyz.s = base_material_data_ijkl[..., np.newaxis, np.newaxis, np.newaxis] * \
                                         np.power(phase_field_at_quad_poits_1qnxyz, p)[0, :, 0, ...]

    # strain_fluctuation_ijqxyz = discretization.apply_gradient_operator_symmetrized(displacement_field_fnxyz)
    # strain_ijqxyz = macro_gradient_field_ijqxyz + strain_fluctuation_ijqxyz
    # stress_fluctuation_ijqxyz = discretization.apply_material_data(material_data_field_rho_ijklqxyz,
    #                                                                strain_fluctuation_ijqxyz)
    # stress_field_ijqxyz = discretization.apply_material_data(material_data_field_rho_ijklqxyz,
    #                                                          strain_ijqxyz)
    # dmaterial_data_field_drho_ijklqxyz = material_data_field_ijklqxyz[..., :, :, :] * np.power(
    #     p * phase_field_at_quad_poits_1qnxyz, p - 1)[0, :, 0, ...]
    # compute strain field from to displacement and macro gradient
    # strain_fluctuation_ijqxyz = discretization.apply_gradient_operator_symmetrized(displacement_field_fnxyz)
    # compute strain field from to displacement and macro gradient

    # strain_fluctuation_ijqxyz = discretization.apply_gradient_operator_symmetrized(displacement_field_fnxyz)
    # strain_ijqxyz = macro_gradient_field_ijqxyz + strain_fluctuation_ijqxyz

    # # stress_difference_ijqxyz = discretization.get_gradient_size_field()
    # # stress_difference_ijqxyz[:, :, ...] = stress_difference_ij[
    # #     (...,) + (np.newaxis,) * (stress_difference_ijqxyz.ndim - 2)]
    # stress_difference_ijqxyz = stress_field_ijqxyz - target_stress_ij[
    #     (...,) + (np.newaxis,) * (stress_field_ijqxyz.ndim - 2)]
    # # stress_difference_ijqxyz[:, :, ...] = stress_difference_ij[
    # #     (...,) + (np.newaxis,) * (stress_difference_ijqxyz.ndim - 2)]

    # f_sigmas_energy -= target_energy
    # d_stress_d_rho phase field gradient potential for a phase field without perturbation

    dstress_drho = partial_derivative_of_energy_equivalence_wrt_phase_field_FE(discretization=discretization,
                                                                               phase_field_1nxyz=phase_field_1nxyz,
                                                                               target_stress_ij=target_stress_ij,
                                                                               actual_stress_ij=actual_stress_ij,
                                                                               base_material_data_ijkl=base_material_data_ijkl,
                                                                               displacement_field_fnxyz=displacement_field_inxyz,
                                                                               macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
                                                                               left_macro_gradient_ij=left_macro_gradient_ij,
                                                                               target_energy=target_energy,
                                                                               p=p)
    stress_difference_ij = target_stress_ij - actual_stress_ij
    f_sigmas_energy = np.einsum('ij,ij->...',
                                left_macro_gradient_ij,
                                stress_difference_ij)
    dstress_drho = f_sigmas_energy * dstress_drho
    # Adjoint problem

    # compute strain field from to displacement and macro gradient
    # strain_fluctuation_ijqxyz = discretization.apply_gradient_operator_symmetrized(displacement_field_fnxyz)
    # compute strain field from to displacement and macro gradient

    # df_du_field_CE = -(discretization.get_rhs(material_data_field_ijklqxyz=material_data_field_rho_ijklqxyz,
    #                                           macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz)
    #                    / discretization.cell.domain_volume)
    # df_du_field_Cgradu = -(discretization.get_rhs(material_data_field_ijklqxyz=material_data_field_rho_ijklqxyz,
    #                                           macro_gradient_field_ijqxyz=stress_difference_ijqxyz)
    #                    / discretization.cell.domain_volume)
    # df_du_field_BtSdiff = -(discretization.get_rhs(material_data_field_ijklqxyz=material_data_field_rho_ijklqxyz,
    #                                               macro_gradient_field_ijqxyz=stress_difference_ijqxyz)
    #                        / discretization.cell.domain_volume)
    # stress_difference_ijqxyz = discretization.apply_quadrature_weights_on_gradient_field(stress_difference_ijqxyz)
    # df_du_field_BtSdiff = 2 * discretization.apply_gradient_transposed_operator(stress_difference_ijqxyz)
    left_macro_gradient_ijqxyz = discretization.get_gradient_size_field(
        name='left_macro_gradient_ijqxyz_in_sensitivity_stress_and_adjoint_FE_NEW')
    left_macro_gradient_ijqxyz = discretization.get_macro_gradient_field(macro_gradient_ij=left_macro_gradient_ij,
                                                                         macro_gradient_field_ijqxyz=left_macro_gradient_ijqxyz
                                                                         )

    df_du_field = discretization.get_unknown_size_field(
        name='adjoint_problem_rhs_in_sensitivity_elastic_energy_and_adjoint_FE_NEW')

    df_du_field = discretization.get_rhs(material_data_field_ijklqxyz=material_data_field_rho_ijklqxyz,
                                         macro_gradient_field_ijqxyz=left_macro_gradient_ijqxyz,
                                         rhs_inxyz=df_du_field)

    df_du_field.s = -2 * df_du_field.s / discretization.cell.domain_volume
    # Normalization
    # df_du_field = weight * (2 * f_sigmas_energy * 2 * df_du_field_BtSdiff) / target_energy ** 2
    df_du_field.s = weight * (f_sigmas_energy * df_du_field.s) / (target_energy ** 2)

    # adjoint_field_inxyz = discretization.get_unknown_size_field(
    #     name='adjoint_field_inxyz_sensitivity_elastic_energy_and_adjoint_FE_NEW')
    adjoint_field_inxyz.s, adjoint_norms = solvers.PCG(Afun=system_matrix_fun,
                                                       B=df_du_field.s,
                                                       x0=adjoint_field_inxyz.s,
                                                       P=preconditioner_fun,
                                                       steps=int(10000),
                                                       toler=1e-14)
    if disp and MPI.COMM_WORLD.rank == 0:
        nb_it_comb = len(adjoint_norms['residual_rz'])
        norm_rz = adjoint_norms['residual_rz'][-1]
        print(' nb_ steps CG adjoint =' f'{nb_it_comb}, residual_rz = {norm_rz}')

    dadjoin_drho = discretization.get_scalar_field(name='dadjoin_drho_in_sensitivity_elastic_energy_and_adjoint_FE_NEW')
    dadjoin_drho = partial_derivative_of_adjoint_potential_wrt_phase_field_FE(
        discretization=discretization,
        base_material_data_ijkl=base_material_data_ijkl,
        displacement_field_fnxyz=displacement_field_inxyz,
        macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
        phase_field_1nxyz=phase_field_1nxyz,
        adjoint_field_inxyz=adjoint_field_inxyz,
        output_field_inxyz=dadjoin_drho,
        p=p)

    stress_field_ijqxyz = discretization.get_gradient_size_field(
        name='stress_field_ijqxyz_in_sensitivity_stress_and_adjoint_FE_NEW')
    stress_field = discretization.get_stress_field(
        material_data_field_ijklqxyz=material_data_field_rho_ijklqxyz,
        displacement_field_inxyz=displacement_field_inxyz,
        macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
        output_stress_field_ijqxyz=stress_field_ijqxyz,
        formulation='small_strain')

    adjoint_energy = adjoint_potential(
        discretization=discretization,
        stress_field_ijqxyz=stress_field,
        adjoint_field_inxyz=adjoint_field_inxyz)
    test = True
    if disp and MPI.COMM_WORLD.rank == 0:
        print({'dfstress_drho': np.linalg.norm(dstress_drho),
               'df_du_field': np.linalg.norm(df_du_field),
               'f_sigmas_energy': np.linalg.norm(f_sigmas_energy),
               # 'df_du_field_BtSdiff': np.linalg.norm(df_du_field_BtSdiff),
               'dstress_drho': np.linalg.norm(dstress_drho),
               'adjoint_energy': adjoint_energy})
    return weight * dstress_drho + dadjoin_drho, adjoint_field_inxyz, adjoint_energy


def sensitivity_phase_field_term_FE_NEW(discretization,
                                        base_material_data_ijkl,
                                        phase_field_1nxyz,
                                        p,
                                        eta,
                                        double_well_depth=1):
    # Input:
    #        material_data_field_ijklqxyz [d,d,d,d,q,x,y,z] - elasticity tensors without applied phase field -- C_0
    #        displacement_field_fnxyz [f,n,x,y,z]
    #        phase_field_1nxyz [1,n,x,y,z]
    #        macro_gradient_field_ijqxyz [d,d,q,x,y,z]
    #        target_stress_ij [d,d]
    #        actual_stress_ij [d,d] # homogenized stress
    #        formulation  - 'finite_strain', 'small_strain'
    #        p [1]  # polynomial order of a material interpolation
    #        eta [1] # weight parameter for balancing the phase field terms

    # Output:
    #        df_drho_fnxyz [1,n,x,y,z]
    # -- -- -- -- -- -- -- -- -- -- --

    # -----    stress difference potential ----- #
    # Gradient of material data with respect to phase field
    phase_field_at_quad_poits_1qnxyz, N_at_quad_points_qnijk = discretization.evaluate_field_at_quad_points(
        nodal_field_fnxyz=phase_field_1nxyz.s,
        quad_field_fqnxyz=None,
        quad_points_coords_iq=None)

    material_data_field_rho_ijklqxyz = discretization.get_material_data_size_field(
        name='data_field_in_sensitivity_phase_field_term_FE_NEW', )

    # material_data_field_rho_ijklqxyz = material_data_field_ijklqxyz[..., :, :, :] * (
    #     np.power(phase_field_at_quad_poits_1qnxyz, p)[0, :, 0, ...])

    material_data_field_rho_ijklqxyz.s = base_material_data_ijkl[..., np.newaxis, np.newaxis, np.newaxis] * \
                                         np.power(phase_field_at_quad_poits_1qnxyz, p)[0, :, 0, ...]

    # -----    Double well potential ----- #

    # Derivative of the double-well potential with respect to phase-field
    # phase field potential = int ( rho^2(1-rho)^2 )/eta   dx
    # gradient phase field potential = int ((2 * phase_field( + 2 * phase_field^2  -  3 * phase_field +1 )) )/eta   dx
    # d/dρ(ρ^2 (1 - ρ)^2) = 2 ρ (2 ρ^2 - 3 ρ + 1)

    ddw_drho = partial_der_of_double_well_potential_wrt_density_analytical(discretization=discretization,
                                                                           phase_field_1nxyz=phase_field_1nxyz)

    # -----    phase field gradient potential ----- #
    # partial derivative of  phase field gradient potential = 2/eta int (  (grad(rho))^2 )    dx
    #  (D rho, D I) ==  ( D I, D rho) and thus  == I D_t D rho
    dgradrho_drho = discretization.get_scalar_field(name='dgradrho_drho_sensitivity_phase_field_term_FE_NEW')
    dgradrho_drho.s.fill(0)
    dgradrho_drho = partial_derivative_of_gradient_of_phase_field_potential(discretization=discretization,
                                                                            phase_field_1nxyz=phase_field_1nxyz,
                                                                            output_1nxyz=dgradrho_drho)

    # sum of all parts of df_drho
    dphase_drho = (dgradrho_drho.s * eta + double_well_depth * ddw_drho / eta)

    return dphase_drho


def sensitivity_with_adjoint_problem_FE_weights(discretization,
                                                material_data_field_ijklqxyz,
                                                displacement_field_fnxyz,
                                                macro_gradient_field_ijqxyz,
                                                phase_field_1nxyz,
                                                target_stress_ij,
                                                actual_stress_ij,
                                                formulation,
                                                p,
                                                eta,
                                                weight):
    # Input:
    #        material_data_field_ijklqxyz [d,d,d,d,q,x,y,z] - elasticity tensors without applied phase field -- C_0
    #        displacement_field_fnxyz [f,n,x,y,z]
    #        phase_field_1nxyz [1,n,x,y,z]
    #        macro_gradient_field_ijqxyz [d,d,q,x,y,z]
    #        target_stress_ij [d,d]
    #        actual_stress_ij [d,d] # homogenized stress
    #        formulation  - 'finite_strain', 'small_strain'
    #        p [1]  # polynomial order of a material interpolation
    #        eta [1] # weight parameter for balancing the phase field terms

    # Output:
    #        df_drho_fnxyz [1,n,x,y,z]
    # -- -- -- -- -- -- -- -- -- -- --

    # -----    stress difference potential ----- #
    # Gradient of material data with respect to phase field
    phase_field_at_quad_poits_1qnxyz, N_at_quad_points_qnijk = discretization.evaluate_field_at_quad_points(
        nodal_field_fnxyz=phase_field_1nxyz,
        quad_field_fqnxyz=None,
        quad_points_coords_iq=None)
    # dmaterial_data_field_drho_ijklqxyz = material_data_field_ijklqxyz[..., :, :] * (
    #         p * np.power(phase_field_1nxyz[0, 0], (p - 1)))

    dmaterial_data_field_drho_ijklqxyz = material_data_field_ijklqxyz[..., :, :, :] * (
            p * np.power(phase_field_at_quad_poits_1qnxyz[0, :, 0, ...], (p - 1)))

    # compute strain field from to displacement and macro gradient
    strain_ijqxyz = discretization.apply_gradient_operator_symmetrized(displacement_field_fnxyz)
    if p == 1:
        # TODO CHANGE strain_ijqxyz field adjustment
        strain_ijqxyz = strain_ijqxyz * np.where(phase_field_at_quad_poits_1qnxyz > 0.0001, 1, 0)

    strain_ijqxyz = macro_gradient_field_ijqxyz + strain_ijqxyz

    # compute stress field
    stress_field_ijqxyz = np.einsum('ijkl...,lk...->ij...', dmaterial_data_field_drho_ijklqxyz, strain_ijqxyz)

    # apply quadrature weights
    stress_field_ijqxyz = discretization.apply_quadrature_weights_on_gradient_field(stress_field_ijqxyz)

    # ---  part that is unique for  df_drho ---
    # stress difference
    stress_difference_ij = actual_stress_ij - target_stress_ij

    double_contraction_stress_qxyz = np.einsum('ij,ijqxy...->qxy...',
                                               stress_difference_ij,
                                               stress_field_ijqxyz)
    # Average over quad points in pixel !!!
    partial_derivative_xyz = np.zeros(phase_field_1nxyz.shape)
    for pixel_node in np.ndindex(
            *np.ones([discretization.domain_dimension], dtype=int) * 2):  # iteration over all voxel corners
        pixel_node = np.asarray(pixel_node)
        if discretization.domain_dimension == 2:
            # N_at_quad_points_qnijk
            div_fnxyz_pixel_node = np.einsum('qn,qxy->nxy',
                                             N_at_quad_points_qnijk[(..., *pixel_node)],
                                             double_contraction_stress_qxyz)

            partial_derivative_xyz += np.roll(div_fnxyz_pixel_node, 1 * pixel_node, axis=(1, 2))

        elif discretization.domain_dimension == 3:

            div_fnxyz_pixel_node = np.einsum('dqn,dqxyz->nxyz',
                                             N_at_quad_points_qnijk[(..., *pixel_node)],
                                             double_contraction_stress_qxyz)

            partial_derivative_xyz += np.roll(div_fnxyz_pixel_node, 1 * pixel_node, axis=(1, 2, 3))
            warnings.warn('Gradient transposed is not tested for 3D.')

    dfstress_drho = 2 * partial_derivative_xyz / discretization.cell.domain_volume / np.sum(target_stress_ij ** 2)

    # -----    phase field gradient potential ----- #
    # partial derivative of  phase field gradient potential = 2/eta int (  (grad(rho))^2 )    dx
    #  (D rho, D I) ==  ( D I, D rho) and thus  == I D_t D rho
    # I implement it in the way = 2/eta (  I D_t D rho )
    phase_field_gradient = discretization.apply_gradient_operator(phase_field_1nxyz)
    phase_field_gradient = discretization.apply_quadrature_weights_on_gradient_field(phase_field_gradient)
    Dt_D_rho = discretization.apply_gradient_transposed_operator(phase_field_gradient)

    dgradrho_drho = 2 * Dt_D_rho

    # -----    Double well potential ----- #

    # Derivative of the double-well potential with respect to phase-field
    # phase field potential = int ( rho^2(1-rho)^2 )/eta   dx
    # gradient phase field potential = int ((2 * phase_field( + 2 * phase_field^2  -  3 * phase_field +1 )) )/eta   dx
    # d/dρ(ρ^2 (1 - ρ)^2) = 2 ρ (2 ρ^2 - 3 ρ + 1)

    # integrant_fnxyz = (2 * phase_field_1nxyz * (2 * phase_field_1nxyz * phase_field_1nxyz - 3 * phase_field_1nxyz + 1))

    # integral_fnxyz = (integrant_fnxyz / np.prod(integrant_fnxyz.shape)) * discretization.cell.domain_volume
    # there is no sum here
    # ddouble_well_drho_drho = integral_fnxyz
    ddouble_well_drho_drho = partial_der_of_double_well_potential_wrt_density_analytical(discretization=discretization,
                                                                                         phase_field_1nxyz=phase_field_1nxyz,
                                                                                         eta=1)
    # sum of all parts of df_drho
    df_drho = weight * dfstress_drho + (dgradrho_drho * eta + ddouble_well_drho_drho / eta)

    # --------------------------------------
    # Solve adjoint problem ∂f/∂u=-∂g/∂u
    # Dt C D lambda = - 2/|omega| Dt: C : sigma_diff
    # material_data_field_C_0_rho_ijklqxyz = material_data_field_ijklqxyz[..., :, :] * np.power(phase_field_1nxyz,
    #                                                                                          p)

    material_data_field_C_0_rho_ijklqxyz = material_data_field_ijklqxyz[..., :, :, :] * (
        np.power(phase_field_at_quad_poits_1qnxyz[0, :, 0, ...], (p)))
    # stress difference potential
    # rhs=-Dt*wA*E  -- we can use it to assemble df_du_field

    stress_difference_ijqxyz = discretization.get_gradient_size_field()
    stress_difference_ijqxyz[:, :, ...] = stress_difference_ij[
        (...,) + (np.newaxis,) * (stress_difference_ijqxyz.ndim - 2)]

    df_du_field = 2 * discretization.get_rhs(material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                                             macro_gradient_field_ijqxyz=stress_difference_ijqxyz) / discretization.cell.domain_volume  # minus sign is already there
    # Normalization
    df_du_field = df_du_field / np.sum(target_stress_ij ** 2)
    #
    K_fun = lambda x: discretization.apply_system_matrix(material_data_field=material_data_field_C_0_rho_ijklqxyz,
                                                         displacement_field=x,
                                                         formulation=formulation)
    # M_fun = lambda x: 1 * x
    preconditioner = discretization.get_preconditioner(
        reference_material_data_field_ijklqxyz=material_data_field_ijklqxyz)
    M_fun = lambda x: discretization.apply_preconditioner(preconditioner_Fourier_fnfnxyz=preconditioner,
                                                          nodal_field_fnxyz=x)

    # solve the system
    adjoint_field_fnxyz, adjoint_norms = solvers.PCG(Afun=K_fun, B=df_du_field, x0=None, P=M_fun,
                                                     steps=int(500),
                                                     toler=1e-6)

    # gradient of adjoint_field
    adjoint_field_gradient_ijqxyz = discretization.apply_gradient_operator_symmetrized(adjoint_field_fnxyz)

    # ddot22 = lambda A2, B2:  np.einsum('ijqxyz  ,jiqxyz  ->qxyz    ', A2, B2)
    double_contraction_stress_qxyz = np.einsum('ij...,ij...->...',
                                               adjoint_field_gradient_ijqxyz,
                                               stress_field_ijqxyz)

    dg_drho_nxyz = np.zeros(phase_field_1nxyz.shape)
    for pixel_node in np.ndindex(
            *np.ones([discretization.domain_dimension], dtype=int) * 2):  # iteration over all voxel corners
        pixel_node = np.asarray(pixel_node)
        if discretization.domain_dimension == 2:
            # N_at_quad_points_qnijk
            div_fnxyz_pixel_node = np.einsum('qn,qxy->nxy',
                                             N_at_quad_points_qnijk[(..., *pixel_node)],
                                             double_contraction_stress_qxyz)

            dg_drho_nxyz += np.roll(div_fnxyz_pixel_node, 1 * pixel_node, axis=(1, 2))

        elif discretization.domain_dimension == 3:

            div_fnxyz_pixel_node = np.einsum('dqn,dqxyz->nxyz',
                                             N_at_quad_points_qnijk[(..., *pixel_node)],
                                             double_contraction_stress_qxyz)

            dg_drho_nxyz += np.roll(div_fnxyz_pixel_node, 1 * pixel_node, axis=(1, 2, 3))
            warnings.warn('Gradient transposed is not tested for 3D.')

    return df_drho + dg_drho_nxyz


def sensitivity_with_adjoint_problem_FE_testing(discretization,
                                                material_data_field_ijklqxyz,
                                                displacement_field_fnxyz,
                                                macro_gradient_field_ijqxyz,
                                                phase_field_1nxyz,
                                                target_stress_ij,
                                                actual_stress_ij,
                                                formulation,
                                                p,
                                                eta,
                                                weight):
    # Input:
    #        material_data_field_ijklqxyz [d,d,d,d,q,x,y,z] - elasticity tensors without applied phase field -- C_0
    #        displacement_field_fnxyz [f,n,x,y,z]
    #        phase_field_1nxyz [1,n,x,y,z]
    #        macro_gradient_field_ijqxyz [d,d,q,x,y,z]
    #        target_stress_ij [d,d]
    #        actual_stress_ij [d,d] # homogenized stress
    #        formulation  - 'finite_strain', 'small_strain'
    #        p [1]  # polynomial order of a material interpolation
    #        eta [1] # weight parameter for balancing the phase field terms

    # Output:
    #        df_drho_fnxyz [1,n,x,y,z]
    # -- -- -- -- -- -- -- -- -- -- --

    # -----    stress difference potential ----- #
    # Gradient of material data with respect to phase field
    phase_field_at_quad_poits_1qnxyz, N_at_quad_points_qnijk = discretization.evaluate_field_at_quad_points(
        nodal_field_fnxyz=phase_field_1nxyz,
        quad_field_fqnxyz=None,
        quad_points_coords_iq=None)
    # dmaterial_data_field_drho_ijklqxyz = material_data_field_ijklqxyz[..., :, :] * (
    #         p * np.power(phase_field_1nxyz[0, 0], (p - 1)))

    dmaterial_data_field_drho_ijklqxyz = material_data_field_ijklqxyz[..., :, :, :] * (
            p * np.power(phase_field_at_quad_poits_1qnxyz[0, :, 0, ...], (p - 1)))

    # compute strain field from to displacement and macro gradient
    strain_ijqxyz = discretization.apply_gradient_operator_symmetrized(displacement_field_fnxyz)
    strain_ijqxyz = macro_gradient_field_ijqxyz + strain_ijqxyz
    # TODO CHANGE strain_ijqxyz field adjustment

    strain_ijqxyz = strain_ijqxyz * np.where(phase_field_at_quad_poits_1qnxyz > 0.01, 1, 0)

    # compute stress field
    stress_field_ijqxyz = np.einsum('ijkl...,lk...->ij...', dmaterial_data_field_drho_ijklqxyz, strain_ijqxyz)

    # apply quadrature weights
    stress_field_ijqxyz = discretization.apply_quadrature_weights_on_gradient_field(stress_field_ijqxyz)

    # ---  part that is unique for  df_drho ---
    # stress difference
    stress_difference_ij = actual_stress_ij - target_stress_ij

    double_contraction_stress_qxyz = np.einsum('ij,ijqxy...->qxy...',
                                               stress_difference_ij,
                                               stress_field_ijqxyz)
    # Average over quad points in pixel !!!
    partial_derivative_xyz = np.zeros(phase_field_1nxyz.shape)
    for pixel_node in np.ndindex(
            *np.ones([discretization.domain_dimension], dtype=int) * 2):  # iteration over all voxel corners
        pixel_node = np.asarray(pixel_node)
        if discretization.domain_dimension == 2:
            # N_at_quad_points_qnijk
            div_fnxyz_pixel_node = np.einsum('qn,qxy->nxy',
                                             N_at_quad_points_qnijk[(..., *pixel_node)],
                                             double_contraction_stress_qxyz)

            partial_derivative_xyz += np.roll(div_fnxyz_pixel_node, 1 * pixel_node, axis=(1, 2))

        elif discretization.domain_dimension == 3:

            div_fnxyz_pixel_node = np.einsum('dqn,dqxyz->nxyz',
                                             N_at_quad_points_qnijk[(..., *pixel_node)],
                                             double_contraction_stress_qxyz)

            partial_derivative_xyz += np.roll(div_fnxyz_pixel_node, 1 * pixel_node, axis=(1, 2, 3))
            warnings.warn('Gradient transposed is not tested for 3D.')

    dfstress_drho = 2 * partial_derivative_xyz / discretization.cell.domain_volume / np.sum(target_stress_ij ** 2)

    # -----    phase field gradient potential ----- #
    # partial derivative of  phase field gradient potential = 2/eta int (  (grad(rho))^2 )    dx
    #  (D rho, D I) ==  ( D I, D rho) and thus  == I D_t D rho
    # I implement it in the way = 2/eta (  I D_t D rho )
    phase_field_gradient = discretization.apply_gradient_operator(phase_field_1nxyz)
    phase_field_gradient = discretization.apply_quadrature_weights_on_gradient_field(phase_field_gradient)
    Dt_D_rho = discretization.apply_gradient_transposed_operator(phase_field_gradient)

    dgradrho_drho = 2 * Dt_D_rho

    # -----    Double well potential ----- #

    # Derivative of the double-well potential with respect to phase-field
    # phase field potential = int ( rho^2(1-rho)^2 )/eta   dx
    # gradient phase field potential = int ((2 * phase_field( + 2 * phase_field^2  -  3 * phase_field +1 )) )/eta   dx
    # d/dρ(ρ^2 (1 - ρ)^2) = 2 ρ (2 ρ^2 - 3 ρ + 1)

    # integrant_fnxyz = (2 * phase_field_1nxyz * (2 * phase_field_1nxyz * phase_field_1nxyz - 3 * phase_field_1nxyz + 1))

    # integral_fnxyz = (integrant_fnxyz / np.prod(integrant_fnxyz.shape)) * discretization.cell.domain_volume
    # there is no sum here
    # ddouble_well_drho_drho = integral_fnxyz
    ddouble_well_drho_drho = partial_der_of_double_well_potential_wrt_density_analytical(discretization=discretization,
                                                                                         phase_field_1nxyz=phase_field_1nxyz,
                                                                                         eta=1)
    # sum of all parts of df_drho
    df_drho = dfstress_drho + weight * (dgradrho_drho * eta + ddouble_well_drho_drho / eta)

    # --------------------------------------
    # Solve adjoint problem ∂f/∂u=-∂g/∂u
    # Dt C D lambda = - 2/|omega| Dt: C : sigma_diff
    # material_data_field_C_0_rho_ijklqxyz = material_data_field_ijklqxyz[..., :, :] * np.power(phase_field_1nxyz,
    #                                                                                          p)

    material_data_field_C_0_rho_ijklqxyz = material_data_field_ijklqxyz[..., :, :, :] * (
        np.power(phase_field_at_quad_poits_1qnxyz[0, :, 0, ...], (p)))
    # stress difference potential
    # rhs=-Dt*wA*E  -- we can use it to assemble df_du_field

    stress_difference_ijqxyz = discretization.get_gradient_size_field()
    stress_difference_ijqxyz[:, :, ...] = stress_difference_ij[
        (...,) + (np.newaxis,) * (stress_difference_ijqxyz.ndim - 2)]

    df_du_field = 2 * discretization.get_rhs(material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                                             macro_gradient_field_ijqxyz=stress_difference_ijqxyz) / discretization.cell.domain_volume  # minus sign is already there

    df_du_field = df_du_field / np.sum(target_stress_ij ** 2)
    #
    K_fun = lambda x: discretization.apply_system_matrix(material_data_field=material_data_field_C_0_rho_ijklqxyz,
                                                         displacement_field=x,
                                                         formulation=formulation)
    # M_fun = lambda x: 1 * x
    preconditioner = discretization.get_preconditioner(
        reference_material_data_field_ijklqxyz=material_data_field_ijklqxyz)
    M_fun = lambda x: discretization.apply_preconditioner(preconditioner_Fourier_fnfnxyz=preconditioner,
                                                          nodal_field_fnxyz=x)

    # solve the system
    adjoint_field_fnxyz, adjoint_norms = solvers.PCG(Afun=K_fun, B=df_du_field, x0=None, P=M_fun,
                                                     steps=int(500),
                                                     toler=1e-6)

    # gradient of adjoint_field
    adjoint_field_gradient_ijqxyz = discretization.apply_gradient_operator_symmetrized(adjoint_field_fnxyz)

    # ddot22 = lambda A2, B2:  np.einsum('ijqxyz  ,jiqxyz  ->qxyz    ', A2, B2)
    double_contraction_stress_qxyz = np.einsum('ij...,ij...->...',
                                               adjoint_field_gradient_ijqxyz,
                                               stress_field_ijqxyz)

    dg_drho_nxyz = np.zeros(phase_field_1nxyz.shape)
    for pixel_node in np.ndindex(
            *np.ones([discretization.domain_dimension], dtype=int) * 2):  # iteration over all voxel corners
        pixel_node = np.asarray(pixel_node)
        if discretization.domain_dimension == 2:
            # N_at_quad_points_qnijk
            div_fnxyz_pixel_node = np.einsum('qn,qxy->nxy',
                                             N_at_quad_points_qnijk[(..., *pixel_node)],
                                             double_contraction_stress_qxyz)

            dg_drho_nxyz += np.roll(div_fnxyz_pixel_node, 1 * pixel_node, axis=(1, 2))

        elif discretization.domain_dimension == 3:

            div_fnxyz_pixel_node = np.einsum('dqn,dqxyz->nxyz',
                                             N_at_quad_points_qnijk[(..., *pixel_node)],
                                             double_contraction_stress_qxyz)

            dg_drho_nxyz += np.roll(div_fnxyz_pixel_node, 1 * pixel_node, axis=(1, 2, 3))
            warnings.warn('Gradient transposed is not tested for 3D.')

    sensitivity_parts = {'dfstress_drho': dfstress_drho,
                         'dgradrho_drho': dgradrho_drho,
                         'ddouble_well_drho_drho': ddouble_well_drho_drho,
                         'dg_drho_nxyz': dg_drho_nxyz,
                         'sensitivity': df_drho + dg_drho_nxyz}
    return sensitivity_parts


def objective_function_small_strain_FE_testing(discretization,
                                               actual_stress_ij,
                                               target_stress_ij,
                                               phase_field_1nxyz,
                                               eta,
                                               w):
    # evaluate objective functions
    # f = (flux_h -flux_target)^2 + w*eta* int (  (grad(rho))^2 )dx  +    int ( rho^2(1-rho)^2 ) / eta   dx
    # f =  f_sigma + w*eta* f_rho_grad  + f_dw/eta

    # stress difference potential: actual_stress_ij is homogenized stress
    # stress_difference_ij = actual_stress_ij - target_stress_ij
    stress_difference_ij = (actual_stress_ij - target_stress_ij)

    f_sigma = np.sum(stress_difference_ij ** 2) / np.sum(target_stress_ij ** 2)

    # double - well potential
    # integrant = (phase_field_1nxyz ** 2) * (1 - phase_field_1nxyz) ** 2
    # f_dw = (np.sum(integrant) / np.prod(integrant.shape)) * discretization.cell.domain_volume
    f_dw = compute_double_well_potential_analytical(discretization=discretization,
                                                    phase_field_1nxyz=phase_field_1nxyz)

    phase_field_gradient = discretization.apply_gradient_operator(phase_field_1nxyz)
    f_rho_grad = np.sum(discretization.integrate_over_cell(phase_field_gradient ** 2))

    # gradient_of_phase_field = compute_gradient_of_phase_field(phase_field_gradient)

    f_rho = eta * f_rho_grad + f_dw / eta

    # print('f_sigma linear=  {} '.format(f_sigma))
    # print('f_rho_grad linear=  {} '.format(f_rho_grad))
    # print('f_dw =  linear {} '.format(f_dw))

    # print('f_rho   linear = {} '.format(f_rho))
    # print('w * f_rho linear =  {} '.format(w * f_rho))
    # print('objective_function linear = {} '.format(f_sigma + w * f_rho))

    return f_sigma + w * f_rho  # / discretization.cell.domain_volume
