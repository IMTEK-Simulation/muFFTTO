import warnings

import numpy as np
import scipy as sc
import time

from muFFTTO import domain
from muFFTTO import solvers


def objective_function_small_strain(discretization,
                                    actual_stress_ij,
                                    target_stress_ij,
                                    phase_field_1nxyz,
                                    eta=1,
                                    w=1):
    # evaluate objective functions
    # f = (flux_h -flux_target)^2 + w*eta* int (  (grad(rho))^2 )dx  +    int ( rho^2(1-rho)^2 ) / eta   dx
    # f =  f_sigma + w*eta* f_rho_grad  + f_dw/eta

    # stress difference potential: actual_stress_ij is homogenized stress
    # stress_difference_ij = actual_stress_ij - target_stress_ij
    stress_difference_ij = (actual_stress_ij - target_stress_ij)

    f_sigma = np.sum(stress_difference_ij ** 2)

    # double - well potential
    # integrant = (phase_field_1nxyz ** 2) * (1 - phase_field_1nxyz) ** 2
    # f_dw = (np.sum(integrant) / np.prod(integrant.shape)) * discretization.cell.domain_volume
    f_dw = compute_double_well_potential_analytical(discretization=discretization,
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


def compute_double_well_potential_Gauss_quad(discretization, phase_field_1nxyz, eta=1):
    # The double-well potential
    # phase field potential = int ( rho^2(1-rho)^2 ) / eta   dx
    # double - well potential
    # with interpolation for more precise integration
    # integrant = (phase_field_1nxyz ** 2) * (1 - phase_field_1nxyz) ** 2
    # integral = (np.sum(integrant) / np.prod(integrant.shape)) * discretization.cell.domain_volume
    # (ρ^2 (1 - ρ)^2) = ρ^2 - 2ρ^3 + ρ^4
    if discretization.element_type != 'linear_triangles':
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
        quad_points_coords_dq=quad_points_coord)

    quad_field_fqnxyz = (quad_field_fqnxyz ** 2) * (1 - quad_field_fqnxyz) ** 2
    # Multiply with quadrature weights
    quad_field_fqnxyz = np.einsum('fq...,q->fq...', quad_field_fqnxyz, quad_points_weights)

    return np.sum(quad_field_fqnxyz) / eta


def compute_double_well_potential_analytical(discretization, phase_field_1nxyz, eta=1):
    # The double-well potential
    # phase field potential = int ( rho^2(1-rho)^2 ) / eta   dx
    # double - well potential
    # with interpolation for more precise integration
    # (ρ^2 (1 - ρ)^2) = ρ^2 - 2ρ^3 + ρ^4
    # Phase field rho is considered as a linear combination of nodal values phase_field_1nxyz and shape FE functions

    if discretization.element_type != 'linear_triangles':
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

    rho_squared = np.sum(rho_squared_pixel(phase_field_1nxyz[0, 0],
                                           np.roll(phase_field_1nxyz[0, 0], -1, axis=(0)),
                                           np.roll(phase_field_1nxyz[0, 0], -1, axis=(1)),
                                           np.roll(phase_field_1nxyz[0, 0], -1 * np.array([1, 1]),
                                                   axis=(0, 1)))) * Jacobian_det

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
    rho_qubed = np.sum(rho_qubed_pixel(phase_field_1nxyz[0, 0],
                                       np.roll(phase_field_1nxyz[0, 0], -1, axis=(0)),
                                       np.roll(phase_field_1nxyz[0, 0], -1, axis=(1)),
                                       np.roll(phase_field_1nxyz[0, 0], -1 * np.array([1, 1]),
                                               axis=(0, 1)))) * Jacobian_det

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
    rho_quartic = np.sum(rho_quartic_pixel(phase_field_1nxyz[0, 0],
                                           np.roll(phase_field_1nxyz[0, 0], -1, axis=(0)),
                                           np.roll(phase_field_1nxyz[0, 0], -1, axis=(1)),
                                           np.roll(phase_field_1nxyz[0, 0], -1 * np.array([1, 1]),
                                                   axis=(0, 1)))) * Jacobian_det
    # (ρ^2 (1 - ρ)^2) = ρ^2 - 2ρ^3 + ρ^4
    integral = rho_squared - 2 * rho_qubed + rho_quartic

    return integral / eta


def compute_double_well_potential_analytical_fast(discretization, phase_field_1nxyz, eta=1):
    # The double-well potential
    # phase field potential = int ( rho^2(1-rho)^2 ) / eta   dx
    # double - well potential
    # with interpolation for more precise integration
    # (ρ^2 (1 - ρ)^2) = ρ^2 - 2ρ^3 + ρ^4
    # Phase field rho is considered as a linear combination of nodal values phase_field_1nxyz and shape FE functions

    if discretization.element_type != 'linear_triangles':
        raise ValueError(
            'Analytical evaluation works only for linear triangles. You provided {} '.format(
                discretization.element_type))

    Jacobian_matrix = np.diag(discretization.pixel_size)
    Jacobian_det = np.linalg.det(
        Jacobian_matrix)  # this is product of diagonal term of Jacoby transformation matrix

    # # TODO [ask Lars] is the array copied or is it just "view" on array?
    rho0 = phase_field_1nxyz[0, 0]
    rho1 = np.roll(phase_field_1nxyz[0, 0], -1, axis=(0))
    rho2 = np.roll(phase_field_1nxyz[0, 0], -1, axis=(1))
    rho3 = np.roll(phase_field_1nxyz[0, 0], -1 * np.array([1, 1]))

    '''
    rho_squared_pixel = (1 / 12) * np.sum((rho0 ** 2 + rho3 ** 2) + 2 * (rho1 ** 2 + rho2 ** 2) 
                                          + rho1 * (rho0 + rho2 + rho2 + rho3) 
                                          + rho2 * (rho3 + rho0)
                                          )

    rho_qubed_pixel = (1 / 20) * np.sum((rho0 ** 3 + rho3 ** 3) + 2 * (rho1 ** 3 + rho2 ** 3)
                                        + (rho3 ** 2 + rho0 ** 2) * (rho1 + rho2)
                                        + rho1 ** 2 * (rho0 + 2 * rho2 + rho3)
                                        + rho2 ** 2 * (rho0 + 2 * rho1 + rho3)
                                        + rho1 * rho2 * (rho0 + rho3)
                                        )

    rho_quartic_pixel = (1 / 30) * np.sum(rho0 ** 4 + rho3 ** 4+ 2 * (rho1 ** 4 + rho2 ** 4) 
                                          + rho1 ** 3 * (rho0 + 2 * rho2 + rho3)
                                          + rho2 ** 3 * (rho0 + 2 * rho1 + rho3)
                                          + rho0 ** 3 * (rho1 + rho2)
                                          + rho3 ** 3 * (rho1 + rho2)
                                          + rho1 ** 2 * (rho0 ** 2 + 2 * rho2 ** 2 + rho3 ** 2)
                                          + (rho2 ** 2 + rho1 * rho2) * (rho0 ** 2 + rho3 ** 2)
                                          + (rho0 + rho3) * (rho1 ** 2 * rho2 + rho1 * rho2 ** 2)
                                          )
    '''
    rho_pixel = (((1 / 12) * np.sum((rho0 ** 2 + rho3 ** 2) + 2 * (rho1 ** 2 + rho2 ** 2)
                                    + rho1 * (rho0 + rho2 + rho2 + rho3)
                                    + rho2 * (rho3 + rho0)
                                    )
                  -
                  (2 / 20) * np.sum((rho0 ** 3 + rho3 ** 3) + 2 * (rho1 ** 3 + rho2 ** 3)
                                    + (rho3 ** 2 + rho0 ** 2) * (rho1 + rho2)
                                    + rho1 ** 2 * (rho0 + 2 * rho2 + rho3)
                                    + rho2 ** 2 * (rho0 + 2 * rho1 + rho3)
                                    + rho1 * rho2 * (rho0 + rho3)
                                    ))
                 +
                 (1 / 30) * np.sum(rho0 ** 4 + rho3 ** 4 + 2 * (rho1 ** 4 + rho2 ** 4)
                                   + rho1 ** 3 * (rho0 + 2 * rho2 + rho3)
                                   + rho2 ** 3 * (rho0 + 2 * rho1 + rho3)
                                   + rho0 ** 3 * (rho1 + rho2)
                                   + rho3 ** 3 * (rho1 + rho2)
                                   + rho1 ** 2 * (rho0 ** 2 + 2 * rho2 ** 2 + rho3 ** 2)
                                   + (rho2 ** 2 + rho1 * rho2) * (rho0 ** 2 + rho3 ** 2)
                                   + (rho0 + rho3) * (rho1 ** 2 * rho2 + rho1 * rho2 ** 2)
                                   ))
    # (ρ^2 (1 - ρ)^2) = ρ^2 - 2ρ^3 + ρ^4
    integral = (rho_pixel) * Jacobian_det
    return integral / eta


def compute_double_well_potential(discretization, phase_field_1nxyz, eta=1):
    # The double-well potential
    # phase field potential = int ( rho^2(1-rho)^2 ) / eta   dx
    # double - well potential
    integrant = (phase_field_1nxyz ** 2) * (1 - phase_field_1nxyz) ** 2
    integral = (np.sum(integrant) / np.prod(integrant.shape)) * discretization.cell.domain_volume
    return integral / eta


def partial_der_of_double_well_potential_wrt_density_NEW(discretization, phase_field_1nxyz, eta=1):
    # Derivative of the double-well potential with respect to phase-field
    # phase field potential = int ( rho^2(1-rho)^2 )/eta   dx
    # gradient phase field potential = int ((2 * phase_field - 6 * phase_field^2  +  4 * phase_field^3 )) )/eta   dx
    # d/dρ(ρ^2 (1 - ρ)^2) = 2ρ -6ρ^2 + 4ρ^3
    # TODO[Martin]: do this part first ' integration of double well potential
    if discretization.element_type != 'linear_triangles':
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
        quad_points_coords_dq=quad_points_coord)
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


def partial_der_of_double_well_potential_wrt_density(discretization, phase_field_1nxyz, eta=1):
    # Derivative of the double-well potential with respect to phase-field
    # phase field potential = int ( rho^2(1-rho)^2 )/eta   dx
    # gradient phase field potential = int ((2 * phase_field( + 2 * phase_field^2  -  3 * phase_field +1 )) )/eta   dx
    # d/dρ(ρ^2 (1 - ρ)^2) = 2 ρ (2 ρ^2 - 3 ρ + 1)

    integrant_fnxyz = (2 * phase_field_1nxyz * (2 * phase_field_1nxyz * phase_field_1nxyz - 3 * phase_field_1nxyz + 1))

    integral_fnxyz = (integrant_fnxyz / np.prod(integrant_fnxyz.shape)) * discretization.cell.domain_volume
    # there is no sum here
    return integral_fnxyz / eta


def partial_der_of_double_well_potential_wrt_density_analytical(discretization, phase_field_1nxyz, eta=1):
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

    # integral_fnxyz =  (2 * phase_field_1nxyz[0, 0] \
    #                 - 6 * phase_field_1nxyz[0, 0] ** 2 * 6 / 12 \
    #                 + 4 * phase_field_1nxyz[0, 0] ** 3 * 6 / 20) *Jacobian_det
    # NODAL VALUES OF CONNECTED POINTS
    rho_00 = phase_field_1nxyz[0, 0]
    rho_10 = np.roll(phase_field_1nxyz[0, 0], np.array([-1, 0]), axis=(0, 1))
    rho_m10 = np.roll(phase_field_1nxyz[0, 0], np.array([1, 0]), axis=(0, 1))
    rho_01 = np.roll(phase_field_1nxyz[0, 0], np.array([0, -1]), axis=(0, 1))
    rho_0m1 = np.roll(phase_field_1nxyz[0, 0], np.array([0, 1]), axis=(0, 1))
    rho_m11 = np.roll(phase_field_1nxyz[0, 0], np.array([1, -1]), axis=(0, 1))
    rho_1m1 = np.roll(phase_field_1nxyz[0, 0], np.array([-1, 1]), axis=(0, 1))

    drho_squared = (rho_00 \
                    + (4 / 24) * rho_10 \
                    + (4 / 24) * rho_m10 \
                    + (4 / 24) * rho_01 \
                    + (4 / 24) * rho_0m1 \
                    + (4 / 24) * rho_m11 \
                    + (4 / 24) * rho_1m1) * Jacobian_det

    drho_cubed = ((18 / 20) * rho_00 ** 2 \
                  + (6 / 60) * (rho_10 ** 2 + rho_m10 ** 2 + rho_01 ** 2 + rho_0m1 ** 2 + rho_m11 ** 2 + rho_1m1 ** 2) \
                  + (12 / 60) * rho_00 * (rho_10 + rho_m10 + rho_01 + rho_0m1 + rho_m11 + rho_1m1) \
                  + (6 / 120) * (rho_10 * rho_01 + rho_01 * rho_m11 + rho_m11 * rho_m10 \
                                 + rho_m10 * rho_0m1 + rho_0m1 * rho_1m1 + rho_1m1 * rho_10) \
                  ) * Jacobian_det
    drho_quartic = ((24 / 30) * rho_00 ** 3 \
                    + (8 / 120) * (
                            rho_10 ** 3 + rho_m10 ** 3 + rho_01 ** 3 + rho_0m1 ** 3 + rho_m11 ** 3 + rho_1m1 ** 3) \
                    + (24 / 120) * rho_00 ** 2 * (rho_10 + rho_m10 + rho_01 + rho_0m1 + rho_m11 + rho_1m1) \
                    + (24 / 180) * rho_00 * (
                            rho_10 ** 2 + rho_m10 ** 2 + rho_01 ** 2 + rho_0m1 ** 2 + rho_m11 ** 2 + rho_1m1 ** 2) \
                    + (12 / 360) * (rho_10 ** 2 * rho_01 + rho_01 ** 2 * rho_m11 + rho_m11 ** 2 * rho_m10 \
                                    + rho_m10 ** 2 * rho_0m1 + rho_0m1 ** 2 * rho_1m1 + rho_1m1 ** 2 * rho_10) \
                    + (12 / 360) * (rho_10 * rho_01 ** 2 + rho_01 * rho_m11 ** 2 + rho_m11 * rho_m10 ** 2 \
                                    + rho_m10 * rho_0m1 ** 2 + rho_0m1 * rho_1m1 ** 2 + rho_1m1 * rho_10 ** 2) \
                    + (24 / 360) * rho_00 * (rho_10 * rho_01 + rho_01 * rho_m11 + rho_m11 * rho_m10 \
                                             + rho_m10 * rho_0m1 + rho_0m1 * rho_1m1 + rho_1m1 * rho_10)
                    ) * Jacobian_det

    return drho_squared - 2 * drho_cubed + drho_quartic / eta


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


def compute_gradient_of_phase_field_potential(discretization, phase_field_1nxyz, eta=1):
    # Input: phase_field [1,n,x,y,z]
    # Output: potential [1]
    # phase field gradient potential = int (  (grad(rho))^2 )    dx
    phase_field_gradient_ijqxyz = discretization.apply_gradient_operator(phase_field_1nxyz)
    f_rho_grad = np.sum(discretization.integrate_over_cell(phase_field_gradient_ijqxyz ** 2))
    return eta * f_rho_grad


def partial_derivative_of_gradient_of_phase_field_potential(discretization, phase_field_1nxyz, eta=1):
    # Input: phase_field [1,n,x,y,z]
    # Output: ∂ potential/ ∂ pha adjoint_potential ase_field [1,n,x,y,z] # Note: one potential per phase field DOF

    # partial derivative of  phase field gradient potential = 2/eta int (  (grad(rho))^2 )    dx
    # Compute       grad (rho). grad I  without using I
    #
    #  (D rho, D I) ==  ( D I, D rho) and thus  == I D_t D rho
    # I try to implement it in the way = 2/eta (int I D_t D rho )
    phase_field_gradient = discretization.apply_gradient_operator(phase_field_1nxyz)
    phase_field_gradient = discretization.apply_quadrature_weights_on_gradient_field(phase_field_gradient)
    Dt_D_rho = discretization.apply_gradient_transposed_operator(phase_field_gradient)

    # integrated_Dt_D_rho =  #/ np.prod(Dt_D_rho.shape)) * discretization.cell.domain_volume
    return 2 * Dt_D_rho * eta


##
def objective_function_stress_equivalence(discretization, actual_stress_ij, target_stress_ij):
    # Input: phase_field [1,n,x,y,z]
    # Output: f_sigma  [1]   == stress difference =  (Sigma_target-Sigma_homogenized,Sigma_target-Sigma_homogenized)

    # stress difference potential: actual_stress_ij is homogenized stress
    stress_difference_ij = actual_stress_ij - target_stress_ij

    # f_sigma = np.sum(stress_difference_ij ** 2)
    f_sigma = np.einsum('ij,ij->', stress_difference_ij, stress_difference_ij)
    # can be done np.tensordot(stress_difference, stress_difference,axes=2)
    return f_sigma


def partial_derivative_of_objective_function_stress_equivalence_wrt_phase_field(discretization,
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
    # I consider linear interpolation of material  C_ijkl= p*rho**(p-1) C^0_ijkl
    # so  ∂ C_ijkl/ ∂ rho = 1* C^0_ijkl
    # p = 2
    material_data_field_ijklqxyz = material_data_field_ijklqxyz[..., :, :] * (
            p * np.power(phase_field_1nxyz[0, 0], (p - 1)))

    # int(∂ C/ ∂ rho_i  * (macro_grad + micro_grad)) dx / | domain |

    # compute strain field from to displacement and macro gradient
    strain_ijqxyz = discretization.apply_gradient_operator_symmetrized(displacement_field_fnxyz)
    strain_ijqxyz = macro_gradient_field_ijqxyz + strain_ijqxyz

    # compute stress field
    # ddot42 = lambda A4, B2: np.einsum('ijklxyz,lkxyz  ->ijxyz  ', A4, B2)
    stress_field_ijqxyz = np.einsum('ijkl...,lk...->ij...', material_data_field_ijklqxyz, strain_ijqxyz)

    # apply quadrature weights
    stress_ijqxyz = discretization.apply_quadrature_weights_on_gradient_field(stress_field_ijqxyz)

    # stress difference
    stress_difference_ij = actual_stress_ij - target_stress_ij

    double_contraction_stress_qxyz = np.einsum('ij,ijqxy...->qxy...',
                                               stress_difference_ij,
                                               stress_ijqxyz)
    # np.einsum('ijqxyz  ,jiqxyz  ->xyz    ', A2, B2)
    # Average over quad points in pixel !!!
    # partial_derivative = partial_derivative.mean(axis=0)
    partial_derivative_xyz = double_contraction_stress_qxyz.sum(axis=0)

    return 2 * partial_derivative_xyz / discretization.cell.domain_volume


def partial_derivative_of_objective_function_wrt_phase_field_OLD(discretization,
                                                                 material_data_field_ijklqxyz,
                                                                 displacement_field_fnxyz,
                                                                 macro_gradient_field_ijqxyz,
                                                                 phase_field_1nxyz,
                                                                 target_stress_ij,
                                                                 actual_stress_ij,
                                                                 p):
    df_drho = partial_derivative_of_objective_function_stress_equivalence_wrt_phase_field(discretization,
                                                                                          material_data_field_ijklqxyz,
                                                                                          displacement_field_fnxyz,
                                                                                          macro_gradient_field_ijqxyz,
                                                                                          phase_field_1nxyz,
                                                                                          target_stress_ij,
                                                                                          actual_stress_ij,
                                                                                          p)

    dgradrho_drho = partial_derivative_of_gradient_of_phase_field_potential(discretization,
                                                                            phase_field_1nxyz,
                                                                            eta=1)

    ddouble_well_drho_drho = partial_der_of_double_well_potential_wrt_density(discretization,
                                                                              phase_field_1nxyz,
                                                                              eta=1)

    return df_drho + dgradrho_drho + ddouble_well_drho_drho


def partial_derivative_of_objective_function_wrt_phase_field(discretization,
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


def adjoint_potential(discretization, stress_field_ijqxyz, adjoint_field_fnxyz):
    # g = (grad lambda, stress)
    # g = (grad lambda, C grad displacement)
    # g = (grad lambda, C grad displacement)  == lambda_transpose grad_transpose C grad u

    # Input: adjoint_field [f,n,x,y,z]
    #        stress_field  [d,d,q,x,y,z]

    # Output: g  [1] == 0
    # -- -- -- -- -- -- -- -- -- -- --
    # apply quadrature weights
    stress_field_ijqxyz = discretization.apply_quadrature_weights_on_gradient_field(stress_field_ijqxyz)
    force_field_inxyz = discretization.apply_gradient_transposed_operator(stress_field_ijqxyz)
    adjoint_potential_field = np.einsum('i...,i...->...', adjoint_field_fnxyz, force_field_inxyz)

    return np.sum(adjoint_potential_field)


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


def partial_derivative_of_adjoint_potential_wrt_phase_field(discretization,
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


def sensitivity_OLD(discretization,
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

    dg_drho = partial_derivative_of_adjoint_potential_wrt_phase_field(
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


def sensitivity_with_adjoint_problem(discretization,
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
    phase_field_at_quad_poits_1qnxyz = discretization.evaluate_field_at_quad_points(nodal_field_fnxyz=phase_field_1nxyz,
                                                                                    quad_field_fqnxyz=None,
                                                                                    quad_points_coords_dq=None)[0]
    # dmaterial_data_field_drho_ijklqxyz = material_data_field_ijklqxyz[..., :, :] * (
    #         p * np.power(phase_field_1nxyz[0, 0], (p - 1)))

    dmaterial_data_field_drho_ijklqxyz = material_data_field_ijklqxyz[..., :, :, :] * (
            p * np.power(phase_field_at_quad_poits_1qnxyz[0, :, 0, ...], (p - 1)))
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
    df_drho = dfstress_drho + weight * (dgradrho_drho * eta + ddouble_well_drho_drho / eta)

    # --------------------------------------
    # Solve adjoint problem ∂f/∂u=-∂g/∂u
    # Dt C D lambda = - 2/|omega| Dt: C : sigma_diff
    # material_data_field_C_0_rho_ijklqxyz = material_data_field_ijklqxyz[..., :, :] * np.power(phase_field_1nxyz,
    #                                                                                          p)
    # TODO delete if phase field at quad points wokrs
    material_data_field_C_0_rho_ijklqxyz = material_data_field_ijklqxyz[..., :, :, :] * np.power(
        phase_field_at_quad_poits_1qnxyz, p)[0, :, 0, ...]

    # stress difference potential
    # rhs=-Dt*wA*E  -- we can use it to assemble df_du_field

    stress_difference_ijqxyz = discretization.get_gradient_size_field()
    stress_difference_ijqxyz[:, :, ...] = stress_difference_ij[
        (...,) + (np.newaxis,) * (stress_difference_ijqxyz.ndim - 2)]

    df_du_field = 2 * discretization.get_rhs(material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                                             macro_gradient_field_ijqxyz=stress_difference_ijqxyz) / discretization.cell.domain_volume  # minus sign is already there
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

    dg_drho = double_contraction_stress_qxyz.sum(axis=0)

    return df_drho + dg_drho
