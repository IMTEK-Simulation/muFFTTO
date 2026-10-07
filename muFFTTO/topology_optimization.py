"""
Objective functions and sensitivities for phase-field topology optimization.

This module collects the building blocks of the phase-field based topology
optimization of periodic microstructures (unit cells) that is driven by
FFT-accelerated FEM homogenization (see :mod:`muFFTTO.domain` and
:mod:`muFFTTO.solvers`).

The design variable is a nodal scalar phase field ``rho`` (field layout
``[1, n, x, y(, z)]``, ``n`` = nodal points per pixel) with values ideally in
``[0, 1]`` (0 = void, 1 = solid). The (weighted) objective has the generic form

.. math::

    f(\\rho) = w\\, f_\\sigma(u(\\rho), \\rho)
               + \\eta \\int_\\Omega |\\nabla\\rho|^2\\,dx
               + \\frac{c_{dw}}{\\eta} \\int_\\Omega \\rho^2 (1-\\rho)^2\\,dx ,

where

* ``f_sigma`` is a mechanical term, e.g. the normalized squared difference
  between a target and the homogenized stress
  ``|Sigma_t - Sigma_h|^2 / |Sigma_t|^2`` (stress equivalence) or the squared
  difference of an energy-like contraction ``(E_L : (Sigma_t - Sigma_h))^2 /
  W_t^2`` (energy equivalence),
* the gradient term ``eta * int |grad rho|^2`` penalises interfaces
  (Modica-Mortola / Cahn-Hilliard type regularisation),
* the double-well term ``int rho^2 (1 - rho)^2 / eta`` drives ``rho`` towards
  0 or 1; ``eta`` controls the width of the diffuse interface and
  ``c_dw`` (``double_well_depth``) scales the well depth.

The material is interpolated with SIMP,
``C(rho) = (C_1 - C_0) rho^p + C_0`` (``C_1`` base/solid, ``C_0`` void),
evaluated at the quadrature points.

Sensitivities ``df/drho`` are obtained with the adjoint method. With the
discrete equilibrium (FE, small strain)

.. math::

    g(u, \\rho) = D^T W\\, C(\\rho) : (E + D u) = 0 ,

(``D`` = (symmetrised) gradient operator nodal -> quadrature points,
``W`` = quadrature weights, ``E`` = macroscopic strain), the adjoint field
``lambda`` solves ``K(rho) lambda = -df/du`` and the total derivative reads

.. math::

    \\frac{df}{d\\rho} = \\frac{\\partial f}{\\partial\\rho}
                       + \\lambda^T \\frac{\\partial g}{\\partial\\rho},
    \\qquad
    \\lambda^T \\frac{\\partial g}{\\partial\\rho}
       = N^T W \\big[ (D\\lambda) : \\tfrac{\\partial C}{\\partial\\rho} : (E + D u) \\big],

where ``N`` interpolates nodal values to quadrature points.

Index / layout conventions used in variable names
--------------------------------------------------
``i, j, k, l``  tensor indices (0..d-1), ``d`` spatial dimension (2 or 3),
``f`` field components, ``n`` nodal points per pixel, ``q`` quadrature points
per pixel, ``x, y, z`` pixel indices of the (MPI-local) grid. E.g.
``material_data_field_ijklqxyz`` has shape ``[d, d, d, d, q, nx, ny(, nz)]``,
``displacement_field_inxyz`` has shape ``[d, n, nx, ny(, nz)]`` and a phase
field ``phase_field_1nxyz`` has shape ``[1, n, nx, ny(, nz)]``.

Fields are mostly muGrid fields: ``.s`` is the view without ghost layers and
``.sg`` the view including ghost layers (needed for stencil/roll operations
in MPI-parallel runs). Global reductions are performed with
``discretization.mpi_reduction`` (NuMPI) or ``discretization.communicator``.

Functions suffixed ``_mugrid``/``_FE_NEW`` are the current, muGrid-based
implementations. Several older functions (``*_pixel``, ``*_FE_weights``,
``*_FE_testing``, ``*_NEW`` double-well derivative, Gauss-quadrature double
well) rely on legacy NumPy-array APIs of the discretization object that no
longer exist / changed signature and are kept for reference only.
"""
import warnings
import gc

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
    """
    Evaluate the full small-strain objective (stress equivalence + phase-field terms).

    Computes

    ``f = w * f_sigma + eta * f_rho_grad + double_well_depth * f_dw / eta``

    with ``f_sigma = |Sigma_t - Sigma_h|^2 / |Sigma_t|^2`` (see
    :func:`compute_stress_equivalence_potential`),
    ``f_rho_grad = int |grad rho|^2 dx`` and
    ``f_dw = int rho^2 (1 - rho)^2 dx`` (exact integration for linear
    triangles, :func:`compute_double_well_potential_analytical`).

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        Discretization of the periodic unit cell.
    actual_stress_ij : ndarray, shape [d, d]
        Homogenized (volume-averaged) stress ``Sigma_h`` of the current design.
    target_stress_ij : ndarray, shape [d, d]
        Target homogenized stress ``Sigma_t``.
    phase_field_1nxyz : muGrid Field, shape [1, n, x, y(, z)]
        Nodal phase field ``rho``. Its ghost layers are updated (MPI
        communication) as a side effect.
    eta : float
        Phase-field interface-width parameter.
    w : float
        Weight of the stress-equivalence term.
    double_well_depth : float, optional
        Scaling ``c_dw`` of the double-well term (default 1).
    disp : bool, optional
        If True, rank 0 prints the individual contributions.

    Returns
    -------
    float
        Global (MPI-reduced) objective value.

    Notes
    -----
    ``f_sigma`` is always printed on rank 0, independent of ``disp``.
    Only valid for ``linear_triangles``/``linear_triangles_tilled`` elements,
    because of the analytical double-well integration.
    """
    # evaluate objective functions
    # f = (flux_h -flux_target)^2 + w*eta* int (  (grad(rho))^2 )dx  +    int ( rho^2(1-rho)^2 ) / eta   dx
    # f =  f_sigma + w*eta* f_rho_grad  + f_dw/eta
    # NOTE: the code below actually evaluates
    #       f = w * f_sigma + eta * f_rho_grad + double_well_depth * f_dw / eta
    #       i.e. the weight w multiplies the stress term, not the phase-field terms.

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

    return w * f_sigma + f_rho  # / discretization.cell.domain_volume


def objective_function_phase_field(discretization,
                                   phase_field_1nxyz,
                                   eta,
                                   double_well_depth=1,
                                   disp=False,
                                   split_results=False):
    """
    Evaluate the phase-field (regularisation) part of the objective.

    Computes

    ``f_rho = eta * int |grad rho|^2 dx + double_well_depth / eta * int rho^2 (1 - rho)^2 dx``.

    The double-well integral is evaluated with nodal quadrature
    (:func:`compute_double_well_potential_nodal`); the analytical variant is
    available behind the hard-coded switch ``anal_double_well``. The matching
    sensitivity is :func:`sensitivity_phase_field_term_FE_NEW`.

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        Discretization of the periodic unit cell.
    phase_field_1nxyz : muGrid Field, shape [1, n, x, y(, z)]
        Nodal phase field ``rho``. Ghost layers are updated (MPI) as a side effect.
    eta : float
        Phase-field interface-width parameter.
    double_well_depth : float, optional
        Scaling of the double-well term (default 1).
    disp : bool, optional
        If True, rank 0 prints the individual contributions.
    split_results : bool, optional
        If True, also return the individual (unscaled) contributions.

    Returns
    -------
    f_rho : float
        Global value of the phase-field objective.
    f_rho_grad : float
        Only if ``split_results``: ``int |grad rho|^2 dx`` (not multiplied by ``eta``).
    f_dw : float
        Only if ``split_results``: ``int rho^2 (1 - rho)^2 dx`` (not scaled).
    """
    # evaluate objective functions
    # f =  w*eta* int (  (grad(rho))^2 )dx  +    int ( rho^2(1-rho)^2 ) / eta   dx
    # f =  eta* f_rho_grad  + f_dw/eta

    # double - well potential
    # use anlytical expresion ?
    # False -> nodal (lumped) quadrature, consistent with the derivative used in
    #          sensitivity_phase_field_term_FE_NEW (partial_der_..._nodal)
    # True  -> exact integration of the P1 interpolant (linear triangles only)
    anal_double_well = False
    if anal_double_well:
        f_dw = compute_double_well_potential_analytical(discretization=discretization,
                                                        phase_field_1nxyz=phase_field_1nxyz)
    else:
        f_dw = compute_double_well_potential_nodal(discretization=discretization,
                                                   phase_field_1nxyz=phase_field_1nxyz)
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

    if split_results:
        return f_rho, f_rho_grad, f_dw

    return f_rho


def objective_function_phase_field_3D(discretization,
                                      phase_field_1nxyz,
                                      eta,
                                      double_well_depth=1,
                                      disp=False,
                                      split_results=False):
    """
    Evaluate the phase-field (regularisation) part of the objective (3D variant).

    Same as :func:`objective_function_phase_field` but always uses the nodal
    quadrature for the double-well term (the analytical variant only exists
    for 2D linear triangles):

    ``f_rho = eta * int |grad rho|^2 dx + double_well_depth / eta * int rho^2 (1 - rho)^2 dx``.

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        Discretization of the periodic unit cell.
    phase_field_1nxyz : muGrid Field, shape [1, n, x, y, z]
        Nodal phase field ``rho``. Ghost layers are updated (MPI) as a side effect.
    eta : float
        Phase-field interface-width parameter.
    double_well_depth : float, optional
        Scaling of the double-well term (default 1).
    disp : bool, optional
        If True, rank 0 prints the individual contributions.
    split_results : bool, optional
        If True, also return ``f_rho_grad`` and ``f_dw`` (unscaled).

    Returns
    -------
    f_rho : float or tuple
        ``f_rho`` or ``(f_rho, f_rho_grad, f_dw)`` if ``split_results``.
    """
    # evaluate objective functions
    # f =  w*eta* int (  (grad(rho))^2 )dx  +    int ( rho^2(1-rho)^2 ) / eta   dx
    # f =  eta* f_rho_grad  + f_dw/eta

    # double - well potential
    f_dw = compute_double_well_potential_nodal(discretization=discretization,
                                               phase_field_1nxyz=phase_field_1nxyz)

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

    if split_results:
        return f_rho, f_rho_grad, f_dw

    return f_rho


def compute_stress_equivalence_potential(actual_stress_ij,
                                         target_stress_ij,
                                         disp=False):
    """
    Normalized squared stress mismatch (stress-equivalence potential).

    ``f_sigma = (Sigma_t - Sigma_h) : (Sigma_t - Sigma_h) / (Sigma_t : Sigma_t)``

    Parameters
    ----------
    actual_stress_ij : ndarray, shape [d, d]
        Homogenized stress ``Sigma_h`` of the current design.
    target_stress_ij : ndarray, shape [d, d]
        Target homogenized stress ``Sigma_t`` (must not be zero).
    disp : bool, optional
        If True, rank 0 prints the value.

    Returns
    -------
    float
        Dimensionless stress mismatch. No MPI reduction is needed because the
        homogenized stresses are already global quantities.
    """
    # evaluate objective functions
    # f_sigma = ( stress_target-stress_h)^2

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
    """
    Normalized squared energy mismatch (energy-equivalence potential).

    ``f_sigma = (E_L : (Sigma_t - Sigma_h))^2 / W_t^2``

    where ``E_L`` is a "left" macroscopic strain used to contract the stress
    difference into a scalar (energy-like) quantity.

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        Unused, kept for a uniform interface.
    actual_stress_ij : ndarray, shape [d, d]
        Homogenized stress ``Sigma_h``.
    target_stress_ij : ndarray, shape [d, d]
        Target homogenized stress ``Sigma_t``.
    left_macro_gradient_ij : ndarray, shape [d, d]
        Macroscopic strain ``E_L`` used in the contraction.
    target_energy : float
        Normalisation ``W_t`` (typically ``E_L : Sigma_t``).
    disp : bool, optional
        If True (default), rank 0 prints intermediate values.

    Returns
    -------
    float
        Dimensionless energy mismatch.
    """
    # evaluate objective functions
    # stress_difference_ij = actual_stress_ij - target_stress_ij
    stress_difference_ij = target_stress_ij - actual_stress_ij
    # double contraction E_L : (Sigma_t - Sigma_h) -> scalar
    actual_energy_difference = np.einsum('ij,ij->...',
                                         left_macro_gradient_ij,
                                         stress_difference_ij
                                         )
    if disp and MPI.COMM_WORLD.rank == 0:
        print('actual_energy_difference = '          ' {} '.format(actual_energy_difference))
    f_sigma = (actual_energy_difference ** 2) / (target_energy ** 2)
    if disp and MPI.COMM_WORLD.rank == 0:
        print('f_sigma = '          ' {} '.format(f_sigma))  # good in MPI
    return f_sigma


def objective_function_small_strain_pixel(discretization,
                                          actual_stress_ij,
                                          target_stress_ij,
                                          phase_field_1nxyz,
                                          eta,
                                          w):
    """
    Legacy pixel-based small-strain objective (stress equivalence + phase field).

    Computes ``f = f_sigma + w * (eta * f_rho_grad + f_dw / eta)``. Note that,
    unlike :func:`objective_function_small_strain`, the weight ``w`` multiplies
    the phase-field terms here.

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        Discretization of the periodic unit cell.
    actual_stress_ij : ndarray, shape [d, d]
        Homogenized stress ``Sigma_h``.
    target_stress_ij : ndarray, shape [d, d]
        Target homogenized stress ``Sigma_t``.
    phase_field_1nxyz : muGrid Field or ndarray, shape [1, n, x, y(, z)]
        Nodal phase field.
    eta : float
        Phase-field interface-width parameter.
    w : float
        Weight of the phase-field terms.

    Returns
    -------
    float
        Objective value.

    Notes
    -----
    Legacy: relies on ``discretization.apply_gradient_operator`` (NumPy API),
    which is no longer provided by :mod:`muFFTTO.domain`.
    """
    # evaluate objective functions
    # f = (flux_h -flux_target)^2 + w*eta* int (  (grad(rho))^2 )dx  +    int ( rho^2(1-rho)^2 ) / eta   dx
    # f =  f_sigma + w*eta* f_rho_grad  + f_dw/eta

    # stress difference potential: actual_stress_ij is homogenized stress
    # stress_difference_ij = actual_stress_ij - target_stress_ij
    stress_difference_ij = (actual_stress_ij - target_stress_ij)

    f_sigma = np.sum(stress_difference_ij ** 2) / np.sum(target_stress_ij ** 2)

    # double - well potential (nodal quadrature, not divided by eta here: eta=1)
    f_dw = compute_double_well_potential_nodal(discretization=discretization,
                                               phase_field_1nxyz=phase_field_1nxyz,
                                               eta=1)

    phase_field_gradient = discretization.apply_gradient_operator(phase_field_1nxyz)
    f_rho_grad = np.sum(discretization.integrate_over_cell(phase_field_gradient ** 2))

    f_rho = eta * f_rho_grad + f_dw / eta

    return f_sigma + w * f_rho  # / discretization.cell.domain_volume


def compute_double_well_potential_Gauss_quad(discretization, phase_field_1nxyz):
    """
    Double-well integral ``int rho^2 (1 - rho)^2 dx`` by high-order Gauss quadrature.

    The P1 interpolant of the nodal phase field is evaluated at 18 Gauss
    points per pixel (two triangles) and the (quartic) integrand is integrated
    numerically.

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        Discretization; must use ``linear_triangles`` or ``linear_triangles_tilled``.
    phase_field_1nxyz : muGrid Field, shape [1, n, x, y]
        Nodal phase field.

    Returns
    -------
    float
        Global (MPI-reduced) value of the integral (not divided by ``eta``).

    Raises
    ------
    ValueError
        If the element type is not a linear-triangle type.

    Notes
    -----
    Legacy: expects ``discretization.evaluate_field_at_quad_points`` to accept
    arbitrary quadrature-point coordinates and to return a tuple
    ``(quad_field, N_at_quad_points)``; the current implementation in
    :mod:`muFFTTO.domain` returns a single field.
    """
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

    # Map reference-pixel weights to the physical pixel: for an axis-aligned
    # rectangular pixel the Jacobian is diag(h_x, h_y) and det J = pixel area.
    Jacobian_matrix = np.diag(discretization.pixel_size)
    Jacobian_det = np.linalg.det(
        Jacobian_matrix)  # this is product of diagonal term of Jacoby transformation matrix
    quad_points_weights = quad_points_weights * Jacobian_det
    # Evaluate field on the quadrature points
    quad_field_fqnxyz, N_at_quad_points_qnijk = discretization.evaluate_field_at_quad_points(
        nodal_field_fnxyz=phase_field_1nxyz,
        quad_field_fqnxyz=None,
        quad_points_coords_iq=quad_points_coord)

    # pointwise double-well integrand rho^2 (1 - rho)^2 at quadrature points
    quad_field_fqnxyz = (quad_field_fqnxyz ** 2) * (1 - quad_field_fqnxyz) ** 2
    # Multiply with quadrature weights
    quad_field_fqnxyz = np.einsum('fq...,q->fq...', quad_field_fqnxyz, quad_points_weights)

    return discretization.mpi_reduction.sum(quad_field_fqnxyz)


def compute_double_well_potential_analytical(discretization, phase_field_1nxyz):
    """
    Exact double-well integral ``int rho^2 (1 - rho)^2 dx`` for P1 triangles (2D).

    The phase field is the piecewise-linear interpolant of its nodal values on
    a mesh where every pixel is split into two triangles,
    ``T1 = (rho_00, rho_10, rho_01)`` and ``T2 = (rho_11, rho_10, rho_01)``.
    Using ``rho^2 (1 - rho)^2 = rho^2 - 2 rho^3 + rho^4`` each monomial is
    integrated exactly with the barycentric-coordinate formula

    ``int_T L1^a L2^b L3^c dA = 2 |T| a! b! c! / (a + b + c + 2)!``,

    with ``|T| = h_x h_y / 2``. The coefficients in the lambdas below are
    these exact integrals already multiplied by the multinomial coefficients
    and divided by the pixel area ``h_x h_y`` (= ``Jacobian_det``), e.g. for
    ``rho^2``: ``h_x h_y * (1/12 sum_i rho_i^2 + 1/12 sum_{i<j} rho_i rho_j)``.

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        2D discretization with ``linear_triangles`` or ``linear_triangles_tilled``.
    phase_field_1nxyz : muGrid Field, shape [1, 1, x, y]
        Nodal phase field (one node per pixel). Its ghost layers are updated
        (MPI communication) as a side effect.

    Returns
    -------
    float
        Global (MPI-reduced) value of the integral (not divided by ``eta``).

    Raises
    ------
    ValueError
        If the element type is not a linear-triangle type.

    Notes
    -----
    Uses the temporary scalar fields ``rho_00, rho_10, rho_01, rho_11`` of the
    discretization's field collection.
    """
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
    # Fill ghost layers so that the periodic shifts (np.roll) below see the
    # correct neighbour values across MPI subdomain boundaries.
    discretization.fft.communicate_ghosts(phase_field_1nxyz)

    # Nodal values of the four pixel corners, stored at the index of the
    # pixel's lower-left node (x, y):
    #   rho_00 = rho(x, y), rho_10 = rho(x+1, y), rho_01 = rho(x, y+1),
    #   rho_11 = rho(x+1, y+1)
    # (np.roll by -1 brings the value of the right/upper neighbour to (x, y)).
    # Rolling is performed on the ghost-padded array (.sg); only the interior
    # (.s) values are used afterwards, so the wrap-around of the ghost buffer
    # does not matter.
    rho_00 = discretization.get_scalar_field(name='rho_00')
    rho_10 = discretization.get_scalar_field(name='rho_10')
    rho_01 = discretization.get_scalar_field(name='rho_01')
    rho_11 = discretization.get_scalar_field(name='rho_11')

    rho_00.sg[0, 0] = phase_field_1nxyz.sg[0, 0]  # [..., 1:-1]
    rho_10.sg[0, 0] = np.roll(phase_field_1nxyz.sg[0, 0], np.array([-1, 0]), axis=(0, 1))  # [..., 1:-1]
    # rho_m10 = np.roll(phase_field_1nxyz.sg[0, 0], np.array([1, 0]), axis=(0, 1))
    rho_01.sg[0, 0] = np.roll(phase_field_1nxyz.sg[0, 0], np.array([0, -1]), axis=(0, 1))  # [..., 1:-1]
    # rho_0m1 = np.roll(phase_field_1nxyz.sg[0, 0], np.array([0, 1]), axis=(0, 1))
    # rho_m11 = np.roll(phase_field_1nxyz.sg[0, 0], np.array([1, -1]), axis=(0, 1))
    # rho_1m1 = np.roll(phase_field_1nxyz.sg[0, 0], np.array([-1, 1]), axis=(0, 1))
    rho_11.sg[0, 0] = np.roll(phase_field_1nxyz.sg[0, 0], np.array([-1, -1]), axis=(0, 1))  # [..., 1:-1]

    # int_pixel rho^2 / (h_x h_y): first two terms -> triangle (rho0, rho1, rho2),
    # last two terms -> triangle (rho3, rho1, rho2)
    rho_squared_pixel = lambda rho0, rho1, rho2, rho3: (1 / 12) * (rho0 ** 2 + rho1 ** 2 + rho2 ** 2) \
                                                       + (2 / 24) * (rho0 * rho1 + rho0 * rho2 + rho2 * rho1) \
                                                       + (1 / 12) * (rho3 ** 2 + rho1 ** 2 + rho2 ** 2) \
                                                       + (2 / 24) * (rho3 * rho1 + rho3 * rho2 + rho2 * rho1)

    rho_squared = discretization.mpi_reduction.sum(rho_squared_pixel(rho_00.s[0, 0],
                                                                     rho_10.s[0, 0],
                                                                     rho_01.s[0, 0],
                                                                     rho_11.s[0, 0])) * Jacobian_det

    # int_pixel rho^3 / (h_x h_y), same split into the two triangles
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

    rho_qubed = discretization.mpi_reduction.sum(rho_qubed_pixel(rho_00.s[0, 0],
                                                                 rho_10.s[0, 0],
                                                                 rho_01.s[0, 0],
                                                                 rho_11.s[0, 0])) * Jacobian_det

    # int_pixel rho^4 / (h_x h_y), same split into the two triangles
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

    rho_quartic = discretization.mpi_reduction.sum(rho_quartic_pixel(rho_00.s[0, 0],
                                                                     rho_10.s[0, 0],
                                                                     rho_01.s[0, 0],
                                                                     rho_11.s[0, 0])) * Jacobian_det

    # (ρ^2 (1 - ρ)^2) = ρ^2 - 2ρ^3 + ρ^4
    integral = rho_squared - 2 * rho_qubed + rho_quartic

    del rho_00
    del rho_10
    del rho_01
    del rho_11
    return integral


def get_nb_of_entries_global(discretization, field_s):
    """
    Global number of entries of a (distributed) field array.

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        Provides ``domain_dimension`` and ``nb_of_pixels_global``.
    field_s : ndarray, shape [..., x, y(, z)]
        MPI-local array (e.g. ``field.s``); the leading (component/node)
        dimensions are taken from it, the spatial ones from the global grid.

    Returns
    -------
    int
        ``prod(field_s.shape[:-dim]) * prod(nb_of_pixels_global)``.
    """
    # Number of entries that a field  [f,n,x,y,z] has over the whole unit cell.
    # field_s.shape holds only the pixels of the current MPI rank, so the number of
    # pixels has to be taken from the global grid.
    dim = discretization.domain_dimension
    return int(np.prod(field_s.shape[:-dim]) * np.prod(discretization.nb_of_pixels_global))


def compute_double_well_potential_nodal(discretization,
                                        phase_field_1nxyz,
                                        eta=1):
    """
    Double-well integral ``int rho^2 (1 - rho)^2 dx / eta`` by nodal quadrature.

    The integrand is evaluated at the nodes and integrated with equal weights
    ``|Omega| / N_global`` (lumped / trapezoidal-type rule on a regular grid):

    ``f_dw = |Omega| / N * sum_nodes rho_n^2 (1 - rho_n)^2 / eta``.

    Works in 2D and 3D and for any element type.

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        Discretization of the periodic unit cell.
    phase_field_1nxyz : muGrid Field, shape [1, n, x, y(, z)]
        Nodal phase field.
    eta : float, optional
        Divisor (default 1, i.e. the plain integral is returned).

    Returns
    -------
    float
        Global (MPI-reduced) value.
    """
    # The double-well potential
    # phase field potential = int ( rho^2(1-rho)^2 ) / eta   dx
    # double - well potential
    integrant = (phase_field_1nxyz.s ** 2) * (1 - phase_field_1nxyz.s) ** 2
    integral = discretization.mpi_reduction.sum(integrant)
    integral = (integral / get_nb_of_entries_global(discretization, integrant)) * discretization.cell.domain_volume
    return integral / eta


def partial_der_of_double_well_potential_wrt_density_NEW(discretization, phase_field_1nxyz, eta=1):
    """
    Legacy: nodal derivative of the double-well integral via Gauss quadrature.

    Computes ``d/d rho_a int rho^2 (1 - rho)^2 dx / eta
    = int (2 rho - 6 rho^2 + 4 rho^3) N_a dx / eta`` for every node ``a`` by
    evaluating the integrand at 8 Gauss points per pixel and applying the
    transposed interpolation (sum of ``N_a(x_q) w_q`` contributions of all
    pixels adjacent to the node).

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        Discretization with ``linear_triangles``/``linear_triangles_tilled``.
    phase_field_1nxyz : muGrid Field, shape [1, n, x, y(, z)]
        Nodal phase field.
    eta : float, optional
        Divisor (default 1).

    Returns
    -------
    ndarray, shape [1, n, x, y(, z)]
        Nodal gradient of the double-well term.

    Raises
    ------
    ValueError
        If the element type is not a linear-triangle type.

    Notes
    -----
    Legacy: relies on the tuple-returning
    ``evaluate_field_at_quad_points(..., quad_points_coords_iq)`` API that no
    longer exists in :mod:`muFFTTO.domain`. The 3D branch is untested.
    """
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
    quad_field_fqnxyz = np.einsum('fq...,q->fq...', quad_field_fqnxyz, quad_points_weights)
    nodal_field_u_fnxyz = np.zeros(phase_field_1nxyz.s.shape)
    # Transposed interpolation N^T: each pixel corner (pixel_node in {0,1}^d)
    # receives sum_q N_corner(x_q) * w_q * integrand(x_q); the contribution is
    # then shifted (rolled) by the corner offset so that it lands on the
    # global node it belongs to (periodic assembly).
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

    return nodal_field_u_fnxyz / eta


def partial_der_of_double_well_potential_wrt_density_nodal(discretization,
                                                           phase_field_1nxyz,
                                                           output_1nxyz,
                                                           eta=1):
    """
    Nodal derivative of the nodal-quadrature double-well term.

    Exact derivative of :func:`compute_double_well_potential_nodal` w.r.t.
    each nodal value ``rho_a``:

    ``d f_dw / d rho_a = |Omega| / N * 2 rho_a (2 rho_a^2 - 3 rho_a + 1) / eta``
    ``= |Omega| / N * (2 rho_a - 6 rho_a^2 + 4 rho_a^3) / eta``.

    Purely local (no neighbour coupling, no MPI communication).

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        Discretization of the periodic unit cell.
    phase_field_1nxyz : muGrid Field, shape [1, n, x, y(, z)]
        Nodal phase field.
    output_1nxyz : muGrid Field, shape [1, n, x, y(, z)]
        Output field, overwritten in place (component ``[0, 0]``).
    eta : float, optional
        Divisor (default 1).

    Returns
    -------
    muGrid Field
        ``output_1nxyz`` (same object).
    """
    # Derivative of the double-well potential with respect to phase-field
    # phase field potential = int ( rho^2(1-rho)^2 )/eta   dx
    # gradient phase field potential = int ((2 * phase_field( + 2 * phase_field^2  -  3 * phase_field +1 )) )/eta   dx
    # d/dρ(ρ^2 (1 - ρ)^2) = 2 ρ (2 ρ^2 - 3 ρ + 1)

    integrant_1nxyz = (
            2 * phase_field_1nxyz.s * (2 * phase_field_1nxyz.s * phase_field_1nxyz.s - 3 * phase_field_1nxyz.s + 1))
    # integral=discretization.mpi_reduction.sum(integrant_1nxyz)

    integral_fnxyz = (integrant_1nxyz / get_nb_of_entries_global(discretization,
                                                                 integrant_1nxyz)) * discretization.cell.domain_volume
    # there is no sum here
    output_1nxyz.s[0, 0] = integral_fnxyz / eta
    return output_1nxyz


def partial_der_of_double_well_potential_wrt_density_analytical(discretization,
                                                                phase_field_1nxyz,
                                                                output_1nxyz):
    """
    Exact nodal derivative of the analytical double-well integral (P1 triangles, 2D).

    Derivative of :func:`compute_double_well_potential_analytical` w.r.t.
    each nodal value ``rho_0``:

    ``d/d rho_0 int rho^2 (1 - rho)^2 dx
    = d/d rho_0 [int rho^2 - 2 int rho^3 + int rho^4]``.

    In the triangulation used (each pixel split along the (1,0)-(0,1)
    diagonal) a node is shared by 6 triangles and coupled to the 6
    neighbours ``(+-1, 0), (0, +-1), (-1, +1), (+1, -1)``. Each of
    ``drho_squared``, ``drho_cubed`` and ``drho_quartic`` is the exact
    derivative of the corresponding monomial integral, written in terms of
    the node value ``rho_00`` and its 6 neighbours, e.g.

    ``d/d rho_0 int rho^2 = 2 int rho N_0 = h_x h_y (rho_0 + 1/6 sum_nb rho_nb)``.

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        2D discretization with ``linear_triangles``/``linear_triangles_tilled``.
    phase_field_1nxyz : muGrid Field, shape [1, 1, x, y]
        Nodal phase field. Ghost layers are updated (MPI) as a side effect.
    output_1nxyz : muGrid Field, shape [1, 1, x, y]
        Output field, overwritten in place.

    Returns
    -------
    muGrid Field
        ``output_1nxyz`` (same object). Not divided by ``eta``.

    Raises
    ------
    ValueError
        If the element type is not a linear-triangle type.
    """
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

    # communicate ghost before rolling
    discretization.fft.communicate_ghosts(phase_field_1nxyz)

    rho_00 = discretization.get_scalar_field(name='rho_00')
    rho_10 = discretization.get_scalar_field(name='rho_10')
    rho_m10 = discretization.get_scalar_field(name='rho_m10')
    rho_01 = discretization.get_scalar_field(name='rho_01')
    rho_0m1 = discretization.get_scalar_field(name='rho_0m1')
    rho_m11 = discretization.get_scalar_field(name='rho_m11')
    rho_1m1 = discretization.get_scalar_field(name='rho_1m1')

    # NODAL VALUES OF CONNECTED POINTS
    # rho_ab holds, at index (x, y), the value rho(x + a, y + b) ('m' = minus 1),
    # e.g. rho_m11 = rho(x - 1, y + 1). np.roll by -s brings the value at +s to the origin.
    rho_00.sg[0, 0] = phase_field_1nxyz.sg[0, 0]  # [..., 1:-1]
    rho_10.sg[0, 0] = np.roll(phase_field_1nxyz.sg[0, 0], np.array([-1, 0]), axis=(0, 1))  # [..., 1:-1]
    rho_m10.sg[0, 0] = np.roll(phase_field_1nxyz.sg[0, 0], np.array([1, 0]), axis=(0, 1))  # [..., 1:-1]
    rho_01.sg[0, 0] = np.roll(phase_field_1nxyz.sg[0, 0], np.array([0, -1]), axis=(0, 1))  # [..., 1:-1]
    rho_0m1.sg[0, 0] = np.roll(phase_field_1nxyz.sg[0, 0], np.array([0, 1]), axis=(0, 1))  # [..., 1:-1]
    rho_m11.sg[0, 0] = np.roll(phase_field_1nxyz.sg[0, 0], np.array([1, -1]), axis=(0, 1))  # [..., 1:-1]
    rho_1m1.sg[0, 0] = np.roll(phase_field_1nxyz.sg[0, 0], np.array([-1, 1]), axis=(0, 1))  # [..., 1:-1]

    # d/d rho_0 int rho^2 dx  (Jacobian_det = pixel area h_x h_y)
    drho_squared = (rho_00.s[0, 0] + 1 / 6 * (rho_10.s[0, 0] + rho_m10.s[0, 0] + rho_01.s[0, 0]
                                              + rho_0m1.s[0, 0] + rho_m11.s[0, 0] + rho_1m1.s[0, 0])) * Jacobian_det

    # d/d rho_0 int rho^3 dx; the (1/20) products run over neighbour pairs that
    # share a triangle with node 0 (consecutive neighbours around the node)
    drho_cubed = ((9 / 10) * rho_00.s[0, 0] ** 2 \
                  + (1 / 10) * (
                          rho_10.s[0, 0] ** 2 + rho_m10.s[0, 0] ** 2 + rho_01.s[0, 0] ** 2 + rho_0m1.s[0, 0] ** 2 +
                          rho_m11.s[0, 0] ** 2 + rho_1m1.s[0, 0] ** 2) \
                  + (2 / 10) * rho_00.s[0, 0] * (
                          rho_10.s[0, 0] + rho_m10.s[0, 0] + rho_01.s[0, 0] + rho_0m1.s[0, 0] + rho_m11.s[0, 0] +
                          rho_1m1.s[0, 0]) \
                  + (1 / 20) * (rho_10.s[0, 0] * rho_01.s[0, 0] + rho_01.s[0, 0] * rho_m11.s[0, 0] + rho_m11.s[0, 0] *
                                rho_m10.s[0, 0] \
                                + rho_m10.s[0, 0] * rho_0m1.s[0, 0] + rho_0m1.s[0, 0] * rho_1m1.s[0, 0] + rho_1m1.s[
                                    0, 0] * rho_10.s[0, 0]) \
                  ) * Jacobian_det

    # d/d rho_0 int rho^4 dx
    drho_quartic = ((24 / 30) * rho_00.s[0, 0] ** 3 \
                    + (2 / 30) * (
                            rho_10.s[0, 0] ** 3 + rho_m10.s[0, 0] ** 3 + rho_01.s[0, 0] ** 3 + rho_0m1.s[0, 0] ** 3 +
                            rho_m11.s[0, 0] ** 3 + rho_1m1.s[0, 0] ** 3) \
                    + (6 / 30) * rho_00.s[0, 0] ** 2 * (
                            rho_10.s[0, 0] + rho_m10.s[0, 0] + rho_01.s[0, 0] + rho_0m1.s[0, 0] + rho_m11.s[0, 0] +
                            rho_1m1.s[0, 0]) \
                    + (4 / 30) * rho_00.s[0, 0] * (
                            rho_10.s[0, 0] ** 2 + rho_m10.s[0, 0] ** 2 + rho_01.s[0, 0] ** 2 + rho_0m1.s[0, 0] ** 2 +
                            rho_m11.s[0, 0] ** 2 + rho_1m1.s[0, 0] ** 2) \
                    + (1 / 30) * (rho_10.s[0, 0] ** 2 * rho_01.s[0, 0] + rho_01.s[0, 0] ** 2 * rho_m11.s[0, 0] +
                                  rho_m11.s[0, 0] ** 2 * rho_m10.s[0, 0] \
                                  + rho_m10.s[0, 0] ** 2 * rho_0m1.s[0, 0] + rho_0m1.s[0, 0] ** 2 * rho_1m1.s[0, 0] +
                                  rho_1m1.s[0, 0] ** 2 * rho_10.s[0, 0]) \
                    + (1 / 30) * (rho_10.s[0, 0] * rho_01.s[0, 0] ** 2 + rho_01.s[0, 0] * rho_m11.s[0, 0] ** 2 +
                                  rho_m11.s[0, 0] * rho_m10.s[0, 0] ** 2 \
                                  + rho_m10.s[0, 0] * rho_0m1.s[0, 0] ** 2 + rho_0m1.s[0, 0] * rho_1m1.s[0, 0] ** 2 +
                                  rho_1m1.s[0, 0] * rho_10.s[0, 0] ** 2) \
                    + (2 / 30) * rho_00.s[0, 0] * (
                            rho_10.s[0, 0] * rho_01.s[0, 0] + rho_01.s[0, 0] * rho_m11.s[0, 0] + rho_m11.s[0, 0] *
                            rho_m10.s[0, 0] \
                            + rho_m10.s[0, 0] * rho_0m1.s[0, 0] + rho_0m1.s[0, 0] * rho_1m1.s[0, 0] + rho_1m1.s[
                                0, 0] * rho_10.s[0, 0])
                    ) * Jacobian_det
    # d/d rho_0 int (rho^2 - 2 rho^3 + rho^4) dx
    output_1nxyz.s[0, 0] = (drho_squared - 2 * drho_cubed + drho_quartic)

    del rho_00
    del rho_10
    del rho_m10
    del rho_01
    del rho_0m1
    del rho_m11
    del rho_1m1

    return output_1nxyz


def compute_gradient_of_phase_field_potential(discretization, phase_field_1nxyz):
    """
    Gradient (interface) energy ``int_Omega |grad rho|^2 dx``.

    The gradient of the nodal phase field is evaluated at the quadrature
    points with the muGrid gradient operator ``D`` and the squared components
    are integrated with the quadrature weights:

    ``f_rho_grad = sum_q sum_pixels w_q |D rho|_q^2 = rho^T D^T W D rho``.

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        Discretization of the periodic unit cell.
    phase_field_1nxyz : muGrid Field, shape [1, n, x, y(, z)]
        Nodal phase field. Ghost layers are updated (MPI) as a side effect.

    Returns
    -------
    float
        Global value (``integrate_over_cell`` already performs the MPI sum).
        Not multiplied by ``eta``.
    """
    # Input: phase_field [1,n,x,y,z]
    # Output: potential [1]
    # phase field gradient potential = int (  (grad(rho))^2 )    dx
    # (re) allocate field  for gradient
    phase_field_gradient_ijqxyz = discretization.get_gradient_of_scalar_field(
        name='compute_gradient_of_phase_field_potential')
    phase_field_gradient_ijqxyz.s.fill(0)

    discretization.fft.communicate_ghosts(phase_field_1nxyz)
    discretization.apply_gradient_operator_mugrid(u_inxyz=phase_field_1nxyz,
                                                  grad_u_ijqxyz=phase_field_gradient_ijqxyz)

    # this is without mpi. Integrate already has mpi sum
    # integrate_over_cell returns the [1, d] array int (d rho/d x_j)^2 dx;
    # summing over j gives int |grad rho|^2 dx
    f_rho_grad = np.sum(
        discretization.integrate_over_cell(phase_field_gradient_ijqxyz.s ** 2))
    # print(f'rank = {MPI.COMM_WORLD.rank}' + 'f_rho_grad= {} '.format(
    #     f_rho_grad))
    return f_rho_grad


def partial_derivative_of_gradient_of_phase_field_potential(discretization,
                                                            phase_field_1nxyz,
                                                            output_1nxyz):
    """
    Nodal derivative of the gradient energy ``int |grad rho|^2 dx``.

    Since ``f_rho_grad = rho^T D^T W D rho`` is quadratic,

    ``d f_rho_grad / d rho = 2 D^T W D rho``,

    which is evaluated with the muGrid gradient operator and its weighted
    transpose (a discrete, weak Laplacian).

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        Discretization of the periodic unit cell.
    phase_field_1nxyz : muGrid Field, shape [1, n, x, y(, z)]
        Nodal phase field. Ghost layers are updated (MPI) as a side effect.
    output_1nxyz : muGrid Field, shape [1, n, x, y(, z)]
        Output field, overwritten in place with ``2 D^T W D rho``.

    Returns
    -------
    None
        The result is written into ``output_1nxyz``. Not multiplied by ``eta``.
    """
    # Input: phase_field [1,n,x,y,z]
    # Output: ∂ potential/ ∂ pha adjoint_potential ase_field [1,n,x,y,z] # Note: one potential per phase field DOF

    # partial derivative of  phase field gradient potential = 2/eta int (  (grad(rho))^2 )    dx
    # Compute       grad (rho). grad I  without using I
    #
    #  (D rho, D I) ==  ( D I, D rho) and thus  == I D_t D rho
    # I try to implement it in the way = 2/eta (int I D_t D rho )
    phase_field_grad_1jqxyz = discretization.get_gradient_of_scalar_field(
        name='temporary_partial_derivative_of_gradient_of_phase_field_potential')

    discretization.fft.communicate_ghosts(phase_field_1nxyz)

    # D rho : nodal -> quadrature-point gradient [1, d, q, x, y(, z)]
    discretization.gradient_op.apply(nodal_field=phase_field_1nxyz,
                                 quadrature_point_field=phase_field_grad_1jqxyz)

    weights = discretization.quadrature_weights

    # ghosts of the quadrature field are needed by the transposed stencil
    discretization.fft.communicate_ghosts(phase_field_grad_1jqxyz)

    # D^T W (D rho) : weighted transposed gradient back to the nodes
    discretization.gradient_op.transpose(quadrature_point_field=phase_field_grad_1jqxyz,
                                     nodal_field=output_1nxyz,
                                     weights=weights)

    # phase_field_gradient = discretization.apply_gradient_operator(phase_field_1nxyz)
    # phase_field_gradient = discretization.apply_quadrature_weights_on_gradient_field(phase_field_gradient)
    # Dt_D_rho = discretization.apply_gradient_transposed_operator(phase_field_gradient)
    output_1nxyz.s[...] *= 2


##
def objective_function_stress_equivalence(discretization,
                                          actual_stress_ij,
                                          target_stress_ij):
    """
    Normalized squared stress mismatch ``|Sigma_t - Sigma_h|^2 / |Sigma_t|^2``.

    Identical to :func:`compute_stress_equivalence_potential` (without printing).

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        Unused, kept for a uniform interface.
    actual_stress_ij : ndarray, shape [d, d]
        Homogenized stress ``Sigma_h``.
    target_stress_ij : ndarray, shape [d, d]
        Target stress ``Sigma_t``.

    Returns
    -------
    float
        Dimensionless stress mismatch.
    """
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
    """
    Legacy: partial derivative (fixed displacement) of the energy-equivalence term.

    For ``f = (E_L : (Sigma_t - Sigma_h))^2 / W_t^2`` with
    ``Sigma_h = 1/|Omega| int C(rho) : (E + grad^s u) dx`` this function returns
    the nodal field

    ``-2 / (|Omega| W_t^2) * N^T W [ E_L : dC/drho : (E + grad^s u) ]``,

    i.e. ``d f / d rho`` **without** the scalar factor ``E_L : (Sigma_t - Sigma_h)``
    (the caller, :func:`sensitivity_elastic_energy_and_adjoint_FE_NEW`,
    multiplies by it).

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        Discretization of the periodic unit cell.
    base_material_data_ijkl : ndarray, shape [d, d, d, d]
        Base (solid) stiffness ``C_1``.
    displacement_field_fnxyz : muGrid Field, shape [d, n, x, y(, z)]
        Displacement fluctuation ``u``.
    macro_gradient_field_ijqxyz : muGrid Field, shape [d, d, q, x, y(, z)]
        Macroscopic strain ``E`` at quadrature points.
    phase_field_1nxyz : muGrid Field, shape [1, n, x, y(, z)]
        Nodal phase field.
    target_stress_ij, actual_stress_ij : ndarray, shape [d, d]
        Unused here (kept for interface symmetry).
    left_macro_gradient_ij : ndarray, shape [d, d]
        Left macroscopic strain ``E_L`` of the energy contraction.
    target_energy : float
        Normalisation ``W_t``.
    p : int or float
        SIMP exponent.

    Returns
    -------
    ndarray, shape [1, n, x, y(, z)]
        Nodal partial derivative (see above).

    Notes
    -----
    Legacy: uses ``evaluate_field_at_quad_points`` returning a tuple,
    ``get_material_data_size_field``/``apply_gradient_operator_symmetrized``
    NumPy-style APIs and assembles ``N^T`` via an explicit loop over pixel
    corners with FFT-based rolls. The material derivative is computed as
    ``C_1 * (p * rho)^(p - 1)``, which equals the SIMP derivative
    ``p rho^(p-1) C_1`` only for ``p = 1`` (void stiffness is ignored).
    """
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
    dmaterial_data_field_drho_ijklqxyz = discretization.get_material_data_size_field(
        name='data_field_in_partial_derivative_of_energy_equivalence_wrt_phase_field_FE')
    dmaterial_data_field_drho_ijklqxyz.s[...] = base_material_data_ijkl[..., np.newaxis, np.newaxis, np.newaxis] * \
                                                np.power(
                                                    p * phase_field_at_quad_poits_1qnxyz, p - 1)[0, :, 0, ...]
    # I consider linear interpolation of material  C_ijkl= p*rho**(p-1) C^0_ijkl
    # so  ∂ C_ijkl/ ∂ rho = 1* C^0_ijkl

    # int(∂ C/ ∂ rho_i  * (macro_grad + micro_grad)) dx / | domain |

    # compute strain field from to displacement and macro gradient
    strain_ijqxyz = discretization.get_displacement_gradient_sized_field(name='strain_ijqxyz_local_at_pdofsewpf')
    strain_ijqxyz = discretization.apply_gradient_operator_symmetrized(u_inxyz=displacement_field_fnxyz,
                                                                       grad_u_ijqxyz=strain_ijqxyz)
    strain_ijqxyz.s[...] = macro_gradient_field_ijqxyz.s + strain_ijqxyz.s

    # Get the stress field (in the strain field name)
    strain_ijqxyz.s[...] = discretization.apply_material_data_elasticity(
        material_data=dmaterial_data_field_drho_ijklqxyz,
        gradient_field=strain_ijqxyz)
    # apply quadrature weights
    # TODO not sure if this should be here
    strain_ijqxyz.s[...] = discretization.apply_quadrature_weights_on_gradient_field(grad_field=strain_ijqxyz.s)

    # stress difference
    # E_L : (w_q dC/drho : eps) at every quadrature point -> [q, x, y(, z)]
    double_contraction_stress_qxyz_FE = np.einsum('ij,ijqxy...->qxy...',
                                                  left_macro_gradient_ij,
                                                  strain_ijqxyz.s)

    dfstress_drho_OLD = discretization.get_scalar_field(name='dfstress_drho_OLD_')
    dfstress_drho_OLD.s.fill(0)
    # Explicit N^T assembly: loop over the 2^d pixel corners, contract with the
    # shape-function values N_corner(x_q) and shift the result to the global
    # node of that corner (periodic roll via FFT phase shift).
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
                                                                                   void_material_data_ijkl,
                                                                                   displacement_field_fnxyz,
                                                                                   macro_gradient_field_ijqxyz,
                                                                                   phase_field_1nxyz,
                                                                                   target_stress_ij,
                                                                                   actual_stress_ij,
                                                                                   p):
    """
    Partial derivative (at fixed displacement) of the stress-equivalence term w.r.t. the phase field.

    For ``f_sigma = |Sigma_t - Sigma_h|^2 / |Sigma_t|^2`` with
    ``Sigma_h = 1/|Omega| int C(rho) : (E + grad^s u) dx`` and the SIMP law
    ``C(rho) = (C_1 - C_0) rho^p + C_0`` one obtains for every node ``a``

    ``d f_sigma / d rho_a = -2 / (|Omega| |Sigma_t|^2)
    * int (Sigma_t - Sigma_h) : dC/drho : (E + grad^s u) N_a dx``

    with ``dC/drho = p rho^(p-1) (C_1 - C_0)`` evaluated at the quadrature
    points (``rho`` interpolated with ``N``). The integral is computed as
    ``N^T W [...]`` (weighted transposed interpolation).

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        Discretization of the periodic unit cell.
    material_data_field_ijkl : ndarray, shape [d, d, d, d]
        Base (solid) stiffness ``C_1`` (constant, not a field).
    void_material_data_ijkl : ndarray, shape [d, d, d, d]
        Void stiffness ``C_0``.
    displacement_field_fnxyz : muGrid Field, shape [d, n, x, y(, z)]
        Displacement fluctuation ``u`` of the equilibrium solution.
    macro_gradient_field_ijqxyz : muGrid Field, shape [d, d, q, x, y(, z)]
        Macroscopic strain ``E`` at quadrature points.
    phase_field_1nxyz : muGrid Field, shape [1, n, x, y(, z)]
        Nodal phase field.
    target_stress_ij : ndarray, shape [d, d]
        Target stress ``Sigma_t``.
    actual_stress_ij : ndarray, shape [d, d]
        Homogenized stress ``Sigma_h`` of the current design.
    p : int or float
        SIMP exponent.

    Returns
    -------
    muGrid Field, shape [1, n, x, y(, z)]
        Nodal partial derivative (temporary field ``'dfstress_drho_output'``
        of the field collection; it is overwritten on the next call).

    Notes
    -----
    Uses and overwrites several named temporary fields of the discretization
    (e.g. ``'phase_field_at_quads_reusable'``, ``'temp_at_quads'``).
    Involves ghost communication (MPI) inside the muGrid operators.
    """
    # Input: phase_field [1,n,x,y,z]
    #        material_data_field [d,d,d,d,q,x,y,z] - elasticity
    #        target_stress_ij [d,d]
    #        actual_stress_ij [d,d]
    # -- -- -- -- -- -- -- -- -- -- --
    dim = discretization.domain_dimension
    # Gradient of material data with respect to phase field
    # % interpolation of rho into quad points
    # -----    stress difference potential ----- #
    # Gradient of material data with respect to phase field
    phase_field_at_quad_poits_1qxyz = discretization.get_quad_field_scalar(
        name='phase_field_at_quads_reusable')
    discretization.apply_N_operator_mugrid(phase_field_1nxyz, phase_field_at_quad_poits_1qxyz)

    dmaterial_data_field_drho_ijklqxyz_FE = discretization.get_material_data_size_field_mugrid(
        name='data_field_drho_partial_derivative_of_objective_function_stress_equivalence_wrt_phase_field_FE')
    # dmaterial_data_field_drho_ijklqxyz_FE.s[...] = material_data_field_ijkl[..., np.newaxis, np.newaxis, np.newaxis] * \
    #                                           np.power(
    #                                               p * phase_field_at_quad_poits_1qxyz.s, p - 1)[0, 0, :, ...]
    # TODO{WARNING} here is missing Cvoidd becouse of derivateive
    # dmaterial_data_field_drho_ijklqxyz_FE.s[...] = (material_data_field_ijkl - void_material_data_ijkl)[
    #                                                    ..., np.newaxis, np.newaxis, np.newaxis] * \
    #                                                (p * np.power(phase_field_at_quad_poits_1qxyz.s, p - 1))[
    #                                                    0, 0, :, ...]
    # broadcast the constant [d,d,d,d] tensor over the trailing (q, x, y[, z]) axes
    expand = (...,) + (np.newaxis,) * (dim + 1)

    # dC/drho at quadrature points: p * rho_q^(p-1) * (C_1 - C_0)  -> [d,d,d,d,q,x,y(,z)]
    dmaterial_data_field_drho_ijklqxyz_FE.s[...] = \
        (material_data_field_ijkl - void_material_data_ijkl)[expand] \
        * (p * np.power(phase_field_at_quad_poits_1qxyz.s, p - 1))[0, 0, :, ...]
    # I consider linear interpolation of material  C_ijkl= p*rho**(p-1) C^0_ijkl
    # so  ∂ C_ijkl/ ∂ rho = 1* C^0_ijkl
    # (more precisely: SIMP C = (C_1 - C_0) rho^p + C_0  =>  dC/drho = p rho^(p-1) (C_1 - C_0);
    #  for p = 1 this reduces to the constant C_1 - C_0)

    # int(∂ C/ ∂ rho_i  * (macro_grad + micro_grad)) dx / | domain |

    # compute strain field from to displacement and macro gradient
    strain_ijqxyz = discretization.get_displacement_gradient_sized_field(name='strain_ijqxyz_local_at_pdofsewpf')
    discretization.apply_gradient_operator_symmetrized_mugrid(u_inxyz=displacement_field_fnxyz,
                                                              grad_u_ijqxyz=strain_ijqxyz)
    strain_ijqxyz.s[...] = macro_gradient_field_ijqxyz.s + strain_ijqxyz.s

    # compute stress field
    # dsigma/drho_ij = dC/drho_ijkl : eps_lk  (eps symmetric, so lk == kl)
    strain_ijqxyz.s[...] = np.einsum('ijkl...,lk...->ij...',
                                     dmaterial_data_field_drho_ijklqxyz_FE.s,
                                     strain_ijqxyz.s)
    # this is actually stress

    # stress difference
    stress_difference_ij = target_stress_ij - actual_stress_ij

    # (Sigma_t - Sigma_h) : dsigma/drho at every quadrature point -> [1, 1, q, x, y(, z)]
    double_contraction_stress_qxyz_FE = discretization.get_quad_field_scalar(name='temp_at_quads')
    double_contraction_stress_qxyz_FE.s[0, 0] = np.einsum('ij,ijqxy...->qxy...',
                                                          stress_difference_ij,
                                                          strain_ijqxyz.s)
    # N^T W [...] : integrate against the nodal shape functions -> nodal field
    dfstress_drho = discretization.get_scalar_field(name='dfstress_drho_output')
    discretization.apply_N_transposed_operator_mugrid(
        quad_field_ijqxyz=double_contraction_stress_qxyz_FE,
        nodal_field_inxyz=dfstress_drho,
        apply_weights=True)
    # chain rule of |Sigma_t - Sigma_h|^2 / |Sigma_t|^2 with dSigma_h = 1/|Omega| int dsigma
    dfstress_drho.s[...] = -2 * dfstress_drho.s / discretization.cell.domain_volume / np.sum(target_stress_ij ** 2)
    return dfstress_drho


def partial_derivative_of_objective_function_stress_equivalence_wrt_phase_field_pixel(discretization,
                                                                                      material_data_field_ijklqxyz,
                                                                                      displacement_field_fnxyz,
                                                                                      macro_gradient_field_ijqxyz,
                                                                                      phase_field_1nxyz,
                                                                                      target_stress_ij,
                                                                                      actual_stress_ij,
                                                                                      p):
    """
    Legacy pixel-based partial derivative of the stress-equivalence term.

    Pixel-wise (piecewise constant) design variant of
    :func:`partial_derivative_of_objective_function_stress_equivalence_wrt_phase_field_FE`:
    ``rho`` lives on pixels, ``dC/drho = p rho^(p-1) C`` and the contribution
    of all quadrature points of a pixel is summed:

    ``d f / d rho_e = -2 / (|Omega| |Sigma_t|^2) sum_q w_q (Sigma_t - Sigma_h) : dC/drho : eps_q``.

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        Discretization of the periodic unit cell.
    material_data_field_ijklqxyz : ndarray, shape [d, d, d, d, q, x, y(, z)]
        Base material data field (without phase field applied).
    displacement_field_fnxyz : ndarray, shape [d, n, x, y(, z)]
        Displacement fluctuation.
    macro_gradient_field_ijqxyz : ndarray, shape [d, d, q, x, y(, z)]
        Macroscopic strain at quadrature points.
    phase_field_1nxyz : ndarray, shape [1, n, x, y(, z)]
        Pixel-wise phase field.
    target_stress_ij, actual_stress_ij : ndarray, shape [d, d]
        Target and homogenized stress.
    p : int or float
        SIMP exponent.

    Returns
    -------
    ndarray, shape [x, y(, z)]
        Pixel-wise partial derivative.

    Notes
    -----
    Legacy: uses ``apply_gradient_operator_symmetrized`` (NumPy API) which no
    longer exists in :mod:`muFFTTO.domain`.
    """
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


def adjoint_potential(discretization,
                      stress_field_ijqxyz,
                      adjoint_field_inxyz):
    """
    Evaluate the adjoint (Lagrangian) term ``lambda^T g(u, rho)``.

    ``g = D^T W sigma`` is the discrete equilibrium residual (nodal internal
    forces) for the stress field ``sigma = C(rho) : (E + grad^s u)``; hence

    ``adjoint_potential = lambda^T D^T W sigma = int grad(lambda) : sigma dx``.

    At an exact equilibrium solution ``g = 0`` and this value vanishes; it is
    used as a diagnostic of how well the equilibrium problem was solved.

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        Discretization of the periodic unit cell.
    stress_field_ijqxyz : muGrid Field, shape [d, d, q, x, y(, z)]
        Stress field at quadrature points (without quadrature weights).
        Its ghost layers are updated (MPI) as a side effect.
    adjoint_field_inxyz : muGrid Field, shape [d, n, x, y(, z)]
        Adjoint field ``lambda``.

    Returns
    -------
    float
        Global (MPI-reduced) scalar ``lambda^T D^T W sigma``.
    """
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
    # stress_field_ijqxyz.s[...] = discretization.apply_quadrature_weights_on_gradient_field(stress_field_ijqxyz.s)
    force_field_inxyz = discretization.get_displacement_sized_field(
        name='force_field_inxyz_in_adjoint_potential_temporary')
    discretization.fft.communicate_ghosts(stress_field_ijqxyz)

    discretization.gradient_op.transpose(quadrature_point_field=stress_field_ijqxyz,
                                     nodal_field=force_field_inxyz,
                                     weights=weights)

    # pointwise dot product lambda_i * f_i (summed over components i), local to this rank
    adjoint_potential_field = np.einsum('i...,i...->...', adjoint_field_inxyz.s, force_field_inxyz.s)

    # Reductor_numpi = discretization.mpi_reduction(MPI.COMM_WORLD)
    integral = discretization.mpi_reduction.sum(adjoint_potential_field)  #

    return integral


def partial_derivative_of_adjoint_potential_wrt_phase_field_FE(discretization,
                                                               base_material_data_ijkl,
                                                               void_material_data_ijkl,
                                                               displacement_field_fnxyz,
                                                               macro_gradient_field_ijqxyz,
                                                               phase_field_1nxyz,
                                                               adjoint_field_inxyz,
                                                               output_field_inxyz,
                                                               p=1
                                                               ):
    """
    Adjoint contribution ``lambda^T dg/drho`` to the sensitivity.

    With the equilibrium residual ``g(u, rho) = D^T W C(rho) : (E + D u)`` and
    the SIMP law ``C(rho) = (C_1 - C_0) rho^p + C_0`` one has for every node ``a``

    ``lambda^T dg/drho_a = int grad^s(lambda) : dC/drho : (E + grad^s u) N_a dx``
    ``= [N^T W ( grad^s(lambda) : p rho^(p-1) (C_1 - C_0) : (E + grad^s u) )]_a``.

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        Discretization of the periodic unit cell.
    base_material_data_ijkl : ndarray, shape [d, d, d, d]
        Base (solid) stiffness ``C_1``.
    void_material_data_ijkl : ndarray, shape [d, d, d, d]
        Void stiffness ``C_0``.
    displacement_field_fnxyz : muGrid Field, shape [d, n, x, y(, z)]
        Displacement fluctuation ``u`` of the equilibrium problem.
    macro_gradient_field_ijqxyz : muGrid Field, shape [d, d, q, x, y(, z)]
        Macroscopic strain ``E`` at quadrature points.
    phase_field_1nxyz : muGrid Field, shape [1, n, x, y(, z)]
        Nodal phase field.
    adjoint_field_inxyz : muGrid Field, shape [d, n, x, y(, z)]
        Solution ``lambda`` of the adjoint problem.
    output_field_inxyz : muGrid Field, shape [1, n, x, y(, z)]
        Output field, overwritten in place.
    p : int or float, optional
        SIMP exponent (default 1).

    Returns
    -------
    muGrid Field
        ``output_field_inxyz`` (same object).

    Notes
    -----
    Uses and overwrites named temporary fields of the discretization
    (``'phase_field_at_quads_reusable'``, ``'data_field_in_sensitivity_reusable'``, ...).
    Ghost communication (MPI) happens inside the muGrid operators.
    """
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
    dim = discretization.domain_dimension
    # Gradient of material data with respect to phasse field   % interpolation of rho into quad points
    # I consider linear interpolation of material  C_ijkl= p*rho**(p-1) C^0_ijkl
    # so  ∂ C_ijkl/ ∂ rho = 1* C^0_ijkl
    # Gradient of material data with respect to phase field
    phase_field_at_quad_poits_1qxyz = discretization.get_quad_field_scalar(
        name='phase_field_at_quads_reusable')
    discretization.apply_N_operator_mugrid(phase_field_1nxyz, phase_field_at_quad_poits_1qxyz)

    dmaterial_data_field_drho_ijklqxyz = discretization.get_material_data_size_field_mugrid(
        name='data_field_in_sensitivity_reusable')

    # dmaterial_data_field_drho_ijklqxyz.s[...] = (base_material_data_ijkl - void_material_data_ijkl)[
    #                                                 ..., np.newaxis, np.newaxis, np.newaxis] * (
    #                                                     p * np.power(phase_field_at_quad_poits_1qxyz.s[0, 0, :, ...],
    #                                                                  (p - 1)))
    # broadcast the constant [d,d,d,d] tensor over the trailing (q, x, y[, z]) axes
    expand = (...,) + (np.newaxis,) * (dim + 1)

    # dC/drho = p rho_q^(p-1) (C_1 - C_0) at every quadrature point
    dmaterial_data_field_drho_ijklqxyz.s[...] = \
        (base_material_data_ijkl - void_material_data_ijkl)[expand] \
        * (p * np.power(phase_field_at_quad_poits_1qxyz.s[0, 0, :, ...], p - 1))
    # compute strain field from to displacement and macro gradient
    # -> stress_ijqxyz = dC/drho : (E + grad^s u)  ("stress derivative")
    stress_ijqxyz = discretization.get_displacement_gradient_sized_field(name='stress_ijqxyz_local_at_pdapwpf')

    discretization.get_stress_field_mugrid(
        material_data_field_ijklqxyz=dmaterial_data_field_drho_ijklqxyz,
        displacement_field_inxyz=displacement_field_fnxyz,
        macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
        output_stress_field_ijqxyz=stress_ijqxyz,
        formulation='small_strain')

    # local_stress=np.einsum('ijkl,lk->ij',material_data_field_ijklqxyz[...,0,0,0] , strain_ijqxyz[...,0,0,0])
    # apply quadrature weights
    # discretization.apply_quadrature_weights_on_gradient_field_mugrid(strain_ijqxyz)

    # gradient of adjoint_field
    adjoint_field_gradient_ijqxyz = discretization.get_displacement_gradient_sized_field(
        name='adjoint_field_gradient_ijqxyz__pdapwpf')
    discretization.apply_gradient_operator_symmetrized_mugrid(u_inxyz=adjoint_field_inxyz,
                                                              grad_u_ijqxyz=adjoint_field_gradient_ijqxyz, )
    # TODO: should this be symmetric gradient?
    # ddot22 = lambda A2, B2:  np.einsum('ijqxyz  ,jiqxyz  ->qxyz    ', A2, B2)
    double_contraction_stress_qxyz = discretization.get_quad_field_scalar(name='double_contraction_stress_qxyz')
    double_contraction_stress_qxyz.s[0, 0] = np.einsum('ij...,ij...->...',
                                                       adjoint_field_gradient_ijqxyz.s,
                                                       stress_ijqxyz.s)  # this is stress, It just cupy the same name

    output_field_inxyz.s.fill(0)
    discretization.fft.communicate_ghosts(double_contraction_stress_qxyz)

    # N^T W [grad^s(lambda) : dsigma/drho] -> nodal sensitivity contribution
    discretization.apply_N_transposed_operator_mugrid(
        quad_field_ijqxyz=double_contraction_stress_qxyz,
        nodal_field_inxyz=output_field_inxyz,
        apply_weights=True)

    return output_field_inxyz


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
    """
    Legacy: full adjoint sensitivity for a pixel-wise (piecewise constant) phase field.

    Computes ``df/drho`` of
    ``f = f_sigma + weight * (eta * int |grad rho|^2 + int rho^2 (1-rho)^2 / eta)``
    (note: ``weight`` multiplies the phase-field terms here) as

    ``df/drho = df_sigma/drho + weight * (eta * 2 D^T W D rho + ddw/drho / eta) + lambda^T dg/drho``

    where the adjoint field solves ``K(rho) lambda = -df_sigma/du`` with
    ``-df_sigma/du = 2/(|Omega| |Sigma_t|^2) D^T W C(rho) : (Sigma_t - Sigma_h)``.

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        Discretization of the periodic unit cell.
    material_data_field_ijklqxyz : ndarray, shape [d, d, d, d, q, x, y(, z)]
        Base material data field ``C`` (without phase field applied).
    displacement_field_fnxyz : ndarray, shape [d, n, x, y(, z)]
        Displacement fluctuation of the equilibrium problem.
    macro_gradient_field_ijqxyz : ndarray, shape [d, d, q, x, y(, z)]
        Macroscopic strain at quadrature points.
    phase_field_1nxyz : ndarray, shape [1, n, x, y(, z)]
        Phase field.
    target_stress_ij, actual_stress_ij : ndarray, shape [d, d]
        Target and homogenized stress.
    formulation : str
        ``'small_strain'`` or ``'finite_strain'`` (passed to the system matrix).
    p : int or float
        SIMP exponent (``C(rho) = rho^p C``).
    eta : float
        Phase-field interface-width parameter.
    weight : float
        Weight of the phase-field terms.

    Returns
    -------
    ndarray
        Sensitivity ``df/drho``.

    Notes
    -----
    Legacy: relies on NumPy-array APIs of the discretization
    (``apply_gradient_operator``, ``apply_gradient_operator_symmetrized``,
    ``apply_gradient_transposed_operator``, ``get_rhs`` returning an array)
    and on ``solvers.PCG``, which are no longer available, and calls
    :func:`partial_der_of_double_well_potential_wrt_density_nodal` without
    the required ``output_1nxyz`` argument.
    """
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
    # (the constant stress difference is broadcast to every quadrature point and
    #  plays the role of "E" in -D^T W C : E)

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
    # lambda^T dg/drho: grad^s(lambda) : (w_q dC/drho : eps) summed over quadrature points of a pixel
    double_contraction_stress_qxyz = np.einsum('ij...,ij...->...',
                                               adjoint_field_gradient_ijqxyz,
                                               stress_field_ijqxyz)

    dg_drho = double_contraction_stress_qxyz.sum(axis=0)

    return df_drho + dg_drho



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
    """
    Full adjoint sensitivity of stress-equivalence + phase-field objective (older muGrid version).

    Objective (consistent with :func:`objective_function_small_strain`):

    ``f = weight * f_sigma + eta * int |grad rho|^2 + double_well_depth / eta * int rho^2 (1-rho)^2``.

    Steps:

    1. ``df_sigma/drho`` at fixed ``u``
       (:func:`partial_derivative_of_objective_function_stress_equivalence_wrt_phase_field_FE`),
    2. analytical double-well derivative and gradient-energy derivative,
    3. adjoint problem ``K(rho) lambda = -weight * df_sigma/du`` with
       ``-df_sigma/du = 2/(|Omega| |Sigma_t|^2) D^T W C(rho) : (Sigma_t - Sigma_h)``,
       solved by preconditioned CG,
    4. adjoint term ``lambda^T dg/drho``
       (:func:`partial_derivative_of_adjoint_potential_wrt_phase_field_FE`).

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        Discretization of the periodic unit cell.
    base_material_data_ijkl : ndarray, shape [d, d, d, d]
        Base (solid) stiffness ``C_1``; here ``C(rho) = rho^p C_1`` (no void term).
    displacement_field_inxyz : muGrid Field, shape [d, n, x, y(, z)]
        Displacement fluctuation of the equilibrium solution.
    macro_gradient_field_ijqxyz : muGrid Field, shape [d, d, q, x, y(, z)]
        Macroscopic strain ``E`` at quadrature points.
    phase_field_1nxyz : muGrid Field, shape [1, n, x, y]
        Nodal phase field.
    target_stress_ij, actual_stress_ij : ndarray, shape [d, d]
        Target and homogenized stress.
    preconditioner_fun : callable
        Preconditioner ``P(r, z)`` for the CG solver.
    system_matrix_fun : callable
        Linear operator ``K(rho)`` (Hessian-vector product) for the CG solver.
    formulation : str
        Unused (small strain is assumed).
    p : int or float
        SIMP exponent.
    eta : float
        Phase-field interface-width parameter.
    weight : float
        Weight of the stress term.
    double_well_depth : float, optional
        Scaling of the double-well term (default 1).

    Returns
    -------
    sensitivity : ndarray, shape [1, n, x, y]
        ``df/drho``.
    sensitivity_parts : dict
        Norms of the individual contributions (returned because the
        hard-coded flag ``test`` is True).

    Notes
    -----
    Rank 0 prints the number of adjoint CG iterations. Out of sync with the
    current helper signatures: the helpers called here require
    ``void_material_data_ijkl``, which is not passed, so this function will
    raise a ``TypeError``; use :func:`sensitivity_stress_and_adjoint_FE_NEW`
    together with :func:`sensitivity_phase_field_term_FE_NEW` instead.
    """
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
    phase_field_at_quad_poits_1qxyz = discretization.get_quad_field_scalar(
        name='phase_field_at_quad_poits_1qxyzsensitivity_with_adjoint_problem_FE_NEW')
    discretization.apply_N_operator_mugrid(phase_field_1nxyz, phase_field_at_quad_poits_1qxyz)

    material_data_field_rho_ijklqxyz = discretization.get_material_data_size_field_mugrid(
        name='data_field_in_sensitivity_with_adjoint_problem_FE_NEW')
    # C(rho) = rho_q^p C_1 at quadrature points (three new axes -> 2D only: q, x, y)
    material_data_field_rho_ijklqxyz.s[...] = base_material_data_ijkl[..., np.newaxis, np.newaxis, np.newaxis] * \
                                              np.power(phase_field_at_quad_poits_1qxyz.s, p)[0, 0, :, ...]

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
    # d/dρ(ρ^2 (1 - ρ)^2) = 2 ρ (2 ρ^2 - 3 ρ + 1)
    ddw_drho = discretization.get_scalar_field(name='ddw_drho')
    partial_der_of_double_well_potential_wrt_density_analytical(discretization=discretization,
                                                                phase_field_1nxyz=phase_field_1nxyz,
                                                                output_1nxyz=ddw_drho)

    # -----    phase field gradient potential ----- #
    # partial derivative of  phase field gradient potential = 2/eta int (  (grad(rho))^2 )    dx
    #  (D rho, D I) ==  ( D I, D rho) and thus  == I D_t D rho
    dgradrho_drho = discretization.get_scalar_field(name='dgradrho_drho')
    dgradrho_drho.s.fill(0)
    partial_derivative_of_gradient_of_phase_field_potential(discretization=discretization,
                                                            phase_field_1nxyz=phase_field_1nxyz,
                                                            output_1nxyz=dgradrho_drho)

    # sum of all parts of df_drho
    df_drho = weight * dstress_drho + (dgradrho_drho.s * eta + double_well_depth * ddw_drho.s / eta)

    stress_difference_ij = target_stress_ij - actual_stress_ij
    # Adjoint problem
    # RHS b = -d(weight f_sigma)/du = 2 weight / (|Omega| |Sigma_t|^2) D^T W C(rho) : (Sigma_t - Sigma_h).
    # get_rhs_mugrid computes -D^T W C : X for a constant "macro gradient" X, so
    # X = Sigma_t - Sigma_h is broadcast to all quadrature points and the result
    # is multiplied by -2/|Omega| (and the normalisation) below.
    stress_difference_ijqxyz = discretization.get_gradient_size_field(
        name='stress_difference_ijqxyz_in_sensitivity_with_adjoint_problem')

    discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=stress_difference_ij,
                                                   macro_gradient_field_ijqxyz=stress_difference_ijqxyz
                                                   )
    # minus sign is already there
    df_du_field = discretization.get_unknown_size_field(name='adjoint_problem_rhs')
    discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_rho_ijklqxyz,
                                  macro_gradient_field_ijqxyz=stress_difference_ijqxyz,
                                  rhs_inxyz=df_du_field)  # minus sign is already there
    df_du_field.s[...] = -2 * df_du_field.s / discretization.cell.domain_volume  # this is now on the right hand side
    # Normalization
    df_du_field.s[...] = weight * df_du_field.s / np.sum(target_stress_ij ** 2)

    # solve adjoint problem
    norms_cg_adjoint = dict()
    norms_cg_adjoint['residual_rr'] = []
    norms_cg_adjoint['residual_rz'] = []

    def callback_adjoint(it, x, r, p, z, stop_crit_norm):
        """CG callback: record global squared residual norms r.r and r.z (MPI-reduced)."""
        # global norms_cg_mech
        norm_of_rr = discretization.communicator.sum(np.dot(r.ravel(), r.ravel()))
        norm_of_rz = discretization.communicator.sum(np.dot(r.ravel(), z.ravel()))
        norms_cg_adjoint['residual_rr'].append(norm_of_rr)
        norms_cg_adjoint['residual_rz'].append(norm_of_rz)

    adjoint_field_inxyz = discretization.get_unknown_size_field(
        name='adjoint_field_inxyz_in_sensitivity_with_adjoint_problem')
    solvers.conjugate_gradients_mugrid(
        comm=discretization.communicator,
        fc=discretization.field_collection,
        hessp=system_matrix_fun,  # linear operator
        b=df_du_field,
        x=adjoint_field_inxyz,
        P=preconditioner_fun,
        tol=1e-10,
        maxiter=int(10000),
        callback=callback_adjoint,
        # norm_metric=res_norm
    )

    if MPI.COMM_WORLD.rank == 0:
        nb_it_comb = len(norms_cg_adjoint['residual_rr'])
        norm_rz = norms_cg_adjoint['residual_rr'][-1]
        print(' nb_ steps CG adjoint =' f'{nb_it_comb}, residual_rz = {norm_rz}')

    # lambda^T dg/drho
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

    # Diagnostic only: lambda^T g(u, rho) should be ~0 at equilibrium
    stress_field_ijqxyz = discretization.get_gradient_size_field(
        name='stress_field_ijqxyz_in_sensitivity_stress_and_adjoint_FE_NEW')
    discretization.get_stress_field_mugrid(
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
                             'dgradrho_drho': np.linalg.norm(dgradrho_drho.s),
                             'ddouble_well_drho_drho': np.linalg.norm(ddw_drho.s),
                             'dphase_drho': np.linalg.norm(dgradrho_drho.s + ddw_drho.s),
                             'df_drho_': np.linalg.norm(dstress_drho + dgradrho_drho.s + ddw_drho.s),
                             'df_drho': np.linalg.norm(df_drho),
                             'dg_drho_nxyz_mpi': np.linalg.norm(dadjoin_drho.s),
                             'sensitivity': np.linalg.norm(df_drho + dadjoin_drho.s),
                             'adjoint_energy': adjoint_energy}
        # print(sensitivity_parts)

        return df_drho + dadjoin_drho.s, sensitivity_parts

    else:
        return df_drho + dadjoin_drho.s


def sensitivity_stress_and_adjoint_FE_NEW(discretization,
                                          base_material_data_ijkl,
                                          void_material_data_ijkl,
                                          displacement_field_inxyz,
                                          adjoint_field_inxyz,
                                          macro_gradient_field_ijqxyz,
                                          phase_field_1nxyz,
                                          target_stress_ij,
                                          actual_stress_ij,
                                          preconditioner_fun,
                                          system_matrix_fun, p,
                                          weight,
                                          formulation=None,
                                          disp=False,
                                          **kwargs):
    """
    Adjoint sensitivity of the (weighted) stress-equivalence term for one load case.

    Computes the total derivative of ``weight * f_sigma`` with
    ``f_sigma = |Sigma_t - Sigma_h(u(rho), rho)|^2 / |Sigma_t|^2`` and the SIMP
    law ``C(rho) = (C_1 - C_0) rho^p + C_0``:

    ``d(weight f_sigma)/drho = weight * df_sigma/drho|_u + lambda^T dg/drho``

    1. ``df_sigma/drho|_u`` -- explicit dependence through ``C(rho)``
       (:func:`partial_derivative_of_objective_function_stress_equivalence_wrt_phase_field_FE`).
    2. Adjoint problem ``K(rho) lambda = b`` with
       ``b = -weight * df_sigma/du = 2 weight / (|Omega| |Sigma_t|^2) D^T W C(rho) : (Sigma_t - Sigma_h)``,
       solved with :func:`muFFTTO.solvers.conjugate_gradients_mugrid`
       (``adjoint_field_inxyz`` is used as initial guess -> warm start).
    3. Adjoint term ``lambda^T dg/drho = N^T W [grad^s(lambda) : dC/drho : (E + grad^s u)]``
       (:func:`partial_derivative_of_adjoint_potential_wrt_phase_field_FE`).

    The phase-field (regularisation) part of the sensitivity is computed
    separately by :func:`sensitivity_phase_field_term_FE_NEW`.

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        Discretization of the periodic unit cell.
    base_material_data_ijkl : ndarray, shape [d, d, d, d]
        Base (solid) stiffness ``C_1``.
    void_material_data_ijkl : ndarray, shape [d, d, d, d]
        Void stiffness ``C_0``.
    displacement_field_inxyz : muGrid Field, shape [d, n, x, y(, z)]
        Displacement fluctuation ``u`` of the equilibrium solution for this load case.
    adjoint_field_inxyz : muGrid Field, shape [d, n, x, y(, z)]
        On input: initial guess for the adjoint field. On output: the
        adjoint solution ``lambda`` (modified in place).
    macro_gradient_field_ijqxyz : muGrid Field, shape [d, d, q, x, y(, z)]
        Macroscopic strain ``E`` at quadrature points.
    phase_field_1nxyz : muGrid Field, shape [1, n, x, y(, z)]
        Nodal phase field.
    target_stress_ij : ndarray, shape [d, d]
        Target homogenized stress ``Sigma_t``.
    actual_stress_ij : ndarray, shape [d, d]
        Homogenized stress ``Sigma_h`` of the current design.
    preconditioner_fun : callable
        Preconditioner for the CG solver (signature as expected by
        ``conjugate_gradients_mugrid``).
    system_matrix_fun : callable
        Linear operator ``K(rho)`` (stiffness/Hessian-vector product) of the
        current design, same operator as used for the equilibrium problem.
    p : int or float
        SIMP exponent.
    weight : float
        Weight of the stress-equivalence term in the objective.
    formulation : str, optional
        Unused (small strain is assumed).
    disp : bool, optional
        If True, rank 0 prints CG iteration info.
    **kwargs
        ``cg_tol`` (float, default 1e-7): CG tolerance;
        ``r_tol`` (bool, default True): passed as ``rtol`` to the CG solver
        (relative vs. absolute stopping criterion).

    Returns
    -------
    sensitivity : ndarray, shape [1, n, x, y(, z)]
        ``weight * df_sigma/drho|_u + lambda^T dg/drho`` (MPI-local part).
    adjoint_field_inxyz : muGrid Field
        The adjoint solution (same object as the input).
    adjoint_energy : float
        Diagnostic ``lambda^T D^T W sigma`` (:func:`adjoint_potential`),
        should be ~0 if the equilibrium was solved accurately.
    info_adjoint_ : dict
        ``'residual_rz'`` -- list of ``r.z`` per CG iteration (filled on
        rank 0 only); ``'num_iteration_adjoint'`` -- number of CG iterations
        (rank 0 only).

    Notes
    -----
    All reductions inside CG and :func:`adjoint_potential` are global (MPI).
    Several named temporary fields of the discretization are overwritten.
    """
    cg_tol = kwargs.get('cg_tol', 1e-7)
    r_tol = kwargs.get('r_tol', True)
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
    dim = discretization.domain_dimension
    # -----    stress difference potential ----- #
    # Gradient of material data with respect to phase field
    phase_field_at_quad_poits_1qxyz = discretization.get_quad_field_scalar(
        name='phase_field_at_quads_reusable')
    discretization.apply_N_operator_mugrid(phase_field_1nxyz, phase_field_at_quad_poits_1qxyz)

    material_data_field_rho_ijklqxyz = discretization.get_material_data_size_field_mugrid(
        name='data_field_in_sensitivity_sensitivity_stress_and_adjoint_FE_NEW')
    # material_data_field_rho_ijklqxyz.s[...] = base_material_data_ijkl[..., np.newaxis, np.newaxis, np.newaxis] * \
    #                                      np.power(phase_field_at_quad_poits_1qxyz.s, p)[0, 0, :, ...]

    # material_data_field_rho_ijklqxyz.s[...] = (base_material_data_ijkl - void_material_data_ijkl)[
    #                                               ..., np.newaxis, np.newaxis, np.newaxis] * \
    #                                           np.power(phase_field_at_quad_poits_1qxyz.s, p)[0, 0, :, ...] + \
    #                                           void_material_data_ijkl[
    #                                               ..., np.newaxis, np.newaxis, np.newaxis]

    # dim = 2 or 3 (number of spatial dimensions)
    # quad axis (q) + spatial axes (x, y[, z]) -> dim + 1 trailing axes
    expand = (...,) + (np.newaxis,) * (dim + 1)

    # SIMP: C(rho_q) = (C_1 - C_0) rho_q^p + C_0 at every quadrature point
    # (same as material_interpolation_simp)
    material_data_field_rho_ijklqxyz.s[...] = \
        (base_material_data_ijkl - void_material_data_ijkl)[expand] \
        * np.power(phase_field_at_quad_poits_1qxyz.s, p)[0, 0, :, ...] \
        + void_material_data_ijkl[expand]

    # d_stress_d_rho phase field gradient potential for a phase field without perturbation
    # (explicit part df_sigma/drho at fixed u)
    dstress_drho = partial_derivative_of_objective_function_stress_equivalence_wrt_phase_field_FE(
        discretization=discretization,
        phase_field_1nxyz=phase_field_1nxyz,
        target_stress_ij=target_stress_ij,
        actual_stress_ij=actual_stress_ij,
        material_data_field_ijkl=base_material_data_ijkl,
        void_material_data_ijkl=void_material_data_ijkl,
        displacement_field_fnxyz=displacement_field_inxyz,
        macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
        p=p)

    # Adjoint problem
    # compute strain field from to displacement and macro gradient
    # strain_fluctuation_ijqxyz = discretization.apply_gradient_operator_symmetrized(displacement_field_fnxyz)
    stress_difference_ij = target_stress_ij - actual_stress_ij

    # RHS b = -d(weight f_sigma)/du = 2 weight / (|Omega| |Sigma_t|^2) D^T W C(rho) : (Sigma_t - Sigma_h).
    # get_rhs_mugrid computes -D^T W C : X for a constant "macro gradient" X;
    # here X = Sigma_t - Sigma_h broadcast to all quadrature points.
    # NOTE: this buffer has the same name as stress_field_ijqxyz below, i.e. the
    # same memory is reused once the RHS has been assembled.
    stress_difference_ijqxyz = discretization.get_gradient_size_field(
        name='stress_field_in_sensitivity_stress_and_adjoint_FE_NEW')
    discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=stress_difference_ij,
                                                   macro_gradient_field_ijqxyz=stress_difference_ijqxyz
                                                   )
    # minus sign is already there
    df_du_field = discretization.get_unknown_size_field(
        name='adjoint_problem_rhs_in_sensitivity_stress_and_adjoint_FE_NEW')
    discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_rho_ijklqxyz,
                                  macro_gradient_field_ijqxyz=stress_difference_ijqxyz,
                                  rhs_inxyz=df_du_field)
    # minus sign is already there
    df_du_field.s[...] = -2 * df_du_field.s / discretization.cell.domain_volume
    # Normalization
    df_du_field.s[...] = weight * df_du_field.s / np.sum(target_stress_ij ** 2)

    info_adjoint_ = {}
    # if MPI.COMM_WORLD.rank == 0:
    norms_cg_adjoint = dict()
    norms_cg_adjoint['residual_rr'] = []
    norms_cg_adjoint['residual_rz'] = []

    def callback_adjoint(it, x, r, p, z, stop_crit_norm):
        """CG callback: record global r.r and r.z (MPI-reduced on all ranks, stored on rank 0)."""
        # global norms_cg_mech
        # communicator.sum is collective -> must be called on every rank
        norm_of_rr = discretization.communicator.sum(np.dot(r.ravel(), r.ravel()))
        norm_of_rz = discretization.communicator.sum(np.dot(r.ravel(), z.ravel()))
        if MPI.COMM_WORLD.rank == 0:
            norms_cg_adjoint['residual_rr'].append(norm_of_rr)
            norms_cg_adjoint['residual_rz'].append(norm_of_rz)

    solvers.conjugate_gradients_mugrid(
        comm=discretization.communicator,
        fc=discretization.field_collection,
        hessp=system_matrix_fun,  # linear operator
        b=df_du_field,
        x=adjoint_field_inxyz,
        P=preconditioner_fun,
        tol=cg_tol,
        rtol=r_tol,
        maxiter=int(10000),
        callback=callback_adjoint,
        # norm_metric=res_norm
    )

    info_adjoint_['residual_rz'] = norms_cg_adjoint['residual_rz']
    if MPI.COMM_WORLD.rank == 0:
        nb_it = len(norms_cg_adjoint['residual_rr'])
        info_adjoint_['num_iteration_adjoint'] = nb_it

        del norms_cg_adjoint
        gc.collect()

        if disp:
            print(f" nb_steps CG adjoint = {nb_it}, residual_rz = {info_adjoint_['residual_rz']}")
    # lambda^T dg/drho  (adjoint contribution to the sensitivity)
    dadjoin_drho = discretization.get_scalar_field(name='dadjoin_drho_in_sensitivity_stress_and_adjoint_FE_NEW')
    dadjoin_drho = partial_derivative_of_adjoint_potential_wrt_phase_field_FE(
        discretization=discretization,
        base_material_data_ijkl=base_material_data_ijkl,
        void_material_data_ijkl=void_material_data_ijkl,
        displacement_field_fnxyz=displacement_field_inxyz,
        macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
        phase_field_1nxyz=phase_field_1nxyz,
        adjoint_field_inxyz=adjoint_field_inxyz,
        output_field_inxyz=dadjoin_drho,
        p=p)

    # Diagnostic only: adjoint "energy" lambda^T D^T W sigma (~0 at equilibrium)
    stress_field_ijqxyz = discretization.get_gradient_size_field(
        name='stress_field_in_sensitivity_stress_and_adjoint_FE_NEW')
    discretization.get_stress_field_mugrid(
        material_data_field_ijklqxyz=material_data_field_rho_ijklqxyz,
        displacement_field_inxyz=displacement_field_inxyz,
        macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
        output_stress_field_ijqxyz=stress_field_ijqxyz,
        formulation='small_strain')

    adjoint_energy = adjoint_potential(
        discretization=discretization,
        stress_field_ijqxyz=stress_field_ijqxyz,
        adjoint_field_inxyz=adjoint_field_inxyz)

    return weight * dstress_drho.s + dadjoin_drho.s, adjoint_field_inxyz, adjoint_energy, info_adjoint_


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
    """
    Legacy: adjoint sensitivity of the (weighted) energy-equivalence term.

    For ``f = (E_L : (Sigma_t - Sigma_h))^2 / W_t^2`` (see
    :func:`compute_elastic_energy_equivalence_potential`) with
    ``C(rho) = rho^p C_1``:

    ``d(weight f)/drho = weight * df/drho|_u + lambda^T dg/drho``,

    * ``df/drho|_u = (E_L : (Sigma_t - Sigma_h)) * [-2/(|Omega| W_t^2) N^T W (E_L : dC/drho : eps)]``,
    * adjoint problem ``K(rho) lambda = -weight df/du
      = 2 weight (E_L : (Sigma_t - Sigma_h)) / (|Omega| W_t^2) D^T W C(rho) : E_L``,
    * ``lambda^T dg/drho`` from
      :func:`partial_derivative_of_adjoint_potential_wrt_phase_field_FE`.

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        Discretization of the periodic unit cell.
    base_material_data_ijkl : ndarray, shape [d, d, d, d]
        Base (solid) stiffness ``C_1``.
    displacement_field_inxyz : muGrid Field, shape [d, n, x, y(, z)]
        Displacement fluctuation of the equilibrium solution.
    adjoint_field_inxyz : muGrid Field, shape [d, n, x, y(, z)]
        Initial guess / output for the adjoint field (overwritten).
    macro_gradient_field_ijqxyz : muGrid Field, shape [d, d, q, x, y(, z)]
        Macroscopic strain ``E`` at quadrature points.
    left_macro_gradient_ij : ndarray, shape [d, d]
        Left macroscopic strain ``E_L`` of the energy contraction.
    phase_field_1nxyz : muGrid Field, shape [1, n, x, y]
        Nodal phase field.
    target_stress_ij, actual_stress_ij : ndarray, shape [d, d]
        Target and homogenized stress.
    preconditioner_fun : callable
        Preconditioner for the CG solver.
    system_matrix_fun : callable
        Linear operator ``K(rho)``.
    formulation : str
        Unused.
    target_energy : float
        Normalisation ``W_t``.
    p : int or float
        SIMP exponent.
    weight : float
        Weight of the energy term.
    disp : bool, optional
        If True, rank 0 prints diagnostics.

    Returns
    -------
    sensitivity : ndarray / Field
        ``weight * df/drho|_u + lambda^T dg/drho``.
    adjoint_field_inxyz : muGrid Field
        Adjoint solution.
    adjoint_energy : float
        Diagnostic ``lambda^T D^T W sigma``.

    Notes
    -----
    Legacy: relies on ``solvers.PCG``, ``discretization.get_stress_field`` and
    the tuple-returning ``evaluate_field_at_quad_points`` (no longer
    available), and calls helpers without the now required
    ``void_material_data_ijkl`` argument.
    """
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
    material_data_field_rho_ijklqxyz.s[...] = base_material_data_ijkl[..., np.newaxis, np.newaxis, np.newaxis] * \
                                              np.power(phase_field_at_quad_poits_1qnxyz, p)[0, :, 0, ...]

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
    # scalar E_L : (Sigma_t - Sigma_h): outer factor of the chain rule of (.)^2
    f_sigmas_energy = np.einsum('ij,ij->...',
                                left_macro_gradient_ij,
                                stress_difference_ij)
    dstress_drho = f_sigmas_energy * dstress_drho
    # Adjoint problem
    # RHS = -weight df/du = 2 weight (E_L : dSigma) / (|Omega| W_t^2) D^T W C(rho) : E_L
    # (get_rhs gives -D^T W C : E_L, multiplied by -2/|Omega| and the factors below)
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

    df_du_field.s[...] = -2 * df_du_field.s / discretization.cell.domain_volume
    # Normalization
    df_du_field.s[...] = weight * (f_sigmas_energy * df_du_field.s) / (target_energy ** 2)

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
                                        phase_field_1nxyz,
                                        p,
                                        eta,
                                        output_array,
                                        double_well_depth=1):
    """
    Sensitivity of the phase-field (regularisation) part of the objective.

    Derivative of the objective of :func:`objective_function_phase_field`,

    ``f_rho = eta * int |grad rho|^2 dx + double_well_depth / eta * int rho^2 (1-rho)^2 dx``,

    w.r.t. the nodal phase field:

    ``df_rho/drho = eta * 2 D^T W D rho + double_well_depth / eta * |Omega|/N * (2 rho - 6 rho^2 + 4 rho^3)``.

    The double-well part uses the nodal-quadrature derivative
    (:func:`partial_der_of_double_well_potential_wrt_density_nodal`),
    consistent with the nodal evaluation in
    :func:`objective_function_phase_field`; the analytical variant is kept
    commented out.

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        Discretization of the periodic unit cell.
    phase_field_1nxyz : muGrid Field, shape [1, n, x, y(, z)]
        Nodal phase field. Ghost layers are updated (MPI) as a side effect.
    p : int or float
        Unused (SIMP exponent; kept for interface symmetry).
    eta : float
        Phase-field interface-width parameter.
    output_array : muGrid Field, shape [1, n, x, y(, z)]
        Output field, overwritten in place with ``df_rho/drho``.
    double_well_depth : float, optional
        Scaling of the double-well term (default 1).

    Returns
    -------
    None
        The result is written into ``output_array``.
    """
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
    dim = discretization.domain_dimension
    # Gradient of material data with respect to phase field
    # (the interpolated quadrature field below is not used further in this function)
    phase_field_at_quad_poits_1qxyz = discretization.get_quad_field_scalar(name='phase_field_at_quads_reusable')
    discretization.apply_N_operator_mugrid(phase_field_1nxyz, phase_field_at_quad_poits_1qxyz)

    # -----    Double well potential ----- #

    # Derivative of the double-well potential with respect to phase-field
    # phase field potential = int ( rho^2(1-rho)^2 )/eta   dx
    # gradient phase field potential = int ((2 * phase_field( + 2 * phase_field^2  -  3 * phase_field +1 )) )/eta   dx
    # d/dρ(ρ^2 (1 - ρ)^2) = 2 ρ (2 ρ^2 - 3 ρ + 1)
    ddw_drho = discretization.get_scalar_field(name='ddw_drho_sensitivity_phase_field_term_FE_NEW')
    ddw_drho.s.fill(0)
    # ddw_drho = partial_der_of_double_well_potential_wrt_density_analytical(discretization=discretization,
    #                                                                        phase_field_1nxyz=phase_field_1nxyz,
    #                                                                        output_1nxyz=ddw_drho
    #                                                                        )
    ddw_drho = partial_der_of_double_well_potential_wrt_density_nodal(discretization=discretization,
                                                                      phase_field_1nxyz=phase_field_1nxyz,
                                                                      output_1nxyz=ddw_drho
                                                                      )

    # -----    phase field gradient potential ----- #
    # partial derivative of  phase field gradient potential = 2/eta int (  (grad(rho))^2 )    dx
    #  (D rho, D I) ==  ( D I, D rho) and thus  == I D_t D rho
    dgradrho_drho = discretization.get_scalar_field(name='dgradrho_drho_sensitivity_phase_field_term_FE_NEW')
    dgradrho_drho.s.fill(0)
    partial_derivative_of_gradient_of_phase_field_potential(discretization=discretization,
                                                            phase_field_1nxyz=phase_field_1nxyz,
                                                            output_1nxyz=dgradrho_drho)

    # sum of all parts of df_drho
    output_array.s[...] = (dgradrho_drho.s * eta + double_well_depth * ddw_drho.s / eta)


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
    """
    Legacy: full adjoint sensitivity (nodal FE phase field), weight on the stress term.

    Computes ``df/drho`` of
    ``f = weight * f_sigma + eta * int |grad rho|^2 + int rho^2 (1-rho)^2 / eta``
    with ``C(rho) = rho^p C``:

    * ``df_sigma/drho|_u = 2/(|Omega| |Sigma_t|^2) N^T W [(Sigma_h - Sigma_t) : dC/drho : eps]``
      (``N^T`` assembled by an explicit loop over pixel corners),
    * ``2 D^T W D rho`` for the gradient term and the analytical double-well derivative,
    * adjoint problem ``K lambda = 2/(|Omega| |Sigma_t|^2) D^T W C : (Sigma_t - Sigma_h)``
      solved with ``solvers.PCG`` and an FFT preconditioner,
    * adjoint term ``N^T W [grad^s(lambda) : w_q dC/drho : eps]``.

    For ``p == 1`` the fluctuation strain is masked to zero where
    ``rho_q <= 1e-4`` (experimental adjustment, see TODO in the code).

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        Discretization of the periodic unit cell.
    material_data_field_ijklqxyz : ndarray, shape [d, d, d, d, q, x, y(, z)]
        Base material data field ``C`` (without phase field applied).
    displacement_field_fnxyz : ndarray, shape [d, n, x, y(, z)]
        Displacement fluctuation of the equilibrium problem.
    macro_gradient_field_ijqxyz : ndarray, shape [d, d, q, x, y(, z)]
        Macroscopic strain at quadrature points.
    phase_field_1nxyz : ndarray, shape [1, n, x, y(, z)]
        Nodal phase field.
    target_stress_ij, actual_stress_ij : ndarray, shape [d, d]
        Target and homogenized stress.
    formulation : str
        ``'small_strain'`` or ``'finite_strain'`` (passed to the system matrix).
    p : int or float
        SIMP exponent.
    eta : float
        Phase-field interface-width parameter.
    weight : float
        Weight of the stress term.

    Returns
    -------
    ndarray, shape [1, n, x, y(, z)]
        Sensitivity ``df/drho``.

    Notes
    -----
    Legacy: depends on NumPy-array APIs of the discretization
    (``apply_gradient_operator*``, ``get_preconditioner``,
    ``apply_preconditioner``, tuple-returning ``evaluate_field_at_quad_points``)
    and ``solvers.PCG`` which no longer exist, and calls
    :func:`partial_der_of_double_well_potential_wrt_density_analytical` with
    an unsupported ``eta`` argument.
    """
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

    dmaterial_data_field_drho_ijklqxyz = material_data_field_ijklqxyz[..., :, :, :] * (
            p * np.power(phase_field_at_quad_poits_1qnxyz[0, :, 0, ...], (p - 1)))

    # compute strain field from to displacement and macro gradient
    strain_ijqxyz = discretization.apply_gradient_operator_symmetrized(displacement_field_fnxyz)
    if p == 1:
        # TODO CHANGE strain_ijqxyz field adjustment
        # For p == 1, zero the fluctuation strain where rho_q <= 1e-4 (almost void)
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
    # d/dρ(ρ^2 (1 - ρ)^2) = 2 ρ (2 ρ^2 - 3 ρ + 1)
    ddouble_well_drho_drho = partial_der_of_double_well_potential_wrt_density_analytical(discretization=discretization,
                                                                                         phase_field_1nxyz=phase_field_1nxyz,
                                                                                         eta=1)
    # sum of all parts of df_drho
    df_drho = weight * dfstress_drho + (dgradrho_drho * eta + ddouble_well_drho_drho / eta)

    # --------------------------------------
    # Solve adjoint problem ∂f/∂u=-∂g/∂u
    # Dt C D lambda = - 2/|omega| Dt: C : sigma_diff
    material_data_field_C_0_rho_ijklqxyz = material_data_field_ijklqxyz[..., :, :, :] * (
        np.power(phase_field_at_quad_poits_1qnxyz[0, :, 0, ...], (p)))
    # stress difference potential: rhs=-Dt*wA*E

    stress_difference_ijqxyz = discretization.get_gradient_size_field()
    stress_difference_ijqxyz[:, :, ...] = stress_difference_ij[
        (...,) + (np.newaxis,) * (stress_difference_ijqxyz.ndim - 2)]

    df_du_field = 2 * discretization.get_rhs(material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                                             macro_gradient_field_ijqxyz=stress_difference_ijqxyz) / discretization.cell.domain_volume  # minus sign is already there
    # Normalization
    df_du_field = df_du_field / np.sum(target_stress_ij ** 2)
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
    """
    Legacy/testing: full adjoint sensitivity returning the individual contributions.

    Same algorithm as :func:`sensitivity_with_adjoint_problem_FE_weights`, but
    with ``weight`` multiplying the phase-field terms,
    ``f = f_sigma + weight * (eta * int |grad rho|^2 + int rho^2 (1-rho)^2 / eta)``,
    and with the total strain ``E + grad^s u`` masked to zero wherever
    ``rho_q <= 0.01`` (experimental adjustment, see TODO in the code).

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        Discretization of the periodic unit cell.
    material_data_field_ijklqxyz : ndarray, shape [d, d, d, d, q, x, y(, z)]
        Base material data field ``C`` (without phase field applied).
    displacement_field_fnxyz : ndarray, shape [d, n, x, y(, z)]
        Displacement fluctuation of the equilibrium problem.
    macro_gradient_field_ijqxyz : ndarray, shape [d, d, q, x, y(, z)]
        Macroscopic strain at quadrature points.
    phase_field_1nxyz : ndarray, shape [1, n, x, y(, z)]
        Nodal phase field.
    target_stress_ij, actual_stress_ij : ndarray, shape [d, d]
        Target and homogenized stress.
    formulation : str
        ``'small_strain'`` or ``'finite_strain'``.
    p : int or float
        SIMP exponent.
    eta : float
        Phase-field interface-width parameter.
    weight : float
        Weight of the phase-field terms.

    Returns
    -------
    dict
        ``'dfstress_drho'``, ``'dgradrho_drho'``, ``'ddouble_well_drho_drho'``,
        ``'dg_drho_nxyz'`` (adjoint term) and ``'sensitivity'`` (total), all
        nodal arrays of shape [1, n, x, y(, z)].

    Notes
    -----
    Legacy: same API limitations as
    :func:`sensitivity_with_adjoint_problem_FE_weights`.
    """
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

    dmaterial_data_field_drho_ijklqxyz = material_data_field_ijklqxyz[..., :, :, :] * (
            p * np.power(phase_field_at_quad_poits_1qnxyz[0, :, 0, ...], (p - 1)))

    # compute strain field from to displacement and macro gradient
    strain_ijqxyz = discretization.apply_gradient_operator_symmetrized(displacement_field_fnxyz)
    strain_ijqxyz = macro_gradient_field_ijqxyz + strain_ijqxyz
    # TODO CHANGE strain_ijqxyz field adjustment
    # Mask the strain in (almost) void regions rho_q <= 0.01

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
    # d/dρ(ρ^2 (1 - ρ)^2) = 2 ρ (2 ρ^2 - 3 ρ + 1)
    ddouble_well_drho_drho = partial_der_of_double_well_potential_wrt_density_analytical(discretization=discretization,
                                                                                         phase_field_1nxyz=phase_field_1nxyz,
                                                                                         eta=1)
    # sum of all parts of df_drho
    df_drho = dfstress_drho + weight * (dgradrho_drho * eta + ddouble_well_drho_drho / eta)

    # --------------------------------------
    # Solve adjoint problem ∂f/∂u=-∂g/∂u
    # Dt C D lambda = - 2/|omega| Dt: C : sigma_diff
    material_data_field_C_0_rho_ijklqxyz = material_data_field_ijklqxyz[..., :, :, :] * (
        np.power(phase_field_at_quad_poits_1qnxyz[0, :, 0, ...], (p)))
    # stress difference potential: rhs=-Dt*wA*E

    stress_difference_ijqxyz = discretization.get_gradient_size_field()
    stress_difference_ijqxyz[:, :, ...] = stress_difference_ij[
        (...,) + (np.newaxis,) * (stress_difference_ijqxyz.ndim - 2)]

    df_du_field = 2 * discretization.get_rhs(material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                                             macro_gradient_field_ijqxyz=stress_difference_ijqxyz) / discretization.cell.domain_volume  # minus sign is already there

    df_du_field = df_du_field / np.sum(target_stress_ij ** 2)

    K_fun = lambda x: discretization.apply_system_matrix(material_data_field=material_data_field_C_0_rho_ijklqxyz,
                                                         displacement_field=x,
                                                         formulation=formulation)
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
    """
    Legacy/testing objective: stress equivalence + phase field (analytical double well).

    Computes ``f = f_sigma + w * (eta * int |grad rho|^2 + int rho^2 (1-rho)^2 / eta)``
    with ``f_sigma = |Sigma_h - Sigma_t|^2 / |Sigma_t|^2``; counterpart of
    :func:`sensitivity_with_adjoint_problem_FE_testing`.

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        2D discretization with linear triangles.
    actual_stress_ij, target_stress_ij : ndarray, shape [d, d]
        Homogenized and target stress.
    phase_field_1nxyz : muGrid Field, shape [1, 1, x, y]
        Nodal phase field.
    eta : float
        Phase-field interface-width parameter.
    w : float
        Weight of the phase-field terms.

    Returns
    -------
    float
        Objective value.

    Notes
    -----
    Legacy: uses ``discretization.apply_gradient_operator`` (no longer available).
    """
    # evaluate objective functions
    # f = (flux_h -flux_target)^2 + w*eta* int (  (grad(rho))^2 )dx  +    int ( rho^2(1-rho)^2 ) / eta   dx
    # f =  f_sigma + w*eta* f_rho_grad  + f_dw/eta

    # stress difference potential: actual_stress_ij is homogenized stress
    # stress_difference_ij = actual_stress_ij - target_stress_ij
    stress_difference_ij = (actual_stress_ij - target_stress_ij)

    f_sigma = np.sum(stress_difference_ij ** 2) / np.sum(target_stress_ij ** 2)

    # double - well potential
    f_dw = compute_double_well_potential_analytical(discretization=discretization,
                                                    phase_field_1nxyz=phase_field_1nxyz)

    phase_field_gradient = discretization.apply_gradient_operator(phase_field_1nxyz)
    f_rho_grad = np.sum(discretization.integrate_over_cell(phase_field_gradient ** 2))

    f_rho = eta * f_rho_grad + f_dw / eta

    return f_sigma + w * f_rho  # / discretization.cell.domain_volume


def material_interpolation_simp(phase_field_at_quad_poits_1qxyz,
                                output_material_data_field_rho_ijklqxyz,
                                p, C0_ijkl,
                                C1_ijkl, dim):
    """
    Populate the material data field with the SIMP interpolation.

    ``C(rho(x_q)) = (C1 - C0) * rho(x_q)^p + C0`` at every quadrature point.

    Parameters
    ----------
    phase_field_at_quad_poits_1qxyz : muGrid Field, shape [1, 1, q, x, y(, z)]
        Phase field interpolated to the quadrature points
        (e.g. by ``discretization.apply_N_operator_mugrid``).
    output_material_data_field_rho_ijklqxyz : muGrid Field, shape [d, d, d, d, q, x, y(, z)]
        Output material data field, overwritten in place.
    p : int or float
        SIMP penalisation exponent.
    C0_ijkl : ndarray, shape [d, d, d, d]
        Void (weak) material stiffness.
    C1_ijkl : ndarray, shape [d, d, d, d]
        Base (solid) material stiffness.
    dim : int
        Spatial dimension (2 or 3), used to broadcast the constant tensors
        over the ``dim + 1`` trailing axes ``(q, x, y[, z])``.

    Returns
    -------
    None
        The result is written into ``output_material_data_field_rho_ijklqxyz``.
    """

    # dim = 2 or 3 (number of spatial dimensions)
    # quad axis (q) + spatial axes (x, y[, z]) -> dim + 1 trailing axes

    expand = (...,) + (np.newaxis,) * (dim + 1)

    output_material_data_field_rho_ijklqxyz.s[...] = \
        (C1_ijkl - C0_ijkl)[expand] \
        * np.power(phase_field_at_quad_poits_1qxyz.s, p)[0, 0, :, ...] \
        + C0_ijkl[expand]


def dmaterial_interpolation_simp(phase_field_at_quad_poits_1qxyz,
                                 output_dmaterial_data_field_rho_ijklqxyz,
                                 p, C0_ijkl,
                                 C1_ijkl, dim):
    """
    Populate the derivative of the SIMP material data w.r.t. the phase field.

    ``dC/drho (x_q) = (C1 - C0) * p * rho(x_q)^(p - 1)`` at every quadrature
    point, i.e. the derivative of ``C(rho) = (C1 - C0) rho^p + C0``
    (see :func:`material_interpolation_simp`).

    Parameters
    ----------
    phase_field_at_quad_poits_1qxyz : muGrid Field, shape [1, 1, q, x, y(, z)]
        Phase field interpolated to the quadrature points.
    output_dmaterial_data_field_rho_ijklqxyz : muGrid Field, shape [d, d, d, d, q, x, y(, z)]
        Output field, overwritten in place.
    p : int or float
        SIMP penalisation exponent.
    C0_ijkl : ndarray, shape [d, d, d, d]
        Void (weak) material stiffness.
    C1_ijkl : ndarray, shape [d, d, d, d]
        Base (solid) material stiffness.
    dim : int
        Spatial dimension (2 or 3).

    Returns
    -------
    None
        The result is written into ``output_dmaterial_data_field_rho_ijklqxyz``.
    """

    # broadcast the constant [d,d,d,d] tensors over the trailing (q, x, y[, z]) axes
    expand = (...,) + (np.newaxis,) * (dim + 1)

    output_dmaterial_data_field_rho_ijklqxyz.s[...] = \
        (C1_ijkl - C0_ijkl)[expand] \
        * (p * np.power(phase_field_at_quad_poits_1qxyz.s, p - 1))[0, 0, :, ...]
