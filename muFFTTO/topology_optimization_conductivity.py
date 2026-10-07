"""
Objective function and adjoint sensitivities for topology optimization of
(linear, steady-state) heat conductivity of periodic microstructures.

Problem setting
---------------
The design variable is a nodal phase field ``rho`` (``phase_field_1nxyz``),
interpolated to quadrature points with the FE interpolation operator ``N``.
The local conductivity follows a SIMP-like power-law interpolation::

    K(rho) = rho^p (K_base - K_void) + K_void,
    dK/drho = p rho^(p-1) (K_base - K_void).

For a prescribed macroscopic temperature gradient ``E`` the periodic
temperature fluctuation ``u`` solves the cell problem (equilibrium residual)::

    r(u, rho) = B^T W K(rho) (E + B u) = 0,

with ``B`` the discrete gradient operator and ``W`` the quadrature weights.
The homogenized (macroscopic) flux is
``q_h = 1/|Omega| int K(rho) (E + grad u) dOmega``.

Objective ("flux equivalence")::

    f_sigma = || q_target - q_h ||^2 / || q_target ||^2.

Sensitivities are computed with the adjoint method. With the Lagrangian
``L = w f_sigma + lambda^T r`` the adjoint field ``lambda`` solves::

    A lambda = - w df_sigma/du,         A = B^T W K(rho) B   (symmetric),

and the total derivative reads::

    dL/drho = w df_sigma/drho|_explicit + lambda^T dr/drho
            = w df_sigma/drho|_explicit + N^T W [grad(lambda) . dK/drho (E + grad u)].

The phase-field regularization (double-well + gradient terms) is identical to
the elasticity case; the corresponding functions are re-exported from
:mod:`muFFTTO.topology_optimization`.

Array naming convention
-----------------------
Suffixes encode the axis layout of the ``.s`` array of a muGrid field:
``i, j`` component/tensor axes (for a scalar temperature field ``i = 1``),
``n`` nodal point within a pixel, ``q`` quadrature point, ``x, y, z`` pixels,
``kl`` additional material-tensor axes (names such as ``*_ijkl`` are inherited
from the elasticity module; for conductivity the material tensor is ``[d, d]``).
"""
import warnings
import gc

import muGrid
import numpy as np
import scipy as sc
import time

from muFFTTO import domain
from muFFTTO import solvers

from NuMPI.Tools import Reduction
from mpi4py import MPI

# phase field part is identical to the elasticity part
from muFFTTO.topology_optimization import objective_function_phase_field
from muFFTTO.topology_optimization import sensitivity_phase_field_term_FE_NEW


def compute_flux_equivalence_potential(actual_flux_ij: np.ndarray,
                                       target_flux_ij: np.ndarray,
                                       disp: bool = False):
    """
    Evaluate the flux-equivalence objective (relative squared flux mismatch).

    .. math:: f_\\sigma = \\frac{\\|q_{target} - q_h\\|^2}{\\|q_{target}\\|^2}

    Parameters
    ----------
    actual_flux_ij : numpy.ndarray
        Homogenized flux of the current design ``q_h``, shape ``(1, d)`` (same
        shape as ``target_flux_ij``).
    target_flux_ij : numpy.ndarray
        Target homogenized flux ``q_target``; must not be identically zero.
    disp : bool, optional
        If ``True``, print the value on MPI rank 0. Default ``False``.

    Returns
    -------
    f_sigma : float
        Objective value (dimensionless, >= 0).

    Notes
    -----
    The inputs are already global (homogenized) quantities, so no MPI
    reduction is performed here.
    """
    # evaluate objective functions
    # f_sigma = (flux_target-flux_h)^2/flux_target^2

    # stress difference potential: actual_stress_ij is homogenized stress
    # stress_difference_ij = actual_stress_ij - target_stress_ij
    stress_difference_ij = target_flux_ij - actual_flux_ij

    f_sigma = np.sum(stress_difference_ij ** 2) / np.sum(target_flux_ij ** 2)
    if disp and MPI.COMM_WORLD.rank == 0:
        print('f_sigma = '          ' {} '.format(f_sigma))  # good in MPI
    return f_sigma


def adjoint_potential(discretization,
                      flux_field_ijqxyz,
                      adjoint_field_inxyz):
    """
    Evaluate the adjoint term ``g = lambda^T B^T W q`` of the Lagrangian.

    For the discrete equilibrium residual ``r = B^T W q`` (with
    ``q = K (E + B u)`` the flux at quadrature points) this computes the
    scalar product ``lambda . r``. At an exactly solved state ``r = 0`` and
    therefore ``g`` should vanish up to solver tolerance; it is mainly a
    diagnostic / Lagrangian contribution.

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        Discretization providing the gradient operator, quadrature weights,
        field collection and MPI reduction.
    flux_field_ijqxyz : muGrid.Field
        Flux field at quadrature points, ``.s`` of shape ``(1, d, q, x, y[, z])``.
    adjoint_field_inxyz : muGrid.Field
        Adjoint (Lagrange multiplier) nodal field ``lambda``, ``.s`` of shape
        ``(1, n, x, y[, z])``.

    Returns
    -------
    integral : float
        Global (MPI-summed) value of ``lambda^T B^T W q``.

    Notes
    -----
    Uses/overwrites the temporary nodal field
    ``'temporary_field_inxyz_in_adjoint_potential_temporary'`` of the field
    collection.
    """
    # g = (grad lambda, flux)
    # g = (grad lambda, C grad displacement)
    # g = (grad lambda, C grad displacement)  == lambda_transpose grad_transpose C grad u

    # Input: adjoint_field [f,n,x,y,z]
    #        stress_field  [d,d,q,x,y,z]

    # Output: g  [1] == 0
    # -- -- -- -- -- -- -- -- -- -- --
    # apply quadrature weights

    weights = discretization.quadrature_weights
    # apply B^transposed via the convolution operator "div (flux)"
    force_field_inxyz = discretization.get_scalar_field(
        name='temporary_field_inxyz_in_adjoint_potential_temporary')
    discretization.fft.communicate_ghosts(force_field_inxyz)

    discretization.gradient_op.transpose(quadrature_point_field=flux_field_ijqxyz,
                                     nodal_field=force_field_inxyz,
                                     weights=weights)

    # pointwise product lambda_un * f_un summed over the component axis u and
    # nodal axis n (here 'i' plays the role of n) -> field over (x, y[, z])
    adjoint_potential_field = np.einsum('ui...,ui...->...', adjoint_field_inxyz.s, force_field_inxyz.s)

    # Reductor_numpi = discretization.mpi_reduction(MPI.COMM_WORLD)
    # sum over local pixels and over all MPI ranks
    integral = discretization.mpi_reduction.sum(adjoint_potential_field)  #

    return integral


def sensitivity_flux_and_adjoint(discretization,
                                 base_material_data_ijkl,
                                 void_material_data_ijkl,
                                 displacement_field_inxyz,
                                 adjoint_field_inxyz,
                                 macro_gradient_field_ijqxyz,
                                 phase_field_1nxyz,
                                 target_flux_ij,
                                 actual_flux_ij,
                                 preconditioner_fun,
                                 system_matrix_fun, p,
                                 weight,
                                 formulation=None,
                                 disp=False,
                                 **kwargs):
    """
    Total sensitivity of the (weighted) flux-equivalence objective with respect
    to the nodal phase field, computed with the adjoint method.

    Steps
    -----
    1. Build ``K(rho)`` at quadrature points from the interpolated phase field.
    2. Explicit partial derivative ``df_sigma/drho`` (state ``u`` fixed), see
       :func:`partial_derivative_of_objective_function_flux_equivalence_wrt_phase_field`.
    3. Assemble the adjoint right-hand side ``-w df_sigma/du`` and solve
       ``A lambda = -w df_sigma/du`` with preconditioned CG (result written into
       ``adjoint_field_inxyz``).
    4. Adjoint contribution ``lambda^T dr/drho``, see
       :func:`partial_derivative_of_adjoint_potential_wrt_phase_field`.
    5. Evaluate the adjoint potential ``lambda^T r`` for diagnostics.

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        Discretization of the periodic cell (conductivity problem).
    base_material_data_ijkl : numpy.ndarray
        Conductivity tensor of the solid (base) phase, shape ``(d, d)``.
    void_material_data_ijkl : numpy.ndarray
        Conductivity tensor of the void (weak) phase, shape ``(d, d)``.
    displacement_field_inxyz : muGrid.Field
        Solved temperature fluctuation ``u`` (named "displacement" for
        consistency with the elasticity module), ``.s`` shape ``(1, n, x, y[, z])``.
    adjoint_field_inxyz : muGrid.Field
        Adjoint field ``lambda``; used as the initial guess of CG and
        overwritten in-place with the adjoint solution.
    macro_gradient_field_ijqxyz : muGrid.Field
        Macroscopic temperature gradient ``E`` at quadrature points,
        ``.s`` shape ``(1, d, q, x, y[, z])``.
    phase_field_1nxyz : muGrid.Field
        Nodal phase field (design variable) ``rho``, ``.s`` shape
        ``(1, n, x, y[, z])``.
    target_flux_ij : numpy.ndarray
        Target homogenized flux, shape ``(1, d)``.
    actual_flux_ij : numpy.ndarray
        Homogenized flux of the current design, shape ``(1, d)``.
    preconditioner_fun : callable
        Preconditioner application for CG, signature as expected by
        :func:`muFFTTO.solvers.conjugate_gradients_mugrid`.
    system_matrix_fun : callable
        Application of the system matrix ``A = B^T W K(rho) B`` (Hessian-vector
        product) for the current design. Because ``A`` is symmetric the same
        operator is used for the adjoint problem.
    p : int or float
        Exponent of the power-law material interpolation.
    weight : float
        Weight ``w`` of the flux-equivalence term in the total objective.
    formulation : str, optional
        Unused (kept for interface compatibility with the elasticity version).
    disp : bool, optional
        If ``True``, print the number of adjoint CG iterations on rank 0.
    **kwargs
        ``cg_tol`` (float, default ``1e-7``): CG tolerance;
        ``r_tol`` (bool, default ``True``): passed as ``rtol`` to the CG
        solver (whether the tolerance is relative).

    Returns
    -------
    sensitivity : numpy.ndarray
        ``w * df_sigma/drho|_explicit + lambda^T dr/drho``, nodal array of shape
        ``(1, n, x, y[, z])`` (a NumPy view ``.s`` of internal muGrid fields;
        copy it if it must survive further calls that reuse those field names).
    adjoint_field_inxyz : muGrid.Field
        The solved adjoint field (same object as the input).
    adjoint_energy : float
        ``lambda^T B^T W q`` (should be ~0 at equilibrium).
    info_adjoint_ : dict
        ``'residual_rz'``: list of preconditioned residual norms ``r.z`` per CG
        iteration (filled on rank 0 only); on rank 0 additionally
        ``'num_iteration_adjoint'``: number of CG iterations.

    Notes
    -----
    Phase-field regularization terms are *not* included; add
    :func:`sensitivity_phase_field_term_FE_NEW` separately.
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

    # dim = 2 or 3 (number of spatial dimensions)
    # quad axis (q) + spatial axes (x, y[, z]) -> dim + 1 trailing axes
    expand = (...,) + (np.newaxis,) * (dim + 1)

    # K(rho) = rho^p (K_base - K_void) + K_void at every quadrature point;
    # (d, d) material tensors are broadcast against the (q, x, y[, z]) field
    material_data_field_rho_ijklqxyz.s[...] = \
        (base_material_data_ijkl - void_material_data_ijkl)[expand] \
        * np.power(phase_field_at_quad_poits_1qxyz.s, p)[0, 0, :, ...] \
        + void_material_data_ijkl[expand]

    # d_stress_d_rho phase field gradient potential for a phase field without perturbation
    dstress_drho = partial_derivative_of_objective_function_flux_equivalence_wrt_phase_field(
        discretization=discretization,
        phase_field_1nxyz=phase_field_1nxyz,
        target_flux_ij=target_flux_ij,
        actual_flux_ij=actual_flux_ij,
        base_material_data_ij=base_material_data_ijkl,
        void_material_data_ij=void_material_data_ijkl,
        temperature_field_inxyz=displacement_field_inxyz,
        macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
        p=p)

    # Adjoint problem
    # compute strain field from to displacement and macro gradient
    # strain_fluctuation_ijqxyz = discretization.apply_gradient_operator_symmetrized(displacement_field_fnxyz)
    # stress_difference_ij = target_stress_ij - actual_stress_ij
    flux_difference_ij = target_flux_ij - actual_flux_ij

    # df_sigma/du = -2/(|Omega| ||q_t||^2) B^T W K(rho) (q_t - q_h):
    # the constant flux difference is spread to all quadrature points and
    # used in place of the macro gradient in the standard RHS routine
    flux_difference_ijqxyz = discretization.get_gradient_size_field(
        name='flux_field_in_sensitivity_flux_and_adjoint')
    discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=flux_difference_ij,
                                                   macro_gradient_field_ijqxyz=flux_difference_ijqxyz
                                                   )
    # minus sign is already there
    df_du_field = discretization.get_unknown_size_field(
        name='adjoint_problem_rhs_in_sensitivity_flux_and_adjoint')
    discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_rho_ijklqxyz,
                                  macro_gradient_field_ijqxyz=flux_difference_ijqxyz,
                                  rhs_inxyz=df_du_field)
    # minus sign is already there
    # (get_rhs_mugrid returns -B^T W K dq, so after multiplying by -2/|Omega| the
    #  field equals +2/|Omega| B^T W K dq = -df_sigma/du * ||q_t||^2, i.e. the
    #  adjoint RHS -df/du up to the normalization below)
    df_du_field.s[...] = -2 * df_du_field.s / discretization.cell.domain_volume
    # Normalization
    df_du_field.s[...] = weight * df_du_field.s / np.sum(target_flux_ij ** 2)

    info_adjoint_ = {}
    # if MPI.COMM_WORLD.rank == 0:
    norms_cg_adjoint = dict()
    norms_cg_adjoint['residual_rr'] = []
    norms_cg_adjoint['residual_rz'] = []

    def callback_adjoint(it, x, r, p, z, stop_crit_norm):
        """Record global residual norms r.r and r.z of each adjoint CG iteration (rank 0)."""
        # global norms_cg_mech
        norm_of_rr = discretization.communicator.sum(np.dot(r.ravel(), r.ravel()))
        norm_of_rz = discretization.communicator.sum(np.dot(r.ravel(), z.ravel()))
        if MPI.COMM_WORLD.rank == 0:
            norms_cg_adjoint['residual_rr'].append(norm_of_rr)
            norms_cg_adjoint['residual_rz'].append(norm_of_rz)

    # solve A lambda = -w df/du (A symmetric -> same operator as the forward problem)
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
    dadjoin_drho = discretization.get_scalar_field(name='dadjoin_drho_in_sensitivity_stress_and_adjoint_FE_NEW')
    dadjoin_drho = partial_derivative_of_adjoint_potential_wrt_phase_field(
        discretization=discretization,
        base_material_data_ijkl=base_material_data_ijkl,
        void_material_data_ijkl=void_material_data_ijkl,
        displacement_field_fnxyz=displacement_field_inxyz,
        macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
        phase_field_1nxyz=phase_field_1nxyz,
        adjoint_field_inxyz=adjoint_field_inxyz,
        output_field_inxyz=dadjoin_drho,
        p=p)

    flux_field_ijqxyz = discretization.get_gradient_size_field(
        name='flux_field_in_sensitivity_flux_and_adjoint_FE_NEW')

    discretization.get_flux_field_mugrid(
        material_data_field_ijqxyz=material_data_field_rho_ijklqxyz,
        temperature_field_inxyz=displacement_field_inxyz,
        macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
        output_flux_field_ijqxyz=flux_field_ijqxyz)

    # diagnostic: lambda^T r(u, rho), should be ~0 at the solved state
    adjoint_energy = adjoint_potential(
        discretization=discretization,
        flux_field_ijqxyz=flux_field_ijqxyz,
        adjoint_field_inxyz=adjoint_field_inxyz)

    return weight * dstress_drho.s + dadjoin_drho.s, adjoint_field_inxyz, adjoint_energy, info_adjoint_


def partial_derivative_of_objective_function_flux_equivalence_wrt_phase_field(discretization,
                                                                              base_material_data_ij: np.ndarray,
                                                                              void_material_data_ij: np.ndarray,
                                                                              temperature_field_inxyz,
                                                                              macro_gradient_field_ijqxyz,
                                                                              phase_field_1nxyz,
                                                                              target_flux_ij: np.ndarray,
                                                                              actual_flux_ij: np.ndarray,
                                                                              p: int):
    """
    Explicit partial derivative of the flux-equivalence objective with respect
    to the nodal phase field (temperature field held fixed).

    With ``q_h = 1/|Omega| int K(rho) (E + grad u)`` and
    ``f = ||q_t - q_h||^2 / ||q_t||^2``::

        df/drho_explicit = -2 / (|Omega| ||q_t||^2)
                           N^T W [ (q_t - q_h) . dK/drho (E + grad u) ]

    where ``N^T W`` maps the quadrature-point integrand to the nodes.

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        Discretization of the periodic cell.
    base_material_data_ij : numpy.ndarray
        Conductivity tensor of the base phase, shape ``(d, d)``.
    void_material_data_ij : numpy.ndarray
        Conductivity tensor of the void phase, shape ``(d, d)``.
    temperature_field_inxyz : muGrid.Field
        Temperature fluctuation ``u``, ``.s`` shape ``(1, n, x, y[, z])``.
    macro_gradient_field_ijqxyz : muGrid.Field
        Macroscopic gradient ``E`` at quadrature points, ``(1, d, q, x, y[, z])``.
    phase_field_1nxyz : muGrid.Field
        Nodal phase field ``rho``, ``(1, n, x, y[, z])``.
    target_flux_ij : numpy.ndarray
        Target homogenized flux, shape ``(1, d)``.
    actual_flux_ij : numpy.ndarray
        Current homogenized flux, shape ``(1, d)``.
    p : int
        Exponent of the power-law interpolation.

    Returns
    -------
    dfflux_drho : muGrid.Field
        Nodal scalar field (``.s`` shape ``(1, n, x, y[, z])``) named
        ``'dfflux_drho_output'``; overwritten on every call.

    Notes
    -----
    The ``weight`` of the objective is *not* applied here (it is applied in
    :func:`sensitivity_flux_and_adjoint`).
    """
    # Input: phase_field [1,n,x,y,z]
    #        material_data_field [d,d,q,x,y,z] - conductivity
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

    # I consider polynomial interpolation of material
    # C_ij= rho**(p) (C_base_ij-C_void_ij) + C_void_ij
    # ∂ C_ijkl/ ∂ rho = p*rho**(p-1) (C_base_ij-C_void_ij)
    expand = (...,) + (np.newaxis,) * (dim + 1)
    dmaterial_data_field_drho_ijklqxyz_FE.s[...] = \
        (base_material_data_ij - void_material_data_ij)[expand] \
        * (p * np.power(phase_field_at_quad_poits_1qxyz.s, p - 1))[0, 0, :, ...]

    # compute gradient of temperature field from to temperature and macro gradient
    flux_ijqxyz = discretization.get_gradient_size_field(name='temperature_ijqxyz_local_at_pdofsewpf')
    discretization.apply_gradient_operator_mugrid(u_inxyz=temperature_field_inxyz,
                                                  grad_u_ijqxyz=flux_ijqxyz)
    flux_ijqxyz.s[...] = macro_gradient_field_ijqxyz.s + flux_ijqxyz.s

    # compute heat flux field
    # dq_ui = dK_ij/drho * (E + grad u)_uj  (u = 1 for a scalar temperature field)
    flux_ijqxyz.s[...] = np.einsum('ij...,uj...->ui...', dmaterial_data_field_drho_ijklqxyz_FE.s,
                                   flux_ijqxyz.s)
    # this is actually flux

    # flux difference
    flux_difference_ij = target_flux_ij - actual_flux_ij

    # pointwise scalar (q_t - q_h) . dq/drho at every quadrature point
    double_contraction_flux_qxyz_FE = discretization.get_quad_field_scalar(name='temp_at_quads')
    double_contraction_flux_qxyz_FE.s[0, 0] = np.einsum('uj,ujqxy...->qxy...',
                                                        flux_difference_ij,
                                                        flux_ijqxyz.s)
    # integrate against the nodal shape functions: N^T W (...) -> nodal field
    dfflux_drho = discretization.get_scalar_field(name='dfflux_drho_output')
    discretization.apply_N_transposed_operator_mugrid(
        quad_field_ijqxyz=double_contraction_flux_qxyz_FE,
        nodal_field_inxyz=dfflux_drho,
        apply_weights=True)
    # chain rule of ||q_t - q_h||^2 (factor -2), homogenization 1/|Omega| and normalization
    dfflux_drho.s[...] = -2 * dfflux_drho.s / discretization.cell.domain_volume / np.sum(target_flux_ij ** 2)
    return dfflux_drho


def partial_derivative_of_adjoint_potential_wrt_phase_field(discretization,
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
    Adjoint contribution ``lambda^T dr/drho`` to the total sensitivity.

    For the residual ``r = B^T W K(rho) (E + B u)``::

        lambda^T dr/drho = N^T W [ grad(lambda) . dK/drho (E + grad u) ]

    evaluated at quadrature points and mapped to the nodes with the weighted
    transposed interpolation operator.

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        Discretization of the periodic cell.
    base_material_data_ijkl : numpy.ndarray
        Conductivity tensor of the base phase, shape ``(d, d)``.
    void_material_data_ijkl : numpy.ndarray
        Conductivity tensor of the void phase, shape ``(d, d)``.
    displacement_field_fnxyz : muGrid.Field
        Temperature fluctuation ``u``, ``.s`` shape ``(1, n, x, y[, z])``.
    macro_gradient_field_ijqxyz : muGrid.Field
        Macroscopic gradient ``E`` at quadrature points, ``(1, d, q, x, y[, z])``.
    phase_field_1nxyz : muGrid.Field
        Nodal phase field ``rho``.
    adjoint_field_inxyz : muGrid.Field
        Solved adjoint field ``lambda``, ``(1, n, x, y[, z])``.
    output_field_inxyz : muGrid.Field
        Nodal scalar field, zeroed and overwritten in-place with the result.
    p : int or float, optional
        Exponent of the power-law interpolation. Default 1.

    Returns
    -------
    output_field_inxyz : muGrid.Field
        The same object as the ``output_field_inxyz`` argument.
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
    # (outdated note from the linear-interpolation version; the code below uses
    #  the power-law interpolation documented a few lines further down)
    # I consider linear interpolation of material  C_ijkl= p*rho**(p-1) C^0_ijkl
    # so  ∂ C_ijkl/ ∂ rho = 1* C^0_ijkl
    # Gradient of material data with respect to phase field
    phase_field_at_quad_poits_1qxyz = discretization.get_quad_field_scalar(
        name='phase_field_at_quads_reusable')
    discretization.apply_N_operator_mugrid(phase_field_1nxyz, phase_field_at_quad_poits_1qxyz)

    dmaterial_data_field_drho_ijklqxyz = discretization.get_material_data_size_field_mugrid(
        name='data_field_in_sensitivity_reusable')

    # I consider polynomial interpolation of material
    # C_ij= rho**(p) (C_base_ij-C_void_ij) + C_void_ij
    # ∂ C_ijkl/ ∂ rho = p*rho**(p-1) (C_base_ij-C_void_ij)
    expand = (...,) + (np.newaxis,) * (dim + 1)

    dmaterial_data_field_drho_ijklqxyz.s[...] = \
        (base_material_data_ijkl - void_material_data_ijkl)[expand] \
        * (p * np.power(phase_field_at_quad_poits_1qxyz.s[0, 0, :, ...], p - 1))

    # compute strain field from to displacement and macro gradient
    # (here: "flux sensitivity" dK/drho (E + grad u) at quadrature points)
    flux_ijqxyz = discretization.get_gradient_size_field(name='flux_ijqxyz_local_at_pdapwpf')

    discretization.get_flux_field_mugrid(
        material_data_field_ijqxyz=dmaterial_data_field_drho_ijklqxyz,
        temperature_field_inxyz=displacement_field_fnxyz,
        macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
        output_flux_field_ijqxyz=flux_ijqxyz)

    # local_stress=np.einsum('ijkl,lk->ij',material_data_field_ijklqxyz[...,0,0,0] , strain_ijqxyz[...,0,0,0])
    # apply quadrature weights
    # discretization.apply_quadrature_weights_on_gradient_field_mugrid(strain_ijqxyz)

    # gradient of adjoint_field
    adjoint_field_gradient_ijqxyz = discretization.get_gradient_of_scalar_field(
        name='adjoint_field_gradient_ijqxyz__pdapwpf')
    discretization.apply_gradient_operator_mugrid(u_inxyz=adjoint_field_inxyz,
                                                  grad_u_ijqxyz=adjoint_field_gradient_ijqxyz)

    # pointwise scalar grad(lambda) . dq/drho, summed over both component axes
    double_contraction_stress_qxyz = discretization.get_quad_field_scalar(name='double_contraction_stress_qxyz')
    double_contraction_stress_qxyz.s[0, 0] = np.einsum('ij...,ij...->...',
                                                       adjoint_field_gradient_ijqxyz.s,
                                                       flux_ijqxyz.s)

    output_field_inxyz.s.fill(0)
    discretization.fft.communicate_ghosts(double_contraction_stress_qxyz)

    # N^T W (...) maps the quadrature integrand to nodal sensitivities
    discretization.apply_N_transposed_operator_mugrid(
        quad_field_ijqxyz=double_contraction_stress_qxyz,
        nodal_field_inxyz=output_field_inxyz,
        apply_weights=True)

    return output_field_inxyz
