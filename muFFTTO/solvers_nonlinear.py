"""
Nonlinear solvers for muFFTTO.

Currently contains :func:`solve_finite_strain_newton_cg`, an incremental
Newton--Raphson solver for (finite-strain) hyperelastic cell problems on a
periodic muFFTTO discretisation. Each Newton step solves the linearised
equilibrium equations with the preconditioned conjugate gradient solver
:func:`muFFTTO.solvers.conjugate_gradients_mugrid` (Newton--Krylov / Newton-CG
scheme).
"""
import numpy as np

from muFFTTO import solvers

def solve_finite_strain_newton_cg(
        discretization,
        material,
        macro_gradient_ij,
        ninc=1,
        newton_tol=1e-8,
        newton_atol=1e-12,
        newton_max_iter=50,
        cg_tol=1e-6,
        cg_max_iter=10000,
        preconditioner_type='Green',
        reference_material_data_ijkl=None,
        formulation='finite_strain',
        verbose=True,
):
    """
    Newton-CG solver for finite strain hyperelasticity.

    Solves the nonlinear equilibrium problem:
        find u_fluc such that:  div( P(F) ) = 0
    where F = I + grad(macro_gradient) + grad(u_fluc)

    The outer loop is Newton's method.
    The inner loop is a preconditioned conjugate gradient (CG) solver
    that solves the linearized system at each Newton step:
        K(u) * du = R(u),   R(u) = -B^T W P(F(u))
    i.e. R is the out-of-balance force (the negative internal force; since
    B^T W is the discrete -div, R is the discrete +div P).

    Parameters
    ----------
    discretization : muFFTTO discretization object
        Provides field allocation, gradient / transposed-gradient operators,
        system-matrix application, preconditioners, the field collection and
        the MPI communicator.
    material : MaterialModelElasticity instance (e.g. NeoHookean)
        Must provide ``get_stress(grad_field, stress_field)`` and
        ``get_algorithmic_tangent(grad_field, tangent_field)``, both writing
        into the output fields in place. The input is the accumulated
        ``total_strain_field``, i.e. the deformation gradient
        ``F = I + H + grad(u_fluc)``, reset to ``I`` at the start.
    macro_gradient_ij : np.ndarray, shape (dim, dim)
        Prescribed macroscopic (displacement) gradient applied over all
        increments.
    ninc : int, optional
        Number of equal load increments; ``macro_gradient_ij / ninc`` is
        added at the start of each. Default 1.
    newton_tol : float, optional
        Newton convergence tolerance on ``||R|| / ||R_0||`` (Euclidean norm
        of the nodal residual, relative to its value at the start of the
        increment). Default 1e-8.
    newton_atol : float, optional
        Absolute floor on ``||R||``: if the residual is already below it,
        the state is accepted without a linear solve. This handles cells
        that are (numerically) in equilibrium, e.g. a homogeneous material,
        where ``R`` is pure round-off and a relative CG solve on it would be
        meaningless. Default 1e-12.
    newton_max_iter : int, optional
        Maximum Newton iterations per increment. Default 50.
    cg_tol : float, optional
        *Relative* CG tolerance (CG is called with ``rtol=True``, i.e. it
        stops when ``||r_k|| <= cg_tol * ||r_0||``). Default 1e-6.
    cg_max_iter : int, optional
        Maximum CG iterations per Newton step. Default 10000.
    preconditioner_type : str, optional
        ``'Green'`` (Fourier-space Green-operator preconditioner of the
        reference material), ``'Green_Jacobi'`` (Green preconditioner
        symmetrically scaled by the Jacobi factors of the current tangent);
        any other value gives the identity (no preconditioning).
        Default ``'Green'``.
    reference_material_data_ijkl : np.ndarray, shape (dim, dim, dim, dim), optional
        Reference stiffness for the Green preconditioner. Defaults to the
        symmetric fourth-order identity ``I4s``.
    formulation : str, optional
        Passed to the system-matrix application and the Jacobi
        preconditioner assembly. Default ``'finite_strain'``.
    verbose : bool, optional
        Print convergence info on rank 0. Default True.

    Returns
    -------
    results : dict
        ``'newton_residuals_per_increment'`` : list (per increment) of lists
        of ``||R||`` (initial value followed by one entry per Newton step);
        ``'cg_iterations_per_newton_step'`` : list of CG iteration counts
        (all increments concatenated);
        ``'total_newton_iterations'``, ``'total_cg_iterations'`` : int;
        ``'displacement_fluctuation_field'``, ``'total_strain_field'``,
        ``'stress_field'``, ``'tangent_field'`` : the muGrid fields holding
        the converged state.

    Notes
    -----
    Classical (full) Newton--Raphson with the consistent algorithmic tangent
    and no line search; the linear solves are inexact (relative CG
    tolerance), i.e. an inexact Newton / Newton--Krylov method. The
    convergence test requires at least two Newton iterations per increment
    (``iiter > 0``). If Newton does not converge, a warning is printed and the
    next increment starts from the unconverged state.
    """

    dim = discretization.domain_dimension

    # ------------------------------------------------------------------
    # allocate fields
    # ------------------------------------------------------------------
    macro_gradient_inc_field       = discretization.get_gradient_size_field(
                                         name='macro_gradient_inc_field')
    displacement_fluctuation_field = discretization.get_unknown_size_field(
                                         name='displacement_fluctuation_field')
    displacement_increment_field   = discretization.get_unknown_size_field(
                                         name='displacement_increment_field')
    strain_fluc_field              = discretization.get_displacement_gradient_sized_field(
                                         name='strain_fluctuation_field')
    total_strain_field             = discretization.get_displacement_gradient_sized_field(
                                         name='total_strain_field')
    stress_field                   = discretization.get_displacement_gradient_sized_field(
                                         name='stress_field')
    tangent_field                  = discretization.get_material_data_size_field_mugrid(
                                         name='tangent_field')
    rhs_field                      = discretization.get_unknown_size_field(
                                         name='rhs_field')
    energy_field                   = discretization.get_quad_field_scalar(
                                         name='energy_field')

    # ------------------------------------------------------------------
    # macroscopic gradient increment
    # ------------------------------------------------------------------
    macro_gradient_inc_ij = macro_gradient_ij / float(ninc)
    discretization.get_macro_gradient_field_mugrid(
        macro_gradient_ij=macro_gradient_inc_ij,
        macro_gradient_field_ijqxyz=macro_gradient_inc_field
    )

    # ------------------------------------------------------------------
    # identity tensors for reference material and preconditioner
    # ------------------------------------------------------------------
    if reference_material_data_ijkl is None:
        i = np.eye(dim)
        I4s = 0.5 * (np.einsum('il,jk->ijkl', i, i)
                     + np.einsum('ik,jl->ijkl', i, i))
        reference_material_data_ijkl = I4s

    # ------------------------------------------------------------------
    # build Green preconditioner (displacement-independent, built once)
    # ------------------------------------------------------------------
    preconditioner = discretization.get_preconditioner_Green_mugrid(
        reference_material_data_ijkl=reference_material_data_ijkl
    )

    # ------------------------------------------------------------------
    # helper: apply preconditioner
    # ------------------------------------------------------------------
    def apply_green_preconditioner(x, Px):
        # Px = M_Green^{-1} x, applied in Fourier space; ghosts of the input
        # are refreshed first so that the nodal stencil sees periodic data
        discretization.fft.communicate_ghosts(x)
        discretization.apply_preconditioner_mugrid(
            preconditioner_Fourier_fnfnqks=preconditioner,
            input_nodal_field_fnxyz=x,
            output_nodal_field_fnxyz=Px
        )

    def apply_green_jacobi_preconditioner(x, Px):
        # Px = D M_Green^{-1} D x with the Jacobi scaling D = K_diag computed
        # from the current tangent. NOTE: K_diag (and the scratch field) are
        # recomputed at every application, i.e. in every CG iteration.
        K_diag = discretization.get_preconditioner_Jacobi_mugrid(
            material_data_field_ijklqxyz=tangent_field,
            formulation=formulation
        )
        x_scaled = discretization.get_unknown_size_field(name='x_scaled')
        x_scaled.s[...] = K_diag.s * x.s
        discretization.fft.communicate_ghosts(x_scaled)
        discretization.apply_preconditioner_mugrid(
            preconditioner_Fourier_fnfnqks=preconditioner,
            input_nodal_field_fnxyz=x_scaled,
            output_nodal_field_fnxyz=Px
        )
        Px.s[...] = K_diag.s * Px.s
        discretization.fft.communicate_ghosts(Px)

    if preconditioner_type == 'Green':
        M_fun = apply_green_preconditioner
    elif preconditioner_type == 'Green_Jacobi':
        M_fun = apply_green_jacobi_preconditioner
    else:
        # no preconditioning: identity copy
        def M_fun(x, Px):
            Px.s[...] = x.s

    # ------------------------------------------------------------------
    # helper: apply system matrix  K * x -> Ax
    # ------------------------------------------------------------------
    def K_fun(x, Ax):
        # Ax = B^T C_tangent B x  (linearised equilibrium operator with the
        # current algorithmic tangent), ghosts refreshed afterwards
        discretization.apply_system_matrix_mugrid(
            material_data_field=tangent_field,
            input_field_inxyz=x,
            output_field_inxyz=Ax,
            formulation=formulation
        )
        discretization.fft.communicate_ghosts(Ax)

    # ------------------------------------------------------------------
    # helper: compute residual R = -B^T W P  (discrete div P) from current total strain
    # ------------------------------------------------------------------
    def compute_residual():
        # Evaluates stress and algorithmic tangent at the current total
        # gradient (the tangent is thus updated as a side effect), then
        # rhs = -B^T W P. B^T W is the discrete -div, so rhs is the discrete
        # +div P (out-of-balance force); it is the right-hand side of K du = rhs.
        material.get_stress(total_strain_field, stress_field)
        material.get_algorithmic_tangent(total_strain_field, tangent_field)
        # CG needs a symmetric K, i.e. a tangent with major symmetry A_ijkl = A_klij
        # (true for hyperelastic materials); one pass over the data per Newton step
        discretization.assert_material_symmetry(tangent_field, name='algorithmic tangent')
        discretization.fft.communicate_ghosts(stress_field)
        discretization.apply_gradient_transposed_operator_mugrid(
            gradient_field_ijqxyz=stress_field,
            div_u_fnxyz=rhs_field,
            apply_weights=True
        )
        rhs_field.s[...] *= -1

    # ------------------------------------------------------------------
    # helper: MPI norm
    # ------------------------------------------------------------------
    def mpi_norm(field):
        # global Euclidean norm of a field (sum of squares reduced over ranks)
        return np.sqrt(
            discretization.communicator.sum(
                np.dot(field.s.ravel(), field.s.ravel())
            )
        )

    # ------------------------------------------------------------------
    # tracking
    # ------------------------------------------------------------------
    results = {
        'newton_residuals_per_increment' : [],
        'cg_iterations_per_newton_step'  : [],
        'total_newton_iterations'        : 0,
        'total_cg_iterations'            : 0,
        'displacement_fluctuation_field' : displacement_fluctuation_field,
        'total_strain_field'             : total_strain_field,
        'stress_field'                   : stress_field,
        'tangent_field'                  : tangent_field,
    }

    # ------------------------------------------------------------------
    # initial state: undeformed configuration F = I, u_fluc = 0
    # ------------------------------------------------------------------
    # The fields are looked up by name in the field collection, so they may
    # still hold values from an earlier call; reset them explicitly.
    displacement_fluctuation_field.s[...] = 0.0
    total_strain_field.s[...] = 0.0
    for d in range(dim):
        total_strain_field.s[d, d] = 1.0

    # ------------------------------------------------------------------
    # incremental loading loop
    # ------------------------------------------------------------------
    for inc in range(ninc):
        if verbose and discretization.communicator.rank == 0:
            print(f'\nIncrement {inc + 1} / {ninc}')
            print('=' * 60)

        # apply macroscopic strain increment
        total_strain_field.s[...] += macro_gradient_inc_field.s[...]

        # compute initial residual for this increment
        compute_residual()
        norm_rhs_0 = mpi_norm(rhs_field)

        if verbose and discretization.communicator.rank == 0:
            print(f'  Initial residual: {norm_rhs_0:.4e}')

        newton_residuals = [norm_rhs_0]

        # --------------------------------------------------------------
        # Newton loop
        # --------------------------------------------------------------
        for iiter in range(newton_max_iter):

            # already in equilibrium (up to round-off): nothing to solve
            if newton_residuals[-1] <= newton_atol:
                if verbose and discretization.communicator.rank == 0:
                    print(f'  Newton converged in {iiter} iterations '
                          f'(||R|| <= newton_atol).')
                break

            # solve linearised system:  K * du = R   (R = rhs_field = -B^T W P)
            # (zero initial guess; CG tolerance relative to ||R||)
            displacement_increment_field.s.fill(0.0)

            cg_iter_count = [0]

            # the callback only records the last CG iteration index; it is
            # stored in a list so the closure can mutate it. If CG returns
            # before the loop (initial residual below tol) the count stays 0.
            def cg_callback(it, x, r, p, z, stop_crit_norm):
                cg_iter_count[0] = it

            solvers.conjugate_gradients_mugrid(
                comm=discretization.communicator,
                fc=discretization.field_collection,
                hessp=K_fun,
                b=rhs_field,
                x=displacement_increment_field,
                P=M_fun,
                tol=cg_tol,
                maxiter=cg_max_iter,
                callback=cg_callback,
                rtol=True,
            )

            results['cg_iterations_per_newton_step'].append(cg_iter_count[0])
            results['total_cg_iterations'] += cg_iter_count[0]

            # update displacement and strain:
            # H <- H + grad(du),  u_fluc <- u_fluc + du  (full Newton step, no line search)
            discretization.apply_gradient_operator_mugrid(
                u_inxyz=displacement_increment_field,
                grad_u_ijqxyz=strain_fluc_field
            )
            total_strain_field.s[...]             += strain_fluc_field.s[...]
            displacement_fluctuation_field.s[...] += displacement_increment_field.s[...]

            # recompute residual (also updates the tangent for the next step)
            compute_residual()
            norm_rhs = mpi_norm(rhs_field)
            newton_residuals.append(norm_rhs)

            results['total_newton_iterations'] += 1

            if verbose and discretization.communicator.rank == 0:
                print(f'  Newton it {iiter + 1:3d} | '
                      f'||R|| = {norm_rhs:.4e} | '
                      f'||R||/||R_0|| = {norm_rhs / (norm_rhs_0 + 1e-30):.4e} | '
                      f'CG its = {cg_iter_count[0]}')

            # Newton convergence check: relative residual, at least 2 iterations
            # (1e-30 avoids division by zero for a vanishing initial residual)
            if norm_rhs / (norm_rhs_0 + 1e-30) < newton_tol and iiter > 0:
                if verbose and discretization.communicator.rank == 0:
                    print(f'  Newton converged in {iiter + 1} iterations.')
                break

        else:
            # for-else: reached only if the Newton loop finished without `break`
            if verbose and discretization.communicator.rank == 0:
                print(f'  WARNING: Newton did not converge in {newton_max_iter} iterations.')

        results['newton_residuals_per_increment'].append(newton_residuals)

    return results