"""
Nonlinear solvers for muFFTTO.

Cell problems
    :func:`solve_finite_strain_newton_cg`, an incremental Newton--Raphson solver for (finite-strain) hyperelastic
    cell problems on a periodic muFFTTO discretisation. Each Newton step solves the linearised equilibrium equations
    with the preconditioned conjugate gradient solver :func:`muFFTTO.solvers.conjugate_gradients_mugrid`
    (Newton--Krylov / Newton-CG scheme).

Optimization (energy minimization, topology optimization)
    :func:`minimize` is one front end for all optimizers, with the same call and the same result fields:

    method         algorithm                           needs                         constraints                source
    -------------  ----------------------------------  ----------------------------  -------------------------  -------
    'l_bfgs'       limited-memory BFGS, line search    f, gradient                   box, box + <a,x> = V       NuMPI
    'mma'          method of moving asymptotes         f, gradient                   box, box + <a,x> <= / = V  muFFTTO
    'tr_newton'    trust-region Newton, Steihaug CG    f, gradient, Hessian-vector   box                        NuMPI

    :func:`mma_bounded` is the MMA implementation (also usable directly).

MMA
    Method of Moving Asymptotes (MMA, Svanberg 1987 / 2007) for box-constrained problems with an optional single
    linear (volume) constraint.

        min f(x)   subject to   lo <= x <= hi,   <a, x> <= V   (or = V)

    Same stopping test as NuMPI.Optimization.l_bfgs_bounded (inf-norm of the projected KKT gradient < gtol, or
    largest step < xtol). MPI: x, the gradient, the bounds and the constraint weights are local slices, the energy
    and the constraint target are global scalars (see the MPI contract of NuMPI.Optimization).

    Subproblem. The objective is replaced by the convex separable MMA approximation

        f~(x) = f(x_k) + sum_j  p_j / (U_j - x_j) + q_j / (x_j - L_j) - const,      alpha_j <= x_j <= beta_j

    * box only: closed-form minimizer  x_j = (sqrt(p_j) L_j + sqrt(q_j) U_j) / (sqrt(p_j) + sqrt(q_j)), clipped;
    * with the linear constraint (kept exact, not approximated): for a multiplier lam, every x_j minimizes
      f~_j(x_j) + lam a_j x_j, i.e. solves the monotone 1D equation p_j/(U_j-x_j)^2 - q_j/(x_j-L_j)^2 + lam a_j = 0
      (vectorized bisection). <a, x(lam)> is non-increasing in lam (a >= 0), so lam follows from a bisection on the
      dual; lam >= 0 for '<=' and free for '='. Every iterate satisfies the constraint (to the bisection tolerance).

    Asymptotes (Svanberg 2007): L, U = x -+ asyinit (hi - lo) in the first two iterations; afterwards they move away
    from x (factor asyincr) where x_j kept its direction and towards x (asydecr) where it oscillated.
    Move limit: |x_new - x| <= move (hi - lo).

    Plain MMA is not globally convergent (no line search). As in GCMMA, a step is accepted only when the
    approximation was conservative, f(x_new) <= f~(x_new) (this implies f(x_new) <= f(x_k)); otherwise it is retried
    with more curvature (raa x 10) and tighter asymptotes (closer to x).
"""
import numpy as np
from mpi4py import MPI

from muFFTTO import solvers

try:
    from scipy.optimize import OptimizeResult
except ModuleNotFoundError:  # NuMPI defines the same dict-like class
    from NuMPI.Optimization.Result import OptimizeResult


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


# ---------------------------------------------------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------------------------------------------------

METHODS = ('l_bfgs', 'mma', 'tr_newton')


def minimize(fun, x0, method='l_bfgs', bounds=None, constraint_weights=None, constraint_target=None,
             constraint_type='eq', hessp=None, precond=None, callback=None, comm=None, disp=False, **options):
    """
    Minimize f(x) subject to box bounds and an optional linear (volume) constraint, with one of several optimizers.

        min f(x)   subject to   lo <= x <= hi,   <a, x> = V   (or <a, x> <= V, method 'mma')

    Parameters
    ----------
    fun : callable
        ``fun(x)`` returns ``(f, gradient)``; f is the global (MPI-reduced) value, x and the gradient are the local
        slices (flat arrays).
    x0 : ndarray
        Initial guess (local slice; any shape, numpy or cupy for 'l_bfgs' / 'tr_newton'; passed on unchanged).
    method : {'l_bfgs', 'mma', 'tr_newton'}
        Optimizer, see the table in the module docstring.
    bounds : (lo, hi) or None
        Box bounds, scalars or local arrays; None (default): unbounded (not for 'mma').
    constraint_weights, constraint_target, constraint_type :
        Linear constraint <a, x> = V ('eq') or <a, x> <= V ('ineq'): a = constraint_weights (local slice, a >= 0),
        V = constraint_target (global). None: no constraint. 'l_bfgs' supports 'eq' (projected L-BFGS),
        'mma' both, 'tr_newton' none.
    hessp : callable, optional
        ``hessp(x, v)`` returns the Hessian (or a symmetric approximation) applied to v. Required by 'tr_newton'.
    precond : callable, optional
        ``precond(r)`` returns M^-1 r for 'tr_newton' (only without bounds, see NuMPI.Optimization.tr_newton_bounded).
    callback : callable, optional
        ``callback(x)`` after every accepted iterate.
    comm : MPI communicator, optional
        Default MPI.COMM_WORLD.
    disp : bool
        Print the iteration table of the optimizer.
    **options
        Passed unchanged to the optimizer (its own defaults apply otherwise), e.g. gtol, xtol, maxiter (all);
        maxcor, c1, max_halvings, zero_mask (l_bfgs); move, asyinit, max_retries (mma); delta0, delta_max,
        inner_tol, inner_stop, inner_maxiter, fun_error, request_accuracy, zero_mask (tr_newton).

    Returns
    -------
    OptimizeResult
        The optimizer's result (x, fun, jac, nit, success, message, max_grad, ...) plus, for every method, the
        fields ``method``, ``nfev`` (calls of fun) and ``nhessp`` (calls of hessp).
    """
    from NuMPI import Optimization
    from NuMPI.Tools import Reduction

    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose one of {METHODS}")
    comm = MPI.COMM_WORLD if comm is None else comm
    constrained = constraint_weights is not None
    if constraint_type not in ('eq', 'ineq'):
        raise ValueError("constraint_type must be 'eq' or 'ineq'")
    lo, hi = (None, None) if bounds is None else bounds

    # count the evaluations uniformly (the NuMPI optimizers do not all report them)
    counts = {'nfev': 0, 'nhessp': 0}

    def counted_fun(x, *args):
        counts['nfev'] += 1
        return fun(x)

    def counted_hessp(x, v, *args):
        counts['nhessp'] += 1
        return hessp(x, v)

    common = dict(callback=callback, disp=disp)
    if method == 'l_bfgs':
        if precond is not None or hessp is not None:
            raise ValueError("'l_bfgs' takes neither hessp nor precond")
        if constrained and constraint_type == 'ineq':
            raise ValueError("'l_bfgs' supports only an equality constraint; use method='mma' for <a, x> <= V")
        if constrained:
            constraint = Optimization.LinearConstraint(constraint_weights, constraint_target, pnp=Reduction(comm))
            result = Optimization.l_bfgs_projected(counted_fun, x0, constraint, jac=True, bounds_lo=lo, bounds_hi=hi,
                                                   comm=comm, **common, **options)
        else:
            result = Optimization.l_bfgs_bounded(counted_fun, x0, jac=True, bounds_lo=lo, bounds_hi=hi, comm=comm,
                                                 **common, **options)
    elif method == 'mma':
        if precond is not None or hessp is not None:
            raise ValueError("'mma' takes neither hessp nor precond")
        if bounds is None:
            raise ValueError("'mma' needs finite bounds (the asymptotes are scaled with hi - lo)")
        result = mma_bounded(counted_fun, x0, jac=True, bounds_lo=lo, bounds_hi=hi,
                             constraint_weights=constraint_weights, constraint_target=constraint_target,
                             constraint_type=constraint_type, comm=comm, **common, **options)
    else:  # tr_newton
        if hessp is None:
            raise ValueError("'tr_newton' needs hessp(x, v)")
        if constrained:
            raise ValueError("'tr_newton' supports box bounds only; use 'l_bfgs' or 'mma' with a linear constraint")
        result = Optimization.tr_newton_bounded(counted_fun, x0, counted_hessp, jac=True, bounds_lo=lo,
                                                bounds_hi=hi, precond=precond, comm=comm, **common, **options)
    result['method'] = method
    result['nfev'] = counts['nfev']
    result['nhessp'] = counts['nhessp']
    return result


def mma_bounded(fun, x0, args=(), jac=True, bounds_lo=0.0, bounds_hi=1.0, constraint_weights=None,
                constraint_target=None, constraint_type='ineq', gtol=1e-5, xtol=0.0, maxiter=500, move=0.2,
                asyinit=0.5, asyincr=1.2, asydecr=0.7, raa0=1e-5, accept_increase=False, max_retries=8,
                comm=None, callback=None, disp=False):
    """
    Parameters
    ----------
    constraint_weights : array_like, optional
        Local slice of the non-negative weights a of the linear constraint <a, x> (e.g. pixel volumes / total
        volume, then <a, x> is the volume fraction). None: no constraint (box only).
    constraint_target : float, optional
        Global right-hand side V.
    constraint_type : {'ineq', 'eq'}
        '<a, x> <= V' or '<a, x> = V'.
    Other parameters as NuMPI.Optimization.l_bfgs_bounded; move, asyinit, asyincr, asydecr, raa0 are the
    MMA parameters of Svanberg (2007).
    """
    comm = MPI.COMM_WORLD if comm is None else comm
    x = np.clip(np.asarray(x0, dtype=float).ravel().copy(), bounds_lo, bounds_hi)
    lo = np.broadcast_to(np.asarray(bounds_lo, dtype=float), x.shape)
    hi = np.broadcast_to(np.asarray(bounds_hi, dtype=float), x.shape)
    span = np.maximum(hi - lo, 1e-12)
    constrained = constraint_weights is not None
    if constrained:
        a = np.asarray(constraint_weights, dtype=float).ravel()
        if np.any(a < 0):
            raise ValueError('constraint_weights must be non-negative')
        if constraint_type not in ('ineq', 'eq'):
            raise ValueError("constraint_type must be 'ineq' or 'eq'")
        V = float(constraint_target)

    def sum_global(values):
        return comm.allreduce(float(np.sum(values)), op=MPI.SUM)

    def max_global(values):
        return comm.allreduce(float(np.max(values)) if values.size else 0.0, op=MPI.MAX)

    def projected_gradient_norm(x, g):
        # a variable at a bound with the gradient pushing it out of the box is not free
        free = ~(((x <= lo) & (g > 0)) | ((x >= hi) & (g < 0)))
        return max_global(np.abs(np.where(free, g, 0.0)))

    def evaluate(x):
        if jac is True:
            f, g = fun(x, *args)
        else:
            f, g = fun(x, *args), jac(x, *args)
        return float(f), np.asarray(g, dtype=float).ravel()

    def minimize_separable(p, q, L, U, alpha, beta, lam):
        """argmin of p/(U-x) + q/(x-L) + lam a x on [alpha, beta], element by element."""
        if not constrained or lam == 0.0:
            sp, sq = np.sqrt(p), np.sqrt(q)
            return np.clip((sp * L + sq * U) / (sp + sq), alpha, beta)

        def slope(x):  # increasing in x on (L, U)
            return p / (U - x) ** 2 - q / (x - L) ** 2 + lam * a

        left, right = alpha.copy(), beta.copy()
        for _ in range(50):
            mid = 0.5 * (left + right)
            positive = slope(mid) > 0
            right = np.where(positive, mid, right)
            left = np.where(positive, left, mid)
        # exactly on the bound where the minimum is there (the bisection only approaches it)
        return np.where(slope(alpha) >= 0, alpha, np.where(slope(beta) <= 0, beta, 0.5 * (left + right)))

    def solve_subproblem(p, q, L, U, alpha, beta):
        """Minimizer of the subproblem and the multiplier of the linear constraint."""
        x_free = minimize_separable(p, q, L, U, alpha, beta, 0.0)
        if not constrained:
            return x_free, 0.0

        lam = bisect_multiplier(lambda lam: sum_global(a * minimize_separable(p, q, L, U, alpha, beta, lam)),
                                max_global(np.abs(g)) / max(max_global(a), 1e-300))
        return minimize_separable(p, q, L, U, alpha, beta, lam), lam

    def bisect_multiplier(volume_of, lam_scale):
        """Root lam of volume_of(lam) - V (non-increasing in lam); lam >= 0 for 'ineq'."""
        excess_0 = volume_of(0.0) - V
        if constraint_type == 'ineq' and excess_0 <= 0:
            return 0.0
        # bracket the root: lam > 0 decreases <a, x>, lam < 0 increases it
        direction = 1.0 if excess_0 > 0 else -1.0
        lam_inner, lam_outer = 0.0, direction * lam_scale
        while direction * (volume_of(lam_outer) - V) > 0:
            lam_inner, lam_outer = lam_outer, 2.0 * lam_outer
            if abs(lam_outer) > 1e30:  # the box and the move limit do not allow <a, x> = V
                return lam_outer
        for _ in range(100):
            lam = 0.5 * (lam_inner + lam_outer)
            excess = volume_of(lam) - V
            if abs(excess) <= 1e-12 * max(abs(V), 1.0):
                return lam
            if direction * excess > 0:
                lam_inner = lam
            else:
                lam_outer = lam
            if abs(lam_outer - lam_inner) <= 1e-14 * abs(lam_outer):
                break
        return lam_outer

    if constrained:  # start on the constraint (Euclidean projection): feasibility is kept from here on
        x0_projected = x.copy()
        mu = bisect_multiplier(lambda mu: sum_global(a * np.clip(x0_projected - mu * a, lo, hi)),
                               max_global(span) / max(max_global(a), 1e-300))
        x = np.clip(x0_projected - mu * a, lo, hi)

    f, g = evaluate(x)
    nfev = 1
    x_old1 = x_old2 = None
    L = U = None
    raa = raa0

    def kkt_gradient(x, g):
        """Gradient of the Lagrangian g + lam a, lam estimated from the variables strictly inside the box."""
        if not constrained:
            return g, 0.0
        inside = (x > lo) & (x < hi)
        lam = -sum_global(np.where(inside, a * g, 0.0)) / max(sum_global(np.where(inside, a * a, 0.0)), 1e-300)
        if constraint_type == 'ineq':
            lam = max(lam, 0.0) if sum_global(a * x) >= V - 1e-9 * max(abs(V), 1.0) else 0.0
        return g + lam * a, lam

    def volume(x):
        return sum_global(a * x) if constrained else np.nan

    g_kkt, lam = kkt_gradient(x, g)
    history = {'fun': [f], 'max_grad': [projected_gradient_norm(x, g_kkt)],
               'constraint': [volume(x)], 'nfev': [nfev]}
    if disp and comm.rank == 0:
        print(f"{'it':<5} {'f':<14} {'max|pg|':<12} {'max|dx|':<10} {'<a,x>':<10} retries")
        print(f"{0:<5d} {f:<14.6e} {history['max_grad'][-1]:<12.4e} {'':<10} {history['constraint'][-1]:<10.5g}")

    def finish(success, nit, message):
        return OptimizeResult({'success': bool(success), 'x': x, 'fun': f, 'jac': g, 'nit': int(nit),
                               'nfev': int(nfev), 'message': message, 'max_grad': history['max_grad'][-1],
                               'multiplier': lam, 'history': history})

    for it in range(1, maxiter + 1):
        # --- asymptotes ---------------------------------------------------------------------
        if x_old2 is None:
            L = x - asyinit * span
            U = x + asyinit * span
        else:
            trend = (x - x_old1) * (x_old1 - x_old2)
            gamma = np.where(trend > 0, asyincr, np.where(trend < 0, asydecr, 1.0))
            L = x - gamma * (x_old1 - L)
            U = x + gamma * (U - x_old1)
            L = np.clip(L, x - 10.0 * span, x - 0.01 * span)
            U = np.clip(U, x + 0.01 * span, x + 10.0 * span)

        # --- convex separable approximation and its minimizer ---------------------------------
        raa = max(0.1 * raa, raa0)
        tighten = 1.0
        for retry in range(max_retries + 1):
            L_k = x - tighten * (x - L)
            U_k = x + tighten * (U - x)
            alpha = np.maximum.reduce([lo, L_k + 0.1 * (x - L_k), x - move * span])
            beta = np.minimum.reduce([hi, U_k - 0.1 * (U_k - x), x + move * span])
            g_plus, g_minus = np.maximum(g, 0.0), np.maximum(-g, 0.0)
            p = (U_k - x) ** 2 * (1.001 * g_plus + 0.001 * g_minus + raa / span)
            q = (x - L_k) ** 2 * (0.001 * g_plus + 1.001 * g_minus + raa / span)
            x_new, _ = solve_subproblem(p, q, L_k, U_k, alpha, beta)
            f_new, g_new = evaluate(x_new)
            nfev += 1
            # value of the approximation f~ at x_new; f~ <= f, so a conservative step also decreases f
            f_approx = f + sum_global(p * (1 / (U_k - x_new) - 1 / (U_k - x))
                                      + q * (1 / (x_new - L_k) - 1 / (x - L_k)))
            if accept_increase or f_new <= f_approx + 1e-14 * abs(f) or retry == max_retries:
                break
            # not conservative: more curvature (GCMMA) and asymptotes closer to x
            raa *= 10.0
            tighten *= 0.5

        step = max_global(np.abs(x_new - x))
        x_old2, x_old1 = x_old1, x.copy()
        L, U = L_k, U_k
        x, f, g = x_new, f_new, g_new
        g_kkt, lam = kkt_gradient(x, g)
        history['fun'].append(f)
        history['max_grad'].append(projected_gradient_norm(x, g_kkt))
        history['constraint'].append(volume(x))
        history['nfev'].append(nfev)
        if callback is not None:
            callback(x)
        if disp and comm.rank == 0:
            print(f"{it:<5d} {f:<14.6e} {history['max_grad'][-1]:<12.4e} {step:<10.3e} "
                  f"{history['constraint'][-1]:<10.5g} {retry}")
        if history['max_grad'][-1] < gtol:
            return finish(True, it, 'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_GTOL')
        if xtol > 0 and step < xtol:
            return finish(True, it, 'CONVERGENCE: MAX_STEP_<=_XTOL')
    return finish(False, maxiter, 'NO CONVERGENCE: MAXITERATIONS REACHED')
