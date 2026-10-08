"""
Linear (and simple first-order) iterative solvers used throughout muFFTTO.

The module contains matrix-free Krylov solvers for the symmetric positive
definite systems that arise from the FFT/FEM discretisation of periodic cell
problems (homogenisation) and their adjoints (topology optimisation):

* :func:`conjugate_gradients_mugrid` -- preconditioned conjugate gradients
  (PCG) operating directly on muGrid fields, MPI-parallel through a muGrid
  ``Communicator``. This is the production solver.
* :func:`conjugate_gradients_mugrid_experimental` -- the same PCG with
  additional a-posteriori energy-norm error estimates (Hestenes--Stiefel
  lower bound with adaptive delay, Gauss--Radau upper bound) and selectable
  stopping criteria. Used for studying stopping criteria.
* :func:`___PCG` -- legacy PCG on plain NumPy arrays (kept for reference).
* :func:`Richardson` -- (preconditioned) Richardson iteration on NumPy arrays.
* :func:`dr_pbcg_mugrid` -- preconditioned block CG with several right-hand
  sides (DR-PBCG, Meurant & Tichy).
* :func:`adam` / :func:`update_parameters_with_adam` -- the Adam
  first-order optimiser (Kingma & Ba, 2015) with MPI-aware convergence checks.

Helper functions: :func:`donothing` (default no-op callback),
:func:`findS` (safety factor of the adaptive-delay error estimator) and
:func:`scalar_product_mpi` (global Euclidean inner product of distributed
NumPy arrays).

Conventions
-----------
For the muGrid-based solvers, the operator and preconditioner are given as
*in-place* callables ``hessp(x, Ax)`` and ``P(r, z)`` that read the input
field and write the result into the (preallocated) output field. All inner
products are local ``np.dot`` products of the ``.s`` (sub-point) arrays,
summed over MPI ranks with ``comm.sum`` -- so every rank obtains the same
global scalar and takes the same control-flow decisions.
"""
import numpy as np
import warnings

from NuMPI.Tools import Reduction
from mpi4py import MPI
from muGrid import Communicator, Field, GlobalFieldCollection


def donothing(*args, **kwargs):
    """
    No-op function accepting any arguments.

    Used as the default ``callback`` of the legacy NumPy solvers so that the
    callback can be invoked unconditionally.
    """
    pass


def findS(curve, Delta, l):
    """
    Safety factor ``S`` of the adaptive-delay energy-error estimator.

    Part of the adaptive choice of the delay ``d`` in the Hestenes--Stiefel
    lower bound on the A-norm (energy) error of CG, see
    G. Meurant, J. Papez, P. Tichy, "Accurate error estimation in CG",
    Numer. Algorithms 88 (2021). The estimate of ``||x - x_l||_A^2`` with
    delay ``d = k - l`` is accepted when ``S * Delta_k / sum_{j=l}^{k-1}
    Delta_j <= tau``; ``S`` guards against an overly optimistic estimate
    during phases of stagnation.

    Parameters
    ----------
    curve : list of float, length k+1
        ``curve[j] = sum_{i=j}^{k} Delta_i``, i.e. the current (delay ``k-j``)
        estimate of the squared energy error at iteration ``j``.
    Delta : list of float, length k+1
        ``Delta[j] = alpha_j * (r_j, z_j)`` -- the CG step contributions,
        ``Delta_j = ||x_{j+1} - x_j||_A^2``.
    l : int
        Index of the iteration whose error is currently being estimated.

    Returns
    -------
    S : float
        ``max_j curve[j] / Delta[j]`` over ``j`` from the last index at which
        ``curve`` was at least 1e4 times larger than ``curve[l]`` (or from 0
        if no such index exists) up to, but excluding, the last entry.

    Notes
    -----
    The inputs are global scalars (identical on every rank), so no MPI
    reduction is needed.
    """
    # function to compute safety factor S
    curve = np.array(curve)
    # indices j where the estimate at j is (much) larger than the one at l:
    # curve[l] / curve[j] <= 1e-4  <=>  curve[j] >= 1e4 * curve[l]
    ind = np.where((curve[l] / curve) <= 1e-4)[0]  # , 1, 'last')
    # MATLAB-style find(..., 1, 'last'): take the last such index (0 if none)
    last_index = ind[-1] if ind.size > 0 else None
    if last_index is None:
        last_index = 0
    # S = max_j  curve[j] / Delta[j]  over the relevant window (last entry excluded)
    S = np.max(curve[last_index:-1] / np.asarray(Delta)[last_index:-1])

    return S


def conjugate_gradients_mugrid(
        comm: Communicator,
        fc: GlobalFieldCollection,
        hessp: callable,
        b: Field,
        x: Field,
        P: callable,
        tol: float = 1e-6,
        rtol: bool = False,
        maxiter: int = 1000,
        callback: callable = None,
        norm_metric: callable = None,
        **kwargs
):
    """
    Preconditioned conjugate gradient (PCG) method for the matrix-free
    solution of the symmetric positive definite linear problem ``A x = b``.

    ``A`` is represented by the function ``hessp`` (which computes the product
    of A with a field) and the preconditioner ``M^{-1}`` by ``P``. The method
    iteratively refines the solution ``x`` until the squared stopping
    quantity (by default the squared Euclidean residual norm ``(r, r)``) is
    below ``tol**2``, or until ``maxiter`` iterations are reached.

    Parameters
    ----------
    comm : muGrid.Communicator
        Communicator for parallel processing. ``comm.sum`` is used to reduce
        all local dot products to global scalars.
    fc : muGrid.GlobalFieldCollection
        Collection holding temporary fields of the CG algorithm. Must provide
        the ``'nodal_points'`` sub-point type. The work fields
        ``cg-search-direction``, ``cg-hessian-product``, ``cg-residual``,
        ``cg-preconditioned_residual`` (and ``cg-custom_metric_residual`` if
        ``norm_metric`` is given) are created in / fetched from it.
    hessp : callable
        ``hessp(x, Ax)``: applies the system matrix (Hessian) A to the field
        ``x`` and writes the result into the field ``Ax`` (in place). A must
        be symmetric positive definite on the iterated subspace.
    b : muGrid.Field
        Right-hand side field (same component shape as ``x``).
    x : muGrid.Field
        Initial guess for the solution; overwritten in place with the
        approximate solution.
    P : callable
        ``P(r, z)``: applies the (SPD) preconditioner ``M^{-1}`` to the field
        ``r`` and writes the result into the field ``z`` (in place). Use an
        identity copy for unpreconditioned CG.
    tol : float, optional
        Tolerance for convergence. The stopping quantity (a *squared* norm)
        is compared with ``tol**2``. The default is 1e-6.
    rtol : bool, optional
        If True, the tolerance is relative: the loop stops when the stopping
        quantity drops below ``tol**2`` times its initial value. The default
        is False (absolute tolerance).
    maxiter : int, optional
        Maximum number of iterations. The default is 1000.
    callback : callable, optional
        ``callback(iteration, x_s, r_s, p_s, z_s, stop_crit)``, called once
        with ``iteration = 0`` before the loop and after each iteration with
        the arrays ``.s`` of the current solution, residual, search direction
        and preconditioned residual, and the current (squared) stopping
        quantity. Note: in the loop the callback is invoked *before* the
        search direction is updated, so ``p_s`` is the direction used in the
        step just taken.
    norm_metric : callable, optional
        ``norm_metric(r, Pr)``: writes ``G r`` into ``Pr`` for some SPD
        metric ``G``. If given, the stopping quantity becomes ``(r, G r)``
        instead of the Euclidean ``(r, r)``.
    **kwargs
        Ignored; accepted for interface compatibility with other solvers.

    Returns
    -------
    x : muGrid.Field
        Approximate solution to the system Ax = b. (Same object as input field
        x.)

    Raises
    ------
    RuntimeError
        If ``(p, A p) <= 0`` is encountered, i.e. A is not positive definite
        (or has lost positive definiteness due to round-off).

    Warns
    -----
    RuntimeWarning
        On rank 0, if the method did not converge within ``maxiter``.

    Notes
    -----
    This is the classical Hestenes--Stiefel PCG (M. R. Hestenes, E. Stiefel,
    J. Res. Nat. Bur. Standards 49, 1952), e.g. Algorithm 9.1 in Y. Saad,
    *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM 2003::

        r_0 = b - A x_0,  z_0 = M^{-1} r_0,  p_0 = z_0
        alpha_k   = (r_k, z_k) / (p_k, A p_k)
        x_{k+1}   = x_k + alpha_k p_k
        r_{k+1}   = r_k - alpha_k A p_k
        z_{k+1}   = M^{-1} r_{k+1}
        beta_k    = (r_{k+1}, z_{k+1}) / (r_k, z_k)
        p_{k+1}   = z_{k+1} + beta_k p_k

    The residual is updated recursively (not recomputed as ``b - A x``).
    Each iteration costs one ``hessp``, one ``P`` and two or three global
    reductions (``comm.sum``). With ``rtol=True`` the initial guess is only
    accepted as converged if its residual is exactly zero.
    """

    # all stopping tests compare squared norms, hence tol**2
    tol_sq = tol * tol
    # --- allocate (or fetch existing) work fields in the field collection ---
    p = fc.real_field(
        name="cg-search-direction",  # name of the field
        components=(*x.components_shape,),  # shape of components
        sub_pt='nodal_points'  # sub-point type
    )
    Ap = fc.real_field(
        name="cg-hessian-product",  # name of the field
        components=(*x.components_shape,),  # shape of components
        sub_pt='nodal_points'  # sub-point type
    )
    r = fc.real_field(
        name="cg-residual",  # name of the field
        components=(*x.components_shape,),  # shape of components
        sub_pt='nodal_points'  # sub-point type
    )
    z = fc.real_field(
        name="cg-preconditioned_residual",  # name of the field
        components=(*x.components_shape,),  # shape of components
        sub_pt='nodal_points'  # sub-point type
    )

    # --- initialisation: r_0 = b - A x_0, z_0 = M^{-1} r_0, p_0 = z_0 ---
    hessp(x, Ap)
    r.s[...] = b.s - Ap.s
    P(r, z)
    p.s[...] = np.copy(z.s)  # residual  (first search direction = preconditioned residual)


    rr = comm.sum(np.dot(r.s.ravel(), r.s.ravel()))  # initial residual dot product
    rz = comm.sum(np.dot(r.s.ravel(), z.s.ravel()))  # initial residual dot product

    # --- choose the (squared) norm used in the stopping test ---
    #     (r, G r) with a user metric G, or the Euclidean (r, r)
    if norm_metric is not None:
        Pr = fc.real_field(
            name="cg-custom_metric_residual",  # name of the field
            components=(*x.components_shape,),  # shape of components
            sub_pt='nodal_points'  # sub-point type
        )
        norm_metric(r, Pr)
        stop_crit = comm.sum(np.dot(r.s.ravel(), Pr.s.ravel()))  # initial residual dot product


    elif norm_metric is None:
        stop_crit = rr

    # initial guess already good enough: absolute test, or (for rtol) an
    # exactly zero residual -- a relative test against itself is meaningless
    if stop_crit == 0 or (not rtol and stop_crit < tol_sq):
        return x

    # relative tolerance: scale by the initial (squared) stopping quantity
    if rtol:
        tol_sq = tol_sq * stop_crit

    if callback:
        #callback(0, x.s, r.s, p.s, z.s, stop_crit)
        callback(0, x.s, r.s, p.s, z.s, stop_crit)
    for iteration in range(maxiter):
        # Compute Hessian product
        hessp(p, Ap)

        # Update x (and residual)
        # curvature (p, A p); must be > 0 for an SPD operator
        pAp = comm.sum(np.dot(p.s.ravel(), Ap.s.ravel()))
        if pAp <= 0:
            raise RuntimeError("Hessian is not positive definite")

        # step length alpha_k = (r_k, z_k) / (p_k, A p_k): minimises the
        # A-norm (energy) error along p_k
        alpha = rz / pAp
        x.s[...] += alpha * p.s
        # recursive residual update r_{k+1} = r_k - alpha A p_k (no extra hessp)
        r.s[...] -= alpha * Ap.s

        # z_{k+1} = M^{-1} r_{k+1}
        P(r, z)

        # Check convergence
        next_rr = comm.sum(np.dot(r.s.ravel(), r.s.ravel()))
        next_rz = comm.sum(np.dot(r.s.ravel(), z.s.ravel()))
        if norm_metric is not None:
            norm_metric(r, Pr)
            stop_crit = comm.sum(np.dot(r.s.ravel(), Pr.s.ravel()))  # current residual in the custom metric, (r, G r)

        elif norm_metric is None:
            stop_crit = next_rr

        if callback:
            callback(iteration + 1, x.s, r.s, p.s, z.s, stop_crit)

        if stop_crit < tol_sq:
            return x

        # Update search direction
        # standard PCG beta: beta_k = (r_{k+1}, z_{k+1}) / (r_k, z_k)
        # (the commented line is the unpreconditioned variant)
        # beta = next_rr / rr
        beta = next_rz / rz

        rz = next_rz

        # p_{k+1} = z_{k+1} + beta_k p_k  (A-conjugate to all previous directions)
        p.s[...] = z.s + beta * p.s
        # p.s *= beta
        # p.s += z.s

    if comm.rank == 0:
        warnings.warn("Conjugate gradient algorithm did not converge", RuntimeWarning)
    return x

def conjugate_gradients_mugrid_experimental(
        comm: Communicator,
        fc: GlobalFieldCollection,
        hessp: callable,
        b: Field,
        x: Field,
        P: callable,
        tol: float = 1e-6,
        maxiter: int = 1000,
        callback: callable = None,
        rtol: bool = False,
        norm_metric: callable = None,
        lambda_min: float = None,
        stop_crit_norm: str = "rr",
        **kwargs
):
    """
    Conjugate gradient method for matrix-free solution of the linear problem
    Ax = b, where A is represented by the function hessp (which computes the
    product of A with a vector). The method iteratively refines the solution x
    until the quantity selected by stop_crit_norm is less than tol**2, or until
    maxiter iterations are reached.

    The PCG iteration itself is identical to :func:`conjugate_gradients_mugrid`
    (Hestenes--Stiefel PCG); this variant additionally computes a-posteriori
    estimates/bounds of the energy (A-norm) error ``||x - x_k||_A^2`` and
    lets the user choose which quantity drives the stopping test.

    Parameters
    ----------
    comm : muGrid.Communicator
        Communicator for parallel processing (global reductions via
        ``comm.sum``).
    fc : muGrid.GlobalFieldCollection
        Collection holding temporary fields of the CG algorithm (must provide
        the ``'nodal_points'`` sub-point type).
    hessp : callable
        ``hessp(x, Ax)``: writes ``A x`` into the field ``Ax`` (in place).
        A must be SPD.
    b : muGrid.Field
        Right-hand side field.
    x : muGrid.Field
        Initial guess for the solution; overwritten in place with the
        approximate solution.
    P : callable
        ``P(r, z)``: writes ``M^{-1} r`` into the field ``z`` (in place).
    tol : float, optional
        Tolerance for convergence; the selected quantity is compared with
        ``tol**2``. The default is 1e-6.
    maxiter : int, optional
        Maximum number of iterations. The default is 1000.
    callback : callable, optional
        ``callback(iteration, x_s, r_s, p_s, z_s, stop_crit)``, called before
        the loop (iteration 0) and after each iteration (before the search
        direction update). ``stop_crit`` is ``(r, r)`` or, if ``norm_metric``
        is given, ``(r, G r)`` -- independent of ``stop_crit_norm``.
    rtol : bool, optional
        If True, ``tol**2`` is multiplied by the initial value of ``(r, r)``
        (or ``(r, G r)`` with ``norm_metric``). Note that this same scaled
        tolerance is then used for every criterion, including the energy
        ones. The default is False.
    norm_metric : callable, optional
        ``norm_metric(r, Pr)``: writes ``G r`` into ``Pr`` for an SPD metric
        ``G``; enables the ``'custom'`` criterion ``(r, G r)``.
    lambda_min : float, optional
        mu_min of the Gauss-Radau upper bound. Must satisfy mu_min <= lambda_min
        of the preconditioned operator; pass 0.9 * eigen_LB to be safe. None
        disables the bound.
    stop_crit_norm : str, optional
        Which quantity the stopping test compares against tol**2:

          'rr'                 Euclidean residual  (r, r)          [default]
          'rz'                 preconditioned residual  (r, z)
          'custom'             (r, norm_metric(r)); needs norm_metric
          'energy_lower_estim' Meurant-Papez-Tichy delayed lower bound
                               on ||e_k||_K^2
          'energy_upper_estim' that lower bound divided by (1 - tau); needs
                               0 <= tau < 1. Computed on the fly, not stored
          'energy_upper_bound' Gauss-Radau upper bound; needs lambda_min
          'all'                iterate until EVERY applicable criterion above
                               is satisfied, i.e. until the slowest one is

        The three energy criteria are not available at every iteration: the
        estimator appends only when its delay condition is met, and the
        Gauss-Radau recursion stops after a breakdown. While the selected
        quantity is unavailable the loop simply does not stop, so an
        unsatisfiable criterion runs to maxiter rather than failing silently.

        NOTE: tol means a different thing in each mode. 'rr' and 'rz' are
        residual norms, the three energy criteria are squared energy errors.
        Comparing iteration counts across modes is only meaningful once the
        tolerances are chosen to target the same achieved ||e_k||_K^2.
    **kwargs
        ``tau`` (float, default 0.25): tolerance of the adaptive-delay test
        of the Hestenes--Stiefel lower bound; must satisfy ``0 <= tau < 1``
        for ``'energy_upper_estim'`` / ``'all'``. Other keys are ignored.

    Returns
    -------
    x : muGrid.Field
        Approximate solution to the systems Ax = b. (Same as input field x.)
    norms : dict
        'energy_lower_estim' and 'energy_upper_bound' series, as before, plus
        'stop_iteration': {criterion -> first iteration at which it fell below
        tol_sq, or None}. This is filled in EVERY mode, not just 'all', since
        all the values are computed anyway: one run yields the whole
        stopping-criterion comparison table.

        Neither stored series runs to the last iteration. 'energy_lower_estim'
        lags by the current delay, and 'energy_upper_bound' stops at a
        Gauss-Radau breakdown. Entry i of both corresponds to iteration i, so
        they may be plotted against their array index; they simply end early.
        'energy_upper_estim' is not stored at all - derive it from
        'energy_lower_estim' / (1 - tau) if a plot needs it.

    Raises
    ------
    ValueError
        If ``stop_crit_norm`` is unknown or its prerequisites
        (``norm_metric``, ``lambda_min``, ``0 <= tau < 1``) are not met.
    RuntimeError
        If ``(p, A p) <= 0`` (operator not positive definite).

    Warns
    -----
    RuntimeWarning
        On rank 0: if the Gauss--Radau recursion breaks down, and if
        ``maxiter`` is reached without satisfying the stopping test.

    Notes
    -----
    Energy-error estimates (A-norm error of the *preconditioned* CG, i.e.
    of ``||x - x_k||_A^2``):

    * Hestenes--Stiefel lower bound with delay ``d``:
      ``||x - x_l||_A^2 >= sum_{j=l}^{l+d} alpha_j (r_j, z_j)``.
      The delay is chosen adaptively following G. Meurant, J. Papez,
      P. Tichy, "Accurate error estimation in CG", Numer. Algorithms 88
      (2021): the estimate for iteration ``l`` is accepted once
      ``S * Delta_k / sum_{j=l}^{k-1} Delta_j <= tau`` with safety factor
      ``S`` from :func:`findS`.
    * Gauss--Radau upper bound ``||x - x_k||_A^2 <= mu_k (r_k, z_k)`` with
      the scalar recursion ``mu_0 = 1/lambda_min``,
      ``mu_{k+1} = (mu_k - alpha_k) / (lambda_min (mu_k - alpha_k) + beta_k)``
      (G. Meurant, P. Tichy, "Error Norm Estimation in the Conjugate
      Gradient Algorithm", SIAM 2024). It is a guaranteed bound only if
      ``lambda_min`` underestimates the smallest eigenvalue of ``M^{-1}A``.

    If the initial guess already satisfies the (absolute) test, ``norms``
    is returned empty.
    """
    tol_sq = tol * tol

    # tau is needed by the validation below, so it is read here rather than
    # just before the iteration loop
    tau = kwargs.get("tau", 0.25)

    # ---- stopping-criterion selection -------------------------------------
    _VALID_STOP = ("rr", "rz", "custom",
                   "energy_lower_estim", "energy_upper_estim",
                   "energy_upper_bound",
                   "all")
    if stop_crit_norm not in _VALID_STOP:
        raise ValueError(f"stop_crit_norm must be one of {_VALID_STOP}, "
                         f"got {stop_crit_norm!r}")
    if stop_crit_norm == "custom" and norm_metric is None:
        raise ValueError("stop_crit_norm='custom' requires norm_metric")
    if stop_crit_norm == "energy_upper_bound" and lambda_min is None:
        raise ValueError("stop_crit_norm='energy_upper_bound' requires lambda_min")
    if stop_crit_norm in ("energy_upper_estim", "all") and not (0.0 <= tau < 1.0):
        raise ValueError(f"stop_crit_norm={stop_crit_norm!r} requires "
                         f"0 <= tau < 1, got {tau}")

    p = fc.real_field(
        name="cg-search-direction",  # name of the field
        components=(*x.components_shape,),  # shape of components
        sub_pt='nodal_points'  # sub-point type
    )
    Ap = fc.real_field(
        name="cg-hessian-product",  # name of the field
        components=(*x.components_shape,),  # shape of components
        sub_pt='nodal_points'  # sub-point type
    )
    r = fc.real_field(
        name="cg-residual",  # name of the field
        components=(*x.components_shape,),  # shape of components
        sub_pt='nodal_points'  # sub-point type
    )
    z = fc.real_field(
        name="cg-preconditioned_residual",  # name of the field
        components=(*x.components_shape,),  # shape of components
        sub_pt='nodal_points'  # sub-point type
    )

    # --- initialisation: r_0 = b - A x_0, z_0 = M^{-1} r_0, p_0 = z_0 ---
    hessp(x, Ap)
    r.s[...] = b.s[...] - Ap.s[...]
    P(r, z)
    p.s[...] = np.copy(z.s[...])  # residual  (first search direction = preconditioned residual)

    norms = dict()

    rr = comm.sum(np.dot(r.s.ravel(), r.s.ravel()))  # initial residual dot product
    rz = comm.sum(np.dot(r.s.ravel(), z.s.ravel()))  # initial residual dot product

    if norm_metric is not None:
        Pr = fc.real_field(
            name="cg-custom_metric_residual",  # name of the field
            components=(*x.components_shape,),  # shape of components
            sub_pt='nodal_points'  # sub-point type
        )
        norm_metric(r, Pr)
        stop_crit = comm.sum(np.dot(r.s.ravel(), Pr.s.ravel()))  # initial residual dot product

    elif norm_metric is None:
        stop_crit = rr

    # initial guess already good enough: absolute test, or (for rtol) an
    # exactly zero residual; norms is still empty
    if stop_crit == 0 or (not rtol and stop_crit < tol_sq):
        return x, norms

    # relative tolerance: scaled by the initial (r, r) or (r, G r), whatever
    # stop_crit_norm is -- the same tol_sq is then used for all criteria
    if rtol:
        tol_sq = tol_sq * stop_crit

    if callback:
        callback(0, x.s, r.s, p.s, z.s, stop_crit)

    # --- state of the adaptive-delay Hestenes--Stiefel estimator ---
    # l     : iteration whose error is estimated next
    # d     : current delay (k - l)
    # Delta : Delta[j] = alpha_j (r_j, z_j) = ||x_{j+1} - x_j||_A^2
    # curve : curve[j] = sum_{i=j}^{k} Delta[i] (running tail sums)
    # estim : unused here (kept from the legacy implementation)
    #  % in the paper this is denoted as k
    l = 0
    d = 0
    Delta = []
    curve = []
    estim = []
    norms['energy_lower_estim'] = []
    norms['energy_upper_bound'] = []
    # --- Gauss--Radau upper bound: ||e_k||_A^2 <= mu_k (r_k, z_k), mu_0 = 1/lambda_min ---
    compute_upper_bound = lambda_min is not None
    if compute_upper_bound:
        mu_gr = 1.0 / lambda_min
        norms['energy_upper_bound'].append(mu_gr * rz)
    delay = []

    # Criteria whose prerequisites are met in this call. 'all' waits for every
    # one of them; every mode records when each first crosses tol_sq.
    _applicable = ["rr", "rz"]
    if norm_metric is not None:
        _applicable.append("custom")
    _applicable.append("energy_lower_estim")
    if 0.0 <= tau < 1.0:
        _applicable.append("energy_upper_estim")
    if lambda_min is not None:
        _applicable.append("energy_upper_bound")
    norms['stop_iteration'] = {c: None for c in _applicable}

    def _all_stop_values(rr_, rz_, custom_):
        """
        Every applicable criterion's current value. None means 'not usable at
        this iteration', which never counts as satisfied.
        """
        values = {}
        lo = norms["energy_lower_estim"]
        ub = norms["energy_upper_bound"]
        for c in _applicable:
            if c == "rr":
                values[c] = rr_
            elif c == "rz":
                values[c] = rz_
            elif c == "custom":
                values[c] = custom_
            elif c == "energy_lower_estim":
                values[c] = lo[-1] if lo else None
            elif c == "energy_upper_estim":
                # tau-dependent upper estimate, computed on the fly from the
                # latest lower bound rather than stored
                values[c] = lo[-1] / (1.0 - tau) if lo else None
            else:
                # Gauss-Radau: trust it only while the recursion is alive and
                # positive. After a breakdown the stored value is stale, and a
                # lambda_min above the true minimum can make it negative,
                # which a naive '< tol_sq' test reads as instant convergence.
                values[c] = (ub[-1] if (compute_upper_bound and ub and ub[-1] > 0.0)
                             else None)
        return values

    for iteration in range(maxiter):
        # Compute Hessian product
        hessp(p, Ap)

        # Update x (and residual)
        # curvature (p, A p); must be > 0 for an SPD operator
        pAp = comm.sum(np.dot(p.s.ravel(), Ap.s.ravel()))
        if pAp <= 0:
            raise RuntimeError("Hessian is not positive definite")

        # alpha_k = (r_k, z_k) / (p_k, A p_k);  x_{k+1} = x_k + alpha_k p_k;
        # r_{k+1} = r_k - alpha_k A p_k  (recursive residual)
        alpha = rz / pAp
        x.s[...] += alpha * p.s[...]
        r.s[...] -= alpha * Ap.s[...]

        # z_{k+1} = M^{-1} r_{k+1}
        P(r, z)

        # Check convergence
        next_rr = comm.sum(np.dot(r.s.ravel(), r.s.ravel()))
        next_rz = comm.sum(np.dot(r.s.ravel(), z.s.ravel()))
        if norm_metric is not None:
            norm_metric(r, Pr)
            stop_crit = comm.sum(np.dot(r.s.ravel(), Pr.s.ravel()))  # current residual in the custom metric, (r, G r)

        elif norm_metric is None:
            stop_crit = next_rr

        if callback:
            callback(iteration + 1, x.s, r.s, p.s, z.s, stop_crit)

        # Update search direction
        # beta_k = (r_{k+1}, z_{k+1}) / (r_k, z_k);  p_{k+1} = z_{k+1} + beta_k p_k
        # (beta is also needed by the Gauss--Radau recursion below)
        # beta = next_rr / rr
        beta = next_rz / rz
        p.s[...] = z.s + beta * p.s

        # Energy - error upper bound (Gauss-Radau)
        # mu_{k+1} = (mu_k - alpha_k) / (lambda_min (mu_k - alpha_k) + beta_k);
        # both numerator and denominator must stay positive, otherwise
        # lambda_min was not a lower bound on the spectrum of M^{-1}A
        if compute_upper_bound:
            gr_t = mu_gr - alpha
            gr_den = lambda_min * gr_t + beta
            if gr_t <= 0.0 or gr_den <= 0.0:
                if comm.rank == 0:
                    warnings.warn(
                        f"Gauss-Radau recursion broke down at iteration {iteration + 1}; "
                        f"lambda_min = {lambda_min:g} is probably not below the smallest "
                        f"eigenvalue. Upper bound not computed further.", RuntimeWarning)
                compute_upper_bound = False
            else:
                mu_gr = gr_t / gr_den
                norms['energy_upper_bound'].append(mu_gr * next_rz)

        # Energy - error estimator
        # Delta_k = alpha_k (r_k, z_k) (rz still holds (r_k, z_k) here);
        # add it to every running tail sum curve[j] = sum_{i=j}^{k} Delta_i
        Delta.append(alpha * rz)
        curve.append(0)
        curve = (np.asarray(curve) + Delta[-1]).tolist()

        if iteration > 1:
            # safety factor
            S = findS(curve, Delta, l)

            # Adaptive delay (Meurant, Papez, Tichy 2021): accept the estimate
            # of ||x - x_l||_A^2 = sum_{j=l}^{k} Delta_j as soon as the newest
            # contribution is relatively small, S Delta_k / sum_{j=l}^{k-1} Delta_j <= tau.
            # Several l may be accepted in one iteration (delay d decreases).
            # Delta entries are already global scalars (identical on every
            # rank), so a plain sum is used -- no MPI reduction.
            num = S * Delta[-1]
            den = float(np.sum(Delta[l:-1]))
            while (d >= 0) and (den > 0) and (num / den <= tau):
                delay.append(d)
                norms['energy_lower_estim'].append(den + Delta[-1])
                l = l + 1
                d = d - 1
                den = float(np.sum(Delta[l:-1]))

            d = d + 1

        # shift (r_k, z_k) <- (r_{k+1}, z_{k+1}) for the next iteration
        rz = next_rz
        # p.s *= beta
        # p.s += z.s

        # Stopping test. Placed here, after the energy estimator and the
        # Gauss-Radau update, so that the energy criteria see this iteration's
        # values. A None means the selected quantity is not available yet.
        stop_values = _all_stop_values(next_rr, next_rz, stop_crit)

        # Record the first crossing of every criterion, whichever one is
        # driving the loop. Free: all the values are already computed.
        for _name, _val in stop_values.items():
            if (norms['stop_iteration'][_name] is None
                    and _val is not None and _val < tol_sq):
                norms['stop_iteration'][_name] = iteration + 1

        if stop_crit_norm == "all":
            if all(v is not None and v < tol_sq for v in stop_values.values()):
                return x, norms
        else:
            stop_value = stop_values[stop_crit_norm]
            if stop_value is not None and stop_value < tol_sq:
                return x, norms

    if comm.rank == 0:
        _never = [c for c, k in norms.get('stop_iteration', {}).items() if k is None]
        if stop_crit_norm == "all" and _never:
            warnings.warn(
                f"Reached maxiter in stop_crit_norm='all': {_never} never "
                f"satisfied the tolerance, so the run could not finish even "
                f"though the other criteria did. Check lambda_min (a "
                f"Gauss-Radau breakdown freezes the upper bound) and tau.",
                RuntimeWarning)
        else:
            warnings.warn("Conjugate gradient algorithm did not converge", RuntimeWarning)

    return x, norms

def ___PCG(Afun, B, x0, P, steps=int(500), toler=1e-6, norm_energy_upper_bound=False,
        lambda_min=None, norm_type='rz',
        callback=None, **kwargs):
    # print('I am in PCG')
    """
    Legacy preconditioned conjugate gradients solver using NumPy arrays.

    Kept for reference; the muGrid-based :func:`conjugate_gradients_mugrid`
    and :func:`conjugate_gradients_mugrid_experimental` supersede it. The
    triple-underscore name marks it as deprecated/private.

    Parameters
    ----------
    Afun : callable
        ``Afun(x) -> A x``; returns a new array with the matrix-vector product
        (the original documentation also mentions Matrix/LinOper objects).
    B : numpy.ndarray
        Right-hand side of the linear system (any shape; MPI-distributed
        arrays are supported through :func:`scalar_product_mpi`).
    x0 : numpy.ndarray or None
        Initial approximation of the solution; zeros if None. Not modified.
    P : callable
        ``P(r) -> M^{-1} r``; returns the preconditioned residual.
    steps : int, optional
        The loop runs for at most ``steps - 1`` iterations. Default 500.
    toler : float, optional
        Tolerance compared *directly* (not squared) with the selected
        (squared) norm, see ``norm_type``. Default 1e-6.
    norm_energy_upper_bound : bool, optional
        If True, compute the Gauss--Radau upper bound on the energy error
        (requires ``lambda_min``). Default False.
    lambda_min : float, optional
        Lower estimate of the smallest eigenvalue of ``M^{-1}A`` used by the
        Gauss--Radau bound.
    norm_type : str, optional
        Stopping criterion: ``'rz'`` ((r, z) < toler), ``'rz_rel'``,
        ``'rr'``, ``'rr_rel'`` (relative to the initial value),
        ``'energy'`` (Gauss--Radau bound < toler; needs
        ``norm_energy_upper_bound=True``), ``'data_scaled_rz'``
        ((r, M^{-1} G r)) or ``'data_scaled_rr'`` ((r, G r)), with
        ``G = kwargs['norm_metric']``. Default ``'rz'``.
    callback : callable, optional
        Called as ``callback(x0)`` once before the loop and as
        ``callback(x_k, r)`` in each iteration (``r`` is the residual *before*
        the update in that iteration).
    **kwargs
        ``exact_solution`` (array): if given, the true energy error
        ``(e, A e)`` is recorded in ``norms['energy_iter_error']``;
        ``norm_metric`` (callable ``G(r) -> G r``) for the data-scaled norms;
        ``tau`` (float, default 0.25) for the adaptive delay estimator;
        ``energy_lower_bound`` (any value) -- if present, the delayed
        Hestenes--Stiefel lower-bound estimates are returned in
        ``norms['energy_lower_bound']``.

    Returns
    -------
    x_k : numpy.ndarray
        Approximate solution.
    norms : dict
        Lists of per-iteration values: ``'residual_rr'`` ((r, r)),
        ``'residual_rz'`` ((r, z)), ``'data_scaled_rz'``,
        ``'data_scaled_rr'``, ``'energy_upper_bound'`` and, optionally,
        ``'energy_iter_error'`` and ``'energy_lower_bound'``.

    Notes
    -----
    Hestenes--Stiefel PCG (see Saad, *Iterative Methods for Sparse Linear
    Systems*, Alg. 9.1) with the energy-error estimates described in
    :func:`conjugate_gradients_mugrid_experimental`.
    """
    if x0 is None:
        x0 = np.zeros(B.shape)
    if callback is None:
        callback = donothing
    norms = dict()
    norms['residual_rr'] = []
    norms['residual_rz'] = []
    norms['data_scaled_rz'] = []
    norms['data_scaled_rr'] = []

    norms['energy_upper_bound'] = []
    if "exact_solution" in kwargs:
        norms['energy_iter_error'] = []
        error = x0 - kwargs['exact_solution']
        norms['energy_iter_error'].append(scalar_product_mpi(error, Afun(error)))

    ##
    k = 0
    x_k = np.copy(x0)
    callback(x_k)
    ##
    # initial residual r_0 = B - A x_0 and preconditioned residual z_0 = M^{-1} r_0
    Ax = Afun(x0)

    r_0 = B - Ax
    z_0 = P(r_0)

    # scalar_product = lambda a, b: np.sum(a * b)

    r_0z_0 = scalar_product_mpi(r_0, z_0)

    norms['residual_rr'].append(scalar_product_mpi(r_0, r_0))
    norms['residual_rz'].append(r_0z_0)

    # Gauss--Radau upper bound: ||e_0||_A^2 <= mu_0 (r_0, z_0), mu_0 = 1/lambda_min
    if norm_energy_upper_bound:
        gamma_mu = 1 / lambda_min
        norms['energy_upper_bound'].append(gamma_mu * r_0z_0)
    if norm_type == 'data_scaled_rz':
        r_1_C_z_1 = scalar_product_mpi(r_0, P(kwargs['norm_metric'](r_0)))
        norms['data_scaled_rz'].append(r_1_C_z_1)
    if norm_type == 'data_scaled_rr':
        r_1_C_r_1 = scalar_product_mpi(r_0, kwargs['norm_metric'](r_0))
        norms['data_scaled_rr'].append(r_1_C_r_1)

    # first search direction p_0 = z_0
    p_0 = np.copy(z_0)
    # state of the adaptive-delay estimator (see conjugate_gradients_mugrid_experimental)
    #  % in the paper this is denoted as k
    l = 0
    d = 0
    Delta = []
    curve = []
    estim = []
    delay = []
    if "tau" in kwargs:
        tau = kwargs['tau']
    else:
        tau = 0.25

    for k in np.arange(1, steps):
        Ap_0 = Afun(p_0)
        # print('callback Ap_0 = {}'.format(np.linalg.norm(Ap_0)))
        # print('ID Ap_0 = {}'.format(id(Ap_0)))

        # step length alpha_k = (r_k, z_k) / (p_k, A p_k); x_{k+1} = x_k + alpha_k p_k
        alpha = float(r_0z_0 / scalar_product_mpi(p_0, Ap_0))
        x_k = x_k + alpha * p_0
        # print('callback xo = {}'.format(np.linalg.norm(x_k)))
        callback(x_k, r_0)
        # print('callback xo = {}'.format(np.linalg.norm(x_k)))
        # print('callback Ap_0 = {}'.format(np.linalg.norm(Ap_0)))
        # print('ID Ap_0 = {}'.format(id(Ap_0)))
        # if xCG.val.mean() > 1e-10:
        #     print('iteration left zero-mean space {} \n {}'.format(xCG.name, xCG.val.mean()))

        # recursive residual update and preconditioning
        # (variables named *_0 always hold the *current* iterate's quantities)
        r_0 = r_0 - alpha * Ap_0

        z_0 = P(r_0)

        r_1z_1 = scalar_product_mpi(r_0, z_0)

        # beta_k = (r_{k+1}, z_{k+1}) / (r_k, z_k)
        beta = r_1z_1 / r_0z_0

        if "exact_solution" in kwargs:
            error = x_k - kwargs['exact_solution']
            norms['energy_iter_error'].append(scalar_product_mpi(error, Afun(error)))

        if norm_energy_upper_bound:
            # updade upper bound on energy error estim parameter
            # mu_{k+1} = (mu_k - alpha_k) / (lambda_min (mu_k - alpha_k) + beta_k)
            gamma_mu = (gamma_mu - alpha) / (lambda_min * (gamma_mu - alpha) + beta)
            norms['energy_upper_bound'].append(gamma_mu * r_1z_1)

        norms['residual_rr'].append(scalar_product_mpi(r_0, r_0))
        norms['residual_rz'].append(r_1z_1)

        # if r_1z_1 < toler:  # TODO[Solver] check out stopping criteria
        #     break
        if norm_type == 'rz':
            if norms['residual_rz'][-1] < toler:  # TODO[Solver] check out stopping criteria
                break
        if norm_type == 'rz_rel':
            if norms['residual_rz'][-1] / norms['residual_rz'][0] < toler:  # TODO[Solver] check out stopping criteria
                break
        if norm_type == 'rr':
            if norms['residual_rr'][-1] < toler:  # TODO[Solver] check out stopping criteria
                break
        if norm_type == 'rr_rel':
            if norms['residual_rr'][-1] / norms['residual_rr'][0] < toler:  # TODO[Solver] check out stopping criteria
                break
        if norm_type == 'energy':
            if norms['energy_upper_bound'][-1] < toler:  # TODO[Solver] check out stopping criteria
                break
        if norm_type == 'data_scaled_rz':
            r_1_C_z_1 = scalar_product_mpi(r_0, P(kwargs['norm_metric'](r_0)))
            norms['data_scaled_rz'].append(r_1_C_z_1)
            if norms['data_scaled_rz'][-1] < toler:  # TODO[Solver] check out stopping criteria
                break
        if norm_type == 'data_scaled_rr':
            r_1_C_r_1 = scalar_product_mpi(r_0, kwargs['norm_metric'](r_0))
            norms['data_scaled_rr'].append(r_1_C_r_1)
            if norms['data_scaled_rr'][-1] < toler:  # TODO[Solver] check out stopping criteria
                break

        # new search direction p_{k+1} = z_{k+1} + beta_k p_k
        p_0 = z_0 + beta * p_0

        # Energy - error estimator
        # Delta_k = alpha_k (r_k, z_k); curve[j] = sum_{i=j}^{k} Delta_i
        Delta.append(alpha * r_0z_0)
        curve.append(0)
        curve = (np.asarray(curve) + Delta[-1]).tolist()

        if k > 1:
            # safety factor
            S = findS(curve, Delta, l)

            num = S * Delta[-1]
            den = float(np.sum(Delta[l:-1]))
            while (d >= 0) and (den > 0) and (num / den <= tau):
                delay.append(d)
                estim.append(den + Delta[-1])
                l = l + 1
                d = d - 1
                den = float(np.sum(Delta[l:-1]))

            d = d + 1

        # shift (r_k, z_k) <- (r_{k+1}, z_{k+1})
        r_0z_0 = r_1z_1

        if "energy_lower_bound" in kwargs:
            norms['energy_lower_bound'] = estim

    return x_k, norms

def Richardson(Afun, B, x0, omega, P=None, steps=int(500), toler=1e-6):
    """
    (Preconditioned) Richardson iteration using NumPy arrays.
    𝑥𝑘+1=𝑥𝑘−𝜏(𝐴𝑥𝑘−𝑓)

    Here implemented as ``x_{k+1} = x_k + omega * P(B - A x_k)``.

    Parameters
    ----------
    Afun : callable
        ``Afun(x) -> A x``; returns the matrix-vector product as a new array.
    B : numpy.ndarray
        Right-hand side of the linear system.
    x0 : numpy.ndarray or None
        Initial approximation of the solution; zeros if None. Copied, not
        modified.
    omega : float
        Relaxation parameter (step size). For SPD ``P A`` convergence
        requires ``0 < omega < 2 / lambda_max(P A)``.
    P : callable, optional
        ``P(r) -> M^{-1} r`` preconditioner; identity if None.
    steps : int, optional
        At most ``steps - 1`` iterations are performed. Default 500.
    toler : float, optional
        Stop when the squared Euclidean residual ``(r, r)`` < ``toler``.
        Default 1e-6.

    Returns
    -------
    x_k : numpy.ndarray
        Approximate solution.
    norms : dict
        ``'residual_rr'``: list of ``(r_k, r_k)`` (global, via MPI), where
        ``r_k`` is the residual of the iterate *before* the update of that
        step.
    """
    if x0 is None:
        x0 = np.zeros(B.shape)
    if P is None:
        P = lambda x: 1 * x

    norms = dict()
    norms['residual_rr'] = []
    ##
    k = 0
    x_k = np.copy(x0)
    ##
    for k in np.arange(1, steps):
        Ax = Afun(x_k)

        # r_k = B - A x_k;  x_{k+1} = x_k + omega M^{-1} r_k  (in place)
        r_0 = B - Ax
        x_k += omega * P(r_0)

        norms['residual_rr'].append(scalar_product_mpi(r_0, r_0))
        # print(norms['residual_rr'][-1])
        if norms['residual_rr'][-1] < toler:
            break

    return x_k, norms


def dr_pbcg_mugrid(
        comm: Communicator,
        fc: GlobalFieldCollection,
        hessp: callable,
        b_list: list,
        x_list: list,
        P: callable,
        tol: float = 1e-6,
        rtol: bool = False,
        maxiter: int = 1000,
        callback: callable = None,
        **kwargs
):
    """
    DR-PBCG: Preconditioned Block Conjugate Gradient with deflation-based restart.

    Solves A X = B where B is a block of m right-hand sides simultaneously,
    following Algorithm 5 of Meurant & Tichy (2026).

    All m systems share the same SPD operator A and preconditioner M^{-1}.
    The block Krylov space generated from all right-hand sides is used
    jointly, which typically needs fewer iterations than m separate PCG runs
    (one block iteration costs m ``hessp`` and m ``P`` applications).

    Parameters
    ----------
    comm : muGrid.Communicator
        Communicator used for global reductions (``comm.sum``).
    fc : muGrid.GlobalFieldCollection
        Collection that holds all working fields (must have 'nodal_points' sub-pt).
        7*m work fields named ``dr-pbcg-<name>-<j>`` are created/fetched.
    hessp : callable
        hessp(p, Ap): computes Ap = A p, writing the result into Ap.
    b_list : list of muGrid.Field, length m
        Block of m right-hand-side fields.
    x_list : list of muGrid.Field, length m
        Initial guesses, overwritten in-place with the approximate solutions.
    P : callable
        P(r, z): applies preconditioner M^{-1} to r, writing the result into z.
    tol : float
        Convergence tolerance on ||R_k||_F (absolute, or relative when rtol=True).
        Internally the *squared* Frobenius norm is compared with ``tol**2``.
    rtol : bool
        If True, tolerance is relative to the initial ||R_0||_F.
    maxiter : int
        Maximum number of iterations.
    callback : callable, optional
        callback(iteration, x_list, stop_crit) called after each iteration
        (and once with iteration 0); ``stop_crit`` is the squared Frobenius
        norm (or its upper bound, in the early-exit branch).
    **kwargs
        Ignored.

    Returns
    -------
    x_list : list of Field
        Approximate solutions (same objects as input).
    norms : dict
        'residual_frobenius': list of the *squared* Frobenius norms
        ||R_k||_F^2 (entry 0 is the initial residual, then one per
        iteration). The last entry of an early exit is the upper bound
        ||R||_F^2 ||Sigma||_F^2 rather than ||R_k||_F^2.

    Raises
    ------
    RuntimeError
        If a column collapses during the QR factorisation (linearly
        dependent residual block / right-hand sides).
    numpy.linalg.LinAlgError
        If one of the small m x m systems is singular.

    Warns
    -----
    RuntimeWarning
        On rank 0, if the method did not converge within ``maxiter``.

    Notes
    -----
    The residual block is kept in factored form ``R_k = Q_k Sigma_k`` with
    ``Q_k`` orthonormal (thin QR, here by Modified Gram--Schmidt), so that
    ``||R_k||_F^2 = trace(Sigma_k^T Sigma_k)`` is available from an m x m
    matrix. Restart/deflation of converged columns is not implemented here;
    a rank-deficient residual block raises an error instead.
    Reference: G. Meurant, P. Tichy, block CG / DR-PBCG, Algorithm 5 (2026).
    """
    m    = len(b_list)
    comp = (*b_list[0].components_shape,)

    Q     = [fc.real_field(name=f'dr-pbcg-Q-{j}',    components=comp, sub_pt='nodal_points') for j in range(m)]
    S     = [fc.real_field(name=f'dr-pbcg-S-{j}',    components=comp, sub_pt='nodal_points') for j in range(m)]
    Q_new = [fc.real_field(name=f'dr-pbcg-Qnew-{j}', components=comp, sub_pt='nodal_points') for j in range(m)]
    S_new = [fc.real_field(name=f'dr-pbcg-Snew-{j}', components=comp, sub_pt='nodal_points') for j in range(m)]
    MiQ   = [fc.real_field(name=f'dr-pbcg-MiQ-{j}',  components=comp, sub_pt='nodal_points') for j in range(m)]
    R     = [fc.real_field(name=f'dr-pbcg-R-{j}',    components=comp, sub_pt='nodal_points') for j in range(m)]
    AS    = [fc.real_field(name=f'dr-pbcg-AS-{j}',   components=comp, sub_pt='nodal_points') for j in range(m)]

    def dot(u, v):
        """Global (MPI-reduced) Euclidean inner product of two fields."""
        return comm.sum(np.dot(u.s.ravel(), v.s.ravel()))

    def gram(u_list, v_list):
        """m × m matrix G[i,j] = <u_list[i], v_list[j]>."""
        return np.array([[dot(u_list[i], v_list[j]) for j in range(m)]
                         for i in range(m)])

    def householder_qr(r_list, q_list):
        """Thin QR via Householder reflections (MPI-aware).
        Overwrites q_list with orthonormal columns. Returns upper-triangular Psi.

        Currently unused (see the commented-out call below). Mechanically it
        QR-factors the m x m Gram matrix G = Q_coeff Psi and sets
        q_new = r @ Q_coeff. Note that then q_new^T q_new = Q_coeff^T G Q_coeff
        = Psi, which is in general *not* the identity, so the columns are not
        orthonormal and Psi is not the R-factor of r.
        """
        Psi = np.zeros((m, m))

        for j in range(m):
            q_list[j].s[...] = r_list[j].s

        # Build the m x m Gram matrix G[i,j] = dot(r_list[i], r_list[j])
        # Then QR-factor G's Cholesky-like structure via Householder on the normal eqs.
        # Instead: work with the "tall" matrix implicitly via its m x m inner product matrix.

        # Step 1: collect all pairwise inner products (the m x m matrix A where A[:,j] is
        # the j-th vector expressed in terms of inner products with all others).
        # Householder QR on this m x m matrix gives us the mixing coefficients.
        A = np.zeros((m, m))
        for i in range(m):
            for j in range(i, m):
                A[i, j] = dot(q_list[i], q_list[j])
                A[j, i] = A[i, j]

        # Step 2: Householder QR of A (plain numpy, O(m^3), cheap)
        Q_coeff, Psi = np.linalg.qr(A)  # A = Q_coeff @ Psi

        # Step 3: form new q_list as linear combinations
        old_fields = [q_list[j].s.copy() for j in range(m)]
        for j in range(m):
            q_list[j].s[...] = sum(Q_coeff[k, j] * old_fields[k] for k in range(m))

        return Psi

    def modified_gram_schmidt(r_list, q_list):
        """Thin QR via Modified Gram-Schmidt (MPI-aware).
        Overwrites q_list with orthonormal columns. Returns upper-triangular Psi."""
        Psi = np.zeros((m, m))
        for j in range(m):
            q_list[j].s[...] = r_list[j].s
        # for each column j: subtract its projections onto the already
        # orthonormalised columns i < j (using the *updated* q_j -> MGS),
        # then normalise. Psi[i, j] are the R-factor entries.
        for j in range(m):
            for i in range(j):
                Psi[i, j] = dot(q_list[i], q_list[j])
                q_list[j].s[...] -= Psi[i, j] * q_list[i].s
            nrm = np.sqrt(dot(q_list[j], q_list[j]))
            if nrm < 1e-14:
                raise RuntimeError(
                    f"DR-PBCG: column {j} collapsed during QR (linearly dependent RHS?)"
                )
            Psi[j, j] = nrm
            q_list[j].s[...] /= nrm
        return Psi

    def cholesky_qr(r_list, q_list):
        """Tall-and-thin QR via CholeskyQR (MPI-aware).
        Overwrites q_list with orthonormal columns. Returns upper-triangular Psi.

        Needs only one block of global reductions but loses orthogonality
        for ill-conditioned blocks (error ~ cond(R)^2 * eps). Currently
        unused (see the commented-out call below).
        """
        for j in range(m):
            q_list[j].s[...] = r_list[j].s

        # Step 1: Gram matrix G = R^T R  (one global reduction)
        G = np.zeros((m, m))
        for i in range(m):
            for j in range(i, m):
                G[i, j] = dot(q_list[i], q_list[j])
                G[j, i] = G[i, j]

        # Step 2: Cholesky G = L L^T  →  Psi = L^T (upper triangular R-factor)
        try:
            Psi = np.linalg.cholesky(G).T
        except np.linalg.LinAlgError:
            raise RuntimeError(
                "DR-PBCG: Gram matrix not positive definite (linearly dependent RHS?)"
            )

        # Step 3: Q = R Psi^{-1}
        Psi_inv = np.linalg.inv(Psi)
        old_fields = [q_list[j].s.copy() for j in range(m)]
        for j in range(m):
            q_list[j].s[...] = sum(Psi_inv[k, j] * old_fields[k] for k in range(m))

        return Psi
    # ------------------------------------------------------------------
    # Initialisation  (Algorithm 5, lines 2-5)
    # ------------------------------------------------------------------
    for j in range(m):
        hessp(x_list[j], R[j])                        # R[j] = A X0[j]
        R[j].s[...] = b_list[j].s - R[j].s            # R0 = B - A X0  (line 2)

    # Check convergence before QR: zero residual means X0 is already the solution.
    tol_sq     = tol * tol
    R0_norm_sq = comm.sum(sum(np.dot(R[j].s.ravel(), R[j].s.ravel()) for j in range(m)))

    if rtol:
        tol_sq = tol_sq * R0_norm_sq

    norms = {'residual_frobenius': [R0_norm_sq]}

    if R0_norm_sq == 0 or R0_norm_sq < tol_sq:
        return x_list, norms
    # Alternative QR variants kept for reference (Householder/CholeskyQR)
    #np.qr(R, mode='reduced')
    Sigma = modified_gram_schmidt(R, Q)                # [Q0, Sigma0] = qr(R0)  (line 3)
    #Sigma = householder_qr(R, Q)                # [Q0, Sigma0] = qr(R0)  (line 3)
    #Sigma = cholesky_qr(R, Q)                # [Q0, Sigma0] = qr(R0)  (line 3)


    for j in range(m):
        P(Q[j], S[j])                                  # S0 = M^{-1} Q0  (line 4)

    Theta = gram(Q, S)                                 # Theta0 = Q0^T M^{-1} Q0  (line 5)

    if callback:
        callback(0, x_list, R0_norm_sq)

    # ------------------------------------------------------------------
    # Main loop  (Algorithm 5, lines 6-13)
    # ------------------------------------------------------------------
    for iteration in range(maxiter):

        # Line 7: Pi = (S^T A S)^{-1} Theta
        for j in range(m):
            hessp(S[j], AS[j])
        StAS = gram(S, AS)                             # S^T A S  (m × m, SPD)
        Pi   = np.linalg.solve(StAS, Theta)            # (S^T A S)^{-1} Theta

        # Line 8: X_k = X_{k-1} + S Pi Sigma
        C = Pi @ Sigma                                 # m × m
        for j in range(m):
            for i in range(m):
                x_list[j].s[...] += C[i, j] * S[i].s

        # Line 9: R = Q - A S Pi,  then [Q_new, Psi] = qr(R)
        # (R here is the residual in the Q-basis; the true residual block is
        #  R_k = R Sigma_{k-1})
        for j in range(m):
            R[j].s[...] = Q[j].s
            for i in range(m):
                R[j].s[...] -= Pi[i, j] * AS[i].s

        # Pre-QR convergence check: ||R_k||_F ≤ ||R||_F * ||Sigma||_F (submultiplicativity).
        # If R ≈ 0 the system is solved; avoid QR of near-zero columns.
        R_norm_sq     = comm.sum(sum(np.dot(R[j].s.ravel(), R[j].s.ravel()) for j in range(m)))
        Sigma_norm_sq = np.trace(Sigma.T @ Sigma)
        stop_crit     = R_norm_sq * Sigma_norm_sq
        if stop_crit < tol_sq:
            norms['residual_frobenius'].append( stop_crit)
            if callback:
                callback(iteration + 1, x_list, stop_crit)
            return x_list, norms

        Psi = modified_gram_schmidt(R, Q_new)

        # Line 10: Theta_new = Q_new^T M^{-1} Q_new
        for j in range(m):
            P(Q_new[j], MiQ[j])
        Theta_new = gram(Q_new, MiQ)

        # Line 11: S_new = M^{-1} Q_new + S Theta^{-1} Psi^T Theta_new
        C2 = np.linalg.solve(Theta, Psi.T @ Theta_new)  # m × m
        for j in range(m):
            S_new[j].s[...] = MiQ[j].s
            for i in range(m):
                S_new[j].s[...] += C2[i, j] * S[i].s

        # Line 12: Sigma_k = Psi Sigma_{k-1}
        Sigma = Psi @ Sigma

        # Convergence: ||R_k||_F^2 = tr(Sigma^T Sigma)
        stop_crit = np.trace(Sigma.T @ Sigma)
        norms['residual_frobenius'].append(stop_crit)

        if callback:
            callback(iteration + 1, x_list, stop_crit)

        if stop_crit < tol_sq:
            return x_list, norms

        # Advance: swap buffers instead of copying field data
        # (Q_k <- Q_new, S_k <- S_new, Theta_k <- Theta_new; the old buffers
        #  are reused as scratch space in the next iteration)
        Q, Q_new = Q_new, Q
        S, S_new = S_new, S
        Theta = Theta_new

    if comm.rank == 0:
        warnings.warn("DR-PBCG did not converge", RuntimeWarning)
    return x_list, norms


def scalar_product_mpi(a, b):
    """
    Global Euclidean inner product of two MPI-distributed NumPy arrays.

    Parameters
    ----------
    a, b : numpy.ndarray
        Local parts of the distributed arrays (same/broadcastable shapes).

    Returns
    -------
    float
        ``sum(a * b)`` summed over all ranks of ``MPI.COMM_WORLD``.
    """
    return Reduction(MPI.COMM_WORLD).sum(a * b)


# Update parameters using Adam
def update_parameters_with_adam(x, grads, m, v,
                                t, learning_rate,
                                beta1, beta2,
                                epsilon=1e-8):
    """
    One Adam update step (Kingma & Ba, "Adam: A Method for Stochastic
    Optimization", ICLR 2015).

    Parameters
    ----------
    x : numpy.ndarray
        Current parameters (design variables).
    grads : numpy.ndarray
        Gradient of the objective at ``x`` (same shape as ``x``).
    m, v : numpy.ndarray
        Exponential moving averages of the gradient (1st moment) and of the
        squared gradient (2nd raw moment) from the previous step.
    t : int
        Zero-based iteration counter; ``t + 1`` is used in the bias
        correction.
    learning_rate : float
        Step size ``alpha``.
    beta1, beta2 : float
        Decay rates of the first / second moment estimates (typically 0.9 and
        0.999).
    epsilon : float, optional
        Small constant avoiding division by zero. Default 1e-8.

    Returns
    -------
    x, m, v : numpy.ndarray
        Updated parameters and moment estimates (new arrays; inputs are not
        modified in place).
    """
    # biased moment estimates
    m = beta1 * m + (1.0 - beta1) * grads
    v = beta2 * v + (1.0 - beta2) * grads ** 2
    # bias correction (moments are initialised with zero)
    m_hat = m / (1.0 - beta1 ** (t + 1))
    v_hat = v / (1.0 - beta2 ** (t + 1))
    # element-wise scaled gradient step
    x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return x, m, v


# Adam optimization algorithm
def adam(f, df, x0,
         n_iter, alpha, beta1, beta2, eps=1e-8, callback=None, gtol=1e-5, ftol=2.2e-9):
    """
    Minimise ``f`` with the Adam optimiser (Kingma & Ba, ICLR 2015).

    Parameters
    ----------
    f : callable
        ``f(x) -> float`` objective function (global value, identical on all
        ranks).
    df : callable
        ``df(x) -> numpy.ndarray`` gradient of ``f`` (local part, same shape
        as ``x``).
    x0 : numpy.ndarray
        Initial point (local part of the MPI-distributed design vector).
    n_iter : int
        Maximum number of iterations.
    alpha : float
        Learning rate (step size).
    beta1, beta2 : float
        Moment decay rates, see :func:`update_parameters_with_adam`.
    eps : float, optional
        Regularisation constant of the update. Default 1e-8.
    callback : callable
        ``callback([phi, phi_change, max_grad, abs_grad, x, t])`` called
        every iteration. Although it defaults to None it is called
        unconditionally, so it must be provided.
    gtol : float, optional
        Stop when the global max-norm of the gradient ``max|grad|`` < gtol.
        Default 1e-5.
    ftol : float, optional
        Stop when the decrease ``phi_old - phi <= ftol * max(1, |phi|,
        |phi_old|)`` (scipy L-BFGS-B-like criterion). Default 2.2e-9.

    Returns
    -------
    list
        ``[x, phi, t]``: final point, its objective value and the last
        (zero-based) iteration index.

    Notes
    -----
    The gradient criterion uses the gradient evaluated at the point *before*
    the last update. Norms are reduced over ``MPI.COMM_WORLD``. Progress is
    printed on rank 0 only. The function-tolerance test uses the magnitude of
    the change, ``|phi_old - phi| <= ftol * max(1, |phi|, |phi_old|)``, so an
    uphill step does not count as convergence unless it is also tiny.
    """
    # phi=[]
    # phi_change=[]
    # Generate an initial point
    x = x0
    phi_old = f(x)

    # Initialize Adam moments
    # m, v = initialize_adam()
    m = np.zeros_like(x0)
    v = np.zeros_like(x0)
    # Run the gradient descent updates
    for t in range(n_iter):
        # Calculate gradient g(t)
        grad = df(x)

        # Update parameters using Adam
        x, m, v = update_parameters_with_adam(x=x, grads=grad, m=m,
                                              v=v, t=t, learning_rate=alpha,
                                              beta1=beta1, beta2=beta2,
                                              epsilon=eps)

        # Evaluate candidate point
        phi = f(x)

        # decrease of the objective in this step (positive = improvement)
        phi_change = phi_old - phi

        # global gradient norms: max-norm and Euclidean norm over all ranks

        max_grad = Reduction(MPI.COMM_WORLD).max(np.abs(grad))
        # abs_grad = np.linalg.norm(grad)
        abs_grad = np.sqrt(Reduction(MPI.COMM_WORLD).sum(grad ** 2))
        norms__ = [phi, phi_change, max_grad, abs_grad, x, t]
        if callback is not None:
            callback(norms__)
        is_root = MPI.COMM_WORLD.rank == 0
        # Report progress
        if is_root:
            print('-------------- >>%d = %.5f' % (t, phi))

        if (max_grad < gtol):
            if is_root:
                print("CONVERGED because gradient tolerance was reached")
            return [x, phi, t]
        if (abs(phi_change) <= ftol * max((1, abs(phi), abs(phi_old)))):
            if is_root:
                print("CONVERGED because function tolerance was reached")
            return [x, phi, t]
        phi_old = phi

    return [x, phi, t]


