"""
Linear solvers of muFFTTO.

* :func:`conjugate_gradients` -- preconditioned conjugate gradients on muGrid fields, stopped on the estimate of the
  energy error ``||x* - x_k||_K`` (Meurant-Papez-Tichy lower estimate with adaptive delay, :class:`EnergyErrorEstimate`).
  The solver of the library: every muFFTTO solve uses it. A residual-based PCG is in muGrid
  (``muGrid.Solvers.conjugate_gradients``).
* :func:`dr_pbcg_mugrid` -- preconditioned block CG with several right-hand sides (DR-PBCG, Meurant & Tichy).

Conventions
-----------
The operator and the preconditioner are *in-place* callables ``hessp(x, Kx)`` and ``P(r, z)`` that read the input
field and write the result into the (preallocated) output field. Dot products are local over the interior of the
fields (muGrid ``linalg``, ghosts excluded) and summed over the MPI ranks with ``comm.sum``, so every rank obtains the
same global scalar and takes the same decisions.
"""
import warnings

import numpy as np
from muGrid import Communicator, GlobalFieldCollection, linalg


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




class EnergyErrorEstimate:
    """
    Estimate of the energy error of PCG (Meurant, Papez, Tichy 2021).

    With Delta_j = alpha_j (r_j, z_j) = ||x_{j+1} - x_j||_K^2, the error of the iterate l is bounded from below by

        ||x - x_l||_K^2 >= sum_{j=l}^{k} Delta_j            (Hestenes-Stiefel, delay d = k - l).

    The delay is adaptive: the estimate for l is accepted at iteration k once the newest contribution is small,
    S Delta_k / sum_{j=l}^{k-1} Delta_j <= tau, with the safety factor S of :func:`findS`; with d_min > 0 also the
    delay k - l must be at least d_min ("adaptive delay, minimum d_min").

    Call :meth:`update` once per PCG iteration with alpha_k and (r_k, z_k). All inputs are global scalars, so every
    MPI rank takes the same decisions.
    """

    def __init__(self, tau=0.25, d_min=0):
        self.tau = tau
        self.d_min = d_min
        self.Delta = []              # Delta_j = alpha_j (r_j, z_j)
        self.curve = []              # curve[j] = sum_{i=j}^{k} Delta_i
        self.l = 0                   # iterate whose error is estimated next
        self.d = 0                   # current delay
        self.estimates = []          # accepted (l, estimate of ||x - x_l||_K^2), in order of acceptance

    @property
    def latest(self):
        """The most recent accepted estimate (None before the first one)."""
        return self.estimates[-1][1] if self.estimates else None

    @property
    def initial_error(self):
        """Estimate of the initial error ||x - x_0||_K^2 (a lower bound): the sum of all Delta_j so far."""
        return self.curve[0] if self.curve else 0.0

    def update(self, alpha, rz):
        """Add the step of one PCG iteration; returns the newly accepted (l, estimate) pairs."""
        self.Delta.append(alpha * rz)
        self.curve = [value + self.Delta[-1] for value in self.curve] + [self.Delta[-1]]
        accepted = []
        if len(self.Delta) > 2:
            safety_factor = findS(self.curve, self.Delta, self.l)
            newest = safety_factor * self.Delta[-1]
            tail = float(np.sum(self.Delta[self.l:-1]))
            while (self.d >= 0 and tail > 0 and newest / tail <= self.tau
                   and len(self.Delta) - self.l >= self.d_min):
                accepted.append((self.l, tail + self.Delta[-1]))
                self.l += 1
                self.d -= 1
                tail = float(np.sum(self.Delta[self.l:-1]))
            self.d += 1
        self.estimates += accepted
        return accepted


def conjugate_gradients(comm, fc, hessp, b, x, P, tol=1e-6, d_min=10, tau=0.25, maxiter=10000, callback=None,
                        stopping_rule=None):
    """
    Preconditioned conjugate gradients, stopped on the estimate of the energy error.

    Solves ``K x = b`` (K symmetric positive definite) from the start ``x`` and stops once the estimate of the energy
    error ``||x* - x_l||_K`` (:class:`EnergyErrorEstimate`: lower estimate with adaptive delay, minimum ``d_min``) is
    below ``tol`` times

    * ``||x_0||_K`` for a warm start (the energy of the start stands in for the energy of the solution: the same
      accuracy at every solve, however good the start already is), or
    * the estimate of the initial error ``||x* - x_0||_K`` for a cold start ``x_0 = 0``.

    The energy norm does not depend on the preconditioner, so iteration counts of different preconditioners are
    comparable, and it controls the error of energies and stresses.

    Periodic problems are singular (rigid translations / constants): the preconditioner must map into the mean-free
    fields (the Green preconditioners do), otherwise the iteration diverges once the rest has converged.

    Parameters
    ----------
    comm : muGrid.Communicator
        Global sums of the dot products (every rank takes the same decisions).
    fc : muGrid field collection
        Holds the four work fields of CG (``cg-residual``, ``cg-preconditioned-residual``, ``cg-search-direction``,
        ``cg-hessian-product``; requesting them again returns the same fields).
    hessp, P : callable
        ``hessp(v, Kv)`` and ``P(r, z)`` write the operator / preconditioner product into the second field.
    b, x : muGrid.Field
        Right-hand side and start vector (overwritten with the solution).
    tol : float
        Relative energy error (default 1e-6). CG also stops when ``(r, z)`` reaches machine precision,
        ``eps^2 (r_0, z_0)`` (small problems get there before the delayed estimate comes in).
    d_min : int
        Minimal delay of the error estimate (default 10). ``0`` is the plain adaptive delay.
    tau : float
        Tolerance of the adaptive-delay test (default 0.25).
    maxiter : int
        Maximal number of iterations; reaching it gives a warning.
    callback : callable, optional
        ``callback(iteration, x.s, r.s, p.s, z.s, estimate)`` for the start (iteration 0) and after every iteration
        (``estimate``: latest accepted estimate of the squared energy error, None before the first one; ``p`` is the
        direction of the step just taken).
    stopping_rule : callable, optional
        ``stopping_rule(iteration, x, alpha, rz) -> bool`` replaces the energy test (e.g.
        :class:`muFFTTO.error_bounds.TotalErrorStoppingRule`).

    Returns
    -------
    x : muGrid.Field
        The solution (the field passed in).
    info : dict
        ``iterations`` (operator products in the loop), ``converged`` (bool), ``estimate`` (latest accepted estimate
        of ``||x* - x_l||_K^2``), ``estimator`` (the :class:`EnergyErrorEstimate`).
    """
    def work_field(name):
        return fc.real_field(name=name, components=(*x.components_shape,), sub_pt='nodal_points')

    r = work_field('cg-residual')
    z = work_field('cg-preconditioned-residual')
    p = work_field('cg-search-direction')
    Kp = work_field('cg-hessian-product')

    def dot(u, v):
        return comm.sum(np.float64(linalg.vecdot(u, v)))

    estimator = EnergyErrorEstimate(tau=tau, d_min=d_min)
    info = dict(iterations=0, converged=True, estimate=None, estimator=estimator)

    hessp(x, Kp)
    start_energy = dot(x, Kp)                     # ||x_0||_K^2, the scale of the test for a warm start
    linalg.copy(b, r)
    linalg.axpy(-1.0, Kp, r)                      # r = b - K x_0
    if dot(r, r) == 0.0:
        return x, info
    P(r, z)
    linalg.copy(z, p)
    rz = dot(r, z)
    rz_start = rz
    round_off = np.finfo(float).eps ** 2          # (r, z) relative to the start at machine precision
    if callback:
        callback(0, x.s, r.s, p.s, z.s, None)

    for iteration in range(1, maxiter + 1):
        hessp(p, Kp)
        pKp = dot(p, Kp)
        if pKp <= 0:
            raise RuntimeError('conjugate_gradients: (p, K p) <= 0, the operator is not positive definite')
        alpha = rz / pKp
        linalg.axpy(alpha, p, x)
        linalg.axpy(-alpha, Kp, r)
        estimator.update(alpha, rz)
        info.update(iterations=iteration, estimate=estimator.latest)
        if stopping_rule is not None:
            done = stopping_rule(iteration, x, alpha, rz)          # rz is still (r_k, z_k) of this step
        else:
            scale = start_energy if start_energy > 0 else estimator.initial_error
            done = estimator.latest is not None and estimator.latest < tol ** 2 * scale
        if done and callback is None:
            return x, info
        P(r, z)
        if callback:
            callback(iteration, x.s, r.s, p.s, z.s, estimator.latest)
        if done:
            return x, info
        rz_new = dot(r, z)
        if rz_new <= round_off * rz_start:        # solved to machine precision (small problems: before the
            return x, info                        # delayed estimate comes in); further steps are noise
        linalg.axpby(1.0, z, rz_new / rz, p)    # p = z + beta p
        rz = rz_new

    info['converged'] = False
    if comm.rank == 0:
        warnings.warn('conjugate_gradients did not converge in maxiter iterations', RuntimeWarning)
    return x, info


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
        # float() copies each (possibly GPU) scalar to the host -> G is always a small numpy matrix
        return np.array([[float(dot(u_list[i], v_list[j])) for j in range(m)]
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


