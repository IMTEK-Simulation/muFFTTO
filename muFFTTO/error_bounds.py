"""
Error bounds for stopping preconditioned conjugate gradients (PCG) in FFT-based homogenization.

The homogenized energy E^T A* E of a cell problem is approximated by the energy U_k of the PCG iterate u_k. By
Galerkin orthogonality its error splits into an algebraic and a discretization part,

    U_k - E^T A* E  =  ||u_h - u_k||_K^2  +  E^T (A*_h - A*) E,
       total error       algebraic error       discretization error

and all three are nonnegative. On a single grid, PCG should stop once the algebraic error is a small fraction of
the total error: further iterations cannot improve the homogenized energy by more than that fraction.

This module provides

* :class:`DelayedEnergyErrorEstimate` -- the PT estimate of the algebraic error: the Hestenes-Stiefel lower bound
  sum_{j=l}^{k} alpha_j (r_j, z_j) of ||u_h - u_l||_K^2 with the adaptive delay of Meurant, Papez and Tichy
  (Numer. Algorithms 88, 2021). It needs no bound on the spectrum and is accurate (within a factor of about
  0.16-1.33 of the true error for the Green preconditioner, Ladecky et al.), but it is not guaranteed.
* :class:`TwoSidedBounds2D` -- guaranteed bounds L_k <= E^T A* E <= U_k for 2D scalar (conductivity) problems from
  the primal and the complementary energy principle (Gaynutdinova, Ladecky, Pultarova, Zeman, Comput. Math. Appl.
  220, 2026): U_k - L_k is a guaranteed upper bound on the total error.
* :class:`TotalErrorStoppingRule` -- stop PCG once

      PT_k <= theta (U_k - L*_k)   and   |L*_k - L*_{k-1}| <= theta (U_k - L*_k),

  with L*_k the best lower bound so far. This is the stopping rule of Ladecky et al., "Certified homogenized
  coefficients in FFT-based homogenization" (Sec. 5.1, as implemented in the paper's code, AdaptivePrimalDual
  without dual steps), with the PT estimate in place of the Gauss-Radau bound.
"""
import numpy as np

from muFFTTO.solvers import findS


class DelayedEnergyErrorEstimate:
    """
    PT estimate of the energy error of PCG (Meurant, Papez, Tichy 2021).

    With Delta_j = alpha_j (r_j, z_j) = ||u_{j+1} - u_j||_K^2, the error of the iterate l is bounded from below by

        ||u_h - u_l||_K^2 >= sum_{j=l}^{k} Delta_j.

    The estimate for l is accepted at iteration k once the newest contribution is small,
    S Delta_k / sum_{j=l}^{k-1} Delta_j <= tau, with the safety factor S of :func:`muFFTTO.solvers.findS`; with
    d_min > 0 also the delay k - l must be at least d_min. This is the same estimator as in
    :func:`muFFTTO.solvers.conjugate_gradients_mugrid_experimental` (stop_crit_norm='energy_lower_estim').

    Call :meth:`update` once per PCG iteration with alpha_k and (r_k, z_k).
    """

    def __init__(self, tau=0.25, d_min=0):
        self.tau = tau
        self.d_min = d_min
        self.Delta = []              # Delta_j = alpha_j (r_j, z_j)
        self.curve = []              # curve[j] = sum_{i=j}^{k} Delta_i
        self.l = 0                   # iterate whose error is estimated next
        self.d = 0                   # current delay
        self.estimates = []          # accepted (l, estimate of ||u_h - u_l||_K^2), in order of acceptance

    @property
    def latest(self):
        """The most recent accepted estimate (None before the first one)."""
        return self.estimates[-1][1] if self.estimates else None

    @property
    def initial_error(self):
        """Estimate of the initial error ||u_h - u_0||_K^2 (a lower bound): the sum of all Delta_j so far."""
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


class TwoSidedBounds2D:
    """
    Guaranteed two-sided bounds L_k <= E^T A* E <= U_k on the homogenized energy of a 2D scalar (conductivity)
    problem, from any iterate u_k (Gaynutdinova, Ladecky, Pultarova, Zeman, Comput. Math. Appl. 220 (2026)).

    Upper bound: U_k = < (E + grad u_k) . A (E + grad u_k) >, the energy of an admissible field.

    Lower bound (complementary energy): for every periodic divergence-free flux w,

        E^T A* E >= < 2 w . E - w . A^-1 w >.

    The flux q_k = A (E + grad u_k) is not divergence-free; it is L2-projected onto R grad psi_h,
    R = [[0, 1], [-1, 0]], with psi_h from the same linear-triangle space as u (exactly divergence-free). The
    projection is a Laplace problem, solved by one application of the Green operator with identity reference
    material (one FFT pair). Then w = c + t R grad psi_h with the constant c and the scaling t that maximize the
    bound (a 3 x 3 system, no further FFTs); this contains the projected bound (c = <q_k>, t = 1) and the Reuss
    bound (t = 0).

    Requirements: linear triangles and material data constant per element. Then all integrals are exact and both
    bounds are guaranteed, whatever the accuracy of u_k and of the projection.

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        2D scalar discretization with linear triangles.
    material_data : muGrid field
        Conductivity per quadrature point, ``.s`` of shape (2, 2, q, x, y) (as for apply_system_matrix_mugrid).
    macro_gradient_field : muGrid field
        The macroscopic gradient E at the quadrature points (a gradient-size field).
    macro_gradient : array_like, shape (2,)
        The macroscopic gradient E.
    """

    def __init__(self, discretization, material_data, macro_gradient_field, macro_gradient):
        if discretization.domain_dimension != 2:
            raise ValueError('TwoSidedBounds2D supports only 2D problems')
        self.discretization = discretization
        self.comm = discretization.communicator
        self.E = np.asarray(macro_gradient, dtype=float)
        self.E_field = macro_gradient_field
        self.A = material_data.s                                                     # (2, 2, q, x, y)
        self.A_inv = np.moveaxis(np.linalg.inv(np.moveaxis(self.A, (0, 1), (-2, -1))), (-2, -1), (0, 1))
        self.weights = discretization.quadrature_weights
        # a private copy: get_preconditioner_Green_mugrid returns one shared field for every call, so a later
        # preconditioner would overwrite this one
        self.green_identity = _ArrayField(np.array(discretization.get_preconditioner_Green_mugrid(
            reference_material_data_ijkl=np.eye(2)).s, copy=True))
        self.gradient = discretization.get_gradient_size_field(name='two_sided_bounds_gradient')
        self.rotated_flux = discretization.get_gradient_size_field(name='two_sided_bounds_rotated_flux')
        self.stream_gradient = discretization.get_gradient_size_field(name='two_sided_bounds_stream_gradient')
        self.rhs = discretization.get_unknown_size_field(name='two_sided_bounds_rhs')
        self.stream_function = discretization.get_unknown_size_field(name='two_sided_bounds_stream_function')
        self.reuss = float(self.E @ np.linalg.inv(self._mean(self.A_inv)) @ self.E)

    def _mean(self, field):
        """Cell average <field> of a quadrature-point field (..., q, x, y), |Y| = 1, summed over MPI ranks."""
        local = np.einsum('...qxy,q->...', field, self.weights)
        if np.ndim(local) == 0:
            return self.comm.sum(float(local))
        return np.vectorize(self.comm.sum)(local)

    def upper(self, u):
        """U_k, the energy of E + grad u, and the flux A (E + grad u)."""
        self.discretization.apply_gradient_operator_mugrid(u_inxyz=u, grad_u_ijqxyz=self.gradient)
        strain = self.gradient.s[0] + self.E_field.s[0]                               # (2, q, x, y)
        flux = np.einsum('ijqxy,jqxy->iqxy', self.A, strain)
        return float(self._mean(np.einsum('iqxy,iqxy->qxy', strain, flux))), flux

    def lower(self, flux):
        """The optimized complementary-energy bound from the flux of an iterate (one FFT pair)."""
        d = self.discretization
        # L2 projection of the flux onto R grad psi:  K_I psi = B^T W R^T q
        self.rotated_flux.s[0] = np.stack([-flux[1], flux[0]])                       # R^T q
        d.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=self.rotated_flux,
                                                    div_u_fnxyz=self.rhs, apply_weights=True)
        d.fft.communicate_ghosts(self.rhs)
        d.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=self.green_identity,
                                      input_nodal_field_fnxyz=self.rhs,
                                      output_nodal_field_fnxyz=self.stream_function)
        d.apply_gradient_operator_mugrid(u_inxyz=self.stream_function, grad_u_ijqxyz=self.stream_gradient)
        s = np.stack([self.stream_gradient.s[0, 1], -self.stream_gradient.s[0, 0]])   # R grad psi, <s> = 0
        # w = c + t s maximizes 2 c.E - [c; t]^T Q [c; t]  for  Q [c; t] = [E; 0]
        A_inv_s = np.einsum('ijqxy,jqxy->iqxy', self.A_inv, s)
        Q = np.zeros((3, 3))
        Q[:2, :2] = self._mean(self.A_inv)
        Q[:2, 2] = Q[2, :2] = self._mean(A_inv_s)
        Q[2, 2] = self._mean(np.einsum('iqxy,iqxy->qxy', s, A_inv_s))
        c_t = np.linalg.solve(Q, np.r_[self.E, 0.0])
        return float(2.0 * c_t[:2] @ self.E - c_t @ Q @ c_t)

    def __call__(self, u):
        """(U_k, L_k) for the nodal field u (a muGrid field)."""
        upper, flux = self.upper(u)
        return upper, self.lower(flux)


class _ArrayField:
    """Minimal stand-in for a muGrid field: apply_preconditioner_mugrid only reads ``.s``."""

    def __init__(self, array):
        self.s = array


class TotalErrorStoppingRule:
    """
    Stop PCG when the algebraic error is a small fraction of the total error:

        PT_k <= theta (U_k - L*_k)   and   |L*_k - L*_{k-1}| <= theta (U_k - L*_k).

    PT_k is the latest PT estimate of the algebraic error (:class:`DelayedEnergyErrorEstimate`, for an iterate
    l <= k, whose error is at least that of u_k), U_k and L_k are the two-sided bounds of the iterate u_k
    (:class:`TwoSidedBounds2D`) and L*_k = max(Reuss bound, L_0, ..., L_k) the best lower bound so far; U_k - L*_k
    is a guaranteed upper bound on the total error. The second condition requires the lower bound to have
    stagnated: while it is still far from its limit the interval, and with it the admitted algebraic error, is
    too wide. Before the first PT estimate is accepted the rule cannot fire.

    As in the paper, U_k and L_k are evaluated at every iteration; the lower bound costs one FFT pair, i.e. about
    as much as one PCG iteration.

    Usage with :func:`muFFTTO.solvers.conjugate_gradients_mugrid`::

        bounds = TwoSidedBounds2D(discretization, material, macro_gradient_field, E)
        rule = TotalErrorStoppingRule(bounds, theta=0.1)
        conjugate_gradients_mugrid(..., stop_crit_norm='total_error', stopping_rule=rule)
        lower, upper = rule.best_lower, rule.history['upper'][-1]        # guaranteed interval for E^T A* E

    Parameters
    ----------
    bounds : TwoSidedBounds2D
        Or any object with ``upper(u) -> (U, flux)``, ``lower(flux) -> L`` and the attribute ``reuss``.
    theta : float
        Admitted fraction of the total-error bound for the algebraic error (default 0.1).
    algebraic_estimate : str
        'lower' (default): the PT lower bound itself; 'upper': the PT upper estimate PT / (1 - tau), as the
        paper's code with alg='estimate'. Neither is guaranteed.
    tau, d_min : float, int
        Parameters of the adaptive delay of the PT estimate (defaults 0.25 and 0).
    """

    def __init__(self, bounds, theta=0.1, algebraic_estimate='lower', tau=0.25, d_min=0):
        if algebraic_estimate not in ('lower', 'upper'):
            raise ValueError(f"algebraic_estimate must be 'lower' or 'upper', got {algebraic_estimate!r}")
        self.bounds = bounds
        self.theta = theta
        self.estimate = DelayedEnergyErrorEstimate(tau=tau, d_min=d_min)
        self.scale = 1.0 if algebraic_estimate == 'lower' else 1.0 / (1.0 - tau)
        self.best_lower = bounds.reuss
        self.previous_best_lower = None   # L*_{k-1}
        self.history = {'iteration': [], 'upper': [], 'lower': [], 'best_lower': [], 'algebraic_estimate': []}

    def __call__(self, iteration, u, alpha, rz):
        """
        True if PCG may stop at the iterate u = u_k. alpha and rz are alpha_{k-1} and (r_{k-1}, z_{k-1}), the
        coefficients of the step that produced u_k.
        """
        self.estimate.update(alpha, rz)
        algebraic = None if self.estimate.latest is None else self.scale * self.estimate.latest
        upper, lower = self.bounds(u)
        self.best_lower = max(self.best_lower, lower)
        total_error_bound = upper - self.best_lower
        for key, value in zip(self.history, (iteration, upper, lower, self.best_lower, algebraic)):
            self.history[key].append(value)

        small_algebraic_error = algebraic is not None and algebraic <= self.theta * total_error_bound
        stagnated = (self.previous_best_lower is not None
                     and abs(self.best_lower - self.previous_best_lower) <= self.theta * total_error_bound)
        self.previous_best_lower = self.best_lower
        return small_algebraic_error and stagnated
