"""Method of moving asymptotes (Svanberg 1987, with the asymptote update of Svanberg 2002) for
    min f(x)   subject to   x_min <= x <= x_max
(box constraints only, as the inverse-homogenization problem needs). Each iteration replaces f by the separable convex
approximation
    f(x) ~ f(x_k) + sum_j  p_j / (U_j - x_j) + q_j / (x_j - L_j)   (+ const),
    p_j = (U_j - x_j)^2 (1.001 g_j^+ + 0.001 g_j^- + r),   q_j = (x_j - L_j)^2 (0.001 g_j^+ + 1.001 g_j^- + r),
g = grad f(x_k), r = 1e-5 max|g| / (x_max - x_min). Without general constraints its minimizer is explicit per variable,
    x_j = (sqrt(p_j) L_j + sqrt(q_j) U_j) / (sqrt(p_j) + sqrt(q_j)),
clipped to the move limits [alpha_j, beta_j] (exact, the approximation is convex and separable).
Asymptotes: initially x -/+ asy_init (x_max - x_min); then widened (x 1.2) where x_j moves monotonically and tightened
(x 0.7) where it oscillates.

Conservative steps (in the spirit of GCMMA): if the new point does not decrease f, the caller rejects it and calls
retry(), which solves the subproblem again from the same point and gradient with the asymptotes pulled halfway in and
the move limit halved (a more conservative, i.e. more curved, approximation). No new gradient is needed.
"""
import numpy as np


class MMA:
    def __init__(self, n, x_min=0.0, x_max=1.0, move=0.2, asy_init=0.5, asy_incr=1.2, asy_decr=0.7, albefa=0.1):
        self.x_min, self.x_max = np.broadcast_to(x_min, n).astype(float), np.broadcast_to(x_max, n).astype(float)
        self.move, self.asy_init, self.asy_incr, self.asy_decr, self.albefa = move, asy_init, asy_incr, asy_decr, albefa
        self.x_old1 = self.x_old2 = self.low = self.upp = None
        self.iteration = 0
        self._step = None  # (x, gradient, move) of the last update, for retry()

    def update(self, x, gradient):
        """One MMA step from x with gradient grad f(x); returns the new x."""
        x = np.asarray(x, dtype=float)
        x_range = self.x_max - self.x_min
        if self.iteration < 2:
            self.low = x - self.asy_init * x_range
            self.upp = x + self.asy_init * x_range
        else:
            trend = (x - self.x_old1) * (self.x_old1 - self.x_old2)
            factor = np.where(trend > 0, self.asy_incr, np.where(trend < 0, self.asy_decr, 1.0))
            self.low = x - factor * (self.x_old1 - self.low)
            self.upp = x + factor * (self.upp - self.x_old1)
            self.low = np.clip(self.low, x - 10 * x_range, x - 0.01 * x_range)
            self.upp = np.clip(self.upp, x + 0.01 * x_range, x + 10 * x_range)
        self.x_old2, self.x_old1 = self.x_old1, x.copy()
        self.iteration += 1
        self._step = (x, np.asarray(gradient, dtype=float), self.move)
        return self._solve()

    def retry(self):
        """The rejected step again, from the same point, with tighter asymptotes and half the move limit."""
        x, gradient, move = self._step
        x_range = self.x_max - self.x_min
        self.low = x - np.maximum(0.5 * (x - self.low), 0.01 * x_range)
        self.upp = x + np.maximum(0.5 * (self.upp - x), 0.01 * x_range)
        self._step = (x, gradient, move / 2)
        return self._solve()

    def _solve(self):
        x, gradient, move = self._step
        x_range = self.x_max - self.x_min
        alpha = np.maximum.reduce([self.x_min, self.low + self.albefa * (x - self.low), x - move * x_range])
        beta = np.minimum.reduce([self.x_max, self.upp - self.albefa * (self.upp - x), x + move * x_range])
        positive, negative = np.maximum(gradient, 0), np.maximum(-gradient, 0)
        regularization = 1e-5 * np.max(np.abs(gradient)) / x_range + 1e-300
        p = (self.upp - x) ** 2 * (1.001 * positive + 0.001 * negative + regularization)
        q = (x - self.low) ** 2 * (0.001 * positive + 1.001 * negative + regularization)
        sqrt_p, sqrt_q = np.sqrt(p), np.sqrt(q)
        return np.clip((sqrt_p * self.low + sqrt_q * self.upp) / (sqrt_p + sqrt_q), alpha, beta)


class GCMMA:
    """Globally convergent MMA (Svanberg 2002; implementation notes "MMA and GCMMA", 2007), box constraints only.

    Outer iteration k at x_k: asymptotes L, U as in MMA (initially x -/+ 0.5 (x_max - x_min), then x 1.2 / x 0.7 where
    x_j moves monotonically / oscillates, clamped to 0.01 ... 10 (x_max - x_min) from x_j), and the approximation
        f~(x) = r + sum_j p_j / (U_j - x_j) + q_j / (x_j - L_j),
        p_j = (U_j - x_j)^2 (1.001 g_j^+ + 0.001 g_j^- + rho / (x_max_j - x_min_j)),
        q_j = (x_j - L_j)^2 (0.001 g_j^+ + 1.001 g_j^- + rho / (x_max_j - x_min_j)),
    with r such that f~(x_k) = f(x_k), and rho = max(1e-6, 0.1 / n sum_j |g_j| (x_max_j - x_min_j)) at the start of
    each outer iteration. Its minimizer over [alpha, beta], alpha = max(x_min, L + 0.1 (x_k - L)), beta = min(x_max,
    U - 0.1 (U - x_k)), is explicit per variable (no general constraints).
    Inner iterations: the trial point is accepted if the approximation is conservative there,
    f~(x_trial) + 0.5 eps |f(x_k)| >= f(x_trial) (eps relative to f, as J spans many decades here); otherwise rho is increased (Svanberg's update: by the defect
    (f(x_trial) - f~(x_trial)) / d(x_trial), d = sum_j (U_j - L_j) (x_j - x_k)^2 / ((U_j - x_j)(x_j - L_j)(x_max_j - x_min_j)),
    times 1.1, at most x 10) and the subproblem is solved again with the same asymptotes and gradient. A conservative
    trial point always satisfies f(x_trial) <= f~(x_trial) <= f~(x_k) = f(x_k): monotone descent.
    """

    def __init__(self, n, x_min=0.0, x_max=1.0, asy_init=0.5, asy_incr=1.2, asy_decr=0.7, albefa=0.1, eps=1e-7):
        self.x_min, self.x_max = np.broadcast_to(x_min, n).astype(float), np.broadcast_to(x_max, n).astype(float)
        self.asy_init, self.asy_incr, self.asy_decr, self.albefa, self.eps = asy_init, asy_incr, asy_decr, albefa, eps
        self.x_old1 = self.x_old2 = self.low = self.upp = None
        self.iteration = 0

    def outer(self, x, f, gradient):
        """Start outer iteration at x (value f, gradient); returns the first trial point."""
        x = np.asarray(x, dtype=float)
        x_range = np.maximum(self.x_max - self.x_min, 1e-5)
        if self.iteration < 2:
            self.low = x - self.asy_init * x_range
            self.upp = x + self.asy_init * x_range
        else:
            trend = (x - self.x_old1) * (self.x_old1 - self.x_old2)
            factor = np.where(trend > 0, self.asy_incr, np.where(trend < 0, self.asy_decr, 1.0))
            self.low = np.clip(x - factor * (self.x_old1 - self.low), x - 10 * x_range, x - 0.01 * x_range)
            self.upp = np.clip(x + factor * (self.upp - self.x_old1), x + 0.01 * x_range, x + 10 * x_range)
        self.x_old2, self.x_old1 = self.x_old1, x.copy()
        self.iteration += 1
        self.x, self.f, self.gradient, self.x_range = x, f, np.asarray(gradient, dtype=float), x_range
        self.rho = max(1e-6, 0.1 / x.size * np.sum(np.abs(self.gradient) * x_range))
        self.alpha = np.maximum(self.x_min, self.low + self.albefa * (x - self.low))
        self.beta = np.minimum(self.x_max, self.upp - self.albefa * (self.upp - x))
        return self._solve()

    def _approximation(self):
        positive, negative = np.maximum(self.gradient, 0), np.maximum(-self.gradient, 0)
        p = (self.upp - self.x) ** 2 * (1.001 * positive + 0.001 * negative + self.rho / self.x_range)
        q = (self.x - self.low) ** 2 * (0.001 * positive + 1.001 * negative + self.rho / self.x_range)
        r = self.f - np.sum(p / (self.upp - self.x) + q / (self.x - self.low))
        return p, q, r

    def _solve(self):
        p, q, r = self._approximation()
        sqrt_p, sqrt_q = np.sqrt(p), np.sqrt(q)
        self.trial = np.clip((sqrt_p * self.low + sqrt_q * self.upp) / (sqrt_p + sqrt_q), self.alpha, self.beta)
        self.f_approx = r + np.sum(p / (self.upp - self.trial) + q / (self.trial - self.low))
        return self.trial

    def conservative(self, f_trial):
        return self.f_approx + 0.5 * self.eps * abs(self.f) >= f_trial  # eps relative: J spans many decades here

    def inner(self, f_trial):
        """The trial point was not conservative (f(trial) = f_trial): increase rho, return the new trial point."""
        x, z = self.x, self.trial
        d = np.sum((self.upp - self.low) * (z - x) ** 2 / ((self.upp - z) * (z - self.low) * self.x_range))
        d = max(d, 1e-12)
        self.rho = min(1.1 * (self.rho + (f_trial - self.f_approx) / d), 10 * self.rho)
        return self._solve()
