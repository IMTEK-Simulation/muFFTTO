This file lists all changes to the code

Unreleased
----------

* `muFFTTO/density_filter.py`: Helmholtz density filter (exact FFT solve) and Heaviside projection with chain rule
  and beta continuation, an alternative to the phase-field regularization (tests: `test/test_density_filter.py`,
  docs: theory section 7.5). `RobustFilterAndProjection`: eroded / intermediate / dilated designs (robust
  formulation, minimum feature size).
* Fix: `sensitivity_stress_and_adjoint_FE_NEW` (elasticity) and its conductivity counterpart measured the relative
  adjoint CG tolerance (`r_tol=True`, default) against the residual of the warm start instead of the right-hand
  side. When the warm start was already (nearly) the adjoint solution -- the objective evaluated twice at one
  design, or at the tiny steps of a backtracking line search -- that residual was round-off, CG iterated in noise
  and returned a wrong adjoint (sensitivities off by more than 100 %, or "Hessian is not positive definite").
  L-BFGS then stalled. Now `|r| <= cg_tol |b|`. Test: `test/test_adjoint_warm_start.py` (fails on the old code).

v0.0.1
------

* Initial release