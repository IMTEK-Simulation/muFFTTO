This file lists all changes to the code

Unreleased
----------

* Consolidation (2026-11-18, `maintenance/CONSOLIDATION_PLAN.md`): muFFTTO is the readable prototyping library;
  the studies moved to separate repositories (`muFFTTO-dual-preconditioner`, `muFFTTO-multigrid`, `muFFTTO-contact`,
  `muFFTTO-stefanus`, each with a frozen copy of the library at tag `pre-consolidation`).
  - Requires muGrid >= 1.6.1. Global reductions through the muGrid communicator (`Discretization.mpi_reduction` is a
    `GlobalReduction`: whole arrays or along axes, numpy or cupy, on the discretization's own communicator).
  - One CG: `solvers.conjugate_gradients`, stopped on the energy-error estimate (adaptive delay, minimum 10,
    relative tolerance 1e-6, warm start relative to `||x_0||_K`), returns `(x, info)`. Removed
    `conjugate_gradients_mugrid` (residual CG: `muGrid.Solvers.conjugate_gradients`),
    `conjugate_gradients_mugrid_experimental`, `___PCG`, `Richardson`, `adam`, the `cg_stop` / `r_tol` options and
    `topology_optimization.energy_stop_kwargs`.
  - New `muFFTTO/preconditioners.py`: `Green` (muGrid's preconditioner for one node per pixel), `Jacobi`,
    `GreenJacobi` (muGrid), `Dual` (`G K(C_ref C^-1 C_ref) G`), `dual_material`.
  - New `muFFTTO/phase_field_fracture.py`: AT1/AT2 material models (degradation, crack density and damage
    operator, none/Amor/spectral splits, history field); example
    `examples/phase_field_fracture/example_2D_phase_field_fracture_AT1_AT2.py`.
  - `geometry.correlated_random_field` (2D/3D smooth periodic random field).
  - Merged the multigrid preconditioner (`muFFTTO/multigrid.py`) and the fused GPU kernels of the large-3D-TO work.
  - Examples curated to one per feature (26 + AT1/AT2); run outputs are no longer tracked.
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