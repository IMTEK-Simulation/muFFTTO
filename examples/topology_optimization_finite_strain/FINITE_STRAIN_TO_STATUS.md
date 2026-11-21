# Finite strain topology optimization: status

Started 2026-10-09. Combines `examples/topology_optimization/example_2D_elasticity_TO.py` (phase field,
SIMP, adjoint, L-BFGS-B) with the neo-Hookean finite strain cell problem
(`examples/homogenization/finite_strain_elasticity/`).

## Files

| file | what |
|---|---|
| `example_2D_finite_strain_TO.py` | the optimization (formulation and equations in the module docstring). Neo-Hookean (library `NeoHookean`), SIMP on both Lame parameters, P(F; rho) = s(rho) P_1(F). Target: homogeneous neo-Hookean with the K_t, G_t of the small-strain example (nu_t = 0, G_t = 3/20 E). Load cases F = I + 0.1 H (uniaxial x, y, simple shear). State: Newton-CG (inexact, backtracking with det F > 0), warm start from the previous design, fallback load stepping. Adjoint with the tangent of the converged state. Output `data/<run name>/` (history.nc, final.nc, design_*.png, response.png). |
| `check_sensitivity.py` | finite-difference check of the gradient |

## Checks

- Gradient vs central differences (32^2, random rho in [0.3, 0.9]): full objective rel. error
  3.6e-4 / 3.6e-6 / 3.6e-8 / 5e-9 for h = 1e-2..1e-5 (O(h^2)); mismatch term alone (phase-field term
  off) 3.4e-6 / 3.4e-8 / 1.1e-9.

## First run: `data/N32_Green_Jacobi_strain0.1_w5_p2_20261009-094501`

236 L-BFGS iterations, 135 s (serial). Normal stresses matched (P00 0.0367 vs 0.0368); transverse
stress 0.007 instead of 0 (nu_t = 0 not reached), shear 0.0175 vs 0.0193. The phase-field term (1.0)
dominates the objective (mismatch 0.14); design still grey in places. At 10 % strain the response is
almost linear.

## Next

- Larger strains (0.2-0.3) and/or several load levels per load case (match the nonlinear curve).
- Weight / eta continuation for a crisper 0-1 design; 64^2.
- Documentation page in docs/examples once the setup is settled.

## Curve matching: `example_2D_finite_strain_TO_curve.py` (2026-10-09)

- Several strain levels per load case (default 0.05..0.30, 6 levels), target curve
  P = sign(tau)(E1 e + E2 <e - e_c>^2) in the matched components (others free), mismatch summed over all
  (case, level) pairs, scale S_l = max_k |P_t|^2 per case. Each level is its own equilibrium (hyperelastic).
- Equilibrium = energy minimization with NuMPI's trust-region Newton-CG (Steihaug, handles buckling);
  the first Newton-CG version crashed on negative curvature (thin member buckling). Path following over
  the levels when the warm start fails.
- Switch `third_medium_contact` (default False = plain finite strain: SIMP void 1e-4, no regularization,
  tension, J-curve target). True = TMC (RAMP gamma_0 1e-6, HuHu-LuLu, compression, linear-then-contact
  target); NOT working yet: no equilibrium beyond 20 % compression on the initial grey design.
- CLI: `-n`, `--weight`. Post-processing: `plot_deformation.py <run> [--movie]` (2x2 deformed cells along
  the load path per load case + curves).
- CG stopping (adjoint; Newton steps of the single-point script): Meurant-Papez-Tichy lower bound of the
  energy error, adaptive delay with d_min = cg_delay = 10 (user), relative like a relative residual:
  new option `energy_rtol=True` in solvers.conjugate_gradients_mugrid_experimental (tol^2 x the running
  estimate of |x - x_0|_K^2 = sum_j Delta_j; no extra operator application; `energy_atol` = optional floor). The trust-region inner CG (NuMPI) still uses its own
  forcing rule.
- Gradient checks (32^2): O(h^2), rel. error 1e-8..1e-9 for both scripts, also with PT stopping.

Results (32^2, J-curve target):
- weight 5 (`curve_fs_N32_..._w5_20261009-100312`): diamond checkerboard, concave curve; phase-field term dominates.
- weight 50 (`curve_fs_N32_..._w50_20261009-100934`): convex J-curve, fits up to 0.25, 0.039 vs 0.042 at 0.3;
  mechanism: slanted struts rotating toward the load (shear P01 ~ 0.015-0.02 free). 300 iterations (cap).
- 64^2 weight 50 (`curve_fs_N64_..._w50_20261009-100937`): running; wavy (S-shaped) ligaments at it 40.
- Trust-region inner CG (2026-10-09): patched the INSTALLED NuMPI
  (.venv/.../NuMPI/Optimization/BoundedTRNewtonCG.py; the repo ~/Programming/NuMPI has neither `precond`
  nor this; backup of the file before this patch in the session scratchpad): new `inner_stop='energy'`,
  `inner_delay=d` in tr_newton_bounded: Hestenes-Stiefel lower bound with FIXED delay d,
  sum_{k-d..k} Delta <= tol^2 sum_{0..k} Delta (tol = forcing / inner_tol); default 'residual' unchanged.
  Curve script uses it (tr_inner_stop = 'energy', delay cg_delay = 10). Cold-start objective, optimized
  32^2 design: 2090 vs 4241 Hessian products, 56 vs 693 TR iterations, 4.0 vs 9.5 s; random grey design
  601 vs 384 products (the delay costs >= 10 per solve); F equal to 5e-9, gradient to 1.4e-7.
- `inner_stop='energy'` is now the DEFAULT of the patched tr_newton_bounded (user, 2026-10-09): all
  trust-region users switch (contact-fracture regular/adapted, internal_contact examples). Check on the
  contact-fracture example (regular 32^2, lam 0 -> 0.3, damage + contact): identical response (P10, min J,
  max d to all printed digits), 1396 vs 1863 Hessian products, 18.9 vs 23.9 s.

## Third medium contact TO (2026-10-09 afternoon)

Setup: `--contact --init sinusoidal_ligaments --levels 12 --max-strain 0.24 --normalize --target 0.10 2.0 0.10`
(RAMP gamma_0 = 1e-5, HuHu alpha 1e-6, multiplicative predictor, lower-energy start per level, tr_maxiter 2000).
Results 32^2 (tmc4, 13:19):
- w10 dw10 (`..._w10_dw10_..._131916`): 65 it, 9000 s, mismatch 0.056 (dipped to 0.17 mid-run, recovered).
  Crisp bone-shaped voids; curve too stiff below 0.18, too soft above 0.22 (0.069 vs 0.10 at 0.29). Best so far.
- w50 dw10: 26 it, mismatch 0.197; w50 dw1: grey, 0.29. 64^2 w10 dw10: ~25 min/it, stopped at it 10 (0.094).
- New option `--init-file <run folder | final.nc>` (continuation; run tag `restart`).
  Running (tmc5): restarts from the w10 dw10 final design with weight 50 and 100 (dw10).
- tmc5 restart w50 (`..._w50_dw10_..._norm_restart_20261009-172946`): L-BFGS stopped after 6 it (line search
  steps 1/16, 1/64), mismatch 0.040 on the w10 scale (0.056 before); design and curve almost unchanged.
  restart w100: at 18:14 it 24, mismatch 0.028 on the w10 scale, design visually unchanged; still running.
  Conclusion: the double-well term (~5.2) locks the topology; next idea: double well ~3 + filter/projection.
- `--init-file` also accepts history.nc (last 'rho_iterate' frame; works for killed runs).
- tmc6 (18:18, user: "run it on 64"): 64^2 continuations from the stopped 64^2 w10 dw10 run
  (`..._N64_..._w10_dw10_..._sinusoidal_ligaments_20261009-131919/history.nc`, mismatch 0.094), weight 10 and 50,
  dw10, cpus 0 and 4. First evaluation reproduces F 5.5965 / mismatch 0.09436 (9 min).
- 2026-10-10 07:15: 32^2 restart w100 finished (67 it, mismatch 0.025 on the w10 scale; 0.056 vs 0.063 at 24 %).
  64^2 restart w10 stopped after 7 it (no progress, 0.108). 64^2 restart w50 running, it 26, 0.065 (from 0.094):
  much better at large strain (0.051/0.056 vs 0.063 at 24 %; initial 0.036 / no equilibrium in y), but up to
  10 % too stiff at 10-15 %. New tools: `compare_designs.py <run>` (initial geometry vs optimized: designs,
  deformed cells, curves, error), `plot_deformation.py` now uses the run's target/levels and never overwrites.
