# Contact + phase-field fracture: status and next steps

Working notes to continue the work on `example_2D_phase_field_fracture_contact.py`
(state of 2026-10-08). The model and the numerical scheme are documented in
`docs/examples/phase_field_fracture_contact.md`.

## Current defaults in the example

| setting | value | note |
|---|---|---|
| element | `bilinear_rectangle` | Q2 works too |
| `mechanics_preconditioner` | `'Green'` in the file, **use `'Green_Jacobi'`** | GJ = D^-1/2 G D^-1/2, D = diag of the current tangent, rebuilt per trust-region solve; 2-3x fewer Hessian products, identical results |
| `SOLVER_GTOL` | `3e-6` | on max\|g\|/s, s = E h^(dim-1) (force scale): resolution independent; the solver sees Pi/s, g/s, H/s (steps unchanged) |
| `SOLVER_INNER_TOL` | `None` = inexact Newton | forcing eta = min(0.5, sqrt(max\|g\|/s)) of NuMPI `tr_newton_bounded`; folder tag `inner-EW`. **Always use it.** |
| `damage_in_medium` | `False` | damage problem assembled on the solid only (grad d . n = 0 at the cavity surface); nodes touching only the medium pinned to d = 0. `True` = old behaviour (folder tag `dmg-all`) |
| `length_scale` | `domain_size[0] / 32` | fixed physically (2026-10-08): 1 px at 32^2, 2 px at 64^2, 4 px at 128^2. Runs before that used 2 px |
| `staggered_tol`, `max_staggered_iterations` | `1e-3`, `300` | 1e-3 kept by decision (see below). At the cap the increment is rejected and retried with half the step (was: accepted silently) |
| output | own folder per run | `exp_data/phase_field_fracture_contact/<geometry>_<load>_<element>_N<n>_Gc<Gc>_prec-<prec>_inner-<tol>[_dmg-all]_<timestamp>/` with `contact_fracture.nc`, `run.log`, `response.png`, `contact_fracture.mp4` |

The plot call at the end of the run strips the MPI launcher variables from the
environment (a child of `mpirun` otherwise aborts in `MPI_Init`).

Plotting: `plot_phase_field_fracture_contact_output.py <run>/contact_fracture.nc [--save] [--animate]`.

## Runs (all in `exp_data/phase_field_fracture_contact/`)

Geometry `sinusoidal_ligaments` (library default parameters), compression
H = [[0, 0], [0, -0.3]], Gc = 0.1, l = 2 px, bilinear.

| folder (prefix `sinusoidal_ligaments_F11-0.3_bilinear_`) | physics | solver | result |
|---|---|---|---|
| `N32_Gc0.1_prec-Green_20261007-223844` | damage everywhere | Green, inner 1e-6 | 35 min serial, peak P11 -1.112 at F11 0.826 |
| `N32_..._prec-Green_Jacobi_inner-1e-06_dmg-all_20261008-064456` | damage everywhere | GJ, inner 1e-6 | 19 min, identical curve |
| `N32_..._prec-Green_Jacobi_inner-1e-06_20261008-064457` | **solid only** | GJ, inner 1e-6 | 30 min, 342k hessp |
| `N32_..._prec-Green_Jacobi_inner-EW_20261008-064456` | **solid only** | GJ, **inexact** | **15 min, 134k hessp**, same curve as above (\|dP\| < 1.4e-5) |
| `N64_Gc0.1_prec-Green_Jacobi_20261007-230639` | damage everywhere | GJ, inner 1e-6 | 2 ranks, 6.1 h, 1.21M hessp, peak -1.408 at 0.826 |
| `N64_Gc0.1_prec-Green_20261007-230639` | damage everywhere | Green, inner 1e-6 | 2 ranks, >9 h, stopped/slow, same curve as GJ |

Also: `contact_fracture_s_gap_F10+0.3_bilinear_N32_Gc0.1_20261007-214229`
(S-gap geometry in shear, stopped after full separation).

Comparisons (scripts and figures) in `exp_data/phase_field_fracture_contact/comparisons/`:
- `compare_iterations.py out.png run1.nc run2.nc ...` - response, Hessian products / staggered
  iterations / CG length / damage CG / wall time per increment.
- `compare_damage_domain.py out.png all.nc solid.nc` - damage everywhere vs solid only.

## Findings

- **Cost** is almost entirely the mechanics CG (Hessian products); the damage solve is negligible.
- **Inexact Newton**: 2x faster wall time, 2.5x fewer Hessian products, same converged answer.
- **Damage on the solid only** (the correct, variational choice): the medium was a damage sink
  that made the surfaces/ligaments tougher. With the fix at 32^2: peak -0.922 at F11 0.850
  (17 % lower, earlier), first crack 0.838 instead of 0.802, cracks start at the ligament
  junctions instead of mid-ligament, more brittle (15 % vs 7 % of the solid with d > 0.9).
  Still the point-symmetric double crack.
- **Staggered loop** is now the main cost driver: up to ~63 staggered iterations (each a full
  mechanics solve to gtol 1e-5) at crack events.
- **Scaling problem**: per Hessian product ~4-6 ms at 32^2 serial but 16-18 ms at 64^2 on
  2 ranks (expected ~8-12 ms). To be profiled (see below).

## Before the 128^2 run (agreed plan)

1. Fix l physically: l = 1/32 (2 px at 64^2, 4 px at 128^2), so 128^2 refines 64^2. **Done.**
2. Check the symmetry break: H10 = +0.01 on 32^2; increase it or use a geometric imperfection
   if it still breaks at two points at once.
3. Cheaper staggered loop: loose mechanics gtol while the damage still changes a lot, tight at
   the end; must not change the curve (test on 32^2).
4. Restart from the last frame of `contact_fracture.nc` (u~, d, H, lam, dlam). **Done** (`restart_file`): new folder `..._restart_<time>`, old frames copied; 32^2 test 0->0.3 + restart ->0.5 bit-identical to a straight run 0->0.5.
5. Instead of stopping at lam ~ 0.9: lower the load to H11 = -0.27 (F11 = 0.73 at lam = 1), with shear
   H10 = 0.01 (decided 2026-10-08).
6. (optional) damage Green preconditioner from the masked operator (its CG doubled).
7. Timing run 64^2 / 2 ranks with all of the above -> estimate 128^2.
8. 128^2: H = [[0, 0], [0.01, -0.3]] (or what step 2 gives), Green_Jacobi, inexact Newton,
   damage on solid only, 4 ranks.

Test scripts are copies of the example with changed parameters; 32^2 runs serially.

## Profiling (2026-10-08)

Files in `profiling/`: `benchmark_kernels.py` (standard muFFTTO elasticity kernels, any N, any
ranks) and `cprofile_contact_N32_lam0.3_serial.txt` (cProfile of the contact example, 32^2,
Green_Jacobi, inexact Newton, up to lam = 0.3). Machine: laptop i7-1260P, hybrid CPU:
4 P-cores (CPUs 0-7 with HT) + 8 E-cores (CPUs 8-15). The old Green 64^2 run occupied 2 cores
during the measurements, so the MPI numbers are somewhat pessimistic.

**1. The contact example wastes most of each Hessian product** (32^2 serial, 4.4 ms per hessp):

| part of hessp | share |
|---|---|
| `split_neo_hookean(..., tangent=True)` - tangent rebuilt on every product | 54 % |
| `set_F` - F rebuilt on every product | 8 % |
| `add_regularization` (HuHu + LuLu; the LuLu part is identically 0 for Q1) | 19 % |
| `apply_system_matrix_mugrid` (B^T A B, the actual work) | 17 % |

x is fixed during a whole Steihaug-CG solve (67 solves vs 2033 products in this profile), so
F and the tangent should be computed once per trust-region iterate and cached: expected ~3x
cheaper hessp; skipping LuLu for Q1 saves a bit more. JAX (discretization_library) only
compiles element matrices at setup (~2 s), irrelevant for long runs.

**2. Standard kernels (serial, ns per pixel)**: FFT Green preconditioner ~40-60; material
contraction (einsum `ddot42`) ~300 at any N; muGrid stencil B ~130 at N <= 64 but ~380 at
128-256 and ~650 at 512, B^T 70 -> 500 (out of cache). K = B^T C B: 690 (N 32-64), ~1050
(128-256), ~1600 (512). The stencil operators and the einsum contraction, not the FFT,
dominate; they are memory bound and get worse per pixel with N.

**3. MPI scaling is poor on this laptop**: K at 512^2: 1 rank 418 ms, 2 ranks 287 ms (1.46x),
4 ranks 182 ms (2.3x); even the purely local material contraction only gains 1.3x / 2.1x
-> shared memory bandwidth, turbo drops with more active cores, ranks landing on E-cores.
Contact example 64^2 to lam 0.3: 20.4 / 15.4 / 11.2 ms per hessp on 1 / 2 / 4 ranks
(1.33x, 1.82x). The 16-18 ms at 64^2 / 2 ranks seen before = serial cost / 1.3, plus two
jobs sharing the machine.

**Done (2026-10-08)**: (a) and (b) below are implemented (`update_tangent` caches F and A per
(x, lam, d); `laplacian_acts` skips L^T L when it is identically zero). 32^2 serial, same settings as
the `inner-EW_20261008-064456` run: bit-identical result (same curve, same 134 325 Hessian
products), 889 s -> 201 s, 6.6 -> 1.5 ms per Hessian product (run folder
`..._inner-EW_20261008-074133`; the 889 s run shared the machine with two other runs, so the
clean gain is ~3-4x). 2 ranks: no deadlock, same iteration counts up to round-off.

**Fixes, in order**: (a) cache F and the tangent per trust-region iterate (example); (b) skip
LuLu for Q1 (example); (c) one job at a time, ranks pinned to P-cores
(`mpirun -n 4 --map-by core --bind-to core`; NOT `--cpu-list 0,2 --bind-to cpu-list:ordered`, which put 2 ranks on the hyperthreads 0 and 1 of one core), proper scaling tests on a
cluster node; (d) library level: compiled material contraction instead of einsum, check the
muGrid stencil performance for large grids.

## CG convergence, Green vs Green_Jacobi (2026-10-08)

32^2 serial runs `..._prec-Green_inner-EW_20261008-081712` and `..._prec-Green_Jacobi_inner-EW_20261008-081712`
(each with `cg_convergence.pkl`); figure `comparisons/N32_cg_convergence_Green_vs_Green_Jacobi_20261008-082404.png`
(script `comparisons/plot_cg_convergence.py`). Logging hooks (test only, inserted into a copy of the
example before the load loop / at the end): `profiling/cg_logging_hook_before_loop.py`,
`profiling/cg_logging_hook_after.py`.

- Diagnostic PCG to 1e-10 on the same Newton system: Green ~800 iterations (0.97 reduction per
  iteration) vs Green_Jacobi ~180-215 (0.88-0.90) before/around the peak; after cracking
  (lam 0.8) 2130 vs 595. Green drops fast to ~1e-2 and then stalls in a long slow tail (modes of the
  1e5 solid/medium contrast and of the damaged zones that a homogeneous reference cannot see);
  the Jacobi scaling removes most of it.
- In the runs (inexact Newton): Green median 92 / mean 133 iterations per solve, Green_Jacobi
  median 25 / mean 60. But GJ needs more trust-region iterations (2151 vs 1606) and 45 % of its
  solves end on the trust-region boundary (Green 24 %): the radius is measured in the
  preconditioner norm, so the outer path differs. Net wall time 352 s (Green) vs 245 s (GJ) = 1.4x,
  not the 4x of a single solve -> the trust-region radius handling with GJ is a possible improvement.

## Resolution-independent tolerances (2026-10-08)

The trust region now works on Pi/s with s = E_solid h^(dim-1), so `SOLVER_GTOL = 3e-6` and the forcing
term are relative (residual stress / E) and mean the same at 32^2, 64^2, 128^2 (before: absolute
gtol 1e-5 = 3.2e-6 relative at 32^2, but 6.4e-6 at 64^2 and 1.3e-5 at 128^2). 32^2 GJ test
`..._prec-Green_Jacobi_inner-EW_20261008-085415`: same curve as before (|dP11| <= 1.4e-5), 227 s,
142k hessp, eta median 0.023 (was 0.039). The trust-radius parameters of NuMPI (delta0 = 0.05,
delta_max = 0.5, times sqrt(n), in the M-norm) are still preconditioner dependent -> likely cause of
the many boundary hits with Green_Jacobi.

## Trust-region norm and staggered tolerance (2026-10-08)

- Green_Jacobi trust-region norm normalized by the (uniform) reference diagonal d_ref = diag K_ref
  (`reference_diagonal`): M_GJ = (D/d_ref)^1/2 K_ref (D/d_ref)^1/2, i.e. the reference energy norm
  where the tangent is the reference one (before it was ~11-26x larger than the Green norm). CG
  iterates unchanged. Effect at 32^2: crack increment 808 -> 317 solves, 27k -> 9.3k CG iterations
  (now 2x cheaper than Green there).
- **Staggered tolerance 1e-3 is not converged after the first crack**: with 1e-3, Green and GJ
  follow different post-crack curves (|dP11| up to 0.09) and the reload is too stiff by up to ~0.1;
  with **1e-4** Green and GJ agree to 1e-5 everywhere (also after the crack), the peak changes by
  5e-4 and the second crack moves from F11 0.755 to 0.751. Cost 2x (stagger 487 -> 1050,
  max 166 per increment). Runs `..._inner-EW_stag0.0001_20261008-093605` (Green 677 s, GJ 474 s);
  figure `comparisons/N32_staggered_tol_1e-3_vs_1e-4_20261008-094813.png`.
  Folder names now carry `_stag<tol>` when staggered_tol != 1e-3.
- 1e-5 check (GJ, `..._stag1e-05_20261008-095245`): vs 1e-4 max |dP11| 9e-5 before and 2.8e-3
  after the crack (mean 2e-4), same peak (-0.92185 vs -0.92190), same crack points, final -0.36318 vs
  -0.36323; +10 % Hessian products (294k vs 267k), up to 229 staggered iterations per increment.
  -> 1e-4 is converged; figure `comparisons/N32_staggered_tol_1e-3_1e-4_1e-5_20261008-095913.png`.
- Decision (2026-10-08, user): keep staggered_tol = 1e-3 as the default (cheaper: 157k vs 267k
  Hessian products; pre-crack response and peak are converged; post-crack branch accepted as approximate).
  Use 1e-4 only for a deliberate convergence check.

## New defaults test: l = 1/32, H = [[0, 0], [0.01, -0.27]], staggered cap 300 with step halving (2026-10-08)

Run `sinusoidal_ligaments_F10+0.01_F11-0.27_bilinear_N32_Gc0.1_prec-Green_Jacobi_inner-EW_20261008-110452`
(32^2 serial, staggered_tol 1e-3): 63 s, 55 increments, 60k Hessian products; the cap was never hit
(max 100 staggered iterations, in the crack increment); only det F halvings after the crack.
- Peak P11 = -1.177 at F11 = 0.827 (was -0.922 with l = 2 px: smaller l -> higher strength,
  AT2 sigma_c ~ sqrt(E Gc / l) predicts x1.41). l = 1 px at 32^2 is under-resolved; 32^2 is a smoke test only.
- One single drop at F11 = 0.806 (P11 -1.158 -> -0.550) instead of the two drops before; the cell then
  slides in shear (P10 jumps from +0.009 to -0.077). Cracks are still two (point-symmetric pair)
  but break in the same increment. Whether H10 = 0.01 or the new l caused the change is not separated.

## 64^2 timing run, new defaults (2026-10-08)

Run `sinusoidal_ligaments_F10+0.01_F11-0.27_bilinear_N64_Gc0.1_prec-Green_Jacobi_inner-EW_20261008-110939`
(2 ranks, `mpirun -n 2 --map-by core --bind-to core`; response.png, contact_fracture.mp4,
response_partial_lam0.925_*.png). 3445 s, 72 increments, 410k Hessian products.
- Peak P11 -1.166 at F11 0.860 (32^2: -1.177 at 0.827); small drops at 0.854/0.849, main drop at
  0.825 (to -0.48), shear sliding after it; second crack pair at F11 0.737 (P11 -0.894 -> -0.304).
- Up to F11 0.75: 671 s, ~2.3 ms per Hessian product. The second-crack increment (inc 69) alone:
  2681 s, 141 staggered iterations, 478 trust-region iterations, 110k Hessian products at 24 ms each
  (10x the normal cost per product, not explained yet; damage CG only 2180 iterations). To investigate
  before 128^2.
- Aborted start with wrong pinning: folder `..._N64_..._20261008-110915` (6 increments, can be deleted).

## 128^2 and 256^2 (2026-10-08), H = [[0, 0], [0.01, -0.27]], l = 1/32, Green_Jacobi, 4 ranks

- 128^2: `..._N128_..._20261008-122957`, stopped by hand at lam 0.9725 (95 frames; 128^2 had reached
  a second drop at F11 0.755 to P11 -0.22). One increment at lam 0.91 took 50 min (trust region hit
  maxiter 500 in staggered iteration 49, step halved). ~6-7 ms per Hessian product (4 ranks); 2 ranks:
  15 ms (test `..._N128_..._20261008-122840`, 9 increments). Killed run `..._20261008-122721` (SIGSTOP on
  mpirun kills the job: pause only the python ranks).
- 256^2: `..._N256_..._20261008-150323`, to lam 0.56 (F11 0.849): 6868 s, 28 increments, ~455 MB per rank.
  To F11 0.849: 64^2 61 s / 29k hessp, 128^2 497 s / 62k, 256^2 5209 s / 112k.
- Resolution: 128^2 and 256^2 agree (curve and damage: corner/outer-junction cracks first, small drop
  at F11 0.876, P11 -1.05 at 0.854). 64^2 (l = 2 px) differs from F11 ~0.88 on (inner ligament
  localizes first) -> l needs >= 4 px; 128^2 is resolved.
- Figure/movie: `comparisons/N64_N128_N256_damage_before_peak_20261008-165817.png`, `..._damage_20261008-165817.mp4`
  (script `comparisons/compare_resolutions.py`).
- 256^2 continued with restart_file from `..._N256_..._20261008-150323` (lam 0.56 -> 1), started 2026-10-08 ~17:52,
  folder `..._N256_..._inner-EW_restart_20261008-175012`; stopped by the user at ~17:55 before its first new
  increment (holds only the 28 copied frames, lam 0.56). Continue later with restart_file = the original
  `..._N256_..._20261008-150323/contact_fracture.nc` (or this copy). Estimate to lam 1: ~23 h on 4 P-cores.
