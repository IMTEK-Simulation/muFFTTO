# Linear elastic TO from a bow-tie initial design: status

Started 2026-10-10 (user: standard linear elastic TO, initial design = bow-tie geometry, test its Poisson
ratio, then use it as the initial guess and tune the geometry for different targets).

## Files

| file | what |
|---|---|
| `example_2D_elasticity_TO_bowtie.py` | everything (docstring = formulation). Bow-tie = `geometry.reentrant_honeycomb` (square cell, `--theta` 30, `--thickness` 0.15). `--evaluate-only`: homogenized C (Voigt, engineering shear), E_x, E_y, G_xy, nu_xy, nu_yx. `--nu <nu_xy>`: target = orthotropic with the bow-tie's E_x, E_y, G_xy and the new nu_xy; stress mismatch of 3 unit strains (as examples/topology_optimization/example_2D_elasticity_TO.py), SIMP p 2, void 1e-5, phase field (eta = 1 pixel, double well 1), L-BFGS-B. Output `data/<run>/` (history.nc, final.nc, design.png, design_it*.png, properties.txt). |

## Initial design (theta 30, thickness 0.15, solid fraction 0.30)

| grid | E_x | E_y | G_xy | nu_xy | nu_yx |
|---|---|---|---|---|---|
| 64^2 | 0.0244 | 0.0072 | 0.00079 | -1.604 | -0.471 |
| 128^2 | 0.0241 | 0.0073 | 0.00092 | -1.600 | -0.484 |

Positive definite targets need |nu_xy| < sqrt(E_x/E_y) = 1.83.

## Checks

- Gradient vs central differences (32^2, random rho): rel. error 2e-6 / 2e-8 / 5e-10 (h = 1e-3..1e-5), also the
  mismatch term alone.
- CG tolerance 1e-3 (original example) + adding the adjoint energy to the objective: line search stalls, the
  "mismatch" became negative. Now cg_tol 1e-8 and the adjoint energy is not added.
  (Folders `*_20261010-073502` (crash) and `*_20261010-073541` (stalled, stopped) are these first attempts.)

## Runs (64^2, w 5, dw 1)

nu_xy = -1.0, -0.5, 0.0, +0.5 started 2026-10-10 07:37: folders `*_20261010-073727`.
- Grey designs (07:37 runs, p 2, dw 1, w 5): the target is very compliant (E ~ 2 % of solid); uniform grey
  rho 0.14 has the same stiffness at phase-field cost 0.9 vs 8.4 for the crisp bow-tie (perimeter). Not a bug
  of the mechanics. With p 3 and dw 10 grey costs 24.5.
- p 3, dw 10, w 50 without normalization (07:45): all four collapsed to rho = 0 (first L-BFGS step along a
  gradient of 15 to the bound; dC/drho = 0 at rho = 0 for p > 1). Also the 'moduli' target moved C_11 by 4x.
- Fixes: objective normalized by its initial value; default target `--keep stiffness` (C_11, C_22, C_66 of the
  bow-tie, C_12 = nu C_22). Tool `compare_targets.py <runs>` (overview figure).

## Results (07:48, 64^2, p 3, dw 10, keep stiffness): `data/*_stiffness_w{50,10}_dw10_p3_20261010-074831`

| target nu_xy | w 50 achieved | w 10 achieved |
|---|---|---|
| -1.0 | -1.339 | -1.611 (stopped at 13 it) |
| -0.5 | -0.504 | -1.343 |
|  0.0 | -0.001 | -0.006 |
| +0.5 | +0.493 | +0.519 |

w 50: crisp (grey 11-14 %, interfaces), C_11/C_22 within 2-12 %. Mechanism: inclined walls become wavy
horizontal ligaments; the vertical walls are cut into pieces joined by one-pixel hinges (suspicious: thin
one-node connections, check at 128^2 / with a filter). Overview `data/compare_targets_20261010-075613.png`.

## Thin ligaments (user: "very thin ligaments that can't hold anything"), 2026-10-10 08:30

- New options: `--eta <pixels>` (phase-field width; min member size), `--strain` (load amplitude, default 0.05;
  CG tolerances scaled with it, result identical up to CG noise: histories agree to 6-7 digits),
  `--cases` (matched load cases; others free).
- eta 2 / 3 px (all cases): all four targets hit (nu -1.04/-1.06, -0.51, 0.00, +0.51/+0.53), but the vertical
  walls stay pinched/grey at the joints (grey 26-38 %). `compare_targets_20261010-082227/082230.png`.
- eta 2, shear free (`--cases xx yy`): same pinched/grey vertical walls; nu +0.5 failed (0.87, grey).
  `compare_targets_20261010-082730.png`. So the low shear target is not the cause.
- Likely cause: the target stiffness itself. Solid vertical walls of the min thickness are too stiff for the
  bow-tie's C_22 = 0.028 (~2 % of solid), so the optimizer thins or greys them. Options: free stiffness scale
  (match only nu / ratios), thicker initial bow-tie (stiffer target), volume constraint, filter + projection.

## Density filter + projection in the library (user: "implement meaningfully into the library, tested, documented")

- `muFFTTO/density_filter.py`: `HelmholtzFilter` (periodic (I + r^2 L)^-1, exact FFT, r = R / (2 sqrt 3), L = FD
  Laplacian = P1 FE with lumped mass), `heaviside_projection(_derivative)`, `FilterAndProjection` (forward /
  backward chain rule, beta settable), `RobustFilterAndProjection` (eroded 0.75 / intermediate 0.5 / dilated 0.25,
  weighted backward), `measure_of_non_discreteness`. 2D / 3D, MPI.
- Tests `test/test_density_filter.py` (38, all pass): Fourier modes 2D / 3D, sparse direct solve, constants / mean /
  symmetry / bounds, projection limits and derivative, chain rule FD (plain and robust), robust ordering, eroded
  removes a 1-px bar, mpirun -n 1 vs -n 2 (subprocess, OMPI env stripped).
- Docs: theory section 7.5, docs/examples/topology_optimization_bowtie.md (+ README link), CHANGELOG.
- Example: `--regularization filter --filter-radius <px> [--robust] [--beta ...] [--stage-iterations N] [--xtol]`;
  objective refactored (mismatch_and_gradient per physical design; warm-start fields per robust design). FD checks
  O(h^2) for phase field, filter (beta 1, 8) and robust (beta 1, 8).
- `--free-scale` (suggestion 1): Poisson targets hit but stiffness drifts down (nu +0.5: C_11 40x too small,
  islands); not useful.
- `feature_size.py`: min solid / void size in px by morphological opening (largest disk whose opening only trims
  the boundary and keeps the connected pieces); validated on bars (exact), disk, 1-px hinge (1), 6-px link (6);
  bow-tie 5 / 4 px. Phase-field eta 2 designs: min solid 1-3 px (hinges confirmed). In compare_targets.py.
- Bug found: example passed xtol = 1e-3 to L-BFGS -> filter stages stopped after 2-8 iterations (small smooth
  steps). Now `--xtol`, default 0 for the filter. Runs bowtie10 / bowtie11 (filter R3, robust R4 with xtol 1e-3)
  were stopped; restarted 09:10 as `*_robustR4_*` and `*_filterR4_*` (xtol 0).

## Results 13:13 (bowtie12, R = 4 px, p 3, w 50, 64²)

Overviews: `data/compare_targets_20261010-131323.png` (plain filter), `data/compare_targets_20261010-131330.png` (robust).

| target ν | plain filter: ν achieved, min solid/void px | robust: ν achieved, C11/C22 (target 0.0937/0.0278), min solid/void px |
|---|---|---|
| −1.0 | −0.995, 1/1 | −1.21, 0.0915/0.0297, 3/9 |
| −0.5 | −0.500, 1/2 | −1.47, 0.0691/0.0004, 2/2 (vertical links detached) |
| 0.0 | −0.000, 1/1 | still running at 13:13 (β 8, mismatch 26, failing) |
| +0.5 | −0.271 (fails), 1/2 | +0.57 but C22 0.00003 (vertical links detached), 5/10 |

* Plain filter hits ν −1, −0.5, 0 and C exactly but closes with 1 px hinges at β 32.
* Robust R4: thick, crisp members; only ν −1 is usable. For −0.5 and +0.5 the vertical
  members separate from the horizontal waves (E_y ≈ 0), and the mismatch ends far above the initial one
  (2.85). The eroded design probably breaks the thin vertical links, and the mean objective then gives up
  on them. Next ideas: smaller R (2–3 px), a hinge-free initial design (thicker bow-tie), or a volume/
  stiffness lower bound on C22.

14:12: stopped robust R4 nu 0 (PID 1350134, beta 16, mismatch 24, failing). Started bowtie13 robust R3 and R2 (all four targets) at 13:40; at 14:11 all in beta 1, designs still connected. Snapshot: data/progress_robustR2R3_20261010-141213.png

14:20: stopped bowtie13 R2 runs (1-px necks). Started bowtie14: --thickness 0.25 robust R3, all four targets (CPUs 8-11). Target from the t 0.25 bow-tie: C11 0.223, C22 0.059, C66 0.0054, nu_xy -1.29.

14:25: new options --init original (sin guess + 0.5 rand of topology_optimization/example_2D_elasticity_TO.py, clipped), --keep isotropic (2D isotropic, G = --shear-fraction E_solid, nu_xy = --nu exactly; the original example's 3D conversion gives nu -0.33 for -0.5), --gtol (stage 1 stopped after 3 it at gtol 1e-3). bowtie15: init original, nu -0.5, plain filter R3, p3, gtol 1e-8; shear fraction 0.15 (original level, very stiff for an auxetic) and 0.02.
14:28: phase-field comparison runs bowtie15 pf (dw 10, eta 1 px, p3, w50, gtol 1e-8) for both shear fractions.
15:01 snapshot data/progress_init-original_20261010-150140.png: filter G0.02 beta 8 mismatch 1.6e-3, 1 px features; pf G0.15 254 it crisp stars (min solid 3 px), mismatch 3.9; filter G0.15 stuck beta 1 (mismatch 6-7, 26 % grey). bowtie14 (t 0.25) and bowtie13 R3 at beta 1-2.
15:37: stopped bowtie13 robust R3 nu -0.5 (mismatch 14.7 > start 10.1, beta 2) and bowtie14 t0.25 nu 0.5 (25 > best 1.4, beta 2). Finished: init-original G0.02 filter R3: C exact (mismatch 1.7e-11), min solid/void 1/1 px; init-original G0.15 phase field: nu -0.21 (target -0.5), G 0.136 (0.193), min solid 3 px.
16:01: stopped robust R3 nu 0 (thin and t0.25) and nu 0.5 (thin): mismatch climbing in beta 2 (13-26). Robust fails for nu >= 0 targets of the bow-tie stiffness. Started bowtie16: init original, isotropic G0.02, nu -0.5, robust R3 (compare with plain filter run).

## Results 17:54
`data/compare_init-original_20261010-175402.png` (init original, nu -0.5):
* soft G0.02: plain filter R3 exact (nu -0.500), min solid/void 1/1 px; phase field nu -0.485, C within 2 %, min solid/void 1/3 px, 8 % grey.
* stiff G0.15 (original level): both fail similarly: filter nu -0.22, phase field nu -0.21, C66 ~0.12-0.14 vs 0.193. Target likely not attainable.
`data/compare_robust_thick-bowtie_*.png` (bow-tie t 0.25, robust R3):
* nu -1: nu -0.973, C within ~10 %, min solid 5 px, min void 7 px, connected: best result so far.
* nu -0.5: nu -0.82 (missed), min solid 5 px, void 2 px (small pin holes).

## 18:51 report.py
New `report.py` (one row per run: initial, optimized, convergence with beta stages and grey measure, C error bars,
summary). Figures: data/report_init-original_20261010-185100.png, data/report_bowtie_plain-filter_20261010-185103.png,
data/report_bowtie_robust_20261010-185107.png.
Finding: in the robust runs (gtol 1e-3, default) the stages beta >= 4 end after 1-5 iterations (normalized gradient
below gtol), so the design is never re-optimized after sharpening -> mismatch jumps and stays. Started bowtie17:
t 0.25 robust R3 nu -0.5, 0, 0.5 with --gtol 1e-8 (CPUs 4-6).

## 2026-10-10 evening: gradient bug found and fixed (all earlier runs affected)
* Symptom: robust runs stalled at the start of beta stages (line search finds no decrease, |r| jumps 10x after a
  fallback step). Replays from history.nc reproduced the jump with warm starts, not with cold starts.
* Cause (library): adjoint CG in `sensitivity_stress_and_adjoint_FE_NEW` used `rtol` relative to the residual of the
  warm start; near the solution that residual is round-off -> wrong adjoint (up to >100 % gradient error).
  Fixed in muFFTTO/topology_optimization.py and topology_optimization_conductivity.py (|r| <= tol |b|);
  regression test test/test_adjoint_warm_start.py (4/5 fail on old code, all pass now); 98 TO/filter tests pass.
* Example: state CG tolerance now also relative to |b| (absolute tol could fall below round-off); adjoint cg_tol no
  longer multiplied by the free-scale factor.
* New `check_gradients.py`: FD check of every term (filter, projections, filter+projection, each load case, each
  robust design, free scale, full plain/robust/phase-field objectives, phase-field gradient and double-well terms).
* All runs before this fix used wrong gradients after short line-search steps; results above are not trustworthy
  for the late beta stages (plain filter mostly worked, robust stalled). Killed all running jobs at ~20:50.

20:34: started night runs (fixed gradients, gtol 1e-6): robust R3 from t0.25 and t0.15 bow-tie (nu -1,-0.5,0,0.5), init-original G0.02 nu -0.5 robust R3 / plain R3 / phase field. Logs scratchpad/night_*.log.

### 2026-10-11 01:01: first night results (fixed gradients)
- Thick bow-tie (t 0.25) robust R3, nu -1 (`..._robustR3_..._20261010-203434`): nu_xy -1.024, C11/C22/C12 within 5 %,
  C66 12 %; connected; min solid 7 px, min void 9 px; 600 it, 266 min. Best design so far.
- Original guess, isotropic G 0.02, nu -0.5: plain filter R3 nu -0.502 (all C < 0.3 %) but 1-px features;
  phase field nu -0.484 (C within 2 %), min solid/void 1/2 px.
- Figure: `data/report_fixed_first3_20261011-010139.png`.
- Launched 01:02 on CPUs 11-13: thick bow-tie robust R2, nu -0.5 / 0.0 / 0.5 (logs `night_t0.25_robustR2_nu*.log`).

### 2026-10-11 03:40: robust R3 misses nu -0.5 / 0 / +0.5 -> new option --robust-shape
- Thick bow-tie robust R3: nu -0.5 -> -0.733, nu 0 -> -0.132, nu +0.5 -> +1.66 (only nu -1 on target).
  Thin (t 0.15) robust R3 nu 0 -> -1.47. Per-design mismatch at the end: the intermediate is small, eroded and dilated
  3-20x larger and dominate the mean. The eroded / dilated designs are necessarily softer / stiffer and cannot match one
  absolute C, so the mean drags the intermediate off target.
- Stopped (not promising): thin robust R3 nu -0.5 and +0.5, thick robust R2 nu 0.
- `--robust-shape`: the eroded and dilated designs are matched up to their own optimal stiffness factor (closed form, as
  --free-scale, envelope theorem), the intermediate design absolutely. FD check (check_gradients.py, new entry): rel. error 2e-7.
- Launched 03:50: thick bow-tie robust-shape R3 (CPUs 0-3) and R2 (CPUs 6,7,9,10), nu -1 / -0.5 / 0 / +0.5,
  logs `night_t0.25_robustshapeR{3,2}_nu*.log`.
- Report of the 4 thick robust R3 runs: `data/report_thick_robustR3_20261011-034909.png`; feature sizes nu -1 7/9, -0.5 5/5, 0 3/3, +0.5 1/4 px (solid/void).

### 2026-10-11 06:45: robust-shape hits the targets exactly, but loses the feature-size guarantee
- Thick bow-tie robust-shape: R3 nu -1 -> -1.0004 (mismatch 2e-6), nu -0.5 -> -0.5000 (7e-11); R2 nu -1 -> -1.0000,
  nu -0.5 -> -0.5000. All C entries within 2e-4. Figure `data/report_thick_robustshape_20261011-064318.png`.
- But min solid / void 1-4 / 1 px: 1-px holes and slits survive (the eroded / dilated designs only need the shape of
  C, so holes that the dilated design closes cost almost nothing). Eroded stiffness factor 2-3 for nu -1 / -0.5,
  340 for nu 0 (eroded design nearly disconnected).
- New: `density_filter.length_scale_constraints` (geometric constraints of Zhou, Lazarov, Wang & Sigmund 2015 on
  the filtered field, skeleton indicator exp(-R^4 |grad x|^2)), tests in test_density_filter.py (41 pass), example
  option `--length-scale KAPPA` (penalty, active from `--length-scale-beta` 4). FD check 29/29.
- Stopped the old-formulation robust runs (R2 nu -0.5/+0.5, original guess robust).
- Launched 06:52: thick bow-tie, kappa 100: plain filter R3 + length scale (CPUs 0,1,4,5) and robust-shape R3 + length
  scale (CPUs 6,7,8,11), nu -1 / -0.5 / 0 / +0.5, logs `morning_t0.25_*_ls100_nu*.log`.

### 2026-10-11 07:56: plain filter R3 + length scale (kappa 100): all four targets, clean designs, hinges remain
- nu -1 -> -1.0002, -0.5 -> -0.5000, 0 -> 0.0002, +0.5 -> +0.5001; all C within 5e-4; ~1 h per run (450 it).
  Figure `data/report_thick_filterR3_lengthscale_20261011-075558.png`.
- Min solid / void: nu -1 5/2 px, others 1/1 px. The 1-px features are hinges (corner-to-corner contact of the two
  voids at the centre, necks of the vertical links), `data/where_thin_filterR3_ls100_20261011-0756.png`. The final
  penalty (3e-4 .. 1e-2) is far above the mismatch (1e-6): kappa 100 too weak.
- 07:56: launched plain filter R3 with kappa 1e3 and 1e4 (8 runs, logs `morning_t0.25_filterR3_ls{1000,10000}_nu*.log`).
  Stopped robust-shape (no length scale) R2 nu 0 / +0.5 (eroded design 15-50x softer, nearly disconnected).

### 2026-10-11 09:25: plain filter R3 + length scale, kappa 1e3 / 1e4
| kappa | nu -1 | nu -0.5 | nu 0 | nu +0.5 |
|---|---|---|---|---|
| 100  | -1.0002, 5/2 px | -0.5000, 1/1 | 0.0002, 1/1 | 0.5001, 1/1 |
| 1e3  | -1.0001, 5/2 px | -0.5002, 1/1 | -0.0025, 1/1 | 0.5015, 1/1 |
| 1e4  | -0.9998, **7/7 px** | -0.459, 2/1 | -0.0055, 2/1 | 0.508, 1/2 |
- nu -1 at kappa 1e4: exact (C within 3e-4) and clean, min solid and void 7 px = guaranteed size. Best result.
- nu -0.5 / 0 / +0.5: the hinges (centre crossing of the voids, necks of the vertical links) persist even at kappa 1e4,
  and the fit degrades (nu -0.5 -> -0.459). With C11, C22, C66 of the bow-tie kept, these targets seem to need hinges
  at R 3 px: target and minimum feature size conflict. Options: free stiffness level (--free-scale) or isotropic target.
- Figures `data/report_thick_filterR3_ls{1000,10000}_20261011-0925*.png`, hinge maps `data/where_thin_filterR3_ls*`.

### 2026-10-11 10:38: Poisson-only objective (`--objective poisson`)
- f = weight (nu_xy(C) - nu)^2, nu_xy = -S12/S11, S = C^-1; gradient d(A:Sigma_k)/drho via the library adjoint with
  target Sigma + A and weight -|Sigma + A|^2/2 (`stress_functional_gradient`). FD check ok (robust: needs cg_tol 1e-14,
  the eroded design's nu is noisy at 1e-11).
- Overview `data/overview_poisson_only_20261011-103753.png` (rows: Poisson-only filter R3 + LS 1e4, Poisson-only robust R3,
  stiffness target filter R3 + LS 1e4 for comparison):
  - filter + LS: nu -1 exact (5/1 px, void corner), -0.5 hinges (1/1), **0 exact and clean (3/7 px)**, +0.5 hexagon-like
    (4/2 px, a few islands). 260-410 L-BFGS it, 38-57 min.
  - robust R3: **nu -1 exact, clean, 6/4 px** (199 it, 46 min); nu 0 degenerate: vertical bars only, C11 = C12 = C66 ~ 0
    (nu_xy = 0 trivially). Poisson-only needs a safeguard against losing stiffness in one direction.
    nu -0.5 / +0.5 still running.
- New: `--preconditioner {green-jacobi, green, jacobi}`, CG iterations per evaluation logged (log_cg_state/adjoint),
  `compare_preconditioners.py`, `what_cg_sees.py` (stiffness field seen by CG per beta + cold-start CG counts),
  `overview.py` (methods x targets).

### 2026-10-11 11:30: preconditioners, Poisson-only degeneracy, session closed
- 32^2, random initial guess (--init original), Poisson only nu -0.5, filter R2 + LS 1e4:
  Green-Jacobi 155k CG iterations total (272 per evaluation), 477 s; Green only 1.69M (2109 per evaluation), 2639 s
  (11x iterations, 5.5x time). Jacobi only: ~6x more than Green-Jacobi in the first evaluation, stopped (user: worthless).
  Figures `data/compare_preconditioners_32_20261011-104706.png`, `data/report_preconditioners_32_20261011-112203.png`.
- BUT both 32^2 runs collapsed within ~5 L-BFGS iterations to vertical stripes (C11 = C12 = C66 = 0, nu reported 0,
  f stuck at 12.5, zero gradient). Same degeneracy: 64^2 robust Poisson-only nu 0 (vertical bars) and nu +0.5
  (all moduli ~5e-5, disconnected). 64^2 robust Poisson-only nu -0.5 finished at nu -0.5000, E_x 0.025, E_y 0.029
  (not inspected yet).
- NEXT (proposed, not done): safeguard for --objective poisson, e.g. penalty if E_x or E_y < 20 % of the initial value
  (or a small-weight stiffness term); rerun the 32^2 random-init case and the 64^2 robust nu 0 / +0.5.
- Left running at close (write their own folders): robust-shape R3 + LS 100, nu 0 and +0.5 (PIDs 1589246, 1589247,
  logs scratchpad morning_t0.25_robustshapeR3_ls100_nu{0.0,0.5}.log, beta 32 stage).
- Best results so far (target + min feature size): stiffness target nu -1 filter R3 + LS 1e4 (exact, 7/7 px);
  Poisson only: nu 0 filter R3 + LS 1e4 (exact, 3/7 px), nu -1 robust R3 (exact, 6/4 px).
  Overviews `data/overview_thick_all_methods_20261011-093614.png`, `data/overview_poisson_only_20261011-103753.png`.

### 2026-10-11 14:00: runs left at close finished
- Overview `data/overview_robustshape_ls100_and_poisson_robust_20261011-140027.png`.
- robust-shape R3 + LS 100 (stiffness target): nu -1 -1.007 (5/2 px), -0.5 -0.513 (2/2), 0 -0.031 (1/1), +0.5 0.549
  (1/1, C 11 % off). LS 100 too weak: hinges everywhere, fit worse than robust-shape without LS. Dead end at kappa 100.
- Poisson-only robust R3: nu -0.5 exact but hinges (2/1 px, small holes at the joints, 319 it, 83 min); nu +0.5 exact
  but thin wavy struts and islands (1/2 px, degenerate-looking); nu 0 = vertical bars (degenerate, overview marks it
  black because nu is trivially 0). Confirms NEXT: Poisson-only safeguard against losing stiffness.
