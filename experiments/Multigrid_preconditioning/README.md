# Multigrid preconditioning combined with the FFT preconditioners

Question: how can geometric multigrid (MG) and the FFT-based preconditioners of muFFTTO (Green, Green-Jacobi,
dual) be combined, and when does a combination beat the plain FFT preconditioners?

Started 2026-11-13. Self-contained: `problems.py` holds the Paper-1 problems (copied from
`experiments/dual_preconditioner/paper1_problems`, data in `data/`).

## Building blocks (`mg_lib.py`)

* Every grid level is a muFFTTO `Discretization` (linear triangles, anti-diagonal split), N, N/2, ..., 4.
* P = P1 interpolation of the nested triangulations, R = P^T.
* Coarse operator: muFFTTO stiffness on the coarse grid with the material **averaged over the 4 fine
  triangles** of each coarse triangle. This is exactly the Galerkin operator P^T A P (checked to 3e-16 in
  `test_mg_lib.py`), so no stored stencils are needed. In homogenization terms each coarse level is a
  Voigt (KUBC-type) homogenisation of 2x2 cells: too stiff wherever the fine field wants to fluctuate.
* Smoother: Chebyshev polynomial (degree 2 pre/post) in (M A) on [lambda_max / 30, lambda_max], with lambda_max
  from power iteration (Green: the guaranteed bound g_max).
* Coarsest grid 4^2: exact pseudo-inverse.

## Combinations (preconditioners of the library CG on the fine grid)

| name | operator | idea |
|---|---|---|
| `green`, `green_jacobi`, `dual` | one-level FFT | baselines (references per [[preconditioner-reference]]) |
| `mg` | V(2,2), Chebyshev-Jacobi | plain geometric MG |
| `mgS-<M>` | V-cycle, smoother Chebyshev(M A) on every level | **FFT preconditioner as the MG smoother** |
| `bnn<k>-<M>` | Q + (I - QA) M (I - AQ), Q = P V_1^k R | **FFT inside, MG coarse correction outside** (balancing / A-DEF2, Tang et al. 2009) |
| `add<k>-<M>` | M + Q | additive two-level |

All are fixed SPD operators (checked: symmetry ~1e-13, positive on random vectors), so the CG and its
energy stopping test are unchanged.

## Protocol

CG stops on the Meurant-Papez-Tichy lower estimate, relative 1e-6 (squared energy 1e-12), adaptive delay with
minimum 10. In addition `k*` = first iteration whose TRUE relative energy error (vs. a dual reference solved
to (r, z) < 1e-26) is <= 1e-12 -- the delay adds >= 10 iterations, which distorts ratios between fast methods.
Work unit = fine-level K applications + w_G Green + w_T transfers, coarse levels weighted 4^-l.
Measured serial CPU (`timing.py`): w_G ~ 0.2, w_T ~ 0.02-0.07 -- the muFFTTO operator dominates.

## Stage 1: 64^2, P1-P3, contrasts 1e+-1..3 (P3: 1e-1..1e-3, 1e-6)

`run_stage1.sh` -> `exp_data/`, `summarize.py --metric it|kstar|work`.

Figure: `figures/stage1_overview_N64.png` (numbers = CG iterations, colour = work / best FFT preconditioner).

* Sharp two-phase (P1, P2): Green / dual cheapest. MG needs similar iterations (P2 soft: MG with dual smoother
  38 vs dual 107), but one MG iteration costs ~7 K, so work is 2-5x higher. No combination wins on work.
* Graded design (P3): MG iterations contrast-independent (20-22 for kappa 1e-1 ... 1e-6); work 148 vs GJ 92 at 64^2.
* FFT inside + MG coarse correction (`bnn`, `add`): always worse than the FFT preconditioner alone, also with a
  near-exact coarse solve. The coarse projection does not commute with the clustered Green / dual spectrum and
  spreads it.
* FFT as MG smoother (`mgS-*`): fewest iterations on P2, but the most expensive per iteration.

## Stage 2: user's Green two-grid on a phase-field geometry (256^2)

`phasefield_twogrid.py`, `plot_phasefield_twogrid.py` -> `figures/phasefield_twogrid_N256_eta2.png`.
TO design of the phase-field interface study, tanh interface eta = 2h, k = 1e-4, E_xx.
Iterations / work to the true error 1e-12:

| method | iterations | work |
|---|---|---|
| Green-PCG | 528 | 634 |
| Green-Jacobi-PCG | 152 | 182 |
| dual-PCG | 185 | 444 |
| **MG-PCG (Chebyshev-Jacobi, Galerkin)** | **13** | **84** |
| MG-PCG, Chebyshev-Green smoother | 136 | 1024 |
| 2-grid: nu Green-CG on 256 + nu_c Green-CG on 128, (2,2) / (5,5) / (5,20) | not converged | > 3000 (error 1e-5 ... 1e-7) |

Energy spectrum of the error (exact by Parseval, `phasefield_spectra_N256_eta2.npz`): the initial error is 93 % at
|k| < 8. **Green-CG is a low-frequency solver, not a smoother**: after 5 iterations the |k| < 8 band is down
50x, while |k| 32-64 grows 7x; after 20 iterations |k| > 128 has grown 48x. The coarse correction after
Green-CG removes more low frequencies but injects high frequencies (x30-80 in |k| > 64; the Voigt-type Galerkin
coarse operator is too stiff at the interface), which restarted Green-CG then cannot remove. Restarting CG every
2-5 iterations also discards the Krylov space that Green-CG needs (528 iterations in one run).

Consequence: the roles are reversed compared with classical MG -- an FFT preconditioner acts as a global
(coarse-grid-like) correction, the high-frequency interface error needs a local smoother.

## Stage 3: eta sweep at 256^2 (`run_eta_sweep.sh`, `plot_eta_sweep.py` -> `figures/eta_sweep_N256.png`)

Same design, k = 1e-4, iterations / work to the true error 1e-12:

| eta/h | Green | GJ | dual | MG |
|---|---|---|---|---|
| 0 (sharp at quad points) | 39 / 47 | 525 / 630 | 20 / 48 | 15 / 97 |
| 0.25 | 375 / 450 | 509 / 611 | 244 / 586 | 14 / 91 |
| 0.5 | 488 / 586 | 421 / 505 | 253 / 607 | 14 / 91 |
| 1 | 520 / 624 | 272 / 326 | 237 / 569 | 14 / 91 |
| 1.5 | 524 / 629 | 193 / 232 | 213 / 511 | 14 / 91 |
| 2 | 528 / 634 | 152 / 182 | 185 / 444 | 13 / 84 |
| 3 | 524 / 629 | 106 / 127 | 124 / 298 | 12 / 78 |
| 4 | 502 / 602 | 84 / 101 | 80 / 192 | 12 / 78 |

MG is independent of the interface width (12-15 iterations, work 78-97). Sharp interface: Green / dual 2x
cheaper than MG. Any diffuse interface: MG cheapest, by 5-6x at eta/h 0.25-1 (where every FFT preconditioner
is bad), 3.6x at 1.5, 2.2x at 2, 1.3x at 4 (GJ catches up as the interface widens).

## Stage 4: mesh study 64 / 128 / 256 (`run_mesh_study.sh`, `plot_mesh_study.py` -> `figures/mesh_study_64_128_256.png`)

256^2 design block-averaged to 128^2 / 64^2 (same geometry), eta/h 0 ... 8. Iterations / work to the true 1e-12:

| case | Green | GJ | dual | MG |
|---|---|---|---|---|
| sharp, 64 / 128 / 256 | 42 / 40 / 39 | 150 / 277 / 525 | 24 / 21 / 20 | 15 / 15 / 15 |
| eta = 1h, 64 / 128 / 256 | 460 / 523 / 520 | 68 / 141 / 272 | 71 / 179 / 237 | 12 / 13 / 14 |
| eta = 2h, 64 / 128 / 256 | 93 / 491 / 528 | 39 / 80 / 152 | 18 / 76 / 185 | 11 / 12 / 13 |
| fixed eta = 1/32 (2h, 4h, 8h) | 93 / 109 / 120 | 39 / 42 / 44 | 18 / 19 / 19 | 11 / 11 / 10 |

Work at eta = 2h: MG 71 / 78 / 84 vs best FFT 43 (dual) / 96 (GJ) / 182 (GJ) -> MG loses 1.65x at 64, wins 1.2x at
128 and 2.2x at 256. At eta = 1h: equal at 64, MG 2x at 128, 3.6x at 256. Fixed physical eta: everything is
mesh-independent and dual is cheapest (46 vs MG 65).

Reading: MG is mesh- and width-independent (10-15). The FFT preconditioners are mesh-independent under true
refinement of a fixed geometry, but their cost grows ~N at fixed eta/h, because the interface becomes physically
thinner (GJ doubles per refinement; also on the sharp interface, 150 -> 525). So MG's advantage is a phase-field
practice effect (eta tied to h) and grows ~2x per refinement in that setting.

Addendum eta < h (`run_mesh_small_eta.sh`; 256^2 eta 0.25 from the eta sweep, same geometry), work MG vs best FFT:

| eta/h | 64 | 128 | 256 |
|---|---|---|---|
| 0 | MG 1.9x dearer | 2.0x dearer | 2.1x dearer |
| 0.1 | MG 1.6x cheaper | 1.5x | - |
| 0.25 | 1.8x | 3.6x | 5.0x |
| 0.5 | 1.5x | 2.9x | 5.6x |
| 1 | 1.1x | 2.0x | 3.6x |

Already eta = 0.1h triples Green (42 -> 120 at 64^2): the nodal phi is interpolated to the quadrature points, so
the triangles cut by the interface get grey values (~0.1-0.4) no matter how small eta is. The switch is
"exactly two-phase at the quadrature points" (eta = 0 branch thresholds at the quad points) vs anything else.
MG: 14 iterations for every eta < 1 and every grid.

## Stage 5: NEMO scaling 64^2 ... 1024^2 on one L40S (`scaling_case.py`, `sbatch_mg_scaling.sh`, jobs 8801835-41)

`sync_nemo_scaling.sh` -> `exp_data/nemo_scaling/`, `plot_scaling.py` -> `figures/nemo_scaling.png`.
Library CG, energy stop rtol 1e-6, adaptive delay with d_min = 0 (user decision: the floor of 10 doubled MG's cost);
grids only up to 1024 (user). GPU solve time at 1024^2 (iterations):

| eta/h | Green | GJ | dual | MG |
|---|---|---|---|---|
| 0 | 0.57 s (46) | 17.5 s (2530) | 0.31 s (24) | 1.04 s (20) |
| 0.1 | 1.06 s (119) | 13.8 s (2006) | 1.24 s (99) | 1.04 s (20) |
| 0.25 | 2.59 s (353) | 15.5 s (2289) | 3.28 s (264) | 0.99 s (19) |
| 0.5 | 3.37 s (460) | 13.2 s (1930) | 3.48 s (279) | 1.01 s (19) |
| 1 | 3.52 s (504) | 8.7 s (1300) | 3.44 s (281) | 0.82 s (18) |
| 2 | 3.73 s (538) | 5.0 s (737) | 3.39 s (277) | 0.77 s (17) |
| 4 | 3.95 s (568) | 2.5 s (369) | 3.17 s (258) | 0.80 s (17) |
| 8 (= fixed eta 1/128) | 3.97 s (572) | 1.3 s (185) | 2.45 s (200) | 0.73 s (16) |

Scaling at fixed eta/h 64 -> 1024: MG 13-18 -> 16-20 iterations (+~1 per refinement); GJ doubles per refinement;
dual grows to 256^2 then saturates at ~260-280 (eta >= 0.25h); Green flat (~45 sharp, ~500 diffuse).
Fixed physical eta = 1/128: everything mesh-independent (MG 16, GJ 113 -> 185, dual 182 -> 200, Green ~570).
GPU: MG iteration ~45 ms vs Green ~7 ms at 1024^2 (= the operator-count ratio, launch overhead gone).
=> diffuse interfaces: MG 2.6-4.4x faster than the best FFT preconditioner at 1024^2, a roughly constant factor
(Green / dual saturate); eta = 0.1h tie; sharp: dual 3.4x faster than MG.

Other tests (laptop, 256^2 eta 2h, `green_then_mg.py`): k Green-CG then MG-CG = no gain (84 -> 84-90 work);
alternating restarts 2x worse; V-Green-V preconditioner 8 iterations but 113 work. Memory (`mem_check.py`, peak RSS
at 256^2): Green 1.00, GJ 1.11, MG 1.22, dual 1.35.

User idea "multigrid on ONE grid with Green as the coarse correction" (`green_then_mg.py --variants mg SMS`,
256^2, true error 1e-12, work): one cycle = 2 Chebyshev-Jacobi steps, M correction, 2 steps (SPD).

| eta/h | MG | Green | S-Green-S | GJ | S-GJ-S | dual | S-dual-S |
|---|---|---|---|---|---|---|---|
| 2 | 13 / 84 | 528 / 634 | **35 / 182** | 152 / 182 | 179 / 931 | 185 / 444 | 140 / 896 |
| 0.5 | 14 / 91 | 488 / 586 | **42 / 218** | 421 / 505 | 248 / 1290 | 253 / 607 | 111 / 710 |

S-Green-S: 12-15x fewer iterations than Green, 2.3x cheaper than the best one-level FFT preconditioner at
eta = 0.5h, equal to GJ at 2h; plain MG still ~2x cheaper. GJ / dual in the middle (scaled 1/lambda_max) are bad.
Attraction: no coarse grids, only Jacobi + FFT -> runs with muFFTTO's existing MPI/GPU machinery. Mesh scaling untested.

NEMO smooth-Green-smooth (`sgs-green`, jobs 8802909-13), iterations 64 / 128 / 256 / 512 / 1024:
eta 0: 22 / 29 / 49 / 88 / 144; eta 0.5h: 20 / 26 / 47 / 90 / 159; eta 2h: 19 / 23 / 39 / 81 / 159; fixed eta 1/128
(8h @ 1024): 129. NOT mesh-independent: ~x1.8 per refinement, the same for every eta (also sharp) -> the bottleneck is
the void (k = 1e-4), where Green (reference = solid) corrects nothing and only the Jacobi smoother works, which is
mesh-dependent like Jacobi-CG. At 1024^2: 4.1-4.3 s, slower than Green / dual (3.4-3.5 s), 5x slower than MG (0.8 s).
The coarse grids carrying the averaged (void) material are what makes MG scalable.

## Stage 6: topology optimization with MG (`example_2D_elasticity_TO.py` copy + 'MG', `run_TO_test.sh`, `plot_TO_test.py`)

Laptop TO script (= NEMO version + RANDOM_INIT / XTOL; library identical on both) with NEMO production settings
(sbatch_TO2D_start_gpu.sh): contrast 1e4, PF_NORMALIZED=1, PF_WEIGHT=0.2, energy CG with warm starts, eta = 4h;
CG_DELAY=0; 64^2 from the default start. MG hierarchy built once, material updated per design (setup 3.1 s total).
All four runs: 56 evaluations, identical final objective 1.4925.

| | CG / solve | CG total | work | laptop wall |
|---|---|---|---|---|
| Green | 362 | 121512 | 145814 | 147 s |
| Green-Jacobi | 59 | 19680 | **23616** | **44 s** |
| dual | 54 | 18270 | 43848 | 53 s |
| MG | **13** | 4391 | 28542 | 69 s |

64^2: MG 4.5x fewer CG iterations than GJ, but 1.2x more work and 1.6x more wall time (small-grid Python overhead)
-- as in the scaling study, GJ wins at 64^2; MG is expected to win from ~256^2 on the GPU.

## Stage 7: design optimizers for TO (`design_optimizers.py`, `run_design_opt_test.sh`, `plot_design_opt.py`)

Local optimizers (NuMPI / muFFTTO untouched): bounded L-BFGS with H_0 = gamma P M P (reproduces NuMPI's iterates
exactly with M = I) and projected PR+ nonlinear CG; M = Helmholtz smoother (I + eta^2 (-Lap_h))^-1 of the nodal design
gradient via the muFFTTO FFT, P = mask of box-active nodes. TO copy: env DESIGN_OPT, HELM_C. CH start bi_s0, eta = 4h,
production settings, CG_DELAY=0, Green-Jacobi mechanics, xtol stop. Evaluations = 6 linear solves each.

| N | L-BFGS (NuMPI) | L-BFGS + Helmholtz | NCG | NCG + Helmholtz |
|---|---|---|---|---|
| 32 | 65 evals, f 1.36194, 36 s | **41, 1.36192, 26 s** | 119, 1.36192, 52 s | 59, 1.36194, 31 s |
| 64 | 266, f 1.20952, 202 s | **135, 1.17776, 108 s** | 803, 1.21854, 511 s | 458, 1.17648, 296 s |

Helmholtz H_0 halves the evaluations of L-BFGS (1.6x at 32^2, 2.0x at 64^2) and at 64^2 ends 2.6 % lower (plain
L-BFGS stops on xtol at a worse design). NCG is 1.4-3.4x behind L-BFGS in either form -> keep L-BFGS, add the
Helmholtz H_0. Gain grows with N (expected from the eta/h^2 conditioning of the phase-field term): check at 128-256.

Why the Helmholtz H_0 helps (`why_helmholtz.py` -> `figures/why_helmholtz_N64.png`, `run_helm_c_test.sh`):
the objective's curvature over plane design waves k = 0 ... 32 (64^2, converged design) spreads only 5x, and the
Helmholtz M with c = 1 over-corrects it to 18.6x -> NOT a conditioning effect. The raw gradient is spiky (max/mean 18,
57 % of its energy at |k| <= 4) vs smoothed (7.5, 95 %). Smoothing strength HELM_C (evaluations / final f):

| N | plain | c = 1/16 (curvature-matched) | c = 1 | c = 4 |
|---|---|---|---|---|
| 32 | 65 / 1.36194 | 39 / 1.36191 | 41 / 1.36192 | 40 / 1.36193 |
| 64 | 266 / 1.20952 | 332 / 1.18343 | 135 / 1.17776 | **97 / 1.17765** |

Stronger smoothing is better -> the gain is the Sobolev (coherent-step) effect: smoothed directions move whole
regions on the eta scale instead of being step-limited by a few spiky nodes under the projected line search.

Helmholtz strength sweep (`plot_helmholtz.py` -> `figures/helmholtz_c_dependence.png`), evaluations to the stop:

| c (l = sqrt(c) eta) | 0 | 1/64 | 1/16 | 1/4 | 1 | 4 | 16 | 64 |
|---|---|---|---|---|---|---|---|---|
| 32^2 | 65 | 46 | 39 | 46 | 41 | 40 | 69 | 73 |
| 64^2 | 266 | 294 | 332 | 446 | 135 | **97** | **97** | 112 |

Final objectives within 0.01 % at 32^2; at 64^2 every c > 0 ends 1-3 % below plain. Weak smoothing (c <= 1/4) does
not save evaluations at 64^2; c = 4-16 is best (l = 2-4 eta); too strong (l ~ half the cell) degrades again. In
physical units the best l is ~1/8-1/4 of the cell at both grids -> with eta tied to h the best c may grow with N
(the NEMO runs use c = 1).

NEMO, multigrid mechanics, c = 1 (`figures/helmholtz_nemo.png`), evaluations plain -> Helmholtz:
eta 2h: 64 329 -> 141, 128 467 -> 300, 256 761 -> 402, 512 1283 -> 517 (both running);
eta 4h: 64 305 -> 122, 128 471 -> 222, 256 724 -> 322, 512 1052 -> 413 (Helmholtz running, already below plain's
final f). 1.6-2.5x fewer evaluations, same or better designs (one exception: 2h 64^2, +0.8 %).

Designs vs c (`plot_helmholtz_designs.py` -> `figures/helmholtz_designs.png`; IoU of the solid, best periodic shift and
square symmetry): 32^2 all c give the same design (IoU 0.96-0.99 vs plain; raw IoU 0.25-0.93 was only translation),
88 % grey (eta = 1/8 of the cell). 64^2: all c >= 1 give one design (IoU 0.97-0.98 vs c = 4), less grey (54 % vs 60 %)
and 2.6 % lower objective than plain L-BFGS, which ends in a different local optimum (IoU 0.90); weak c in between.

Helmholtz vs interface width (`run_helm_eta_test.sh`, `plot_helmholtz_eta.py` -> `figures/helmholtz_eta.png`),
evaluations plain -> Helmholtz (factor) / final f relative to plain:

| eta, N | c = 4 | c = 16 | c = 64 |
|---|---|---|---|
| 4h, 32 | 65 -> 40 (1.6x) / 0.0 % | 69 (0.9x) / 0.0 % | 73 (0.9x) / 0.0 % |
| 4h, 64 | 266 -> 97 (2.7x) / -2.6 % | 97 (2.7x) / -2.4 % | 112 (2.4x) / -2.8 % |
| 2h, 32 | 206 -> 131 (1.6x) / -4.6 % | 119 (1.7x) / -4.8 % | - |
| 2h, 64 | 272 -> 220 (1.2x) / +0.1 % | 119 (2.3x) / +0.3 % | - |
| 1h, 32 | 281 -> 94 (3.0x) / +0.5 % | - | 95 (3.0x) / -2.6 % |
| 1h, 64 | 446 -> 121 (3.7x) / +0.8 % | - | 274 (1.6x) / -0.3 % |

Helmholtz saves evaluations at every eta (1.2-3.7x), most at the thinnest interface (1h: 3-3.7x); final designs
within about +-1 % of plain or clearly better (2h 32^2, 4h 64^2: 2.5-5 % lower). Neither a fixed c nor a fixed
physical length l is uniformly best; c = 4-16 is a robust choice.

## Library port of the Helmholtz H_0 (2026-11-13, uncommitted)

* NuMPI (`feat/tr_precond` working tree): `l_bfgs_bounded(..., precond=None)`; initial inverse Hessian
  gamma P M P (zero on box-active components), `_twoloop_direction(..., H0=None)`, `_free_mask`; 7 new tests
  (identity, exact inverse = 1 Newton step, box = clip, FFT Helmholtz on a Laplacian problem, n-D shapes, MPI,
  many active bounds). All 180 optimization tests pass.
* muFFTTO: `solvers_nonlinear.minimize(method='l_bfgs', precond=...)` (box bounds, no linear constraint) + tests.
* TO copy: `DESIGN_OPT=lbfgs_lib_helm` uses the library path; it reproduces the local `lbfgs_helm` exactly
  (32/64^2, c = 1 and 4: identical evaluations and objectives).
* Rejected variant: identity on the epsilon-active components (Bertsekas) -- safer on a bound-heavy toy problem
  but it destroys the TO gain (64^2 c = 4: 282 evaluations vs 97), because the spiky gradient returns near 0/1.
* Caveat (documented): with a near-exact inverse of a strongly coupled Hessian and many active bounds the final
  approach to a tight gtol can be slow (P M P is not the inverse of the reduced Hessian).

## Stage 8: 3D (option (a) re-discretised coarse grids), NEMO 32³ / 64³ (l40s) / 128³ (h200)

New random CH geometries (`make_new_CH_geometries.py`), `scaling_case_3d.py`, `sbatch_mg_scaling_3d.sh`,
`plot_scaling_3d.py` -> `figures/nemo_scaling_3d.png`. MG 12-16 iterations everywhere; at 128³ 2.1-2.3x faster than
the best FFT preconditioner on sharp interfaces and 4.3-6.4x at eta/h = 0.5-1, 1.2x at eta = 4h. Details and the full
table: OVERNIGHT_NOTES.md.
