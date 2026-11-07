# ASR gel expansion in a concrete meso-structure (2D, AT2 phase field)

Prototype study: alkali-silica reaction (ASR) damage driven by expanding gel pockets inside aggregates, set up after
the ASR example of Falsafi, Leute, Ladecky, Junge, *Non-convex, ringing-free, FFT-accelerated solver using an
incremental approximate energy functional* (arXiv 2207.10657, Sec. 3.3, Fig. 7, Table 2; material data from
Gallyamov et al., IJSS 2020, `Gallyamov2020_ASR_multiscale_arXiv2006.13859.pdf`). They used crack-band damage; here
AT2 phase field. Purpose: compare the elastic preconditioners **Green**, **Green-Jacobi** and **dual** on many small
damage zones nucleating at gel pockets in a heterogeneous (aggregate/paste stiffness ratio 4.9) medium.

## Model

* Cell 0.1 m x 0.1 m, periodic (length unit 0.1 m, stress unit E_paste = 12 GPa, Gc -> Gc / (12 GPa * 0.1 m)).
* Phases (Table 2 of the paper), plane strain, true plane-strain Lame constants:

  | phase        | E [GPa] | nu   | Gc [J/m^2] | damage                              |
  |--------------|---------|------|------------|-------------------------------------|
  | cement paste | 12      | 0.30 | 60         | AT2                                 |
  | aggregate    | 59      | 0.30 | 160        | AT2                                 |
  | ASR gel      | 11      | 0.18 | (160)      | none: never degraded, does not drive |

* Geometry (`make_asr_geometry.py`, SEED=1): 32 elliptical aggregates (aspect 1-1.5), Fuller grading
  P(d) = (d/d_max)^0.5 with d = 8-20 mm, 43 % area, random sequential addition with a 1.5 mm gap; 12 gel pockets,
  each one cell of the 1/32 grid (3.1 mm square: 1 pixel at 32^2, 2x2 at 64^2, 4x4 at 128^2), inside aggregates,
  >= 3 cells apart; gel area 1.2 % (2.7 % of the aggregate area). Written on a 512^2 grid, every grid samples it at
  its pixels' lower-left nodes, so the gel pockets are identical on all grids.
  Gallyamov et al. used 0.5 mm gel elements at 0.1 % of the aggregate area with eigenstrain up to 6.5 %; at our
  resolution a pocket must be >= 1 pixel at 32^2, so the pockets are ~40x larger in area and the expansion range was
  reduced accordingly (see below). Their 1-16 mm aggregates at 70 % packing (mortar incl. sand) are not resolvable here.
* Gel load: isotropic eigenstrain eps_g I in the gel pixels, tr = 2 eps_g = dV/V, raised linearly
  0 -> DVV_MAX = 0.024 in 48 steps (dV/V step 5e-4). With DVV_MAX = 0.064 (paper value) the 32^2 cell is fully
  damaged by dV/V = 0.016 (because of the larger gel pockets), 0.024 covers nucleation -> saturation.
* AT2 exactly as the base script (`damage_AT2_small_strain/schneider2025_notch/example_2D_phase_field_fracture_AT2.py`):
  hybrid formulation (C_d = g(d) C_0, g = (1-d)^2 + 1e-4), Amor split psi^+, history field, staggered
  elasticity/damage with tol max|d - d_old| < 1e-3, damage CG with Green-Jacobi. psi^+ uses the **elastic** strain
  eps - eps_g. l = 1/16 (6.25 mm) fixed on all grids: l/h = 2, 4, 8 at 32^2, 64^2, 128^2. With this l the AT2
  strength 3/16 sqrt(3 E Gc / l) is 3.5 MPa (paste) and 12.6 MPa (aggregate), close to ft = 3 / 10 MPa of Table 2.
* Free expansion (zero mean stress), exact at every staggered iteration by superposition: with the current damage
  solve 3 unit-macro-strain problems (u_xx, u_yy, u_xy) and the eigenstrain problem u_g (E = 0); then
  <sigma> = C_eff E + <sigma_g> = 0 gives E, and u~ = u_g + sum E_k u_k. 4 elastic solves per staggered iteration,
  all with the same C_d and preconditioner, each warm-started from its own previous solution; C_eff gives the
  stiffness reduction ||C_eff|| / ||C_eff,0|| (Mandel Frobenius norm) for free. Logged <sigma> residual ~1e-15.
* Preconditioners of the elastic solves (argv[1]): Green with C_ref = undamaged paste, Green-Jacobi (2 mu_paste I_sym
  + Jacobi of K), dual G K(C_ref C_d^-1 C_ref) G. C_d is stored as `ScaledMaterial` with two terms
  s_vol P_vol + s_dev P_dev, so the dual's C_ref C_d^-1 C_ref = (2 kappa_r)^2/s_vol P_vol + (2 mu_r)^2/s_dev P_dev
  is exact also for the gel's different Poisson ratio. CG stops on the energy-error estimate, tol 1e-6, delay 10.
* MAX_STAGGERED (env, default 500): a step that hits it is ACCEPTED and logged ("NOT CONVERGED: step ACCEPTED");
  never hit in these runs.

## Files

* `asr_AT2_gel_expansion.py` - the solver (adapted copy of the base script). env: NPIX, SEED, LENGTH_SCALE,
  DVV_MAX, LOAD_STEPS, MAX_STAGGERED, K_RESIDUAL, GEOMETRY_FILE, RUN_SUFFIX.
* `make_asr_geometry.py` - geometry generator -> `geometry/asr_meso_s<SEED>.npy` (0 paste, 1 aggregate, 2 gel).
* `run_asr_grid_study.sh [grids]` - G / GJ / dual on the given grids (default "32 64 128"), 3 processes at once
  (NPAR), CPU, `systemd-run --user --scope -p MemoryMax=4G`. Runs done:
  `./run_asr_grid_study.sh "32 64 128"` (l = 1/16) and `LENGTH_SCALE=0.03125 ./run_asr_grid_study.sh "64 128"`.
  Plots: `python plot_asr_results.py` and `LENGTH_SCALE=0.03125 GRIDS=64,128 python plot_asr_results.py`.
* `plot_asr_results.py` - all figures (env LENGTH_SCALE, GRIDS, SUFFIX).
* `exp_data/ASR_N<N>_<P>_s1_l<l>_iterations.csv` - per staggered iteration: CG of the eigenstrain solve and the
  three unit solves, damage CG, times, damage change, macro strain E, mean-stress check, stiffness ratio, max d.
  `..._steps.npz` - per load step: damage field, E, stiffness ratio, mean CG, staggered iterations.
* `logs/` - run logs. `figures/` - see below.

## Results

All runs: 48 load steps completed, every step converged (max 47 / 109 / 145 staggered iterations per step at
l = 1/16 / 64^2 l = 1/32 / 128^2 l = 1/32; MAX_STAGGERED = 500 never hit), <sigma> = 0 to 1e-14 relative.
The damage path is the same for all three preconditioners (stiffness ratios differ by <= 3e-8).

### Physics (figures `asr_damage_evolution_dual_l*.png`, `asr_stiffness_reduction_l*.png`)

* Damage nucleates at every gel pocket (max d 0.1 at dV/V = 0.0045, 0.5 at 0.008), grows gradually until
  dV/V ~ 0.008, then each pocket's damage zone saturates quickly (0.008-0.011: stiffness ratio 0.85 -> 0.45 at
  l = 1/16) and afterwards grows only slowly (0.24 / 0.32 / 0.38 at dV/V = 0.024 on 32^2 / 64^2 / 128^2).
* The damage zones are compact, nearly round blobs (radius ~ 2-3 l) around the pockets, not radial cracks; only
  close pockets coalesce (pair at (0.36, 0.55)-(0.55, 0.58)), no long cracks through the paste as in Fig. 7 of
  the paper. Halving l (1/32) only shrinks the blobs. Likely cause: the hybrid formulation degrades the elastic
  stiffness isotropically, so the radial compression around the expanding pocket is released together with the
  hoop tension and an isotropic damage ring is the cheapest mechanism; plus the pocket (3.1 mm) is not small
  compared with l (6.25 / 3.1 mm).
* Stiffness reduction is not grid-converged at fixed l after the onset (l/h = 2 softens most): AT2 needs l/h >= 4.

### Preconditioners: mean CG iterations per elastic solve (mean of the 4 solves; `asr_cg_vs_grid_l*.png`)

'before cracking' = staggered iterations with max d < 0.9, 'crack growth' = max d >= 0.9.

l = 1/16 fixed (l/h = 2 / 4 / 8):

| grid  | Green pre / crack | Green-Jacobi pre / crack | dual pre / crack | wall time G / GJ / dual [s] |
|-------|-------------------|--------------------------|------------------|-----------------------------|
| 32^2  | 25.7 / 73.7       | 38.1 / 52.1              | 16.1 / 39.3      | 32 / 29 / 25                |
| 64^2  | 24.5 / 100.0      | 57.9 / 80.3              | 15.2 / 43.5      | 169 / 135 / 151             |
| 128^2 | 24.8 / 104.2      | 96.9 / 138.4             | 15.1 / 48.8      | 1434 / 1991 / 1525          |

l = 1/32 fixed (l/h = 2 / 4):

| grid  | Green pre / crack | Green-Jacobi pre / crack | dual pre / crack | wall time G / GJ / dual [s] |
|-------|-------------------|--------------------------|------------------|-----------------------------|
| 64^2  | 21.7 / 61.5       | 53.6 / 64.9              | 14.8 / 39.5      | 331 / 389 / 379             |
| 128^2 | 21.9 / 78.3       | 91.0 / 115.9             | 14.7 / 53.8      | 2086 / 2749 / 2556          |

* Before cracking (aggregate/paste contrast 4.9, gel) all are grid-independent except Green-Jacobi, which grows
  ~1.7x per refinement (38 -> 58 -> 97); dual ~15, Green ~25.
* During crack growth the dual is lowest everywhere and nearly grid-independent (39 -> 44 -> 49 at fixed l = 1/16;
  per load step it saturates at ~75 on all grids), Green grows to 100-210 per solve late in the run, Green-Jacobi
  ~1.6x per refinement (52 -> 80 -> 138).
* Wall time: one dual iteration costs ~2.2x a Green iteration on CPU (2 FFT pairs + K(C_dual)): 17.4 vs 7.8 ms
  at 128^2, so the dual's ~2x fewer iterations only break even with Green in wall time here (1525 vs 1434 s);
  Green-Jacobi is slowest at 128^2. (The laptop was shared with other jobs: timings indicative only.)

### Figures

* `asr_geometry_l0.0625.png` - the three phases on the exact pixels of 32^2 / 64^2 / 128^2.
* `asr_damage_evolution_dual_l0.0625.png`, `..._l0.03125.png` - damage at dV/V = 0, 0.006, ..., 0.024 (cf. Fig. 7).
* `asr_stiffness_reduction_l*.png` - ||C_eff|| / ||C_eff,0|| vs dV/V (cf. Fig. 8).
* `asr_cg_per_solve_vs_load_l*.png` - mean CG per elastic solve per load step, G / GJ / dual.
* `asr_cg_vs_grid_l*.png` - mean CG per elastic solve vs grid, before cracking and during crack growth.

### Open issues / next steps

* To get radial cracks from the pockets (Fig. 7), use the Amor split also in the elastic operator
  (C(d) = g(d) C^+ + C^-, nonlinear; Newton per staggered iteration, or the secant/'hybrid-with-active-set'
  linearisation) - the dual then needs C_ref C_d^-1 C_ref of the anisotropic tangent (Mandel inverse, as in the
  base script's full-field branch). Alternatively smaller gel pockets (1 pixel at 128^2) with larger eigenstrain.
* The 3 unit-strain solves per staggered iteration cost 3/4 of the elastic work; a cheaper zero-mean-stress
  scheme (update E with the C_eff of the previous iteration, one solve per staggered iteration, converging with
  the staggered loop) would make the study 3-4x cheaper.

## Real-parameter AT2 runs (Falsafi setting, laptop, 2026-11-06)

Same AT2 script with `GEOMETRY_NPZ=geometry/falsafi_s1.npz` (`make_asr_geometry_falsafi.py`: 0.1 m cell, 53 ellipses,
5-26 mm, Fuller, 50 % aggregate; 75 gel sites as counted in their Fig. 7a; each pocket a 1.5625 mm square = 1 px at
64^2, 2x2 at 128^2; at 32^2 one 3.1 mm pixel with eigenstrain x 1/4, i.e. the same expansion volume, and Voigt-mixed
gel/aggregate stiffness). Table 2 materials, zero mean stress (exact superposition), dV/V 0 -> 0.064 in 64 steps
(their Delta eps_eig = 5e-4). l chosen from the materials: AT2 strength 3/16 sqrt(3 E Gc / l) = ft for the paste ->
**l = 8.44 mm** (the aggregate then has 10.9 MPa vs ft = 10 MPa); fixed on all grids, l/h = 2.7 / 5.4 / 10.8.
Run: `GEOMETRY_NPZ=geometry/falsafi_s1.npz RUN_SUFFIX=_real DVV_MAX=0.064 LOAD_STEPS=64 LENGTH_SCALE=0.084375
./run_asr_grid_study.sh "32 64 128"`; plots `SUFFIX=_real DVV_MAX=0.064 LENGTH_SCALE=0.084375 python plot_asr_results.py`;
video `python make_asr_video.py 128 dual` -> `figures/asr_damage_video_N128_dual_real_l0.084375.mp4`.

Stiffness ratio ||C||/||C0|| (dual run; Falsafi Fig. 8a, digitised: 0.93 / 0.84 / 0.71 / 0.52 / 0.38 / 0.29):

| dV/V        | 0.005 | 0.010 | 0.016 | 0.032 | 0.048 | 0.064 |
|-------------|-------|-------|-------|-------|-------|-------|
| 32^2        | 0.89  | 0.57  | 0.22  | 0.044 | 0.025 | 0.017 |
| 64^2        | 0.93  | 0.22  | 0.10  | 0.061 | 0.050 | 0.044 |
| 128^2       | 0.91  | 0.27  | 0.12  | 0.078 | 0.069 | 0.064 |
| 64^2, l = 3.1 mm | 0.98 | 0.77 | 0.39 | 0.27 | 0.24 | 0.23 |

* Up to dV/V = 0.5 % the curve follows the paper; then every aggregate that contains gel is damaged as a whole
  (l = 8.4 mm is comparable to the 5-26 mm aggregates: the damage zones are as large as the aggregates) and the cell
  loses > 90 % of its stiffness by dV/V = 2 % (paper: 0.29 at 6.4 %). No crack network like their Fig. 7.
  With l = 3.1 mm (64^2, l/h = 2) the final stiffness is close (0.23 vs 0.29) but the drop is abrupt at ~1 %
  instead of gradual, and the damage still forms blobs around the gel clusters, not thin cracks.
* All steps converged; 128^2 Green-Jacobi stopped after 40 steps (> 60 min); CG tables use the first 40 steps there.

Mean CG per elastic solve, before (max d < 0.9) / during cracking:

| grid (l/h)    | Green        | Green-Jacobi | dual        |
|---------------|--------------|--------------|-------------|
| 32^2 (2.7)    | 25.2 / 184.3 | 39.2 / 236.7 | 17.3 / 123.0 |
| 64^2 (5.4)    | 28.0 / 211.3 | 65.1 / 230.3 | 16.3 / 70.3  |
| 128^2 (10.8)  | 24.6 / 224.7 | 121.8 / 440.7 | 15.4 / 67.5 |
| 64^2, l = 3.1 mm (2) | 21.6 / 147.0 | 59.7 / 137.8 | 15.6 / 60.7 |

Dual ~3x fewer iterations than Green during cracking and grid-independent; Green-Jacobi doubles per refinement.
Wall time 128^2: dual 962 s (all 64 steps 1292 s), Green 1700 s (40 steps), Green-Jacobi 2917 s (40 steps).

## NEMO mesh-size study (2D and 3D, l = 1.5625 mm fixed) - started 2026-11-06 evening

Script `asr_AT2_mesh_study.py` (generalised copy: DIM 2/3, DEVICE cpu/gpu, parametric geometry evaluated on every
grid, checkpoint/restart, cheap zero-stress control, damage snapshots to `exp_data/<run>_damage/`).

* **l = 1.5625 mm** (LENGTH_SCALE = 1/64), fixed on all grids: aggregates 5-26 mm = 3-16 l, gel pocket 1.5625 mm
  (Falsafi's coarse pixel; Gallyamov 0.5 mm elements) = 1 l. The AT2 strength 3/16 sqrt(3 E Gc / l) is then
  7.0 MPa (paste, ft = 3) and 25 MPa (aggregate, ft = 10), i.e. 2.3-2.5x ft; their crack-band reference size is
  lc = 0.5 mm, which would need h <= 0.25 mm (400^2+).
* Grids: 2D 32^2 ... 1024^2 (l/h = 0.5, 1, 2, 4, 8, 16), 3D 32^3 ... 256^3 (l/h = 0.5, 1, 2, 4). l/h < 2
  (32, 64) is under-resolved (kept, flagged in tables/plots). Gel pockets below 64^d: one pixel, eigenstrain scaled to
  the same expansion volume (see the script docstring).
* 3D geometry `make_asr_geometry_3D.py` (seed 1): 802 ellipsoids 5-25 mm, 34.8 % volume, 3378 gel pockets (same
  pocket / aggregate ratio as 2D).
* Cheap zero-stress control validated locally (2D 64^2 dual, all 64 steps): stiffness ratio differs from the exact
  superposition by <= 1.6e-4, max d by <= 2e-3; 16 % more staggered iterations, 2.7x less time (112 vs 302 s).
* GPU smoke test (job 8657565, 2D 64^2 dual, L40S): CG counts identical to the laptop run step by step.
* `sbatch_asr_gpu.sh <DIM> <N> <P>` (l40s; `-p h200 --mem=200G` for 256^3), 48 h segments, self-continues from
  `<run>_checkpoint.npz` (hourly), MAXSEG 3; outputs `/work/classic/fr_ml1145-muffto/runs/asr/{exp_data,logs,slurm}`
  (symlinked from the experiment folder on NEMO), job list `runs/asr/jobs.txt`.
* `submit_asr_queue.sh` (laptop) feeds the queue: <= 12 ASR jobs on l40s, <= 2 on h200; order 2D 32-256, 2D 512,
  3D 32-128, 2D 1024, then 3D 256^3 on h200 once the 3D 128^3 runs are going. `fetch_asr_results.sh` copies the
  results back (no checkpoints / 3D full fields); `plot_asr_mesh_study.py` makes the figures `figures/asrmesh_*`.

### Interim results (2026-11-07 ~01:30; `./fetch_asr_results.sh && python plot_asr_mesh_study.py` to refresh)

Done: 2D 32^2-512^2 (all 3), 2D 1024^2 dual, 3D 32^3 and 64^3 (all 3); every step converged (no MAX_STAGGERED hit).
Running: 2D 1024^2 Green (step 29/64), Green-Jacobi (13/64, ~25 min per step, will use its 48 h segments);
3D 128^3 G / GJ / dual (steps 18-20 of 64, cracking steps 1-1.4 h each); 3D 256^3 dual + Green pending on h200;
3D 256^3 Green-Jacobi not yet submitted (h200 budget 2; `submit_asr_queue.sh` submits it when a slot frees).

Mean CG per elastic solve, 2D (before cracking max d < 0.9 / cracking), over the load steps all three runs have:

| grid (l/h)        | Green        | Green-Jacobi | dual        | wall G / GJ / dual [s] |
|-------------------|--------------|--------------|-------------|------------------------|
| 32^2 (0.5, u-r)   | 21.9 / 78.7  | 33.6 / 79.0  | 16.0 / 76.7 | 134 / 144 / 198        |
| 64^2 (1, u-r)     | 22.6 / 133.7 | 54.4 / 122.1 | 15.9 / 65.7 | 160 / 161 / 136        |
| 128^2 (2)         | 22.5 / 155.8 | 107.2 / 194.6 | 15.7 / 59.0 | 254 / 328 / 178       |
| 256^2 (4)         | 21.0 / 190.6 | 204.0 / 396.9 | 15.3 / 68.9 | 618 / 1283 / 411      |
| 512^2 (8)         | 20.2 / 254.6 | 381.6 / 911.0 | 15.1 / 76.1 | 2094 / 7491 / 1244    |
| 1024^2 (16), 13 steps | 20.1 / 265.3 | 748.9 / 839.5 | 15.0 / 84.6 | (running)          |

(1024^2 dual, all 64 steps: 15.0 / 76.7 CG, 6323 s; Green 29 steps: 20.1 / 269.2.) u-r = under-resolved (l/h < 2).

3D (32^3, 64^3 all steps; 128^3 first 19 steps): Green 20.1 / 39.7, 19.7 / 93.4, 18.2 / 90.7;
Green-Jacobi 28.5 / 48.9, 42.8 / 72.1, 82.3 / 104.3; dual 15.9 / 38.3, 15.5 / 75.6, 15.1 / 43.9.

Stiffness ratio (2D dual): dV/V = 1 / 1.6 / 3.2 / 6.4 %: 128^2 0.94 / 0.66 / 0.53 / 0.50, 256^2 0.94 / 0.68 / 0.60 / 0.59,
512^2 0.94 / 0.70 / 0.64 / 0.63, 1024^2 0.94 / 0.71 / 0.66 / 0.65 (converging with h, increments 0.08, 0.04, 0.02);
Falsafi Fig. 8a: 0.84 / 0.74 / 0.50 / 0.29. Damage = halos around the gel pockets inside the aggregates that stop
growing after dV/V ~ 1.5 %; no cracks into the paste. Cause (most likely): with the hybrid isotropic degradation the
fully damaged shell around a pocket has stiffness k = 1e-4, so the gel expands into it without loading the
surrounding material any more. The paper's crack band also degrades isotropically but its cracks are single-element
bands; a tension/compression split in the elastic operator (Amor in C(d)) is the next modelling step.

## Phase field with Fourel parameters (2026-11-07, laptop, 2D)

Goal: the AT2 runs above give damage halos around the gel pockets, no propagating cracks. Adapt the phase-field model
to the set-up of Fourel, Gunn, Molinari 2026 (`../damage_AT2_small_strain/fourel.pdf`) and test which ingredient
produces radial, propagating, coalescing cracks; keep the Green / Green-Jacobi / dual comparison. (Their own nonlocal
threshold model is a separate script by another agent: `fourel_nonlocal_damage.py`, outputs `FOUREL_*`.)

Files: `asr_pf_fourel.py` (solver, copy of `asr_AT2_mesh_study.py`; every switch an env variable, VARIANT presets),
`run_pf_fourel.sh <VARIANT> <N> <P> [ENV=...]` (systemd scope, MemoryMax 12G, log `logs/pf/`),
`plot_asr_pf_fourel.py` (figures + tables), `make_asr_pf_video.py` (mp4). Outputs `exp_data/ASRpf_<variant>_N<N>_<P>*`,
figures `figures/ASRpf_*`.

### Set-up

* Geometry `geometry/fourel2D_s1.npz` (`make_fourel_geometry_2D.py`): 35 mm periodic cell, 91 circular aggregates
  1.5-7 mm (57 %), 28 gel pockets d = 0.75 mm inside aggregates (1 %). h = 0.547 / 0.273 / 0.137 mm at 64^2 / 128^2 /
  256^2 (128^2 = their voxel). Per-pocket eigenstrain factor (true pocket area / its pixel area) keeps the expansion
  volume grid-independent. Plane strain, in-plane gel eigenstrain eps_ASR I, 0 -> EPS_MAX, step 0.025 % (their step).
* Mean stress -1 MPa in x and y ("Free", Table 6), shear 0. Superposition is invalid with the nonlinear stress: the
  macro strain E is an unknown of the staggered loop, E <- E - C_eff^-1 (<sigma> - sigma_bar), C_eff = effective
  tangent from 3 unit-strain solves with the frozen (damage, active set) material of the last converged step.
* Amor split in the stress (Fourel eq. 11): sigma = g (K <tr e>+ I + 2 mu e_D) + K <tr e>- I. Piecewise linear and
  positively homogeneous: sigma = C(e) : e with C = s_vol P_vol + s_dev P_dev, s_vol switching on sign(tr e) per quad
  point. Semismooth Newton = solve the linear problem with the active set frozen (PCG, warm start), update the active
  set, repeat (merged with the E update and the damage staggering into one loop). The dual's C_ref C^-1 C_ref =
  (2K_r)^2 / s_vol P_vol + (2 mu_r)^2 / s_dev P_dev stays exact. C_ref (Green, dual) = undamaged paste, Green-Jacobi
  2 mu_paste I_sym. A load step is converged when max |d - d_old| < 1e-3, |<sigma> - sigma_bar| < 1e-4 (relative) and
  no quad point changed its active set. CG: energy-error stop 1e-6, delay 10.
* History monotone also within the staggered iterations (H = max(H, psi+), H starts the step at H_n): with
  max(H_n, psi+) as in the base script the brittle variants 2-cycled (max d alternating 0.75 / 0.49, 500 iterations).
* Driving force: 'vol' (Fourel eq. 13) psi+ = K/2 <tr e>+^2, or 'amor' psi+ = K/2 <tr e>+^2 + mu e_D : e_D.

### Fracture parameters (every variant)

| variant | damage model | materials (E GPa, nu) | Gc [J/m^2] paste / agg / gel | l or b [mm] | threshold | k_res |
|---|---|---|---|---|---|---|
| A `fourel` (Fourel-matched) | AT1 | 12 / 60 / 60, nu 0.2, gel damageable | 0.533 / 1.6 / 1.6 | l = 0.5 | psi_c = 3Gc/(16l) = 200 / 600 / 600 J/m^3 | 1e-4 |
| A, `_k1e-5` | AT1 | same | same | 0.5 | same | 1e-5 |
| B `czm` (real concrete) | Wu PF-CZM, Cornelissen | same | 60 / 180 / 180 | b = 0.5 | ft = 3 / 11.6 / 11.6 MPa (psi0 = ft^2/2E = 375 / 1125 J/m^3), l_ch = E Gf / ft^2 = 80 / 80 / 80 mm | 1e-4 |
| `old` | AT2, hybrid, Amor psi+ | Falsafi: 12 / 59 / 11 (nu 0.3/0.3/0.18), gel never damaged | 60 / 160 / - | l = 1.5625 | none | 1e-4 |
| `hybrid` | A without the stress split | as A | as A | 0.5 | as A | 1e-4 |
| `at2` | A with AT2 | as A | 1.90 / 5.69 / 5.69 (same critical energy 27Gc/(512l) = psi_c) | 0.5 | none | 1e-4 |
| `amor` | A with Amor psi+ (vol + dev) | as A | as A | 0.5 | as A | 1e-4 |
| `softgel` | A with gel 11 GPa, nu 0.18, never damaged | | as A | 0.5 | as A | 1e-4 |
| `czmR` | B with Wu's Rankine-type driving force (sigma_eq = (9 <s1> + sqrt(3 J2)) / 10 of the undamaged stress, Y = sigma_eq^2 / 2E) | as A | 60 / 180 / 180 | b = 0.5 | ft = 3 / 11.6 / 11.6 MPa | 1e-4 |
| `fourelR` | A with the Rankine-type driving force | as A | 0.533 / 1.6 / 1.6 | 0.5 | psi_c on Y | 1e-4 |
| `czm`, `_b1` | B with b = 1 mm (b/h = 3.7 at 128^2) | as A | 60 / 180 / 180 | b = 1.0 | as B | 1e-4 |

* AT1 mapping: the homogeneous AT1 elastic limit is psi+ = psi_c = 3 Gc / (16 l), so Gc = 16 l psi_c / 3 from
  Fourel's psi_c = 200 (paste) / 600 (aggregate, gel) J/m^3. l = 0.5 mm (their l_c = 0.25 mm, crack thickness
  +-2 l_c = 1 mm; AT1 band width 4 l = 2 mm): l/h = 0.91 / 1.83 / 3.66 at 64^2 / 128^2 / 256^2 (64^2 under-resolved).
  With l = 0.25 mm the Gc would be 0.267 / 0.8 J/m^2 (l/h = 1.83 only at 256^2). AT1 is solved with the
  threshold-history substitution H_eff = max(H, psi_c): 2 H_eff d - 3/4 Gc l Lap d = 2 (H_eff - psi_c), linear, zero
  right-hand side below the threshold (no damage before psi+ reaches psi_c); d clipped to [0, 1]. The tiny Gc (vs real
  50-200 J/m^2) is what makes it brittle like their model: psi_c is their calibrated (nonlocal, macro ft = 3 MPa) value.
* PF-CZM (B): Wu 2017 as `../localization_softening_bands/pfczm_concrete.py` (alpha = 2d - d^2, rational omega,
  Cornelissen p = 2, a2 = 1.3868, a3 = 0.6567, a1 = 4 l_ch / (pi b), projected convexified Newton, d_n <= d <= 1), with
  the volumetric driving force and the stress split of this study instead of Wu's Rankine sigma_eq. Onset when
  K/2 <tr e>+^2 > psi0 = ft^2 / (2E); ft_agg from psi0_agg = 3 psi0_paste (Fourel's ratio): 11.6 MPa. Because
  l_ch = 80 mm >> b, a1 ~ 200 and omega(d) ~ 0 already at d ~ 0.1-0.2: a PF-CZM crack is visible in 1 - omega(d), not
  in d; plots and crack counts of B use 1 - omega(d) (> 0.5 = crack).

### Results (2D, laptop; 64^2 ablations, 128^2 variants, 256^2 mesh check)

Figures: `figures/ASRpf_summary_N128.png` (damage on the exact pixels, old / A / B / B-Rankine / Fourel's nonlocal
model of the parallel script, + tr E and CG per solve), `figures/ASRpf_damage_variants_N64_dbg.png` (64^2 ablation),
`figures/ASRpf_damage_variants_N128.png`, `figures/ASRpf_response_N128.png`, `figures/ASRpf_damage_czm_b_check_N128.png`,
`figures/ASRpf_damage_rankine_check_N64.png`, `figures/ASRpf_cg_{fourel,czm}.png`, `figures/ASRpf_mesh_cg_vs_lh_{A,B}.png`;
videos `figures/ASRpf_video_fourel_N128_dual.mp4` (A), `figures/ASRpf_video_czmR_N128_dual.mp4` (B, Rankine).
Damage paths are identical for the three preconditioners (stiffness ratios equal to 4 digits). No load step hit
MAX_STAGGERED (500) after the history fix.

**Which change produces cracks (64^2 ablation, confirmed at 128^2):**

* old model on the Fourel geometry: halos (initiation 0.73 %, blobs radius ~2 l around the pockets).
* **Stress split is decisive.** `hybrid` (A without the split): every pocket gets a fully damaged shell by 0.15 %,
  then nothing grows (stiffness 0.81 at 1 %, tr E ~ 0): the broken shell lets the gel expand freely - the halo mechanism.
* Threshold (AT1 vs AT2) only shifts the onset: AT2 with the same critical energy cracks too, later (0.7 % vs 0.5 %).
* Driving force: Amor psi+ (vol + dev) is wrong here: the deviatoric energy of the confined paste drives broad damage
  bands everywhere from 0.05 %. Fourel's volumetric psi+ gives thin cracks.
* Gel: soft undamageable gel delays cracking (~0.1 %) but gives the same pattern.
* **A (full Fourel set-up)**: damage at the pockets from 0.1 %, cracks leave the pockets at ~0.5 %, run along / through
  aggregates into the paste and link up; 64^2 percolates at 0.675 % (paper free case 0.375 %); 128^2 forms the
  network between 0.55 and 0.75 % with jumps in tr E and stiffness (0.80 -> 0.57 at 0.6 %, 0.55 -> 0.43 at 0.72 %),
  stiffness ratio 0.85 / 0.43 / 0.29 at eps 0.5 / 0.75 / 1 % (paper: 30-60 % drop; ours larger, 2D). tr E 0.04 % at
  1 % (2D, 1 % gel area; the paper's 3D plateau 0.2 % is not comparable directly). k = 1e-5: identical pattern up to
  0.55 % (stopped): k = 1e-4 releases the stress sufficiently.
* **B (PF-CZM, real Gf, volumetric driving)**: cohesive cracks start at 0.15 % but are X-shaped (four arms at exactly
  45 deg from every pocket), grow slowly and link only at ~1.5-3 % (wrap at 2.9 %); stiffness 0.90 / 0.66 / 0.32 at
  1 / 1.5 / 3 %. The X is not a crack-band resolution artefact (b = 1 mm, b/h = 3.7: same X, wider): outside a
  circular pocket under isotropic expansion tr e = 0 (Lame: pure shear, radial compression + hoop tension), so a
  volumetric-only driving force is blind to the hoop tension; damage starts only where the pixelated pocket corners
  create volumetric tension, i.e. on the grid diagonals. Fourel's driving force has the same blindness (their brittle
  threshold hides it because cracks then follow the aggregate interfaces). More quadrature points do not help.
  Fix: a principal-stress (Rankine, Wu's sigma_eq) driving force, `czmR`: cracks leave the pockets in all directions
  and follow the hoop tension, coalesce, wrap the cell at 1.8 % (128^2; 1.0 % at 64^2); stiffness 0.68 / 0.52 / 0.43
  at 1 / 1.5 / 2 %. The same driving force in A (`fourelR`) gives wide bands (threshold too low for sigma_eq).
* Fourel's own nonlocal threshold model (parallel script, FOUREL_N128_MS_free) gives thin percolating cracks already
  at 0.5 %, the closest to A (A is smoother, 2 mm band vs their 1 mm).

**Elastic CG per solve (before / during cracking = load steps ending with max d (1-omega) < / >= 0.9), total elastic
CG incl. unit solves, dual x 2 = cost in Green iterations; wall times on a shared, oversubscribed laptop (indicative)**

| variant, grid, window | Green | Green-Jacobi | dual | total G / GJ / dual(x2) |
|---|---|---|---|---|
| A 64^2, eps <= 1 % | 21.1 / 104 | 54.7 / 134 | 16.4 / 143 | 190k / 242k / 506k |
| A 128^2, eps <= 1 % | 21.8 / 168 | 96.7 / 278 | 16.5 / 156 | 277k / 448k / 474k |
| B 64^2, eps <= 1.5 % | 19.0 / 26.2 | 49.8 / 59.2 | 15.1 / 19.1 | 12.7k / 29.5k / 18.1k |
| B 128^2, eps <= 1.5 % | 18.8 / 27.3 | 93.9 / 118 | 14.8 / 18.4 | 23.5k / 101k / 30.6k |
| B-Rankine 128^2, eps <= 1 % | 19.6 / 25.5 | 106 / 110 | 15.1 / 19.2 | 11.3k / 49.9k / 16.2k |

* A (brittle, tiny Gc, switching tension/compression tangent): cracking solves cost 100-280 CG for all three; the dual
  is NOT better than Green during crack growth (early growth, eps <= 0.5 % at 128^2: 87 vs 36 CG per solve; in the
  percolation steps single iterations were ~220 (dual) vs ~430 (Green)), so in work the dual is
  1.7-2.7x Green. Likely reason: with the stress split a cracked point under compression is stiff in volume but has
  k-soft shear, a mixed stiff/void anisotropy for which neither Green (C_ref = paste) nor the dual's exact inverse is
  a good fit. Green-Jacobi grows ~1.7x per refinement as before.
* Mesh check A at 256^2 (l/h = 3.7, resolved), eps <= 0.5 % (`figures/ASRpf_mesh_cg_vs_lh_A.png`): CG per solve before /
  during cracking Green 19.6 / 31.4, GJ 156 / 152, dual 15.8 / 37.5; work (dual x2) 11.1k / 60.3k / 23.2k. The dual's
  cracking penalty at 64^2 / 128^2 (99 / 87 vs Green 53 / 36) is largely an under-resolution effect (l/h < 2); at
  l/h = 3.7 dual ~ Green per solve, Green-Jacobi grows ~1.6x per refinement. Green is cheapest in work here.
* B (cohesive, 1 - omega spread over thin bands): all solves stay cheap; dual lowest per solve (15-19), Green 19-27,
  GJ ~2x per refinement; in work (dual x2) Green is ~25 % cheaper than the dual.
* Mesh check B at 256^2 (b/h = 3.7), eps <= 1 % (`figures/ASRpf_mesh_cg_vs_lh_B.png`): CG per solve before / during
  cracking Green 18.3 / 23.7, dual 14.4 / 16.2 (both grid-independent), Green-Jacobi 150 / 165 (only to 0.5 %, run
  still going); work Green 24.9k, dual (x2) 33.4k, GJ > 55k. Staggered iterations grow with the grid (180 / 369 / 964).
* Staggered iterations dominate the cost of A: 1400-1600 to eps = 1 % (vs ~370-730 for B); a crack advance needs
  many staggered iterations (unstable brittle propagation).

### NEMO port (prepared 2026-11-07, not submitted except one smoke job)

* `asr_pf_fourel.py` runs 2D and 3D (DIM), CPU or GPU (DEVICE=gpu, cupy), checkpoint at a load-step end every
  CHECKPOINT_SECONDS (3600) -> `<run>_checkpoint.npz`, RESTART_FROM continues (validated: restart reproduces the
  next step bit-for-bit, 64^2 B). 3D: trilinear hexahedra, Voigt/Mandel 6x6, eps_ASR I (3x3), mean stress -1 MPa in
  x, y, z; 3D Rankine via the principal-stress formula of pfczm_concrete.py; damage snapshots = mid-plane z = 1/2 every
  step + the full field every SNAP_FULL (10) steps. OUT_DIR redirects the results. 2D output identical to the laptop
  runs (checked 64^2 A and B). 3D smoke-tested on the CPU (32^3, A, 2 steps, snapshots + checkpoint).
* 3D geometry: `make_fourel_geometry_3D.py` / `geometry/fourel3D_s1.npz` written by the parallel Fourel-nonlocal agent
  (spheres, Fuller, 35 mm cube, 1046 aggregates 1.5-8 mm ~40 %, 1941 gel pockets d = 0.75 mm, 1 %); reused as is.
  Pockets that hit no voxel node (most at 32^3 / 64^3) get their nearest node with eigenstrain x (true volume / voxel):
  harmless because the gel has aggregate stiffness, only the expansion volume matters.
* `sbatch_asrpf_gpu.sh <DIM> <N> <VARIANT> <P>` (l40s, 40G, 48 h; `-p h200 --mem=200G` for 256^3; exclude g3606,g3608),
  runs from `~/software/muFFTTO/experiments/dual_preconditioner/asr_gel_expansion`, results in
  `/work/classic/fr_ml1145-muffto/runs/asr_pf_fourel/{exp_data,logs,slurm}`, job list `jobs.txt`; budget 47 h 30 min,
  self-resubmits from the checkpoint, MAXSEG 3. Defaults: A eps 0 -> 2 % in 80 steps, B 0 -> 3 % in 60 steps.
* `sync_nemo_asrpf.sh push` (scripts + geometry), `sync_nemo_asrpf.sh fetch` (-> `exp_data/nemo_pf/`, `logs/nemo_pf/`).
* `plot_asrpf_mesh.py` -> `figures/ASRpf_mesh_cg_vs_lh_{A,B}.png` (format of the fatigue study: CG per solve before /
  during cracking and total work vs l/h, rows 2D / 3D, grey band l/h < 2), common eps window A <= 0.5 %, B <= 1 %;
  reads `exp_data/` and `exp_data/nemo_pf/`.

## Fourel model (separate script), solver comparison (2026-11-07, laptop 2D + NEMO port)

Question: can our CG preconditioners make the set-up of Fourel, Gunn, Molinari 2026 (`../damage_AT2_small_strain/
fourel.pdf`, nonlocal continuum damage, one Gaussian crack spot per elastic solve, Moulinec-Suquet fixed point)
cheaper than their own solver? Script `fourel_nonlocal_damage.py` (docstring = full description); nothing shared
with the AT2 / phase-field scripts.

### Set-up
* Model as their Sec. 2 / Alg. 1: Amor split in the stress sigma = g (kappa tr+ I + 2 mu dev) + kappa tr- I,
  g = 1 - phi (1 - K_RESIDUAL); psi+ = 1/2 g kappa (tr+)^2, nonlocal psi_nl = alpha * psi+ (normalised Gaussian,
  l_c = 0.25 mm, FFT); while max psi_nl >= psi_c(phase): Gaussian spot (peak 1) at the argmax of psi_nl among the
  points above their threshold, re-solve. Table 5: paste 12 GPa, aggregate = gel 60 GPa, nu 0.2, psi_c 200 / 600 /
  600 J/m^3 (gel damageable). 2D: plane-strain Lame constants, 2D split (kappa = lambda + mu), in-plane eigenstrain.
  phi and materials per pixel. Free loading: <sigma_xx> = <sigma_yy> = -1 MPa, <sigma_xy> = 0 (mixed control);
  eps_ASR 0 -> 5 % in 200 steps.
* Geometry `geometry/fourel2D_s1.npz` (`make_fourel_geometry_2D.py`): 35 mm cell, 91 circular aggregates 1.5-7 mm
  (57 %), 28 gel pockets 0.75 mm (1 %); l_c/h = 0.46 / 0.91 / 1.83 at 64^2 / 128^2 / 256^2.
* Solvers, all on the same bilinear FE discretisation (4 quad points):
  - **MS** = their basic scheme as preconditioned Richardson u <- u - G0 B^T sigma(eps), C0 = mean of the min / max
    undamaged phase moduli, built once (their Alg. 1); mean stress as Kabel et al.: E <- E + C0^-1 (sigma_bar -
    <sigma>); stop on their strain increment ||eps_k - eps_{k-1}|| / ||eps_k|| < TOL_MS (1e-4 = theirs, 1e-6 =
    equal accuracy). Willot's rotated scheme is the same iteration with another discrete gradient (not proven here).
    C0 from the *current* damaged min (MS_C0=current) does not converge: kappa0 -> kappa_max / 2 and the stiff phase
    is no longer contracted (oscillation, rel. increment stuck at 0.15).
  - **Newton + PCG** (Green C_ref = aggregate, Green-Jacobi 2 mu I + Jacobi, dual G K(C_ref C_t^-1 C_ref) G): the
    stress is piecewise linear, Newton = active set on sign(tr eps_e) at every quad point, energy line search when
    the pattern changes (plain active set cycles: 199 iterations, ~2e4 CG per solve), kink stop (<= 2 points on
    tr = 0 and the PCG at its delay floor), CG energy stop 1e-6 / delay 10; mean stress by E <- E + C_eff^-1 r with
    a superposition warm start from 3 unit solves per load step (counted), Broyden update in between.
  - Cost unit = 1 Green-PCG iteration = 1 MS iteration (1 B, 1 B^T, 1 FFT pair; same measured time per iteration on
    one CPU thread); dual counts 2.
* Accuracy: CHECK_EVERY = n measures every n-th solve's energy error ||u - u_ref||_K / ||u_ref||_K against a
  1e-10 Green-CG reference of the same linear problem.

### Results (laptop, one thread per run, load 16-30 on 16 cores: wall times only indicative)
`plot_fourel_mesh.py` -> `figures/FOUREL_mesh_cg_vs_lh.png` (table printed), `make_fourel_summary.py` ->
`figures/FOUREL_summary.png`, `plot_fourel.py` -> `figures/FOUREL_cost_*`, `FOUREL_expansion_free.png`,
`FOUREL_cracks_free.png`, `FOUREL_damage_N*_free.png`, `figures/FOUREL_cost_table_free.txt`;
video `make_fourel_video.py exp_data/FOUREL_N128_Green_free_k0.0001` -> `figures/FOUREL_N128_Green_free_k0.0001.mp4`.

Total cost (Green-iteration units, dual x 2, Newton incl. unit solves) up to the common eps_ASR window of each grid,
with the median measured energy error (`plot_fourel_mesh.py`):

| grid (l_c/h), window | MS tol 1e-4 | MS tol 1e-6 | Newton + Green | Newton + GJ | Newton + dual |
|---|---|---|---|---|---|
| 64^2 (0.46), 1.35 % | 61k (err 6e-3) | 1625k (1e-4) | **549k (2e-7)** | 901k (2e-7) | 2308k (2e-7) |
| 128^2 (0.91), 0.35 % | 26k (2e-3) | 465k (1e-4) | **255k (1e-7)** | - | 790k (1e-7) |
| 256^2 (1.83), 0.2 % | 5.5k (6e-3) | 88k (5e-5) | **38k (8e-8)** | - | 149k (3e-7) |

Cost per elastic solve (units): before cracking MS 22 / 34 (tol 1e-4 / 1e-6), Green 36, GJ 82, dual 63; during
cracking (one solve per spot) MS 14-82 / 850-1290, Green 260-610, GJ 450, dual 1170-2100, and growing with the
open-crack network (128^2 Green: 2000-4500 per solve at eps_ASR 0.5-0.9 %).

* **Answer**: at equal accuracy (MS tol 1e-6, still ~1e-4 energy error vs 1e-7 for CG) Newton + Green is 1.8-3x
  cheaper than their scheme; Green-Jacobi and dual do not help here (dual 3-4x Green: the cracks are 1-2 pixels wide
  at l_c/h <= 1.8, consistent with the earlier finding that dual needs l/h >~ 2-3). But with **their** criterion
  (strain increment 1e-4) MS is 10-20x cheaper than Newton + Green: once cracks are open ||eps|| is dominated by the
  crack-opening strains, the relative increment passes after 1-10 iterations per spot and the solution is ~0.2-1 % off
  in energy. That is how a one-spot-per-solve scheme is affordable for them (their 2D 1024^2 in 5 s on an H100). The
  crack patterns of MS 1e-4 and Green agree up to ~0.25 % and differ in 2-4 % of the pixels at 0.5-1.4 % (64^2);
  expansion and damaged fraction differ by < 1 %.
* Newton needs ~2.5-3 PCG solves per spot (sign switches, the 10-iteration CG delay floor of each solve); the PCG work
  per solve then grows strongly once cracks open (near-void, mixed tension / compression tangent).
* MS with C0 from the current damaged moduli does not converge (see above); C0 fixed as in their Alg. 1 converges.
* K_RESIDUAL 1e-4 vs 1e-6 (64^2, to 1 %): tr E within 4 %, damaged fraction within 2 %; Green 154-184 its / solve
  (k = 1e-6 needed the kink stop), MS error stays ~1e-3.
* Batched spots (BATCH=1, 64^2): to 1 % 2.5x fewer solves, 1.7x less work (Green), tr E within 0.2 %, crack pattern
  differs in 4-5 % of the pixels (MS vs Green one-spot: 2.5 %); beyond 1.4 % (fragmented) more spots, and the batched
  Newton run stalls (thousands of sign flips, mean-stress residual not met; stopped). MS batched runs to 5 %.

Physics vs the paper (3D, free): cracks start at the gel pockets at eps_ASR = 0.1 % (64^2, 128^2) / 0.125 % (256^2)
(paper 0.3-0.5 %), grow radially through the aggregate into the paste and coalesce; first face-spanning crack at
0.425 % (64^2, paper percolation 0.375 %), at 128^2 the network is visually connected at ~0.8 % (MS: spanning at
1.4 %); 64^2 tr E rises to 0.25 % at 1.5 %, falls to 0.17-0.18 % at 2-3 % and rises again to 0.28 % at 5 % (paper
plateau ~0.2 % for 1.5-3 %); 128^2 expands slower (0.12 % at 2 %). Tangent stiffness C_xxxx 30 -> 3 GPa (64^2 at 2 %),
-> 7 GPa (128^2 at 0.9 %): the tangent of the open-crack state, not their compressive-unit-strain stiffness (they
report 30-40 % loss). Uniaxial tension check (128^2, E_xx control): E_eff 27.7 GPa, first spot at <sigma_xx> =
1.9-2.2 MPa -> our 2D cell with their psi_c is ~30 % weaker than their 3 MPa calibration (2D split, 2D cell).
Earlier initiation is consistent with that and with the 2D gel pockets (cylinders) and coarse pocket resolution.


### NEMO port (prepared, not submitted except one smoke job)
* `fourel_nonlocal_damage.py` is device and dimension generic: DEVICE=gpu (cupy), DIM=3 (trilinear hexahedra, 8 quad
  points, 3D split K = lambda + 2 mu / 3, 6 Voigt components, 26-neighbour batch maxima), checkpoint at the end of a
  load step every CHECKPOINT_SECONDS (phi, mask, u, unit solutions, E, records), RESTART_FROM continues and trims the
  CSV, OUT_DIR / MUFFTTO_ROOT for the self-contained NEMO folder, 3D output: mid-plane phi every step, full field
  every SNAP_FULL steps. 2D CPU results identical to the earlier version (CSV diff: round-off 1e-13 in sxy);
  checkpoint/restart reproduces an uninterrupted run exactly.
* `make_fourel_geometry_3D.py` -> `geometry/fourel3D_s1.npz`: 35 mm periodic cube, Fuller spheres d = 1.5-8 mm by
  random sequential addition (1046 aggregates, 45 % - RSA jams below their 64 %), 1941 gel pockets of 0.75 mm =
  1 % of the volume (their text: 2000 pockets in a 70 mm cube = 0.13 %, inconsistent with "1 %"; we keep 1 %).
  128^3 = their voxel size 0.273 mm. Their own set-up: CELL_MM=70 D_MIN=0.5 D_MAX=15 at 256^3.
* `sbatch_fourel_gpu.sh <DIM> <N> <solver>` (l40s default, 3D 256^3: `-p h200 --mem=200G`; excludes g3606,g3608;
  47 h 30 min budget, self-resubmit from the checkpoint, MAXSEG 3), run folder
  `/work/classic/fr_ml1145-muffto/runs/asr_fourel/` (scripts + geometry copied by `push_nemo_fourel.sh`, library
  `~/software/muFFTTO`), `sync_nemo_fourel.sh` fetches CSV / steps / snapshots / logs into `exp_data/`,
  `logs/nemo_fourel/` (FULL=1 also the 3D full fields).
* Smoke job 8658309 (2D 128^2 Green, 6 steps; moved to rtx because the l40s QOS was full for ~9 h): RTX PRO 6000,
  15.6 s, spots / CG iterations identical to the laptop CPU run, ~1 ms per Green-iteration unit at 128^2 (laptop
  under load ~12 ms).

### Proposed NEMO run matrix (NOT submitted; the user decides)
Free loading, k = 1e-4, eps_ASR 0 -> 5 % in 200 steps, one spot per solve unless noted.
Estimates: spots to 5 % ~ 5-10k in 2D for N >= 256 (64^2: 16k with over-cracking, 128^2: 2.7k at 2 %; the spot is
physical, ~ l_c). Cost per solve: ~250-600 units early in cracking, but 2000-4500 for Green once the network is open
(128^2 at 0.5-0.9 %), assume ~2000 average for Green / MS 1e-6 / GJ, ~4000 for dual (x 2 included), ~20-80 for MS
1e-4. GPU time per unit: ~1 ms at 128^2 (rtx smoke job); L40S per CG iteration measured in the ASRmesh runs
1.8 / 4.0 / 16.5 ms at 256^2 / 512^2 / 1024^2, 117 ms at 3D 128^3 (H200 256^3 ~0.5 s, extrapolated). Hours = spots x
units / solve x ms, uncertain by ~2-3x; 48 h segments self-continue (MAXSEG 3 = 142 h).

| case | partition | l_c/h | MS 1e-4 | MS 1e-6 | Green | GJ | dual | batched (BATCH=1) |
|---|---|---|---|---|---|---|---|---|
| 2D 256^2 | l40s | 1.8 | < 0.5 h | 5-15 h | 5-15 h | 5-15 h | 10-30 h | Green, MS 1e-6: ~0.5x |
| 2D 512^2 | l40s | 3.7 | ~1 h | 10-35 h | 10-35 h | 10-35 h | 20-70 h | Green, MS 1e-6 |
| 2D 1024^2 | l40s | 7.3 | 2-5 h | 40-140 h (1-3 segm.) | 40-140 h | 40-140 h | > 142 h: window <= 2 % | Green, MS 1e-6 |
| 3D 64^3 (35 mm) | l40s | 0.46 | probe | - | probe | - | probe | - |
| 3D 128^3 (35 mm) | l40s | 0.91 | 20-60 h (spots unknown) | >> 142 h | >> 142 h | >> 142 h | >> 142 h | needed, window <= 1 % |
| 3D 256^3 (35 mm or their 70 mm) | h200 200G | 1.8 / 0.91 | > 142 h | infeasible | infeasible | - | - | needed |

* 2D: 15 single-GPU l40s jobs (5 solvers x 3 grids) + 6 batched (Green, MS 1e-6 x 3 grids), CHECK_EVERY=100 for the
  energy-error samples. Total ~1000+ GPU h if all go to 5 %; a cheaper first pass: all
  five solvers with STEPS_RUN=60 (eps <= 1.5 %, past percolation), ~1/3 of that.
* 3D: first one probe each, MS 1e-4 and Green at 128^3 with STEPS_RUN=40 (eps <= 1 %, ~20-50 h) to measure spots per
  step and cost per solve before deciding the 3D matrix; one-spot runs to 5 % in 3D are probably only affordable with
  MS at their loose tolerance (as they did), which our accuracy check shows is ~1e-3 in energy. Memory: 128^3 ~ 10 GB,
  256^3 ~ 80-100 GB (gradient-size fields 9.6 GB each: eps, sigma, macro, line-search copy, MS copies) -> h200.
* Submit e.g.: `ssh nemo "cd /work/classic/fr_ml1145-muffto/runs/asr_fourel && sbatch --job-name=FOUREL_2D_512_Green
  --export=ALL,CHECK_EVERY=100 sbatch_fourel_gpu.sh 2 512 Green"`, `--export=ALL,TOL_MS=1e-6 ... MS`,
  `--export=ALL,BATCH=1 ...`; 3D probe `--export=ALL,STEPS_RUN=40 sbatch_fourel_gpu.sh 3 128 MS`.

**Proposed NEMO run matrix (NOT submitted; the user decides).** Cost model: L40S ms per elastic CG iteration from the
ASRmesh runs (2D Green/GJ 1.8 / 4.0 / 16.5 ms at 256^2 / 512^2 / 1024^2, dual 2x; 3D 13.8 / 117 ms at 64^3 / 128^3,
dual 2x; 256^3 on H200 ~0.5 s, extrapolated) x elastic CG totals extrapolated from the laptop runs (A to 1 %: 190k /
277k at 64^2 / 128^2 for Green, x2.5 to 2 %, assumed to grow ~1.6x per refinement; B to 3 %: ~70k at 128^2, CG per
solve grid-independent for Green/dual). Uncertain by a factor ~3 (staggered iterations of A per crack advance). Caveat for B: its staggered iterations per load
step grow with refinement (steps 0-10: 64^2 3-10, 128^2 3-34, 256^2 3-121; the PF-CZM band of b/h pixels advances
~1 pixel per iteration), so the B estimates below may be low by 2-4x at 1024^2.

| run | grid | Green [h] | GJ [h] | dual [h] | partition |
|---|---|---|---|---|---|
| A (fourel, eps 0-2 %, 80 steps) | 2D 256^2 | 1 | 2 | 2 | l40s |
| | 2D 512^2 | 3 | 7 | 6 | l40s |
| | 2D 1024^2 | 20 | 50-80 (2 segments) | 40 | l40s |
| | 3D 64^3 | 3 | 6 | 6 | l40s |
| | 3D 128^3 | 30 | 60+ (2 seg.) | 60 (2 seg.) | l40s |
| | 3D 256^3 | not feasible (> 3 x 47 h) | | | |
| B (czm, eps 0-3 %, 60 steps) | 2D 256^2 / 512^2 / 1024^2 | 0.2 / 0.5 / 2 | 0.5 / 2 / 8 | 0.3 / 1 / 4 | l40s |
| | 3D 64^3 / 128^3 | 0.5 / 4 | 2 / 15 | 1 / 8 | l40s |
| | 3D 256^3 | ~30 | > 100 (skip) | ~60 (2 seg.) | h200, --mem=200G |
| B-Rankine (czmR), recommended in addition | same grids as B | ~ as B | | | |

Recommendation: 2D 256^2-1024^2 x {A, B, czmR} x {G, GJ, dual} (27 jobs, ~250 GPU h), 3D 64^3 + 128^3 x {A, B} x 3,
3D 256^3 B Green + dual on h200. Submit command per job:
`sbatch [-p h200 --mem=200G] sbatch_asrpf_gpu.sh <DIM> <N> <fourel|czm|czmR> <Green|Green_Jacobi|dual>`.
Smoke job 8658308 (2D 128^2 czm dual, 6 steps, EXTRA_SUFFIX=_smoke) submitted 2026-11-07 ~08:40, still PENDING at
~10:00 (QOSMaxJobsPerUserLimit: 21 other jobs of the user running). Check: `./sync_nemo_asrpf.sh fetch`, then compare
`exp_data/nemo_pf/ASRpf_czm_N128_dual_smoke_steps.npz` (stiffness_ratio, max_d per step) with the first 6 steps of
the laptop run `exp_data/ASRpf_czm_N128_dual_steps.npz` (expect agreement to CG tolerance).
