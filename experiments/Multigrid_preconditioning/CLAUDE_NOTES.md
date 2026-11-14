# Multigrid preconditioning study — CLAUDE NOTES

## RESUME HERE (state 2026-11-14 ~06:40)

Read this section first after an interruption; details and all tables are in `README.md` (stages 1-8) and
`OVERNIGHT_NOTES.md` (overnight + morning results). Nothing in this folder is committed (`git status`: untracked).

### What is decided (user decisions in **bold**)
* Mechanics: CG + **geometric multigrid** V(2,2) (Chebyshev-Jacobi, `mg_lib.py`), coarse grids by averaged material
  (2D linear triangles: exact Galerkin; 3D trilinear hex: **option (a) re-discretisation**, option (b) Galerkin
  stencils = fallback, not needed), **hierarchy stored as ScaledMaterial** (s(x) per Gauss point; identical results,
  16x / 81x less material memory in 2D / 3D, 2.2x faster per evaluation in 3D TO, neutral in 2D on the GPU).
* CG stop: energy estimate rtol 1e-6, **d_min = 0** (the floor of 10 doubled MG's cost), warm starts.
* Optimizer: L-BFGS with **Helmholtz H_0** (I + c eta^2 (-Lap))^-1. NuMPI commit **6561d66** on `feat/tr_precond`
  (local + NEMO, **not pushed**); muFFTTO `solvers_nonlinear.minimize(precond=)` pass-through + tests NOT committed.
* Big-run settings so far: **contrast 1e3** (SOFT_EXP=3), **eta = 2h**, **c = 4** (prototype: c = 1 marginally
  cheaper, c = 4 close; same designs), Cahn-Hilliard starts, H200 (`-p h200`) for large runs.
* **Grids only up to 1024 for the static scaling**; Green/dual dropped from TO comparisons (GJ = production baseline).

### Open questions / next steps
1. **Auxetic target** (user: reduce target stiffness to get a meaningfully negative Poisson ratio): current designs
   meet the shear target but reach only nu12 = -0.13 / -0.20 / -0.23 at 256 / 512 / 1024² (target -1/3).
   Prototype `run_auxetic_proto.sh` RUNNING on the laptop (unit `mg-aux`, logs `logs/auxetic/`): 128², TARGET_NU
   {-0.5, -0.7} x TARGET_SCALE {1, 0.3, 0.1}. When done: `python design_properties.py --soft 1e-3
   data/example_2D_elasticity_TO/N128_soft0.001_*_aux_nu*_final.nc` -> nu12, mu, solid fraction per target; pick the
   target for the big runs. Compare with the 3D stress-weight lever (job TO3D_128_sw25 chain, `$WS/runs/TO3D/logs`).
2. V(1,1) vs V(2,2): V(1,1) ~8-10 % cheaper per evaluation (128² prototype) -> optional (`MG_DEGREE=1`).
3. Library port of the multigrid into muFFTTO (`mg_lib.py` -> `muFFTTO/multigrid.py`): single GPU first; MPI needs
   aligned coarse decompositions + ghost transfers (not started). Then 3D 256³ TO on an H200.
4. 1024³ 3D TO is NOT feasible now (memory, see below; and MG is single-GPU).
5. When everything is decided: the "nice" large runs (2D 2048²/4096², 3D 128³/256³) with the chosen auxetic target.

### Auxetic target and objective (2026-11-14)
* Stress objective, 128² eta 2h contrast 1e3 (`run_auxetic_proto.sh`, `design_properties.py --soft 1e-3`):
  TARGET_SCALE 1 -> nu12 -0.10 (nu* -0.5) / -0.15 (-0.7), mu 0.20, 80 % solid, anisotropic (A_Z 1.35);
  **TARGET_SCALE 0.3 -> nu12 -0.26 / -0.30, mu 0.055 (on target), 59 % solid, isotropic (A_Z 0.99)**; 0.1 -> -0.24 /
  -0.29 (resolution-limited at 128²). Best so far: TARGET_NU -0.7, TARGET_SCALE 0.3 (nu12 -0.30 ~ isotropic nu -0.43).
* **TARGET_SCALE meaning (REVISIT LATER, user 2026-11-14)**: multiplies the whole target tensor C* at fixed nu*:
  G* = scale * 0.15 E_0, E* = 2 G* (1 + nu*), K* = E* / (3 (1 - 2 nu*)) (solid K_0 = 1, mu_0 = 0.5, E_0 = 1.29).
  scale 1, nu* -0.5: G* 0.193 (39 % of mu_0), E* 0.193 (15 % E_0), K* 0.032 (3 %) -> unreachable (stiff + no bulk);
  scale 0.3: G* 0.058 (12 %), E* 0.058 (4.5 %), K* 0.0096 (1 %); nu* -0.7 scale 0.3: E* 0.035 (2.7 %), K* 0.0048.
  Open: how the target should be chosen properly (bounds on attainable (K, mu) for solid/void, resolution, eta).
* **Energy objective** (user request): `OBJECTIVE=energy` in the 2D TO copy = match the target energies C_1111,
  C_2222, C_1122, C_1212 (sum (W - W*)^2 / W*^2); gradient via the library stress adjoint with a modified target
  (exact: FD check `check_objective_gradient.py`, rel diff ~1e-7). Prototype `run_auxetic_energy.sh` RUNNING
  (unit mg-auxE, logs `logs/auxetic_energy/`, designs `*_auxE_nu*_final.nc`) -> compare nu12 with the stress objective.
* **Under-integrated 3D elements** (user request, 8x less quadrature memory): `trilinear_hexahedron_1Q` supported in
  mg_lib (fine level 1Q, coarse levels fully integrated) and in the 3D TO copy (`ELEMENT=...`); static test MG only
  (`run_1Q_test.sh`, unit mg-1q2, `logs/1Q/`, CSV `exp_data/scaling3d_1Q`) RUNNING: iterations, C11 accuracy vs 8-point,
  peak memory. **RESULT: not usable with MG as is.** MG iterations 32³ eta 0/1h/2h: 3013 / 318 / 198 (8-point 14 / 12 /
  12); 64³ eta 2h: > 600 (cap) vs 12. Per CG iteration 4x cheaper (62 vs 256 ms at 64³), peak memory 408 vs 1080 MB
  (2.6x; only quad fields shrink 8x, vectors do not) -> overall ~12x SLOWER. Cause: hourglass (checkerboard, near
  zero-energy) modes, neither smoothed by Jacobi nor representable on the coarse grids. Accuracy: C11 -19 % on sharp
  interfaces (eta 0), < 0.3 % for eta >= 1h. Fix to try later: hourglass stabilisation (Flanagan-Belytschko-type
  stiffness for the hourglass modes) in a new muFFTTO element. 3D TO with 1Q skipped.
* **Hourglass control prototype** (2026-11-14, `hourglass_prototype.py`, SUPERSEDED): FB with kappa = eps c0 s (lambda + 2mu),
  c0 calibrated on 8³ -> BUG: did not scale with h, true eps = nominal x N/8 (the "eps 0.1" runs were eps 0.4 at 32³,
  0.8 at 64³). Results of that run: logs/1Q/run_hg*.log, peak_memory_hg.txt.
* **Hourglass control in muFFTTO (library, 2026-11-14, uncommitted)**: element `'trilinear_hexahedron_1Q_hg'`
  (+ `Discretization(..., hourglass_eps=0.1)`), module `muFFTTO/hourglass_control.py` (`HourglassControl`: muGrid
  GenericLinearOperator stencil [4,1,1,2,2,2] = h_alpha/8; kappa_{i alpha}(x) = sum_de M_alpha[d,e] C_{idie}(x) = exact
  2x2x2-Gauss energy of the unit mode on one voxel at the actual pixel size -> eps = 1 reproduces the 8-point hourglass
  energy mode by mode on any grid; ScaledMaterial / full field / constant tensor). Hook: end of
  `apply_system_matrix_mugrid` (Jacobi diagonal + Green reference pick it up). TO: hourglass term in
  `partial_derivative_of_adjoint_potential_wrt_phase_field_FE` (via `design_derivative`) + hourglass force in the
  adjoint-energy diagnostic of `sensitivity_stress_and_adjoint_FE_NEW` (that diagnostic is ADDED to the objective in
  the TO scripts -> without it FD check failed at 4e-3). Tests `test/test_hourglass_control.py` (7 pass, incl. FD of
  design_derivative); regression test_discretization + test_scaled_material pass. 3D TO FD check
  `check_objective_gradient_3d.py` (8³, MG): 8-point 1.3e-8, 1Q_hg 1.1e-8 / 4e-10. mg_lib + scaling_case_3d now use
  the library element (`--element trilinear_hexahedron_1Q_hg --hg-eps EPS`, CSV tag `_1Qhg<eps>`).
  **Corrected eps study** (`run_hg_test.sh`, ch_s12, contrast 1e4, MG only; CSV exp_data/scaling3d_1Q/csv/*_1Qhg<eps>.csv):
  MG iterations 8-point 14/12/12 (32³) 14/13/12 (64³) for eta 0/1h/2h; eps 0.3: 16/12/12, 15/13/12 (= 8-point);
  eps 0.1: 18/13/13, 17/13/12; eps 0.03: 28/19/18, 26/17/17; eps 1: 18/12/12, 17/13/12. C11 vs 8-point, eta >= 1h:
  eps 0.3 -0.18..+0.05 %, eps 0.1 -0.27..-0.02 %, eps 1 +0.13..+0.78 %; sharp (eta 0) -8 % (64³) / -14 % (32³) for
  any eps (one-point error, converges with h). 64³ solve 1.1-1.3 s vs 3.1-3.6 s (2.5-2.8x), peak GPU 480 vs 1080 MB.
  -> **DEFAULT eps = 0.3** (the user's "eps 0.1 best" was the prototype's mislabelled eps ~0.4-0.8). Speed (128³ GPU,
  one K apply): 8-point 331 ms, 1Q 36 ms, 1Q_hg 72 ms (control = muGrid H 9 ms + H^T 11 ms + scaling; one pass for
  kappa). Small 3D TO test `run_TO3D_hg_test.sh` (32³, eta 2h, contrast 1e3, MG + Helmholtz c4, bi_s11 3D CH start,
  8-point vs 1Q_hg eps 0.3, unit mg-to3dhg, logs/TO3D_hg/) RUNNING -> compare f, evals, s/eval, peak memory, design.
* User: **run only the fast preconditioner (MG)** in these tests.

### Memory (measured on the laptop GPU, 3D TO with 6 load cases, `TO3D` 32³ -> 64³ slope)
MG ~7.4 KB/voxel, GJ ~6.4 KB/voxel (+ ~0.3 GB CUDA context). -> 128³ ~16 GB, 256³ ~125 GB (fits one H200, tight;
LOW_MEMORY=1 halves the gradient temporaries), 512³ ~1 TB (>= 8 H200 with MPI), 1024³ ~8 TB (beyond NEMO's 16 H200).
48³ MG fails: grid sizes must halve to 4 (powers of 2 x 4, e.g. 32, 64, 128, 256; 192 = 3*64 untested).

### Running on NEMO (sync: `./sync_nemo_TO.sh`, `./sync_nemo_scaling3d.sh`; job list `$WS/runs/MG_overnight_jobs.txt`)
| job | GPU | what | log ($WS/runs/MG_TO/) |
|---|---|---|---|
| 8810349 | h200 | 2D TO 4096², chain bi_s11, MG + Helmholtz c4, eta 4h, contrast 1e4 | TO_N4096_MG_bi_s11_eta4h_lbfgs_helm_c4.log |
| 8813562 | h200 | 3D TO 128³ MG + Helmholtz c4, ScaledMaterial, from 64³ design | TO3D_N128_MG_bi3d_s11_eta4h_lbfgs_helm_c4_scaled.log |
| 8810501 | h200 | 3D TO 128³ Green-Jacobi + NuMPI L-BFGS (production) | TO3D_N128_Green_Jacobi_bi3d_s11_eta4h.log |
| 8813561 | l40s | 3D TO 64³ MG + Helmholtz c4, ScaledMaterial | TO3D_N64_MG_bi3d_s11_eta4h_lbfgs_helm_c4_scaled.log |
| 8813729 | l40s | earlier session: 3D TO 128³ GJ stress weight 25 chain, segment 6 (self-continuing) | $WS/runs/TO3D/logs |
| 8812764 | l40s | earlier session: AT1 bounded study 1024² | $WS/logs/AT1b-8812764.out |
Comparison to make when the 3D 128³ jobs finish: time to the converged design MG + Helmholtz (new) vs GJ (production)
(first ~13 evaluations: MG 27 s/eval vs GJ 34 s/eval on the H200).

### How to run things (all scripts in this folder; NEMO copy in ~/software/muFFTTO/experiments/Multigrid_preconditioning)
* Sync code to NEMO (never data): `rsync -a --exclude exp_data --exclude logs --exclude figures --exclude __pycache__
  --exclude data/example_2D_elasticity_TO --exclude data/example_3D_elasticity_TO ./ nemo:software/muFFTTO/experiments/Multigrid_preconditioning/`
  (queued jobs import mg_lib / the TO scripts at start -> keep changes backward compatible).
* TO on NEMO: `sbatch [-p h200] sbatch_TO_mg.sh <MG|Green_Jacobi|Green|dual> <N> [start.nc]` with env ETA_H, DESIGN_OPT
  (lbfgs | lbfgs_helm), HELM_C, LABEL, SCRIPT (example_3D_elasticity_TO.py for 3D), MAXITER, LOG_SUFFIX; add
  SOFT_EXP / TARGET_NU / TARGET_SCALE / MG_DEGREE by editing the NPIX=... line (not yet exported by the script).
  Smoke test every new/edited script first (MAXITER=2-3, N 16/64, `--time=00:20:00`), dependents with afterok.
* Static scaling: `scaling_case.py` (2D) / `scaling_case_3d.py` via `sbatch_mg_scaling.sh` / `sbatch_mg_scaling_3d.sh`.
* Laptop TO: `example_2D_elasticity_TO.py MG` with env NPIX, ETA, SOFT_EXP, PF_NORMALIZED=1, PF_WEIGHT=0.2,
  CG_MODE=energy, CG_DELAY=0, GTOL=0, DESIGN_OPT=lbfgs_helm, HELM_C, MG_DEGREE, TARGET_NU, TARGET_SCALE, INIT_FROM.
  Heavy runs via `systemd-run --user -p MemoryMax=... bash <script>.sh`.

### File map
| file | purpose |
|---|---|
| `mg_lib.py` | multigrid (2D/3D, transfers, Chebyshev smoother, V-cycle, combinations, ScaledMaterial hierarchy, GPU) |
| `design_optimizers.py` | local L-BFGS with preconditioned H_0 (= NuMPI 6561d66) and projected PR+ NCG |
| `example_2D/3D_elasticity_TO.py` | TO copies with MG, DESIGN_OPT, CG_DELAY, MG_DEGREE, TARGET_NU / TARGET_SCALE, EVAL/TOTAL lines |
| `scaling_case(_3d).py`, `sbatch_mg_scaling(_3d).sh` | static preconditioner scaling (NEMO) |
| `sbatch_TO_mg.sh`, `submit_overnight.sh`, `submit_TO3D.sh` | NEMO TO jobs |
| `make_new_CH_geometries.py` | new random Cahn-Hilliard starts (data/CH_starts_new, data/CH3D) |
| `design_properties.py`, `elastic_properties.py` | effective nu12 / mu of a 2D design |
| `plot_*.py` | figures (stage overview, eta sweep, mesh/NEMO scaling, TO scaling, Helmholtz, designs, prototype) |
| `test_mg_lib.py`, `test_mg_lib_3d.py`, `test_scaled_material.py` | checks (Galerkin, SPD, ScaledMaterial == full) |
