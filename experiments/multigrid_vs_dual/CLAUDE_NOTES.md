# multigrid_vs_dual — CLAUDE NOTES

## RESUME HERE (2026-11-15)

**STATUS 2026-11-15 ~13:30: STUDY DONE.** l/h sweep `_lb` finished (180 runs, NEMO 8820953-70 + laptop), synced,
`plot_lh_sweep.py` run, **README.md = results + verdict** (read it first). Short: MG 2.5-7x fewer CG than the dual,
flat in N and l/h; wall time MG wins on CPU (1.5-2.3x) and on GPU 1024² (2.3-3.6x), dual wins at 256² GPU (<=1.2x),
512² tie. Rigorous smoother bound needed (max CG/solve 117 vs 12084 with the warm power estimate). Nothing running.
Possible next: Galerkin Q1 coarse operators, fused GPU cycle, MG_LMAX default -> gershgorin (ask), library Multigrid
has the same power-estimate weakness (other session owns it: report, don't edit).

User request (2026-11-15 09:00): "compare multigrid preconditioner with dual preconditioner on AT2 phase-field
fracture; new folder; AT2 with pores; which preconditioner is more powerful". Then: "prototype on laptop but run
larger grids on NEMO so we do not have to wait; results this evening", "work autonomously" (user away).

### Where / what
* This folder lives in the **muFFTTO_fable worktree** (branch `fable/large-3d-to`) because the library multigrid
  `muFFTTO/multigrid.py` exists only on that branch. Another session works in the same worktree on the large 3D TO
  (`experiments/Multigrid_preconditioning`, NEMO jobs `FB_*`): do not touch its files or jobs.
* `example_2D_phase_field_fracture_AT2.py`: copy of `../dual_preconditioner/damage_AT2_small_strain/` (sys.path
  fixed), new elastic preconditioner `MG`, work counters in the iterations CSV, `CG_DELAY` env (d_min, default 10).
  Damage solve unchanged (Green-Jacobi) for every elastic preconditioner.
* `mg_q1.py`: `MultigridQ1(Multigrid)` = library cycle + 2D bilinear-quad coarsening (re-discretisation: coarse
  Gauss point q = i + 2j <- mean of fine pixel (2I+i, 2J+j)), `warm_power_iterations` (env `MG_POWER`, library 15).
  `MG_DEGREE` env: V(nu, nu) (default 2).
* MG setup after EVERY damage update (every staggered iteration): coarse materials, Jacobi diagonals (Dirac combs),
  warm power iterations, dense pinv of the 4x4 grid. Counted in `kwork_setup_u`, `time_setup_u` (inside `time_u`).
* Cost columns: `kwork_u` = K applications in fine-grid equivalents (MG levels weighted 4^-l, dual's K(C_dual),
  CG's own products), `gwork_u` = Green (FFT pair) applications.

### Problem
polydisperse pores (soft, 1e-3), k = 1e-4, l = 2h (l/h = 2), uniaxial strain xx to 0.08 in 80 steps, stop 3 steps
after cut-through, CG energy test rtol 1e-6. Two series:
* `fixed` (no tag): Gc = 5e-4 on every grid. **Problem found:** AT2 strength ~ sqrt(E Gc / l) grows with N at fixed
  l/h -> 1024² never cracks (max d 0.21 at E 0.08), 512² barely -> only the undamaged stage is compared there.
* `sGc` (tag `_sGc`, MAIN series, my decision while the user was away): Gc = 0.016 l = 0.032/N (Gc/l of the 64² run),
  i.e. fixed strength, every grid cracks in the load path. 64² is identical in both series.

### Runs
* Laptop (CPU, one thread per run, concurrent batches -> times comparable only within a batch):
  `run_laptop.sh` (64/128 fixed, MG + dual, 64² also MG_POWER=3 and d_min 0), `run_laptop_v11.sh` (64² MG V(1,1)
  pow3, d_min 10/0), `run_laptop_sGc.sh` (128² sGc MG + dual, d_min 10/0), `run_laptop_sGc_v11.sh` (128² sGc
  MG V(1,1)). Logs `logs/`, CSV `exp_data/`. (run_laptop.sh log names have a doubled `_lh2_lh2`, harmless.)
* NEMO (L40S, `sbatch_mg_vs_dual.sh <N> <MG|dual>`, env GC / CG_DELAY / MG_DEGREE / MG_POWER / SUFFIX; smoke
  8820310/11 ok): study folder rsynced ALONE to `~/software/muFFTTO_fable/experiments/multigrid_vs_dual` (library
  md5-identical to this worktree; never `fable_sync_nemo.sh` from here: it pushes the other session's uncommitted
  files). Output `$WS/runs/MG_vs_dual/{exp_data,logs,figures,slurm}`, job list `jobs.txt` there.
  fixed: 8820318-29 (256/512/1024 x MG, dual x d_min 10/0); sGc: 8820339-50; sGc MG V(1,1) pow3: 8820353-58.
* `./sync_nemo.sh` (CSV, logs, png only) then `python plot_mg_vs_dual.py sGc|fixed` ->
  `figures/mg_vs_dual_summary_<series>.png`, `figures/mg_vs_dual_stress_<series>.png`; `python summarize.py`.

### Findings so far (details + tables in README.md)
* Same physics: stress-strain curves of MG and dual identical to 5-6 digits on every grid.
* 64² CPU: MG V(2,2) 18.9 CG/solve vs dual 60.0; elastic time MG 113 s vs dual 96 s; MG_POWER=3 same iterations,
  102 s; d_min 0: MG 11.9 (82 s) vs dual 57.6 (94 s) -> the floor of 10 costs MG ~35 %, the dual ~4 %.
  MG V(1,1) + MG_POWER=3: 20.4 CG/solve, 94 K/solve (V(2,2) 158; dual 123 K + 122 FFT) -> 64 s vs dual 96 s.

### l/h SWEEP (user 2026-11-15 ~09:20: "make proper study with l/h sweep from 0.5 to 4") — the MAIN study now
* l/h = 0.5, 1, 1.5, 2, 3, 4; Gc = 0.016 l (fixed strength); load to 0.12 in 120 steps (increment as before);
  configurations dual, MG V(1,1), MG V(2,2) (MG_POWER=3 for both) x d_min 10 / 0; OUTPUT_NC=0 (final damage .npy).
* Laptop 64², 128²: `run_lh_sweep_laptop.sh` (unit `mgdual-sweep`, 8 concurrent single-thread runs, skips finished).
* NEMO 256/512/1024: `sbatch_lh_sweep.sh <N> <lh>` = one job per (N, l/h), six configurations sequential on one
  L40S (comparable timings, finished ones skipped on resubmission); smoke 8820389 ok; jobs 8820416-33.
* `./sync_nemo.sh && python plot_lh_sweep.py` -> figures/lh_sweep_{iterations,time_ratio,damage}.png,
  exp_data/lh_sweep_summary.csv. Logs logs/(nemo/)AT2_N<N>_<config>_sw_lh<lh>.log.
* The earlier series (`_lh2`, `_lh2_sGc`, `_sGc_v11`; plot_mg_vs_dual.py) are the l/h = 2 prototype, load to 0.08.

### l/h = 2 prototype results (fixed strength, load 0.08; plot_mg_vs_dual.py sGc) — done 2026-11-15 ~09:50
* dual CG/solve grows once the fine grids crack: 60 / 58 / 66 / 92 / 118 (64-1024); MG V(2,2) 19-26, V(1,1) 20-32.
* GPU elastic time (L40S): 256² dual 58 s, MG22 100, MG11 80; 512² dual 178, MG22 237, MG11 169;
  1024² dual 1100, MG22 700, MG11 546 (d_min 0: MG22 595, MG11 530, dual 1098).
* MG V(1,1) at 1024²: 3 solves with 1017-3559 CG at load step 35 (crack cuts through the cell, staggered its
  111-113), then back to ~40. MG V(2,2) (15 warm power its) max 69.
* Adaptive power iteration (MG_POWER_TOL=0.01, mg_q1 power_tol) test 8820454: average 3.0 warm steps, still 2 solves
  >1000 -> NOT a lambda_max lag. Hypothesis: re-discretised (non-Galerkin) Q1 coarse operators over-correct when the
  crack just separates the cell (coarse grids do not see the crack), V(1,1) smoothing cannot compensate.
  Test: linear triangles (library coarsening = exact Galerkin) 1024² l/h 2, MG V(1,1) + dual: jobs 8820536/37
  (SUFFIX _sGc_v11_tri / _sGc_tri). Adaptive test files: exp_data/adaptive_power_test, logs/adaptive_power_test.
* Sweep keeps MG_POWER=3 fixed (8 cancelled jobs resubmitted unchanged: 8820527-34); laptop unit mgdual-sweep3.

### MG robustness fix (2026-11-15 ~10:50) -> sweep rerun as `_lb`
* Triangle test (8820536/37, exact Galerkin coarsening): MG V(1,1) no extreme spikes (max 222) but 77 CG/solve
  (Q1: 32); dual tri 107. Not a fix.
* First sweep pass (power estimate, MG_POWER=3) at 1024²: breakdowns also with V(2,2): l/h 0.5 one solve 10615
  (V(1,1) 12084), l/h 2 6778 (V(1,1) 1709), l/h 4 V(1,1) 1820 -- always the staggered iterations where the crack
  cuts through. Earlier V(2,2) with 15 warm power its: max 69. Diagnosis: a NEW dominant mode of D^-1 A localised at
  the crack; the warm-started power vector has ~no component of it, the estimate looks converged at the old value
  (adaptive tolerance useless) -> smoother amplifies the mode -> MG preconditioner (near) indefinite.
* FIX (mg_q1.gershgorin_constant, env MG_LMAX=gershgorin; name kept, it is NOT Gershgorin): rigorous bound for any
  s >= 0 per Gauss point: lambda_max(D^-1 A) <= max_q lambda_max(D_q^-1 K_q) (mediant inequality, D linear in s)
  = 3.523 for bilinear quads, E 1, nu 0.2 (homogeneous 2.10, random extreme <= 3.1; row-sum Gershgorin 10.4 useless).
  No power iteration at all. 64² l/h 2: V(2,2) 19.0 CG/solve (power 18.9), V(1,1) 22.0 (20.4); K/solve unchanged.
* Sweep rerun with MG_LMAX=gershgorin, SUFFIX _lb, ALL 6 configs (dual too: same-job timing): NEMO 8820953-70
  (old sweep jobs cancelled, their finished runs = '<config>_power' in plot_lh_sweep.py), laptop unit
  `mgdual-sweep-lb` (run_lh_sweep_laptop.sh, SUFFIX default _lb). figures/lh_sweep_robustness.png = max CG/solve.
