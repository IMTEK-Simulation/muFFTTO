# Multigrid vs dual preconditioner on AT2 phase-field fracture with pores

Question (2026-11-15): which elastic preconditioner is stronger on the AT2 fracture problem with pores, geometric
multigrid (MG) or the dual preconditioner G K(C_ref C_d^-1 C_ref) G? Main study: l/h sweep 0.5 to 4 on grids 64² to
1024².

## Setup

* `example_2D_phase_field_fracture_AT2.py` (copy of `../dual_preconditioner/damage_AT2_small_strain/`): staggered
  AT2 with the Amor split, history field and hybrid formulation, bilinear quads (2x2 Gauss), polydisperse soft pores
  (stiffness 1e-3), g(d) = (1-d)² + 1e-4, E = 1, nu = 0.2, uniaxial strain xx to 0.12 in 120 steps. The run stops
  3 steps after the crack cuts through the cell.
* Elastic solve K u = f with C_d = s(x) C_iso: CG, energy-error stopping test rtol 1e-6, adaptive delay with
  d_min = 10 (standard) or 0. The damage solve is the same for every elastic preconditioner (Green-Jacobi).
* **dual**: G K(C_ref C_d^-1 C_ref) G, so each iteration costs 2 K + 2 FFT (CG's own K included).
* **MG**: library `muFFTTO.multigrid.Multigrid` plus `mg_q1.MultigridQ1`, which adds the 2D bilinear-quad
  coarsening. Each coarse Gauss point q = i + 2j takes the mean scale of fine pixel (2I+i, 2J+j); this is a
  re-discretisation, not Galerkin. V(nu,nu) cycle with nu = 1 (**MG11**) or 2 (**MG22**), Chebyshev-Jacobi smoother
  on [lmax/30, lmax], exact solve on the 4x4 coarsest grid, multilinear P, R = P^T. The setup is redone after every
  damage update, and its cost is included in every MG time below.
* **Fixed strength, Gc = 0.016 l**: Gc/l is kept at its value from the 64² run with l = 2h. With a fixed Gc the AT2
  strength ~ sqrt(E Gc / l) grows under refinement at fixed l/h, and 1024² never cracked (`_lh2` prototype series).
* Grids: 64² and 128² ran on the laptop CPU, one thread per run, 8 concurrent runs (times are comparable within the
  sweep). 256², 512² and 1024² ran on NEMO, one L40S per (N, l/h) job with the six configurations run one after
  another, so their timings are comparable.
* 180 runs (5 grids x 6 l/h x {dual, MG11, MG22} x d_min {10, 0}). All of them cracked and finished. The peak stress
  is identical in all 6 configurations of every (N, l/h): relative spread 0.0 at the printed precision.

## MG robustness fix: rigorous smoother bound (`MG_LMAX=gershgorin`)

The library estimates lmax(D^-1 A) with a power iteration that is warm-started from the previous dominant vector. In
the first pass of the sweep this failed at 1024²: when the crack cut through the cell, single solves took 1709 to
12084 CG iterations with both V(1,1) and V(2,2). A new dominant mode localised at the crack appeared. The warm vector
had almost no component in it, so the estimate stayed at the old value, the Chebyshev smoother amplified the mode and
the preconditioner became (nearly) indefinite. An adaptive power tolerance did not help, because the estimate *looks*
converged.

The fix is a contrast-independent bound that holds for any s >= 0 per Gauss point. Since D is linear in s, the
mediant inequality gives
lmax(D^-1 A) <= max_q lmax(D_q^-1 K_q) = **3.523** (bilinear quad, nu = 0.2).
For comparison: homogeneous material gives 2.10, random extreme-contrast fields stay <= 3.1, and the Gershgorin row
sum gives 10.4, which is useless. The bound is computed once (`mg_q1.gershgorin_constant`; the name is historical, it
is not Gershgorin) and needs no power iteration.

* With the bound, the largest CG count of any solve is **117** (1024², l/h 0.5, V(1,1)). With the power estimate it
  was 12084. See `figures/lh_sweep_robustness.png`.
* Cost over the 105 complete pairs: median +4 % CG iterations (V(1,1) +10 %, V(2,2) +1 %), median total time
  ratio 1.00. The bound saves the power iterations in every setup.
* The whole sweep was rerun with the bound (tag `_lb`). Everything below is `_lb`. The power-estimate runs appear as
  `<config>_power` in `exp_data/lh_sweep_summary.csv`.

## Results (d_min = 10 unless stated)

CG iterations per elastic solve, dual / MG11 / MG22:

| N \ l/h | 0.5 | 1 | 1.5 | 2 | 3 | 4 |
|---|---|---|---|---|---|---|
| 64   | 66 / 24 / 21 | 65 / 23 / 20 | 65 / 23 / 20 | 60 / 22 / 19 | 56 / 22 / 19 | 54 / 21 / 19 |
| 128  | 75 / 25 / 21 | 63 / 24 / 21 | 63 / 24 / 20 | 58 / 23 / 20 | 61 / 22 / 19 | 75 / 21 / 19 |
| 256  | 89 / 27 / 23 | 78 / 26 / 22 | 65 / 26 / 22 | 66 / 26 / 22 | 69 / 25 / 21 | 72 / 23 / 20 |
| 512  | 124 / 39 / 32 | 94 / 33 / 28 | 91 / 28 / 24 | 92 / 25 / 22 | 89 / 24 / 21 | 85 / 21 / 19 |
| 1024 | 228 / 42 / 35 | 186 / 32 / 27 | 128 / 30 / 26 | 118 / 32 / 26 | 98 / 23 / 20 | 96 / 24 / 20 |

Max CG per solve: dual 133-688, MG11 29-117, MG22 24-100. d_min = 0 changes the dual's iterations by 0 to -6 %
(no systematic time change); MG then needs 14-39 (V(1,1)) and 12-31 (V(2,2)) CG per solve.

Whole-run elastic wall time, best MG over best dual (minimum over d_min and V(1,1)/V(2,2); in nearly every case the
best MG is MG11 with d_min 0):

| N \ l/h | 0.5 | 1 | 1.5 | 2 | 3 | 4 | hardware |
|---|---|---|---|---|---|---|---|
| 64   | 0.50 | 0.48 | 0.53 | 0.61 | 0.65 | 0.63 | laptop CPU |
| 128  | 0.49 | 0.66 | 0.63 | 0.60 | 0.54 | 0.44 | laptop CPU |
| 256  | 0.97 | 1.03 | 1.23 | 1.20 | 1.13 | 0.93 | L40S |
| 512  | 0.94 | 1.05 | 0.90 | 0.81 | 0.77 | 0.73 | L40S |
| 1024 | 0.31 | 0.28 | 0.38 | 0.43 | 0.37 | 0.37 | L40S |

With the standard d_min = 10, MG11/dual: CPU 0.56-0.89, 256² 1.13-1.54, 512² 0.90-1.19, 1024² 0.31-0.48.
1024² absolute times, dual vs MG11 d_min 0: 1398 vs 429 s (l/h 0.5) and 2123 vs 783 s (l/h 3).

Work per solve (fine-grid K equivalents, MG coarse levels weighted 4^-level): dual 111-458 K **plus** the same number
of FFT pairs; MG11 93-170 K; MG22 135-238 K.

## Verdict

1. **MG is the more powerful preconditioner.** It needs 2.5-4.5x fewer CG iterations on 64²-512² and 3.7-6.9x
   fewer on 1024². Its count is almost flat in N and l/h (V(1,1): 21-42). The dual grows with the grid at fixed l/h and with
   decreasing l/h (54 -> 228; at 1024² and l/h 0.5 it needs 228 per solve with a max of 688). The thin, nearly broken
   crack band (contrast 1e-4 in g(d), times 1e-3 in the pores) hurts the dual much more than MG.
2. **MG also uses less operator work**: MG11 needs 0.35-0.85x the K applications of the dual (MG22 0.5-1.2x), and
   the dual needs an FFT pair on top of each K.
3. **Wall time depends on the hardware and grid size:**
   * CPU (64², 128²): MG wins, 1.5-2.3x faster.
   * GPU L40S at 256²: the dual wins by up to 1.2x (MG11 with d_min 10 is 1.1-1.5x slower). The MG cycle (about
     10 small kernels per level, Python-driven) is latency-bound: about 1.4 ms per K-equivalent against 0.4 ms per
     dual operator.
   * 512²: roughly a tie (0.73-1.05). MG wins at l/h >= 1.5.
   * 1024²: MG wins clearly, **2.3-3.6x faster**, because the dual's iteration count blows up and the GPU is
     bandwidth-bound.
   The trend favours MG further on larger and 3D grids.
4. **V(1,1) is faster than V(2,2)** in 59 of 60 cases (the exception, 256² l/h 2 d_min 0, is 1 s), despite
   12-22 % more iterations.
5. **d_min = 0 helps only MG** (-7 to -39 % time, most on the small grids). The dual's iteration counts are far above the floor of 10. MG
   with d_min 10 wastes iterations once its counts get close to the floor.
6. **MG needs the rigorous smoother bound** in fracture. The warm power estimate breaks down at crack cut-through. The
   same failure mode can hit the library `Multigrid` in any problem with a sudden local material change, such as TO
   with SIMP (D is linear in s there too, so the same bound applies).

## Follow-ups (not done)
* Galerkin Q1 coarse operators (P^T A P, 9-point stencils): possibly fewer iterations at small l/h on large grids.
  The linear-triangle (Galerkin) test at 1024², l/h 2 needed 77 CG/solve vs 32 for Q1, so it is not a priority.
* Fused or graph-captured GPU MG cycle to remove the latency penalty at 256²-512².
* Default `MG_LMAX` in the script is still `power` (reproduces the prototype). Use `MG_LMAX=gershgorin` for new runs.

## Files
* `run_lh_sweep_laptop.sh` (64², 128²) and `sbatch_lh_sweep.sh <N> <lh>` (NEMO; env MG_LMAX, SUFFIX, CONFIGS). Then
  `./sync_nemo.sh` and `python plot_lh_sweep.py`.
* `figures/lh_sweep_iterations.png`: CG per solve vs l/h, per method and d_min. `figures/lh_sweep_time_ratio.png`:
  MG/dual time heatmaps. `figures/lh_sweep_robustness.png`: max CG per solve, bound vs power.
  `figures/lh_sweep_damage.png`: final damage.
* `exp_data/lh_sweep_summary.csv`: all runs (incl. `_power`). Iteration CSVs are in `exp_data/` (laptop) and
  `exp_data/nemo/`, logs in `logs/` and `logs/nemo/`.
* l/h = 2 prototype (load 0.08; `plot_mg_vs_dual.py sGc|fixed`, `run_laptop*.sh`, `sbatch_mg_vs_dual.sh`): see
  CLAUDE_NOTES.md.
