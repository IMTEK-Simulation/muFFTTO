# Talk outline: Green, Green-Jacobi and the dual preconditioner for FFT-based solvers

Scope: linear elasticity benchmarks, topology optimization, small-strain phase-field fracture. No contact.
Paths are relative to `experiments/`. Cost always means PCG iterations x applications per iteration:
Green 1, Green-Jacobi 1, dual 2 (G K(.) G); the measured wall-time ratio dual / Green is 1.5-2.9.

**Message:** there is no universal winner, so the right choice is predictable from the problem.
- **Green** is the cheapest when the reference fits the material.
- **Green-Jacobi** is the cheapest with void or smoothed interfaces.
- **The dual** has a guaranteed spectral interval and the most stable iteration counts. It pays off only where Green
  degrades.

---

## 1. Setting (1 slide)
- FFT-Galerkin / FE on a regular grid, preconditioned CG. The preconditioner is a reference problem inverted by FFT.
- The difficulty: contrast, void, damage, mesh refinement.
- The question: which reference, and how do we judge "better": iterations or cost?

## 2. Three preconditioners (1 slide)

| Preconditioner | Operator | Reference |
|---|---|---|
| Green | G | C_ref = solid tangent |
| Green-Jacobi | D^-1/2 G_I D^-1/2 | 2 mu I, local magnitude in the Jacobi diagonal |
| dual | G K(C_ref A^-1 C_ref) G | C_ref = solid tangent |

- Cost per iteration: 1 / 1 / 2.
- Remark: the dual with C_ref = 2 mu I is 2-6x worse. The reference matters.

## 3. Theory: why the dual is robust (1 slide)
- Spectral bounds: lambda_min(PK) >= 1 (complementary energy), lambda_max <= max g / min g, with g the eigenvalues of
  C_ref^-1 A. These are a priori, so the iteration count is predictable.
- The flip side: near-void (g -> 0) makes the bound blow up, and true void (A^-1 undefined) is impossible.
- Figure: `dual_preconditioner/spectral_study/results/iterations_N32.png`.

## 4. Linear benchmarks P1-P3 (2 slides)
- **Slide A, iterations.** The dual has the fewest iterations everywhere and is flat in contrast and grid. Green-Jacobi
  explodes (up to 6818 iterations).
  - `dual_preconditioner/paper1_problems/figures_elasticity/iterations_vs_contrast.png`
  - `dual_preconditioner/paper1_problems/figures_elasticity/grid_study_kappa1e-3.png`
- **Slide B, the same in cost** (`dual_preconditioner/figures/talk_cost.png`, top row and bottom-left pair):
  - P1: Green is cheaper (23 against 38).
  - P2: tie.
  - P3: the dual wins (110 against 206).
  - Take-away: the factor 2 decides. The dual wins where Green degrades.

## 5. Topology optimization (2 slides)
- **The interface decides between Green and Green-Jacobi:**
  `dual_preconditioner/topology_optimization/figures/TO_green_cost_vs_interface_N64.png`.
  - Sharp at the quadrature points: Green 76, Green-Jacobi 247.
  - Any smoothed band: Green 600-1250, Green-Jacobi 77-166.
- **Scaling with the grid:** `dual_preconditioner/TO_JCTAM_setup/figures/JCTAM_scaling.png` and the bottom-right panel
  of `talk_cost.png`.
  - True void (kappa = 0): only Green-Jacobi works.
  - kappa_void = 1e-5: Green-Jacobi is the cheapest (121), dual 450, Green 841.
  - kappa_void = 1e-3: Green-Jacobi 46, dual 73, Green 102.
- Honest point: this is Green-Jacobi territory. The dual's bound degrades with near-void.

## 6. Phase-field fracture, small strain (2-3 slides)
- **Periodic AT2 with pores:**
  - Movie or final frame: `dual_preconditioner/damage_AT2_small_strain/figures/AT2_N1024_dual_polydisperse_pores_l0.03125_final.png`.
  - Cost: `dual_preconditioner/figures/talk_fracture_cost.png` (a, b). The dual is flat at about 166 per solve from
    256^2 to 1024^2. Green and Green-Jacobi grow to about 330 at 256^2, so the dual is 2x cheaper in cost and also
    cheaper in total.
- **Dirichlet BCs with the corrected Green** (`linear_elastic_mesh_study/report/corrected_green.pdf`):
  - Sketch of M_GD = (Q K_0 Q)^-1 and the capacitance system: `report/figs/fixed_sets.png`.
  - Clamped cantilever: `linear_elastic_mesh_study/exp_data/cantilever_fracture_study.png`. The load-deflection curve
    is mesh converged; the dual needs 2x fewer iterations.
  - In cost (`talk_fracture_cost.png` c, d), the corrected dual equals the corrected Green, both for the cantilever and
    for the 3D house. Corrected Green-Jacobi is 2.5x worse.
- Optional: the house movie `linear_elastic_mesh_study/exp_data/house_damage_h0.125_ft0.2.mp4`.

## 7. Robustness summary (1 slide)
- Figure: `dual_preconditioner/figures/talk_regret.png`. For 48 cases it shows each preconditioner's cost divided by
  the cost of the best one in that case.
- Worst case / median, all cases:

  | Preconditioner | Worst | Median | Best in |
  |---|---|---|---|
  | Green | 10x | 1.0x | 25/48 cases |
  | Green-Jacobi | 130x | 2.7x | 16/48 cases |
  | dual | 16x | 1.5x | 8/48 cases |

- Worst case / median, by problem class:

  | Problem class | Green | Green-Jacobi | dual |
  |---|---|---|---|
  | linear | 3.1 / 1.0 | 130 / 4.6-16 | 2.6 / 1.3-1.6 |
  | topology optimization | 10 / 2.4 | 1.0 / 1.0 | 16 / 3.7 |
  | fracture | 2.0 / 1.0 | 2.7 / 2.4 | 1.8 / 1.0 |

- Reading: outside near-void topology optimization, the dual is never worse than 2.6x the best. Green-Jacobi is
  unbeatable with void and poor elsewhere.

## 8. Conclusions and open problems (1 slide)
- Choose by the material.
  - Sharp two-phase, moderate contrast: Green.
  - Void or smoothed interfaces: Green-Jacobi.
  - Strongly heterogeneous with no void, or damage at fine grids: the dual. It has guaranteed bounds and is flat in
    grid and contrast.
- Open:
  1. Cut the dual's factor 2 per iteration. At 1024^2 the measured ratio is already 1.47.
  2. A dual that tolerates void.
  3. 3D.
  4. Corrected Green for large clamped sets: the FFT capacitance solve from `corrected_green.pdf`.

---

## Data caveats (do not show, but know them)
- **Cost factor 2 is an operation count.** Measured wall time varies 1.5-2.9 (hardware noise at 256^2).
- **TO kappa_void = 1e-3:** Green and Green-Jacobi beyond 64^2 are short-run estimates from a formed design.
- **AT2:** Green and Green-Jacobi stop at 256^2; only the dual ran at 512^2 and 1024^2.
- **Cantilever:** there is no Green-Jacobi run.
- **House:** a frozen-damage comparison (five load steps), not a full fracture run per preconditioner.
- **Results before 2026-10-16** with two Green operators on one discretization are invalid
  (`topology_optimization/figures/invalid_shared_green_field/`); none are used here.
- **Not in the talk:**
  - Finite-strain / third-medium contact, where the dual loses.
  - Adapted grids, where the dual degrades.
