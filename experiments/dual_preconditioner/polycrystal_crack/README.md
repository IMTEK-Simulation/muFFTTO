# Polycrystal with an edge crack: elastic preconditioners

Test of the elastic preconditioners of the dual-preconditioner study (Green, Green-Jacobi, dual) on the cracked
polycrystal of

> N. B. Nkoumbou Kaptchouang, L. Gélébart, *Multiscale coupling of FFT-based simulations with the LDC approach*,
> Comput. Methods Appl. Mech. Engrg. 394 (2022) 114921, Sec. 5.3, Figs. 18–19.
> (`NkoumbouKaptchouang_Gelebart_2022_CMAME_LDC_FFT.pdf`, local copy, not in git)

## The problem in the paper

* Sec. 5.2: a 3D Voronoi polycrystal of 15 × 15 × 15 grains, isotropic linear elastic grains with Young modulus
  drawn at random in **[1, 3] GPa**, **ν = 0.3** (contrast 3). Fully refined grid ≈ 35 voxels per grain (525³).
* Sec. 5.3 (used here): a **2D section** of it, with
  * a **crack** of one line of voxels with **C = 0**, vertical, from the top of the cell down to its centre (Fig. 18);
  * a **layer of 2 void voxels (C = 0) along the top**, which makes the top/bottom of the periodic cell stress-free,
    so the cell is in simple tension;
  * **load: tension in x** (horizontal, perpendicular to the crack).
* Their topic is local refinement (LDC multigrid, FD/Willot discretisation, basic Moulinec–Suquet type solver).
  The result shown is the von Mises field (Fig. 19): concentration at the crack tip in the cell centre, an unloaded
  wedge above it. They report no solver-iteration data for this case, so there is nothing to compare iteration counts
  against: here the problem only serves as a hard benchmark geometry.

## Our setup (`polycrystal_geometry.py`, `polycrystal_crack_linear.py`)

* Periodic Voronoi with 225 seeds in the unit square (seed 2022), E per grain uniform in [1, 3], ν = 0.3; sampled at
  the pixel centres, so every grid sees the same grains.
* Crack: one pixel column at x = 1/2 from y = 1/2 to the top; void layer: the top 2 pixel rows (in pixels, as in the
  paper, so both get thinner with h). Crack and void have the stiffness **k × E_min** (k = 1e-2, 1e-4, 1e-6) instead
  of 0, because the dual needs C⁻¹.
* Q1 finite elements (bilinear_rectangle), small strain, periodic fluctuation, macroscopic strain E_xx = 1e-3
  (other components 0; the void layer takes the Poisson contraction).
* All materials are s(x) C_iso, so C is a `domain.ScaledMaterial`; preconditioners exactly as in the AT2 study:
  Green (C_ref = C_iso), Green-Jacobi (D^-1/2 G(2μ I) D^-1/2), dual G K(C_ref C⁻¹ C_ref) G (two FFT pairs + one K
  per iteration: **cost 2** per iteration).
* CG: energy-error estimate ≤ 1e-6 × energy of the solution, cold start, adaptive delay d_min = 10.
* Assumptions not fixed by the paper: crack length (taken: half the cell, tip at the centre, as drawn in Fig. 18),
  plane formulation (2D Lamé of the code), the random grain layout.

## Stage 1: linear solve, small grids (`run_stage1_small.sh`, `plot_stage1.py`)

32², 64², 128² (≈ 2, 4, 8.5 pixels per grain; the paper's fine grid has ≈ 35), CPU, serial.
Figure: `figures/polycrystal_stage1.png`.

CG iterations (dual: work = 2 × iterations); σ_xx/E_xx agrees between the preconditioners to all printed digits:

| N | k | Green | Green-Jacobi | dual | σ_xx/E_xx |
|---|---|---|---|---|---|
| 32 | 1e-2 | 53 | 75 | 57 | 1.342 |
| 64 | 1e-2 | 66 | 121 | 62 | 1.442 |
| 128 | 1e-2 | 81 | 200 | 64 | 1.589 |
| 32 | 1e-4 | 56 | 81 | 107 | 1.217 |
| 64 | 1e-4 | 71 | 138 | 173 | 1.215 |
| 128 | 1e-4 | 96 | 262 | 274 | 1.217 |
| 32 | 1e-6 | 56 | 82 | 166 | 1.216 |
| 64 | 1e-6 | 71 | 139 | 279 | 1.212 |
| 128 | 1e-6 | 96 | 264 | 516 | 1.211 |

### Comments

1. **Green is the clear winner here** – the opposite of the gypsum stiff-grain study. Its iterations do not depend on k
   at all (56/71/96 for k ≤ 1e-4) and grow only mildly with the grid (≈ ×1.3 per refinement). At 128², k = 1e-6 it
   needs 96 iterations against 1032 work units of the dual (≈ 11× cheaper).
2. **Green-Jacobi** doubles per refinement (∝ 1/h), as in the gypsum study, and is independent of k for k ≤ 1e-4.
3. **The dual depends on the void stiffness**: with a moderate void (k = 1e-2) it is flat in the grid (57/62/64) and
   the best per iteration; for k → 0 its iterations grow both with 1/k and with the grid (k = 1e-6: 166 → 279 → 516).
   The dual's metric C_ref C⁻¹ C_ref is 1/k in the void pixels; here the void is thin (1–2 pixels) and the top layer
   spans the whole cell. In the gypsum pores (k = 1e-4, round pores of many pixels) the dual stayed competitive with
   Green; the thin, percolating void seems to be what hurts it – a hypothesis, not yet tested (see next steps).
4. The physics behaves as expected: the von Mises field reproduces Fig. 19 (tip concentration, unloaded wedge above
   the crack, grain-wise variation below); σ_xx/E_xx for k ≤ 1e-4 converges to ≈ 1.21 already on these grids. For
   k = 1e-2 the 2-pixel void layer still carries load and gets thinner with h, so σ_xx grows with N.
5. Consistent with the earlier observation that **Green is cheap where every quadrature point is in exactly one
   phase**: here all phases are pixel-wise constant (no smoothed bands) and the grain contrast is only 3; the only
   strong contrast is a sharp void, which Green handles well.

## Stage 2: AT2 crack growth from the pre-crack (`example_2D_phase_field_fracture_AT2.py`, small scale)

Copy of the gypsum AT2 script with `POLYCRYSTAL=1` (grain stiffness E/E_min per pixel, ν = 0.3; pre-crack + void layer
as pores of stiffness PORE_CONTRAST = 1e-4, damage free there) and `GC_GB=<factor>` (Gc × factor on the 2-pixel
grain-boundary band, 38 % of the cell). Gc = 1e-5 (units E_min = 1), LOAD_MAX = 6e-3 in 60 steps, damage solver
Green-Jacobi in every run, CPU serial. Runs: `run_stage2_small.sh 64|128`, `run_stage2_gb.sh 128 0.3`;
crack path: `plot_stage2_crack_path.py` → `figures/polycrystal_stage2_crack_path.png`.

Total elastic work (CG iterations of the elastic solves over the whole run, dual × 2); all three give the same
stress-strain curve (6 digits):

| run | l (l/h) | peak σ_xx | Green | Green-Jacobi | dual |
|---|---|---|---|---|---|
| 64², constant Gc | 0.031 (2) | 3.80e-3 | **43.7k** | 47.8k | 221.6k |
| 128², constant Gc | 0.031 (4) | 3.78e-3 | **53.5k** | 65.1k | 198.9k |
| 128², GB Gc × 0.3 | 0.016 (2) | 3.30e-3 | **66.4k** | 111.3k | 416.4k |

* The crack grows from the tip of the pre-crack down to the bottom (the periodic image of the void layer) and cuts
  the cell; with constant Gc it is almost straight (the grains differ only in E, contrast 3). Most of the cost is the
  break step (one step with 136–274 staggered iterations).
* Weak grain boundaries (Gc × 0.3): the crack zig-zags and breaks in two stages; 61 % of its pixels lie on the
  boundary band (44 % with constant Gc, 38 % = band fraction): partly intergranular, but still mostly straight –
  at 128² a grain is only ≈ 8.5 pixels ≈ 4 l, the 2-pixel band is coarse.
* **Green stays the cheapest in every run**, also with the zig-zag crack; Green-Jacobi 1.1–1.7×, the dual 4–6×.
  The dual is slow already before any damage (≈ 140 iterations per solve at 64², as in stage 1 with k = 1e-4):
  the thin void (pre-crack + void layer), not the damage, is what slows it down here. Unlike the gypsum stiff
  grains, the damage band does not hurt Green here (grain contrast only 3).

### Possible next steps

* Diagnostic for comment 3 (small, linear): dual with k = 1e-6 and (a) the crack only, no void layer; (b) a thicker
  crack (3–5 pixels) – which of the thin crack / the full-width layer drives the dual's growth.
* Larger grids (256², 512² ≈ the paper's 35 pixels per grain) – only after agreement (laptop GPU is busy with the gypsum
  queue; NEMO GPUs are in maintenance).
* Clearer intergranular cracking: fewer, larger grains (e.g. 6 × 6) at 128², or 256², and/or GC_GB = 0.1.
* Literature (Matti Schneider, KIT): Ernesti, Schneider, Böhlke 2020 (CMAME 363, FFT phase-field fracture solvers on
  heterogeneous microstructures), Ernesti & Schneider 2023 (Arch. Appl. Mech. 93, effective crack energy with weak
  interfaces), Risthaus & Schneider 2025 (IJNME 126, FFT solvers with voids and pores) – compare once the PDFs are here.
