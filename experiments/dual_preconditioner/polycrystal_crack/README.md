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

### Possible next steps

* Diagnostic for comment 3 (small, linear): dual with k = 1e-6 and (a) the crack only, no void layer; (b) a thicker
  crack (3–5 pixels) – which of the thin crack / the full-width layer drives the dual's growth.
* Larger grids (256², 512² ≈ the paper's 35 pixels per grain) – only after agreement (laptop GPU is busy with the gypsum
  queue; NEMO GPUs are in maintenance).
* Stage 2: AT2 crack growth from the pre-crack through the polycrystal (the AT2 script with a per-pixel stiffness
  field, optionally Gc per grain / weaker grain boundaries) – here the damage creates smooth soft bands, where Green
  struggled in the gypsum grains, so the ranking may change again.
