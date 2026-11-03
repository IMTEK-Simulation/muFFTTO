# Random UD-fibre fracture benchmark of Ernesti, Schneider & Böhlke (2020): Green vs dual

> F. Ernesti, M. Schneider, T. Böhlke, *Fast implicit solvers for phase-field fracture problems on heterogeneous
> microstructures*, CMAME 363 (2020) 112793, Sec. 4.4 (local copy of the PDF, not in git).

## The paper
* AT2-type phase field (quadratic degradation k0 + (1-k0)(1-d)², crack term Gc[d²/(4η) + η|∇d|²]), damage only in
  the matrix, trivial split (Amor / Miehe give the same curves in tension), no history field by default.
* Staggered-grid FD for u, Moulinec–Suquet for d; elastic preconditioner always Green (isotropic reference).
* Outer schemes compared: **alternating minimisation** (damage exact, then elasticity by Green-CG with Eisenstat–Walker
  forcing) vs **heavy ball** (damage exact, then one momentum gradient step in u); basic scheme too slow.
* Sec. 4.4: 8 random arrangements of 16 glass fibres (45 %) in polyamide, 256² pixels, h = 2 µm, η = 3 µm,
  k0 = 1e-4, E_xx in 0.1 % steps. Failure at 1.3 %, peak ≈ 114 MPa. **Alternating minimisation failed to converge
  (30 000-iteration cap) in the break step on 7 of 8 structures**; heavy ball converged (≈ 8k strain iterations).

## Our setup (`example_2D_phase_field_fracture_AT2.py` with `FIBERS=1`, `fiber_geometry.py`)
* 16 fibres, 45 %, random sequential addition (min. gap 2 µm, `FIBER_GAP_UM`), seed 0; cell 512 µm.
* Polyamide E = 3.45 GPa, ν = 0.39 (brittle); E-glass E = 72 GPa, ν = 0.22 (elastic, never degraded, no crack
  driving); plane strain (3D Lamé constants). C_d = g(d) χ_m C_m + χ_f C_f as a two-term `ScaledMaterial`; the dual
  C_m C_d⁻¹ C_m = χ_m/g C_m + χ_f C_m C_f⁻¹ C_m (on the fly / precomputed, no stored tensor field).
* l = 2η = 6 µm (from matching the crack term; the paper's text says η = 2l, its formulas give l = 2η),
  Gc = 0.1 N/mm, k = 1e-4, E_xx 0.1 % steps. Our AT2: Amor split, Miehe history field, staggered (alternating
  minimisation) with the elastic CG to a fixed energy tolerance 1e-6, damage CG with Green-Jacobi.
* Runs: `run_fibers_small.sh N seed` (`PRECS="Green dual"` to choose), CPU serial.

## Results (seed 0)

Physics: the pre-peak curve matches the paper (0.4 %: 52–54 MPa vs ≈ 52; 0.8 %: 91–94 vs ≈ 92), but our cell breaks
at **0.9 %** on every grid (peak 94.4 / 91.8 / 90.9 MPa at 64² / 128² / 256²) instead of 1.3 % / 114 MPa. Not a
resolution effect; suspected: the fibre arrangement (RSA allows near-touching fibres; the paper's Torquato–Jiao
packings look more even) – open.

Elastic preconditioners (alternating minimisation, all give the same curve):

| grid (l/h) | | before the crack its/solve | break step: staggered its, CG its (per solve) | total elastic work | wall time |
|---|---|---|---|---|---|
| 64² (0.75) | Green | 33 | 264, 45.7k (173) | **49.9k** | 48 s |
| | Green-Jacobi | 130 | 264, 52.3k (198) | 60.5k | 60 s |
| | dual | 21 | 264, 57.3k (217) | 126.5k | 101 s |
| 128² (1.5) | Green | 33 | 199, 71.3k (358) | **76.5k** | 282 s |
| | Green-Jacobi | 240 | 199, 76.4k (384) | 92.2k | 334 s |
| | dual | 21 | 199, 81.6k (410) | 173.0k | 526 s |
| 256² (3) | Green | 28–36 | 118, 97.5k (**826**) | 104.6k | 1598 s |
| | dual | 20–21 | 118, 38.5k (**326**) | **83.8k** | **1332 s** |

(work: CG iterations, dual × 2; break step = the load step at 0.9 %, > 90 % of the work.)

### Comments
1. **At the paper's resolution (256², l/h = 3) the dual wins**: in the break step it needs 2.5× fewer iterations per
   solve than Green (326 vs 826), 20 % less total work and 17 % less wall time. At 64²–128² Green was cheaper.
2. **Green's break-step iterations grow with the resolution** (173 → 358 → 826 per solve, ≈ ×2.2 per refinement,
   ∝ 1/h), the dual's do not (217 → 410 → 326). This is the growth behind the paper's non-convergence of Green-CG
   alternating minimisation. Before the crack both are grid-independent (Green ≈ 30, dual ≈ 21).
3. Explanation consistent with the gypsum study: at l/h ≤ 1.5 the crack is a sharp, almost void band (Green's good
   case, the dual's 1/g weighting hurts); at l/h = 3 the damaged band is smooth over several pixels next to stiff
   fibres – Green's bad case, the dual's good case. Extrapolating, the gap should widen at 512².
4. The number of staggered iterations of the break step falls with the resolution (264 → 199 → 118) and is the same
   for all preconditioners – the outer scheme, not the preconditioner, sets it (the paper's heavy ball attacks this).

### Open
* Failure strain 0.9 % vs 1.3 %: packing (Torquato–Jiao-like, larger gaps), trivial split / no history like the paper.
* More seeds (paper: 8), 512², Green-Jacobi at 256², a dual heavy-ball outer scheme.
