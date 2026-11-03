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

## 8 realisations at 256² (seeds 0–7, Green and dual; `run_fibers_seeds.sh`, gallery `figures/fibers_gallery_N256.png`)

| seed | peak σ_xx (strain) | Green work | dual work | Green / dual | worst step: Green its/solve | dual its/solve |
|---|---|---|---|---|---|---|
| 0 | 90.9 MPa (0.8 %) | 104.6k | **83.8k** | 1.25 | 826 | 326 |
| 1 | 82.2 (0.7) | 237.4k | **232.8k** | 1.02 | 531 | 284 |
| 2 | 97.2 (0.9) | 202.4k | **163.0k** | 1.24 | 732 | 299 |
| 3 | 91.0 (0.8) | **178.0k** | 211.3k | 0.84 | 413 | 275 |
| 4 | 85.6 (0.7) | 199.4k | **144.9k** | 1.38 | 610 | 234 |
| 5 | 91.2 (0.8) | **212.0k** | 272.7k | 0.78 | 790 | 508 |
| 6 | 90.0 (0.8) | 111.6k | **63.2k** | 1.77 | 984 | 261 |
| 7 | 96.8 (0.9) | 228.6k | **123.8k** | 1.85 | 908 | 242 |
| mean | 90.6 ± 5.0 (0.80 ± 0.08) | 184k ± 50k | **162k ± 73k** | **1.26** | | |

Wall time (14 runs in parallel on 16 cores): Green 6556 s, dual 5785 s on average (dual 12 % faster).

* **The dual is cheaper on 6 of 8 structures** (up to 1.85×), Green on 2 (#3, #5: 0.84, 0.78); on average 12 % less
  work and wall time.
* In the expensive load step the dual always needs fewer iterations per solve (1.3–3.8×), but it pays off only when
  the ratio exceeds 2 (its cost per iteration): #3 and #5 have ratios 1.5 and 1.6.
* Both preconditioners see the same staggered iterations; the worst load step needs 92k–220k CG iterations with
  Green and 25k–132k with the dual (the paper's cap was 30 000 per step, with an adaptive inner tolerance – not
  directly comparable to our fixed 1e-6).
* Physics: 0.80 ± 0.08 % / 90.6 ± 5.0 MPa vs the paper's 1.29 ± 0.03 % / 114.1 ± 0.8 MPa – lower and with 6× more
  scatter: our RSA packings (near-touching fibres, visible as damage concentrations in narrow ligaments) differ
  from the paper's Torquato–Jiao packings. Crack paths are like the paper's Fig. 13 (one winding crack per cell,
  perpendicular to the load, around the fibres).

### Open
* Failure strain 0.9 % vs 1.3 %: packing (Torquato–Jiao-like, larger gaps), trivial split / no history like the paper.
* 512² (on the cluster), Green-Jacobi at 256², a Torquato–Jiao-like packing, a dual heavy-ball outer scheme.
