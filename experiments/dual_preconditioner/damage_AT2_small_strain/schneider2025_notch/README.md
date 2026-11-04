# Single notch of Schneider & Kästner (2025): residual stiffness vs preconditioner

Reference: T. Schneider, M. Kästner, "FFT-Based Phase-Field Fracture Modeling of Periodic Inhomogeneous
Microstructures", Fatigue & Fracture of Engineering Materials & Structures 48 (2025) 1782-1805,
https://doi.org/10.1111/ffe.14553 (open access; PDF in this folder).

## What they do

- AT2 phase field (Miehe), hybrid formulation (Amor split optional), finite strain (St. Venant-Kirchhoff), history
  field or "hybrid" irreversibility (d frozen above 0.95), inhomogeneous Gc(X); implemented in DAMASK / PETSc.
- Mechanics: basic Lippmann-Schwinger (Moulinec-Suquet) scheme, Green operator with the reference
  0.5 (min + max) stiffness, Willot's rotated finite-difference scheme for the gradients, matrix-free nonlinear GMRES.
  Phase field: Chen et al.-type FFT fixed-point split, Anderson mixing, numerical viscosity eta. Fully implicit
  staggered scheme, adaptive load stepping.
- Sec. 4.1: periodic single notch (50 x 50 mm, notch 10 mm), tension and shear, against an ABAQUS UMAT (small strain):
  peak ~187 MPa, crack initiation at E_22 = 8.2e-4; their Fig. 5: mechanical iterations per increment 10-20 before
  the crack, ~1e4 at the break ("very high stiffness contrast").
- Sec. 4.3.2 / Fig. 14 (two-phase microstructure, 501^2, E 210 / 21 GPa): k_res = 1e-6 and 5e-6 did not finish in
  24 h, 1e-5 ~330 min, 1e-4 ~200 min, 5e-4 ~110 min -> they recommend k_res = 1e-4 (contrast is the bottleneck of
  their Green-based solver).
- Sec. 4.3.3: 501^2 - 1201^2, iterations per increment roughly constant with resolution (fixed l = 6 mm, l/h grows);
  Sec. 4.3.4: 3D 256^3 in 8 h on 256 tasks.

## What we run (`run_AT2_notch_kres.sh`)

Their single notch under tension (Sec. 4.1.1, Fig. 3) in our small-strain code (`example_2D_phase_field_fracture_AT2.py`,
a copy of the study script with the options NOTCH, LAME, GC, LOAD_DIR, LOAD_MAX, LOAD_STEPS, PORE_CONTRAST, K_RESIDUAL):

| | paper | here (length unit 50 mm, stress unit E) |
|---|---|---|
| cell | 50 x 50 mm, periodic, plane strain | 1 x 1, periodic, plane strain |
| notch | x = 5 ... 15 mm at mid height, one voxel row of stiffness 1e-6 | `NOTCH=0.1,0.3,0.5`, one pixel row, stiffness = k_res, damage solved in it |
| l | 1 mm | `LENGTH_SCALE=0.02` |
| Gc | 2.7 N/mm | `GC=2.5714e-7` (= 2.7 / (210000 * 50)) |
| E, nu | 210 GPa, 0.3 | `LAME=0.576923,0.384615` (true plane-strain lambda, mu for E = 1, nu = 0.3) |
| load | E_22 (F_22) linear to 1.1e-3 in 100 increments | `LOAD_DIR=yy LOAD_MAX=1.1e-3 LOAD_STEPS=100` |
| k_res | 1e-6 (Fig. 14: 1e-6 ... 5e-4) | 1e-6, 1e-5, 1e-4 |
| grid | 250 x 250 (h = 0.2 mm, l/h = 5) | 64^2 (l/h = 1.28), 128^2 (l/h = 2.56) first |
| staggered tol. | eps_a = 1e-2, eps_r = 1e-3 | max abs. damage change 1e-3; CG energy-error rtol 1e-6 |

Preconditioners of the elastic problem: Green (C_ref = C_iso), Green-Jacobi, dual (2 K applications per iteration).
Difference to the paper: small strain (their FEM reference is small strain; the difference is negligible here),
Q1 FE discretization instead of the rotated scheme, CG instead of NGMRES, history-field irreversibility.

`plot_AT2_notch_kres.py` -> `figures/AT2_notch_kres.png`: stress-strain curves, CG iterations per solve over the load
at k_res = 1e-6, total elastic work against k_res.

## Results (2026-11-01, 64^2 and 128^2)

Physics: peak 187.5 MPa at E_yy = 0.85e-3 on 128^2 (paper 187 MPa, crack initiation 8.2e-4), 192.6 MPa on 64^2;
independent of k_res and of the preconditioner. The cell breaks within one load step, as in the paper.

Total elastic work of the run (CG iterations, dual x 2):

| N | k_res | Green | Green-Jacobi | dual |
|---|---|---|---|---|
| 64 | 1e-6 | 9 882 | 8 751 | 92 108 |
| 64 | 1e-5 | 9 879 | 8 753 | 65 810 |
| 64 | 1e-4 | 9 902 | 8 758 | 42 434 |
| 128 | 1e-6 | 14 133 | 12 612 | 197 600 |
| 128 | 1e-5 | 14 205 | 12 727 | 116 486 |
| 128 | 1e-4 | 13 773 | 12 299 | 61 488 |

CG iterations per elastic solve, 128^2: before the crack Green 27, Green-Jacobi 29, dual 34 (k_res = 1e-4) / 45
(1e-6); after the break Green ~100, Green-Jacobi ~75, dual 274 (1e-4) / 1362 (1e-6).

- Green and Green-Jacobi do not care about k_res at all here: the cell is homogeneous apart from one soft pixel row
  (and later the crack, of the same stiffness k_res): an exactly two-phase field with sharp interfaces, the case in which
  Green is efficient. With our Q1 discretization and CG, k_res = 1e-6 is no problem -- the paper's difficulty (Fig. 14)
  was on its two-phase microstructure (stiffness ratio 10 plus cracks), not on the notch.
- The dual loses 5-15x and degrades with decreasing k_res: its lambda_max grows with the stiffness contrast it sees
  (1/k_res at a one-pixel notch and along the crack); contrast-dependent level, as in the 2D contrast study.
- Consistent with the rule of the main study: two-phase sharp field -> Green; the dual pays off with three or more
  stiffness levels / graded fields (inclusions, damage in pores, smooth cracks). Next: the paper's two-phase
  microstructure (E 210 / 21 GPa, Gc 2.7 / 4.7) is the relevant test for the dual.

## Two-phase Cahn-Hilliard microstructure (the paper's Sec. 4.3 test), `run_AT2_cahn_hilliard_kres.sh`

Their exact image (Seibert et al. 2024) is not available; `make_cahn_hilliard_geometry.py` generates a similar periodic
spinodal (Cahn-Hilliard) structure (512^2, phase 2 fraction 0.42, seed 0, `geometry/cahn_hilliard_512.npy`,
`figures/cahn_hilliard_geometry.png`), sampled on the simulation grids (option `GEOMETRY_FILE`). Phase 1 E = 210 GPa
(cracks), phase 2 E = 21 GPa (stiffness ratio 10, as in the paper), here tough (no crack driving force; the paper:
Gc 4.7 vs 2.7 N/mm, cracks in phase 1); 1 x 1 m cell, Gc = 2.7 N/mm, l = 1/32 (31 mm; paper 6 mm, not resolvable on
these grids), true plane strain, E_yy to 2e-3 in 200 steps. Stiffness contrast: 10 before cracking, ~1/k_res after
(measured 9.4e5 at k_res = 1e-6, 1.2e4 at 1e-4). Stepwise cracking, peak 31.2 MPa (128^2), residual stress ~6-7 MPa at
2e-3, increasing with k_res (as their Fig. 14a).

Total elastic work (CG iterations, dual x 2), `figures/AT2_cahn_hilliard_kres.png`:

| N | k_res | Green | Green-Jacobi | dual | hybrid Green -> dual |
|---|---|---|---|---|---|
| 64 | 1e-6 | 888k | **213k** | 1081k | |
| 64 | 1e-5 | 704k | **217k** | 508k | |
| 64 | 1e-4 | 657k | **300k** | 365k | |
| 128 | 1e-6 | 1992k | **580k** | 1725k | (running) |
| 128 | 1e-5 | 1668k | **611k** | 809k | 809k |
| 128 | 1e-4 | 1204k | 770k | **536k** | 535k |

- 64^2 -> 128^2: Green-Jacobi x2.6-2.7, Green x1.8-2.4, dual x1.5-1.6: the dual has the best grid scaling, and at
  128^2 it wins at the paper's recommended k_res = 1e-4 (1.44x vs Green-Jacobi, 2.25x vs Green).
- The dual's cost roughly doubles per decade of k_res (contrast-dependent level); Green-Jacobi hardly depends on k_res
  and wins at k_res <= 1e-5 on these grids. Green is the most expensive everywhere (Fig. 14 of the paper: their
  Green-based solver could not finish k_res <= 5e-6).
- Before the crack: Green ~25, dual ~17 (x2 = 34), Green-Jacobi ~180 iterations per solve; ~96 % of the solves are in
  the long stepwise cracking phase, so the hybrid Green -> dual (`green_dual`, switch at max d >= 0.95; GPU,
  `run_AT2_cahn_hilliard_green_dual_gpu.sh`) gives the same work as the dual -- dropped (64^2 hybrid runs not done).
- Note 2026-11-01 19:20: NuMPI was rebased (now imports array_api_compat; installed into the muffto-gpu env); runs that
  started after it failed on import and were restarted.

### Larger grids at k_res = 1e-4 (GPU, `run_AT2_cahn_hilliard_large_gpu.sh`; fixed l = 1/32: l/h = 2 / 4 / 8)

Total elastic work (dual x 2), work per elastic solve in brackets:

| N | Green | Green-Jacobi | dual |
|---|---|---|---|
| 64 | 657k (211) | **300k** (97) | 365k (117) |
| 128 | 1204k (276) | 770k (177) | **536k** (123) |
| 256 | 756k (318) | 861k (362) | **262k** (110) |
| 512 | - | 2119k (706) | **339k** (113) |

The dual's work per solve is grid independent (117 / 123 / 110), Green-Jacobi x3.7 and Green x1.5 from 64^2 to 256^2;
crossover at ~128^2, at 256^2 the dual is 3.3x cheaper than Green-Jacobi and 2.9x than Green. 512^2 (dual,
Green-Jacobi): dual 339k, 113 per solve (flat), 3002 solves (2376 at 256^2), 1.4 h on the GPU (2026-11-03); Green-Jacobi
1.6 h on NEMO, 2119k, 706 per solve: the dual is 6.3x cheaper at 512^2 (data from NEMO, 2026-11-04; the local run was stopped).
Figure: `plot_AT2_cahn_hilliard_large.py` -> `figures/AT2_cahn_hilliard_large.png` (work vs grid, work per solve,
stress-strain, CG iterations per load step). The runs stop on their own once sigma < 0.2 peak for 3 steps: 64^2 / 128^2
keep a residual stress above that and run to 2e-3, 256^2 stops at 1.09e-3, 512^2 at 0.84e-3. The figure therefore cuts
every run at the same load step, the last one all finished runs reached (84 steps, strain 0.84e-3). Same load steps,
total work (work per solve):

| N | Green | Green-Jacobi | dual |
|---|---|---|---|
| 64 | 173k (140) | **116k** (94) | 123k (100) |
| 128 | 327k (234) | 245k (175) | **151k** (108) |
| 256 | 354k (265) | 412k (309) | **125k** (93) |
| 512 | - | 2119k (706) | **339k** (113) |

512^2 dual: work per solve flat, the total grows because the break needs up to ~1000 staggered iterations.

### Phase 2 cracks too, with the paper's toughness (`run_AT2_cahn_hilliard_gc2.sh`, 2026-11-03)

Option `PHASE2_GC` (script copy here): phase 2 (stiffness 0.1) is no longer tough but cracks with Gc_2 = 1.7407 Gc
(paper 4.7 / 2.7 N/mm); per-pixel Gc in both terms of the damage operator, the damage Green preconditioner keeps the
matrix Gc as reference. k_res = 1e-4, CPU. Physics: one straight crack through both phases in a single break at
E_yy = 4e-4, the cell separates (final stress 3-4 % of the peak, against ~16-25 % with tough phase-2 bridges), peak
1.38e-4 (64^2) / 1.35e-4 (128^2) against 1.50e-4; the run stops after 45 load steps.

Total work (dual x 2), work per solve before / after the peak in brackets:

| N | Green | Green-Jacobi | dual | solves (after the peak) |
|---|---|---|---|---|
| 64 | 43k (24 / 127) | 47k (87 / 112) | **42k** (34 / 120) | 446 (317) |
| 128 | 117k (24 / 283) | 98k (159 / 193) | **64k** (34 / 148) | 528 (405) |
| 256 | 506k (24 / 622) | 272k (298 / 290) | **143k** (34 / 172) | 934 (808) |
| 512 | 912k (24 / 929), 41 min NEMO GPU | 567k (577 / 499), 27 min NEMO GPU | **164k** (34 / 162), 6.6 h laptop CPU | 1114 (978) |

256^2 (overnight 2026-11-03, CPU, `NS="512 256"`): dual 1.2 h, Green-Jacobi 2.1 h, Green 4.4 h of run time. All grids
break at load step 42 and stop at step 44; the break step needs 695 staggered iterations at 256^2 (most of the solves).
Figure: `GC2=1 python plot_AT2_cahn_hilliard_large.py` -> `figures/AT2_cahn_hilliard_gc2.png`.

- Most solves are the staggered iterations of the break step. 64^2: all three equal. 128^2: the dual is 1.5x cheaper than
  Green-Jacobi and 1.8x cheaper than Green. After the peak, from 64^2 to 128^2: Green x2.2, Green-Jacobi x1.7, dual x1.2
  per solve. Before the peak: Green ~24 (two-phase field), Green-Jacobi grows with the grid.
- Same ranking as with the tough phase 2 (dual from 128^2), the run is just much shorter.
- Videos: `figures/AT2_N{64,128}_dual_incl0.1_gc1.7407_l0.03125_k0.0001_ch_tile2.mp4` (2 x 2 periodic copies).
