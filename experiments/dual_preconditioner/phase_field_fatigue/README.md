# Phase-field fatigue fracture: Green vs Green-Jacobi vs dual preconditioner (2D prototype)

Reference: P. Carrara, M. Ambati, R. Alessi, L. De Lorenzis, "A framework to model the fatigue behavior of brittle
materials based on a variational phase-field approach", CMAME 361 (2020) 112731 (`Carrara2020_fatigue_phase_field.pdf`).

Question: fatigue = many elastic solves in an already cracked, slowly changing state. The dual preconditioner was nearly
grid independent on cracks (~40 CG per solve) while Green-Jacobi grows ~2x per refinement and Green ~sqrt(contrast);
does that pay off over a fatigue life?

## Model (`example_2D_phase_field_fatigue_AT2.py`)

Copy of `../damage_AT2_small_strain/schneider2025_notch/example_2D_phase_field_fracture_AT2.py` (AT2, Amor split, hybrid
formulation, history field H = max psi^+, staggered scheme, CG with the energy-error stopping rule rtol 1e-6, delay 10),
extended with the fatigue model of Carrara et al.:

- fatigue variable alpha = g(d) psi^+ (degraded tensile energy, their Eq. (13), alpha = psi_el^+), accumulated with the
  mean-load-independent rule (their Eqs. (39), (45)): alpha_bar_n = alpha_bar_{n-1} + <alpha_n - alpha_{n-1}>_+
- toughness Gc -> f(alpha_bar) Gc in both terms of the crack density (per quadrature point), asymptotic f (Eq. (41),
  `FATIGUE=asymptotic`, default) or logarithmic (Eq. (42), `FATIGUE=logarithmic`, `KAPPA`)
- alpha_T = Gc(x) / (12 l) (`ALPHA_T_FACTOR` x), their Eq. (47) as evaluated in their examples (5.625e1 N/mm^2 for
  Gc = 2.7 N/mm, l = 0.004 mm)
- damage operator: [B^T W f Gc l B + N^T W (f Gc / l + 2H) N] d = N^T W 2H, then d <- min(d, 1) (`CLIP_DAMAGE=1`; the
  consistent Q1 mass lets d overshoot to ~1.3 where f Gc -> 0)
- cyclic load (`CYCLES`, `AMPLITUDE`): R = 0, constant strain amplitude, ONE load step per cycle at the peak. The trough
  E = 0 is not solved: the periodic solution there is u~ = 0, alpha = 0, so each cycle adds the peak alpha (Carrara et
  al. use 8 steps per cycle; with R = 0 proportional loading the only approximation is that d is frozen during the
  load-up). Warm start of u~ from the previous peak.
- `MAX_STAGGERED` (default 500): a step that hits it is accepted (logged "ACCEPTED") and the run goes on.
- `NOTCH=x0,x1,y` together with `GEOMETRY_FILE`: notch laid over the microstructure, as a one-pixel row of stiffness
  `NOTCH_STIFFNESS` (default k_res, `NOTCH_MODE=stiffness`), or as a phase-field pre-crack: matrix stiffness and initial
  history H_0 = 1000 Gc / l in the notch pixels (`NOTCH_MODE=history`, Borden et al. 2012)
- outputs per run in `exp_data/`: `*_cycles.csv` (one line per cycle: peak stress, d_max, crack functional Gamma,
  projected crack length (columns with d > 0.95, notch counted), broken area, min f, staggered iterations, CG sums,
  times), `*_iterations.csv` (one line per staggered iteration, as in the base script), `.nc` (fields every 10th cycle),
  `*_damage.npz` (first / final damage).

## Set-up (`run_FAT_cahn_hilliard_notch.sh`)

The two-phase Cahn-Hilliard cell of the Schneider & Kaestner test (`geometry/cahn_hilliard_512.npy`, copied from
`../damage_AT2_small_strain/schneider2025_notch/geometry/`): unit periodic cell, matrix E = 1 (true plane strain Lame
0.5769, 0.3846, i.e. nu = 0.3), phase 2 stiffness 0.1 with Gc_2 = 1.7407 Gc (both phases crack and fatigue),
Gc = 1.2857e-8, l = 1/16 FIXED on all grids (l/h = 2 / 4 / 8 on 32^2 / 64^2 / 128^2), k_res = 1e-4. Notch x = 0.375 ...
0.625 at y = 0.5 (two tips, periodic), uniaxial strain E_yy. Monotonic peak strain 3.4e-4 (32^2,
`logs/mono_N32_GJ.log`); fatigue amplitude 8e-5 (~0.24 of the peak strain; 1e-4 / 1.2e-4 / 1.5e-4 gave 95 / 55 / 27
cycles on 32^2, too fast). Run stops once the peak stress < 0.2 x max for 3 cycles.

    ./run_FAT_cahn_hilliard_notch.sh                       # NS="32 64 128", 3 runs at a time, CPU, MemoryMax 4G
    NOTCH_MODE=history ./run_FAT_cahn_hilliard_notch.sh    # phase-field pre-crack instead of the soft pixel row
    ~/micromamba/envs/muffto-gpu/bin/python plot_FAT_cahn_hilliard_notch.py      # figures + summary table
    TAG=_incl0.1_gc1.7407_l0.0625_k0.0001_precrack_asymptotic_amp8e-05_cyc1500 FIG_SUFFIX=_precrack ...  # pre-crack runs

## Results A: Cahn-Hilliard cell, scaled parameters (2026-11-06)

`plot_FAT_cahn_hilliard_notch.py` -> `figures/FAT_crack_growth*.png` (a: crack length, peak stress, da/dN vs cycle),
`FAT_cg_per_cycle*.png` (b), `FAT_cg_vs_grid*.png` (c), `FAT_damage_maps*.png` (mid-growth and final damage, dual runs).
Suffix `_precrack`: NOTCH_MODE=history; no suffix: soft notch row. "pre" = cycles before the crack is > 2 pixels
longer than after cycle 1, "growth" = from then to the final break (first > 30 % peak-stress drop in one cycle). CG per
solve = mean over all staggered elastic solves of the phase; work = CG iterations, dual x 2.

Physics: slow, stable, stepwise growth over ~100 cycles (both tips; the crack hops between phase-1 / phase-2 segments),
then an unstable final break; life 150-200 cycles, identical for the three preconditioners (same solution).
Initiation is grid dependent with the soft-row notch (crack leaves the notch at cycle 115 / 79 / 55 on 32 / 64 / 128,
sharper notch tip on finer grids), much less with the pre-crack (61 / 36 / 32). Mean da/dN in the growth phase
(pre-crack) 2.4e-3 / 2.8e-3 / 4.6e-3 L per cycle on 32 / 64 / 128.

Pre-crack (`NOTCH_MODE=history`), CG per elastic solve pre / growth, total work:

| N (l/h) | Green | Green-Jacobi | dual | work G / GJ / dual (x2) |
|---|---|---|---|---|
| 32 (2) | 92 / 93 | 48 / 55 | 49 / 51 | 116k / 67k / 127k |
| 64 (4) | 129 / 148 | 88 / 101 | 48 / 53 | 232k / 157k / 164k |
| 128 (8) | 142 / 268 | 147 / 205 | 40 / 56 | 480k / 371k / **202k** |

Soft notch row (one pixel row of stiffness k_res = 1e-4):

| N (l/h) | Green | Green-Jacobi | dual | work G / GJ / dual (x2) |
|---|---|---|---|---|
| 32 (2) | 32 / 43 | 44 / 51 | 61 / 94 | **74k** / 85k / 289k |
| 64 (4) | 36 / 142 | 69 / 96 | 81 / 139 | 240k / **181k** / 502k |
| 128 (8) | 45 / 224 | 125 / 198 | 102 / 168 | 472k / **438k** / 741k |

- With the phase-field pre-crack the dual is grid independent (~50 CG per solve before and during growth), Green-Jacobi
  doubles per refinement (55 -> 101 -> 205 during growth), Green grows too (93 -> 148 -> 268). At 128^2 the dual is
  1.8x cheaper than Green-Jacobi and 2.4x cheaper than Green in total work; crossover at ~64^2.
- With a sharp one-pixel soft notch row the dual loses its advantage (102-168 per solve at 128^2): the 1/k_res jump
  across a single pixel row is the dual's bad case (as in schneider2025_notch). Green is cheapest before growth there
  (exactly two-phase sharp field). Pre-cracks should be modelled as phase-field cracks for the dual.
- No step hit MAX_STAGGERED = 500 (max ~130 staggered iterations at the final break).
- Mistake 2026-11-06: editing run_FAT_cahn_hilliard_notch.sh while it ran made bash restart the 128^2 soft-row runs and
  overwrite their outputs; they were rerun (`logs/runner_stiffness_128_rerun.txt`). Do not edit a running .sh.

## Results B: Carrara's real parameters, glass beads in a polymer matrix (local, 2026-11-06)

Matrix = Carrara et al. Sec. 4.3 (Paris-law set): E = 6 GPa, nu = 0.22, Gc = 2.28 N/mm, l = 0.2 mm, alpha_T = Gc / (12 l)
= 0.95 N/mm^2, plane stress (lambda* = E nu / (1 - nu^2)), AT2, asymptotic f, R = 0. Inclusions: glass beads (assumption,
typical handbook values: E = 70 GPa, nu = 0.22 -> stiffness ratio 11.67, same nu so C_bead = 11.67 C_matrix), tough (no
crack driving force), ~20 %, periodic random non-overlapping disks (`make_glass_bead_geometry.py`). Phase-field pre-crack
0.25 L long at y = L/2 (NOTCH_MODE=history). k_res = 1e-4, stop when the peak stress < 0.3 x max for 3 cycles.

**B1 (`run_FAT_carrara_beads16.sh`, the valid set):** cell L = 16 l = 3.2 mm, l/h = 2 / 4 / 8 on 32^2 / 64^2 / 128^2,
bead diameters 0.38-0.64 mm (`geometry/glass_beads_r0.06-0.1_512.npy`), pre-crack 0.8 mm. Monotonic peak strain 1.4e-2
(69 MPa, `logs/mono_beads16_N64_GJ.log`), fatigue amplitude E_yy = 2.8e-3 (0.2 x peak, peak stress ~20 MPa).
Figures `figures/*_carrara_beads16.png`; video `figures/FAT_video_N128_dual_..._beads16_..._cyc1500.mp4`
(`make_FAT_video.py`; every cycle stored with NC_EVERY=1 OUTPUT_FIELDS=damage).

| N (l/h) | life [cycles] | growth from / to | da/dN in growth | CG/solve pre, growth: Green | Green-Jacobi | dual | work G / GJ / dual(x2) |
|---|---|---|---|---|---|---|---|
| 32 (2) | 191 | 158 / 189 | 23 um/cycle | 93, 101 | 54, 65 | 58, 68 | 110k / **67k** / 143k |
| 64 (4) | 158 | 117 / 156 | 15 um/cycle | 145, 179 | 87, 104 | 52, 58 | 197k / **116k** / 132k |
| 128 (8) | 155 | 109 / 153 | 17 um/cycle | 225, 321 | 148, 180 | 41, 50 | 412k / 227k / **134k** |

- Physics: after ~110 cycles of fatigue accumulation at the tips the crack grows stably for ~40 cycles at ~15-17 um per
  cycle (grid converged at l/h >= 4; l/h = 2 gives a later, slower, more diffuse growth), deflects around the beads,
  then the remaining ligament breaks in one cycle. Same cycles/stresses for all preconditioners.
- Dual: grid independent / slightly decreasing (58 -> 52 -> 41 before growth, 68 -> 58 -> 50 during growth); Green-Jacobi
  ~1.7x per refinement; Green ~1.6-1.8x per refinement (stiff beads + crack). At 128^2 (l/h = 8) the dual is 1.7x
  cheaper than Green-Jacobi and 3.1x cheaper than Green in total work; crossover at 64^2.

**B0 (`run_FAT_carrara_beads.sh`, superseded):** the same with a cell of 64 l (12.8 mm, small beads r = 0.45-0.77 mm,
amplitude 2.5e-3), i.e. l/h = 0.5 / 1 / 2 only (design error: under-resolved). Dual 64 / 71 / 71 (pre) and 91 / 111 /
122 (growth) CG per solve, Green 49 / 64 / 101 and 60 / 93 / 174, Green-Jacobi 56 / 91 / 165 and 73 / 140 / 258; Green
cheapest in total work (89k / 96k / 169k) -- the dual needs a resolved crack band (l/h >= 2-4) to be grid independent.
Figures `figures/*_carrara_beads_lh0.5-2.png`. Life 225 / 123 / 108 cycles, da/dN 41 / 125 / 132 um per cycle.

## NEMO mesh study (2026-11-07, `sbatch_fatigue_gpu.sh`, `run_FAT_nemo2D_local.sh`, `plot_FAT_mesh_study.py`)

Carrara Sec. 4.3 parameters (as B), fixed l = 0.2 mm = L/64: cell L = 64 l = 12.8 mm, glass beads r = 5-8 l (1.0-1.6 mm),
20 % (`geometry/beads2D_2048.npy`: 6 disks; `geometry/beads3D_256.npy`: 52 spheres; uint8), phase-field pre-crack
x = 0.375 ... 0.625 L (3.2 mm = 16 l) at y = L/2 (in 3D a through crack, all z). 2D plane stress, 3D full 3D Lame.
R = 0 amplitude 0.25 x the monotonic peak strain: 2D 2.0e-3 (peak 8e-3 at 128^2, `logs/mono_nemo2D_N128_GJ.log`),
3D 2.2e-3 (peak ~9e-3 at 32^3, `logs/mono_nemo3D_N32_GJ.log`, the laptop 3D smoke test). l/h = N / 64: 2D 32^2 ...
1024^2 = 0.5 ... 16, 3D 32^3 ... 256^3 = 0.5 ... 4; l/h < 2 is under-resolved (flagged in table / figure).
OUTPUT_FIELDS=damage, NC_EVERY=5, checkpoint every 10 cycles (+ fatigue variable; restart tested bit-identical at 32^2),
segments of 47.5 h, MAXSEG 3.

- GPU smoke test 8657569 (64^2 dual, 15 cycles, L40S): stresses identical to the laptop, CG counts identical in the
  first cycles, +-3 iterations later (round-off).
- Local (CPU): 2D 32^2 / 64^2 / 128^2 (`run_FAT_nemo2D_local.sh`, done 2026-11-06).
- NEMO l40s: 2D 256^2, 512^2, 3D 32^3, 64^3 (12 jobs, 2026-11-07 00:32); next 3D 128^3, then 2D 1024^2 and 3D 256^3
  (h200, --mem 200G). Job list: /work/classic/fr_ml1145-muffto/runs/fatigue/jobs.txt; outputs in
  /work/classic/fr_ml1145-muffto/runs/fatigue/{exp_data,logs,figures} (symlinked); CSVs/logs copied back to exp_data/, logs/.

    ssh nemo 'cd ~/software/muFFTTO/experiments/dual_preconditioner/phase_field_fatigue && sbatch --job-name=FAT_2D_256_dual sbatch_fatigue_gpu.sh 2D 256 dual'
    ssh nemo '... sbatch -p h200 --mem 200G --job-name=FAT_3D_256_dual sbatch_fatigue_gpu.sh 3D 256 dual'
    rsync -a --include='*/' --include='*_cycles.csv' --include='*_iterations.csv' --exclude='*' nemo:/work/classic/fr_ml1145-muffto/runs/fatigue/exp_data/ exp_data/
    python plot_FAT_mesh_study.py      # -> figures/FAT_mesh_cg_vs_lh.png, FAT_mesh_crack_growth.png + table

### 2D mesh-study results (32^2-512^2 done 2026-11-07 03:00; 1024^2 queued)

| N | l/h | life | growth (cycles) | da/dN [um/cycle] | CG/solve pre: G / GJ / dual | growth: G / GJ / dual | work G / GJ / dual(x2) |
|---|---|---|---|---|---|---|---|
| 32 | 0.5* | 250 | (one jump) | - | 52 / 44 / 55 | - | 87k / 74k / 201k |
| 64 | 1* | 201 | 156-199 | 148 | 73 / 64 / 68 | 104 / 94 / 93 | 192k / 173k / 348k |
| 128 | 2 | 179 | 101-178 | 96 | 107 / 100 / 63 | 167 / 156 / 84 | 413k / 384k / 422k |
| 256 | 4 | 169 | 69-168 | 77 | 160 / 165 / 49 | 283 / 276 / 70 | 871k / 842k / **432k** |
| 512 | 8 | 168 | 54-166 | 68 | 230 / 285 / 36 | 500 / 502 / 58 | 2037k / 1842k / **457k** |

\* under-resolved (l/h < 2). Life converges (179 / 169 / 168 cycles), da/dN converges slowly (96 / 77 / 68 um per
cycle); the growth phase (~100 cycles) is stable, the last cycle is the unstable break. Green and Green-Jacobi double
their CG per solve with every refinement (growth: 167 -> 283 -> 500); the dual stays at 58-84 (decreasing) and is
4x cheaper than either at 512^2 in total work (dual iterations counted twice).

### NEMO job state at 2026-11-07 06:00 (then ssh to NEMO timed out; the jobs keep running there)

Done: smoke 8657569; 2D 512^2 8658019-21, 256^2 8658022-24; 3D 32^3 8658028/29/31, 3D 64^3 Green 8658025.
Running: 3D 64^3 Green-Jacobi 8658026 (near the end), dual 8658027. Pending (l40s): 3D 128^3 dual / Green /
Green-Jacobi 8658095 / 8658100 / 8658105, 2D 1024^2 dual / Green-Jacobi / Green 8658106 / 8658107 / 8658108;
(h200, 200G): 3D 256^3 dual 8658109, Green-Jacobi 8658110. Not yet submitted: 3D 256^3 Green (h200 budget 2).
To collect: the rsync line above, then `python plot_FAT_mesh_study.py`.

3D so far (all under-resolved, l/h = 0.5 / 1): CG per solve before / during growth 32^3: Green 54 / 76, GJ 53 / 67,
dual 227 / 240; 64^3: Green 74 / 109, GJ 87 / 116, dual 179 / 191 -- the dual is poor at l/h <= 1 (as in 2D B0 and the
Ernesti benchmark); the resolved 3D grids (128^3, 256^3: l/h = 2, 4) are the decisive ones.
