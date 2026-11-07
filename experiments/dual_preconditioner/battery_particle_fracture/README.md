# Chemo-mechanical fracture of a core-shell battery particle: FFT preconditioners

A prototype 2D study. It compares the elastic preconditioners **Green**, **Green-Jacobi (GJ)** and **dual** on a
coupled diffusion / swelling / AT2 phase-field fracture problem. The diffusion solve also gets a Green / GJ /
dual-like comparison. The model and parameters follow

> Y. Tu, B. Wu, E. Martínez-Pañeda, *Phase field modelling of cracking and capacity fade in core-shell cathode
> particles for lithium-ion batteries*, Applied Energy 403 (2026) 127105 (`Tu2026_core_shell_cathode_phase_field.pdf`)

and are cast into a periodic FFT cell. Script `example_2D_battery_particle_fracture_AT2.py` is adapted from (copied,
not imported) `../damage_AT2_small_strain/schneider2025_notch/example_2D_phase_field_fracture_AT2.py`.

## Model

The particle is an NMC811 core (radius R = 4 µm) with an NMC532 shell (thickness h = 0.3 R = 1.2 µm). It sits in the
centre of a periodic 12 µm x 12 µm cell, and the rest of the cell is a soft "matrix" (binder / electrolyte).
**Lithiation** runs at 1C from SOL 10 %.

| field | equation (discrete, per time step, implicit Euler) |
|---|---|
| Li filling c = c_Li / c_max,i (continuous at the interface) | `(1/dt) M_m c + B^T W m D(d) B c = (1/dt) M_m c_old + N^T W q` |
| displacement u~ (periodic, small strain, plane strain) | `B^T W g(d) C_0(x) B u~ = B^T W g(d) C_0(x) eps_Li` |
| damage d (AT2, Amor split, history H) | `[Gc l B^T W B + N^T W (Gc/l + 2H) N] d = N^T W 2H` |

* `m = c_max,i / c_max,core` (capacity weight); `D = D_i ((1-d)^2 + K_DIFF)`. Damage blocks Li transport (Tu
  Sec. 2.4). The matrix has `D_MATRIX = 1e-3`.
* `eps_Li = Omega_i c_max,i / 3 (c - c0) I` (Tu eq. 6). Core: 0.0136, shell: 0.0079 per unit filling. The core swells
  more, so the shell goes into hoop tension at the interface. The matrix gets no Li strain and does not crack
  (its psi^+ is not a driving force).
* `g(d) = (1-d)^2 + 1e-4` for the stiffness. Gc is set per phase (core 0.299 N/m, shell 0.408 N/m). The interface value
  is `Gc_I = GC_INTERFACE x Gc_ave`, blended with `zeta = exp(-|r-R|/L_ZETA)` (Tu eq. 20-23; default Gc_I = Gc_ave).
* The initial surface crack (Tu eq. 32) is a history value `H = alpha0 = 1e12 J/m^3` on one pixel row along +x. It
  runs from the outer surface inwards over 0.5 x the shell thickness.
* Charging: **CC** (Tu eq. 31) is a source spread uniformly over a surface band of width 1/32 inside the shell, giving
  the full capacity in 1/C hours. Once c reaches 1 in the band the run switches to **CV**: `q = KAPPA (1 - c)` in the
  band, KAPPA = 1e3. The run ends when the current falls below 10 % of the CC current or at `T_MAX`.
* Scales: length L = 12 µm, stress E_core = 230 GPa, time L^2/D_core = 4417 s, concentration c_max,core.
* **Macroscopic BC: zero mean strain (E = 0).** This models a particle constrained by its neighbours in a compacted
  electrode, and it is the simplest correct choice: `eigen_rhs = -eps_Li` goes straight into `get_rhs_mugrid`, and the
  elastic strain is `sym(grad u~) - eps_Li`. With a soft matrix (1e-2) the constraint on the particle is weak.
* Staggered scheme per time step: diffusion -> elasticity -> history -> damage, repeated until max |d - d_old| < 1e-3.
  If `MAX_STAGGERED` (default 500) is reached, the step is **accepted** and the log line says
  `MAX_STAGGERED HIT, step ACCEPTED`.
* All CG solves stop on the energy-error estimate (tol 1e-6, warm start, delay d_min 10), as in the base script.

### Deviations from Tu et al.

| Tu et al. 2026 | here | reason |
|---|---|---|
| 3D axisymmetric sphere, free surface | 2D plane-strain disk (a cylinder) in a periodic cell with a soft matrix (stiffness 1e-2 x core) | FFT / periodic cell; the soft matrix stands in for the free surface |
| l = 0.23 / 0.27 µm | l = 1/16 L = 0.75 µm, fixed on all grids (l/h = 2, 4, 8 at 32^2, 64^2, 128^2) | resolution: l/h >= 2 at 32^2. Effective strength sigma_c ~ sqrt(E Gc / l) is ~1.8x lower |
| reference shell h/R = 0.2 (assumed) | h/R = 0.3 (in the paper's range 0.1-0.3) | so the shell (1.2 µm, 3.2 px at 32^2) is wider than l |
| interface condition: continuous chemical potential (OCP curves) | continuous normalised filling c/c_max | no OCP data; single scalar field |
| stress-assisted flux -D c Omega/RT grad(sigma_h) | omitted (one-way coupling c -> u, plus d -> D) | keeps the diffusion solve linear and SPD |
| surface flux J0 | volumetric source / Robin (CV) term in a 1/32 surface band | no sharp boundary in the FFT cell |
| diffuse interface l_zeta = 0.1 µm | 0.375 µm (1/32 L) | resolution (only matters if GC_INTERFACE != 1) |
| initial crack a = 0.3 h, Gaussian H profile | a = 0.5 h, H = alpha0 on one pixel row | resolution |
| Li in the matrix: none | the matrix carries c with D = 1e-3 (slight leakage into the matrix, not counted in SOL) | the dual-like diffusion preconditioner needs D > 0 everywhere |

## Preconditioners

* Elasticity (argv[1]):
  * `Green`: C_ref = core tangent.
  * `Green_Jacobi`: C_ref = 2 mu I_sym, plus Jacobi scaling of the current K.
  * `dual`: `G K(C_ref C_d^-1 C_ref) G`, with C_d^-1 = C_iso / s evaluated on the fly.
* Damage: Green-Jacobi (as in the base script).
* Diffusion (`DIFF_PRECOND`, default Green): the reference operator is D_core Laplace + 1/dt mass, with the zero mode
  inverted.
  * `Green_Jacobi` adds Jacobi scaling.
  * `dual` is `G [N^T W (r_ref^2/r) N + B^T W (D_ref^2/D) B] G`.
  * With `DIFF_SHADOW=1` (on in the dual runs), the other two diffusion preconditioners also solve the same system
    from the same warm start. Their iterations are counted and the solutions thrown away, so all three are compared
    on an identical trajectory.
* Every Green operator has its own `name=` (`green_u`, `green_d`, `green_c`), and so does every Jacobi diagonal.

## How to run

```bash
./run_battery_grids.sh Green & ./run_battery_grids.sh Green_Jacobi & ./run_battery_grids.sh dual &   # 32, 64, 128
./run_battery_contrast.sh 1e-1 dual Green_Jacobi Green; ./run_battery_contrast.sh 1e-3 dual Green_Jacobi Green   # 64^2 contrast study
python plot_battery.py
# series 2, Tu et al. length scales (64^2, 128^2), figures with suffix _tu, video
./run_battery_tu.sh Green & ./run_battery_tu.sh Green_Jacobi & ./run_battery_tu.sh dual &      # dual 128^2 for the video: GRIDS=128 SNAP_EVERY=2
SUFFIX=_tu L_CELL_UM=6 GRIDS="64 128" python plot_battery.py
python make_battery_video.py 128 dual
```

Each run uses `systemd-run --user --scope -p MemoryMax=4G`, `OMP_NUM_THREADS=1`, `JAX_PLATFORMS=cpu`. The main
environment variables are `NPIX`, `MATRIX_STIFFNESS`, `LENGTH_SCALE`, `DT` (0.005 = 22 s), `T_MAX`, `C_RATE`,
`H_BAR`, `R_CORE_UM`, `A_BAR`, `GC_INTERFACE`, `K_DIFF`, `D_MATRIX`, `MAX_STAGGERED`, `DIFF_PRECOND`, `DIFF_SHADOW`.

Output in `exp_data/battery_N{N}_{P}_ms{matrix}`:

* `_steps.csv`: one row per time step, with SOL, a_c, Gamma, mean CG.
* `_iterations.csv`: one row per staggered iteration, with CG counts for u, d, c and the shadow c solves.
* `_snapshots.npz`: c and d every 10 steps.

Logs are in `logs/` and figures in `figures/`.

## Results, series 1 (12 µm cell, l = 0.75 µm, 32^2 - 128^2)

> **Caveat (found later):** these runs switched the whole band from CC to CV at max c = 1 (now `CV_SWITCH=cmax`).
> That caused a **9.5x current surge** at t = 2098 s, which coincides with the big debonding burst described below.
> The physics of series 1 is therefore partly an artefact; the preconditioner comparison (identical trajectories)
> is unaffected. Series 2 below uses Tu et al.'s length scales and the fixed pixel-wise CC -> CV switch.

All runs below are 1C lithiation to t = 1.0 x 4417 s, with dt = 22 s (200 steps). No step hit MAX_STAGGERED; the
largest was 437 staggered iterations, at 128^2. Green, GJ and dual give identical trajectories (the same a_c, SOL and
number of staggered iterations), so the CG counts compare like with like.

### Crack evolution (figures `battery_snapshots_N128.png`, `battery_crack_growth.png`)

The model reproduces the P + DB pattern of Tu et al. (propagation of the surface crack through the shell, then
core-shell debonding):

* **0 - 1800 s:** AT2 has no threshold, so a diffuse low damage band d ~ 0.2-0.4 builds up at the interface and Gamma
  rises smoothly.
* **~1870 s:** the first fully broken segments appear: an interface debond segment and the seed crack running through
  the shell.
* **2000 - 2450 s:** debonding spreads around the interface in a few rapid events (bursts of up to 437 staggered
  iterations per step at 128^2). Tu et al. also describe debonding as rapid.
* **2450 - 3400 s:** slower growth, with one more jump.
* **After that:** saturation.

Growth is therefore **gradual overall but punctuated by jumps**, not uniformly slow. The debond blocks Li transport
into the core (D = D_0 g(d)), and final SOL at t = 4417 s falls from 0.91 (32^2) to 0.89 (128^2). Final broken area
fraction a_c is 0.163 / 0.160 / 0.158 at 32^2 / 64^2 / 128^2: grid-converged in the integral measure, although the
order of the debond events shifts with the grid (l/h = 2 is coarse).

### Elastic CG per solve, matrix stiffness 1e-2 (`battery_cg_vs_grid.png`, `battery_cg_per_step.png`)

"Before" means only the seed crack is present (a_c within 0.002 of its start value). "During" runs from the onset to
95 % of the final a_c growth. Both windows are taken from the dual run and applied to all three runs.

| mean CG / elastic solve | 32^2 before | 64^2 before | 128^2 before | 32^2 during | 64^2 during | 128^2 during |
|---|---|---|---|---|---|---|
| Green        | 64 | 74 | 96 | 211 | 291 | 310 |
| Green-Jacobi | 40 | 49 | 76 | 84 | 166 | 357 |
| **dual**     | **36** | **34** | **33** | **51** | **53** | **55** |

Total elastic CG iterations per run:

| | 32^2 | 64^2 | 128^2 |
|---|---|---|---|
| Green | 272k | 459k | 547k |
| Green-Jacobi | 116k | 314k | 644k |
| dual | 72k | 93k | 103k |

Total elastic solve wall time at 128^2 is 2125 s (Green), 2336 s (GJ) and 737 s (dual). A dual iteration costs about
1.9x a Green iteration, so the dual is still about 3x faster. Timings were taken on a shared, loaded laptop and are
indicative only.

* **Dual:** grid-independent before and during cracking.
* **GJ:** doubles per refinement while cracks are present (84 -> 166 -> 357). After the debonding (steps > 145) it
  sits at ~800 per solve at 128^2.
* **Green:** limited by the contrast and grows mildly with the grid.

### Matrix contrast at 64^2 (`battery_cg_vs_matrix_contrast.png`)

| matrix / core stiffness | 0.1 | 0.01 | 0.001 |
|---|---|---|---|
| Green before / during | 80 / 226 | 74 / 291 | 133 / 206 |
| GJ before / during | 35 / 108 | 49 / 166 | 54 / 196 |
| dual before / during | 34 / 50 | 34 / 53 | 55 / 60 |
| final a_c, SOL | 0.214, 0.921 | 0.160, 0.892 | 0.063, 0.956 |

The soft matrix fills 41 % of the cell. At 1e-3 the dual loses its advantage **before** cracking: 55 iterations,
equal to GJ. This is the predicted "large soft region" effect, here only 1.6x because the contrast is moderate. During
cracking the dual remains 3.3x better than GJ.

The cracking itself depends on the matrix. A stiffer matrix constrains the particle more and gives more cracking, so
the "during" windows differ between the columns.

### Diffusion solve (shadow solves on the dual trajectory, `battery_cg_per_step.png` right panel)

| mean CG / diffusion solve | 32^2 | 64^2 | 128^2 | 64^2, matrix 0.1 | 64^2, matrix 0.001 |
|---|---|---|---|---|---|
| Green | 25 | 36 | 60 | 39 | 30 |
| Green-Jacobi | **23** | **31** | **51** | **31** | **29** |
| dual-like | 68 | 67 | 61 | 66 | 47 |

* **Dual-like:** grid-independent but higher. The diffusion problem's contrast is dominated by the *matrix*
  D_m = 1e-3 over 41 % of the cell, plus the cracks at K_DIFF = 1e-4. That is exactly the regime where the dual loses.
* **Green / GJ:** grow with the grid, because the 1/dt mass term regularises less as h falls.
* **Overall:** GJ is the best choice for the diffusion solve here. The dual-like preconditioner would only pay off on
  grids finer than 128^2.

The damage solve (GJ) takes 14 / 18 / 24 iterations per solve at 32^2 / 64^2 / 128^2.

### Problems / caveats

* **Length scale:** l = 0.75 µm (3x the paper) with a 1.2 µm shell, so the crack band is as wide as the shell. The
  absolute crack pattern is resolution-limited; the 32^2 run (l/h = 2) is coarse.
* **Unstable debonding:** the debond events need hundreds of staggered iterations per step (max 437 < 500).
  MAX_STAGGERED = 500 was never reached, but a 256^2 run or a faster C-rate may hit it.
* **2D and periodicity:** a plane-strain disk, not a sphere. The neighbouring particle images sit only 1.6 µm apart
  through the matrix.
* **Li leakage:** Li leaks slightly into the matrix (D_m = 1e-3). It is not counted in SOL.

## Results, series 2: Tu et al.'s length scales (`run_battery_tu.sh`, suffix `_tu`)

### Setup and deviations from series 1

* **Length scales:** Tu et al.'s **l = 0.23 µm (core) / 0.27 µm (shell)**, per pixel (`L_CORE_UM`, `L_SHELL_UM`;
  the matrix takes 0.27). **l_zeta = 0.1 µm** (`L_ZETA_UM`).
* **Geometry:** shell **h/R = 0.2**, initial crack **a/h = 0.3** (at least one pixel).
* **Particle:** to keep l/h >= 2 with 64^2 / 128^2, the core radius is **R = 2 µm** (a value of the paper's R study)
  in a **6 µm periodic cell**. h = 0.094 / 0.047 µm, so min l/h = 2.45 / 4.9.
* **32^2 is not run:** l/h would be 1.2.
* **Unchanged from series 1:** zero mean strain, matrix stiffness 1e-2, dt = 22 s, 1C, T_MAX = 4969 s (the 10 %
  cut-off is not reached; final current 0.19-0.24 x CC).
* **CC -> CV pixel by pixel:** a band pixel goes from q_cc to KAPPA (1 - c) once its c >= 1 - q_cc / KAPPA. The current
  never exceeds the CC current, and max c = 1.014. The global switches tried first gave 1.2x - 9.5x surges and are no
  longer used.

### Crack evolution (`battery_snapshots_N128_tu.png`, `battery_crack_growth_tu.png`, video)

* **Before ~2900 s (SOL < 0.76):** only the diffuse interface damage band grows; Gamma rises smoothly. The small
  particle has an almost uniform c (core and shell SOL differ by < 0.02), so the driver is the Omega mismatch.
* **2900-4130 s:** cracking happens in a few bursts (max 272 / 389 staggered iterations per step at 64^2 / 128^2):
  * the seed crack runs through the shell;
  * new shell cracks form at the top, bottom and left;
  * debond arcs form along the interface.

The final pattern is **P + I + DB**, as Tu et al. report for R = 2 µm (Table 2). Final SOL is 0.995 / 0.999 and
final a_c is 0.082 / 0.054 at 64^2 / 128^2. The pattern is **not grid-converged** at l/h = 2.45: the 64^2 run debonds
more.

### CG per solve (`battery_cg_vs_grid_tu.png`, `battery_cg_per_step_tu.png`)

| mean CG / solve | 64^2 before | 128^2 before | 64^2 during | 128^2 during |
|---|---|---|---|---|
| elastic Green        | 70 | 76 | 233 | 270 |
| elastic Green-Jacobi | 43 | 64 | 128 | 206 |
| **elastic dual**     | **33** | **30** | **53** | **51** |
| diffusion Green       | 36 | 38 | 47 | 87 |
| diffusion Green-Jacobi | 50 | 77 | **37** | 62 |
| diffusion dual-like   | 37 | **34** | 44 | **38** |
| damage (GJ)           | 13 | 14 | 18 | 23 |

Total elastic CG iterations:

| | 64^2 | 128^2 |
|---|---|---|
| Green | 363k | 378k |
| Green-Jacobi | 202k | 297k |
| dual | 90k | 82k |

Elastic solve wall time at 128^2 is 1623 s (Green), 1328 s (GJ) and 712 s (dual).

With the small particle and fine h, the dual-like diffusion preconditioner is now the best at 128^2 and grid-independent.
The 1/dt mass term no longer dominates, and the matrix region is smaller, at 50 % of the cell.

Video: `figures/battery_video_N128_dual_tu.mp4` (`make_battery_video.py 128 dual`; 128^2 dual run with SNAP_EVERY=2;
c and d on exact pixels with the core and shell outlines, t / SOL / a_c in the title, elastic CG of G / GJ / dual with
a moving marker).

## NEMO mesh-size study (2D 32^2 - 1024^2, 3D 32^3 - 256^3, fixed l)

Settings are those of series 2 (Tu et al.):
* 6 µm cell, R_core = 2 µm, shell 0.4 µm.
* **l = 0.23 / 0.27 µm fixed** on every grid, so l/h (core) = 1.2, 2.5, 4.9, 9.8, 20, 39 for N = 32 ... 1024.
* **N = 32 is under-resolved** (l/h < 2): grey band and open markers in the figure.
* 3D is a core-shell sphere in the periodic cube (`DIM=3`, trilinear hexahedra, a planar seed patch). The soft
  matrix fills 73 % of the cube in 3D, against 50 % of the cell in 2D.
* GPU results agree with the laptop CPU (64^2 smoke test, job 8657568): the same staggered iterations, elastic CG
  within +-2, diffusion within +-4 (round-off).

How to submit (from the laptop):

```bash
rsync -a example_2D_battery_particle_fracture_AT2.py sbatch_battery_gpu.sh nemo:software/muFFTTO/experiments/dual_preconditioner/battery_particle_fracture/
ssh nemo 'cd ~/software/muFFTTO/experiments/dual_preconditioner/battery_particle_fracture && sbatch --job-name=battery_2D_256_dual sbatch_battery_gpu.sh 2 256 dual'
nohup ./nemo_queue_battery.sh > logs/nemo_queue_battery.log 2>&1 &   # the whole ordered list, <= 12 l40s / <= 2 h200 queued
./sync_nemo_battery.sh && python plot_battery_mesh.py                 # CSVs -> exp_data/nemo, logs -> logs/nemo, figure
```

* **Script:** `sbatch_battery_gpu.sh <DIM> <N> <P>`, one GPU (l40s by default, `-p h200 --mem=200G` for 256^3).
* **Outputs:** `exp_data` and `logs` are symlinks into `/work/classic/fr_ml1145-muffto/runs/battery/`.
* **Segmenting:** a checkpoint is written every 30 min of wall time. After 47.5 h the job resubmits itself with
  `RESTART=1` (`SEG` +1), up to `MAXSEG=3` segments.
* **Job list:** `/work/classic/fr_ml1145-muffto/runs/battery/jobs.txt` (copy in `logs/nemo/jobs.txt`).

Job IDs:

| run | jobs |
|---|---|
| 2D 32^2 / 64^2 / 128^2 / 256^2 (Green, Green-Jacobi, dual) | 8657600 - 8657611 |
| 3D 32^3 | 8657671 - 8657673 |
| 3D 64^3 | 8657674, 8657693, 8657694 |
| 2D 512^2 | 8657695 - 8657697 |
| 3D 128^3 (dual, Green-Jacobi, Green) | 8657718 - 8657720 |
| 3D 256^3 on h200 (dual, Green-Jacobi) | 8657721, 8657722 |
| 2D 1024^2 (dual, Green-Jacobi, Green) | 8657745, 8657746, 8657765 |
| 3D 256^3 Green on h200 | submitted by the feeder when an h200 slot frees |

Figure: `figures/battery_mesh_study_nemo.png` (results: see the end of this README / the final report).
