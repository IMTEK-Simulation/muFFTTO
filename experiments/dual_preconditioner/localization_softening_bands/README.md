# Strain localisation in softening bands (implicit-gradient damage), 2D

Prototype study: FFT preconditioners Green, Green-Jacobi (GJ) and dual for the elasticity solve when the tangent
stiffness collapses gradually inside a band of width ~ l in a heterogeneous microstructure.

## Model choice

Candidate (a), **implicit-gradient enhanced damage** (Peerlings, de Borst, Brekelmans & de Vree 1996) with exponential
softening, was chosen because it fits the existing AT2 code best:

* the nonlocal equation is a scalar, linear, SPD screened-Poisson (Helmholtz) problem on the same 'conductivity'
  discretization that the AT2 script uses for the damage equation (diffusion `B^T W B` + mass `N^T W N`);
* its coefficients are constant, so the FFT Green operator of the homogeneous operator (zero mode inverted) is its
  **exact** inverse: the nonlocal solve is one FFT solve per staggered iteration (relative residual ~1e-14 checked at
  start-up), and the elastic solve is the only iterative solve -- the cleanest test of the three preconditioners;
* the elastic problem with the secant stiffness `[(1-D) + k] C_0(x)` is linear-symmetric per staggered iteration, so
  the same PCG / ScaledMaterial / dual machinery is reused unchanged;
* softening is gradual by construction (no brittle snap): the local stress after the peak is
  `E kappa_0 exp(-beta (kappa - kappa_0))`; with `beta kappa_0 = 0.1` and a band of a few l, the macroscopic response has
  no snap-back. PF-CZM (b) would need a nonlinear, bound-constrained phase-field equation; (c) is just a load case of (a)
  (`LOAD_DIR=xy`).

## Equations

```
sigma    = s C_iso : eps,   s = stiffness(x) [(1 - D) + k]          stiffness = 1 matrix, 10 inclusions
eps      = E + sym(grad u~)                                         periodic fluctuation u~, macro strain E
eps_eq   = sqrt(2 psi^+(eps) / E_young)                             energy equivalent strain (Amor split:
           psi^+ = K/2 <tr eps>_+^2 + mu eps_dev:eps_dev)            tension + shear drive damage)
ebar - l^2 lap ebar = eps_eq                                        nonlocal equivalent strain
kappa    = max(kappa_n, ebar, kappa_0)                              history (irreversibility)
D        = 1 - kappa_0/kappa [1 - alpha + alpha exp(-beta (kappa - kappa_0))]   in the matrix, D = 0 in inclusions
```

Staggered scheme per load step: elastic PCG (warm start, energy-error stop rtol 1e-6, delay 10, as in the AT2 script)
-> eps_eq -> Helmholtz FFT solve -> kappa, D, s; repeat until `max|D - D_old| < STAGGERED_TOL`; after `MAX_STAGGERED`
iterations the step is **accepted** and logged `ACCEPTED NOT CONVERGED`.

## Parameters (defaults)

| env | default | meaning |
|---|---|---|
| NPIX | 64 | grid N x N (unit cell) |
| MICROSTRUCTURE | polydisperse_pores | 25 % disks (radii 0.05-0.14), used as stiff tough inclusions |
| GEOMETRY_FILE | - | optional .npy phase image instead (e.g. the Cahn-Hilliard image of schneider2025_notch) |
| INCLUSION_STIFFNESS | 10 | inclusion / matrix stiffness |
| LENGTH_SCALE | 1/16 | l, fixed across grids (l/h = 2, 4, 8 at 32², 64², 128²) |
| KAPPA0, ALPHA, BETA | 1e-3, 1, 100 | damage threshold and exponential softening |
| K_RESIDUAL | 1e-3 | residual stiffness k (band stiffness floor; contrast of the band) |
| LOAD_DIR | xx | `xx`/`yy` uniaxial strain, `xy` simple shear (load = gamma) |
| LOAD_MAX, LOAD_STEPS | 0.012, 120 | load path |
| STAGGERED_TOL, MAX_STAGGERED | 1e-3, 500 | staggered loop |
| STOP_FRACTION, STOP_STEPS | 0.05, 3 | stop when stress < 5 % of the peak for 3 steps |

E = 1, nu = 0.2 (muFFTTO 2D Lame quirk, see the AT2 study); argv[1] = `Green`, `Green_Jacobi` or `dual`.

## How to run

```
./run_GD_series.sh xx 1e-3            # tension, k = 1e-3: 32², 64², 128² x 3 preconditioners, 3 processes at once
./run_GD_series.sh xx 1e-4
./run_GD_series.sh xy 1e-3 0.02 120   # simple shear
python plot_GD_results.py xx 0.001    # figures/ + table on stdout
```

Outputs: `logs/GD_N<N>_<P>_<dir>_l0.0625_b100_k<k>.log`, `exp_data/..._iterations.csv` (every staggered iteration:
CG its, times, max|dD|, max D, min s), `..._steps.csv` (every load step: stress, max D, min s, band fractions,
staggered its, convergence, CG), `..._fields.npz` (pixel-mean D of every load step).

## Results (2026-11-06, CPU, 3 serial runs at a time: wall times only relative)

Runs: tension `xx` k = 1e-3 (to 0.012, 120 steps), tension `xx` k = 1e-4 (to 0.02, 200 steps, `STOP_FRACTION=0`;
an earlier k = 1e-4 series stopped at 5 % of the peak with min s = 4e-4 is in `exp_data/k1e-4_stopped/`), shear `xy`
k = 1e-3 (gamma to 0.02, 120 steps). Every staggered loop converged (max 28 iterations per step, cap 500 never hit).

**Localisation is gradual.** Diffuse damage between the inclusions before the peak, one band (normal to the load for
tension, a vertical band for shear) selected right after the peak, then the band stiffness drops smoothly over ~100
load steps (1 -> 0.1 by 2e-3, 0.01 by 4e-3, 1e-3 by 10e-3 for k = 1e-3); stress-strain curves without snap-back,
identical for the three preconditioners and nearly identical on the three grids (fixed l). Caveat: the band of the
implicit-gradient model **widens** (classic spurious damage growth): the area with s < 0.1 grows to ~40 % of the matrix;
it does not travel through the microstructure.

Mean elastic CG iterations per solve, binned by the current minimum band stiffness s (`plot_GD_results.py`, tension):

| k | P | 32² | 64² | 128² | (bins s >= 0.1 / 0.1-0.01 / 0.01-1e-3 / < 1e-3) |
|---|---|---|---|---|---|
| 1e-3 | Green | 30/68/209 | 30/68/212 | 29/67/212 | flat in N, ~sqrt(contrast) |
| 1e-3 | GJ | 55/81/92 | 83/128/140 | 142/224/243 | ~1.7x per refinement |
| 1e-3 | dual | 19/35/80 | 19/34/81 | 19/34/81 | flat in N, grows with contrast |
| 1e-4 | Green | 30/68/173/624 | 30/69/179/664 | 29/68/182/680 | |
| 1e-4 | GJ | 55/81/87/80 | 83/128/133/111 | 142/223/237/180 | GJ drops once the band is uniform |
| 1e-4 | dual | 19/35/64/184 | 19/34/64/211 | 19/34/64/217 | |

Whole path (tension): k = 1e-3 total CG Green 92k/100k/102k, GJ 57k/91k/162k, dual 39k/41k/42k (32²/64²/128²);
128² elastic time Green 205 s, GJ 395 s, dual 166 s. k = 1e-4: Green 395k/455k/475k, GJ 83k/129k/221k,
dual 123k/150k/157k; 128² time Green 2080 s, GJ 885 s, dual 1303 s. Shear k = 1e-3: dual 41k/44k/43k,
Green 93k/100k/99k, GJ 95k/159k/263k; 128² time dual 441, Green 517, GJ 1100 s.

## Model alternative: PF-CZM (Wu 2017, `Wu2017_unified_phase_field_JMPS.pdf`)

Not implemented. alpha(d) = 2d - d^2, g(d) = (1-d)^p / ((1-d)^p + a1 d (1 + a2 d + a2 a3 d^2)), p = 2,
a1 = 4 l_ch / (pi l); linear softening a2 = -0.5, a3 = 0; exponential a2 = 2^(5/3) - 3 = 0.1748, a3 = 0; Cornelissen
a2 = 1.3868, a3 = 0.6567. Its damage band keeps a fixed width ~ pi l and propagates (no widening), so it would give the
narrow, travelling band; it needs a nonlinear d-equation (rational g, linear alpha term -> bounds 0 <= d <= 1),
i.e. a Newton/projected solve per staggered iteration instead of the linear AT2 one.

---

# Part 2: PF-CZM (Wu 2017) in a concrete meso-cell

User verdict on part 1: the gradient-damage band widens instead of moving. Part 2 uses Wu's phase-field regularised
cohesive zone model (`Wu2017_unified_phase_field_JMPS.pdf`): alpha(d) = 2d - d^2, c0 = pi,
omega(d) = (1-d)^p / ((1-d)^p + a1 d (1 + a2 d + a2 a3 d^2)), a1 = 4 l_ch/(pi b); softening laws
linear (p 2, a2 -0.5, a3 0), exponential (p 2.5, a2 2^(5/3)-3, a3 0), Cornelissen (p 2, a2 1.3868, a3 0.6567, default).
Driving force Y = sigma_eq^2/(2E) with Wu's modified equivalent effective stress (eq. 2.28, fc/ft = 10), history
H = max(Y0, H_n, Y), Y0 = ft^2/(2E). Damage per staggered iteration: projected (active-set) Newton on d_n <= d <= 1,
local Hessian omega'' H - 2Gf/(pi b) clipped from below (convexified), reduced Newton systems by PCG with the
Green-Jacobi preconditioner of the homogeneous damage operator, energy backtracking. Elastic solve unchanged
(secant s = stiffness [omega(d) + k], Green / Green-Jacobi / dual, energy-error stop).

Scripts: `example_2D_pfczm_concrete.py` (first local version, b = 5 mm, E0 = 30 GPa), `pfczm_concrete.py`
(mesh-study version: 2D/3D, CPU/GPU, generated aggregates, checkpoint/restart), `run_CZM_series.sh`,
`plot_CZM_results.py`, `make_CZM_video.py`, `sbatch_czm_gpu.sh`, `sync_nemo.sh`, `plot_pfczm_mesh.py`.

## Local runs (example_2D_pfczm_concrete.py, 2026-11-06)

Wu's concrete E0 = 30 GPa, nu = 0.2, ft = 3 MPa, Gf = 0.12 N/mm, b = 5 mm (Wu's bar / L-panel value); cell 80 mm
(ASR meso geometry scaled, aggregates 6.4-16 mm, 44 %, E 70 GPa, ft and Gf 2x mortar -- assumed values, typical
of mesoscale concrete studies (aggregate E 50-70 GPa); no source checked), k = 1e-3, b/h = 2, 4, 8 at 32², 64², 128².

| tension k = 1e-3 | 32² | 64² | 128² | (mean elastic CG per solve before peak / softening) |
|---|---|---|---|---|
| Green | 22/120 | 22/124 | 21/124 | |
| GJ | 32/41 | 44/60 | 61/80 | |
| dual | 15/32 | 15/32 | 15/32 | grid independent |

Shear (k = 1e-3, gamma to 6e-3): Green 30-43/120-125, GJ 46-131/86-167, dual 18-20/32. All staggered loops converged.
Videos: `figures/CZM_video_N128_xx.mp4`, `figures/CZM_video_N128_xy.mp4`.

**Finding: k must be small in a cohesive model.** The stress stalled at about 1/3 ft (tension) instead of
softening to 0. A homogeneous verification cell (one weak disk, `logs/verify_homog_128*.log`, 128², b/h = 1.9) shows
why: the band strain at full opening is ~ w_c / (pi b) ~ 0.05, so the residual stiffness carries k E 0.05 -- 0.4 ft
for k = 1e-3. With k = 1e-5 the stress drops to 0.1 MPa (4 % ft) at w ~ 0.85 w_c, as Wu's Cornelissen curve. Aggregate
bridging is a smaller part (aggregates as strong as the mortar: 0.81 vs 1.02 MPa at the end). Therefore the mesh study
uses k = 1e-5 (elastic contrast 1e5 in the crack: the regime where the dual's level grows).

## Mesh-size study on NEMO (pfczm_concrete.py, k = 1e-5)

Parameters: Wu Sec. 4.3 (Rots 1988 notched beam) E0 = 20 GPa, nu = 0.2, ft = 2.4 MPa, Gf = 0.113 N/mm, b = 1.5 mm
(Wu's smaller value there), l_ch = 392 mm, Cornelissen softening. Cell 100 mm, periodic, generated aggregates
(random sequential addition, Fuller grading, diameters 8-20 mm = 5-13 b, gap >= 1.5 mm; 2D disks 45 %, 3D spheres 35 %),
E_agg = 70 GPa, ft/Gf 2x mortar. Fixed b: b/h = 0.48, 0.96, 1.92, 3.84, 7.68, 15.4 at 32 ... 1024 (32-128 under-resolve
b, flagged in the plots). Uniaxial strain E_xx to 3e-3 in 300 steps (2D shear to 6e-3 for 32-128²), stop at 2 % of
the peak stress, MAX_STAGGERED 500 (accept).

GPU check (job 8657599): 64² dual identical to the laptop CG counts per step (step 19: 164, step 59: 172; total
78053 GPU vs 78014 CPU); 3D 32³ dual on the L40S 580 s.

NEMO: code in ~/software/muFFTTO/experiments/dual_preconditioner/localization_softening_bands (scp, no git),
outputs /work/classic/fr_ml1145-muffto/runs/czm/{exp_data,logs}, job list runs/czm/jobs.txt.
`sbatch sbatch_czm_gpu.sh DIM:N:P:DIR [...]` (several specs = one bundle job, run one after another; a single-spec job
writes a checkpoint after 47 h and resubmits itself, at most 3 segments). Copy back with `./sync_nemo.sh`,
plot with `python plot_pfczm_mesh.py xx`.

Jobs submitted 2026-11-07 00:32 (l40s): 8658030 2D 32-128² tension bundle (9 runs), 8658032 2D 32-128² shear bundle,
8658033-5 2D 256² (G, GJ, dual), 8658036-8 2D 512², 8658039 3D 32³+64³ bundle (6 runs), 8658040-2 3D 128³.
Pending: 2D 1024² and 3D 256³ (h200) once the smaller grids run fine.

The 2D 32-128² bundles (8658030, 8658032) were cancelled (QOS full) and run on the laptop CPU instead
(`run_pfczm_small_cpu.sh`, same script, outputs in exp_data/; plot_pfczm_mesh.py reads exp_data_nemo/ and exp_data/).

### Results 2D 32-128² (b/h 0.48 / 0.96 / 1.92, all under-resolved), mean elastic CG per solve before peak / cracking

| | 32² | 64² | 128² |
|---|---|---|---|
| tension Green | 24 / 162 | 22 / 316 | 25 / 510 |
| tension GJ | 39 / 77 | 58 / 132 | 99 / 271 |
| tension dual | 16 / 71 | 16 / 101 | 17 / 110 |
| shear Green | 46 / 190 | 49 / 341 | 61 / 554 |
| shear GJ | 69 / 137 | 130 / 417 | 293 / 810 |
| shear dual | 23 / 74 | 24 / 114 | 27 / 138 |

All 18 runs reach the end of the load path, every staggered loop converged. One narrow crack (width ~ pi b) meanders
between the aggregates; the stress softens smoothly to 0.1 MPa (4 % ft); peak 2.9 / 2.65 / 2.55 MPa (32/64/128²),
curves converge from 64² on. Figures: figures/PFCZM_mesh_*.png.

### Results 2D tension 32-512² (fixed b = 1.5 mm), mean elastic CG per solve before peak / during cracking

| b/h | 0.48 (32²) | 0.96 (64²) | 1.92 (128²) | 3.84 (256²) | 7.68 (512²) |
|---|---|---|---|---|---|
| Green | 24 / 162 | 22 / 316 | 25 / 510 | 26 / 662 | 27 / 706 |
| GJ | 39 / 77 | 58 / 132 | 99 / 271 | 179 / 498 | 328 / 836 |
| dual | 16 / 71 | 16 / 101 | 17 / 110 | 17 / 108 | 18 / 100 |

The dual's count is grid-independent once b is resolved (100-110 during cracking, 16-18 before the peak; a dual iteration costs
~2 Green iterations). GJ roughly doubles per refinement, and Green grows until b is resolved (contrast ~1e5 inside the
crack) and then levels off at ~700. 512² GPU wall time (L40S): dual 786 s, GJ 1655 s, Green 1956 s. Peak stress 2.9 / 2.65 /
2.55 / 2.50 / 2.50 MPa (32 ... 512²): converged from 256². One crack, curving between the aggregates, the same
path on every grid from 64².
3D 32³ Green (b/h 0.48): 24 / 216, 683 s.

NEMO state when the ssh ControlMaster dropped (2026-11-07 morning): running or queued: 3D 32³+64³ bundle 8658039,
3D 128³ 8658040-2 (~40-60 s per load step early on), 2D 1024² 8658120 (dual), 8658121 (GJ), 8658213 (Green), 2D 256² shear
bundle 8658214, 3D 256³ h200 8658122 (dual), 8658123 (GJ). Long jobs resubmit themselves (checkpoint, at most 3
segments). To continue: re-open the master (`ssh -fN nemo`), `./sync_nemo.sh`, `python plot_pfczm_mesh.py xx` (and xy).
3D 256³ Green not submitted yet (h200 budget 2).
