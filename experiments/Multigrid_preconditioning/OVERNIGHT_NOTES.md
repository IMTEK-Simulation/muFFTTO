# Overnight 2026-11-13 → 14: what was done, what ran, where the results are

Read this first in the morning. Tables are filled in as the jobs finish (sections marked *pending* are not done yet).

## Done in the evening

1. **NuMPI** — commit `6561d66` "ENH: Preconditioned initial inverse Hessian in l_bfgs_bounded (precond)" on
   `feat/tr_precond` (local, **not pushed**). Installed on NEMO: `~/software/NuMPI` fast-forwarded to 6561d66 (git
   bundle; editable for `.venv`, `.venv-gpu-h200`, `-rtx`, `-mi300a`); `.venv-gpu` (l40s, plain copy) got the two files,
   backups `$WS/runs/tmp_scripts/*.bak_20261113`. Default behaviour unchanged (`precond=None`).
   muFFTTO `solvers_nonlinear.minimize(..., precond=)` pass-through + tests: in the working tree, **not committed**.
2. **3D multigrid** in `mg_lib.py` (experiment): trilinear hexahedra, trilinear transfers (R = P^T), coarse grids by
   **option (a) re-discretisation** (coarse Gauss point = mean material of the fine voxel in its octant). Option (b),
   exact Galerkin 27-point stencils, is described in the `mg_lib.py` docstring as the fallback. Checks
   (`test_mg_lib_3d.py`): transfers adjoint (6e-16), homogeneous re-discretisation = Galerkin (4e-16), V-cycle SPD;
   smooth 3D phase field: MG 12 → 11 CG from 16³ to 32³ vs Green-Jacobi 19 → 28, Green 17 → 39.
3. **New random Cahn-Hilliard structures** (`make_new_CH_geometries.py`, new seeds, figures `CH_starts_new_2D.png`,
   `CH3D_new.png`): 2D TO starts `bi_s11`, `bi_s12` (bicontinuous), `drop_s11` (soft droplets, 69 % stiff);
   3D `bi_s11` (50 %), `ch_s12`, `ch_s13` (57 % stiff).

## Submitted to NEMO (`submit_overnight.sh`, job list `$WS/runs/MG_overnight_jobs.txt`)

Smoke jobs passed (3D driver on l40s: identical counts to the laptop; TO + MG on h200: identical objective sequence).

* **3D scaling** (`scaling_case_3d.py`, `sbatch_mg_scaling_3d.sh`): MG vs Green-Jacobi / Green / dual, contrast 1e-4,
  E_xx, energy stop d_min 0. Resolved interfaces only (32³: η/h ≤ 1, 64³: ≤ 2, 128³: ≤ 4 — wider bands wash out the
  CH features). `ch_s12`: η/h 0, 0.5, 1 (32³), +2 (64³), +4 (128³, h200); `bi_s11`, `ch_s13`: η/h 0, 1 on all grids.
  → `./sync_nemo_scaling3d.sh && python plot_scaling_3d.py` → `figures/nemo_scaling_3d.png`. **Done (21:10)**.
* **2D TO chains** (η = 4h hierarchy, MG + Helmholtz L-BFGS c = 4, each grid INIT_FROM the previous final design):
  256² → 512² (l40s) → 1024² → 2048² (h200) for `bi_s11`, `bi_s12`, `drop_s11`; `bi_s11` also 4096² (h200, 24 h).
  Baseline at 256²: production Green-Jacobi + NuMPI L-BFGS from the same starts. → `./sync_nemo_TO.sh`. *pending*
* **3D TO with MG** (`example_3D_elasticity_TO.py` copy with 'MG' and `DESIGN_OPT=lbfgs_helm`, `submit_TO3D.sh`),
  η = 4h, start = new 3D CH structure `bi_s11` blurred (`data/CH3D/CH3D_start_bi_s11.nc`):
  MG + Helmholtz c = 4: 64³ (l40s) → 128³ (h200); production Green-Jacobi + NuMPI L-BFGS: 64³ → 128³ (h200);
  MG + NuMPI L-BFGS: 64³. Laptop 16³ check: MG ~15 CG/solve vs GJ ~38. Logs `$WS/runs/MG_TO/TO3D_*`. *pending*

## Results

### 3D scaling (all 9 jobs done in 0.5-13 min; 128³ on H200)

Multigrid with option (a) re-discretised coarse grids works in 3D: **12-16 CG iterations on every grid, interface
width and geometry** -> option (b) (Galerkin stencils) is not needed, no parameter change needed.

GPU solve time at 128³ (H200), iterations in brackets:

| geometry, eta/h | Green | Green-Jacobi | dual | **MG** | MG vs best FFT |
|---|---|---|---|---|---|
| ch_s12, 0 (sharp) | 18.9 s (185) | 63.5 s (630) | 18.8 s (94) | **8.5 s (14)** | 2.2x |
| ch_s12, 0.5 | 51.6 s (512) | 49.2 s (487) | 51.5 s (259) | **7.7 s (13)** | 6.4x |
| ch_s12, 1 | 54.3 s (538) | 32.8 s (322) | 46.4 s (233) | **7.7 s (13)** | 4.3x |
| ch_s12, 2 | 55.8 s (554) | 19.5 s (190) | 33.1 s (166) | **7.7 s (13)** | 2.5x |
| ch_s12, 4 | 33.2 s (328) | 9.4 s (89) | 8.7 s (43) | **7.1 s (12)** | 1.2x |
| bi_s11, 0 / 1 | 18.9 / 54.5 s | 79.6 / 42.5 s | 18.6 / 45.3 s | **9.0 / 8.8 s** | 2.1x / 4.8x |
| ch_s13, 0 / 1 | 18.1 / 54.9 s | 62.7 / 33.0 s | 18.0 / 46.3 s | **7.7 / 7.7 s** | 2.3x / 4.3x |

* Unlike 2D, MG wins **also on the sharp interface** in 3D (2.1-2.3x vs dual): Green / dual need ~90-200 iterations
  on the sharp 3D Cahn-Hilliard structures (2D: 24-46).
* Green-Jacobi doubles per refinement at fixed eta/h (sharp: 171 -> 325 -> 630); Green and dual saturate
  (Green ~185 sharp / ~540 diffuse; dual ~95 sharp / ~230 diffuse); MG flat.
* Fixed physical eta = 1/32 (eta/h = 1, 2, 4 at 32³, 64³, 128³): everything mesh-independent (MG 12, dual 34-43,
  GJ 65-89, Green 213-328); MG still the cheapest at 128³ (7.1 s vs dual 8.7 s).
* GPU time per iteration at 128³: MG ~0.6 s, Green ~0.1 s (the ~6x operator-count ratio).


### 2D TO chains from the new CH starts (status 05:30; MG + Helmholtz c = 4, eta = 4h, INIT_FROM coarser grid)

| start | 256² | 512² | 1024² (H200) | 2048² (H200) | 4096² (H200) |
|---|---|---|---|---|---|
| bi_s11 | 704 ev, 0.27 h, f 0.8031 | 147 ev, 0.08 h, 0.7457 | 386 ev, 0.36 h, 0.7227 | 596 ev, 1.45 h, 0.7127 | running: 752 ev, 6.4 h, 30.6 s/ev, f 0.7075 |
| bi_s12 | 288, 0.12 h, 0.8041 | 146, 0.08 h, 0.7455 | 282, 0.28 h, 0.7227 | 730, 1.82 h, 0.7127 | - |
| drop_s11 | 258, 0.10 h, 0.9866 | 423, 0.23 h, 0.8952 | 560, 0.55 h, 0.8505 | 900, 2.23 h, 0.8259 | - |

MG: 13-15 CG per solve on every grid up to 4096². Time per evaluation (H200): 3.4-3.6 s (1024²), 8.8-9.0 s (2048²),
30.6 s (4096²). Whole hierarchy 256² -> 2048²: 2.2-3.1 h per start. Production baseline at 256² (GJ + NuMPI L-BFGS):
138-161 CG/solve, 0.30-0.34 h (MG + Helmholtz: 0.10-0.27 h), same objectives (within 0.3 %).

### Helmholtz c = 1 vs 4 on NEMO (bi_s0, MG mechanics): no clear winner

Evaluations c = 1 / c = 4: eta 2h: 141/229 (64²), 300/266 (128²), 402/476 (256²), 1106/1080 (512²); eta 4h: 122/99,
222/232, 322/418, 717/702. Both 1.5-2.7x fewer than plain L-BFGS; c in [1, 4] is fine.

**Time to a converged design at 512² (L40S):** production GJ + NuMPI L-BFGS 3.02 h (eta 2h) / 1.11 h (4h) vs
MG + Helmholtz 0.62 h / 0.39 h -> **4.9x / 2.8x faster**, same designs.

### eta = 1h, Green vs MG (bi_s0): Green 428-439 CG/solve, MG 13-15; time 0.47 / 0.77 / 1.77 / 4.10 h (Green) vs
0.11 / 0.20 / 0.66 / 1.20 h (MG) for 64² ... 512² -> 3.4-4.3x, same objectives.

### 3D TO (bi3d_s11, eta 4h) -- MG is NOT faster here yet, and why

| | evaluations | CG/solve | s/evaluation | time | f |
|---|---|---|---|---|---|
| 64³ MG + Helmholtz (L40S) | 1173 | 10.1 | 16.2 | 5.3 h | 2.2299 |
| 64³ GJ + NuMPI L-BFGS (L40S) | 2007 | 69.4 | 11.9 | 6.6 h | 2.2252 |
| 64³ MG + NuMPI L-BFGS (L40S) | 1927+ (running) | 9.4 | 15.1 | 8.1 h+ | 2.2253 |
| 128³ MG + Helmholtz (H200) | 154+ (running) | 10.0 | 64.1 | 2.7 h+ | 1.7729 |
| 128³ GJ + NuMPI L-BFGS (H200) | 122+ (running) | 115.9 | 41.3 | 1.4 h+ | 1.7709 |

MG needs 7-12x fewer CG iterations, but one evaluation costs MORE than with GJ. Cause: the TO script applies K with
the cheap ScaledMaterial (a scalar s(x) times C_0), while mg_lib's V-cycle uses full C_ijkl material fields on every
level (3D: 81 components x 8 Gauss points per voxel) -> each smoothing K is several times slower than the CG's K.
In the static 3D study all methods used the full material, so the comparison there was fair. **Fix for the library
version: ScaledMaterial on all MG levels** (expected to make MG the faster one in 3D TO as well). Helmholtz still
cuts the evaluations 1.7x at 64³ (1173 vs 2007).

## Morning 2026-11-14

* **ScaledMaterial hierarchy** (user request): mg_lib's scalar path now stores every MG level as a muFFTTO
  ScaledMaterial (s(x) per Gauss point instead of the full C_ijkl: 16x less material memory in 2D, 81x in 3D; 128³:
  ~11 GB -> 134 MB). Identical results (K, Jacobi, dual, V-cycle to 1e-16; `test_scaled_material.py`), K 2x faster on
  the CPU. NEMO: 3D TO re-runs tagged `_scaled` (64³ l40s, 128³ h200 from the finished 64³ design), smoke passed
  with the identical objective sequence.
* **How much Helmholtz helped** (`plot_helmholtz_summary.py` -> `figures/helmholtz_summary_nemo.png`), time to the
  converged design, L40S: 512² eta 2h: production GJ + NuMPI L-BFGS 181 min, MG + plain 73 min, MG + Helmholtz 37 min
  (4.9x); eta 4h: 67 / 35 / 23 min (2.9x); 256²: 30 / 19 / 10 min (2h), 19 / 16 / 7 min (4h).
* **Designs of the chains** (`plot_chain_designs.py`): `figures/TO_chains_new_CH.png` (overview) and exact-pixel
  `figures/TO_chain_<start>_N2048.png`.
* **3D TO with the ScaledMaterial hierarchy** (first ~12-33 evaluations): 128³ H200 27.2 s/eval (old full-C_ijkl
  hierarchy 60.6 s, production GJ 33.6 s) -> MG now 1.24x faster per evaluation than GJ, and Helmholtz needs ~1.7x
  fewer evaluations -> expected ~2x faster to the design. 64³ L40S: 10.0 s/eval (old 14.6, GJ 9.4). The old 128³ MG run
  (8810499) was cancelled as superseded (it would have needed ~10 more H200-hours).

## Settings prototype (2026-11-14, `run_proto.sh`, `plot_proto.py --soft-exp 3` -> `figures/proto_N128_soft3.png`)

2D TO 128², contrast 1e3 (user), CH start bi_s11, MG (ScaledMaterial) + Helmholtz L-BFGS. Evaluations / laptop s:

| | c 1 | c 4 | c 16 |
|---|---|---|---|
| eta 2h V(2,2) / V(1,1) | 358, 1010 / 302, 778 | 414, 1149 / 472, 1195 | 420, 1168 / 434, 1127 |
| eta 4h V(2,2) / V(1,1) | 253, 683 / 257, 639 | 282, 764 / 303, 756 | 473, 1266 / 449, 1115 |

Same design within each eta (2h: f 0.854, grey 0.20; 4h: f 1.000, grey 0.34). c = 1 cheapest, c = 4 close, c = 16
worse at 4h. V(1,1): 18-19 vs 13 CG/solve, ~8-10 % cheaper per evaluation. eta 2h ~1.4x more evaluations than 4h.

Existing designs (target nu = -0.5 -> nu12 = -1/3, mu* = 0.193): bi_s11 nu12 -0.135 / -0.195 / -0.228 at 256 / 512 /
1024² (mu 0.203, 83-85 % solid) -> shear target met, Poisson not; auxetic prototype with lower TARGET_SCALE running.
