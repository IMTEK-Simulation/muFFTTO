# Large 3D TO (512³ / 1024³) — FABLE NOTES

Task: `FABLE_BRIEF.md`. Worktree `/home/martin/PycharmProjects/muFFTTO_fable`, branch `fable/large-3d-to` (base 3a56b48),
NEMO clone `~/software/muFFTTO_fable`, NEMO runs/logs `$WS/runs/FABLE/`, job names `FB_*`.

## RESUME HERE (state 2026-11-14 ~09:30)

User is away until the evening of 2026-11-14 and said: work on autonomously, keep the NEMO rules (own single-node jobs
<= 24 h; ask before > 24 h, full-node 8-GPU production, deletions in `$WS`). Also: use several GPUs and muGrid's features.

Done
* WP0: unit tests 192 passed (`logs/fable/unit_tests_baseline.log`); baseline histories `logs/fable/baseline_N32.log`
  (21 evaluations) and `baseline_N64.log` (11 evaluations) = the reference for every later change.
* WP1: memory tables (`fable_memory.py`, `fable_profile_TO.py`): `logs/fable/profile_N{32,64}.log`, 128³ on NEMO
  (`~/software/muFFTTO_fable/.../logs/fable/profile_N128.log`, job 8814588). Table below.
* `muFFTTO/multigrid.py` (new): field-based V-cycle on muGrid `GridTransfer` + `linalg`, MPI-capable with
  agglomerated serial tail on rank 0. Serial: equal to `mg_lib` to round-off (`fable_test_multigrid.py`, rel diff 2.8e-16).
  `domain.py`: the MPI block of `Discretization.__init__` now follows the discretization's communicator (a serial
  discretization inside a parallel run deadlocked in a COMM_WORLD reduction).

In progress / next
1. MPI CPU tests of the multigrid with 2/4/8 ranks (`mpirun -n 4 python fable_test_multigrid.py 32`): same iterations
   as serial.
2. Lean CG (local copy of the library algorithm on muGrid linalg) + lean large-run TO driver that shares all scratch
   fields; validate against the baseline histories.
3. Warm starts on host / float32, cubic symmetry (WP3), NEMO measurements 256³, 512³, multi-GPU (WP4), sbatch chain (WP5).

Running jobs: none on NEMO. Laptop: none of mine (`systemctl --user list-units 'fable-*'`).

## Decisions
* **Order changed: the multigrid rewrite comes first** (it is both the largest memory item of WP2 and the core of WP4):
  one field-based implementation that shares the TO's scratch fields and runs under MPI, instead of dieting the serial
  `mg_lib` and rewriting it again for MPI.
* muGrid features used: `GridTransfer` (nested-slab P / P^T), `linalg` (fused axpy / axpby / vecdot, no temporaries).
  NEMO muGrid 1.6.0-15 has both; NEMO Open MPI reports `opal_built_with_cuda_support: true` (job 8814587).
  `IsotropicStiffnessOperator3D` (fused K) is a 5-tetrahedra element, not our hexahedron -> not usable (results must
  not change).
* Reference setup of all runs here (`fable_run_TO.sh`): 1Q_hg eps 0.3, MG, 6 load cases, `OBJECTIVE=energy`,
  `TARGET_NU=-0.5`, `TARGET_SCALE=0.3`, eta 2h, contrast 1e3, `lbfgs_helm` c 4, start `CH3D_start_bi_s11.nc`.

## WP1 — where the memory goes (existing code, 1Q_hg, MG, 6 load cases)

Peak per process: 212 / 876 MiB (32³ / 64³, laptop), 5452 MiB (128³, L40S) = 2.7 KiB/voxel at 128³.

| owner (128³) | B/voxel | content |
|---|---|---|
| TO discretization, muGrid fields | 1280 | u, λ of 6 load cases 306; gradient-sized scratch x2 153; constant `macro_gradient_field` 77; hourglass strain + force 128; Green operator + impulse fields (unused with MG) 149; `dual_temp_1/2` (unused) 51; CG 102; rhs + adjoint rhs + energy_stop_Kx + 4 single-use nodal scratch 180; 14 scalar fields 119 |
| MG level 0 (own Discretization) | 392 | duplicate gradient + hourglass scratch 204; Jacobi comb fields 51; diagonal 26; 4 level vectors 102; scale 9 |
| MG levels 1.. | 133 | 8-point gradient scratch of level 1 alone 81 |
| CuPy pool, live | 87 | `_dhalf`, `_dinv`, `_power_start` per level; Helmholtz symbol |
| CuPy pool, reserved but free | 565 | high-water mark of temporaries: `.copy()` per K / M call in the smoother, `ravel()` copies in every CG dot product, `alpha * p` etc. in CG, hourglass `kappa` arrays (96 B) and `q_u.copy()`, ScaledMaterial `g @ C` (2 x 72 B) |
| rest | 270 | CUDA context (~300 MiB fixed) + cuFFT plans / work areas (grow with N: ~120 B/voxel) |
| **total** | **2726** | |

Essential state (float64): u/λ of the active load case 48, CG 4 vectors 96, rhs 24, MG fine 5 vectors 120, one
gradient scratch 72, hourglass strain 96, design fields ~50 -> ~500 B/voxel + stored warm starts.
