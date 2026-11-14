# Large 3D TO (512³ / 1024³) — FABLE NOTES

Task: `FABLE_BRIEF.md`. Worktree `/home/martin/PycharmProjects/muFFTTO_fable`, branch `fable/large-3d-to` (base 3a56b48),
NEMO clone `~/software/muFFTTO_fable`, NEMO runs/logs `$WS/runs/FABLE/`, job names `FB_*`.

## RESUME HERE (state 2026-11-14 ~11:00)

User is away until the evening of 2026-11-14 and said: work on autonomously, keep the NEMO rules (own single-node jobs
<= 24 h; ask before > 24 h, full-node 8-GPU production, deletions in `$WS`); use several GPUs and muGrid's features;
give short status messages from time to time.

### What exists now (all committed on `fable/large-3d-to`, nothing pushed)
| file | what |
|---|---|
| `muFFTTO/multigrid.py` (+ `test/test_multigrid.py`) | field-based V-cycle (muGrid `linalg`, `GridTransfer` on host, own CuPy transfer kernels on the GPU), MPI with agglomerated serial tail on rank 0, `coarse_element` option |
| `muFFTTO/domain.py` | `ScaledMaterial.apply_to_field` (one contiguous GPU kernel, no temporaries), `Discretization` MPI block follows its own communicator, one duplicate ghost exchange removed from K |
| `muFFTTO/hourglass_control.py` | in-place scaling by the mode stiffness (GPU: contiguous component vectors), force added by `linalg.axpy` |
| `muFFTTO/tensor_operations.py` | `point_view(field)`: contiguous `[components, points]` view of a GPU field buffer |
| `to3d_large.py` | the lean large-run TO driver (one shared set of fields; `WARM_STORE`, `SYMMETRY=cubic`, `MG_COARSE_ELEMENT`, `CHECKPOINT_EVERY`, `MEM_REPORT`, `PROFILE`) |
| `cg_energy.py` | the library energy-stop PCG on muGrid linalg kernels |
| `cubic_symmetry.py` | cubic projection, serial and slab-distributed (`SlabRemap`) |
| `fable_run_TO.sh`, `fable_run_TO_mpi.sh`, `fable_job.sh`, `fable_sync_nemo.sh`, `fable_env.sh` | run one TO (laptop / NEMO 1 GPU), several GPUs (srun, one rank per GPU), generic sbatch wrapper, code sync, environment |
| `fable_check_to3d_large.py`, `fable_check_symmetry.py`, `fable_test_multigrid.py`, `fable_compare.py` | validation: objective + gradient vs the reference script, symmetry checks, multigrid vs `mg_lib` / MPI, history comparison |
| `fable_memory.py`, `fable_profile_TO.py`, `fable_time_K.py` | memory table, profile of the old script, timing of the pieces of K |
| `fable_reference/` | baseline histories (32^3, 64^3) of the existing code |

### Validation status
* Unit tests: baseline 192 passed; after the library changes `test_hourglass_control`, `test_scaled_material`,
  `test_multigrid` pass (rerun the full list of the brief before the final report).
* Multigrid == `mg_lib` to round-off (serial CPU + GPU); CPU MPI 1/2/4/8 ranks: same CG iterations, same energy.
* `to3d_large` objective and gradient == `example_3D_elasticity_TO.py` to 4e-16 / 3e-16 (8^3, 1Q_hg and 8-point,
  energy and stress); FD check 1e-8.
* 32^3 TO history vs baseline: max rel diff 2e-8 over 21 evaluations (4e-10 at the first), CG 3655 vs 3657.
  All `WARM_STORE` modes (device, device32, host, host32): the same iterations and history.
* Full TO under MPI on CPU, 1/2/4 ranks: identical CG counts, f equal to <= 3e-11. NEMO 2 x L40S: f equal to the
  serial run to all 12 printed digits (CUDA-aware MPI and the distributed GPU FFT of the Helmholtz step work).
* Symmetry: 2-load-case f and projected gradient == 6-load-case ones to 1e-16 (serial, 2 and 4 ranks), FD 1e-10.

### Measurements (reference setup, 1Q_hg, MG, energy objective)
| grid, GPU | code | peak MiB / process | B/voxel all-in | s / evaluation |
|---|---|---|---|---|
| 32^3 laptop | baseline | 212 | | 13.95 |
| 32^3 laptop | lean, 6 load cases | 152 | | 3.56 |
| 32^3 laptop | lean, cubic (2 load cases) | 144 | | 1.1 |
| 128^3 H200 | baseline | 5824 | 2777 | ~15 |
| 128^3 H200 | lean, warm starts on device | 3134 | 1494 | ~10 (job leanB) |
| 256^3 H200 | lean, warm starts on host | 15154 | 947 | ~70 (first version ~120) |
| 256^3 H200 | lean, cubic, host | 15344 | 959 | ~28 (noisy, shared node) |
| 256^3 L40S x 1 / 2 / 4 | lean, host | 15013 / 8607 / 4877 per GPU | | 185 / 110 / 87 |
(256^3 numbers are before the GPU transfer kernels.) 256^3 fields: 825 B/voxel (fine 701, coarse levels 124), CuPy pool
72-84, context + FFT ~50.

### Open issues being worked on
1. **MPI strong scaling is weak** (1.7x on 2, 2.1x on 4 L40S at 256^3). Profiling jobs FB_prof_1 / FB_prof_4
   (8815571 / 8815572, 128^3, `PROFILE=1`) -> `logs/fable/prof_N128_np{1,4}.log` on NEMO. Suspects: ghost exchanges
   (6-7 per K, 3 needed) and the small levels under MPI; idea: agglomerate levels below ~32^3 on rank 0's GPU.
2. **Symmetric start**: the symmetrised CH start is nearly grey and the 32^3 symmetric run stalls at f 6.15, nu +0.06
   (6 load cases: f 2.94, nu -0.16 after 400 iterations, still improving). Laptop unit `fable-wp3s`
   (`fable_wp3_starts.sh`) tries cos / product / symmetrised-converged starts -> `logs/fable/wp3_starts_report.txt`.
3. Then: 512^3 on one H200 (few evaluations), 1/2/4(/8) H200 scaling, sbatch chain 256 -> 512 -> 1024 (WP5), final
   report. H200 partition is busy (jobs wait).

### NEMO jobs of this study (all short, single node)
8814587 features, 8814588 profile 128, 8814660 smoke, 8814666-69 base/lean 128 + 256, 8814703 time K,
8814723 MPI smoke (2 L40S), 8814724-26 re-measure, 8814731-34 strong scaling L40S, 8814795 GridTransfer check,
8815571-72 profiles. Logs `$WS/runs/FABLE/logs/`, run logs `~/software/muFFTTO_fable/experiments/Multigrid_preconditioning/logs/fable/`,
data `$WS/runs/FABLE/data/`.

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
