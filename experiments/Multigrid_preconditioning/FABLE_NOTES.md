# Large 3D TO (512³ / 1024³) — FABLE NOTES

Task: `FABLE_BRIEF.md`. Worktree `/home/martin/PycharmProjects/muFFTTO_fable`, branch `fable/large-3d-to` (base 3a56b48),
NEMO clone `~/software/muFFTTO_fable`, NEMO runs/logs `$WS/runs/FABLE/`, job names `FB_*`.

## RESUME HERE: read `HANDOFF.md` first (state 2026-11-15 08:40); the text below is the older state

## Older state (2026-11-14 ~20:15)

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

### Measurements (reference setup, 1Q_hg, MG, energy objective; s/evaluation = steady evaluations after the first two)
| grid, GPU | code | peak MiB / process | B/voxel all-in | s / evaluation |
|---|---|---|---|---|
| 32^3 laptop | baseline | 212 | | 13.95 |
| 32^3 laptop | lean, 6 load cases | 140-152 | | 2.2 (first lean version 7.3) |
| 32^3 laptop | lean, cubic (2 load cases) | 144 | | ~1.0 |
| 128^3 H200 | baseline | 5824 | 2777 | ~15 |
| 128^3 H200 | lean, warm starts on device | 3134 | 1494 | ~10 (leanB; before lean K / transfer kernels) |
| 256^3 H200 | baseline (8816167) | 40922 | 2438 | ~120 |
| 256^3 H200 | lean, host warm starts | 15154 | 947 | 62 (leanC; first lean version 120) |
| 256^3 L40S x 1 / 2 / 4 | lean, host (strong2) | 15013 / 8619 / 4891 per GPU | | 164 / 90 / 62 (first version 185 / 110 / 87) |
| **512^3 H200 x 1** | **lean, cubic, host (job 8815590)** | **116172** | **908** | **~105** (2 load cases) |
| **512^3 H200 x 1** | **lean, 6 load cases, host (job 8815591)** | **114638** | **896** | 550 before the transfer fix |
| 128^3 H200 | lean, final code, device / host | 3134 / 2510 | 1494 / 1197 | 8.5 |
| 512^3 H200 x 1 | lean, 6 load cases, pinned host buffers (8815826) | 114638 | 896 | 377 cold / 147 / 463 (2.0 s per CG iteration) |
| 512^3 H200 x 2 | lean, 6 load cases (8815895, 8816168) | 57319-60875 per GPU | | 84 / 59 after the first (1.14 s per CG iteration) |
| 512^3 H200 x 4 | lean, cubic (8815617) | 29527 per GPU | | 44-78 (0.84 s per CG iteration) |
| 512^3 L40S x 4 | lean, 6 load cases (8816090) | 29393 per GPU | | 316 cold / 127 / 89 (1.69 s per CG iteration) |
| 256^3 H200 | lean, one-point coarse levels | 14076 | 880 | 55 |
512^3 fields: 810 B/voxel (fine level 691, MG level 1 104, rest 15), CuPy pool 84, context + FFT plans 13.
K on one H200 (fable_time_K.py, job 8815618): 256^3 fine K 28 ms (library K at the start: 54 ms), level-1 (8-point)
K 13 ms, V-cycle 185 ms; 128^3: 3.7 / 2.1 / 30 ms (small levels are ~25 % of the cycle at 128^3, negligible at 512^3).

### Findings
* Time is K-bound: one CG iteration = 5 fine K + 4 level-1 K + small levels. What made it faster, in order:
  CuPy kernels on contiguous buffers instead of strided views (material, hourglass scaling, Jacobi product),
  a single kernel for the grid transfer, three instead of six-seven ghost exchanges per K (matters under MPI:
  128^3 on 4 L40S 58825 -> 22043 exchanges per run, 18 s -> 5 s), small levels agglomerated on rank 0's GPU.
* One-point coarse levels (`MG_COARSE_ELEMENT=trilinear_hexahedron_1Q_hg MG_COARSE_HG_EPS=1`): CG iterations
  174 -> 185 per evaluation at 32^3 (eps 0.3: 214, eps 3: 202), level-1 work / 8, memory -85 B/voxel. Changes the
  preconditioner, so f differs at the solver-tolerance level (not the default). H200 256^3 timing: job 8815628.
* **SYMMETRY=cubic is numerically correct but leads to poor designs so far**: 32^3 from 7 symmetric starts (symmetrised
  CH, cos, product, symmetrised converged unsymmetric design ...) and 64^3 / 128^3 from the symmetrised CH start all
  converge (XTOL) to f 6.1-7.0 with nu12 = +0.055..0.066, C11 ~ target, shear 0.012-0.021 (target 0.058): a
  frame-like non-auxetic basin. Without symmetry: 32^3 f 2.94, nu -0.15..-0.17 after 400 iterations; 64^3 f 2.10
  after 307 evaluations (job 8815596 running). Laptop unit `fable-wp3r` tries 8 random symmetric starts
  (`logs/fable/wp3_random_report.txt`). Until a start is found that reaches the auxetic basin, production runs
  should use 6 load cases.
* WP5: `sbatch_TO3D_large.sh` + `submit_TO3D_large_chain.sh` smoke-tested on 2 L40S (jobs 8815608-12): time-budget
  stop, continuation from the checkpoint (3 segments at 32^3), hand-over 32^3 final -> 64^3 start.

* **512^3 with 6 load cases on one H200 works** (job 8815591): peak 114638 MiB (896 B/voxel), 4 evaluations,
  929 / 373 / 552 / 116 s. Half of that time was warm-start traffic (48 fields of 3.2 GB per evaluation into
  freshly allocated pageable host arrays) -> now persistent pinned host buffers; rerun 8815826 (`fullB_host_N512`).
* One-point coarse levels at 256^3 on an H200 (8815628): 1393 vs 1305 CG iterations, 0.211 vs 0.242 s per iteration
  -> 7 % faster per evaluation, 67 B/voxel less. Not worth a different default: default stays the 8-point coarse
  levels (reproduces the baseline), option documented.
* `SYMMETRY=cubic_chiral` (24 rotations, no mirror planes; same cubic stiffness form, same 2 load cases; validated
  like cubic): 6 starts at 32^3 -> the same basin (f 6.12-6.24, nu +0.05). So with either symmetry the optimiser
  finds no auxetic design for this target; unsymmetric 64^3: f 2.10 (553 evaluations, job 8815596).
* **MPI at 512^3 is slow so far**: cubic on 2 H200 (8815616) 169 / 125 / 170 s per evaluation vs 115 / 59 / 105 s on
  one, plus ~100 s between evaluations (one GPU: 3 s) -> the distributed GPU FFT of the Helmholtz step, the symmetry
  remap, and probably the ghost exchanges of the big fields. Done: Helmholtz by CG on the 7-point stencil
  (`HELMHOLTZ=cg`, default under MPI; same history as the FFT to 2e-12), timers for Helmholtz / symmetry / warm
  transfers in the TOTAL line. Profile job 8815895 (512^3, 2 H200, 6 load cases, `PROFILE=1`) queued.
* H200 has a per-user GPU limit (QOSMaxGRESPerUser): my jobs queue behind each other.
* Note: `fable_run_TO.sh` always sets DESIGN_OPT=lbfgs_helm = NuMPI's L-BFGS in `to3d_large.py`; the baseline used
  `design_optimizers.py`. The histories agree to 2e-8, i.e. the two optimizers are equivalent as expected.

* 512^3, 6 load cases, one H200 with pinned warm-start buffers (8815826): 377 / 147 / 463 / 105 s per evaluation,
  solves 984 s for 532 CG iterations (1.85 s per iteration), warm-start transfers 22 s in total (before: ~880 s).
* MPI on 2 H200 at 512^3 (8815895, 6 load cases, profile): after the first evaluation 84 and 59 s per evaluation,
  solves 1.14 s per CG iteration (one GPU: 2.0) = 1.75x; per GPU 60.9 GiB (one GPU 113 GiB: +3.6 GiB per rank).
  Ghost benchmark `fable_time_ghosts.py` on 2 H200 at 512^3 (8815944): exchanges 2.2 / 6.2 / 8.5 ms (nodal, stress,
  hourglass strain), K 126 ms (one GPU 217), V-cycle 853 ms (1393), multigrid setup 18 s first / 8 s later.
  **But the first evaluation stalls ~10 min somewhere** (8815895: "MG setup" 626 s; the cubic run 8815616: 640 s
  before the first evaluation) -- not reproduced by the benchmark. Per-stage timing added (`setup:` lines);
  tests 8816011 (2 H200) and 8816090 (4 L40S), logs `stall_N512_np{2,4}.log`.

* **512^3 on 4 L40S** (8816090, 6 load cases): no stall (setup 24 s; the stall is an H200-run effect), evaluations
  316 / 127 / 89 s, solves 1.69 s per CG iteration (one H200: 2.0), 29.4 GiB per GPU (48 GiB cards).

### What 512^3 and 1024^3 need (estimates from the measurements above; 6 load cases, 8-point coarse levels)
* CG iterations per evaluation over a whole run: ~150 (64^3 run 8815596: 82784 iterations / 553 evaluations).
* **512^3 on one H200**: 113-115 GiB of 140 GiB; ~2 s per CG iteration -> ~330 s per evaluation -> 500 evaluations
  ~46 h = two 24 h segments (chain script). On 2 H200: 1.14 s per iteration -> ~27 h. Host RAM ~120 GB.
* **1024^3 on 8 H200 (one full node)**: per GPU the load of 512^3 on one GPU plus MPI overhead and two 1024^2 ghost
  planes: ~120 GiB of 140 GiB. Per CG iteration ~2.0 s x (1 + ~25 % exchange) = ~2.5 s -> ~400 s per evaluation ->
  500 evaluations ~56 h = three 24 h segments. Host RAM: warm starts 310 GB (pinned; `WARM_STORE=host32`: 155 GB)
  + L-BFGS memory (2 x 20 + ~10 vectors of 8.6 GB) ~430 GB -> ~800 GB of 1.5 TB.
  Not yet demonstrated: 8 ranks on GPUs (needs a full node = the user's OK), and the first-evaluation stall.

* **The "stall" was the start-design interpolation on the host**, not MPI: timed run 8816168 (2 H200, per-rank
  timing) showed rank 1 arriving ~570 s after rank 0 at the first evaluation; the 8-corner numpy interpolation of
  the start design is memory-heavy and erratically slow on the loaded H200 nodes (one GPU: 218-322 s of start-up).
  Replaced by a separable (axis-by-axis) interpolation, identical histories. Check run 8816779 (2 H200, 512^3,
  6 load cases): start-up 7 s, evaluations 217 (cold) / 84 / 269 s (one H200: 377 / 147 / 463), f as on one GPU.
* 512^3 cubic on 4 H200 (8815617): 0.84 s per CG iteration (1 GPU 2.0, 2 GPUs 1.14 -> speed-up 1.75 / 2.4),
  29.5 GiB per GPU; distributed symmetry projection 14 s per evaluation; Helmholtz by CG 1.5 s per evaluation.
* Baseline at 256^3 on an H200 (8816167): 40922 MiB, ~120 s per evaluation; lean: 15154 MiB, 62 s.
* Unit tests after all library changes: 200 passed (192 + 8 new); reference gradient checks unchanged.

### Open
1. 8 GPUs = full node: needs the user's OK (also for a short test). CPU MPI with 8 ranks is validated; GPU MPI
   validated with 2 and 4 ranks (L40S and H200).
2. Design symmetry gives non-auxetic designs for this target (15 mirror-symmetric + 6 chiral starts at 32^3, plus
   64^3 / 128^3): the 3x saving is available but not useful until a start / target is found that works.
3. Speed is bound by muGrid's stencil passes (fine K 28 ms at 256^3 on an H200: gradient 4, divergence 6,
   hourglass H 5.5 + H^T 8.6, material + scaling ~3). A fused hexahedron stiffness kernel inside muGrid (as its
   `IsotropicStiffnessOperator3D` does for tetrahedra) would be the next factor ~2-3; outside this task's rules.
4. Possible small gains not done: analytic Jacobi diagonal (setup is 4-7 % of a run), merged gradient+hourglass
   stencil (~10 %), one-point coarse levels (7 %, option exists).

### STATE 2026-11-14 20:15 (after the user's "yes for both", the muTopOpt comparison and the fused / float32 request)
* **128^3 production stage done** (8817007, one H200, 59 min, 545 evaluations, MAXITER): f 1.41,
  nu12 -0.32 nu21 -0.33 nu13 -0.30 nu23 -0.31, C11 0.0938 (target 0.0868), mu 0.0566 (target 0.0579), solid 0.50.
  File copied to `data/nemo_fable/`, opened for the user in muEye.
* h200 allows 8 GPUs per user in total -> the 256^3 / 512^3 stages were moved to l40s (4 GPUs each). First attempt
  8817132 died on g3606 (uncorrectable GPU ECC error; `SBATCH_EXCLUDE` had no effect) -> chain script got `EXCLUDE=`.
  Running / queued now:
  - approved chain `aux_s0.3` (one-point element): 256^3 **8817466, 8817467** (4 L40S, 2 x 12 h) -> 512^3
    **8817468, 8817469** (4 L40S, 2 x 24 h);
  - the same from the same 128^3 design with `OPERATOR=fused MG_PRECISION=single`, label `aux_s0.3_fused`:
    256^3 **8817470** (1 L40S, 12 h) -> 512^3 **8817471** (4 L40S, 24 h).
  - 1024^3 test 8817006 still queued (needs a whole free H200 node and no other h200 job of the user).
* **User request 2026-11-14: "keep lbfgs that we have but implement fused kernel and single precision"** -> done:
  `muFFTTO/multigrid.py` `operator='fused'`, `precision='single'`; `to3d_large.py` `OPERATOR=fused`,
  `MG_PRECISION=single`; description in `LARGE_3D_TO.md` section 8. Checks: fused kernel == 8-point hexahedron with
  voxel-wise material to round-off (`fable_fused_check.py`); gradient FD 1e-8 (`fable_check_fused.py`); double vs
  single multigrid: same iterations, histories equal to 8e-13; MPI 1 vs 4 ranks identical; unit tests 205 passed.
  256^3 on one L40S, 7 evaluations: one-point element 891 s / 15.0 GiB; fused double 515 s / 9.1 GiB;
  **fused + float32 multigrid 155 s / 8.5 GiB** (532 B/voxel all-in; CG iterations +7 %).
  Not decided by the user: whether production switches to it (objective differs by a few per cent: other element).
* **Comparison with muTopOpt 1.5** (my clone `~/software/muTopOpt_fable`, the user's `~/software/muTopOpt` untouched):
  - state solve on the same design, rtol 1e-6, one L40S (`fable_bench_vs_mutopopt.py`, results
    `logs/fable/bench_vs_mutopopt_l40s.txt` on NEMO). 128^3 smooth design: muTopOpt double 643 it / 5.78 s, single
    764 it / 0.78 s; mine one-point 14.5 it / 1.02 s, fused double 15 it / 0.64 s, fused + float32 MG 15 it / 0.18 s.
    Sharpened design: 11.9 s / 1.61 s / 2.08 s / 1.45 s / 0.41 s. 256^3 smooth: muTopOpt double 1895 it / 145 s, single
    not converged in 2000 it (30.7 s); mine one-point 17.5 it / 10.3 s. Memory 128^3: muTopOpt 658 (double) / 486
    (single) B/voxel, mine 904 / 614 / 570.
  - full optimisation, same problem (stress objective = muTopOpt's), same start, 128^3, one L40S each, jobs
    8817298 (muTopOpt trust region double), 8817299 (single), 8817300 (L-BFGS), 8817301 (mine, one-point element,
    `OBJECTIVE=stress`); after ~46 min wall: f 1.62 / 1.15 / 1.48 / 1.03, nu -0.10 / -0.19 / -0.12 / (not printed).
    Logs `logs/fable/mutopopt_*_N128.log`, `vs_mutopopt_stress_N128.log`; still running (4 h limit).

### 2026-11-14 21:35: 256^3 stopped early by the user ("geometry no longer changing"), 512^3 started from its checkpoint
* 256^3 fused run 8817470 cancelled at evaluation 290 (iteration ~240 of 500, f 1.1878, 14-16 s per evaluation on
  one L40S). No `_final.nc`; the design is `N256_soft0.001_eta0.0078125_energy_aux_s0.3_fused_checkpoint.nc`
  (iteration 220-240; copy on the laptop in `data/nemo_fable/N256_aux_s0.3_fused_checkpoint.nc`, opened in muEye).
* New jobs (old dependents 8817471, 8817511, 8817513, 8817514 cancelled): 512^3 **8817830, 8817831** (4 L40S,
  2 x 24 h, start = that checkpoint) -> 1024^3 **8817832, 8817833** (8 H200, 2 x 24 h). Test 8817512 still queued.
* Head-to-head with muTopOpt at 128^3 (stress objective, same start, one L40S each): ours fused + float32 multigrid
  500 iterations in 1187 s, f 0.964, nu -0.19, 2128 MiB; ours one-point 6271 s, f 0.954, nu -0.19, 2984 MiB;
  muTopOpt trust region single converged after 185 iterations, 6354 s, f 1.109, nu -0.21, 2328 MiB; muTopOpt double
  runs (trust region, L-BFGS) at f 1.25 / 1.18 after 2 h 19 min, still running. Solve benchmark 256^3: ours fused +
  float32 multigrid 19 iterations / 1.75 s (sharpened 53.5 / 4.83 s).

### PRODUCTION SWITCHED 2026-11-14 20:25 (user: "Yes of course use new super fast and efficient version")
All production now `OPERATOR=fused MG_PRECISION=single`, label `aux_s0.3_fused`, from the 128^3 one-point design:
* 256^3 **8817470** (1 L40S, running, ~20 s per evaluation) -> 512^3 **8817471** + safety segment **8817511**
  (4 L40S, 24 h each) -> 1024^3 **8817513, 8817514** (8 H200 = full node, 2 x 24 h, afterok 8817511).
* 1024^3 full-node test with the new operator: **8817512** (1.5 h limit, 1 iteration) ->
  `logs/fable/test_fused_N1024_np8.log`. If it fails (memory, MPI), cancel 8817513/14 and fix first.
* Cancelled (my own jobs): the one-point chain 8817466-69 and the old test 8817006. A partial one-point 256^3
  checkpoint may exist under the label `aux_s0.3` (not deleted; ignore).
* Logs `$WS/runs/FABLE/TO3D_N<N>_aux_s0.3_fused.log`, data `$WS/runs/FABLE/data/N<N>_*_aux_s0.3_fused_*`.

### SUBMITTED 2026-11-14 18:25 (user: "Yes for both" = short 8-GPU 1024^3 test + production chain)
* 1024^3 full-node test: job **8817006** (`FB_1024_test`, 8 H200, 1.5 h limit, 1 iteration, 6 load cases, host warm
  starts) -> `logs/fable/test_N1024_np8.log`, peak line in `logs/fable/peak.txt` (NEMO clone).
* Production chain `aux_s0.3` (6 load cases, eta 2h, nu* -0.5, scale 0.3, MAXITER 500, HISTORY_EVERY 50):
  128^3 **8817007** (1 H200, 4 h) -> 256^3 **8817008, 8817009** (1 H200, 2 x 12 h) -> 512^3 **8817010, 8817011**
  (2 H200, 2 x 24 h). Logs `$WS/runs/FABLE/TO3D_N<N>_aux_s0.3.log`, data `$WS/runs/FABLE/data/N<N>_*_aux_s0.3_*`.
* TODO when the 1024^3 test has passed: attach the 1024^3 stage to the chain,
  `sbatch -p h200 --gres=gpu:8 --ntasks=8 --mem=900G --time=24:00:00 -J FB_TO3D_1024 --dependency=afterok:8817011
  sbatch_TO3D_large.sh 1024 $WS/runs/FABLE/data/N512_soft0.001_eta0.00390625_energy_aux_s0.3_final.nc aux_s0.3`
  (TIME_BUDGET=84900, HISTORY_EVERY=0; 3 segments, the later ones afterany). If the 512^3 run needs a third
  segment, submit one more `sbatch_TO3D_large.sh 512 ...` (it continues from the checkpoint).
* Evaluate finals with `report_TO3D_nu.py` (rsync the _final.nc to the laptop).

### Production recipe
```
cd ~/software/muFFTTO_fable/experiments/Multigrid_preconditioning      # on NEMO, after ./fable_sync_nemo.sh
DRY_RUN=1 ./submit_TO3D_large_chain.sh aux_s0.3 $PWD/data/CH3D/CH3D_start_bi_s11.nc \
    "128 1 1 04:00:00 32G" "256 1 1 12:00:00 64G" "512 1 2 24:00:00 160G" "1024 8 3 24:00:00 900G"
```
(remove DRY_RUN=1 to submit; the 1024^3 stage is a full node for 3 x 24 h -> ask the user first; a 512^3 stage on
4 H200 instead of 1: "512 4 1 24:00:00 160G"). Evaluate a result with
`python ../dual_preconditioner/topology_optimization/report_TO3D_nu.py <..._final.nc>`.

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

## 2026-11-15 06:15  stopping test on the energies; 512^3 progress

- 2D prototype `stop_criterion_2D.py` (128^2, 600 iterations, scale 0.3 and 1; `logs/stop_criterion/`,
  `figures/stop_criterion_2D.png`): "energies close to target" fires too early (it 18-43, design diffuse) or never
  (unreachable target). "Energies settled" (max change of an energy / its target over 10 iterations) is robust:
  1e-3 -> ~20 % of the run, nu within 0.005 of final; 3e-4 -> 40-50 %, design within 1-3 % of final. The step test
  (xtol 1e-3) fires at 45-55 %.
- `to3d_large.py`: `ENERGY_TOL`, `FTOL`, `MISFIT_TOL`, `FTOL_WINDOW` (all off by default; history unchanged). Each
  iteration prints an `ITER` line with these measures. Commit 801cc17, synced to NEMO.
- Thresholds for production NOT yet chosen by the user (suggested: 1e-3 intermediate grids, 3e-4 final grid).
- 512^3 `aux_s0.3_fused` (job 8817830, 4 L40S, g3611, old code without ENERGY_TOL): 06:13 eval 467 / iteration 371,
  f 1.0822 (1.1142 @50, 1.0999 @200, 1.0871 @300, 1.0835 @400), ~35 s per evaluation, no errors. Time limit 21:31.
- 1024^3 full-node test 8817512 still pending (Resources), estimated start 14:47 on 15 Nov.

## 2026-11-15 06:20  production resubmitted with ENERGY_TOL (user: 1e-3 for 512^3, 3e-4 for 1024^3)

- Cancelled 8817830 (512^3, at iteration 371, f 1.0822) and 8817831-33. Resubmitted, same label `aux_s0.3_fused`,
  `OPERATOR=fused MG_PRECISION=single HISTORY_EVERY=50`:
  512^3 `ENERGY_TOL=1e-3`: 8819699 (running, continued from the checkpoint of iteration 371), 8819700 (afterany);
  1024^3 `ENERGY_TOL=3e-4`: 8819701, 8819702 (8 H200, 24 h each; afterok 8819700 AND the full-node test 8817512).
- User's test "512^3 directly from a Cahn-Hilliard start, no iteration limit": label `fromCH_s0.3_fused`, start
  `data/CH3D/CH3D_start_bi_s11.nc` (the 128^3 CH field the chain started from), `MAXITER=100000 ENERGY_TOL=1e-3`,
  4 L40S, jobs 8819707 + 8819708 (24 h each). Compare wall time / result with the chain 128 -> 256 -> 512.
- Local copy of the 512^3 design at iteration 370: `data/nemo_fable/N512_aux_s0.3_fused_checkpoint_it0616.nc` (muEye).

## 2026-11-15 08:15  wild L-BFGS trial designs (the "hang"), STEP_GROWTH guard, video, mesh-independence check

- The 512^3 restarts (8819699, 8819700) "hung" at evaluation 5: a line-search trial design with MEAN change 0.13 from
  the accepted design (normal accepted steps on the plateau: 2e-5); its solves take 130-320 CG iterations instead of
  ~10. Seen via the new `TRIAL design` line. Also hit by the from-CH test at iteration 29 (old code), and it explains
  why the first 512^3 segment averaged 67 s per evaluation instead of 35 s.
- Guards in `to3d_large.py` (off by default): `MAX_STEP` (absolute mean change) and `STEP_GROWTH` (mean change >
  STEP_GROWTH x largest of the last 10 accepted steps): rejected without solving, the line search halves the step.
  MAX_STEP=0.05 alone was too loose on the plateau (job 8819947: still ~10 slow evaluations per iteration).
- User: "current 512 is sufficient as an init solution". 512^3 stage ended at the checkpoint of iteration 370
  (f 1.0822); 1024^3 = 8820095 + 8820096 (`ENERGY_TOL=3e-4 STEP_GROWTH=20`, start = that checkpoint, afterok the
  full-node test 8817512). All older 512^3 / 1024^3 production jobs are cancelled.
- From-CH 512^3 test restarted with the guard: 8820101 + 8820102 (`fromCH_s0.3_fused`, continues from its checkpoint
  of iteration 20; the first attempt 8819707 spent 58 min on 28 iterations and then stalled). Wall time = sum of logs.
- Video 128^3 -> 256^3 -> 512^3: `render_TO3D_multigrid_video.py` + `sbatch_render_TO3D_multigrid.sh` (NEMO,
  ~/software/render-venv), result `figures/video/aux_s0.3_grids.mp4` (NEMO: `$WS/runs/FABLE/video/`).
- 2D mesh-independence check (`run_mesh_independence_2D.sh`, `plot_mesh_independence_2D.py`): fixed eta -> objective,
  interface penalty and nu agree between 64^2 and 128^2; eta = 2h -> a different problem per grid (nu -0.21, -0.27,
  -0.30 at 32^2, 64^2, 128^2). 256^2 runs on the laptop GPU (unit fable-meshindep-gpu) pending.
- `fable_sync_nemo.sh` now excludes `data/nemo_fable` (it had started to upload the downloaded histories).

## 2026-11-15 09:15  Opus 5.5 took over (Fable closed); cause of the wild L-BFGS steps found and fixed

- Cause: after a (re)start the L-BFGS memory is empty and NuMPI uses H_0 = Helmholtz with scale 1, but the true
  scale (gamma = s.y / y.H_0 y) is ~6e3 at 128^3 (grows with the grid). The first step is ~1e5x too small (128^3
  restart: mean change 9e-8; 512^3: f change 4e-10), its curvature pair is CG noise (pair gamma 0.22 instead of
  ~6e3; cos(s, y) 0.08 looks normal, so a cautious cos test does not catch it), and the next directions blow up
  (mean change 0.11 at 128^3, 0.13 at 512^3). Afterwards the line search can fail completely (128^3: 40 halvings,
  accepted step 1e-13, then the XTOL test ended the run after 4 iterations and 49 evaluations).
- Test `plot_lbfgs_restart.py` -> `figures/lbfgs_restart_128.png`: 128^3 restart from the converged chain design,
  logs `logs/fable/rst_<tag>_N128.log`: base 4 it / 49 evals (stuck); H_0 scale carried over (LBFGS_GAMMA0=6000,
  local optimiser) 15 it / 22 evals, f 1.4165 -> 1.4077; first step capped at a largest change of 0.05 (local
  `LBFGS_FIRST_STEP` and production NuMPI path) 15 it / 22 evals, identical iterates in both paths.
- Fix `LBFGS_FIRST_STEP=<x>` in `to3d_large.py` (off by default): H_0 = c x Helmholtz with c fixed at the first
  call so that the first step changes the design by at most x. NuMPI's gamma makes all later directions independent
  of c. User approved it for the 1024^3 production: resubmitted as **8820336 + 8820337** (same settings as
  8820095/96 + `LBFGS_FIRST_STEP=0.05`: `OPERATOR=fused MG_PRECISION=single HISTORY_EVERY=50 ENERGY_TOL=3e-4
  STEP_GROWTH=20`, afterok the test 8817512); 8820095/96 cancelled. From-CH test and video jobs unchanged.
- Note: the STEP_GROWTH guard needs 2 accepted steps, so it never caught a wild direction in iterations 2-3.
- Video: renderer takes `HISTORY_EVERY=2,2,5` (one value per file). 128^3 dense preview (render job 8820289,
  8 min, 250 designs): `figures/video/video_s0.3_N128_preview.mp4`. The video rerun 128^3 ended at MAXITER 500,
  f 1.5007 (the fused chain run reached 1.412: a different path / local optimum).
