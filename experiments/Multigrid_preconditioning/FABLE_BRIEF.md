# Brief for Fable: 3D phase-field topology optimisation at 512³ and 1024³

Written 2026-11-14 by the Claude (Opus) session that built the current code. The user (Martin) wants to see how far you
can engineer this. You work **independently** of that session, which keeps running on other tasks. The coordination
rules in section 7 keep the two of you from colliding. Read the whole brief before starting.

## 1. Goal

Run the 3D auxetic topology optimisation (TO) described in section 2 at **512³**, and ideally **1024³**, on NEMO2's
H200 GPUs. Each run must fit NEMO's job limits (section 4). The code has to stay **clear, reproducible, human-readable
and fast** (the user's words). Operators go through **muGrid**, the way the library already does it.

The physics and the numerical results must not change. Every optimisation has to reproduce the baseline (section 6).

## 2. What exists (base commit: see section 7)

Repository: `/home/martin/PycharmProjects/muFFTTO` (Python library `muFFTTO/` + experiments). The study folder is
`experiments/Multigrid_preconditioning/`. Read its `CLAUDE_NOTES.md` ("RESUME HERE") and `README.md` first.

**The TO problem** (`example_3D_elasticity_TO.py`, a self-contained copy of the production script):
- Periodic unit cell, N³ voxels. Nodal phase field ρ in [0, 1] is the design variable. Material at the quadrature
  points: s(x) C_0 with s = ρ_q² (1 − k) + k and k = 1e-3 (contrast 1e3, `SOFT_EXP=3`). It is stored as
  `domain.ScaledMaterial` (scalar per quadrature point times a constant tensor).
- Objective: match the target stiffness of an isotropic auxetic material, plus a phase-field (Cahn-Hilliard type)
  regulariser with interface width η = 2h.
  - `OBJECTIVE=energy` matches the energies C1111, C1122, C1133, C2222, C2233, C3333, C2323, C1313, C1212.
  - Target: `TARGET_NU=-0.5`, `TARGET_SCALE=0.3`.
- **6 load cases** (xx, yy, zz, yz, xz, xy). Each evaluation does one mechanical solve and one adjoint solve per load
  case, with warm starts. The displacement and adjoint fields of all 6 load cases are kept between evaluations.
- Linear solver: the library CG `muFFTTO.solvers.conjugate_gradients_mugrid_experimental`.
  - Stopping: energy-error estimate, rtol 1e-6, adaptive delay with `d_min=0`.
  - Preconditioner: geometric multigrid V(2,2) from `mg_lib.py` (Chebyshev-Jacobi smoother, re-discretised coarse
    levels down to 4³, exact pseudo-inverse there, ScaledMaterial hierarchy). Use `'MG'` on the command line.
  - It needs 12-16 CG iterations at any N; grids must be 4·2^k.
- Design update: L-BFGS with box bounds and a Helmholtz initial inverse Hessian (I + c η² (−Δ))⁻¹, c = 4, applied
  with the muFFTTO FFT (`DESIGN_OPT=lbfgs_helm HELM_C=4`, `design_optimizers.py`). It runs on host numpy with
  global (serial) arrays.
- Element: `ELEMENT=trilinear_hexahedron_1Q_hg` (one quadrature point per voxel, 8x smaller quadrature fields).
  - Flanagan-Belytschko hourglass control, `HG_EPS=0.3`, in `muFFTTO/hourglass_control.py`. The hourglass operator H
    is a muGrid `GenericLinearOperator`. It is hooked into `Discretization.apply_system_matrix_mugrid` and into the
    TO sensitivity.
  - The 8-point `trilinear_hexahedron` is the reference element.
- Start design: `data/CH3D/CH3D_start_bi_s11.nc` (Cahn-Hilliard pattern, interpolated to N). Large runs must start
  from the converged design of the next coarser grid (`INIT_FROM`), never from noise.
- NEMO driver: `sbatch_TO_mg.sh <P> <N> [start.nc]`. Its environment switches are documented in its header.

**Measured** (peak GPU memory per process, CUDA context included; `mem_TO3D_hg.sh` → `logs/mem_hg/peak.txt`):

| | 32³ | 64³ | 128³ | slope |
|---|---|---|---|---|
| 3D TO, 1Q_hg, 6 load cases | 212 MiB | 876 MiB | (being measured) | **~3.0 KiB/voxel** |
| 3D TO, 8-point | 362 MiB | 1980 MiB | — | ~7.2 KiB/voxel |
| one static MG-CG solve, 1Q_hg | 148 MiB | 430 MiB | 2456 MiB | ~1.1 KiB/voxel |

Speed:
- One stiffness application at 128³ on the laptop GPU: 8-point 331 ms, 1Q 36 ms, 1Q_hg 72 ms. In 1Q_hg, the muGrid H
  apply is 9 ms and H^T is 11 ms.
- 3D TO at 128³ on an H200 with the 8-point element: MG + Helmholtz 19.3 s/evaluation vs Green-Jacobi 34.0.

## 3. The arithmetic you are fighting

- 512³ = 134 M voxels. 1024³ = 1.07 G voxels.
- One nodal displacement field is 24 B/voxel in float64.
- At today's 3.0 KiB/voxel: 512³ needs ~400 GB and 1024³ needs ~3.2 TB.
- One H200 has 141 GB. The largest job has 8 H200 = 1.13 TB (section 4).
- So:
  - **512³ on one H200** needs ≤ ~1 KiB/voxel. At 3 KiB/voxel it needs 4 H200 with an MPI multigrid.
  - **1024³** needs both ≤ ~0.9 KiB/voxel **and** a multigrid running on 8 GPUs.
- The 3.0 KiB/voxel is about 128 nodal vectors per voxel; the essential state is far smaller.

My rough floor for the essential state (float64, 1Q_hg, all on the GPU):

| item | B/voxel |
|---|---|
| u and λ for 6 load cases | 288 |
| CG vectors | ~100 |
| MG fine-level vectors | ~120 |
| quadrature temporaries (gradient 72 B, hourglass strain 96 B) | ~170 |
| design, gradient, scale | ~50 |
| **total** | **~750** |

Moving the 6 warm-start pairs to host RAM (1.5 TB per node) brings this to ~500 B.

Time estimate at 1024³ on 8 H200, assuming ideal scaling from the measured 128³ numbers and 1Q_hg ~3x faster than
8-point: ~7 min per evaluation with 6 load cases. A few hundred evaluations are needed from a good start, so 1-2 days.
That is why the cubic-symmetry lever (WP3) matters.

## 4. Hardware, environments, rules

**Laptop**
- 16 cores, 60 GB RAM, one 8 GB RTX PRO 1000 (Blackwell) GPU **shared with the other session**.
  - Check `nvidia-smi` and keep your laptop GPU jobs ≤ ~3 GB while `mg-*` systemd units are active.
  - Do the heavy measuring on NEMO.
- Python: `~/micromamba/envs/muffto-gpu/bin/python` (CUDA muGrid built from `~/PycharmProjects/muGrid`, CuPy, OpenMPI).
- Set `CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux`, `JAX_PLATFORMS=cpu`,
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `OMP_NUM_THREADS=1`.
- Conda OpenMPI on the laptop is **not CUDA-aware**: passing GPU buffers to MPI segfaults (reduce on device, `.get()`,
  then Allreduce).
- **Never** start long or memory-heavy runs with `&`/nohup from your shell. An OOM kill takes down the whole terminal
  scope, including other Claude sessions. Use
  `systemd-run --user --unit=fable-<name> -p MemoryMax=<cap> -p MemorySwapMax=0 --working-directory=<dir> bash <script>`.
- Never `pkill -f`/`pgrep -f` a pattern that appears in your own command line. Kill by PID.
- Do not rebuild or modify the installed muGrid in `muffto-gpu`. If muGrid itself must change, build a separate copy
  in a separate env and say so.

**NEMO2** (Freiburg)
- Access: `ssh -o BatchMode=yes nemo '...'` over the user's ControlMaster. If it is closed, ask the user to run
  `! ssh -fN nemo`. Workspace: `WS=/work/classic/fr_ml1145-muffto`.
- Partition `h200`: 2 nodes (g4401, g4402) × 8 H200 (141 GB each), 1.5 TB host RAM and 192 cores per node.
  **MaxNodes = 1 per job**, MaxTime 2 days. The `l40s` partition has many 1-GPU nodes; use it for development.
- Environment: `source ~/software/muffto_env_gpu.sh` (picks the venv by `$SLURM_JOB_PARTITION`; each partition has its
  own stack). CUDA partitions load `lib/ucx-cuda`. Whether that MPI is CUDA-aware is **unverified**; check it first.
- Rules from the user:
  - You may sbatch and scancel **your own** single-node jobs ≤ 24 h freely, but report them.
  - **Ask the user before** jobs > 24 h, more than one node, or deleting anything in `$WS`.
  - Never compute on the login node. Never run `srun` in the foreground; use `sbatch`.
  - **Smoke test** every new or edited job script with a tiny job first, including any restart path.
- Both h200 nodes currently run one GPU job each from the other session, so a full-node 8-GPU job waits until a node
  frees up. Short tests with 2-4 GPUs on one node can run now.
- Use your own clone and run folders on NEMO (section 7). **Do not touch `~/software/muFFTTO`**: running jobs and the
  other session use it.

**User preferences** (from earlier sessions, they apply to you too)
- CG stops on the energy-error estimate, rtol 1e-6. Do not add options or variants to the library CG
  (`solvers.py`); test variants in the experiment folder.
- Every FFT and operator goes through the muFFTTO discretization or muGrid. Never use numpy.fft or your own engine.
- Exploratory tests run at 32³-128³. Bigger runs only as targeted measurements.
- Plots: plain tick labels, font ≥ 12, colours by method family. Open every figure you produce with `xdg-open`.

## 5. Work packages, in this order

**WP0 — set up and reproduce**
- Create the worktree (section 7).
- Reproduce the baseline memory numbers (`mem_TO3D_hg.sh` or your own per-process measurement) and one 32³ TO
  objective history (the first 10 evaluations) as the reference for WP2-WP4.

**WP1 — memory breakdown**
- Instrument the 3D TO (1Q_hg, MG, 6 load cases) at 64³ and 128³. Every muGrid field in every field collection
  (TO discretization, every MG level's discretization), every CuPy allocation (mempool), cuFFT plans and CUDA context
  → a table of bytes/voxel by owner.
- Write it into `FABLE_NOTES.md`. This decides what WP2 attacks.

**WP2 — memory diet (single GPU)**
- Goal: ≤ 1 KiB/voxel for the 3D TO with 6 load cases (lower is better), with identical results.
- Candidates:
  - the MG fine level builds its own `Discretization` next to the TO's, so fields are duplicated;
  - named temporaries that live forever;
  - full-size `kappa` or `stiffness()` arrays in the hourglass control;
  - gradient/stress scratch fields;
  - the warm-start u/λ of all 6 load cases on the GPU (move them to host and stream the current load case);
  - float32 storage of warm starts only (compute stays float64);
  - L-BFGS history placement;
  - `LOW_MEMORY` / `quad_point_blocks` in domain.py.
- Keep speed: report s/evaluation before and after.
- Demonstrate on an H200: measure 256³, then **run 512³ on one H200** for a few evaluations, if your per-voxel number
  allows it.

**WP3 — cubic design symmetry (approved by the user as an option)**
- Add an opt-in switch (`SYMMETRY=cubic`, default off). The design ρ is kept invariant under the 48 symmetry operations
  of the cube about the cell centre. On the periodic nodal grid a reflection maps node i → (N − i) mod N.
  - Project the design and the gradient onto the symmetric subspace. The start design must be symmetrised too.
  - Then C_h has cubic symmetry, and 2 load cases (xx and one shear) determine the whole objective: C1111, C1122 from
    xx and C1212 from the shear.
- Validate:
  - a finite-difference gradient check;
  - for a symmetric design, the 2-load-case objective equals the 6-load-case objective to round-off;
  - a 32³ TO with the switch converges to a sensible design (report ν12 with
    `../dual_preconditioner/topology_optimization/report_TO3D_nu.py`).
- This cuts the state and the work per evaluation by 3x.

**WP4 — multigrid on several GPUs (MPI)**
- `mg_lib.py` is serial: grid transfers and coarse levels use global arrays. muFFTTO's `Discretization` is
  MPI-capable (muGrid domain decomposition, ghost exchange).
- Make the whole TO path run with one MPI rank per GPU:
  - every MG level is a distributed Discretization with aligned decompositions;
  - restriction and prolongation are local stencils with ghost exchange;
  - power iteration, dot products and norms use MPI reductions;
  - the coarsest levels are agglomerated or replicated (4³ pseudo-inverse);
  - the Helmholtz H0 FFT and the L-BFGS dot products work on distributed arrays.
- Check first whether muGrid's FFT engine supports distributed FFTs on the GPU. If not, apply the Helmholtz H0 another
  way (e.g. a few Chebyshev or MG steps) and show that the TO results do not get worse.
- Validate on CPU MPI (1/2/4 ranks) against serial (same iterations, same objective history), then on NEMO with
  1/2/4/8 H200: correctness plus strong and weak scaling at 256³ and 512³.

**WP5 — production readiness**
- Smoke-tested sbatch scripts for the chain 256³ → 512³ (→ 1024³), each level starting from the coarser final design,
  restartable within the 2-day limit.
- **Do not launch production runs > 24 h or full-node jobs without the user's OK.** Prepare them and report the
  estimated memory per GPU, time per evaluation and wall time.

You may reorder WP2-WP4 if your measurements show a better path. Write down why.

## 6. Validation, required for every change

- Unit tests stay green: `test/test_hourglass_control.py`, `test/test_discretization.py`, `test/test_scaled_material.py`,
  `test/test_topology_optimization_elasticity.py`, `test/test_adjoint_warm_start.py`, `test/test_material_data_evaluation.py`.
  Run them with the muffto-gpu python, `JAX_PLATFORMS=cpu`.
- Gradient: `check_objective_gradient_3d.py` (8³; set `ELEMENT`, `OBJECTIVE=energy`, `TIGHT=1`). The relative
  difference to finite differences must stay ~1e-8.
- TO reproducibility at 32³ and 64³: objective history of the first 10-20 evaluations equal to the baseline (relative
  ~1e-10, or explain any difference, e.g. float32 warm starts changing CG iteration counts), CG iterations per solve
  unchanged.
- Memory and time: report peak GPU memory per process (`nvidia-smi --query-compute-apps`) and s/evaluation, before and
  after, at the same grid.

## 7. Coordination with the other session (important)

**Your workspace**
- Laptop git worktree on your own branch, from the base commit named in the prompt the user gave you:
  ```
  cd /home/martin/PycharmProjects/muFFTTO
  git worktree add /home/martin/PycharmProjects/muFFTTO_fable -b fable/large-3d-to <base commit>
  ```
  Work only in `/home/martin/PycharmProjects/muFFTTO_fable`.
- Make sure imports resolve there: `python -c "import muFFTTO; print(muFFTTO.__file__)"` from inside it. The
  experiment scripts put the repo root first on sys.path. For pytest use `PYTHONPATH=$PWD`.
- Commit on `fable/large-3d-to` as you go. **Do not push, merge or rebase other branches.**
- NEMO: your own clone `~/software/muFFTTO_fable` (rsync or git bundle from your worktree), runs and logs under
  `$WS/runs/FABLE/`, job name prefix `FB_`.

**Do not touch**
- The main working tree `/home/martin/PycharmProjects/muFFTTO`.
- The laptop systemd units named `mg-*`.
- NEMO's `~/software/muFFTTO`.
- NEMO jobs that are not yours: the `MG_TO` jobs 8813561, 8813562, 8810501 and 8814559-62, and the other studies'
  jobs (`TO3D_128_sw25`, `AT1b_*`).

**Notes**
- Keep `experiments/Multigrid_preconditioning/FABLE_NOTES.md` in your worktree, with a "RESUME HERE" section:
  decisions, measurements, running jobs, next steps. Update it after every milestone so work survives interruptions.
- End with a short report for the user: memory/voxel and s/evaluation before and after, the largest N demonstrated,
  what 512³ and 1024³ need (GPUs, hours), what is validated, and what is still open.
