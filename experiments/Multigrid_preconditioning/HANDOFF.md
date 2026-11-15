# Hand-off: large 3D topology optimisation (state 2026-11-15 08:40)

Read this file only; it is self-contained. Details if needed: `LARGE_3D_TO.md` (how the method works),
`FABLE_NOTES.md` (chronological log, measurements). Do not reload the previous session's transcript.

## 1. What this is

3D phase-field topology optimisation of an auxetic unit cell (target Poisson ratio -0.5, `TARGET_SCALE=0.3`) on grids
up to 1024^3 on the NEMO2 cluster. Driver: `to3d_large.py` (multigrid-preconditioned CG, 6 load cases, L-BFGS with a
Helmholtz initial Hessian, one MPI rank per GPU). The engineering is done; what remains is running, monitoring,
post-processing, and one open optimiser question (section 5).

## 2. Rules (the user's; keep them)

- Work only in the worktree `/home/martin/PycharmProjects/muFFTTO_fable` (branch `fable/large-3d-to`). Commit there.
  Never push, merge or rebase. Never edit `/home/martin/PycharmProjects/muFFTTO` (read only).
- NEMO: `ssh -o BatchMode=yes nemo '...'`. Own clone `~/software/muFFTTO_fable` (sync code with
  `./fable_sync_nemo.sh`; commit first, it copies the working tree). Runs under `WS=/work/classic/fr_ml1145-muffto`,
  `$WS/runs/FABLE/{data,logs,video}`, logs `$WS/runs/FABLE/TO3D_N<N>_<label>.log`. Job names start with `FB_`.
- Own single-node jobs <= 24 h: submit / cancel freely, but tell the user. ASK FIRST: jobs > 24 h, more than one
  node, any new full-node (8 GPU) job beyond the approved 1024^3 production + test, deleting anything in `$WS`.
- Do not touch jobs that are not `FB_*` (`AT1b_*`, `TO3D_128_sw25*`, `MG_TO` belong to another session), the NEMO
  clone `~/software/muFFTTO`, or the laptop systemd units `mg-*`.
- Never compute on the login node. Never run `srun` in the foreground; use `sbatch`. Smoke-test new or changed job
  scripts with a tiny job first.
- Laptop: long or heavy runs through `systemd-run --user --unit=fable-<name> -p MemoryMax=<cap> -p MemorySwapMax=0
  --working-directory=<dir> bash <script>`. Never `pkill -f` / `pgrep -f`; kill by PID. Python:
  `~/micromamba/envs/muffto-gpu/bin/python` (`source fable_env.sh` sets `$PY`). Keep GPU use <= ~3 GB if `mg-*`
  units are active.
- Figures: plain tick labels, font >= 12, exact pixels for designs; open every figure / video with `xdg-open` and do
  NOT read image files. muEye: `setsid nohup ~/PycharmProjects/muEye/run_mueye_cuda.sh <file.nc>`; never kill muEye.
- User preferences: keep L-BFGS (no optimiser swap without asking); CG stops on the energy-error estimate (rtol 1e-6);
  no variants inside library solvers (`muFFTTO/solvers*.py`, NuMPI) - experiment-local code only; exploratory tests at
  32^3-128^3 (2D: up to 256^2); give short status messages; the user wants to save tokens.
- Commit trailer: `Co-Authored-By: <your model name> <noreply@anthropic.com>`.

## 3. Jobs on NEMO right now (all mine, all `FB_*`)

| job | what | state at 08:40 | settings |
|---|---|---|---|
| 8817512 | 1024^3 full-node test, 8 H200, 1.5 h | pending, est. start 14:47 | - |
| (cancelled) | 1024^3 production 8820336/37: test 8817512 timed out (see FABLE_NOTES 20:00) | - | - |
| 8820185, 8820186 | user's test: 512^3 directly from the Cahn-Hilliard start, label `fromCH_s0.3_fused`, 4 L40S, 2 x 24 h | pending | `ENERGY_TOL=1e-3 STEP_GROWTH=20 MAXITER=100000`, continues from its checkpoint (iteration 20) |
| 8820129 | video rerun 128^3, label `video_s0.3`, 1 L40S | running (iteration ~200 after 10 min) | `HISTORY_EVERY=2 ENERGY_TOL=1e-3 STEP_GROWTH=20` |
| 8820130 | video rerun 256^3 (afterok 8820129) | pending | `HISTORY_EVERY=2`, same tests |
| 8820131, 8820132 | video rerun 512^3 (afterok 8820130), 4 L40S | pending | `HISTORY_EVERY=5` (about 1 GB per frame) |

All use `OPERATOR=fused MG_PRECISION=single`. The 512^3 production stage is finished by user decision ("current 512
is sufficient as an init solution"): its result is
`$WS/runs/FABLE/data/N512_soft0.001_eta0.00390625_energy_aux_s0.3_fused_checkpoint.nc` (iteration 370, f 1.0822;
local copy `data/nemo_fable/N512_aux_s0.3_fused_checkpoint_it0616.nc`). There is no 512^3 `_final.nc`.

Bad L40S nodes (GPU ECC errors): g3606, g3608 and, new today, **g3610**. Always submit L40S jobs with
`EXCLUDE=g3606,g3608,g3610`. The H200 queue allows 8 GPUs per user in total, and both H200 nodes are busy with
other users, hence the wait.

How to submit (on NEMO, in `~/software/muFFTTO_fable/experiments/Multigrid_preconditioning`):
`OPERATOR=fused MG_PRECISION=single STEP_GROWTH=20 ENERGY_TOL=<tol> HISTORY_EVERY=<n> PARTITION=l40s
EXCLUDE=g3606,g3608,g3610 [AFTER=<job>[:<job>]] ./submit_TO3D_large_chain.sh <label> <start.nc>
"<N> <GPUs> <segments> <time> <mem>"`. A segment continues from the run's own checkpoint if one exists. The next grid
starts from `<...>_final.nc`, which is only written when a run ends by itself (stopping test or MAXITER=500).

## 4. What to do next (in this order)

1. **Watch the jobs.** Healthy log: `EVAL` lines every 30-50 s at 512^3, `ITER` lines, CG counts around 10-20 per
   solve, `TRIAL design` lines with small mean change. Signs of trouble: no new line for > 10 min; a traceback with
   "ECC" (bad node: cancel, add the node to `EXCLUDE`, resubmit); many `rejected without solving` lines per iteration.
2. **1024^3 test 8817512.** If it fails, cancel 8820336/8820337, fix, and tell the user. If it passes, the production
   starts by itself. First 1024^3 evaluation and memory: note them in `FABLE_NOTES.md`.
3. **Smooth video** (the user asked for more frames). When the `video_s0.3` stages finish, render on NEMO:
   `OUT=$WS/runs/FABLE/video/video_s0.3 HISTORY_EVERY=2 HOLD=3 sbatch sbatch_render_TO3D_multigrid.sh <N128
   history> <N256 history> ...` then rsync the `.mp4` to `figures/video/` and `xdg-open` it. CAUTION:
   `render_TO3D_multigrid_video.py` assumes ONE `HISTORY_EVERY` for all files; the 512^3 stage uses 5, the others 2.
   Either render in two calls or add a per-file step (small edit). `HOLD` = video frames per design (45 was used for
   the sparse video; use 2-4 for dense histories). Test with `TEST=1` first (writes three stills).
   The first (sparse, 24 designs) video is `figures/video/aux_s0.3_grids.mp4`.
4. **From-CH test**: when it ends, report to the user: total wall time (sum over its segments in the log, including
   the first attempt: 58 min for 28 iterations, then stalled on the bad node), iterations, objective, compared with
   the chain 128 -> 256 -> 512 (59 min + ~1.5 h + ~8.8 h, the last inflated by slow evaluations, see section 5).
5. **Final results**: for every `_final.nc` evaluate the Poisson ratio with `report_TO3D_nu.py` (rsync the file to
   `data/nemo_fable/` first) and open the design in muEye. The 256^3 and 512^3 Poisson ratios have NOT been
   evaluated yet (last measured: 128^3, nu12 -0.30 ... -0.33).

## 5. Open question: L-BFGS proposes wild steps

Symptom (seen at 512^3 in `TRIAL design` lines): accepted steps change the design by ~2e-5 on average, then the
next search direction changes it by 0.13 on average (nodes flip between void and solid). The backtracking line search
evaluates that trial design; its linear solves need 130-320 CG iterations instead of ~10, so one evaluation takes
many minutes ("hang"). It happened after every restart from a checkpoint on the plateau (L-BFGS memory is lost at a
restart). Part of the stalls of the from-CH test were instead the ECC fault on g3610.

Current workaround in `to3d_large.py` (off by default, on in all queued jobs): `STEP_GROWTH=20` rejects a trial
design whose mean change exceeds 20 x the largest of the last 10 accepted steps, without solving (returns f = inf,
the line search halves). `MAX_STEP=<x>` is the absolute variant; 0.05 was too loose on the plateau.

Not yet known: the cause. Hypotheses: (1) NuMPI's L-BFGS keeps any curvature pair with s.y > 1e-12 |s||y| and uses
Armijo backtracking only (no Wolfe condition), so a near-zero pair blows up the direction; (2) the set of free nodes
(bounds 0 / 1) changes between iterations; (3) gradient noise (CG tolerance 1e-6) on the plateau.
Diagnosis started: `design_optimizers.py` (experiment-local serial copy of the optimiser, `DESIGN_OPT=
lbfgs_helm_local`) now prints per iteration `L-BFGS pair it ...: cos(s, y), gamma, mean|d|, mean|s|, free nodes
changed`. A 64^3 run with it is in `logs/fable/lbfgsdiag_N64.log` (laptop unit `fable-lbfgsdiag2`; may be finished).
At 64^3 with the production optimiser no wild step occurred in 260 iterations, so 64^3 may not reproduce it; a restart
from a converged 128^3 design (`INIT_FROM=<final.nc>`, same grid) is the cheapest likely reproduction.
Possible proper fix (ask the user before changing production behaviour): cautious pair update (skip pairs with small
cos(s, y)) plus a trust-region-like step bound, in the experiment-local optimiser first, compared in 2D / at 64^3.
The user asked whether muTopOpt's trust-region Newton would behave better. Data from 14 Nov (128^3, stress
objective): ours L-BFGS 500 it, 1187 s, f 0.964; muTopOpt TR-Newton single 185 it (converged), 6354 s, f 1.109;
double not finished in 4 h. Newton needs Hessian-vector products (~12 linear solves each for our objective). My
recommendation to the user was: keep L-BFGS, fix it; they have not answered yet.

## 6. Things finished today (for answering questions)

- **Stopping test on the energies** (`ENERGY_TOL`): stop when every homogenised energy of the objective changed by
  less than tol of its target over 10 iterations. 2D prototype `stop_criterion_2D.py`,
  `figures/stop_criterion_2D.png`: 1e-3 stops after ~20 % of a full run with nu within 0.005 of final; 3e-4 after
  40-50 % with the design within 1-3 % of final. "Energies close to target" does not work as a stop. User decision:
  1e-3 at 512^3, 3e-4 at 1024^3. Also `FTOL`, `MISFIT_TOL` (unused).
- **Mesh independence** (2D, `run_mesh_independence_2D.sh`, `plot_mesh_independence_2D.py`,
  `figures/mesh_independence_2D.png`, `figures/mesh_independence_2D_interfaces.png`): the interface penalty is
  normalised to interface area, so with a FIXED physical interface width eta the objective and Poisson ratio agree
  between grids (64^2 -> 256^2: < 0.5 % / 0.002). With eta = 2h (what the 3D chain does) every grid is a different
  problem and the design keeps improving (nu -0.21, -0.27, -0.30, -0.32 at 32^2 ... 256^2). Two nodes across the
  interface are under-resolved; four are enough. `ETA_H=<eta / h>` in the job script sets it in 3D.
- **Convergence of the chain** (`aux_s0.3[_fused]`): 128^3 500 it, f 66.9 -> 1.412, misfit 0.167; 256^3 ~230 it,
  1.392 -> 1.188, misfit 0.085 (stopped by hand); 512^3 370 it, 1.333 -> 1.082, misfit 0.054.
- 2D script on the GPU: `DEVICE=gpu` works with `stop_criterion_2D.py` (5 x faster than the single-threaded CPU).

## 7. Housekeeping

- Uncommitted at hand-off: nothing important (check `git status`; `logs/` are untracked on purpose).
- `fable_sync_nemo.sh` excludes `data/nemo_fable` (downloaded results); keep large downloads there.
- Update `FABLE_NOTES.md` (append a dated section) when something happens; keep this file's sections 3-5 current
  or replace it with a new hand-off.
