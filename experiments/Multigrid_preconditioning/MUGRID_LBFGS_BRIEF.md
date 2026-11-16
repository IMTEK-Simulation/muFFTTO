# Brief: bound-constrained L-BFGS in muGrid (fast, GPU, MPI)

Written 2026-11-16 for a new Claude session. The user (Martin) wants the design optimiser of our topology
optimisation (TO) in **muGrid**, next to its CG and preconditioners, so it uses muGrid's fields, kernels, GPU and
MPI. Requirements: clear, reproducible, human-readable and fast. Read the whole brief before starting. Keep your
messages to the user short.

## 1. What and why

- The optimiser: L-BFGS with box bounds (0 ≤ ρ ≤ 1) and a **preconditioned initial inverse Hessian** H0 = γ P M P.
  - M = the user's preconditioner, in TO a Helmholtz smoother (I + c η² (−Δ))⁻¹ applied with muGrid's FFT, c = 4.
  - P = the 0/1 mask of the free variables; γ = ⟨s, y⟩ / ⟨y, P M P y⟩.
  - It cut objective evaluations 2-3x at the same or better designs.
- Why muGrid: at 1024³ the vector work of NuMPI's host L-BFGS took ~2500 s per iteration. A GPU version with fixed
  memory takes ~0.3 s per iteration at 512³/4 GPUs (laptop measurement of the reference below).

## 2. References (read them)

- **Algorithm (the specification):** NuMPI `l_bfgs_bounded` with `precond`.
  - PR https://github.com/muSpectre/NuMPI/pull/112, local checkout `/home/martin/PycharmProjects/NuMPI_pr`, branch
    `enh/lbfgs-precond`.
  - Files: `NuMPI/Optimization/BoundedLBFGS.py`, `_lbfgs_helpers.py`, tests in `test/Optimization/test_bounded_lbfgs.py`.
  - Details: masks, two-loop recursion, curvature test s·y > 1e-12 |s||y|, projected Armijo backtracking,
    steepest-descent retry, xtol/ftol/gtol tests, no xtol test on the first step.
- **Fast version:** `/home/martin/PycharmProjects/muFFTTO_fable/experiments/Multigrid_preconditioning/lbfgs_xp.py`.
  - Same algorithm on numpy or CuPy arrays, fixed memory: every vector allocated once, here as muGrid fields; in-place
    updates; one allreduce per inner product.
  - Optional curvature pairs in pinned host memory (`pair_store='host'`) when they do not fit on the GPU.
  - `reduction='sum'` reproduces NuMPI bitwise for checks.
- **How the TO driver uses it:** `to3d_large.py` in the same folder (`DESIGN_OPT=lbfgs_xp LBFGS_DEVICE=gpu`).
- **Open algorithmic issue:** section 5 of `.../Multigrid_preconditioning/HANDOFF.md` in `muFFTTO_fable`.
  - After a restart (lost memory) the first step is tiny, its curvature pair is noise, and a few iterations later
    the step is ~5000x too large.
  - Candidate fixes: carry γ across restarts, or a cautious pair test cos(s, y).
  - Do not change the algorithm silently. Propose a fix as an option with a test, and ask the user.

## 3. What to build

1. **Design note** (`docs/lbfgs.md`), shown to the user before coding.
   - API, e.g. `muGrid.Optimizers.l_bfgs_bounded(fun, x, lower, upper, precond=None, maxcor=20, gtol, xtol, ftol,
     maxiter, callback)` with x a muGrid field or a field-shaped array.
   - Where it lives (`Solvers.py` or a new `Optimizers.py`).
   - Memory: 2 maxcor + about 10 field-sized vectors.
   - **First check whether the muGrid maintainers want an optimiser in muGrid.** Look for similar code and the
     contribution docs. If unclear, ask the user; the fallback location is muTopOpt.
2. **Serial CPU** implementation on muGrid fields with `linalg` kernels (dot, axpy, axpby, copy); no temporaries.
3. **GPU**: device fields; only scalars go to the host.
4. **MPI**: inner products through muGrid's communicator. The conda OpenMPI is not CUDA-aware: reduce on the device,
   copy the scalar, then Allreduce.
5. **Optional:** pair storage on the host for very large grids; the restart fix (after the user's OK).

## 4. Validation, required

- **Iterates equal NuMPI's** (`precond=None` and with a Helmholtz `precond`): bitwise with a sum-reduction check
  mode, round-off otherwise. Same `nit`, `nfev` and message on NuMPI's test problems: quadratic, box solution,
  Laplacian with FFT Helmholtz, many active bounds.
- CPU vs GPU, 1/2/4 MPI ranks: same iterates up to round-off.
- **Speed:** vector work per iteration at 33.5 M values (512³ / 4 ranks) vs `lbfgs_xp.py` (target ≤ 0.3 s on the
  laptop GPU with pairs on the device) and vs NuMPI (~11 s).
- **TO check:** run a 64³ 3D TO from the study with the new optimiser in place of `lbfgs_xp`. Same objective history
  up to round-off, at least for the first 20 iterations.
  - The 3D TO driver: `to3d_large.py`.
  - Default settings: `OPERATOR=fused MG_PRECISION=single`, energy objective, target scale 0.3.
  - Read `HANDOFF.md` section 2 for how to run it on the laptop.

## 5. Workspace and rules

**Your workspace**
- Your own **new** muGrid clone. Do not use `/home/martin/PycharmProjects/muGrid` (user), `muGrid_mg` (multigrid
  port session), `muGrid_cghg` (CG/hourglass session) or `muGrid_precond`.
  ```
  git clone https://github.com/muSpectre/muGrid.git /home/martin/PycharmProjects/muGrid_lbfgs
  cd /home/martin/PycharmProjects/muGrid_lbfgs && git checkout -b feat/lbfgs-bounded
  ```
- Commit on your branch. Do not push or open PRs unless the user asks.
- Read only: `muFFTTO`, `muFFTTO_fable`, `NuMPI`, `NuMPI_pr`.

**Python environment**
- Your own venv on top of the micromamba env `muffto-gpu`:
  ```
  eval "$(~/.local/bin/micromamba shell hook -s bash)"; micromamba activate muffto-gpu
  python -m venv --system-site-packages ~/venvs/muGrid_lbfgs && source ~/venvs/muGrid_lbfgs/bin/activate
  pip install --no-build-isolation --no-deps -e . -Cbuild-dir=build-cuda \
      -Ccmake.define.MUGRID_ENABLE_CUDA=ON -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=120 \
      -Ccmake.define.MUGRID_ENABLE_MPI=ON -Ccmake.define.CMAKE_BUILD_TYPE=Release
  python -c "import muGrid; print(muGrid.__file__)"      # must point into muGrid_lbfgs
  ```
- **Never `pip install` into `muffto-gpu` itself.** Production code and other sessions use it.
- Run muGrid's test suite once first, as the baseline.
- For NuMPI comparisons use `PYTHONPATH=/home/martin/PycharmProjects/NuMPI_pr`.

**Machine**
- One 8 GB GPU (sm_120), **shared** with other sessions: check `nvidia-smi`, stay ≤ 2 GB. 16 cores, 60 GB RAM.
- Builds and long runs: `systemd-run --user --unit=lbfgs-<name> -p MemoryMax=20G -p MemorySwapMax=0
  --working-directory=<dir> bash <script>`, never `&`/nohup. An OOM kill takes down all sessions. Build with `-j6`.
- Never `pkill -f`/`pgrep -f` a pattern in your own command line; kill by PID.
- No NEMO cluster use.

**Notes**
- Keep `docs/lbfgs.md` with a "Status / RESUME HERE" section updated after every milestone.
- End with a short report: what is in muGrid, validation, speed vs `lbfgs_xp` and NuMPI, open points.
