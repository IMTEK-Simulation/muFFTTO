# Brief: port our heterogeneous multigrid preconditioner into muGrid

Written 2026-11-16 for a new Claude session. The user (Martin) wants the multigrid preconditioner that Fable
engineered for muFFTTO moved into the **muGrid** library: clear, reproducible, human-readable, and fast. Read the
whole brief before starting.

## 1. What to port

**Reference implementation**: `/home/martin/PycharmProjects/muFFTTO_fable/muFFTTO/multigrid.py`, class `Multigrid`
(628 lines, branch `fable/large-3d-to` of muFFTTO, commit `8e5d7be`). Its tests: `test/test_multigrid.py` in the same
worktree. Read the module docstring first: it is the specification.

- A geometric multigrid V(ν, ν)-cycle used as a fixed SPD preconditioner for plain CG.
- Problem: periodic small-strain elasticity with a **heterogeneous** material `A(x) = s(x) C_0`, contrast up to
  1e3-1e4, phase-field and sharp interfaces.
- Measured, in the production 3D topology optimisation (TO): 12-16 CG iterations independent of the grid
  (32³ … 512³, run up to 1024³) and of the contrast. The multigrid adds about 120 B/voxel on the fine grid.
- Algorithm:

| component | what it is |
|---|---|
| Smoother | Chebyshev-Jacobi, `degree` = 2 steps on [λmax/α, λmax] with α = 30; λmax(D⁻¹A) by warm-started power iteration × 1.1 |
| Grid transfer | grids N, N/2, …, `n_coarsest` = 4; P = multilinear interpolation, R = Pᵀ |
| Coarse operators | **re-discretisation with averaged material**: in 3D the coarse Gauss point in octant (i, j, k) takes the mean scale of fine voxel (2I+i, 2J+j, 2K+k); in 2D linear triangles take the mean of the 4 fine triangles, which is exactly the Galerkin PᵀAP |
| Coarsest grid | exact dense pseudo-inverse (periodic, so singular) |
| `operator='fused'` | muGrid's `IsotropicStiffnessOperator2D/3D` with one (λ, μ) per voxel |
| `precision='single'` | float32 cycle between the double-precision CG vectors |
| MPI | every level on the same communicator; slab decomposition, nested while every rank keeps an even number of planes; coarse levels (≤ `n_agglomerate` = 16) gathered and solved on rank 0 |
| GPU grid transfer | CuPy RawKernels |

**Production setting** (the one to make fast in muGrid): `operator='fused'`, `precision='single'`, trilinear
hexahedra, isotropic C_0, scale per voxel.

**A known flaw to fix while porting.** The warm-started power estimate of λmax can miss a newly localised mode: in an
AT2 fracture run a crack cut through the cell and CG went to 1000-12000 iterations. Fix: a rigorous,
contrast-independent bound c = max over Gauss points q of λmax(D_q⁻¹ K_q) (the mediant inequality; see
`gershgorin_constant` in `/home/martin/PycharmProjects/muFFTTO_fable/experiments/multigrid_vs_dual/mg_q1.py`,
c = 3.523 for 2D Q1 with ν = 0.2). Offer it as an option, `lmax='bound'`, and make it the default if the iteration
counts allow (measure).

**Other context, read on demand**
- `/home/martin/PycharmProjects/muFFTTO_fable/experiments/multigrid_vs_dual/mg_q1.py`: subclass adding the 2D
  bilinear-quadrilateral coarsening (quadrant rule).
- `/home/martin/PycharmProjects/muFFTTO/experiments/Multigrid_preconditioning/mg_lib.py`: the serial prototype;
  scaling-study results in `README.md` there.
- Engineering log and results: `HANDOFF.md`, `FABLE_NOTES.md` in `muFFTTO_fable/experiments/Multigrid_preconditioning/`.
- MG vs dual on fracture: `muFFTTO_fable/experiments/multigrid_vs_dual/README.md`.

## 2. What muGrid already has (check, reuse, do not duplicate)

You work in a **fresh clone** of muGrid (section 5), from `muSpectre/muGrid` `main`, release 1.6.1 or newer. The
user's existing checkout `/home/martin/PycharmProjects/muGrid` is older and has unpushed work; do not use it.

- **`MultigridReferencePreconditioner`** and `_MultigridLevel` in `language_bindings/python/muGrid/Preconditioners.py`,
  with the plan and measurements in `docs/multigrid_preconditioner_plan.md`.
  - It is a multigrid for the **uniform reference** operator inside Green-Jacobi, replacing the FFT for MPI scaling.
    It is serial and host-only.
  - **Ours is different:** a multigrid on the heterogeneous operator itself, with no FFT anywhere.
  - Reuse its pieces where they fit, avoid name clashes, and document how the two relate.
- `GridTransfer2D/3D` (C++, `src/libmugrid/operators/transfer.hh`), `IsotropicStiffnessOperator2D/3D`
  (`src/libmugrid/operators/solids/isotropic_stiffness.hh`), `GenericLinearOperator`, `FEMGradientOperator`,
  `linalg` (axpy, …), `CartesianDecomposition`, `Solvers.conjugate_gradients`, `JacobiPreconditioner`, and the
  `Preconditioner` base class.
- Fable's code ran against an older muGrid (built from 1.4.1-9-g14b3209b) and wrote its own CuPy kernels for the GPU
  grid transfer. Check what `origin/main` supports on the device now. Prefer C++/CUDA kernels in muGrid over CuPy
  RawKernels.

## 3. Suggested stages (reorder if your findings say so, and write down why)

1. **Design.** Write a short note (`docs/heterogeneous_multigrid.md`): API, algorithm, relation to
   `MultigridReferencePreconditioner`.
   - Suggested API: a `Preconditioner` subclass, e.g. `HeterogeneousMultigridPreconditioner(decomposition,
     grid_spacing, lambda_0, mu_0, element=FEMElement.q1, degree=2, alpha=30, n_coarsest=4, precision='double',
     lmax='bound')`.
   - Methods: `set_scale(field)` / `setup()` after each material change, and `apply(r, z)`.
   - Start with the isotropic, per-voxel case (`IsotropicStiffnessOperator`), 2D and 3D.
2. **Serial CPU** implementation and tests.
3. **GPU** (device fields, transfer kernels, no host round trips in the cycle).
4. **float32 cycle.**
5. **MPI** (nested decompositions, agglomeration on rank 0).
6. *Optional:* general material, i.e. per-quadrature-point scale or anisotropic C_0 through operator stencils, and the
   one-point hexahedron with hourglass control (muFFTTO's `trilinear_hexahedron_1Q_hg`, `muFFTTO/hourglass_control.py`).
7. *Only after the user's OK:* make muFFTTO's `Multigrid` a thin wrapper of the muGrid class.

## 4. Validation, required

- muGrid unit tests in the style of `tests/python_*_tests.py`:
  - P and R are exact adjoints, and P reproduces linear fields;
  - in 2D, the triangle coarse operator equals PᵀAP;
  - the cycle is symmetric and positive definite;
  - CG iterations are grid-independent on a phase-field geometry;
  - 1/2/4 MPI ranks give the same iteration counts (`mpiexec`);
  - CPU and GPU give the same results;
  - float32 vs float64 iteration difference is reported.
- **Cross-check against the muFFTTO reference:** same geometry (e.g.
  `/home/martin/PycharmProjects/muFFTTO/experiments/Multigrid_preconditioning/data/CH3D/ch3d_ch_s12.npy` with a tanh
  interface, or a seeded smooth random field), contrast 1e3-1e4, same parameters.
  - CG iteration counts must be the same; the reference's `power_start` hook reproduces its power iteration.
  - The homogenised stress must agree to round-off.
- **Report:** time per V-cycle, CG time and GPU memory per voxel at 64³/128³ (and 256³ if it fits) vs the muFFTTO
  implementation, measured per process with `nvidia-smi --query-compute-apps`.

## 5. Workspace and rules

**Your workspace: a new copy of muGrid**
- Clone muGrid fresh from GitHub. The repo is public; use https, or ssh if https fails:
  ```
  git clone https://github.com/muSpectre/muGrid.git /home/martin/PycharmProjects/muGrid_mg
  cd /home/martin/PycharmProjects/muGrid_mg && git checkout -b feat/heterogeneous-multigrid
  ```
- Work only in `/home/martin/PycharmProjects/muGrid_mg`. Commit on your branch. Do not push or open PRs unless the
  user asks.
- **Do not touch:**
  - the user's old checkout `/home/martin/PycharmProjects/muGrid` (unpushed work), which you may read;
  - the muFFTTO repos (`muFFTTO`, `muFFTTO_fable`), which you only read.

**Your own Python environment for the new muGrid**
- The compilers and CUDA come from the micromamba env `muffto-gpu` (conda gcc 14, CUDA 12.9, OpenMPI, PnetCDF,
  CuPy, muFFTTO's dependencies):
  `eval "$(~/.local/bin/micromamba shell hook -s bash)"; micromamba activate muffto-gpu`.
- **Never `pip install` anything into `muffto-gpu` itself.** muFFTTO's production code and other sessions use its
  muGrid (an older 1.4.1-based build).
- Make a venv on top of it that sees the env's packages but has its own muGrid:
  ```
  python -m venv --system-site-packages ~/venvs/muGrid_mg
  source ~/venvs/muGrid_mg/bin/activate          # after activating muffto-gpu
  cd /home/martin/PycharmProjects/muGrid_mg
  pip install --no-build-isolation --no-deps -e . -Cbuild-dir=build-cuda \
      -Ccmake.define.MUGRID_ENABLE_CUDA=ON -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=120 \
      -Ccmake.define.MUGRID_ENABLE_MPI=ON -Ccmake.define.CMAKE_BUILD_TYPE=Release
  python -c "import muGrid; print(muGrid.__file__, muGrid.__version__)"   # must point into muGrid_mg
  ```
  If the editable install misbehaves, fall back to a plain cmake build and `runtests.py` with `PYTHONPATH`
  (`docs/multigrid_preconditioner_plan.md` §0.3 describes this; it was written on another machine, so adapt its paths).
- Run muGrid's test suite once **before** changing anything, so you know the baseline.
- The muFFTTO reference (`muFFTTO_fable`) imports muGrid too. Run the cross-checks in this venv, so muFFTTO and the
  new class use the same new muGrid. If muFFTTO does not work with muGrid 1.6.x, report what breaks and run the
  reference in `muffto-gpu` instead. Do not edit muFFTTO.

**GPU and MPI**
- GPU: RTX PRO 1000 (Blackwell, sm_120, 8 GB). It is **shared** with other sessions, so check `nvidia-smi` and stay
  ≤ 3 GB.
- The conda OpenMPI is **not CUDA-aware**: never pass device buffers to MPI.

**Running things**
- Run builds, long tests and benchmarks with `systemd-run --user --unit=mgport-<name> -p MemoryMax=24G
  -p MemorySwapMax=0 --working-directory=<dir> bash <script>`, never `&`/nohup. An OOM kill takes down the whole
  terminal scope with all sessions. The machine has 16 cores and 60 GB RAM; build with `-j8`.
- Never `pkill -f`/`pgrep -f` a pattern that appears in your own command line. Kill by PID.
- If muFFTTO is imported: `JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1`,
  `CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux`.
- No NEMO cluster use unless the user asks.

**User preferences**
- Small tests first (32³-128³).
- Figures: plain tick labels, font ≥ 12, open each with `xdg-open`.
- Keep `docs/heterogeneous_multigrid.md` with a "Status / RESUME HERE" section updated after every milestone, so the
  work survives interruptions.
- End with a short report: what is in muGrid, the validation results, the speed and memory vs the muFFTTO version,
  and what is open.
