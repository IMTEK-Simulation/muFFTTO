# Brief: energy-error CG stopping test and one-point hexahedron with hourglass control in muGrid

Written 2026-11-16 for a new Claude session. The user (Martin) wants two pieces of our topology-optimisation setup
in **muGrid**: clear, reproducible, human-readable and fast. Do A first, then B, each on its own branch. Read the
whole brief before starting. Keep your messages to the user short.

## A. CG with the energy-error stopping test

**What.** CG stops when an estimate of the energy-norm error ‖x − x_k‖_A falls below `tol` relative to the solution's
energy, not on the residual. This is the stopping rule of all our studies: rtol 1e-6, adaptive delay.

**References** (read them; the iterates and iteration counts must match):
- `/home/martin/PycharmProjects/muFFTTO/muFFTTO/solvers.py`: `conjugate_gradients_mugrid_experimental` with
  `stop_crit_norm='energy_lower_estim'`, `energy_rtol=True`, `d_min`, `tau=0.25`; the delay rule `findS`.
- `/home/martin/PycharmProjects/muFFTTO/muFFTTO/topology_optimization.py`: `energy_stop_kwargs` (the reference
  energy used as the denominator).
- `/home/martin/PycharmProjects/muFFTTO_fable/experiments/Multigrid_preconditioning/cg_energy.py`: lean version, same
  iterates. It uses muGrid `linalg` kernels, no temporaries, and one K product and one reduction less per solve.
  This is the speed target.
- Method:
  - Strakoš & Tichý (2002): lower estimate Σ_{j=k}^{k+d−1} α_j (r_j, z_j) of ‖x − x_k‖²_A.
  - Meurant, Papež & Tichý (2021): adaptive choice of the delay d (parameter τ = 0.25), with a minimal delay `d_min`.

**Where in muGrid.** `language_bindings/python/muGrid/Solvers.py` has `conjugate_gradients` (residual test).
- Propose in a short design note how to add the energy test without cluttering CG. Options: a stopping-criterion
  argument or object, or a separate function. Show the note to the user before coding.
- `d_min` must be a parameter. Our studies use 0 (multigrid) and 10 (elsewhere); ask the user for the default.

**Validation**
- Same iterates and iteration counts as the muFFTTO reference on the same SPD problem and preconditioner.
- On problems with a known solution, the estimate is ≤ the true error.
- MPI: only global reductions, same counts on 1/2/4 ranks. Also GPU, float32 and float64.
- No extra operator products or temporaries compared with `cg_energy.py`.

## B. One-point hexahedron with hourglass control

**What.** A trilinear hexahedron integrated at one point (the voxel centre), so all quadrature fields are 8x smaller,
plus Flanagan-Belytschko hourglass control against its zero-energy modes:
- `K = B₁ᵀ w C B₁ + Hᵀ (ε κ) H`.
- H: the 4 hourglass base vectors (ηζ, ζξ, ξη, ξηζ at the corners) / 8, applied per displacement component.
- κ_{iα} = Σ_{d,e} M_α[d,e] C_{idie}: the exact 2×2×2-Gauss energy of the unit mode on one voxel, so ε = 1 reproduces
  the 8-point hourglass energy mode by mode, on any grid.
- Default ε = 0.3. Measured: multigrid iterations equal to the 8-point element; homogenised stiffness within 0.2 % for
  phase-field interfaces of width ≥ h; sharp voxel interfaces −8 % (64³) to −14 % (32³), shrinking with h.

**Reference.** `/home/martin/PycharmProjects/muFFTTO/muFFTTO/hourglass_control.py`; the docstring is the
specification. Also read:
- the element `'trilinear_hexahedron_1Q_hg'` in `discretization_library.py`;
- the hook at the end of `Discretization.apply_system_matrix_mugrid` in `domain.py`;
- the TO sensitivity term `design_derivative`;
- tests in `test/test_hourglass_control.py`;
- results in `experiments/Multigrid_preconditioning/CLAUDE_NOTES.md` (search "hourglass").

**What to build in muGrid, in this order**
1. **Stencil version for any material**: one-point quadrature for the q1 hexahedron gradient operator (check what
   `FEMGradientOperator` supports), the hourglass operator H (a `GenericLinearOperator` stencil works; the reference
   does that), and κ from the material (per voxel, any anisotropic C). Plus the design derivative of the hourglass
   energy for topology optimisation.
2. **Measure before building the fused kernel.** Our production runs use muGrid's fused `IsotropicStiffnessOperator3D`
   (8-point, one isotropic material per voxel). It is exact, has no quadrature fields, and its memory is already as
   low as a one-point element. A fused one-point + hourglass kernel only pays off if it is clearly faster: about 4x
   fewer flops, and the 8-point kernel may be compute-bound in float64.
   - Estimate and measure on the laptop GPU at 128³, in double and single.
   - Report to the user and ask before building it.
   - If built: per-voxel λ, μ (κ ∝ μ, λ), `apply` and `assemble_diagonal` (the multigrid needs it), CPU and GPU,
     float32 and float64, ghost handling as the existing fused operator.
3. *Optional, ask first:* the 2D analogue (bilinear quadrilateral, one hourglass mode ξη per component).

**Validation**
- Port the reference tests:
  - zero on linear fields;
  - symmetric positive semidefinite;
  - ε = 1 reproduces the 8-point hourglass energy per mode (rel 1e-12);
  - the design derivative matches finite differences.
- K u equal to muFFTTO's implementation to round-off on the same grid and material.
- Fused vs stencil equality; CPU vs GPU; float32; MPI ghosts.
- Speed and memory per voxel vs the fused 8-point operator.

## Workspace and rules

**Your workspace**
- Your own **new** muGrid clone. Do not use `/home/martin/PycharmProjects/muGrid` (user's work), `muGrid_mg`
  (another session porting the multigrid) or `muGrid_precond`.
  ```
  git clone https://github.com/muSpectre/muGrid.git /home/martin/PycharmProjects/muGrid_cghg
  cd /home/martin/PycharmProjects/muGrid_cghg
  git checkout -b feat/cg-energy-stop        # part A; part B: git checkout -b feat/one-point-hexahedron-hourglass main
  ```
- Commit on your branches. Do not push or open PRs unless the user asks. Read the muFFTTO repos (`muFFTTO`,
  `muFFTTO_fable`) only; never edit them.

**Python environment**
- Your own venv on top of the micromamba env `muffto-gpu`:
  ```
  eval "$(~/.local/bin/micromamba shell hook -s bash)"; micromamba activate muffto-gpu
  python -m venv --system-site-packages ~/venvs/muGrid_cghg && source ~/venvs/muGrid_cghg/bin/activate
  pip install --no-build-isolation --no-deps -e . -Cbuild-dir=build-cuda \
      -Ccmake.define.MUGRID_ENABLE_CUDA=ON -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=120 \
      -Ccmake.define.MUGRID_ENABLE_MPI=ON -Ccmake.define.CMAKE_BUILD_TYPE=Release
  python -c "import muGrid; print(muGrid.__file__)"      # must point into muGrid_cghg
  ```
- **Never `pip install` into `muffto-gpu` itself.** Production code and other sessions use it.
- Run muGrid's test suite once before changing anything, as the baseline.
- To run the muFFTTO references, use this venv if muFFTTO works with the new muGrid; otherwise use `muffto-gpu` and
  report what breaks. With muFFTTO: `JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1
  CUDA_PATH=$HOME/micromamba/envs/muffto-gpu/targets/x86_64-linux`.

**Machine**
- One 8 GB GPU (sm_120), **shared** with other sessions: check `nvidia-smi` and stay ≤ 2 GB. 16 cores, 60 GB RAM.
- The conda OpenMPI is not CUDA-aware: never pass device buffers to MPI.
- Builds and long runs go through `systemd-run --user --unit=cghg-<name> -p MemoryMax=20G -p MemorySwapMax=0
  --working-directory=<dir> bash <script>`, never `&`/nohup. An OOM kill takes down all sessions. Build with `-j6`.
- Never `pkill -f`/`pgrep -f` a pattern in your own command line; kill by PID.
- No NEMO cluster use.

**Notes**
- Keep `docs/cg_energy_and_hourglass.md` with a "Status / RESUME HERE" section updated after every milestone.
- End with a short report: what is in muGrid, validation, speed vs references, open points.
