# Large 3D topology optimisation: how the procedure works

Written 2026-11-14. This describes the optimisation procedure of `to3d_large.py` (the driver for 256³ – 1024³ runs)
from the problem statement down to the GPU kernels, and how to run it. Measurements, job numbers and the history of
the work are in `FABLE_NOTES.md`; the task description is `FABLE_BRIEF.md`.

## 1. The problem

A periodic unit cell [0, 1]³ on a grid of N³ voxels. The design variable is a **nodal phase field** ρ ∈ [0, 1]
(one value per grid node: 1 = solid, 0 = void).

**Material.** At every quadrature point the stiffness is a scalar times the solid tensor,

    A(x) = s(x) C_0,      s = ρ_q² (1 − k) + k,      k = 10⁻³  (void stiffness, `SOFT_EXP=3`)

with ρ_q the phase field interpolated to the quadrature point and C_0 the isotropic solid (K_0 = 1, G_0 = 0.5).
Only the scalar field s is stored, never a full tensor per point.

**Element.** `trilinear_hexahedron_1Q_hg`: one quadrature point per voxel with Flanagan–Belytschko hourglass
control of strength 0.3 (`muFFTTO/hourglass_control.py`). The stiffness is K = Bᵀ W s C_0 B + Hᵀ (ε κ s) H.

**Objective.**

    f(ρ) = Σ_load cases  Σ_k c_k ((W*_k − W_k) / W*_k)²          stiffness misfit, 9 energy terms
         + w_pf / (c_W V^(2/3)) · [ η ∫|∇ρ|² + (1/η) ∫ρ²(1−ρ)² ]     phase-field interface penalty

- W_k = E_k : C_h : E are energies of the homogenised stiffness C_h; the nine terms are C1111, C1122, C1133 (load
  case xx), C2222, C2233 (yy), C3333 (zz) and the shears C2323, C1313, C1212.
- W*_k are the same energies of the **target**: an isotropic material with Poisson ratio ν* = −0.5 and shear
  modulus G* = `TARGET_SCALE` · 0.15 · E_0 (scale 0.3: G* = E* = 0.0579, K* = 0.0096, C1111* = 0.0868,
  C1122* = −0.0289).
- Interface penalty: weight w_pf = 0.2, interface width η = 2 voxels (`ETA_H=2`), c_W = 1/3.

## 2. One evaluation of f and its gradient

`TopologyOptimization.evaluate(ρ)` in `to3d_large.py`:

1. **Material.** Interpolate ρ to the quadrature points, s = ρ_q²(1 − k) + k.
2. **Multigrid setup** for this material (section 4).
3. **Phase-field term** and its gradient (gradient energy with the element quadrature, double well at the nodes).
4. **For each of the 6 load cases** E (xx, yy, zz, yz, xz, xy):
   - mechanical problem K u = −Bᵀ W s C_0 : E, solved by CG, started from the u of the previous evaluation;
   - homogenised stress Σ_h = (1/V) Σ w s C_0 : (E + ∇u) and the load case's misfit terms;
   - adjoint problem K λ = (2/V) Bᵀ W s C_0 : D, with D built from the misfits; also warm started;
   - sensitivity at the quadrature points: (∇λ − (2/V) D) : C_0 : (E + ∇u), plus the hourglass part.
5. **Chain rule**: multiply the accumulated quadrature sensitivity by ds/dρ_q = 2 ρ_q (1 − k) and bring it to the
   nodes with the transposed interpolation.

So one evaluation costs 12 linear solves. The value f also contains the term λ·(K u − rhs) of every load case, zero
for an exact solve; it keeps f consistent with its gradient (as in the reference script).

## 3. The linear solver

`cg_energy.py`: preconditioned conjugate gradients, **stopped on the energy error**: the Meurant–Papež–Tichý lower
estimate of ‖u − u_k‖_K with adaptive delay, compared with 10⁻⁶ ‖u‖_K (for a warm start ‖u‖_K is taken from the
start vector). This is the algorithm of the library's `conjugate_gradients_mugrid_experimental`; the local copy
only replaces array expressions by muGrid `linalg` kernels (`axpy`, `axpby`, `vecdot`) so that no temporaries are
created. Typical cost: 15 iterations from a cold start, 3–12 with a warm start.

## 4. The preconditioner: geometric multigrid

`muFFTTO/multigrid.py`, a symmetric V(2,2)-cycle:

- **Levels** N, N/2, …, 4. Every level is a muFFTTO `Discretization`; the fine level is the TO's own
  discretization and shares its scratch fields.
- **Coarse operators** by re-discretisation: fully integrated (8-point) hexahedra whose Gauss point in octant
  (i, j, k) of a coarse voxel takes the mean scale of the fine voxel in that octant.
- **Transfers**: trilinear interpolation P and R = Pᵀ (`muGrid.GridTransfer` on the host, one CuPy kernel on the GPU).
- **Smoother**: 2 Chebyshev steps in D⁻¹K on [λ_max/30, λ_max]. The diagonal D comes from Dirac-comb products,
  λ_max from a power iteration (40 steps the first time, 15 warm-started steps after each design change).
- **Coarsest level** 4³: exact, dense pseudo-inverse.

One CG iteration costs 5 fine-level products with K (1 in CG, 4 in the cycle) plus 4 on each coarser level.
The iteration count does not grow with N.

## 5. The design update

Bound-constrained L-BFGS (NuMPI `l_bfgs_bounded`, memory 20, bounds [0, 1], projected Armijo line search) with the
**Helmholtz initial inverse Hessian** H_0 = (I + c η² (−Δ_h))⁻¹, c = 4, Δ_h the 7-point Laplacian. On one GPU H_0 is
applied by FFT; on several GPUs by CG on the stencil to a relative residual of 10⁻¹⁰ (same result to 2·10⁻¹²).
Stop: largest change of ρ in an iteration below 10⁻³, or `MAXITER` (500), or one of the optional tests of
section 13.

This is the only place where an FFT is used, and only in single-GPU runs.

## 6. Grid continuation

A large run never starts from noise. Each grid starts from the converged design of the next coarser grid,
interpolated trilinearly (`INIT_FROM`): 128³ → 256³ → 512³ → 1024³, the first from the 3D Cahn–Hilliard pattern
`data/CH3D/CH3D_start_bi_s11.nc`. The interface width stays 2 voxels on every grid, so it halves in physical units
from grid to grid. What that means for the comparability of grids is in section 15.

## 7. What makes it fit and run fast

**Memory** (about 900 bytes per voxel on the GPU, all-in; the previous script needed about 2700):

- one set of fields, each named after its role and reused: u and λ of the load case being solved, one right-hand
  side, 4 CG vectors, 5 multigrid vectors, two gradient-sized scratch fields, the hourglass strain, a few scalars;
- the solutions of all load cases (the warm starts) live in page-locked host memory and are streamed to the GPU
  one load case at a time (`WARM_STORE=host`);
- no stored tensors, no constant fields, no array temporaries in the solver.

**Speed** (one evaluation at 256³ on an H200: 62 s; previous script 120 s):

- pointwise operations (material law, hourglass scaling, Jacobi product) run as single CuPy kernels over the
  contiguous buffer of a field (`tensor_operations.point_view`), not over strided views;
- K exchanges ghost layers three times (displacement, stress, hourglass strain) instead of six or seven;
- the stencils themselves (gradient, divergence, hourglass operator) are muGrid `GenericLinearOperator`s.

**Several GPUs** (one MPI rank per GPU, the grid cut into slabs along z):

- every multigrid level is distributed on the same slabs, so the transfers are local up to a ghost exchange;
- levels of 16³ and coarser are gathered on rank 0 and solved there (on small grids an exchange is pure latency);
- the power-iteration start vectors are seeded per grid plane, so a run gives the same iteration counts on any
  number of ranks.

## 8. Optional: fused stiffness kernel and single-precision multigrid

`OPERATOR=fused` replaces the element. The mechanics then use trilinear hexahedra with **one material value per
voxel**, applied by muGrid's fused `IsotropicStiffnessOperator` (the kernel muTopOpt uses). This is muFFTTO's
8-point `trilinear_hexahedron` with voxel-wise constant material -- verified to round-off for K u, the diagonal,
the right-hand side and the average stress (`fable_fused_check.py`, `test/test_multigrid.py`) -- but not the
one-point element of section 1, so objective values differ by a few per cent (2·10⁻³ in a test at 8³).

- Every multigrid level uses the kernel; a coarse voxel takes the mean scale of its 8 fine voxels. The Jacobi
  diagonal comes from the kernel in one pass (instead of 24 operator products).
- Right-hand sides, the average stress and the sensitivity contractions also come from the kernel, so there are
  no gradient or hourglass fields at all: about 530 bytes per voxel on the GPU instead of about 900.
- `MG_PRECISION=single` runs the multigrid cycle in float32 on its own fields, between the double-precision
  vectors of CG. A preconditioner need not be exact: the solutions, the stopping test and the objective stay in
  double precision, and the iteration counts do not change. The design fields and the optimiser are untouched.

Measured (256³, one L40S, 7 evaluations): one-point element 891 s and 15.0 GiB; fused kernel 515 s and 9.1 GiB;
fused kernel with float32 multigrid 155 s and 8.5 GiB. CG iterations +7 % against the one-point element.

## 9. Optional: design symmetry

`SYMMETRY=cubic` (48 symmetries of the cube) or `cubic_chiral` (its 24 rotations) projects the start design and
every gradient onto symmetric designs (`cubic_symmetry.py`). The homogenised stiffness is then cubic and two load
cases (xx and xy) give the whole objective: a third of the work. It is verified numerically, but for the auxetic
target above every start tried so far converged to a non-auxetic design (ν ≈ +0.05). Not used in production.

## 10. Files

| file | role |
|---|---|
| `to3d_large.py` | the driver: objective, gradient, warm starts, optimiser call, checkpoints, output |
| `cg_energy.py` | energy-stopped PCG on muGrid linalg kernels |
| `muFFTTO/multigrid.py` | the multigrid preconditioner (tests: `test/test_multigrid.py`) |
| `cubic_symmetry.py` | symmetry projections, serial and distributed |
| `sbatch_TO3D_large.sh` | one job segment on NEMO (continues from its checkpoint) |
| `submit_TO3D_large_chain.sh` | chain of segments and grids with job dependencies |
| `fable_run_TO.sh`, `fable_run_TO_mpi.sh` | short test runs with memory sampling (1 GPU / several) |
| `fable_check_to3d_large.py`, `fable_check_symmetry.py`, `fable_test_multigrid.py`, `fable_compare.py` | validation against the reference script, `mg_lib` and the baseline histories in `fable_reference/` |
| `fable_memory.py`, `fable_time_K.py`, `fable_time_ghosts.py` | memory table, timing of K and of ghost exchanges |

## 11. Running it

Settings are environment variables; the full list is in the docstring of `to3d_large.py`.

Laptop, one GPU, 32³ for 20 iterations:

    SCRIPT=to3d_large.py ./fable_run_TO.sh mytest 32 20        # log: logs/fable/mytest_N32.log

NEMO, a chain over grids (run on the login node in the clone; `DRY_RUN=1` only prints the sbatch commands):

    ./fable_sync_nemo.sh                                        # on the laptop: code -> ~/software/muFFTTO_fable
    ./submit_TO3D_large_chain.sh <label> <start.nc> "128 1 1 04:00:00 32G" "256 1 2 12:00:00 64G" "512 2 2 24:00:00 200G"
    #                                                 grid  GPUs  segments  time limit  host memory

- A segment stops cleanly 25 minutes before its time limit and writes a checkpoint; the next segment continues.
- A run is finished when `<…>_final.nc` exists; the next grid starts from it.
- Logs: `$WS/runs/FABLE/TO3D_N<N>_<label>.log`; data: `$WS/runs/FABLE/data/`.
- Each evaluation prints `EVAL <n> f <value> wall <s> eval <s>`; each load case prints its CG iterations.

Poisson ratios and stiffness of a result (copy the `_final.nc` to the laptop first):

    python ../dual_preconditioner/topology_optimization/report_TO3D_nu.py <…_final.nc>

## 12. How it was checked

- Objective and gradient equal those of `example_3D_elasticity_TO.py` to round-off (4·10⁻¹⁶ / 3·10⁻¹⁶ at 8³), and
  the gradient passes a finite-difference check (10⁻⁸).
- The 32³ optimisation history equals the baseline of the previous code to 2·10⁻⁸ over 21 evaluations, with the
  same CG iteration counts (3655 vs 3657).
- The multigrid equals the study's `mg_lib` to round-off on CPU and GPU.
- 1, 2 and 4 MPI ranks give identical iteration counts and histories on the CPU; on GPUs the first evaluations
  agree with the single-GPU run to all printed digits.
- Fused kernel: gradient passes the finite-difference check (10⁻⁸); double- and single-precision multigrid give the
  same history (8·10⁻¹³) and iteration counts; 1 and 4 MPI ranks agree exactly.
- Unit tests of the library: 205 passed.

## 13. Stopping the design iteration on the energies (added 2026-11-15)

The objective has two parts: the stiffness misfit (squared relative misfits of the homogenised energies) and the
interface penalty. The stiffness settles long before the interfaces are clean, so a test on the energies stops much
earlier than the step test. `ENERGY_TOL=<tol>` stops when every energy of the objective changed by less than tol of
its target over the last 10 iterations (`FTOL_WINDOW`). Every iteration prints this measure (`ITER` line).
Also available: `FTOL` (relative decrease of f over the window), `MISFIT_TOL` (absolute misfit). All off by default;
with all off a run is unchanged.

2D prototype (`stop_criterion_2D.py`, 128², 600 iterations, figure `figures/stop_criterion_2D.png`):

| test | stops after (share of the full run) | state there |
|---|---|---|
| energies within 5-10 % of the target | 4-7 %, or never if the target is out of reach | design still diffuse: not usable |
| energies settled to 10⁻³ | about 20 % | Poisson ratio within 0.005 of final, design 5-10 % from final |
| energies settled to 3·10⁻⁴ | 40-50 % | design within 1-3 % of final |
| step below 10⁻³ (old test) | 45-55 % | - |

Used in production: 10⁻³ on grids that only feed the next grid, 3·10⁻⁴ on the final grid.

## 14. Guard against wild line-search steps (added 2026-11-15)

Observed at 512³: L-BFGS occasionally proposes a direction thousands of times larger than its recent steps (mean
design change 0.13 after accepted steps of 2·10⁻⁵), typically right after a restart from a checkpoint. The trial
design is then close to noise and its linear solves need 130-320 CG iterations instead of about 10, so a single
evaluation takes many minutes. Every evaluation now prints the max and mean change of its trial design
(`TRIAL design` line). Two guards, off by default:

- `STEP_GROWTH=<g>`: reject a trial design whose mean change exceeds g × the largest mean change of the last 10
  accepted steps (needs two accepted steps). Production uses 20.
- `MAX_STEP=<x>`: the same with an absolute limit on the mean change.

A rejected trial is not solved; the objective returns infinity and the line search halves the step.

**Cause (found 2026-11-15).** After a (re)start the L-BFGS memory is empty and the first step uses H₀ = Helmholtz
with scale 1, but the right scale (γ = s·y / y·H₀y) is ~6·10³ at 128³ and grows with the grid. The first step is then
~10⁵ times too small, its curvature pair is CG noise (pair γ 0.22 instead of ~6·10³; cos(s, y) looks normal, so a
cautious-update test does not catch it), and the next directions blow up. 128³ restart test
(`plot_lbfgs_restart.py`, `figures/lbfgs_restart_128.png`): unfixed 4 iterations in 49 evaluations, then stuck;
fixed 15 iterations in 22 evaluations. **Fix: `LBFGS_FIRST_STEP=<x>`** (0.05 in production): H₀ = c × Helmholtz
with c chosen at the first call so that the first step changes the design by at most x. γ makes all later directions
independent of c. Note that `STEP_GROWTH` needs two accepted steps, so it never catches a wild direction in
iterations 2-3.

## 15. Is the result mesh-size independent? (2D check, 2026-11-15)

The interface penalty is normalised so that, once the interface is resolved, it equals w × interface area: its value
does not depend on the grid. What depends on the grid is the interface width η = 2h, which sets the smallest feature.
2D check (`run_mesh_independence_2D.sh`, `plot_mesh_independence_2D.py`; same start, 600 iterations; figures
`figures/mesh_independence_2D.png` and `figures/mesh_independence_2D_interfaces.png`):

| interface width | grid | nodes across | objective | misfit | interface penalty | Poisson ratio |
|---|---|---|---|---|---|---|
| 1/16 fixed | 64² | 4 | 1.131 | 0.266 | 0.866 | −0.216 |
| 1/16 fixed | 128² | 8 | 1.113 | 0.249 | 0.864 | −0.220 |
| 1/16 fixed | 256² | 16 | 1.108 | 0.244 | 0.864 | −0.221 |
| 1/32 fixed | 64² | 2 | 0.873 | 0.035 | 0.839 | −0.270 |
| 1/32 fixed | 128² | 4 | 0.859 | 0.030 | 0.830 | −0.279 |
| 1/32 fixed | 256² | 8 | 0.855 | 0.029 | 0.827 | −0.281 |
| 2h | 32² | 2 | 1.197 | 0.200 | 0.997 | −0.210 |
| 2h | 64² | 2 | 0.873 | 0.035 | 0.839 | −0.270 |
| 2h | 128² | 2 | 0.719 | 0.009 | 0.710 | −0.300 |
| 2h | 256² | 2 | 0.682 | 0.003 | 0.679 | −0.316 |

With a fixed physical η the result is mesh-independent (4 nodes across the interface are enough; 2 are
under-resolved). With η = 2h every grid solves a different problem, and the thinner interface is what lets the
design become more auxetic. In 3D `ETA_H=<η / h>` of `sbatch_TO3D_large.sh` chooses between the two.

## 16. Video of a run through the grids

`render_TO3D_multigrid_video.py` renders the solid of every stored design of a sequence of history / checkpoint
files (different grids allowed), with the objective curve; `sbatch_render_TO3D_multigrid.sh` runs it headless on one
L40S with `~/software/render-venv`. `TEST=1` writes three stills. A smooth video needs a dense history
(`HISTORY_EVERY=2` at 128³ / 256³, 5 at 512³: a 512³ frame is about 1 GB). `HISTORY_EVERY` and `HOLD` (video frames
per design) accept one value per file, e.g. `HISTORY_EVERY=2,2,5 HOLD=3,3,8` (export them before `sbatch`: its
`--export` splits at commas). Result: `figures/video/video_s0.3_smooth.mp4` (289 designs, 128³ → 256³ → 512³).


## 17. What the runs cost: time, memory, energy (2026-11-15)

Sources: Slurm accounting (`sacct -j <jobs> -o JobID,Partition,State,Elapsed,AllocTRES,MaxRSS,NodeList`), the run
logs `$WS/runs/FABLE/TO3D_N<N>_<label>.log` (one `EVAL` line per evaluation with its time, one `load case` line per
solve with the CG counts, the `TOTAL` line at the end) and `logs/fable/peak.txt` (GPU memory of the test runs).
NEMO records no energy (`ConsumedEnergy` is 0), so energy is estimated from GPU-hours.

**Production chain `aux_s0.3[_fused]`** (energy objective, ν* = −0.5, scale 0.3, 6 load cases, η = 2h)

| grid | job | hardware | wall time | iterations | evaluations | CG iterations (mech + adj) | per evaluation | final f | stiffness misfit |
|---|---|---|---|---|---|---|---|---|---|
| 128³ | 8817007 | 1 H200 | 59.5 min | 500 | 545 | 86.9 k | 6.4 s | 1.412 | 0.167 |
| 256³ | 8817470 | 1 L40S | 1 h 39 min | 229 | 290 | 47.0 k | 16.8 s | 1.188 | 0.085 |
| 512³ | 8817830 | 4 L40S | 8 h 45 min | 373 | 470 | 84.4 k | 40.8 s | 1.082 | 0.054 |
| total | | | 11 h 23 min | 1102 | 1305 | 218 k | | | |

GPU time: ~1 H200-h + 36.6 L40S-h. About 16 CG iterations per solve on every grid (1024³ too). At 512³, 61 % of the
wall time is evaluations and 39 % is outside them, mostly NuMPI's L-BFGS vector work on the host (section 18).

**Memory**

| grid | GPU memory (≈ 900 B per voxel all-in) | host memory per rank (MaxRSS) | design file |
|---|---|---|---|
| 128³ | 1.9 GB | 3.9 GB | 17 MB |
| 256³ | 15 GB | 18.6 GB | 134 MB |
| 512³ | 121 GB (4 × 29.4 GiB on L40S; 112 GiB on one H200) | ≤ 77 GB (4 ranks) | 1.07 GB |
| 1024³ | ≈ 970 GB (8 × ≈ 121 GB on H200, estimate) | 69 GB after one iteration (test 8817512); + 43 GB per rank for 20 L-BFGS pairs on the host | 8.6 GB |

**Energy** (GPUs only, at their rated power H200 700 W / L40S 350 W, i.e. an upper bound; a realistic load of ~60 %
gives about 60 % of it; CPUs and node not included): chain ≤ 13.5 kWh.

**The run shown in the video** (`video_s0.3`: 128³ → 256³ → 512³, dense history, `ENERGY_TOL=1e-3` at 256³ / 512³,
NuMPI L-BFGS; `figures/video/video_s0.3_smooth.mp4`)

| stage | job | hardware | wall time | iterations | evaluations | CG iterations (mech + adj) | final f | misfit | host memory per rank |
|---|---|---|---|---|---|---|---|---|---|
| 128³ | 8820129 | 1 L40S | 24 min | 500 | 556 | 96.1 k | 1.501 | 0.204 | 7.5 GB |
| 256³ | 8820130 | 1 L40S | 2 h 48 min | 54 | 68 | 106.4 k | 1.261 | 0.090 | 19.4 GB |
| 512³ | 8820131 | 4 L40S | 2 h 20 min | 56 | 87 | 20.2 k | 1.171 | 0.050 | 46.6 GB |
| render | 8826297 | 1 L40S | 11 min | | | | | | |
| total | | | 5 h 43 min | 610 | 711 | 223 k | | | |

12.7 L40S-h, ≤ 4.4 kWh GPU energy at rated power (≈ 2.7 kWh at 60 % load). Waste in it: at 256³ ONE wild trial design
(evaluation 13, iteration 4) took 8678 s = 2 h 25 min of the 2 h 48 min (its solves needed thousands of CG
iterations; no `LBFGS_FIRST_STEP` in that run); at 512³ two wild trials took 222 s and 296 s, and ~40 % of the wall
time is NuMPI's host-side vector work. Without the stall and with `lbfgs_xp` on the GPU the same video run would take
roughly 2 h 15 min and ~6 L40S-h.

**Other runs of 15 Nov**: 512³ restarts lost to wild steps 7.7 L40S-h; cancelled 256³ attempts 2.3 L40S-h; 1024³
full-node test 8817512 (timed out, section 18) 12 H200-h ≈ 8 kWh at rated power; dense-history video rerun 128³ /
256³ / 512³ (0.4 + 2.8 + 9.3 L40S-h, the 256³ stage includes a 1.5 h wild-step stall) and renders 0.3 L40S-h;
512³ directly from the Cahn-Hilliard start: ≈ 56 L40S-h and ≈ 14 h wall by 20:00, at f 1.37 / misfit 0.086, against
f 1.082 / misfit 0.054 of the chain after 11.4 h: grid continuation is clearly cheaper.

## 18. L-BFGS vector work on the GPU: `lbfgs_xp.py` (2026-11-15)

Between two evaluations, after every accepted step, the 512³ runs on 4 L40S spend ~37 s (0 s after a rejected trial):
NuMPI's L-BFGS works on host numpy arrays and allocates a temporary for every inner product (laptop, 33.5 M values =
512³ / 4 ranks: ~11 s per iteration with a full memory). At 1024³ on 8 H200 (4× the data per rank, 8 ranks on one
node's memory bandwidth, host side of the H200 nodes erratically slow) the first iteration took ~2500 s and the
evaluation after it had a 303 s material setup; the full-node test timed out.

`lbfgs_xp.py` is NuMPI's `l_bfgs_bounded` with `precond`, the same algorithm, for numpy (CPU, MPI reductions) or cupy
arrays (everything on the GPU; only scalars go over PCIe / MPI). All vectors are allocated once (on the GPU as muGrid
fields of a `GlobalFieldCollection([n_local], device='cuda')`), updates are in place, dots use BLAS / cuBLAS:
fixed memory, nothing in the cupy pool. `LBFGS_PAIRS=host` keeps the 2 × 20 pair vectors in pinned host memory and
streams them (for 1024³, where they do not fit on the H200 next to the solver).

    DESIGN_OPT=lbfgs_xp LBFGS_DEVICE=gpu [LBFGS_PAIRS=host]

Checks: with `reduction='sum'` (NuMPI's rounding) it reproduces NuMPI bitwise (synthetic box problem, 465
evaluations); in the fast mode it agrees to 12 digits until a near-tie (slope ≈ 0) where round-off decides. 64³ TO:
NuMPI, CPU, GPU (muGrid fields) and GPU with host pairs give the same objective at iterations 1/10/20/30 and the same
CG counts. Vector work per iteration at 33.5 M values (laptop): NuMPI ~11 s, CPU ~5 s, GPU ~0.3 s (pairs on the
device) / ~0.9 s (host pairs). 512³ check on 4 L40S: job 8826405 (pending at the time of writing).
