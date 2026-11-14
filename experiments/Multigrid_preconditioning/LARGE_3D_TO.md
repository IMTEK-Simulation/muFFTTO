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
Stop: largest change of ρ in an iteration below 10⁻³, or `MAXITER` (500).

This is the only place where an FFT is used, and only in single-GPU runs.

## 6. Grid continuation

A large run never starts from noise. Each grid starts from the converged design of the next coarser grid,
interpolated trilinearly (`INIT_FROM`): 128³ → 256³ → 512³ → 1024³, the first from the 3D Cahn–Hilliard pattern
`data/CH3D/CH3D_start_bi_s11.nc`. The interface width stays 2 voxels on every grid, so it halves in physical units
from grid to grid.

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
