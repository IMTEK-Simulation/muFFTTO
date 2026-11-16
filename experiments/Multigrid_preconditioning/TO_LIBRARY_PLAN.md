# Plan: move our best topology-optimisation setup into muGrid, NuMPI and muTopOpt

2026-11-16. Inventory of everything the best 3D TO runs (256³-1024³, `muFFTTO_fable/.../to3d_large.py`) needed, and where
each piece belongs. Status: ✔ exists there, ◐ prototype in muFFTTO/experiments, ✘ missing.

## muGrid — solver building blocks

| # | Piece | Where it is now | Status in muGrid |
|---|---|---|---|
| G1 | Heterogeneous multigrid preconditioner: element-free (Galerkin), guaranteed λmax bound, GPU, float32, MPI | `muFFTTO_fable/muFFTTO/multigrid.py` | ✘ (port running in a separate session, `MUGRID_PORT_BRIEF.md`) |
| G2 | One-point hexahedron + hourglass control, ideally inside the fused stiffness kernel | `muFFTTO/hourglass_control.py` | ✘ (muGrid has the fused 8-point `IsotropicStiffnessOperator`) |
| G3 | CG with the energy-error stopping test (lower estimate, adaptive delay) | muFFTTO `conjugate_gradients_mugrid_experimental`, Fable's `cg_energy.py` | ✘ (`Solvers.conjugate_gradients` stops on the residual) |
| G4 | GPU kernels: grid transfer, scaled-material and Jacobi pointwise products; fewer ghost exchanges per K | Fable's CuPy kernels | part of G1 |
| G5 | float32 fields and fused operator | — | ✔ (check linalg coverage) |

## NuMPI — the optimiser

| # | Piece | Where it is now | Status |
|---|---|---|---|
| N1 | Bounded L-BFGS with a preconditioned initial inverse Hessian (`precond=`, H0 = P M P) | NuMPI `feat/tr_precond` commit 6561d66 (local + NEMO, **not pushed**) | ◐ → PR |
| N2 | Restart fix: tiny first step after a restart, then a ~5000x too large step (carry γ across restarts or cautious pair test) | finding in `muFFTTO_fable/.../HANDOFF.md` | ✘ (test experiment-locally first) |
| N3 | Array-module agnostic L-BFGS (numpy or CuPy, device buffers, fixed memory) — host vector work was ~2500 s per iteration at 1024³ | Fable's `lbfgs_xp.py` | ✘ |
| N4 | MPI reductions with device arrays on non-CUDA-aware MPI (reduce on device, then host Allreduce) | Fable's driver | ✘ (with N3) |

## muTopOpt — the TO formulation and driver

| # | Piece | Where it is now | Status |
|---|---|---|---|
| T1 | Phase-field regulariser: normalised weight 0.2, interface width η = 2h, double well | muFFTTO `topology_optimization.py` | check `regularization.py` |
| T2 | Energy objective (match target stiffness energies) and auxetic target with TARGET_SCALE; stress objective | study copies (2D/3D) | ◐ (muTopOpt: stress objective) |
| T3 | Helmholtz preconditioner for L-BFGS, H0 = (I + c η² (−Δ))⁻¹ by muGrid FFT, c = 4 | `design_optimizers.py`, NuMPI N1 | ✘ |
| T4 | Option: multigrid preconditioner (G1) instead of Green-Jacobi; one-point hourglass element (G2) | Fable's driver | ✘ |
| T5 | Adjoint sensitivity incl. the hourglass term (FD-checked to 1e-8) | muFFTTO `topology_optimization.py` | ✘ (with G2) |
| T6 | Coarse-to-fine chain (start from the coarser converged design) and Cahn-Hilliard starts | study scripts | ◐ (`restart.py`, `fourier_resample`) |
| T7 | Memory diet: warm starts on host or in float32, one load case on the GPU at a time (512³ on one H200 at ~0.9 KiB/voxel) | Fable's driver | ✘ |
| T8 | Multi-GPU MPI run on one 8×H200 node, checkpoint/restart within the 2-day limit | Fable's sbatch scripts | ◐ |
| T9 | Optional: cubic design symmetry (2 load cases) — numerically correct, but gave non-auxetic designs | Fable's `cubic_symmetry.py` | optional |

## Order

1. **N1 → NuMPI PR** (small, ready, tested). Then N2 with a test.
2. **muGrid**: G1 (running), G3, then G2.
3. **NuMPI N3/N4** (needed only from 512³ on).
4. **muTopOpt**: T1-T3 first (they change the designs), then T4-T5 (speed), T6-T8 (scale).
5. Validation at every step: reproduce a 64³ run of the study (same objective history, same CG counts), then 256³ timing.
