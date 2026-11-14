# Pressurised hole / hollow bar / channel — what we did

Study of Green-type preconditioners (pure Green, C_ref = solid tangent) for finite-strain
neo-Hookean solids with a pressurised zero-stiffness cavity and clamped faces, on a regular voxel grid
(no adapted grid, no regularisation, no contact). Trust-region Newton-CG, CG stops on the energy
estimate (rtol 1e-6, delay 10). Plans and open ideas: see `TODO.md`.

## Model

- Material: W = mu/2 (J^(-2/3) I1 - 3) + kappa/2 ln^2 J (2D plane strain and 3D), E = 100.
- Gas pressure: follower energy -∫ p(X) J dX over the gas region (cofactor derivative), or dead load
  (f0 = dV/du at u = 0, right-hand side only).
- Cavity and outer void: zero stiffness; nodes touching only void are pinned.
- Clamps: full node lines (2D) / planes (3D) at the left and right faces, running through the void.
- Void rule: the outer void has a fixed physical thickness VOID_T (not a fixed number of pixels).
  VOID_T = 1/16 is sufficient (void sweep: small effect).

## Preconditioners

- `G` projected Green: Q M Q (Green with the clamped dofs projected out).
- `G_corr` corrected Green, dense capacitance system [S B; B^T 0], S = E^T M E. Correct only the real
  clamp dofs (`fixed & ~void_node`); including void nodes in S made it worse.
- `G_wall` corrected Green with the exact wall FFT: full-plane clamp through the void, so S is
  translation-invariant along the wall: 1D FFT in 2D (4x4 symbol per wave number), 2D FFT in 3D
  (6x6 symbol, 9x9 at k = 0). No dense solve; feasible in 3D.

## Scripts

| script | what |
|---|---|
| `pressurized_hole.py` | 2D plate with a pressurised hole, free or clamped (CLAMP=1), PRECOND G_corr / G_wall / G / GJ / D |
| `bar3d_geometry.py` | voxel geometry of the 3D hollow bar (sealed by caps) or periodic channel |
| `pressurized_bar_3D.py` | 3D bar / channel solver: matrix-free Hessian, GPU, wall FFT, pressure in space and time, muEye output |
| `bar3d_video.py`, `display_fill.py` | video (3 sections + p-volume / pressure-profile panel); harmonic fill of cavity nodes for display |
| `bar3d_to_mueye.py` | state.npz -> muEye NetCDF (runs now write `bar3d_mueye.nc` directly) |
| `plot_*.py` | figures (see below) |
| `run_*.sh`, `sbatch_bar3d_gpu.sh` | laptop runs / NEMO L40S job |

Main options of `pressurized_bar_3D.py` (env): NNN, P_MAX, NB_INC, PRECOND (G_wall | G), DEVICE,
MATRIX_FREE, PERIODIC, BAR_L, P_VAR, P_OFFSET, LOAD (follower | dead), WAVE, PULSE, PULSE_LEN,
PULSE_TRAVEL, NB_T, MUEYE_NC, OUTPUT_EVERY, QBLOCK (1 = quadrature-point blocking, needed at 1/128).

## Results

Mean Hessian products per Newton solve.

2D clamped plate, hole, VOID_T = 1/16 (h = 1/32, 1/64, 1/128):
corrected 43 / 46 / 50, projected 55 / 70 / 92. Free plate limit pressure p ≈ 17.5.

3D, x faces clamped (figure `bar3d_scaling.png`):

| case | precond | 1/32 | 1/64 | 1/128 |
|---|---|---|---|---|
| sealed bar, p 0 -> 12 | corrected (wall FFT) | 50.9 | 53.6 | 56.1 |
| | projected | 77.7 | 104.8 | 142.8 |
| channel pulse (periodic, length 3) | corrected (wall FFT) | 69.5 | 71.2 | 74.2 |
| | projected | 148.3 | 201.0 | not run |

- The corrected Green is nearly mesh-independent (+4-5 % per refinement); projected grows ~1.35x.
- Advantage projected / corrected: 1.53 -> 1.96 -> 2.55 (sealed bar), 2.13 -> 2.82 (pulse).
- Sealed bar limit pressure ≈ 15.7. Follower vs dead load: ~7 % difference in volume change at p = 12.
- Pulse: one sine pulse over the first third of the channel, amplitude ramp in 12 steps, then it travels
  2/3 of the length in 48 steps. Per solve ~47 during the ramp, flat ~77 while travelling (1/128).
- 1/128 pulse on NEMO L40S: 7.7 h, 25.8 GB GPU with QBLOCK=1 (without it: out of memory).

## Data

- Laptop runs: `runs/bar3d_n*`; NEMO logs copied to `runs/nemo/logs/`.
- NEMO run folders: `/work/classic/fr_ml1145-muffto/runs/bar3d/` (the 1/128 pulse muEye file,
  4.8 GB, is only there).

## Commits

- 0180975 pressurised hole in a clamped plate, 1D wall FFT
- 706e209 3D hollow bar / periodic channel, wall-FFT corrected Green
- abec79a QBLOCK memory fix, mesh-scaling plot
