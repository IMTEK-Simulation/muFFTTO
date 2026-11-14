# Stress-driven (pressure-assisted) sintering: handover notes

Written 2026-11-09 as a plan; implemented and run 2026-11-09 to 2026-11-11. `sintering_phase_field.py` here is now a
MODIFIED copy of the base model (rigid-body approach, contracting cell, profile output). The base study is unchanged.

## RESUME HERE (state 2026-11-12, session closed by the user)
Bottom line: pressure-assisted sintering works in a 2-grain periodic chain (stage 2) and is converged in t_r, dt and mesh.
Shrinkage rate is linear in P. Every deviation from the analytical Coble rate 3 B sigma / a^2 has a source (details in
Status below):
- the excess GB transport (1.3 B at W = 4 nm) is a diffuse-interface effect: B_eff/B -> 1 as W -> 0 (W = 4/3/2 nm);
- the 0.8 driving-force factor = neck-edge term (finite surface diffusion, real physics, ~0.1 sigma, 0.02 at 4x D_surf)
  + w-taper weighting bias (diffuse, -> 0 with W) + t_r residual;
- sharp-limit estimate: rate about 0.9 x stress-only Coble at our D_s/D_gb. At W = 4 nm the match with Coble (1.07) is a
  cancellation. Absolute rates at practical W uncertain 10-30 %. Not compared with experiment. 2D only.
Use: trend and mechanism only. Not for the Freiburg talk beyond an outlook (talk 12-13 Nov 2026).

Recommended settings: CHAIN=1, GEOMETRY=geometry/two_chain.npz, NPIX=96, WIDTH=4, RB_TAU=1e-7, DT=1e-8, MECH_EVERY=2500,
Green preconditioner, laptop CPU (2 ms = 200000 steps, minutes to ~15 min). W = 2 nm needs N = 192 (did one, user OK'd);
at W = 2, t_r = 1e-7 is less converged (residual 0.07 sigma) -> use 5e-8 with dt 5e-9 if repeating.

Files (all in this folder):
- Runners: `run_rb_pressure.sh` (stage 1, isolated pair), `run_chain_pressure.sh` (stage 2, old t_r = 2e-6),
  `run_chain_convergence.sh` (t_r / mesh), `run_profile_pressure.sh` (mu profiles 0/25/100 MPa), `run_edge_tests.sh` (W = 2 nm,
  4x D_surf). The final converged P series (0/25/50/100 MPa, t_r 1e-7, dt 1e-8, 2 ms) is in exp_data
  `SINT_N96_Green_W4_P{P}u_k0.0001_chain_P{P}_tau1e-7_dt1e-8`.
- Env added to the model: RB_TAU, RB_PRESSURE, SKIP_MECH, CHAIN, SURF_FACTOR (D_surf multiplier, default 1).
- Analysis: `analyse_two_particles.py`, `plot_rate_vs_P.py`, `analyse_chain.py`, `analyse_convergence.py`,
  `plot_final_pressure.py` (main result), `plot_gb_profile.py`, `analyse_Beff.py`, `decompose_mu.py` (0.8 factor split),
  `gb_mobility_split.py`, `make_chain_video.py` (uses the imageio_ffmpeg binary; there is no system ffmpeg).
- Figures: `figures/chain_final.png` (L, neck, V/V_Coble vs t), `chain_final_snapshots.png`, `chain_gb_profile.png`,
  `chain_convergence.png`, `chain_tr_convergence.png`, `chain_coble.png`, `chain_video_last_frame.png`; video
  `videos/chain_pressure.mp4` (4 pressures, physical coordinates, L(t) strip).
- Snapshot filenames get a 6th digit after step 99999: always sort numerically (the scripts do).

Open (ask the user before any of these):
1. A figure for the W-dependence / D_surf results (only tables so far, in Status).
2. Measure the triple-junction short-circuit length delta directly (now only inferred from (a/(a-delta))^2 = B_eff/B).
3. Mesh check at pressures other than 50 MPa; t_r check at W = 2 nm.
4. Calibration / comparison with experiment, 3D, 8-particle packing `particles_s2.npz` (plan step 4): only if asked.
5. Literature: Seiz 2024 (arXiv 2308.09682) = the RB law used; `Wang.pdf` is really Zhao 2022, `Zhao_2022.pdf` is the Yi review.

## Status 2026-11-09 (supersedes the plan below where they differ)
Literature: Seiz 2024 (Sci. Rep., arXiv 2308.09682) = pressure as a shift of the GB vacancy equilibrium, acting through
the rigid-body velocity, with no stress field; Zhao 2022 (file `Wang.pdf`, names swapped with `Zhao_2022.pdf`, which is
the Yi review) = GB-weighted 1/2 eCe - rho beta2 tr(eps) in mu, no particle approach, density from a neck formula.
Done, N = 128 CPU, h = 1 nm, W = 4 nm, dt = 1e-7, 2 ms, r = 24 nm, isolated pair (`make_two_particles.py`):
- Step 0: P = 0 mass conserved, but centres DO move in the diffusion-only model (centroid distance 45.4 -> 34.8, tip
  93.1 -> 81.0, neck 16 -> 50 nm): shape relaxation towards the sharp equilibrium shape (tip 74.8, neck 55.2 nm; dihedral
  from gamma_gb/gamma_s = 1.18/2, hand calculation). The notes' "no centre motion" was wrong.
- Stage 1 (`RB_TAU`, `RB_PRESSURE`, `SKIP_MECH`, `run_rb_pressure.sh`, `analyse_two_particles.py`, `plot_rate_vs_P.py`):
  V = 2 I_gb chi (mu_s + P/e0 - <mu>_gb) / t_r. Tip shrinkage rate at equal neck 48 nm [nm/ms]: 7.1 / 11.3 / 15.6 / 22.7
  at 0 / 50 / 100 / 200 MPa (no RB: 6.0), near linear. t_r = 1e-6 and 2e-6 give the same 400-1200 us tip rate at 100 MPa,
  2e-5 is about 19 % lower. Seiz's t_r = 1e-12 is out of reach of the explicit coupling (need t_r >> dt).
- 200 MPa: GB drifts from about 900 us, one grain is consumed by 2 ms (the end of that run is not a valid pair).
- Stage 2 (`CHAIN=1`, `geometry/two_chain.npz`, `run_chain_pressure.sh`, `analyse_chain.py`): 2-grain periodic chain,
  N = 96, contracting cell y = s(t) Y (Y grid fixed, y part of Laplacian / s^2, elastic discretisation rebuilt at each
  solve), load in V = -<sigma_yy>_GB from the elastic solve. Period L 90.2 -> 73.6 / 63.6 / 50.9 nm at 25 / 50 / 100 MPa in
  2 ms (P = 0 with RB: 89.0). Rate at neck 42 nm: 7.1 / 14.2 / 29.0 nm/ms. <sigma>_GB matches force balance P L_x / neck to
  2-4 % (end of run). Mass s<rho> drifts up to 2e-5. No grain consumed.
- CAVEAT: linearity in P is largely built into the law (V proportional to sigma + capillary term). The chain rate is NOT
  converged in t_r at 50 MPa: 16.7 / 14.2 / 4.1 nm/ms for t_r = 1e-6 / 2e-6 / 2e-5, so absolute rates are set by t_r
  (explicit coupling needs t_r >> dt). The stage 1 t_r check used a fixed-time window and only 100 MPa.
  One run at t_r = 2e-7, dt = 1e-8 (50 MPa): L_end 57.8 nm (61.05 at 1e-6, 63.6 at 2e-6), rate at neck 42 nm 20.2 nm/ms,
  still rising; 1/rate is roughly linear in t_r, extrapolates to about 21 nm/ms at t_r -> 0 (2e-7 is about 5 % off; three
  points, t_r and dt changed together). `figures/chain_tr_convergence.png`.
- CONVERGED RESULT (supersedes the t_r = 2e-6 numbers above; `analyse_convergence.py`, `plot_final_pressure.py`):
  use t_r = 1e-7, dt = 1e-8, N = 96, W = 4 nm. 50 MPa, dt = 1e-8: rate at neck 42 = 17.3 / 18.9 / 20.2 / 20.8 nm/ms for t_r =
  1e-6 / 5e-7 / 2e-7 / 1e-7 (about 3 % from the extrapolated limit). N = 128: L_end 57.97 vs 57.83. dt 1e-7 -> 1e-8 at
  t_r = 1e-6: rate +4 %. Final series P = 0 / 25 / 50 / 100 MPa: L_end 87.0 / 67.8 / 57.5 / 45.4 nm; rate at neck 42 = - / 10.2 /
  20.8 / 41.6 nm/ms. VALIDATION with no fit: V / (3 B sigma_gb / a^2), B = D_gb delta Omega / kT, a = neck/2, is 1.05-1.19
  for 25-100 MPa. Linearity in the small-t_r limit is set by GB diffusion, not by the law's prefactor.
- GB profile at 50 MPa, 100-400 us (`plot_gb_profile.py`, `figures/chain_gb_profile.png`): mu(x) - mu(edge) is parabola-like,
  but centre value only 0.81-0.83 and mean 0.74-0.77 of V (a^2 - x^2) / 2B. Elastic -sigma_yy(x) is nearly flat (98-117 MPa),
  NOT the Coble parabola. So do NOT add the welded elastic sigma_n(x) to mu: GB diffusion should relax it (a^3 / (E' B) ~ 7 us,
  estimate) and mu already carries the diffusive profile. Mean of mu - edge is 0.8 sigma_gb and V / V_Coble = 1.09, which implies
  an effective GB transport about 1.35 x B: the Coble agreement is partly a cancellation of two 20-35 % offsets (source not
  identified; candidates: diffuse-interface width W, edge definition, M_surf where rho < 1 on the GB).
- W dependence (`analyse_Beff.py`, 50 MPa, 100-400 us, same W/h = 4.26): B_eff/B (from mean mu - edge) = 1.31-1.37 at W = 4 nm,
  1.23 at W = 3 nm; linear extrapolation gives about 1 at W -> 0 (two widths only). V/V_Coble = 1.07-1.11 -> 0.98-1.02.
  mean(mu - edge)/sigma_gb stays 0.80-0.83 at both W, so that offset is not an interface-width effect (unexplained).
- The 0.8 factor explained (`decompose_mu.py`, profile runs at P = 0 / 25 / 50 / 100 MPa via `run_profile_pressure.sh`,
  100-400 us). The law drives <mu>_gb (weight 4 eta0 eta1) to mu_s + sigma, mu_s = mean over ALL free surface. Exactly:
  mean_in - edge = sigma - (edge - mu_s) - (<mu>_gb - mean_in) - residual. Fractions of sigma:
  (i) edge - mu_s: 0.06-0.13 at 25/50 MPa, 0.07 -> 0.04 at 100 MPa (falls as the neck grows), 0.09-0.12 at W = 3 nm, ZERO at
  P = 0 (equilibrium). Load-induced surface effect at the neck: the edge is not at the mean surface mu, as the Coble formula
  assumes. Plausibly real (finite surface diffusion / non-equilibrium neck shape), but offset/(V a) is not constant at 100
  MPa, so it is not a pure flux bottleneck. Not proven physical.
  (ii) <mu>_gb - mean_in: 0.05-0.08 at W = 4, 0.05-0.06 at W = 3. Every row across the GB has the same mean, but w tapers
  along x at the neck edges, where mu is lowest, so the weighted mean is biased high. Diffuse-interface artefact, O(W/a).
  (iii) finite-t_r residual 0.01-0.03.
  So in the sharp limit (B_eff -> B, (ii) -> 0): V / V_Coble about 0.88-0.95, set by the edge term (supersedes "0.75-0.8").
- Third width + surface test (user OK 2026-11-10; `run_edge_tests.sh`, `SURF_FACTOR` env multiplies D_surf; 50 MPa, 200-400 us):
  W = 4 / 3 / 2 nm (N = 96 / 128 / 192, W/h fixed): B_eff/B = 1.31-1.32 / 1.23 / 1.13-1.15 (linear -> about 0.96-1.0 at W = 0);
  V/V_Coble = 1.07 / 0.98-0.99 / 0.87-0.89; edge term (i) 0.10-0.12 at all three W; weighting (ii) 0.05-0.08 / 0.05 / 0.04;
  t_r residual (iii) 0.015 / 0.03 / 0.07 (t_r = 1e-7 is less converged at W = 2 nm).
  4x D_surf (W = 4): edge term 0.10 -> 0.02 (so it is a surface-diffusion bottleneck: matter leaving the GB is spread over
  the surface at finite D_s; Coble assumes D_s infinite; this is physics, set by D_s/D_gb), but B_eff/B 1.32 -> 1.49-1.53 and
  V +30 %. Mid-neck rho = 1 exactly (`gb_mobility_split.py`: zero surface mobility there), so the excess GB transport is at
  the diffuse triple junction: surface mobility (43 x GB) short-circuits the last delta of the GB. Inferred, not measured:
  (a / (a - delta))^2 = B_eff/B gives delta about 0.7 W at all three W, larger for larger D_surf; vanishes as W -> 0.
  Summary: sharp-limit rate about 0.9 x stress-only Coble at our D_s/D_gb, the 10 % deficit being finite surface diffusion.
Open: the 0.8 driving-force factor is explained (above); mesh check only at 50 MPa; 2D, no
comparison with experiment.

## Goal
A SMALL exploratory test: does adding stress-driven diffusion and particle approach to our phase-field sintering model
give a sensible pressure effect and real shrinkage? Two particles, 64^2-128^2, laptop, minutes per run.
Not quantitative yet. Not for the Freiburg GAMM talk (12-13 Nov 2026), at most an "outlook" if the trend is clean.
Do not scale up without asking.

## Why: what the current model lacks
Base study: `../sintering_pore_evolution/` (its README.md has the model, parameters, findings). There:
- The microstructure is driven by capillarity. T = 700 K is a constant (mobilities only, no heat equation).
- Elasticity enters only via `d psi_el/d rho`, psi_el = 1/2 eps : C(rho) : eps, SECOND order in stress.
  At 50 MPa: about 1e4 J/m^3, about 1e-4 of the capillary force (e0 = 2.7e8 J/m^3). README: P = 0 gives the same
  microstructure, and the CG cost is unchanged because the unit-strain solves do not depend on P.
- Fixed periodic cell, no rigid-body motion: mean density constant (no shrinkage). Elasticity is effectively a passive
  solver benchmark.

Real hot pressing is FIRST order in stress: compressive stress at the necks changes the chemical potential
(mu ~ -sigma Omega, Coble creep along grain boundaries, Nabarro-Herring through the bulk), atoms move along the GB into
the pore, particle centres approach. Size at 50 MPa: sigma ~ 5e7 J/m^3, comparable to capillary 2 gamma/r
(gamma = 2 J/m^2, r = 15-27 nm: 1.5-2.7e8 J/m^3). These estimates are back-of-envelope, not results.

## Plan (each step small, check before the next)
0. Two-particle geometry: write a small generator (two discs, small initial neck), one `geometry/*.npz` with keys
   `centres`, `radii`, `cell` (format of `make_particles.py`). Load via env `GEOMETRY`. First reproduce P = 0 behaviour
   of the unmodified copy: neck growth, no centre motion.
1. First-order stress term in the chemical potential. Options, discuss with the user:
   a. Eigenstrain eps* = beta (rho - rho0) I, so psi_el is linear in sigma: the term is -sigma : d eps*/d rho. Same
      structure as Li swelling in `../battery_particle_fracture/`. Needs a 4th solve per mechanics step (eigenstrain RHS).
   b. Direct term in mu, -Omega sigma_n, with sigma_n the normal stress on the grain boundary, active on the GB only
      (weight sum_{i != j} eta_i eta_j). Closer to Coble; no extra solve, but needs a GB normal.
2. Particle approach (the essential step, no densification without it). Rigid-body velocity per grain from the
   force on it, advecting rho and eta_i. Wang, Acta Mater. 54 (2006) 953 has such a term, from memory: CHECK the paper,
   our model dropped it. Alternative: a macroscopic strain rate on the cell. Advecting diffuse fields can oscillate;
   use the muFFTTO FFT engine for any spectral shift.
3. Test: P = 0, 50, 200 MPa. Pass = centre distance decreases faster with P, neck grows faster, mass conserved, no
   oscillations. Plot centre distance and neck width vs time, plus density snapshots.
4. Only if asked: calibration of the rigid-body mobility (free parameter), comparison with a known densification rate
   or two-particle benchmark, 8-particle packing `particles_s2.npz`.

## Code pointers in the copied `sintering_phase_field.py` (line numbers as copied)
- Parameters 79-107; env inputs 64-75 and docstring line 35; time step scheme docstring 26-30.
- `mechanics()` 309-335: 3 unit solves, `C_eff`, `E = C_eff^-1 Sigma` (Sigma at 242), total strain, and `psi_q` at 334
  (`d psi/d rho` at quad points in e0 units): the place to add a first-order term.
- `pf_step()` 347-390: `mu` at about 355, mobility incl. GB term, semi-implicit CH update, Allen-Cahn loop. Advection
  would go here. Mobility stabilisation `M0` is fixed at t = 0.
- With an eigenstrain the problem is no longer homogeneous in E: `u = sum E_k u_k` needs an extra particular solution.
- Main loop from 442. Run names use `RUN_SUFFIX`; `sys.path` works from this folder (same depth as the base study).

## User's working rules (full versions in the memory index, MEMORY.md)
- "Try X" = one small test, ask before sweeps or scaling up. Minute-scale jobs on the laptop, larger on NEMO.
- Heavy runs via `systemd-run --user --scope -p MemoryMax=...` (OOM kills the session).
- Python: `~/micromamba/envs/muffto-gpu/bin/python` (set CUDA_PATH for GPU, see run_sintering_study.sh in the base study).
- CG: energy-error stopping, rtol 1e-6, adaptive delay d_min 10 (already in the script).
- All FFTs/operators through the muFFTTO discretization, never numpy.fft.
- Library CG (`solvers.py`) stays clean: no variant options.
- Plots: exact pixels, no smoothing; plain tick labels, labels on every panel, font >= 12; Green = green,
  Green-Jacobi = blue, dual = red. Open every PNG/video with xdg-open; do not Read images.
- Never edit the talk .tex unless asked. Keep this folder self-contained; do not change the base study.

## Honest framing
Without calibration this shows a trend, not a rate. Do not call it a realistic model of hot pressing until step 4.
