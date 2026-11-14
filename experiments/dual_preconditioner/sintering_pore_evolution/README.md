# Phase-field sintering / pore evolution: elastic CG cost of Green, Green-Jacobi and dual

This study is part of the evolving-diffuse-interface series (TO, AT2, fatigue, ASR, battery, PF-CZM). A Cu nano-powder
compact sinters under hot pressing. The pore band is diffuse and has high contrast: `C_pore = k_res C_Cu`. The band
moves at every time step. We compare the CG cost of the elastic solve over the whole run for three preconditioners:
Green, Green-Jacobi and dual.

## Files
| file | what |
|---|---|
| `calibrate_wang.py` | 1D offline calibration of the Wang free energy. It gives the dimensionless gamma_s, gamma_gb, width and mobility integrals, written to `geometry/wang_calibration.npz` |
| `make_particles.py` | Periodic packing of 8 discs in a 128 nm cell (radii 14.8 / 21.1 / 27.4 nm, solid fraction 0.656, 11 contacts). Writes `geometry/particles_s<seed>.npz`; the study uses seed 2 |
| `sintering_phase_field.py` | The model (env vars in its docstring: NPIX, DEVICE, STEPS, DT, WIDTH, PRESSURE, LOAD, K_RES, MECH_EVERY, SNAP, GEOMETRY, CHECKPOINT_SECONDS, RESTART_FROM, RUN_SUFFIX) |
| `run_sintering_study.sh` | `run_sintering_study.sh <N> <cpu|gpu> [preconditioners]`. Each run gets its own memory-capped systemd scope |
| `plot_sintering.py` | CG summary, physics vs time and snapshots, written to `figures/` |
| `make_sintering_video.py` | mp4 of rho, the grain map and CG vs time |

## Model
This is the model of Wang, Acta Mater. 54 (2006) 953, with elasticity added. The fields are the density rho (conserved)
and one grain order parameter eta_i per particle.

* Free energy:
  `f = A rho^2(1-rho)^2 + B[rho^2 + 6(1-rho) S2 - 4(2-rho) S3 + 3 S2^2]`, with Wang's A = 16, B = 1,
  k_rho = 10, k_eta = 5. Elastic energy: `psi_el = 1/2 eps : C(rho) : eps`, with
  `C = (h + (1-h) k_res) C_Cu` and `h = rho^3(10-15rho+6rho^2)`.
* Mechanics, which is the compared solve:
  - The macroscopic stress is prescribed, `Sigma = -P e_y e_y` (die hot pressing, P = 50 MPa), with periodic BC.
  - There is no eigenstrain, so the problem is linear in E. Each mechanics step does 3 unit-strain solves, then
    `C_eff`, `E = C_eff^-1 Sigma` and `u = sum E_k u_k`. This is the exact control of the ASR script.
  - Each solve is warm-started from the previous mechanics step and solved with PCG. CG stops on the energy-error
    estimate with rtol 1e-6 and an adaptive delay with d_min 10.
  - Preconditioner reference media: Green and dual use C_ref = C_Cu; Green-Jacobi uses 2 mu_Cu I plus Jacobi.
* Cahn-Hilliard:
  - `d rho/dt = div(M grad mu)`, with `mu = f_rho - k_rho Lap rho + dpsi_el/drho`.
  - Wang mobility: `M = M_vol h + M_vap(1-h) + M_surf rho^2(1-rho)^2 + M_gb sum_{i!=j} eta_i eta_j`.
* Allen-Cahn: `d eta_i/dt = -L (f_eta_i - k_eta Lap eta_i)`.
* Numerics:
  - Everything uses muFFTTO operators: the Q1 gradient, its weighted transpose and the N operator, plus the FFT
    engine of the phase-field `Discretization` (`disc_pf.fft`). There is no numpy.fft and no separate engine, so a
    step runs on GPU exactly like the solver.
  - Discretisation: `div(M grad mu) = -(G^T W M G mu)/h^2` with a lumped mass, and `-Lap` is the Q1 stiffness divided
    by h^2. The Fourier symbol of `-Lap` comes from the FFT of its impulse response.
  - Time stepping is semi-implicit and stabilised in Fourier space, with `M0 = max M` at t = 0.
* **Not implemented:** Wang's rigid-body advection (particle translation/rotation from grain-boundary forces). Its
  parameters (kappa, m_t) are phenomenological, with no real-material values. Consequences:
  - The cell is periodic with a fixed size and rho is conserved, so the **solid fraction is constant (65.6 %)**.
  - Densification shows only as neck (grain-boundary) growth, pore rounding/closure, a shrinking free surface and
    stiffening of C_eff.
  - Hot pressing enters only through `dpsi_el/drho`. At the real 50 MPa this is about 1e-4 of the capillary driving
    force (psi_el ~ 1e4 J/m^3 vs e0 = 2.7e8 J/m^3). Its physical effect is therefore negligible: a stress-free run
    (P = 0) gives the same microstructure. Its CG cost is also unchanged, because the unit solves do not depend on P.

## Parameters (Cu, ~700 K) and sources
| quantity | value | source |
|---|---|---|
| surface energy gamma_s | 2 J/m^2 | Seiz, Hierl, Nestler, Rheinheimer, *Revealing process and material parameter effects on densification via phase-field studies*, arXiv:2308.09682 (2024), Table 1 ("approximating copper at ~700 K") |
| GB energy | 1 J/m^2 (Seiz); the model gives 1.18 J/m^2 (Wang ratio gamma_gb/gamma_s = 0.59) | same |
| D_vol, D_gb, D_surf | 1e-15, 5.5e-11, 1.69e-10 m^2/s | same |
| diffusion width delta | 2e-10 m | same |
| atomic volume Omega | 1.22e-29 m^3 | same |
| particle radius | 8-12 nm in Seiz; here 14.8-27.4 nm | same order |
| GB mobility m_gb | 1/tau_gv = 1.25e-11 m^4/(J s) | Seiz surface inverse mobility, **assumed** for the GB |
| M_vap / M_vol | 0.1 | **assumption** (small vapour transport, as in Wang 2006) |
| E, nu | 120 GPa, 0.34 (plane strain) | Cu polycrystal average |
| P | 50 MPa uniaxial | typical hot-press / SPS pressure |
| k_res | 1e-4 (also tested 1e-6) | |
| surface width W (10-90 %) | 4 nm, the same on all grids | this study (Seiz uses about 10 nm) |

Mapping:
* Lengths: l0 = W / w*, with w* = 2.40 the Wang 10-90 % width, so l0 = 1.667 nm.
* Energy density: e0 = gamma_s / (gamma_s* l0), with gamma_s* = 4.37, so e0 = 2.75e8 J/m^3.
* Mobilities follow the Deng/Ahmed calibration: `M_vol = D_v Omega/kT`, `M_surf = D_s delta Omega/(kT I_s)` and
  `M_gb = D_gb delta Omega/(kT I_gb)`. Here I_s and I_gb are the model's own integrals of rho^2(1-rho)^2 and of
  sum eta_i eta_j across an interface.
* Allen-Cahn mobility: `L = m_gb gamma_gb / k_eta`, so that the AC curvature velocity equals m_gb gamma_gb K.

Time step:
* dt = 1e-7 s. A 64^2 convergence test to 50 us gives a surface length of 430 / 435 / 445 / 508 nm for
  dt = 2.5e-8 / 1e-7 / 2.5e-7 / 1e-6 s. The stabilised scheme slows the evolution artificially when dt is too large.
* 5000 steps cover 500 us. Mechanics runs every 25 steps (MECH_EVERY), which gives 201 mechanics steps and
  603 elastic solves per run. In between, dpsi_el/drho is lagged.

Mesh study at fixed W = 4 nm, 128 nm cell:

| grid | h | W/h |
|---|---|---|
| 32^2 | 4 nm | 1 |
| 64^2 | 2 nm | 2 |
| 128^2 | 1 nm | 4 |
| 256^2 | 0.5 nm | 8 |

## Results (2026-11-07)
Run setup:
* 500 us, 201 mechanics steps, 603 elastic solves per run.
* 32^2 and 64^2 ran on CPU. 128^2 and 256^2 ran on the laptop GPU, with 3 jobs sharing it, so the wall times are only
  indicative.
* "Work" counts FFT pairs per CG iteration: Green 1, GJ 1, dual 2.

**Mean CG iterations per elastic solve (k_res = 1e-4, P = 50 MPa):**

| grid | W/h | Green | Green-Jacobi | dual | dual work | wall mech [s] G / GJ / dual |
|---|---|---|---|---|---|---|
| 32^2 | 1 | 105 | 31 | 79 | 158 | 22 / 7 / 34 (CPU) |
| 64^2 | 2 | 332 | **110** | 177 | 354 | 197 / 51 / 187 (CPU) |
| 128^2 | 4 | 333 | **163** | 168 | 336 | 1463 / 617 / 1474 (GPU) |
| 256^2 | 8 | 334 | **200** | **166** | 331 | 2246 / 1789 / 2376 (GPU) |

**k_res = 1e-6 at 64^2:** Green 2280, GJ **130**, dual 1331 (work 2662).

Findings:
* **Green** costs about 333 iterations from 64^2 up. That is the resolved-interface plateau; it does not depend on h.
  It grows strongly with the contrast: about 7x from k_res 1e-4 to 1e-6.
* **Dual** is also flat in h: 177 -> 168 -> 166. It needs about half as many iterations as Green at every resolved
  grid, but costs the same work as Green.
* **Green-Jacobi** is the cheapest in work on every grid and almost insensitive to the contrast (110 -> 130 for
  k_res 1e-4 -> 1e-6). Its iteration count grows with resolution: 110 -> 163 -> 200, about +40 per refinement.
  - At 256^2 (W/h = 8) dual needs fewer iterations than GJ (166 vs 200), but still about 1.65x more work.
  - Extrapolating the +40 per doubling, GJ would reach the dual work (about 330) only around 2048^2 and finer.
* CG per solve vs time: the cost is highest at t = 0, when the particles have sharp contacts and thin necks. It
  decreases as the necks grow (warm start plus a stiffer network) and then stays at a plateau.

Physics (identical on all three preconditioners, to the printed digits):
* Fast neck formation in the first ~30 us: the GB length goes from 0 to 290 nm.
* Pore count drops from 10-14 to 2, by closure of the small inter-particle gaps (about 30-50 us).
* The pores round off and the free surface shrinks from 714 to 385 nm. C_yy/C_s rises from 0.25 to 0.32.
* Small grains are consumed by grain growth: the 8 grains become 4-5 visible ones, and the GB length falls back to
  about 210 nm.
* 128^2 and 256^2 agree closely, so the physics is mesh-converged at W/h >= 4. 64^2 is close; 32^2 (W/h = 1) is
  pinned by the grid and the microstructure freezes.
* The stress-free run (P = 0, 64^2 GJ) gives the same final surface (399.82 vs 399.84 nm) and the same CG mean
  (109.5).

Figures:
* `figures/sintering_cg_summary.png`: CG per solve, work, wall time and CG vs time.
* `figures/sintering_physics.png`: neck, surface, stiffness and pores.
* `figures/sintering_snapshots_N128.png`: rho and grains at t = 0 / 10 / 100 / 500 us, on exact pixels.
* `figures/sintering_cg_summary_W4_P50u_k1e-06.png`: the k_res = 1e-6 comparison at 64^2.
* `figures/sintering_N128_dual.mp4`, `figures/sintering_N64_dual.mp4`: videos.

Caveats:
* There is no rigid-body motion, and the cell is periodic with a fixed size, so the relative density is constant.
* At 50 MPa the elastic coupling is about 1e-4 of the capillary driving force.
* The GB mobility and the vapour mobility are assumptions.
* Wang's free energy gives gamma_gb/gamma_s = 0.59 (Seiz: 0.5).
* The particles are about 2x larger than in Seiz, and W = 4 nm is thinner than Seiz's 10 nm.
* The time step of 1e-7 s was checked only at 64^2.
* Log names from the first 64^2 k_res = 1e-4 runs were overwritten by the k_res = 1e-6 runs. The data (npz) are
  separate and complete.
