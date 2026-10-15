# Primal–dual Green (PD) preconditioner: prototype results

Standalone prototype (numpy/scipy, `src/`, `tests/`, `experiments/`), independent of muFFTTO. 2D periodic unit cell,
linear triangles (two per pixel), matrix-free with `np.roll`, PCG with u0 = 0.

- **Preconditioners:** G = Green (C_ref = solid / c_ref = 1), GJ = J^1/2 G J^1/2, PD = G K_S G with
  S_eff = C_ref C⁻¹ C_ref.
- **Stopping test:** ‖r_k‖ ≤ 1e-5 ‖r_0‖ (unpreconditioned residual), maxiter 2000. All counts below are for this test.
  The energy-error estimate (relative 1e-5, delay 10) is recorded in the CSVs as `it_energy`; it gives the same ranking,
  with 0–30 % fewer iterations.
- **Mean:** all preconditioner outputs are mean-subtracted, except in T5.
- **Material:** scalar c = ρ, c_ref = 1. Elastic C = ρ C0 (μ0 = 1, K0 = 2), C_ref = C0, macroscopic strain [1, 1, 1].
  Scalar macroscopic gradient E = [1, 0], across the laminate layers.
- **Tests:** all 26 sanity tests pass (`pytest tests`): B/Bᵀ adjoint, K symmetric, G K_ref = I − mean, Γ0 projector,
  comb = brute-force Jacobi, 1 iteration for constant material, PD symmetric positive, PD mean-free without subtraction,
  modal PD = pointwise PD for C = ρC₀, isotropic projection exact.

Full tables: `results/tables.md`; raw data: `results/*.csv`, `results/*.jsonl`; plots: `results/*.png`.

## Conclusion: where PD applies

PD is a preconditioner for **C(x) = ρ(x)·C_ref**: a scalar density field times one fixed tensor, with ρ pixelated or
sharp, in a **linear** solve. There it is mesh-independent and very fast:
- 1 iteration on laminates with the macroscopic load
- 5–8 on the sharp disk (G 10–15, GJ 62–172)
- 7–24 on the cosine geometry, flat in n

It does **not** apply when the two phases have different Poisson ratios (T10), nor to nonlinear tangents (T7–T7″). There
it is about level with Green at best and up to 11× worse. Modal matching of the dual material (T9, T7″) fixes the bulk
spectrum but not the interfaces and makes things worse in practice.

**It does not pay off for density-based topology optimization either (T11).** On real phase-field designs GJ needs about
5× fewer FFTs, and on thresholded designs Green is cheapest. PD has the lowest FFT count only on smooth gray random
designs and on sharp proportional-phase data with a macroscopic load, the setting of the T1–T3 and T10 benchmarks.

Recommendation for muFFTTO:
- port PD with S = C_ref C⁻¹ C_ref and a documented `rho_clip`
- add a runtime check that falls back to Green or GJ when max over quadrature points of ‖C/‖C‖ − C_ref/‖C_ref‖‖ > ~0.1
- keep GJ for data that are smooth at the grid scale (T3′: GJ 21–57 where PD needs 99–1046)
- keep Green for nonlinear Newton tangents (T7′, T7″: Green has the fewest iterations and FFTs in every case)

## Verdict on the predictions

| # | prediction | result |
|---|---|---|
| 1 | PD symmetric, positive on mean-free vectors, kernel = constants, no mean drift | **holds.** Symmetry/positivity tests pass; without any mean subtraction the PD solution mean stays at 1e-18 (T5), and its output is mean-free to round-off. |
| 2 | laminate: M⁻¹K = I − rank ≤ d, PCG ≤ 3 (scalar) / ≤ 2d+1 (elastic), independent of n, p, χ | **holds for the homogenization load, fails at the operator level.** With the macroscopic load PD converges in **1 iteration** in all 132 laminate solves (66 scalar + 66 elastic: n = 16…512, p = 4…n, χ = 10, 1e4). But the spectrum (T4, n = 32, p = 8) has 249 of 1023 eigenvalues ≠ 1 (λ_max = 2.5e3), and with a random right-hand side (T1b) PD needs 7–8 / 9 iterations at χ = 10 but 12–54 (scalar) and 24–168 (elastic) at χ = 1e4, growing with p and χ. The rank-≤ d structure does not hold on this discretization; the macroscopic rhs lies in the eigenspace of eigenvalue 1. |
| 3 | smooth 2D data: PD similar to GJ | **holds roughly, and PD is more mesh-robust.** Cosine χ = 1e4: PD 7–16 (scalar) / 9–24 (elastic) iterations, nearly flat in n; GJ 5–87 / 6–228, grows with n/p; G 8–411 / 11–456. Smoothed disk (T3, i ≥ 4): GJ best (12–35 scalar), PD 13–95, G 305–449. With true voids (χ = ∞) PD needs 12–29 (scalar) / 18–63 (elastic) at rho_clip = 1e-8, more than GJ on the diagonal p = n but fewer at coarse p. |
| 4 | sharp high contrast: PD worse than Green, λ_max ~ χ | **spectrum holds, iteration count does not.** λ_max(PD) = 2.5e3 ≈ χ/4 for the sharp disk (T4). But it is a tight cluster of n-proportional size (31 eigenvalues at n = 16, 63 at n = 32), so PCG handles it cheaply: sharp disk n = 256, PD **5** iterations vs G 10, GJ 62 (scalar); PD 8 vs G 15, GJ 172 (elastic). PD is worst only for lightly smoothed data (i = 1–2): 77/141 vs G 38/106 and GJ 51/47 (scalar). |

Other findings:
- **T5, T5b:** there is no mean drift of GJ in any case (mean of u ≤ 2e-7 relative). With finite contrast GJ's error falls
  with the tolerance like PD's. With **true voids** (cosine, χ = ∞) GJ's displacement error stalls at 6e-2 (scalar) /
  8e-2 (elastic) and its strain error at 0.99. This is the void null space, not the mean: J^½ (zero diagonal replaced
  by 1) passes components inside the voids through, which G and PD annihilate. Errors restricted to ρ > 0, or measured
  in the energy norm, would remove this difference; the paper's Fig. 7 discussion should say so.
- **T6:** PD with true voids depends on rho_clip. The iterations roughly double from 1e-4 to 1e-10 (e.g. n = 256, p = 16:
  14 → 32 scalar, 22 → 73 elastic). G and GJ have no such parameter.
- **Cost:** one PD application = 4d FFTs + one K_S application, against 2d FFTs for G and GJ. Measured at n = 512, PD costs
  1.7–1.8× more per iteration than G or GJ (scalar 58 vs 32 ms, elastic 109 vs 65 ms). These timings were taken with
  several jobs running in parallel, so only the ratio is meaningful.

## T1 laminate (macroscopic load), iterations G / GJ / PD

Scalar, χ = 1e4 (elastic is the same pattern, PD = 1 everywhere; `results/t1_laminate_elastic.png`):

| p \ n | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|
| 512 |  |  |  |  |  | 121 / 9 / 1 |
| 256 |  |  |  |  | 86 / 9 / 1 | 86 / 12 / 1 |
| 128 |  |  |  | 60 / 7 / 1 | 60 / 11 / 1 | 60 / 17 / 1 |
| 64 |  |  | 41 / 7 / 1 | 41 / 11 / 1 | 41 / 16 / 1 | 41 / 21 / 1 |
| 32 |  | 27 / 6 / 1 | 27 / 9 / 1 | 27 / 14 / 1 | 27 / 17 / 1 | 27 / 23 / 1 |
| 16 | 15 / 5 / 1 | 15 / 9 / 1 | 15 / 13 / 1 | 15 / 15 / 1 | 15 / 20 / 1 | 15 / 26 / 1 |
| 8 | 7 / 8 / 1 | 7 / 11 / 1 | 7 / 13 / 1 | 7 / 17 / 1 | 7 / 19 / 1 | 7 / 21 / 1 |
| 4 | 3 / 8 / 1 | 3 / 8 / 1 | 3 / 8 / 1 | 3 / 8 / 1 | 3 / 10 / 1 | 3 / 10 / 1 |

As expected, G grows with p (≈ p − 1 iterations up to p = 16) and is flat in n, while GJ grows with n/p.

![T1](results/t1_laminate_scalar.png)

## T1b laminate, random mean-free right-hand side (added: tests prediction 2 at the operator level)

Scalar, χ = 1e4:

| p \ n | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| 256 |  |  |  |  | 187 / 61 / 54 |
| 128 |  |  |  | 130 / 39 / 40 | 176 / 82 / 41 |
| 64 |  |  | 83 / 26 / 31 | 117 / 51 / 31 | 119 / 82 / 31 |
| 32 |  | 55 / 18 / 23 | 81 / 32 / 24 | 81 / 51 / 25 | 81 / 84 / 25 |
| 16 | 33 / 13 / 19 | 50 / 23 / 19 | 50 / 36 / 19 | 49 / 55 / 17 | 50 / 96 / 17 |
| 8 | 31 / 16 / 17 | 29 / 26 / 18 | 29 / 37 / 14 | 27 / 57 / 14 | 28 / 96 / 14 |
| 4 | 13 / 18 / 12 | 13 / 25 / 12 | 13 / 39 / 12 | 13 / 60 / 12 | 13 / 103 / 12 |

PD stays flat in n but grows with p. In the elastic case PD is no better than G (e.g. n = 256: 26–168 vs 26–191).

![T1b](results/t1b_laminate_random_rhs_scalar.png)

## T2 cosine

![T2 scalar](results/t2_cosine_scalar.png)
![T2 elastic](results/t2_cosine_elastic.png)

## T3 smoothed disk, n = 256, χ = 1e4

| scalar | i=0 | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|---|
| G | 10 | 38 | 106 | 305 | 360 | 384 | 411 | 428 | 449 |
| GJ | 62 | 51 | 47 | 35 | 28 | 21 | 18 | 15 | 12 |
| PD | **5** | 77 | 141 | 95 | 64 | 43 | 27 | 19 | 13 |

| elastic | i=0 | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|---|
| G | 15 | 48 | 135 | 338 | 377 | 400 | 427 | 449 | 471 |
| GJ | 172 | 141 | 118 | 95 | 73 | 53 | 39 | 28 | 23 |
| PD | **8** | 119 | 192 | 157 | 116 | 86 | 64 | 47 | 32 |

![T3](results/t3_disk_scalar.png)

## T4 spectra (scalar, χ = 1e4, n = 32)

| case | method | λ_min | λ_max | cond | eigenvalues = 1 | ≠ 1 |
|---|---|---|---|---|---|---|
| laminate p=8 | G | 1.0 | 1.0e4 | 1.0e4 | 96 | 927 |
| laminate p=8 | GJ | 0.033 | 1.57 | 47 | 0 | 1023 |
| laminate p=8 | PD | 1.0 | 2.5e3 | 2.5e3 | 774 | 249 |
| cosine p=4 | G | 1e-4 | 1.0 | 1.0e4 | 0 | 1023 |
| cosine p=4 | GJ | 0.064 | 1.90 | 29 | 0 | 1023 |
| cosine p=4 | PD | 1.0 | 1.3e3 | 1.3e3 | 899 | 124 |
| sharp disk | G | 1e-4 | 1.0 | 1.0e4 | 783 | 240 |
| sharp disk | GJ | 0.059 | 1.58 | 27 | 0 | 1023 |
| sharp disk | PD | 1.0 | 2.5e3 | 2.5e3 | 960 | 63 |

All PD eigenvalues are ≥ 1. The largest imaginary part is ≤ 6e-12 for all methods, so the eigenvalues are real to round-off.

![T4](results/t4_spectra_scalar_n32.png)

## T5 mean drift (cosine n = 256, p = 16, χ = 1e4; no mean subtraction inside GJ/PD; error vs Green at the same tolerance)

| tol | GJ displacement | GJ strain | GJ max abs mean(u) | PD displacement | PD strain | PD max abs mean(u) |
|---|---|---|---|---|---|---|
| 1e-4 | 1.2e-5 | 1.8e-4 | 3e-12 | 8.8e-6 | 3.6e-5 | 0 |
| 1e-6 | 4.1e-8 | 7.8e-7 | 2e-12 | 2.7e-8 | 2.9e-7 | 4e-19 |
| 1e-8 | 7.8e-10 | 1.8e-8 | 1e-13 | 2.7e-10 | 2.5e-9 | 0 |
| 1e-10 | 7.2e-12 | 1.8e-10 | 1e-14 | 2.1e-12 | 2.3e-11 | 4e-19 |

The elastic case behaves the same way (`results/t5_mean_drift_elastic.csv`).

![T5](results/t5_mean_drift_scalar.png)

## T6 rho_clip (PD, cosine χ = ∞), iterations

| scalar n, p | 1e-4 | 1e-6 | 1e-8 | 1e-10 | G | GJ |
|---|---|---|---|---|---|---|
| 64, 4 | 10 | 12 | 17 | 22 | 10 | 9 |
| 256, 4 | 11 | 13 | 17 | 23 | 12 | 18 |
| 256, 16 | 14 | 18 | 25 | 32 | 49 | 17 |
| 256, 64 | 10 | 16 | 24 | 31 | 150 | 12 |
| 256, 256 | 7 | 11 | 14 | 18 | 293 | 6 |

Elastic: `results/t6_rho_clip_elastic.csv` (same trend, 10 → 73 iterations).

## T3′ contrast and clip sweep (disk, n = 256, rho_clip relative to max ρ = 1)

Scalar, χ = 1e12 (the table for χ = 1e8 is almost identical; ">2000" = not converged):

| i | G | GJ | PD clip 1e-2 | 1e-3 | 1e-4 | 1e-6 | no clip |
|---|---|---|---|---|---|---|---|
| 0 | 10 | 68 | **5** | 5 | 5 | 5 | 5 |
| 1 | 38 | 58 | **23** | 50 | 78 | 122 | 256 |
| 2 | 111 | 55 | **43** | 54 | 142 | 374 | 1048 |
| 4 | 850 | **51** | 315 | 257 | 245 | 726 | >2000 |
| 8 | >2000 | **57** | 723 | 597 | 633 | 579 | >2000 |
| 16 | >2000 | **45** | 959 | 657 | 475 | 363 | >2000 |
| 32 | >2000 | **34** | 915 | 555 | 330 | 187 | 1896 |
| 64 | >2000 | **24** | 1046 | 518 | 267 | 99 | 376 |

- **Without a clip PD fails** for χ ≥ 1e8 at i ≥ 4. With a clip it becomes independent of the contrast (1e8 and 1e12 give
  the same counts).
- **No single clip value works everywhere.** 1e-2 is best for nearly sharp data (i ≤ 2), 1e-6 for well-resolved smooth
  data (i ≥ 32).
- **On smooth data PD never reaches GJ.** Elastic is the same picture, but worse for PD (χ = 1e12, i = 64: PD 396 at
  best, GJ 65; `results/t3b_contrast_clip_*.csv`).
- **The smoothing width is a number of pixels**, so the transition band sits at the grid scale for every n. That is the
  bad band for PD as well as for GJ. Refining with the width fixed in physical units would be the physically meaningful
  protocol.

## T7 / T7′ Newton, power-law elasticity (2D version of the paper's Sec. 4.5)

- **Setup:** compliant linear inclusion (κ, μ) = (2, 1) in a power-law matrix (K = 20, σ0 = 5, ε0 = 0.01), macroscopic
  shear 0.025 applied in one step, full Newton steps, Newton tolerance ‖B δu‖ ≤ 1e-5 ‖E‖.
- **Inclusion:** T7 uses a square of area 1/4; T7′ a 0.5 × 0.25 rectangle (volume fraction 1/8, pixel-aligned on every
  grid).
- **Reference materials:** G uses the inclusion, GJ the identity (as in the paper), PD the inclusion or the identity.

T7, total CG iterations over all Newton steps (`results/t7_nonlinear_elastic.csv`):

| ω | n | G | GJ | PD (C_ref inclusion) | PD (C_ref I) | Newton steps |
|---|---|---|---|---|---|---|
| 3 | 32 | 466 | 480 | 593 | 586 | 12 |
| 3 | 256 | 1568 | 2967 | **1455** | 1558 | 16 |
| 4 | 256 | **2921** | 4255 | 7107 | 7408 | 18 |
| 5 | 32 | 856 | **729** | 8131 | 8178 | 15 |
| 5 | 256 | **5033** | 5628 | 18749 | 25368 | 19 (PD inclusion: 24) |

T7′, ω = 3 (`results/t7b_newton.jsonl`): G 318 / 431 / 474 / 533 and GJ 359 / 646 / 1148 / 2068 for n = 32 / 64 / 128 /
256. PD with C_ref = inclusion: 456 / 533 / 570 / 628 (C_ref = I and C_ref = matrix tangent at the first step within
±10 %). PD is flat in n but needs about 2.4× the FFTs of Green (5024 vs 2132 at 256²). The ω = 5 runs of T7′ were
not completed (a scheduling mistake on my side); T7″ covers ω = 5.

## T9 spectrum check: PD with a varying Poisson ratio (elastic, n = 16)

Matrix material only, κ = 20 fixed, C_ref = inclusion (κ_r, μ_r) = (2, 1):

| μ field | pointwise PD | modal PD | G | GJ |
|---|---|---|---|---|
| uniform 2000 | [1, 50.5] (predicted: 1 and 45) | **[1, 1.33]** | cond 3.0 | cond 2.0 |
| smooth 500 … 40000 | [1, 827] (predicted up to 818) | [0.20, 10.4], cond 53 | cond 154 | cond 29 |

This confirms the zeroth-order error of pointwise PD when κ/μ differs from κ_r/μ_r: the longitudinal symbol is
(κ_r²/κ + μ_r²/μ)(κ+μ)/(κ_r+μ_r)². The **modal** dual material S = s_κ I⊗I + 2 s_μ P_dev with s_μ = μ_r²/μ and
s_κ + s_μ = (κ_r+μ_r)²/(κ+μ) (from the isotropic projection of the tangent) removes it. For C = ρC₀ the modal S equals
the pointwise one (pytest).

## T7″ Newton with modal PD (T7′ problem; total CG, FFTs in brackets)

Modal PD needs κ_r/μ_r ≥ max κ/μ to keep s_κ > 0. Along the Newton path max κ/μ reaches ~70 (ω = 3) and ~450 (ω = 5),
so κ_r is raised per Newton step ("auto reference"). The clip variant also limits μ ≥ 1e-2 μ_max in S.

| ω | n | CG stop | G | GJ | PD pointwise | PD modal | PD modal + clip | Newton steps |
|---|---|---|---|---|---|---|---|---|
| 3 | 64 | residual | **431** (1724) | 646 (2584) | 533 (4264) | 1506 (12048) | 644 (5152) | 8 |
| 3 | 64 | energy | **499** (1996) | 732 (2928) | 579 (4632) | 1501 (12008) | 690 (5520) | 8 |
| 3 | 256 | residual | **533** (2132) | 2068 (8272) | 628 (5024) | 1821 (14568) | 789 (6312) | 8 |
| 3 | 256 | energy | **574** (2296) | 2089 (8356) | 651 (5208) | 1733 (13864) | 793 (6344) | 8 |
| 5 | 64 | residual | **748** (2992) | 1140 (4560) | 1305 (10440) | 9124 (72992) | 1213 (9704) | 10 |
| 5 | 64 | energy | **816** (3264) | 1195 (4780) | 1305 (10440) | 8178 (65424) | 1223 (9784) | 10 |
| 5 | 256 | residual | **1059** (4236) | 4121 (16484) | 1510 (12080) | 28568 (228544) | 1561 (12488) | 10 |
| 5 | 256 | energy | **1117** (4468) | 3973 (15892) | 1497 (11976) | 25249 (201992) | 1534 (12272) | 10 |

- **Green has the fewest iterations and FFTs in every case.**
- **Modal PD is the worst** (up to 27× Green). The clip brings it back to about pointwise PD.
- **Both stopping rules give the same ranking and the same Newton steps.** The energy stop costs 0–15 % more CG here.

Isolating the first Newton step (uniform matrix, `t7c` side test) shows why modal PD loses:
- modal PD is worse than pointwise PD even when the tangent is replaced by its isotropic projection (ω = 3, 256²: 160 vs
  80; G 76)
- the modal S raises the dual's bulk contrast between the phases (matrix s_κ ≈ 0.004 instead of 0.2, a ratio of ~500),
  which damages the interface modes from the first step on
- the anisotropy of the power-law tangent is secondary: removing it lowers every preconditioner by 10–35 %

## T8 random right-hand side (proxy for Newton residuals)

PD iterations for n = 16 → 512 (G / GJ in `results/t8_random_rhs_*.csv`):

| geometry | scalar | elastic |
|---|---|---|
| cosine χ = 1e4, p = 4 | 14 → 15 | 22 → 24 |
| cosine, p = 16 | 10 → 14 | 18 → 35 |
| cosine, p = n | 9 → 8 | 14 → 14 |
| sharp disk | 8 → 8 | 13 → 13 |
| disk smoothed i = 8 (fixed number of pixels) | 5 → 159 | 5 → 200 |

PD is flat in n for pixelated and sharp data with a general rhs. With the transition band at the grid scale (i = 8) it
grows; GJ grows there too (5 → 80), while G stays at ~600.

## T10 linear test of the applicability condition (sharp disk, n = 256, elastic)

Inclusion (κ, μ) = (2, 1) inside the disk; C_ref = inclusion:

| matrix | load | G | GJ | PD pointwise | PD modal |
|---|---|---|---|---|---|
| (20, 2000): different Poisson ratio | macro E | **77** | 523 | 109 | 235 |
| (20, 2000) | random rhs | 132 | 861 | **129** | 345 |
| (4000, 2000): same Poisson ratio, C = ρ C_inc | macro E | 14 | 142 | **7** | **7** |
| (4000, 2000) | random rhs | 24 | 185 | **12** | **12** |

This is a linear problem, so the Newton outcome is fully explained by the phases not being proportional. When C ∝ C_ref
everywhere, PD halves Green's count; when the Poisson ratios differ, the interface clusters smear and PD loses its edge.

## T11 density-based topology optimization designs (elastic, C = ρ_mat C0, loads e1, e2, e3; iterations / 2D FFTs summed over the three loads; ✗ = not converged in 5000 iterations per load)

**Part A**: designs along a real muFFTTO phase-field optimization at 128² (`example_2D_elasticity_TO.py`, random U(0,1)
start, L-BFGS, η = one pixel, converged after 272 iterations). The designs are pictured in `results/t11_designs_A.png`.

| design | material | G | GJ | PD clip 1e-2 | PD 1e-4 | PD none |
|---|---|---|---|---|---|---|
| k = 0 | max(ρ², 1e-15) | **164 / 656** | 1007 / 4028 | 88 / 704 | 309 / 2472 | 310 / 2480 |
| k = 50 | max(ρ², 1e-15) | 15000 ✗ | **1778 / 7112** | 12669 / 101352 | 14764 / 118128 | 15000 ✗ |
| k = 200 / 250 / 270 | max(ρ², 1e-15) | 15000 ✗ | **681–714 / 2724–2856** | ~14000 / ~112000 | 15000 ✗ | 15000 ✗ |
| k = 0 | ρ²(1−1e-5) + 1e-5 (the optimizer's law) | **164 / 656** | 1008 / 4032 | 88 / 704 | 303 / 2424 | 303 / 2424 |
| k = 50 | ρ²(1−1e-5) + 1e-5 | 3508 / 14032 | **924 / 3696** | 1361 / 10888 | 1178 / 9424 | 1464 / 11712 |
| k = 200 / 250 / 270 | ρ²(1−1e-5) + 1e-5 | ~3700 / ~14800 | **514–519 / 2056–2076** | ~1430 / ~11450 | ~1270 / ~10180 | ~1500 / ~12000 |
| final thresholded, void 1e-4 | 1 / 1e-4 | **89 / 356** | 745 / 2980 | 59 / 472 | 104 / 832 | 104 / 832 |
| final thresholded, void 0 | 1 / 0 | **89 / 356** | 276 / 1104 | 59 / 472 | 104 / 832 | 107 / 856 |

**Part B**: filtered random noise on a design grid p, analyzed at n > p:

| design | n | G | GJ | PD clip 1e-2 | PD 1e-4 | PD none |
|---|---|---|---|---|---|---|
| p = 32, SIMP ρ³ ≥ 1e-4 | 128 | 1061 / 4244 | 265 / 1060 | 186 / 1488 | **97 / 776** | 97 / 776 |
| p = 32, SIMP | 256 | 1153 / 4612 | 469 / 1876 | 194 / 1552 | **100 / 800** | 100 / 800 |
| p = 64, SIMP | 128 | 798 / 3192 | 312 / 1248 | 171 / 1368 | **118 / 944** | 118 / 944 |
| p = 64, SIMP | 256 | 839 / 3356 | 568 / 2272 | 173 / 1384 | **118 / 944** | 118 / 944 |
| p = 32, thresholded 1 / 0 | 128 | **288 / 1152** | 946 / 3784 | 149 / 1192 | 218 / 1744 | 228 / 1824 |
| p = 32, thresholded | 256 | **303 / 1212** | 1697 / 6788 | 155 / 1240 | 217 / 1736 | 229 / 1832 |
| p = 64, thresholded | 128 | **599 / 2396** | 2179 / 8716 | 302 / 2416 | 312 / 2496 | 380 / 3040 |
| p = 64, thresholded | 256 | **601 / 2404** | 3790 / 15160 | 301 / 2408 | 313 / 2504 | 382 / 3056 |

**Pass criterion (PD lowest FFT count on the thresholded designs at n > p in Part B): not met.** Green is lowest in all
four cases. PD with clip 1e-2 needs half of Green's iterations but 0.2–3 % more FFTs. **The topology-optimization use case
is therefore dropped.** The other findings:
- On the designs of a real phase-field optimization (diffuse interfaces about one pixel wide, contrast 1e5–1e15)
  **GJ is best by a factor of about 5 in FFTs.** G and every PD clip are far behind or do not converge.
- PD wins only on smooth gray random designs on a coarse design grid (Part B SIMP: 776–944 FFTs vs GJ 1060–2272).
  That is not what a phase-field optimization produces after its first iterations.
- On the thresholded final design G is best (356 FFTs), with PD clip 1e-2 second (472).

## Reproduce

```
cd prototypes/pd_preconditioner
python -m pytest tests
cd experiments
python t1_laminate.py 1; python t1_laminate.py 2      # argument: 1 scalar, 2 elastic
python t1b_laminate_random_rhs.py 1   ...   t6_rho_clip.py 2
python t3b_contrast_clip.py 1 1e12; python t5b_mean_drift_voids.py 1; python t8_random_rhs.py 1
python t7_nonlinear.py 16 32 64 128 256                # T7
python t7b_newton.py 3 256                              # T7'  (OMEGA N)
python t7c_modal_newton.py 5 256 energy                 # T7'' (OMEGA N residual|energy)
python t9_modal_spectrum.py; python t10_poisson_ratio.py
python t11_topology_optimization.py B; python t11_topology_optimization.py A t11_to_run/data/to128/*_history.nc [_soft]
python t4_spectra.py; python make_tables.py > ../results/tables.md
```
