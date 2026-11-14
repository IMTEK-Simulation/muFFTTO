# What makes the dual preconditioner better than Green, and when it is not (2026-11-08)

## Short answer

- **Green** (P_G = K_ref⁻¹, C_ref = solid) reasons in strains with a stiff reference. In 1D, the eigenvalues of the
  preconditioned operator are the stiffness values themselves. CG needs about one iteration per distinct stiffness
  value:
  - a sharp two-phase field is cheap at any contrast;
  - a diffuse band of intermediate values is expensive, and gets worse with contrast.
- **Dual** (P_D = K_ref⁻¹ · Dᵀ(C_ref C⁻¹ C_ref)D · K_ref⁻¹) reasons in stresses, through the compliance C⁻¹. In 1D it is
  the exact inverse up to one rank-1 term, the macro-stress mode, so CG takes **2 iterations at any contrast**,
  whether the band is sharp or diffuse.
- **Every macro load of a laminate gives a 1D problem:** tension across the layers, along them, and shear alike. So
  the dual is near-exact on laminate-like geometry **for any load direction**.
  - Cracks are long, thin, locally flat soft bands, i.e. locally laminate-like. That is why the dual wins in the
    cracked phase.
  - Its weak case is a **compact soft inclusion enclosed by solid**, which is not laminate-like in any direction.
    There Green-Jacobi is best.
- **The load direction does not matter** for any preconditioner; only the geometry does. My earlier wording,
  "series vs parallel load", was wrong.

## 1. 1D derivation, checked numerically (`dual_exact_1d.py`)

Setup: periodic bar, forward difference D, K = Dᵀ C D, c_ref = 1. On the mean-free strains e = D u, with Π the
projection onto mean-free vectors:

    Green:  P_G K  ~  Π C Π                          eigenvalues = the (compressed) stiffness values
    dual:   P_D K  ~  Π C⁻¹ Π C  =  I − (Π c⁻¹)(mean(c e))
                   =  identity except ONE eigenvalue mean(c)·mean(1/c)

Diffuse AT2 band (l = 4 cells), N = 128:

| contrast | Green: eigenvalues ≠ 1 | Green range | dual: eigenvalues ≠ 1 | the one dual eigenvalue (= mean(c)·mean(1/c)) |
|---|---|---|---|---|
| 1e2 | 127 of 127 | [0.015, 1] | **1** | 2.015 |
| 1e4 | 127 of 127 | [0.0052, 1] | **1** | 72.16 |
| 1e8 | 127 of 127 | [0.0051, 1] | **1** | 7.08e5 |

The rank-1 term comes from the dual's projection removing the mean of the stress.

Caution: the nodal form P_D = pinv(DᵀD)·DᵀC⁻¹D·pinv(DᵀD) is numerically useless here, because the pseudo-inverse of
the singular D loses about 1e11. Use the strain-space form.

## 2. 2D test: geometry vs load direction (`crack_orientation_test.py`)

Setup:
- stiffness s = (1−d)²(1−k) + k with the AT2 profile d = exp(−dist/l);
- contrast 1/k = 1e4, l = 4h;
- linear triangles, CG from zero with the energy-error stop 1e-6, as in all studies;
- loads E_xx (across the band x = 1/2), E_yy (along it), E_xy (shear).

**256², d_min = 10** (the study setting; `crack_orientation_results.csv`):

| geometry | Green (xx / yy / xy) | dual (xx / yy / xy) | dual ×2 |
|---|---|---|---|
| straight crack through the cell (laminate), diffuse | 24 / 24 / 24 | **14 / 14 / 14** | 28 |
| the same, sharp two-phase | 11 / 11 / 11 | 11 / 11 / 11 | 22 |
| finite crack with two tips | 74 / 76 / 71 | **27 / 27 / 27** | 54 |
| two crossing straight cracks | 87 / 87 / 84 | **33 / 33 / 30** | 60–66 |
| soft round inclusion, diffuse rim | 434 / 434 / 425 | 205 / 204 / 202 | 404–410 |

**128², d_min = 1** (no delay: shows the true speed of a near-exact preconditioner;
`crack_orientation_results_N128_dmin1_Green_Green_Jacobi_dual.csv`):

| geometry | Green | Green-Jacobi | dual |
|---|---|---|---|
| laminate, diffuse | 17 | 8–9 | **7** |
| laminate, sharp | 4 | 5 | 4 |
| finite crack | 67–76 | 41–43 | **20** |
| crossing cracks | 82–85 | 55–91 | **24–26** |
| soft inclusion | 432–446 | **58–73** | 177–184 |

Observations:
- **Load direction:** no effect. For every geometry and every preconditioner, the three loads are within about 10 %.
  Green-Jacobi on the crossing cracks is the only exception (shear 55 vs 88–91).
- **Discretization:** on the FE grid the laminate is near-exact for the dual (7 iterations), not 2 as in 1D. The
  elements have several quadrature points per pixel, so the discrete problem is only approximately 1D.
- **The minimum count:** the delayed energy stop (d_min = 10) puts a floor of about 11–15 under all counts. Near-exact
  preconditioners look alike in the study numbers.
- **Complementarity:**
  - the dual is best for band-like soft features (cracks, crossings);
  - Green-Jacobi is best for compact soft inclusions (58–76 vs the dual's 177–184);
  - Green is best only for sharp two-phase data.

## 2b. The same test with the classical spectral discretization (`spectral_orientation_test.py`)

This is the Fourier / collocation discretization of the original FFT solvers (Moulinec–Suquet; the CG form of
Zeman et al. 2010; the Galerkin view of Vondřejc et al. 2014; the Newton–CG form of de Geus et al. 2017).
- Unknown: the displacement at the N² grid points, in a trigonometric polynomial basis.
- Gradient and divergence by the Fourier symbol iξ (Nyquist frequency set to zero); C evaluated at the grid points.
- Same CG, same energy stop, same Green and dual. Only the derivative operators differ.
- muFFTTO's own `'Fourier'` discretization is only a stub, so the operators are built here on muFFTTO's FFT engine.
- Green-Jacobi is left out: with a global (spectral) derivative, diag(K) is nearly constant and carries no local
  stiffness information, so a Jacobi scaling has no FE-like meaning. A first try with J = s^(−1/2) gave 264–1728
  iterations and is not meaningful.

**128², contrast 1e4, l = 4h, d_min = 1, compared with FE (linear triangles, same setting):**

| geometry | spectral Green (xx / yy / xy) | spectral dual (xx / yy / xy) | FE Green | FE dual |
|---|---|---|---|---|
| homogeneous (random rhs) | 4 | 4 | – | – |
| laminate, diffuse | 18 / 18 / 18 | **5 / 5 / 5** | 17 | 7 |
| laminate, sharp | 5 | 6 | 4 | 4 |
| finite crack | 66 / 67 / 49 | 59 / 60 / 48 | 67–76 | **20** |
| crossing cracks | 124 / 124 / 53 | **57 / 57 / 35** | 82–85 | 24–26 |
| soft inclusion | 523 / 523 / 495 | 238 / 238 / 227 | 432–446 | 177–184 |

What changes with the spectral discretization:
- **Laminate:** the dual is near-exact in both. Spectrally it is at the minimum count (5, like the homogeneous 4),
  because the spectral gradient of a laminate field is exactly 1D: one value per grid point, no quadrature-point
  mismatch.
- **Finite crack (with tips):** the dual's advantage almost disappears spectrally (59 vs 66), while in FE it is 3.4×
  (20 vs 67). The likely cause, not checked: Gibbs ringing of the spectral derivative at the crack tips. The
  trigonometric basis cannot represent the tip singularity locally, so the ringing spreads over the cell.
- **Crossing cracks and soft inclusion:** the dual still about halves Green, but every count is higher than in FE.
- **Load direction:** spectrally it matters a little for shear. Green's shear count is lower (49 vs 66, 53 vs 124);
  in FE all loads are within about 10 %.
- **In short:** the "dual is exact for laminates" mechanism holds in both discretizations. The FE discretization is
  clearly kinder to cracks with tips.

## 2c. Smooth fields, where Green-Jacobi wins: FE vs spectral

New geometries:
- `smooth`: ρ = ½ + ½ cos 2πx cos 2πy, s = ρ²(1−k) + k, like the sine start of the TO runs;
- `wide_disc`: a soft disc with a wide rim, l = 16h, like a sintering pore.

The spectral Green-Jacobi uses the **exact** diagonal of the spectral K: a unit displacement at one grid point gives
the strain e(r) = sym(d(r) ⊗ e_a) everywhere, so diag(K) = s ∗ (e:C₀:e), a periodic convolution done by FFT.

128², contrast 1e4, d_min = 1 (xx / yy / xy ranges):

| geometry | discretization | Green | Green-Jacobi | dual (raw) |
|---|---|---|---|---|
| smooth | FE | 501–565 | **21–24** | 20–24 |
| smooth | spectral | 498–567 | 119–120 | **49–70** |
| wide_disc (l = 16h) | FE | 469–470 | **42–48** | 100–119 |
| wide_disc (l = 16h) | spectral | 480–491 | 289–309 | **149–166** |
| disc (l = 4h) | FE | 432–446 | **58–73** | 177–184 |
| disc (l = 4h) | spectral | 495–523 | 564–592 | **227–238** |

Findings:
- **Green-Jacobi's strength depends on the FE discretization.**
  - With FE it is the best preconditioner on smooth fields and soft inclusions: 21–73 iterations, about 10× better
    than Green. Counting the dual ×2, it also beats the dual on the smooth field.
  - Spectrally it loses most of that: 119 on the smooth field (5× worse than FE), and on the l = 4h disc it is worse
    than Green.
  - Reason: the FE stencil is local, so diag(K) at a node is the local stiffness, and the Jacobi scaling is exactly
    the local correction that a constant-reference Green operator misses. The spectral derivative is global (its
    kernel decays like 1/r), so diag(K) is a smeared average of s over the neighbourhood, and the scaling no longer
    matches the local stiffness.
- **Green is indifferent to the discretization** (about 430–570 in both).
- **The dual survives the change better** (smooth 49–70, discs 149–238). On the spectral grid it is the best of the
  three everywhere tested, but it is still 2–6× worse than FE Green-Jacobi on these smooth fields.
- **Consequence:** the data-aware choice in the studies (Green-Jacobi for smooth, wide, high-contrast fields) is tied
  to the FE discretization used by muFFTTO. With the classical spectral discretization the options shrink to Green
  vs dual.

## 3. Where this comes from

- **Green ≈ Voigt-type, strain-based:** exact only for homogeneous C.
- **Dual ≈ Reuss-type, stress/compliance-based:** exact for 1D and laminate fields, and near-exact where the soft
  phase forms thin, locally flat bands.
- **The underlying reason:** in 1D every mean-free strain field is compatible. So the compliance-based guess C⁻¹σ is
  automatically a valid strain field, and the dual inverts K exactly. In 2D/3D the compatibility projection is
  non-trivial. The dual stays exact where the field is locally 1D (laminates, flat bands) and loses accuracy where the
  geometry is genuinely 2D/3D: crack tips, crossings, and especially closed inclusions.

## 4. Can we build one that is also good for compact inclusions? (open)

- **Tried: additive combination M = P_dual + P_GJ** (`PRECS=dual+GJ`). It is no better than the dual alone
  (128², d_min = 10: laminate 14, finite 26–27, cross 30–32, inclusion 177–186), and it **fails on the sharp band**:
  no convergence within 2000 iterations.
  - Reason: scale. In soft regions the dual term contains C⁻¹, up to 1e4 larger than the Green-Jacobi term, so the sum
    is the dual.
  - A working combination needs a balanced scaling, or a multiplicative / two-level construction. That is a research
    question, not done here.
- **Practical answer for now:** choose by the data.
  - band-like soft features (cracks, fatigue, battery debond) → dual;
  - compact soft inclusions in a stiff matrix (2D TO holes, sintering pores) → Green-Jacobi;
  - sharp two-phase or under-resolved data → Green.

  This matches every study (fracture slides, TO, sintering).

## References (spectral FFT solvers; check the details)

- J. Zeman, J. Vondřejc, J. Novák, I. Marek, Accelerating a FFT-based solver for numerical homogenization of
  periodic media by conjugate gradients, J. Comput. Phys. 229 (2010).
- J. Vondřejc, J. Zeman, I. Marek, An FFT-based Galerkin method for homogenization of periodic media, Comput. Math.
  Appl. 68 (2014).
- T.W.J. de Geus, J. Vondřejc, J. Zeman, R.H.J. Peerlings, M.G.D. Geers, Finite strain FFT-based non-linear solvers
  made simple, Comput. Methods Appl. Mech. Eng. 318 (2017).
- H. Moulinec, P. Suquet, A numerical method for computing the overall response of nonlinear composites with
  complex microstructure, Comput. Methods Appl. Mech. Eng. 157 (1998).

## Files

| file | what |
|---|---|
| `dual_exact_1d.py` | 1D spectral check (section 1) |
| `crack_orientation_test.py` | 2D FE test (sections 2, 2c). Options: `[N]`; env `DMIN`, `PRECS` (Green, Green_Jacobi, dual, dual+GJ), `MAXIT`, `GEOMS` (band, band2ph, finite, cross, disc, smooth, wide_disc). At 128² or 256² it runs in a few minutes |
| `crack_orientation_results*.csv` | results |
| `spectral_orientation_test.py` | the same test with the spectral (Fourier / collocation) discretization (sections 2b, 2c). Options: `[N]`; env `DMIN`, `PRECS`, `MAXIT`, `GEOMS`. Green-Jacobi with the exact spectral diagonal. 128² in a few minutes |
| `spectral_orientation_results_N*_dmin*.csv` | results |
