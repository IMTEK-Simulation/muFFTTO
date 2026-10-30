# Night log 29/30 Oct 2026: optimizing the solver for third-medium contact (Composite A)

## SUMMARY (for the morning)

**Winner: dual preconditioner (reference lambda/mu = 10) inside the Steihaug CG + trust region measured in the Green
norm + third medium graded inside the voids (k = 1e-4 deep inside -> 1e-3 at the wall, matrix/inclusion sharp) + fast
Hessian code.** 6.4x (CPU) / 6.5x (GPU) faster than Green with the same code, 10.6x faster than yesterday evening's
set-up; response within 2 % of the sharp interface incl. buckling (128 px; at 64 px use a 1-px ramp).

| (third medium 1e-4, simplified material) | linear test 128 px | nonlinear 2 increments |
|---|---|---|
| Green, sharp | 6 661 its | 76 075 CG, 501 s (GPU 128) / 424 s (CPU 64) |
| dual ref 10, sharp | 1 698 its | 15 707 CG, 130 s (CPU 64) |
| **winner (dual ref 10 + graded medium)** | **913 its** | **8 639 CG, 77 s (GPU 128) / 6 746 CG, 67 s (CPU 64)** |

What made the difference (each verified separately):
1. trust region in the Green norm (NuMPI `precond_metric`): the dual works in the nonlinear solver (before: failed).
2. code: closed-form dual middle operator; SRI tangent as explicit 2x2 products; skip the (zero) Q1 Laplacian
   term -> Hessian 13.1 -> 6.9 ms; one dual iteration costs only ~1.2x a Green iteration.
3. dual reference lambda/mu = 10 instead of the matrix 100: -22 %.
4. graded third medium: removes the 1e4 jump at the void wall: dual 1 698 -> 913. Ramps through the matrix also
   help the dual but change the physics (too soft or delayed buckling) -> rejected.
5. dual refresh per load increment (effective moduli from the converged state): -28 to -37 % CG after buckling.

**Caveats found later in the night (see sections 14-17):**
- the gain is largest before/through buckling (6x CG) and shrinks after (1.4x at eps 0.03-0.14, 1.1x beyond):
  near contact the systems get much harder for every preconditioner.
- **the Steihaug CG hits its iteration cap (= number of unknowns) in the contact phase, for Green from eps 0.035 on:
  those Newton steps were inexact** (Newton converged anyway). With the cap raised (exact 1e-6 solves) one
  contact-phase increment costs Green 99 547 CG vs dual + refresh 54 364 (1.8x). Decision needed: raise the cap or
  accept/state the inexact late steps.
- the graded medium with CAP 1e-3 is fine through buckling (< 2 %) but resists slit closure: +10 % stress at eps 0.2
  and no contact by 0.2 (64 px, 2-px ramp). Gentler grade (CAP 3e-4, 1-px ramp at 64 px, sec. 18): physics OK
  (+1.2 % at eps 0.2, contact from 0.195 vs 0.185) but total CG = Green's (2.37 M); 3.7x fewer CG before eps 0.03,
  equal 0.03-0.14, 1.4x MORE after 0.14. Wall 3.69 h vs 5.95 h only thanks to the faster Hessian code.
  => With a medium that gives the right contact, the dual's advantage is limited to the pre-buckling phase.
- full path 64 px (eps 0 -> 0.2): winner 1.63 M CG / 2.75 h vs Green 2.38 M / 5.95 h (2.2x faster; part of it the
  faster Hessian code).

Did not help: Green-Jacobi (any interface treatment), linear N-interpolation of the moduli, augmented Lagrangian,
mixed u-p + MINRES, larger trust-region radii, full anisotropic dual refresh (only ~5 % over the isotropic one),
refresh at every Newton step.
All in `diffuse_interface/stefanus_diffuse.py` (test copy; main script unchanged).

Setting for all tests unless noted: simplified material (inclusion contrast 10, matrix Lambda/mu 100), voids filled with
third medium k = 1e-4 (nu = 0.3), SRI Q1, 1x2 cell, CG stop = energy error 1e-6 with adaptive delay >= 10.
"Linear test" = one Newton system at F = I solved by plain PCG (`--linear_test`). Small grids only (64 / 128 px).

## Starting point (before tonight)

| 128 px, linear test       | Green | Green-Jacobi | dual (K applications) | stress at eps 0.005 (1 increment) |
|---------------------------|-------|--------------|-----------------------|-----------------------------------|
| sharp interfaces          | 6 661 | 19 372       | 2 087 (4 174)         | 0.00166 MPa (reference)           |
| log ramp centred, w 2 px  | 7 179 |              | 1 337 (2 674)         | 0.00049 (3.4x too soft)           |
| log ramp one-sided, w 1   | 6 816 |              | 1 242 (2 484)         | 0.00263 (+58 %)                   |
| log ramp one-sided, w 2   | 6 726 |              | 957 (1 914)           | 0.00346 (2.1x too stiff)          |
| linear N interpolation    | >50k  | >50k         | 5 349 (w 0)           |                                   |

The dual works in the nonlinear trust-region solver when the trust region is measured in the Green norm
(`--tr_metric green`, NuMPI `precond_metric`).

## Tonight

### 1. Cost of one CG iteration (CPU, 64 px, `--time_precond`)
- Hessian application 13.1 ms, of which the SRI tangent contraction (einsum) 9.0 ms, regularization 2.4 ms, the
  muGrid gradient + transpose only 0.5 ms. Green 0.2 ms, dual middle operator (closed form at F = I) 2.1 ms.
- => one dual iteration costs ~1.2x a Green iteration (not 2x): the Hessian dominates both.
- Optimizations in `diffuse_interface/stefanus_diffuse.py` (all checked against the old code to 1e-16, FD check ok):
  - dual middle operator in closed form at F = I: mu (grad v + grad v^T) + lam tr(mean grad v) I
  - SRI tangent: F^-T dF^T F^-T written out for 2x2 tensors (no einsum, no stacking): 9.0 -> 3.0 ms
  - regularization: skip the L^T L term (the Q1 Laplacian is identically zero)
  - Hessian application 13.1 -> 6.9 ms (1.9x), for every preconditioner

### 2. Ramp position (in progress)
Log profile shifted into the circles by c w h (c = 0 centred: too soft; c = 3 one-sided: too stiff). 128 px, stress
after one increment (sharp: 0.00166 MPa) and dual linear-test count:

| w | c = 0.75        | c = 1.5         | dual its (c 0.75 / 1.5) |
|---|-----------------|-----------------|-------------------------|
| 1 | 0.00147 (-11 %) | 0.00186 (+12 %) | 1 420 / 1 387           |
| 2 | 0.00122 (-26 %) | 0.00198 (+19 %) | 1 194 / 1 062           |

Interpolated match at c ~ 1.1 (w 1), 1.2 (w 2). Calibrated, linear test 128 px:

|                    | dual  | Green |
|--------------------|-------|-------|
| sharp              | 2 087 | 6 661 |
| w 1, c 1.1         | 1 435 | 7 121 |
| w 2, c 1.2         | 1 112 | 6 981 |

=> with a stress-matched ramp the dual needs 6x fewer iterations than Green (Green does not react to the ramp).
Load-path check (6 increments to eps 0.03, sharp vs w 2 c 1.2): see below.

### 3. Dual reference medium (64 px, sharp, linear test)
lambda_ref / mu_ref = 100 (matrix, default): 2 291 | 30: 1 936 | 10: 1 795 | 3: 1 869  -> ~10 is best (-22 %).

### 4. Nonlinear wall time, 64 px, sharp, 2 increments, CPU (both still with the old einsum Hessian)
| | Newton | CG | wall |
|---|---|---|---|
| Green | 5 / 6 | 31 537 / 39 417 | 709 s |
| dual + Green-norm trust region, closed-form middle operator | 7 / 6 | 11 961 / 17 604 | **398 s (1.8x faster)** |
Same stress (0.00158 / 0.00312 MPa).

### 5. Combined: dual with reference lambda/mu 10 (+ ramp)
Linear test 128 px: dual ref 10 on sharp 1 698 (vs 2 087 with ref 100); on the stress-matched ramp (w 2, c 1.2): 839,
i.e. 8x fewer than Green (6 981).

Nonlinear, 64 px, SHARP interfaces, 2 increments, CPU, both with the fast Hessian:
| | Newton | CG | wall |
|---|---|---|---|
| Green | 5 / 6 | 31 805 / 38 908 | 424 s |
| **dual, ref lambda/mu 10, Green-norm trust region** | 5 / 6 | 6 905 / 8 802 | **130 s** |
=> 3.3x faster than Green with the same code, 5.4x faster than the starting point (Green + einsum Hessian, 709 s).

### 6. Robustness through buckling (64 px, sharp, dual ref 10 + Green-norm trust region)
| eps | Green (earlier run): Newton / CG | dual ref 10: Newton / CG |
|---|---|---|
| 0.015 (buckling) | 35 / 102 112 | 43 / 46 913 |
| 0.020 | 18 / 82 042 | 24 / 37 509 |
| 0.025 | 12 / 84 237 | 13 / 34 391 |
=> the Green-norm trust region handles the indefinite Hessian; dual still ~2.2x fewer CG after buckling.

### 7. The matrix ramp changes buckling
Load path 128 px (dual): sharp 0.00166 / 0.00329 / 0.00472 (buckled at 0.015); ramp w 2 c 1.2: 0.00168 / 0.00333 /
0.00497 / 0.00659 / 0.00754 / 0.00765 -> not buckled by eps 0.03: the stiff skin around the voids stabilizes the
ligaments. Ramps through the matrix are matched only before buckling -> discarded.

### 8. Graded third medium (matrix and inclusion sharp, `--void_grade CAP`)
Inside each void ln mu rises from ln k (1e-4) deep inside to ln CAP at the wall over ~6 w px (w = 2), lambda/mu of the
medium kept; matrix bit-identical to sharp. Linear test 128 px:
| | dual ref 10 | Green |
|---|---|---|
| sharp | 1 698 | 6 661 |
| graded, CAP 1e-2 | **828** | 6 335 |
| graded, CAP 1e-3 | 913 | 6 571 |
Stress after 1 increment, CAP 1e-2: 0.00169 MPa vs sharp 0.00166 (+1.8 %). Buckling check running.
Load path 128 px (dual ref 10 + Green-norm trust region), stress [MPa]:
| eps | sharp | graded CAP 1e-3 | graded CAP 1e-2 |
|---|---|---|---|
| 0.005 | 0.00166 | 0.00167 (+0.6 %) | 0.00169 (+1.8 %) |
| 0.010 | 0.00329 | 0.00330 (+0.3 %) | 0.00336 (+2.1 %) |
| 0.015 | 0.00472 | 0.00481 (+1.9 %) | 0.00499 (+5.7 %) |
| 0.020 | 0.00480 (buckled) | 0.00489 (+1.9 %, buckled) | 0.00543 (+13 %) |
CAP 1e-3 keeps the response (incl. buckling) within 2 % and the dual at 913 linear-test its (vs 1 698 sharp).
Cost CAP 1e-3, 128 px, per increment Newton / CG: 5 / 3 796, 6 / 4 843, 41 / 21 536 (buckling), 16 / 16 011.

=> WINNER so far: dual (ref lambda/mu 10) + Green-norm trust region + graded third medium (1e-4 -> 1e-3 at the
   walls, w 2) + fast Hessian.

### 9. Wall time of the winner, 64 px, 2 increments, CPU
| | CG | wall | stress eps 0.005 / 0.010 |
|---|---|---|---|
| starting point: Green, einsum Hessian, sharp | 70 954 | 709 s | 0.00158 / 0.00312 |
| Green, fast Hessian, sharp | 70 713 | 424 s | 0.00158 / 0.00312 |
| dual ref 10, sharp | 15 707 | 130 s | 0.00158 / 0.00312 |
| **winner (dual ref 10 + graded CAP 1e-3)** | **6 746** | **67 s** | 0.00159 / 0.00314 |
=> 6.4x faster than Green (same code), 10.6x faster than the starting point.

### 10. Probes of the winner (linear test, dual ref 10)
- grid, graded CAP 1e-3 w 2: 64 px 770 | 128 px 913 | 256 px 1 186 (Green 256 sharp: 6 793). Mild growth: the ramp is
  fixed at 2 px, i.e. physically thinner on finer grids (a fixed physical width would likely be flat).
- sharp 256 px: 1 661 (128 px: 1 698) -> grid independent.
- w 1 at 128 px: 1 096 (w 2: 913).
- softer medium k 1e-5 (128 px): Green 11 319 | dual sharp 5 195 | dual graded to 1e-3: 2 589 (4.4x below Green).
- contact run (winner, 64 px): buckling increment 32 Newton / 17 021 CG vs Green 35 / 101 797 (6x fewer CG).
- ramp at FIXED PHYSICAL width (w = 1 / 2 / 4 px at 64 / 128 / 256): dual 948 / 913 / 1 021 -> grid independent.

### 11. Trust-region radius (Green norm), winner, 64 px, 3 increments through buckling
(delta0, delta_max) = (0.05, 0.5): buckling increment 32 Newton / 17 021 CG, 246 s | (0.2, 2): 37 / 21 351, 267 s |
(0.5, 5): 41 / 17 290, 256 s -> keep the default.
NB: at 64 px with w = 2 the winner is +6 % at eps 0.015 (0.00425 vs 0.00400 sharp); at 128 px (w 2) +1.9 %. Use a fixed
physical width (w = 1 at 64 px).

### 12. GPU wall time, 128 px, 2 increments (fast Hessian)
| | CG | GPU time | stress eps 0.005 / 0.010 |
|---|---|---|---|
| Green, sharp | 76 075 | 501 s | 0.00166 / 0.00329 |
| **winner** | **8 639** | **77 s** | 0.00167 / 0.00330 |
=> 6.5x faster on the GPU (6.4x on the CPU).

### 13. Probes
- reference lambda/mu on the graded medium (128 px, linear): 3: 1 176 | 10: 913 | 30: 997 -> 10 stays best.
- paper material (contrast 1000, Lambda/mu 1000) + graded medium, 128 px, linear: Green > 50 000 (not converged) |
  dual ref 10: 9 492 | dual ref 100: 9 550 -> the dual works there too, but the simplified material is 10x cheaper.
- Green-Jacobi on the graded medium (128 px, linear): 19 011 (sharp 19 372) -> no help.
- dual-norm trust region with the winner (64 px, 2 increments): identical counts to the Green norm before buckling
  (the radius never binds there); the Green norm stays the safe choice for buckling.
- winner with the 1-px ramp at 64 px (fixed physical width), 4 increments through buckling: stress 0.00158 / 0.00313
  / 0.00409 / 0.00417 vs sharp 0.00158 / 0.00312 / 0.00400 / 0.00408 (+2.2 % after buckling; w 2: +6 %);
  52 434 CG vs 100 129 (dual, sharp) and ~255 000 (Green, sharp).

### 14. The dual degrades with deformation -> refresh it
Per Newton step, contact run 64 px (Green earlier run vs winner): eps 0.005: 6 307 vs 600 | 0.025: 7 020 vs 1 068 |
0.05: 5 641 vs 2 317 | 0.08: 5 643 vs 4 669 -> the dual (built at F = I) loses its advantage as the voids flatten:
the compressed third medium stiffens ~1/J.
Refresh (`--dual_refresh`): at each load increment, from the last converged state, per pixel mu_eff = max(mu - lam ln
J0, 1e-3 mu) / J0, lam_eff = lam / J0 (isotropic part of the neo-Hookean tangent); trust region stays in the fixed Green
norm. Test: one increment from the eps = 0.08 frame (`--restart_frame`, reproduces the original run: 43 909 vs 44 042):
| | Newton | CG | CG / Newton |
|---|---|---|---|
| no refresh | 9 | 43 909 | 4 879 |
| refresh per increment, mu_eff from J0 | 9 | **27 784** | 3 087 |
| refresh per increment, mu_eff with tr(b0)/2 | 9 | 27 579 | 3 064 |
| refresh at every Newton step | 9 | 36 891 | 4 099 (worse: the first iterate is the predictor state) |
| Green (earlier run) | 10 | ~56 400 | 5 640 |
| full anisotropic dual of the current SRI tangent (C_ref abs-pinv(A) C_ref, `--dual_refresh_mode full`) | 11 | 26 353 | 2 396 |
-> only ~5 % below the isotropic refresh: the growth after buckling is not (only) the dual's middle material; the Green
part stays the F = I reference and the Newton steps go through negative curvature. Keep the isotropic J refresh.
(full mode verified: first increment from F = I 2 982 CG vs 3 002 without refresh.)

### 15. Where the CG goes in the contact phase (trace, one increment from eps 0.115, winner, no refresh)
Newton steps 1-5: 2-90 CG each (trust-region limited; 2 trial points inadmissible, J <= 0). Steps 6-9: full solves to
1e-6 with 11 620-13 292 CG each (vs ~600-900 per full solve at the start of the load path). The linear systems become
much harder near contact: the squeezed third medium stiffens ~1/J^2 (the contact barrier), orders of magnitude, in thin
oriented zones. Neither the F = I dual nor the refreshed (isotropic or full anisotropic) dual follows this; Green is hit
just as hard (~5 600 CG / Newton step there).
Contact phase, CG per increment (64 px): eps 0.115: winner 44 111 | winner + refresh 44 177 | Green (earlier) ~56 400.

### 16. IMPORTANT: the Steihaug CG hits its iteration cap in the contact phase
The default cap of the Steihaug CG (NuMPI `inner_maxiter`) is the number of unknowns: 2*64*110 + 1 = 14 081 at 64 px.
- Green, earlier contact run (k 1e-4): 56 399-56 472 CG per increment from eps 0.035 on = 4 capped solves (4 x 14 081 =
  56 324) + ~100 for the short steps; 70 533 from eps 0.185 = 5 capped solves. -> Green's late Newton steps never
  reached the 1e-6 energy tolerance (Newton converged anyway).
- winner: 56 432 per increment from eps ~0.15 on -> also capped there. The winner delays the onset of capping from
  eps 0.035 (Green) to ~0.15; that is where its savings come from.
Linear solve at the CONVERGED state eps 0.15 (Hessian positive definite there), cap 100 000 (`--linear_test_state`):
| | CG to 1e-6 | at F = I |
|---|---|---|
| Green | 22 528 | 6 661 (128 px) |
| dual ref 10 | 10 786 | 913 (128 px) / 770 (64 px) |
| dual ref 10 + refresh | 7 762 | |
-> no stagnation, just slow: the systems near contact need more than the cap. The dual stays 2-3x below Green but
   degrades more than Green from F = I (12x vs 3.4x). The refresh helps (-28 %); in the nonlinear run this was hidden
   by the cap. DECISION NEEDED: raise `inner_maxiter` (exact 1e-6 solves, comparable counts, more cost) or keep the
   cap and state that the late Newton steps are inexact.

### 17. Full contact path, 64 px (winner, w 2, CAP 1e-3, no refresh) vs Green (plain medium, earlier run, old Hessian)
| | Green | winner |
|---|---|---|
| CG eps <= 0.03 | 409 506 | 68 694 (6.0x fewer) |
| CG 0.03-0.14 | 1 241 476 | 893 988 (1.4x) |
| CG > 0.14 | 733 866 | 662 972 (1.1x; both capped) |
| total CG / wall | 2.38 M / 5.95 h | 1.63 M / 2.75 h (2.2x faster) |
| stress eps 0.20 | 0.00887 MPa | 0.00974 MPa (+10 %) |
| contact (min J < 0.1) | from eps 0.185 | not reached (min J 0.13 at 0.20) |
PHYSICS CAVEAT: the graded medium (up to 1e-3 at the walls) resists slit closure -> +10 % stress at eps 0.2 and delayed
contact (at 64 px the 2-px ramp is also physically 2x too wide). CAP 1e-3 is fine through buckling (< 2 %), too stiff
for contact.
One contact-phase increment (eps 0.15 -> 0.155) with the cap raised to 60 000 (exact solves), GPU, 64 px:
Green 99 547 CG / 541 s | dual 65 031 / 509 s | dual + refresh 54 364 / 435 s (same stress 0.00777).
Gentler grade, 64 px, linear test: w 1 CAP 3e-4: 1 175 | w 1 CAP 1e-3: 948 | sharp: 1 706.
Running: full path 64 px with w 1, CAP 3e-4, refresh (`log_gentle_contact64.txt`) to check stress / contact at large strain.

### 18. Full contact path, 64 px, gentle grade (w 1, CAP 3e-4, dual ref 10 + Green-norm TR + refresh)
| | Green (plain medium) | winner CAP 1e-3 (sec. 17) | gentle CAP 3e-4 |
|---|---|---|---|
| CG eps <= 0.03 | 409 506 | 68 694 | 111 696 (3.7x fewer than Green) |
| CG 0.03-0.14 | 1 241 476 | 893 988 | 1 233 456 (1.0x) |
| CG > 0.14 | 733 866 | 662 972 | 1 024 233 (0.7x, worse) |
| total CG / wall | 2.38 M / 5.95 h | 1.63 M / 2.75 h | 2.37 M / 3.69 h |
| stress eps 0.20 | 0.00887 MPa | 0.00974 (+10 %) | 0.00898 (+1.2 %) |
| contact (min J < 0.1) | from eps 0.185 | not reached | from eps 0.195 |
-> the gentle grade gets the physics right (stress +1.2 %, contact one increment later), but its CG advantage is gone
   after buckling: total CG equal to Green, the remaining 1.6x wall-clock gain is only the faster Hessian code (the Green
   run used the old code). The cheap phase is the one before buckling; in the contact phase all set-ups are expensive.
   Conclusion: with a physically acceptable medium, the dual + grading helps only up to buckling. Options for the
   contact phase are still open (switch back to Green after buckling = free hybrid; or raise inner_maxiter first to get
   honest exact-solve counts before comparing).
