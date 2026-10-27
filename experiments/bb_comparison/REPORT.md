# Barzilai-Borwein vs (trust-region) Newton-CG in muFFTTO

Comparison of the basic Barzilai-Borwein scheme of Schneider (2019, *Int. J. Numer. Meth. Engng.* 118:482-494) with
our Newton-CG and trust-region Newton-CG, on every applicable experiment, with all three of our preconditioners:
Green (C_ref = solid tangent), Green-Jacobi (C_ref = 2 mu I, D^-1/2 G D^-1/2) and dual G K(C0 A^-1 C0) G.
Everything here is reproducible from this folder (2026-10-25).

## 1. The method and our implementation

Schneider's BB scheme is preconditioned gradient descent, x_{k+1} = x_k - alpha_k G grad f(x_k), with no line search
and the Barzilai-Borwein step

    BB1: alpha = (s, M s) / (s, y)          BB2: alpha = (s, y) / (y, M^-1 y)
    s = x_{k+1} - x_k,  y = grad f(x_{k+1}) - grad f(x_k),  M = G^-1 (preconditioner)

For a quadratic it costs one gradient (one material evaluation + B/B^T) and one preconditioner application per
iteration, the same count as one CG iteration, and it never needs the tangent. Schneider reports BB about 1.5x the CG
iteration count on linear problems, and recommends the plain (non-globalized) scheme.

**Library:** `muFFTTO/solvers_nonlinear.py`, `minimize(..., method='bb')` → `barzilai_borwein()`.

- `variant=1|2` selects the BB1 or BB2 step.
- `globalize=False` (default) gives plain BB.
  - `'energy'` is the nonmonotone GLL test f_new ≤ max(last 10 f) − θ t (g, z).
  - `'gradient'` is the Fletcher test on ‖g‖_G.
- Inadmissible states (det F ≤ 0, energy above `inadmissible`) are always backtracked (t ← t/2).
- `converged(x, g, z)` lets a driver use its own stopping test; the default is max|g| ≤ gtol.
- The last step length is returned in `res.alpha`, and contact drivers warm-start the next increment with it.
- Any of our preconditioners plugs in through `precond=`.

**Globalization on a quadratic test** (κ = 1e4, n = 400; CG needs 1200 iterations):

| BB variant | iterations | backtracks | converged? |
|---|---|---|---|
| plain BB1 | 1511 | 0 | yes |
| plain BB2 | 1541 | 0 | yes |
| GLL (energy) BB1 | 3191 (5161 evaluations) | 1969 | yes |
| GLL (energy) BB2 | 2782 | 71 | yes |
| Fletcher (gradient) BB1 | did not converge in 5000 | 28 556 | no |
| Fletcher (gradient) BB2 | did not converge in 5000 | 3 281 | no |

The energy test stalls near the minimum, because the energy differences drop to round-off. This agrees with Schneider:
plain BB is the default.

## 2. Linear elasticity P1–P3 (128², phase contrast 1e-6 … 1e6)

**Setup:**

- Script: `linear_P123.py`, a copy of the paper-1 setup; results in `linear_data/`, table in `linear_table.md`,
  figure `linear_cg_vs_bb.png`.
- Error measure: the exact energy error e_k/e_0 for CG and for BB1/BB2.
- Cost: one iteration costs the same for both methods (one K product + one preconditioner application).

**BB / CG iteration ratio** to reach e_k/e_0 ≤ 1e-10 (BB1), over κ = 1e-6 … 1e6:

| preconditioner | P1 | P2 | P3 |
|---|---|---|---|
| Green | 1.2–2.9 | 1.4–20 | 1.2–2.4 |
| dual | 1.7–3.0 | 1.2–3.6 | 1.2–2.9 |
| Green-Jacobi | 2.0–28 | 2.4–3.4, **3 of 6 never reach 1e-10** | 2.2–5.3 |

- With the Green and dual preconditioners BB is typically 1.3–3× CG. This is consistent with the paper's ~1.5×.
- **Dual is the best preconditioner for BB as for CG.**
  - It has the smallest iteration counts in almost every case: P2 κ=1e6 takes 147 CG / 529 BB iterations, against
    238/798 for Green and 7990/no convergence for Green-Jacobi.
- **Green-Jacobi is a poor partner for BB.** The BB/CG ratio grows to 28 (P1, κ=1e4).
  - On P2 with κ ∈ {1e-6, 1e4, 1e6}, BB had not reached 1e-10 after 50 000 iterations.
  - This is a slow tail, not round-off stagnation. P2 GJ κ=1e6 is still at 4e-7 after 50 000 BB iterations, while
    CG reaches 1e-12 in 11 372.
  - BB uses only two-point spectral information. A wide spread of eigenvalues in the preconditioned operator, which
    is what GJ leaves on these geometries, costs BB much more than it costs CG.
- At the engineering tolerance 1e-6, the ratios are similar or smaller.

## 3. Finite-strain P1–P3 (neo-Hookean, contrast 1e±3, 32² and 64²)

**Setup:**

- Script: `finite_strain_P123.py --solver newton|bb1|bb2`; table `fs_table.md`, figure `fs_newton_vs_bb.png`.
- Load: 10 increments of macroscopic shear plus tension.
- Newton side: CG energy stop 1e-6, delay 10.
- BB side: BB per increment, with the preconditioner rebuilt at the start of the increment, stopping when
  ‖∇(G0 g)‖/‖F‖ < 1e-8.
- Work counts:
  - Newton = CG iterations + Newton steps;
  - BB = gradient evaluations.
- **The mean stress history from BB matches Newton to 1e-11 … 6e-6** in every completed case.

**Medians over the 10 completed (geometry, κ, N) cases per preconditioner:**

| preconditioner | work, best BB / Newton | wall time, BB2 / Newton | BB cheaper in work |
|---|---|---|---|
| Green | 0.45 | 1.46 | 9 / 10 |
| Green-Jacobi | 0.85 | 2.78 | 5 / 10 |
| dual | **0.40** | **0.99** | **10 / 10** |

Findings:

- **BB needs fewer operator applications than Newton-CG.** The difference is up to 5× (P2 κ=1e-3 32², Green:
  23 631 against 4435).
  - Newton pays for re-solving every linearization to 1e-6, and BB does not.
  - With the dual preconditioner BB is cheaper in every case.
- **In wall time BB is not faster here.** A BB iteration re-evaluates the constitutive law. A CG iteration applies a
  stored tangent, which is ~2.5× cheaper per application in our code.
  - Only with dual (whose application cost dominates both) does BB reach parity.
  - Schneider makes the same point the other way round: BB wins when the tangent is expensive to form or store.
- **Green-Jacobi + BB is again weak.** P1 κ=1e3 64²: 44 031 BB evaluations against 5403 for Newton.
- **Robustness:**
  - P2 κ=1e-3 64² (soft inclusion that buckles): Newton-CG stops at increment 3 with an indefinite Hessian for all
    three preconditioners. BB1/BB2 complete the load path with every preconditioner.
    - BB with f = 0 only seeks a stationary point; it does not test stability.
  - P2 κ=1e3 64² fails for all solvers (elements invert at increment 6–7; dual crashes in the eigen-decomposition).
    This is a problem limit, not a solver difference.

## 4. Third-medium contact (32², all at K_V = 1e-6, α = 1e-7, HuHu regularization only in the gap)

**Problems:**

- patch test (40 increments);
- flat punch (B = 6 px, 40 increments);
- Hertz cylinder on an adapted grid (35 increments);
- disc between plates (adapted, grade 0.6, 50 increments).

**Setup:**

- 72 jobs: problems × {Green, Green-Jacobi, dual} × {TR Newton-CG, Newton-CG, BB1, BB2, BB1 with gtol/10,
  BB1 + GLL}.
- Run with `run_queue.sh jobs_32.txt`; tables in `contact_table.md` (`summarize_contact.sh` → `compare_runs.py`);
  figure `contact_work.png`.
- Work = Hessian-vector products + gradient evaluations, over accepted increments.
- dP = force deviation at common load levels.
- du = final displacement deviation.
- Reference: TR Newton-CG with Green-Jacobi.

**Work over the whole load path (best variant of each family):**

| problem | TR Newton-CG GJ | Newton-CG GJ | best BB (GJ) | BB dP / du | Newton G | best BB (G) |
|---|---|---|---|---|---|---|
| patch | 3 179 | 3 140 | 2 329 (BB2) | 2.9e-3 / 1.4e-4 | 3 458 | 3 476 (BB2) |
| punch | 3 016 | 2 963 | 900 (BB2) | 2.1e-4 / 1.2e-5 | 3 388 | 1 299 (BB2) |
| hertz | 15 091 | 22 925 | 5 338 (BB2) | 1.5e-3 / 3.6e-5 | 91 119 (TR) | 74 805 (BB2) |
| disc | 24 187 | 30 700 | 9 238 (BB2) | 2.6e-3 / 8.1e-4 | 98 093 (TR) | 11 226 (BB2) |

Findings:

- **On contact, BB2 with Green-Jacobi is the cheapest combination on every problem.** It does 1.4–3.4× less work
  than TR Newton-CG GJ.
  - Wall time: 15.5 against 15.7 s, 8.6 / 20, 46 / 78, 62 / 127 s.
  - Here a BB iteration costs about the same as a Newton-CG iteration, because the third-medium material evaluation
    is cheap relative to the FFT/preconditioner.
- **BB with Green is poor where the void is large** (Hertz: 75 000–430 000 evaluations, du = 13%).
  - The Green reference uses the solid stiffness everywhere, so the gradient in the K_V = 1e-6 void is ~1e-6 smaller.
  - An absolute gradient stop accepts void displacements that are far from converged.
  - Green-Jacobi scales the void by its own stiffness and does not have this problem. This is the same mechanism
    that makes GJ 6× better than Green for Newton-CG on Hertz.
- **Accuracy at equal gtol (3e-6): BB is looser than Newton.** dP is ~1e-3 for BB against ≤ 1e-9 for Newton.
  - Newton's quadratic convergence overshoots the tolerance; BB stops just at it.
  - With gtol/10 and GJ, dP drops to 2e-5 … 3e-4. The work changes by −13% (punch) to +40% (patch).
  - On the disc, all BB runs (BB2, and BB1 with gtol/10) show the same 2.6e-3 force deviation at λ ≈ 0.88–0.92.
    This is not a tolerance effect: BB lands on a neighbouring equilibrium while a new contact row closes, and it
    rejoins the reference by λ = 1 (du 3e-4 … 8e-4).
- **The GLL globalization never pays off.** It costs 1.2–2.2× plain BB1, except one cheaper disc run whose discarded
  failed attempt is not counted in the work. On the disc a GLL attempt hit the 50 000-iteration cap
  (539 s against 98 s for plain BB1).
- **Dual on contact:**
  - Newton-CG/TR: 12–40× the GJ work (patch 118 943; punch 36 571). Hertz and disc did not finish λ = 1 within the
    2 h limit (stopped at λ ≈ 0.6–0.7 after 1.3–1.5 million Hessian products).
  - **BB + dual does not converge at all:** 50 000 iterations without reaching gtol at λ = 0.0008 on the patch test,
    so the other dual-BB jobs were skipped.
  - The dual operator G K(C0 A^-1 C0) G has A^-1 ~ 1/K_V in the void. Its spectrum then spans ~1e6, which two-point
    BB steps cannot handle. Dual is the right preconditioner for bounded-contrast composites, not for a near-empty
    third medium.

## 5. Recommendations

1. **Contact (third medium): use BB2 + Green-Jacobi, plain, warm-started step, gtol ≈ 3e-7.**
   - It is cheaper than TR Newton-CG in work and time on all four benchmarks, at force accuracy ~1e-4.
   - Keep TR Newton-CG + GJ as the reference / fallback when tight equilibrium is needed (fracture staggering,
     convergence studies).
2. **Finite-strain composites: BB with the dual preconditioner** needs 2.5× fewer operator applications than
   Newton-CG.
   - It reaches wall-time parity now, and would win once the constitutive evaluation is cheaper (GPU, vectorized
     laws) or the tangent is expensive (plasticity, damage).
3. **Linear problems: keep CG.** BB is 1.2–3× slower in iterations with the best preconditioner (dual), as the paper
   reports.
4. **Do not combine BB with Green-Jacobi on high-contrast composites**, nor dual with a third medium. In both cases the
   preconditioned spectrum is too wide for two-point steps.
5. Default to plain BB; never use gradient-norm (Fletcher) globalization. Energy (GLL) only as a safeguard against
   inadmissible states, which plain BB already handles by halving.

## Caveats

- Contact at 32² only (per the no-big-runs rule). The 64² contact subset of the best combinations is a natural
  next step.
- Wall times come from 7 (contact) or 3 (finite strain) concurrent single-thread jobs on 16 cores, so they are
  indicative only.
- Work counts only accepted increments; rejected (halved) attempts appear in wall time only.
- AT2 fracture was not run with BB. Its staggered damage solve is unchanged, and the mechanics step is the same
  unconstrained problem as contact.

## Files

| file | content |
|---|---|
| `linear_P123.py`, `run_linear.sh`, `plot_linear.py` | linear CG vs BB |
| `finite_strain_P123.py`, `run_fs.sh`, `jobs_fs.txt`, `plot_fs.py` | finite strain Newton vs BB |
| `{patch,punch,hertz,disc}_contact.py` + geometries, `run_queue.sh`, `jobs_32.txt` | contact drivers (`MECH_SOLVER=bb BB_VARIANT=1/2 BB_GLOBALIZE=energy GTOL=...`) |
| `compare_runs.py`, `summarize_contact.sh`, `plot_contact.py` | contact tables and figure |
| `linear_table.md`, `fs_table.md`, `contact_table.md` | full tables |
