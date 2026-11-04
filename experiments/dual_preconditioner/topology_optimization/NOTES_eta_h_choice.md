# Choosing η and h for the phase-field topology optimisation

**Rule in one line** (unit cell, grid N², h = 1/N, contrast 1e4):

    4/N  ≤  η  ≤  0.035/P  ≈  1/128   (this target, P ≈ 4.5)

- Figure: `figures/nemo/TO_eta_window.png`, made by `python plot_TO_eta_window.py [csv] [P]`. It shows the window in the (N, η) plane, with all contrast-1e4 runs drawn on top.
- The window is non-empty only for **N ≥ 512**.
- Inside it, take the lower end, **η = 4/N**: a smaller η is closer to the sharp-interface optimum.
- The absolute minimum is η = 2/N, which leaves a design difference of up to about 10 %.

**Rule.** η is the only free parameter: the weights are fixed, with stress match 5 and F_pf 1. Two conditions:
1. **η ≥ 4h**, so the interface is resolved and the design is grid-converged. η ≥ 2h is the minimum.
2. **η ≤ 0.035 / P**, so the interface is thin and F_pf ≈ P/3 is a pure perimeter. Take P from a coarse run; it hardly depends on η.

For this target (P ≈ 4.5) that means η ≤ 1/128, so N ≥ 512 with η = 4/N.

**What P is.** P is the total length of all interfaces in the unit cell: the length of the ρ = 0.5 contour, where the design switches from soft to stiff. It is measured on the final design, with the cell side as the unit of length (cell 1 × 1, cell outline = 4).
- **This target:** four diamond-shaped soft inclusions per cell, each with a boundary of about 1.1–1.2, counting the pieces cut by the periodic boundary. So P ≈ 4.5 (measured 4.46–4.69, depending on η). The interfaces in one cell are about 4.5 cell edges long.
- **Why it enters the rule:** the interface band is about 4.4 η wide and runs along all interfaces. It therefore covers an area of about **4.4 · η · P** of the cell:

  | η | band area (P = 4.5) |
  |---|---|
  | 1/128 | 16 % of the cell |
  | 1/64 | 31 % |
  | 1/32 | 61 % |

  Condition 2 keeps the band below about 15 % of the cell, so the interfaces do not crowd each other and F_pf measures the perimeter. For a cell of side L the condition is η · P ≤ 0.035 L².
- **How to get P for a new target:** column `P` of `design_convergence.csv`, written by `analyze_TO_design_convergence.py` (bilinear contouring of ρ = 0.5 on the periodic cell). A coarse run is enough, because P changes by only about 5 % with η.

Below that bound, a smaller η still moves the design towards the sharp-interface optimum. Each halving of η improves the stiffness error by about 0.025 and adds 1.5–2 % perimeter. η does **not** set how strongly the perimeter is penalised; that strength is fixed by the weights.

Results of the η / contrast study on NEMO2 (2026-11-03/04). The 100 final designs cover:
- contrasts 1e2 … 1e5;
- grids 32² … 1024²;
- η = 1/4 … 1/1024.

Data: `$MUFFTTO_WS/runs/TO_study/analysis_contrast/design_convergence.csv`, with a copy in `figures/nemo/`.

Figure: `figures/nemo/TO_perimeter_indicator_readable.png`, made by `python plot_TO_perimeter_indicator.py`.
- Left: R against the band area.
- Middle: interface width against the band area.
- Right: grid convergence at fixed η.

The figure replaces the hard-to-read `TO_perimeter_indicator.png` of `analyze_TO_design_convergence.py`.

## What is being measured

The objective contains the phase-field term

    F_pf = η ∫|∇ρ|² + (1/η) ∫ ρ²(1−ρ)²

For a sharp, well-resolved two-phase design, theory (Modica–Mortola) gives:
- **F_pf → P/3**, where P is the length of the ρ = 0.5 contour (the perimeter);
- an interface profile ρ(s) = 1/(1+e^(−s/η)), whose 0.1–0.9 band is 2 ln 9 · η = 4.39 η wide.

Quantities per final design:
- **R = 3 F_pf / P.** R = 1 means F_pf is exactly a perimeter penalty, weighted 1/3.
- **Band area = 4.39 · η · P**, the share of the unit cell covered by the interface band. η·P measures how thick the interface is compared with the features of the design.
- **w / η**, the measured 0.1–0.9 width (band area / P) relative to η. Theory gives 4.39.
- **RMS(ρ_N − ρ_finest)**, the difference to the finest grid at the same η, compared after the best periodic shift. It measures grid convergence.

## Result: two independent conditions

### 1. Thin band: η·P ≲ 0.035, i.e. band ≲ 15 % of the cell

This condition decides whether F_pf is a perimeter.
- All designs, at every contrast and grid, collapse onto one curve of R against band area.
- What controls the curve is band area, not resolution.

Contrast 1e5, finest grid of each η (perimeter P ≈ 4.5):

| η | grid | η/h | η·P | band [% cell] | R = 3F_pf/P | w/η (theory 4.39) |
|---|---|---|---|---|---|---|
| 1/32 | 1024² | 32 | 0.139 | 61 | 1.107 | 3.65 |
| 1/64 | 1024² | 16 | 0.071 | 31 | 1.050 | 3.92 |
| 1/128 | 1024² | 8 | 0.036 | 16 | 1.021 | 4.14 |
| 1/256 | 1024² | 4 | 0.018 | 8 | 1.009 | 4.24 |
| 1/512 | 512² | 1 | 0.009 | 4 | 0.998 | 4.28 |

What this shows:
- **Wide bands (η ≥ 1/64):** the band covers ≥ 30 % of the cell. Neighbouring interfaces overlap, so the profile is squeezed to 3.3–3.9 η and R is 5–11 % above 1. F_pf then acts as a smoothing regulariser rather than a perimeter penalty.
- **Thin bands (η ≤ 1/128):** R = 1.00–1.02 and the width is within 6 % of theory.
- **Grey designs** (no two-phase split, grey crosses in the figure) have R ≈ 0.75 and lie outside the theory. Most are at η ≥ 1/16 on coarse grids.

### 2. Resolved band: η ≥ 2h, better 4h

This condition decides whether the design is grid-converged at fixed η (contrast 1e5, right panel).

| η/h | RMS(ρ_N − ρ_finest) | comment |
|---|---|---|
| 1 | 0.14–0.29 | different design, snapped to the lattice (e.g. 1/128 at 128² has a different topology) |
| 2 | 0.02–0.10 | close |
| 4 | 0.02–0.09 | close; η = 1/32 is the slowest, its wide band still squeezes |
| 8 | ≈ 0.02 | converged; this ≈ 2 % floor is the optimizer's own scatter, from the 500-iteration cap and the L-BFGS restarts at segment boundaries |

Resolution alone does not make F_pf a perimeter: at η/h = 32 with η = 1/32, R is still 1.11. Condition 1 is needed as well.

## Recommendation

Choose **η from the feature size**, so that η·P ≲ 0.035, i.e. η ≈ 0.035 / P. Then choose **h ≤ η/4**, or h ≤ η/2 if the grid is limited.

For these designs (P ≈ 4.5):

| η | grid at η/h = 4 (minimum: η/h = 2) | verdict |
|---|---|---|
| 1/64 | 256² (128²) | smoothing regulariser, R = 1.05; acceptable only if a perimeter penalty is not the goal |
| **1/128** | **512² (256²)** | **sweet spot: R = 1.02, w = 4.1 η, design within 2–5 %** |
| 1/256 | 1024² (512²) | cleanest (R = 1.01), but needs 1024² |

## Which contrast is meaningful

The soft phase stands in for void. The contrast k = E_stiff / E_soft is high enough once the design no longer changes with k.

Figure: `figures/nemo/TO_contrast_effect.png`, made by `python plot_TO_contrast_effect.py`.
- Left: stiffness error against k.
- Middle: perimeter against k.
- Right: CG work per solve against k at η = 4h.

Final designs at η = 4h:

| grid | | 1e2 | 1e3 | 1e4 | 1e5 |
|---|---|---|---|---|---|
| 1024² (η = 1/256) | stiffness error | 0.295 | 0.162 | 0.155 | 0.151 |
| 1024² (η = 1/256) | P | 4.81 | 4.73 | 4.69 | 4.69 |
| 512² (η = 1/128) | stiffness error | 0.309 | 0.153 | 0.176 | 0.174 |
| 512² (η = 1/128) | P | 3.84 | 4.73 | 4.63 | 4.62 |

What the contrasts give:
- **1e2 is not meaningful.** The soft phase carries load, so the optimizer finds other designs: the stiffness error is 1.5–2× larger, P is off by up to 20 %, and some topologies differ.
- **1e3 is nearly converged.** It is within about 0.01–0.02 of the 1e4 stiffness error, with some run-to-run scatter.
- **1e4 is converged.** It agrees with 1e5 to about 0.003 in stiffness error and 0.2 % in P.
- **1e5 only costs more:** 2–3× the work per solve of 1e4, for every preconditioner.

The same physical η on two grids gives the same curves (η = 1/64: 256² at 4h and 512² at 8h), which confirms grid convergence.

**Use contrast 1e4**, or 1e3 for a cheaper approximation.

## Preconditioners at the recommended setting (η = 4h)

CG work per linear solve (dual counted ×2), Green / Green-Jacobi / dual. The cheapest is in bold. Green was not run at 1024²: its work per solve is flat in N, so the bracketed value is its 512² value.

| contrast | 128² | 256² | 512² | 1024² |
|---|---|---|---|---|
| 1e2 | 53 / 53 / **49** | 56 / 77 / **55** | **54** / 151 / 60 | (≈55) / 315 / **64** |
| 1e3 | 153 / **73** / 84 | 157 / 108 / **107** | 147 / 184 / **123** | (≈150) / 324 / **134** |
| 1e4 | 463 / **90** / 148 | 466 / **149** / 227 | 444 / **262** / 316 | (≈450) / 478 / **374** |
| 1e5 | 972 / **91** / 255 | 1377 / **201** / 515 | 1303 / **371** / 799 | (≈1300) / **831** / 1126 |

How the three scale:
- **Green:** cost flat in the grid, but steep in contrast (≈ k^0.45). Only competitive at contrast 1e2, where Green ≈ dual.
- **Green-Jacobi:** nearly contrast-independent (≈ k^0.15), but its cost roughly doubles with every grid refinement (h^-1). **Cheapest at high contrast (1e4–1e5) up to 512²**, by 1.2–2.2× over the dual.
- **Dual:** grows with contrast (≈ k^0.4) and mildly with the grid (≈ 1.1–1.5× per refinement). Cheapest at contrast 1e2–1e3 on fine grids, and at 1e4 from 1024².

**Crossover dual ↔ Green-Jacobi on the grid:**
- contrast 1e3: about 256²;
- contrast 1e4: about 1024²;
- contrast 1e5: beyond 1024², extrapolated around 2048²;
- contrast 1e2: the dual or Green win on every grid ≥ 256².

η/h = 8 gives the same picture with lower numbers (`contrast_summary.csv`).

Caveat: Green-Jacobi can break down at the extreme. At 1e5 on 1024² with η = 1h, CG hit maxiter in every solve; at 1e5 with η = 2h it needed about 22k CG per solve. At η ≥ 4h no breakdown was seen.

## Link to the preconditioner study

- **Green** is cheap only on nearly two-phase quadrature points, i.e. small η/h. At η/h = 1 Green ≈ dual.
- **Green-Jacobi** is nearly independent of contrast, but its cost grows strongly with the grid.
- **Dual against Green only:** the dual beats Green from contrast 1e3 on, and the more so the larger η/h is. Against Green-Jacobi, see the table above: at high contrast Green-Jacobi is cheaper up to 512².

  Green / dual work per solve (dual counted ×2; > 1 means the dual is cheaper), from `figures/nemo/contrast_summary.csv`:

  | grid | η/h | 1e2 | 1e3 | 1e4 | 1e5 |
  |---|---|---|---|---|---|
  | 256² | 2 | 0.90 | 1.16 | 1.39 | 1.64 |
  | 256² | 4 | 1.02 | 1.46 | 2.05 | 2.68 |
  | 512² | 2 | 0.87 | 1.03 | 1.10 | 1.17 |
  | 512² | 4 | 0.91 | 1.19 | 1.41 | 1.63 |
  | 512² | 8 | 0.98 | 1.42 | 2.00 | 2.75 |

  At the recommended setting (η = 1/128 on 512², η/h = 4) the dual is 1.2–1.6× cheaper from contrast 1e3 on, and Green is about 10 % cheaper at 1e2. The dual's lead shrinks on finer grids at fixed η/h.

  See `figures/nemo/TO_contrast_green_dual_ratio.png` and `TO_contrast_vs_k.png`.

## Caveats

- **Few runs actually converged.** The L-BFGS stop is `GTOL=0` with `xtol = 1e-3` on the phase field, capped at 500 iterations. All η ≤ 2h runs at 1024², and several 4h runs, hit the cap. Their final designs are the iterate at step 500, though they hardly change any more.
- **Only one family of designs is covered:** one load case set (target stiffness) and one volume fraction. P ≈ 4.5 is specific to this target. For other targets, apply the rule η·P ≲ 0.035 with that design's own perimeter.

## Objective normalisation: phase-field term = perimeter (2026-11-04)

The study above used the raw bracket with weight 1, against a stress term of weight 5/3 per load case:

    f = (5/3) Σ_Γ ‖σ̄ − σ*‖²/‖σ*‖²  +  [η ∫|∇ρ|² + (1/η) ∫ρ²(1−ρ)²]

Here the bracket is ≈ P/3, not P.

muTopOpt (≥ v1.1.0) normalises the phase-field term so that its weight is the penalty per unit relative interface measure:

    f = a Σ_Γ ‖σ̄ − σ*‖²/‖σ*‖²  +  w / (c_W V^((d−1)/d)) · [bracket],   c_W = 1/3

As η → 0 the phase-field term → w · P, where P is the interface length in 2D or the interface area in 3D, in units of the cell size.

`example_2D_elasticity_TO.py` now has this as an option:
- `PF_NORMALIZED=1` switches it on;
- `PF_WEIGHT` sets w (default 0.2);
- `STRESS_WEIGHT` sets a (default 1).

**w = 0.2 with a = 1 has the same minimiser as the old objective**, and its value is 3/5 of the old one. This was checked at 32²: the first objective and gradient are exactly 3/5 of the old values. Runs with the new objective are named `..._pfw<w>_...` instead of `..._w_5.0_...`.

The 2D default stays `PF_NORMALIZED=0` while the old study still runs on NEMO: its launchers and analysis scripts expect the old objective and file names. The 3D script uses the normalised form by default.

muTopOpt's default w = 1 penalises the perimeter 5× more than this study. That gives a smaller P, so the upper bound 0.035/P on η moves up.
