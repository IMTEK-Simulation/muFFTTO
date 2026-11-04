# AT2 phase-field fracture at practical crack resolution (l/h = 2-4): dual vs Green vs Green-Jacobi

Study of 2026-10-31 / 11-01 in `damage_AT2_small_strain/`. Hybrid AT2 (g(d) = (1 - d)^2 + k, k = 1e-4, Amor split in
the history field only), staggered scheme (tol 1e-3), E_xx = 0.001 ... 0.08 in 80 steps, stop when sigma_xx < 0.2 peak
for 3 steps. Elastic preconditioners: Green (C_ref = C_iso), Green-Jacobi (2 mu I_sym + Jacobi diagonal), dual
G K(C_iso / s) G (cost 2 per iteration); damage problem always Green-Jacobi. CG: energy-error stop, rtol 1e-6, delay 10.
Material: C_d = s C_iso with s = g(d) * stiffness (scalar per quadrature point, `ScaledMaterial`). Note the 2D lambda
quirk: "E = 1, nu = 0.2" is plane strain with lambda = 5/36, mu = 5/12.

Microstructures (nine polydisperse disks, `benchmark/data/disks.csv`): pores (stiffness 1e-3, damage solved in them) or
stiff tough inclusions (10x, `INCLUSION_STIFFNESS=10`: no crack driving force inside).

## 1. Practical setting of the crack length scale

| setting | l/h | used for |
|---|---|---|
| minimum | ~2 | classic rule h <= l/2 (Miehe et al. 2010); coarser misrepresents the crack |
| typical | 2-4 | most engineering studies, uniform grids (FFT / voxel images) where memory limits resolution |
| well resolved | 4-10 | verification / convergence studies, adaptive meshes refined along the crack |

- AT2 profile d = exp(-|x| / l): the visible band (d > 0.1) is ~4.6 l wide, i.e. ~9-10 elements at l/h = 2.
- l as a regularization parameter: as small as the grid allows (l/h ~ 2). l as a material parameter: fitted to the
  strength (AT2: sigma_c = 9/16 sqrt(E' Gc / (3 l))), then the mesh must satisfy h <= l/2.
- FFT / voxel studies of heterogeneous microstructures: typically l = 1-4 voxels.
- Discretization bias: the crack energy is overestimated by a factor ~ (1 + c h / l); some authors correct Gc.
  Our benchmark (`benchmark/benchmark.pdf`): crack length Gamma 4-30 % high at l/h = 2-4, peak stress within 3.1 %
  (l/h = 2) and 2.3 % (l/h = 4).
- (from memory of Miehe 2010, Borden 2012, Bourdin et al., FFT phase-field papers; exact citations to be checked)
- Stricter, for mesh objectivity: element size five times smaller than l (l/h >= 5), Navidtehrani, Duddu,
  Martinez-Paneda, "Damage Mechanics Challenge: Predictions based on the phase field fracture model", Eng. Fract. Mech.
  301 (2024) 110046 (AT2, vol./dev. split, monolithic; `damage_AT2_small_strain/2024efm.pdf`), citing Kristensen,
  Niordson, Martinez-Paneda, "An assessment of phase field fracture: crack initiation and growth", Phil. Trans. R. Soc.
  A 379 (2021) 20210021. Consistent with our benchmark: peak stress converged at l/h = 2-4, the crack energy not.
- Hybrid formulation used here: Ambati, Gerasimov, De Lorenzis, "A review on phase-field models of brittle fracture and
  a new fast hybrid formulation", Comput. Mech. 55 (2015) 383-405 (`damage_AT2_small_strain/CM_Paper_2015_published.pdf`;
  only states that the mesh must resolve l, refers to Miehe et al. 2010 for mesh-size effects).

So l/h = 2 and 4 cover common practice, l/h >= 5 the mesh-objective setting (not yet run at fixed l/h); the fixed-l series (l = 1/32 up to l/h = 32 at 1024^2) is a convergence
reference only.

### Damage profile across the crack (`plot_AT2_damage_profile.py` -> `figures/AT2_damage_profile.png`)

B1 (one pore), fixed l = 1/32, final state, row y = 0 across the vertical crack, xi = (x - x_0) / l:

- The core (|xi| < 2) follows exp(-|xi|) already at l/h = 2 (the nodes lie on the curve), converged at l/h = 4
  (128^2 indistinguishable from 1024^2). l/h = 1 fails: a two-node plateau at d = 1, the band twice too wide.
- AT2 has no elastic threshold: background damage d_bg ~ 0.095-0.10 everywhere (history of the peak load).
- Beyond |xi| ~ 2 the tail is heavier than exponential ((d - d_bg) / (1 - d_bg) ~ 0.05 at xi = 5 instead of 0.007):
  the history field is elevated around the crack path (stress concentration of the passing crack tip), a source term
  the 1D profile does not have. Fitted decay length on 0.5 <= xi <= 3: 1.3-1.4 l (pulled by the tail).
- At l/h = 2 the crack centre can fall between two nodes (two nodes at d ~ 1): slight asymmetry.

## 2. 2D at fixed l/h, 64^2 -> 512^2 (`plot_AT2_lh_iterations.py` -> `figures/AT2_lh_iterations.png`)

Mean elastic CG per solve (dual cost = 2x); Green-Jacobi not run at 512^2 (never fastest at 256^2):

| case | Green | Green-Jacobi | dual (cost at 512^2) |
|---|---|---|---|
| pores l/h = 2 | 113 / 119 / 123 / 107 | 107 / 195 / 373 / - | 61 / 57 / 56 / 55 (109) |
| pores l/h = 4 | 136 / 209 / 209 / 254 | 105 / 185 / 350 / - | 56 / 75 / 68 / 69 (137) |
| inclusions l/h = 2 | 77 / 97 / 230 / 515 | 81 / 132 / 248 / - | 30 / 32 / 40 / 70 (140) |
| inclusions l/h = 4 | 84 / 129 / 305 / 879 | 77 / 134 / 228 / - | 23 / 30 / 40 / 69 (138) |

Elastic time at 512^2, Green / dual: pores l/h 2: 1607 / 1617 s (tie), pores l/h 4: 10877 / 5596 s, inclusions
l/h 2: 7211 / 1785 s, inclusions l/h 4: 26911 / 6103 s. At 256^2 the dual was fastest in all four cases (inclusions
2.5-3.5x, pores 1.2-1.9x). Pores at l/h = 2 are not cut through by E_xx = 0.08 at 256^2 / 512^2 (all preconditioners).

### When Green and the dual cost the same

Split into calm load steps (<= 3 staggered iterations) and crack steps:

| case (64^2 -> 512^2) | calm: Green / dual (cost) | crack: Green / dual (cost) |
|---|---|---|
| pores l/h = 2 | 91->34 / 45->22 (89->44) | 118->133 / 65->66 (130->132): tie |
| pores l/h = 4 | 48->35 / 27->22 (54->44) | 156->270 / 62->72 (125->144) |
| inclusions l/h = 2 | 38->35 / 21->18 (42->35) | 94->728 / 33->93 (66->186) |
| inclusions l/h = 4 | 50->22 / 20->16 (40->31) | 96->950 / 25->74 (49->147) |

Equal cost whenever the stiffness field is effectively two-valued (stiff vs nearly void): all calm steps (matrix +
disks, sharp interfaces), and pores at l/h = 2 during cracking (the 1-2 element wide band acts as a sharp void phase,
the cracks link the pores). The dual wins with smeared bands (l/h = 4) or three stiffness levels (inclusions 10,
matrix 1, crack ~1e-4). Consistent with "Green is cheap only on exactly two-phase quadrature points".

## 3. 3D at l/h = 2 (`plot_AT2_3D_iterations.py` -> `figures/AT2_3D_iterations.png`, GPU)

| case | Green | Green-Jacobi | dual (cost) | elastic time G / GJ / dual |
|---|---|---|---|---|
| inclusions 32^3 | 97 | 57 | 37 (74) | 202 / 154 / 146 s |
| inclusions 64^3 | 273 | 69 | 102 (204) | 15241 / 5202 / 11317 s |
| pores 32^3 | 125 | 47 | 102 (204) | 309 / 171 / 495 s |
| pores 64^3 | (queued) | 80 | (running: calm 28, crack 197) | - / 4963 / > 6771 s |

In 3D Green-Jacobi wins once the crack runs: the dual is best in the calm steps (64^3 inclusions 17 vs GJ 74) but the
crack-propagation steps dominate the solve count (dual ~155, GJ ~210 per solve, dual cost 2x; one dual iteration costs
~1.5x a GJ iteration on the GPU). 96^3 inclusions dual queued after the 64^3 pores Green run.

## 4. Why the dual is not mesh-independent after cracking (`diag_dual_mesh_dependence.py`)

Single elastic solves (cold start) on a FROZEN damage field of the fixed-l run (512^2, inclusions, l = 1/32) sampled
at the nodes of 64^2-512^2; Ritz values of the preconditioned operator from the CG coefficients.

| state, k | dual its (lambda_max), 64 -> 512 | Green | Green-Jacobi |
|---|---|---|---|
| before the break (max d 0.48), all k | 22, kappa 4.6, flat | 46 flat | 110 -> 594 |
| cut through, k = 1e-4 | 48 / 62 / 93 / 104 (53 -> 617) | 173 -> 1845 | 171 -> 1376 |
| cut through, k = 1e-3 | 46 / 52 / 56 / 57 | 167 -> 646 | 169 -> 1085 |
| cut through, k = 1e-2 | 37 / 38 / 38 / 38 (22 -> 25) | 132 -> 217 | 163 -> 842 |

- The dual is exactly mesh-independent at bounded contrast; lambda_min ~ 1 always (the guarantee lambda_min >= 1).
- Only lambda_max grows, and only for small k: g = (1 - d)^2 + k has its k-plateau of width ~ l sqrt(k) (3e-4 for
  k = 1e-4) far below h, so each refinement resolves a deeper stiffness valley, until h < l sqrt(k).
- Also at fixed l the crack-step iterations of the full runs grow (inclusions 33 -> 57, 64^2 -> 512^2), so it is not
  an artefact of fixed l/h.
- Capped dual C_iso / max(s, 1e-2) (`DIAG_PRECONDS=dual_cap`): WORSE (k = 1e-4: 41 / 58 / 124 / 250): lambda_max
  stays ~30 but lambda_min drops 1 -> 0.012; the large 1/s in the band is needed.

### Reference material

- Scaling C_ref by a constant changes nothing (PCG is invariant to scaling of the preconditioner; the dual
  G_ref K(C_ref C^-1 C_ref) G_ref is exactly scale-invariant). Two references "mean of C" and "harmonic mean of C" for
  the two Green operators give the identical dual here (all phases are multiples of C_iso); with phases of different
  shape A != B the operator G_A K(A C^-1 B) G_B is non-symmetric (no plain CG) and not exact for a homogeneous material.
- Identity reference C_ref = I_sym (`Green_I`, `dual_I`): at nu = 0.2 (C_iso close to a multiple of I, eigenvalue ratio
  1.33) within 1-2 iterations of C_iso. At nu = 0.49 (ratio ~50), 64^2 -> 256^2:

| state | dual | dual_I | Green | Green_I |
|---|---|---|---|---|
| before the break | 22 / 22 / 22 | 77 / 76 / 75 | 47 / 47 / 47 | 137 / 148 / 154 |
| cut through, k = 1e-2 | 42 / 41 / 42 | 109 / 110 / 108 | 161 / 193 / 214 | 481 / 670 / 755 |
| cut through, k = 1e-4 | 57 / 74 / 114 | 144 / 196 / 338 | 256 / 486 / 1213 | 561 / 1185 / 2814 |

  dual_I keeps lambda_min ~ 1 and the mesh behaviour, only lambda_max (the shape mismatch) and the constant grow ~3x;
  the reference must have the right shape, not the right scale. With the correct C_iso the dual is insensitive to
  nu = 0.49.

## Files

- runs: `run_AT2_incl_lh_{64,128,256}.sh`, `run_overnight_2D_512_cpu.sh`, `run_overnight_3D_gpu.sh`,
  `run_overnight_3D_96_gpu.sh`; logs in `logs/`, data in `exp_data/`
- figures: `AT2_lh_iterations.png`, `AT2_3D_iterations.png`, `diag_dual_mesh_dependence{,_cap,_identity,_identity_nu049}.png`,
  `AT2_damage_profile.png`
- benchmark for other groups: `benchmark/` (B0-B3, `benchmark.pdf`)
- figures/AT2_3D_crack.png (plot_AT2_3D_crack.py): 3D crack surfaces at 64^3, l/h = 2: inclusions -> one continuous surface deflected around the balls (x = 0.52-0.70); pores -> the crack links the pores (blank holes in the face-on map).

## 5. 3D frozen-field test: 96^3 iterations without a 96^3 run (`diag_3D_frozen.py`)

Damage of the 64^3, l/h = 2 runs in three phases (calm = half the peak load, peak = last step before the break,
broken = cut through), transferred to 96^3 (periodic trilinear interpolation of ln d, band narrowed to l = 1/48 by
d -> d^1.5), balls regenerated; one cold-start elastic solve each. The 64^3 counts reproduce the full runs.

| case | phase | dual 64 -> 96 | Green-Jacobi 64 -> 96 | work ratio GJ / (2 dual) at 96^3 |
|---|---|---|---|---|
| inclusions | calm | 20 -> 19 | 110 -> 148 | 3.9 |
| inclusions | peak | 28 -> 25 | 127 -> 155 | 3.1 |
| inclusions | broken | 239 -> 216 | 348 -> 480 | 1.1 |
| pores | calm | 39 -> 32 | 116 -> 170 | 2.7 |
| pores | peak | 51 -> 47 | 118 -> 171 | 1.8 |
| pores | broken | 436 -> 770 | 222 -> 330 | 0.21 |

- Dual mesh-independent before the break (lambda_min 1.00-1.05); Green-Jacobi grows 30-45 % per 1.5x refinement.
- Inclusions broken: the dual flat, GJ grows -> the dual wins all phases at 96^3.
- Pores broken: the dual grows strongly (lambda_max 8.1e3 -> 2.5e4): pore contrast 1e-3 times the residual 1e-4 in the
  damaged pores (s ~ 1e-7), the cracked ligament lattice; GJ 4.7x cheaper in work at 96^3.
- The opt-in `dual_switch` preconditioner (dual; in break steps whichever of dual (2 per iteration) / GJ needs less CG
  work, trial after SWITCH_AFTER staggered iterations) exists in the example script; only a small 2D smoke test, not
  part of the results.

## 6. 2D: crack smoothness vs grid scaling (`run_AT2_smoothness_2D.sh`, `plot_AT2_smoothness_2D.py`)

Full runs, N = 32/64/128 x l = 1/8 ... 1/64 (l/h = 1 ... 16), pores and inclusions, all three preconditioners; work per
elastic solve (dual 2 per iteration), mean over the run (`figures/AT2_smoothness_2D.png`).

- Green-Jacobi: ~1.75x per grid doubling (pores 61 -> 105 -> 185, inclusions 54 -> 79 -> 131) in every setting; at fixed
  grid independent of l/h. Its growth is a grid effect; a smoother crack does not help it.
- Dual: nearly flat in both settings (pores 95-150, inclusions 40-65 over all runs); at fixed grid 128^2 cheaper for
  smoother cracks (pores 150 -> 104 for l/h 4 -> 16, inclusions 63 -> 40 for l/h 2 -> 16).
- Green: grows with the grid, erratic with l/h (sensitive to the crack pattern).
- The dual's work is below Green-Jacobi's from 64^2 on (pores) and on every grid (inclusions).
- l = 1/8 is larger than the disks: the damage is smeared over the microstructure (a different physical problem).

## 7. 2D: stiffness contrast x crack length x grid (`plot_AT2_contrast_map_2D.py`)

Five material settings, each on the N = 32/64/128 x l = 1/8 ... 1/64 matrix with all three preconditioners (165 runs):
pores = k = 1e-3 (`HOLES=1 HOLE_STIFFNESS=1e-3 K_RESIDUAL=1e-3`, contrast 1e3 throughout), pores 1e-3 + damage (old
model, 1e3, 1e7 in a few cracked pore voxels), pores = k = 1e-4 (`HOLES=1 HOLE_STIFFNESS=1e-4`, 1e4), inclusions 10x
(10 before / 1e5 after cracking), inclusions 1000x (1e3 / 1e7). New options: `HOLE_STIFFNESS`, `K_RESIDUAL`.
Figures: `AT2_contrast_map_2D.png` (work per solve over grid x l), `AT2_contrast_scaling_2D.png` (grid exponent b).

Grid exponents b (work ~ N^b), sharp crack l/h = 2 | smooth crack l = 1/16:

| setting (contrast before / after) | Green | Green-Jacobi | dual |
|---|---|---|---|
| pores = k = 1e-3 (1e3 / 1e3) | 0.55 / 0.47 | 0.73 / 0.79 | 0.10 / 0.06 |
| pores 1e-3 + damage (1e3 / 1e7 few) | 0.19 / 0.60 | 0.84 / 0.77 | 0.09 / 0.13 |
| pores = k = 1e-4 (1e4 / 1e4) | 0.63 / 1.00 | 0.84 / 0.80 | 0.11 / 0.20 |
| inclusions 10x (10 / 1e5) | 0.35 / 0.73 | 0.64 / 0.63 | 0.22 / 0.09 |
| inclusions 1000x (1e3 / 1e7) | 0.34 / 0.94 | 0.62 / 0.64 | 0.43 / 0.30 |

- Green-Jacobi: b = 0.6-0.85 in every setting; the contrast sets its level (inclusions 1000x: 214 -> 589), not its growth.
- Dual: the contrast sets the level (pores k = 1e-4 ~2x pores 1e-3) and, through the stiffness range after cracking, the
  growth: b ~ 0.1 up to 1e4, 0.2 at 1e5, 0.3-0.4 at 1e7 (inclusions 1000x). Still the cheapest there (92-197 vs
  GJ 214-589, Green 169-642).
- High-contrast stiff inclusions hurt Green and Green-Jacobi most (both ~3x the 10x case); the dual degrades less.

## 8. Summary: which preconditioner when (2D, 165 runs of section 7)

Work per elastic solve at 128^2 (range over l; dual 2 per iteration):

| setting | Green | Green-Jacobi | dual | cheapest |
|---|---|---|---|---|
| pores = k = 1e-3 | 71-101 | 166-181 | 104-118 | Green |
| pores 1e-3 + damage (old model) | 119-219 | 179-195 | 104-150 | dual (Green ties at l/h = 2) |
| pores = k = 1e-4 | 117-197 | 190-208 | 249-321 | Green; the dual slowest |
| inclusions 10x | 80-165 | 128-134 | 40-63 | dual, 2-3x |
| inclusions 1000x | 278-642 | 490-589 | 92-190 | dual, 3-5x |

Grid scaling (work ~ N^b): dual b ~ 0.1-0.2 (0.4 only at contrast 1e7), Green-Jacobi 0.6-0.85 everywhere, Green 0.2-1.0.

Why the dual is sometimes slower:
1. A higher starting level and 2 K applications per iteration (1.5x in GPU time): on small grids Green / Green-Jacobi
   have not grown yet.
2. Pores of one uniform stiffness (pores = k) make the field nearly exactly two-phase with sharp interfaces -- where
   Green is very efficient.
3. The dual is mesh-independent but not contrast-independent: lambda_min ~ 1 always, lambda_max grows with the stiffness
   contrast it sees (~250 at 1e3, ~2500 at 1e4); pores = k = 1e-4 put a 1e4 jump on every pore boundary from the first
   step, the dual's level ~doubles.

Where the dual wins: inclusions at any contrast; graded or multi-level stiffness fields (smeared bands, three stiffness
levels, damage inside pores); larger grids -- every case it loses at 128^2 flips around 256^2 (rough estimate from
three grids).

One line: Green-Jacobi's cost is set by the grid, Green's by how two-phase the field is, the dual's by the stiffness
contrast (flat in the grid, level rising with the contrast). Lowering the contrast (k = 1e-3, pores at the same
stiffness) makes the dual both flat and competitive. In 3D the crack is a surface: a sharp crack (fixed l/h) adds more
high-contrast elements per refinement, so the dual grows faster there than in 2D, especially with pores.

### Extrapolation to 256^2 - 1024^2 (`plot_AT2_contrast_extrapolation_2D.py` -> `figures/AT2_contrast_extrapolation_2D.png`)

Fit work ~ a N^b over 32^2 - 128^2, sharp crack (l/h = 2) | smooth crack (l = 1/16), work at 1024^2:

| setting | Green | Green-Jacobi | dual |
|---|---|---|---|
| pores = k = 1e-3 | 317 / 244 | 753 / 909 | 144 / 124 |
| pores 1e-3 + damage | 180 / 733 | 1107 / 898 | 143 / 161 |
| pores = k = 1e-4 | 432 / 1716 | 1120 / 1003 | 334 / 419 |
| inclusions 10x | 203 / 715 | 492 / 471 | 104 / 62 |
| inclusions 1000x | 577 / 4491 | 1782 / 1894 | 502 / 295 |

By 1024^2 the dual is the cheapest in every setting; crossovers: Green-Jacobi passes the dual at ~64^2 (pores 1e-3)
to ~256^2 (pores = k = 1e-4); Green passes the dual at ~256^2 (pores = k = 1e-3) and ~600^2 (pores = k = 1e-4, sharp).

Check against the real 256^2 / 512^2 runs (sharp crack): pores 1e-3 + damage well predicted (dual 126 / 134 vs real
112 / 109, GJ 346 vs 373, Green 138 / 158 vs 123 / 107); inclusions 10x under-predicted -- the growth accelerates beyond
128^2 (dual 76 / 89 vs real 79 / 140, GJ 202 vs 248, Green 125 / 159 vs 230 / 515). The fits are lower bounds where
the crack pattern changes with refinement, Green most of all.

### Split by phase (`PHASE=pre|cracking|post`, figures `AT2_contrast_extrapolation_2D_{pre,cracking,post}.png`)

Phases from the stress-strain curve: pre = up to the peak stress; cracking = after the peak up to and including the
step in which sigma_xx drops below 0.2 peak (the breaking step, most of the solves); post = later steps.
Grid exponents b (sharp l/h = 2 | smooth l = 1/16):

| phase | Green | Green-Jacobi | dual |
|---|---|---|---|
| pre-crack | -0.12 ... 0.00 (flat; 22-50) | 0.61 ... 0.83 | -0.10 ... -0.02 (flat; 30-67) |
| cracking, pores (all models) | 0.25 ... 1.07 | 0.71 ... 0.86 | 0.07 ... 0.21 |
| cracking, inclusions | 0.42 ... 1.01 | 0.56 ... 0.87 | sharp 0.58 / 0.77, smooth 0.17 / 0.39 |
| post-crack, pores | 0.17 ... 1.01 | 0.71 ... 0.89 | -0.05 ... 0.03 (flat) |
| post-crack, inclusions | 0.31 ... 0.96 | 0.75 ... 0.81 | sharp 0.33, smooth 0.12 / 0.20 |

- Pre-crack: Green and dual are grid independent (Green cheapest for pores: the field is exactly two-phase); only
  Green-Jacobi grows. Real 256^2 runs within 0-10 % of the fits.
- The dual's grid dependence comes only from cracking / post-crack with stiff inclusions and a sharp crack (three
  stiffness levels: inclusion, matrix, crack); with pores it is flat in every phase.
- Green-Jacobi grows in every phase (b 0.6-0.9); Green grows once the crack forms and is hard to predict (real/fit
  0.4-2.3).

### Crack path vs preferred preconditioner (`plot_AT2_crack_gallery.py` -> `figures/AT2_crack_gallery.png`, videos
`figures/AT2_N256_dual_polydisperse_pores*_smooth.mp4`, `AT2_N256_dual_polydisperse_pores_l0.015625_lh.mp4`)

256^2, cracking-phase work (dual 2x), best other / dual: pores with damage inside, stepwise crack 1.60, smooth 2.21;
pores = k: 0.86 (k = 1e-3, smooth), 0.40 (k = 1e-4, sharp); inclusions 10x 1.97 / 2.46, 1000x 1.90 / 3.56.
The crack path alone does not decide; the number of stiffness levels of the cracked field does: a crack linking pores
of the same stiffness as broken material keeps the field two-phase (Green); a crack around stiff inclusions, or into
pores of a different stiffness (damage in the pores), makes it multi-level and graded (dual, more so for smooth cracks
and higher contrast). Green-Jacobi is never the cheapest.
