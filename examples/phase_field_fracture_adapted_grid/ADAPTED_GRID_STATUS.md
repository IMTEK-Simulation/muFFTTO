# Contact + fracture on an interface-fitted grid: status

Started 2026-10-08. Goal: a better representation of the geometry (`sinusoidal_ligaments`, the
contact surfaces) with an adapted grid; refinement for the crack comes from the resolution chosen
later (l = 1/32 needs >= 4 px, see `../phase_field_fracture/CONTACT_FRACTURE_STATUS.md`).
Later: larger runs on the cluster.

## Files

| file | what |
|---|---|
| `adapted_grid.py` | interface-fitted periodic grid for any `muFFTTO.geometry` name: signed distance from a fine image, nodes within `snap_distance` h snapped onto the interface (bisection with the exact indicator), periodic Laplace relaxation of the others, release of snapped nodes of degenerate elements, phase at the deformed element centroid. `python adapted_grid.py 32` plots regular vs adapted grid. Written for this example (the library `grid_adaptation_arbitrary` workflow is not used). |
| `example_2D_phase_field_fracture_contact_adapted_grid.py` | the contact-fracture example with all operators pulled back with F_g = I + grad d_g, J_g (mechanics, tangent, damage, mean stress); preconditioners on the regular grid; HuHu regularization in reference coordinates. `adapt_grid_to_geometry = False` -> regular grid. Grids cached in `exp_data/grids/*.npz`. |
| `plot_contact_fracture_adapted_grid_output.py` | response + damage on the deformed adapted grid (+ `--animate`) |
| `exp_data/comparisons/compare_runs.py` | several runs (regular / adapted, any N) matched by F11, figure or movie |

Defaults: sinusoidal_ligaments, H = [[0, 0], [0.01, -0.27]], l = 1/32, Green_Jacobi, inexact Newton,
staggered_tol 1e-3 (cap 300, then halve), output `exp_data/runs/` (short path: muGrid's NetCDF writer
crashes with a buffer overflow for file paths longer than ~245 characters; the example checks 230).

## Tests (32^2, serial)

- d_g = 0 (`adapt_grid_to_geometry = False`): bit-identical to the regular-grid example
  (`../phase_field_fracture/exp_data/phase_field_fracture_contact/..._N32_..._20261008-110452`), all fields, 55 frames.
- Patch test on the adapted grid (k_v = 1, k_residual = 0, any u~ = 0): residual 4e-16, mean P = P(F_bar)
  to all digits, physical volume 1, mean F = I + lam H.
- Finite differences on the adapted grid with the medium: gradient and Hessian-vector product O(eps^2),
  1e-8 at eps 1e-6; damage operator symmetric, integrates a constant to the cell area.

## First result: adapted 32^2 (`exp_data/runs/..._adapted_N32_..._20261008-184143`)

Grid: 196 snapped nodes (36 released), min corner J/h^2 0.31, max |d_g| 0.47 h. Misassigned area
0.0255 (regular 32^2: 0.0493), but solid fraction 0.5510 vs exact 0.5434: the narrow medium wedges
(cusps) at the ligament crossings are thinner than an element and end up solid.

Figure `exp_data/comparisons/adapted32_vs_regular_32_64_128_256_20261008-184809.png`:
- elastic part: adapted 32^2 too stiff (P11 -0.905 at F11 0.92), regular 32^2 too soft (-0.77),
  reference 128^2/256^2 -0.84: similar error, opposite sign (follows the solid fraction).
- peak -1.255 at F11 0.854, main drop at 0.827; failure through the inner ligaments like regular
  32^2/64^2, not at the outer junctions like 128^2/256^2 (l = 1 px at 32^2 anyway).
- cost: 337 s, 323k Hessian products vs 63 s / 60k on the regular 32^2 grid (preconditioner does
  not see the grid map; 27 det F step halvings).

## Next

1. Cusps: phase by element area fraction instead of the centroid, and/or keep snapped nodes at the
   cusps; measure geometry representation by the elastic response (before damage) against 256^2.
2. Cost: preconditioner with the grid map (e.g. Green of J_g-averaged / pulled-back reference tangent).
3. Resolution study adapted vs regular at 64^2 / 128^2; then cluster job scripts.

## Grid adaptation moved to the library (2026-10-08, evening)

`muFFTTO/grid_adaptation.py`: `adapt_grid(nb_pixels, geometry, method='zecevic' | 'snap')`,
`project_nodes_to_interface`, `grid_quality`, `plot_grid`. `adapted_grid.py` in this folder is superseded
(only the first run `..._adapted_N32_..._20261008-184143` used it).
- 'zecevic' (Zecevic, Lebensohn, Capolungo, Mech. Mater. 212 (2026) 105512; paper as
  `grid_addapt_zecevic.pdf` here): pixel phases of the regular grid (single-pixel features flipped), the
  staircase boundary nodes are moved onto the interface (chain springs + projection, gradual, interior
  relaxed with k/(dist+1)^0.5 springs every iteration), detected cusps get a node, safeguard J >= 0.01 h^2
  at the Gauss points only (user: "use min J test just for quad points"; elements may be triangles or
  slightly non-convex, needed at wedge tips narrower than a pixel).
- 32^2: misassigned area 0.00028 (regular 0.0493), solid fraction 0.5437 (exact 0.5434), min J_g 0.045 h^2,
  ~1 min. Figures `exp_data/grids/mesh_detail_N32_zecevic_v3_20261008-201002.png` (user: "good enough").
- Plot scripts: `plot_adapted_grids.py` (grids + .npz), `plot_mesh_detail.py` (zooms), `project_saved_grid.py`.
- Slowness was the exact geometry indicator (~45 us/point for sinusoidal_ligaments): projection now uses
  the signed-distance image during the iterations and the exact bisection only at the end.

## Run on the zecevic grid, 32^2 (`exp_data/runs/..._adapted_N32_..._20261008-201420`)

322 s, 64 increments, 195k Hessian products (regular 32^2: 63 s, 60k).
Figure `exp_data/comparisons/zecevic32_vs_regular_32_64_128_256_20261008-202013.png`.
- P11 at F11 0.92: zecevic 32^2 -0.894, regular 32^2 -0.763, reference 64/128/256^2 -0.84/-0.833/-0.838:
  with the exact geometry the 32^2 grid is too stiff by ~7 %; the regular 32^2 was too soft by ~9 % (phase
  at pixel corners, solid fraction 0.535). So at 32^2 the element (bilinear Q1, ~4.5 elements across a
  ligament, distorted elements) dominates the error, not the geometry.
- Peak -1.255 at 0.849, main drop 0.826; still the inner-ligament failure of 32^2/64^2 (l = 1 px).

## 2026-10-08 evening: physical regularization, SVK void, CG study (running overnight)

- Regularization in physical coordinates (`regularization_in_physical_coordinates = True`, default):
  H_x u = F_g^-T [H_X u - (grad_x u)_c H_X x_c] F_g^-1, energy k_r/2 sum w J_g (H_x:H_x - L_x.L_x/d).
  Checks (32^2): affine-in-x field H_x = 1e-9 (old X-Hessian 84); lam=0 energy/residual exactly 0;
  regular grid identical to the old term (6e-16); FD of gradient/Hessian O(eps^2). It does NOT explain
  the extra stiffness of adapted grids (elastic 32^2 at F11 0.92: adapted -0.998 without regularization,
  -1.003 X-reg, -1.007 x-reg; regular 128^2 -0.931). It is ~10x slower on the adapted grid (1/J_g^2).
- Elastic check (no damage), P11 at F11 0.92: regular 32/64/128: -0.838/-0.937/-0.931; adapted
  32/64/128 (X-reg): -1.003/-0.981/-0.961. Adapted converges slowly from the stiff side; open question:
  stiff thin Q1 elements, or the staircase makes regular grids too soft (regular 256^2 elastic would tell).
- Zecevic grid: safeguard bug fixed (undo widened until all elements valid; 128^2 was invalid before),
  adapt_grid raises if the final grid is invalid. Grids 32/64/128: misassigned 0.00028/0.0012/0.0004.
- `void_model = 'saint_venant_kirchhoff'`: no-contact voids (super-soft SVK, k_v 1e-6, finite for any F,
  J > 0 required in the solid only); default 'neo_hookean' (third medium, contact). FD checks pass.
- Tension H11 = +0.5 (adapted 32^2, `..._F11+0.5_..._20261008-211542`, stopped at F11 1.30): brittle,
  peak P11 1.04 at F11 1.09, vertical ligaments break at the cell edge. Compression in x
  (`..._F00-0.27_..._20261008-213416`): peak P00 -1.15 at F00 0.86, cracks at the apex of the
  horizontal ligaments.
- adapted 64^2 damage (compression y, `..._adapted_N64_..._20261008-202544`): corner (outer-junction)
  cracks first like regular 128^2/256^2, but stopped at F11 0.822 (trust region maxiter 500; use 2000).
- CG study (16 cases: regular/adapted x elastic/damage/damage+contact/contact x Green/Green_Jacobi;
  no contact = SVK void, alpha 0): 32^2 in `scratchpad/cgstudy`, 64^2 and 128^2 started by the
  overnight driver `scratchpad/cgstudy64/overnight.py` (log overnight.log); analysis
  `exp_data/comparisons/cg_study.py` -> `cg_study_N32/N64/N128_*.png/.txt`.
- Paper draft (math only): ~/Documents/contact_fracture_adapted_grid/main.tex.

## 2026-10-09: CG study, first results and restart (v2)

- 32^2 night results (`exp_data/comparisons/cg_study_N32_partial_20261009-065947.*`,
  `cg_study_response_N32_partial_20261009-070639.png`): Green_Jacobi 2-5x fewer CG per TR iteration on the
  regular grid; contact is the expensive part; adapted + contact ~400-500 CG/TR with both preconditioners
  (preconditioner does not see the grid map). Curves identical for both preconditioners.
- Stuck: damage without contact with the SVK void (after the first crack the fully damaged elements distort
  without bound, SVK softens in compression -> TR failures of 2000 iterations, hours per step); adapted
  elastic Green_Jacobi (SVK void) 9 h CPU for 31 steps, time in rejected TR attempts (not in the frame counts).
- New `void_model = 'linear'` (super-soft small-strain linear elastic void, convex, constant tangent; FD
  checked). All no-contact cases of v2 use it.
- v2 study: `scratchpad/cgstudy_v2/` (gen.py writes case scripts per N, driver.py: 64^2 all 16 cases, then
  32^2 missing/no-contact cases, then 128^2 per finished 64^2 case, max 3; wall-time limits 2/8/12 h;
  analysis -> `cg_study_v2_N*` and `cg_study_response_v2_N*`). Run tags `_cg2-{el,dm,dc,ct}`.
- Regularization cost test (adapted 64^2, contact, Green, 6 steps, `scratchpad/cgstudy_v2/regtest`): CG per
  step physical ~9000, reference-grid (regX) ~1000, none ~300-500; P10 equal to 4 digits (regX vs physical).
  The physical-coordinate HuHu term (1/J_g^2 on thin elements) is not in the preconditioner.
  Decision (user, 2026-10-09): adapted contact cases of the study use `regularization_in_physical_coordinates
  = False`; the four 64^2 runs with the physical term were stopped (logs `*_physreg_stopped.log`, partial
  run folders kept) and restarted; `driver2.py` adopted the running jobs. Later: physical term in the
  preconditioner.

## PAUSED 2026-10-09 09:39 (user: new task, keep the study state)

- driver2.py killed; the 10 running 64^2 runs SIGSTOPped (state T), list in
  `scratchpad/cgstudy_v2/paused.txt` (pid case; cwd scratchpad/cgstudy_v2/N64; adapted_contact_* and
  adapted_damage_contact_* are the reference-grid-regularization restarts, the others started 07:15-09:00).
- Resume: `kill -CONT $(awk '{print $1}' paused.txt)`, then a driver (copy of driver2.py) that adopts these
  pids with fresh start times (the pause must not count against the wall-time limits) and the remaining queue:
  64^2 regular_damage_{Green,Green_Jacobi}, regular_elastic_{Green,Green_Jacobi}; then 32^2 all cases except
  regular contact/damage_contact (KEEP32); then 128^2 for every finished 64^2 case (adapted_elastic_* already
  finished), max 3 at once.
- If the processes are gone (reboot): each run folder holds all finished frames; continue with
  `restart_file` in the case script, or rerun.
- Interim figures: `exp_data/comparisons/cg_study_v2_N64_partial_20261009-093518.*`,
  `cg_study_response_v2_N64_partial_20261009-093518.png`.
