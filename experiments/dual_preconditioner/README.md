# Dual preconditioner study

The dual preconditioner `P = G K(C_ref A^-1 C_ref) G` compared with Green `G` (reference `C_ref`) and Green-Jacobi
`D^-1/2 G_I D^-1/2` (reference `2 mu I`). Reference material, as everywhere in this study: `C_ref` = the solid
(matrix) tangent for Green and dual, `2 mu I` for Green-Jacobi. Guarantees: `lambda_min(PK) >= 1` (complementary
energy, trial stress `C_ref B G f`), `lambda_max(PK) <= max g / min g` with `g` the eigenvalues of `C_ref^-1 A`.
CG stops on the energy-error estimate (relative 1e-6, delay 10) unless stated otherwise.

| folder | what |
|---|---|
| `spectral_study/` | exact spectra (dense) on 16^2 / 32^2: conductivity and small-strain elasticity, several references and Schur / Neumann variants (`study.py`, `summarize.py`) |
| `paper1_problems/` | test problems P1-P3 of the error-estimates paper (Ladecky, Papez, Tichy, Pastewka): conductivity (`run_experiment.py`) and small-strain elasticity (`run_experiment_elasticity.py`), contrast and grid studies, `plot_results.py [elasticity]` |
| `topology_optimization/` | a self-contained copy of `examples/topology_optimization/example_2D_elasticity_TO.py` with its data (`data/`) and the three preconditioners: overnight queue, comparison plots, all final designs, 1024^2 video run script |
| `slide_problems/runs_2026-10-15/` | the dual runs of the phase-field slide problems (`examples/phase_field_fracture_adapted_grid/`): run folders `N*_dual*`, the alpha = 0 set `N32_noreg_*` / `runs_noreg/`, Green without R in G (`N32_Green_noR_in_G`, `runs_green_noR/`), earlier dual variants (`runs_dual_v1_stopped/`, `runs_dual_v2_Isym/`, `runs_dual_solid_R/`), comparison figures, `make_dual_videos.sh`; the Green / Green-Jacobi reference runs stay in the slides folder |
| `slide_problems/recheck_32/` | 32^2 rerun (2026-10-16, current script) of trust region without regularization, contact and fracture for all three preconditioners against the 2026-10-13/15 runs (`compare.py`) |

Every part is self-contained: the scripts it runs are copies inside this folder, not the files in `examples/`.

Results produced before 2026-10-16 with two Green operators on one discretization are invalid for their Green and
dual columns (`get_preconditioner_Green_mugrid` shared one field; it now takes `name=`); they are kept in
`topology_optimization/figures/invalid_shared_green_field/`.

## Notes

- `notes/finite_strain_tangent_explained.md`: the finite-strain tangent (material and geometric stiffness, rotational stiffness = stress), why single points can have negative stiffness while the structure is stable, and why the dual preconditioner must not invert the rotation part (plain inverse vs |eigenvalue| pseudo-inverse vs symmetric block).
