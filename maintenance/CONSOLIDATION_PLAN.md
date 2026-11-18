# muFFTTO consolidation plan (2026-11-18, not started)

**Goal.** muFFTTO is the human-readable prototyping library. Ready features go to muSpectre (muGrid, muTopOpt).
Keep our own energy-error CG. Anything muGrid already has comes from muGrid. Features the experiments keep
rebuilding become proper library code. The experiments move to separate repos.

## What the survey found (numbers)

| | now |
|---|---|
| Library `muFFTTO/` | 17k lines; `domain.py` 3.8k lines (82 methods); dead code in `solvers.py` (`___PCG`, `donothing`, `findS`, `adam`), `apply_preconditioner_NEW`, 3 grid-adaptation modules |
| muGrid use | only `Field`, `FFTEngine`, `GenericLinearOperator`, `FileIONetCDF`, `Communicator`. Our own Green / GJ / Jacobi / plain CG / reductions (NuMPI `Reduction`) duplicate muGrid main (v1.6.x) |
| Fable branch | library: `multigrid.py` (698 lines), fused GPU ScaledMaterial kernels, hourglass `apply_to_field`, `test_multigrid.py`; the rest is experiment code (`to3d_large.py`, `lbfgs_xp.py`, `cg_energy.py`, `cubic_symmetry.py`), most of it already being ported to muGrid/muTopOpt (`TO_LIBRARY_PLAN.md`) |
| Experiments | 448 .py files, 115k lines, 168 GB data on disk (not in git) |
| Copy-paste families | `solve_damage` in 190 files, `correlated_random_field` 135, `apply_dual` / `global_sum` / `field_to_optimizer` / neo-Hookean split ~49, `hessp`/`fun_grad` (TR Newton contact) ~64, 16 local CG copies, 335 `sys.path` hacks |
| Examples | 83 scripts + 84 run snapshots inside `examples/phase_field_fracture_adapted_grid/exp_data`; 6.3 GB output data in the folder |
| Git | current branch 67 commits ahead of `main` (main last changed 2026-10-08), 20 unpushed; fable 75 unpushed + uncommitted `experiments/multigrid_vs_dual`; `feat/fracture_and_contact` fully merged; `worktree-polycrystal-crack` 6 and `StefanusMichael_Dev` 18 unmerged commits |

## Target library layout (proposal)

```
muFFTTO/
  discretization.py      grid, fields, gradient / B^T C B application   (split out of domain.py)
  elements.py            element library (today discretization_library.py, JAX) + quadrature
  materials.py           material_models + ScaledMaterial
  preconditioners.py     Green, Green-Jacobi, Jacobi (thin wrappers on muGrid), dual G K(C_ref A^-1 C_ref) G,
                         dual_switch, corrected Green (wall FFT), multigrid (muGrid once released)
  solvers.py             ONE CG: energy-error stop, fixed rtol, adaptive delay (ours, kept)
  solvers_nonlinear.py   TR Newton-CG, BB, MMA; L-BFGS from NuMPI/muGrid
  parallel.py            global sum/min/max/dot, root_print, field <-> optimizer vector (muGrid Communicator)
  phase_field_fracture.py  AT1/AT2, splits (amor, spectral), degradation, history, staggered + bounded damage solve
  contact.py             third-medium contact (regularisation restricted to the gap)
  topology_optimization.py (+ conductivity), Helmholtz L-BFGS preconditioner, energy objective
  microstructures.py     geometry, correlated random fields, Cahn-Hilliard starts, Fourier resampling
  grid_adaptation.py     one module instead of three
  io_utils.py, visualization.py   (exact-pixel rendering, Green=green / GJ=blue / dual=red, readable axes)
```

## Phases

**0. Safety first (no code change).**
Commit the uncommitted fable work. Tag the current state `pre-consolidation`. Run the test suite and record
reference numbers (CG counts, energies) on 5-6 small cases: homogenization 2D/3D, TO 64², AT2 64², contact 32²,
multigrid 32³. Check which NEMO jobs run from which clone before moving anything.

**1. One branch.** Merge `fable/large-3d-to` into the current branch; that branch becomes the new `main`.
Delete `feat/fracture_and_contact`; decide about `worktree-polycrystal-crack` and `StefanusMichael_Dev`.

**2. Split the experiments out (before refactoring).** Every study repo pins muFFTTO to the `pre-consolidation`
tag, so the 115k lines of study code keep running unchanged while the library API moves. Data stays on disk/NEMO
(git ignores it anyway); only the folders move. `Stefanus/`, `prototypes/`, the run snapshots in `examples/` and
the PDFs leave the library repo.

**3. Reuse muGrid (no behaviour change).** Reductions → `muGrid.Communicator` (drops NuMPI `Reduction`, works with
GPU fields). Green → `make_reference_stiffness_preconditioner` (impulse route with our K_ref, so every element
works). GJ / Jacobi → muGrid classes. Field algebra → `muGrid.linalg`. Grid transfer → `muGrid.GridTransfer`. The
plain CG is replaced by our energy-error CG. Each swap must reproduce the phase-0 numbers.

**4. Move what the experiments rebuild into the library.** One feature at a time, each with a test, and each
validated by switching one real study script to it:
dual + dual_switch → parallel helpers → phase-field fracture core → third-medium contact / TR Newton functional →
random fields + CH starts → Helmholtz L-BFGS + energy objective → plotting style.

**5. Clean up.** Delete dead code; split `domain.py`; merge the grid-adaptation modules; about 20 curated examples
(one per feature, small grids, output outside the repo); docs.

**6. Port pipeline (continuing).** Ready pieces go to muGrid/muTopOpt (multigrid, 1Q_hg, L-BFGS already in
flight); muFFTTO then switches to the muGrid version and deletes its own.

## Decided 2026-11-18

- Split by topic into 4 study repos (grouping in item 1 below).
- Fresh start: one initial commit per study repo (old history stays in muFFTTO up to `pre-consolidation`).
- Hosting: GitHub, private repos under the user's account.
- Multigrid: take fable's `multigrid.py` + tests into the library now; replace by muGrid's once released.
- Elements: our own element path (JAX element library, generic B^T C B) stays as the readable reference; the
  muGrid fused operator is an optional fast path.
- Phase field: the material models (AT1, AT2 and their splits, degradation) are library code; `examples/` gets a
  small example of using the AT1 and AT2 models. The whole study (all variants: fatigue, ASR, battery, PF-CZM,
  sintering, ...) goes to the separate study repo.

## Decision log

1. ~~Study-repo grouping~~ (decided: by topic) —
   `dual-preconditioner` (dual_preconditioner/ incl. all phase-field families, corrected_green,
   linear_elastic_mesh_study, polycrystal branch), `multigrid` (Multigrid_preconditioning + fable
   multigrid_vs_dual), `contact` (patch test, flat punch, Hertz, disc, Wriggers 2025, tree_on_house,
   bubble_inflation, bb_comparison), `stefanus`.
2. Keep each study's git history (`git filter-repo`, needs a pip install) or start fresh.
3. Hosting of the new repos (GitHub private under your account / imtek-simulation / local only).
4. ~~Element path~~ (decided: own path = reference, muGrid fused = optional fast path).
5. ~~Multigrid~~ (decided: take it now).
6. ~~Phase-field scope~~ (decided: material models in the library + AT1/AT2 example; study to its repo).

## Progress (2026-11-18, branch `consolidate/library`, not pushed)

| Phase | State |
|---|---|
| 0 Safety | ✔ fable work committed; `fable/large-3d-to` and `main` merged; tag `pre-consolidation` = `cb9dbbd`; local `main` moved there (not pushed); baseline: 950 passed (muffto-gpu muGrid 1.4.1 and dev env muGrid 1.6.1) |
| 1 One branch | ✔ (merge commits `67b53d6`, `cb9dbbd`); `feat/fracture_and_contact`, `worktree-polycrystal-crack`, `StefanusMichael_Dev` left as they are |
| 2 Split | ✔ four private repos `MartinLadecky/muFFTTO-{dual-preconditioner,multigrid,contact,stefanus}` (pushed), laptop `/home/martin/PycharmProjects/<repo>`, same layout, frozen `muFFTTO/`, `env.sh`; data moved, fable data copied; polycrystal/fibre folders moved out of the polycrystal worktree. NEMO clones untouched |
| 3 Reuse muGrid | ✔ reductions (`GlobalReduction` on the communicator), Green (muGrid for one node per pixel), Green-Jacobi (muGrid), residual CG (muGrid); ◐ grid transfer / fused operator still muFFTTO's own (multigrid.py) until muGrid releases the multigrid |
| 4 Unify | ✔ one energy-error CG, `preconditioners` (incl. dual), `phase_field_fracture` (AT1/AT2 + example), `geometry.correlated_random_field`; ✘ third-medium contact functional, MPI/optimizer-vector helpers, Helmholtz L-BFGS start as a function, multigrid wrapper in `preconditioners` |
| 5 Clean up | ◐ dead CG variants and two legacy preconditioner methods removed, examples curated (26 + AT1/AT2, outputs untracked), docs updated; ✘ split `domain.py`, merge the grid-adaptation modules, remove the muFFTTO Green/Jacobi field API once the examples use `preconditioners` |
| 6 Port | ongoing in the muGrid/muTopOpt sessions (`TO_LIBRARY_PLAN.md` now in `muFFTTO-multigrid`) |

Environments: library dev venv `~/venvs/muFFTTO_dev` (muffto-gpu + muGrid 1.6.1 with the GPU multinode fix,
worktree `/home/martin/PycharmProjects/muGrid_161`, branch `muffto/1.6.1-gpu-multinode`); the studies keep `muffto-gpu`
(muGrid 1.4.1) with their frozen library copies. Old-code reference: worktree `muFFTTO_pre-consolidation`.

Known: `examples/topology_optimization_finite_strain/example_2D_finite_strain_TO.py` hits an indefinite Newton
tangent (also before the consolidation).
