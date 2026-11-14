# Fracture-Bench periodic analogues: Green vs dual (2026-11-11/12)

Paper: `../damage_AT2_small_strain/Krishnan.pdf` (Krishnan et al. 2026, arXiv:2611.03473, "Fracture-Bench", ML surrogates
for phase-field fracture). Their two datasets rebuilt with THEIR parameters as PERIODIC analogues (our solver is
periodic; their plates are Dirichlet-loaded). Question: does Green or the dual preconditioner win under mesh refinement?

## Scripts
| file | what |
|---|---|
| `fb_dataset1_AT2.py` | Dataset 1: copy of the AT2 script + `MICROSTRUCTURE=fb_hole` (hole r 0.5/2.2, 8 notches 0.05/2.2, notch width fixed at the 128² pixel), `LOAD_DIR=biax` |
| `fb_dataset2_pfczm.py` | Dataset 2: copy of `../localization_softening_bands/pfczm_concrete.py` (CPU/GPU, checkpoint), 4×8 mm cell `GRID=NX,2NX`, two inclusions (`INCL=hard,soft`), pre-crack (`CRACK_TIP`), `DRIVER=spectral` |
| `sbatch_fb_gpu.sh` | NEMO L40S job: `sbatch sbatch_fb_gpu.sh <1|2> <N> <Green|dual>` (24 h, NO self-resubmit) |
| `plot_fb_mesh_study.py` | 3 panels: D2 stress, D2 CG per solve per load step, Green/(2 dual) vs l/h for D1+D2; `STEP_MAX=13` -> `_s13` |
| `make_fb_dataset2_video.py` | video of D2 512×1024 dual (damage exact pixels + stress + CG) |
| `make_fb_dataset1_video.py` | video of D1 128² (`SRC=nemo` for the NEMO run to the break) |
| `plot_fb_dataset1.py`, `plot_fb_dataset2_progress.py` | early laptop plots (D2 one is from the WRONG-geometry run, ignore) |

## Parameters
- **D1** (AT2): μ = 1, λ = 0, Gc = 1.721e-5/2.2, l = 0.015/2.2 (l/h = 0.87 at 128²), residual η = 1e-3 (= `K_RESIDUAL`
  = `HOLE_STIFFNESS`), equal biaxial strain in steps of 1e-4. Their 44 steps give NO crack here (max d 0.43); the cell
  breaks at step ~47 (ε = 4.7e-3) -> NEMO runs use up to 120 steps, stop when cut through.
- **D2** (PF-CZM): E = 377 GPa, ν = 0.3, Gf = 10 N/m, b = l0 = 0.04 mm, linear softening, hard/soft inclusions ×10/×0.1
  on E, Gf, ft; ft not given by them -> 153.5 MPa (l_ch = 4 l0); residual k = 1e-4 (theirs unknown); load yy in steps
  1.25e-5, 80 steps.

## Results (NEMO, data in `exp_data_nemo/`, logs in `logs_nemo/`)
Total elastic CG, Green / (dual × 2):

| | grids (l/h or b/h) | Green/(2 dual) |
|---|---|---|
| D1, full runs (break at step ~46, one-step brittle break) | 128²–1024² (0.87, 1.75, 3.5, 7) | 0.40, 0.46, 0.87, 0.80 -> Green wins |
| D2, load steps 1–13 (peak + break; the physically meaningful part) | 128×256, 256×512, 512×1024 (1.28, 2.56, 5.12) | **2.25, 4.80, 4.88** -> dual wins |
| D2, all 80 steps | same | 2.75, 3.28, 3.53 |

D2 steps 1–13: Green 543k / 2.46M / 2.86M CG (718 / 1437 / 1259 per solve), dual×2 242k / 513k / 587k (320 / 299 / 258
per solve); 756 / 1714 / 2275 solves (identical for Green and dual).

**User (2026-11-12): D2 makes sense only up to load step 13** — after the break the stress sits at a ~2.3 MPa (8 %)
residual for the rest of the 80 steps (stop criterion 2 % never hit) = nonsense tail. Use `STEP_MAX=13` numbers.

## Caveats / open
- Periodic cells under macro strain, not their clamped plates; D1 small strain + scalar AT2 (theirs finite strain,
  vector phase field, random boundary profile).
- Unconverged steps (500 staggered-iteration cap, same steps for Green and dual): D2 steps 1–13: 0 / 2 / 4; full runs many.
- D2 512×1024 Green (job 8762983) was still running on 2026-11-12, will stop at its 23.3 h budget (~step 45, checkpoint
  written, no resubmit). Not needed for the step-13 numbers (it passed step 13). Sync later with
  `rsync nemo:/work/classic/fr_ml1145-muffto/runs/fracture_bench/exp_data/FB2_* exp_data_nemo/`.
- `exp_data/wrong_geometry_cpu_v1/`: laptop D2 runs of 2026-11-11 with WRONG geometry (`fft.coords` are cell
  fractions; the old CPU script scaled x by 8 mm instead of 4) — do not use.
