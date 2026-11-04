# Gypsum RVE: a physical phase-field length scale that has to be resolved

Purpose: a practical application in which the AT2 length scale l is a material property (fixed by strength and
toughness), the RVE is large compared to l, and resolving l with >= 5 elements (mesh objectivity) therefore needs large
grids -- and the cost of Green / Green-Jacobi / dual to get there.

## Sources (PDFs in this folder)

- Navidtehrani, Duddu, Martinez-Paneda, "Damage Mechanics Challenge: Predictions based on the phase field fracture
  model", Eng. Fract. Mech. 301 (2024) 110046 (`Navidtehrani_Duddu_MartinezPaneda_2024_EFM_damage_mechanics_challenge.pdf`):
  3D-printed gypsum calibrated to experiments: E = 600 MPa, nu = 0.2, Gc = 0.13 kJ/m^2 = 0.13 N/mm, sigma_c = 4.05 MPa,
  l = 0.5 mm (AT2: sigma_c = sqrt(27 E Gc / (256 l))). p. 6: "the characteristic element size has to be five times
  smaller than l to ensure mesh objectivity [31]" -- [31] Kristensen, Niordson, Martinez-Paneda, "An assessment of phase
  field fracture: crack initiation and growth", Phil. Trans. R. Soc. A 379 (2021) 20210021. They meet it with local
  mesh refinement along the expected crack; a uniform FFT grid has to meet it everywhere.
- Ambati, Gerasimov, De Lorenzis, "A review on phase-field models of brittle fracture and a new fast hybrid
  formulation", Comput. Mech. 55 (2015) 383-405 (`Ambati_Gerasimov_DeLorenzis_2015_CM_hybrid_phase_field.pdf`): the
  hybrid formulation used in our code.

## Setup (`run_AT2_gypsum_rve.sh`, script copy `example_2D_phase_field_fracture_AT2.py`)

| | value | code (length unit 25.6 mm, stress unit E) |
|---|---|---|
| RVE | 25.6 x 25.6 mm, periodic, plane strain | unit cell |
| microstructure | nine polydisperse disks, radii 1.3 - 3.6 mm (`benchmark/data/disks.csv` scaled) | `MICROSTRUCTURE=polydisperse_pores` |
| variant pores | porous gypsum: pores of stiffness k = 1e-4, no damage in the pores | `HOLES=1 HOLE_STIFFNESS=1e-4` |
| variant grains | stiff tough grains, 10x the matrix stiffness | `INCLUSION_STIFFNESS=10` |
| l | 0.5 mm (from sigma_c, Gc, E) | `LENGTH_SCALE=0.0195915` |
| Gc | 0.13 N/mm | `GC=8.4635e-6` (= 0.13 / (600 * 25.6)) |
| E, nu | 600 MPa, 0.2 (true plane strain) | `LAME=0.277778,0.416667` |
| k_res | 1e-4 | default |
| load | uniaxial strain E_xx to 0.015 in 150 steps (sigma_c / E = 6.75e-3) | `LOAD_MAX=0.015 LOAD_STEPS=150` |
| grids | 64^2 / 128^2 / 256^2 / 512^2: h = 0.4 / 0.2 / 0.1 / 0.05 mm, l/h = 1.25 / 2.5 / 5 / 10 | `NPIX` |

Mesh objectivity (l/h >= 5) starts at 256^2 for this RVE (in 3D at 256^3). First 64^2 - 256^2 on the CPU, all three
preconditioners; 512^2 on the GPU after.

## Results (2026-11-01, 64^2 - 256^2; `plot_AT2_gypsum.py` -> `figures/AT2_gypsum.png`)

Total elastic work of the run (CG iterations, dual x 2), peak stress, number of elastic solves:

| variant | N | l/h | Green | Green-Jacobi | dual | peak MPa | solves |
|---|---|---|---|---|---|---|---|
| pores | 64 | 1.25 | **43k** | 70k | 157k | 1.103 | 645 |
| pores | 128 | 2.5 | **70k** | 120k | 159k | 1.105 | 599 |
| pores | 256 | 5 | **385k** | 416k | 467k | 1.122 | 1469 |
| pores | 512 | 10 | **565k** | 855k | 596k | 1.127 | 1708 (GPU: dual 2.1 h, Green 2.0 h, GJ 3.1 h) |
| pores | 1024 | 20 | 566k | 1517k | **559k** | 1.128 | 1594 (GPU: dual 8.5 h, Green 9.0 h; GJ 4.6 h on NEMO) |
| grains | 64 | 1.25 | **14.7k** | 21.3k | 15.2k | 4.064 | 261 |
| grains | 128 | 2.5 | 30.1k | 37.5k | **17.7k** | 4.053 | 279 |
| grains | 256 | 5 | 131.5k | 77.4k | **30.4k** | 4.044 | 360 |
| grains | 512 | 10 | 1624k | 423k | **225k** | 4.041 | 1671 (GPU, 2026-11-02) |
| grains | 1024 | 20 | 1330k | 505k | **164k** | 4.041 | 1367 (GPU: dual 2.7 h, GJ 8.1 h; Green 3.9 h on NEMO) |

- Physics: grains -- one sudden break, peak 4.04-4.06 MPa (= sigma_c of the material), converging with l/h; pores --
  stepwise cracking (three drops), macroscopic peak ~1.1 MPa (porosity, stress concentrations), the crack sequence
  settles from l/h = 2.5 on, the peak changes by 1.7 % from l/h = 2.5 to 5.
- Grains: the dual wins at the mesh-objective resolution l/h = 5 by 2.5x (Green-Jacobi) and 4.3x (Green); 64^2 -> 256^2
  dual x2, Green-Jacobi x3.6, Green x9.
- Pores (pores = k = 1e-4, nearly two-phase field with a 1e4 jump on every pore boundary): Green is the cheapest on all
  three grids; per elastic solve the dual grows least (265 -> 318 from 128^2 to 256^2, Green x2.3, Green-Jacobi x1.4)
  but from a higher level, and the number of staggered solves grows for all (599 -> ~1460). No crossover up to 256^2.
- Grains 512^2 (l/h = 10): dual 225k, Green-Jacobi 423k (1.9x), Green 1624k (7.2x). 1488 of the 1671 solves are
  staggered iterations in the single break step; there Green needs ~1000 CG iterations per solve, Green-Jacobi ~600,
  the dual ~75 (x2). Before the peak: dual ~20, Green ~25, Green-Jacobi ~400 per solve.
- Consistent with the main study: graded / multi-level fields (stiff grains) -> dual; uniform pores at the broken
  stiffness -> Green.

1024^2 grains Green and pores Green-Jacobi ran on NEMO (GPU, 2026-11-03/04, data copied from
`/work/classic/fr_ml1145-muffto/runs/AT2/`); the NEMO pores Green run reproduces the local one (same 1594 solves). At
l/h = 20 the dual is 8.1x cheaper than Green and 3.1x than Green-Jacobi with grains; with pores Green and dual tie
(566k / 559k) and Green-Jacobi is 2.7x more expensive. The killed local partial runs are kept as `*_localkilled*`.
