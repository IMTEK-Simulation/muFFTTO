# Grid-aware Green preconditioners: 2x2 nodal block Jacobi

Question: can the Green preconditioner take the deformation of an adapted grid into account?

The Green operator is FFT-diagonal, so it cannot contain the spatially varying geometric coefficients
J_g F_g^-1 (.) F_g^-T of the pulled-back operator. A nodal scaling S can, P = S G S. Compared on real Newton
systems (first Newton step of an increment, predictor state) of the slide runs, 32², energy stop 1e-6, delay 10:

| variant | G | scaling S |
|---|---|---|
| Green | solid reference | none |
| Green-Jacobi | identity reference | scalar diagonal of the tangent (the runs) |
| Green + block Jacobi | identity reference | 2x2 nodal blocks of the tangent |
| Green + grid Jacobi | solid reference | diagonal of the pulled-back reference operator (grid only) |
| Green + grid block Jacobi | solid reference | its 2x2 nodal blocks |

The blocks are probed (one component on every 4th node, 32 operator applications); the deviation from exact
unit-vector columns is ~1e-17.

## Results (CG iterations to the energy stop)

| grid, problem | Green | Green-Jacobi | + block Jacobi | + grid Jacobi | + grid block Jacobi |
|---|---|---|---|---|---|
| adapted, trust region (5 states) | 4086 - 4480 | 585 - 652 | 585 - 650 | 2040 - 2179 | 2135 - 2324 |
| adapted, contact (3 states) | 4300 - 5926 | 542 - 882 | 543 - 865 | 2108 - 2291 | 2113 - 2336 |
| regular, trust region (5 states) | 374 - 734 | 70 - 87 | 70 - 86 | 369 - 734 | 369 - 734 |

- 2x2 blocks instead of the scalar diagonal: no gain anywhere (within round-off). The x-y coupling of the stretched
  elements does not limit the convergence.
- The grid-only scaling halves the cost of Green on the adapted grid (and does nothing on the regular grid, as it
  must), but stays 3.5x behind Green-Jacobi: most of the Green-Jacobi gain comes from the material contrast.
- The adapted grid itself costs about 8x more CG for every variant (Green-Jacobi ~600 vs ~75, Green ~4300 vs ~600).
  No local (nodal) scaling removes that: the difficulty is non-local, the FFT reference does not see how the grid
  map varies over the cell. A remedy would need a non-local correction (e.g. an inner solve with the pulled-back
  reference operator) or a milder grid adaptation (larger min_jacobian).
- Round-off: on the adapted grid two versions of Green-Jacobi that differ only by round-off (1e-16) can need 652 vs
  884 CG (loss of orthogonality); single-state differences below ~30 % are noise.
- On the adapted grid every variant stops with true energy errors of 1e-6 ... 1e-5: the delayed estimate is less
  reliable there than on the regular grid.

## Files

- `block_jacobi_study.py`: the study (`--replot OUT` redraws a figure from its npz)
- `run_all.sh`: the three studies above
- `results_*.log|npz|png`: output
