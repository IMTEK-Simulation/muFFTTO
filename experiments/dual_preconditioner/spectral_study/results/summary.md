# Dual preconditioner study

PCG iterations to ||e_k||_K <= 1e-6 ||e_0||_K (true energy error); "-" = P indefinite. Cost per application in Green (FFT-pair) applications: Green, Green-Jacobi 1; dual 2 (+1 K); Schur Neumann k: k+3; Schur A: 4.

## 16²

| physics | geometry | k_v | Green | Green-Jacobi | dual (C_ref = I) | dual (C_ref = solid) | Schur, Neumann 1 | Schur, Neumann 3 | Schur, S~ = Γ_S A Γ_S |
|---|---|---|---|---|---|---|---|---|---|
| scalar | bubble_array | 0.1 | 6 | 7 | 5 | 5 | 6 | 5 | - |
| scalar | bubble_array | 0.001 | 6 | 9 | 5 | 5 | 6 | 6 | - |
| scalar | bubble_array | 1e-05 | 6 | 9 | 5 | 5 | 6 | 6 | - |
| scalar | square_inclusion | 0.1 | 5 | 9 | 3 | 3 | 5 | 5 | - |
| scalar | square_inclusion | 0.001 | 5 | 11 | 3 | 3 | 6 | 7 | - |
| scalar | square_inclusion | 1e-05 | 5 | 11 | 3 | 3 | 6 | 7 | - |
| elastic | bubble_array | 0.1 | 11 | 14 | 13 | 7 | 11 | 8 | - |
| elastic | bubble_array | 0.001 | 13 | 20 | 28 | 9 | 29 | 29 | - |
| elastic | bubble_array | 1e-05 | 13 | 20 | 36 | 9 | 39 | 45 | - |
| elastic | square_inclusion | 0.1 | 9 | 17 | 13 | 6 | 11 | 8 | - |
| elastic | square_inclusion | 0.001 | 10 | 22 | 29 | 7 | 33 | 33 | - |
| elastic | square_inclusion | 1e-05 | 10 | 22 | 37 | 7 | 39 | 42 | - |

Spectral interval [lambda_min, lambda_max] of P K, 16²:

| physics | geometry | k_v | Green | Green-Jacobi | dual (C_ref = I) | dual (C_ref = solid) | Schur, Neumann 1 | Schur, Neumann 3 |
|---|---|---|---|---|---|---|---|---|
| scalar | bubble_array | 0.1 | [0.1, 1] | [0.068, 0.75] | [1, 3] | [1, 3] | [1, 2.1] | [1, 1.5] |
| scalar | bubble_array | 0.001 | [0.001, 1] | [0.037, 1.2] | [1, 2.5e+02] | [1, 2.5e+02] | [1, 1.5e+02] | [1, 83] |
| scalar | bubble_array | 1e-05 | [1e-05, 1] | [0.035, 1.3] | [1, 2.5e+04] | [1, 2.5e+04] | [1, 1.5e+04] | [1, 8.2e+03] |
| scalar | square_inclusion | 0.1 | [0.1, 1] | [0.12, 0.56] | [1, 3] | [1, 3] | [1, 2.1] | [1, 1.4] |
| scalar | square_inclusion | 0.001 | [0.001, 1] | [0.085, 0.82] | [1, 2.5e+02] | [1, 2.5e+02] | [1, 1.5e+02] | [1, 70] |
| scalar | square_inclusion | 1e-05 | [1e-05, 1] | [0.083, 0.85] | [1, 2.5e+04] | [1, 2.5e+04] | [1, 1.5e+04] | [1, 6.9e+03] |
| elastic | bubble_array | 0.1 | [0.1, 1] | [0.033, 1.2] | [1, 5.9] | [1, 3] | [1, 3.6] | [1, 2.1] |
| elastic | bubble_array | 0.001 | [0.001, 1] | [0.0084, 2.2] | [1, 5.3e+02] | [1, 2.5e+02] | [1, 2.9e+02] | [1, 1.4e+02] |
| elastic | bubble_array | 1e-05 | [1e-05, 1] | [0.0079, 2.4] | [1, 5.3e+04] | [1, 2.5e+04] | [1, 2.9e+04] | [1, 1.4e+04] |
| elastic | square_inclusion | 0.1 | [0.1, 1] | [0.088, 0.87] | [1, 5.1] | [1, 3] | [1, 3.4] | [1, 2.1] |
| elastic | square_inclusion | 0.001 | [0.001, 1] | [0.057, 1.3] | [1, 4.5e+02] | [1, 2.5e+02] | [1, 2.8e+02] | [1, 1.5e+02] |
| elastic | square_inclusion | 1e-05 | [1e-05, 1] | [0.055, 1.4] | [1, 4.5e+04] | [1, 2.5e+04] | [1, 2.8e+04] | [1, 1.5e+04] |

## 32²

| physics | geometry | k_v | Green | Green-Jacobi | dual (C_ref = I) | dual (C_ref = solid) | Schur, Neumann 1 | Schur, Neumann 3 | Schur, S~ = Γ_S A Γ_S |
|---|---|---|---|---|---|---|---|---|---|
| scalar | bubble_array | 0.1 | 10 | 13 | 6 | 6 | 7 | 6 | - |
| scalar | bubble_array | 0.001 | 11 | 19 | 7 | 7 | 12 | 22 | - |
| scalar | bubble_array | 1e-05 | 11 | 20 | 7 | 7 | 13 | 24 | - |
| scalar | square_inclusion | 0.1 | 6 | 12 | 4 | 4 | 5 | 5 | - |
| scalar | square_inclusion | 0.001 | 6 | 15 | 4 | 4 | 6 | 8 | - |
| scalar | square_inclusion | 1e-05 | 6 | 16 | 4 | 4 | 6 | 8 | - |
| elastic | bubble_array | 0.1 | 15 | 28 | 15 | 8 | 12 | 9 | - |
| elastic | bubble_array | 0.001 | 23 | 57 | 46 | 12 | 55 | 65 | - |
| elastic | bubble_array | 1e-05 | 23 | 60 | 69 | 12 | 82 | 117 | - |
| elastic | square_inclusion | 0.1 | 9 | 21 | 13 | 6 | 11 | 8 | - |
| elastic | square_inclusion | 0.001 | 11 | 33 | 35 | 7 | 39 | 44 | - |
| elastic | square_inclusion | 1e-05 | 11 | 34 | 47 | 7 | 55 | 61 | - |

Spectral interval [lambda_min, lambda_max] of P K, 32²:

| physics | geometry | k_v | Green | Green-Jacobi | dual (C_ref = I) | dual (C_ref = solid) | Schur, Neumann 1 | Schur, Neumann 3 |
|---|---|---|---|---|---|---|---|---|
| scalar | bubble_array | 0.1 | [0.1, 1] | [0.041, 1.1] | [1, 3] | [1, 3] | [1, 2.1] | [1, 1.5] |
| scalar | bubble_array | 0.001 | [0.001, 1] | [0.018, 2.3] | [1, 2.5e+02] | [1, 2.5e+02] | [1, 1.5e+02] | [1, 83] |
| scalar | bubble_array | 1e-05 | [1e-05, 1] | [0.017, 2.4] | [1, 2.5e+04] | [1, 2.5e+04] | [1, 1.5e+04] | [1, 8.2e+03] |
| scalar | square_inclusion | 0.1 | [0.1, 1] | [0.083, 0.81] | [1, 3] | [1, 3] | [1, 2.1] | [1, 1.4] |
| scalar | square_inclusion | 0.001 | [0.001, 1] | [0.05, 1.4] | [1, 2.5e+02] | [1, 2.5e+02] | [1, 1.5e+02] | [1, 73] |
| scalar | square_inclusion | 1e-05 | [1e-05, 1] | [0.048, 1.4] | [1, 2.5e+04] | [1, 2.5e+04] | [1, 1.5e+04] | [1, 7.3e+03] |
| elastic | bubble_array | 0.1 | [0.1, 1] | [0.018, 1.9] | [1, 6.1] | [1, 3] | [1, 3.7] | [1, 2.1] |
| elastic | bubble_array | 0.001 | [0.001, 1] | [0.0023, 4.3] | [1, 5.6e+02] | [1, 2.5e+02] | [1, 3e+02] | [1, 1.5e+02] |
| elastic | bubble_array | 1e-05 | [1e-05, 1] | [0.0021, 4.6] | [1, 5.5e+04] | [1, 2.5e+04] | [1, 3e+04] | [1, 1.5e+04] |
| elastic | square_inclusion | 0.1 | [0.1, 1] | [0.059, 1.3] | [1, 5.1] | [1, 3] | [1, 3.5] | [1, 2.2] |
| elastic | square_inclusion | 0.001 | [0.001, 1] | [0.033, 2.2] | [1, 4.5e+02] | [1, 2.5e+02] | [1, 2.9e+02] | [1, 1.6e+02] |
| elastic | square_inclusion | 1e-05 | [1e-05, 1] | [0.031, 2.3] | [1, 4.5e+04] | [1, 2.5e+04] | [1, 2.9e+04] | [1, 1.6e+04] |

