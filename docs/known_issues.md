# Known issues

These issues were noticed while documenting the library in October 2026,
mostly by reading the code. The first section lists what has been fixed since;
each fix has a test in `test/test_regressions.py`. Line numbers are left out
because they change; search for the function name instead. Issues that belong
to a single example are listed at the end of that example's walkthrough page in
[`examples/`](examples/).

## Fixed

| Where | What was wrong |
|-------|----------------|
| `domain.py` `Discretization.get_flux_field_mugrid` | The einsum `'ij...,uj...->uj...'` computed $(\sum_i A_{ij})\,g_j$ instead of $A_{ij} g_j$, which is wrong for non-diagonal conductivity. It now uses `->ui...`, matching `apply_material_data_conductivity_mugrid`. |
| `domain.py` (16 places), `microstructure_library.get_geometry` | `raise ("...")` raised a string, so Python gave a generic `TypeError` and the message was lost. These now raise `TypeError("...")` / `NotImplementedError("...")` with the message. |
| `domain.py` `integrate_field`, `integrate_flux_field` | Both einsums were invalid in NumPy, so the functions always failed. They now sum over all quadrature and pixel axes. |
| `domain.py` | Removed 7 legacy methods that failed as soon as they were called: `get_system_matrix`, `get_preconditioner_Jacoby`, `apply_system_matrix`, `apply_system_matrix_explicit_stress`, `get_preconditioner_Green_fast`, `get_preconditioner_NEW`, `get_preconditioner_Jacoby_fast`. The working `_mugrid` versions remain. `apply_preconditioner_NEW` is kept because `apply_preconditioner_Green_Jacobi_full` uses it. That pair is unused, and its Jacobi input came from the removed `get_preconditioner_Jacoby_fast`. |
| `solvers_nonlinear.solve_finite_strain_newton_cg` | The identity was never added, so $F = H + \nabla\tilde u$. The solver now starts from $F = I$, $\tilde u = 0$. A new `newton_atol` argument accepts a state that is already in equilibrium, such as a homogeneous cell, without solving a linear system for round-off noise. |
| `solvers.py` `conjugate_gradients_mugrid`, `conjugate_gradients_mugrid_experimental`, `dr_pbcg_mugrid` | With `rtol=True`, the initial-guess check still used the absolute `tol`. A small right-hand side therefore returned the initial guess unsolved. With `rtol=True`, the initial guess is now accepted only if its residual is exactly zero. |
| `solvers.py` experimental CG, `___PCG`, `findS` | `Reduction.sum`/`.max` was applied to values that were already global, which multiplied the energy-error estimate by the number of ranks. Plain sums are now used. `___PCG` also no longer crashes on `list + float`, and checks `den > 0`. |
| `solvers.adam` | It crashed when `callback=None`, and it stopped as "converged" after any uphill step. It also tested against the new $\varphi$ instead of the previous one, and printed on every rank. All four are fixed. |
| `material_models.get_orthotropic_stiffness_tensor_plane_strain` | `nu21 = nu12*E1/E2` contradicted the symmetry $\nu_{21}/E_2 = \nu_{12}/E_1$. It is now `nu12*E2/E1`. |
| `material_models.compute_Voigt_notation_4order` | Now raises `ValueError` for dim ∉ {2, 3} instead of `UnboundLocalError`. |
| `microstructure_library` (superseded, see the next row) | `'abs_val'` now works in 3D. `'symmetric_linear'` no longer modifies the caller's `coordinates`. `'uniform_x1'`, which had no implementation, is no longer accepted. `check_equal_number_of_voxels` now rejects non-cubic grids. `visualize_voxels(figure=...)` no longer raises `NameError`. |
| `topology_optimization.py` | Deleted 11 legacy functions that could no longer run, about 1,500 lines: `objective_function_small_strain_pixel`, `objective_function_small_strain_FE_testing`, `compute_double_well_potential_Gauss_quad`, `partial_der_of_double_well_potential_wrt_density_NEW`, `partial_derivative_of_energy_equivalence_wrt_phase_field_FE`, `partial_derivative_of_objective_function_stress_equivalence_wrt_phase_field_pixel`, `sensitivity_with_adjoint_problem_pixel`, `sensitivity_with_adjoint_problem_FE_NEW`, `sensitivity_elastic_energy_and_adjoint_FE_NEW`, `sensitivity_with_adjoint_problem_FE_weights`, `sensitivity_with_adjoint_problem_FE_testing`. They called removed methods, expected an old tuple-returning `evaluate_field_at_quad_points`, or left out required arguments. None of them was used by the examples or tests. |
| `microstructure_library.py` | The 2,600-line geometry catalogue was replaced by the new `muFFTTO/geometry.py`. That module has composable periodic shapes, evaluates point-wise on local coordinates (so it is MPI-parallel), and keeps a registry of named geometries. Only the 5 geometries used by the examples remain: `square_inclusion`, `circle_inclusion`, `random_distribution`, `contact_test_geometry_1` and `contact_test_geometry_2`. They are bit-identical to before (`test/test_geometry.py`). The exception is `random_distribution`, which is now hash-based, so it gives the same field for any number of MPI ranks. `get_geometry` is kept as a thin wrapper, and `visualize_voxels` moved to `visualization_utils`. |
| all contractions | The code mixed the reversed contraction $C_{ijkl}\varepsilon_{lk}$ with silent symmetry assumptions. It now uses the standard $C_{ijkl}\varepsilon_{kl}$ (and $A_{ij}g_j$) everywhere, and stores the Neo-Hookean tangent as $\partial P_{ij}/\partial F_{kl}$. Explicit transposes are used where the maths needs them: $\mathbb C^T$ / $A^T$ in the adjoint right-hand sides, and the full gradient of $\lambda$ in the sensitivity. `Discretization.assert_material_symmetry` checks the symmetries the methods require (major for CG/adjoint, minor for small strain) where material data enters a solve. See theory §2.3. |
| `topology_optimization_conductivity.adjoint_potential` | Ghosts were refreshed on the empty output field instead of the flux that $B^T$ reads, so the `adjoint_energy` diagnostic was wrong. The sensitivities were not affected. |
| `domain.get_preconditioner_Green_mugrid` | Removed debug prints. |
| tests | `test/conftest.py` selects the non-interactive matplotlib backend, so tests that call `plt.show()` no longer block `pytest` or the pre-commit hook. |

## Open: may affect results

| Where | Issue |
|-------|-------|
| `material_models.get_elastic_material_tensor` vs `get_lame_parameters_from_bulk_and_shear(dim=2)` | In 2D, λ is computed as $K-\tfrac23\mu$ in one and $K-G$ in the other. The first treats $K$ as the 3D bulk modulus (plane strain); the second treats it as a 2D bulk modulus. This is a convention to decide, not a typo, so it has not been changed. |
| `material_models.get_orthotropic_stiffness_tensor_plane_strain` | The formulas give the reduced plane-*stress* stiffness, despite the function name. |
| `domain.py` `get_preconditioner_Green_mugrid` | The zero-frequency handling uses real-space `fft.icoords` to decide whether the local Fourier index 0 is $k=0$. This may be wrong on ranks that do not own $k=0$ under a pencil decomposition. It needs an MPI test with more than 2 ranks. |
| `domain.py` `Discretization.__init__` | `mpi_reduction` always uses `MPI.COMM_WORLD`, ignoring the communicator that is passed in. muGrid's `Communicator.mpi4py_comm` is `None` in the installed builds, so there is no simple fix. |
| `solvers.conjugate_gradients_mugrid_experimental` | With `rtol=True`, every criterion, including the energy-error ones, is scaled by $(r_0, r_0)$. |
| `visualization_utils.get_deformed_grid_coords_two_dim` | The periodic copy nodes (last row/column) do not get the grid displacement. |

## Open: legacy code that cannot run

Not removed yet.

- `grid_adaptation_methods_Zecevic.py`: the default `ref_grid_coords_ixyz=None` is not handled. Several helpers assume square grids (`N = shape[0]`).

## Open: questionable geometry and semantics

- `grid_adaptation_arbitrary.py`:
  - Node coordinates are laid out `(2, Ny, Nx)`, the transpose of the usual `[xy, nx, ny]`.
  - The nearest-edge projection ignores periodicity.
  - `fix_boundary` is unused.
- `domain.py`: the default `communicator=` argument is evaluated at import time.
- `domain.py`: the 3D node offsets in `__init__` look inconsistent (node 7 gets the z half-pixel twice).
- `discretization_library.trilinear_hex_1Q`: Q1 elements with one-point integration have hourglass modes. This may be intentional.
