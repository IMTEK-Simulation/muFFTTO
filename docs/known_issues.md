# Known issues (suspected)

These issues were noticed while documenting the library in October 2026. They
were found by reading the code, not by running it, and **none have been fixed**.
Verify each one before changing anything. Line numbers are approximate because
the comments added to the modules moved the code. Search for the function name.
Issues that are specific to one example are listed at the end of that example's
walkthrough page in [`examples/`](examples/).

## Likely to affect results

| Where | Issue |
|-------|-------|
| `domain.py` `Discretization.get_flux_field_mugrid` | einsum `'ij...,uj...->uj...'` computes $(\sum_i A_{ij})\,g_j$ rather than $A_{ij} g_j$. This is only correct for diagonal conductivity. The explicit term in `topology_optimization_conductivity.py` uses the correct `->ui...`, so the two sensitivity terms are inconsistent for anisotropic $A$. |
| `topology_optimization.py` (legacy SIMP derivative) | `np.power(p * rho, p - 1)` computes $(p\rho)^{p-1}$ instead of $p\rho^{p-1}$. |
| `material_models.py` `get_elastic_material_tensor` vs `get_lame_parameters_from_bulk_and_shear(dim=2)` | 2D λ is computed two ways ($K-\tfrac23\mu$ vs $K-G$), so the same $(K,G)$ gives a different material in each. |
| `material_models.py` `get_orthotropic_stiffness_tensor_plane_strain` | `nu21 = nu12*E1/E2` contradicts the symmetry $\nu_{21}/E_2=\nu_{12}/E_1$. The formulas are the plane-*stress* reduced stiffness. |
| `solvers_nonlinear.py` `solve_finite_strain_newton_cg` | Never adds the identity to `'total_strain_field'`, so a fresh field gives $F = H + \nabla\tilde u$. The examples work around this by initialising the field to $I$. |
| `solvers.py` experimental CG / `___PCG` | `Reduction(...).sum(Delta[l:-1])` reduces over ranks values that are already global, so it is P times too large on P ranks. |
| `solvers.py` `conjugate_gradients_mugrid*` | The initial convergence test uses the absolute `tol` even when `rtol=True`. |
| `domain.py` `get_preconditioner_Green_fast` / `_mugrid` | The zero-frequency handling may be wrong on ranks that do not own $k=0$ (pencil decomposition). |
| `domain.py` `integrate_field` | `'fdqxy...->fd...'` keeps the `z` axis in 3D. |
| `visualization_utils.py` `get_deformed_grid_coords_two_dim` | The periodic copy nodes (last row/column) do not get the grid displacement. |

## Will crash if reached

- `raise ("...")` / `raise "..."` raises a string, which gives a `TypeError` and loses the message. This happens in about 16 places in `domain.py` and in `microstructure_library.get_geometry`.
- `domain.py`: `apply_system_matrix`, `apply_system_matrix_explicit_stress`, `get_system_matrix` and `get_preconditioner_Jacoby` call `apply_gradient_operator*` methods that no longer exist.
- `topology_optimization.py`: many legacy functions (`*_pixel`, `*_FE_weights`, `*_FE_testing`, `*_FE_NEW`) call removed methods (`apply_gradient_operator`, `get_preconditioner`, `solvers.PCG`, …). Some unpack `evaluate_field_at_quad_points` as a tuple, or omit the required `void_material_data_ijkl` / `output_1nxyz` arguments.
- `microstructure_library.get_geometry`: `'uniform_x1'` is allowed but has no `case`. In the chiral parameter defaults, only the first missing key is filled. `visualize_voxels(figure=...)` raises `NameError`.
- `material_models.compute_Voigt_notation_4order`: `UnboundLocalError` for dim ∉ {2, 3}.
- `solvers.py` `___PCG`: `curve` is a list, so `curve + Delta[-1]` fails. `adam` calls `callback` even when it is `None`.
- `grid_adaptation_methods_Zecevic.py`: `ref_grid_coords_ixyz=None` is the default but is not handled. Several helpers assume square grids (`N = shape[0]`).

## Questionable geometry / semantics

- `microstructure_library.get_geometry`:
  - `'abs_val'` (3D) never assigns its result.
  - `'sine_wave_rapid'` / `'tanh'` (3D) return shape `(3, ...)`.
  - `'cos_wave'` (3D) is not periodic in `z`.
  - `'sine_wave_inv'` (3D) is not inverted.
  - `'symmetric_linear'` changes the caller's `coordinates` in place.
  - `'circles'` takes the cell size from local coordinates, which is wrong under MPI.
- Chiral metamaterials:
  - The input-validation asserts are logically inverted.
  - `hy` is checked twice where `hz` was meant.
  - `chiral_metamaterial_2` uses `t_half_x` for a z-range.
- `check_equal_number_of_voxels` uses `a != b != c`, which does not test all-equal.
- `grid_adaptation_arbitrary.py`:
  - Node coordinates are laid out `(2, Ny, Nx)`, the transpose of the usual `[xy, nx, ny]`.
  - The nearest-edge projection ignores periodicity.
  - `fix_boundary` is unused.
- `domain.py`:
  - `mpi_reduction` always uses `MPI.COMM_WORLD`, ignoring the communicator passed in.
  - The default `communicator=` argument is evaluated at import time.
  - The 3D node offsets in `__init__` look inconsistent (node 7 gets the z half-pixel twice).
- `discretization_library.trilinear_hex_1Q`: one-point integration of Q1, so it has hourglass modes. This may be intentional.
