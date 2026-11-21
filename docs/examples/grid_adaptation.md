# Homogenization on deformed (transformed) grids

Examples: [`examples/grid_adaptation/`](../../examples/grid_adaptation/)

| File | Physics |
|---|---|
| `example_2D_homogenization_conductivity_transformed_grid.py` | stationary heat conduction |
| `example_2D_homogenization_elasticity_transformed_grid.py` | small-strain linear elasticity |

Both scripts solve the standard periodic cell problem, but on a pixel grid whose
nodes have been moved. The unknowns, the FFT and the Green-operator
preconditioner stay on the regular grid, and the effect of the distortion is
pulled back into the operator.

Background: [cell problem](../theory.md#1-periodic-homogenization-the-cell-problem),
[physical problems](../theory.md#2-physical-problems-conductivity-and-small-strain-elasticity),
[FE on a pixel grid](../theory.md#4-finite-element-discretization-on-a-pixel-grid),
[linear solve](../theory.md#5-assembling-and-solving-the-linear-system),
[effective properties](../theory.md#6-effective-homogenized-properties),
[grid adaptation](../theory.md#8-grid-adaptation-deformed-grids).

## What PDE is solved

The physical (deformed) cell $\Omega_x$ is the image of the regular reference cell
$\Omega_X=[0,1)^2$ under a periodic map

$$x=\varphi(X)=X+d(X),\qquad F=\nabla_X\varphi=I+\nabla_X d,\qquad J=\det F ,$$

where $d$ is a periodic grid-node displacement. On $\Omega_x$ we solve the usual cell
problem for a periodic fluctuation $\tilde u$ under a macroscopic gradient $E$
(a vector for conductivity, a $2\times 2$ tensor for elasticity):

$$\nabla_x\cdot\big(C(x):(E+\nabla_x\tilde u)\big)=0 \quad\text{in } \Omega_x,\qquad \tilde u \text{ periodic}.$$

With $\nabla_x v=\nabla_X v\cdot F^{-1}$ and $\mathrm dx=J\,\mathrm dX$, the weak form is
transported to $\Omega_X$:

$$\int_{\Omega_X} J\,\big[C:\big((E+\nabla_X\tilde u)\cdot F^{-1}\big)\big]\cdot F^{-T} : \nabla_X v \;\mathrm dX = 0\qquad\forall v .$$

So the deformed-grid problem is a regular-grid problem with a spatially varying,
modified "material" $J\,F^{-1}\!\cdot C\cdot F^{-T}$. The discrete operator implemented
in `Discretization.apply_system_matrix_mugrid_deformed_grid` is

$$K u = B^T W\big[J\,\big(C:\mathrm{sym}(Bu\cdot F^{-1})\big)\cdot F^{-T}\big],$$

($\mathrm{sym}$ only when `formulation='small_strain'`), and the right-hand side
(`get_rhs_mugrid_deformed_grid`) is

$$f=-B^T W\big[J\,\big(C:(E\cdot F^{-1})\big)\cdot F^{-T}\big].$$

Note that $E$ is also multiplied by $F^{-1}$: the imposed affine field is $E\cdot X$
(linear in the *reference* coordinates), not $E\cdot x$. The two differ by the periodic
field $E\cdot d(X)$, which the fluctuation absorbs, and for a periodic map
$\frac{1}{|\Omega|}\int J F^{-T}\,\mathrm dX = I$ (Piola identity), so the mean physical
gradient is still $E$.

The effective tensor (`get_homogenized_stress_mugrid_deformed_grid`) is the
volume average of the flux/stress over the physical cell:

$$\bar\sigma=\frac{1}{|\Omega_X|}\sum_q w_q\,J_q\,C_q:\big((E+\nabla_X\tilde u)\cdot F_q^{-1}\big),$$

divided by the reference volume, which equals $|\Omega_x|$ because $d$ is periodic.

## How the grid is generated

Neither example calls a function from `muFFTTO.grid_adaptation_*` or
`muFFTTO.analytical_grid_adaptation`. The grid displacement is a hard-coded
analytical, periodic sine field in the $x$ direction only:

$$d_x(X)=a\,\sin(2\pi X_1)\sin(2\pi X_2),\qquad d_y=0,$$

with $a=0.1$ (conductivity) and $a=0.01$ (elasticity). It is a test of the
pulled-back operators, not a boundary-fitted mesh. The library's real adaptation tools
(`analytical_grid_adaptation.adapt_grid_to_circle`,
`grid_adaptation_methods_Zecevic.adapt_grid_to_circle` and
`grid_adaptation_arbitrary.run_grid_adaptation_workflow`, which do interface
projection and spring relaxation) produce node positions that could be used for
$d$ in the same way. See [theory §8](../theory.md#8-grid-adaptation-deformed-grids).

## Walkthrough: `example_2D_homogenization_conductivity_transformed_grid.py`

**1. Cell and discretization** (`:19-33`). A unit cell with 32×32 pixels and linear
triangles (two triangles per pixel, so two quadrature points).

**2. Material on the reference grid** (`:37-58`). Isotropic conductivity, $A=I$
everywhere, scaled by `mat_contrast_2 = 1e2` in the matrix (`phase > 0`) and by
`mat_contrast = 1` in the centred square inclusion $[0.25,0.75)^2$
(`'square_inclusion'`). The phase is assigned per **reference** pixel, so the
inclusion is carried along by the grid: the physical inclusion is the image of the
square under $\varphi$.

```python
material_data_field_C_0.s[...] = conductivity_C_1[:, :, np.newaxis, np.newaxis, np.newaxis]
material_data_field_C_0.s[..., matrix_mask] = mat_contrast_2 * material_data_field_C_0.s[..., matrix_mask]
```

**3. Grid displacement $d$ and deformed coordinates** (`:61-74`).

```python
grid_nodes_displacement_inxyz.s[0, 0, ...] = (0.1 * np.sin(2 * np.pi * ref_grid_coords_ixyz[0, ...]) *
                                              np.sin(2 * np.pi * ref_grid_coords_ixyz[1, ...]))
def_grid_coords_inxyz.s[:, 0, ...] = ref_grid_coords_ixyz[...] + grid_nodes_displacement_inxyz.s[:, 0, ...]
```

`def_grid_coords_inxyz` is computed but not used afterwards. Lines `:79-84`
build plotting coordinates with the periodic boundary nodes and show the
material on the deformed grid.

**4. Grid deformation gradient $F=I+\nabla_X d$, $J$, $F^{-1}$** (`:87-96`). The FE
gradient operator is applied to the nodal displacement, so $F$ is constant on
each triangle (one value per quadrature point). This is consistent with mapping
each element affinely:

```python
discretization.apply_gradient_operator_mugrid(grid_nodes_displacement_inxyz, F_ijqxy)
F_ijqxy.s[...] += np.eye(2)[:, :, None, None, None]
det_F.s[0,0,...] = np.linalg.det(F_ijqxy.s.transpose(2, 3, 4, 0, 1))
inv_F.s[...] = np.linalg.pinv(F_ijqxy.s.transpose(2, 3, 4, 0, 1)).transpose(3, 4, 0, 1, 2)
```

$J$ is plotted (`:99`; first quadrature point only).

**5. Operator and preconditioner** (`:102-126`). `K_fun` wraps
`apply_system_matrix_mugrid_deformed_grid(..., det_of_deformation_gradient=det_F,
inv_of_deformation_gradient=inv_F)`. The preconditioner is the ordinary
regular-grid Green operator for the reference conductivity $A_0=I$
(`get_preconditioner_Green_mugrid(reference_material_data_ijkl=conductivity_C_1)`).
It ignores the grid distortion, so the iteration count grows with the distortion,
but it stays FFT-diagonal.

**6. Loop over unit macroscopic gradients** (`:136-199`). For $E=e_i$, $i=1,2$:

- `get_macro_gradient_field_mugrid` broadcasts $E$ to all quadrature points (`:142`);
- `get_rhs_mugrid_deformed_grid` builds $f$ (`:150`);
- `muGrid.Solvers.conjugate_gradients` solves $K\tilde u=f$ with PCG, `rtol=1e-6`,
  `maxiter=2000` (`:166-175`);
- in serial the solution is plotted on the deformed grid (`:177-181`);
- `get_homogenized_stress_mugrid_deformed_grid` returns the averaged flux, which is
  row $i$ of $A^{\mathrm{eff}}$ (`:188-193`).

**7. Comparison** (`:206-208`). The script prints an analytical value

$$A^{\mathrm{eff}}_{11}=\sigma_m\sqrt{\frac{\sigma_m+3\sigma_i}{3\sigma_m+\sigma_i}},\qquad\sigma_m=100,\ \sigma_i=1,$$

(the closed-form result for a periodic array of square inclusions with
volume fraction 1/4). This value holds for the **undeformed** square inclusion. Because
the material is mapped with the grid, it is an exact reference only when the
grid displacement amplitude is set to 0 (`:70`). With the default $a=0.1$ the
printed values differ by the change in geometry, not only by discretization error.
To compare against a regular grid, set the amplitude to 0 and rerun. With $d=0$
the operators reduce exactly to the regular-grid ones ($F=I$, $J=1$).

## Variant: `example_2D_homogenization_elasticity_transformed_grid.py`

Differences from the conductivity script:

- Small-strain elasticity, isotropic $E=1,\ \nu=0.2$ via
  `material_models.get_bulk_and_shear_modulus` and
  `get_elastic_material_tensor(kind='linear')` (`:38-51`). The contrast is reversed:
  inclusion ×`1e2`, matrix ×1 (`:59-66`).
- Grid amplitude $a=0.01$ (`:79`).
- `K_fun` passes `formulation='small_strain'`, so the transformed gradient is
  symmetrized (`:117-122`).
- Four solves, $E=e_i\otimes e_j$, fill $C^{\mathrm{eff}}_{ijkl}$ (`:147-215`), and the
  result is printed in Voigt notation (`:218-222`).
- The CG call uses `tol=1e-6` instead of `rtol` (`:186`).
- The deformed configuration $x=\varphi(X)+E\cdot x+\tilde u$ is plotted with
  `visualization_utils.get_deformed_grid_coords_two_dim` (`:192-202`).
- There is no analytical reference value.

The rhs and the homogenized stress are called without `formulation='small_strain'`,
so they are not symmetrized. For a tensor with minor symmetries
$C:G=C:\mathrm{sym}\,G$, so this does not change the result.

## Summary

| File | Dim | Physics | Key method | Outputs |
|---|---|---|---|---|
| `example_2D_homogenization_conductivity_transformed_grid.py` | 2D | conductivity | pulled-back operator (`*_deformed_grid`), Green-PCG on reference grid | plots (material, $J$, solutions); printed $A^{\mathrm{eff}}$ and analytical $A^{\mathrm{eff}}_{11}$ |
| `example_2D_homogenization_elasticity_transformed_grid.py` | 2D | small-strain elasticity | same, `formulation='small_strain'` | plots (material, $J$, deformed solutions); printed $C^{\mathrm{eff}}$ (Voigt) |

## Key tunable parameters

| Parameter | Where | Meaning |
|---|---|---|
| `number_of_pixels` | `:25` / `:27` | grid resolution |
| amplitude `0.1` / `0.01` | `:70` / `:79` | grid distortion; 0 recovers the regular grid |
| `mat_contrast`, `mat_contrast_2` | `:37-38` / `:59-60` | phase contrast (inclusion / matrix) |
| `geometry_ID` | `:22` / `:24` | microstructure from `microstructure_library.get_geometry` |
| `element_type` | `:21` / `:22` | `'linear_triangles'` (other element types are untested here) |
| `rtol` / `tol`, `maxiter` | CG call | solver tolerance |

## How to run

```bash
python examples/grid_adaptation/example_2D_homogenization_conductivity_transformed_grid.py
python examples/grid_adaptation/example_2D_homogenization_elasticity_transformed_grid.py
```

Both scripts open several matplotlib windows. Run them serially. The solve and
reductions use the MPI communicator, but the material and $J$ plots (`:84`, `:99` /
`:91`, `:107`) are not guarded by a rank or size check.

## Known issues

- No grid-adaptation algorithm is used. The grid map is a hard-coded sine field,
  so "transformed grid" here means a synthetic distortion.
- The printed analytical $A^{\mathrm{eff}}_{11}$ (conductivity) refers to the undeformed
  square inclusion. With the default amplitude it is not a valid reference
  (see step 7).
- `def_grid_coords_inxyz` is computed but unused.
- The elasticity script prints the elapsed time on every rank (`:226`).
