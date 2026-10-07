# Homogenization of heat conductivity (examples walkthrough)

This page explains the scripts in `examples/homogenization/conductivity/`. Each script
computes the **effective (homogenized) conductivity tensor** $\mathbf{A}^{\mathrm{eff}}$
of a periodic two-phase unit cell. It discretizes the cell problem with finite elements on a
pixel/voxel grid and solves it with an FFT-preconditioned conjugate gradient (PCG) method.

For background, see the theory notes:
[cell problem](../theory.md#1-periodic-homogenization-the-cell-problem),
[conductivity](../theory.md#2-physical-problems-conductivity-and-small-strain-elasticity),
[FE on a pixel grid](../theory.md#4-finite-element-discretization-on-a-pixel-grid),
[linear system & preconditioner](../theory.md#5-assembling-and-solving-the-linear-system),
[effective properties](../theory.md#6-effective-homogenized-properties),
[array conventions](../theory.md#9-notation-and-array-conventions).

## Which PDE is solved

The unit cell is $Y=[0,L_1)\times[0,L_2)(\times[0,L_3))$ and the local conductivity is $\mathbf{A}(\mathbf{x})$.
For a prescribed macroscopic temperature gradient $\mathbf{E}\in\mathbb{R}^d$, the temperature splits as
$\theta(\mathbf{x}) = \mathbf{E}\cdot\mathbf{x} + \tilde\theta(\mathbf{x})$, where the fluctuation $\tilde\theta$ is $Y$-periodic.

**Strong form.** Find a periodic $\tilde\theta$ such that

$$
-\nabla\cdot\big(\mathbf{A}(\mathbf{x})\,(\mathbf{E}+\nabla\tilde\theta(\mathbf{x}))\big) = 0 \quad\text{in } Y .
$$

**Weak form.** Find a periodic $\tilde\theta$ such that for all periodic test functions $v$:

$$
\int_Y \nabla v \cdot \mathbf{A}\,\big(\mathbf{E}+\nabla\tilde\theta\big)\,\mathrm{d}\mathbf{x} = 0 .
$$

**Discrete system.** $\mathsf{B}$ is the FE gradient operator (nodal values to gradients at quadrature
points), $\mathsf{W}$ holds the quadrature weights and $\mathsf{A}$ the material data at quadrature points. The discrete system is

$$
\underbrace{\mathsf{B}^\top\mathsf{W}\mathsf{A}\mathsf{B}}_{\mathsf{K}}\;\tilde{\boldsymbol\theta}
= \underbrace{-\,\mathsf{B}^\top\mathsf{W}\mathsf{A}\,\mathbf{E}}_{\mathbf{b}} .
$$

$\mathsf{K}$ is symmetric positive semi-definite. Its null space holds the constant fields, so it is
solved with PCG. The preconditioner is the inverse of a reference operator
$\mathsf{K}_{\mathrm{ref}} = \mathsf{B}^\top\mathsf{W}\mathbf{A}_{\mathrm{ref}}\mathsf{B}$
for a homogeneous material. The FFT block-diagonalizes it, so it can be inverted
frequency by frequency (a "Green's function" preconditioner).

**Effective tensor.** Solve once for each unit load $\mathbf{E}=\mathbf{e}_i$. Then the $i$-th row of the effective tensor is the
average flux

$$
\mathbf{A}^{\mathrm{eff}}\mathbf{e}_i = \frac{1}{|Y|}\int_Y \mathbf{A}\,(\mathbf{e}_i+\nabla\tilde\theta_i)\,\mathrm{d}\mathbf{x}
\approx \frac{1}{|Y|}\sum_{\text{pixels}}\sum_q w_q\,\mathbf{A}_q(\mathbf{e}_i+\nabla\tilde\theta_i)_q .
$$

## Overview of the examples

| File | Dim | Element (`element_type`) | Grid (default) | Microstructure | Linear solver | Preconditioner | Key outputs |
|---|---|---|---|---|---|---|---|
| `example_2D_homogenization_conductivity.py` | 2D | `linear_triangles` (2 triangles/pixel, 1 node/pixel) | 128×128 | `square_inclusion` | `muGrid.Solvers.conjugate_gradients`, `rtol=1e-6` | Green (FFT), $\mathbf{A}_{\mathrm{ref}}=\mathbf{I}$ | $2\times2$ $\mathbf{A}^{\mathrm{eff}}$, analytical comparison for $A^{\mathrm{eff}}_{11}$, one plot of $\tilde\theta$ per load case |
| `example_2D_homogenization_conductivity_quadratic_basis.py` | 2D | `biquadratic_rectangle` (Q9, 4 nodes/pixel, 3×3 Gauss) | 64×64 | `square_inclusion` | same as above | Green (FFT) | same as above |
| `example_2D_homogenization_conductivity_block_CG.py` | 2D | `linear_triangles` | 128×128 | `square_inclusion` (or phase-scaled) | muFFTTO `solvers.conjugate_gradients_mugrid`, then block CG `solvers.dr_pbcg_mugrid` | Green (FFT) | $\mathbf{A}^{\mathrm{eff}}$ from both solvers, timings, block-CG residual history |
| `example_3D_homogenization_conductivity.py` | 3D | `trilinear_hexahedron` (2×2×2 Gauss) | 128³ | `square_inclusion` (centred cube) | `muGrid.Solvers.conjugate_gradients`, `rtol=1e-6` | Green (FFT) | $3\times3$ $\mathbf{A}^{\mathrm{eff}}$, mid-plane plots |
| `example_3D_homogenization_conductivity_block_CG.py` | 3D | `trilinear_hexahedron` | 64³ | `random_distribution` | muFFTTO PCG then block CG (absolute `tol=1e-5`) | Green (FFT) | $3\times3$ $\mathbf{A}^{\mathrm{eff}}$ from both solvers, convergence plot (single vs. block CG) |

All scripts are linear problems with one load case per spatial direction ($d$ load cases).

---

## Walkthrough: `example_2D_homogenization_conductivity.py`

### 1. Unit cell and discretization

```python
problem_type = 'conductivity'          # :16
element_type = 'linear_triangles'      # :18
domain_size = [1, 1]                   # :21
number_of_pixels = (128, 128)          # :22
my_cell = domain.PeriodicUnitCell(domain_size=domain_size, problem_type=problem_type)   # :24
discretization = domain.Discretization(cell=my_cell, nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)                     # :27
```

- `PeriodicUnitCell` (`:24`) defines $Y=[0,1)^2$. `problem_type='conductivity'` makes the unknown a
  scalar field (one component, $f=1$).
- `Discretization` (`:27`) builds the reference element: shape functions, quadrature and the
  $\mathsf{B}$ stencil. It also creates the distributed `muGrid.FFTEngine` with one ghost layer per side. With
  `linear_triangles` each pixel is split into two linear triangles with one quadrature point each
  (2 quadrature points per pixel). Each pixel owns one node, so the unknown has
  $128\times128$ degrees of freedom.

### 2. Material data at quadrature points

```python
mat_contrast = 1                                                                  # :34
mat_contrast_2 = 1e2                                                              # :35
conductivity_C_1 = np.array([[1., 0], [0, 1.0]])                                  # :36
material_data_field_C_0 = discretization.get_material_data_size_field_mugrid(name='conductivity_tensor')  # :38
material_data_field_C_0.s[...] = conductivity_C_1[:, :, np.newaxis, np.newaxis, np.newaxis]               # :41
```

The material field has layout `[i, j, q, x, y]`: one $d\times d$ tensor $\mathbf{A}_q$ at every quadrature
point $q$ of every pixel. Line `:41` broadcasts the identity tensor to all quadrature points.

### 3. Microstructure (phase field)

```python
phase_field_geom = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                       microstructure_name=geometry_ID,
                                                       coordinates=discretization.fft.coords)  # :44
matrix_mask = phase_field_geom > 0                                                             # :50
inc_mask = phase_field_geom == 0                                                               # :51
material_data_field_C_0.s[..., matrix_mask] = mat_contrast_2 * material_data_field_C_0.s[..., matrix_mask]  # :55
material_data_field_C_0.s[..., inc_mask] = mat_contrast * material_data_field_C_0.s[..., inc_mask]          # :56
```

`'square_inclusion'` returns a pixel field. It is 1 in the matrix and 0 in the centred square
$[0.25,0.75)^2$, so the inclusion volume fraction is $1/4$. Each rank gets only its local subdomain, because
`discretization.nb_of_pixels` and `fft.coords` are local. The masks scale the conductivity:
$\mathbf{A} = 100\,\mathbf{I}$ in the matrix and $\mathbf{A}=\mathbf{I}$ in the inclusion. The material is
constant per pixel, so all quadrature points in a pixel get the same value.

### 4. System matrix $\mathsf{K}$ (matrix-free)

```python
def K_fun(x, Ax):                                                                  # :58
    discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0,
                                              input_field_inxyz=x, output_field_inxyz=Ax)  # :64
    discretization.fft.communicate_ghosts(Ax)                                      # :67
```

`apply_system_matrix_mugrid` computes $\mathsf{K}\mathbf{x} = \mathsf{B}^\top\mathsf{W}\mathsf{A}\mathsf{B}\mathbf{x}$
with stencils:

1. gradient at the quadrature points;
2. multiply by $\mathbf{A}_q$;
3. multiply by the weights $w_q$;
4. apply the transposed gradient (divergence).

The matrix is never assembled. The ghost update (`:67`) keeps the halo layers consistent for the next stencil
application when the grid is split across MPI ranks.

### 5. Preconditioner

```python
preconditioner = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=conductivity_C_1)  # :70
def M_fun(x, Px):                                                                  # :72
    discretization.fft.communicate_ghosts(x)
    discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                               input_nodal_field_fnxyz=x, output_nodal_field_fnxyz=Px)  # :78
```

- `get_preconditioner_Green_mugrid` (`:70`) computes the response of $\mathsf{K}_{\mathrm{ref}}$ (with
  $\mathbf{A}_{\mathrm{ref}}=\mathbf{I}$) to a unit impulse and Fourier-transforms it. It then inverts the
  small block for every wave vector $\mathbf{k}$. For one node per pixel, the $\mathbf{k}=0$ block (the constant mode) is
  not inverted. It stays numerically zero, so the preconditioner removes the mean.
- `apply_preconditioner_mugrid` (`:78`) computes $\mathsf{M}^{-1}\mathbf{r} = \mathcal{F}^{-1}[\hat{\mathsf{K}}_{\mathrm{ref}}(\mathbf{k})^{-1}\hat{\mathbf{r}}(\mathbf{k})]$
  with one forward FFT and one inverse FFT.
- Scaling $\mathbf{A}_{\mathrm{ref}}$ by a constant does not change the PCG iterates. What matters is the
  spectral equivalence between $\mathsf{K}$ and $\mathsf{K}_{\mathrm{ref}}$, so the iteration count grows
  with the phase contrast (here 100), not with the grid size.

### 6. Loop over the macroscopic load cases $\mathbf{E}=\mathbf{e}_i$

```python
for i in range(dim):                                                               # :90
    macro_gradient = np.zeros([dim]); macro_gradient[i] = 1                        # :92-93
    discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient,
                                                   macro_gradient_field_ijqxyz=macro_gradient_field)  # :96
    discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0,
                                  macro_gradient_field_ijqxyz=macro_gradient_field,
                                  rhs_inxyz=rhs_field)                             # :102
```

- `get_macro_gradient_field_mugrid` (`:96`) copies the constant vector $\mathbf{E}$ to every quadrature point. The field has
  layout `[1, d, q, x, y]`.
- `get_rhs_mugrid` (`:102`) computes $\mathbf{b} = -\mathsf{B}^\top\mathsf{W}\mathsf{A}\mathbf{E}$ and refreshes
  the ghost layers.

### 7. PCG solve

```python
Solvers.conjugate_gradients(comm=discretization.communicator, fc=discretization.field_collection,
                            hessp=K_fun, b=rhs_field, x=solution_field, prec=M_fun,
                            rtol=1e-6, maxiter=2000, callback=callback)            # :115
```

This calls muGrid's matrix-free PCG. It stops when $\|\mathbf{b}-\mathsf{K}\mathbf{x}\|\le 10^{-6}\|\mathbf{b}\|$.
The callback (`:106`) prints `fields['rr']` on rank 0. That value is the **squared** residual norm
$\mathbf{r}\cdot\mathbf{r}$. `solution_field` is not reset between load cases, so the solution of the
previous load case is the initial guess.

### 8. Post-processing

```python
if discretization.communicator.size == 1:   # :126  serial only: pcolormesh of the fluctuation θ̃
    ...
sum_sol = discretization.mpi_reduction.sum(solution_field.s, axis=tuple(range(-3, 0)))   # :141
homogenized_A_ij[:, i] = discretization.get_homogenized_stress_mugrid(
    material_data_field_ijklqxyz=material_data_field_C_0,
    displacement_field_inxyz=solution_field,
    macro_gradient_field_ijqxyz=macro_gradient_field)                                    # :145
```

- **Plot (`:126-138`).** Shows $\tilde\theta$ with `plt.show()`. It only runs when the script runs on a single rank.
- **Sanity check (`:141`).** `sum_sol` is the global sum of $\tilde\theta$. It should be close to 0, because the
  preconditioner removes the mean.
- **Homogenized flux (`:145`).** `get_homogenized_stress_mugrid` computes
  $\frac{1}{|Y|}\sum w_q\mathbf{A}_q(\mathbf{E}+\nabla\tilde\theta)_q$ and reduces it over all MPI ranks.
  This fills row $i$ of $\mathbf{A}^{\mathrm{eff}}$.
- **Analytical check (`:157-164`).** The script prints the elapsed time, then compares $A^{\mathrm{eff}}_{11}$
  with the closed-form value hard-coded at `:162`:
  $A^{\mathrm{eff}} = A_m\sqrt{(A_m+3A_i)/(3A_m+A_i)}$, with $A_m=100$ and $A_i=1$. This is the formula for a
  square array of square inclusions at volume fraction $1/4$.
- **Homogenized energy.** None of the conductivity examples compute it.
  `discretization.get_homogenized_energy_mugrid(...)` exists in the library and would give
  $\mathbf{E}\cdot\mathbf{A}^{\mathrm{eff}}\mathbf{E}$.

For reference, one run with the default settings gave
$A^{\mathrm{eff}}_{11}\approx 58.546$, against the analytical value $58.497$. The off-diagonal entries were
zero to 8 digits.

### 9. MPI aspects

- The `Discretization` uses `MPI.COMM_WORLD` by default. The FFT engine splits the pixel grid into
  subdomains, and every field (`.s` arrays) holds only the local subdomain plus ghost layers.
- The stencils need neighbour values. Hence the `communicate_ghosts` calls in `K_fun` and `M_fun` and after
  building $\mathbf{E}$ and the solution.
- Global quantities use collective reductions: `mpi_reduction.sum` and the dot products inside the solver.
  The FFTs in the preconditioner are also collective.
- Plots only run when `communicator.size == 1`. Most prints are guarded by `rank == 0`. The `sum_sol`
  and intermediate tensor prints (`:143`, `:151`) are not guarded, so every rank prints them.

### 10. How to run

```bash
python examples/homogenization/conductivity/example_2D_homogenization_conductivity.py
mpirun -n 4 python examples/homogenization/conductivity/example_2D_homogenization_conductivity.py
```

- Running from a different directory works, because the script adds the repository root to `sys.path` (`:3`).
- In serial, `plt.show()` blocks once per load case. To run headless, set `MPLBACKEND=Agg`.
- The MPI run was checked with 2 ranks and gave the same $\mathbf{A}^{\mathrm{eff}}$.

### Tunable parameters

| Parameter | Line | Meaning |
|---|---|---|
| `element_type` | `:18` | Element family: `'linear_triangles'`, `'bilinear_rectangle'`, `'biquadratic_rectangle'`, ... |
| `geometry_ID` | `:19` | Any name accepted by `microstructure_library.get_geometry` |
| `number_of_pixels` | `:22` | Grid resolution (discretization error vs. cost) |
| `mat_contrast`, `mat_contrast_2` | `:34-35` | Inclusion and matrix conductivity multipliers (phase contrast) |
| `conductivity_C_1` | `:36` | Base tensor; also used as the reference material of the preconditioner (`:70`) |
| `rtol`, `maxiter` | `:122-123` | PCG stopping criteria |

---

## Variants

### `example_2D_homogenization_conductivity_quadratic_basis.py`: biquadratic elements

This script is the same as the main example except for:

- `element_type = 'biquadratic_rectangle'` (`:18`). This is a 9-node Q9 element with 3×3 Gauss quadrature. Each
  pixel owns **4 nodes**, so the unknown field has shape `[1, 4, x, y]` and the Green preconditioner
  inverts a $4\times4$ block per wave vector. The $\mathbf{k}=0$ block uses a pseudo-inverse.
- The grid is `number_of_pixels = (64, 64)` (`:22`). This gives about the same number of DOFs as 128² linear
  elements.
- `M_fun` also refreshes the ghost layers of the output (`discretization.fft.communicate_ghosts(Px)`, `:82`).

One run gave $A^{\mathrm{eff}}_{11}\approx 58.508$, closer to the analytical value 58.497 than the
linear triangles on 128².

### `example_2D_homogenization_conductivity_block_CG.py`: single PCG vs. block PCG

The script solves the same problem twice and times both solves:

1. **One load case at a time (`:95-161`).** It uses muFFTTO's own `solvers.conjugate_gradients_mugrid`
   (`:120`) with `tol=1e-5, rtol=True`. Here the criterion is relative: $\|\mathbf{r}\|^2 \le 10^{-10}\|\mathbf{r}_0\|^2$.
   The callback signature is different: `callback(it, x, r, p, z, stop_crit_norm)` (`:112`). It receives
   the raw arrays and computes $\mathbf{r}\cdot\mathbf{r}$ with `communicator.sum`.
2. **All load cases at once (`:174-213`).** It allocates $d$ right-hand sides and solutions (`:174-178`) and
   builds every $\mathbf{b}_i$ (`:183-195`). Then `solvers.dr_pbcg_mugrid` (`:197`) solves
   $\mathsf{K}[\mathbf{x}_1\dots\mathbf{x}_d]=[\mathbf{b}_1\dots\mathbf{b}_d]$ with a
   preconditioned **block CG** (DR-PBCG, Meurant & Tichý), using `tol=1e-10, rtol=True`. All
   right-hand sides share one block Krylov space, which usually means fewer iterations than separate
   solves. The returned `norms['residual_frobenius']` is printed (`:205-207`). It holds the squared
   Frobenius norms $\|R_k\|_F^2$.

The material assignment (`:55-59`) has an extra `else` branch. For geometries other than
`'square_inclusion'`, it uses $\mathbf{A}=\phi(\mathbf{x})\,\mathbf{I}$, the phase field scaled by its value.

### `example_3D_homogenization_conductivity.py`: 3D, trilinear hexahedra

The 3D version of the main example:

- `element_type = 'trilinear_hexahedron'` (`:15`), with 8 Gauss points per voxel and 1 node per voxel.
- `domain_size = [1, 1, 1]`, `number_of_pixels = 3 * (128,)` (`:17-18`). This is about 2.1 M DOFs, so it is
  much heavier than the 2D runs.
- The material tensor is $3\times3$ with an extra broadcast axis (`:34-39`). `'square_inclusion'` gives a
  centred cube $[0.25,0.75)^3$ with volume fraction $1/8$.
- $\mathbf{A}^{\mathrm{eff}}$ is computed right after each solve (`:130`).
- The serial plot shows the mid-plane $z = L/2$ (`:139-141`).
- No analytical comparison is printed.
- `init_x_0` (`:86`) is allocated but never used.

### `example_3D_homogenization_conductivity_block_CG.py`: 3D, random microstructure, block CG

The 3D version of the block-CG comparison:

- The grid is `3 * (64,)` (`:19`) and `geometry_ID = 'random_distribution'` (`:21`), with uniform random values
  $\phi\in[0,1)$ per voxel. The `else` branch (`:57`) sets $\mathbf{A} = 100\,\phi(\mathbf{x})\,\mathbf{I}$.
- Both solvers use an **absolute** tolerance `tol=1e-5`. The `rtol=True` arguments are commented out
  (`:139`, `:208`).
- Before each single solve the solution is reset to zero (`:129`). The callback records the residual
  history of each load case in `norms_single` (`:98-102`, `:124`).
- At the end, a semilog plot (`:233-238`) compares the block-CG history with the three single-PCG histories.
  All of them are squared norms.
- The analytical 2D square-inclusion value is still printed at the end (`:250-252`). It does not apply to
  this microstructure.

---

## Known issues

- **`example_3D_homogenization_conductivity_block_CG.py`, `:233-238`.** The final convergence plot is outside
  the `communicator.size == 1` guard, and `matplotlib.pyplot` is only imported inside that guard. Under
  `mpirun -n >1` the script will fail with `NameError: plt` at that point. Also, `norms` is only converted
  on rank 0 (`:210-211`).
- **Same file, `:156-158` and `:224-226`.** The 2D slice plots take the coordinates at $z$ = mid-plane
  (`coords[..., nz//2]`), but the solution at $y$ = mid-plane (`s[0, 0, ..., ny//2, :]` and
  `s[0, 0, :, ny//2, :]`). The shapes match only because the grid is cubic, so the plots are mislabelled.
- **Same file, `:250-252`.** The analytical $A^{\mathrm{eff}}_{11}$ comparison refers to the 2D square
  inclusion, not the random 3D microstructure used here.
- **Residual printouts (all files).** "norm of residual" is actually the **squared** residual norm
  (`fields['rr']` or $\mathbf{r}\cdot\mathbf{r}$). The block-CG history is likewise $\|R\|_F^2$.
- **3D examples.** Some `print("Elapsed time: ...")` lines are not rank-guarded (`example_3D_homogenization_conductivity.py:157-158`, `example_3D_homogenization_conductivity_block_CG.py:174-175`), so every rank prints them.
