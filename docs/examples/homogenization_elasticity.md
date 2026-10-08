# Homogenization of elasticity (examples walkthrough)

This page explains the scripts in

- `examples/homogenization/small_strain_elasticity/`: linear, small-strain elasticity, effective stiffness
  $\mathbb{C}^{\mathrm{eff}}$;
- `examples/homogenization/finite_strain_elasticity/`: nonlinear Neo-Hookean, finite strain, Newton–CG.

For background, see the theory notes:
[cell problem](../theory.md#1-periodic-homogenization-the-cell-problem),
[small-strain elasticity](../theory.md#2-physical-problems-conductivity-and-small-strain-elasticity),
[finite strain](../theory.md#3-finite-strain-elasticity),
[FE on a pixel grid](../theory.md#4-finite-element-discretization-on-a-pixel-grid),
[linear system & preconditioner](../theory.md#5-assembling-and-solving-the-linear-system),
[effective properties](../theory.md#6-effective-homogenized-properties),
[array conventions](../theory.md#9-notation-and-array-conventions).
The conductivity page ([homogenization_conductivity.md](homogenization_conductivity.md)) explains the
common machinery (Green preconditioner, ghosts, MPI) in more detail. This page focuses on what changes for
vector-valued unknowns.

## Which PDE is solved

### Small strain (linear elasticity)

For a prescribed macroscopic strain $\mathbf{E}$, the displacement is $\mathbf{u}=\mathbf{E}\mathbf{x}+\tilde{\mathbf{u}}$,
where the fluctuation $\tilde{\mathbf{u}}$ is $Y$-periodic. The strain is $\boldsymbol\varepsilon = \mathbf{E} + \nabla^s\tilde{\mathbf{u}}$,
with $\nabla^s$ the symmetrized gradient.

**Strong form.** Find a periodic $\tilde{\mathbf{u}}$ such that

$$
-\nabla\cdot\big(\mathbb{C}(\mathbf{x}) : (\mathbf{E}+\nabla^s\tilde{\mathbf{u}})\big) = \mathbf{0}\quad\text{in } Y .
$$

**Weak form.** For all periodic test functions $\mathbf{v}$:

$$
\int_Y \nabla^s\mathbf{v} : \mathbb{C} : (\mathbf{E}+\nabla^s\tilde{\mathbf{u}})\,\mathrm{d}\mathbf{x} = 0 .
$$

**Discrete system.**

$$
\mathsf{K}\tilde{\mathbf{u}} = \mathsf{B}^\top\mathsf{W}\mathsf{C}\mathsf{B}\,\tilde{\mathbf{u}} = -\mathsf{B}^\top\mathsf{W}\mathsf{C}\,\mathbf{E} .
$$

The system is solved with PCG and an FFT-based Green preconditioner built from a homogeneous reference stiffness $\mathbb{C}_{\mathrm{ref}}$.

**Effective stiffness.** Solve for the unit loads $\mathbf{E}=\mathbf{e}_k\otimes\mathbf{e}_l$, for all $d^2$ pairs $(k,l)$. Then

$$
\mathbb{C}^{\mathrm{eff}}_{ijkl} = \frac{1}{|Y|}\int_Y \big[\mathbb{C}:(\mathbf{e}_k\otimes\mathbf{e}_l+\nabla^s\tilde{\mathbf{u}}_{kl})\big]_{ij}\,\mathrm{d}\mathbf{x} .
$$

The scripts print it in Voigt notation with `material_models.compute_Voigt_notation_4order`. This is a pure
index map with no factors of 2.

### Finite strain (hyperelasticity)

The unknown is the deformation gradient $\mathbf{F} = \bar{\mathbf{F}} + \nabla\tilde{\mathbf{u}}$, where $\bar{\mathbf{F}}$ is the macroscopic part and $\tilde{\mathbf{u}}$ is periodic. Equilibrium is written in terms of the first Piola–Kirchhoff stress $\mathbf{P}(\mathbf{F})=\partial W/\partial\mathbf{F}$:

$$
-\nabla\cdot\mathbf{P}(\mathbf{F}) = \mathbf{0},\qquad
\int_Y \nabla\mathbf{v} : \mathbf{P}(\bar{\mathbf{F}}+\nabla\tilde{\mathbf{u}})\,\mathrm{d}\mathbf{X} = 0\quad\forall\,\mathbf{v}\ \text{periodic}.
$$

Each Newton step solves the linearized system

$$
\mathsf{B}^\top\mathsf{W}\,\mathbb{A}(\mathbf{F}^{(n)})\,\mathsf{B}\;\delta\tilde{\mathbf{u}} = -\mathsf{B}^\top\mathsf{W}\,\mathbf{P}(\mathbf{F}^{(n)}),
\qquad \mathbb{A}=\partial\mathbf{P}/\partial\mathbf{F},
$$

with PCG, and then updates $\mathbf{F}^{(n+1)} = \mathbf{F}^{(n)} + \mathsf{B}\,\delta\tilde{\mathbf{u}}$.

The material is compressible Neo-Hookean (Simo–Pister form):

$$
W = \tfrac{\lambda}{2}(\ln J)^2 + \tfrac{\mu}{2}(\mathbf{F}:\mathbf{F}-d) - \mu\ln J ,
$$

$$
\mathbf{P} = \lambda\ln J\,\mathbf{F}^{-\top} + \mu(\mathbf{F}-\mathbf{F}^{-\top}) .
$$

## Overview of the examples

| File | Dim | Physics | Element | Grid (default) | Material input | Solver | Preconditioner | Key outputs |
|---|---|---|---|---|---|---|---|---|
| `small_strain_elasticity/example_2D_homogenization_elasticity.py` | 2D | linear elastic, small strain | `bilinear_rectangle` (2×2 Gauss) | 32×32 | `LinearElastic` → tangent field $\mathbb{C}_q$ | `muGrid.Solvers.conjugate_gradients`, `tol=1e-6` | Green, $\mathbb{C}_{\mathrm{ref}}(K_0,G_0)$ | $3\times3$ Voigt $\mathbb{C}^{\mathrm{eff}}$, plots of $\tilde u_x,\tilde u_y$ on the deformed grid |
| `small_strain_elasticity/example_2D_homogenization_elasticity_no_data_field.py` | 2D | linear elastic, small strain | `linear_triangles` | 32×32 | explicit stress function $\boldsymbol\sigma(\boldsymbol\varepsilon)$ from $\lambda_q,\mu_q$ (no $\mathbb{C}$ field) | same | Green, $\mathbb{C}(\lambda_0,\mu_0)$ | $3\times3$ Voigt $\mathbb{C}^{\mathrm{eff}}$, plots of $\tilde u_0,\tilde u_1$ |
| `small_strain_elasticity/example_3D_homogenization_elasticity.py` | 3D | linear elastic, small strain | `trilinear_hexahedron` (2×2×2 Gauss) | 17³, cell $4\times3\times5$ | constant $\mathbb{C}$ field scaled by masks | same | Green, $\mathbb{C}(K_0,G_0)$ | $6\times6$ Voigt $\mathbb{C}^{\mathrm{eff}}$, mid-plane plots of $\tilde u_{0,1,2}$ |
| `finite_strain_elasticity/example_2D_homogenization_finite_strain_elasticity_NeoHookean.py` | 2D | Neo-Hookean, finite strain | `bilinear_rectangle` | 64×64 (`-n`) | `NeoHookean` (stress + tangent + energy) | Newton + muFFTTO `solvers.conjugate_gradients_mugrid`, absolute `tol=1e-5` | Green, small-strain $\mathbb{C}(\lambda_m,\mu_m)$ | Newton/CG iteration log, `.npy` fields per Newton iteration, `info_log_final.npz` |
| `finite_strain_elasticity/example_3D_homogenization_finite_strain_elasticity_NeoHookean.py` | 3D | — | — | — | — | — | — | **The file is empty (0 bytes).** |

---

## Walkthrough: `example_2D_homogenization_elasticity.py`

### 1. Unit cell and discretization

```python
problem_type = 'elasticity'                                     # :16
element_type = 'bilinear_rectangle'                             # :18
formulation = 'small_strain'                                    # :19 (informational; see K_fun)
domain_size = [1, 1]; number_of_pixels = (32, 32)               # :21-22
my_cell = domain.PeriodicUnitCell(domain_size=domain_size, problem_type=problem_type)   # :24
discretization = domain.Discretization(cell=my_cell, nb_of_pixels_global=number_of_pixels, ...)  # :27
print(f'{MPI.COMM_WORLD.rank:6} ... {discretization.fft.subdomain_locations}')         # :32
```

- With `problem_type='elasticity'`, the unknown is the displacement fluctuation $\tilde{\mathbf{u}}$, with
  $d$ components per node (layout `[d, n, x, y]`).
- `bilinear_rectangle` is the Q4 element with 2×2 Gauss points and one node per pixel.
- Line `:32` prints how each MPI rank's subdomain sits in the global grid: global grid, local grid, offset.

### 2. Microstructure and Lamé fields at quadrature points

```python
phase_field.s[0, 0] = microstructure_library.get_geometry(..., microstructure_name='square_inclusion', ...)  # :40
matrix_mask = phase_field.s[0, 0] > 0; inc_mask = phase_field.s[0, 0] == 0               # :43-44
K_0, G_0 = material_models.get_bulk_and_shear_modulus(E=1, poisson=0.2)                 # :51
lam, mu = material_models.get_lame_parameters_from_bulk_and_shear(K_0, G_0, dim=2)      # :52
lam_11qxyz = discretization.get_quad_field_scalar(name='lam_first_lame')                 # :56
lam_11qxyz.s[..., matrix_mask] = mat_contrast_2 * lam                                    # :60
...
```

- The matrix (phase value 1) gets $100\lambda$ and $100\mu$. The square inclusion $[0.25,0.75)^2$ gets $\lambda$ and $\mu$.
- $\lambda$ and $\mu$ are stored as scalar quadrature-point fields `[1, 1, q, x, y]`.
- With `dim=2`, the Lamé conversion uses the 2D bulk-modulus relation. For $E=1$, $\nu=0.2$ this gives $\lambda=K-G\approx0.139$, $\mu\approx0.417$.

### 3. Material model → tangent field $\mathbb{C}_q$

```python
material = material_models.LinearElastic(discretization=discretization,
                                         lam_1qxyz=lam_11qxyz, mu_1qxyz=mu_11qxyz, ...)   # :66
material_data_field_C_0 = discretization.get_material_data_size_field_mugrid(name='elastic_tensor')  # :70
strain_ijqxyz = discretization.get_strain_sized_field(name='macro_gradient_field')           # :73
material.get_algorithmic_tangent(strain_ijqxyz, material_data_field_C_0)                   # :75
```

`get_algorithmic_tangent` fills the field `[i, j, k, l, q, x, y]` with
$\mathbb{C}_{ijkl}=\lambda\delta_{ij}\delta_{kl}+\mu(\delta_{ik}\delta_{jl}+\delta_{il}\delta_{jk})$ at every quadrature point.
The strain argument is only used to infer the dimension, because the model is linear.

### 4. System operator, preconditioner and right-hand side

```python
def K_fun(x, Ax):
    discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0,
                                              input_field_inxyz=x, output_field_inxyz=Ax,
                                              formulation='small_strain')               # :82
    discretization.fft.communicate_ghosts(Ax)
elastic_C_ref = material_models.get_elastic_material_tensor(dim=2, K=K_0, mu=G_0, kind='linear')  # :90
preconditioner = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=elastic_C_ref)  # :94
```

- **System operator.** `formulation='small_strain'` makes the operator use the symmetrized gradient
  $\nabla^s$, so it computes $\mathsf{K}\mathbf{x}=\mathsf{B}_s^\top\mathsf{W}\mathsf{C}\mathsf{B}_s\mathbf{x}$.
- **Preconditioner.** For elasticity, the Green preconditioner inverts a $d\times d$ block per wave
  vector. The reference material is the unscaled (soft-phase) isotropic stiffness.
- **Right-hand side.** `get_rhs_mugrid` (`:132`) computes $-\mathsf{B}^\top\mathsf{W}\mathsf{C}:\mathbf{E}$.
  $\mathbb{C}$ has minor symmetries, so a non-symmetric unit load $\mathbf{e}_0\otimes\mathbf{e}_1$ acts like
  its symmetric part.

### 5. Loop over the $d^2$ unit strains and PCG solve

```python
for i in range(dim):
    for j in range(dim):                                                                # :123-124
        macro_gradient_ij = np.zeros([dim, dim]); macro_gradient_ij[i, j] = 1          # :126-127
        discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient_ij,
                                                       macro_gradient_field_ijqxyz=macro_gradient_field)  # :129
        discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0,
                                      macro_gradient_field_ijqxyz=macro_gradient_field,
                                      rhs_inxyz=rhs_field)                             # :132
        Solvers.conjugate_gradients(comm=discretization.communicator, fc=discretization.field_collection,
                                    hessp=K_fun, b=rhs_field, x=solution_field, prec=M_fun,
                                    tol=1e-6, maxiter=2000, callback=callback)         # :136
```

- **Stopping criterion.** The `tol=` argument of muGrid's CG is deprecated. Passing it selects a purely
  **absolute** criterion $\|\mathbf{r}\|<10^{-6}$, and muGrid emits a `DeprecationWarning`. Use `rtol=` for a
  relative one.
- **Initial guess.** The solution of the previous load case is the initial guess, because `solution_field`
  is not reset.

### 6. Post-processing

```python
if discretization.communicator.size == 1:
    x_plot_ixyz = visualization_utils.get_deformed_grid_coords_two_dim(discretization,
                    macro_gradient_ij=macro_gradient_ij, displacement_fluctuation=solution_field)  # :151
    visualization_utils.plot_field_on_grid(coordinates_for_plot=x_plot_ixyz,
                                           field_to_plot=solution_field.s[0, 0], ...)               # :155
homogenized_C_ijkl[:, :, i, j] = discretization.get_homogenized_stress_mugrid(
    material_data_field_ijklqxyz=material_data_field_C_0, displacement_field_inxyz=solution_field,
    macro_gradient_field_ijqxyz=macro_gradient_field, formulation='small_strain')                   # :167
print(... material_models.compute_Voigt_notation_4order(homogenized_C_ijkl) ...)                    # :176
```

- **Plots (`:151-164`).** In serial, $\tilde u_x$ and $\tilde u_y$ are drawn on the deformed grid
  $\mathbf{x}+\mathbf{E}\mathbf{x}+\tilde{\mathbf{u}}$. The plotting code is wrapped in a bare
  `try/except`.
- **Effective stiffness (`:167`).** `get_homogenized_stress_mugrid` computes the volume-averaged stress
  $\langle\mathbb{C}:(\mathbf{E}+\nabla^s\tilde{\mathbf{u}})\rangle$ and reduces it over all MPI ranks. For
  $\mathbf{E}=\mathbf{e}_i\otimes\mathbf{e}_j$ this is the slice $\mathbb{C}^{\mathrm{eff}}_{ij\,\cdot\,\cdot}$.
- **Output (`:176`).** Rank 0 prints the $3\times3$ Voigt matrix.
- **Energy.** The homogenized energy is not computed. `get_homogenized_energy_mugrid` is available for that.

For reference, one serial run with the default settings printed

```
[[53.16409517 6.10414508 -0.00000000]
 [6.10414511 53.16409517 -0.00000000]
 [0.00000000 -0.00000003 11.17067961]]
```

### 7. MPI and running

```bash
python examples/homogenization/small_strain_elasticity/example_2D_homogenization_elasticity.py
mpirun -n 4 python examples/homogenization/small_strain_elasticity/example_2D_homogenization_elasticity.py
```

The same MPI rules apply as for conductivity:

- the grid is split by the FFT engine;
- ghost layers are refreshed in `K_fun` and `M_fun`;
- reductions are collective;
- plots run only on a single rank.

To run headless, set `MPLBACKEND=Agg`. Otherwise `plt.show()` blocks after each load case.

### Tunable parameters

| Parameter | Line | Meaning |
|---|---|---|
| `element_type` | `:18` | `'bilinear_rectangle'`, `'biquadratic_rectangle'`, `'linear_triangles'` (the alternatives are listed in a comment) |
| `number_of_pixels` | `:22` | Resolution |
| `geometry_ID` | `:37` | Microstructure name for `get_geometry` |
| `mat_contrast`, `mat_contrast_2` | `:48-49` | Inclusion and matrix stiffness multipliers |
| `E`, `poisson` | `:51` | Base Young's modulus and Poisson ratio (also the preconditioner reference) |
| `tol`, `maxiter` | `:143-144` | CG stopping (absolute, see above) |

---

## Variants

### `example_2D_homogenization_elasticity_no_data_field.py`: explicit stress, no $\mathbb{C}$ field

This script never stores the 4th-order tensor field $\mathbb{C}_q$ (`[d,d,d,d,q,x,y]`), which saves memory.
Instead, the constitutive law is a function:

```python
lam_0, mu_0 = material_models.get_lame_parameters(E=1, poisson=0.2)                     # :36
def constitutive_model(strain, stress):                                                 # :68
    material_models.linear_isotropic_elasticity_stress_from_strain_lame(
        strain_ijqxyz=strain, lam_1qxyz=lam_1qxyz, mu_1qxyz=mu_1qxyz, output_stress_ijqxyz=stress)
def K_fun(x, Ax):
    discretization.apply_system_matrix_mugrid_explicit_stress(constitutive=constitutive_model, ...,
                                                              formulation='small_strain')   # :76
```

| Step | Function | Lines |
|---|---|---|
| $\mathsf{K}\mathbf{x}=\mathsf{B}_s^\top\mathsf{W}\,\boldsymbol\sigma(\mathsf{B}_s\mathbf{x})$ | `apply_system_matrix_mugrid_explicit_stress` | `:76` |
| $\mathbf{b}=-\mathsf{B}^\top\mathsf{W}\,\boldsymbol\sigma(\mathbf{E})$ | `get_rhs_explicit_stress_mugrid` | `:126` |
| $\langle\boldsymbol\sigma(\mathbf{E}+\nabla^s\tilde{\mathbf{u}})\rangle$ | `get_homogenized_stress_mugrid_explicit_stress` | `:170` |

Other differences from the main example:

- `element_type = 'linear_triangles'` (`:18`).
- $\lambda$ and $\mu$ are set to $(\lambda_0,\mu_0)$ everywhere and multiplied by 100 in the matrix (`:61-65`).
- The preconditioner uses `get_elastic_tensor_from_lame(lam_0, mu_0)` (`:38`, `:84`).
- The plots are plain `pcolormesh` with a fixed colour range $[-0.2,0.2]$ (`:141-166`).

This variant does **not** give the same $\mathbb{C}^{\mathrm{eff}}$ as the main example. See Known issues.

### `example_3D_homogenization_elasticity.py`: 3D, trilinear hexahedra

The 3D version of the main example:

- **Grid and cell.** `element_type = 'trilinear_hexahedron'` (`:17`). The cell is **non-cubic**:
  `domain_size = [4, 3, 5]` (`:20`), discretized with `(17, 17, 17)` voxels (`:21`), so the voxels are
  anisotropic.
- **Microstructure.** `geometry_ID = 'circle_inclusion'` (`:22`). In 3D this is a centred spherical void
  region (value 0), which gets the inclusion material.
- **Material.** The material field is set directly from a constant isotropic tensor and then scaled by the
  masks (`:43-64`). No `LinearElastic` object is used. Its Voigt form is printed twice (`:47`, `:51`).
- **Load cases.** The fixed `macro_gradient` (`:36`) and the right-hand side built at `:66-76` are later
  overwritten inside the $3\times3$ load loop (`:110-138`). Only the loop matters.
- **Output.** The result is a $6\times6$ Voigt matrix, with index order 11, 22, 33, 23, 13, 12. In serial,
  each load case also shows mid-plane ($z$) plots of the three displacement components (`:140-171`).

The default settings run in a few seconds in serial.

### `example_2D_homogenization_finite_strain_elasticity_NeoHookean.py`: Newton–CG at finite strain

**Command line (`:22-39`).**

| Option | Default | Meaning |
|---|---|---|
| `-n` | 64 | pixels per direction |
| `-inc` | 50 | number of load increments |
| `--save_per_it` | saving on | **Disables** saving, because it is declared with `action='store_false'` |

Run it with:

```bash
python examples/homogenization/finite_strain_elasticity/example_2D_homogenization_finite_strain_elasticity_NeoHookean.py -n 64 -inc 50
mpirun -n 4 python examples/homogenization/finite_strain_elasticity/example_2D_homogenization_finite_strain_elasticity_NeoHookean.py -n 64 -inc 50
```

The script needs `NuMPI` for `save_npy`. It uses `sys.path.append` (`:11`), so an installed `muFFTTO`
package takes precedence over the repository copy.

**Setup (`:43-165`).**

- **Discretization and microstructure.** `bilinear_rectangle` elements and `formulation='finite_strain'`
  (`:49-50`), which means the full gradient is used, not $\nabla^s$. The microstructure is the square
  inclusion: the matrix is soft ($E=1$) and the inclusion is stiff ($E=10$), both with $\nu=0.3$ (`:92-101`).
- **Lamé parameters.** Computed with the 3D/plane-strain formulas and stored as quadrature fields (`:132-138`).
- **Material model.** `material_models.NeoHookean` (`:147`).
- **Preconditioner reference.** `ref_mat` (`:109`) is the *small-strain* isotropic stiffness of the matrix.
  This is the linearization of the Neo-Hookean model at $\mathbf{F}=\mathbf{I}$, so the Green preconditioner
  (`:177-187`) is built once and reused for all Newton steps.
- **Fields.** The script uses separate fields for:
  - $\tilde{\mathbf{u}}$ and $\delta\tilde{\mathbf{u}}$;
  - $\nabla\delta\tilde{\mathbf{u}}$;
  - $\mathbf{F}$ (`total_strain_field`, initialized to $\mathbf{I}$ at `:170-172`);
  - $\mathbf{P}$, $\mathbb{A}$ and $W$.

**Loading (`:192-199`).** The macroscopic increment is
$\Delta\bar{\mathbf{F}}=\frac{1}{n_{\mathrm{inc}}}\begin{pmatrix}0&0.8\\0&0.3\end{pmatrix}$.
After all increments, $\bar{\mathbf{F}}-\mathbf{I}$ has simple shear $0.8$ in component $F_{01}$ and a stretch of $0.3$ in $F_{11}$.

**Increment loop (`:219-381`).** Each increment:

1. **Predictor (`:225-238`).** Adds $\Delta\bar{\mathbf{F}}$ to $\mathbf{F}$ at every quadrature point.
   Evaluates $\mathbf{P}$ and $\mathbb{A}$. Assembles the residual
   $\mathbf{b}=-\mathsf{B}^\top\mathsf{W}\mathbf{P}$ with `apply_gradient_transposed_operator_mugrid(..., apply_weights=True)`.
2. **Newton loop (`:257-369`).** Each step:
   - solves $\mathsf{B}^\top\mathsf{W}\mathbb{A}\mathsf{B}\,\delta\tilde{\mathbf{u}}=\mathbf{b}$ with
     `solvers.conjugate_gradients_mugrid` (`:278`), with absolute `tol=1e-5` and `maxiter=1000`. `K_fun`
     uses the current tangent field;
   - updates $\mathbf{F}\leftarrow\mathbf{F}+\mathsf{B}\delta\tilde{\mathbf{u}}$ and
     $\tilde{\mathbf{u}}\leftarrow\tilde{\mathbf{u}}+\delta\tilde{\mathbf{u}}$ (`:301-312`);
   - re-evaluates $\mathbf{P}$, $\mathbb{A}$ and $W$, and recomputes the residual (`:315-330`).

   Newton stops when $\|\mathsf{B}\delta\tilde{\mathbf{u}}\|/\|\mathbf{F}\|<10^{-8}$, or after 100 iterations (`:366-369`).
3. **Plot (`:370-381`).** In serial only, plots one tangent component $\mathbb{A}_{0000}$ at the first
   quadrature point on the deformed grid.

**Outputs.**

| Output | Location |
|---|---|
| Per Newton iteration (when saving is on): the pixel-averaged $\tilde{\mathbf{u}}$, $W$ and $\mathbf{P}$ | `.npy` files in `finite_strain_elasticity/exp_data/<script>/Nx=..Ny=.._Green/` (`:343-363`) |
| Total CG and Newton iteration counts and timing | printed (`:393-400`) |
| Run summary: parameters, the history of $\|\nabla\delta\tilde{\mathbf{u}}\|$ and the counters | `info_log_final.npz` (`:401`) |

No homogenized stress or effective tangent is computed. The library function
`muFFTTO.solvers_nonlinear.solve_finite_strain_newton_cg` implements the same Newton–CG scheme as a
reusable routine, but this example does not call it.

**Tunable parameters.**

| Parameter | Lines |
|---|---|
| Grid size and increments | `-n`, `-inc` |
| Material constants | `:92-101` |
| Load direction and magnitude | `:193-194` |
| CG tolerance | `:285` |
| Newton tolerance and cap | `:366-368` |

### `example_3D_homogenization_finite_strain_elasticity_NeoHookean.py`

The file is empty (0 bytes). There is no 3D finite-strain example yet. The directories
`exp_data/` and `figures/` with the same name exist next to it.

---

## Known issues

- **`example_2D_homogenization_elasticity_no_data_field.py` disagrees with the tensor-field version.**
  Two separate effects:
  1. `get_lame_parameters` (`:36`) gives $\lambda\approx0.278$, the 3D/plane-strain value. The main example
     uses the 2D bulk relation with $\lambda\approx0.139$. These are different materials.
  2. `linear_isotropic_elasticity_stress_from_strain_lame` does **not** symmetrize its input. The
     symmetrization line is commented out in `material_models.py`. For the non-symmetric unit loads
     $\mathbf{E}=\mathbf{e}_0\otimes\mathbf{e}_1$ (`:119-120`), this gives $\boldsymbol\sigma=2\mu\mathbf{E}$
     instead of $\mathbb{C}:\mathbf{E}$.

  In a check run with the Lamé parameters matched to the main example (both `linear_triangles`, 32²):
  - the shear Voigt entry came out $\approx 42.8$ instead of $\approx 11.43$, and the Voigt matrix was
    slightly non-symmetric;
  - symmetrizing the strain inside `constitutive_model` reproduced the tensor-field result exactly.
- **Deprecated `tol=` in muGrid CG (all small-strain examples).** muGrid's `Solvers.conjugate_gradients` is
  called with the deprecated `tol=` keyword, which means an absolute criterion and a `DeprecationWarning`.
  The conductivity examples use `rtol=` instead.
- **Finite-strain example: Newton stopping criterion.** CG uses an absolute tolerance of $10^{-5}$. Once
  $\|\mathbf{b}\|<10^{-5}$, CG returns a zero increment and the Newton test
  $\|\nabla\delta\tilde{\mathbf{u}}\|/\|\mathbf{F}\|<10^{-8}$ passes trivially. In a small test run the last
  Newton step reported `CG its = 0`. The effective equilibrium tolerance is therefore the CG `tol`.
- **Finite-strain example: output and plotting quirks.**
  - Saving is on by default, and `--save_per_it` turns it **off**.
  - The plot at `:378-381` shows a tangent component but is titled $\tilde u_x$.
  - In serial it is shown (blocking) after **every** increment, which means 50 windows by default.
  - `figures/...` is created (`:86`) but nothing is written to it.
  - Only rank 0 creates the output directories, and there is no barrier before the collective `save_npy`.
- **Finite-strain 3D file.** `example_3D_homogenization_finite_strain_elasticity_NeoHookean.py` is empty.
