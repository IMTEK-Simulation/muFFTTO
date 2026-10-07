# Topology optimization examples

This page explains the scripts in `examples/topology_optimization/`. Each script designs a
periodic unit cell whose **homogenized** response (effective stiffness or conductivity)
matches a prescribed target. It uses a phase-field formulation, FFT-preconditioned
finite-element cell problems, adjoint sensitivities and a bound-constrained L-BFGS
optimizer from NuMPI.

The background theory is in [theory.md](../theory.md), in particular
[periodic homogenization](../theory.md#1-periodic-homogenization-the-cell-problem),
[the physical problems](../theory.md#2-physical-problems-conductivity-and-small-strain-elasticity),
[FE discretization](../theory.md#4-finite-element-discretization-on-a-pixel-grid),
[solving the linear system](../theory.md#5-assembling-and-solving-the-linear-system),
[effective properties](../theory.md#6-effective-homogenized-properties),
[topology optimization](../theory.md#7-topology-optimization) and
[notation](../theory.md#9-notation-and-array-conventions).

| File | Dim | Physics | Element (`element_type`) | Grid | Optimizer | Target |
|---|---|---|---|---|---|---|
| `example_2D_elasticity_TO.py` | 2D | small-strain elasticity | `linear_triangles` | 64x64, cell 1x1 | `l_bfgs_bounded`, box [0, 1] | isotropic, $\nu_t = 0$, $G_t = \tfrac{3}{20}E_0$ |
| `example_2D_elasticity_TO_tilled_grid.py` | 2D | small-strain elasticity | `linear_triangles_tilled` (sheared, equilateral triangles) | 32x32, cell $1\times\sqrt3/2$ | `l_bfgs` (**unbounded**) | isotropic, $\nu_t = -0.5$ (auxetic), $G_t = \tfrac{3}{20}E_0$ |
| `example_3D_elasticity_TO.py` | 3D | small-strain elasticity | `trilinear_hexahedron_1Q` | 15x15x15, cell 1x1x1 | `l_bfgs_bounded`, box [0, 1] | isotropic, $\nu_t = -0.3$, $G_t = \tfrac{3}{20}E_0$ |
| `example_2D_conductivity_TO.py` | 2D | conductivity | `linear_triangles` | 64x64, cell 1x1 | `l_bfgs_bounded`, box [0, 1] | anisotropic $\mathrm{diag}(0.5,\,0.1)$ |
| `example_2D_conductivity_TO_discrete.py` | 2D | conductivity | inherited from the conductivity example | inherited | custom discrete descent (serial only) | same as the conductivity example |

---

## 1. The optimization problem as implemented

### Design variable

The design variable is a **nodal phase field** $\rho$ (`phase_field_1nxyz`, shape
`[1, n, x, y(, z)]`). All elements used here have one node per pixel ($n = 1$), so
$\rho$ has one value per pixel and the optimizer sees it as a flat vector of length
$N_x N_y (N_z)$ (the MPI-local part, see [MPI handling](#mpi-handling-of-the-design-vector)).
$\rho = 1$ is solid and $\rho = 0$ is void. The operator $N$ interpolates $\rho$ to the
quadrature points (`apply_N_operator_mugrid`).

### Material interpolation (SIMP)

At every quadrature point $q$:

$$
\mathbb C(\rho_q) = (\mathbb C_1 - \mathbb C_0)\,\rho_q^{\,p} + \mathbb C_0,
\qquad
\mathbb C_0 = 10^{-s}\,\mathbb C_1,
\qquad
\frac{\partial\mathbb C}{\partial\rho} = p\,\rho^{p-1}(\mathbb C_1 - \mathbb C_0),
$$

where the scripts use $p = 2$ (`p`) and $s$ = `soft_phase_exponent` (5 in 2D elasticity,
3 in 3D, 4 in conductivity). For elasticity $\mathbb C_1$ is isotropic with $K_0 = 1$ and
$G_0 = 0.5$ (`material_models.get_elastic_material_tensor`, which uses the 3D/plane-strain
Lamé relation). For conductivity $\mathbb C_1 = \mathbf I$, a $2\times2$ tensor.

### State problems (one per load case)

For each macroscopic strain (or temperature gradient) $\mathbf E_l$, $l = 1..L$, the
periodic fluctuation $u_l$ solves the discrete cell problem

$$
g_l(u_l,\rho) = D^{T} W\, \mathbb C(\rho) : (\mathbf E_l + D u_l) = 0
\quad\Longleftrightarrow\quad
K(\rho)\,u_l = -D^{T} W\, \mathbb C(\rho) : \mathbf E_l ,
$$

where $D$ is the (symmetrized) gradient from nodes to quadrature points and $W$ holds the
quadrature weights. The equation is solved with preconditioned CG
(`solvers.conjugate_gradients_mugrid`). The homogenized stress (or flux) is

$$
\boldsymbol\Sigma_h^{l}(\rho) = \frac{1}{|\Omega|}\int_\Omega \mathbb C(\rho) : (\mathbf E_l + \nabla^s u_l)\,dx
$$

(`get_homogenized_stress_mugrid`).

### Target

A target effective tensor $\mathbb C_t$ is chosen, and the target response for each load
case is $\boldsymbol\Sigma_t^{l} = \mathbb C_t : \mathbf E_l$ (`target_stresses`,
`target_fluxes`).

* **Elasticity.** $\mathbb C_t$ is isotropic with $G_t = \tfrac{3}{20}E_0$, where
  $E_0 = 9K_0G_0/(3K_0+G_0)$. Also $E_t = 2G_t(1+\nu_t)$ and
  $K_t = E_t / (3(1-2\nu_t))$ (`get_bulk_and_shear_modulus`, 3D definition). Only
  $\nu_t$ (`poison_target` / `poisson_target`) differs between the elasticity scripts.
* **Conductivity.** $\mathbb C_t = \mathrm{diag}(0.5,\;0.1)$, which is strongly anisotropic.

### Objective

$$
F(\rho) = \sum_{l=1}^{L} w_l\,
\underbrace{\frac{\|\boldsymbol\Sigma_t^{l} - \boldsymbol\Sigma_h^{l}(\rho)\|^2}{\|\boldsymbol\Sigma_t^{l}\|^2}}_{f_\sigma^l}
\;+\;
\underbrace{\eta \int_\Omega |\nabla\rho|^2\,dx \;+\; \frac{c_{dw}}{\eta}\int_\Omega \rho^2(1-\rho)^2\,dx}_{f_\rho}
\;\Big[+\;\sum_l \lambda_l^{T} g_l\Big]
$$

* $f_\sigma^l$ is the normalized squared mismatch between the homogenized and target
  stress or flux (`compute_stress_equivalence_potential`,
  `compute_flux_equivalence_potential`).
* $f_\rho$ is the Modica–Mortola phase-field regularization
  (`objective_function_phase_field`, or `objective_function_phase_field_3D` in 3D). The
  gradient term is integrated with the FE gradient and quadrature
  ($\rho^T D^T W D\rho$). The double-well term uses nodal quadrature with equal weights
  $|\Omega|/N$. The scripts use $c_{dw} = 1$ (`double_well_depth_test`) and
  $\eta$ = one pixel size (`eta = max(pixel_size)`).
* Weights: elasticity uses $w_l = \texttt{weight}/L = 5/3$. Conductivity uses
  $w_l = \texttt{weights}[l] = 5$, not divided by $L$.
* The bracketed term is the adjoint "energy" $\lambda_l^T g_l$ (the value of the
  Lagrangian residual term). It vanishes at exact equilibrium. The **elasticity** scripts
  add it to the returned objective. The conductivity script has this line commented out.
  See [Known issues](#known-issues).
* No volume-fraction constraint is active: a `LinearConstraint` is built in the 2D
  elasticity script but never passed to the optimizer. Bounds are $0 \le \rho \le 1$,
  except in the tilled-grid example.

### Adjoint problem and sensitivity

Because $K(\rho)$ is symmetric, each load case needs a single adjoint solve with the same
operator and preconditioner (`sensitivity_stress_and_adjoint_FE_NEW`,
`sensitivity_flux_and_adjoint`):

$$
K(\rho)\,\lambda_l = -w_l\frac{\partial f_\sigma^l}{\partial u}
= \frac{2 w_l}{|\Omega|\,\|\boldsymbol\Sigma_t^{l}\|^2}\, D^{T} W\, \mathbb C(\rho) : (\boldsymbol\Sigma_t^{l} - \boldsymbol\Sigma_h^{l}) .
$$

Here $\boldsymbol\Sigma_t - \boldsymbol\Sigma_h$ is broadcast to all quadrature points as a
constant "macro gradient". The nodal sensitivity assembled in `s_sensitivity_field` is

$$
\frac{dF}{d\rho_a} = \sum_l \Big[
\underbrace{-\frac{2 w_l}{|\Omega|\,\|\boldsymbol\Sigma_t^l\|^2}\int (\boldsymbol\Sigma_t^l-\boldsymbol\Sigma_h^l) : \mathbb C'(\rho) : (\mathbf E_l+\nabla^s u_l)\,N_a\,dx}_{w_l\,\partial f_\sigma^l/\partial\rho\ \text{(explicit)}}
+ \underbrace{\int \nabla^s\lambda_l : \mathbb C'(\rho) : (\mathbf E_l+\nabla^s u_l)\,N_a\,dx}_{\lambda_l^T\,\partial g_l/\partial\rho}
\Big]
+ \underbrace{2\eta\,(D^TWD\rho)_a + \frac{c_{dw}}{\eta}\frac{|\Omega|}{N}\,(2\rho_a - 6\rho_a^2 + 4\rho_a^3)}_{\partial f_\rho/\partial\rho\ \text{(sensitivity\_phase\_field\_term\_FE\_NEW)}} .
$$

The integrals are evaluated as $N^T W[\cdots]$. Both the state field $u_l$ and the adjoint
field $\lambda_l$ are kept between objective evaluations
(`displacement_field_load_case`, `adjoint_field_load_case`) and serve as CG warm starts.

### Linear solver and preconditioner

The preconditioner is selected with `preconditioner_type`. All scripts use
`'Green_Jacobi'`:

* `'Green'`: the FFT-diagonal Green operator of the homogeneous reference material
  $\mathbb C_1$ (`get_preconditioner_Green_mugrid`). It is built once.
* `'Jacobi'`: $\mathrm{diag}(K)^{-1}$, applied as `K_diag**2 * x`, because
  `get_preconditioner_Jacobi_mugrid` returns $\mathrm{diag}(K)^{-1/2}$.
* `'Green_Jacobi'`: $M^{-1} = J\,G\,J$ with $J = \mathrm{diag}(K(\rho))^{-1/2}$. Because the
  material changes with $\rho$, $J$ is recomputed in every objective evaluation.

CG tolerances (`cg_setup`). Note that `conjugate_gradients_mugrid` compares $(r,r)$ with
`tol**2`. The state solves use an **absolute** criterion (the default `rtol=False`), and
the adjoint solves use a **relative** one unless `r_tol=False` is set:

| Script | `cg_tol` | state solve | adjoint solve |
|---|---|---|---|
| 2D elasticity | 1e-3 | absolute | relative |
| tilled grid | 1e-6 | absolute | relative |
| 3D elasticity | 1e-6, `r_tol=False` | absolute | absolute |
| conductivity | 1e-6 | absolute | relative |

### Optimizer

All continuous examples except the tilled grid call NuMPI's bound-constrained L-BFGS:

```python
Optimization.l_bfgs_bounded(fun=objective_function_multiple_load_cases, x0=phase_field_0.s.ravel(),
                            jac=True, bounds_lo=0., bounds_hi=1., gtol=..., xtol=..., maxiter=500,
                            maxcor=20, c1=1e-4, max_halvings=40, comm=MPI.COMM_WORLD,
                            callback=my_callback, disp=True)
```

* `fun` returns `(F, dF/drho)` (`jac=True`). $F$ is a global scalar; the gradient is the
  rank-local slice.
* `gtol` is the tolerance on the infinity norm of the box-projected gradient. `xtol` stops
  when $\max|x_{k+1}-x_k|$ drops below it.
* The line search is Armijo backtracking (`c1`, at most `max_halvings` halvings).
* `x0` is projected onto the box before the first iteration.

There is **no continuation**: $\eta$, $p$, $w$ and the contrast stay fixed for the whole
run, and there is no outer loop.

### MPI handling of the design vector

The phase field lives on the muFFT domain decomposition. Each rank passes only its
subdomain (`phase_field_0.s.ravel()`, without ghost layers) to the optimizer, and NuMPI
reduces dot products over `comm`. Inside the objective:

* the flat vector is reshaped to `[1, 1, *discretization.nb_of_pixels]` (local pixels);
* all integrals and CG reductions are global (`discretization.mpi_reduction`,
  `discretization.communicator`);
* ghost layers are refreshed (`communicate_ghosts`) before stencil operations and at the
  end of the sensitivity computation.

Results are written collectively with `NuMPI.IO.save_npy`, which places each rank's block
at `subdomain_locations` in one global array. The `.npz` log is written by rank 0.

### Outputs

Each script creates `examples/topology_optimization/data/<script_name>/` and
`figures/<script_name>/`. Only `data/` is written to.

* `<preconditioner>_eta_<eta>_w_<weight>_final.npy` holds the optimized global phase field.
* `<preconditioner>_eta_<eta>_w_<weight>_log.npz` holds CG iteration counts
  (`num_iteration_mech`, `num_iteration_adjoint`), the objective history
  (`norms_sigma`, `norms_pf`, `norms_adjoint_energy`), `nb_iterations`, target and
  achieved homogenized stress or flux per load case, and the full homogenized tensor next
  to the target (`homogenized_C_ijkl`/`target_C_ijkl` in Voigt notation for elasticity,
  `homogenized_C_ij`/`target_C_ij` for conductivity).

Plotting happens only in the optimizer callback, as a `pcolormesh` of $\rho$ (or pyvista
slices in 3D) shown with `plt.show()`. Nothing is saved to `figures/`.

---

## 2. Walkthrough: `example_2D_elasticity_TO.py`

**Setup.**

* `example_2D_elasticity_TO.py:18-34`: problem and optimization parameters (64x64 pixels,
  `soft_phase_exponent = 5`, `preconditioner_type = "Green_Jacobi"`,
  `eta = max(pixel_size)` = 1/64, `weight = 5.`, `cg_tol = 1e-3`).
* `:37-50`: the periodic cell and the FE discretization on the pixel grid
  ([theory §4](../theory.md#4-finite-element-discretization-on-a-pixel-grid)). Each rank
  prints its subdomain.
* `:55-61`: the solid tensor $\mathbb C_1$ (`elastic_C_0`, $K_0 = 1$, $G_0 = 0.5$) and the
  void tensor $\mathbb C_0 = 10^{-5}\mathbb C_1$ (`elastic_C_void`).
* `:64-72`: the Green preconditioner for the reference material $\mathbb C_1$ and its
  application `M_fun_Green`
  ([theory §5](../theory.md#5-assembling-and-solving-the-linear-system)).

**Load cases and target.**

* `:76-80`: three macroscopic strains: $\mathbf E_1 = \mathbf e_1\otimes\mathbf e_1$,
  $\mathbf E_2 = \mathbf e_2\otimes\mathbf e_2$, and $\mathbf E_3$ = symmetric shear with
  $E_{12} = E_{21} = 0.5$. `left_macro_gradients` (`:83-86`) and `target_energy`
  (`:111, 115`) are computed but not used.
* `:95-104`: the isotropic target tensor with $\nu_t = 0$ and $G_t = \tfrac{3}{20}E_0$.
* `:113-114`: the target stress for each load case:
  ```python
  target_stresses[load_case] = np.einsum('ijkl,kl->ij', elastic_C_target, macro_gradients[load_case])
  ```

**Allocations.** `:121-122` allocate the persistent state and adjoint fields, one per
load case. `:124` sets $p = 2$ and `:139` sets $w = \texttt{weight}/L$.

**Objective and gradient** (`objective_function_multiple_load_cases`, `:145-327`).

1. `:151` unpacks the flat design vector into the nodal field $\rho$.
2. `:154` interpolates $\rho$ to the quadrature points ($N\rho$).
3. `:157-160` applies SIMP, $\mathbb C(\rho_q) = (\mathbb C_1-\mathbb C_0)\rho_q^p + \mathbb C_0$.
4. `:163-175` computes the phase-field term $f_\rho$ and its gradient. The gradient
   initializes `s_sensitivity_field`.
5. `:181-211` builds the preconditioner. For `Green_Jacobi` the Jacobi scaling is
   recomputed for the current $\mathbb C(\rho)$.
6. `:213-217` defines `K_fun`, the matrix-free product $K(\rho)x$ in small strain.
7. The load-case loop (`:227-322`):
   * `:229-236` broadcasts $\mathbf E_l$ to the quadrature points and assembles the right-hand
     side $-D^TW\mathbb C:\mathbf E_l$
     ([theory §1](../theory.md#1-periodic-homogenization-the-cell-problem));
   * `:250-260` solves the **state problem** with PCG, warm-started from the previous $u_l$;
   * `:277-281` computes the homogenized stress $\boldsymbol\Sigma_h^l$
     ([theory §6](../theory.md#6-effective-homogenized-properties));
   * `:283-285` computes the mismatch $f_\sigma^l$;
   * `:287-305` calls `sensitivity_stress_and_adjoint_FE_NEW`, which solves the **adjoint
     problem** and returns $w_l\,\partial f_\sigma^l/\partial\rho + \lambda_l^T\partial g_l/\partial\rho$
     together with the diagnostic $\lambda_l^Tg_l$
     ([theory §7](../theory.md#7-topology-optimization));
   * `:307-310` accumulates the sensitivity and the objective:
     ```python
     s_sensitivity_field.s[0, 0] += s_stress_and_adjoint_load_case.s[0, 0]
     objective_function += w * f_sigmas[load_case]
     objective_function += adjoint_energies[load_case]
     ```
8. `:323-327` refreshes the ghost layers of the sensitivity and returns `(F, grad)` as a
   flat local array.

**Driver** (`if __name__ == '__main__'`, `:330-551`).

* `:357-365` builds the initial guess. `random_init = False` gives
  $\rho_0 = \tfrac14(\sin 4\pi x + \sin 4\pi y + 2) + 0.5\,\mathcal U[0,1)$, with the RNG
  seeded by MPI rank. This lies in $[0, 1.5)$ and is projected onto $[0,1]$ by the optimizer.
  `apply_filter` (`:343-354`) is defined but not used.
* `:370-385` defines `my_callback`, which counts iterations and, in serial runs only,
  opens a blocking matplotlib window of $\rho$ at **every** iteration.
* `:395-413` runs L-BFGS-B (`gtol=1e-3`, `xtol=1e-3`, `maxiter=500`, `maxcor=20`).
* `:417-436` saves the optimum to `..._final.npy` with `save_npy`.
* `:442-502` post-processes the optimum. It rebuilds $\mathbb C(\rho^\*)$, re-solves the
  three load cases with the plain Green preconditioner and `tol=1e-5`, and prints the
  target and achieved $\boldsymbol\Sigma_h$.
* `:504-546` computes the full homogenized tangent column by column with unit strains
  $\mathbf e_i\otimes\mathbf e_j$, and prints it next to $\mathbb C_t$ in Voigt notation.
* `:549-551` writes the `_log.npz`.

---

## 3. The other examples (differences only)

### `example_2D_conductivity_TO.py` (current working-tree version)

* **Physics** (`:14, 18`): `problem_type = 'conductivity'`. The script imports
  `topology_optimization_conductivity as topology_optimization`, which re-exports the same
  phase-field functions. The unknown is a scalar temperature fluctuation
  (`get_temperature_sized_field`, `:105-106`). Material tensors are $2\times2$
  (`:57-61`, contrast $10^{-4}$). `K_fun` uses no `formulation` argument (`:196-199`).
* **Load cases** (`:76-79`): two unit temperature gradients, $\mathbf e_1$ and $\mathbf e_2$.
  **Target** (`:88-99`): $\mathbb C_t = \mathrm{diag}(0.5, 0.1)$, so the target fluxes are
  $(0.5, 0)$ and $(0, 0.1)$.
* **Objective**: the flux mismatch `compute_flux_equivalence_potential` with per-load-case
  weight `weights[l] = 5` (`:32, 210, 289`). The adjoint term is **not** added
  (`:290` is commented out). Sensitivity: `sensitivity_flux_and_adjoint` (`:268-285`).
* **Solver**: `cg_tol = 1e-6` (`:33`).
* **Initial guess** (`:316, 340-359`): `random_init = True` draws uniform $[0,1)$ values
  from `default_rng(42)`. Each rank skips ahead by the row-major offset of its subdomain,
  so the field does not depend on the number of ranks for slab decompositions.
* **Plotting** is off by default (`show_plots = False`, `:317, 373`).
* **Optimizer** (`:394-412`): `l_bfgs_bounded` with `gtol=1e-4` and `xtol=1e-4`; the other
  settings match 2D elasticity.
* **Rounding** (`:419-430`): with `nb_phase_levels = 10`, the continuous optimum is first
  saved as `..._smooth.npy`, then rounded to $\{0, 0.1, \dots, 1\}$. The `_final.npy`
  file and the post-processing therefore describe the **rounded** design. Set
  `nb_phase_levels = None` to keep the continuous field.
* **Post-processing** (`:514-554`) computes the $2\times2$ effective conductivity row by
  row with unit gradients. The printout still says "elastic tangent".
* File names use `weights[0]` as `w`.

### `example_2D_conductivity_TO_discrete.py`

This script re-optimizes the conductivity result with $\rho$ restricted to the discrete
set $\{0, 1/n, \dots, 1\}$. It is **serial only** (`assert MPI.COMM_WORLD.size == 1`, `:23`).

* `:19` imports `example_2D_conductivity_TO as base`. That runs the module-level setup of
  the base script (discretization, targets, objective) but not its optimizer.
* `:39-40` load `data/example_2D_conductivity_TO/<prec>_eta_<eta>_w_<w0>_smooth.npy`.
  **Run the base script first** with the same `number_of_pixels`, `eta`,
  `preconditioner_type` and `weights`.
* `:25-28` read the command line: `[n_levels]` (default 20) and `--flux-only`. With
  `--flux-only`, `full_objective` (`:84-97`) subtracts $f_\rho$ and its gradient, so only
  $\sum_l w_l f_\sigma^l$ is minimized.
* `:108-161` run a sensitivity-guided discrete descent on integer levels $k$, with
  $\rho = k/n$:
  * start from the smooth optimum rounded to the nearest level;
  * each step moves the `nb_moves` pixels with the largest $|\partial F/\partial\rho|$ by
    one level against the sign of the gradient, keeping $0\le k\le n$;
  * a step is accepted only if $F$ decreases;
  * on success `nb_moves` doubles, on failure it halves;
  * a pixel whose single-pixel move fails is blocked, and blocked pixels are released
    after any later progress;
  * the loop stops at `max_steps = 2000`, when no candidate is left, or after 200
    consecutive rejected single-pixel moves.
* `:46-77` (`homogenized_conductivity`) recomputes the effective tensor (CG `tol=1e-8`)
  for the smooth, rounded and re-optimized fields. `:176-181` print the objective,
  $C_{11}$, $C_{22}$, $C_{12}$ and the relative errors against the target.
* `:183-186` save `..._levels_<n>[_flux_only]_discrete.npy` and `..._log.npz` to
  `data/example_2D_conductivity_TO_discrete/`.

### `example_2D_elasticity_TO_tilled_grid.py`

* **Element and cell** (`:20, 24-25`): `linear_triangles_tilled` puts two P1 triangles in
  each pixel, under a shear map $x = h_x\xi + \tfrac{h_x}{2}\eta$, $y = h_y\eta$. The cell is
  $1 \times \sqrt3/2$ with 32x32 pixels, so $h_y = \tfrac{\sqrt3}{2}h_x$ and the triangles
  are **equilateral** (a hexagonal-lattice mesh on a sheared periodic cell).
* **Target** (`:95`): $\nu_t = -0.5$ (auxetic). **Solver** (`:34`): `cg_tol = 1e-6`.
* **Initial guess**: same as 2D elasticity, but `np.random.seed` is removed, so it is not
  reproducible.
* **Optimizer** (`:391-401`): **unconstrained** `Optimization.l_bfgs` with `gtol=1e-3`,
  `ftol=1e-5`, `maxiter=1000` and `maxcor=20`. The bounds $[0,1]$ are **not** enforced;
  only the double-well term keeps $\rho$ near $\{0,1\}$.
* **Plotting** (`:367-388`): the pixel coordinates are sheared for display. The callback
  plots on rank 0 at every iteration (blocking `plt.show()`), even under MPI.
* `save_npy` uses `discretization.subdomain_locations_no_buffers` (`:421`).

### `example_3D_elasticity_TO.py`

* **Discretization** (`:21, 25-26`): `trilinear_hexahedron_1Q` (8-node hexahedron, one
  Gauss point) on a 15x15x15 grid of the unit cube, giving $\eta = 1/15$. The one-point
  rule is rank deficient (hourglass modes); see `discretization_library.py`.
* **Material**: contrast $10^{-3}$ (`:31`). The SIMP broadcast has one extra `np.newaxis`
  for $z$ (`:159-161`, `:490-495`).
* **Load cases** (`:79-81`): only the three uniaxial strains
  $\mathbf e_i\otimes\mathbf e_i$. There are **no shear load cases**, so the shear part of
  $\mathbb C_t$ is not targeted. **Target**: $\nu_t = -0.3$ (`:96`).
* **Phase-field term**: `objective_function_phase_field_3D` (`:164`).
* **Solver** (`:35, 259`): `cg_setup = {'cg_tol': 1e-6, 'r_tol': False}`, so both the state
  and adjoint solves use an absolute tolerance.
* **Initial guess**: as in 2D. The sinusoid depends only on $x$ and $y$, and noise is added.
* **Optimizer** (`:436-454`): `l_bfgs_bounded` with `gtol=1e-5` and `xtol=1e-3`.
* **Plotting** (`:379-413`): in serial runs, every 10th iteration renders three
  orthogonal pyvista slices off-screen and shows them with matplotlib. This needs
  `pyvista`.
* **Post-processing** builds the full $3\times3\times3\times3$ homogenized tangent
  (`:554-555`, `range(dim)`).

---

## 4. Key tunable parameters

| Parameter | Where | Meaning |
|---|---|---|
| `number_of_pixels`, `domain_size` | top of each script | grid resolution and cell size; `eta` follows from them |
| `eta` | `max(1 * pixel_size)` | interface width of the phase-field regularization |
| `weight` / `weights` | elasticity / conductivity | weight $w$ of the mismatch term relative to $f_\rho$ |
| `double_well_depth_test` | objective | $c_{dw}$; must match the hard-coded `double_well_depth=1` in the sensitivity call |
| `p` | `p = 2` | SIMP exponent |
| `soft_phase_exponent` | top | void stiffness or conductivity $10^{-s}$ |
| `poison_target` / `poisson_target`, `G_target_auxet`, `conductivity_C_target` | target block | target effective tensor |
| `macro_gradients`, `nb_load_cases` | load-case block | which responses are matched |
| `preconditioner_type` | top | `'Green'`, `'Jacobi'` or `'Green_Jacobi'` |
| `cg_setup` | top | CG tolerance (`cg_tol`) and relative/absolute switch for the adjoint (`r_tol`) |
| `gtol`, `xtol`, `ftol`, `maxiter`, `maxcor` | optimizer call | L-BFGS stopping and memory |
| `random_init`, `show_plots` | `__main__` | initial guess, per-iteration plotting |
| `nb_phase_levels` | conductivity `__main__` | rounding of the final design (None = off) |
| `n_levels`, `--flux-only`, `max_steps`, `nb_moves_init` | discrete script | discrete re-optimization |

## 5. How to run

Run from the repository root with the project's Python environment. Each script adds the
repository root to `sys.path`.

```bash
python examples/topology_optimization/example_2D_elasticity_TO.py
mpirun -n 4 python examples/topology_optimization/example_2D_elasticity_TO.py   # MPI-parallel

python examples/topology_optimization/example_2D_conductivity_TO.py
mpirun -n 4 python examples/topology_optimization/example_2D_conductivity_TO.py

python examples/topology_optimization/example_3D_elasticity_TO.py               # or with mpirun
python examples/topology_optimization/example_2D_elasticity_TO_tilled_grid.py

# serial only; needs the *_smooth.npy written by example_2D_conductivity_TO.py
python examples/topology_optimization/example_2D_conductivity_TO_discrete.py 20
python examples/topology_optimization/example_2D_conductivity_TO_discrete.py 20 --flux-only
```

In serial runs the 2D elasticity, tilled-grid and 3D scripts open blocking plot windows
from the optimizer callback. Close each window to continue, or use a non-interactive
matplotlib backend (`MPLBACKEND=Agg`).

## Known issues

* **Objective vs. gradient in elasticity.** The elasticity scripts add the diagnostic
  $\lambda_l^Tg_l$ to $F$. With the loose absolute state tolerance in 2D (`cg_tol = 1e-3`),
  this term and the error in $\boldsymbol\Sigma_h$ are not negligible, which can disturb
  the Armijo line search. The conductivity script leaves the term out.
* **Unbounded tilled-grid run.** `example_2D_elasticity_TO_tilled_grid.py` uses
  unconstrained `l_bfgs`, so $\rho$ can leave $[0,1]$. SIMP with $p = 2$ is then not
  monotone, since $\rho^2$ is also large for $\rho < 0$.
* **Plotting under MPI.** The tilled-grid callback plots on rank 0 with the rank-local
  `x_current` and the global-size `shift` array. Under `mpirun` it shows only a part of the
  field and may fail on a shape mismatch. Its initial guess is also unseeded.
* **Blocking windows.** In serial runs, `my_callback` in 2D elasticity opens a blocking
  `plt.show()` window at every iteration. The conductivity script has a `show_plots`
  switch; the other scripts do not.
* **3D load cases.** The 3D example uses only normal-strain load cases, so the shear
  modulus of the result is not controlled.
* **Rounded conductivity output.** The conductivity `_final.npy` and its post-processed
  tensor refer to the design rounded to 11 levels, not to the L-BFGS optimum. The optimum
  is in `_smooth.npy`.
* **Dead code.**
  * `apply_filter` is unused.
  * `left_macro_gradients` and `target_energy` are unused.
  * The `LinearConstraint` in 2D elasticity is never passed to the optimizer.
  * `info_mech['num_iteration_adjoint']` actually stores the **state** CG iteration counts.
  * The "Data saved to" message omits the preconditioner prefix of the real file name.
* **Discrete script dependency.** The discrete script imports the base script, so any
  edit to the base parameters changes the expected `_smooth.npy` name. Every objective
  evaluation also prints the CG iteration counts of the base objective.
