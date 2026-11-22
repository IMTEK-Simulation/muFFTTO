# Internal contact with the third-medium method

Examples: [`examples/internal_contact/`](../../examples/internal_contact/)

| File | Role |
|---|---|
| `Example_2d_third_medium_contact_tr_minimal.py` | main example: monotonic loading, trust-region Newton-CG |
| `Example_2d_third_medium_contact_tr_minimal_step_lenght_control.py` | adds per-pixel load throttling |
| `Example_2d_third_medium_contact_tr_minimal_step_lenght_control_retract.py` | throttling plus an arbitrary load path (load, then retract) |
| `Example_2d_third_medium_contact_tr_minimal_net_cdf.py` | main example plus NetCDF output for muEye |
| `read_tmc.py` | reader and summary/plot tool for the NetCDF output of all four scripts |

Background: [cell problem](../theory.md#1-periodic-homogenization-the-cell-problem),
[finite-strain elasticity](../theory.md#3-finite-strain-elasticity),
[FE on a pixel grid](../theory.md#4-finite-element-discretization-on-a-pixel-grid),
[solving](../theory.md#5-assembling-and-solving-the-linear-system),
[notation](../theory.md#9-notation-and-array-conventions).

## Physics: third-medium contact (TMC)

The cell contains a stiff solid (a square frame with two "sticks" reaching into
an internal cavity). When the cell is sheared or compressed, the sticks move
towards each other and touch. The method does not detect contact surfaces.
Instead, the cavity is filled with a **third medium**: the same hyperelastic
material, but $k_v=10^{-5}$ times softer. While the gap is open, this medium carries
practically no load. As the gap closes it is compressed towards $\det F\to 0$. The
$\ln J$ terms of the neo-Hookean energy then blow up, and the medium transmits the
contact pressure through the continuum. The whole cell remains a single
periodic finite-strain boundary-value problem on the pixel grid.

A very soft medium under extreme compression gets badly distorted elements
and spurious modes. A **regularization** penalizes the curvature of the
displacement (the "HuHu-LuLu" term). It uses the second gradient
$\mathbb H u=\nabla\nabla u$ minus its trace part, the Laplacian $\mathbb L u=\Delta u$:

$$\Pi_r(\tilde u)=\frac{k_r}{2}\int_\Omega\Big(\mathbb H\tilde u:\mathbb H\tilde u-\frac1d\,\mathbb L\tilde u\cdot\mathbb L\tilde u\Big)\,\mathrm dx,
\qquad k_r=\alpha\,L^2\Big(K+\tfrac43 G\Big),\ \alpha=10^{-6}.$$

In these scripts $k_r$ is uniform over the whole cell (matrix and third medium).
It acts only on the fluctuation, since the affine part has zero second gradient.

## What PDE is solved

Unknown: periodic displacement fluctuation $\tilde u$. Deformation gradient with
macroscopic load factor $\lambda$:

$$F=I+\lambda H+\nabla\tilde u,\qquad \bar F=I+\lambda H ,$$

with $H_{10}=-0.6$ (shear $\partial u_y/\partial x$) in the main example. The material is the
compressible neo-Hookean model `material_models.NeoHookean`:

$$W(F)=\frac{\lambda_L}{2}(\ln J)^2+\frac{\mu}{2}(F:F-d)-\mu\ln J,\qquad P=\lambda_L\ln J\,F^{-T}+\mu(F-F^{-T}).$$

Here $\lambda_L,\mu$ are the Lamé fields: matrix $E=100,\ \nu=0.3$, third medium
$E=10^{-3}$. The equilibrium state for each $\lambda$ is the minimizer of

$$\Pi(\tilde u)=\int_\Omega W(F)\,\mathrm dx+\Pi_r(\tilde u),$$

$$\nabla\Pi=G^TWP+R\tilde u,\qquad \nabla^2\Pi=G^TW\,\mathbb C\,G+R,\qquad R=k_r\big(H^TH-\tfrac1d L^TL\big),$$

where $G$ is the FE gradient operator, $W$ the quadrature weights and
$\mathbb C=\partial P/\partial F$ the tangent. The macroscopic response reported is
$\bar P=\langle P\rangle$.

## Walkthrough: `Example_2d_third_medium_contact_tr_minimal.py`

**0. Dependency check** (`:30-38`). `NuMPI.Optimization.tr_newton_bounded` must
accept a `precond` argument. That exists only in a hand-patched NuMPI
`BoundedTRNewtonCG.py`, and the script raises `RuntimeError` otherwise.

**1. Discretization** (`:51-78`). 32×32 pixels, `'bilinear_rectangle'` elements,
`formulation='finite_strain'`, `ninc = 100` load increments.

**2. Materials** (`:93-116`). Matrix and third-medium Lamé constants,
$k_r$, and the reference tensor `ref_mat` $=\lambda I\otimes I+2\mu\,\mathbb I^s$ of the matrix
for the preconditioner.

**3. Geometry and material fields** (`:121-149`). `'contact_test_geometry_2'`:
a solid frame (border width 0.15) around a cavity, with a lower-left stick
($x\in[0.15,0.52),\ y\in[0.35,0.45)$) and an upper-right stick
($x\in[0.47,0.95),\ y\in[0.55,0.65)$) that overlap in $x$ and are separated by a gap
of 0.1 in $y$. Phase $>0$ is solid and phase $=0$ is the third medium. $\lambda_L,\mu$ are
stored per quadrature point and passed to `NeoHookean`.

**4. Deformation gradient** (`:221-228`). `set_F` rebuilds $F=I+\lambda H+G\tilde u$ from
scratch at every evaluation.

**5. Regularization $R\tilde u$** (`:234-248`). Uses the Hessian operator and its
transpose, plus the Laplacian and its transpose:

```python
discretization.apply_hessian_operator_to_vector_field_mugrid(u_inxyz=x, hess_u_ijkqxyz=hess_u_ijkqxyz)
discretization.apply_hessian_operator_transposed_to_vector_field_mugrid(..., nodal_field_inxyz=HtH_field, apply_weights=True)
discretization.laplacian.apply(nodal_field=x, quadrature_point_field=lap_u_inxyz)
discretization.laplacian.transpose(..., nodal_field=LtL_field, weights=discretization.quadrature_weights)
out.s[...] += scale * k_r * (HtH_field.s - inv_tr_I * LtL_field.s)
```

**6. Green preconditioner** (`:258-283`).
`get_preconditioner_Green_mugrid(reference_material_data_ijkl=ref_mat, operator=operator_for_preconditioner)`
inverts, in Fourier space, the constant-coefficient operator
$M=G^TW\,C_{\mathrm{ref}}\,G+R$, so the regularization is part of the preconditioner. $M$
is built once and never updated. The trust region is measured in the $M$-norm, and
changing $M$ during the run would change what the radius means. `precond(v)`
returns $M^{-1}v$.

**7. Objective, gradient, Hessian-vector product** (`:292-347`).

- `fun_grad(x)` sets $F$, checks $\min J$, and returns
  $\Pi=\sum_q w_qW+\tfrac12\langle\tilde u,R\tilde u\rangle$ and $\nabla\Pi=G^TWP+R\tilde u$.
  If any $J\le0$ (or $\Pi$ is not finite), it returns the finite penalty
  `INADMISSIBLE_ENERGY = 1e30`. The trust-region step is then rejected and the
  radius shrinks; `np.inf` would give `nan` ratios (`:294-297`).
- `hessp(x, v)` evaluates the algorithmic tangent $\mathbb C(F(x))$ and returns
  $(G^TW\mathbb C G+R)v$ via `apply_system_matrix_mugrid(material_data_field=tangent_field, formulation='finite_strain')`.

All dot products and sums are MPI-reduced (`global_sum`, `dot_global`, `:192-205`).

**8. Load stepping and Newton solve** (`:353-458`). $\lambda=k/n_{inc}$, $k=1..100$. Each
increment is one call, warm-started from the previous $\tilde u$:

```python
res = tr_newton_bounded(fun=fun_grad, x0=u, hessp=hessp, precond=precond, jac=True,
                        gtol=1e-4, maxiter=500, inner_tol=1e-6, inner_maxiter=None, comm=comm)
```

This is a trust-region Newton method whose inner solver is a Steihaug
(truncated) preconditioned CG on $\nabla^2\Pi\,\delta=-\nabla\Pi$. It stops when
$\|\nabla\Pi\|_\infty<$ `gtol`. Each increment prints the outer iterations, Hessian
products, $\|\nabla\Pi\|_\infty$, $\Pi$, $\min J$ and the mean $P_{xx}$, $P_{yx}$. Every
`plot_every = 10` increments (serial only), the phase is drawn on the deformed grid
$x=X+\lambda H X+\tilde u$.

**9. Response curves** (`:470-493`). $\bar P_{10}$ and $\bar P_{xx}$ against
$\bar F_{10}=\lambda H_{10}$, $\min\det F$ (log scale) and $\Pi$ against $\lambda$. Contact
shows up as a stiffening in $\bar P_{10}(\bar F_{10})$, together with $\min\det F$
dropping by orders of magnitude in the third medium.

## Variant: `..._step_lenght_control.py` (per-pixel load throttling)

This variant does not add trust-region step-length control. It limits how much
macroscopic load each quadrature point receives. The imposed deformation becomes
an accumulated field $H_{\mathrm{imp}}(q,x)$ (`imposed_field`), and
$F=I+H_{\mathrm{imp}}+\nabla\tilde u$ (`set_F`, `:243-254`). Before each solve, starting from the
converged $F$:

- `admissible_scale_field` (`:257-294`) computes, per point, the largest $s>0$ with
  $\det(F+sH)>0$. In 2D this is exact, because $\det(F+sH)=\det F+s\,b+s^2\det H$ is quadratic in
  $s$. It uses the linear root when $\det H=0$, as for pure shear $H_{10}$.
- `apply_throttled_load` (`:297-319`) adds
  $\theta\,\Delta\lambda\,H$ with $\theta=\min\big(1,\ \texttt{PIXEL\_SAFETY}\cdot s/\Delta\lambda\big)$,
  `PIXEL_SAFETY = 0.5` (`:82`).

As a consequence, $H_{\mathrm{imp}}$ is no longer uniform, so $\langle F\rangle$ drifts from
$I+\lambda H$, and the imposed part is in general not a compatible gradient. The script
reports the number of throttled points, $\theta_{\min}$, and the target and
actual mean of $F_{10}$ (drift). The plots show $\bar P_{10}$ against both the actual and
nominal $\bar F_{10}$, the drift, $\min\det F$ and the throttled-point count. The
deformed-mesh plot uses the mean imposed gradient and is only indicative.

## Variant: `..._step_lenght_control_retract.py` (load path with unloading)

Same throttling as above, plus:

- **Geometry and load**: `'contact_test_geometry_1'` (sticks offset in $y$ with a
  horizontal gap) under compression $H_{00}=-0.3$ (`:142`, `:448`). The
  reported component is chosen automatically as the largest $|H_{ij}|$ (`:454-458`).
- **Load path**: `waypoints = [0.0, 1.0, 0.0]` and `per_leg = ninc` (`:475-494`)
  build a schedule of signed $\Delta\lambda$ (load to 1, then back to 0). Other paths
  such as `[0, 1, 0.3, 1]` work as well, but the path must start at 0.
- **Signed throttling**: the admissible scale is measured along
  $\mathrm{sign}(\Delta\lambda)H$ and $\theta$ scales $|\Delta\lambda|$ (`:300-326`).
- **Return check**: if the path ends at $\lambda=0$, it prints the residual imposed
  $F$, $\Pi$, $\bar P$ and $\|\tilde u\|_\infty$, all of which should be 0. Because throttling
  is one-way, $H_{\mathrm{imp}}$ need not return to zero.
- `plot_every = 1`. Curves are coloured by leg, with leg boundaries marked.

## Variant: `..._net_cdf.py` (NetCDF output for muEye)

Same solver as the main example (no throttling), with these differences:
`nnn = 128`, `H_macro[1, 0] = -0.3`, `plot_every = 0`. Solver tolerances are
named constants (`SOLVER_GTOL`, `SOLVER_MAXITER`, `SOLVER_INNER_TOL`), and the run
writes

```
examples/internal_contact/exp_data/Example_2d_third_medium_contact_tr_minimal_net_cdf/Nx=<nnn>Ny=<nnn>/tmc_run.nc
                                                                                         /response.png
```

using `muFFTTO.io_utils.FieldWriter` (`:519`, see [Saving and loading fields](../io.md)):

- **Per-frame fields** (pixel-level, one component axis, `_pixel_field`, `:430`).
  `u_total` $=(\lambda H\cdot X+\tilde u)/h$ in grid-point units, so the muEye warp scale 1 is
  the physical deformation. Also written: `u_fluc_only`, `phase_field`, `detF`, and
  `F_flat` / `P_flat` with component $c=i\,d+j$. $F$ and $P$ are **averaged over quadrature
  points**, which smooths out a single collapsing point (`update_view_fields`, `:470`).
  The raw fluctuation `u_fluc` (physical units) is stored as well, so a frame can be
  loaded back into the discretization with `io_utils.load_fields`.
- **Frame variables** (the increment history, one value per frame): `increment`, `lam`,
  `applied_deformation_gradient` $=I+\lambda H$, `F10`, `P10`, `Pxx`, `min_det_F`, `energy`,
  `nb_outer`, `nb_hessp`, `nb_precond`, `grad_inf`, `converged` (0: the increment did
  not reach `gtol` and is not an equilibrium) and `elapsed_time`.
- **Global attributes**: run parameters (`k_v`, `alpha`, `k_r`, `H_macro`, tolerances,
  `command_line`, ...), the phase legend, and `deformation_gradient = I`. muEye reads
  the cell shape from that attribute once, so the macro deformation is folded into
  `u_total` instead.
- A frame is written every `dump_every = 1` increments (`write_frame`, `:574`) and
  synced to disk, so a killed run leaves a complete file up to its last frame. The last
  increment is always written. With `dump_every > 1` the history is stored only for
  the written increments.

### Output of the other three scripts

With `output_name = 'tmc_run.nc'` (`None` switches it off), the minimal and the two
throttling scripts write `exp_data/<script>/Nx=<nnn>Ny=<nnn>/tmc_run.nc` with one frame
per increment: the fields `phase_field` and `u_fluc` (plus `H_imposed`, the throttled
imposed gradient, in the throttling variants) and the same frame variables as above
(without `applied_deformation_gradient` and `nb_precond`). The throttling script adds
`F10_nominal` and `throttled_points`. The retract script stores the driven component
as `F_driven`, `F_driven_nominal`, `P_driven`, plus `mean_P`, `branch` and `leg`.

### `read_tmc.py`

```python
from read_tmc import load
r = load('tmc_run.nc')
r.hist['min_det_F']; r.attrs['k_v']; r.frames; r.phase; r.detF(-1); r.bad_increments
r.field('u_fluc', -1)        # [2, 1, x, y] at the last stored frame
```

`TMCRun` reads the file with `io_utils.read_file`: `hist` are the frame variables,
`frames` the increment of each stored frame, and `field(name, frame)` any stored field
on the global grid. It lists increments that did not converge (those are not
equilibria) and gives the static phase field. It reads the files of all four scripts.
From the command line:

```bash
python examples/internal_contact/read_tmc.py path/to/tmc_run.nc          # summary
python examples/internal_contact/read_tmc.py path/to/tmc_run.nc --plot   # response curves
```

Files written by the earlier version of the `_net_cdf` script (direct `FileIONetCDF`,
histories as `hist_*` attributes) are not read by the new `read_tmc.py`.

## Summary

| File | Dim | Physics | Key method | Outputs |
|---|---|---|---|---|
| `..._tr_minimal.py` | 2D | finite-strain neo-Hookean + TMC, HuHu-LuLu | energy minimization, `tr_newton_bounded` + Green-preconditioned Steihaug CG, uniform $\Delta\lambda$ | console log, deformed-mesh plots, response curves, `tmc_run.nc` |
| `..._step_lenght_control.py` | 2D | same | + per-point load throttling ($\det F>0$ admissible scale) | + drift / throttled-count plots |
| `..._step_lenght_control_retract.py` | 2D | same, compression, geometry 1 | + signed waypoint load path (load/unload) | + return-to-origin report, per-leg curves |
| `..._net_cdf.py` | 2D | same as minimal (128², $H_{10}=-0.3$) | + muEye view fields via `io_utils.FieldWriter` | `exp_data/.../tmc_run.nc`, `response.png` |
| `read_tmc.py` | – | post-processing | `io_utils.read_file` | summary, response plots |

## Key tunable parameters

| Parameter | Meaning |
|---|---|
| `nnn`, `ninc` | grid size and number of load increments |
| `H_macro` | macroscopic load direction (shear `[1,0]` or compression `[0,0]`) |
| `geometry_name` | `'contact_test_geometry_1/2/3'` |
| `E_matrix`, `nu_matrix` | solid material |
| `k_v` | third-medium stiffness ratio (smaller means less parasitic stiffness but harder to solve) |
| `alpha` (→ `k_r`) | regularization strength |
| `gtol`, `maxiter`, `inner_tol` | trust-region Newton tolerances |
| `PIXEL_SAFETY` | throttling fraction of the admissible scale (throttling variants) |
| `waypoints`, `per_leg` | load path (retract variant) |
| `plot_every`, `dump_every`, `output_name` | plotting and output (`dump_every`: `_net_cdf` only) |

## How to run

```bash
python examples/internal_contact/Example_2d_third_medium_contact_tr_minimal.py
mpirun -n 4 python examples/internal_contact/Example_2d_third_medium_contact_tr_minimal_net_cdf.py
```

All scripts reduce energies, dot products and $\min J$ over `MPI.COMM_WORLD` and
pass `comm` to the optimizer, so `mpirun` is supported for the solve. The
deformed-mesh plots run only in serial. Requirements: the patched NuMPI (see
step 0); `muGrid` with NetCDF support for the output; and `netCDF4` for
`read_tmc.py`.

## Known issues

- **Patched NuMPI required.** Stock NuMPI 0.15.1 lacks `precond` in
  `tr_newton_bounded`, and all four solvers abort at import.
- The variant name "step_lenght_control" is misspelled and describes load
  throttling, not trust-region step control.
- `sys.path.append` (not `insert`) means an installed `muFFTTO` takes precedence
  over the repository copy.
- The retract variant computes `dlam_max` and `lamn` but never uses them.
