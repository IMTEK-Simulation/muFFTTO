# Theory and numerics of muFFTTO

This page explains the mathematics behind muFFTTO: what PDE the examples solve,
how it is discretized and solved, and how the solver is used inside topology
optimization. It is written for a new PhD student who knows basic continuum
mechanics and finite elements but has not met FFT-accelerated homogenization
before. Each concept is linked to the code that implements it, using function
and attribute names (e.g. `Discretization.apply_gradient_operator_mugrid`) and
file paths (e.g. `muFFTTO/domain.py`). Where the code departs from the textbook
version (sign conventions, which norm CG uses, index order), the text says so.

The per-example walkthroughs in [`docs/examples/`](examples/) link back to the
sections below.

**Contents**

1. [Periodic homogenization: the cell problem](#1-periodic-homogenization-the-cell-problem)
2. [Physical problems: conductivity and small-strain elasticity](#2-physical-problems-conductivity-and-small-strain-elasticity)
3. [Finite-strain elasticity](#3-finite-strain-elasticity)
4. [Finite-element discretization on a pixel grid](#4-finite-element-discretization-on-a-pixel-grid)
5. [Assembling and solving the linear system](#5-assembling-and-solving-the-linear-system)
6. [Effective (homogenized) properties](#6-effective-homogenized-properties)
7. [Topology optimization](#7-topology-optimization)
8. [Grid adaptation (deformed grids)](#8-grid-adaptation-deformed-grids)
9. [Notation and array conventions](#9-notation-and-array-conventions)
10. [References](#references)

**The pipeline at a glance.** Almost every example follows the same steps:

```
PeriodicUnitCell  ->  Discretization  ->  material field C(x_q)  ->  rhs f = -B^T W C E
      ->  PCG with Green (FFT) preconditioner  ->  fluctuation u~  ->  <sigma> = <C (E + grad u~)>
```

Topology optimization wraps this loop in an outer optimizer and adds an adjoint
solve for each load case. Section 5 gives a runnable sketch of the
pipeline.

---

## 1. Periodic homogenization: the cell problem

### 1.1 Scale separation and the unit cell

Think of a composite whose microstructure has a characteristic length $\ell$,
used in a component of size $L \gg \ell$. Resolving every inclusion in a
simulation of the whole component is far too expensive. *Homogenization*
replaces the heterogeneous material with an equivalent homogeneous one, whose
**effective** (homogenized) properties come from a boundary value problem
posed on one representative piece of the microstructure.

If the microstructure is periodic, or if we simply assume periodicity for a
representative volume, that piece is a **periodic unit cell**
$Y = [0, Y_1] \times \dots \times [0, Y_d]$ with $d \in \{2, 3\}$. In muFFTTO
the cell is a `PeriodicUnitCell` (`muFFTTO/domain.py`):

| attribute | meaning |
|---|---|
| `domain_size` | edge lengths $(Y_1, \dots, Y_d)$ |
| `domain_dimension` | $d$ |
| `domain_volume` | $\lvert Y \rvert = \prod_i Y_i$ |
| `problem_type` | `'conductivity'` (scalar unknown) or `'elasticity'` (vector unknown) |

### 1.2 Macroscopic gradient and periodic fluctuation

Scale separation lets us assume that the macroscopic field is locally affine at
the microscale. The microscopic field $u$ (a temperature or a displacement) is
split into an affine part, driven by a prescribed **macroscopic gradient** $E$,
and a **periodic fluctuation** $\tilde u$:

$$
u(x) = E \cdot x + \tilde u(x), \qquad \tilde u \ \text{is } Y\text{-periodic}.
$$

Its gradient is therefore

$$
\nabla u = E + \nabla \tilde u, \qquad \langle \nabla \tilde u \rangle = 0,
$$

where $\langle \cdot \rangle = \lvert Y\rvert^{-1}\int_Y \cdot \, dx$ is the
volume average. The average of a gradient of a periodic function vanishes, so
$\langle \nabla u\rangle = E$: the prescribed $E$ is exactly the average
microscopic gradient.

In muFFTTO the unknown is always the fluctuation $\tilde u$ (often named
`solution_field`, `displacement_field` or `displacement_fluctuation`). The
macroscopic gradient is stored as a constant quadrature-point field and is
built by `Discretization.get_macro_gradient_field_mugrid(macro_gradient_ij,
macro_gradient_field_ijqxyz)`, which broadcasts a small `[d, d]` (or `[d]` for
conductivity) array to every quadrature point.

### 1.3 Strong and weak form of the cell problem

Write $\sigma(x) = \mathbb C(x) : \nabla u$ for the flux or stress, where
$\mathbb C$ is the material tensor (Section 2). Equilibrium without body
forces, posed on the cell, reads:

$$
\text{find periodic } \tilde u:\quad
-\nabla\cdot\big(\mathbb C(x) : (E + \nabla\tilde u(x))\big) = 0 \quad \text{in } Y .
$$

Periodicity of $\tilde u$ together with anti-periodic tractions replaces the
boundary conditions. The **weak form** is: find $\tilde u \in H^1_{\rm per}(Y)$
such that

$$
\int_Y \nabla v : \mathbb C : \big(E + \nabla \tilde u\big)\,dx = 0
\qquad \forall\, v \in H^1_{\rm per}(Y).
$$

Moving the known part to the right-hand side gives the linear problem

$$
\underbrace{\int_Y \nabla v : \mathbb C : \nabla\tilde u \,dx}_{a(v,\tilde u)}
= \underbrace{-\int_Y \nabla v : \mathbb C : E\,dx}_{\ell(v)} .
$$

The bilinear form $a$ is symmetric and positive semidefinite. Its kernel is
made of the constant fields (constant temperature, or rigid translations).
Section 5.4 explains how the solver deals with this kernel.

### 1.4 Effective tensor

Once $\tilde u$ is known for a given $E$, the **macroscopic flux/stress** is
the volume average

$$
\Sigma = \langle \sigma \rangle = \frac{1}{\lvert Y\rvert}\int_Y \mathbb C : (E + \nabla\tilde u)\,dx .
$$

$\tilde u$ depends linearly on $E$, so $\Sigma = \mathbb C^{\rm eff} : E$ for
an effective tensor $\mathbb C^{\rm eff}$. Two equivalent characterizations
are useful:

* **Stress averaging.** Solve one cell problem per unit load case
  $E = e_k \otimes e_l$ (or $E = e_k$ for conductivity). The average
  $\langle\sigma\rangle$ is then one column of $\mathbb C^{\rm eff}$.
* **Energy equivalence (Hill–Mandel).** Testing the weak form with
  $v = \tilde u$ shows $\langle \nabla \tilde u : \sigma \rangle = 0$, hence
  $$E : \mathbb C^{\rm eff} : E = \langle (E+\nabla\tilde u) : \mathbb C : (E+\nabla\tilde u) \rangle = E : \langle \sigma \rangle .$$

Both are implemented, see [Section 6](#6-effective-homogenized-properties).

---

## 2. Physical problems: conductivity and small-strain elasticity

`PeriodicUnitCell.__init__` fixes the component shapes of the unknown, its
gradient and the material tensor from `problem_type`:

| `problem_type` | unknown $\tilde u$ | `unknown_shape` | gradient | `gradient_shape` | material | `material_data_shape` |
|---|---|---|---|---|---|---|
| `'conductivity'` | temperature $\theta$ (scalar) | `[1]` | $\nabla\theta$ | `[1, d]` | $A_{ij}$ | `[d, d]` |
| `'elasticity'` | displacement $u_i$ (vector) | `[d]` | $\nabla u$ or $\varepsilon$ | `[d, d]` | $C_{ijkl}$ | `[d, d, d, d]` |

The scalar case keeps a leading component axis of size 1 (`[1, d]` rather than
`[d]`). This lets the same code path, the gradient operator plus a material
contraction, serve both problems.

### 2.1 Heat (or electric, or diffusive) conduction

* Unknown: temperature fluctuation $\tilde\theta$; the macroscopic gradient
  $E \in \mathbb R^d$.
* Flux: $q_i = A_{ij}\,(E_j + \partial_j\tilde\theta)$, with $A$ symmetric
  positive definite.
* Code: `Discretization.apply_material_data_conductivity_mugrid` computes
  `g[u, i] <- A[i, j] g[u, j]` in place (the `u` axis is the size-1 component
  axis).

**Sign convention.** The physical Fourier law is $q = -A\nabla\theta$. muFFTTO
drops the minus sign everywhere and calls $A\nabla\theta$ the "flux". This
does not change the effective tensor, but keep it in mind when comparing flux
fields with literature.

### 2.2 Small-strain (linearized) elasticity

* Unknown: displacement fluctuation $\tilde u_i$; macroscopic strain $E_{ij}$.
* Kinematics: $\varepsilon = \operatorname{sym}(\nabla u) = \tfrac12(\nabla u + \nabla u^T)$.
  In code, `Discretization.apply_gradient_operator_symmetrized_mugrid` first
  calls the gradient operator and then sets `g <- (g + g^T)/2`. Any routine
  that takes `formulation='small_strain'` uses this symmetrized gradient. Any
  other value, or `None`, uses the full gradient.
* Hooke's law: $\sigma_{ij} = C_{ijkl}\,\varepsilon_{kl}$. For an isotropic
  material, $C_{ijkl} = \lambda\delta_{ij}\delta_{kl} + \mu(\delta_{ik}\delta_{jl} + \delta_{il}\delta_{jk})$.

Material helpers in `muFFTTO/material_models.py`:

| function | returns |
|---|---|
| `get_bulk_and_shear_modulus(E, poisson)` | $K = E/(3(1-2\nu))$, $G = E/(2(1+\nu))$ (3D formulas) |
| `get_lame_parameters(E, poisson)` | $(\lambda, \mu)$ |
| `get_lame_parameters_from_bulk_and_shear(K, G, dim)` | $\mu=G$, $\lambda = K - \tfrac{2}{\rm dim}G$ |
| `get_elastic_material_tensor(dim, K, mu)` | $K\,\delta\delta + \mu(\delta\delta + \delta\delta - \tfrac23 \delta\delta)$, a full `[d,d,d,d]` array |
| `get_elastic_tensor_from_lame(dim, lam, mu)` | $\lambda\delta\delta + \mu(\dots)$ |
| `LinearElastic(discretization, lam_1qxyz, mu_1qxyz)` | field-based model: `get_stress`, `get_algorithmic_tangent` fill $C(x_q)$ |

> **Careful with 2D.** `get_elastic_material_tensor` always uses the 3D factor
> $2/3$, so in 2D it is plane strain with a 3D bulk modulus.
> `get_lame_parameters_from_bulk_and_shear(K, G, dim=2)` instead treats $K$
> as the 2D bulk modulus $\lambda+\mu$. The same pair $(K, G)$ therefore gives
> *different* $\lambda$ in the two helpers. The small-strain elasticity
> example builds its material with the second helper and its reference
> material for the preconditioner with the first. This affects only the
> preconditioner quality, not the result. It does matter when you compare with
> analytical values.

### 2.3 The index convention of the contraction

`muFFTTO/tensor_operations.py` contracts the *inner* indices:

$$
\texttt{ddot42}:\quad \sigma_{ij} = C_{ijkl}\,\varepsilon_{lk} .
$$

`Discretization.apply_material_data_elasticity_mugrid` follows the same rule.
For tensors with minor symmetry this is the usual $C_{ijkl}\varepsilon_{kl}$.
It matters for non-symmetric tangents such as the finite-strain
$\partial P/\partial F$ (Section 3), which the code stores with the last two
indices swapped.

### 2.4 Material data layout per quadrature point

Material properties are **not** stored per pixel but per quadrature point:

* conductivity: `material_data_field_ijqxyz`, `.s` shape `[d, d, q, x, y(, z)]`
* elasticity: `material_data_field_ijklqxyz`, `.s` shape `[d, d, d, d, q, x, y(, z)]`

Allocate the field with `Discretization.get_material_data_size_field_mugrid(name)`.
The examples fill it in one of two ways:

1. Broadcast a constant tensor and scale it by phase masks:
   `C.s[...] = C_1[..., np.newaxis, np.newaxis, np.newaxis]`, then
   `C.s[..., matrix_mask] *= contrast`.
2. Evaluate a constitutive model: `LinearElastic.get_algorithmic_tangent(strain, C)`,
   or `topology_optimization.material_interpolation_simp(rho_q, C, p, C0, C1, dim)`
   for a phase field.

Because data lives at quadrature points, a pixel can hold different materials
at different quadrature points. With `linear_triangles`, for example, the two
triangles of a pixel can differ. The phase field of the topology-optimization
examples is interpolated to the quadrature points with $N$ (Section 4.4)
before the material law is evaluated.

---

## 3. Finite-strain elasticity

### 3.1 Kinematics and stress

At finite strain the unknown is still a periodic displacement fluctuation, and
the macroscopic loading is a prescribed **macroscopic displacement gradient**
$H$:

$$
F = I + H + \nabla_X \tilde u, \qquad
\text{equilibrium: } \nabla_X\cdot P(F) = 0 \text{ in } Y,
$$

with $F$ the deformation gradient and $P$ the **first Piola–Kirchhoff**
stress. In the weak form, $\int_Y \nabla v : P(F)\,dX = 0$ for all periodic
$v$. No symmetrization is applied (`formulation='finite_strain'` means the
full gradient).

### 3.2 Neo-Hookean model

`material_models.NeoHookean` (compressible Simo–Pister form), with
$J = \det F$:

$$
W(F) = \tfrac{\lambda}{2}(\ln J)^2 + \tfrac{\mu}{2}(F:F - d) - \mu\ln J ,\qquad
P = \frac{\partial W}{\partial F} = \lambda \ln J\, F^{-T} + \mu\,(F - F^{-T}).
$$

* `get_energy_density(F, W)`, `get_stress(F, P)` and
  `get_algorithmic_tangent(F, A)` work on quadrature-point fields.
  $\lambda,\mu$ are scalar quadrature fields (`get_quad_field_scalar`).
* The tangent is the consistent linearization
  $\partial P_{ij}/\partial F_{kl} = \lambda F^{-T}_{ij}F^{-T}_{kl} + (\mu - \lambda\ln J)F^{-T}_{il}F^{-T}_{kj} + \mu\,\delta_{ik}\delta_{jl}$,
  stored with $k \leftrightarrow l$ swapped to match the contraction
  convention of Section 2.3.
* For $F \to I$ the model linearizes to `LinearElastic` with the same
  $\lambda,\mu$.

### 3.3 Newton–CG

`solvers_nonlinear.solve_finite_strain_newton_cg(discretization, material, macro_gradient_ij, ninc, ...)`
implements an incremental, inexact Newton–Krylov method:

```
for each load increment:  H_tot += macro_gradient_ij / ninc
    R = -B^T W P(F)                       # compute_residual(): also refreshes the tangent A(F)
    repeat (Newton):
        solve  K(F) du = R  with PCG       # K(F) = B^T W A(F) B, relative tol cg_tol
        u~ += du ;  F += B du              # full step, no line search
        R  = -B^T W P(F)
    until ||R|| / ||R_0|| < newton_tol (and at least 2 Newton steps)
```

* The linearized operator is `apply_system_matrix_mugrid` with the current
  tangent field as "material". This is the same matrix-free operator as in
  the linear case, with $\mathbb C$ replaced by $\mathbb A = \partial P/\partial F$.
* Preconditioners: `'Green'` (a reference material built once, default the
  symmetric identity $\mathbb I^s$), `'Green_Jacobi'`, or none (Section 5).
* CG is called with `rtol=True`, so each Newton step solves only to a
  *relative* tolerance (inexact Newton).

The finite-strain example scripts in
`examples/homogenization/finite_strain_elasticity/` write out the same loop by
hand. They start from `total_strain_field = I` and add the increment of $H$,
so the field holds $F$ itself.

> **Note (code vs. theory).** `solve_finite_strain_newton_cg` gets its
> `total_strain_field` from the field collection and only adds the macro
> increments. It does not add the identity. On a fresh field this gives
> $F = H + \nabla\tilde u$ instead of $I + H + \nabla\tilde u$, and
> `NeoHookean` then takes $\ln\det F$ of a nearly singular tensor. Either
> pre-fill the field named `'total_strain_field'` with $I$, as the examples
> do, or include $I$ in the load path.

---

## 4. Finite-element discretization on a pixel grid

### 4.1 Regular periodic grid

`Discretization(cell, nb_of_pixels_global, discretization_type='finite_element', element_type=...)`
splits the cell into $N_1\times\dots\times N_d$ identical pixels (voxels in
3D) of size `pixel_size` $= Y_i/N_i$. Every pixel carries the same reference
element. Because of periodicity, each pixel **owns** only the nodes at its
lower-left corner, plus edge or centre nodes for quadratic elements. Nodes on
the right and top boundaries are periodic images. So:

* a nodal field has local shape `[f, n, x, y(, z)]`, with `n = nb_nodes_per_pixel`;
* a quadrature field has shape `[i, j, q, x, y(, z)]`, with `q = nb_quad_points_per_pixel`.

`discretization_type='Fourier'` is accepted by the constructor, but it does not
set up the field collections or operators. Only the finite-element path is
functional.

### 4.2 Element library

`muFFTTO/discretization_library.py` defines the elements. Each `Element`
factory writes `shape_functions(xi)` in `jax.numpy` and gets first and second
derivatives by automatic differentiation. Every pixel is an affine image of
the reference element with constant Jacobian `jacobian_of_pixel`, so the
operators are computed once for one pixel.

| `element_type` | dim | description | quad. points `q` | nodes/pixel `n` |
|---|---|---|---|---|
| `linear_1D` | 1 | 2-node linear | 1 (midpoint) | 1 |
| `quadratic_1D` | 1 | 3-node quadratic | 3 Gauss | 2 |
| `linear_triangles` | 2 | pixel split into two P1 triangles | 2 (one centroid per triangle) | 1 |
| `linear_triangles_tilled` | 2 | P1 triangles on a sheared (hexagonal) lattice, $J=\begin{pmatrix}h_x & h_x/2\\ 0 & h_y\end{pmatrix}$ | 2 | 1 |
| `bilinear_rectangle` | 2 | Q1 quadrilateral | 4 (2×2 Gauss) | 1 |
| `biquadratic_rectangle` | 2 | Q9 quadrilateral | 9 (3×3 Gauss) | 4 |
| `trilinear_hexahedron` | 3 | Q1 hexahedron | 8 (2×2×2 Gauss) | 1 |
| `trilinear_hexahedron_1Q` | 3 | Q1 hex, one-point (under-)integration | 1 | 1 |

`get_shape_function_gradient_matrix(domain, element_type)` attaches the
element data to the `Discretization`:

* `quadrature_weights` `(q,)`: **physical** weights that already include the
  pixel measure, so $\sum_{q,\text{pixels}} w_q f(x_q) \approx \int_Y f\,dx$.
  For `linear_triangles`, $w_q = h_xh_y/2$.
* `quad_points_coord`, `quad_points_coord_parametric`
* `B_grad_at_pixel_dqnijk` `(d, q, n, 2, 2[, 2])`: the gradient stencil.
  Entry `[d, q, n, i, j, k]` is $\partial N/\partial x_d$ at quadrature point
  `q` for nodal DOF `n` of the pixel at offset `(i, j, k)` ∈ {0,1}^d.
* `N_at_quad_points_dqnijk` `(1, q, n, 2, 2[, 2])`: the interpolation stencil.
* `H_hess_at_pixel_deqnijk`, `L_laplace_at_pixel_eqnijk`: Hessian and
  Laplacian stencils, used by higher-order (Cahn–Hilliard type) problems.

### 4.3 The gradient operator $B$ is a convolution

The discrete gradient at quadrature point $q$ of pixel $\mathbf x$ is

$$
(\nabla u)_{i j}(\mathbf x, q) = \sum_{n}\sum_{\mathbf o \in\{0,1\}^d} B_{j q n \mathbf o}\; u_{i n}(\mathbf x + \mathbf o) .
$$

The same stencil is used in every pixel, so $B$ is a **periodic convolution**.
It is block-circulant and therefore block-diagonalized by the FFT. That is the
key structural fact behind the Green preconditioner (Section 5.4).
`Discretization` wraps the stencils in muGrid `GenericLinearOperator` objects:

| operator | attribute | apply (nodes → quad points) | transpose (quad points → nodes) |
|---|---|---|---|
| gradient $B$ | `gradient_op` | `apply_gradient_operator_mugrid(u_inxyz, grad_u_ijqxyz)` | `apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz, div_u_fnxyz, apply_weights=True)` = $B^TW\sigma$ |
| sym. gradient | — | `apply_gradient_operator_symmetrized_mugrid` | (same transpose) |
| interpolation $N$ | `interpolation_op` | `apply_N_operator_mugrid(nodal_field_inxyz, quad_field_ijqnxyz)` | `apply_N_transposed_operator_mugrid(quad_field_ijqxyz, nodal_field_inxyz, apply_weights=True)` = $N^TWg$ |
| Hessian | `hessian_op` | `apply_hessian_operator_to_{scalar,vector}_field_mugrid` | `apply_hessian_operator_transposed_to_*` |

**$B^T W$ is the discrete divergence.** For a stress field $\sigma$ at the
quadrature points,

$$
(B^T W\sigma)_{i}^{a} = \sum_{q,\text{pixels}} w_q\,\frac{\partial N^a}{\partial x_j}(x_q)\,\sigma_{ij}(x_q) \approx \int_Y \nabla N^a\cdot \sigma\,dx ,
$$

which is the internal nodal force vector. With `apply_weights=False` it is the
pure transpose $B^T$.

### 4.4 Interpolation $N$: nodal vs. quadrature fields

There are two kinds of fields:

* **Nodal fields** are the unknowns (temperature, displacement), the adjoint
  field, the phase field $\rho$ and right-hand sides. They live on
  `sub_pt='nodal_points'`.
* **Quadrature fields** are gradients, strains, stresses, material data,
  $\det F$, and $\rho$ evaluated at quadrature points. They live on
  `sub_pt='quad_points'`.

$N$ maps nodal values to quadrature-point values,
$u(x_q) = \sum_a N^a(x_q)u^a$. Its weighted transpose $N^TW$ pulls a
quadrature-point density back to the nodes. Topology optimization uses this to
turn $\partial f/\partial\rho(x_q)$ into a nodal sensitivity (chain rule
through the interpolation). `evaluate_field_at_quad_points` does the same job
as `apply_N_operator_mugrid`.

### 4.5 muGrid fields, `.s`/`.sg`, ghost layers

All fields are `muGrid.Field` objects from
`Discretization.field_collection` (real space) or `ffield_collection`
(Fourier space). Factories:
`get_unknown_size_field`, `get_gradient_size_field`,
`get_material_data_size_field_mugrid`, `get_scalar_field` (`[1, n, x, y]`),
`get_quad_field_scalar` (`[1, 1, q, x, y]`), `get_strain_sized_field`,
`get_displacement_gradient_sized_field`, and others.

* `field.s` is a NumPy view of the locally owned data (no ghosts).
  `field.sg` includes the ghost layers.
* **Fields are registered by name.** A second request with the same name
  returns the *same* field and its old content. muFFTTO relies on this for
  scratch fields (e.g. `'grad_field_temporary'`). It also means two callers
  that use the same name share memory, and a "new" field is zero only the
  first time it is created.
* **Ghost layers.** The stencils reach one pixel into the neighbour. The
  `muGrid.FFTEngine` is created with one ghost layer on each side, and
  `fft.communicate_ghosts(field)` fills it: an MPI halo exchange plus the
  periodic wrap-around. All `*_mugrid` operators call it themselves. The
  user-written `K_fun`/`M_fun` wrappers in the examples call it once more on
  their output, which is harmless.

### 4.6 MPI domain decomposition

`muGrid.FFTEngine(nb_domain_grid_pts, communicator, nb_ghosts_left, nb_ghosts_right)`
splits the grid into slabs or pencils across ranks.

* `nb_of_pixels_global` is the global grid size; `nb_of_pixels`
  (= `fft.nb_subdomain_grid_pts`) is the local one.
* `fft.subdomain_locations` gives the offset of the local subdomain.
  `fft.coords` and `fft.icoords` are the fractional and integer coordinates
  of the local pixels.
* Global reductions use `Discretization.mpi_reduction` (NuMPI `Reduction`:
  `.sum`, `.max`) or `communicator.sum(...)`. The CG solvers reduce every dot
  product, so all ranks follow the same control flow.
* FFTs (`fft.fft`, `fft.ifft`) are parallel and unnormalized. Multiply by
  `fft.normalisation` $=1/N_{\rm total}$ after a forward–inverse pair.

---

## 5. Assembling and solving the linear system

### 5.1 The discrete system

Insert the finite-element approximation into the weak form of Section 1.3.
Let $W$ be the diagonal matrix of quadrature weights and $\mathbb C$ the block
diagonal of material tensors at the quadrature points. Then

$$
\boxed{\;K\,\tilde u = f,\qquad K = B^T W \mathbb C B,\qquad f = -B^T W \mathbb C\,E\;}
$$

For small strain, replace $B$ by $\operatorname{sym}\circ B$. Since $\mathbb C$
has minor symmetry, $\mathbb C\,\operatorname{sym}(g) = \mathbb C g$ and
$B^T$ acting on a symmetric $\sigma$ equals $(\operatorname{sym}B)^T\sigma$.

| quantity | code |
|---|---|
| $E$ at quadrature points | `get_macro_gradient_field_mugrid(macro_gradient_ij, macro_gradient_field_ijqxyz)` |
| $f = -B^TW\mathbb C E$ | `get_rhs_mugrid(material_data_field_ijklqxyz, macro_gradient_field_ijqxyz, rhs_inxyz)` |
| $K x$ | `apply_system_matrix_mugrid(material_data_field, input_field_inxyz, output_field_inxyz, formulation=None)` |
| nonlinear residual $-B^TW\sigma(\nabla u)$ | `get_rhs_explicit_stress_mugrid(stress_function, gradient_field_ijqxyz, rhs_inxyz)` |
| $K x$ with a constitutive callable | `apply_system_matrix_mugrid_explicit_stress(constitutive, input_field_inxyz, output_field_inxyz)` |
| dense $K$ (debugging, tiny grids, 1 rank) | `get_system_matrix_mugrid(material_data_field, formulation)` |

### 5.2 Matrix-free operator

$K$ is never assembled. `apply_system_matrix_mugrid` performs three steps:

1. `g = B u` (or `sym(B u)`): gradient at the quadrature points, one stencil
   pass.
2. `g <- C : g`: a pointwise contraction (`apply_material_data_mugrid`).
3. `out = B^T W g`: the transposed stencil with weights.

The cost is $\mathcal O(N)$ per application and the memory is a few
quadrature fields. The material can be a full field or a single tensor
`[d,d]`/`[d,d,d,d]`. A single tensor is how the reference operator of the
Green preconditioner is built.

### 5.3 Preconditioned conjugate gradients

$K$ is symmetric positive (semi)definite, so the solver of choice is
**preconditioned CG** (Hestenes–Stiefel; Saad, Alg. 9.1):

```
r0 = b - A x0;  z0 = M^{-1} r0;  p0 = z0
alpha_k = (r_k, z_k) / (p_k, A p_k)
x_{k+1} = x_k + alpha_k p_k
r_{k+1} = r_k - alpha_k A p_k          # recursive residual, no extra A-apply
z_{k+1} = M^{-1} r_{k+1}
beta_k  = (r_{k+1}, z_{k+1}) / (r_k, z_k)
p_{k+1} = z_{k+1} + beta_k p_k
```

There are two implementations, and the examples use both:

* **`muFFTTO.solvers.conjugate_gradients_mugrid(comm, fc, hessp, b, x, P, tol=1e-6, rtol=False, maxiter=1000, callback=None, norm_metric=None)`**
  * `hessp(x, Ax)` and `P(r, z)` are in-place callables on muGrid fields.
  * **Stopping test.** By default it uses the *squared Euclidean* residual,
    $(r,r) < \texttt{tol}^2$, which is **absolute** unless `rtol=True`. With
    `rtol=True` the test becomes $(r,r) < \texttt{tol}^2\,(r_0,r_0)$. It is
    *not* the preconditioned norm $(r,z)$ that many FFT papers use. You can
    pass `norm_metric(r, Gr)` to stop on $(r, Gr)$ instead, for example
    $G=M^{-1}$.
  * `callback(it, x, r, p, z, stop_crit)` receives `.s` arrays. The examples
    use it to record $(r,r)$ and $(r,z)$.
  * It raises `RuntimeError` if $(p,Ap)\le 0$.
* **`muGrid.Solvers.conjugate_gradients(comm, fc, b, x, hessp, prec, rtol, atol, tol, maxiter, callback)`**
  (muGrid's own CG). It stops when $\lVert r\rVert \le \max(\texttt{rtol}\,\lVert b\rVert, \texttt{atol})$.
  `tol` is a deprecated alias for an *absolute* `atol`. The conductivity
  example passes `rtol=1e-6`; the small-strain elasticity example passes
  `tol=1e-6`, which is absolute.

`solvers.conjugate_gradients_mugrid_experimental` adds a-posteriori
energy-norm error estimates: a Hestenes–Stiefel lower bound with adaptive
delay (`findS`) and a Gauss–Radau upper bound. It is meant for research on
stopping criteria.

**Block CG.** `solvers.dr_pbcg_mugrid(comm, fc, hessp, b_list, x_list, P, tol, rtol)`
solves all $m$ load cases at once with a shared block Krylov space (DR-PBCG,
Meurant–Tichý). The residual block is kept as a thin QR factor and the method
stops on its Frobenius norm. `examples/homogenization/conductivity/*_block_CG.py`
use it for the $d$ conductivity load cases.

### 5.4 Green (FFT, reference-material) preconditioner

Pick a homogeneous **reference material** $\mathbb C^{\rm ref}$. The reference
stiffness $K^{\rm ref} = B^TW\mathbb C^{\rm ref}B$ uses the same stencil in
every pixel, so it is block-circulant. The DFT $\mathcal F$
block-diagonalizes it:

$$
K^{\rm ref} = \mathcal F^{-1}\,\hat K^{\rm ref}(\xi)\,\mathcal F ,
$$

where $\hat K^{\rm ref}(\xi)$ is a small dense $(f n)\times(f n)$ matrix for
every wave vector $\xi$: $f$ components times $n$ nodes per pixel, e.g.
$2\times 2$ for 2D elasticity with linear elements. The preconditioner is

$$
M^{-1} = \mathcal F^{-1}\,\big[\hat K^{\rm ref}(\xi)\big]^{-1}\,\mathcal F ,
$$

whose cost is two FFTs and a batch of tiny matrix-vector products. This is
the *discretization-consistent* Green operator of Ladecký et al. (2023) and
Leute et al. (2022). It is not the continuous Moulinec–Suquet $\Gamma^0$
operator, and the consistency is what removes ringing artifacts. Their
analysis shows that the spectrum of $M^{-1}K$ is bounded by the extreme
eigenvalues of $\mathbb C(x)$ relative to $\mathbb C^{\rm ref}$. The number of
CG iterations is therefore independent of the grid size and depends only on
the phase contrast.

The code:

* `get_preconditioner_Green_mugrid(reference_material_data_ijkl, formulation=None, operator=None)`
  builds $\hat K^{\rm ref}$ column by column. It applies $K^{\rm ref}$ to a
  unit impulse at DOF `(f, n)` of the origin pixel and FFTs the response.
  It then inverts each block (`np.linalg.inv`, batched) and returns a complex
  field `'Greens_diagonal_fast'` of shape `[f, n, f, n, ξ...]`. A custom
  translation-invariant `operator` can replace $K^{\rm ref}$.
* `apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks, input_nodal_field_fnxyz, output_nodal_field_fnxyz)`
  does FFT → `einsum('cdab...,cd...->ab...')` → iFFT → `*= fft.normalisation`.
* **Zero frequency.** $\hat K^{\rm ref}(0)$ is singular because constants
  form the kernel. With one node per pixel its block is left un-inverted, and
  it is numerically zero because the stencil sums to zero. The
  preconditioner therefore **annihilates the mean**. The right-hand side has
  zero mean too, since $\sum_a\nabla N^a = 0$. Starting from $x_0=0$, every
  CG iterate stays in the zero-mean subspace, which fixes the free
  translation. With several nodes per pixel (Q9) the zero block is singular
  but not zero, and the code uses its pseudo-inverse.
* **Choice of $\mathbb C^{\rm ref}$.** The examples use the matrix or solid
  phase (e.g. `elastic_C_0` or `conductivity_C_0`). The finite-strain solver
  defaults to $\mathbb I^s$. A reference far from the actual phases costs
  iterations but never changes the converged solution.

### 5.5 Jacobi and Green–Jacobi preconditioners

* `get_preconditioner_Jacobi_mugrid(material_data_field_ijklqxyz, formulation=None)`
  returns the field `'jacobi_diagonal_inxyz'` holding $D^{-1/2}$, where
  $D = \operatorname{diag}(K)$. It never assembles $K$. Instead it applies
  $K$ to **Dirac combs**: ones on every second pixel in each direction, for
  one component at a time. Linear-element stencils reach only one pixel
  away, so the response at the comb points equals $K_{ii}$. This takes
  $f\cdot 2^d$ operator applications, needs even grid sizes, and assumes one
  node per pixel. Zero diagonal entries (pure void) are replaced by
  `zero_threshold`.
* Jacobi alone: `Px = D^{-1/2} * D^{-1/2} * x` (see `M_fun_Jacobi` in the
  TO examples).
* **Green–Jacobi**: $M^{-1} = D^{-1/2}\,G\,D^{-1/2}$, with $G$ the Green
  preconditioner, which keeps it symmetric. The diagonal scaling handles
  local stiffness variations, for example SIMP void regions with stiffness
  $10^{-5}$, that a single reference material cannot capture. It is the
  default in the topology-optimization examples
  (`preconditioner_type = "Green_Jacobi"`) and an option in
  `solve_finite_strain_newton_cg`. `apply_preconditioner_Green_Jacobi_full`
  is a legacy NumPy variant with per-pixel blocks.

### 5.6 Pipeline sketch

The sketch below uses only functions whose signatures were checked against
the code:

```python
import numpy as np
from muFFTTO import domain, material_models, solvers, microstructure_library

# 1. cell + discretization
cell = domain.PeriodicUnitCell(domain_size=[1, 1], problem_type='elasticity')
disc = domain.Discretization(cell=cell, nb_of_pixels_global=(64, 64),
                             discretization_type='finite_element',
                             element_type='linear_triangles')
dim = disc.domain_dimension

# 2. material field C(x_q), shape [d,d,d,d,q,x,y]
K, G = material_models.get_bulk_and_shear_modulus(E=1.0, poisson=0.2)
C_1 = material_models.get_elastic_material_tensor(dim=dim, K=K, mu=G)
geom = microstructure_library.get_geometry(nb_voxels=disc.nb_of_pixels,
                                           microstructure_name='square_inclusion',
                                           coordinates=disc.fft.coords)
C = disc.get_material_data_size_field_mugrid(name='C')
C.s[...] = C_1[..., np.newaxis, np.newaxis, np.newaxis]
C.s[..., geom > 0] *= 100.0                       # stiff matrix, soft inclusion

# 3. operator K and Green preconditioner M^{-1}
def K_fun(x, Ax):
    disc.apply_system_matrix_mugrid(material_data_field=C, input_field_inxyz=x,
                                    output_field_inxyz=Ax, formulation='small_strain')

G_hat = disc.get_preconditioner_Green_mugrid(reference_material_data_ijkl=C_1)
def M_fun(r, z):
    disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=G_hat,
                                     input_nodal_field_fnxyz=r,
                                     output_nodal_field_fnxyz=z)

# 4. one cell problem per unit load case -> columns of C_eff
E_field = disc.get_gradient_size_field(name='E')
rhs = disc.get_unknown_size_field(name='rhs')
u = disc.get_unknown_size_field(name='u')
C_eff = np.zeros(4 * [dim])
for k in range(dim):
    for l in range(dim):
        E = np.zeros([dim, dim]); E[k, l] = 1.0
        disc.get_macro_gradient_field_mugrid(macro_gradient_ij=E,
                                             macro_gradient_field_ijqxyz=E_field)
        disc.get_rhs_mugrid(material_data_field_ijklqxyz=C,
                            macro_gradient_field_ijqxyz=E_field, rhs_inxyz=rhs)
        u.s.fill(0)
        solvers.conjugate_gradients_mugrid(comm=disc.communicator, fc=disc.field_collection,
                                           hessp=K_fun, b=rhs, x=u, P=M_fun,
                                           tol=1e-6, rtol=True, maxiter=1000)
        C_eff[k, l] = disc.get_homogenized_stress_mugrid(
            material_data_field_ijklqxyz=C, displacement_field_inxyz=u,
            macro_gradient_field_ijqxyz=E_field, formulation='small_strain')

print(material_models.compute_Voigt_notation_4order(C_eff))
```

For conductivity, use `problem_type='conductivity'`, a `[d, d]` tensor, a
macro gradient `E` of shape `[d]` (a single loop over `k`), and
`formulation=None`. `get_homogenized_stress_mugrid` then returns a `[1, d]`
flux.

---

## 6. Effective (homogenized) properties

### 6.1 Stress / flux averaging

`Discretization.get_homogenized_stress_mugrid(material_data_field_ijklqxyz, displacement_field_inxyz, macro_gradient_field_ijqxyz, formulation=None)`
computes

$$
\Sigma = \frac{1}{\lvert Y\rvert}\sum_{\text{pixels}}\sum_q w_q\,\mathbb C(x_q):\big(E + B\tilde u\big)(x_q),
$$

reduced over all MPI ranks. It returns `[d, d]` for elasticity and `[1, d]`
for conductivity. Remember `formulation='small_strain'` for elasticity. The
fluctuation term matters, so the gradient must be the same one used in the
solve. Related helpers:

* `get_stress_field_mugrid` and `get_flux_field_mugrid` return the local
  fields $\sigma(x_q)$.
* `integrate_over_cell_mugrid(stress_field)` gives $\sum w_q\sigma$ without
  dividing by $\lvert Y\rvert$. Note that it scales its argument in place.
* `get_homogenized_stress_mugrid_explicit_stress(constitutive, ...)` handles
  nonlinear constitutive callables.

### 6.2 Energy

`get_homogenized_energy_mugrid(...)` returns
$\langle (E+\nabla\tilde u) : \mathbb C : (E+\nabla\tilde u)\rangle = E:\mathbb C^{\rm eff}:E$.
This value is *quadratic* in $\tilde u$, while the stress average is linear.
The two agree, $E:\Sigma$, only when the discrete equilibrium holds, i.e. at
the exact discrete solution or at a CG iterate started from $x_0 = 0$.
Comparing them is a cheap sanity check on solver accuracy.

### 6.3 Assembling $\mathbb C^{\rm eff}$ from load cases

For each unit load $E = e_k\otimes e_l$ the examples store
`homogenized_C_ijkl[k, l] = Σ`. With the contraction convention of
Section 2.3 this array is
$\Sigma_{ij} = C^{\rm eff}_{ijlk}$, which by major and minor symmetry equals
$C^{\rm eff}_{klij} = C^{\rm eff}_{ijkl}$. The same holds for conductivity:
`homogenized_A_ij[k, :] = Σ` stores $A^{\rm eff}_{ik}$ in row $k$, i.e. the
transpose, which equals $A^{\rm eff}$ because it is symmetric.

The non-symmetric load case $e_0\otimes e_1$ needs no symmetrization, because
$\mathbb C:E = \mathbb C:\operatorname{sym}E$. A $d=2$ elasticity cell
therefore takes 4 solves, or 3 if you exploit symmetry; the TO examples use
the 3 loads $e_0\otimes e_0$, $e_1\otimes e_1$ and
$\tfrac12(e_0\otimes e_1+e_1\otimes e_0)$.

### 6.4 Voigt notation

`material_models.compute_Voigt_notation_4order(C_ijkl)` maps the index pairs

* 2D: $(00, 11, 01)$
* 3D: $(00, 11, 22, 12, 02, 01)$

with **no** factors of 2 or $\sqrt2$. This is the standard Voigt stiffness
matrix, which acts on engineering shear strains $\gamma = 2\varepsilon_{ij}$;
it is not Mandel notation. `compute_Voigt_notation_2order` and
`compute_Voigt_notation` handle second-order tensors.

---

## 7. Topology optimization

The goal is to find a periodic microstructure whose homogenized tensor matches
a target, for example an auxetic material with $\nu^{\rm eff}<0$. Material
lives in `muFFTTO/topology_optimization.py` (elasticity) and
`muFFTTO/topology_optimization_conductivity.py` (conductivity).

### 7.1 Design variable and material interpolation

The design variable is a **nodal phase field** $\rho\in[0,1]$
(`phase_field_1nxyz`, shape `[1, n, x, y]`), where 0 is void and 1 is solid.
It is interpolated to the quadrature points with $N$
(`apply_N_operator_mugrid`). The material follows **SIMP**:

$$
\mathbb C(\rho) = (\mathbb C_1 - \mathbb C_0)\,\rho^p + \mathbb C_0,\qquad
\frac{\partial\mathbb C}{\partial\rho} = p\,\rho^{p-1}(\mathbb C_1-\mathbb C_0),
$$

with `material_interpolation_simp` and `dmaterial_interpolation_simp`. The
examples write the same formula inline. They use $p=2$ and a void stiffness
$\mathbb C_0 = 10^{-s}\mathbb C_1$ (`soft_phase_exponent` $s$ = 5 in 2D
elasticity, 4 in conductivity, 3 in 3D), so that $K$ stays positive
definite. Naming caveat: in the module docstrings $\mathbb C_1$ is the solid,
while in the examples the solid is called `elastic_C_0` and the void
`elastic_C_void`.

### 7.2 Objective

For load cases $E^{(m)}$, $m=1..M$, with targets
$\Sigma_t^{(m)} = \mathbb C^{\rm target}:E^{(m)}$, the objective is

$$
f(\rho) = \frac{w}{M}\sum_{m} \underbrace{\frac{\lVert\Sigma_t^{(m)} - \Sigma_h^{(m)}(\rho)\rVert^2}{\lVert\Sigma_t^{(m)}\rVert^2}}_{f_\sigma^{(m)}}
\;+\; \eta\int_Y\lvert\nabla\rho\rvert^2dx \;+\; \frac{c_{dw}}{\eta}\int_Y \rho^2(1-\rho)^2dx .
$$

* **Stress-equivalence term**:
  `compute_stress_equivalence_potential(actual_stress_ij, target_stress_ij)`,
  equivalently `objective_function_stress_equivalence`. An energy variant,
  `compute_elastic_energy_equivalence_potential`, contracts with a "left"
  strain $E_L$ instead.
* **Conductivity**:
  `topology_optimization_conductivity.compute_flux_equivalence_potential`,
  with fluxes $q_t = A^{\rm target}E$.
* **Phase-field regularization** (Modica–Mortola / Cahn–Hilliard type):
  `objective_function_phase_field(discretization, phase_field_1nxyz, eta, double_well_depth)`.
  * The gradient term is `compute_gradient_of_phase_field_potential`,
    $= \rho^TB^TWB\rho$.
  * The double well uses **nodal (lumped) quadrature**
    (`compute_double_well_potential_nodal`,
    $\lvert Y\rvert/N\sum_a\rho_a^2(1-\rho_a)^2$). An exact P1 variant,
    `compute_double_well_potential_analytical`, sits behind a hard-coded
    switch.
  * $\eta$ sets the diffuse-interface width (the examples use one pixel
    size), and the double well pushes $\rho$ toward 0 or 1.

The examples use $w=5$ for elasticity. They also add the adjoint term
$\lambda^Tg$ (`adjoint_energies`) to $f$. It vanishes at equilibrium and only
reports how accurately the state was solved.

### 7.3 Adjoint sensitivity

The state equation is $g(\tilde u,\rho) = B^TW\,\mathbb C(\rho):(E + B\tilde u) = 0$.
With the Lagrangian $\mathcal L = f + \lambda^Tg$, requiring stationarity in
$\tilde u$ gives the **adjoint problem**

$$
K(\rho)\,\lambda = -\frac{\partial f}{\partial\tilde u},\qquad
-\frac{\partial f_\sigma}{\partial \tilde u} = \frac{2}{\lvert Y\rvert\,\lVert\Sigma_t\rVert^2}\,B^TW\,\mathbb C(\rho):(\Sigma_t-\Sigma_h).
$$

$K$ is symmetric, so this is the same operator, preconditioner and PCG as the
state problem. Each load case needs one extra solve, and the previous
$\lambda$ is a warm start. The total derivative is

$$
\frac{df}{d\rho} = \underbrace{\frac{\partial f}{\partial\rho}\Big|_{\tilde u}}_{\text{explicit}}
+ \underbrace{N^TW\Big[(B\lambda) : \frac{\partial\mathbb C}{\partial\rho} : (E + B\tilde u)\Big]}_{\lambda^T\,\partial g/\partial\rho},
$$

with the explicit part

$$
\frac{\partial f_\sigma}{\partial\rho_a}\Big|_{\tilde u} = -\frac{2}{\lvert Y\rvert\lVert\Sigma_t\rVert^2}\int_Y(\Sigma_t-\Sigma_h):\frac{\partial\mathbb C}{\partial\rho}:(E+\nabla^s\tilde u)\,N_a\,dx .
$$

| step | elasticity | conductivity |
|---|---|---|
| explicit + adjoint solve + adjoint term, per load case | `sensitivity_stress_and_adjoint_FE_NEW(...)` | `sensitivity_flux_and_adjoint(...)` |
| explicit part | `partial_derivative_of_objective_function_stress_equivalence_wrt_phase_field_FE` | `partial_derivative_of_objective_function_flux_equivalence_wrt_phase_field` |
| $\lambda^T\partial g/\partial\rho$ | `partial_derivative_of_adjoint_potential_wrt_phase_field_FE` | `partial_derivative_of_adjoint_potential_wrt_phase_field` |
| regularization: $2\eta B^TWB\rho + \tfrac{c_{dw}}{\eta}\tfrac{\lvert Y\rvert}{N}(2\rho-6\rho^2+4\rho^3)$ | `sensitivity_phase_field_term_FE_NEW` | (re-exported from the elasticity module) |

The `sensitivity_*` functions take `preconditioner_fun` and
`system_matrix_fun`, the same `M_fun`/`K_fun` used for the state, plus
`cg_tol` and `r_tol` keyword arguments. Functions suffixed `_pixel`,
`_FE_weights` or `_FE_testing` are legacy.

### 7.4 Optimizer

The examples (`examples/topology_optimization/example_2D_elasticity_TO.py`,
`example_2D_conductivity_TO.py` and `example_3D_elasticity_TO.py`) minimize
$f$ with **`NuMPI.Optimization.l_bfgs_bounded`**. This is an MPI-parallel,
bound-constrained L-BFGS with Armijo backtracking, run with box bounds
$0\le\rho\le1$ and called with `jac=True`, so the objective returns
`(f, df/drho)`. It is similar in spirit to scipy's L-BFGS-B (Byrd et al.).
The objective function, per call:

1. Interpolate $\rho\to\rho(x_q)$ and build $\mathbb C(\rho)$.
2. Compute the phase-field energy and its gradient.
3. Build the Jacobi diagonal for the current $\mathbb C(\rho)$.
4. For each load case: state solve, $\Sigma_h$, $f_\sigma$, then the adjoint
   solve and the sensitivity.
5. Sum everything up.

An `Optimization.LinearConstraint` object is created in the elasticity
example but never passed to the optimizer. A first-order alternative,
`solvers.adam` with `update_parameters_with_adam` (Kingma–Ba), exists in
`muFFTTO/solvers.py`; the listed examples do not use it.

After optimization the examples re-solve the cell problems with the final
$\rho$ and print the obtained and target $\mathbb C^{\rm eff}$ in Voigt
notation.

---

## 8. Grid adaptation (deformed grids)

### 8.1 Why

On a regular pixel grid a curved interface becomes a staircase. Quadrature
points on the wrong side of the true interface cause local stress and flux
errors and slow, oscillating convergence of $\mathbb C^{\rm eff}$.
*Conformal* grid adaptation (Zecevic, Lebensohn & Capolungo) moves the grid
nodes so that element edges follow the interface. It keeps the topology,
periodicity and number of nodes of the regular grid unchanged, so the FFT
machinery still applies.

### 8.2 Pull-back to the regular grid

Let $X$ be the regular reference grid and $x=\varphi(X) = X + w(X)$ the
deformed grid, where $w$ is a periodic nodal displacement of the grid nodes.
Its gradient at the quadrature points,

$$
F = I + \nabla_X w ,
$$

is computed in the examples (`apply_gradient_operator_mugrid` applied to the grid displacement, then $+I$) together with $\det F$ and $F^{-1}$
(`np.linalg.det` and `np.linalg.pinv` per quadrature point). Standard
change-of-variables rules,
$\nabla_x v = \nabla_X v\cdot F^{-1}$ and $dx = \det F\,dX$, turn the weak
form on the deformed grid into one on the regular grid:

$$
\int_{Y}\big(\nabla_X v\,F^{-1}\big) : \mathbb C : \operatorname{sym}\big((E+\nabla_X\tilde u)F^{-1}\big)\,\det F\,dX = 0 .
$$

So $K u = B^TW\big[\det F\;\big(\mathbb C : \operatorname{sym}(Bu\,F^{-1})\big)\,F^{-T}\big]$.
It uses the same regular-grid operators $B$, $B^TW$ and a spatially varying
"effective material". That is why the Green preconditioner of the undeformed
problem still works, with the distortion of $F$ entering the spectral bounds
like extra material contrast.

| code (`muFFTTO/domain.py`) | formula |
|---|---|
| `apply_system_matrix_mugrid_deformed_grid(material_data_field, input_field_inxyz, output_field_inxyz, det_of_deformation_gradient, inv_of_deformation_gradient, formulation)` | $B^TW[\det F\,(\mathbb C:\operatorname{sym}(Bu\,F^{-1}))F^{-T}]$ |
| `get_rhs_mugrid_deformed_grid(...)` | $-B^TW[\det F\,(\mathbb C:(E F^{-1}))F^{-T}]$ |
| `get_homogenized_stress_mugrid_deformed_grid(material_data_field_ijklqxyz, temperature_field_inxyz, macro_gradient_field_ijqxyz, det_of_deformation_gradient, inv_of_deformation_gradient, formulation)` | $\lvert Y\rvert^{-1}\sum w_q\det F\,\mathbb C:((E+B\tilde u)F^{-1})$ |

**Interpretation of $E$.** The code multiplies $E$ by $F^{-1}$ as well. In
other words it prescribes $u = E\cdot X + \tilde u(X)$, affine in the
*reference* coordinates. Because $x - X = w$ is periodic,
$E\cdot x = E\cdot X + E\cdot w$ differs only by a periodic function, which is
absorbed into the fluctuation. The physical average gradient is still
$\langle\nabla_xu\rangle_x = E$. Likewise
$\lvert\varphi(Y)\rvert = \lvert Y\rvert$, so dividing by the reference
volume is correct.

Two details are easy to miss:

* The homogenized-stress routine only symmetrizes for
  `formulation='small_strain'`, and the grid-adaptation elasticity example
  does not pass it. For a minor-symmetric $\mathbb C$ the result is the same.
* The right-hand side is never symmetrized, for the same reason.

### 8.3 Spring-relaxation methods

All grid-adaptation methods produce nodal coordinates. Their difference from
`fft.coords` is the grid displacement $w$ used above.

* `muFFTTO/grid_adaptation_methods_Zecevic.py` (periodic, one circular
  inclusion; layout `P[xy, i, j]`). `adapt_grid_to_circle(ref_grid_coords_ixyz, center, radius)` runs five steps:
  1. `cell_labels`: inside/outside per cell, decided at the cell centre.
  2. `interface_node_mask`: nodes whose four cells disagree.
  3. `project_points_to_circle`: radial snap onto the circle.
  4. `manhattan_distance_to_interface` and `stiffness_from_distance`: nodal
     spring stiffness $k=\max(k_0/(d+a)^b, k_{\min})$.
  5. `spring_relax_weighted`: interface nodes stay fixed, and the others
     relax by weighted Laplacian smoothing, $x_a \leftarrow \sum_b k_b x_b/\sum_b k_b$.

  Cell phase labels are *not* recomputed after the move; each cell keeps its
  reference phase.
* `muFFTTO/grid_adaptation_arbitrary.py`: the image-driven version for
  arbitrary multi-phase microstructures, `run_grid_adaptation_workflow(...)`.
  It works as follows:
  * Otsu thresholding and edge detection on a fine image
    (`muFFTTO/otsu.py`).
  * Majority-vote coarse labels.
  * Snapping interface nodes to the nearest fine edge point.
  * Periodic distance-weighted Jacobi spring relaxation
    (`spring_relax_projected_grid`).

  Its arrays are **row-major in y**, `P[xy, j, i]`, which is the transpose of
  the rest of muFFTTO.
* `muFFTTO/analytical_grid_adaptation.py`: a non-periodic reference
  implementation on $[-L,L]^2$ with a fixed outer boundary, layout
  `P[i, j, xy]`.

Examples: `examples/grid_adaptation/example_2D_homogenization_{conductivity,elasticity}_transformed_grid.py`.
These use an analytical sinusoidal grid displacement to demonstrate the
pull-back.

---

## 9. Notation and array conventions

### 9.1 Index letters in variable names

Variable names carry their array layout as a suffix. For example,
`u_inxyz` has `.s` shape `[i, n, x, y, z]`.

| letter | meaning | size |
|---|---|---|
| `i, j, k, l` | tensor components (also `f`, `d`, `a`, `b`, `c` in some places) | $d$, or 1 for a scalar unknown |
| `f` | components of the unknown | `unknown_shape[0]` (1 or $d$) |
| `d`, `e` | derivative direction (in stencils) | $d$ |
| `n` | nodal sub-point within a pixel | `nb_nodes_per_pixel` |
| `q` | quadrature sub-point within a pixel (real-space fields) | `nb_quad_points_per_pixel` |
| `x, y, z` | pixel indices (local MPI subdomain) | `nb_of_pixels` |
| `q, k, s` **after** the components in Fourier fields (e.g. `_fnfnqks`) | wave-vector indices, **not** quadrature points | Fourier grid |
| `i, j, k` **after** `n` in stencils (`_dqnijk`) | pixel offsets 0/1 per direction | 2 each |
| `1` | a size-1 axis (scalar) | 1 |

### 9.2 Field shapes (`.s` view)

| field | shape | factory |
|---|---|---|
| unknown, rhs, adjoint | `[f, n, x, y(, z)]` | `get_unknown_size_field` |
| phase field / nodal scalar | `[1, n, x, y(, z)]` | `get_scalar_field` |
| gradient / strain / stress / flux | `[f, d, q, x, y(, z)]` | `get_gradient_size_field`, `get_strain_sized_field`, `get_stress_sized_field` |
| scalar at quadrature points ($\lambda$, $\mu$, $\det F$, $\rho(x_q)$) | `[1, 1, q, x, y(, z)]` | `get_quad_field_scalar` |
| conductivity tensor | `[d, d, q, x, y(, z)]` | `get_material_data_size_field_mugrid` |
| stiffness / tangent | `[d, d, d, d, q, x, y(, z)]` | `get_material_data_size_field_mugrid` |
| Green preconditioner | `[f, n, f, n, ξ...]` (complex) | `get_preconditioner_Green_mugrid` |
| Jacobi scaling $D^{-1/2}$ | `[f, n, x, y(, z)]` | `get_preconditioner_Jacobi_mugrid` |
| gradient stencil `B_grad_at_pixel_dqnijk` | `[d, q, n, 2, 2(, 2)]` | element data |

### 9.3 Glossary

| term | meaning |
|---|---|
| cell problem | periodic boundary value problem on $Y$ for the fluctuation $\tilde u$ under a macro gradient $E$ |
| fluctuation $\tilde u$ | periodic part of the field; the solver's unknown |
| macro gradient $E$ | prescribed average gradient (temperature gradient, small strain, or displacement gradient $H$) |
| effective / homogenized tensor | $\mathbb C^{\rm eff}$ with $\langle\sigma\rangle = \mathbb C^{\rm eff}:E$ |
| $B$, $B^TW$ | discrete gradient (nodes → quadrature points) and weighted divergence (quadrature points → nodes) |
| $N$, $N^TW$ | interpolation (nodes → quadrature points) and its weighted transpose |
| $K = B^TW\mathbb CB$ | system matrix (stiffness/conductance), applied matrix-free |
| reference material $\mathbb C^{\rm ref}$ | homogeneous material defining the Green preconditioner |
| Green preconditioner | $\mathcal F^{-1}[\hat K^{\rm ref}(\xi)]^{-1}\mathcal F$ |
| Jacobi | diagonal scaling $\operatorname{diag}(K)^{-1}$ (stored as $D^{-1/2}$) |
| ghost layer | halo of one pixel, filled by `communicate_ghosts` |
| `.s` / `.sg` | muGrid views without / with ghost layers |
| SIMP | $\mathbb C(\rho) = (\mathbb C_1-\mathbb C_0)\rho^p+\mathbb C_0$ |
| double well | $\rho^2(1-\rho)^2$, penalizes intermediate densities |
| $\eta$ | phase-field interface width |
| adjoint field $\lambda$ | solution of $K\lambda=-\partial f/\partial\tilde u$ |
| $F$, $\det F$ (grid adaptation) | gradient of the reference → deformed grid map |
| $F$, $P$ (finite strain) | deformation gradient and first Piola–Kirchhoff stress |

---

## References

* H. Moulinec, P. Suquet. *A numerical method for computing the overall
  response of nonlinear composites with complex microstructure.* Computer
  Methods in Applied Mechanics and Engineering 157 (1998) 69–94.
* J. Zeman, T. W. J. de Geus, J. Vondřejc, R. H. J. Peerlings,
  M. G. D. Geers. *A finite element perspective on nonlinear FFT-based
  micromechanical simulations.* International Journal for Numerical Methods
  in Engineering 111 (2017) 903–926.
* R. J. Leute, M. Ladecký, A. Falsafi, I. Jödicke, I. Pultarová, J. Zeman,
  T. Junge, L. Pastewka. *Elimination of ringing artifacts by finite-element
  projection in FFT-based homogenization.* Journal of Computational Physics
  453 (2022) 110931.
* M. Ladecký, R. J. Leute, A. Falsafi, I. Pultarová, L. Pastewka, T. Junge,
  J. Zeman. *An optimal preconditioned FFT-accelerated finite element solver
  for homogenization.* Applied Mathematics and Computation 446 (2023) 127835.
* M. R. Hestenes, E. Stiefel. *Methods of conjugate gradients for solving
  linear systems.* Journal of Research of the National Bureau of Standards 49
  (1952) 409–436.
* Y. Saad. *Iterative Methods for Sparse Linear Systems*, 2nd ed. SIAM, 2003.
* G. Meurant, J. Papež, P. Tichý. *Accurate error estimation in CG.*
  Numerical Algorithms 88 (2021). (Basis of `findS` and
  `conjugate_gradients_mugrid_experimental`.)
* G. Meurant, P. Tichý. Block CG / DR-PBCG. This is the Algorithm 5 cited in
  `solvers.dr_pbcg_mugrid`; see that docstring for the reference used.
* M. P. Bendsøe, O. Sigmund. *Topology Optimization: Theory, Methods and
  Applications.* Springer, 2003.
* R. H. Byrd, P. Lu, J. Nocedal, C. Zhu. *A limited memory algorithm for bound
  constrained optimization.* SIAM Journal on Scientific Computing 16 (1995)
  1190–1208.
* D. P. Kingma, J. Ba. *Adam: A method for stochastic optimization.* ICLR
  2015. (`solvers.adam`)
* M. Zecevic, R. A. Lebensohn, L. Capolungo. *Achieving geometric accuracy in
  FFT-based micromechanical models using conformal grid.* Mechanics of
  Materials 212 (2026) 105512, doi:10.1016/j.mechmat.2025.105512.
* N. Otsu. *A threshold selection method from gray-level histograms.* IEEE
  Transactions on Systems, Man, and Cybernetics 9 (1979) 62–66. (`muFFTTO/otsu.py`)
