# Phase-field fracture with third-medium contact

Example: [`examples/phase_field_fracture/example_2D_phase_field_fracture_contact.py`](../../examples/phase_field_fracture/example_2D_phase_field_fracture_contact.py)
Plotting: [`plot_phase_field_fracture_contact_output.py`](../../examples/phase_field_fracture/plot_phase_field_fracture_contact_output.py)

A periodic solid cell contains an S-shaped cavity between two interlocking
teeth (geometry `contact_fracture_s_gap`). The cell is sheared. The narrow
channel between the teeth closes, the teeth come into contact, and the contact
load concentrates the stresses until the solid cracks. Contact is modelled
with the **third-medium method**: the cavity is filled with a very soft
material that transmits pressure once it is crushed. Fracture is modelled with
the **AT2 phase field**, combining
[Phase-field fracture](phase_field_fracture.md) with
[Internal contact](internal_contact.md) at finite strain.

![geometry](../../examples/phase_field_fracture/figures/contact_fracture_geometry.png)

## 1. Geometry

`geometry.get('contact_fracture_s_gap', coords, **parameters)` returns 1 in the
solid and 0 in the cavity. The cavity is the union of three parts:

| Part | Definition (fractional coordinates) | Default |
|---|---|---|
| channel | $[c_x, c_x + s) \times [c_y, c_y + g)$ | $c = (0.42, 0.57)$, $g = 0.05$, $s = 0.12$ |
| upper lobe | quarter ellipse, centre $(c_x + s, c_y)$, radii $(r_{1x}, r_{1y})$, opening up-right | $(0.26, 0.28)$ |
| lower lobe | quarter ellipse, centre $(c_x, y_b)$, radii $(r_{2x}, c_y + g - y_b)$, opening up-left | $r_{2x} = 0.31$, $y_b = 0.18$ |

The channel lies between the top face of the right tooth (corner $c$) and the
bottom face of the upper-left tooth (corner $c + (s, g)$). The defaults were
fitted to a hand sketch (82 % overlap) and rounded to $0.01$.

## 2. Model

### Kinematics and load

The unknown is the periodic displacement fluctuation $\tilde u$:

$$
F = I + \lambda H + \nabla \tilde u, \qquad J = \det F, \qquad
H = \begin{pmatrix} 0 & 0 \\ 0.3 & 0 \end{pmatrix}, \quad \lambda: 0 \to 1 .
$$

This is simple shear $\bar F_{10} = 0.3\lambda$. With $H_{10} > 0$ the right
part of the cell moves up relative to the left part, which closes the channel.

### Energy

$$
\Pi(\tilde u, d) =
\int_\Omega g_s(d)\, W_{\mathrm{iso}}(F) + g_v(d, F)\, W_{\mathrm{vol}}(F) \, dx
+ \frac{k_r}{2} \int_\Omega \mathbb{H}\tilde u \mathbin{\vdots} \mathbb{H}\tilde u
  - \frac{1}{d_{\mathrm{dim}}}\, \mathbb{L}\tilde u \cdot \mathbb{L}\tilde u \, dx
+ \frac{G_c}{2} \int_\Omega \frac{d^2}{\ell} + \ell\, |\nabla d|^2 \, dx .
$$

**Elastic energy: neo-Hookean with a volumetric–isochoric split.** In plane
strain, with the out-of-plane stretch equal to 1:

$$
W_{\mathrm{iso}} = \frac{\mu}{2}\left( J^{-2/3} (I_1 + 1) - 3 \right), \qquad
W_{\mathrm{vol}} = \frac{\kappa}{2} (\ln J)^2, \qquad I_1 = F : F .
$$

$I_1 + 1$ is the first invariant of the 3D right Cauchy–Green tensor, and
$\kappa = \lambda_L + \tfrac{2}{3}\mu$ is the bulk modulus ($\lambda_L$ the first
Lamé parameter). Both parts are non-negative and vanish at $F = I$.

**Degradation**, with $g(d) = (1-d)^2 + k$:

$$
g_s = \begin{cases} g(d) & \text{solid} \\ 1 & \text{third medium} \end{cases}
\qquad
g_v = \begin{cases} g_s & J > 1 \text{ (expansion)} \\ 1 & J \le 1 \text{ (compression)} \end{cases}
$$

Shape change and expansion are degraded, but compression is not. A crack can
therefore open and slide, but its faces cannot interpenetrate, and the crushed
third medium keeps its contact stiffness. This is the finite-strain analogue of
the Amor split of the small-strain example.

**Third medium.** The same law with $\mu$ and $\kappa$ scaled by $k_v = 10^{-5}$,
never degraded. As the channel closes, $J \to 0^+$ in the medium and
$W_{\mathrm{vol}} = \tfrac{\kappa}{2} (\ln J)^2 \to \infty$. This barrier is the
contact force; no contact search is needed.

**HuHu–LuLu regularization** (Faltus et al.). $\mathbb{H}\tilde u$ is the second
gradient of each displacement component and $\mathbb{L}\tilde u$ its trace, the
Laplacian. Since $|\mathbb{H}u_i|^2 \ge \tfrac{1}{d_{\mathrm{dim}}} (\mathrm{tr}\, \mathbb{H}u_i)^2$,
the term is non-negative. It penalizes the curvature of the displacement
field, which stops the soft medium from collapsing into zig-zag (hourglass)
modes before real contact. It is scaled as
$k_r = \alpha L^2 (\kappa + \tfrac{4}{3}\mu)$ with $\alpha = 10^{-6}$.

> With bilinear (Q1) elements on axis-aligned pixels the Laplacian of the
> interpolation vanishes, and $\mathbb{H}$ keeps only the mixed derivative
> $\partial_{xy}$, so the regularization is weak. With Q2
> (`biquadratic_rectangle`) both parts act. Q1 is the default because it is 4×
> cheaper and works for this geometry.

**AT2 crack density.** $G_c$ is the toughness and $\ell$ the length scale
($1/32$ of the cell width, fixed in physical units so that a finer grid
refines the same problem: 1 pixel at $32^2$, 4 pixels at $128^2$). The crack-density term is integrated over the whole cell, but the
driving force is zero in the third medium (below), so damage there only
diffuses in from the solid and has no mechanical effect ($g_s = 1$).

### Driving force and irreversibility

$$
\psi^+ = W_{\mathrm{iso}} + \langle J > 1 \rangle\, W_{\mathrm{vol}} \quad \text{(solid only)},
\qquad
\mathcal{H}(x, t) = \max_{s \le t} \psi^+(F(x, s)) .
$$

The history field $\mathcal{H}$ (Miehe et al. 2010) makes the damage
irreversible and replaces $\psi^+$ in the damage equation.

### Stress and tangent

With $a = J^{-2/3}$, $b = (I_1 + 1)/3$ and $F^{-T}$:

$$
P = \frac{\partial W}{\partial F}
  = g_s\, \mu\, a \left( F - b\, F^{-T} \right) + g_v\, \kappa \ln J\; F^{-T},
$$

$$
\mathbb{A}_{ijkl} = \frac{\partial P_{ij}}{\partial F_{kl}}
= g_s \mu a \Big[ \delta_{ik}\delta_{jl}
  - \tfrac{2}{3} F^{-T}_{kl} \big(F_{ij} - b F^{-T}_{ij}\big)
  - \tfrac{2}{3} F_{kl} F^{-T}_{ij}
  + b\, F^{-T}_{il} F^{-T}_{kj} \Big]
+ g_v \kappa \Big[ F^{-T}_{ij} F^{-T}_{kl} - \ln J\; F^{-T}_{il} F^{-T}_{kj} \Big] .
$$

The switch in $g_v$ is not differentiated. This is consistent: at $J = 1$ both
$W_{\mathrm{vol}}$ and its derivative vanish, so the energy is $C^1$ and only
the tangent jumps. $P$ and $\mathbb{A}$ were checked against central finite
differences (relative error about $10^{-9}$).

## 3. Discretization

Finite elements on the pixel grid ([theory §4](../theory.md#4-finite-element-discretization-on-a-pixel-grid)):

- **Displacement:** an `elasticity` discretization.
- **Damage:** a `conductivity` (scalar) discretization with the same element.

Both have the same quadrature points and MPI decomposition. All constitutive
quantities live at the quadrature points; integrals are $\sum_q w_q (\cdot)$, and
the weights of one pixel sum to the pixel area.

| Operator | Meaning |
|---|---|
| $B$ | gradient: $\nabla \tilde u$ at the quadrature points |
| $N$ | interpolation: $d$ at the quadrature points |
| $\mathbb{H}$, $\mathbb{L}$ | Hessian and Laplacian stencils of the regularization |
| $W$ | quadrature weights |

## 4. Numerical scheme

### 4.1 Load stepping (adaptive, with a predictor)

$\lambda$ runs from 0 to 1. The largest step is $1/50$ and the smallest is
$1/(50 \cdot 64)$. For each increment $\lambda_n \to \lambda_n + \Delta\lambda$:

1. **Predictor.** Extrapolate linearly from the last two converged states,
   $\tilde u^{\mathrm{pred}} = \tilde u_n + \tfrac{\Delta\lambda}{\Delta\lambda_{n-1}}(\tilde u_n - \tilde u_{n-1})$.
   If it is not admissible ($J \le 0$ at some quadrature point), use
   $\tilde u_n$ instead.
2. **Admissibility.** If $\tilde u_n$ is not admissible at
   $\lambda_n + \Delta\lambda$ either, halve $\Delta\lambda$ and retry.
3. **Staggered solve** (4.2). If the trust-region solve fails, or the
   staggered loop has not converged after `max_staggered_iterations`, restore
   $\tilde u_n$ and $d_n$, halve $\Delta\lambda$ and retry. An unconverged
   increment is never accepted.
4. **Accept.** Set $\mathcal{H}_n \leftarrow \mathcal{H}$, write a frame, and
   double the step for the next increment, up to the largest step.

The predictor matters near contact. Without it, the medium in the closing
channel is so thin that the old $\tilde u_n$ at a larger load already has
$J \le 0$, and the step got stuck at $\Delta\lambda = 0.0025$. With it, the step
stays at $0.02$ through contact.

### 4.2 Staggered (alternate minimization) loop

Repeat until $\max |d - d_{\mathrm{old}}| < 10^{-3}$, at most 300 times
(otherwise the increment is retried with half the step, 4.1):

**(a) Mechanics, damage frozen.** Minimize
$\Pi(\cdot, d)$ over $\tilde u$ with NuMPI's trust-region Newton–CG
(`tr_newton_bounded`, $\|\nabla\Pi\|_\infty < 10^{-5}$):

$$
\nabla \Pi = B^T W P(F) + R\,\tilde u, \qquad
\nabla^2 \Pi\, v = B^T W\, \mathbb{A}(F)\, B\, v + R\, v, \qquad
R = k_r \left( \mathbb{H}^T W \mathbb{H} - \tfrac{1}{d_{\mathrm{dim}}} \mathbb{L}^T W \mathbb{L} \right).
$$

- **Inner solver.** Steihaug CG uses only Hessian–vector products, so no matrix
  is assembled. It stops at the trust-region boundary or at negative curvature;
  $\Pi$ is non-convex in contact.
- **Inadmissible trial points.** Any trial point with $J \le 0$ returns
  $\Pi = 10^{30}$. That makes the reduction ratio very negative, so the step is
  rejected and the radius shrinks. A finite value is used instead of $\infty$,
  because $\infty - \infty$ gives NaN and stalls the radius update.
- **Preconditioner.** A Green operator, built once, for the undamaged solid
  linearized at $F = I$ plus the regularization:
  $M = B^T W \mathbb{A}(I) B + R$, inverted blockwise in Fourier space. It must
  not change during a solve, because the trust region is measured in the
  $M$-norm.

**(b) History.** $\mathcal{H} = \max(\mathcal{H}_n, \psi^+(F))$ at every
quadrature point of the solid, and 0 in the medium.

**(c) Damage, displacement frozen.** The stationarity of $\Pi$ in $d$, with
$\psi^+ \to \mathcal{H}$, is linear and SPD:

$$
\left[ G_c \ell\, B^T W B + N^T W \left( \tfrac{G_c}{\ell} + 2\mathcal{H} \right) N \right] d
= N^T W\, 2\mathcal{H} .
$$

It is solved with CG (relative tolerance $10^{-8}$) and the Green–Jacobi
preconditioner $D^{-1/2} G D^{-1/2}$:
- $G$ inverts the homogeneous operator with $\mathcal{H} = 0$. Its zero
  frequency is regular because of the mass term, so it is inverted too
  (`invert_zero_mode=True`).
- $D$ is the diagonal of the current operator. It is recomputed in every
  iteration with Dirac combs, because $\mathcal{H}$ changes.

**(d) Degradation.** $g_s$ is updated from $N d$ for the next mechanics solve.

The mechanics problem at the end of an increment is solved with the damage of
the previous staggered iteration. The loop tolerance on $d$ controls this
lag, as in the small-strain example.

### 4.3 Output

`io_utils.FieldWriter` writes one frame per accepted increment to
`exp_data/phase_field_fracture_contact/Nx=<n>Ny=<n>/contact_fracture.nc`:

- **Fields:**
  - `damage` and `u_fluc` (nodal; sub-point 0 is the pixel corner);
  - `history` (quadrature points);
  - `detF` (pixel average) and `phase` (pixel).
- **Frame variables:** `lam`, `dlam`, `F10`, `P10`, `Pxx`, `mean_P`
  ($\langle P \rangle = \sum_q w_q P / |\Omega|$), `min_det_F`, `energy`,
  `damage_max`, `staggered_iterations`, `nb_outer`, `nb_hessp`, `cg_damage`,
  `converged`, `elapsed_time`.
- **Attributes:** all parameters.

`plot_phase_field_fracture_contact_output.py` plots the $\bar P_{10}$–$\bar F_{10}$
curve with the maximal damage, and the damage on the deformed cell. `--animate`
saves an MP4. The history names match the contact examples, so
`examples/internal_contact/read_tmc.py` can read the file as well.

## 5. Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `nnn` | 32 | pixels per direction |
| `element_type` | `bilinear_rectangle` | or `biquadratic_rectangle` (Q2) |
| `H_macro` | $H_{10} = 0.3$ | shear at $\lambda = 1$ |
| `nb_increments`, `min_step` | 50, $1/3200$ | largest and smallest load step |
| `E_solid`, `nu_solid` | 100, 0.3 | solid |
| `k_v`, `alpha` | $10^{-5}$, $10^{-6}$ | third-medium stiffness, regularization |
| `Gc`, `length_scale`, `k_residual` | 0.1, $1/32$ of the cell (fixed, not in pixels), $10^{-4}$ | AT2 |
| `staggered_tol`, `max_staggered_iterations` | $10^{-3}$, 300 | staggered loop |
| `SOLVER_GTOL`, `SOLVER_MAXITER`, `SOLVER_INNER_TOL` | $3\cdot10^{-6}$, 500, `None` | trust-region Newton–CG on $\Pi / s$, $s = E h^{d-1}$: `SOLVER_GTOL` bounds $\max\|g\| / s$ (a residual stress over $E$, independent of the resolution); `None` = inexact Newton, $\eta_k = \min(0.5, \sqrt{\max\|g_k\| / s})$ |

## 6. Limitations

- **Plane-strain split.** The isochoric part uses the 3D invariants with
  $F_{33} = 1$. Under in-plane compression $W_{\mathrm{iso}} > 0$, so shape change
  under compression still drives damage. Only the volumetric part is protected.
- **Damage in the medium.** $d$ diffuses a short distance into the third medium,
  where it has no effect. The crack-density integral therefore slightly
  overestimates the crack energy next to the cavity.
- **Hybrid character.** The degradation depends on the sign of $\ln J$, so the
  mechanics problem is not exactly the minimization of one fixed energy across
  $J = 1$. The energy is $C^1$ there, and the trust region handles the kink in
  the tangent.
- **Cost.** Every staggered iteration is a full trust-region solve. Once the
  crack runs, the number of staggered iterations rises. Use small grids (32²,
  serial) for testing.
- **Patched NuMPI.** The trust-region solver needs the `precond` argument
  (see [Internal contact](internal_contact.md)).
