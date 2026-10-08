# Phase-field fracture (AT2)

Example: [`examples/phase_field_fracture/example_2D_phase_field_fracture_AT2.py`](../../examples/phase_field_fracture/example_2D_phase_field_fracture_AT2.py)

A periodic cell with a soft circular pore (stiffness ratio $10^{-3}$) is loaded
in uniaxial macroscopic strain $\bar{\varepsilon} = t\, e_x \otimes e_x$, which
increases step by step. A crack nucleates at the poles of the pore and runs
through the ligaments perpendicular to the load. The example prints the
homogenized stress–strain curve and plots it together with damage snapshots.

![AT2 result](../../examples/phase_field_fracture/figures/phase_field_fracture_AT2.png)

## 1. Model

The standard second-order (AT2) regularization of brittle fracture
(Bourdin, Francfort & Marigo 2000; Miehe, Hofacker & Welschinger 2010).
The unknowns are the periodic displacement fluctuation $\tilde u$ and the
damage $d \in [0, 1]$:

$$
\Pi(\tilde u, d) = \int_\Omega g(d)\, \psi^+(\varepsilon) + \psi^-(\varepsilon) \, dx
 + \frac{G_c}{2} \int_\Omega \frac{d^2}{\ell} + \ell\, |\nabla d|^2 \, dx,
\qquad
\varepsilon = \bar{\varepsilon} + \nabla^s \tilde u,
\qquad
g(d) = (1-d)^2 + k .
$$

Here $G_c$ is the fracture toughness and $\ell$ the regularization length
(2 pixels in the example). The small residual stiffness $k = 10^{-4}$ keeps the
elastic problem regular in the broken material.

**Energy split (Amor, Marigo & Maurini 2009).** Only the tensile part drives
the crack:

$$
\psi^+ = \frac{K}{2} \langle \mathrm{tr}\, \varepsilon \rangle_+^2 + \mu\, \varepsilon_{\mathrm{dev}} : \varepsilon_{\mathrm{dev}},
\qquad K = \lambda + \frac{2\mu}{d_{\mathrm{dim}}}, \qquad
\varepsilon_{\mathrm{dev}} = \varepsilon - \frac{\mathrm{tr}\,\varepsilon}{d_{\mathrm{dim}}} I .
$$

**Irreversibility** comes from the history field of Miehe et al. (2010):
$H(x, t) = \max_{s \le t} \psi^+(\varepsilon(x, s))$.

**Hybrid formulation (Ambati, Gerasimov & De Lorenzis 2015).** The elastic
problem uses the isotropic degradation $\mathbb{C}(d) = g(d)\, \mathbb{C}_0$.
It therefore stays linear and symmetric, and the standard homogenization solver
applies unchanged. The split enters only through $H$.

## 2. Discrete equations (staggered scheme)

Within a load step the two subproblems are solved alternately until
$\max |d - d_{\mathrm{old}}| < 10^{-3}$:

1. **Elasticity** with the damage frozen. This is the cell problem of
   [theory §5](../theory.md#5-assembling-and-solving-the-linear-system) with
   material data $g(d)\,\mathbb{C}_0$:

   $$
   B^T W\, g(d)\, \mathbb{C}_0\, B\, \tilde u = - B^T W\, g(d)\, \mathbb{C}_0 : \bar{\varepsilon}.
   $$

   It is solved with PCG and the Green preconditioner of the undamaged
   reference $\mathbb{C}_0$.

2. **History update** at the quadrature points:
   $H = \max(H_{n-1}, \psi^+(\varepsilon))$, where $H_{n-1}$ is the value from
   the last converged load step.

3. **Damage** with $H$ fixed. Setting the variation of $\Pi$ with respect to
   $d$ to zero (with $\psi^+$ replaced by $H$) gives
   $(G_c/\ell + 2H)\, d - G_c \ell\, \Delta d = 2H$. In discrete form:

   $$
   \left[ G_c \ell\, B^T W B + N^T W \left(\frac{G_c}{\ell} + 2H\right) N \right] d = N^T W\, 2H .
   $$

   The matrix is SPD and the equation is linear in $d$. $B$ is the gradient
   operator and $N$ the interpolation from nodes to quadrature points
   (`apply_gradient_operator_mugrid`, `apply_N_operator_mugrid` and their
   transposes). The solver is CG with a Green preconditioner of the
   homogeneous operator ($H = 0$), built through the `operator=` argument of
   `get_preconditioner_Green_mugrid`.

The mass term makes the zero-frequency block $\hat K(0)$
regular, unlike in a pure stiffness problem. The example therefore passes
`invert_zero_mode=True`, so the mean of $d$ is preconditioned exactly as well.

## 3. Code structure

- **Two discretizations of the same grid.** `disc_u` (`'elasticity'`) holds
  $\tilde u$ and `disc_d` (`'conductivity'`, scalar) holds $d$. Both have the
  same MPI decomposition, so the arrays of quadrature-point and nodal fields
  are copied between them with `.s[...]`.
- **`update_degraded_material`** computes $C_d = g(N d)\, \mathbb{C}_0$.
- **`tensile_energy`** computes $\psi^+$ (Amor split) at the quadrature points.
- **`apply_damage_operator`** is the damage matrix above. It is used both by
  CG and, with $H = 0$, to build the preconditioner.
- **Homogenized stress.** After each load step, $\bar\sigma$ is computed with
  `get_homogenized_stress_mugrid` from the degraded material data.

### Preconditioner

`preconditioner_type` selects `'Green'` ($M = G$) or `'Green_Jacobi'`
($M = D^{-1/2} G D^{-1/2}$ with $D = \mathrm{diag}(K)$). The default is
`'Green_Jacobi'`. The diagonal is recomputed before every elastic and every
damage solve, because $K$ changes with $d$ and $H$. Each recomputation costs
a few operator applications (Dirac combs). For the damage equation it is
computed with `get_preconditioner_Jacobi_mugrid(operator=...)`.

On the 64×64 run both choices give the same stress–strain curve. The total
number of CG iterations is:

| | elasticity | damage | wall time |
|---|---|---|---|
| Green | 22 099 | 2 002 | 48 s |
| Green–Jacobi | 11 331 | 1 592 | 31 s |

On a 128×128 grid ($\ell$ is again 2 pixels, so the peak moves to
$\bar\sigma_{xx} \approx 0.032$ at $\bar\varepsilon_{xx} \approx 0.057$) the
picture is mixed:

| 128×128 | CG per elastic solve, before / after the crack | elasticity total | damage total |
|---|---|---|---|
| Green | ≈ 40 / ≈ 305 | 24 565 | 2 442 |
| Green–Jacobi | ≈ 68 / ≈ 114 | 20 600 | 1 702 |

At 128², Jacobi scaling slows the intact phase and speeds up the cracked
phase, so the overall saving drops to about 16%.
Whichever phase dominates the run decides which preconditioner is better.

The gain is in the cracked state. There the stiffness ranges from $k$ to 1,
and the Jacobi scaling removes most of this contrast: about 64 instead of
about 270 elastic CG iterations per solve. Before the crack forms, the two
are about the same.

### Watching the damage evolve

Set `live_plot = 'iteration'` (redraw after every staggered iteration),
`'step'` (after every load step) or `None` at the top of the script. A Tk
window then shows the damage field and the stress–strain curve, redrawn in
place. The script forces the `TkAgg` backend, because PyCharm's plot panel
cannot redraw a figure. The live plot is only drawn in serial runs.

### Output

With `output_file` set (default `exp_data/phase_field_fracture_AT2.nc`, `None`
switches it off), the run is written to a NetCDF file, see
[Saving and loading fields](../io.md). `output_every` sets how often:

- `'step'` (default): one frame per load step, after the staggered loop has converged.
- `'iteration'`: one frame per staggered iteration, so you can follow how the crack
  grows inside the unstable step.

Each frame holds the damage, $\tilde u$ and $H$, and the frame variables `load`,
`macro_stress`, `step`, `staggered_iteration`, `damage_change` and `converged`
(1 for the last frame of a load step). The run parameters are stored as attributes.

Replot the stored run afterwards, also for parallel runs:

```
python plot_phase_field_fracture_output.py [file.nc] [frame ...]   # curve + damage of chosen frames
python plot_phase_field_fracture_output.py [file.nc] --animate     # movie of all frames, saved as file.mp4
```

The stress–strain curve uses the converged frames; without frame numbers the
peak and the last frame are shown.

## 4. Results and remarks

- **Before the peak.** AT2 has no elastic threshold, so damage grows from the
  first step, slowly at first. The curve softens gradually up to the peak
  $\bar\sigma_{xx} \approx 0.025$ at $\bar{\varepsilon}_{xx} \approx 0.049$.
- **Failure.** In the next step the crack crosses the cell. This propagation
  is unstable under strain control: about 45 staggered iterations, after which
  the stress drops to the residual level set by $k$ and the pore.
- **Solver cost.** On the intact material PCG needs about 35–50 iterations.
  On the cracked cell, Green alone needs about 250–270, because the stiffness
  contrast is then about $1/k$; Green–Jacobi needs about 60 (see above).
- **Run time.** 64×64 grid, 80 load steps: about 30 s serial with Green–Jacobi. A 2-rank MPI run
  gives the same numbers.
- **Things to vary.** Grid, $G_c$, $\ell$, $k$, the geometry
  (`microstructure_name`) and the loading direction `macro_strain_direction`.
  Under shear or compressive loading the split matters, because only
  $\psi^+$ drives the crack.
- **Limitations.** The staggered scheme cannot follow snap-back under pure
  strain control, and the drop at failure happens within one load step. A
  path-following (arc-length) control or a monolithic Newton scheme would be
  needed to resolve the post-peak branch.
