# Phase-field fracture (AT1 and AT2)

Example: [`examples/phase_field_fracture/example_2D_phase_field_fracture_AT1_AT2.py`](../../examples/phase_field_fracture/example_2D_phase_field_fracture_AT1_AT2.py)

A periodic cell with a soft circular pore (stiffness ratio $10^{-3}$) is loaded in uniaxial macroscopic strain
$\bar{\varepsilon} = t\, e_x \otimes e_x$, which increases step by step. A crack nucleates at the poles of the pore
and runs through the ligaments perpendicular to the load. The example prints the homogenized stress–strain curve and
plots it with the final damage field.

```
MODEL=AT2 SPLIT=amor N=64 STEPS=40 python example_2D_phase_field_fracture_AT1_AT2.py      # defaults
MODEL=AT1 mpirun -n 4 python example_2D_phase_field_fracture_AT1_AT2.py
```

## 1. Model (`muFFTTO.phase_field_fracture`)

The unknowns are the periodic displacement fluctuation $\tilde u$ and the damage $d \in [0, 1]$:

$$
\Pi(\tilde u, d) = \int_\Omega g(d)\, \psi^+(\varepsilon) + \psi^-(\varepsilon) \, dx
 + \frac{G_c}{c_w} \int_\Omega \frac{w(d)}{\ell} + \ell\, |\nabla d|^2 \, dx,
\qquad \varepsilon = \bar{\varepsilon} + \nabla^s \tilde u,
\qquad g(d) = (1-d)^2 + k .
$$

* **AT2** (Bourdin, Francfort & Marigo 2000): $w(d) = d^2$, $c_w = 2$.
* **AT1** (Pham, Amor, Marigo & Maurini 2011): $w(d) = d$, $c_w = 8/3$; elastic up to $\psi_c = 3 G_c / (16 \ell)$.

`CrackDensity(model, Gc, l)` holds the coefficients; `degradation(d, k)` is $g$.

**Energy splits** (`elastic_energy(strain, lam, mu, split)`), only $\psi^+$ drives the crack:

* `'amor'` (Amor, Marigo & Maurini 2009): $\psi^+ = \frac{K}{2} \langle \mathrm{tr}\, \varepsilon \rangle_+^2 + \mu\,
  \varepsilon_{\mathrm{dev}} : \varepsilon_{\mathrm{dev}}$, $K = \lambda + 2\mu / d_{\mathrm{dim}}$;
* `'spectral'` (Miehe, Welschinger & Hofacker 2010): $\psi^+ = \frac{\lambda}{2} \langle \mathrm{tr}\, \varepsilon
  \rangle_+^2 + \mu \sum_i \langle \varepsilon_i \rangle_+^2$;
* `'none'`: $\psi^+ = \psi_0$.

**Irreversibility**: the history field $H = \max(H_{n-1}, \psi^+, \psi_c)$ (`update_history`, Miehe, Hofacker &
Welschinger 2010).

**Hybrid formulation** (Ambati, Gerasimov & De Lorenzis 2015): the elastic problem uses the isotropic degradation
$\mathbb C(d) = g(d)\, s(x)\, \mathbb C_0$, so it stays linear; only $H$ drives the crack.

## 2. Staggered scheme

Per load step, repeated until $\max|d - d_{\rm old}| < 10^{-3}$:

1. **elasticity** $B^T W\, g(d) s \mathbb C_0\, B \tilde u = -B^T W\, g(d) s \mathbb C_0\, \bar\varepsilon$: CG with the
   dual preconditioner `preconditioners.Dual(disc_u, C_0, material)`, $G\,K(\mathbb C_0 / (g s))\,G$ (the material is
   the `ScaledMaterial` $g(d) s(x) \mathbb C_0$, the dual follows it without an update);
2. **history** $H = \max(H_{n-1}, \psi^+(\varepsilon), \psi_c)$;
3. **damage** (`CrackDensity.apply_operator`, `CrackDensity.rhs`)

   $$
   \big[\,2 G_c \ell / c_w\; B^T W B + N^T W (\text{reaction} + 2H) N\,\big]\, d = N^T W\, (2H - \text{offset}),
   $$

   AT2: $G_c \ell$, reaction $G_c/\ell$, offset 0; AT1: $\tfrac34 G_c \ell$, reaction 0, offset $3G_c/(8\ell)$, $d$
   clipped to $[0, 1]$. CG with the Green preconditioner of the homogeneous operator
   (`preconditioners.Green(disc_d, operator=..., invert_zero_mode=True)`; its mass term makes the zero mode regular).

Every solve is `solvers.conjugate_gradients`: energy-error stop, relative tolerance $10^{-6}$, adaptive delay with
minimum 10.

The damage lives on a scalar ('conductivity') discretization of the same grid as the displacement; both have the
same MPI decomposition, so quadrature arrays are shared with `.s[...]`.

The studies built on this model (fatigue, ASR, battery particles, PF-CZM, sintering, fibres, polycrystals) are in
the separate repository `muFFTTO-dual-preconditioner`.
