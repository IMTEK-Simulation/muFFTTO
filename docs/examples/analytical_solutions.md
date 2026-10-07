# Analytical verification: Hashin coated cylinder

Examples: [`examples/analytical_solutions/`](../../examples/analytical_solutions/)

| File | Role |
|---|---|
| `analytical_solution_2D_elasticity_Hashin_composite_sphere.py` | closed-form strain field `solution_hashin(...)` and its plot |
| `example_2D_homogenization_elasticity_Hashin_composite_sphere.py` | the same configuration solved numerically with muFFTTO |
| `hashin_strain_field_2d_analytical_v2.png`, `hashin_strain_field_2d_FFT_solver.png` | the saved plots of the two scripts (same colour scale) |

Background: [cell problem](../theory.md#1-periodic-homogenization-the-cell-problem),
[small-strain elasticity](../theory.md#2-physical-problems-conductivity-and-small-strain-elasticity),
[FE on a pixel grid](../theory.md#4-finite-element-discretization-on-a-pixel-grid),
[linear solve](../theory.md#5-assembling-and-solving-the-linear-system),
[effective properties](../theory.md#6-effective-homogenized-properties).

## What PDE is solved

Small-strain periodic homogenization in 2D (plane strain, $d=2$): find a periodic
fluctuation $\tilde u$ such that

$$\nabla\cdot\big(C(x):(E+\nabla^s\tilde u)\big)=0\ \text{in }\Omega=[0,1)^2,\qquad \bar\sigma=\langle C:(E+\nabla^s\tilde u)\rangle ,$$

for a cell made of three isotropic phases: a core $r\le R_1$, a shell
$R_1<r<R_2$ and a surrounding matrix (centre $(0.5,0.5)$, $R_1=0.2$, $R_2=0.4$).

**Idea of the test (Hashin's composite sphere/cylinder).** If the matrix has the
bulk modulus $\kappa_3$ equal to the effective bulk modulus of the coated
inclusion, then inserting the coated inclusion into a uniformly, hydrostatically
strained matrix does not disturb the matrix at all. For $E=I$ the exact solution is
then known in closed form everywhere. The matrix strain is exactly $I$, which is
compatible with periodicity, so the closed-form field is also the exact periodic
solution. The homogenized stress for $E=I$ is exactly $\bar\sigma=d\,\kappa_3\,I$.
This only holds for hydrostatic loading. The matrix shear modulus $\mu_3$ is
arbitrary, and the shear entries of $C^{\mathrm{eff}}$ have no analytical reference.

## The analytical formula (as implemented)

`solution_hashin(coordinates, mu1, lam1, mu2, lam2, R1, R2, center)`
(`analytical_solution_2D_elasticity_Hashin_composite_sphere.py:8-75`). The
parameters are the Lamé constants of core (1) and shell (2), and the radii. With
$d$ = `len(center)`:

$$\kappa_i=\lambda_i+\tfrac{2}{d}\mu_i,\qquad \phi=\Big(\frac{R_1}{R_2}\Big)^d,\qquad
\alpha=\frac{d(\kappa_2-\kappa_1)}{2(d-1)\mu_2+d\,\kappa_1},$$

$$a_2=\frac{1}{1+\alpha\phi},\qquad a_1=(1+\alpha)\,a_2,\qquad b_2=\alpha R_1^d\,a_2 .$$

With $r=|x-x_c|$, $e_r=(x-x_c)/r$ and region coefficients
$(a,b)=(a_1,0)$ for the core, $(a_2,b_2)$ for the shell and $(1,0)$ for the matrix, the strain is

$$\varepsilon(x)=\Big(a+\frac{b}{r^d}\Big)I-d\,\frac{b}{r^d}\,e_r\otimes e_r ,$$

i.e. $\varepsilon_{rr}=a-(d-1)b/r^d$ and $\varepsilon_{\theta\theta}=a+b/r^d$. At $r=0$ the
code sets $\varepsilon=a_1 I$ (`:73`). The coefficients make the radial displacement
continuous at $R_1$ ($a_1=a_2+b_2/R_1^d$) and at $R_2$ ($a_2+b_2/R_2^d=1$), and make the
radial traction continuous at $R_1$, which defines $\alpha$.

Radial traction continuity at $R_2$ gives the matrix bulk modulus. It is computed in the
numerical example (`example_..._Hashin_composite_sphere.py:52-55`):

$$\beta=1+\frac{2(d-1)\mu_2}{d\,\kappa_2},\qquad
\kappa_3=\kappa_2\Big(1-\beta\,\frac{\alpha\phi}{1+\alpha\phi}\Big)
= a_2\Big(\kappa_2-\tfrac{2(d-1)}{d}\mu_2\,\alpha\phi\Big).$$

With the default data ($\lambda_1=0.001,\ \mu_1=0.005,\ \lambda_2=1,\ \mu_2=0.5$) this gives
$\kappa_1=0.006$, $\kappa_2=1.5$, $\alpha\approx2.9526$, $\kappa_3\approx0.65065$, so the exact
homogenized stress for $E=I$ is $\bar\sigma\approx1.30131\,I$. The exact strains are
$a_1\approx2.274$ in the soft core and $\varepsilon=I$ in the matrix.

## Walkthrough: `analytical_solution_2D_elasticity_Hashin_composite_sphere.py`

Standalone (numpy and matplotlib only). The `__main__` block (`:78-140`):

1. sets the material data and radii (`:80-90`);
2. builds a 512×512 point cloud on $[0,1)^2$ (`:94-97`);
3. evaluates `eps = solution_hashin(pts, ...)` with shape `(N, N, 2, 2)` (`:103`);
4. plots $\varepsilon_{xx},\varepsilon_{xy},\varepsilon_{yx},\varepsilon_{yy}$ with a fixed colour range
   `[0.2, 1.3]`, overlays the two circles and saves
   `hashin_strain_field_2d_analytical_v2.png` (`:108-140`).

## Walkthrough: `example_2D_homogenization_elasticity_Hashin_composite_sphere.py`

**1. Discretization** (`:14-29`). 512×512 pixels, linear triangles, small strain.

**2. Phases** (`:36-63`). Core and shell tensors from
`material_models.get_elastic_tensor_from_lame`. The matrix uses $\kappa_3$ from the
formula above and $\mu_3=0.3$, $\lambda_3=\kappa_3-\tfrac{2}{d}\mu_3$:

```python
phi = (r_1 / r_2) ** dim
alpha = dim * (kappa_2 - kappa_1) / ((dim - 1) * 2.0 * mu_2 + dim * kappa_1)
beta = 1 + 2 * (dim - 1) * mu_2 / (dim * kappa_2)
kappa_3 = kappa_2 * (1.0 - beta * (alpha * phi) / (1.0 + alpha * phi))
```

**3. Geometry** (`:69-82`). The radius is evaluated at the pixel coordinates
`discretization.fft.coords` (`r <= r_1` core, `r_1 < r < r_2` shell), and the
material is assigned to all quadrature points of a pixel. The circular interfaces are
therefore staircase-approximated.

**4. Operator and preconditioner** (`:86-105`). `K_fun` applies
`apply_system_matrix_mugrid(..., formulation='small_strain')`, i.e.
$B^TW\,C:\nabla^s u$. The Green preconditioner uses the reference tensor
`C_0_ref = np.sqrt(np.einsum('ijkl,klmn->ijmn', C_core, C_shell))` (`:67`), an
entry-wise square root of the product of the core and shell tensors (a heuristic
"geometric mean").

**5. Hydrostatic solve** (`:111-141`). $E=I$ (`:34`), rhs from `get_rhs_mugrid`, and
PCG with `muFFTTO.solvers.conjugate_gradients_mugrid` (`tol=1e-8`).

**6. Strain field and comparison** (`:145-184`). The total strain is
$\varepsilon=E+\nabla^s\tilde u$ (`apply_gradient_operator_symmetrized_mugrid` plus $E$),
averaged over the two quadrature points of each pixel and plotted with the same
colour range and circles as the analytical script. It is saved as
`hashin_strain_field_2d_FFT_solver.png`. The comparison is visual, between the two
PNGs. No pointwise error norm is computed.

**7. Homogenized stress** (`:189-196`). `get_homogenized_stress_mugrid` returns
$\bar\sigma$ for $E=I$. Compare $\bar\sigma_{11}=\bar\sigma_{22}$ with $d\,\kappa_3\approx1.3013$ and
$\bar\sigma_{12}$ with 0. The script prints $\bar\sigma$ and `C_matrix` but not $d\kappa_3$ itself.

**8. Full tangent** (`:203-240`). Four solves with $E=e_i\otimes e_j$ (`tol=1e-6`)
give $C^{\mathrm{eff}}$ in Voigt notation. Check $C^{\mathrm{eff}}_{11}+C^{\mathrm{eff}}_{12}\approx 2\kappa_3$.
The shear-related entries have no closed-form reference here.

## Summary

| File | Dim | Physics | Key method | Outputs |
|---|---|---|---|---|
| `analytical_solution_2D_elasticity_Hashin_composite_sphere.py` | 2D (formula is $d$-generic) | linear elasticity, coated cylinder | closed-form strain `solution_hashin` | `hashin_strain_field_2d_analytical_v2.png`; strain at centre printed |
| `example_2D_homogenization_elasticity_Hashin_composite_sphere.py` | 2D | small-strain periodic homogenization | FE + Green-PCG (`conjugate_gradients_mugrid`), matrix bulk modulus = Hashin $\kappa_3$ | `hashin_strain_field_2d_FFT_solver.png`; $\bar\sigma(E=I)$; $C^{\mathrm{eff}}$ (Voigt) |

## Key tunable parameters

| Parameter | Where (example script) | Meaning |
|---|---|---|
| `number_of_pixels` | `:21` | resolution (default 512×512) |
| `r_1`, `r_2` | `:37-38` | core / shell radii (must match `R1, R2` in the analytical script) |
| `lambda_1, mu_1, lambda_2, mu_2` | `:41-48` | core and shell Lamé constants |
| `mu_3` | `:61` | matrix shear modulus (free; does not affect the hydrostatic test) |
| `macro_gradient` | `:34` | must be hydrostatic ($\propto I$) for the analytical comparison |
| `tol` | `:138`, `:228` | PCG tolerances |

## How to run

```bash
cd examples/analytical_solutions
python analytical_solution_2D_elasticity_Hashin_composite_sphere.py
python example_2D_homogenization_elasticity_Hashin_composite_sphere.py
```

Run them from that directory if you want the PNGs to overwrite the ones stored
there, since both scripts save to the current working directory. Run serially:
the strain plot uses the local subdomain arrays.

## Known issues

- The two scripts are not linked. The numerical example does not import
  `solution_hashin`, so there is no automatic pointwise error and no printed
  $d\kappa_3$ reference. The check is manual (PNGs and $\bar\sigma$ versus $d\kappa_3$).
- The colour range `[0.2, 1.3]` clips the core strain ($a_1\approx2.27$), so the core
  shows as saturated in both plots.
- The shell mask differs slightly at $r=R_2$: `<` in the numerical script, `<=` in
  `solution_hashin`.
- The docstring of `solution_hashin` describes `Ci`/`R` tuple arguments and three
  stiffness objects. The actual signature takes Lamé scalars of core and shell only,
  and the matrix bulk is implied by the formula.
- The `callback` in the numerical script computes a residual norm but never prints it.
