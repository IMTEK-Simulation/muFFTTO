# Topology optimization from a bow-tie design

`examples/topology_optimization_bowtie/example_2D_elasticity_TO_bowtie.py`

Small-strain linear elastic inverse design of a periodic cell. The optimization
starts from the re-entrant (bow-tie) honeycomb, a classic auxetic, and tunes
its geometry to other Poisson ratios. The theory is in
[theory §7](../theory.md#7-topology-optimization).

## Steps

1. **Initial design.** `geometry.get('reentrant_honeycomb', coords, theta, thickness)`
   builds the bow-tie on a square cell: solid walls are 1, voids are 0.
2. **Homogenized properties.** Three macroscopic strains of amplitude
   `--strain` (default 0.05) are applied: $\varepsilon_{xx}$,
   $\varepsilon_{yy}$ and engineering shear $\gamma_{xy}$. Each gives one
   column of the effective Voigt stiffness $C_h$ (stress divided by the
   amplitude). The compliance $S_h = C_h^{-1}$ gives the engineering constants
   $E_x = 1/S_{11}$, $E_y = 1/S_{22}$, $G_{xy} = 1/S_{33}$,
   $\nu_{xy} = -S_{21}/S_{11}$ and $\nu_{yx} = -S_{12}/S_{22}$.
   `--evaluate-only` stops here. For $\theta = 30°$ and thickness 0.15 at 64²:
   $\nu_{xy} = -1.60$, $\nu_{yx} = -0.47$, $E_x \approx 3.4\,E_y$.
3. **Target with Poisson ratio `--nu`.**
   * `--keep stiffness` (default): $C_{11}$, $C_{22}$ and $C_{66}$ of the
     bow-tie, with $C_{12} = \nu\,C_{22}$. The stress targets differ from the
     bow-tie only through $C_{12}$.
   * `--keep moduli`: $E_x$, $E_y$ and $G_{xy}$ of the bow-tie. For a
     strongly auxetic design this changes $C_{11}$ several times.
4. **Objective.** The relative stress mismatch of the matched load cases
   (`--cases`, default all three), times `--weight` divided by the number of
   cases, plus the regularization. It is divided by its initial value, so the
   first L-BFGS step stays moderate. An unscaled first step drove $\rho$ to 0
   everywhere, where the SIMP derivative vanishes.
   * `--free-scale` matches $C_t$ only up to a factor $t$. The factor is
     optimal in closed form,
     $t^* = \sum_k \Sigma_t\!:\!\Sigma_h/|\Sigma_t|^2 \,\big/\, \sum_k |\Sigma_h|^2/|\Sigma_t|^2$,
     so only the Poisson ratio and the stiffness ratios count. By the envelope
     theorem, the gradient is the plain one with target $\Sigma_t/t^*$.
5. **Regularization** (`--regularization`):
   * `phase-field` (default): $\eta\int|\nabla\rho|^2 + \tfrac{c_{dw}}{\eta}\int\rho^2(1-\rho)^2$,
     with `--eta` (interface width in pixels) and `--double-well`.
   * `filter`: [density filter and Heaviside projection](../theory.md#75-alternative-regularization-density-filter-and-heaviside-projection),
     with `--filter-radius` (pixels) and $\beta$ continuation over `--beta`.
     There is one L-BFGS run of at most `--stage-iterations` per stage.
     `--robust` uses the mean mismatch of the eroded, intermediate and dilated
     designs (threshold 0.75 / 0.5 / 0.25). It guarantees a minimum feature
     size of the reported intermediate design.
6. **Output** (`data/<run>/`): `properties.txt` (initial, target and optimized
   $C$ and constants), `design.png`, `design_it*.png`, `history.nc` (design
   and physical density of every iteration) and `final.nc`.
   `compare_targets.py <runs>` draws an overview of several runs.
   `report.py [--logs DIR] <runs>` draws one row per run: initial and optimized design, convergence
   (mismatch, grey measure, $\beta$ stages), the error of each stiffness entry and a summary.
   `feature_size.py <runs>` measures the minimum solid and void feature size
   (pixels) by morphological opening: the largest disk whose opening only
   trims the boundary and keeps the connected pieces. A one-pixel hinge gives
   1 px.

## Notes from the first runs

* The target is very compliant: the bow-tie's stiffness is about 2% of the
  solid. With SIMP $p = 2$ and double well 1, a uniform grey density of 0.14
  costs less phase-field energy (0.9) than the interfaces of the crisp
  bow-tie (8.4), so the designs turned grey. $p = 3$ with double well 10
  avoids this.
* The CG tolerance of the original example ($10^{-3}$, plus the adjoint
  energy in the objective) made the line search stall. The example uses
  $10^{-8}$, with absolute tolerances scaled by the load amplitude.
