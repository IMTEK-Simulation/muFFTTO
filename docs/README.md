# muFFTTO documentation

This folder documents what the examples in [`examples/`](../examples/) compute,
from the point of view of the PDE being solved: the periodic cell problem,
its finite-element discretization on a pixel grid, the FFT-preconditioned
conjugate-gradient solver, the extraction of effective properties, and the
adjoint-based topology optimization built on top of it.

## Where to start

1. **[Theory and numerics](theory.md)**: the mathematics, linked to the
   functions that implement it.
   - [1. Periodic homogenization: the cell problem](theory.md#1-periodic-homogenization-the-cell-problem)
   - [2. Conductivity and small-strain elasticity](theory.md#2-physical-problems-conductivity-and-small-strain-elasticity)
   - [3. Finite-strain elasticity](theory.md#3-finite-strain-elasticity)
   - [4. FE discretization on a pixel grid](theory.md#4-finite-element-discretization-on-a-pixel-grid)
   - [5. Assembling and solving the linear system](theory.md#5-assembling-and-solving-the-linear-system)
   - [6. Effective (homogenized) properties](theory.md#6-effective-homogenized-properties)
   - [7. Topology optimization](theory.md#7-topology-optimization)
   - [8. Grid adaptation (deformed grids)](theory.md#8-grid-adaptation-deformed-grids)
   - [9. Notation and array conventions](theory.md#9-notation-and-array-conventions)
2. **Example walkthroughs**: each page maps the code of an example, block by
   block, to the mathematical steps.

| Page | Examples covered | What is solved |
|------|------------------|----------------|
| [Homogenization: conductivity](examples/homogenization_conductivity.md) | `examples/homogenization/conductivity/` | Scalar diffusion cell problem in 2D/3D, linear and quadratic elements, block CG |
| [Homogenization: elasticity](examples/homogenization_elasticity.md) | `examples/homogenization/small_strain_elasticity/`, `examples/homogenization/finite_strain_elasticity/` | Small-strain linear elasticity in 2D/3D, finite-strain Neo-Hookean with Newton–CG |
| [Topology optimization](examples/topology_optimization.md) | `examples/topology_optimization/` | Phase-field inverse design of microstructures with prescribed effective conductivity/stiffness (adjoint sensitivities + L-BFGS) |
| [Grid adaptation](examples/grid_adaptation.md) | `examples/grid_adaptation/` | Homogenization on deformed (mapped) grids |
| [Analytical solutions](examples/analytical_solutions.md) | `examples/analytical_solutions/` | Hashin coated inclusion: verification against a closed-form solution |
| [Internal contact](examples/internal_contact.md) | `examples/internal_contact/` | Third-medium contact at finite strain with a trust-region Newton solver |
| [Phase-field fracture](examples/phase_field_fracture.md) | `examples/phase_field_fracture/` | AT2 brittle fracture of a periodic cell: staggered elasticity / damage solves, both FFT-preconditioned CG |
| [Phase-field fracture with contact](examples/phase_field_fracture_contact.md) | `examples/phase_field_fracture/` | AT2 fracture + third-medium contact at finite strain: split neo-Hookean, trust-region Newton-CG, staggered damage, adaptive load steps |

3. **[Saving and loading fields](io.md)**: `io_utils`, MPI-parallel NetCDF
   output and restart through muGrid.
4. **[Known issues](known_issues.md)**: suspected bugs and inconsistencies
   found while writing the documentation. None of them has been fixed yet.

## Reading the code

All modules in [`muFFTTO/`](../muFFTTO/) have NumPy-style docstrings, and inline
comments explain the maths. Array names carry index suffixes that encode
their layout. For example, `stress_ijqxyz` is a stress field with tensor indices
`i, j`, quadrature point `q` and pixel coordinates `x, y, z`. The full convention is in
[theory §9](theory.md#9-notation-and-array-conventions).
