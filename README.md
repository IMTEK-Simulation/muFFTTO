# muFFTTO

FFT-based micro-scale topology optimization on periodic unit cells using the [muGrid](https://github.com/muSpectre/muGrid) / [muFFT](https://github.com/muSpectre/muFFT) framework.

## Features

- **Homogenization**: Compute effective material properties (conductivity and elasticity) of periodic microstructures
- **Topology optimization**: Phase-field based optimization with adjoint sensitivity analysis
- **Multi-dimensional**: Supports 1D, 2D, and 3D problems
- **FEM discretization**: Triangular and hexahedral finite elements with quadrature-based integration
- **FFT solvers**: Matrix-free conjugate gradient solvers accelerated by FFT (via muFFT/muGrid)
- **MPI parallelism**: Distributed computing support through muGrid's MPI infrastructure
- **Microstructure library**: Built-in parametric geometries (laminates, inclusions, lattices, etc.)

## Installation

```bash
pip install -e .
```

### Dependencies

- Python >= 3.8
- [numpy](https://numpy.org/), [scipy](https://scipy.org/)
- [muGrid](https://github.com/muSpectre/muGrid) and [muFFT](https://github.com/muSpectre/muFFT) (must be installed separately)

For running tests:

```bash
pip install -e .[test]
```

## Package Structure

| Module | Description |
|--------|-------------|
| `domain.py` | Core classes `PeriodicUnitCell` and `Discretization` for setting up the computational domain, fields, and operators |
| `topology_optimization.py` | Objective functions, sensitivity analysis (adjoint method), and phase-field potentials |
| `solvers.py` | Preconditioned conjugate gradient (PCG) and Adam optimizer for solving linear systems and optimization problems |
| `discretization_library.py` | Shape function gradient matrices for various element types (linear triangles, bilinear rectangles, trilinear hexahedra) |
| `microstructure_library.py` | Parametric geometry definitions for generating periodic microstructures |

## Examples

The `examples/` directory contains working examples:

- `example_2D_homogenization_conductivity_*.py` - 2D thermal conductivity homogenization
- `example_2D_homogenization_elasticity.py` - 2D elasticity homogenization with FEM
- `example_2D_elasticity_TO.py` - 2D topology optimization for elasticity
- `example_3D_homogenization_conductivity.py` - 3D thermal conductivity homogenization
- `example_3D_homogenization_elasticity.py` - 3D elastic homogenization
- `example_2D_homogenization_elasticity_Hashin_composite_sphere.py` - Validation against Hashin analytical bounds

## Tests

```bash
pytest
```

## Authors

- Lars Pastewka (University of Freiburg)
- Martin Ladecky

## License

MIT - see [LICENSE.md](LICENSE.md)
