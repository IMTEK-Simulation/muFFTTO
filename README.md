# muFFTTO

FFT-based micro-scale topology optimization on periodic unit cells using the [muGrid](https://github.com/muSpectre/muGrid) / [muFFT](https://github.com/muSpectre/muFFT) framework.

## Features
 
- **Topology optimization**: Phase-field based optimization with adjoint sensitivity analysis
- **Multi-dimensional**: Supports 1D, 2D, and 3D problems
- **FEM discretization**: Triangular and hexahedral finite elements with quadrature-based integration
- **FFT solvers**: Matrix-free conjugate gradient solvers accelerated by FFT (via muGrid)
- **MPI parallelism**: Distributed computing support through muGrid's MPI infrastructure
- **Microstructure library**: Built-in parametric geometries (laminates, inclusions, lattices, etc.)

 
### Dependencies

- Python >= 3.10
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [muGrid](https://github.com/muSpectre/muGrid)
- [NuMPI](https://github.com/muSpectre/NuMPI)
- [mpi4py](https://mpi4py.readthedocs.io/)
- [JAX](https://github.com/google/jax)
- [matplotlib](https://matplotlib.org/)
 
## Installation

To install `muFFTTO`, you can use `pip`:

```bash
git clone https://github.com/imtek-simulation/muFFTTO.git
cd muFFTTO
pip install .
```

For development, you can install it in editable mode:

```bash
pip install -e .
```

### Installing dependencies

The core dependencies `numpy` and `scipy` will be installed automatically.
  `muGrid`  must be installed separately. You can install them from source:

```bash 
pip install git+https://github.com/muSpectre/muGrid.git
```
 
 

## Package Structure

| Module | Description |
|--------|-------------|
| `domain.py` | Core classes `PeriodicUnitCell` and `Discretization` for setting up the computational domain, fields, and operators |
| `topology_optimization.py` | Objective functions, sensitivity analysis (adjoint method), and phase-field potentials for elasticity |
| `topology_optimization_conductivity.py` | Topology optimization for thermal conductivity problems |
| `solvers.py` | Preconditioned conjugate gradient (PCG) and Adam optimizer for solving linear systems and optimization problems |
| `solvers_nonlinear.py` | Nonlinear solver implementations for advanced optimization |
| `discretization_library.py` | Shape function gradient matrices for various element types (linear triangles, bilinear rectangles, trilinear hexahedra) |
| `material_models.py` | Abstract base classes and implementations for material constitutive models |
| `microstructure_library.py` | Parametric geometry definitions for generating periodic microstructures |
| `grid_adaptation_methods.py` | Methods for adaptive mesh refinement and grid adaptation |
| `grid_adaptation_arbitrary.py` | Arbitrary grid adaptation strategies |
| `analytical_grid_adaptation.py` | Analytical solutions for grid adaptation |
| `tensor_operations.py` | Utility functions for tensor operations and indexing |
| `visualization_utils.py` | Visualization and post-processing utilities |

## Examples

The `examples/` directory contains working examples organized by topic:

**Homogenization** (`examples/homogenization/`)
- Thermal conductivity: 2D and 3D examples with various discretization methods
- Small-strain elasticity: 2D and 3D homogenization with FEM

**Topology Optimization** (`examples/topology_optimization/`)
- 2D and 3D topology optimization for elasticity and conductivity
- Grid tiling and adaptive refinement strategies

**Analytical Solutions** (`examples/analytical_solutions/`)
- Validation against Hashin analytical bounds for composite materials

**Grid Adaptation** (`examples/grid_adaptation/`)
- Examples of mesh refinement and coordinate transformations

**Internal Contact** (`examples/internal_contact/`)
- Contact mechanics and third-medium interaction problems

 

## Authors

- Martin Ladecky (University of Freiburg)
- Lars Pastewka (University of Freiburg)


## License

MIT - see [LICENSE.md](LICENSE.md)

## Funding

This development has received funding from the European Commission (Marie Sklodowska-Curie Fellowship 101106585 — microFFTTO).

