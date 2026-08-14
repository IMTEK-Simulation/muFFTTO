# muFFTTO

FFT-based micro-scale topology optimization on periodic unit cells using the [muGrid](https://github.com/muSpectre/muGrid) / [muFFT](https://github.com/muSpectre/muFFT) framework.

## Features

- **Homogenization**: Compute effective material properties (conductivity and elasticity) of periodic microstructuresclass material_model_elasticity:
    def __init__(self, name='my_basic_material'):
        self.name = name  # Name of the cell

    def get_stress(self, strain_ijqxyz, stress_ijqxyz):
        """
          Function to calculate stress from a given strain. We use index notation with
          full tensors without Voight notation

          Parameters
          ----------
          strain_ijqxyz: muGrid array with shape [i,j,q,x,y,z] representing a global strain field
                        at each quadrature point

          Returns
          -------
          stress_ijqxyz: muGrid array with shape [i,j,q,x,y,z] representing a global stress field
                        at each quadrature point
          """


    def get_algorithmic_tangent(self, strain_ijqxyz, tangent_ijklqxyz):
        """
          Function to calculate stress from a given strain. We use index notation with
          full tensors without Voight notation

          Parameters
          ----------
          strain_ijqxyz: muGrid array with shape [i,j,q,x,y,z] representing a global strain field
                        at each quadrature point

          Returns
          -------
          tangent_ijklqxyz: muGrid array with shape [i,j,q,x,y,z] representing a global algorithmic tangent field
                        at each quadrature point
          """
        stress_ijqxyz=5
        tangent_ijklqxyz=stress_ijqxyz/strain_ijqxyz
- **Topology optimization**: Phase-field based optimization with adjoint sensitivity analysis
- **Multi-dimensional**: Supports 1D, 2D, and 3D problems
- **FEM discretization**: Triangular and hexahedral finite elements with quadrature-based integration
- **FFT solvers**: Matrix-free conjugate gradient solvers accelerated by FFT (via muGrid)
- **MPI parallelism**: Distributed computing support through muGrid's MPI infrastructure
- **Microstructure library**: Built-in parametric geometries (laminates, inclusions, lattices, etc.)

 
### Dependencies

- Python >= 3.8
- [numpy](https://numpy.org/), [scipy](https://scipy.org/)
- [muGrid](https://github.com/muSpectre/muGrid) and [muFFT](https://github.com/muSpectre/muFFT) (must be installed separately)

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

The core dependencies `numpy` and `scipy` will be installed automatically. However, `muGrid` and `muFFT` must be installed separately as they are not currently available on PyPI. You can install them from source:

```bash
pip install git+https://github.com/muSpectre/muFFT.git
pip install git+https://github.com/muSpectre/muGrid.git
```

If you need MPI support, you can install the optional dependencies:

```bash
pip install .[mpi]
```

## Building the Package

To build the source distribution and wheel, you can use `flit`:

```bash
pip install flit
flit build
```

The built files will be located in the `dist/` directory.

 

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

 

## Authors

- Martin Ladecky (University of Freiburg)
- Lars Pastewka (University of Freiburg)


## License

MIT - see [LICENSE.md](LICENSE.md)

## Funding

This development has received funding from the European Commission (Marie Sklodowska-Curie Fellowship 101106585 — microFFTTO).

