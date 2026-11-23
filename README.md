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

Everything installs with `pip` into a virtual environment (Python >= 3.10):

```bash
python3 -m venv venv
source venv/bin/activate
git clone https://github.com/imtek-simulation/muFFTTO.git
cd muFFTTO
pip install -e . openmpi
```

`pip install -e .` installs muFFTTO in editable mode (changes to the code take effect without reinstalling) with all
dependencies: numpy, scipy, muGrid, NuMPI, mpi4py, JAX and matplotlib.

**MPI.** `mpi4py` from pip needs an MPI library at run time; `openmpi` above installs one from PyPI. Without any MPI
library, `import muFFTTO` fails with `RuntimeError: cannot load MPI library`. On a cluster you can use the cluster's
MPI instead: leave out `openmpi` and load the MPI module before running, e.g. on NEMO `module load mpi/openmpi`.
The PyPI `muGrid` is built without MPI, so a pip install runs serially (one process); parallel runs need muGrid built
from source against the cluster's MPI.

**Direct solvers** (`muFFTTO.direct_solvers`, `solvers_nonlinear.trust_region_exact`) work with SciPy out of the box.
Faster backends are used automatically when installed: MUMPS (`conda install -c conda-forge python-mumps`, not
available on pip) or, on an NVIDIA GPU, cuDSS (`pip install "nvmath-python[cu12]" nvidia-cudss-cu12 cupy-cuda12x`).

**GPU.** The PyPI `muGrid` runs on the CPU; for GPU runs install a CUDA build of muGrid
([muGrid](https://github.com/muSpectre/muGrid)).

### Example: a fresh install on NEMO (bwForCluster)

```bash
module load lang/python/3.12.3-gcccore-13.3.0     # do not 'module purge' afterwards: venv would get Python 3.9
cd /work/classic/<your workspace>                  # the home directory is small: install and run in the workspace
python3 -m venv venv && source venv/bin/activate
git clone https://github.com/imtek-simulation/muFFTTO.git
cd muFFTTO && pip install --no-cache-dir -e . openmpi
python -c "import muFFTTO; from muFFTTO import direct_solvers; print(direct_solvers.symmetric_solver().name)"
```

Run on a compute node (`sbatch`), not on the login node; in the job script load the same Python module and
`source venv/bin/activate`. The login node prints a harmless `pml_ucx ... Failed to create UCP worker` message from
the PyPI Open MPI; `export OMPI_MCA_pml=ob1` silences it.


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
| `geometry.py` | Periodic microstructure geometries: composable shapes, phase fields and named geometries (MPI-parallel) |
| `microstructure_library.py` | Backward-compatible `get_geometry` wrapper used by the examples |
| `grid_adaptation_methods.py` | Methods for adaptive mesh refinement and grid adaptation |
| `grid_adaptation_arbitrary.py` | Arbitrary grid adaptation strategies |
| `analytical_grid_adaptation.py` | Analytical solutions for grid adaptation |
| `tensor_operations.py` | Utility functions for tensor operations and indexing |
| `visualization_utils.py` | Visualization and post-processing utilities |
| `io_utils.py` | Save/load fields to NetCDF (MPI-parallel, muGrid `FileIONetCDF`): snapshots, time series, numpy reader |

## Documentation

The [`docs/`](docs/README.md) folder explains the mathematics behind the code
and how each example works:

- [Theory and numerics](docs/theory.md): the periodic cell problem, FE discretization on a pixel grid, the FFT-preconditioned CG solver, effective properties, adjoint topology optimization and grid adaptation
- Example walkthroughs in [`docs/examples/`](docs/examples/), one page per example folder
- [Known issues](docs/known_issues.md): suspected bugs found while documenting the code

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

**Phase-Field Fracture** (`examples/phase_field_fracture/`)
- AT2 brittle fracture of a periodic porous cell under macroscopic strain (staggered scheme, history field)

 

## Authors

- Martin Ladecky (University of Freiburg)
- Lars Pastewka (University of Freiburg)


## License

MIT - see [LICENSE.md](LICENSE.md)

## Funding

This development has received funding from the European Commission (Marie Sklodowska-Curie Fellowship 101106585 — microFFTTO).

