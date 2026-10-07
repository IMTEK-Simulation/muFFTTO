#
# Copyright 2023 Martin Ladecky
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""
The entry point for the code

muFFTTO -- FFT-accelerated finite-element homogenization and topology
optimization of periodic microstructures.

The package discretizes a periodic unit cell on a regular (optionally
adapted/deformed) grid, solves the cell problem of linear conductivity or
(small/finite strain) elasticity matrix-free with preconditioned conjugate
gradients (FFT-based preconditioners via muFFT, fields stored in muGrid), and
computes objective functions and adjoint-based sensitivities for
phase-field topology optimization. Parallel execution uses MPI (mpi4py, NuMPI).

Submodules
----------
domain
    ``PeriodicUnitCell`` and ``Discretization``: grids, fields, gradient /
    interpolation operators, right-hand sides, homogenized quantities and
    preconditioners.
discretization_library
    Finite-element shape-function / gradient-operator definitions (stencils)
    for the supported element types.
solvers, solvers_nonlinear
    Matrix-free (preconditioned) conjugate-gradient solvers and a Newton-CG
    solver for finite-strain hyperelasticity.
material_models
    Constitutive models (stress and algorithmic tangent) for elasticity.
tensor_operations
    Pointwise tensor algebra (contractions, inverse, determinant, trace) on
    muGrid fields with layout ``[i, j, q, x, y, z]``.
geometry
    Periodic microstructure geometries: composable shapes (box, ball, set
    operations), phase fields from shapes, and a registry of named geometries.
    Works point-wise on local pixel coordinates, so it is MPI-parallel.
microstructure_library
    Backward-compatible ``get_geometry`` wrapper around :mod:`geometry`.
topology_optimization
    Objectives, phase-field (double-well + gradient) regularization and
    adjoint sensitivities for elasticity topology optimization.
topology_optimization_conductivity
    Flux-matching objective and adjoint sensitivities for conductivity
    topology optimization.
analytical_grid_adaptation, grid_adaptation_methods_Zecevic,
grid_adaptation_arbitrary
    Grid adaptation (conformal-grid / spring-relaxation, after Zecevic et al.)
    that moves grid nodes onto material interfaces.
otsu
    Otsu-threshold image segmentation and edge/region detection used to
    obtain phase indicators for grid adaptation.
visualization_utils
    Matplotlib helpers for plotting fields on (deformed) 2D grids.
"""

__version__ = '0.0.1'
