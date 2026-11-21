# GPU: `GenericLinearOperator` apply/transpose give wrong results when a pixel has more than one nodal sub-point

## Summary

On a GPU (CUDA) build, `GenericLinearOperator.apply` and `GenericLinearOperator.transpose`
give wrong results whenever the nodal field has **more than one nodal sub-point per pixel**
(`set_nb_sub_pts("nodal_points", n)` with `n > 1`). With one nodal sub-point per pixel, the
GPU result is correct to machine precision. The CPU path is correct in all cases.

The cause is the nodal index of the Structure-of-Arrays sparse operator in
`src/libmugrid/operators/generic.cc`. That operator is only built for device memory. Its
nodal index uses the Array-of-Structures interleaving (`pixel * nb_nodes + node`) instead
of the SoA plane offset (`node * nb_pixels + pixel`). The two only agree when
`nb_nodes == 1`, which is why the existing tests don't catch it.

## Environment

- muGrid `1.4.2.dev6+gf83ce5876`, built from `main` at `f83ce5876285dc44262a8ebcb151fdcbb34e3f47`
  with `-DMUGRID_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120`, MPI and PnetCDF on
- CUDA 12.9 (conda-forge), GCC 14.4, NVIDIA driver 595.91
- GPU: NVIDIA RTX PRO 1000 Blackwell Laptop GPU (compute capability 12.0), 8 GB
- Python 3.13, NumPy 2.5, CuPy 14.2
- Ubuntu 26.04

## Minimal reproduction

```python
import numpy as np
import cupy as cp
import muGrid
from muGrid import GenericLinearOperator

nb_grid = (8, 8)
rng = np.random.default_rng(1)

for nb_nodes, nb_quad in [(1, 4), (2, 3), (4, 9)]:
    stencil = rng.standard_normal((2, nb_quad, nb_nodes, 2, 2))   # [d, q, n, i, j]
    u = rng.standard_normal((1, nb_nodes, *nb_grid))
    out = {}
    for device in ("cpu", "gpu"):
        fft = muGrid.FFTEngine(nb_domain_grid_pts=nb_grid, nb_ghosts_left=[1, 1],
                               nb_ghosts_right=[1, 1], device=device)
        fc = fft.real_space_collection
        fc.set_nb_sub_pts("quad_points", nb_quad)
        fc.set_nb_sub_pts("nodal_points", nb_nodes)
        nodal = fc.real_field("nodal", (1,), "nodal_points")
        quad = fc.real_field("quad", (1, 2), "quad_points")
        nodal.s[...] = cp.asarray(u) if device == "gpu" else u
        fft.communicate_ghosts(nodal)
        GenericLinearOperator([0, 0], stencil).apply(nodal_field=nodal,
                                                     quadrature_point_field=quad)
        out[device] = cp.asnumpy(quad.s) if device == "gpu" else np.array(quad.s)
    print(f"nb_nodes={nb_nodes}: max|cpu - gpu| = {np.abs(out['cpu'] - out['gpu']).max():.3e}")
```

Output:

```
nb_nodes=1: max|cpu - gpu| = 0.000e+00
nb_nodes=4: max|cpu - gpu| = 1.792e+01      (values are O(15))
nb_nodes=2: max|cpu - gpu| = 1.295e+01
```

Compared with a plain NumPy stencil
`q[c,d,q,x,y] = sum_{n,i,j} S[d,q,n,i,j] u[c,n,x+i,y+j]` (periodic):

| nb_nodes | CPU max error | GPU max error |
|---|---|---|
| 1 | 8.9e-16 | 8.9e-16 |
| 2 | 1.3e-15 | 1.2e+01 |
| 4 | 3.6e-15 | 1.8e+01 |

## What triggers it

From the test below, which varies one parameter at a time on the GPU:

| Parameter | Affects the result? |
|---|---|
| nodal sub-points per pixel `> 1` | **yes**: 100% of the output entries are wrong, for both `apply` and `transpose` |
| number of quadrature points (3, 4, 9) | no |
| number of field components (1, 3) | no |
| transpose `weights` given or not | no |
| CPU (host) memory | never affected |

## Where to look

`src/libmugrid/operators/generic.cc`. `create_apply_operator<storage_order>` and
`create_transpose_operator<storage_order>` build the sparse operator. The
`StructureOfArrays` branch is only used on the device, through
`get_device_apply_operator` / `get_device_transpose_operator` in `generic.hh:682ff` with
`DeviceSpace::storage_order`. The host always uses `ArrayOfStructures` (`generic.cc:376`,
`generic.cc:398`).

The actual device layout of a nodal field with `c` components, `n` nodal sub-points and
`npix` pixels (including ghosts) is `comp * (n * npix) + node * npix + pixel`. CuPy reports
it this way: `.sg` with shape `(3, 4, 12, 8)` has element strides `(384, 96, 1, 12)`. So
every nodal sub-point is a separate plane of `npix` pixels. `compute_traversal_params`
assumes the same layout (comment at `generic.cc:337-340`), and the quadrature index in the
same branch follows it correctly. The nodal index does not:

**apply**, `generic.cc:479-480`:

```cpp
index_diff_nodal = i_component * total_nodal_elements
                   + pixel_count * this->nb_pixel_input_components_ + i_node;   // AoS interleaving
```

`pixel_count * nb_nodes + i_node` treats the nodes as interleaved within a pixel. It should
be the node plane plus the pixel offset: `i_node * total_pixels + pixel_count`.

**transpose**, `generic.cc:585`:

```cpp
index_diff_nodal = i_component * total_nodal_elements + i_node;   // missing the plane stride
```

Here the offset to node plane `i_node` should be `i_node * total_pixels`, not `i_node`.

For `nb_nodes == 1`, `i_node` is always 0, so both expressions coincide with the correct
ones. That explains why the bug only shows up with several nodal sub-points.

### How the diagnosis was verified

Before rebuilding, I emulated the device kernel
(`apply_convolution_kernel_impl` in `convolution_kernels_gpu.hh`) in NumPy on the actual
SoA buffer of the GPU field, with both index formulas:

| nb_nodes, nb_quad, nb_comp | GPU vs emulated current formula | GPU vs emulated corrected formula | emulated corrected formula vs NumPy reference |
|---|---|---|---|
| 1, 4, 1 | 8.9e-16 | 8.9e-16 | 0 |
| 2, 3, 1 | 1.8e-15 | 1.2e+01 | 0 |
| 4, 9, 1 | 7.1e-15 | 1.8e+01 | 0 |
| 4, 9, 3 | 7.1e-15 | 1.8e+01 | 0 |

So the GPU output is reproduced exactly by the current index formula, and the corrected
formula gives the right answer. I checked this for `apply` only. For `transpose`, the
same mistake is visible in the code, and the test below fails, but I didn't emulate it.

## Proposed fix

Compiled and tested on the build above (results below the diff):

```diff
--- a/src/libmugrid/operators/generic.cc
+++ b/src/libmugrid/operators/generic.cc
@@ -479,2 +479,2 @@ GenericLinearOperator::create_apply_operator
                                     index_diff_nodal = i_component * total_nodal_elements
-                                                       + pixel_count * this->nb_pixel_input_components_ + i_node;
+                                                       + i_node * total_pixels + pixel_count;
@@ -585,1 +585,2 @@ GenericLinearOperator::create_transpose_operator
-                                    index_diff_nodal = i_component * total_nodal_elements + i_node;
+                                    index_diff_nodal = i_component * total_nodal_elements
+                                                       + i_node * total_pixels;
```

The quadrature indices in both branches already use `i_quad * total_pixels` and stay as
they are.

Results with the fix applied (same machine, same build options):

- The new test: all 36 cases pass (18 CPU, 18 GPU); before the fix 12 GPU cases failed.
- The existing Python test suite: 806 passed, 13 skipped on 1 rank; 769 passed, 32 skipped
  on 2 MPI ranks (the counts include the 36 new tests). No regressions.
- muFFTTO, biquadratic elements (4 nodes per pixel), CPU vs GPU: gradient, system matrix,
  Green preconditioner and its application now agree to <= 3e-14 (before: O(1) errors and
  NaN in the preconditioner). The biquadratic homogenization example runs on the GPU and
  matches the CPU result to 5e-11 relative.

## Test

A test that runs on CPU and GPU in the style of `tests/python_*_gpu_tests.py`. It uses
the helpers in `tests/conftest.py`. On the build above: all 18 CPU cases pass; on the GPU
the 6 cases with one nodal sub-point pass, and all 12 with two or four fail. Every
element of the output is wrong in those, with errors between 9.5 and 53.

<details>
<summary><code>tests/python_generic_operator_multinode_gpu_tests.py</code></summary>

```python
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file    python_generic_operator_multinode_gpu_tests.py

@brief   GenericLinearOperator apply/transpose with several nodal sub-points
         per pixel, compared against a plain NumPy reference on CPU and GPU.

A pixel carrying more than one nodal sub-point (e.g. biquadratic finite
elements: 4 nodes per pixel) exercises the nodal sub-point index of the
sparse operator. With a single nodal sub-point that index is always zero,
so the existing tests do not cover it.
"""

import numpy as np
import pytest

from conftest import cp, get_test_devices, skip_if_gpu_unavailable

import muGrid
from muGrid import GenericLinearOperator

NB_GRID = (8, 6)


def _setup(device, nb_nodes, nb_quad, nb_comp):
    """Periodic FFT engine with one ghost layer, nodal and quadrature fields."""
    fft = muGrid.FFTEngine(nb_domain_grid_pts=NB_GRID, nb_ghosts_left=[1, 1],
                           nb_ghosts_right=[1, 1], device=device)
    fc = fft.real_space_collection
    fc.set_nb_sub_pts("quad_points", nb_quad)
    fc.set_nb_sub_pts("nodal_points", nb_nodes)
    nodal = fc.real_field("nodal", (nb_comp,), "nodal_points")
    quad = fc.real_field("quad", (nb_comp, 2), "quad_points")
    return fft, nodal, quad


def _to_host(a, device):
    return cp.asnumpy(a) if device == "gpu" else np.array(a)


def _to_device(a, device):
    return cp.asarray(a) if device == "gpu" else a


@pytest.mark.parametrize("device", get_test_devices())
@pytest.mark.parametrize("nb_comp", [1, 3])
@pytest.mark.parametrize("nb_nodes, nb_quad", [(1, 4), (2, 3), (4, 9)])
def test_apply_multinode(device, nb_nodes, nb_quad, nb_comp):
    """apply: q[c,d,q,x,y] = sum_{n,i,j} S[d,q,n,i,j] u[c,n,x+i,y+j] (periodic)"""
    skip_if_gpu_unavailable(device)
    rng = np.random.default_rng(1)
    stencil = rng.standard_normal((2, nb_quad, nb_nodes, 2, 2))  # [d, q, n, i, j]
    u = rng.standard_normal((nb_comp, nb_nodes, *NB_GRID))

    expected = np.zeros((nb_comp, 2, nb_quad, *NB_GRID))
    for i in range(2):
        for j in range(2):
            expected += np.einsum("dqn,cnxy->cdqxy", stencil[..., i, j],
                                  np.roll(u, (-i, -j), axis=(-2, -1)))

    fft, nodal, quad = _setup(device, nb_nodes, nb_quad, nb_comp)
    nodal.s[...] = _to_device(u, device)
    fft.communicate_ghosts(nodal)
    GenericLinearOperator([0, 0], stencil).apply(nodal_field=nodal,
                                                 quadrature_point_field=quad)

    np.testing.assert_allclose(_to_host(quad.s, device), expected,
                               rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("device", get_test_devices())
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("nb_comp", [1, 3])
@pytest.mark.parametrize("nb_nodes, nb_quad", [(1, 4), (2, 3), (4, 9)])
def test_transpose_multinode(device, nb_nodes, nb_quad, nb_comp, weighted):
    """transpose: u[c,n,x,y] = sum_{d,q,i,j} w_q S[d,q,n,i,j] q[c,d,q,x-i,y-j]"""
    skip_if_gpu_unavailable(device)
    rng = np.random.default_rng(2)
    stencil = rng.standard_normal((2, nb_quad, nb_nodes, 2, 2))
    g = rng.standard_normal((nb_comp, 2, nb_quad, *NB_GRID))
    weights = rng.uniform(0.5, 1.5, nb_quad) if weighted else np.ones(nb_quad)

    expected = np.zeros((nb_comp, nb_nodes, *NB_GRID))
    for i in range(2):
        for j in range(2):
            expected += np.einsum("dqn,q,cdqxy->cnxy", stencil[..., i, j], weights,
                                  np.roll(g, (i, j), axis=(-2, -1)))

    fft, nodal, quad = _setup(device, nb_nodes, nb_quad, nb_comp)
    quad.s[...] = _to_device(g, device)
    fft.communicate_ghosts(quad)
    GenericLinearOperator([0, 0], stencil).transpose(
        quadrature_point_field=quad, nodal_field=nodal,
        weights=list(weights) if weighted else None)

    np.testing.assert_allclose(_to_host(nodal.s, device), expected,
                               rtol=1e-12, atol=1e-12)
```

</details>

Failures on the GPU (`pytest -k gpu`):

```
test_apply_multinode[2-3-1-gpu]            Mismatched 288/288 (100%)  max abs diff  9.54
test_apply_multinode[2-3-3-gpu]            Mismatched 864/864 (100%)  max abs diff 12.80
test_apply_multinode[4-9-1-gpu]            Mismatched 864/864 (100%)  max abs diff 14.83
test_apply_multinode[4-9-3-gpu]            Mismatched 2592/2592       max abs diff 19.13
test_transpose_multinode[2-3-1-False-gpu]  Mismatched 96/96   (100%)  max abs diff 11.86
test_transpose_multinode[2-3-1-True-gpu]   Mismatched 96/96   (100%)  max abs diff  9.51
test_transpose_multinode[2-3-3-False-gpu]  Mismatched 288/288 (100%)  max abs diff 17.35
test_transpose_multinode[2-3-3-True-gpu]   Mismatched 288/288 (100%)  max abs diff 17.51
test_transpose_multinode[4-9-1-False-gpu]  Mismatched 192/192 (100%)  max abs diff 50.72
test_transpose_multinode[4-9-1-True-gpu]   Mismatched 192/192 (100%)  max abs diff 53.49
test_transpose_multinode[4-9-3-False-gpu]  Mismatched 576/576 (100%)  max abs diff 50.72
test_transpose_multinode[4-9-3-True-gpu]   Mismatched 576/576 (100%)  max abs diff 43.78
```

## Impact

Any finite-element discretization with more than one node per pixel, such as biquadratic
elements, gets wrong gradients and divergences on the GPU, and nothing reports an error.
In the downstream code [muFFTTO](https://github.com/imtek-simulation/muFFTTO), the
biquadratic conductivity homogenization example
(`examples/homogenization/conductivity/example_2D_homogenization_conductivity_quadratic_basis.py`)
produces NaN in the Green preconditioner, and `muGrid.Solvers.conjugate_gradients` then
stops with `ConvergenceError: the curvature p^T A p became NaN at iteration 0`. The same
script runs correctly on the CPU. All linear-element examples (one node per pixel) give
identical results on CPU and GPU.
