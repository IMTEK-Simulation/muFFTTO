"""
Geometric multigrid preconditioner for periodic small-strain elasticity with a scaled material ``A(x) = s(x) C_0``.

A symmetric V(nu, nu)-cycle with Chebyshev-Jacobi smoothing, to be used as the preconditioner of the conjugate
gradient solver. It is the field-based, MPI-capable successor of ``mg_lib.py`` of the multigrid study (repository muFFTTO-multigrid)
(same hierarchy, same smoother, same coarse solve -- the same iteration counts), written for large 3D grids:

* **No copies, no temporaries.** Every vector of the cycle is a muGrid field owned by its level; updates are fused
  muGrid ``linalg`` kernels (``axpy``, ``axpby``, ``copy``) or in-place array operations. The fine level works on the
  caller's ``Discretization`` and shares its scratch fields (gradient, hourglass), so the multigrid adds four nodal
  vectors and a diagonal on the fine grid (120 B per voxel in 3D) and about 1/7 of that again on the coarse grids.
* **MPI.** Every level is a ``Discretization`` on the same communicator. muGrid's FFT engine splits the grid into
  slabs along the last axis; as long as every rank keeps an even number of planes the levels are nested and the
  grid transfer (``muGrid.GridTransfer``) is local up to a ghost exchange. When the slabs become too thin the rest
  (or the grid small, ``n_agglomerate``) the rest of the hierarchy is *agglomerated*: the coarse problem is
  gathered on rank 0 and solved there by a serial multigrid (class :class:`Multigrid` again, on ``MPI.COMM_SELF``).

Hierarchy
---------
Grids N, N/2, ..., ``n_coarsest`` (N = n_coarsest * 2^k).

``P``      multilinear interpolation of the nested nodal spaces, ``R = P^T`` (``muGrid.GridTransfer``)
``A_H``    re-discretisation: the muFFTTO stiffness on the coarse grid with the material averaged over the fine
           voxels. 3D hexahedra: the Gauss point in octant (i, j, k) of a coarse voxel takes the mean scale of the
           fine voxel (2I+i, 2J+j, 2K+k). 2D linear triangles: the mean over the four fine triangles of a coarse
           triangle (this is the exact Galerkin operator P^T A_h P).
           One-point hexahedra (``trilinear_hexahedron_1Q``, ``_1Q_hg``) are used on the fine level only; all coarse
           levels are fully integrated, so they carry no hourglass modes.
smoother   ``degree`` Chebyshev steps in D^-1 A on [lambda_max / alpha, lambda_max], D = diag(A);
           lambda_max: a GUARANTEED upper bound of lambda_max(D^-1 A) (default, see :meth:`Multigrid._lmax_bound`),
           or -- ``lmax='power'``, the old behaviour -- a power-iteration estimate times 1.1 (not safe: a warm
           start can miss a new localised mode, e.g. a crack; 1000-12000 CG iterations in an AT2 run)
coarsest   exact: dense pseudo-inverse of the ``n_coarsest`` grid operator

The cycle is a fixed SPD operator, so it plugs into CG with any stopping test.

Usage
-----
>>> mg = Multigrid(discretization, C_0)
>>> mg.scale.s[0, 0] = s                      # the material scale at the quadrature points, A = s C_0
>>> mg.setup()                                # after every change of the material
>>> solve(K=lambda x, Ax: discretization.apply_system_matrix_mugrid(mg.material, x, Ax, 'small_strain'),
...       M=mg.apply)                         # mg.apply(r, z): z = V-cycle(r)
"""
import time

import numpy as np
import muGrid
from mpi4py import MPI
from muGrid import linalg

from . import domain, tensor_operations
from .material_data import ScaledMaterial

FULLY_INTEGRATED = {'trilinear_hexahedron_1Q': 'trilinear_hexahedron',
                    'trilinear_hexahedron_1Q_hg': 'trilinear_hexahedron'}


# CuPy kernels of the 3D grid transfer for device fields. `i` runs over the buffer of the target field
# (component, z, y, x with x fastest; ghost points are skipped); `source` is the buffer of the other field.
_TRANSFER_INDEX = '''
    long long point = i % points, component = i / points;
    long long x = point % shape_x - ghost_x;
    long long y = (point / shape_x) % shape_y - ghost_y;
    long long z = point / (shape_x * shape_y) - ghost_z;
    if (x >= 0 && x < size_x && y >= 0 && y < size_y && z >= 0 && z < size_z) {
        const T* values = &source[component * other_points];
        long long nx = other_shape_x, nxy = other_shape_x * other_shape_y;
        T sum = 0;
'''
_TRANSFER_KERNELS = {
    'restrict': _TRANSFER_INDEX + '''
        long long x0 = other_ghost_x + 2 * x, y0 = other_ghost_y + 2 * y, z0 = other_ghost_z + 2 * z;
        for (int c = -1; c <= 1; c++) for (int b = -1; b <= 1; b++) for (int a = -1; a <= 1; a++)
            sum += (a ? 0.5 : 1.0) * (b ? 0.5 : 1.0) * (c ? 0.5 : 1.0) * values[(x0 + a) + nx * (y0 + b) + nxy * (z0 + c)];
        target[i] = sum;
    }''',
    'prolong': _TRANSFER_INDEX + '''
        int px = x & 1, py = y & 1, pz = z & 1;
        long long x0 = other_ghost_x + x / 2, y0 = other_ghost_y + y / 2, z0 = other_ghost_z + z / 2;
        for (int c = 0; c <= pz; c++) for (int b = 0; b <= py; b++) for (int a = 0; a <= px; a++)
            sum += values[(x0 + a) + nx * (y0 + b) + nxy * (z0 + c)];
        target[i] = sum / ((1 + px) * (1 + py) * (1 + pz));
    }'''}


class GridTransfer:
    """Multilinear interpolation ``P`` between two nested nodal grids and its transpose ``R = P^T``.

    Host fields use ``muGrid.GridTransfer``. Where the muGrid build has no kernel for device fields, 3D device
    fields use one CuPy kernel per transfer (:data:`_TRANSFER_KERNELS`); the same stencils written as in-place array
    operations on strided views (27 small updates, no full-size temporaries) remain as the reference and for 2D. The caller exchanges the ghosts of the *input* field first: ``restrict`` reads the left ghost
    layer of the fine field, ``prolong`` the right ghost layer of the coarse field.
    """

    def __init__(self, dim):
        self.dim = dim
        self.kernel = muGrid.GridTransfer(dim)
        self.use_kernel = True
        self.use_device_kernel = True
        self._device_kernels = {}

    def restrict(self, fine, coarse):
        """coarse = P^T fine: per axis, coarse[I] = fine[2I] + (fine[2I - 1] + fine[2I + 1]) / 2"""
        if self._try_kernel(self.kernel.restrict, fine, coarse):
            return
        if self.dim == 3 and self.use_device_kernel and fine.is_on_gpu:
            return self._device_kernel('restrict', fine, coarse)
        source, target = fine.sg, coarse.s
        first = True
        for shift in np.ndindex(*(3,) * self.dim):                       # shift - 1 in {-1, 0, 1} per axis
            weight = 0.5 ** sum(1 for a in shift if a != 1)
            view = source[self._window(fine, target.shape, [a - 1 for a in shift], step=2)]
            if first:
                target[...] = view
                target *= weight
                first = False
            else:
                target += weight * view

    def prolong(self, coarse, fine):
        """fine = P coarse: per axis, fine[2I] = coarse[I], fine[2I + 1] = (coarse[I] + coarse[I + 1]) / 2"""
        if self._try_kernel(self.kernel.prolong, coarse, fine):
            return
        if self.dim == 3 and self.use_device_kernel and fine.is_on_gpu:
            return self._device_kernel('prolong', fine, coarse)
        source, target = coarse.sg, fine.s
        coarse_shape = coarse.s.shape
        for parity in np.ndindex(*(2,) * self.dim):                      # fine nodes 2I + parity
            nodes = target[(Ellipsis,) + tuple(slice(a, None, 2) for a in parity)]
            corners = list(np.ndindex(*tuple(a + 1 for a in parity)))    # coarse neighbours I + corner
            for number, corner in enumerate(corners):
                view = source[self._window(coarse, coarse_shape, corner, step=1)]
                if number == 0:
                    nodes[...] = view
                else:
                    nodes += view
            if len(corners) > 1:
                nodes *= 1.0 / len(corners)

    def _device_kernel(self, name, fine, coarse):
        """3D transfer of device fields as one CuPy kernel over the buffer of the target field (one launch instead
        of 27 strided array updates). Buffers are component by component with x fastest, ghost layers included."""
        fine_buffer = tensor_operations.point_view(fine).reshape(-1)
        coarse_buffer = tensor_operations.point_view(coarse).reshape(-1)
        xp = tensor_operations.get_array_module(fine_buffer)
        if name not in self._device_kernels:
            self._device_kernels[name] = xp.ElementwiseKernel(
                'raw T source, int64 points, int64 other_points, int64 shape_x, int64 shape_y, int64 ghost_x, '
                'int64 ghost_y, int64 ghost_z, int64 size_x, int64 size_y, int64 size_z, int64 other_shape_x, '
                'int64 other_shape_y, int64 other_ghost_x, int64 other_ghost_y, int64 other_ghost_z',
                'raw T target', _TRANSFER_KERNELS[name], 'grid_transfer_' + name)
        target, source = (coarse, fine) if name == 'restrict' else (fine, coarse)
        target_buffer, source_buffer = ((coarse_buffer, fine_buffer) if name == 'restrict'
                                        else (fine_buffer, coarse_buffer))
        shape, other_shape = target.sg.shape[-3:], source.sg.shape[-3:]
        ghosts, other_ghosts = target._cpp.collection.nb_ghosts_left, source._cpp.collection.nb_ghosts_left
        self._device_kernels[name](
            source_buffer, int(np.prod(shape)), int(np.prod(other_shape)), shape[0], shape[1], *ghosts,
            *target.s.shape[-3:], other_shape[0], other_shape[1], *other_ghosts, target_buffer,
            size=target_buffer.size)

    def _try_kernel(self, kernel, source, target):
        if self.use_kernel:
            try:
                kernel(source, target)
                return True
            except TypeError:                                            # no overload for device fields
                self.use_kernel = False
        return False

    def _window(self, field, shape, shift, step):
        """index of the ghosted array `field.sg` for the interior nodes `step * I + shift`, I over `shape`"""
        ghosts = field._cpp.collection.nb_ghosts_left
        return (Ellipsis,) + tuple(slice(g + a, g + a + step * n, step)
                                   for g, a, n in zip(ghosts, shift, shape[-self.dim:]))


class Level:
    """One grid of the hierarchy: its discretization, its material ``s C_0`` and its work vectors.

    ``x``, ``b``: solution and right-hand side of the level's problem ``A x = b``; ``r``, ``d``, ``t``: residual,
    Chebyshev direction and scratch; ``dinv``: 1 / diag(A); ``power``: dominant vector of D^-1 A (warm start of
    the power iteration). The fine level takes ``x`` and ``b`` from the caller.

    The operator ``A`` is either muFFTTO's stencil pipeline of the level's element (gradient, material, divergence,
    hourglass control), or -- ``fused=True`` -- muGrid's ``IsotropicStiffnessOperator`` for trilinear hexahedra with
    one material value per voxel: one kernel, no quadrature-point fields, in double or single precision (``dtype``).
    """

    def __init__(self, discretization, C_0, index, own_solution_fields=True, fused=False, dtype=np.float64):
        self.disc = discretization
        self.index = index
        self.dim = discretization.domain_dimension
        self.fused = fused
        self.dtype = np.dtype(dtype)
        self._suffix = '' if self.dtype == np.float64 else '_f32'      # fields of both precisions may coexist
        collection = discretization.field_collection
        self.scale = collection.real_field('mg_scale' + self._suffix, (1, 1), 'quad_points', dtype=self.dtype)
        if fused:
            lam_0, mu_0 = float(C_0[0, 0, 1, 1]), float(C_0[0, 1, 0, 1])
            identity = np.eye(self.dim)
            isotropic = (lam_0 * np.einsum('ij,kl->ijkl', identity, identity) + mu_0 * (
                np.einsum('ik,jl->ijkl', identity, identity) + np.einsum('il,jk->ijkl', identity, identity)))
            assert np.allclose(C_0, isotropic), 'the fused operator needs an isotropic C_0'
            assert self.scale.s.shape[2] == 1, 'the fused operator needs one material value per voxel'
            self.lam_0, self.mu_0 = lam_0, mu_0
            self.lam = collection.real_field('mg_lambda' + self._suffix, dtype=self.dtype)       # s lambda_0 per voxel
            self.mu = collection.real_field('mg_mu' + self._suffix, dtype=self.dtype)            # s mu_0
            operator = muGrid.IsotropicStiffnessOperator3D if self.dim == 3 else muGrid.IsotropicStiffnessOperator2D
            self.operator = operator([float(h) for h in discretization.pixel_size], muGrid.FEMElement.q1)
        else:
            assert self.dtype == np.float64, 'single precision is implemented for the fused operator'
            self.material = ScaledMaterial([(self.scale, C_0)])
            assert self.material.minor_symmetric, 'C_0 must be minor symmetric (C_ijkl = C_ijlk)'
            self.stress = discretization.get_gradient_size_field(name='grad_field_temporary')   # the shared scratch
        self.x = self.vector('mg_x') if own_solution_fields else None
        self.b = self.vector('mg_b') if own_solution_fields else None
        self.r = self.d = self.t = self.dinv = self.power = None
        self.lmax = None

    def vector(self, name):
        """a nodal vector field of this level, in its precision"""
        return self.disc.field_collection.real_field(name + self._suffix, (self.dim,), 'nodal_points',
                                                     dtype=self.dtype)

    def allocate_smoother_fields(self):
        """the vectors of a level that is smoothed (all but the coarsest)"""
        self.r = self.vector('mg_residual')
        self.d = self.vector('mg_direction')
        self.t = self.vector('mg_scratch')
        self.dinv = self.vector('mg_inverse_diagonal')

    def update_material(self):
        """fused operator: the Lame fields s lambda_0, s mu_0 of the current scale (with ghost values)"""
        if self.fused:
            scale = self.scale.s.reshape(self.lam.p.shape)
            self.lam.p[...] = scale * self.lam_0
            self.mu.p[...] = scale * self.mu_0
            self.disc.fft.communicate_ghosts(self.lam)
            self.disc.fft.communicate_ghosts(self.mu)

    def apply_K(self, x, y):
        """y = A x. The ghost layers of y are NOT valid afterwards.

        Fused: one kernel after the ghost exchange of x.  Stencils: B^T W s C_0 : B x (+ hourglass control), as
        ``Discretization.apply_system_matrix_mugrid`` for small strains, with the three ghost exchanges the
        stencils need (x, the stress, the hourglass strain) instead of six or seven. Under MPI every exchange is a
        message, and the cycle applies A ten times per iteration."""
        disc = self.disc
        disc.fft.communicate_ghosts(x)
        if self.fused:
            self.operator.apply(x, self.lam, self.mu, y)
            return
        disc.gradient_op.apply(nodal_field=x, quadrature_point_field=self.stress)       # C_0 is minor symmetric:
        disc.apply_material_data_mugrid(self.material, self.stress)                     # C : grad u = C : sym grad u
        disc.fft.communicate_ghosts(self.stress)
        disc.gradient_op.transpose(quadrature_point_field=self.stress, nodal_field=y, weights=disc.quadrature_weights)
        if disc.hourglass_control is not None:
            disc.hourglass_control.add_stiffness(self.material, x, y, exchange_ghosts=False)

    def global_sum(self, local_value):
        return self.disc.communicator.sum(np.float64(float(local_value)))


class Multigrid:
    """Symmetric V(degree, degree)-cycle for ``A = s(x) C_0`` on ``discretization`` (see the module docstring).

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        The fine grid (elasticity; ``linear_triangles`` in 2D, ``trilinear_hexahedron`` / ``_1Q`` / ``_1Q_hg`` in
        3D). The multigrid allocates its fine-level fields in its field collection and shares its scratch fields.
    C_0 : ndarray [d, d, d, d]
        The constant tensor of the material ``A = s(x) C_0``.
    degree : int
        Chebyshev steps per smoothing (2: the V(2, 2) cycle of the study).
    alpha : float
        Smoothing interval [lambda_max / alpha, lambda_max].
    n_coarsest : int
        Grid of the exact coarse solve.
    coarse_element : str, optional
        Element of the coarse levels. Default: the fully integrated version of the fine element (one-point
        hexahedra on the fine level only). ``'trilinear_hexahedron_1Q_hg'``: one-point hexahedra with hourglass
        control on every level -- a coarse voxel takes the mean scale of its 8 fine voxels; 8x fewer quadrature
        points on the coarse levels (level 1 then costs 1/8 of the fine level instead of about as much).
    coarse_hourglass_eps : float, optional
        Hourglass control of one-point coarse levels (default: that of the fine level; 1 reproduces the hourglass
        energy of the fully integrated element).
    operator : {'stencil', 'fused'}
        'stencil': muFFTTO's operator of the discretization's element on the fine level (default). 'fused': on every
        level muGrid's ``IsotropicStiffnessOperator`` for trilinear hexahedra with ONE material value per voxel --
        identical to ``trilinear_hexahedron`` with a voxel-wise constant scale, but a single kernel without
        quadrature-point fields (about 5x faster than the 8-point stencils). ``discretization`` then only provides
        the grid and the fields and must have one quadrature point per voxel (``trilinear_hexahedron_1Q``);
        ``C_0`` must be isotropic. Coarse voxels take the mean scale of their 8 fine voxels.
    precision : {'double', 'single'}
        'single' (fused operator only): the cycle runs in float32 on its own fields, between the caller's double
        precision vectors. A preconditioner need not be exact, so CG keeps its accuracy and its stopping test; the
        cycle's vectors take half the memory and its kernels move half the bytes.
    n_agglomerate : int
        MPI only: grids of this size and coarser are solved on rank 0 (serial multigrid on its device). On small
        grids a ghost exchange is pure message latency; 16 keeps every distributed level at 32^3 or larger.
    lmax : {'bound', 'power'}
        Upper end of the smoothing interval. 'bound' (default): the guaranteed bound max over the local pieces of
        lambda_max(D_e^-1 A_e) (:meth:`_lmax_bound`) -- valid for any design and contrast because the level material
        is s C_0 with s >= 0 (checked at every setup); computed once per kind of level, no iterations. 'power':
        power-iteration estimate times 1.1 (reproduces ``mg_lib`` and runs before 2026-11-16).
    power_start : callable, optional
        ``power_start(level_index, shape) -> ndarray``: start vector of the power iteration (``lmax='power'``, serial
        runs only; used by the tests to reproduce ``mg_lib``). Default: seeded random numbers per grid plane,
        independent of the MPI decomposition.

    Attributes
    ----------
    scale : muGrid Field [1, 1, q, x, y(, z)]
        The fine-level scale ``s`` at the quadrature points: write it, then call :meth:`setup`.
    material : ScaledMaterial
        ``s C_0`` on the fine level (stencil operator; None for the fused one).
    apply_K : callable
        ``apply_K(x, Ax)``: the fine-level operator the cycle preconditions, in double precision.
    """

    def __init__(self, discretization, C_0, degree=2, alpha=30.0, n_coarsest=4, power_start=None,
                 coarse_element=None, coarse_hourglass_eps=None, n_agglomerate=16, operator='stencil',
                 precision='double', lmax='bound'):
        assert lmax in ('bound', 'power')
        self.lmax = lmax
        self._lmax_bounds = {}                                  # bound per kind of level (see _lmax_bound)
        self.degree = degree
        self.alpha = alpha
        self.C_0 = np.asarray(C_0)
        d2 = self.C_0.shape[0] ** 2                            # the bound needs s C_0 >= 0: C_0 positive semidefinite
        eigenvalues = np.linalg.eigvalsh(0.5 * (self.C_0.reshape(d2, d2) + self.C_0.reshape(d2, d2).T))
        if eigenvalues.min() < -1e-12 * abs(eigenvalues).max():
            raise ValueError('Multigrid: C_0 must be positive semidefinite')
        self.n_coarsest = n_coarsest
        self.power_start = power_start
        self.coarse_element = coarse_element
        self.coarse_hourglass_eps = (coarse_hourglass_eps if coarse_hourglass_eps is not None
                                     else getattr(discretization.hourglass_control, 'eps', None))
        self.n_agglomerate = n_agglomerate
        self.dim = discretization.domain_dimension
        self.comm = discretization.communicator                 # muGrid communicator of all levels
        self.mpi = self.comm.mpi4py_comm if self.comm.size > 1 else MPI.COMM_SELF
        self.transfer = GridTransfer(self.dim)
        assert operator in ('stencil', 'fused') and precision in ('double', 'single')
        self.operator, self.precision = operator, precision
        fused = operator == 'fused'
        dtype = np.float32 if precision == 'single' else np.float64
        # the operator the cycle preconditions, in double precision, on the caller's fields ...
        self.fine_operator = Level(discretization, self.C_0, 0, own_solution_fields=False, fused=fused)
        # ... and the fine level of the cycle: the same object, or its single-precision twin with its own x and b
        self.levels = [self.fine_operator if precision == 'double'
                       else Level(discretization, self.C_0, 0, fused=fused, dtype=dtype)]
        self.tail = None                     # serial multigrid on rank 0 below the last distributed level
        self._tail_slabs = None
        self._coarse_pinv = None
        while self.levels[-1].disc.nb_of_pixels_global[0] > n_coarsest:
            fine = self.levels[-1].disc
            if not self._can_coarsen_distributed(fine):
                break
            element = coarse_element or FULLY_INTEGRATED.get(fine.element_type, fine.element_type)
            if fused:                        # only the grid, the fields and one scale per voxel are used
                element = 'trilinear_hexahedron_1Q'
            coarse = domain.Discretization(
                cell=fine.cell, nb_of_pixels_global=tuple(n // 2 for n in fine.nb_of_pixels_global),
                discretization_type='finite_element', element_type=element, communicator=fine.communicator,
                device=fine.device, low_memory=fine.low_memory,
                hourglass_eps=self.coarse_hourglass_eps if element.endswith('_hg') else None)
            self.levels.append(Level(coarse, self.C_0, len(self.levels), fused=fused, dtype=dtype))
        if self.comm.size > 1:               # the last distributed grid and all coarser ones: serial, on rank 0
            self._build_tail(self.levels[-1].disc)
        for level in self.levels[:-1]:       # the last level is solved exactly (or handed to the tail)
            level.allocate_smoother_fields()
        self.scale = self.fine_operator.scale
        self.material = getattr(self.fine_operator, 'material', None)
        self.apply_K = self.fine_operator.apply_K

    # ------------------------------------------------------------------ hierarchy construction
    def _can_coarsen_distributed(self, disc):
        """True if the next coarser grid is nested on every rank: an even number (>= 4) of planes per rank"""
        local = np.asarray(disc.fft.nb_subdomain_grid_pts)
        ok = (bool(np.all(local % 2 == 0) and np.all(local >= 4)
                   and disc.nb_of_pixels_global[0] // 2 >= self.n_agglomerate) or self.comm.size == 1)
        return bool(self.mpi.allreduce(ok, op=MPI.LAND)) if self.comm.size > 1 else ok

    def _build_tail(self, disc):
        """agglomeration: the grid of `disc` and all coarser ones as a serial multigrid on rank 0 (its device)"""
        if self.comm.rank == 0:
            serial = domain.Discretization(
                cell=disc.cell, nb_of_pixels_global=disc.nb_of_pixels_global, discretization_type='finite_element',
                element_type=disc.element_type, communicator=muGrid.Communicator(MPI.COMM_SELF), device=disc.device,
                hourglass_eps=getattr(disc.hourglass_control, 'eps', None))
            self.tail = Multigrid(serial, self.C_0, degree=self.degree, alpha=self.alpha, n_coarsest=self.n_coarsest,
                                  coarse_element=self.coarse_element, coarse_hourglass_eps=self.coarse_hourglass_eps,
                                  operator=self.operator, precision=self.precision, lmax=self.lmax)
            if self.tail.levels[0].x is None:
                self.tail.levels[0].x = self.tail.levels[0].vector('mg_x')
                self.tail.levels[0].b = self.tail.levels[0].vector('mg_b')
        else:
            self.tail = 'on rank 0'

    # ------------------------------------------------------------------ setup for a material
    def setup(self):
        """Everything that depends on the material ``self.scale``: coarse materials, Jacobi diagonals, smoothing
        intervals and the coarse factorisation. Call after every change of the scale."""
        clock = [time.perf_counter()]

        def lap(stage):                                      # wall time per stage of the last setup
            clock.append(time.perf_counter())
            self.setup_seconds[stage] = self.setup_seconds.get(stage, 0.0) + clock[-1] - clock[-2]

        self.setup_seconds = {}
        if self.lmax == 'bound':                             # the bound holds for s C_0 with s >= 0 only
            xp = tensor_operations.get_array_module(self.scale.s)
            local_min = float(xp.min(self.scale.s)) if self.scale.s.size else np.inf
            if self.mpi.allreduce(local_min, op=MPI.MIN) < 0.0:
                raise ValueError('Multigrid: the material scale s must be >= 0 everywhere (lambda_max bound)')
        self.fine_operator.update_material()
        if self.levels[0] is not self.fine_operator:         # single-precision cycle: its copy of the scale
            self.levels[0].scale.s[...] = self.scale.s
            self.levels[0].update_material()
        for fine, coarse in zip(self.levels[:-1], self.levels[1:]):
            self._coarsen_scale(fine, coarse)
            coarse.update_material()
        lap('coarse materials')
        for level in self.levels:
            if level.dinv is not None:
                self._assemble_inverse_diagonal(level)
                lap('diagonals')
                if self.lmax == 'bound':
                    level.lmax = self._lmax_bound(level)
                    lap('lambda_max bound')
                else:
                    self._estimate_lmax(level)
                    lap('power iterations')
        if self.tail is None:
            self._factorise_coarsest()
        else:
            scale = self._gather(self.levels[-1].scale)
            if self.comm.rank == 0:
                self.tail.scale.s[...] = tensor_operations.get_array_module(self.tail.scale.s).asarray(scale)
                self.tail.setup()
        lap('coarse solve / tail')

    @staticmethod
    def _coarsen_scale(fine, coarse):
        """re-discretisation: mean scale of the fine cells of every coarse quadrature point (local on every rank)"""
        s_fine, s_coarse = fine.scale.s[0, 0], coarse.scale.s[0, 0]
        if fine.dim == 3:
            voxel_mean = s_fine[0] if s_fine.shape[0] == 1 else s_fine.mean(axis=0)
            if s_coarse.shape[0] == 1:                                          # one-point coarse element
                s_coarse[0] = 0.0
                for i, j, k in np.ndindex(2, 2, 2):
                    s_coarse[0] += voxel_mean[i::2, j::2, k::2]
                s_coarse[0] *= 0.125
                return
            for q, (k, j, i) in enumerate(np.ndindex(2, 2, 2)):                 # q = i + 2 j + 4 k
                s_coarse[q] = voxel_mean[i::2, j::2, k::2]
        else:                                               # linear triangles: q = 0 lower, q = 1 upper triangle
            c00, c10 = s_fine[:, 0::2, 0::2], s_fine[:, 1::2, 0::2]
            c01, c11 = s_fine[:, 0::2, 1::2], s_fine[:, 1::2, 1::2]
            s_coarse[0] = (c00[0] + c00[1] + c10[0] + c01[0]) / 4
            s_coarse[1] = (c11[0] + c11[1] + c10[1] + c01[1]) / 4

    @staticmethod
    def _assemble_inverse_diagonal(level):
        """level.dinv = 1 / diag(A). Stencil operator: by Dirac combs -- nodes two pixels apart do not interact, so A
        applied to a comb of unit impulses on every second node per direction returns the diagonal at the comb nodes"""
        if level.fused:                                      # the fused operator assembles its diagonal itself
            level.operator.assemble_diagonal(level.lam, level.mu, level.dinv)
        else:
            comb, response = level.d, level.t
            for component in range(level.dim):
                for offset in np.ndindex(*(2,) * level.dim):
                    nodes = (component, 0) + tuple(slice(o, None, 2) for o in offset)
                    comb.s[...] = 0.0
                    comb.s[nodes] = 1.0
                    level.apply_K(comb, response)
                    level.dinv.s[nodes] = response.s[nodes]
        diagonal = level.dinv.s
        diagonal[diagonal == 0.0] = 1.0                      # no stiffness at a node: leave it alone
        xp = tensor_operations.get_array_module(diagonal)
        xp.divide(1.0, diagonal, out=diagonal)
        level.disc.fft.communicate_ghosts(level.dinv)        # the Jacobi product runs over the ghost points too

    def _lmax_bound(self, level):
        """Guaranteed upper bound of lambda_max(D^-1 A) on ``level`` -- no estimate, no safety factor.

        A is a sum of local pieces A_e, each scaled by ONE material value (one quadrature point of one voxel; for
        one-point elements and the fused operator: one voxel, including its hourglass term), and D = diag(A) =
        sum_e D_e with D_e = diag(A_e). For every x (all terms >= 0, mediant inequality):

            x.Ax / x.Dx = sum_e x.A_e x / sum_e x.D_e x  <=  max_e x.A_e x / x.D_e x  <=  max_e lambda_max(D_e^-1 A_e)

        Here A_e = s_e A_e^0 with s_e >= 0 and a fixed matrix A_e^0 (material s C_0), so the bound depends only on the
        element, the pixel shape, C_0 and the hourglass control -- not on the design or the contrast. It is computed
        once per kind of level: A_e^0 of every quadrature point is assembled on a 4^d grid (scale 1 on that point of
        one voxel, 0 elsewhere) on the voxel's 2^d corner nodes, then a small generalised eigenproblem.
        (2D Q1, nu 0.2: 3.52, against 2.10 for homogeneous material and <= 3.1 for random extreme contrast.)"""
        import scipy.linalg
        disc = level.disc
        eps = getattr(disc.hourglass_control, 'eps', None)
        pixel = tuple(float(h) for h in disc.pixel_size)
        key = (disc.element_type, level.fused, eps, pixel)
        if key in self._lmax_bounds:
            return self._lmax_bounds[key]
        cell = domain.PeriodicUnitCell(domain_size=[4 * h for h in pixel], problem_type='elasticity')
        small = domain.Discretization(cell=cell, nb_of_pixels_global=(4,) * self.dim,
                                      discretization_type='finite_element', element_type=disc.element_type,
                                      hourglass_eps=eps)
        piece = Level(small, self.C_0, 0, fused=level.fused)                 # double precision, on the host
        corners = [(c, 0) + tuple(1 + i for i in offset)                     # the 2^d nodes of voxel (1, ..., 1)
                   for offset in np.ndindex(*(2,) * self.dim) for c in range(self.dim)]
        bound = 0.0
        for q in range(piece.scale.s.shape[2]):
            piece.scale.s[...] = 0.0
            piece.scale.s[(0, 0, q) + (1,) * self.dim] = 1.0
            piece.update_material()
            A_e = np.empty((len(corners), len(corners)))
            for j, dof in enumerate(corners):
                piece.x.s[...] = 0.0
                piece.x.s[dof] = 1.0
                piece.apply_K(piece.x, piece.b)
                A_e[:, j] = [float(piece.b.s[i]) for i in corners]
            A_e = 0.5 * (A_e + A_e.T)
            D_e = np.diag(A_e).copy()
            stiff = D_e > 1e-12 * D_e.max()                  # dofs the piece does not touch (e.g. a triangle's 4th node)
            bound = max(bound, float(scipy.linalg.eigh(A_e[np.ix_(stiff, stiff)], np.diag(D_e[stiff]),
                                                       eigvals_only=True)[-1]))
        self._lmax_bounds[key] = bound
        return bound

    def _estimate_lmax(self, level, iterations=40, warm_iterations=15, safety=1.1):
        """level.lmax: upper estimate of lambda_max(D^-1 A) by power iteration, warm started from the dominant vector
        of the previous material (topology optimization: the design changes little per iteration)"""
        xp = tensor_operations.get_array_module(level.dinv.s)
        if level.power is None:
            level.power = level.vector('mg_power_vector')
            level.power.s[...] = xp.asarray(self._power_start_vector(level))
        else:
            iterations = warm_iterations
        x, y = level.power, level.d
        xx = level.global_sum(linalg.norm_sq(x))
        for _ in range(iterations):
            level.apply_K(x, level.t)
            self._jacobi(level, level.t, y)
            yy = level.global_sum(linalg.norm_sq(y))
            lam = np.sqrt(yy / xx)
            linalg.copy(y, x)
            linalg.scal(1.0 / np.sqrt(yy), x)
            xx = 1.0
        level.lmax = safety * lam

    def _power_start_vector(self, level):
        """random start with zero mean per component, the same for every MPI decomposition"""
        shape = tuple(level.x.s.shape) if level.x is not None else tuple(level.r.s.shape)
        if self.power_start is not None:
            return self.power_start(level.index, shape)
        disc = level.disc
        location, n_global = np.asarray(disc.fft.subdomain_locations), disc.nb_of_pixels_global
        vector = np.empty(shape)
        for plane in range(shape[-1]):                       # one seed per global plane of the last axis
            rng = np.random.default_rng([1234 + level.index, int(location[-1]) + plane])
            full_plane = rng.standard_normal((shape[0], 1) + tuple(n_global[:-1]))
            window = tuple(slice(int(l), int(l) + n) for l, n in zip(location[:-1], shape[2:-1]))
            vector[..., plane] = full_plane[(slice(None), slice(None)) + window]
        axes = tuple(range(2, vector.ndim))
        mean = self.mpi.allreduce(vector.sum(axis=axes), op=MPI.SUM) / np.prod(n_global)
        return vector - mean.reshape(mean.shape + (1,) * len(axes))

    def _factorise_coarsest(self):
        """dense pseudo-inverse of the coarsest operator (singular: rigid translations)"""
        level = self.levels[-1]
        xp = tensor_operations.get_array_module(level.x.s)
        n = int(np.prod(level.x.s.shape))
        A = np.empty((n, n))
        for i, entry in enumerate(np.ndindex(*level.x.s.shape)):
            level.b.s[...] = 0.0
            level.b.s[entry] = 1.0
            level.apply_K(level.b, level.x)
            A[:, i] = tensor_operations.to_host(level.x.s).ravel()
        rcond = 1e-12 if level.dtype == np.float64 else 1e-5         # above the round-off of the assembled matrix
        self._coarse_pinv = xp.asarray(np.linalg.pinv(0.5 * (A + A.T), rcond=rcond, hermitian=True))

    # ------------------------------------------------------------------ the cycle
    def apply(self, r, z):
        """z = V-cycle(r) with zero start: the preconditioner M^-1 r (fields of the fine discretization)"""
        fine = self.levels[0]
        if fine is self.fine_operator:       # (the ghost layers of z are not valid: exchange before a stencil reads them)
            fine.b, fine.x = r, z
            self._vcycle(0)
        else:                                # single-precision cycle between double-precision vectors
            self._convert(r, fine.b)
            self._vcycle(0)
            self._convert(fine.x, z)

    @staticmethod
    def _convert(source, target):
        """target = source for fields of different precision (whole buffers on the GPU)"""
        if source.is_on_gpu:
            tensor_operations.point_view(target)[...] = tensor_operations.point_view(source)
        else:
            target.s[...] = source.s

    def _vcycle(self, l):
        level = self.levels[l]
        if l == len(self.levels) - 1:
            return self._solve_coarsest(level)
        coarse = self.levels[l + 1]
        self._smooth(level, x_is_zero=True)
        level.apply_K(level.x, level.t)                      # r = b - A x
        linalg.copy(level.b, level.r)
        linalg.axpy(-1.0, level.t, level.r)
        level.disc.fft.communicate_ghosts(level.r)
        self.transfer.restrict(level.r, coarse.b)
        self._vcycle(l + 1)
        coarse.disc.fft.communicate_ghosts(coarse.x)
        self.transfer.prolong(coarse.x, level.t)             # x += P x_H
        linalg.axpy(1.0, level.t, level.x)
        self._smooth(level, x_is_zero=False)

    def _smooth(self, level, x_is_zero):
        """`degree` Chebyshev steps for A x = b preconditioned by D^-1 on [lmax / alpha, lmax] (Saad, Alg. 12.1)"""
        xp = tensor_operations.get_array_module(level.dinv.s)
        lmax, lmin = level.lmax, level.lmax / self.alpha
        theta, delta = 0.5 * (lmax + lmin), 0.5 * (lmax - lmin)
        sigma = theta / delta
        rho = 1.0 / sigma
        x, b, r, d, t = level.x, level.b, level.r, level.d, level.t
        linalg.copy(b, r)                                    # r = b - A x
        if not x_is_zero:
            level.apply_K(x, t)
            linalg.axpy(-1.0, t, r)
        self._jacobi(level, r, d)                            # d = D^-1 r / theta
        linalg.scal(1.0 / theta, d)
        for k in range(self.degree):
            if k == 0 and x_is_zero:
                linalg.copy(d, x)
            else:
                linalg.axpy(1.0, d, x)                       # x += d
            if k == self.degree - 1:
                break
            level.apply_K(d, t)                              # r -= A d
            linalg.axpy(-1.0, t, r)
            rho_new = 1.0 / (2.0 * sigma - rho)
            self._jacobi(level, r, t)                        # d = rho' rho d + 2 rho' / delta D^-1 r
            linalg.axpby(2.0 * rho_new / delta, t, rho_new * rho, d)
            rho = rho_new

    @staticmethod
    def _jacobi(level, r, z):
        """z = D^-1 r; on the GPU over the whole contiguous buffers (no index arithmetic of strided views)"""
        if level.disc.device == 'cpu':
            np.multiply(level.dinv.s, r.s, out=z.s)
        else:
            dinv = tensor_operations.point_view(level.dinv)
            xp = tensor_operations.get_array_module(dinv)
            xp.multiply(dinv, tensor_operations.point_view(r), out=tensor_operations.point_view(z))

    def _solve_coarsest(self, level):
        if self.tail is None:
            solution = self._coarse_pinv @ level.b.s.ravel()
            level.x.s[...] = solution.reshape(level.x.s.shape)
            return
        rhs = self._gather(level.b)                          # agglomerated coarse problem: serial cycle on rank 0
        solution = None
        if self.comm.rank == 0:
            tail_level = self.tail.levels[0]
            tail_level.b.s[...] = tensor_operations.get_array_module(tail_level.b.s).asarray(rhs)
            self.tail.apply(tail_level.b, tail_level.x)
            solution = tensor_operations.to_host(tail_level.x.s)
        self._scatter(solution, level.x)

    # ------------------------------------------------------------------ agglomeration: gather / scatter
    def _slab(self, field):
        """index of this rank's interior block in the global array of `field` (spatial axes last)"""
        disc = self.levels[-1].disc
        location, size = disc.fft.subdomain_locations, disc.fft.nb_subdomain_grid_pts
        return (Ellipsis,) + tuple(slice(int(l), int(l) + int(n)) for l, n in zip(location, size))

    def _gather(self, field):
        """the global host array of a distributed field, on rank 0 (None elsewhere)"""
        pieces = self.mpi.gather((self._slab(field), tensor_operations.to_host(field.s)), root=0)
        if self.comm.rank != 0:
            return None
        self._tail_slabs = [slab for slab, _ in pieces]
        n_global = self.levels[-1].disc.nb_of_pixels_global
        full = np.empty(tuple(pieces[0][1].shape[:-self.dim]) + tuple(n_global))
        for slab, piece in pieces:
            full[slab] = piece
        return full

    def _scatter(self, full, field):
        """rank 0's global host array -> the distributed field"""
        pieces = [full[slab] for slab in self._tail_slabs] if self.comm.rank == 0 else None
        xp = tensor_operations.get_array_module(field.s)
        field.s[...] = xp.asarray(self.mpi.scatter(pieces, root=0))
