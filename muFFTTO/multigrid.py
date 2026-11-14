"""
Geometric multigrid preconditioner for periodic small-strain elasticity with a scaled material ``A(x) = s(x) C_0``.

A symmetric V(nu, nu)-cycle with Chebyshev-Jacobi smoothing, to be used as the preconditioner of the conjugate
gradient solver. It is the field-based, MPI-capable successor of ``experiments/Multigrid_preconditioning/mg_lib.py``
(same hierarchy, same smoother, same coarse solve -- the same iteration counts), written for large 3D grids:

* **No copies, no temporaries.** Every vector of the cycle is a muGrid field owned by its level; updates are fused
  muGrid ``linalg`` kernels (``axpy``, ``axpby``, ``copy``) or in-place array operations. The fine level works on the
  caller's ``Discretization`` and shares its scratch fields (gradient, hourglass), so the multigrid adds four nodal
  vectors and a diagonal on the fine grid (120 B per voxel in 3D) and about 1/7 of that again on the coarse grids.
* **MPI.** Every level is a ``Discretization`` on the same communicator. muGrid's FFT engine splits the grid into
  slabs along the last axis; as long as every rank keeps an even number of planes the levels are nested and the
  grid transfer (``muGrid.GridTransfer``) is local up to a ghost exchange. When the slabs become too thin the rest
  of the hierarchy is *agglomerated*: the coarse problem is gathered on rank 0 and solved there by a serial
  multigrid on the host (class :class:`Multigrid` again, on ``MPI.COMM_SELF``).

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
           lambda_max(D^-1 A) by power iteration (warm started after a material update), times 1.1
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
import numpy as np
import muGrid
from mpi4py import MPI
from muGrid import linalg

from . import domain, tensor_operations

FULLY_INTEGRATED = {'trilinear_hexahedron_1Q': 'trilinear_hexahedron',
                    'trilinear_hexahedron_1Q_hg': 'trilinear_hexahedron'}


class GridTransfer:
    """Multilinear interpolation ``P`` between two nested nodal grids and its transpose ``R = P^T``.

    Host fields use ``muGrid.GridTransfer``. Device fields use the same stencils written as in-place array
    operations on strided views (27 small updates per transfer, no full-size temporaries) where the muGrid build has
    no device kernel. The caller exchanges the ghosts of the *input* field first: ``restrict`` reads the left ghost
    layer of the fine field, ``prolong`` the right ghost layer of the coarse field.
    """

    def __init__(self, dim):
        self.dim = dim
        self.kernel = muGrid.GridTransfer(dim)
        self.use_kernel = True

    def restrict(self, fine, coarse):
        """coarse = P^T fine: per axis, coarse[I] = fine[2I] + (fine[2I - 1] + fine[2I + 1]) / 2"""
        if self._try_kernel(self.kernel.restrict, fine, coarse):
            return
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
    """

    def __init__(self, discretization, C_0, index, own_solution_fields=True):
        self.disc = discretization
        self.index = index
        self.dim = discretization.domain_dimension
        self.scale = discretization.get_quad_field_scalar(name='mg_scale')
        self.material = domain.ScaledMaterial([(self.scale, C_0)])
        self.x = discretization.get_unknown_size_field(name='mg_x') if own_solution_fields else None
        self.b = discretization.get_unknown_size_field(name='mg_b') if own_solution_fields else None
        self.r = self.d = self.t = self.dinv = self.power = None
        self.lmax = None

    def allocate_smoother_fields(self):
        """the vectors of a level that is smoothed (all but the coarsest)"""
        self.r = self.disc.get_unknown_size_field(name='mg_residual')
        self.d = self.disc.get_unknown_size_field(name='mg_direction')
        self.t = self.disc.get_unknown_size_field(name='mg_scratch')
        self.dinv = self.disc.get_unknown_size_field(name='mg_inverse_diagonal')

    def apply_K(self, x, y):
        """y = A x"""
        self.disc.apply_system_matrix_mugrid(material_data_field=self.material, input_field_inxyz=x,
                                             output_field_inxyz=y, formulation='small_strain')

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
    power_start : callable, optional
        ``power_start(level_index, shape) -> ndarray``: start vector of the power iteration (serial runs only; used
        by the tests to reproduce ``mg_lib``). Default: seeded random numbers per grid plane, independent of the
        MPI decomposition.

    Attributes
    ----------
    scale : muGrid Field [1, 1, q, x, y(, z)]
        The fine-level scale ``s`` at the quadrature points: write it, then call :meth:`setup`.
    material : domain.ScaledMaterial
        ``s C_0`` on the fine level -- the operator the cycle preconditions.
    """

    def __init__(self, discretization, C_0, degree=2, alpha=30.0, n_coarsest=4, power_start=None):
        self.degree = degree
        self.alpha = alpha
        self.C_0 = np.asarray(C_0)
        self.n_coarsest = n_coarsest
        self.power_start = power_start
        self.dim = discretization.domain_dimension
        self.comm = discretization.communicator                 # muGrid communicator of all levels
        self.mpi = self.comm.mpi4py_comm if self.comm.size > 1 else MPI.COMM_SELF
        self.transfer = GridTransfer(self.dim)
        self.levels = [Level(discretization, self.C_0, 0, own_solution_fields=False)]
        self.tail = None                     # serial multigrid on rank 0 below the last distributed level
        self._tail_slabs = None
        self._coarse_pinv = None
        while self.levels[-1].disc.nb_of_pixels_global[0] > n_coarsest:
            fine = self.levels[-1].disc
            if not self._can_coarsen_distributed(fine):
                break
            fine_element = fine.element_type
            coarse = domain.Discretization(
                cell=fine.cell, nb_of_pixels_global=tuple(n // 2 for n in fine.nb_of_pixels_global),
                discretization_type='finite_element', element_type=FULLY_INTEGRATED.get(fine_element, fine_element),
                communicator=fine.communicator, device=fine.device, low_memory=fine.low_memory)
            self.levels.append(Level(coarse, self.C_0, len(self.levels)))
        if self.comm.size > 1:               # the last distributed grid and all coarser ones: serial, on rank 0
            self._build_tail(self.levels[-1].disc)
        for level in self.levels[:-1]:       # the last level is solved exactly (or handed to the tail)
            level.allocate_smoother_fields()
        self.scale = self.levels[0].scale
        self.material = self.levels[0].material

    # ------------------------------------------------------------------ hierarchy construction
    def _can_coarsen_distributed(self, disc):
        """True if the next coarser grid is nested on every rank: an even number (>= 4) of planes per rank"""
        local = np.asarray(disc.fft.nb_subdomain_grid_pts)
        ok = bool(np.all(local % 2 == 0) and np.all(local >= 4)) or self.comm.size == 1
        return bool(self.mpi.allreduce(ok, op=MPI.LAND)) if self.comm.size > 1 else ok

    def _build_tail(self, disc):
        """agglomeration: the grid of `disc` and all coarser ones as a serial host multigrid on rank 0"""
        if self.comm.rank == 0:
            serial = domain.Discretization(
                cell=disc.cell, nb_of_pixels_global=disc.nb_of_pixels_global, discretization_type='finite_element',
                element_type=disc.element_type, communicator=muGrid.Communicator(MPI.COMM_SELF), device='cpu',
                hourglass_eps=getattr(disc.hourglass_control, 'eps', None))
            self.tail = Multigrid(serial, self.C_0, degree=self.degree, alpha=self.alpha, n_coarsest=self.n_coarsest)
            self.tail.levels[0].x = serial.get_unknown_size_field(name='mg_x')
            self.tail.levels[0].b = serial.get_unknown_size_field(name='mg_b')
        else:
            self.tail = 'on rank 0'

    # ------------------------------------------------------------------ setup for a material
    def setup(self):
        """Everything that depends on the material ``self.scale``: coarse materials, Jacobi diagonals, smoothing
        intervals and the coarse factorisation. Call after every change of the scale."""
        for fine, coarse in zip(self.levels[:-1], self.levels[1:]):
            self._coarsen_scale(fine, coarse)
        for level in self.levels:
            if level.dinv is not None:
                self._assemble_inverse_diagonal(level)
                self._estimate_lmax(level)
        if self.tail is None:
            self._factorise_coarsest()
        else:
            scale = self._gather(self.levels[-1].scale)
            if self.comm.rank == 0:
                self.tail.scale.s[...] = scale
                self.tail.setup()

    @staticmethod
    def _coarsen_scale(fine, coarse):
        """re-discretisation: mean scale of the fine cells of every coarse quadrature point (local on every rank)"""
        s_fine, s_coarse = fine.scale.s[0, 0], coarse.scale.s[0, 0]
        if fine.dim == 3:
            voxel_mean = s_fine[0] if s_fine.shape[0] == 1 else s_fine.mean(axis=0)
            for q, (k, j, i) in enumerate(np.ndindex(2, 2, 2)):                 # q = i + 2 j + 4 k
                s_coarse[q] = voxel_mean[i::2, j::2, k::2]
        else:                                               # linear triangles: q = 0 lower, q = 1 upper triangle
            c00, c10 = s_fine[:, 0::2, 0::2], s_fine[:, 1::2, 0::2]
            c01, c11 = s_fine[:, 0::2, 1::2], s_fine[:, 1::2, 1::2]
            s_coarse[0] = (c00[0] + c00[1] + c10[0] + c01[0]) / 4
            s_coarse[1] = (c11[0] + c11[1] + c10[1] + c01[1]) / 4

    @staticmethod
    def _assemble_inverse_diagonal(level):
        """level.dinv = 1 / diag(A) by Dirac combs: nodes two pixels apart do not interact, so A applied to a comb
        of unit impulses on every second node per direction returns the diagonal at the comb nodes"""
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

    def _estimate_lmax(self, level, iterations=40, warm_iterations=15, safety=1.1):
        """level.lmax: upper estimate of lambda_max(D^-1 A) by power iteration, warm started from the dominant vector
        of the previous material (topology optimization: the design changes little per iteration)"""
        xp = tensor_operations.get_array_module(level.dinv.s)
        if level.power is None:
            level.power = level.disc.get_unknown_size_field(name='mg_power_vector')
            level.power.s[...] = xp.asarray(self._power_start_vector(level))
        else:
            iterations = warm_iterations
        x, y = level.power, level.d
        xx = level.global_sum(linalg.norm_sq(x))
        for _ in range(iterations):
            level.apply_K(x, level.t)
            xp.multiply(level.dinv.s, level.t.s, out=y.s)
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
        self._coarse_pinv = xp.asarray(np.linalg.pinv(0.5 * (A + A.T), rcond=1e-12, hermitian=True))

    # ------------------------------------------------------------------ the cycle
    def apply(self, r, z):
        """z = V-cycle(r) with zero start: the preconditioner M^-1 r (fields of the fine discretization)"""
        fine = self.levels[0]
        fine.b, fine.x = r, z
        self._vcycle(0)
        fine.disc.fft.communicate_ghosts(z)

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
        xp.multiply(level.dinv.s, r.s, out=d.s)              # d = D^-1 r / theta
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
            xp.multiply(level.dinv.s, r.s, out=t.s)          # d = rho' rho d + 2 rho' / delta D^-1 r
            linalg.axpby(2.0 * rho_new / delta, t, rho_new * rho, d)
            rho = rho_new

    def _solve_coarsest(self, level):
        if self.tail is None:
            solution = self._coarse_pinv @ level.b.s.ravel()
            level.x.s[...] = solution.reshape(level.x.s.shape)
            return
        rhs = self._gather(level.b)                          # agglomerated coarse problem: serial cycle on rank 0
        solution = None
        if self.comm.rank == 0:
            tail_level = self.tail.levels[0]
            tail_level.b.s[...] = rhs
            self.tail.apply(tail_level.b, tail_level.x)
            solution = tail_level.x.s
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
