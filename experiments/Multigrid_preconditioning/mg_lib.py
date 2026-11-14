"""
Geometric multigrid on muFFTTO's periodic linear-triangle discretization (small-strain elasticity, 2D) and its
combinations with the FFT preconditioners Green, Green-Jacobi and dual.

Grid hierarchy
--------------
Every level is a muFFTTO Discretization (N, N/2, ..., n_coarsest); all operator, Jacobi and Green applications
go through muFFTTO.  Pixel split of 'linear_triangles': anti-diagonal (1,0)-(0,1), q = 0 lower, q = 1 upper.
The fine triangulation is the red refinement of the coarse one -> nested P1 spaces:

    P    P1 interpolation: coarse node (I, J) = fine node (2I, 2J); edge midpoints (2I+1, 2J), (2I, 2J+1) take
         the average of the edge ends; the cell centre (2I+1, 2J+1) lies on the coarse anti-diagonal and takes
         the average of (I+1, J) and (I, J+1)
    R    = P^T
    A_H  = P^T A_h P exactly, assembled as the muFFTTO stiffness on the coarse grid with the material averaged
         (arithmetic mean) over the 4 fine triangles of each coarse triangle -- the strain of a coarse P1
         function is constant per coarse triangle.  This is the Galerkin operator, not a re-discretisation.
         coarse q=0 of cell (I, J) = fine (2I, 2J) q0+q1, (2I+1, 2J) q0, (2I, 2J+1) q0
         coarse q=1 of cell (I, J) = fine (2I+1, 2J+1) q0+q1, (2I+1, 2J) q1, (2I, 2J+1) q1

One-level preconditioners M (on any level)
------------------------------------------
    jacobi        D^-1                          D = diag(K)
    green         G_C0                          reference C_0 (matrix phase)
    green_jacobi  D^-1/2 G_I D^-1/2             reference 2 mu I
    dual          G_C0 K(C_0 A^-1 C_0) G_C0

3D (trilinear_hexahedron, 2x2x2 Gauss points q = i + 2j + 4k, x fastest)
----------------------------------------------------------------------
    P    trilinear (tensor-product) interpolation of the nested Q1 spaces, R = P^T
    A_H  OPTION (a), implemented: RE-DISCRETISATION -- muFFTTO Q1 stiffness on the coarse grid, coarse Gauss point q
         (octant (i, j, k) of the coarse voxel) gets the mean material of the fine voxel in that octant.  For Q1 this is
         NOT the Galerkin product P^T A_h P (the strain of a coarse function is not constant per fine voxel).
    trilinear_hexahedron_1Q(_hg) (one Gauss point, under-integrated, 8x less quadrature-point memory): only the FINE
         level is one-point; all coarse levels are fully integrated hexahedra whose Gauss point in octant (i, j, k)
         takes the fine voxel (2I+i, 2J+j, 2K+k) -- no hourglass modes in the coarse operators. The fine one-point
         stiffness has zero-energy hourglass modes on even grids: use 'trilinear_hexahedron_1Q_hg' (muFFTTO's
         Flanagan-Belytschko hourglass control, hg_eps = fraction of the fully integrated hourglass stiffness).
    OPTION (b), fallback if (a) converges badly: exact Galerkin coarse operators as stored 27-point stencils with
         3 x 3 blocks per coarse node (P^T A_h P assembled level by level; 243 coefficients per node, applied by a
         custom stencil operator instead of a muFFTTO Discretization).

Combinations (preconditioners of the outer CG on the fine level)
----------------------------------------------------------------
    <M>           one-level M
    mg            V(nu, nu)-cycle, Chebyshev-Jacobi smoother, exact coarsest solve
    mgS-<M>       V-cycle whose smoother is Chebyshev in (M A) on every level  (FFT preconditioner as smoother)
    bnn<k>-<M>    B = Q + (I - Q A) M (I - A Q),  Q = P V_1^(k) R  (k Chebyshev-Jacobi V-cycles from level 1,
                  k = 1 if omitted) -- FFT preconditioner inside, multigrid coarse correction outside
                  (balancing / A-DEF2).  Never worse than M only for exact Q (k -> infinity).
    add<k>-<M>    B = M + Q                      (additive two-level)
    sgs-<M>       one grid: Chebyshev-Jacobi smoothing, M correction, smoothing (I - BA = (I-SA)(I-wMA)(I-SA));
                  M takes the coarse grids' role (user idea 2026-11-13); w = 1 for Green (lambda(GA) <= 1),
                  else 1 / lambda_max(M A)

All combinations are fixed SPD operators (symmetric smoothing with A-contractive Chebyshev polynomials, SPD
middle/coarse components), so they plug into the library CG with the energy stopping test unchanged.

Cost counters (fine-level equivalents, weight 4^-level): 'K' operator applications, 'G' Green (FFT pair)
applications, 'T' grid transfers (P or R).
"""
import re

import numpy as np

from muFFTTO import domain

MANDEL = [(0, 0), (1, 1), (0, 1)]
W_MANDEL = np.array([1.0, 1.0, np.sqrt(2.0)])
I4 = np.einsum('ik,jl->ijkl', np.eye(2), np.eye(2))       # 2 mu I for mu = 1/2 (Green-Jacobi reference)


def identity_ref(dim):
    """2 mu I for mu = 1/2 (Green-Jacobi reference) in `dim` dimensions"""
    return np.einsum('ik,jl->ijkl', np.eye(dim), np.eye(dim))


def xp_of(a):
    """numpy or cupy, whichever holds `a` (field data are cupy arrays on the GPU)"""
    if type(a).__module__.startswith('cupy'):
        import cupy
        return cupy
    return np


def to_host(a):
    return a.get() if xp_of(a) is not np else a


def to_mandel(C):
    """[2, 2, 2, 2, ...] -> [3, 3, ...]"""
    return np.array([[W_MANDEL[a] * W_MANDEL[b] * C[i, j, k, l] for b, (k, l) in enumerate(MANDEL)]
                     for a, (i, j) in enumerate(MANDEL)])


def from_mandel(M):
    """[3, 3, ...] -> [2, 2, 2, 2, ...] with both minor symmetries"""
    C = np.zeros((2, 2, 2, 2) + M.shape[2:])
    for a, (i, j) in enumerate(MANDEL):
        for b, (k, l) in enumerate(MANDEL):
            value = M[a, b] / (W_MANDEL[a] * W_MANDEL[b])
            for ii, jj in {(i, j), (j, i)}:
                for kk, ll in {(k, l), (l, k)}:
                    C[ii, jj, kk, ll] = value
    return C


# ======================================================================================================
# grid transfer and Galerkin coarsening (numpy, serial, periodic)
# ======================================================================================================
def coarsen_material(C):
    """Galerkin coarsening of the quad-point material [..., q=2, N, N] -> [..., 2, N/2, N/2]"""
    c00, c10 = C[..., 0::2, 0::2], C[..., 1::2, 0::2]
    c01, c11 = C[..., 0::2, 1::2], C[..., 1::2, 1::2]
    lower = (c00[..., 0, :, :] + c00[..., 1, :, :] + c10[..., 0, :, :] + c01[..., 0, :, :]) / 4
    upper = (c11[..., 0, :, :] + c11[..., 1, :, :] + c10[..., 1, :, :] + c01[..., 1, :, :]) / 4
    return np.stack([lower, upper], axis=-3)


def prolong(uH):
    """P1 interpolation [..., n, n] -> [..., 2n, 2n]"""
    xp = xp_of(uH)
    sh = uH.shape
    uh = xp.empty(sh[:-2] + (2 * sh[-2], 2 * sh[-1]), dtype=uH.dtype)
    ux = xp.roll(uH, -1, axis=-2)                           # (I+1, J)
    uy = xp.roll(uH, -1, axis=-1)                           # (I, J+1)
    uh[..., 0::2, 0::2] = uH
    uh[..., 1::2, 0::2] = 0.5 * (uH + ux)
    uh[..., 0::2, 1::2] = 0.5 * (uH + uy)
    uh[..., 1::2, 1::2] = 0.5 * (ux + uy)
    return uh


def restrict(rh):
    """P^T [..., 2n, 2n] -> [..., n, n]"""
    a = rh[..., 0::2, 0::2]
    b = rh[..., 1::2, 0::2]                                 # -> (I, J), (I+1, J)
    c = rh[..., 0::2, 1::2]                                 # -> (I, J), (I, J+1)
    d = rh[..., 1::2, 1::2]                                 # -> (I+1, J), (I, J+1)
    xp = xp_of(rh)
    return (a + 0.5 * (b + xp.roll(b, 1, axis=-2)) + 0.5 * (c + xp.roll(c, 1, axis=-1))
            + 0.5 * (xp.roll(d, 1, axis=-2) + xp.roll(d, 1, axis=-1)))


def coarsen_material_hex(C):
    """option (a) re-discretisation, 3D Q1: [..., q=8, N, N, N] -> [..., 8, N/2, N/2, N/2]; coarse Gauss point
    q = i + 2j + 4k = octant (i, j, k) of the coarse voxel = fine voxel (2I+i, 2J+j, 2K+k) -> its mean material"""
    m = C.mean(axis=-4)                                     # per fine voxel
    return np.stack([m[..., i::2, j::2, k::2] for k in (0, 1) for j in (0, 1) for i in (0, 1)], axis=-4)


def _prolong_1d(u, axis):
    xp = xp_of(u)
    n = u.shape[axis]
    shape = list(u.shape)
    shape[axis] = 2 * n
    out = xp.empty(shape, dtype=u.dtype)
    sl_even = [slice(None)] * u.ndim
    sl_odd = [slice(None)] * u.ndim
    sl_even[axis] = slice(0, None, 2)
    sl_odd[axis] = slice(1, None, 2)
    out[tuple(sl_even)] = u
    out[tuple(sl_odd)] = 0.5 * (u + xp.roll(u, -1, axis=axis))
    return out


def _restrict_1d(r, axis):
    xp = xp_of(r)
    sl_even = [slice(None)] * r.ndim
    sl_odd = [slice(None)] * r.ndim
    sl_even[axis] = slice(0, None, 2)
    sl_odd[axis] = slice(1, None, 2)
    odd = r[tuple(sl_odd)]
    return r[tuple(sl_even)] + 0.5 * (odd + xp.roll(odd, 1, axis=axis))


def prolong_tensor(uH, dim=3):
    """multilinear (Q1) interpolation over the last `dim` axes"""
    for ax in range(-dim, 0):
        uH = _prolong_1d(uH, ax)
    return uH


def restrict_tensor(rh, dim=3):
    """P^T of prolong_tensor"""
    for ax in range(-dim, 0):
        rh = _restrict_1d(rh, ax)
    return rh


# ======================================================================================================
# one grid level
# ======================================================================================================
class Level:
    """one grid level.  C_q: full material [d, d, d, d, q, *grid] (host numpy), or with scalar=True the scalar
    s [q, *grid] of A = s C_0, stored as a muFFTTO ScaledMaterial (one scalar per Gauss point instead of d^4: 16x
    less in 2D, 81x in 3D, and a cheaper application); dual material C_0 / s likewise; Green bounds = [min s, max s].
    device='gpu': muFFTTO/muGrid fields on the GPU, all arrays cupy."""

    def __init__(self, C_q, C_0, index, cost, device='cpu', scalar=False, element_type='linear_triangles',
                 hg_eps=None):
        self.N = C_q.shape[-1]
        self.hg_eps = hg_eps            # 'trilinear_hexahedron_1Q_hg': muFFTTO's hourglass control (hourglass_eps)
        self.dim = C_0.shape[0]
        self.scalar = scalar
        if device == 'gpu':
            import cupy
            self.xp = cupy
        else:
            self.xp = np
        self.index = index
        self.w = (2.0 ** self.dim) ** (-index)
        self.cost = cost
        self.C_q = C_q
        self.C_0 = C_0
        cell = domain.PeriodicUnitCell(domain_size=[1] * self.dim, problem_type='elasticity')
        kw = {'device': device} if device != 'cpu' else {}
        if element_type == 'trilinear_hexahedron_1Q_hg':
            kw['hourglass_eps'] = hg_eps
        self.disc = domain.Discretization(cell=cell, nb_of_pixels_global=(self.N,) * self.dim,
                                          discretization_type='finite_element', element_type=element_type, **kw)
        if scalar:
            self.scale = self.disc.get_quad_field_scalar(name='mg_scale')
            self.material = domain.ScaledMaterial([(self.scale, C_0)])
        else:
            self.material = self.disc.get_material_data_size_field_mugrid(name='stiffness')
        self._set_material_field(C_q)
        self.f_in = self.disc.get_unknown_size_field(name='lvl_in')
        self.f_out = self.disc.get_unknown_size_field(name='lvl_out')
        self.f_t1 = self.disc.get_unknown_size_field(name='lvl_t1')
        self.f_t2 = self.disc.get_unknown_size_field(name='lvl_t2')
        self.shape = self.f_in.s.shape
        self._green = {}
        self._dinv = None
        self._dhalf = None
        self._material_dual = None
        self._lmax = {}
        self._g_range = None
        self._power_start = {}

    def _bcast_C0(self):
        return self.xp.asarray(self.C_0)[(...,) + (None,) * (self.dim + 1)]     # [d,d,d,d, q, *grid]

    def _set_material_field(self, C_q):
        if self.scalar:
            self.scale.s[0, 0] = self.xp.asarray(C_q)
        else:
            self.material.s[...] = self.xp.asarray(C_q)

    def set_material(self, C_q):
        """new material on the existing grid (topology optimization: every design); drops every cached setup"""
        self.C_q = C_q
        self._set_material_field(C_q)
        self._dhalf = self._dinv = None
        self._material_dual = None
        self._lmax = {}
        self._g_range = None

    # ---------------------------------------------------------------- operators
    def _apply_K(self, material, x_field, y_field):
        self.disc.apply_system_matrix_mugrid(material_data_field=material, input_field_inxyz=x_field,
                                             output_field_inxyz=y_field, formulation='small_strain')
        self.disc.fft.communicate_ghosts(y_field)
        self.cost['K'] += self.w

    def K(self, x, count=True):
        self.f_in.s[...] = x
        self._apply_K(self.material, self.f_in, self.f_out)
        if not count:
            self.cost['K'] -= self.w
        return self.f_out.s.copy()

    def _green_field(self, key):
        if key not in self._green:
            ref = {'C0': self.C_0, 'I': identity_ref(self.dim)}[key]
            self._green[key] = self.disc.get_preconditioner_Green_mugrid(reference_material_data_ijkl=ref,
                                                                         name=f'green_{key}')
        return self._green[key]

    def _apply_G(self, key, x_field, y_field):
        self.disc.fft.communicate_ghosts(x_field)
        self.disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=self._green_field(key),
                                              input_nodal_field_fnxyz=x_field, output_nodal_field_fnxyz=y_field)
        self.cost['G'] += self.w

    def dhalf(self):
        if self._dhalf is None:
            d = self.disc.get_preconditioner_Jacobi_mugrid(material_data_field_ijklqxyz=self.material,
                                                           formulation='small_strain', name='jacobi_mg')
            self._dhalf = d.s.copy()
            self._dinv = self._dhalf ** 2
        return self._dhalf

    def material_dual(self):
        if self._material_dual is None and self.scalar:                     # C_0 (s C_0)^-1 C_0 = C_0 / s
            dual_scale = self.disc.get_quad_field_scalar(name='mg_dual_scale')
            dual_scale.s[0, 0] = 1.0 / self.xp.asarray(self.C_q)
            self._material_dual = domain.ScaledMaterial([(dual_scale, self.C_0)])
        if self._material_dual is None:
            M_0 = to_mandel(self.C_0)
            M_q = to_mandel(self.C_q)                                          # [3, 3, q, N, N]
            inv = np.linalg.inv(np.moveaxis(M_q, (0, 1), (-2, -1)))            # [q, N, N, 3, 3]
            M_dual = np.einsum('ab,...bc,cd->ad...', M_0, inv, M_0)
            self._material_dual = self.disc.get_material_data_size_field_mugrid(name='stiffness_dual')
            self._material_dual.s[...] = self.xp.asarray(from_mandel(M_dual))
        return self._material_dual

    def g_range(self):
        """[min, max] eigenvalue of C_0^-1 A over the quad points = Green's spectral bounds"""
        if self._g_range is None and self.scalar:
            self._g_range = (float(self.C_q.min()), float(self.C_q.max()))
        if self._g_range is None:
            M_0 = to_mandel(self.C_0)
            L_inv = np.linalg.inv(np.linalg.cholesky(M_0))
            g = np.linalg.eigvalsh(np.einsum('ab,bc...,dc->...ad', L_inv, to_mandel(self.C_q), L_inv))
            self._g_range = (g.min(), g.max())
        return self._g_range

    def M(self, kind, x):
        """one-level preconditioner applied to a numpy array"""
        if kind == 'jacobi':
            self.dhalf()
            return self._dinv * x
        if kind == 'green':
            self.f_in.s[...] = x
            self._apply_G('C0', self.f_in, self.f_out)
            return self.f_out.s.copy()
        if kind == 'green_jacobi':
            self.f_in.s[...] = self.dhalf() * x
            self._apply_G('I', self.f_in, self.f_out)
            return self._dhalf * self.f_out.s
        if kind == 'dual':
            self.f_in.s[...] = x
            self._apply_G('C0', self.f_in, self.f_t1)
            self._apply_K(self.material_dual(), self.f_t1, self.f_t2)
            self._apply_G('C0', self.f_t2, self.f_out)
            return self.f_out.s.copy()
        raise ValueError(kind)

    def lmax(self, kind, iterations=40, safety=1.1, warm_iterations=15):
        """upper estimate of lambda_max(M A): power iteration (setup cost); after a material update it restarts from
        the previous dominant vector with `warm_iterations` steps (TO: the design changes little per iteration)"""
        if kind not in self._lmax:
            if kind in self._power_start:
                x = self._power_start[kind]
                iterations = warm_iterations
            else:
                rng = np.random.default_rng(1234 + self.index)
                x = rng.standard_normal(self.shape)
                x = self.xp.asarray(x - x.mean(axis=tuple(range(-self.dim, 0)), keepdims=True))
            lam = 0.0
            for _ in range(iterations):
                y = self.M(kind, self.K(x))
                yy = float(self.xp.sum(y * y))
                lam = np.sqrt(yy / float(self.xp.sum(x * x)))
                x = y / np.sqrt(yy)
            self._power_start[kind] = x
            if kind == 'green':                                  # guaranteed bound available
                lam = min(lam * safety, self.g_range()[1])
            else:
                lam *= safety
            self._lmax[kind] = lam
        return self._lmax[kind]

    def setup(self, kind):
        """touch everything `kind` needs so that no setup work lands in the solve counters"""
        if kind in ('jacobi', 'green_jacobi'):
            self.dhalf()
        if kind == 'dual':
            self.material_dual()
        self.lmax(kind)


def chebyshev(level, x, b, kind, degree, lmin, lmax, x_is_zero=False):
    """`degree` Chebyshev steps for A x = b preconditioned by M = `kind` on [lmin, lmax] (Saad, Alg. 12.1)"""
    theta, delta = 0.5 * (lmax + lmin), 0.5 * (lmax - lmin)
    sigma = theta / delta
    rho = 1.0 / sigma
    r = b.copy() if x_is_zero else b - level.K(x)
    d = level.M(kind, r) / theta
    for k in range(degree):
        x = x + d
        if k == degree - 1:
            break
        r = r - level.K(d)
        rho_new = 1.0 / (2.0 * sigma - rho)
        d = rho_new * rho * d + (2.0 * rho_new / delta) * level.M(kind, r)
        rho = rho_new
    return x


# ======================================================================================================
# hierarchy, V-cycle and combinations
# ======================================================================================================
class Hierarchy:
    def __init__(self, C_q, C_0, n_coarsest=4, degree=2, alpha=30.0, device='cpu', scalar=False,
                 element_type=None, hg_eps=None):
        """C_q: material at the quadrature points (host numpy); scalar=True: C_q is s [q, *grid], A = s C_0.
        element_type: 'linear_triangles' (2D, default for 2D C_0; exact Galerkin) or 'trilinear_hexahedron'
        (3D default; option (a) re-discretisation)"""
        self.cost = {'K': 0.0, 'G': 0.0, 'T': 0.0}
        self.degree = degree
        self.alpha = alpha
        self.dim = C_0.shape[0]
        self.element_type = element_type or ('linear_triangles' if self.dim == 2 else 'trilinear_hexahedron')
        if self.element_type == 'linear_triangles':
            self._coarsen, self._prolong, self._restrict = coarsen_material, prolong, restrict
        elif self.element_type in ('trilinear_hexahedron', 'trilinear_hexahedron_1Q', 'trilinear_hexahedron_1Q_hg'):
            self._coarsen = coarsen_material_hex            # q = 1 or 8 -> 8 coarse Gauss points
            self._prolong = lambda u: prolong_tensor(u, 3)
            self._restrict = lambda r: restrict_tensor(r, 3)
        else:
            raise ValueError(self.element_type)
        coarse_element = 'trilinear_hexahedron' if '_1Q' in self.element_type else self.element_type
        self.levels = [Level(C_q, C_0, 0, self.cost, device, scalar, self.element_type, hg_eps=hg_eps)]
        while self.levels[-1].N > n_coarsest:
            C_q = self._coarsen(C_q)                        # linear in the material: also coarsens s
            self.levels.append(Level(C_q, C_0, len(self.levels), self.cost, device, scalar, coarse_element))
        self._pinv = None

    def update_material(self, C_q):
        """new material on all levels (Galerkin coarsening), all setups dropped"""
        for lev in self.levels:
            lev.set_material(C_q)
            C_q = self._coarsen(C_q)
        self._pinv = None

    def coarse_pinv(self):
        if self._pinv is None:
            lev = self.levels[-1]
            n = int(np.prod(lev.shape))
            A = np.empty((n, n))
            for i in range(n):
                e = np.zeros(n)
                e[i] = 1.0
                A[:, i] = to_host(lev.K(lev.xp.asarray(e.reshape(lev.shape)))).ravel()
            self._pinv = lev.xp.asarray(np.linalg.pinv(0.5 * (A + A.T), rcond=1e-12, hermitian=True))
        return self._pinv

    def P(self, l, uH):
        self.cost['T'] += self.levels[l].w
        return self._prolong(uH)

    def R(self, l, rh):
        self.cost['T'] += self.levels[l].w
        return self._restrict(rh)

    def smooth(self, l, x, b, kind, x_is_zero=False):
        lev = self.levels[l]
        lmax = lev.lmax(kind)
        return chebyshev(lev, x, b, kind, self.degree, lmax / self.alpha, lmax, x_is_zero=x_is_zero)

    def vcycle(self, b, l=0, kind='jacobi'):
        """symmetric V(degree, degree)-cycle from level l, x_0 = 0"""
        if l == len(self.levels) - 1:
            return (self.coarse_pinv() @ b.ravel()).reshape(b.shape)
        lev = self.levels[l]
        x = self.smooth(l, lev.xp.zeros_like(b), b, kind, x_is_zero=True)
        r = b - lev.K(x)
        x = x + self.P(l, self.vcycle(self.R(l, r), l + 1, kind))
        return self.smooth(l, x, b, kind)

    def Q(self, r, cycles=1):
        """coarse correction on the fine level: P V_1^(cycles) R, V_1^(k) = k V-cycles from level 1 as a fixed
        linear iteration (error propagator (I - V A_1)^k: symmetric, SPD, -> exact A_1^-1 as k grows)"""
        b = self.R(0, r)
        x = self.vcycle(b, 1, 'jacobi')
        for _ in range(cycles - 1):
            x = x + self.vcycle(b - self.levels[1].K(x), 1, 'jacobi')
        return self.P(0, x)

    @staticmethod
    def parse(method):
        """'bnn3-dual' -> ('bnn', 'dual', 3)"""
        family, _, kind = method.partition('-')
        m = re.fullmatch(r'([a-zA-Z_]+?)(\d*)', family)
        return m.group(1), kind, int(m.group(2) or 1)

    def setup(self, method):
        """all setup work (Jacobi diagonals, spectral estimates, coarse factorisation) before the solve"""
        family, kind, _ = self.parse(method)
        if family in ('jacobi', 'green', 'green_jacobi', 'dual'):
            if family in ('jacobi', 'green_jacobi'):
                self.levels[0].dhalf()
            if family == 'dual':
                self.levels[0].material_dual()
        elif family == 'mg':
            for lev in self.levels[:-1]:
                lev.setup('jacobi')
            self.coarse_pinv()
        elif family == 'mgS':
            for lev in self.levels[:-1]:
                lev.setup(kind)
            self.coarse_pinv()
        elif family == 'sgs':
            self.levels[0].setup('jacobi')
            if kind == 'dual':
                self.levels[0].material_dual()
            if kind != 'green':
                self.levels[0].lmax(kind)
        elif family in ('bnn', 'add'):
            for lev in self.levels[1:-1]:
                lev.setup('jacobi')
            if kind in ('jacobi', 'green_jacobi'):
                self.levels[0].dhalf()
            if kind == 'dual':
                self.levels[0].material_dual()
            self.coarse_pinv()
        else:
            raise ValueError(method)
        setup_cost = dict(self.cost)
        for key in self.cost:
            self.cost[key] = 0.0
        return setup_cost

    def preconditioner(self, method):
        """numpy -> numpy, fine level"""
        family, kind, cycles = self.parse(method)
        lev = self.levels[0]
        if family in ('jacobi', 'green', 'green_jacobi', 'dual'):
            return lambda r: lev.M(family, r)
        if family == 'mg':
            return lambda r: self.vcycle(r, 0, 'jacobi')
        if family == 'mgS':
            return lambda r: self.vcycle(r, 0, kind)
        if family == 'bnn':
            def bnn(r):
                z1 = self.Q(r, cycles)
                z2 = lev.M(kind, r - lev.K(z1))
                return z1 + z2 - self.Q(lev.K(z2), cycles)
            return bnn
        if family == 'add':
            return lambda r: lev.M(kind, r) + self.Q(r, cycles)
        if family == 'sgs':
            omega = 1.0 if kind == 'green' else 1.0 / lev.lmax(kind)

            def sgs(r):
                x = self.smooth(0, lev.xp.zeros_like(r), r, 'jacobi', x_is_zero=True)
                x = x + omega * lev.M(kind, r - lev.K(x))
                return self.smooth(0, x, r, 'jacobi')
            return sgs
        raise ValueError(method)
