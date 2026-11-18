"""
Geometric multigrid for the 2D bilinear quadrilateral ('bilinear_rectangle', 2x2 Gauss points) -- experiment-local
extension of the library class ``muFFTTO.multigrid.Multigrid`` (which coarsens linear triangles in 2D and
trilinear hexahedra in 3D).

The AT2 fracture runs of the dual-preconditioner study use bilinear quadrilaterals, so the multigrid here works on the
same discretization as the dual (same K, same CG, same stopping test). Everything of the library cycle is kept
(V(2, 2), Chebyshev-Jacobi smoother on [lambda_max / 30, lambda_max], exact coarsest solve on 4 x 4, bilinear
prolongation P, R = P^T); only the coarse material is new:

    re-discretisation, as option (a) of the 3D hexahedra: the Gauss point q = i + 2 j (quadrant (i, j)) of the coarse
    pixel (I, J) takes the mean scale of the fine pixel (2I + i, 2J + j) (mean over its 4 Gauss points).

(For Q1 this is not the Galerkin product P^T A_h P; in 3D it gave the same iteration counts as the Galerkin
triangles, see ../Multigrid_preconditioning.)
"""
import numpy as np
from muGrid import linalg

from muFFTTO.multigrid import Multigrid


class MultigridQ1(Multigrid):
    """``Multigrid`` with the 2D bilinear-quadrilateral coarsening (other elements: the library behaviour).

    warm_power_iterations: power iterations for lambda_max(D^-1 A) per level after a material update, warm started
    from the previous dominant vector (library: 15). The AT2 damage changes little per staggered iteration, so a few
    may do; the bound is multiplied by 1.1 either way."""

    def __init__(self, *args, warm_power_iterations=15, power_tol=None, lmax_bound='power', **kwargs):
        self.warm_power_iterations = warm_power_iterations
        self.power_tol = power_tol
        self.power_iterations_done = 0            # warm power iterations of all levels so far (diagnostic)
        self.lmax_bound = lmax_bound
        super().__init__(*args, **kwargs)
        if lmax_bound == 'gershgorin':
            self.gershgorin = {}
            for level in self.levels:
                key = level.disc.element_type
                if key not in self.gershgorin:
                    self.gershgorin[key] = gershgorin_constant(level.disc, self.C_0)

    def _estimate_lmax(self, level, iterations=40, warm_iterations=None, safety=1.1):
        if self.lmax_bound == 'gershgorin':           # rigorous bound for every s >= 0: no power iteration
            level.lmax = self.gershgorin[level.disc.element_type]
            return
        self._power_lmax(level, iterations, safety)

    def _power_lmax(self, level, iterations, safety):
        """power_tol None: the library estimate with `warm_power_iterations` warm steps. power_tol > 0 (adaptive):
        at least `warm_power_iterations` warm steps, then until the estimate changes by less than power_tol
        (relative), at most `iterations`. AT2 (2026-11-15): with 3 fixed warm steps the estimate lagged when the crack
        cut through the 1024^2 cell (lambda_max jumps) -> 3 solves of 1000-3500 CG iterations with V(1,1)."""
        if self.power_tol is None or level.power is None:
            super()._estimate_lmax(level, iterations=iterations, warm_iterations=self.warm_power_iterations,
                                   safety=safety)
            return
        x, y = level.power, level.d
        xx = level.global_sum(linalg.norm_sq(x))
        previous = None
        for k in range(iterations):
            level.apply_K(x, level.t)
            self._jacobi(level, level.t, y)
            yy = level.global_sum(linalg.norm_sq(y))
            lam = np.sqrt(yy / xx)
            linalg.copy(y, x)
            linalg.scal(1.0 / np.sqrt(yy), x)
            xx = 1.0
            self.power_iterations_done += 1
            if k + 1 >= self.warm_power_iterations and previous is not None \
                    and abs(lam - previous) <= self.power_tol * lam:
                break
            previous = lam
        level.lmax = safety * lam

    @staticmethod
    def _coarsen_scale(fine, coarse):
        s_fine, s_coarse = fine.scale.s[0, 0], coarse.scale.s[0, 0]
        if fine.dim == 2 and s_fine.shape[0] == 4:                  # bilinear quad: q = i + 2 j, x fastest
            pixel_mean = s_fine.mean(axis=0)
            for q, (j, i) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
                s_coarse[q] = pixel_mean[i::2, j::2]
            return
        Multigrid._coarsen_scale(fine, coarse)



def gershgorin_constant(discretization, C_0):
    """Rigorous, contrast-independent upper bound of lambda_max(D^-1 A) for A = s(x) C_0 with ANY s >= 0 per
    quadrature point (name kept from the first attempt; it is not the Gershgorin bound):

        A = sum_k s_k K_k  (k = one Gauss point of one element, K_k = w B^T C_0 B),  D = diag(A) = sum_k s_k D_k,
        x^T A x / x^T D x = sum_k s_k x^T K_k x / sum_k s_k x^T D_k x <= max_k x^T K_k x / x^T D_k x
                          <= c = max_q lambda_max(D_q^-1 K_q)      (mediant inequality; D_q = diag(K_q))

    bilinear_rectangle, E = 1, nu = 0.2: c = 3.523 (homogeneous material: 2.10; random extreme contrast: <= 3.1;
    300 random 0/1 patterns + greedy search never exceeded c). The row-sum (Gershgorin) bound is 10.4 - useless.
    AT2 (2026-11-15): the power estimate, warm started from the previous dominant vector, missed a new mode localised
    at the crack when it cut through the 1024^2 cell (1000-12000 CG iterations in a few solves, V(1,1) and V(2,2))."""
    import scipy.linalg
    from muFFTTO import domain
    dim = discretization.domain_dimension
    disc = domain.Discretization(cell=discretization.cell, nb_of_pixels_global=(4,) * dim,
                                 discretization_type='finite_element', element_type=discretization.element_type)
    scale = disc.get_quad_field_scalar(name='bound_scale')
    material = domain.ScaledMaterial([(scale, np.asarray(C_0))])
    x = disc.get_unknown_size_field(name='bound_x')
    y = disc.get_unknown_size_field(name='bound_y')
    n = int(np.prod(x.s.shape))
    c = 0.0
    for q in range(scale.s.shape[2]):                  # K_q scales like h^(d-2) as a whole: c does not depend on h
        scale.s[...] = 0.0
        scale.s[(0, 0, q) + (1,) * dim] = 1.0
        K = np.empty((n, n))
        for k, entry in enumerate(np.ndindex(*x.s.shape)):
            x.s[...] = 0.0
            x.s[entry] = 1.0
            disc.apply_system_matrix_mugrid(material_data_field=material, input_field_inxyz=x,
                                            output_field_inxyz=y, formulation='small_strain')
            K[:, k] = np.asarray(y.s).ravel()
        diagonal = np.diag(K)
        rows = diagonal > 1e-12 * diagonal.max()
        scaling = 1.0 / np.sqrt(diagonal[rows])
        K = scaling[:, None] * K[np.ix_(rows, rows)] * scaling[None, :]
        c = max(c, float(scipy.linalg.eigvalsh(0.5 * (K + K.T))[-1]))
    return c
