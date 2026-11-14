"""
Hourglass control for the one-point trilinear hexahedron, element ``'trilinear_hexahedron_1Q_hg'``.

Why
---
One-point (under-integrated) hexahedra store a single quadrature point per voxel instead of eight: the gradient-sized
temporaries, the material scale and every quadrature field are 8x smaller, and one operator application is cheaper.
But the one-point stiffness is rank deficient: displacement patterns whose strain vanishes at the voxel centre (the
"hourglass" modes) cost no energy. On a periodic grid with even size they are global checkerboard-like fields; they
make solvers stall (multigrid needs 15-250x more iterations) and let sharp material interfaces deform too softly.

What
----
The classical remedy of Flanagan & Belytschko (Int. J. Numer. Meth. Engng 17 (1981) 679): add a small stiffness for
the hourglass modes only. On muFFTTO's regular voxel grid the four hourglass base vectors of a voxel are

    h_1 = eta zeta,   h_2 = zeta xi,   h_3 = xi eta,   h_4 = xi eta zeta       (xi, eta, zeta = +-1 at the 8 corners)

They are orthogonal to every linear displacement, so the control never acts on constant strain: right-hand sides,
homogenised stresses and the consistency of the element are unchanged. Per voxel and displacement component i:

    hourglass strain   q_{i alpha} = 1/8 sum_corners h_alpha(corner) u_i(corner)          (4 modes x components)
    control energy     eps * sum_voxels sum_{i alpha} kappa_{i alpha}(x) q_{i alpha}^2
    stiffness          K_hg u = H^T (eps kappa H u)

The mode stiffness kappa is NOT a tuned constant: kappa_{i alpha} is the energy of the unit hourglass mode (alpha in
component i) of one voxel under full 2x2x2 Gauss integration with the local material,

    kappa_{i alpha}(x) = sum_{d, e} M_alpha[d, e] C_{i d i e}(x)        (elasticity)
    kappa_alpha(x)     = sum_{d, e} M_alpha[d, e] A_{d e}(x)            (conductivity)
    M_alpha[d, e]      = sum_g w_g  d_d phi_alpha(x_g)  d_e phi_alpha(x_g),   phi_alpha = sum_corners h_alpha N_corner

so eps = 1 reproduces the hourglass energy of the fully integrated element mode by mode (independent of the grid
size), and eps < 1 is a fraction of it. Default eps = 0.3 (2026-11-14 study, experiments/Multigrid_preconditioning,
3D Cahn-Hilliard geometry, contrast 1e4, 32^3 and 64^3: multigrid needs the same iterations as with the 8-point
element (eps 0.1: up to +4, eps 0.03: up to +14), the homogenised stiffness is within 0.2 % of the 8-point one for
phase-field interfaces of width eta >= h, and one CG iteration is ~3x faster at 64^3. Sharp voxel interfaces are
-8 % (64^3) to -14 % (32^3) too soft whatever eps -- an error of the one-point element itself, which converges with h).

How
---
H is a muGrid ``GenericLinearOperator`` (stencil [4 modes, 1 quadrature point, 1 node, 2, 2, 2 corners]) on fields of
the discretization: GPU and MPI work exactly as for the gradient operator. ``Discretization`` creates one
``HourglassControl`` for the element ``'trilinear_hexahedron_1Q_hg'`` and ``apply_system_matrix_mugrid`` adds K_hg u;
the Jacobi diagonal and Green's reference operator, both built from ``apply_system_matrix_mugrid``, include it.
"""
import itertools

import numpy as np
from muGrid import Field, GenericLinearOperator, linalg

from . import tensor_operations

DEFAULT_EPS = 0.3
CORNERS = list(itertools.product((0, 1), repeat=3))           # voxel corner (i, j, k) at offset (i, j, k)
GAUSS_1D = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))


def hourglass_base_vectors():
    """h[alpha, i, j, k]: the four hourglass base vectors at the 8 voxel corners (xi = 2i - 1 etc.)"""
    h = np.zeros((4, 2, 2, 2))
    for (i, j, k) in CORNERS:
        xi, eta, zeta = 2 * i - 1, 2 * j - 1, 2 * k - 1
        h[:, i, j, k] = (eta * zeta, zeta * xi, xi * eta, xi * eta * zeta)
    return h


def mode_metric(pixel_size):
    """M[alpha, d, e] = sum_g w_g d_d phi_alpha(x_g) d_e phi_alpha(x_g) over the 2x2x2 Gauss points of one voxel of size
    pixel_size, phi_alpha = sum_corners h_alpha(corner) N_corner (trilinear shape functions)"""
    h = hourglass_base_vectors()
    size = np.asarray(pixel_size, dtype=float)
    weight = np.prod(size) / 8.0                               # 8 Gauss points, physical weights
    M = np.zeros((4, 3, 3))
    for gauss_point in itertools.product(GAUSS_1D, repeat=3):
        grad_phi = np.zeros((4, 3))
        for (i, j, k) in CORNERS:
            s = np.array([2 * i - 1, 2 * j - 1, 2 * k - 1], dtype=float)
            factors = (1.0 + s * np.asarray(gauss_point)) / 2.0                # trilinear N = prod of 1D hats
            grad_N = np.array([s[d] / 2.0 * np.prod(np.delete(factors, d)) * 2.0 / size[d] for d in range(3)])
            grad_phi += np.outer(h[:, i, j, k], grad_N)
        M += weight * np.einsum('ad,ae->ade', grad_phi, grad_phi)
    return M


class HourglassControl:
    """Flanagan-Belytschko hourglass stiffness K_hg = H^T (eps kappa) H of a one-point hexahedron discretization."""

    def __init__(self, discretization, eps=DEFAULT_EPS):
        if discretization.domain_dimension != 3:
            raise ValueError('hourglass control is defined for the 3D one-point hexahedron only')
        self.discretization = discretization
        self.eps = DEFAULT_EPS if eps is None else float(eps)
        self.nb_components = discretization.cell.unknown_shape[0]      # 3 (elasticity) or 1 (conductivity)
        stencil = (hourglass_base_vectors() / 8.0)[:, np.newaxis, np.newaxis]   # [4, q = 1, n = 1, 2, 2, 2]
        self.operator = GenericLinearOperator([0, 0, 0], np.ascontiguousarray(stencil))
        self.M = mode_metric(discretization.pixel_size)                # [4, 3, 3]
        self.strain = discretization.field_collection.real_field(
            name='hourglass_strain_temporary', components=(self.nb_components, 4), sub_pt='quad_points')
        self.force = discretization.get_unknown_size_field(name='hourglass_force_temporary')

    # ---------------------------------------------------------------- mode stiffness of a material
    def _kappa_of_tensor(self, C, xp):
        """kappa[i, alpha, ...] of a tensor C[d, d, d, d, ...] (elasticity) or A[d, d, ...] (conductivity)"""
        M = xp.asarray(self.M)
        if C.ndim >= 4 and C.shape[:4] == (3, 3, 3, 3):
            C_diag = xp.stack([C[i, :, i, :] for i in range(3)])        # C_{i d i e}: [i, d, e, ...]
            return xp.einsum('ade,ide...->ia...', M, C_diag)
        return xp.einsum('ade,de...->a...', M, C)[np.newaxis]          # conductivity: one component

    def stiffness(self, material):
        """eps * kappa as an array broadcastable to the hourglass strain field [i, alpha, q = 1, x, y, z]
        (a ScaledMaterial s(x) C_0 costs one pass over the grid: eps kappa(C_0) is a 3 x 4 constant times s(x))"""
        xp = tensor_operations.get_array_module(self.strain.s)
        over_space = (...,) + (np.newaxis,) * (self.discretization.domain_dimension + 1)     # q, x, y, z

        def eps_kappa_constant(tensor):                                 # [i, alpha, 1, 1, 1, 1]
            return self.eps * self._kappa_of_tensor(xp.asarray(tensor_operations.to_host(tensor)), xp)[over_space]

        if hasattr(material, 'terms'):                                  # ScaledMaterial: sum_k s_k(x) kappa(C_k)
            total = None
            for scale, tensor in material.terms:
                s = material._scale_array(scale)
                term = eps_kappa_constant(tensor) * (float(s) if np.ndim(s) == 0 else xp.asarray(s)[np.newaxis, np.newaxis])
                total = term if total is None else total + term
            return total
        if hasattr(material, 's'):                                      # muGrid field [d, d, d, d, q, x, y, z]
            return self.eps * self._kappa_of_tensor(material.s, xp)
        return eps_kappa_constant(material)                             # one constant (reference) tensor

    # ---------------------------------------------------------------- design sensitivity (topology optimization)
    def design_derivative(self, material_derivative, u_field, lambda_field):
        """[q = 1, x, y, z]: eps sum_{i alpha} dkappa_{i alpha} q_{i alpha}(u) q_{i alpha}(lambda), with dkappa the mode
        stiffness of the material derivative dC/drho_q (any material representation) -- the hourglass part of
        lambda^T (dK/drho_q) u at every quadrature point (no quadrature weight)"""
        fft = self.discretization.fft
        fft.communicate_ghosts(u_field)
        self.operator.apply(nodal_field=u_field, quadrature_point_field=self.strain)
        q_u = self.strain.s.copy()
        fft.communicate_ghosts(lambda_field)
        self.operator.apply(nodal_field=lambda_field, quadrature_point_field=self.strain)
        return (self.stiffness(material_derivative) * q_u * self.strain.s).sum(axis=(0, 1))

    def _scale_by_stiffness(self, material):
        """strain *= eps kappa(material), in place. A one-term ScaledMaterial s(x) C_0 (the usual case) needs no
        temporary: eps kappa = s(x) times a 3 x 4 constant, applied as two in-place products."""
        terms = getattr(material, 'terms', None)
        if terms is not None and len(terms) == 1:
            xp = tensor_operations.get_array_module(self.strain.s)
            scale, tensor = terms[0]
            scale = material._scale_array(scale)
            constant = self.eps * self._kappa_of_tensor(np.asarray(tensor_operations.to_host(tensor)), np)
            if xp is not np and isinstance(terms[0][0], Field):         # GPU: contiguous component vectors
                strain = tensor_operations.point_view(self.strain)      # [i, alpha, q = 1, points]
                scale = tensor_operations.point_view(terms[0][0]).reshape(-1)
                for i, alpha in np.ndindex(*constant.shape):
                    component = strain[i, alpha, 0]
                    xp.multiply(component, scale, out=component)
                    component *= float(constant[i, alpha])
            else:
                if np.ndim(scale) == 0:
                    constant = constant * float(scale)
                else:
                    self.strain.s[...] *= xp.asarray(scale)             # [q, x, y, z] broadcasts over (i, alpha)
                self.strain.s[...] *= xp.asarray(constant).reshape(constant.shape + (1,) * (self.strain.s.ndim - 2))
        else:
            self.strain.s[...] *= self.stiffness(material)

    # ---------------------------------------------------------------- K_hg u
    def add_stiffness(self, material, input_field, output_field):
        """output += H^T (eps kappa(material)) H input   (collective: ghost exchange)"""
        fft = self.discretization.fft
        fft.communicate_ghosts(input_field)
        self.operator.apply(nodal_field=input_field, quadrature_point_field=self.strain)       # q = H u
        self._scale_by_stiffness(material)                                                      # eps kappa q
        fft.communicate_ghosts(self.strain)
        self.operator.transpose(quadrature_point_field=self.strain, nodal_field=self.force, weights=[1.0])
        fft.communicate_ghosts(self.force)
        linalg.axpy(1.0, self.force, output_field)
