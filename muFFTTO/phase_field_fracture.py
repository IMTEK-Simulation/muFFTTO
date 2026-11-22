"""
Material models of phase-field fracture (AT1 and AT2), small strain.

    Pi(u, d) = int_Omega  g(d) psi^+(eps) + psi^-(eps)  dx  +  Gc / c_w  int_Omega  w(d) / l + l |grad d|^2  dx

* :func:`degradation` -- ``g(d) = (1 - d)^2 + k`` (``k << 1`` keeps the stiffness regular).
* :class:`CrackDensity` -- the crack surface density of AT2 (``w(d) = d^2``, ``c_w = 2``; Bourdin, Francfort &
  Marigo 2000) and AT1 (``w(d) = d``, ``c_w = 8/3``; Pham, Amor, Marigo & Maurini 2011): the coefficients of the
  damage equation and its operator.
* :func:`elastic_energy` -- ``psi_0 = lam/2 (tr eps)^2 + mu eps : eps`` and its tensile part ``psi^+`` of the
  splits ``'none'``, ``'amor'`` (volumetric-deviatoric, Amor, Marigo & Maurini 2009) and ``'spectral'`` (Miehe,
  Welschinger & Hofacker 2010).
* :func:`update_history` -- ``H = max(H_{n-1}, psi^+)`` (Miehe, Hofacker & Welschinger 2010).

With the history field the damage equation (stationarity of Pi in d, psi^+ replaced by H) is linear,

    AT2:  [Gc l B^T W B + N^T W (Gc / l + 2 H) N] d = N^T W 2 H
    AT1:  [3/4 Gc l B^T W B + N^T W 2 H N] d = N^T W (2 H - 3 Gc / (8 l)),   d >= 0 (bound, or H >= psi_c)

and SPD, solved by :func:`muFFTTO.solvers.conjugate_gradients`. The elastic problem of the 'hybrid' formulation
(Ambati, Gerasimov & De Lorenzis 2015) uses the isotropic degradation ``C(d) = g(d) C_0`` and stays linear; with
``C_0`` a constant tensor it is the :class:`muFFTTO.material_data.ScaledMaterial` ``g(d) C_0``, preconditioned by
:class:`muFFTTO.preconditioners.Dual`. Fields: the damage lives on a scalar ('conductivity') discretization of the
same grid as the displacement; quadrature-point arrays are ``[q, x, y(, z)]``.

``examples/phase_field_fracture/example_2D_phase_field_fracture_AT1_AT2.py`` is the staggered scheme with these
pieces.
"""
import numpy as np

from . import tensor_operations


def degradation(d, k=1e-4):
    """``g(d) = (1 - d)^2 + k``."""
    return (1.0 - d) ** 2 + k


def degradation_derivative(d):
    """``g'(d) = -2 (1 - d)``."""
    return -2.0 * (1.0 - d)


class CrackDensity:
    """
    AT1 or AT2 crack surface density ``Gc / c_w int w(d) / l + l |grad d|^2``, and its damage equation.

    Parameters
    ----------
    model : str
        ``'AT2'`` or ``'AT1'``.
    Gc : float
        Fracture toughness (critical energy release rate).
    length_scale : float
        Regularisation length ``l``.

    Attributes
    ----------
    diffusion : float
        ``2 Gc l / c_w``: coefficient of ``B^T W B`` (``Gc l`` for AT2, ``3/4 Gc l`` for AT1).
    reaction : float
        Constant part of the reaction ``N^T W c N``: ``Gc / l`` (AT2), 0 (AT1; its reaction is ``2 H`` only).
    rhs_offset : float
        ``Gc w'(0) / (c_w l)`` subtracted from ``2 H`` on the right-hand side: 0 (AT2), ``3 Gc / (8 l)`` (AT1).
    threshold : float
        ``psi_c``: tensile energy below which AT1 stays undamaged, ``3 Gc / (16 l)`` (0 for AT2). With
        ``H >= psi_c`` the AT1 right-hand side is nonnegative and the reaction positive, so the damage system stays
        SPD without the bound ``d >= 0``.
    preconditioner_reaction : float
        Reaction of the homogeneous operator that preconditions the damage system (AT2 ``Gc / l``, AT1 with the
        threshold ``2 psi_c = 3 Gc / (8 l)``): it makes the zero mode regular (``invert_zero_mode``).
    """

    def __init__(self, model, Gc, length_scale):
        if model not in ('AT1', 'AT2'):
            raise ValueError(f"model must be 'AT1' or 'AT2', got {model!r}")
        self.model, self.Gc, self.length_scale = model, Gc, length_scale
        l = length_scale
        if model == 'AT2':
            self.diffusion, self.reaction, self.rhs_offset, self.threshold = Gc * l, Gc / l, 0.0, 0.0
            self.preconditioner_reaction = Gc / l
        else:
            self.diffusion, self.reaction = 0.75 * Gc * l, 0.0
            self.rhs_offset, self.threshold = 0.375 * Gc / l, 0.1875 * Gc / l
            self.preconditioner_reaction = 2.0 * self.threshold

    def apply_operator(self, discretization, d, out, reaction_q, work_gradient, work_quad, work_nodal):
        """``out = [diffusion B^T W B + N^T W c N] d`` with the reaction ``c`` at the quadrature points
        (``reaction_q``, ``[q, x, y(, z)]`` or a scalar). The three work fields: gradient size, scalar quadrature
        field, unknown size."""
        discretization.apply_gradient_operator_mugrid(u_inxyz=d, grad_u_ijqxyz=work_gradient)
        work_gradient.s[...] *= self.diffusion
        discretization.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=work_gradient,
                                                                 div_u_fnxyz=out, apply_weights=True)
        discretization.apply_N_operator_mugrid(d, work_quad)
        work_quad.s[...] *= reaction_q
        discretization.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=work_quad, nodal_field_inxyz=work_nodal,
                                                          apply_weights=True)
        out.s[...] += work_nodal.s

    def reaction_field(self, history_q):
        """Reaction coefficient of the damage operator at the quadrature points: ``reaction + 2 H``."""
        return self.reaction + 2.0 * history_q

    def rhs(self, discretization, history_q, out, work_quad):
        """``out = N^T W (2 H - rhs_offset)``."""
        work_quad.s[...] = 2.0 * history_q - self.rhs_offset
        discretization.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=work_quad, nodal_field_inxyz=out,
                                                          apply_weights=True)

    def energy_density(self, d_q, grad_d_q):
        """``Gc / c_w (w(d) / l + l |grad d|^2)`` at the quadrature points (``grad_d_q``: ``[d, q, x, y(, z)]``)."""
        xp = tensor_operations.get_array_module(d_q)
        w, c_w = (d_q ** 2, 2.0) if self.model == 'AT2' else (d_q, 8.0 / 3.0)
        return self.Gc / c_w * (w / self.length_scale + self.length_scale * xp.sum(grad_d_q ** 2, axis=0))


def elastic_energy(strain, lam, mu, split='amor'):
    """
    Elastic energy density and its tensile part.

    Parameters
    ----------
    strain : array ``[d, d, ...]``
        Symmetric strain at the quadrature points.
    lam, mu : float or array broadcastable to ``strain.shape[2:]``
        Lame parameters.
    split : str
        ``'none'`` (``psi^+ = psi_0``), ``'amor'`` (``psi^+ = K/2 <tr eps>_+^2 + mu eps_dev : eps_dev``,
        ``K = lam + 2 mu / dim``) or ``'spectral'`` (``psi^+ = lam/2 <tr eps>_+^2 + mu sum_i <eps_i>_+^2`` with
        the principal strains ``eps_i``).

    Returns
    -------
    psi_0, psi_plus : arrays of shape ``strain.shape[2:]``
    """
    xp = tensor_operations.get_array_module(strain)
    dim = strain.shape[0]
    trace = xp.einsum('ii...->...', strain)
    psi_0 = 0.5 * lam * trace ** 2 + mu * xp.einsum('ij...,ij...->...', strain, strain)
    if split == 'none':
        return psi_0, psi_0
    if split == 'amor':
        bulk = lam + 2.0 * mu / dim
        deviator = strain - trace / dim * xp.eye(dim).reshape(dim, dim, *([1] * trace.ndim))
        return psi_0, 0.5 * bulk * xp.maximum(trace, 0.0) ** 2 + mu * xp.einsum('ij...,ij...->...', deviator,
                                                                                deviator)
    if split == 'spectral':
        principal = xp.linalg.eigvalsh(xp.moveaxis(strain, (0, 1), (-2, -1)))
        return psi_0, (0.5 * lam * xp.maximum(trace, 0.0) ** 2
                       + mu * xp.sum(xp.maximum(principal, 0.0) ** 2, axis=-1))
    raise ValueError(f"split must be 'none', 'amor' or 'spectral', got {split!r}")


def update_history(history_q, history_previous_q, psi_plus_q, threshold=0.0):
    """``H = max(H_{n-1}, psi^+, threshold)`` in place (the history of the last converged load step
    ``H_{n-1}``; the threshold ``psi_c`` of AT1, :attr:`CrackDensity.threshold`)."""
    xp = tensor_operations.get_array_module(history_q)
    history_q[...] = xp.maximum(xp.maximum(history_previous_q, psi_plus_q), threshold)
    return history_q
