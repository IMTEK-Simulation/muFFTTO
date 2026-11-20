"""
Preconditioners for :func:`muFFTTO.solvers.conjugate_gradients`.

Every preconditioner is a callable ``P(r, z)`` (``z = M r``, both muGrid fields of the unknown size of the
discretization) and has the method ``apply(r, z)``:

* :class:`Green` -- reference-material (Green) preconditioner ``G = K_ref^+`` of a homogeneous reference material
  ``C_ref``: block-diagonal in Fourier space. One node per pixel: muGrid's ``BlockFourierPreconditioner``, assembled
  from the impulse response of muFFTTO's ``K_ref`` (works for every element and formulation). Several nodes per pixel
  (e.g. biquadratic) or a regular ``K_ref`` (``invert_zero_mode``): muFFTTO's block inverse
  (:meth:`muFFTTO.domain.Discretization.get_preconditioner_Green_mugrid`).
* :class:`Jacobi` -- ``D^-1``, ``D = diag(K)`` (void entries, ``D = 0``, count as 1).
* :class:`GreenJacobi` -- ``D^-1/2 G D^-1/2`` (muGrid's ``GreenJacobiPreconditioner``); the usual reference is
  ``2 mu I_sym`` (:func:`isotropic_identity`): the Jacobi diagonal carries the local magnitude.
* :class:`Dual` -- the dual preconditioner ``G K(C_ref C^-1 C_ref) G``: Green applied before and after the stiffness
  of the dual material ``C_ref C(x)^-1 C_ref`` (:func:`dual_material`), with ``C_ref`` the stiff (solid) tangent.

Material updates: :meth:`Jacobi.update`, :meth:`GreenJacobi.update` recompute the diagonal for a new material; the
dual material of a :class:`muFFTTO.material_data.ScaledMaterial` follows its scale field (nothing to update), a full field
is recomputed with :meth:`Dual.update`.

Give every preconditioner that has to coexist on one discretization its own ``name``: the names select the work
fields.
"""
import numpy as np
from muGrid import Preconditioners as mugrid_preconditioners

from . import domain, tensor_operations
from .material_data import ScaledMaterial


class Green(mugrid_preconditioners.Preconditioner):
    """
    Green (reference-material) preconditioner ``z = K_ref^+ r``.

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
    reference_material : ndarray
        Homogeneous reference tangent, ``[d, d, d, d]`` (elasticity) or ``[d, d]`` (conductivity).
    formulation : str, optional
        ``'small_strain'`` for elasticity with the symmetrised gradient (as for the system matrix).
    name : str
        Prefix of the work fields.
    invert_zero_mode : bool
        Invert the zero-frequency block too (a regular ``K_ref``, e.g. with a mass term; muFFTTO's block route).
    operator : callable, optional
        ``operator(input_field_inxyz=..., output_field_inxyz=...)``: a translation-invariant reference operator
        instead of the stiffness of ``reference_material`` (e.g. the damage operator of phase-field fracture,
        diffusion + reaction, with ``invert_zero_mode=True``).
    """

    def __init__(self, discretization, reference_material=None, formulation=None, name='green',
                 invert_zero_mode=False, operator=None):
        self.discretization = discretization
        self.reference_material = reference_material
        self.formulation = formulation
        self.name = name
        self.operator = operator
        if discretization.nb_nodes_per_pixel == 1 and not invert_zero_mode:
            self._symbol = None
            self._mugrid = mugrid_preconditioners.make_reference_stiffness_preconditioner(
                discretization.fft, apply_reference_stiffness=self._reference_stiffness(),
                nb_components=int(discretization.unknown_size[0]), name=name)
        else:
            self._mugrid = None
            self._symbol = discretization.get_preconditioner_Green_mugrid(
                reference_material_data_ijkl=reference_material, formulation=formulation, operator=operator,
                invert_zero_mode=invert_zero_mode, name=name)

    def _reference_stiffness(self):
        """``f = K_ref u`` for muGrid's impulse response (its fields are copied into muFFTTO's: setup only)."""
        d = self.discretization
        u_work = d.get_unknown_size_field(name=f'{self.name}-impulse-u')
        f_work = d.get_unknown_size_field(name=f'{self.name}-impulse-f')

        def apply(u, f):
            u_work.s[...] = u.s.reshape(u_work.s.shape)
            if self.operator is not None:
                self.operator(input_field_inxyz=u_work, output_field_inxyz=f_work)
            else:
                d.apply_system_matrix_mugrid(material_data_field=self.reference_material, input_field_inxyz=u_work,
                                             output_field_inxyz=f_work, formulation=self.formulation)
            f.s[...] = f_work.s.reshape(f.s.shape)
        return apply

    def apply(self, r, z):
        if self._mugrid is not None:
            self._mugrid.apply(r, z)
        else:
            self.discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=self._symbol,
                                                            input_nodal_field_fnxyz=r, output_nodal_field_fnxyz=z)


def jacobi_diagonal(discretization, material, formulation=None, name='jacobi-diagonal'):
    """``diag(K)`` of the system matrix with ``material`` (muFFTTO's Dirac combs), as a field of the unknown size.
    Void entries (``K_ii = 0``) are returned as 1."""
    inverse_sqrt = discretization.get_preconditioner_Jacobi_mugrid(material_data_field_ijklqxyz=material,
                                                                   formulation=formulation,
                                                                   name=f'{name}-inverse-sqrt')
    diagonal = discretization.get_unknown_size_field(name=name)
    diagonal.s[...] = 1.0 / inverse_sqrt.s ** 2
    return diagonal


class Jacobi(mugrid_preconditioners.Preconditioner):
    """Jacobi preconditioner ``z = D^-1 r``, ``D = diag(K)`` of the system matrix with ``material``."""

    def __init__(self, discretization, material, formulation=None, name='jacobi'):
        self.discretization = discretization
        self.formulation = formulation
        self.name = name
        self.update(material)

    def update(self, material):
        """Recompute the diagonal for a new material."""
        self.diagonal = jacobi_diagonal(self.discretization, material, self.formulation, name=f'{self.name}-diagonal')

    def apply(self, r, z):
        z.s[...] = r.s / self.diagonal.s


class GreenJacobi(mugrid_preconditioners.Preconditioner):
    """
    Green-Jacobi preconditioner ``z = D^-1/2 G D^-1/2 r`` (muGrid's ``GreenJacobiPreconditioner``).

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
    reference_material : ndarray
        Reference of the Green operator, usually ``2 mu I_sym`` (:func:`isotropic_identity`) or the identity
        (conductivity): the Jacobi diagonal carries the local magnitude.
    material : muGrid field or ScaledMaterial
        Material of the system matrix (for its diagonal ``D``).
    formulation, name
        As for :class:`Green`.
    """

    def __init__(self, discretization, reference_material, material, formulation=None, name='green-jacobi'):
        self.discretization = discretization
        self.formulation = formulation
        self.name = name
        self.green = Green(discretization, reference_material, formulation=formulation, name=f'{name}-green')
        self._mugrid = mugrid_preconditioners.GreenJacobiPreconditioner(
            self.green, self._diagonal(material), name=name, communicator=discretization.communicator)

    def _diagonal(self, material):
        return jacobi_diagonal(self.discretization, material, self.formulation, name=f'{self.name}-diagonal')

    def update(self, material):
        """Recompute the Jacobi diagonal for a new material (the Green part does not change)."""
        self._mugrid.update_diagonal(self._diagonal(material))

    def apply(self, r, z):
        self._mugrid.apply(r, z)


class Dual(mugrid_preconditioners.Preconditioner):
    """
    Dual preconditioner ``z = G K_dual G r`` with ``K_dual = B^T W (C_ref C^-1 C_ref) B``.

    ``G`` is the Green preconditioner of ``C_ref``, the stiff (solid) tangent. For ``C = s(x) C_ref`` (a
    :class:`muFFTTO.material_data.ScaledMaterial`) the dual material is ``C_ref / s(x)``; it is evaluated from the scale of
    ``C`` at every application, so it follows the material without an update. ``C`` must be invertible everywhere
    (no true void: use a small residual stiffness).

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
    reference_material : ndarray
        ``C_ref``.
    material : muGrid field or ScaledMaterial
        The tangent ``C(x)`` of the system matrix.
    formulation : str, optional
        As for :class:`Green`.
    green : Green, optional
        A Green preconditioner of ``C_ref`` to share; default: a new one.
    name : str
        Prefix of the work fields.
    """

    def __init__(self, discretization, reference_material, material, formulation=None, green=None, name='dual'):
        self.discretization = discretization
        self.reference_material = np.asarray(reference_material)
        self.formulation = formulation
        self.name = name
        self.green = green if green is not None else Green(discretization, reference_material,
                                                           formulation=formulation, name=f'{name}-green')
        self._work = [discretization.get_unknown_size_field(name=f'{name}-work-{i}') for i in range(2)]
        self.dual_material = None
        self.update(material)

    def update(self, material):
        """Recompute the dual material ``C_ref C^-1 C_ref`` (a ScaledMaterial needs no update)."""
        self.dual_material = dual_material(self.discretization, self.reference_material, material,
                                           out=self.dual_material, name=f'{self.name}-material')

    def apply(self, r, z):
        work_1, work_2 = self._work
        self.green.apply(r, work_1)
        self.discretization.apply_system_matrix_mugrid(material_data_field=self.dual_material,
                                                       input_field_inxyz=work_1, output_field_inxyz=work_2,
                                                       formulation=self.formulation)
        self.green.apply(work_2, z)


def isotropic_identity(dim):
    """``I_sym``: the symmetric fourth-order identity, ``2 mu I_sym`` is the Green-Jacobi reference of elasticity."""
    identity = np.eye(dim)
    return 0.5 * (np.einsum('ik,jl->ijkl', identity, identity) + np.einsum('il,jk->ijkl', identity, identity))


def _mandel_pairs(dim):
    return [(0, 0), (1, 1), (0, 1)] if dim == 2 else [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def to_mandel(C):
    """Elasticity tensor ``C_ijkl[...]`` (minor symmetric) -> Mandel matrix ``M_ab[...]`` (3x3 in 2D, 6x6 in 3D);
    the inverse of ``M`` is the Mandel matrix of the inverse of ``C`` on symmetric tensors."""
    xp = tensor_operations.get_array_module(C)
    pairs = _mandel_pairs(C.shape[0])
    weight = [1.0 if i == j else np.sqrt(2.0) for i, j in pairs]
    return xp.stack([xp.stack([weight[a] * weight[b] * C[i, j, k, l] for b, (k, l) in enumerate(pairs)])
                     for a, (i, j) in enumerate(pairs)])


def from_mandel(M, dim):
    """Inverse of :func:`to_mandel`."""
    xp = tensor_operations.get_array_module(M)
    pairs = _mandel_pairs(dim)
    weight = [1.0 if i == j else np.sqrt(2.0) for i, j in pairs]
    C = xp.zeros((dim, dim, dim, dim) + M.shape[2:], dtype=M.dtype)
    for a, (i, j) in enumerate(pairs):
        for b, (k, l) in enumerate(pairs):
            for ii, jj in {(i, j), (j, i)}:
                for kk, ll in {(k, l), (l, k)}:
                    C[ii, jj, kk, ll] = M[a, b] / (weight[a] * weight[b])
    return C


def dual_material(discretization, reference_material, material, out=None, name='dual-material'):
    """
    The dual material ``C_ref C(x)^-1 C_ref``.

    * ``material`` a :class:`muFFTTO.material_data.ScaledMaterial` with one term ``s(x) C_ref``: ``C_ref / s(x)`` as a
      ScaledMaterial evaluated from ``s`` at every application (nothing stored).
    * conductivity field ``A_ij``: ``A_ref A^-1 A_ref`` pointwise.
    * elasticity field ``C_ijkl``: pointwise on the symmetric strains (Mandel), 2D and 3D.

    ``out``: the field to write a full-field result into (default: a new field ``name``).
    """
    if isinstance(material, ScaledMaterial):
        if len(material.terms) != 1 or not np.allclose(tensor_operations.to_host(material.terms[0][1]),
                                                       reference_material):
            raise ValueError('dual_material: a ScaledMaterial must be s(x) C_ref (one term with the reference)')
        scale = material.terms[0][0]
        return ScaledMaterial([(lambda: 1.0 / ScaledMaterial._scale_array(scale), reference_material)])
    xp = tensor_operations.get_array_module(material.s)
    if out is None:
        out = discretization.get_material_data_size_field_mugrid(name=name)
    reference = xp.asarray(reference_material)
    C = material.s
    if reference.ndim == 2:                                  # conductivity: A_ij[q, x, y(, z)]
        A = xp.moveaxis(C, (0, 1), (-2, -1))
        out.s[...] = xp.moveaxis(reference @ xp.linalg.inv(A) @ reference, (-2, -1), (0, 1))
        return out
    dim = discretization.domain_dimension
    M_ref = to_mandel(reference)
    M = xp.moveaxis(to_mandel(C), (0, 1), (-2, -1))
    out.s[...] = from_mandel(xp.moveaxis(M_ref @ xp.linalg.inv(M) @ M_ref, (-2, -1), (0, 1)), dim)
    return out
