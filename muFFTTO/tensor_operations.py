"""
tensor_operations.py
===============
Tensor field utilities for muGrid fields with layout [i, j, q, x, y, z].

All operators follow the contraction convention C_ij = A_ijkl B_lk
(innermost indices contracted first), consistent with standard
continuum mechanics notation.

All functions write their result IN-PLACE into the output field,
following the same convention as get_stress and get_algorithmic_tangent.

Operators
---------
trans2   : A_ji   = A_ij                          (transpose)
ddot42   : C_ij   = A_ijkl B_lk                   (4th-2nd double contraction)
ddot44   : C_ijmn = A_ijkl B_lkmn                 (4th-4th double contraction)
dot22    : C_ik   = A_ij B_jk                     (2nd-2nd single contraction)
dot24    : C_ikmn = A_ij B_jkmn                   (2nd-4th single contraction)
dot42    : C_ijkm = A_ijkl B_lm                   (4th-2nd single contraction)
dyad22   : C_ijkl = A_ij B_kl                     (dyadic product)

Field-level linear algebra
--------------------------
inv2     : pointwise inverse of a 2nd-order tensor field
det2     : pointwise determinant of a 2nd-order tensor field
log_field: pointwise natural log of a scalar field (for ln(J))
trace2   : pointwise trace of a 2nd-order tensor field
add_scaled_identity : A_ij += lam * s * delta_ij (e.g. the lambda tr(eps) I
                      term of Hooke's law)

Notes
-----
Every argument is a muGrid field (``muGrid.Field``), not a plain NumPy array.
The functions operate on the field's ``.s`` accessor, which exposes the data as
a NumPy array with the tensor (component) axes first, followed by the
quadrature-point (sub-point) axis ``q`` and the spatial grid axes
``x, y[, z]``:

- 2nd-order tensor field: ``.s`` has shape ``(d, d, nq, nx, ny[, nz])``
- 4th-order tensor field: ``.s`` has shape ``(d, d, d, d, nq, nx, ny[, nz])``
- scalar field:           ``.s`` has shape ``(1, 1, nq, nx, ny[, nz])`` and is
  therefore accessed as ``.s[0, 0]`` to obtain the ``(nq, nx, ny[, nz])`` array.

Because the result is assigned with ``C.s[...] = ...``, the output field must be
pre-allocated with the correct number of components. Avoid passing the same
field as input and output: e.g. ``einsum('ij...->ji...')`` in ``trans2``
returns a *view*, so an in-place transpose would read already-overwritten data.
"""

import numpy as np


# ============================================================================
# Einsum operators — layout [i, j, q, x, y, z]
# q is the quadrature index, x/y/z are grid indices
# The ellipsis '...' covers [q, x, y, z] uniformly for 2D and 3D
# ============================================================================

def trans2(A2, C2):
    """
    Pointwise transpose of a 2nd-order tensor field, C_ji = A_ij.

    Parameters
    ----------
    A2 : muGrid.Field
        Input 2nd-order tensor field, ``A2.s`` of shape ``(d, d, q, x, y[, z])``.
    C2 : muGrid.Field
        Output field of the same shape; overwritten in-place with A^T.

    Returns
    -------
    None
    """
    C2.s[...] = np.einsum('ij...->ji...', A2.s)

def ddot42(A4, B2, C2):
    """
    Double contraction of a 4th-order with a 2nd-order tensor field,
    C_ij = A_ijkl B_lk.

    Typical use: stress = C : strain (for symmetric strain the ordering ``lk``
    vs. ``kl`` makes no difference; for non-symmetric 2nd-order tensors such as
    the deformation gradient it does).

    Parameters
    ----------
    A4 : muGrid.Field
        4th-order tensor field, ``A4.s`` of shape ``(d, d, d, d, q, x, y[, z])``.
    B2 : muGrid.Field
        2nd-order tensor field, ``B2.s`` of shape ``(d, d, q, x, y[, z])``.
    C2 : muGrid.Field
        Output 2nd-order tensor field (overwritten in-place).

    Returns
    -------
    None
    """
    C2.s[...] = np.einsum('ijkl...,lk...->ij...', A4.s, B2.s)

def ddot44(A4, B4, C4):
    """
    Double contraction of two 4th-order tensor fields, C_ijmn = A_ijkl B_lkmn.

    Parameters
    ----------
    A4, B4 : muGrid.Field
        4th-order tensor fields, ``.s`` of shape ``(d, d, d, d, q, x, y[, z])``.
    C4 : muGrid.Field
        Output 4th-order tensor field (overwritten in-place).

    Returns
    -------
    None
    """
    C4.s[...] = np.einsum('ijkl...,lkmn...->ijmn...', A4.s, B4.s)

def dot22(A2, B2, C2):
    """
    Single contraction (matrix product) of two 2nd-order tensor fields,
    C_ik = A_ij B_jk, evaluated independently at every quadrature point.

    Parameters
    ----------
    A2, B2 : muGrid.Field
        2nd-order tensor fields, ``.s`` of shape ``(d, d, q, x, y[, z])``.
    C2 : muGrid.Field
        Output 2nd-order tensor field (overwritten in-place).

    Returns
    -------
    None
    """
    C2.s[...] = np.einsum('ij...,jk...->ik...', A2.s, B2.s)

def dot24(A2, B4, C4):
    """
    Single contraction of a 2nd-order with a 4th-order tensor field over the
    first index of the 4th-order tensor, C_ikmn = A_ij B_jkmn.

    Parameters
    ----------
    A2 : muGrid.Field
        2nd-order tensor field, ``.s`` of shape ``(d, d, q, x, y[, z])``.
    B4 : muGrid.Field
        4th-order tensor field, ``.s`` of shape ``(d, d, d, d, q, x, y[, z])``.
    C4 : muGrid.Field
        Output 4th-order tensor field (overwritten in-place).

    Returns
    -------
    None
    """
    C4.s[...] = np.einsum('ij...,jkmn...->ikmn...', A2.s, B4.s)

def dot42(A4, B2, C4):
    """
    Single contraction of a 4th-order with a 2nd-order tensor field over the
    last index of the 4th-order tensor, C_ijkm = A_ijkl B_lm.

    Parameters
    ----------
    A4 : muGrid.Field
        4th-order tensor field, ``.s`` of shape ``(d, d, d, d, q, x, y[, z])``.
    B2 : muGrid.Field
        2nd-order tensor field, ``.s`` of shape ``(d, d, q, x, y[, z])``.
    C4 : muGrid.Field
        Output 4th-order tensor field (overwritten in-place).

    Returns
    -------
    None
    """
    C4.s[...] = np.einsum('ijkl...,lm...->ijkm...', A4.s, B2.s)

def dyad22(A2, B2, C4):
    """
    Dyadic (outer) product of two 2nd-order tensor fields, C_ijkl = A_ij B_kl.

    Used e.g. to build terms like ``lambda * I (x) I`` of an elastic tangent.

    Parameters
    ----------
    A2, B2 : muGrid.Field
        2nd-order tensor fields, ``.s`` of shape ``(d, d, q, x, y[, z])``.
    C4 : muGrid.Field
        Output 4th-order tensor field (overwritten in-place).

    Returns
    -------
    None
    """
    C4.s[...] = np.einsum('ij...,kl...->ijkl...', A2.s, B2.s)


# ============================================================================
# Field-level linear algebra
# numpy linalg operates on last two axes, so we move [i,j] there and back
# ============================================================================

def inv2(A2, C2):
    """
    Pointwise inverse of a 2nd-order tensor field.
    A2 has shape [i, j, q, x, y, z] -> C2 same shape.

    Parameters
    ----------
    A2 : muGrid.Field
        Input field, ``A2.s`` of shape ``(d, d, q, x, y[, z])``. Every
        ``d x d`` block must be invertible.
    C2 : muGrid.Field
        Output field of the same shape, overwritten with A^{-1} pointwise.

    Returns
    -------
    None

    Raises
    ------
    numpy.linalg.LinAlgError
        If any of the pointwise matrices is singular.
    """
    # move tensor axes (0, 1) to the end -> shape (q, x, y[, z], d, d); np.linalg
    # treats all leading axes as a batch of d x d matrices
    A2_T = np.moveaxis(A2.s, [0, 1], [-2, -1])
    # invert the batch and move the tensor axes back to the front
    C2.s[...] = np.moveaxis(np.linalg.inv(A2_T), [-2, -1], [0, 1])

def det2(A2, c):
    """
    Pointwise determinant of a 2nd-order tensor field.
    A2 has shape [i, j, q, x, y, z] -> c has shape [q, x, y, z].
    c is a scalar muGrid field, accessed via c.s[0, 0].

    Typical use: J = det(F) for a deformation gradient field F.

    Parameters
    ----------
    A2 : muGrid.Field
        Input field, ``A2.s`` of shape ``(d, d, q, x, y[, z])``.
    c : muGrid.Field
        Scalar output field (``c.s`` of shape ``(1, 1, q, x, y[, z])``);
        ``c.s[0, 0]`` is overwritten in-place.

    Returns
    -------
    None
    """
    # batch of d x d matrices in the last two axes, see inv2
    A2_T = np.moveaxis(A2.s, [0, 1], [-2, -1])
    c.s[0, 0] = np.linalg.det(A2_T)

def log_field(a, c):
    """
    Pointwise natural logarithm of a scalar muGrid field.
    a has shape [q, x, y, z] accessed via a.s[0, 0] -> c same convention.

    Parameters
    ----------
    a : muGrid.Field
        Scalar input field; values must be positive (e.g. J = det F).
    c : muGrid.Field
        Scalar output field; ``c.s[0, 0]`` is overwritten with ``ln(a)``.

    Returns
    -------
    None
    """
    c.s[0, 0] = np.log(a.s[0, 0])

def trace2(A2, c):
    """
    Pointwise trace of a 2nd-order tensor field.
    A_ii -> c
    A2 has shape [i, j, q, x, y, z] -> c has shape [1, 1, q, x, y, z].

    Parameters
    ----------
    A2 : muGrid.Field
        Input field, ``A2.s`` of shape ``(d, d, q, x, y[, z])``.
    c : muGrid.Field
        Scalar output field; ``c.s[0, 0]`` is overwritten with ``A_ii``.

    Returns
    -------
    None
    """
    # 'ii...->...' sums the diagonal entries A_00 + A_11 (+ A_22)
    c.s[0, 0] = np.einsum('ii...->...', A2.s)

def add_scaled_identity(scalar_1qxyz, lam_1qxyz, A2, dim):
    """
    A_ij += lam * scalar * delta_ij   (adds scaled identity in-place)
    scalar_1qxyz : scalar field holding tr(ε), shape [1, 1, q, x, y, z]
    lam_1qxyz    : scalar field holding λ,     shape [1, 1, q, x, y, z]
    A2           : 2nd-order field to modify in-place
    dim          : number of spatial dimensions

    Typical use: the volumetric term ``lambda * tr(eps) * I`` of the linear
    elastic stress ``sigma = lambda tr(eps) I + 2 mu eps``.

    Parameters
    ----------
    scalar_1qxyz : muGrid.Field
        Scalar field ``s`` (e.g. tr(eps)), accessed via ``.s[0, 0]``.
    lam_1qxyz : muGrid.Field
        Scalar coefficient field ``lam`` (e.g. Lame parameter lambda),
        accessed via ``.s[0, 0]``.
    A2 : muGrid.Field
        2nd-order tensor field, ``A2.s`` of shape ``(d, d, q, x, y[, z])``;
        its diagonal is incremented in-place.
    dim : int
        Number of spatial dimensions ``d`` (2 or 3).

    Returns
    -------
    None
    """
    # only diagonal entries (i, i) receive the contribution (delta_ij)
    for i in range(dim):
        A2.s[i, i] += lam_1qxyz.s[0, 0] * scalar_1qxyz.s[0, 0]