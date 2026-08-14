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
"""

import numpy as np


# ============================================================================
# Einsum operators — layout [i, j, q, x, y, z]
# q is the quadrature index, x/y/z are grid indices
# The ellipsis '...' covers [q, x, y, z] uniformly for 2D and 3D
# ============================================================================

def trans2(A2, C2):
    """C_ji = A_ij"""
    C2.s[...] = np.einsum('ij...->ji...', A2.s)

def ddot42(A4, B2, C2):
    """C_ij = A_ijkl B_lk"""
    C2.s[...] = np.einsum('ijkl...,lk...->ij...', A4.s, B2.s)

def ddot44(A4, B4, C4):
    """C_ijmn = A_ijkl B_lkmn"""
    C4.s[...] = np.einsum('ijkl...,lkmn...->ijmn...', A4.s, B4.s)

def dot22(A2, B2, C2):
    """C_ik = A_ij B_jk"""
    C2.s[...] = np.einsum('ij...,jk...->ik...', A2.s, B2.s)

def dot24(A2, B4, C4):
    """C_ikmn = A_ij B_jkmn"""
    C4.s[...] = np.einsum('ij...,jkmn...->ikmn...', A2.s, B4.s)

def dot42(A4, B2, C4):
    """C_ijkm = A_ijkl B_lm"""
    C4.s[...] = np.einsum('ijkl...,lm...->ijkm...', A4.s, B2.s)

def dyad22(A2, B2, C4):
    """C_ijkl = A_ij B_kl"""
    C4.s[...] = np.einsum('ij...,kl...->ijkl...', A2.s, B2.s)


# ============================================================================
# Field-level linear algebra
# numpy linalg operates on last two axes, so we move [i,j] there and back
# ============================================================================

def inv2(A2, C2):
    """
    Pointwise inverse of a 2nd-order tensor field.
    A2 has shape [i, j, q, x, y, z] -> C2 same shape.
    """
    A2_T = np.moveaxis(A2.s, [0, 1], [-2, -1])
    C2.s[...] = np.moveaxis(np.linalg.inv(A2_T), [-2, -1], [0, 1])

def det2(A2, c):
    """
    Pointwise determinant of a 2nd-order tensor field.
    A2 has shape [i, j, q, x, y, z] -> c has shape [q, x, y, z].
    c is a scalar muGrid field, accessed via c.s[0, 0].
    """
    A2_T = np.moveaxis(A2.s, [0, 1], [-2, -1])
    c.s[0, 0] = np.linalg.det(A2_T)

def log_field(a, c):
    """
    Pointwise natural logarithm of a scalar muGrid field.
    a has shape [q, x, y, z] accessed via a.s[0, 0] -> c same convention.
    """
    c.s[0, 0] = np.log(a.s[0, 0])

def trace2(A2, c):
    """
    Pointwise trace of a 2nd-order tensor field.
    A_ii -> c
    A2 has shape [i, j, q, x, y, z] -> c has shape [1, 1, q, x, y, z].
    """
    c.s[0, 0] = np.einsum('ii...->...', A2.s)

def add_scaled_identity(scalar_1qxyz, lam_1qxyz, A2, dim):
    """
    A_ij += lam * scalar * delta_ij   (adds scaled identity in-place)
    scalar_1qxyz : scalar field holding tr(ε), shape [1, 1, q, x, y, z]
    lam_1qxyz    : scalar field holding λ,     shape [1, 1, q, x, y, z]
    A2           : 2nd-order field to modify in-place
    dim          : number of spatial dimensions
    """
    for i in range(dim):
        A2.s[i, i] += lam_1qxyz.s[0, 0] * scalar_1qxyz.s[0, 0]