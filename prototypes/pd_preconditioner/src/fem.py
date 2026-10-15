"""Linear triangles on a periodic n x n grid (two per pixel), matrix-free with np.roll.

Unknowns u: shape (d, n, n), d = 1 (scalar) or 2 (elasticity). Node (i, j) sits at (i h, j h), h = 1/n.
Pixel (i, j) is split into
    lower triangle (t = 0), nodes (i, j), (i+1, j), (i, j+1)
    upper triangle (t = 1), nodes (i+1, j+1), (i, j+1), (i+1, j)
One quadrature point per triangle, weight w = h^2 / 2; both triangles carry the material of the pixel.

Gradient:   grad(u) has shape (2, d, 2, n, n)  [triangle, component, derivative, i, j]
Strain-like quantity used by the material ("eps"), shape (2, ds, n, n):
    scalar      eps = grad u                            ds = 2
    elasticity  eps = Mandel strain [e11, e22, sqrt2 e12]   ds = 3
Material field C: shape (ds, ds, n, n) (one block per pixel), reference material Cref: (ds, ds).
"""
import numpy as np

SQRT2 = np.sqrt(2.0)


def shift(a, di, dj):
    """a[i + di, j + dj] (periodic), acting on the last two axes."""
    return np.roll(a, (-di, -dj), axis=(-2, -1))


def gradient(u):
    """grad(u), shape (2, d, 2, n, n)."""
    n = u.shape[-1]
    u10, u01, u11 = shift(u, 1, 0), shift(u, 0, 1), shift(u, 1, 1)
    lower = np.stack([u10 - u, u01 - u], axis=1)            # (d, 2, n, n)
    upper = np.stack([u11 - u01, u11 - u10], axis=1)
    return np.stack([lower, upper]) * n


def gradient_transposed(g):
    """Exact adjoint of gradient: g (2, d, 2, n, n) -> (d, n, n)."""
    n = g.shape[-1]
    lx, ly, ux, uy = g[0, :, 0], g[0, :, 1], g[1, :, 0], g[1, :, 1]
    # lower: (u[i+1,j] - u[i,j]) lx + (u[i,j+1] - u[i,j]) ly
    out = shift(lx, -1, 0) - lx + shift(ly, 0, -1) - ly
    # upper: (u[i+1,j+1] - u[i,j+1]) ux + (u[i+1,j+1] - u[i+1,j]) uy
    out += shift(ux + uy, -1, -1) - shift(ux, 0, -1) - shift(uy, -1, 0)
    return out * n


def strain_size(d):
    return 2 if d == 1 else 3


def B(u):
    """u (d, n, n) -> eps (2, ds, n, n)."""
    g = gradient(u)
    if u.shape[0] == 1:
        return g[:, 0]
    e11, e22 = g[:, 0, 0], g[:, 1, 1]
    e12 = 0.5 * (g[:, 0, 1] + g[:, 1, 0])
    return np.stack([e11, e22, SQRT2 * e12], axis=1)


def BT(s):
    """Exact adjoint of B: s (2, ds, n, n) -> (d, n, n)."""
    if s.shape[1] == 2:
        return gradient_transposed(s[:, None])
    g = np.zeros((2, 2, 2) + s.shape[2:])
    g[:, 0, 0] = s[:, 0]
    g[:, 1, 1] = s[:, 1]
    g[:, 0, 1] = s[:, 2] / SQRT2
    g[:, 1, 0] = s[:, 2] / SQRT2
    return gradient_transposed(g)


def weight(n):
    return 0.5 / n ** 2


def apply_material(C, eps):
    """sigma = C eps. C: (ds, ds) constant, (ds, ds, n, n) per pixel (same block for both triangles) or
    (2, ds, ds, n, n) per triangle (nonlinear tangents)."""
    if C.ndim == 2:
        return np.einsum('ab,tb...->ta...', C, eps)
    if C.ndim == 5:
        return np.einsum('tabij,tbij->taij', C, eps)
    return np.einsum('abij,tbij->taij', C, eps)


def apply_K(u, C):
    """B^T W C B u."""
    return BT(weight(u.shape[-1]) * apply_material(C, B(u)))


def apply_Kref(u, Cref):
    return apply_K(u, Cref)


def rhs(C, E, d, n):
    """f = -B^T W C E for a constant macroscopic eps E (ds,)."""
    ds = strain_size(d)
    eps = np.broadcast_to(np.asarray(E, float)[None, :, None, None], (2, ds, n, n))
    return -BT(weight(n) * apply_material(C, eps))


def mandel_isotropic(K0, mu0):
    """3 x 3 Mandel matrix of C_ijkl = K0 d_ij d_kl + mu0 (d_ik d_jl + d_il d_jk - 2/3 d_ij d_kl) (2D components)."""
    d = np.eye(2)
    C = (K0 * np.einsum('ij,kl->ijkl', d, d)
         + mu0 * (np.einsum('ik,jl->ijkl', d, d) + np.einsum('il,jk->ijkl', d, d)
                  - 2.0 / 3.0 * np.einsum('ij,kl->ijkl', d, d)))
    index = [(0, 0), (1, 1), (0, 1)]
    w = [1.0, 1.0, SQRT2]
    M = np.array([[w[I] * w[J] * C[index[I] + index[J]] for J in range(3)] for I in range(3)])
    assert np.allclose(M, M.T)
    return M


def material_field(rho, C0):
    """C(x) = rho(x) C0, shape (ds, ds, n, n)."""
    return C0[:, :, None, None] * rho[None, None]
