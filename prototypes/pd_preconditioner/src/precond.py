"""Preconditioners: Green (G), Green-Jacobi (GJ), primal-dual Green (PD).

Every preconditioner is a callable r -> z (shapes (d, n, n)) that also counts its applications and 2D FFTs in
`counter` (a dict). With subtract_mean=True the mean of every component is removed from z.
"""
import numpy as np

from fem import apply_K, apply_Kref, BT, B, weight, apply_material


def new_counter():
    return {'K': 0, 'M': 0, 'fft': 0}


def assemble_green(Cref, d, n):
    """Algorithm 1 (Ladecky et al. 2026): impulse responses of K_ref, FFT, invert d x d blocks; k = 0 -> 0."""
    Khat = np.zeros((d, d, n, n), dtype=complex)
    for a in range(d):
        impulse = np.zeros((d, n, n))
        impulse[a, 0, 0] = 1.0
        response = apply_Kref(impulse, Cref)                 # column a: K_ref[b, :, a, 0]
        for b in range(d):
            Khat[b, a] = np.fft.fft2(response[b])
    Ghat = np.zeros_like(Khat)
    blocks = np.moveaxis(Khat, (0, 1), (-2, -1))             # (n, n, d, d)
    nonzero = np.ones((n, n), bool)
    nonzero[0, 0] = False
    inv = np.zeros_like(blocks)
    inv[nonzero] = np.linalg.inv(blocks[nonzero])
    Ghat[...] = np.moveaxis(inv, (-2, -1), (0, 1))
    return Ghat


def apply_green(Ghat, r, counter=None):
    d = r.shape[0]
    rhat = np.fft.fft2(r)
    zhat = np.einsum('ab...,b...->a...', Ghat, rhat)
    if counter is not None:
        counter['fft'] += 2 * d
    return np.real(np.fft.ifft2(zhat))


def jacobi_diagonal_brute(C, d, n):
    diag = np.zeros((d, n, n))
    for a in range(d):
        for i in range(n):
            for j in range(n):
                e = np.zeros((d, n, n))
                e[a, i, j] = 1.0
                diag[a, i, j] = apply_K(e, C)[a, i, j]
    return diag


def jacobi_diagonal(C, d, n):
    """Algorithm 2: stride-2 combs, d 2^d operator applications (nodes couple only within distance 1)."""
    assert n % 2 == 0
    diag = np.zeros((d, n, n))
    for a in range(d):
        for oi in (0, 1):
            for oj in (0, 1):
                comb = np.zeros((d, n, n))
                comb[a, oi::2, oj::2] = 1.0
                diag[a, oi::2, oj::2] = apply_K(comb, C)[a, oi::2, oj::2]
    return diag


def _finish(z, subtract_mean):
    if subtract_mean:
        z = z - z.mean(axis=(1, 2), keepdims=True)
    return z


def make_green(Cref, d, n, counter, subtract_mean=True):
    Ghat = assemble_green(Cref, d, n)

    def M(r):
        counter['M'] += 1
        return _finish(apply_green(Ghat, r, counter), subtract_mean)
    return M


def make_green_jacobi(C, Cref, d, n, counter, subtract_mean=True):
    Ghat = assemble_green(Cref, d, n)
    diag = jacobi_diagonal(C, d, n)
    diag = np.where(diag == 0, 1.0, diag)                    # voids: replace zero diagonal by one (paper)
    Jh = diag ** -0.5

    def M(r):
        counter['M'] += 1
        return _finish(Jh * apply_green(Ghat, Jh * r, counter), subtract_mean)
    return M


def make_primal_dual(rho, C0, Cref, d, n, counter, rho_clip=1e-8, subtract_mean=True):
    """M^-1 = G K_S G, K_S = B^T W S_eff B, S_eff = Cref C^-1 Cref with C = rho C0 (rho clamped by rho_clip)."""
    Ghat = assemble_green(Cref, d, n)
    S0 = Cref @ np.linalg.inv(C0) @ Cref
    inv_rho = 1.0 / np.maximum(rho, rho_clip)
    S_eff = S0[:, :, None, None] * inv_rho[None, None]
    w = weight(n)

    def M(r):
        counter['M'] += 1
        y = apply_green(Ghat, r, counter)
        y = BT(w * apply_material(S_eff, B(y)))
        return _finish(apply_green(Ghat, y, counter), subtract_mean)
    return M


def make_primal_dual_tangent(C, Cref, d, n, counter, subtract_mean=True):
    """PD for a general SPD material field C (per pixel (ds, ds, n, n) or per triangle (2, ds, ds, n, n)):
    S_eff = Cref C^-1 Cref at every quadrature point."""
    Ghat = assemble_green(Cref, d, n)
    blocks = np.moveaxis(C, (-4, -3), (-2, -1)) if C.ndim == 5 else np.moveaxis(C, (0, 1), (-2, -1))
    S_blocks = Cref @ np.linalg.inv(blocks) @ Cref
    S_eff = np.moveaxis(S_blocks, (-2, -1), (-4, -3)) if C.ndim == 5 else np.moveaxis(S_blocks, (-2, -1), (0, 1))
    w = weight(n)

    def M(r):
        counter['M'] += 1
        y = apply_green(Ghat, r, counter)
        y = BT(w * apply_material(S_eff, B(y)))
        return _finish(apply_green(Ghat, y, counter), subtract_mean)
    return M


I_MANDEL = np.array([1.0, 1.0, 0.0])
P_DEV_MANDEL = np.eye(3) - 0.5 * np.outer(I_MANDEL, I_MANDEL)


def isotropic_projection(C):
    """(kappa, mu) of the isotropic projection of a 2D Mandel stiffness C (..., 3, 3), C_iso = kappa I(x)I + 2 mu P_dev:
    kappa = I.C.I / 4, mu = tr(P_dev C P_dev) / 4."""
    kappa = np.einsum('a,...ab,b->...', I_MANDEL, C, I_MANDEL) / 4.0
    mu = np.einsum('ab,...bc,ca->...', P_DEV_MANDEL, C, P_DEV_MANDEL) / 4.0
    return kappa, mu


def modal_dual_moduli(kappa, mu, kappa_r, mu_r):
    """Isotropic dual material S = s_kappa I(x)I + 2 s_mu P_dev whose acoustic moduli match those of C_ref C^-1 C_ref
    for an isotropic pair (2D): transverse s_mu = mu_r^2 / mu, longitudinal s_kappa + s_mu = (kappa_r+mu_r)^2/(kappa+mu)."""
    s_mu = mu_r ** 2 / mu
    s_kappa = (kappa_r + mu_r) ** 2 / (kappa + mu) - s_mu
    return s_kappa, s_mu


def make_primal_dual_modal(C, Cref, d, n, counter, subtract_mean=True, clip=None, auto_reference=False):
    """PD with the modally matched isotropic dual material (elasticity, 2D Mandel). C per pixel (3, 3, n, n) or per
    triangle (2, 3, 3, n, n); Cref isotropic. clip: lower bound for kappa+mu and mu relative to their maxima (only in
    the preconditioner). auto_reference: raise kappa_r (mu_r kept) so that kappa_r/mu_r >= max kappa/mu, which keeps
    the dual bulk modulus positive."""
    assert d == 2
    blocks = np.moveaxis(C, (-4, -3), (-2, -1)) if C.ndim == 5 else np.moveaxis(C, (0, 1), (-2, -1))
    kappa, mu = isotropic_projection(blocks)
    kappa_r, mu_r = isotropic_projection(Cref)
    if clip is not None:
        mu = np.maximum(mu, clip * mu.max())
        kappa = np.maximum(kappa + mu, clip * (kappa + mu).max()) - mu
    if auto_reference:
        kappa_r = max(kappa_r, mu_r * float(np.max(kappa / mu)))
        Cref = kappa_r * np.outer(I_MANDEL, I_MANDEL) + 2 * mu_r * P_DEV_MANDEL
    Ghat = assemble_green(Cref, d, n)
    s_kappa, s_mu = modal_dual_moduli(kappa, mu, kappa_r, mu_r)
    if np.any(s_kappa <= 0):
        raise ValueError('modal dual bulk modulus <= 0: choose C_ref with a larger kappa_r / mu_r')
    S_blocks = (s_kappa[..., None, None] * np.outer(I_MANDEL, I_MANDEL)
                + 2 * s_mu[..., None, None] * P_DEV_MANDEL)
    S_eff = np.moveaxis(S_blocks, (-2, -1), (-4, -3)) if C.ndim == 5 else np.moveaxis(S_blocks, (-2, -1), (0, 1))
    w = weight(n)

    def M(r):
        counter['M'] += 1
        y = apply_green(Ghat, r, counter)
        y = BT(w * apply_material(S_eff, B(y)))
        return _finish(apply_green(Ghat, y, counter), subtract_mean)
    return M
