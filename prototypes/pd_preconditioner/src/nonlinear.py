"""Power-law elasticity (Ladecky et al. 2026, Sec. 4.5) in 2D small strain, Mandel notation [e11, e22, sqrt2 e12].

    sigma = K tr(eps) I + sigma0 (2/3) (eps_eq / eps0)^omega eps_dev / eps_eq,   eps_eq = sqrt(2/3) |eps_dev|

in the nonlinear matrix; a linear-elastic inclusion (K_inc, mu_inc). eps_dev uses the 2D trace (I_M = [1, 1, 0]).
Writing a = (2/3) sigma0 / eps0^omega, sigma_dev = a eps_eq^(omega-1) eps_dev and the consistent tangent is
    C_t = K I_M (x) I_M + a eps_eq^(omega-1) [P_dev + (omega-1) (2/3) eps_dev (x) eps_dev / eps_eq^2].
"""
import numpy as np

I_M = np.array([1.0, 1.0, 0.0])
P_DEV = np.eye(3) - 0.5 * np.outer(I_M, I_M)


def linear_isotropic(K, mu):
    return K * np.outer(I_M, I_M) + 2 * mu * P_DEV


def stress_and_tangent(eps, matrix_mask, K, sigma0, eps0, omega, C_inclusion, eps_eq_floor=1e-12):
    """eps: (2, 3, n, n) per triangle; matrix_mask: (n, n). Returns sigma (2, 3, n, n), C_t (2, 3, 3, n, n)."""
    tr = eps[:, 0] + eps[:, 1]
    dev = eps - 0.5 * tr[:, None] * I_M[None, :, None, None]
    eps_eq = np.maximum(np.sqrt(2.0 / 3.0) * np.linalg.norm(dev, axis=1), eps_eq_floor)
    a = 2.0 / 3.0 * sigma0 / eps0 ** omega
    secant = a * eps_eq ** (omega - 1)
    sigma_matrix = K * tr[:, None] * I_M[None, :, None, None] + secant[:, None] * dev
    C_matrix = (K * np.outer(I_M, I_M)[None, :, :, None, None]
                + secant[:, None, None] * (P_DEV[None, :, :, None, None]
                                          + (omega - 1) * 2.0 / 3.0 * np.einsum('tain,tbin->tabin', dev, dev)
                                          / eps_eq[:, None, None] ** 2))
    sigma_inclusion = np.einsum('ab,tbij->taij', C_inclusion, eps)
    C_inc = np.broadcast_to(C_inclusion[None, :, :, None, None], C_matrix.shape)
    m = matrix_mask[None, None]
    sigma = np.where(m, sigma_matrix, sigma_inclusion)
    C_t = np.where(matrix_mask[None, None, None], C_matrix, C_inc)
    return sigma, C_t
