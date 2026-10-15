"""Material density fields rho on the n x n grid pixels.

Data are defined on p x p data pixels (p <= n, powers of two); each covers (n/p) x (n/p) grid pixels. Values are
sampled at the lower-left corner of the data pixel.
"""
import numpy as np


def expand(rho_data, n):
    p = rho_data.shape[0]
    assert n % p == 0
    return np.repeat(np.repeat(rho_data, n // p, axis=0), n // p, axis=1)


def laminate(n, p, chi):
    """rho(j) = chi + (1 - chi) j/(p-1) for data column j (varies along x1, constant along x2): chi ... 1."""
    j = np.arange(p)
    column = chi + (1.0 - chi) * j / (p - 1)
    return expand(np.repeat(column[:, None], p, axis=1), n)


def cosine(n, p, chi):
    """0.5 + 0.25 (cos 2pi(x1 - x2) + cos 2pi(x1 + x2)) + 1/chi (chi = inf: true voids)."""
    x = np.arange(p) / p
    x1, x2 = np.meshgrid(x, x, indexing='ij')
    rho = 0.5 + 0.25 * (np.cos(2 * np.pi * (x1 - x2)) + np.cos(2 * np.pi * (x1 + x2)))
    rho = np.where(np.abs(rho) < 1e-14, 0.0, rho)            # exact zeros at the void points
    return expand(rho + (0.0 if np.isinf(chi) else 1.0 / chi), n)


def smooth(rho, steps):
    """Periodic 3 x 3 filter [1 2 1]^T [1 2 1] / 16, applied `steps` times."""
    for _ in range(steps):
        a = (np.roll(rho, 1, 0) + 2 * rho + np.roll(rho, -1, 0)) / 4
        rho = (np.roll(a, 1, 1) + 2 * a + np.roll(a, -1, 1)) / 4
    return rho


def disk(n, chi, steps):
    """rho = 1 outside, 1/chi inside the disk of radius 0.25 at the cell centre (pixel centres), smoothed."""
    x = (np.arange(n) + 0.5) / n
    x1, x2 = np.meshgrid(x, x, indexing='ij')
    rho = np.where((x1 - 0.5) ** 2 + (x2 - 0.5) ** 2 < 0.25 ** 2, 1.0 / chi, 1.0)
    return smooth(rho, steps)
