"""
1D calibration of the Wang (2006) sintering free energy (dimensionless A = 16, B = 1, k_rho = 10, k_eta = 5):
surface energy gamma_s*, grain-boundary energy gamma_gb*, 10-90 % surface width w*, and the mobility integrals
I_s = int rho^2 (1 - rho)^2 dx (free surface) and I_gb = int sum_{i != j} eta_i eta_j dx (grain boundary).
Writes geometry/wang_calibration.npz (dimensionless numbers; sintering_phase_field.py maps them to physical units).
"""
import os
import numpy as np

A, B, KR, KE = 16.0, 1.0, 10.0, 5.0
here = os.path.dirname(os.path.abspath(__file__))


def f_bulk(r, etas):
    s2 = sum(e ** 2 for e in etas)
    s3 = sum(e ** 3 for e in etas)
    return A * r ** 2 * (1 - r) ** 2 + B * (r ** 2 + 6 * (1 - r) * s2 - 4 * (2 - r) * s3 + 3 * s2 ** 2)


def df(r, etas):
    s2 = sum(e ** 2 for e in etas)
    s3 = sum(e ** 3 for e in etas)
    dr = A * (2 * r * (1 - r) ** 2 - 2 * r ** 2 * (1 - r)) + B * (2 * r - 6 * s2 + 4 * s3)
    de = [B * (12 * (1 - r) * e - 12 * (2 - r) * e ** 2 + 12 * e * s2) for e in etas]
    return dr, de


L, n = 80.0, 1600
x = np.linspace(0, L, n, endpoint=False)
h = x[1] - x[0]
k = 2 * np.pi * np.fft.rfftfreq(n, d=h)


def relax(r, etas, conserve_r, steps=40000, dt=2e-3):
    for _ in range(steps):
        dr, de = df(r, etas)
        if conserve_r is not None:
            dr = dr - dr.mean()
            r = np.fft.irfft((np.fft.rfft(r) - dt * np.fft.rfft(dr)) / (1 + dt * KR * k ** 2), n)
        etas = [np.fft.irfft((np.fft.rfft(e) - dt * np.fft.rfft(d)) / (1 + dt * KE * k ** 2), n)
                for e, d in zip(etas, de)]
    return r, etas


def energy(r, etas):
    g = lambda u: np.fft.irfft(1j * k * np.fft.rfft(u), n)
    return h * np.sum(f_bulk(r, etas) + KR / 2 * g(r) ** 2 + sum(KE / 2 * g(e) ** 2 for e in etas))


# free surface: solid slab in the middle (two interfaces)
slab = 0.5 * (np.tanh((x - L / 4) / 2) - np.tanh((x - 3 * L / 4) / 2))
r, (e,) = relax(slab.copy(), [slab.copy()], conserve_r=True)
gamma_s = energy(r, [e]) / 2
half = r[: n // 2]
w = abs(x[np.argmin(abs(half - 0.9))] - x[np.argmin(abs(half - 0.1))])
I_s = h * np.sum(r ** 2 * (1 - r) ** 2) / 2
# grain boundary: rho = 1, eta_1 slab, eta_2 complement (rho kept at its equilibrium near 1 by relaxing it too)
e1 = slab.copy()
r2, (e1, e2) = relax(np.ones(n), [e1, 1 - e1], conserve_r=None)
r2, (e1, e2) = relax(np.ones(n), [e1, e2], conserve_r=None)
gamma_gb = energy(r2, [e1, e2]) / 2
I_gb = h * np.sum(2 * e1 * e2) / 2
w_gb = abs(x[np.argmin(abs(e1[: n // 2] - 0.9))] - x[np.argmin(abs(e1[: n // 2] - 0.1))])
print(f'gamma_s* = {gamma_s:.4f}, gamma_gb* = {gamma_gb:.4f} (ratio {gamma_gb / gamma_s:.3f}), '
      f'surface 10-90 width w* = {w:.3f}, GB width {w_gb:.3f}, I_s = {I_s:.4f}, I_gb = {I_gb:.4f}')
np.savez(os.path.join(here, 'geometry', 'wang_calibration.npz'), A=A, B=B, k_rho=KR, k_eta=KE, gamma_s=gamma_s,
         gamma_gb=gamma_gb, width=w, width_gb=w_gb, I_s=I_s, I_gb=I_gb)
