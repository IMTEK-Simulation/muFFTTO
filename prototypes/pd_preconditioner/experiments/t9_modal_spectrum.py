"""T9 spectrum check of the PD diagnosis (elastic, n = 16, dense M^-1 K): matrix material with fixed bulk kappa = 20 and
shear mu (a) uniform 2000, (b) smoothly varying 500 ... 40000; C_ref = inclusion (kappa_r = 2, mu_r = 1).
Prediction for the pointwise PD: eigenvalues 1 (transverse) and (kappa_r^2/kappa + mu_r^2/mu)(kappa+mu)/(kappa_r+mu_r)^2
(longitudinal) ~ 45 in (a), a continuum ~ 11 ... 900 in (b). The modally matched PD should give 1 in (a)."""
import numpy as np
import scipy.linalg

from common import RESULTS, SURFACE, TEXT_1, TEXT_2, plt, style, write_csv
from fem import apply_K
from nonlinear import linear_isotropic
from precond import make_green, make_green_jacobi, make_primal_dual_modal, make_primal_dual_tangent, new_counter

n, d = 16, 2
C_ref = linear_isotropic(2.0, 1.0)
x = (np.arange(n) + 0.5) / n
x1, x2 = np.meshgrid(x, x, indexing='ij')
fields = {'uniform mu = 2000': np.full((n, n), 2000.0),
          'smooth mu 500...40000': 500.0 * 80.0 ** (0.5 * (1 + np.cos(2 * np.pi * x1) * np.cos(2 * np.pi * x2)))}


def dense(apply):
    size = d * n * n
    A = np.empty((size, size))
    for k in range(size):
        e = np.zeros((d, n, n))
        e[np.unravel_index(k, e.shape)] = 1.0
        A[:, k] = apply(e).ravel()
    return A


def predicted(kappa, mu, kappa_r=2.0, mu_r=1.0):
    return (kappa_r ** 2 / kappa + mu_r ** 2 / mu) * (kappa + mu) / (kappa_r + mu_r) ** 2


rows, spectra = [], {}
for name, mu in fields.items():
    C = np.einsum('ab,ij->abij', linear_isotropic(20.0, 1.0) - 2 * 1.0 * (np.eye(3) - 0.5 * np.outer([1, 1, 0], [1, 1, 0])),
                  np.ones((n, n))) + 2 * mu[None, None] * (np.eye(3) - 0.5 * np.outer([1, 1, 0], [1, 1, 0]))[:, :, None, None]
    K = dense(lambda u: apply_K(u, C))
    c = new_counter()
    for method, M in (('G', make_green(C_ref, d, n, c)), ('GJ', make_green_jacobi(C, np.eye(3), d, n, c)),
                      ('PD pointwise', make_primal_dual_tangent(C, C_ref, d, n, c)),
                      ('PD modal', make_primal_dual_modal(C, C_ref, d, n, c))):
        ev = scipy.linalg.eigvals(dense(M) @ K)
        lam = np.sort(ev.real)
        lam = lam[np.abs(lam) > 1e-8 * np.abs(lam).max()]
        spectra[(name, method)] = lam
        distinct = len(np.unique(np.round(np.log10(lam), 6)))
        rows.append(dict(material=name, method=method, nonzero=len(lam), lambda_min=f'{lam.min():.4g}',
                         lambda_max=f'{lam.max():.4g}', cond=f'{lam.max() / lam.min():.4g}',
                         distinct_values=distinct, max_imag=f'{np.abs(ev.imag).max():.1e}',
                         predicted_longitudinal=f'{predicted(20.0, mu.min()):.4g} ... {predicted(20.0, mu.max()):.4g}'))
        print(rows[-1], flush=True)
write_csv('t9_modal_spectrum.csv', rows)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), facecolor=SURFACE)
fig.subplots_adjust(left=0.07, right=0.98, top=0.8, bottom=0.12, wspace=0.2)
fig.text(0.07, 0.94, 'T9 spectra of M^-1 K, elastic n = 16: bulk kappa = 20 fixed, shear mu varies', fontsize=14,
         fontweight='bold', color=TEXT_1)
fig.text(0.07, 0.89, 'C_ref = inclusion (kappa_r = 2, mu_r = 1); pointwise PD: S = C_ref C^-1 C_ref, modal PD: matched '
                     'acoustic moduli', fontsize=10, color=TEXT_2)
colors = {'G': '#2a78d6', 'GJ': '#eb6834', 'PD pointwise': '#1baf7a', 'PD modal': '#eda100'}
for ax, name in zip(axes, fields):
    for method in colors:
        lam = spectra[(name, method)]
        ax.plot(np.arange(len(lam)), lam, color=colors[method], linewidth=2, label=method)
    ax.set_yscale('log')
    ax.grid(axis='y', color='#e6e5e1', linewidth=0.8)
    style(ax, name, 'index (sorted)', 'eigenvalue')
axes[0].legend(frameon=False, fontsize=9.5)
fig.savefig(f'{RESULTS}/t9_modal_spectrum.png', dpi=150, facecolor=SURFACE)
