"""T4 spectra of M^-1 K (scalar, dense, n = 16 and 32) for G, GJ, PD on laminate, cosine and the sharp disk."""
import os
import sys

import numpy as np
import scipy.linalg

from common import COLORS, METHODS, NAMES, RESULTS, SURFACE, TEXT_1, TEXT_2, plt, problem, style, write_csv
from materials import cosine, disk, laminate
from fem import apply_K
from precond import make_green, make_green_jacobi, make_primal_dual, new_counter

d = 1
chi = 1e4
cases = {'laminate p=8': lambda n: laminate(n, 8, chi), 'cosine p=4': lambda n: cosine(n, 4, chi),
         'sharp disk': lambda n: disk(n, chi, 0)}


def dense(apply, size, shape):
    A = np.empty((size, size))
    for k in range(size):
        e = np.zeros(size)
        e[k] = 1.0
        A[:, k] = apply(e.reshape(shape)).ravel()
    return A


rows, spectra = [], {}
for n in (16, 32):
    shape = (d, n, n)
    size = d * n * n
    for case, make_rho in cases.items():
        rho = make_rho(n)
        C0, C, _ = problem(rho, d)
        K = dense(lambda u: apply_K(u, C), size, shape)
        counter = new_counter()
        preconditioners = {'G': make_green(C0, d, n, counter), 'GJ': make_green_jacobi(C, C0, d, n, counter),
                           'PD': make_primal_dual(rho, C0, C0, d, n, counter)}
        for method, M in preconditioners.items():
            Minv = dense(M, size, shape)
            ev = scipy.linalg.eigvals(Minv @ K)
            max_imag = np.abs(ev.imag).max()
            lam = np.sort(ev.real)
            nonzero = lam[np.abs(lam) > 1e-8 * np.abs(lam).max()]
            n_unit = int(np.sum(np.abs(nonzero - 1.0) < 1e-8))
            spectra[(n, case, method)] = nonzero
            rows.append(dict(n=n, case=case, method=method, nonzero=len(nonzero), lambda_min=f'{nonzero.min():.4e}',
                             lambda_max=f'{nonzero.max():.4e}', cond=f'{nonzero.max() / nonzero.min():.4e}',
                             eigenvalues_equal_1=n_unit, not_equal_1=len(nonzero) - n_unit,
                             max_imag=f'{max_imag:.1e}'))
            print(rows[-1], flush=True)
write_csv('t4_spectra_scalar.csv', rows)
np.savez(f'{RESULTS}/t4_spectra_scalar.npz', **{f'{n}|{c}|{m}': v for (n, c, m), v in spectra.items()})

for n in (16, 32):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=SURFACE)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.8, bottom=0.12, wspace=0.22)
    fig.text(0.06, 0.94, f'T4 spectra of M^-1 K, scalar, n = {n}, chi = {chi:g}', fontsize=14, fontweight='bold',
             color=TEXT_1)
    fig.text(0.06, 0.89, 'nonzero eigenvalues, sorted; log scale', fontsize=10, color=TEXT_2)
    for ax, case in zip(axes, cases):
        for method in METHODS:
            lam = spectra[(n, case, method)]
            ax.plot(np.arange(len(lam)), lam, color=COLORS[method], linewidth=2, label=NAMES[method])
        ax.set_yscale('log')
        ax.grid(axis='y', color='#e6e5e1', linewidth=0.8)
        style(ax, case, 'index (sorted)', 'eigenvalue' if case == 'laminate p=8' else None)
    axes[0].legend(frameon=False, fontsize=9.5)
    fig.savefig(f'{RESULTS}/t4_spectra_scalar_n{n}.png', dpi=150, facecolor=SURFACE)
