"""
Spectrum of the preconditioned cantilever operator M^-1 K (Green preconditioner) from the CG runs, and the scaling
of the CG iterations with the pixel size h.

The CG coefficients alpha_k, beta_k stored by cantilever_mesh_study.py give the Lanczos matrix
    T_kk = 1 / alpha_k + beta_{k-1} / alpha_{k-1},   T_k,k+1 = sqrt(beta_k) / alpha_k,
whose eigenvalues (Ritz values) approximate the spectrum of M^-1 K seen by CG; the extreme ones converge first.
Repeated Ritz values (round-off ghosts) are merged.

    1. Ritz values of every mesh (beam L = 1)          -> lambda_min, lambda_max, condition number vs h
    2. CG iterations vs h with the fitted power law, next to sqrt(condition number)
    3. Ritz values for the beam lengths L = 0.5 ... 4 at n_H = 64: how many small eigenvalues a longer beam adds

usage:  python spectrum_study.py
writes exp_data/spectrum_study.png, spectrum_check_dense.png (meshes stored with DENSE_SPECTRUM=1)
"""
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'exp_data', 'runs')
MESHES = (2, 4, 8, 16, 32, 64, 128, 256)
LENGTHS = (0.5, 1.0, 2.0, 4.0)
SMALL = 0.05  # 'small' eigenvalue threshold for the count in panel 3


def ritz_values(r):
    a, b = r['cg_alpha'], r['cg_beta']
    k = len(a)
    T = np.diag(1 / a)
    T[1:, 1:] += np.diag(b[:k - 1] / a[:k - 1])
    off = np.sqrt(b[:k - 1]) / a[:k - 1]
    T += np.diag(off, 1) + np.diag(off, -1)
    w = np.linalg.eigvalsh(T)
    keep = np.concatenate([[True], np.diff(np.log(w)) > 1e-6])  # merge ghosts
    return w[keep]


def load(n_H, tag=''):
    path = os.path.join(RUNS, f'n_H{n_H}_cgtol1e-06{tag}.npz')
    return np.load(path) if os.path.exists(path) else None


rows = []
fig, axes = plt.subplots(1, 4, figsize=(24, 5.4), layout='constrained')
for n_H in MESHES:
    r = load(n_H)
    if r is None or 'cg_alpha' not in r.files:
        continue
    w = ritz_values(r)
    rows.append((n_H, float(r['h']), int(r['cg_its']), w.min(), w.max()))
    axes[0].semilogy(np.full(len(w), n_H), w, '_', ms=14, mew=1.2, color='C0')
n_H, h, its, lam_min, lam_max = map(np.array, zip(*rows))
kappa = lam_max / lam_min
axes[0].set_xscale('log', base=2)
axes[0].set_xlabel('pixels over the beam height $n_H$')
axes[0].set_ylabel('Ritz values of $M^{-1}K$')
axes[0].set_title('spectrum seen by CG (one tick per Ritz value)', fontsize=10)

fit = slice(max(len(h) - 3, 0), len(h))  # the three finest meshes
slope_kappa = np.polyfit(np.log(h[fit]), np.log(kappa[fit]), 1)[0]
slope_min = np.polyfit(np.log(h[fit]), np.log(lam_min[fit]), 1)[0]
slope_its = np.polyfit(np.log(h[fit]), np.log(its[fit]), 1)[0]
axes[1].loglog(h, lam_min, 'o-', label=f'$\\lambda_{{min}}$  (finest: $\\propto h^{{{slope_min:.2f}}}$)')
axes[1].loglog(h, lam_max, 's-', label='$\\lambda_{max}$')
axes[1].loglog(h, kappa, '^-', label=f'condition number $\\kappa$  (finest: $\\propto h^{{{slope_kappa:.2f}}}$)')
axes[1].set_xlabel('pixel size h')
axes[1].set_title('extreme eigenvalues and condition number', fontsize=10)
axes[1].legend(frameon=False, fontsize=9)

axes[2].loglog(h, its, 'o-', label=f'CG iterations  (finest: $\\propto h^{{{slope_its:.2f}}}$)')
axes[2].loglog(h, np.sqrt(kappa) * its[-1] / np.sqrt(kappa[-1]), 'k:', label=r'$\sqrt{\kappa}$ (scaled to the finest)')
axes[2].loglog(h, its[-1] * (h / h[-1]) ** -1, '--', color='0.6', lw=1, label='$\\propto h^{-1}$')
axes[2].set_xlabel('pixel size h')
axes[2].set_ylabel('CG iterations to $10^{-6}$ (energy)')
axes[2].set_title('scaling of the CG iterations', fontsize=10)
axes[2].legend(frameon=False, fontsize=9)

counts = []
for k, L in enumerate(LENGTHS):
    r = load(64, '' if L == 1.0 else f'_L{L:g}')
    if r is None or 'cg_alpha' not in r.files:
        continue
    w = ritz_values(r)
    counts.append((L, int(np.sum(w < SMALL)), w.min(), int(r['cg_its'])))
    axes[3].semilogy(np.full(len(w), L), w, '_', ms=14, mew=1.2, color='C2')
axes[3].axhline(SMALL, color='C3', ls='--', lw=0.8)
axes[3].set_xscale('log', base=2)
axes[3].set_xlabel('beam length L  ($n_H$ = 64)')
axes[3].set_title(f'Ritz values vs beam length; dashed: {SMALL:g}', fontsize=10)
for ax in axes:
    ax.grid(True, color='0.9', lw=0.5)
fig.suptitle('cantilever with void, Green preconditioner: spectrum of $M^{-1}K$ from the CG coefficients',
             fontsize=12)
out = os.path.join(os.path.dirname(RUNS), 'spectrum_study.png')
fig.savefig(out, dpi=150)

print(f'{"n_H":>5s} {"h":>10s} {"CG its":>7s} {"lam_min":>10s} {"lam_max":>8s} {"kappa":>10s}')
for row, k in zip(rows, kappa):
    print(f'{row[0]:5d} {row[1]:10.3e} {row[2]:7d} {row[3]:10.3e} {row[4]:8.3f} {k:10.3e}')
print(f'finest three meshes: its ~ h^{slope_its:.2f}, kappa ~ h^{slope_kappa:.2f}, lam_min ~ h^{slope_min:.2f}')
print(f'{"L":>5s} {"#Ritz < " + str(SMALL):>12s} {"lam_min":>10s} {"CG its":>7s}')
for L, c, m, i in counts:
    print(f'{L:5g} {c:12d} {m:10.3e} {i:7d}')
print(f'wrote {out}')

# check on small meshes: the full spectrum of M^-1 K (dense, column by column; runs with DENSE_SPECTRUM=1)
# against the Ritz values of the CG run
dense_meshes = [n for n in MESHES if load(n, '_dense') is not None]
if dense_meshes:
    fig, ax = plt.subplots(figsize=(9, 5.5), layout='constrained')
    for k, n in enumerate(dense_meshes):
        r = load(n, '_dense')
        d = r['dense_eigenvalues']
        d = d[d > 1e-10 * d.max()]  # drop the zero eigenvalues of the void-only and clamped dofs
        w = ritz_values(r)
        ax.semilogy(np.full(len(d), k - 0.12), d, '_', ms=16, mew=0.8, color='0.55',
                    label='all eigenvalues (dense)' if k == 0 else None)
        ax.semilogy(np.full(len(w), k + 0.12), w, '_', ms=16, mew=1.4, color='C0',
                    label='Ritz values of the CG run' if k == 0 else None)
        print(f'dense n_H = {n}: {len(r["dense_eigenvalues"])} dofs, {len(d)} nonzero eigenvalues '
              f'[{d.min():.3e}, {d.max():.3f}]; Ritz [{w.min():.3e}, {w.max():.3f}]')
    ax.set_xticks(range(len(dense_meshes)), [f'$n_H$ = {n}' for n in dense_meshes])
    ax.set_ylabel('eigenvalues of $M^{-1}K$ (zero eigenvalues of void / clamped dofs left out)')
    ax.set_title('spectrum check: dense eigenvalues (grey) vs CG Ritz values (blue)', fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc='upper center', bbox_to_anchor=(0.3, 1.0))
    ax.grid(True, axis='y', color='0.9', lw=0.5)
    fig.savefig(os.path.join(os.path.dirname(RUNS), 'spectrum_check_dense.png'), dpi=150)
