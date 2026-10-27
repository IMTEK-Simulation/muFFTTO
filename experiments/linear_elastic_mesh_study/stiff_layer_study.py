"""
A stiff layer instead of the clamped face (runs by run_stiff_layer.sh and run_fixed_nodes.sh): beam L = 1, H = 0.2,
void 0.4, tip shear load; left of the beam a layer kappa times stiffer than the beam, held only at its bottom and top
node. Layer thickness fixed in physical units (0.05 = H/4, solid lines) or one pixel (dashed, from
run_fixed_nodes.sh). Green preconditioner with the beam material as reference.

Expectation (eigenvalue = energy in the real material / energy in the reference): the eigenvalues split into a
cluster near 1 (beam) and one up to kappa (layer). A layer of fixed physical thickness keeps both clusters fixed in h,
so CG should need a bounded number of iterations, larger by about log(kappa) rather than by kappa.

    1. lambda_max vs h   2. lambda_min vs h   3. CG iterations vs h   4. how well the layer clamps
    5. CG convergence at n_H = 64   6. Ritz values at n_H = 64 (the clusters)

usage:  python stiff_layer_study.py
writes exp_data/stiff_layer_study.png
"""
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = os.path.dirname(os.path.realpath(__file__))
RUNS = os.path.join(HERE, 'exp_data', 'runs')
VOID, THICKNESS = 0.4, 0.05
KAPPAS = (1, 10, 100, 1000, 10000)
MESHES = (16, 32, 64, 128)


def ritz_values(r):  # Lanczos matrix from the CG coefficients, as in spectrum_study.py
    a, b = r['cg_alpha'], r['cg_beta']
    k = len(a)
    T = np.diag(1 / a)
    T[1:, 1:] += np.diag(b[:k - 1] / a[:k - 1])
    off = np.sqrt(b[:k - 1]) / a[:k - 1]
    T += np.diag(off, 1) + np.diag(off, -1)
    return np.linalg.eigvalsh(T)


def load(n_H, tag):
    path = os.path.join(RUNS, f'n_H{n_H}_cgtol1e-06{tag}.npz')
    return np.load(path) if os.path.exists(path) else None


def exact_error(r):
    Pi = r['energy_functional']
    e = np.sqrt(np.maximum(Pi[:-1] - Pi[-1], 0) / -Pi[-1])
    e[e < 1e-7] = np.nan
    return e


fig, axes = plt.subplots(2, 3, figsize=(20, 11), layout='constrained')
axes = axes.ravel()
colours = dict(zip(KAPPAS, plt.cm.viridis(np.linspace(0, 0.88, len(KAPPAS)))))
print(f'{"layer":>7s} {"kappa":>6s} {"n_H":>4s} {"CG its":>7s} {"lam_min":>10s} {"lam_max":>10s} '
      f'{"compliance / clamp":>19s}')
for kappa in KAPPAS:
    for suffix, ls, name in ((f'_t{THICKNESS:g}', '-', f'{THICKNESS:g}'), ('', '--', '1 px')):
        rows = []
        for n_H in MESHES:
            r = load(n_H, f'_void{VOID:g}_bcstiff{kappa:g}{suffix}')
            ref = load(n_H, f'_void{VOID:g}')
            if r is None or ref is None:
                continue
            w = ritz_values(r)
            rows.append((float(r['h']), int(r['cg_its']), w.min(), w.max(),
                         float(r['compliance'] / ref['compliance'])))
            print(f'{name:>7s} {kappa:6g} {n_H:4d} {rows[-1][1]:7d} {w.min():10.3e} {w.max():10.3f} '
                  f'{rows[-1][4]:19.5f}')
            if n_H == 64 and ls == '-':
                axes[4].semilogy(exact_error(r), color=colours[kappa], lw=1.4, label=f'κ = {kappa:g}')
                axes[5].semilogy(np.full(len(w), np.log10(kappa)), w, '_', ms=18, mew=1.0, color=colours[kappa])
        if not rows:
            continue
        h, its, lam_min, lam_max, ratio = map(np.array, zip(*rows))
        label = f'κ = {kappa:g}' if ls == '-' else None
        axes[0].loglog(h, lam_max, 'o', ls=ls, color=colours[kappa], label=label)
        axes[1].loglog(h, lam_min, 'o', ls=ls, color=colours[kappa], label=label)
        axes[2].loglog(h, its, 'o', ls=ls, color=colours[kappa], label=label)
        axes[3].semilogx(h, 100 * (ratio - 1), 'o', ls=ls, color=colours[kappa], label=label)
for bc, label in (('', 'clamped face (Dirichlet)'), ('_bctraction', 'forces only (no Dirichlet)')):
    rows = [(float(r['h']), int(r['cg_its']), ritz_values(r).max()) for r in
            (load(n, f'_void{VOID:g}{bc}') for n in MESHES) if r is not None]
    style = dict(color='C3' if bc == '' else 'k', marker='s', ls=':')
    if rows:
        h, its, lam_max = map(np.array, zip(*rows))
        axes[0].loglog(h, lam_max, label=label, **style)
        axes[2].loglog(h, its, label=label, **style)
    r = load(64, f'_void{VOID:g}{bc}')
    if r is not None:
        axes[4].semilogy(exact_error(r), lw=1.4, label=label, color=style['color'], ls=':')
for ax in axes[:4]:
    ax.plot([], [], 'k-', label=f'layer {THICKNESS:g} thick (fixed)')
    ax.plot([], [], 'k--', label='layer 1 pixel thick')
    ax.set_xlabel('pixel size h')
    ax.set_xticks([0.2 / n for n in MESHES], [f'1/{5 * n}' for n in MESHES])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axes[0].set_ylabel(r'$\lambda_{max}$ of $M^{-1}K$')
axes[1].set_ylabel(r'$\lambda_{min}$ of $M^{-1}K$')
axes[2].set_ylabel('CG iterations to $10^{-6}$ (energy)')
axes[3].set_ylabel('compliance / clamped compliance − 1  [%]')
axes[4].set_xlabel('CG iteration')
axes[4].set_ylabel(r'relative energy error $\|u - u_k\|_K / \|u\|_K$')
axes[5].set_xlabel(r'$\log_{10}\,\kappa$')
axes[5].set_ylabel('Ritz values of $M^{-1}K$')
for ax, title in zip(axes, ('largest eigenvalue', 'smallest eigenvalue', 'CG iterations',
                            'how well the layer clamps (same mesh)', f'CG convergence at $n_H$ = 64 (layer {THICKNESS:g})',
                            f'Ritz values at $n_H$ = 64 (layer {THICKNESS:g})')):
    ax.set_title(title, fontsize=11)
    ax.grid(True, color='0.9', lw=0.5)
    if ax is not axes[5]:
        ax.legend(frameon=False, fontsize=8)
fig.suptitle(f'stiff layer instead of a clamped face (held at 2 nodes), void {VOID:g}, Green preconditioner with the '
             'beam as reference', fontsize=12)
out = os.path.join(HERE, 'exp_data', 'stiff_layer_study.png')
fig.savefig(out, dpi=150)
print(f'wrote {out}')
