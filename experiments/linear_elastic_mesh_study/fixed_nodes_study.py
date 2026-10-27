"""
Two questions about Dirichlet conditions and the Green preconditioner (runs by run_fixed_nodes.sh; beam L = 1,
H = 0.2, void 0.4 on each side, tip shear load):

    1. Does the conditioning depend on the number of fixed nodes? A centred partial clamp of k nodes (both components)
       on the left face, k = 3 ... n_H + 1, on three meshes. The same k on two meshes separates "how many fixed nodes"
       from "how small the pixels are".
    2. A stiff layer instead of the clamped face: one pixel column left of the beam, kappa times stiffer than the beam,
       held only at its bottom and top node. The reactions that hold the layer come out of those two nodes.

Theory: an eigenvalue of M^-1 K is (energy in the real material) / (energy in the Green reference = the beam
material everywhere). A layer kappa times stiffer than the reference allows lambda up to about kappa.

usage:  python fixed_nodes_study.py
writes exp_data/fixed_nodes_study.png
"""
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = os.path.dirname(os.path.realpath(__file__))
RUNS = os.path.join(HERE, 'exp_data', 'runs')
VOID = 0.4
KAPPAS = (10, 100, 1000, 10000)
MESHES_STIFF = (16, 32, 64, 128)
MESHES_SPAN = (16, 32, 64)


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

# 1. number of fixed nodes
print('1. partial clamp of k nodes')
print(f'{"n_H":>4s} {"k":>4s} {"CG its":>7s} {"lam_min":>10s} {"lam_max":>8s}')
for c, n_H in zip(('C0', 'C1', 'C2'), MESHES_SPAN):
    rows = []
    for k in (3, 5, 9, 17, 33, 65, n_H + 1):
        if k > n_H + 1 or any(row[0] == k for row in rows):
            continue
        span = (k - 1) / n_H
        r = load(n_H, (f'_span{span:g}' if span != 1 else '') + f'_void{VOID:g}')
        if r is None:
            continue
        w = ritz_values(r)
        rows.append((k, int(r['cg_its']), w.min(), w.max()))
        print(f'{n_H:4d} {k:4d} {rows[-1][1]:7d} {w.min():10.3e} {w.max():8.3f}')
    if rows:
        k, its, lam_min, lam_max = map(np.array, zip(*rows))
        axes[0].loglog(k, lam_max, 'o-', color=c, label=f'$n_H$ = {n_H} (h = 1/{5 * n_H})')
        axes[1].semilogx(k, its, 'o-', color=c, label=f'$n_H$ = {n_H}')
axes[0].axhline(1.0, color='0.5', ls=':', lw=1)
axes[0].set_xlabel('number of fixed nodes k (centred on the left face)')
axes[0].set_ylabel(r'$\lambda_{max}$ of $M^{-1}K$')
axes[0].set_title('1. largest eigenvalue vs number of fixed nodes', fontsize=11)
axes[1].set_xlabel('number of fixed nodes k')
axes[1].set_ylabel('CG iterations')
axes[1].set_title('1. CG iterations vs number of fixed nodes', fontsize=11)

# 2. stiff layer
print('2. stiff layer, fixed at 2 nodes')
print(f'{"kappa":>6s} {"n_H":>4s} {"CG its":>7s} {"lam_min":>10s} {"lam_max":>10s} {"compliance / clamp":>19s}')
colours = plt.cm.viridis(np.linspace(0, 0.85, len(KAPPAS)))
for kappa, c in zip(KAPPAS, colours):
    rows = []
    for n_H in MESHES_STIFF:
        r, ref = load(n_H, f'_void{VOID:g}_bcstiff{kappa:g}'), load(n_H, f'_void{VOID:g}')
        if r is None or ref is None:
            continue
        w = ritz_values(r)
        rows.append((float(r['h']), int(r['cg_its']), w.min(), w.max(), float(r['compliance'] / ref['compliance'])))
        print(f'{kappa:6g} {n_H:4d} {rows[-1][1]:7d} {w.min():10.3e} {w.max():10.3f} {rows[-1][4]:19.5f}')
        if n_H == 64:
            axes[5].semilogy(exact_error(r), color=c, lw=1.4, label=f'stiff layer κ = {kappa:g}')
    if rows:
        h, its, lam_min, lam_max, ratio = map(np.array, zip(*rows))
        axes[2].loglog(h, lam_max, 'o-', color=c, label=f'κ = {kappa:g}')
        axes[3].loglog(h, its, 'o-', color=c, label=f'κ = {kappa:g}')
        axes[4].semilogx(h, 100 * (ratio - 1), 'o-', color=c, label=f'κ = {kappa:g}')
for bc, c, label in (('', 'C3', 'clamped face (Dirichlet)'), ('_bctraction', 'k', 'forces only (no Dirichlet)')):
    rows = [(float(r['h']), int(r['cg_its']), ritz_values(r).max()) for r in
            (load(n, f'_void{VOID:g}{bc}') for n in MESHES_STIFF) if r is not None]
    if rows:
        h, its, lam_max = map(np.array, zip(*rows))
        axes[2].loglog(h, lam_max, 's--', color=c, label=label)
        axes[3].loglog(h, its, 's--', color=c, label=label)
    r = load(64, f'_void{VOID:g}{bc}')
    if r is not None:
        axes[5].semilogy(exact_error(r), color=c, ls='--', lw=1.4, label=label)
axes[2].set_ylabel(r'$\lambda_{max}$ of $M^{-1}K$')
axes[2].set_title('2. stiff layer: largest eigenvalue vs h', fontsize=11)
axes[3].set_ylabel('CG iterations')
axes[3].set_title('2. stiff layer: CG iterations vs h', fontsize=11)
axes[4].set_ylabel('compliance / clamped compliance − 1  [%]')
axes[4].set_title('2. how well the stiff layer clamps (same mesh)', fontsize=11)
axes[5].set_xlabel('CG iteration')
axes[5].set_ylabel(r'relative energy error $\|u - u_k\|_K / \|u\|_K$')
axes[5].set_title('2. CG convergence at $n_H$ = 64', fontsize=11)
for ax in axes[2:5]:
    ax.set_xlabel('pixel size h')
    ax.set_xticks([0.2 / n for n in MESHES_STIFF], [f'1/{5 * n}' for n in MESHES_STIFF])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
for ax in axes[:2]:  # node counts as ticks
    ax.set_xticks([3, 5, 9, 17, 33, 65], ['3', '5', '9', '17', '33', '65'])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
for ax in axes:
    ax.grid(True, color='0.9', lw=0.5)
    ax.legend(frameon=False, fontsize=9)
fig.suptitle(f'beam L = 1, H = 0.2, void {VOID:g}, Green preconditioner (beam material as reference): '
             'fixed nodes and a stiff layer', fontsize=12)
out = os.path.join(HERE, 'exp_data', 'fixed_nodes_study.png')
fig.savefig(out, dpi=150)
print(f'wrote {out}')
