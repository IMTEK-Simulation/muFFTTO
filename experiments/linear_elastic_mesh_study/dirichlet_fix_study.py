"""
Can Dirichlet conditions be made harmless for the Green preconditioner? (runs by run_dirichlet_fix.sh; beam L = 1,
H = 0.2, void 0.4, tip shear load)

    A. Green corrected to the exact inverse of the reference problem WITH the fixed dofs (PRECOND=green_dirichlet in
       cantilever_mesh_study.py): z = M (r - E lam) + N alpha with z = 0 on the fixed dofs. Since the real material is
       never stiffer than the reference, lambda_max <= 1 again, for any set of fixed dofs.
    B. A stiff layer (0.05 thick) held only by its reaction forces, no fixed dofs at all, plain Green.

compared with plain Green (projected, Q M Q) and with the cantilever loaded by forces only.

usage:  python dirichlet_fix_study.py
writes exp_data/dirichlet_fix_study.png
"""
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = os.path.dirname(os.path.realpath(__file__))
RUNS = os.path.join(HERE, 'exp_data', 'runs')
VOID, T = 0.4, 0.05
MESHES = (16, 32, 64, 128)


def ritz_values(r):  # Lanczos matrix from the CG coefficients, as in spectrum_study.py
    a, b = r['cg_alpha'], r['cg_beta']
    k = len(a)
    M = np.diag(1 / a)
    M[1:, 1:] += np.diag(b[:k - 1] / a[:k - 1])
    off = np.sqrt(b[:k - 1]) / a[:k - 1]
    M += np.diag(off, 1) + np.diag(off, -1)
    return np.linalg.eigvalsh(M)


def load(n_H, tag):
    path = os.path.join(RUNS, f'n_H{n_H}_cgtol1e-06{tag}.npz')
    return np.load(path) if os.path.exists(path) else None


def exact_error(r):
    Pi = r['energy_functional']
    e = np.sqrt(np.maximum(Pi[:-1] - Pi[-1], 0) / -Pi[-1])
    e[e < 1e-7] = np.nan
    return e


v = f'_void{VOID:g}'
CASES = [  # label, tag, colour, line style
    ('clamped face, plain Green', v, 'C0', '--'),
    ('clamped face, corrected Green', v + '_pgd', 'C0', '-'),
    ('pin + roller, plain Green', v + '_bcsimple', 'C1', '--'),
    ('pin + roller, corrected Green', v + '_bcsimple_pgd', 'C1', '-'),
    ('stiff layer κ=100 at 2 nodes, plain Green', v + f'_bcstiff100_t{T:g}', 'C2', '--'),
    ('stiff layer κ=100 at 2 nodes, corrected Green', v + f'_bcstiff100_t{T:g}_pgd', 'C2', '-'),
    ('stiff layer κ=100 held by forces', v + f'_bcstiffforce100_t{T:g}', 'C4', '-.'),
    ('forces only (no Dirichlet)', v + '_bctraction', 'k', ':'),
]
fig, axes = plt.subplots(2, 3, figsize=(20, 11), layout='constrained')
axes = axes.ravel()
print(f'{"case":48s} {"n_H":>4s} {"CG its":>7s} {"lam_min":>10s} {"lam_max":>10s} {"compliance":>12s}')
for label, tag, c, ls in CASES:
    rows = []
    for n_H in MESHES:
        r = load(n_H, tag)
        if r is None:
            continue
        w = ritz_values(r)
        rows.append((float(r['h']), int(r['cg_its']), w.min(), w.max()))
        print(f'{label:48s} {n_H:4d} {rows[-1][1]:7d} {w.min():10.3e} {w.max():10.3f} {float(r["compliance"]):12.6e}')
        if n_H == max(MESHES):
            axes[3].semilogy(exact_error(r), color=c, ls=ls, lw=1.4, label=label)
    if rows:
        h, its, lam_min, lam_max = map(np.array, zip(*rows))
        for ax, y in zip(axes[:3], (its, lam_max, lam_min)):
            ax.loglog(h, y, 'o', color=c, ls=ls, label=label)

# partial clamps with the corrected Green against plain Green (n_H = 16, 32, 64)
for nodes, c in ((3, 'C5'), (17, 'C6')):
    for pgd, ls in (('', '--'), ('_pgd', '-')):
        rows = []
        for n_H in (16, 32, 64):
            span = (nodes - 1) / n_H
            r = load(n_H, f'_span{span:g}{v}{pgd}')
            if r is not None:
                rows.append((float(r['h']), ritz_values(r).max(), int(r['cg_its'])))
        if rows:
            h, lam_max, its = map(np.array, zip(*rows))
            axes[4].loglog(h, lam_max, 'o', color=c, ls=ls,
                           label=f'{nodes} fixed nodes, ' + ('corrected' if pgd else 'plain') + ' Green')
            print(f'{nodes} fixed nodes {"corrected" if pgd else "plain"}: its {its}, lam_max {np.round(lam_max, 3)}')

# stiffness of the layer: CG iterations at n_H = 64 for the three ways of holding it
for name, tag, c, ls in (('held at 2 nodes, plain Green', '_bcstiff{k:g}_t' + f'{T:g}', 'C2', '--'),
                         ('held at 2 nodes, corrected Green', '_bcstiff{k:g}_t' + f'{T:g}_pgd', 'C2', '-'),
                         ('held by forces, plain Green', '_bcstiffforce{k:g}_t' + f'{T:g}', 'C4', '-.')):
    rows = [(k, int(r['cg_its'])) for k in (1, 10, 100, 1000, 10000)
            if (r := load(64, v + tag.format(k=k))) is not None]
    if rows:
        k, its = map(np.array, zip(*rows))
        axes[5].semilogx(k, its, 'o', color=c, ls=ls, label=name)
for ax in (axes[0], axes[1], axes[2], axes[4]):
    ax.set_xlabel('pixel size h')
    ax.set_xticks([0.2 / n for n in MESHES], [f'1/{5 * n}' for n in MESHES])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axes[1].axhline(1, color='0.5', lw=0.8, ls=':')
axes[4].axhline(1, color='0.5', lw=0.8, ls=':')
axes[0].set_ylabel('CG iterations to $10^{-6}$ (energy)')
axes[1].set_ylabel(r'$\lambda_{max}$')
axes[2].set_ylabel(r'$\lambda_{min}$')
axes[3].set_xlabel('CG iteration')
axes[3].set_ylabel(r'relative energy error $\|u - u_k\|_K / \|u\|_K$')
axes[4].set_ylabel(r'$\lambda_{max}$')
axes[5].set_xlabel('stiffness of the layer κ (× beam)')
axes[5].set_ylabel(f'CG iterations at $n_H$ = 64')
for ax, title in zip(axes, ('CG iterations vs h', 'largest eigenvalue vs h', 'smallest eigenvalue vs h',
                            f'CG convergence at $n_H$ = {max(MESHES)}', 'partial clamps: largest eigenvalue vs h',
                            'stiff layer: CG iterations vs its stiffness')):
    ax.set_title(title, fontsize=11)
    ax.grid(True, color='0.9', lw=0.5)
for ax in (axes[3], axes[4], axes[5]):
    ax.legend(frameon=False, fontsize=8)
fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=4, frameon=False, fontsize=9)
fig.suptitle(f'Dirichlet conditions with the Green preconditioner: plain vs corrected Green, void {VOID:g}',
             fontsize=12)
out = os.path.join(HERE, 'exp_data', 'dirichlet_fix_study.png')
fig.savefig(out, dpi=150)
print(f'wrote {out}')
