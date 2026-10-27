"""
Plain against corrected Green on a beam with holes (runs by run_holes.sh): beam L = 1, H = 0.2 with 4 holes of
radius 0.05 on the centre line, void 0.4, tip shear load; clamped face and pin + roller, each with plain Green (Q M Q)
and corrected Green (exact inverse of the reference problem with the fixed dofs), and the beam loaded by forces only.

    top:    the structure (deformed, n_H = 32);  CG iterations vs h;  largest and smallest eigenvalue vs h
    bottom: CG convergence at the finest mesh;  wall time (setup + CG) vs h;  time per CG iteration vs h

usage:  python holes_study.py [--hetero 10]      (heterogeneous beam of contrast kappa, see HETERO in the solver)
writes exp_data/holes_study[_het<kappa>].png
"""
import argparse
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.colors
from matplotlib.collections import PolyCollection

parser = argparse.ArgumentParser()
parser.add_argument('--hetero', type=float, default=1)
args = parser.parse_args()
het = f'_het{args.hetero:g}' if args.hetero != 1 else ''

HERE = os.path.dirname(os.path.realpath(__file__))
RUNS = os.path.join(HERE, 'exp_data', 'runs')
VOID, NB_HOLES, RADIUS = 0.4, 4, 0.05
L, H = 1.0, 0.2
MESHES = (16, 32, 64, 128)
base = f'_void{VOID:g}'
holes = f'_holes{NB_HOLES}r{RADIUS:g}'
CASES = [('clamped face, plain Green', '', 'C0', '--'),
         ('clamped face, corrected Green', '_pgd', 'C0', '-'),
         ('clamped face, corrected Green, S only', '_pgdS', 'C2', '-.'),
         ('pin + roller, plain Green', '_bcsimple', 'C1', '--'),
         ('pin + roller, corrected Green', '_bcsimple_pgd', 'C1', '-'),
         ('forces only (no Dirichlet)', '_bctraction', 'k', ':')]


def ritz_values(r):  # Lanczos matrix from the CG coefficients, as in spectrum_study.py
    a, b = r['cg_alpha'], r['cg_beta']
    k = len(a)
    T = np.diag(1 / a)
    T[1:, 1:] += np.diag(b[:k - 1] / a[:k - 1])
    off = np.sqrt(b[:k - 1]) / a[:k - 1]
    T += np.diag(off, 1) + np.diag(off, -1)
    return np.linalg.eigvalsh(T)


def load(n_H, bc):
    path = os.path.join(RUNS, f'n_H{n_H}_cgtol1e-06{base}{bc}{holes}{het}.npz')
    return np.load(path) if os.path.exists(path) else None


def exact_error(r):
    Pi = r['energy_functional']
    e = np.sqrt(np.maximum(Pi[:-1] - Pi[-1], 0) / -Pi[-1])
    e[e < 1e-7] = np.nan
    return e


fig, axes = plt.subplots(2, 3, figsize=(20, 11), layout='constrained')
axes = axes.ravel()

# the structure: solid pixels of the deformed beam (clamped, corrected Green), n_H = 32
r = load(32, '_pgd')
if r is not None:
    h, U = float(r['h']), r['u_beam']
    X = np.stack(np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[2]), indexing='ij')) * h
    centre = lambda i, j: ((i + 0.5) * h, (j + 0.5) * h)
    solid = [(i, j) for i in range(U.shape[1] - 1) for j in range(U.shape[2] - 1)
             if all((centre(i, j)[0] - (k + 0.5) * L / NB_HOLES) ** 2 + (centre(i, j)[1] - H / 2) ** 2 > RADIUS ** 2
                    for k in range(NB_HOLES))]
    Xd = X + 0.06 / np.abs(U).max() * U
    quads = lambda Y: [[Y[:, i, j], Y[:, i + 1, j], Y[:, i + 1, j + 1], Y[:, i, j + 1]] for i, j in solid]
    axes[0].add_collection(PolyCollection(quads(X), facecolors='none', edgecolors='0.8', linewidths=0.3))
    if het:  # colour by the stiffness factor of the heterogeneous beam (as in the solver)
        phi = np.array([0.5 * (1 + np.sin(2 * np.pi * centre(i, j)[0] / 0.2 + 0.7) *
                               np.sin(2 * np.pi * centre(i, j)[1] / 0.1 + 0.3)) for i, j in solid])
        pc = PolyCollection(quads(Xd), array=args.hetero ** -phi, cmap='viridis', edgecolors='none',
                            norm=matplotlib.colors.LogNorm(1 / args.hetero, 1))
        axes[0].add_collection(pc)
        fig.colorbar(pc, ax=axes[0], orientation='horizontal', shrink=0.7, pad=0.12).set_label('stiffness / beam')
    else:
        axes[0].add_collection(PolyCollection(quads(Xd), facecolors='#efef9f', edgecolors='#7a7a5a', linewidths=0.3))
    axes[0].plot(np.zeros(U.shape[2]), np.arange(U.shape[2]) * h, 's', color='k', ms=3)
    axes[0].annotate('', xy=(L, H + 0.02), xytext=(L, H + 0.15), arrowprops=dict(arrowstyle='->', color='C3', lw=1.5))
    axes[0].autoscale()
    axes[0].set_aspect('equal')
    axes[0].set_title(f'beam with {NB_HOLES} holes (r = {RADIUS:g}), clamped, $n_H$ = 32 (deformation magnified)',
                      fontsize=10)

print(f'{"case":32s} {"n_H":>4s} {"its":>5s} {"lam_min":>10s} {"lam_max":>8s} {"setup s":>8s} {"CG s":>8s} '
      f'{"ms/it":>7s} {"total s":>8s} {"compliance":>12s}')
for label, bc, c, ls in CASES:
    rows = []
    for n_H in MESHES:
        r = load(n_H, bc)
        if r is None:
            continue
        w = ritz_values(r)
        setup = float(r['setup_time']) if 'setup_time' in r.files else np.nan
        rows.append((float(r['h']), int(r['cg_its']), w.min(), w.max(), setup, float(r['time'])))
        print(f'{label:32s} {n_H:4d} {rows[-1][1]:5d} {w.min():10.3e} {w.max():8.3f} {setup:8.2f} {rows[-1][5]:8.2f} '
              f'{1e3 * rows[-1][5] / rows[-1][1]:7.1f} {setup + rows[-1][5]:8.2f} {float(r["compliance"]):12.6e}')
        if n_H == max(MESHES):
            axes[3].semilogy(exact_error(r), color=c, ls=ls, lw=1.4, label=label)
    if not rows:
        continue
    h, its, lam_min, lam_max, setup, cg = map(np.array, zip(*rows))
    axes[1].loglog(h, its, 'o', color=c, ls=ls, label=label)
    axes[2].loglog(h, lam_max, 'o', color=c, ls=ls, label=f'{label}: max')
    axes[2].loglog(h, lam_min, 's', color=c, ls=ls, ms=4, label=f'{label}: min')
    axes[4].loglog(h, setup + cg, 'o', color=c, ls=ls, label=label)
    axes[4].loglog(h, setup, 'x', color=c, ls=ls, lw=0.8, ms=6)
    axes[5].loglog(h, 1e3 * cg / its, 'o', color=c, ls=ls, label=label)
for ax in axes[[1, 2, 4, 5]]:
    ax.set_xlabel('pixel size h')
    ax.set_xticks([0.2 / n for n in MESHES], [f'1/{5 * n}' for n in MESHES])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axes[2].axhline(1, color='0.5', lw=0.8, ls=':')
axes[1].set_ylabel('CG iterations to $10^{-6}$ (energy)')
axes[2].set_ylabel(r'$\lambda_{max}$ (circles) and $\lambda_{min}$ (squares)')
axes[3].set_xlabel('CG iteration')
axes[3].set_ylabel(r'relative energy error $\|u - u_k\|_K / \|u\|_K$')
axes[4].set_ylabel('wall time [s]: setup + CG (circles), setup only (crosses)')
axes[5].set_ylabel('time per CG iteration [ms]')
for ax, title in zip(axes[1:], ('CG iterations vs h', 'extreme eigenvalues vs h',
                                f'CG convergence at $n_H$ = {max(MESHES)}', 'wall time vs h',
                                'cost of one CG iteration vs h')):
    ax.set_title(title, fontsize=11)
for ax in axes[1:]:
    ax.grid(True, color='0.9', lw=0.5)
for ax in (axes[1], axes[3], axes[5]):
    ax.legend(frameon=False, fontsize=8)
fig.suptitle(f'beam with holes' + (f', heterogeneous (contrast {args.hetero:g})' if het else '') +
             f', void {VOID:g}: plain against corrected Green (times: single thread, ' +
             ('all runs 6 in parallel)' if het else 'clamped face at $n_H$ = 64, 128 timed alone, the other runs 6 '
                                                    'in parallel)'), fontsize=12)
out = os.path.join(HERE, 'exp_data', f'holes_study{het}.png')
fig.savefig(out, dpi=150)
print(f'wrote {out}')
