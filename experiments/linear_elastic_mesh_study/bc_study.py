"""
Supports study (runs by run_bc.sh): does the h-dependence of CG come from the constrained dofs? The same beam
(L = 1, H = 0.2) with a void of fixed physical thickness 0.4 on each side (thick enough that the void no longer
matters, see void_study.py), Green preconditioner, three supports:

    clamp   left face clamped, shear load at the tip                       (2 (n_H + 1) constrained dofs)
    simple  pin + roller at the lower corners, shear load at mid-span      (3 constrained dofs)
    free    nothing constrained, -F at mid-span and +F/2 at both ends      (0; rigid motions are not loaded)
    traction the cantilever without Dirichlet: the clamp replaced by its reactions (parabolic shear, linear bending
            stress) as tractions on the left face, parabolic shear at the tip   (0; exact compliance known)

Without constraints and with the solid as the Green reference, every eigenvalue of M^-1 K is <= 1.

    1. CG iterations vs h        2. lambda_max vs h        3. lambda_min vs h        4. CG energy error vs iteration
    5. discretization error of the compliance: clamp vs its extrapolation, traction vs the exact solution

usage:  python bc_study.py
writes exp_data/bc_study.png
"""
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = os.path.dirname(os.path.realpath(__file__))
RUNS = os.path.join(HERE, 'exp_data', 'runs')
MESHES = (8, 16, 32, 64, 128)
SUPPORTS = {'clamp': 'clamped left face', 'simple': 'pin + roller (3 dofs)', 'free': 'free, balanced load',
            'traction': 'cantilever by tractions (no Dirichlet)'}
E, NU, F, L, H = 1.0, 0.3, -1e-4, 1.0, 0.2
EXACT = F ** 2 * L ** 3 / (3 * E / (1 - NU ** 2) * H ** 3 / 12) + 6 * F ** 2 * L / (5 * E / (2 * (1 + NU)) * H)
VOID = 0.4


def ritz_values(r):  # Lanczos matrix from the CG coefficients, as in spectrum_study.py
    a, b = r['cg_alpha'], r['cg_beta']
    k = len(a)
    T = np.diag(1 / a)
    T[1:, 1:] += np.diag(b[:k - 1] / a[:k - 1])
    off = np.sqrt(b[:k - 1]) / a[:k - 1]
    T += np.diag(off, 1) + np.diag(off, -1)
    return np.linalg.eigvalsh(T)


def load(n_H, bc):
    tag = f'_void{VOID:g}' + ('' if bc == 'clamp' else f'_bc{bc}')
    path = os.path.join(RUNS, f'n_H{n_H}_cgtol1e-06{tag}.npz')
    return np.load(path) if os.path.exists(path) else None


fig, axes = plt.subplots(2, 3, figsize=(20, 11), layout='constrained')
axes = axes.ravel()
print(f'{"supports":8s} {"n_H":>4s} {"h":>9s} {"CG its":>7s} {"lam_min":>10s} {"lam_max":>8s} {"compliance":>12s}')
for k, (bc, label) in enumerate(SUPPORTS.items()):
    rows = []
    for n_H in MESHES:
        r = load(n_H, bc)
        if r is None:
            continue
        w = ritz_values(r)
        rows.append((n_H, float(r['h']), int(r['cg_its']), w.min(), w.max(), float(r['compliance'])))
        print(f'{bc:8s} {n_H:4d} {rows[-1][1]:9.5f} {rows[-1][2]:7d} {w.min():10.3e} {w.max():8.3f} '
              f'{rows[-1][5]:12.6e}')
        if n_H == max(MESHES) or n_H == MESHES[-2] and load(max(MESHES), bc) is None:
            Pi = r['energy_functional']
            e = np.sqrt(np.maximum(Pi[:-1] - Pi[-1], 0) / -Pi[-1])
            e[e < 1e-7] = np.nan
            axes[3].semilogy(e, color=f'C{k}', lw=1.5, label=f'{label}, $n_H$ = {n_H}')
    if not rows:
        continue
    n_H, h, its, lam_min, lam_max, _ = map(np.array, zip(*rows))
    slope = np.polyfit(np.log(h[-3:]), np.log(its[-3:]), 1)[0] if len(h) >= 3 else np.nan
    axes[0].loglog(h, its, 'o-', color=f'C{k}', label=f'{label}  (finest: $\\propto h^{{{slope:.2f}}}$)')
    axes[1].loglog(h, lam_max, 'o-', color=f'C{k}', label=label)
    axes[2].loglog(h, lam_min, 'o-', color=f'C{k}', label=label)
    compliance = np.array([row[5] for row in rows])
    if bc == 'traction':
        reference, ref_label = EXACT, 'exact'
    elif bc == 'clamp' and len(h) >= 3:  # Richardson from the three finest
        p = np.log2((compliance[-3] - compliance[-2]) / (compliance[-2] - compliance[-1]))
        reference, ref_label = compliance[-1] + (compliance[-1] - compliance[-2]) / (2 ** p - 1), 'extrapolated'
    else:
        continue
    error = np.abs(compliance - reference) / reference
    sel = error > 0 if bc == 'traction' else slice(0, len(h) - 1)
    order = np.polyfit(np.log(h[sel]), np.log(error[sel]), 1)[0]
    axes[4].loglog(h[sel], error[sel], 'o-', color=f'C{k}', label=f'{label} vs {ref_label}  ($\\propto h^{{{order:.2f}}}$)')
    print(f'{bc}: compliance reference {reference:.6e} ({ref_label}), errors {np.array2string(error, precision=2)}')
axes[1].axhline(1.0, color='0.5', ls=':', lw=1)
axes[0].set_ylabel('CG iterations to $10^{-6}$ (energy)')
axes[1].set_ylabel(r'$\lambda_{max}$ of $M^{-1}K$ (largest Ritz value)')
axes[2].set_ylabel(r'$\lambda_{min}$ of $M^{-1}K$ (smallest Ritz value)')
axes[3].set_ylabel(r'relative energy error $\|u - u_k\|_K / \|u\|_K$')
axes[3].set_xlabel('CG iteration')
axes[4].set_ylabel('relative error of the compliance')
for ax, title in zip(axes[:5], ('CG iterations vs h', 'largest eigenvalue vs h', 'smallest eigenvalue vs h',
                            'CG convergence on the finest mesh', 'discretization error')):
    ax.set_title(title, fontsize=10)
    ax.grid(True, color='0.9', lw=0.5)
    if ax not in (axes[1], axes[2]):  # those two share the legend in the last panel
        ax.legend(frameon=False, fontsize=9)
    if ax is not axes[3]:
        ax.set_xlabel('pixel size h')
        ax.set_xticks([0.2 / n for n in MESHES], [f'1/{5 * n}' for n in MESHES])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axes[5].axis('off')
axes[5].legend(*axes[1].get_legend_handles_labels(), loc='center', frameon=False, fontsize=12,
               title='supports (panels 2 and 3)')
fig.suptitle(f'beam L = 1, H = 0.2, void {VOID:g} on each side, Green preconditioner: effect of the supports',
             fontsize=12)
out = os.path.join(HERE, 'exp_data', 'bc_study.png')
fig.savefig(out, dpi=150)
print(f'wrote {out}')
