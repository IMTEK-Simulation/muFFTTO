"""
Void-thickness study of the cantilever (runs by run_void.sh): a void of fixed PHYSICAL thickness t on each side of
the beam, instead of a fixed number of void pixels, on several meshes.

Theory (vacuum study of the strip, 2026-10-22): with the Green preconditioner, an eigenvalue of M^-1 K is the ratio
(energy in the real solid) / (energy in the reference material over the whole cell, void included). A mode of
wavelength l along the beam decays into the void like exp(-2 pi d / l); a thin void forces it to undo its jump
across the gap, which makes lambda small. With 2 void pixels the void thickness is 2h -> 0 under refinement; with
a fixed physical t >~ l / (2 pi) lambda_min and the CG iterations should no longer depend on h.

    1. CG iterations vs h, one line per t (and the 2-pixel void for comparison)
    2. smallest Ritz value lambda_min vs h, same lines
    3. lambda_min and CG iterations vs t on the finest mesh
    4. the tip deflection vs t: the void has no stiffness, so the solution must not depend on t

usage:  python void_study.py
writes exp_data/void_study.png
"""
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = os.path.dirname(os.path.realpath(__file__))
RUNS = os.path.join(HERE, 'exp_data', 'runs')
MESHES = (8, 16, 32, 64)
THICKNESSES = (0.025, 0.05, 0.1, 0.2, 0.4, 0.8)


def ritz_values(r):  # as in spectrum_study.py
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


cases = [('2 void pixels (t = 2h)', '')] + [(f't = {t:g}', f'_void{t:g}') for t in THICKNESSES]
colours = ['k'] + list(plt.cm.viridis(np.linspace(0, 0.9, len(THICKNESSES))))
fig, axes = plt.subplots(1, 4, figsize=(24, 5.4), layout='constrained')
table = {}
for (label, tag), c in zip(cases, colours):
    rows = []
    for n_H in MESHES:
        r = load(n_H, tag)
        if r is None or 'cg_alpha' not in r.files:
            continue
        rows.append((float(r['h']), int(r['cg_its']), ritz_values(r).min(), float(r['tip']), int(r['pixels'][0]),
                     int(r['pixels'][1])))
    if not rows:
        continue
    h, its, lam_min, tip, nx, ny = map(np.array, zip(*rows))
    table[label] = rows
    style = dict(color=c, lw=2 if tag == '' else 1.5, ls='--' if tag == '' else '-', marker='o', ms=4)
    axes[0].loglog(h, its, label=label, **style)
    axes[1].loglog(h, lam_min, **style)

finest = MESHES[-1]
t_rows = []
for t in THICKNESSES:
    r = load(finest, f'_void{t:g}')
    if r is not None and 'cg_alpha' in r.files:
        t_rows.append((t, int(r['cg_its']), ritz_values(r).min(), float(r['tip'])))
if t_rows:
    t, its_t, lam_t, tip_t = map(np.array, zip(*t_rows))
    axes[2].loglog(t, lam_t, 'o-', color='C0', label=r'$\lambda_{min}$')
    ax2 = axes[2].twinx()
    ax2.semilogx(t, its_t, 's-', color='C1', label='CG iterations')
    ax2.set_ylabel('CG iterations', color='C1')
    axes[2].set_ylabel(r'$\lambda_{min}$', color='C0')
    axes[2].axvline(1 / (2 * np.pi), color='0.6', ls=':', lw=1)
    axes[2].text(1 / (2 * np.pi), axes[2].get_ylim()[0], r' $L/2\pi$', color='0.4', fontsize=9, va='bottom')
    axes[3].semilogx(t, tip_t / tip_t[-1] - 1, 'o-', color='C2')
    axes[3].set_ylabel(f'tip deflection / (t = {t[-1]:g}) − 1')
axes[0].set_xlabel('pixel size h')
axes[0].set_ylabel('CG iterations to $10^{-6}$ (energy)')
axes[0].set_title('CG iterations vs h', fontsize=10)
axes[0].legend(frameon=False, fontsize=9)
axes[1].set_xlabel('pixel size h')
axes[1].set_ylabel(r'smallest Ritz value $\lambda_{min}$ of $M^{-1}K$')
axes[1].set_title(r'$\lambda_{min}$ vs h', fontsize=10)
axes[2].set_xlabel('void thickness t on each side')
axes[2].set_title(f'finest mesh $n_H$ = {finest}: vs void thickness', fontsize=10)
axes[3].set_xlabel('void thickness t on each side')
axes[3].set_title(f'solution check ($n_H$ = {finest}): the void carries nothing', fontsize=10)
for ax in axes:
    ax.grid(True, color='0.9', lw=0.5)
for ax in axes[2:]:  # void thicknesses as ticks, no minor labels
    ax.set_xticks(THICKNESSES, [f'{t:g}' for t in THICKNESSES])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
fig.suptitle('cantilever L = 1, H = 0.2, Green preconditioner: void of fixed physical thickness t vs 2 void pixels',
             fontsize=12)
out = os.path.join(HERE, 'exp_data', 'void_study.png')
fig.savefig(out, dpi=150)

for label, rows in table.items():
    print(label)
    for h, its, lam, tip, nx, ny in rows:
        print(f'   h = {h:.5f}  grid {nx:4d} x {ny:4d}  CG its {its:5d}  lambda_min {lam:.3e}  tip {tip:.8e}')
print(f'wrote {out}')
