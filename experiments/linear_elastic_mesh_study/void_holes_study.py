"""
How thin may the void around the structure be? (runs by run_void_holes.sh and run_holes.sh) Beam with 4 holes,
void 0.00625 ... 0.4 on each side (fixed in physical units, at least one pixel), clamped face with plain and corrected Green, and the
beam loaded by forces only.

    1. CG iterations vs h (corrected Green solid, plain Green dashed, forces only dotted), one colour per void
    2. smallest eigenvalue vs void thickness (finest mesh)
    3. grid size and wall time at n_H = 128 vs void thickness

usage:  python void_holes_study.py
writes exp_data/void_holes_study.png
"""
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = os.path.dirname(os.path.realpath(__file__))
RUNS = os.path.join(HERE, 'exp_data', 'runs')
VOIDS = (0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.4)
MESHES = (16, 32, 64, 128, 256, 512, 1024)
CASES = [('clamped, plain Green', '', '--', 'o'), ('clamped, corrected Green', '_pgd', '-', 'o'),
         ('forces only', '_bctraction', ':', 's')]


def ritz_values(r):  # Lanczos matrix from the CG coefficients, as in spectrum_study.py
    a, b = r['cg_alpha'], r['cg_beta']
    k = len(a)
    T = np.diag(1 / a)
    T[1:, 1:] += np.diag(b[:k - 1] / a[:k - 1])
    off = np.sqrt(b[:k - 1]) / a[:k - 1]
    T += np.diag(off, 1) + np.diag(off, -1)
    return np.linalg.eigvalsh(T)


def load(n_H, t, bc):
    path = os.path.join(RUNS, f'n_H{n_H}_cgtol1e-06_void{t:g}{bc}_holes4r0.05.npz')
    return np.load(path) if os.path.exists(path) else None


fig, axes = plt.subplots(1, 3, figsize=(20, 7.5), layout='constrained')
colours = dict(zip(VOIDS, plt.cm.plasma(np.linspace(0, 0.8, len(VOIDS)))))
print(f'{"void":>5s} {"case":26s} {"n_H":>4s} {"grid":>12s} {"its":>5s} {"lam_min":>10s} {"lam_max":>8s} '
      f'{"total s":>8s} {"compliance":>12s}')
summary = {}
for t in VOIDS:
    for label, bc, ls, marker in CASES:
        rows = []
        for n_H in MESHES:
            r = load(n_H, t, bc)
            if r is None:
                continue
            w = ritz_values(r)
            total = float(r['time']) + float(r['setup_time']) if 'setup_time' in r.files else float(r['time'])
            rows.append((float(r['h']), int(r['cg_its']), w.min(), w.max(), total, tuple(int(p) for p in r['pixels'])))
            print(f'{t:5g} {label:26s} {n_H:4d} {str(rows[-1][5]):>12s} {rows[-1][1]:5d} {w.min():10.3e} '
                  f'{w.max():8.3f} {total:8.1f} {float(r["compliance"]):12.6e}')
        if not rows:
            continue
        h, its = np.array([x[0] for x in rows]), np.array([x[1] for x in rows])
        axes[0].loglog(h, its, marker=marker, ls=ls, color=colours[t], label=f'void {t:g}, {label}')
        at_128 = [row for row in rows if abs(row[0] - 0.2 / 128) < 1e-12]  # panels 2, 3 compare at n_H = 128
        if at_128:
            summary[(t, bc)] = at_128[0]
for label, bc, ls, marker in CASES:
    pts = [(t, summary[(t, bc)]) for t in VOIDS if (t, bc) in summary]
    if pts:
        axes[1].loglog([p[0] for p in pts], [p[1][2] for p in pts], marker=marker, ls=ls, color='k', label=label)
        axes[2].semilogx([p[0] for p in pts], [p[1][4] for p in pts], marker=marker, ls=ls, color='k', label=label)
ax_px = axes[2].twinx()
pts = [(t, summary[(t, '_pgd')][5]) for t in VOIDS if (t, '_pgd') in summary]
if pts:
    ax_px.semilogx([p[0] for p in pts], [p[1][0] * p[1][1] / 1e6 for p in pts], 'x-', color='C2', label='grid size')
    ax_px.set_ylabel('pixels at $n_H$ = 128 [millions]', color='C2')
axes[0].set_xlabel('pixel size h')
axes[0].set_xticks([0.2 / n for n in MESHES], [f'1/{5 * n}' for n in MESHES])
axes[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axes[0].set_ylabel('CG iterations to $10^{-6}$ (energy)')
axes[0].set_title('CG iterations vs h', fontsize=11)
axes[1].set_xlabel('void thickness on each side')
axes[1].set_ylabel(r'$\lambda_{min}$ at $n_H$ = 128')
axes[1].set_title('smallest eigenvalue vs void thickness', fontsize=11)
axes[2].set_xlabel('void thickness on each side')
axes[2].set_ylabel('wall time at $n_H$ = 128 [s] (6 runs in parallel)')
axes[2].set_title('time and grid size vs void thickness', fontsize=11)
for ax in axes[1:]:
    ax.set_xticks(VOIDS, [f'{t:g}' for t in VOIDS])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
for ax in axes:
    ax.grid(True, color='0.9', lw=0.5)
    if ax is axes[0]:
        ax.legend(frameon=False, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2)
    else:
        ax.legend(frameon=False, fontsize=8)
fig.suptitle('beam with 4 holes: how thin may the void be? (plain Green dashed, corrected Green solid, forces only '
             'dotted)', fontsize=12)
out = os.path.join(HERE, 'exp_data', 'void_holes_study.png')
fig.savefig(out, dpi=150)
print(f'wrote {out}')
