"""
Figures of run_box_patch.sh: the family FFT preconditioner of S with a second level on the crossing patches
(2D square with 4 holes, all four faces clamped through the void).

    exp_data/box_patch_spectra.png   (a) sketch of the families and the crossing patches, (b) the sorted eigenvalues of
                                     C^-1 S for each preconditioner at one mesh, (c) cond(C^-1 S) vs N
    exp_data/box_patch_cost.png      inner steps per outer iteration and total Green applications vs N
also copied as report/figs/box_patch_*.png

usage:  python box_patch_plot.py
"""
import os
import shutil

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Rectangle, FancyBboxPatch

HERE = os.path.dirname(os.path.realpath(__file__))
RUNS = os.path.join(HERE, 'exp_data', 'runs_box')
NS = (32, 64, 128, 256)
N_SPEC = int(os.environ.get('N_SPEC', 128))
plt.rcParams.update({'font.size': 12})
FAMILY, ADD = 'tab:orange', '0.5'
BAL = {k: plt.get_cmap('Purples')(0.35 + 0.6 * i / 4) for i, k in enumerate((0, 2, 4, 8, 16))}
FAM_V, FAM_H, PATCH = 'tab:blue', 'tab:orange', 'tab:green'


def load(n, tag):
    f = os.path.join(RUNS, f'N{n}_box_{tag}.npz')
    return {k: v[()] if v.ndim == 0 else v for k, v in np.load(f).items()} if os.path.exists(f) else None


# ---------------------------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(21, 6.4), layout='constrained', gridspec_kw=dict(width_ratios=[1.05, 1, 1]))

# (a) sketch: periodic grid, two families of lines, crossing nodes in both, patches of +-k nodes around the crossings
ax = axes[0]
nx = ny = 16
sx = sy = (2, 13)
k_sk = 2
for i in range(nx):
    for j in range(ny):
        solid = sx[0] <= i < sx[1] and sy[0] <= j < sy[1]
        ax.add_patch(Rectangle((i, j), 1, 1, facecolor='#c9c9c9' if solid else 'white',
                               edgecolor='white' if solid else '#e8e8e8', lw=0.8))
ax.add_patch(Rectangle((0, 0), nx, ny, fill=False, ls='--', lw=1.2, edgecolor='0.35'))
for cx in sx:
    for cy in sy:
        for (x0_, y0_, w_, h_) in ((cx - 0.45, cy - k_sk - 0.45, 0.9, 2 * k_sk + 0.9),
                                   (cx - k_sk - 0.45, cy - 0.45, 2 * k_sk + 0.9, 0.9)):
            ax.add_patch(FancyBboxPatch((x0_, y0_), w_, h_, boxstyle='round,pad=0.12', facecolor=PATCH, alpha=0.28,
                                        edgecolor=PATCH, lw=1.2, zorder=3))
for x in sx:
    ax.plot([x] * ny, range(ny), 's', ms=6, mfc=FAM_V, mec=FAM_V, zorder=5)
for y in sy:
    xs = [i for i in range(nx) if i not in sx]
    ax.plot(xs, [y] * len(xs), 's', ms=6, mfc=FAM_H, mec=FAM_H, zorder=5)
for x in sx:
    for y in sy:
        ax.plot(x, y, 's', ms=9, mfc=FAM_V, mec=FAM_H, mew=2.4, zorder=6)
ax.plot([], [], 's', mfc=FAM_V, mec=FAM_V, label='vertical family (1D FFT)')
ax.plot([], [], 's', mfc=FAM_H, mec=FAM_H, label='horizontal family (1D FFT)')
ax.add_patch(Rectangle((0, 0), 0, 0, facecolor=PATCH, alpha=0.4, label=f'patch: crossing $\\pm{k_sk}$ nodes\n'
                                                                       '(second level, dense $\\mathbf{S}_{ZZ}$)'))
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=10, frameon=False)
ax.set(xlim=(-0.7, nx), ylim=(-0.7, ny), aspect='equal', xticks=[], yticks=[])
for s_ in ax.spines.values():
    s_.set_visible(False)
ax.set_title('(a) two levels: families by FFT, crossings by a small dense solve\n'
             '$\\mathbf{C}^{-1} = \\mathbf{Q} + (\\mathbf{I} - \\mathbf{Q}\\mathbf{S})\\,\\mathbf{M}^{-1}'
             '(\\mathbf{I} - \\mathbf{S}\\mathbf{Q}),\\ \\ \\mathbf{Q} = \\mathbf{Z}\\mathbf{S}_{ZZ}^{-1}\\mathbf{Z}^{\\top}$',
             fontsize=12)

# (b) the sorted eigenvalues of C^-1 S at N_SPEC
ax = axes[1]
spec = load(N_SPEC, 'dense')
curves = [('none', 'no preconditioner', 'k', ':'), ('family', 'families (overlap at crossings)', FAMILY, '-'),
          ('family_patch2add', '+ patch $\\pm2$, additive', ADD, '--'),
          ('family_patch2bal', '+ patch $\\pm2$, balancing', BAL[2], '-'),
          ('family_patch8bal', '+ patch $\\pm8$, balancing', BAL[8], '-')]
if spec is not None:
    for key, label, colour, ls in curves:
        ev = spec.get(f'eig_{key}')
        if ev is None:
            continue
        ax.plot(np.linspace(0, 1, len(ev)), ev, ls=ls, color=colour, lw=2.2,
                label=f'{label}: [{ev[0]:.2f}, {ev[-1]:.2f}], cond {ev[-1] / ev[0]:.1f}')
ax.set(yscale='log', xlabel='eigenvalue index (fraction of $m$)', ylabel='eigenvalue of $\\mathbf{C}^{-1}\\mathbf{S}$',
       title=f'(b) spectrum of the preconditioned $\\mathbf{{S}}$, N = {N_SPEC}\n'
             'balancing keeps $\\lambda_{\\max}$ near 1 and lifts $\\lambda_{\\min}$')
ax.legend(frameon=False, fontsize=9.5, loc='upper left')

# (c) cond vs N
ax = axes[2]
for key, label, colour, ls in curves[1:] + [('family_patch0bal', '+ crossing node only, balancing', BAL[0], '-'),
                                            ('family_patch8add', '+ patch $\\pm8$, additive', ADD, ':')]:
    pts = []
    for n in NS:
        r = load(n, 'dense')
        if r is not None and f'eig_{key}' in r:
            ev = r[f'eig_{key}']
            pts.append((n, ev[-1] / ev[0]))
    if pts:
        ax.plot(*zip(*pts), ls=ls, marker='o', color=colour, lw=2, ms=6)
        ax.annotate(f'{label}', pts[-1], textcoords='offset points', xytext=(8, -4), fontsize=9.5, color=colour)
ax.set(xscale='log', yscale='log', xlabel='N (pixels over the square)', ylabel='cond($\\mathbf{C}^{-1}\\mathbf{S}$)',
       title='(c) condition number vs mesh')
ax.set_xticks(NS, [str(n) for n in NS])
ax.set_xlim(26, 1500)
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
for ax in axes[1:]:
    ax.grid(True, color='0.92', which='both')
    ax.spines[['top', 'right']].set_visible(False)
out1 = os.path.join(HERE, 'exp_data', 'box_patch_spectra.png')
fig.savefig(out1, dpi=110)

# ---------------------------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout='constrained')
series = [('family1e-06', 'families only', FAMILY, '-'),
          ('family1e-06_patch2add', 'patch $\\pm2$, additive', ADD, '--')] + \
         [(f'family1e-06_patch{k}bal', f'patch $\\pm{k}$, balancing', BAL[k], '-') for k in (0, 2, 4, 8, 16)]
for key, label, colour, ls in series:
    pts = [(n, load(n, key)) for n in NS]
    pts = [(n, r) for n, r in pts if r is not None]
    if not pts:
        continue
    ns = [n for n, _ in pts]
    inner = [r['inner_its'].mean() for _, r in pts]
    total = [int(r['green_apps']) + int(r['setup_apps']) for _, r in pts]
    axes[0].plot(ns, inner, ls=ls, marker='o', color=colour, lw=2, ms=6)
    axes[0].annotate(f'{label}: {inner[-1]:.1f}', (ns[-1], inner[-1]), textcoords='offset points', xytext=(8, -4),
                     fontsize=9.5, color=colour)
    axes[1].plot(ns, total, ls=ls, marker='o', color=colour, lw=2, ms=6)
    axes[1].annotate(f'{label}: {total[-1]}', (ns[-1], total[-1]), textcoords='offset points', xytext=(8, -4),
                     fontsize=9.5, color=colour)
outer = sorted({abs(int(r['cg_its'])) for key, *_ in series for n in NS if (r := load(n, key)) is not None})
axes[0].set(ylim=(0, None), ylabel='inner PCG steps per outer iteration', title='inner steps per outer CG iteration')
axes[1].set(yscale='log', ylabel='Green applications (setup + solve)',
            title='total cost (the patch costs one Green application per patch dof, once)')
for ax in axes:
    ax.set(xscale='log', xlabel='N (pixels over the square)')
    ax.set_xticks(NS, [str(n) for n in NS])
    ax.set_xlim(26, 900)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(True, color='0.92', which='both')
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('2D square clamped on all four faces, inner PCG on S (tolerance 1e-6) with a second level on the crossing '
             f'patches; outer CG iterations unchanged ({min(outer)}-{max(outer)})', fontsize=13)
out2 = os.path.join(HERE, 'exp_data', 'box_patch_cost.png')
fig.savefig(out2, dpi=110)
os.makedirs(os.path.join(HERE, 'report', 'figs'), exist_ok=True)
for f in (out1, out2):
    shutil.copy(f, os.path.join(HERE, 'report', 'figs', os.path.basename(f)))
print('wrote', out1, out2)
