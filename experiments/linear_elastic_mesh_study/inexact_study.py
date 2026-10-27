"""
Corrected Green with an inexact solve of the capacitance system S lam = E^T G r (runs by run_inexact.sh): how the
outer CG reacts to the accuracy of the inner CG, with and without the circulant wall preconditioner of S.
Beam with 4 holes, void 0.00625, clamped face, plain outer CG (energy stop 1e-6, delay 10).

    1. outer CG iterations vs the inner tolerance (relative residual), one line per mesh
    2. outer CG iterations vs a fixed number of inner steps
    3. total Green applications vs mesh: 2 per outer iteration + all inner steps (+ m to assemble S for the direct
       solve), for the direct solve and the cheapest inner variant of each family that keeps the outer count
    4. compliance error relative to the direct solve

usage:  python inexact_study.py      -> exp_data/inexact_study.png, exp_data/inexact_study.md
"""
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = os.path.dirname(os.path.realpath(__file__))
RUNS = os.path.join(HERE, 'exp_data', 'runs')
MESHES = (32, 64, 128, 256)
TOLS = (1e-8, 1e-6, 1e-4, 1e-2)
STEPS = {'': (5, 10, 20, 40), '_ipcirc': (1, 2, 4, 8)}
FAMILIES = {'': ('inner CG', 'Blues'), '_ipcirc': ('inner PCG, circulant wall preconditioner', 'Oranges')}


def load(n_H, tag):
    f = os.path.join(RUNS, f'n_H{n_H}_cgtol1e-06_void0.00625{tag}_holes4r0.05.npz')
    return {k: v[()] for k, v in np.load(f).items()} if os.path.exists(f) else None


def shade(cmap, n_H):
    return plt.get_cmap(cmap)(0.4 + 0.55 * MESHES.index(n_H) / (len(MESHES) - 1))


def green_apps(r, n_H, direct=False):
    outer = abs(int(r['cg_its']))
    return 2 * outer + (int(r['inner_its'].sum()) if 'inner_its' in r else 0) + \
        (2 * int(r['nb_clamped_nodes']) if direct else 0)


direct = {n: load(n, '_pgd') for n in MESHES}
fig, axes = plt.subplots(1, 4, figsize=(21, 5.4), layout='constrained')
rows = []
for fam, (label, cmap) in FAMILIES.items():
    marker, ls = ('o', '-') if fam == '' else ('s', '--')
    for n in MESHES:
        colour = shade(cmap, n)
        for ax, variants, x_of in ((axes[0], [(f'_pgdi{t:g}{fam}', t) for t in TOLS], None),
                                   (axes[1], [(f'_pgdik{k}{fam}', k) for k in STEPS[fam]], None)):
            pts = [(x, load(n, tag)) for tag, x in variants]
            pts = [(x, r) for x, r in pts if r is not None]
            ax.plot([x for x, _ in pts], [abs(int(r['cg_its'])) for _, r in pts], ls=ls, marker=marker, color=colour,
                    mfc=colour, ms=6, label=f'{label}, $n_H$ = {n}' if ax is axes[0] else None)
            for x, r in pts:  # not converged within CG_MAXITER: open marker
                if int(r['cg_its']) < 0:
                    ax.plot(x, -int(r['cg_its']), marker, color=colour, mfc='white', ms=9)
        for tag, x in [(f'_pgdi{t:g}{fam}', f'tol {t:g}') for t in TOLS] + \
                      [(f'_pgdik{k}{fam}', f'{k} steps') for k in STEPS[fam]]:
            r = load(n, tag)
            if r is None:
                continue
            dc = abs(float(r['compliance']) / float(direct[n]['compliance']) - 1) if direct[n] else np.nan
            rows.append((n, label, x, int(r['cg_its']), int(r['inner_its'].sum()), float(r['inner_its'].mean()),
                         green_apps(r, n), float(r['time']), dc))
for n in MESHES:
    if direct[n]:
        rows.append((n, 'direct S (assembled, dense inverse)', '-', int(direct[n]['cg_its']), 0, 0.0,
                     green_apps(direct[n], n, True), float(direct[n]['setup_time']) + float(direct[n]['time']), 0.0))
for ax in axes[:2]:
    ax.axhline(58, color='k', lw=1, ls=':')
    ax.text(ax.get_xlim()[0] if ax is axes[1] else 1.2e-8, 61, 'direct S: 58', fontsize=8)
    ax.set(yscale='log', ylabel='outer CG iterations')
    ax.grid(True, color='0.9', which='both')
axes[0].set(xscale='log', xlabel='inner tolerance (relative residual of S)',
            title='outer iterations vs inner tolerance')
axes[1].set(xscale='log', xlabel='inner steps per outer iteration (fixed)', title='outer iterations vs fixed inner steps')
axes[1].set_xticks([1, 2, 4, 5, 8, 10, 20, 40], ['1', '2', '4', '5', '8', '10', '20', '40'])

# 3: total Green applications of the cheapest variant per family that stays within 10 % of the direct outer count
for fam, (label, cmap) in FAMILIES.items():
    best = []
    for n in MESHES:
        cands = [r for r in rows if r[0] == n and r[1] == label and 0 < r[3] <= 1.1 * 58]
        if cands:
            best.append((n, min(cands, key=lambda r: r[6])))
    axes[2].plot([n for n, _ in best], [r[6] for _, r in best], ('o-' if fam == '' else 's--'),
                 color=shade(cmap, 256), label=f'best {label}')
    for n, r in best:
        axes[2].annotate(r[2], (n, r[6]), textcoords='offset points', xytext=(4, -10), fontsize=7)
d = [(n, r) for n, r in direct.items() if r]
axes[2].plot([n for n, _ in d], [green_apps(r, n, True) for n, r in d], 'v-', color='k',
             label='direct S (m Green applications to assemble + 2 per outer it.)')
axes[2].set(xscale='log', yscale='log', xlabel='$n_H$', ylabel='Green applications in total',
            title='cost: best variant within 10 % of 58 outer its.')
axes[2].set_xticks(MESHES, [str(n) for n in MESHES])
axes[2].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axes[2].legend(frameon=False, fontsize=8)
axes[2].grid(True, color='0.9', which='both')

for fam, (label, cmap) in FAMILIES.items():
    for n in MESHES:
        pts = [(t, load(n, f'_pgdi{t:g}{fam}')) for t in TOLS]
        pts = [(t, r) for t, r in pts if r is not None and direct[n]]
        axes[3].plot([t for t, _ in pts],
                     [max(abs(float(r['compliance']) / float(direct[n]['compliance']) - 1), 1e-12) for _, r in pts],
                     ('o-' if fam == '' else 's--'), color=shade(cmap, n))
axes[3].set(xscale='log', yscale='log', xlabel='inner tolerance', ylabel='|compliance / direct - 1|',
            title='accuracy of the solution (outer stop 1e-6)')
axes[3].grid(True, color='0.9', which='both')
fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=4, frameon=False, fontsize=8)
fig.suptitle('Corrected Green with the capacitance system S solved inexactly by an inner CG (clamped beam with holes, '
             'plain outer CG; open marker = no convergence in 5000)', fontsize=12)
fig.savefig(os.path.join(HERE, 'exp_data', 'inexact_study.png'), dpi=110)
with open(os.path.join(HERE, 'exp_data', 'inexact_study.md'), 'w') as out:
    out.write('| n_H | inner solve | setting | outer its | inner its total | inner/outer | Green apps | time [s] | '
              'compliance err |\n|' + '---|' * 9 + '\n')
    for r in sorted(rows, key=lambda r: (r[0], r[1], r[2])):
        out.write(f'| {r[0]} | {r[1]} | {r[2]} | {r[3] if r[3] > 0 else f"> {-r[3]} (no conv.)"} | {r[4]} | '
                  f'{r[5]:.1f} | {r[6]} | {r[7]:.1f} | {r[8]:.1e} |\n')
print('wrote exp_data/inexact_study.png, exp_data/inexact_study.md')
