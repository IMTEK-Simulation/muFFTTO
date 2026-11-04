"""
2D: does the smoothness of the damage profile (l/h) change how Green, Green-Jacobi and the dual scale with the grid?
Full runs of run_AT2_smoothness_2D.sh (N = 32, 64, 128; l = 1/8 ... 1/64; pores and stiff inclusions 10x) and
run_AT2_smoothness_2D_holesk.sh (pores of stiffness k = 1e-4 without a damage field, tag _holes0.0001) and
run_AT2_smoothness_2D_poresd1.sh (pores with d = 1 imposed: stiffness k C_0 and a damage halo around every pore).

Work per elastic solve = CG iterations x K applications per iteration (Green, Green-Jacobi 1, dual 2), mean over all
elastic solves of the run (lower = cheaper). Rows: pores (stiffness 1e-3 g(d), damage solved in them), pores of
stiffness k without damage (or ROW_POROUS=_poresd1), stiff inclusions; the same y axis in a row:

    left    sharp crack: fixed l/h (2 solid, 4 dashed) -- the crack always spans the same number of elements
    middle  smooth crack: fixed l (1/16 solid, 1/32 dashed) -- the crack spans more elements on finer grids
    right   fixed grid 128^2, l/h = 2 ... 16 -- the smoothness alone

usage:  python plot_AT2_smoothness_2D.py      writes figures/AT2_smoothness_2D.png and prints the table
"""
import os

import numpy as np
import matplotlib
import matplotlib.ticker

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
PRECS = [('Green', 'C2', 'o', 'Green', 1), ('Green_Jacobi', 'C0', 's', 'Green-Jacobi', 1),
         ('dual', 'C3', '^', 'dual (2 per iteration)', 2)]
LENGTHS = [0.125, 0.0625, 0.03125, 0.015625]


def work(n, p, incl, ell):
    pre, _, post = incl.partition('|')                    # tag parts before and after _l<ell>
    base = f'AT2_N{n}_{p}_polydisperse_pores{pre}_l{ell:g}{post}_smooth'
    f = os.path.join(HERE, 'exp_data', base + '_iterations.csv')
    log = os.path.join(HERE, 'logs', base + '.log')
    if not (os.path.exists(f) and os.path.exists(log)) or 'Elapsed time' not in open(log).read():
        return None
    d = np.genfromtxt(f, delimiter=',', names=True, dtype=None, encoding=None)
    return d['cg_u'].mean() * dict((q[0], q[4]) for q in PRECS)[p]


print(f'{"case":12s} {"N":>4s} {"l":>9s} {"l/h":>4s} | {"Green":>6s} {"GJ":>6s} {"dual(2x)":>8s}')
# ROW_POROUS (env): the second row; default the pores of stiffness k without a damage field (_holes0.0001),
# _poresd1 for the pores with d = 1 imposed, '_holes0.001|_k0.001' for pores and k both 1e-3
ROWS = [('', 'pores 1e-3, damage in pores'),
        (os.environ.get('ROW_POROUS', '_holes0.0001'),
         {'_holes0.0001': 'pores k, no damage in pores', '_poresd1': 'pores d = 1 imposed',
          '_holes0.001|_k0.001': 'pores = k = 1e-3, no damage in pores'}[
             os.environ.get('ROW_POROUS', '_holes0.0001')]),
        ('_incl10', 'stiff inclusions (10x)')]
fig, axes = plt.subplots(3, 3, figsize=(19, 17), layout='constrained', dpi=150)
for row, (incl, name) in enumerate(ROWS):
    for n in (32, 64, 128):
        for ell in LENGTHS:
            if n * ell >= 1:
                w = [work(n, p, incl, ell) for p, *_ in PRECS]
                print(f'{name[:12]:12s} {n:4d} {ell:9.6f} {n * ell:4g} | '
                      + ' '.join(f'{x:6.0f}' if x is not None else '     -' for x in w[:2])
                      + (f' {w[2]:8.0f}' if w[2] is not None else '        -'))
    ax_s, ax_l, ax_h = axes[row]
    for p, c, m, label, _ in PRECS:
        for lh, ls in ((2, '-'), (4, '--')):                               # fixed l/h: l = lh / N
            pts = [(n, work(n, p, incl, lh / n)) for n in (32, 64, 128)]
            pts = [q for q in pts if q[1] is not None]
            if pts:
                ax_s.plot(*np.array(pts).T, linestyle=ls, marker=m, ms=8, lw=2.2, color=c,
                          mfc=c if ls == '-' else 'white', label=f'{label}, l/h = {lh}')
        for ell, ls in ((0.0625, '-'), (0.03125, '--')):                   # fixed l
            pts = [(n, work(n, p, incl, ell)) for n in (32, 64, 128)]
            pts = [q for q in pts if q[1] is not None]
            if pts:
                ax_l.plot(*np.array(pts).T, linestyle=ls, marker=m, ms=8, lw=2.2, color=c,
                          mfc=c if ls == '-' else 'white', label=f'{label}, l = 1/{1 / ell:g}')
                for x, y in pts:
                    if ls == '-':
                        ax_l.annotate(f'l/h={x * ell:g}', (x, y), textcoords='offset points', xytext=(5, 5),
                                      fontsize=9, color=c)
        pts = [(128 * ell, work(128, p, incl, ell)) for ell in LENGTHS]      # fixed grid
        pts = sorted(q for q in pts if q[1] is not None)
        if pts:
            ax_h.plot(*np.array(pts).T, '-', marker=m, ms=8, lw=2.2, color=c, label=label)
    for ax, title, xl in ((ax_s, 'sharp crack: fixed l/h\n(the same number of elements across the crack)', 'grid'),
                          (ax_l, 'smooth crack: fixed l\n(more elements across the crack on finer grids)', 'grid'),
                          (ax_h, 'fixed grid 128$^2$\nsmoothness only', 'l/h (elements per length scale)')):
        ax.set_title(f'{name}: {title}', fontsize=12.5)
        ax.set_xscale('log', base=2)
        ticks = (32, 64, 128) if ax is not ax_h else (2, 4, 8, 16)
        ax.set_xticks(ticks), ax.set_xticklabels([f'{t}$^2$' if ax is not ax_h else f'{t}' for t in ticks])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_ylim(0, None), ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6, integer=True))
        ax.set_xlabel(xl, fontsize=12), ax.set_ylabel('work per elastic solve\n(K applications)', fontsize=12)
        ax.tick_params(labelsize=12), ax.grid(True, color='0.9')
        ax.legend(fontsize=9.5, loc='upper left')
    top = max(ax.get_ylim()[1] for ax in axes[row])
    for ax in axes[row]:
        ax.set_ylim(0, top)
out = os.path.join(HERE, 'figures', 'AT2_smoothness_2D' + ('' if ROWS[1][0] == '_holes0.0001'
                                                            else ROWS[1][0].replace('|', '')) + '.png')
fig.savefig(out)
print(f'wrote {out}')
