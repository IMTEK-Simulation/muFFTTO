"""
Cahn-Hilliard two-phase microstructure at k_res = 1e-4 (see README.md, "Larger grids"): Green / Green-Jacobi / dual on
64^2 ... 512^2 (fixed l = 1/32: l/h = 2 / 4 / 8 / 16).

    top     left:   total elastic work (CG iterations x K applications, dual 2) against the grid, all runs cut at the
                    same load step: the last step every finished run reached (the runs stop on their own once the
                    cell is cut through, sigma < 0.2 peak for 3 steps, so the coarse grids run longer)
            middle: work per elastic solve against the grid (same cut)
            right:  macroscopic stress-strain curves of the dual runs
    bottom  mean CG iterations per elastic solve in every load step (raw iterations, dual costs 2 per iteration), one
            panel per grid, the same y limits in every panel; grey bars: staggered iterations of the load step (dual)

Unfinished runs (no 'Elapsed time' in the log) are drawn thin in the bottom row and left out of the top row.

usage:  python plot_AT2_cahn_hilliard_large.py      writes figures/AT2_cahn_hilliard_large.png and prints the table
        GC2=1 python plot_AT2_cahn_hilliard_large.py   the runs with a crackable phase 2 (PHASE2_GC=1.7407,
                                                     run_AT2_cahn_hilliard_gc2.sh) -> figures/AT2_cahn_hilliard_gc2.png
"""
import os

import numpy as np
import netCDF4
import matplotlib
import matplotlib.ticker

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
GRIDS = (64, 128, 256, 512)
GC2 = os.environ.get('GC2', '0') == '1'
TAG = '_incl0.1_gc1.7407_l0.03125_k0.0001_ch' if GC2 else '_incl0.1_l0.03125_k0.0001_ch'
PRECS = [('Green', 'C2', 'o', '--', 1, 'Green'), ('Green_Jacobi', 'C0', 's', ':', 1, 'Green-Jacobi'),
         ('dual', 'C3', '^', '-', 2, 'dual')]


def load(n, p):
    """(iterations, finished, (strain, stress)) or None"""
    base = os.path.join(HERE, 'exp_data', f'AT2_N{n}_{p}{TAG}')
    if not os.path.exists(base + '_iterations.csv'):
        return None
    log = os.path.join(HERE, 'logs', f'AT2_N{n}_{p}{TAG}.log')
    done = os.path.exists(log) and 'Elapsed time' in open(log).read()
    it = np.atleast_1d(np.genfromtxt(base + '_iterations.csv', delimiter=',', names=True, dtype=None, encoding=None))
    curve = None
    if done:
        with netCDF4.Dataset(base + '.nc') as ds:
            curve = np.asarray(ds['load'][:]), np.asarray(ds['macro_stress'][:, 1, 1])
    return it, done, curve


fig = plt.figure(figsize=(26, 12), layout='constrained', dpi=130)
gs = fig.add_gridspec(2, 12)
ax_w, ax_s, ax_c = (fig.add_subplot(gs[0, 4 * i:4 * i + 4]) for i in range(3))
ax_g = [fig.add_subplot(gs[1, 3 * i:3 * i + 3]) for i in range(len(GRIDS))]

print(f'{"N":>4s} {"l/h":>4s} | ' + ' '.join(f'{q[5]:>22s}' for q in PRECS) + '   (total work, work/solve)')
rows = {n: {} for n in GRIDS}
runs = [load(n, p) for n in GRIDS for p, *_ in PRECS]
last_step = min(r[0]['step'].max() for r in runs if r is not None and r[1])   # common load step of the finished runs
for p, c, m, ls, fac, label in PRECS:
    pts = []
    for n in GRIDS:
        r = load(n, p)
        if r is None:
            continue
        it, done, curve = r
        if done:
            cut = it['cg_u'][it['step'] <= last_step]
            pts.append((n, fac * cut.sum() / 1e3, fac * cut.mean()))
            rows[n][p] = pts[-1][1:]
    if pts:
        x, w, ws = np.array(pts).T
        ax_w.plot(x, w, ls='-', marker=m, ms=9, lw=2, color=c, label=label)
        ax_s.plot(x, ws, ls='-', marker=m, ms=9, lw=2, color=c, label=label)
        for xi, wi in zip(x, w):
            ax_w.annotate(f'{wi:.0f}', (xi, wi), textcoords='offset points', xytext=(7, 4), fontsize=11, color=c)
        for xi, wi in zip(x, ws):
            ax_s.annotate(f'{wi:.0f}', (xi, wi), textcoords='offset points', xytext=(7, 4), fontsize=11, color=c)
for n in GRIDS:
    print(f'{n:4d} {n / 32:4.0f} | ' + ' '.join(
        f'{rows[n][p][0]:12.0f}k {rows[n][p][1]:8.0f}' if p in rows[n] else f'{"-":>22s}' for p, *_ in PRECS))

shades = plt.cm.Reds(np.linspace(0.35, 0.95, len(GRIDS)))
for n, sh in zip(GRIDS, shades):
    r = load(n, 'dual')
    if r is not None and r[2] is not None:
        e, s = r[2]
        ax_c.plot(np.r_[0, e] * 1e3, np.r_[0, s] * 1e3, color=sh, lw=2, label=f'{n}$^2$ (l/h = {n / 32:g})')

e_cut = (last_step + 1) * 1e-2           # load increment 2e-3 / 200 = 1e-5 = 1e-2 [10^-3]
for ax, yl, title in ((ax_w, 'total elastic work [thousands]\n(dual 2 per iteration)',
                       f'cost up to the same load step {last_step + 1} ($\\bar\\varepsilon_{{yy}}$ = {e_cut:.2f}e-3)'),
                      (ax_s, 'work per elastic solve\n(dual 2 per iteration)', 'cost per elastic solve (same load steps)')):
    ax.set_xscale('log', base=2), ax.set_xticks(GRIDS), ax.set_xticklabels([f'{n}$^2$' for n in GRIDS])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(0, None), ax.set_xlabel('grid  (l = 1/32 fixed: l/h = 2, 4, 8, 16)', fontsize=13)
    ax.set_ylabel(yl, fontsize=13), ax.set_title(title, fontsize=14), ax.legend(fontsize=12)
ax_c.set_xlabel(r'$\bar\varepsilon_{yy}$ [10$^{-3}$]', fontsize=13)
ax_c.set_ylabel(r'$\bar\sigma_{yy}$ [10$^{-3}$ E]', fontsize=13)
ax_c.set_title('stress-strain (dual runs)', fontsize=14), ax_c.legend(fontsize=12)

twins = []
for ax, n in zip(ax_g, GRIDS):
    tw = ax.twinx()
    for p, c, m, ls, fac, label in PRECS:
        r = load(n, p)
        if r is None:
            continue
        it, done, _ = r
        steps = np.unique(it['step'])
        load_ = np.array([it['load'][it['step'] == s][0] for s in steps]) * 1e3
        ax.plot(load_, [it['cg_u'][it['step'] == s].mean() for s in steps], color=c, ls=ls, lw=2 if done else 1,
                label=label + ('' if done else ' (running)'))
        if p == 'dual':
            stag = np.array([(it['step'] == s).sum() for s in steps])
            tw.bar(load_, stag, width=0.8 * (load_[1] - load_[0]), color='0.8', zorder=0)
            twins.append((tw, stag.max()))
    ax.set_zorder(tw.get_zorder() + 1), ax.patch.set_visible(False)
    tw.set_ylabel('staggered iterations (grey bars)', fontsize=11, color='0.4'), tw.tick_params(labelsize=11, colors='0.4')
    ax.set_xlabel(r'$\bar\varepsilon_{yy}$ [10$^{-3}$]', fontsize=13)
    ax.set_ylabel('CG iterations per elastic solve\n(mean over the load step)', fontsize=12)
    ax.set_title(f'{n}$^2$ (l/h = {n / 32:g})', fontsize=14), ax.legend(fontsize=11, loc='upper left')
top = max(ax.get_ylim()[1] for ax in ax_g)
stag_top = 1.15 * max(v for _, v in twins)
for ax in ax_g:
    ax.set_ylim(0, top)
for ax in [ax_c] + ax_g:
    ax.axvline(e_cut, color='k', lw=1, ls='--', alpha=0.6)
for tw, _ in twins:
    tw.set_ylim(0, stag_top)
for ax in [ax_w, ax_s, ax_c] + ax_g:
    ax.tick_params(labelsize=12), ax.grid(True, color='0.9')
fig.suptitle('Cahn-Hilliard two-phase microstructure (phase contrast 10, k$_{res}$ = 1e-4, '
             + ('phase 2 cracks, Gc$_2$ = 1.74 Gc' if GC2 else 'phase 2 tough') + '): Green / Green-Jacobi / dual',
             fontsize=15)
out = os.path.join(HERE, 'figures', 'AT2_cahn_hilliard_gc2.png' if GC2 else 'AT2_cahn_hilliard_large.png')
fig.savefig(out)
print(f'all runs cut at load step {last_step} (strain {e_cut:.2f}e-3)')
print(f'wrote {out}')
