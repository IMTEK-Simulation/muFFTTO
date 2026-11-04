"""
Gypsum RVE: mean elastic CG iterations per solve in every load step (raw iterations; the dual costs 2 K applications per
iteration) for Green / Green-Jacobi / dual, with the number of staggered iterations of the load step (dual run) as grey bars and
the macroscopic stress as a thin grey line. Rows: porous gypsum, stiff grains; columns: the finished grids (64^2 ...
1024^2); the same y limits in every column of a row.

usage:  python plot_AT2_gypsum_per_step.py      writes figures/AT2_gypsum_per_step.png
"""
import os

import numpy as np
import netCDF4
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
GRIDS = tuple(n for n in (64, 128, 256, 512, 1024)
              if os.path.exists(os.path.join(HERE, 'logs', f'AT2_N{n}_dual_polydisperse_pores_incl10_l0.0195915_gypsum.log'))
              and 'Elapsed' in open(os.path.join(HERE, 'logs',
                                                 f'AT2_N{n}_dual_polydisperse_pores_incl10_l0.0195915_gypsum.log')).read())
PRECS = [('Green', 'C2', '--', 'Green'), ('Green_Jacobi', 'C0', ':', 'Green-Jacobi'), ('dual', 'C3', '-', 'dual')]
VARIANTS = [('_holes0.0001', 'porous gypsum'), ('_incl10', 'stiff grains (10x)')]

fig, axes = plt.subplots(2, len(GRIDS), figsize=(6.7 * len(GRIDS), 10), layout='constrained', dpi=140, squeeze=False)
twins = {}
for row, (tag, name) in enumerate(VARIANTS):
    for col, n in enumerate(GRIDS):
        ax = axes[row, col]
        base = os.path.join(HERE, 'exp_data', f'AT2_N{n}_{{}}_polydisperse_pores{tag}_l0.0195915_gypsum')
        tw = ax.twinx()
        sl = None
        for k, (p, c, ls, label) in enumerate(PRECS):
            f = base.format(p) + '_iterations.csv'
            if not os.path.exists(f):
                continue
            d = np.genfromtxt(f, delimiter=',', names=True, dtype=None, encoding=None)
            steps = np.unique(d['step'])
            load = np.array([d['load'][d['step'] == s][0] for s in steps]) * 1e3
            ax.plot(load, [d['cg_u'][d['step'] == s].mean() for s in steps], color=c, ls=ls, lw=2, label=label)
            if p == 'dual':                   # bars and stress line of the dual run (finished first)
                stag = np.array([(d['step'] == s).sum() for s in steps])
                tw.bar(load, stag, width=0.8 * (load[1] - load[0]), color='0.8', zorder=0)
                for x, y in zip(load, stag):
                    if y > 20:
                        tw.text(x, y, f'{y}', fontsize=8, color='0.4', ha='center', va='bottom')
                twins[row, col] = (tw, stag.max())
                with netCDF4.Dataset(base.format(p) + '.nc') as ds:
                    sl, ss = np.asarray(ds['load'][:]) * 1e3, np.asarray(ds['macro_stress'][:, 0, 0])
        ax.set_zorder(tw.get_zorder() + 1), ax.patch.set_visible(False)
        tw.set_ylabel('staggered iterations of the load step (grey bars)', fontsize=10, color='0.4')
        tw.tick_params(labelsize=10, colors='0.4')
        ts = ax.twinx()
        if sl is not None:
            ts.plot(sl, ss / ss.max(), color='k', lw=1, alpha=0.35)
        else:
            ax.text(0.5, 0.5, 'not yet run', transform=ax.transAxes, ha='center', fontsize=14, color='0.5')
        ts.set_ylim(0, 1.05), ts.set_yticks([])
        ax.set_xlabel(r'$\bar\varepsilon_{xx}$ [10$^{-3}$]', fontsize=11)
        ax.set_ylabel('CG iterations per elastic solve\n(mean over the load step)', fontsize=11)
        ax.set_title(f'{name}, {n}$^2$ (l/h = {0.5 / (25.6 / n):g})   thin line: stress (scaled)', fontsize=12)
        ax.tick_params(labelsize=10), ax.grid(True, color='0.92')
        if row == 0 and col == 0:
            ax.legend(fontsize=11, loc='upper left')
for row in range(2):                       # the same y limits in every column of a row
    top = max(ax.get_ylim()[1] for ax in axes[row])
    for ax in axes[row]:
        ax.set_ylim(0, top)
    stag_top = 1.15 * max(max(v[1] for k, v in twins.items() if k[0] == row), 10)
    for (r, c), (tw, _) in twins.items():
        if r == row:
            tw.set_ylim(0, stag_top)
out = os.path.join(HERE, 'figures', 'AT2_gypsum_per_step.png')
fig.savefig(out)
print(f'wrote {out}')
