"""
Elastic CG iterations against the grid at fixed l/h (2 and 4): pores and stiff tough inclusions (10x), polydisperse
disks, Green / Green-Jacobi / dual. Runs: run_AT2_incl_lh_64.sh, run_AT2_incl_lh_128.sh, run_AT2_incl_lh_256.sh,
run_overnight_2D_512_cpu.sh (512^2: Green and dual only; the 64^2 pores l/h = 2 run is the 2026-10-19 one).
Only finished runs (cell cut through, or the full load path) enter.

Left: mean CG iterations per elastic solve (solid), the dual also in cost = 2 x iterations (dashed, light).
Right: total elastic CG iterations of the whole load path (dual in cost, dashed).

usage:  python plot_AT2_lh_iterations.py       writes figures/AT2_lh_iterations.png and prints the table
"""
import os

import numpy as np
import matplotlib
import matplotlib.ticker

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
GRIDS = (64, 128, 256, 512)
PRECONDS = [('Green', 'C2', 'o', 'Green'), ('Green_Jacobi', 'C0', 's', 'Green-Jacobi'), ('dual', 'C3', '^', 'dual')]


def tag(n, incl, lh):
    ell = lh / n
    t = ('_incl10' if incl else '') + f'_l{ell:g}'
    return t if n == 64 else t + '_lh'


def load(n, p, incl, lh):
    base = f'AT2_N{n}_{p}_polydisperse_pores{tag(n, incl, lh)}'
    log = os.path.join(HERE, 'logs', base + '.log')
    csv = os.path.join(HERE, 'exp_data', base + '_iterations.csv')
    if not (os.path.exists(log) and os.path.exists(csv)) or not any(s in open(log).read() for s in ('cut through', 'step  79 ')):
        return None
    return np.genfromtxt(csv, delimiter=',', names=True)


fig, axes = plt.subplots(2, 4, figsize=(19, 10), layout='constrained', dpi=150, sharex=True)
print(f'{"case":24s} {"N":>5s} | {"mean CG u: G":>13s} {"GJ":>5s} {"dual":>5s} {"(cost)":>7s} | {"total: G":>9s} {"GJ":>8s} '
      f'{"dual cost":>9s}')
handles = {}
for row, incl in enumerate((False, True)):
    for k, lh in enumerate((2, 4)):
        ax_m, ax_t = axes[row, 2 * k], axes[row, 2 * k + 1]
        name = f'{"inclusions 10x" if incl else "pores"}, l/h = {lh}'
        table = {n: {} for n in GRIDS}
        for p, c, m, label in PRECONDS:
            ns, mean, total = [], [], []
            for n in GRIDS:
                d = load(n, p, incl, lh)
                if d is None:
                    continue
                ns.append(n), mean.append(d['cg_u'].mean()), total.append(d['cg_u'].sum())
                table[n][p] = (d['cg_u'].mean(), d['cg_u'].sum())
            if not ns:
                continue
            ns, mean, total = np.array(ns), np.array(mean), np.array(total)
            handles[label] = ax_m.plot(ns, mean, '-', marker=m, ms=7, lw=2, color=c, label=label)[0]
            ax_t.plot(ns, total / 1e3, '-', marker=m, ms=7, lw=2, color=c)
            if p == 'dual':
                handles['dual, cost (2x)'] = ax_m.plot(ns, 2 * mean, '--', marker=m, ms=7, lw=2, mfc='none', color=c,
                                                       alpha=0.5, label='dual, cost (2x)')[0]
                ax_t.plot(ns, 2 * total / 1e3, '--', marker=m, ms=7, lw=2, mfc='none', color=c, alpha=0.5)
        for n in GRIDS:
            t = table[n]
            if not t:
                continue
            f = lambda p, i, s=1: f'{s * t[p][i]:.0f}' if p in t else '-'
            print(f'{name:24s} {n:5d} | {f("Green", 0):>13s} {f("Green_Jacobi", 0):>5s} {f("dual", 0):>5s} '
                  f'{"(" + f("dual", 0, 2) + ")":>7s} | {f("Green", 1):>9s} {f("Green_Jacobi", 1):>8s} {f("dual", 1, 2):>9s}')
        ax_m.set_title(f'{name}\nmean per elastic solve', fontsize=13)
        ax_t.set_title(f'{name}\ntotal over the load path', fontsize=13)
        ax_m.set_ylabel('CG iterations', fontsize=12)
        ax_t.set_ylabel('CG iterations [thousands]', fontsize=12)
        for ax in (ax_m, ax_t):
            ax.set_xscale('log', base=2)
            ax.set_xticks(GRIDS), ax.set_xticklabels([f'{n}$^2$' for n in GRIDS])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_ylim(0, None)
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6, integer=True))
            ax.tick_params(labelsize=12)
            ax.grid(True, color='0.9')
        if row == 1:
            ax_m.set_xlabel('grid', fontsize=12), ax_t.set_xlabel('grid', fontsize=12)
fig.legend(handles=list(handles.values()), loc='outside lower center', ncol=len(handles), fontsize=13)
out = os.path.join(HERE, 'figures', 'AT2_lh_iterations.png')
fig.savefig(out)
print(f'wrote {out}')
