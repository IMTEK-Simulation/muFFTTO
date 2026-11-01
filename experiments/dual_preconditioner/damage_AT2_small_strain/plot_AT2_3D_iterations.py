"""
3D AT2 at practical l/h = 2 (polydisperse balls, trilinear hexahedra, GPU): elastic CG iterations of Green /
Green-Jacobi / dual. Runs: run_overnight_3D_gpu.sh, run_overnight_3D_96_gpu.sh (96^3: dual only; the 32^3 pores point is
the existing AT2_N32_*_polydisperse_pores_l0.0625_3D run). Only finished runs (cut through, or the full load path) enter.

Left, middle: mean CG iterations per elastic solve against the grid, inclusions 10x and pores (dual cost dashed).
Right: 64^3 inclusions, mean CG iterations per elastic solve in each load step against the load.

usage:  python plot_AT2_3D_iterations.py       writes figures/AT2_3D_iterations.png and prints the table
"""
import os

import numpy as np
import matplotlib
import matplotlib.ticker

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
GRIDS = (32, 64, 96)
PRECONDS = [('Green', 'C2', 'o', 'Green'), ('Green_Jacobi', 'C0', 's', 'Green-Jacobi'), ('dual', 'C3', '^', 'dual')]


def load(n, p, incl):
    base = f'AT2_N{n}_{p}_polydisperse_pores{"_incl10" if incl else ""}_l{2 / n:g}_3D'
    log = os.path.join(HERE, 'logs', base + '.log')
    csv = os.path.join(HERE, 'exp_data', base + '_iterations.csv')
    if not os.path.exists(csv):
        return None
    if os.path.exists(log) and not any(s in open(log).read() for s in ('cut through', 'step  79 ')):
        return None
    return np.genfromtxt(csv, delimiter=',', names=True)


fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), layout='constrained', dpi=150)
handles = {}
print(f'{"case":16s} {"N":>4s} | {"mean CG u: G":>13s} {"GJ":>5s} {"dual":>5s} {"(cost)":>7s} | '
      f'{"elastic time [s]: G":>19s} {"GJ":>6s} {"dual":>6s}')
for ax, incl in zip(axes[:2], (True, False)):
    name = 'inclusions 10x' if incl else 'pores'
    table = {n: {} for n in GRIDS}
    for p, c, m, label in PRECONDS:
        ns, mean = [], []
        for n in GRIDS:
            d = load(n, p, incl)
            if d is None:
                continue
            ns.append(n), mean.append(d['cg_u'].mean())
            table[n][p] = (d['cg_u'].mean(), d['time_u'].sum())
        if not ns:
            continue
        handles[label] = ax.plot(ns, mean, '-', marker=m, ms=7, lw=2, color=c, label=label)[0]
        if p == 'dual':
            handles['dual, cost (2x)'] = ax.plot(ns, 2 * np.array(mean), '--', marker=m, ms=7, lw=2, mfc='none',
                                                 color=c, alpha=0.5, label='dual, cost (2x)')[0]
    for n in GRIDS:
        t = table[n]
        if t:
            f = lambda p, i, s=1: f'{s * t[p][i]:.0f}' if p in t else '-'
            print(f'{name:16s} {n:4d} | {f("Green", 0):>13s} {f("Green_Jacobi", 0):>5s} {f("dual", 0):>5s} '
                  f'{"(" + f("dual", 0, 2) + ")":>7s} | {f("Green", 1):>19s} {f("Green_Jacobi", 1):>6s} {f("dual", 1):>6s}')
    ax.set_title(f'3D {name}, l/h = 2\nmean CG iterations per elastic solve', fontsize=13)
    ax.set_xticks(GRIDS), ax.set_xticklabels([f'{n}$^3$' for n in GRIDS])
    ax.set_xlim(24, 104), ax.set_xlabel('grid', fontsize=12), ax.set_ylabel('CG iterations', fontsize=12)

ax = axes[2]
for p, c, m, label in PRECONDS:
    d = load(64, p, True)
    if d is None:
        continue
    loads = np.unique(d['load'])
    mean = np.array([d['cg_u'][d['load'] == e].mean() for e in loads])
    ax.plot(loads, mean, '-', marker=m, ms=4, lw=1.5, color=c)
    if p == 'dual':
        ax.plot(loads, 2 * mean, '--', marker=m, ms=4, lw=1.5, mfc='none', color=c, alpha=0.5)
ax.set_title('3D inclusions 10x, 64$^3$, l/h = 2\nmean CG iterations per elastic solve in each load step', fontsize=13)
ax.set_xlabel(r'load $\bar\varepsilon_{xx}$', fontsize=12), ax.set_ylabel('CG iterations', fontsize=12)
for ax in axes:
    ax.set_ylim(0, None)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6, integer=True))
    ax.tick_params(labelsize=12)
    ax.grid(True, color='0.9')
fig.legend(handles=list(handles.values()), loc='outside lower center', ncol=len(handles), fontsize=13)
out = os.path.join(HERE, 'figures', 'AT2_3D_iterations.png')
fig.savefig(out)
print(f'wrote {out}')
