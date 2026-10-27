"""
Talk figure: phase-field fracture (small strain), Green / Green-Jacobi / dual compared by COST of the elastic solves.

Cost of one PCG iteration in units of (one Green application + one K application): Green 1, Green-Jacobi 1, dual 2
(as talk_cost_plot.py; measured wall-time ratio dual / Green 1.5-2.9 on the paper-1 runs, 1.97 on the house).

    (a) AT2, periodic cell with polydisperse pores (damage_AT2_small_strain): mean cost per elastic solve vs grid
    (b) the same run: total elastic cost of the whole load path vs grid
    (c) AT2 cantilever clamped at one end (Dirichlet; corrected Green / corrected dual, hybrid split,
        ../linear_elastic_mesh_study/exp_data/fracture): mean cost per elastic solve vs n_H
    (d) cracking 3D house clamped at the foundation (Dirichlet), elastic solve of five load steps on frozen damage
        (../linear_elastic_mesh_study/exp_data/house/precond_h0.125.npz): cost per solve

usage:  python talk_fracture_cost_plot.py      -> figures/talk_fracture_cost.png, .pdf
"""
import os
import re

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = os.path.dirname(os.path.realpath(__file__))
AT2 = os.path.join(HERE, 'damage_AT2_small_strain')
MESH = os.path.join(HERE, '..', 'linear_elastic_mesh_study', 'exp_data')
TAG = '_polydisperse_pores_l0.03125'
plt.rcParams.update({'font.size': 13})
COL = {'green': '#2a78d6', 'gj': '#eb6834', 'dual': '#1baf7a'}
MARK = {'green': 'o', 'gj': 's', 'dual': '^'}
FACTOR = {'green': 1, 'gj': 1, 'dual': 2}
STEP = re.compile(r'^step +(\d+) ', re.M)


def at2_solves(N, p):
    """cg_u of every elastic solve of the finished load steps; a restarted run joined at its first restarted step"""
    def part(suffix):
        csv = os.path.join(AT2, 'exp_data', f'AT2_N{N}_{p}{TAG}{suffix}_iterations.csv')
        log = os.path.join(AT2, 'logs', f'AT2_N{N}_{p}{TAG}{suffix}.log')
        if not (os.path.exists(csv) and os.path.exists(log)):
            return None
        it = np.genfromtxt(csv, delimiter=',', names=True, ndmin=1)
        steps = [int(k) for k in STEP.findall(open(log).read())]
        return it[it['step'] <= steps[-1]] if steps else it[:0]
    it, it_r = part(''), part('_restart')
    if it is None:
        return None
    if it_r is not None and len(it_r):
        it = np.concatenate([it[it['step'] < it_r['step'][0]], it_r])
    return it['cg_u']


def line(ax, pts, key, label, dy=0):
    pts = [(x, y) for x, y in pts if y is not None]
    if not pts:
        return
    ax.plot(*zip(*pts), marker=MARK[key], color=COL[key], lw=2.2, ms=7)
    ax.annotate(f'{label}: {pts[-1][1]:,.0f}', pts[-1], textcoords='offset points', xytext=(9, dy - 4), fontsize=11.5,
                color=COL[key])


def logx(ax, ticks, labels, xlim):
    ax.set_xscale('log')
    ax.set_xticks(ticks, labels)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(*xlim)


fig, axes = plt.subplots(1, 4, figsize=(24, 6.3), layout='constrained')
GRIDS = [32, 64, 128, 256, 512, 1024]
for key, p, label in (('green', 'Green', 'Green'), ('gj', 'Green_Jacobi', 'Green-Jacobi'), ('dual', 'dual', 'dual')):
    runs = {N: at2_solves(N, p) for N in GRIDS}
    line(axes[0], [(N, FACTOR[key] * r.mean() if r is not None else None) for N, r in runs.items()], key, label,
         dy=8 if key == 'green' else 0)
    line(axes[1], [(N, FACTOR[key] * r.sum() if r is not None else None) for N, r in runs.items()], key, label,
         dy=8 if key == 'green' else -8 if key == 'gj' else 0)
axes[0].set(ylabel='cost per elastic solve', ylim=(0, None),
            title='(a) AT2, periodic cell with pores\nmean cost per elastic solve')
axes[1].set(ylabel='total elastic cost of the load path', yscale='log',
            title='(b) AT2, periodic cell with pores\ntotal cost of the elastic solves')
for ax in axes[:2]:
    logx(ax, GRIDS, [f'{N}$^2$' for N in GRIDS], (25, 4000))
    ax.set_xlabel('grid')

ax = axes[2]
NH = [32, 64, 128]
for key, p, label in (('green', 'green_c', 'Green, corrected'), ('dual', 'dual_c', 'dual, corrected')):
    pts = []
    for n in NH:
        f = os.path.join(MESH, 'fracture', f'nH{n}_{p}_holes4_het10_l0.0125_ft0.01_iterations.csv')
        pts.append((n, FACTOR[key] * np.genfromtxt(f, delimiter=',', names=True)['cg_u'].mean()
                    if os.path.exists(f) else None))
    line(ax, pts, key, label, dy=8 if key == 'green' else -10)
logx(ax, NH, [str(n) for n in NH], (26, 400))
ax.set(xlabel='$n_H$ (pixels across the beam height)', ylabel='cost per elastic solve', ylim=(0, None),
       title='(c) AT2 cantilever, clamped end (Dirichlet)\nmean cost per elastic solve')

ax = axes[3]
house = np.load(os.path.join(MESH, 'house', 'precond_h0.125.npz'))
labels = ['intact', 'damage\nstarts', 'peak\nwind', 'first\ncrack', 'last\nstep']
x = np.arange(5)
for i, (key, name, label) in enumerate((('green', 'green_c', 'Green, corrected'),
                                        ('gj', 'gj_c', 'Green-Jacobi, corrected'),
                                        ('dual', 'dual_eps_c', 'dual, corrected'))):
    cost = FACTOR[key] * house[name][:, 4]
    ax.bar(x + (i - 1) * 0.27, cost, 0.25, color=COL[key], label=label)
    for xi, c in zip(x, cost):
        ax.annotate(f'{c:.0f}', (xi + (i - 1) * 0.27, c), textcoords='offset points', xytext=(0, 3), ha='center',
                    fontsize=9.5, color=COL[key])
ax.set_xticks(x, labels)
ax.set(ylabel='cost per elastic solve', title='(d) cracking 3D house, clamped foundation\n'
                                               '(Dirichlet), frozen damage')
ax.legend(frameon=False, fontsize=11, loc='upper left')
for ax in axes:
    ax.grid(True, color='0.92', which='both', axis='y' if ax is axes[3] else 'both')
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Phase-field fracture (AT2, small strain): cost of the elastic solves = PCG iterations x applications '
             'per iteration (Green, Green-Jacobi: 1; dual: 2)', fontsize=14)
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(HERE, 'figures', f'talk_fracture_cost.{ext}'), dpi=110)
print('wrote figures/talk_fracture_cost.png')
