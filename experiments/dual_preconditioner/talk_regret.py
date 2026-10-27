"""
Talk figure: robustness of Green / Green-Jacobi / dual over every case we have (no contact): for each case the cost per
solve of each preconditioner divided by the cheapest one in that case (1 = the best; cost counted as in
talk_cost_plot.py, dual x 2). One dot per case; the bar = the worst case.

usage:  python talk_regret.py      -> figures/talk_regret.png, .pdf
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
sys.argv = sys.argv[:1]
import talk_cost_plot as L  # noqa: E402  (its loaders; it also redraws talk_cost.png)
import talk_fracture_cost_plot as F  # noqa: E402

plt.close('all')
cases = []  # (group, {prec: cost per solve})
for problem in ('P1', 'P2', 'P3'):
    for N in L.GRIDS:
        cases.append(('linear, grid', {p: L.p1(problem, N, -3, p) for p, *_ in L.PRECS}))
for problem in ('P1', 'P2'):
    for k in L.KAPPAS:
        cases.append(('linear, contrast', {p: L.p1(problem, 256, k, p) for p, *_ in L.PRECS}))
for e in (5, 3):
    for N in (16, 32, 64, 128, 256):
        cases.append(('topology opt.', {p: L.to_mean(N, e, p) if N <= 64 or p == 'dual' or e == 5 else
                                        L.to_mean(N, e, p, fromdual=True) for p, *_ in L.PRECS}))
for N in (32, 64, 128, 256):
    r = {k: F.at2_solves(N, p) for k, p in (('green', 'Green'), ('green_jacobi', 'Green_Jacobi'), ('dual', 'dual'))}
    cases.append(('fracture', {k: v.mean() if v is not None else None for k, v in r.items()}))
house = np.load(os.path.join(F.MESH, 'house', 'precond_h0.125.npz'))
for i in range(5):
    cases.append(('fracture', {'green': house['green_c'][i, 4], 'green_jacobi': house['gj_c'][i, 4],
                               'dual': house['dual_eps_c'][i, 4]}))
factor = {p: f for p, *_, f in L.PRECS}
ratios = {p: [] for p, *_ in L.PRECS}
groups = []
for g, c in cases:
    if any(c[p] is None for p in factor):
        continue
    cost = {p: c[p] * factor[p] for p in factor}
    best = min(cost.values())
    for p in factor:
        ratios[p].append(cost[p] / best)
    groups.append(g)
GROUPS = ['linear, grid', 'linear, contrast', 'topology opt.', 'fracture']
MARK = {'linear, grid': 'o', 'linear, contrast': 'D', 'topology opt.': 's', 'fracture': '^'}
plt.rcParams.update({'font.size': 14})
fig, ax = plt.subplots(figsize=(11, 6.5), layout='constrained')
rng = np.random.default_rng(0)
for i, (p, label, colour, *_r) in enumerate(L.PRECS):
    r = np.array(ratios[p])
    ax.bar(i, r.max(), 0.6, color=colour, alpha=0.18)
    for g in GROUPS:
        sel = np.array(groups) == g
        ax.scatter(i + rng.uniform(-0.22, 0.22, sel.sum()), r[sel], marker=MARK[g], s=46, color=colour,
                   edgecolor='white', lw=0.6, label=g if i == 0 else None, zorder=3)
    ax.annotate(f'worst {r.max():.1f}x\nmedian {np.median(r):.1f}x', (i, r.max()), textcoords='offset points',
                xytext=(0, 8), ha='center', fontsize=13, color=colour)
    print(f'{label:14s} worst {r.max():6.1f}  median {np.median(r):4.2f}  best in {np.sum(r == 1)}/{len(r)} cases')
ax.set_yscale('log')
ax.set_ylim(0.8, 200)
ax.set_xticks(range(3), [label for _, label, *_ in L.PRECS], fontsize=15)
ax.set_ylabel('cost / cost of the best preconditioner in that case')
ax.axhline(1, color='0.5', lw=1)
leg = ax.legend(frameon=False, fontsize=11.5, loc='upper left', title=f'{len(groups)} cases (marker = problem class)')
for h in leg.legend_handles:
    h.set_color('0.35')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(True, axis='y', color='0.92', which='both')
ax.set_title('How far from the best? (cost = iterations x applications, dual counted 2x)')
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(HERE, 'figures', f'talk_regret.{ext}'), dpi=110)
