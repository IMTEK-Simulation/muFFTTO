"""
CH two-phase AT2 (k_res 1e-4): is the Green-vs-dual balance set by l/h or by the grid?  Fixed l/h = 0.5 and 4 on
64^2, 128^2, 256^2 (run_AT2_cahn_hilliard_fixed_lh.sh + the small-l / l = 1/32 runs). Mean CG per elasticity solve
over the whole run (to cut-through) and the cost ratio Green / (2 x dual).
    python plot_AT2_ch_fixed_lh.py   ->  figures/AT2_ch_fixed_lh.png
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
RUNS = {0.5: {64: '0.0078125', 128: '0.00390625', 256: '0.00195312'}, 4: {64: '0.0625', 128: '0.03125', 256: '0.015625'}}
plt.rcParams.update({'font.size': 14})


def mean_cg(N, P, tag):
    with open(os.path.join(HERE, 'exp_data', f'AT2_N{N}_{P}_incl0.1_l{tag}_k0.0001_ch_iterations.csv')) as f:
        return np.mean([int(r['cg_u']) for r in csv.DictReader(f)])


fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.4), layout='constrained')
grids = [64, 128, 256]
for lh, ls, mk in [(0.5, '--', 'o'), (4, '-', 's')]:
    g = [mean_cg(N, 'Green', RUNS[lh][N]) for N in grids]
    d = [mean_cg(N, 'dual', RUNS[lh][N]) for N in grids]
    a.plot(grids, g, mk + ls, color='tab:green', lw=2.5, ms=9, label=f'Green, l/h = {lh:g}')
    a.plot(grids, d, mk + ls, color='tab:red', lw=2.5, ms=9, label=f'dual, l/h = {lh:g}')
    r = np.array(g) / (2 * np.array(d))
    b.plot(grids, r, mk + ls, color='k', lw=2.5, ms=9, label=f'l/h = {lh:g}')
    for x, y in zip(grids, r):
        b.annotate(f'{y:.2f}', (x, y), textcoords='offset points', xytext=(6, 6), fontsize=12)
    print(lh, np.round(g, 1), np.round(d, 1), np.round(r, 2))
for ax in (a, b):
    ax.set_xscale('log', base=2), ax.set_xticks(grids), ax.set_xticklabels([f'{n}$^2$' for n in grids]), ax.minorticks_off()
    ax.set_xlabel('grid (l/h fixed)'), ax.grid(alpha=0.3), ax.set_ylim(0, None)
a.set(ylabel='CG iterations per elasticity solve', title='(a) mean CG per solve (whole run)')
b.axhline(1, color='0.4', lw=1.5)
b.set(ylabel='Green / (2 x dual)', title='(b) cost ratio: > 1 dual cheaper')
b.set_ylim(0, 2.6), b.set_xlim(54, 330)
a.legend(fontsize=12, loc='upper left'), b.legend(fontsize=12, loc='center right')
fig.suptitle('CH two-phase AT2, k$_{res}$ = 1e-4: at fixed l/h the balance does not depend on the grid', fontsize=15)
fig.savefig(os.path.join(HERE, 'figures', 'AT2_ch_fixed_lh.png'), dpi=120)
