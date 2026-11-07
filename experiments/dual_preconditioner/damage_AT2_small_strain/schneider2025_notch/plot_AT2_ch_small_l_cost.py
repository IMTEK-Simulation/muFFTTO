"""
Cost of the CH small-l study (run_AT2_cahn_hilliard_small_l.sh + the l = 1/32 runs): AT2, 128^2, k_res 1e-4,
l/h = 4, 2, 1, 0.5, 0.25, until the cell is cut through.
(a) mean CG iterations per elasticity solve, (b) total elasticity CG work (dual x 2), (c) wall time.
    python plot_AT2_ch_small_l_cost.py   ->  figures/AT2_ch_small_l_cost.png
"""
import csv
import os
import re

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
N = 128
RUNS = [(0.03125, '0.03125'), (0.015625, '0.015625'), (0.0078125, '0.0078125'), (0.00390625, '0.00390625'),
        (0.001953125, '0.00195312')]
PRE = [('Green', 'Green', 'tab:green', 'o'), ('Green_Jacobi', 'Green-Jacobi', 'tab:blue', 's'),
       ('dual', 'dual', 'tab:red', '^')]
COST = {'Green': 1.0, 'Green_Jacobi': 1.0, 'dual': 2.0}
plt.rcParams.update({'font.size': 13})

lh = np.array([l * N for l, _ in RUNS])
res = {}
for P, *_ in PRE:
    for l, tag in RUNS:
        name = f'AT2_N{N}_{P}_incl0.1_l{tag}_k0.0001_ch'
        with open(os.path.join(HERE, 'exp_data', name + '_iterations.csv')) as f:
            cg = np.array([int(r['cg_u']) for r in csv.DictReader(f)])
        logn = os.path.join(HERE, 'logs', f'AT2_N{N}_{P}_incl0.1_l{l if tag != "0.00195312" else 0.001953125}_k0.0001_ch.log')
        if not os.path.exists(logn):
            logn = os.path.join(HERE, 'logs', name + '.log')
        m = re.findall(r'Elapsed time: ([\d.]+)', open(logn).read())
        res[(P, l)] = (cg.mean(), cg.sum(), float(m[-1]) if m else np.nan, cg.size)
        print(f'l/h {l * N:5g} {P:13s} solves {cg.size:6d}  CG/solve {cg.mean():6.1f}  total {cg.sum():8d}  '
              f'wall {res[(P, l)][2]:7.0f} s')

fig, axs = plt.subplots(1, 3, figsize=(18, 5.4), layout='constrained')
for P, lab, col, mk in PRE:
    mean = [res[(P, l)][0] for l, _ in RUNS]
    work = [res[(P, l)][1] * COST[P] / 1e3 for l, _ in RUNS]
    wall = [res[(P, l)][2] / 60 for l, _ in RUNS]
    axs[0].plot(lh, mean, mk + '-', color=col, lw=2.5, ms=9, label=lab)
    if P == 'dual':
        axs[0].plot(lh, 2 * np.array(mean), mk + '--', color=col, lw=1.8, ms=8, mfc='white', label='dual x 2 (cost)')
    axs[1].plot(lh, work, mk + '-', color=col, lw=2.5, ms=9, label=lab + (' (x 2)' if P == 'dual' else ''))
    axs[2].plot(lh, wall, mk + '-', color=col, lw=2.5, ms=9, label=lab)
titles = ['(a) mean CG per elasticity solve', '(b) total elasticity work to cut-through', '(c) wall time (laptop CPU)']
ylabels = ['CG iterations per solve', 'CG iterations x 1000 (dual x 2)', 'wall time [min]']
for a, t, y in zip(axs, titles, ylabels):
    a.set_xscale('log', base=2)
    a.set_xticks(lh), a.set_xticklabels([f'{v:g}' for v in lh]), a.minorticks_off()
    a.axvspan(0.2, 2, color='0.88', zorder=0)
    a.set_xlim(0.2, 5)
    a.set_ylim(0, None)
    a.set(xlabel='l / h  (128$^2$, grey: l/h < 2 under-resolved)', ylabel=y, title=t)
    a.grid(alpha=0.3)
    a.legend(fontsize=11, loc='upper left', framealpha=0.95)
fig.suptitle('CH two-phase AT2, 128$^2$, k$_{res}$ = 1e-4: shrinking l below the pixel', fontsize=15)
out = os.path.join(HERE, 'figures', 'AT2_ch_small_l_cost.png')
fig.savefig(out, dpi=120)
print('wrote', out)
