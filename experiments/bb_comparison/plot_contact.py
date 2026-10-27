"""
Contact benchmarks (32^2), Newton-CG / trust-region Newton-CG vs Barzilai-Borwein: total operator applications
(hessp + fev) over the load path, one panel per problem, one bar per solver and preconditioner (hue).
Hatched bar = run stopped before lambda = 1 (its work is a lower bound); missing bar = no converged increment.

usage:  ./summarize_contact.sh (writes contact_table.csv);  python plot_contact.py   -> contact_work.png
"""
import csv
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
PRECONDS = {'Green': ('Green', '#3d7fd0'), 'Green_Jacobi': ('Green-Jacobi', '#c4441a'), 'dual': ('dual', '#4d9221')}
SOLVERS = ['tr_newton', 'newton_cg', 'bb1', 'bb2', 'bb1 (gtol 3e-7)', 'bb1energy']
LABELS = {'tr_newton': 'TR Newton-CG', 'newton_cg': 'Newton-CG', 'bb1': 'BB1', 'bb2': 'BB2',
          'bb1 (gtol 3e-7)': 'BB1, gtol/10', 'bb1energy': 'BB1 + GLL'}
PROBLEMS = ['patch', 'punch', 'hertz', 'disc']
rows = list(csv.DictReader(open(os.path.join(HERE, 'contact_table.csv'))))
fig, axes = plt.subplots(1, 4, figsize=(17, 5.2), layout='constrained', sharey=True)
for ax, prob in zip(axes, PROBLEMS):
    for r in rows:
        if not r['run'].startswith(prob + '_') or r['work'] == '' or r['solver'] not in SOLVERS:
            continue
        k = list(PRECONDS).index(r['preconditioner'])
        y = SOLVERS.index(r['solver']) + (k - 1) * 0.27
        colour = PRECONDS[r['preconditioner']][1]
        stopped = 'stopped' in r['increments']
        ax.barh(y, int(r['work']), height=0.25, color='white' if stopped else colour, edgecolor=colour,
                hatch='///' if stopped else None, lw=1)
        ax.text(int(r['work']) * 1.08, y, f'{int(r["work"]):,}'.replace(',', ' ') + (' (stopped)' if stopped else ''),
                va='center', fontsize=7, color='0.25')
    for s in ('bb1', 'bb2', 'bb1 (gtol 3e-7)', 'bb1energy'):
        ax.text(6e2, SOLVERS.index(s) + 0.27, 'dual: no convergence', va='center', fontsize=7, color='0.35') \
            if prob == 'patch' else None
    ax.set(xscale='log', xlim=(5e2, 3e6), title=prob, xlabel='operator applications (hessp + fev)')
    ax.grid(True, axis='x', color='0.9', which='both')
    ax.set_axisbelow(True)
axes[0].set_yticks(range(len(SOLVERS)), [LABELS[s] for s in SOLVERS])
axes[0].invert_yaxis()
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=c, label=n) for n, c in PRECONDS.values()] +
           [Patch(facecolor='white', edgecolor='0.4', hatch='///', label='stopped before $\\lambda=1$')],
           loc='outside lower center', ncol=4, frameon=False)
fig.suptitle('Third-medium contact, 32$^2$: work over the whole load path (dual + BB: no convergence on any '
             'problem, shown for the patch test only)', fontsize=12)
fig.savefig(os.path.join(HERE, 'contact_work.png'), dpi=115)
print('wrote contact_work.png')
