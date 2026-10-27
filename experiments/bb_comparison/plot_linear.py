"""
Linear CG vs Barzilai-Borwein (linear_P123.py results): iterations to reach the relative energy error e_k / e_0 <= 1e-6
(filled) and 1e-10 (open) vs the phase contrast, one panel per problem, colours per preconditioner (hue) and solver
(line style / marker). Also a markdown table of the ratio BB / CG.

usage:  python plot_linear.py      -> linear_cg_vs_bb.png, linear_table.md
"""
import glob
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
PRECONDS = {'green': ('Green', '#3d7fd0'), 'green_jacobi': ('Green-Jacobi', '#c4441a'), 'dual': ('dual', '#4d9221')}
SOLVERS = {'cg': dict(ls='-', marker='o', label='CG'), 'bb1': dict(ls='--', marker='s', label='BB1'),
           'bb2': dict(ls=':', marker='^', label='BB2')}


def first(h, level, e0):
    k = np.nonzero(h <= level * e0)[0]
    return int(k[0]) if len(k) else np.nan


data = {}
for f in glob.glob(os.path.join(HERE, 'linear_data', '*.npz')):
    d = np.load(f, allow_pickle=True)
    key = (str(d['problem']), float(d['kappa']), str(d['preconditioner']))
    data[key] = {s: (first(d[f'error_{s}'], 1e-6, d['e0']), first(d[f'error_{s}'], 1e-10, d['e0']),
                     float(d[f'time_{s}'])) for s in SOLVERS if f'error_{s}' in d}
problems = sorted({k[0] for k in data})
fig, axes = plt.subplots(1, len(problems), figsize=(6.5 * len(problems), 5.8), layout='constrained', squeeze=False)
for ax, prob in zip(axes[0], problems):
    for pc, (pname, colour) in PRECONDS.items():
        kappas = sorted(k[1] for k in data if k[0] == prob and k[2] == pc)
        for s, st in SOLVERS.items():
            for level, fill in ((0, colour), (1, 'white')):
                its = [data[(prob, kp, pc)].get(s, (np.nan,) * 3)[level] for kp in kappas]
                ax.plot(kappas, its, ls=st['ls'], marker=st['marker'], color=colour, mfc=fill, ms=6, lw=1.4,
                        label=f'{pname}, {st["label"]}' if level == 0 else None)
    ax.set(xscale='log', yscale='log', xlabel='phase contrast $\\kappa$', ylabel='iterations',
           title=f'{prob}, 128$^2$: filled $e_k/e_0 \\leq 10^{{-6}}$, open $\\leq 10^{{-10}}$')
    ax.grid(True, color='0.9', which='both')
axes[0][0].legend(frameon=False, fontsize=8, ncol=3)
fig.suptitle('Linear elasticity P1-P3: CG vs Barzilai-Borwein (one iteration = one K product + one preconditioner '
             'application for both)', fontsize=13)
fig.savefig(os.path.join(HERE, 'linear_cg_vs_bb.png'), dpi=115)
with open(os.path.join(HERE, 'linear_table.md'), 'w') as out:
    out.write('| problem | kappa | preconditioner | CG 1e-6 | BB1 1e-6 | BB2 1e-6 | CG 1e-10 | BB1 1e-10 | BB2 1e-10 | '
              'BB1/CG (1e-10) |\n|' + '---|' * 10 + '\n')
    for (prob, kp, pc) in sorted(data):
        r = data[(prob, kp, pc)]
        g = lambda s, i: r.get(s, (np.nan,) * 3)[i]
        ratio = g('bb1', 1) / g('cg', 1) if g('cg', 1) else np.nan
        out.write(f'| {prob} | {kp:g} | {PRECONDS[pc][0]} | {g("cg", 0)} | {g("bb1", 0)} | {g("bb2", 0)} | '
                  f'{g("cg", 1)} | {g("bb1", 1)} | {g("bb2", 1)} | {ratio:.2f} |\n')
print('wrote linear_cg_vs_bb.png, linear_table.md')
