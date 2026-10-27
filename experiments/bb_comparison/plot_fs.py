"""
Finite-strain P1-P3, Newton-CG vs Barzilai-Borwein (finite_strain_P123.py --solver ...): work and wall time per case,
and the deviation of the mean stress history from the Newton run.

    Newton work  = CG iterations + Newton steps (each CG iteration: one tangent product + one preconditioner
                   application; each Newton step: one stress + tangent evaluation)
    BB work      = BB gradient evaluations (each: one stress evaluation + one preconditioner application)

usage:  python plot_fs.py      -> fs_table.md, fs_newton_vs_bb.png
"""
import glob
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
PRECONDS = {'Green': ('Green', '#3d7fd0'), 'Green_Jacobi': ('Green-Jacobi', '#c4441a'), 'dual': ('dual', '#4d9221')}
runs = {}
for f in glob.glob(os.path.join(HERE, 'bb_data', '*.npz')):
    d = np.load(f, allow_pickle=True)
    key = (str(d['geometry']), float(d['kappa']), int(d['N']), str(d['preconditioner']))
    solver = str(d['solver'])
    done = int(d['completed_increments']) == int(d['ninc'])
    work = (int(d['cg_per_increment'].sum()) + int(d['newton_per_increment'].sum())) if solver == 'newton' \
        else int(d['bb_fev'].sum())
    runs.setdefault(key, {})[solver] = dict(work=work, time=float(d['elapsed_time']), done=done,
                                            P=np.asarray(d['mean_P']), failed=str(d['loading_failed']))
rows = []
for key in sorted(runs):
    r = runs[key]
    ref = r.get('newton')
    line = list(key)
    for s in ('newton', 'bb1', 'bb2'):
        if s not in r:
            line += ['-', '-', '-']
            continue
        x = r[s]
        dP = '-'
        if ref is not None and x['done'] and ref['done']:
            dP = f'{np.max(np.abs(x["P"] - ref["P"])) / np.max(np.abs(ref["P"])):.1e}'
        line += [str(x['work']) + ('' if x['done'] else ' (failed)'), f'{x["time"]:.1f}', dP]
    rows.append(line)
with open(os.path.join(HERE, 'fs_table.md'), 'w') as out:
    out.write('| geometry | kappa | N | preconditioner | Newton work | Newton time [s] | - | BB1 work | BB1 time [s] | '
              'BB1 dP | BB2 work | BB2 time [s] | BB2 dP |\n|' + '---|' * 13 + '\n')
    for row in rows:
        out.write('| ' + ' | '.join(str(v) if not isinstance(v, float) else f'{v:g}' for v in row) + ' |\n')

fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout='constrained')
for ax, quantity, label in ((axes[0], 'work', 'operator applications'), (axes[1], 'time', 'wall time [s]')):
    for key, r in runs.items():
        if 'newton' not in r or not r['newton']['done']:
            continue
        colour = PRECONDS[key[3]][1]
        for s, marker in (('bb1', 's'), ('bb2', '^')):
            if s in r and r[s]['done']:
                ax.plot(r['newton'][quantity], r[s][quantity], marker, color=colour,
                        mfc=colour if key[2] == 64 else 'white', ms=7, alpha=0.85)
    lim = ax.get_xlim()[0], ax.get_xlim()[1]
    lo, hi = min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], color='k', lw=1)
    ax.set(xscale='log', yscale='log', xlabel=f'Newton-CG: {label}', ylabel=f'BB: {label}',
           title=f'{label}: below the diagonal = BB cheaper')
    ax.grid(True, color='0.9', which='both')
from matplotlib.lines import Line2D
handles = [Line2D([], [], color=c, marker='o', ls='', label=n) for n, c in PRECONDS.values()] + \
          [Line2D([], [], color='0.4', marker='s', ls='', label='BB1'), Line2D([], [], color='0.4', marker='^', ls='',
                                                                                 label='BB2'),
           Line2D([], [], color='0.4', marker='o', mfc='white', ls='', label='32$^2$ (open), 64$^2$ (filled)')]
axes[0].legend(handles=handles, frameon=False, fontsize=9)
fig.suptitle('Finite-strain P1-P3 (neo-Hookean, contrast $10^{\\pm 3}$): Newton-CG vs Barzilai-Borwein per case',
             fontsize=13)
fig.savefig(os.path.join(HERE, 'fs_newton_vs_bb.png'), dpi=115)
print('wrote fs_table.md, fs_newton_vs_bb.png')
