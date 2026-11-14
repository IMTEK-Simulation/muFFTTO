"""Slide figure: 3D phase-field TO, mean CG iterations per elastic (mechanical) solve against the grid, eta = 4h (+ 2h dashed),
contrast 1e4, w = 0.2, a = 1 (NEMO logs, all segments of a run pooled). Green / Green-Jacobi / dual, the dual counted
x2 (one dual iteration ~ two Green iterations). Filled markers: runs from the sine start; open markers: 192^3 and
256^3 started from the converged 128^3 design (INIT_FROM; later, sharper designs).
    python plot_slide_TO3D_grid.py  ->  figures/slide_TO3D_grid.{png,pdf} (+ copy to the talk figures/)
logs: logs/nemo_TO3D/ (rsync of /work/classic/fr_ml1145-muffto/runs/TO3D/logs)
"""
import glob
import os
import re
import shutil

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
L = os.path.join(HERE, 'logs', 'nemo_TO3D')
TALK = os.path.expanduser('~/Projects/Talks/Slides-Green-Jacobi-topology-optimization-NANTES/figures')
RUNS = {64: ('0.0625', ''), 128: ('0.03125', ''), 192: ('0.0208333333', '_from128'), 256: ('0.015625', '_from128')}
RUNS_2H = {32: ('0.0625', ''), 64: ('0.03125', '')}               # l = 2h (dashed)
PRE = [('Green', 'Green', 'tab:green', 'o', 1), ('Green_Jacobi', 'Green-Jacobi', 'tab:blue', 's', 1),
       ('dual', r'dual ($\times 2$)', 'tab:red', '^', 2)]


def mean_cg(N, P, runs=RUNS):
    eta, tag = runs[N]
    files = [os.path.join(L, f'TO3D_N{N}_eta{eta}_{P}{tag}.log')]
    files += sorted(glob.glob(os.path.join(L, f'TO3D_N{N}_eta{eta}_{P}{tag}_seg*.log')))
    vals = []
    for f in files:
        if os.path.exists(f):
            vals += [int(v) for v in re.findall(r'CG mech\s*=\s*(\d+)', open(f).read())]
    return (float(np.mean(vals)), len(vals)) if vals else (None, 0)


plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 13})
fig, ax = plt.subplots(figsize=(4.4, 3.6), layout='constrained')
for P, lab, col, mk, fac in PRE:
    pts = [(N, *mean_cg(N, P)) for N in RUNS]
    pts = [(N, m * fac, n) for N, m, n in pts if m is not None]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], '-', color=col, lw=2.2, label=lab, zorder=3)
    for N, m, n in pts:
        ax.plot(N, m, mk, color=col, ms=7, mfc='white' if RUNS[N][1] else col, mew=1.6, zorder=5, clip_on=False)
    print(lab, [(N, round(m), n) for N, m, n in pts])
    pts = [(N, *mean_cg(N, P, RUNS_2H)) for N in RUNS_2H]
    pts = [(N, m * fac, n) for N, m, n in pts if m is not None]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], '--', color=col, lw=2.2, zorder=3)
    for N, m, n in pts:
        ax.plot(N, m, mk, color=col, ms=7, mfc=col, mew=1.6, zorder=5, clip_on=False)
    print(lab, 'l=2h', [(N, round(m), n) for N, m, n in pts])
ax.plot([], [], 'o', color='0.3', mfc='0.3', label='sine start')
ax.plot([], [], 'o', color='0.3', mfc='white', label=r'from $128^3$ design')
ax.plot([], [], '-', color='0.3', lw=2.2, label='$l = 4h$')
ax.plot([], [], '--', color='0.3', lw=2.2, label='$l = 2h$')
ax.set_xscale('log', base=2)
TICKS = [32] + list(RUNS)
ax.set_xticks(TICKS), ax.set_xticklabels([f'${n}^3$' for n in TICKS]), ax.minorticks_off()
ax.set_xlim(28, 290), ax.set_ylim(0, 450)
ax.set(xlabel='grid size', ylabel=r'CG per elastic solve (dual $\times 2$)')
ax.grid(alpha=0.3)
fig.legend(loc='outside upper center', ncol=3, fontsize=10, frameon=False)
out = os.path.join(HERE, 'figures', 'slide_TO3D_grid')
for ext in ('png', 'pdf'):
    fig.savefig(f'{out}.{ext}', dpi=150)
    shutil.copy(f'{out}.{ext}', TALK)
print('wrote', out)
