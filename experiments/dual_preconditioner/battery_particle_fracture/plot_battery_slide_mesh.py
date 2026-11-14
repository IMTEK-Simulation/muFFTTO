"""Slide figures (style of ../topology_optimization/plot_slide_green_grid_independence.py): mean CG iterations per
elastic solve vs grid, NEMO battery mesh study (fixed l = 0.23 um), Green / Green-Jacobi / dual.
Solid / circles: during crack growth; dashed / squares: before crack growth (windows from the dual run, identical
trajectories). Grey: N = 32, l/h = 1.2 under-resolved. Open marker: run not finished (3D 256^3 Green-Jacobi stopped
at step 32, its value covers only the first 32 steps of the 'before' window).

    python plot_battery_slide_mesh.py   ->  figures/slides/battery_mesh_cg_{2D,3D}.{png,pdf}
    DUAL_FACTOR=2 (env): dual iterations counted x2 (cost of one dual iteration ~ 2 Green iterations)
                          -> figures/slides/battery_mesh_cg_{2D,3D}_dualx2.{png,pdf}
"""
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
DATA = os.path.join(HERE, 'exp_data', 'nemo')
PRECS = ['Green', 'Green_Jacobi', 'dual']
COLOR = {'Green': '#2ca02c', 'Green_Jacobi': '#1f77b4', 'dual': '#d62728'}
LABEL = {'Green': 'Green', 'Green_Jacobi': 'Green-Jacobi', 'dual': 'dual'}
GRIDS = {2: [32, 64, 128, 256, 512, 1024], 3: [32, 64, 128, 256]}
DUAL_FACTOR = float(os.environ.get('DUAL_FACTOR', 1))


def load(dim, N, P):
    base = os.path.join(DATA, f'battery_N{N}_{P}_ms0.01{"_3D" if dim == 3 else ""}_tu')
    if not os.path.exists(base + '_steps.csv'):
        return None
    steps = np.atleast_1d(np.genfromtxt(base + '_steps.csv', delimiter=',', names=True, dtype=None, encoding=None))
    its = np.atleast_1d(np.genfromtxt(base + '_iterations.csv', delimiter=',', names=True, dtype=None, encoding=None,
                                      missing_values='', filling_values=-1))
    return None if steps.size < 2 else dict(steps=steps, its=its, done=steps.size >= 225)


def windows(steps):                                   # same as plot_battery_mesh.py
    g = steps['a_c'] - steps['a_c'][0]
    on = g > 0.002
    idx = np.arange(len(g))
    i0 = np.argmax(on)
    i1 = np.argmax(g >= 0.95 * g[-1])
    return idx < i0, (idx >= i0) & (idx <= max(i1, i0))


plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 13})
for dim in (2, 3):
    fig, ax = plt.subplots(figsize=(4.4, 3.3), layout='constrained')
    ax.axvspan(32 / 1.25, 32 * 1.25, color='0.88', zorder=0, lw=0)
    rows = []
    for P in ['Green', 'Green_Jacobi', 'dual']:
        for w, (ls, mk) in ((0, ('--', 's')), (1, ('-', 'o'))):
            xs, ys, done = [], [], []
            for N in GRIDS[dim]:
                ref, run = load(dim, N, 'dual'), load(dim, N, P)
                if ref is None or run is None:
                    continue
                mask = windows(ref['steps'])[w]
                vals = run['its']['cg_u'][np.isin(run['its']['step'], ref['steps']['step'][mask])]
                vals = vals[vals >= 0]
                if len(vals) == 0:
                    continue
                xs.append(N), ys.append(float(np.mean(vals)) * (DUAL_FACTOR if P == 'dual' else 1.0)), done.append(run['done'])
                rows.append((P, 'before during'.split()[w], N, ys[-1], run['done']))
            ax.plot(xs, ys, ls, color=COLOR[P], lw=2.2 if w else 1.6, alpha=1.0 if w else 0.75, clip_on=False)
            for n, v, ok in zip(xs, ys, done):
                ax.plot(n, v, mk, color=COLOR[P], ms=6 if w else 5, mfc=COLOR[P] if ok else 'white', mew=1.5,
                        clip_on=False, zorder=5, alpha=1.0 if w else 0.75)
    ax.set_xscale('log', base=2)
    ax.set_xticks(GRIDS[dim]), ax.set_xticklabels([f'{n}$^{dim}$' for n in GRIDS[dim]]), ax.minorticks_off()
    ax.set_xlim(GRIDS[dim][0], GRIDS[dim][-1])
    ax.set_yscale('log')
    yt = [20, 50, 100, 200, 500, 1000, 2000]
    ax.set_yticks(yt), ax.set_yticklabels([str(v) for v in yt]), ax.minorticks_off()
    ax.set_ylim(20, 2000)
    ax.set(xlabel='grid size', ylabel=r'CG per elastic solve (dual $\times 2$)' if DUAL_FACTOR == 2
           else 'CG iterations per elastic solve')
    ax.grid(alpha=0.3)
    h = [plt.Line2D([], [], color=COLOR[P], lw=2.2) for P in PRECS]
    h += [plt.Line2D([], [], color='0.3', ls='-', marker='o', lw=1.8, ms=5),
          plt.Line2D([], [], color='0.3', ls='--', marker='s', lw=1.4, ms=4.5)]
    ax.legend(h, [LABEL[P] for P in PRECS] + ['cracking', 'before'], loc='upper left', fontsize=10, ncol=2,
              handlelength=1.8, columnspacing=0.9, framealpha=0.9)
    out = os.path.join(HERE, 'figures', 'slides', f'battery_mesh_cg_{dim}D' + ('_dualx2' if DUAL_FACTOR == 2 else ''))
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}.{ext}', dpi=200)
    print('wrote', out)
    for r in rows:
        print(f'  {dim}D {LABEL[r[0]]:13s} {r[1]:7s} N={r[2]:5d}  {r[3]:7.1f}{"" if r[4] else "  (unfinished)"}')
