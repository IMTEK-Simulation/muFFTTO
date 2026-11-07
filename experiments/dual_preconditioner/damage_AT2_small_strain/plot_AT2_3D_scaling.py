"""
AT2 preconditioner scaling at fixed crack length scale l, grid refined (l/h = 2, 4, 8, 16): the 2D Cahn-Hilliard series
(l = 1/32, 64^2 ... 512^2) next to its 3D confirmation on NEMO (l = 1/16, 32^3 ... 256^3: Cahn-Hilliard and stiff tough
grains).

Logs: schneider2025_notch/logs/AT2_N<N>_<P>_incl0.1_l0.03125_k0.0001_ch.log (2D),
      schneider2025_notch/logs/nemo_3D/AT2_N<N>_<P>_incl0.1_l0.0625_3D_k0.0001_ch.log,
      logs/nemo_3D/AT2_N<N>_<P>_polydisperse_pores_incl10_l0.0625_3D.log (copied from NEMO).
Per grid, all preconditioners are compared over the same load steps: those every run of that grid has finished
(running runs are cut there; the number of steps is printed). Left: CG iterations per elastic solve. Right: cost
relative to Green-Jacobi (< 1: cheaper): 3D elastic wall time (same GPU type per grid), 2D elastic CG work with dual
iterations counted twice (the 2D runs ran on different machines). Grids with fewer than MIN_STEPS common steps (still
before the crack, e.g. a just started 256^3) are only printed.
usage:  python plot_AT2_3D_scaling.py        writes figures/AT2_3D_scaling.png
"""
import os
import re

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
MIN_STEPS = 35          # the 3D cells crack at load step ~31
plt.rcParams.update({'font.size': 13})
COLOR = {'Green': 'tab:green', 'Green_Jacobi': 'tab:blue', 'dual': 'tab:red'}
LABEL = {'Green': 'Green', 'Green_Jacobi': 'Green-Jacobi', 'dual': 'dual'}
CASES = [  # title, grid -> l/h, log path pattern, cost measure
    ('2D Cahn-Hilliard (l = 1/32)', {64: 2, 128: 4, 256: 8, 512: 16},
     'schneider2025_notch/logs/AT2_N{N}_{P}_incl0.1_l0.03125_k0.0001_ch.log', 'work'),
    ('3D Cahn-Hilliard (l = 1/16)', {32: 2, 64: 4, 128: 8, 256: 16},
     'schneider2025_notch/logs/nemo_3D/AT2_N{N}_{P}_incl0.1_l0.0625_3D_k0.0001_ch.log', 'time'),
    ('3D stiff tough grains (l = 1/16)', {32: 2, 64: 4, 128: 8, 256: 16},
     'logs/nemo_3D/AT2_N{N}_{P}_polydisperse_pores_incl10_l0.0625_3D.log', 'time'),
]


def steps(path):
    """Per finished load step: (solves, elastic CG iterations, elastic time)."""
    rows = []
    for line in open(path):
        if not line.startswith('step'):
            continue
        s = re.search(r'staggered its = +(\d+)', line)
        u = re.search(r'step CG \(u, d\) = \((\d+)', line)
        t = re.search(r'time \(u, d, step\) = \(([0-9.]+)', line)
        rows.append((int(s.group(1)), int(u.group(1)), float(t.group(1))))
    return np.array(rows) if rows else None


fig, axs = plt.subplots(3, 2, figsize=(15, 14))
for row, (title, grids, pattern, measure) in enumerate(CASES):
    per, cost = {P: [] for P in COLOR}, {P: [] for P in COLOR}
    print(f'== {title}')
    for N, lh in grids.items():
        data = {P: steps(os.path.join(HERE, pattern.format(N=N, P=P))) for P in COLOR
                if os.path.exists(os.path.join(HERE, pattern.format(N=N, P=P)))}
        data = {P: d for P, d in data.items() if d is not None and len(d) > 2}
        if 'Green_Jacobi' not in data or len(data) < 2:
            continue
        K = min(len(d) for d in data.values())               # common finished load steps
        if K < MIN_STEPS:                                    # not yet through the peak: not comparable, print only
            print(f'  N {N:4d} (l/h {lh:2d}): only {K} common steps (before the crack), not plotted: ' + '  '.join(
                f'{P} {d[:K, 1].sum() / d[:K, 0].sum():.1f} CG/solve' for P, d in data.items()))
            continue
        line = f'  N {N:4d} (l/h {lh:2d}), first {K} steps:'
        ref = None
        for P, d in data.items():
            s, u, t = d[:K].sum(axis=0)
            c = t if measure == 'time' else u * (2 if P == 'dual' else 1)
            if P == 'Green_Jacobi':
                ref = c
            per[P].append((lh, u / s))
            cost[P].append((lh, c))
            line += f'  {P} {u / s:6.1f} CG/solve ({c:9.0f}{" s" if measure == "time" else ""})'
        for P in COLOR:
            if cost[P] and cost[P][-1][0] == lh:
                cost[P][-1] = (lh, cost[P][-1][1] / ref)
        print(line)
    for P in COLOR:
        if per[P]:
            x, y = np.array(per[P]).T
            axs[row, 0].plot(x, y, 'o-', color=COLOR[P], lw=2.5, ms=8, label=LABEL[P])
            x, y = np.array(cost[P]).T
            axs[row, 1].plot(x, y, 'o-', color=COLOR[P], lw=2.5, ms=8, label=LABEL[P])
    axs[row, 0].set(ylabel='CG iterations per elastic solve', title=f'{title}: iterations', ylim=(0, None))
    axs[row, 1].axhline(1, color='tab:blue', lw=1, ls=':')
    axs[row, 1].set(ylabel=('elastic wall time' if measure == 'time' else 'CG work (dual x2)') + ' / Green-Jacobi',
                    title=f'{title}: cost relative to Green-Jacobi', ylim=(0, None))
    for ax in axs[row]:
        ax.set_xscale('log', base=2)
        ax.set_xticks([2, 4, 8, 16]); ax.set_xticklabels(['2', '4', '8', '16'])
        ax.set_xlim(1.7, 19)
        ax.set_xlabel('l / h')
        ax.grid(alpha=0.3)
        ax.legend()
fig.suptitle('AT2: preconditioner cost at fixed crack length l, grid refined (2D Cahn-Hilliard and its 3D check)',
             fontsize=15)
fig.tight_layout()
out = os.path.join(HERE, 'figures', 'AT2_3D_scaling.png')
fig.savefig(out, dpi=110)
print('saved', out)
