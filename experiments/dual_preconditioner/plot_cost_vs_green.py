"""
Cost of Green-Jacobi and dual relative to Green (= 1, below 1: cheaper than Green) against the element size h, for AT2
fracture and topology optimization, 2D and 3D. Two measures where available:
  solid  -- CG work per solve with a dual iteration counted as 2 (two FFT pairs; pessimistic for the dual),
  dashed -- measured time: elastic wall time per solve (3D AT2, same GPU type per grid) or core-hours of the whole run
            (2D TO, contrast_summary.csv).
Same data and grid selection as plot_mesh_scaling.py (AT2: per grid the load steps all runs finished; TO 3D: whole runs
so far); grids without a Green run are left out.
usage:  python plot_cost_vs_green.py        writes figures/cost_vs_green.png (in this folder)
"""
import csv
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(HERE, 'topology_optimization'))
import plot_TO3D_results as TO3D           # noqa: E402

plt.rcParams.update({'font.size': 13})
COLOR = {'Green_Jacobi': 'tab:blue', 'dual': 'tab:red'}
LABEL = {'Green_Jacobi': 'Green-Jacobi', 'dual': 'dual'}
PRECONDS = ('Green', 'Green_Jacobi', 'dual')
AT2 = os.path.join(HERE, 'damage_AT2_small_strain')
MIN_STEPS = 35


def at2_steps(path):
    rows = []
    for line in open(path):
        if line.startswith('step'):
            s = int(re.search(r'staggered its = +(\d+)', line).group(1))
            u = int(re.search(r'step CG \(u, d\) = \((\d+)', line).group(1))
            t = float(re.search(r'time \(u, d, step\) = \(([0-9.]+)', line).group(1))
            rows.append((s, u, t))
    return np.array(rows) if rows else None


def at2(grids, pattern, timed):
    """{N: {precond: (work per solve, time per solve or None)}} over the common finished load steps."""
    out = {}
    for N in grids:
        data = {}
        for P in PRECONDS:
            p = os.path.join(AT2, pattern.format(N=N, P=P))
            d = at2_steps(p) if os.path.exists(p) else None
            if d is not None and len(d) > 2:
                data[P] = d
        if 'Green' not in data or len(data) < 2:
            continue
        K = min(len(d) for d in data.values())
        if K < MIN_STEPS:
            continue
        out[N] = {P: (d[:K, 1].sum() / d[:K, 0].sum() * (2 if P == 'dual' else 1),
                      d[:K, 2].sum() / d[:K, 0].sum() if timed else None) for P, d in data.items()}
    return out


def to2d():
    out = {}
    with open(os.path.join(HERE, 'topology_optimization', 'figures', 'nemo', 'contrast_summary.csv')) as f:
        for r in csv.DictReader(f):
            if float(r['contrast']) == 1e4 and float(r['eta_h']) == 4 and r['precond'] in PRECONDS:
                # 'work' counts the dual x2; time per solve ~ core-hours / its (its = L-BFGS iterations, similar)
                out.setdefault(int(r['N']), {})[r['precond']] = (float(r['work']),
                                                                 float(r['core_h']) / float(r['its']))
    return {N: v for N, v in out.items() if 'Green' in v}


def to3d():
    out = {}
    for N, eta, sfx in TO3D.SERIES['4h']:
        for P in PRECONDS:
            lg = TO3D.parse_log(N, P, eta, sfx)
            if lg is None or len(lg['it']) < 10:
                continue
            out.setdefault(N, {})[P] = (lg['cg'][1:].sum() / max(lg['ns'][1:].sum(), 1) * (2 if P == 'dual' else 1),
                                        None)
    return {N: v for N, v in out.items() if 'Green' in v}


PANELS = [
    ('AT2 2D Cahn-Hilliard (l = 1/32)',
     at2([64, 128, 256, 512], 'schneider2025_notch/logs/AT2_N{N}_{P}_incl0.1_l0.03125_k0.0001_ch.log', False)),
    ('AT2 3D Cahn-Hilliard (l = 1/16)',
     at2([32, 64, 128, 256], 'schneider2025_notch/logs/nemo_3D/AT2_N{N}_{P}_incl0.1_l0.0625_3D_k0.0001_ch.log', True)),
    ('AT2 3D stiff tough grains (l = 1/16)',
     at2([32, 64, 128, 256], 'logs/nemo_3D/AT2_N{N}_{P}_polydisperse_pores_incl10_l0.0625_3D.log', True)),
    ('TO 2D (contrast 1e4, $\\eta$ = 4h)', to2d()),
    ('TO 3D (contrast 1e4, $\\eta$ = 4h)', to3d()),
]

fig, axs = plt.subplots(2, 3, figsize=(19, 11.5))
for ax, (title, data) in zip(axs.ravel(), PANELS):
    print(f'== {title}')
    Ns = sorted(data)
    h = np.array([1.0 / N for N in Ns])
    for P in COLOR:
        work = [data[N][P][0] / data[N]['Green'][0] if P in data[N] else np.nan for N in Ns]
        ax.plot(h, work, 'o-', color=COLOR[P], lw=2.5, ms=8, label=f'{LABEL[P]}, work (dual x2)')
        times = [data[N][P][1] / data[N]['Green'][1] if P in data[N] and data[N][P][1] else np.nan for N in Ns]
        if not np.all(np.isnan(times)):
            ax.plot(h, times, 's--', color=COLOR[P], lw=2, ms=7, mfc='white', mew=2, label=f'{LABEL[P]}, time')
        print(f'  {P:13s} work/Green ' + '  '.join(f'N{N}: {w:.2f}' for N, w in zip(Ns, work))
              + ('   time/Green ' + '  '.join(f'N{N}: {t:.2f}' for N, t in zip(Ns, times))
                 if not np.all(np.isnan(times)) else ''))
    ax.axhline(1, color='tab:green', lw=2.5, label='Green')
    ax.set_xscale('log')
    ax.set_xticks(h); ax.set_xticklabels([f'1/{N}' for N in Ns]); ax.minorticks_off()
    ax.invert_xaxis()
    ax.set_ylim(0, max(1.3, ax.get_ylim()[1]))
    ax.set(xlabel='element size h (refinement ->)', ylabel='cost per solve / Green', title=title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
ax = axs.ravel()[-1]
ax.axis('off')
ax.text(0.0, 0.95, 'cost relative to Green (green line = 1)\nbelow 1: cheaper than Green\n\n'
        'solid: CG work, dual iteration counted as 2\ndashed: measured time\n'
        '  (3D AT2: elastic wall time per solve,\n   2D TO: core-hours per L-BFGS iteration)\n\n'
        'only grids where Green was run', va='top', fontsize=14, transform=ax.transAxes)
fig.suptitle('Green-Jacobi and dual compared with Green, against the mesh size', fontsize=16)
fig.tight_layout()
out = os.path.join(HERE, 'figures', 'cost_vs_green.png')
fig.savefig(out, dpi=110)
print('saved', out)
