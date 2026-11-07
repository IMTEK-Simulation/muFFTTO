"""
Mesh-size scaling of the three preconditioners: CG iterations per linear (elastic) solve against the element size h,
log-log, with the fitted exponent alpha of  iterations ~ h^-alpha  (alpha = 0: mesh independent; Green-Jacobi of a
Jacobi-type scaling ~ h^-1 expected). The dual's iterations cost ~2x (2D) / ~1.5x (3D) a Green-Jacobi one; that
constant factor does not change the exponent.

AT2 (fixed crack length l, so h -> 0 means l/h grows): per grid the runs are compared over the load steps all of them
finished (damage_AT2_small_strain/plot_AT2_3D_scaling.py); grids still before the crack are left out.
TO (contrast 1e4, eta = 4h, i.e. eta shrinks with h): 2D from topology_optimization/figures/nemo/contrast_summary.csv
(whole runs); 3D from the NEMO logs (whole runs so far, iteration 0 excluded); runs not converged yet: open markers.
Two figures with the same layout:
  figures/mesh_scaling.png        CG iterations per linear solve
  figures/mesh_scaling_total.png  total CG iterations of the whole solution (AT2: over the compared load steps; TO: whole
                                  optimisation; TO 2D estimated as CG per solve x optimizer iterations x 6 solves, as the
                                  3D logs show 2 solves per load case and iteration)
The raw counts count a dual iteration as one; the dashed red line is the dual weighted by its cost per iteration
(DUAL_COST: 2 in 2D, 1.5 in 3D), i.e. Green-Jacobi-iteration equivalents.
usage:  python plot_mesh_scaling.py
"""
import csv
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(HERE, 'topology_optimization'))
import plot_TO3D_results as TO3D           # noqa: E402  (log parser of the 3D TO runs)

plt.rcParams.update({'font.size': 13})
COLOR = {'Green': 'tab:green', 'Green_Jacobi': 'tab:blue', 'dual': 'tab:red'}
LABEL = {'Green': 'Green', 'Green_Jacobi': 'Green-Jacobi', 'dual': 'dual'}
AT2 = os.path.join(HERE, 'damage_AT2_small_strain')
MIN_STEPS = 35
DUAL_COST = {2: 2.0, 3: 1.5}             # cost of a dual iteration in Green-Jacobi iterations
TO2D_SOLVES = 6                            # linear solves per 2D TO iteration (3 load cases, state + adjoint)


def at2_steps(path):
    rows = []
    for line in open(path):
        if line.startswith('step'):
            s = int(re.search(r'staggered its = +(\d+)', line).group(1))
            u = int(re.search(r'step CG \(u, d\) = \((\d+)', line).group(1))
            rows.append((s, u))
    return np.array(rows) if rows else None


def at2_series(grids, pattern, min_steps=MIN_STEPS):
    """{precond: [(N, CG per solve, finished)]} over the common finished load steps of each grid; a grid is used only
    once all its runs have finished min_steps load steps (past the break step, else the grids compare unlike windows)."""
    out = {P: [] for P in COLOR}
    for N in grids:
        data = {}
        for P in COLOR:
            p = os.path.join(AT2, pattern.format(N=N, P=P))
            if os.path.exists(p):
                d = at2_steps(p)
                if d is not None and len(d) > 2:
                    data[P] = (d, 'Elapsed' in open(p).read())
        if len(data) < 2:
            continue
        K = min(len(d) for d, _ in data.values())
        if K < min_steps:
            continue
        for P, (d, done) in data.items():
            out[P].append((N, d[:K, 1].sum() / d[:K, 0].sum(), d[:K, 1].sum(), done))
    return out


def to2d_series():
    out = {P: [] for P in COLOR}
    with open(os.path.join(HERE, 'topology_optimization', 'figures', 'nemo', 'contrast_summary.csv')) as f:
        for r in csv.DictReader(f):
            if float(r['contrast']) == 1e4 and float(r['eta_h']) == 4 and r['precond'] in out:
                out[r['precond']].append((int(r['N']), float(r['cg']), float(r['cg']) * int(r['its']) * TO2D_SOLVES, True))
    return out


def to3d_series():
    out = {P: [] for P in COLOR}
    for N, eta, sfx in TO3D.SERIES['4h']:
        for P in COLOR:
            lg = TO3D.parse_log(N, P, eta, sfx)
            if lg is None or len(lg['it']) < 10:
                continue
            out[P].append((N, lg['cg'][1:].sum() / max(lg['ns'][1:].sum(), 1), lg['cg'][1:].sum(), lg['converged']))
    return out


PANELS = [   # (title, dimension, series)
    ('AT2 2D Cahn-Hilliard (l = 1/32)', 2, at2_series([64, 128, 256, 512],
                                                   'schneider2025_notch/logs/AT2_N{N}_{P}_incl0.1_l0.03125_k0.0001_ch.log')),
    ('AT2 3D Cahn-Hilliard (l = 1/16)', 3, at2_series([32, 64, 128, 256],
                                                   'schneider2025_notch/logs/nemo_3D/AT2_N{N}_{P}_incl0.1_l0.0625_3D_k0.0001_ch.log')),
    ('AT2 3D stiff tough grains (l = 1/16)', 3, at2_series([32, 64, 128, 256],
                                                        'logs/nemo_3D/AT2_N{N}_{P}_polydisperse_pores_incl10_l0.0625_3D.log',
                                                        min_steps=38)),   # break at load step 37
    ('TO 2D (contrast 1e4, $\\eta$ = 4h)', 2, to2d_series()),
    ('TO 3D (contrast 1e4, $\\eta$ = 4h; 192$^3$, 256$^3$ start from the 128$^3$ design)', 3, to3d_series()),
]

def plot(metric):
    """metric 'per_solve' (CG per linear solve) or 'total' (CG of the whole solution)."""
    k = 1 if metric == 'per_solve' else 2
    ylab = 'CG iterations per solve' if metric == 'per_solve' else 'total CG iterations (thousands)'
    scale = 1.0 if metric == 'per_solve' else 1e-3
    fig, axs = plt.subplots(2, 3, figsize=(19, 11.5))
    print(f'#### {metric}')
    for ax, (title, dim, series) in zip(axs.ravel(), PANELS):
        print(f'== {title}')
        allN = sorted({p[0] for pts in series.values() for p in pts})
        for P, pts in series.items():
            if not pts:
                continue
            pts = sorted(pts)
            N = np.array([p[0] for p in pts], float); y = scale * np.array([p[k] for p in pts]); done = [p[-1] for p in pts]
            h = 1.0 / N
            # totals of unfinished runs are partial (shorter window / fewer optimizer iterations): fit finished points only
            use = np.ones(len(pts), bool) if metric == 'per_solve' else np.array(done)
            if metric == 'total' and title.startswith('TO 3D'):
                use[:] = False                  # cold starts (64, 128) and warm starts (192, 256): totals not comparable
            alpha = -np.polyfit(np.log(h[use]), np.log(y[use]), 1)[0] if use.sum() > 1 else np.nan
            ax.plot(h, y, '-', color=COLOR[P], lw=2.5,
                    label=f'{LABEL[P]}:  $\\alpha$ = {alpha:.2f}' if np.isfinite(alpha) else LABEL[P])
            for hi, yi, di in zip(h, y, done):
                ax.plot(hi, yi, 'o', ms=9, color=COLOR[P], mfc=COLOR[P] if di else 'white', mew=2)
            if P == 'dual':
                c = DUAL_COST[dim]
                ax.plot(h, c * y, '--', color=COLOR[P], lw=2, alpha=0.6, marker='s', ms=6,
                        label=f'dual x {c:g} (cost)')
            print(f'  {P:13s} alpha = {alpha:5.2f}   ' + '  '.join(f'N{int(n)}: {v:.4g}' for n, v in zip(N, y)))
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xticks([1 / n for n in allN]); ax.set_xticklabels([f'1/{n}' for n in allN])
        ax.minorticks_off()
        lo, hi_ = ax.get_ylim()
        cand = [m * 10 ** e for e in range(0, 9) for m in (1, 2, 5)]
        yt = [t for t in cand if lo <= t <= hi_]
        ax.set_yticks(yt); ax.set_yticklabels([f'{t:,.0f}'.replace(',', ' ') for t in yt])
        ax.invert_xaxis()                       # refinement to the right
        ax.set(xlabel='element size h (refinement ->)', ylabel=ylab)
        ax.set_title(title, fontsize=13 if len(title) > 45 else 14)
        ax.grid(alpha=0.3, which='major')
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax = axs.ravel()[-1]
    ax.axis('off')
    what = ('CG iterations per linear solve' if metric == 'per_solve' else
            'total CG iterations of the solution\n(AT2: over the load steps all runs finished;\n'
            'TO: whole optimisation, 2D estimated as\nCG/solve x optimizer its x 6 solves)')
    ax.text(0.0, 0.97, f'{what}\n\niterations ~ h$^{{-\\alpha}}$, fitted over the grids shown\n'
            '$\\alpha$ = 0: mesh independent;  $\\alpha$ = 1: doubles per refinement\n\n'
            'solid: raw counts (a dual iteration counted once)\n'
            'dashed red: dual x its cost per iteration\n(2 in 2D, 1.5 in 3D) = Green-Jacobi equivalents\n\n'
            'open markers: run not finished / converged yet' +
            ('' if metric == 'per_solve' else '\n(partial totals: not used in the fit)'),
            va='top', fontsize=13, transform=ax.transAxes)
    sup = 'per linear solve' if metric == 'per_solve' else 'total for the whole solution'
    fig.suptitle(f'Mesh-size scaling of the preconditioners, CG iterations {sup}: AT2 fracture and topology optimization',
                 fontsize=16)
    fig.tight_layout()
    out = os.path.join(HERE, 'figures', 'mesh_scaling.png' if metric == 'per_solve' else 'mesh_scaling_total.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110)
    print('saved', out)


for m in ('per_solve', 'total'):
    plot(m)
