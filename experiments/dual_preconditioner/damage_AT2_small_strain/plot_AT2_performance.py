"""Solver performance of the small-strain AT2 runs over the grid, from the per-staggered-iteration records
exp_data/AT2_N<N>_<prec><TAG>_iterations.csv (one row per elastic + damage solve) and the load step lines of
logs/AT2_N<N>_<prec><TAG>.log (sigma_xx for the phases).

Phases of the fracture process (load steps):
    intact            before the first crack breaks through (first drop of sigma_xx by more than 5 %)
    partly cracked    cracks through, the cell still carries load
    broken            the cell is cut through: sigma_xx < 10 % of the peak
    break steps       the load steps in which sigma_xx drops by more than 5 % (a ligament snaps; many staggered
                      iterations), counted separately

-> figures/AT2_performance_phases<TAG>.png, .pdf   per phase (columns):
       row 1  elastic CG iterations per elastic solve (mean over all staggered iterations of the phase; bars: min-max)
       row 2  elastic CG iterations per load step (sum over its staggered iterations; mean over the steps of the phase)
       row 3  staggered iterations per load step (the same for all preconditioners: one grey curve)
   figures/AT2_performance_totals<TAG>.png, .pdf     per run:
       total elastic CG iterations, mean elastic CG iterations per solve, wall time per elastic CG iteration
       (preconditioner update, operator and preconditioner applications), total time of the elastic solves

Open markers: run not finished yet (the load steps so far).
A run continued from a restart (RUN_SUFFIX=_restart: ..._restart_iterations.csv, ..._restart.log) is joined to the
stopped run: its load steps from the first restarted one on replace those of the stopped run.

    python plot_AT2_performance.py [TAG]      (default _polydisperse_pores_l0.03125)
"""
import os
import re
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
TAG = sys.argv[1] if len(sys.argv) > 1 else '_polydisperse_pores_l0.03125'
PRECS = [('Green', 'Green', '#1baf7a', 'o'), ('Green_Jacobi', 'Green-Jacobi', '#2a78d6', 's'),
         ('dual', 'dual', '#d6402a', '^')]
PHASES = ['intact', 'partly cracked', 'broken', 'break steps']
STEP = re.compile(r'^step +(\d+) .*sigma_xx = (\S+)', re.M)
plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.5, 'legend.fontsize': 6})


def path(N, p, kind, suffix=''):
    return os.path.join(HERE, *{'csv': ('exp_data', f'AT2_N{N}_{p}{TAG}{suffix}_iterations.csv'),
                                'log': ('logs', f'AT2_N{N}_{p}{TAG}{suffix}.log')}[kind])


def restarted(N, p):
    return os.path.exists(path(N, p, 'csv', '_restart')) and os.path.exists(path(N, p, 'log', '_restart'))


def finished(N, p):
    log = path(N, p, 'log', '_restart' if restarted(N, p) else '')
    return os.path.exists(log) and 'Elapsed time' in open(log).read()


def read_part(N, p, suffix=''):
    """per-iteration records of the finished load steps and the (step, sigma_xx) of these steps"""
    it = np.genfromtxt(path(N, p, 'csv', suffix), delimiter=',', names=True, ndmin=1)
    steps = np.array([[int(k), float(s)] for k, s in STEP.findall(open(path(N, p, 'log', suffix)).read())])
    if len(steps) == 0:
        return it[:0], np.zeros((0, 2))
    return it[it['step'] <= steps[-1, 0]], steps


def load_run(N, p):
    """per-iteration records and the phase of every load step, None if the run has no records"""
    if not (os.path.exists(path(N, p, 'csv')) and os.path.exists(path(N, p, 'log'))):
        return None
    it, steps = read_part(N, p)
    if restarted(N, p):
        it_r, steps_r = read_part(N, p, '_restart')
        if len(steps_r):
            first = steps_r[0, 0]
            it = np.concatenate([it[it['step'] < first], it_r])
            steps = np.concatenate([steps[steps[:, 0] < first], steps_r])
    if len(steps) == 0:
        return None
    sigma = steps[:, 1]
    drop = np.r_[False, sigma[1:] < 0.95 * sigma[:-1]]
    phase = np.maximum.accumulate(drop).astype(int)
    phase[(phase == 1) & (sigma < 0.1 * sigma.max())] = 2  # not the first load steps, also small sigma
    phase[drop] = 3
    return it, phase


GRIDS = [N for N in (32, 64, 128, 256, 512, 1024) if any(os.path.exists(path(N, p, 'csv')) for p, *_ in PRECS)]
runs = {(N, p): load_run(N, p) for N in GRIDS for p, *_ in PRECS}


def per_phase(it, phase, i):
    """(CG per solve: mean, min, max), (CG per step: mean, min, max), (staggered its per step: mean, min, max)"""
    steps = np.flatnonzero(phase == i)
    if steps.size == 0:
        return None
    rows = np.isin(it['step'], steps)
    per_step_cg = np.array([it['cg_u'][it['step'] == k].sum() for k in steps])
    per_step_its = np.array([np.sum(it['step'] == k) for k in steps])
    per_solve = it['cg_u'][rows]
    return [(v.mean(), v.min(), v.max()) for v in (per_solve, per_step_cg, per_step_its)]


def style(ax):
    ax.set_xscale('log', base=2)
    ticks = [2 ** k for k in range(int(np.log2(GRIDS[0])), int(np.log2(GRIDS[-1])) + 1)]   # also grids without runs
    # more than four ticks: every other one labelled, counted from the finest grid
    ax.set_xticks(ticks, [str(N) if len(ticks) <= 4 or (len(ticks) - 1 - k) % 2 == 0 else ''
                          for k, N in enumerate(ticks)])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(GRIDS[0] / 1.3, GRIDS[-1] * 1.9)
    ax.set_ylim(0, 1.12 * ax.get_ylim()[1])        # room above the highest marker
    ax.grid(alpha=0.3)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def draw(ax, j, c, m, label, grids, stats, open_mask):
    mean = np.array([s[0] for s in stats])
    lo, hi = mean - [s[1] for s in stats], [s[2] for s in stats] - mean
    x = np.array(grids) * 1.04 ** (j - 1)                  # small offset so the bars do not overlap
    ax.errorbar(x, mean, yerr=(lo, hi), color=c, marker=m, ms=3.5, lw=1.3, elinewidth=0.6, capsize=1.5,
                label=label)
    ax.plot(x[open_mask], mean[open_mask], ls='none', marker=m, ms=2.2, color='white', zorder=3)
    ax.annotate(f'{mean[-1]:,.0f}' if mean[-1] >= 10 else f'{mean[-1]:.2g}', (x[-1], mean[-1]),
                textcoords='offset points', xytext=(6, 0), fontsize=5, va='center', color='0.25')


# ---------------------------------------------------------------------------- per phase
ROWS = ['elastic CG iterations\nper elastic solve', 'elastic CG iterations\nper load step',
        'staggered iterations\nper load step']
# phases without load steps in every run (e.g. 'partly cracked' with a single pore) are not drawn
shown = [i for i in range(len(PHASES))
         if any(run is not None and per_phase(*run, i) is not None for run in runs.values())]
fig, axes = plt.subplots(3, len(shown), figsize=(1.8 * len(shown), 5.4), sharex=True, squeeze=False)
for col, i in enumerate(shown):
    title = PHASES[i]
    axes[0, col].set_title(title)
    for j, (p, label, c, m) in enumerate(PRECS):
        done = [(N, s) for N in GRIDS if runs[N, p] is not None and (s := per_phase(*runs[N, p], i)) is not None]
        if not done:
            continue
        grids = [N for N, _ in done]
        open_mask = np.array([not finished(N, p) for N in grids])
        for r in range(2):
            draw(axes[r, col], j, c, m, label, grids, [s[r] for _, s in done], open_mask)
    # staggered iterations: the damage evolution does not depend on the elastic preconditioner; the run that got
    # furthest on each grid (dual first)
    done = []
    for N in GRIDS:
        run = next((runs[N, p] for p in ('dual', 'Green_Jacobi', 'Green') if runs[N, p] is not None), None)
        if run is not None and (s := per_phase(*run, i)) is not None:
            done.append((N, s[2]))
    if done:
        draw(axes[2, col], 1, '0.45', 'D', 'all preconditioners', [N for N, _ in done], [s for _, s in done],
             np.zeros(len(done), dtype=bool))
for r, ylabel in enumerate(ROWS):
    axes[r, 0].set_ylabel(ylabel)
for ax in axes.flat:
    style(ax)
for ax in axes[-1]:
    ax.set_xlabel(r'grid $N$ ($N^2$ pixels)')
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='lower center', ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=0.6, h_pad=0.8)
out = os.path.join(HERE, 'figures', f'AT2_performance_phases{TAG}')
fig.savefig(out + '.png', dpi=250)
fig.savefig(out + '.pdf')

# ---------------------------------------------------------------------------- per run
PANELS = ['elastic CG iterations\n(total)', 'elastic CG iterations\nper elastic solve',
          'time per elastic\nCG iteration [ms]', 'time of the elastic\nsolves [s]']
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2))
for j, (p, label, c, m) in enumerate(PRECS):
    done = [(N, runs[N, p][0]) for N in GRIDS if runs[N, p] is not None]
    if not done:
        continue
    grids = [N for N, _ in done]
    values = [[it['cg_u'].sum() for _, it in done], [it['cg_u'].mean() for _, it in done],
              [1e3 * it['time_u'].sum() / it['cg_u'].sum() for _, it in done],
              [it['time_u'].sum() for _, it in done]]
    open_mask = np.array([not finished(N, p) for N in grids])
    for ax, v in zip(axes, values):
        draw(ax, j, c, m, label, grids, [(x, x, x) for x in v], open_mask)
for ax, ylabel in zip(axes, PANELS):
    style(ax)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(r'grid $N$ ($N^2$ pixels)')
axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v / 1000:g}k' if v else '0'))
fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=0.8)
out = os.path.join(HERE, 'figures', f'AT2_performance_totals{TAG}')
fig.savefig(out + '.png', dpi=250)
fig.savefig(out + '.pdf')
print(out + '.png')
