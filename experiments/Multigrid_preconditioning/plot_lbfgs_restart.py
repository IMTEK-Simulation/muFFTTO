"""
L-BFGS restart test (2026-11-15): 128^3 run restarted from the converged 128^3 design (same grid), 15-25 iterations.
Variants (logs/fable/rst_<tag>_N128.log): base (H_0 = Helmholtz, unscaled: NuMPI's behaviour), gamma (H_0 scale 6000
carried over), first (first step capped at a largest design change of 0.05), numpi_first (same, production NuMPI path).
Left: decrease of the objective since the restart against the number of objective evaluations (each one = 12 linear
solves). Right: mean design change of every trial design of the line search (dots) and of the accepted ones (line).
    python plot_lbfgs_restart.py   ->  figures/lbfgs_restart_128.png
"""
import os
import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = [('base', 'unscaled H$_0$ (production now)', '#d1242f', '-', 'o'),
        ('gamma', 'H$_0$ scale carried over', '#1f6feb', '-', 's'),
        ('first', 'first step capped (0.05)', '#2da44e', '--', '^'),
        ('numpi_first', 'first step capped, NuMPI path', '#8250df', ':', 'v')]
plt.rcParams.update({'font.size': 14})
fig, (ax_f, ax_s) = plt.subplots(1, 2, figsize=(15, 5.8))
for tag, label, colour, style, marker in RUNS:
    log = os.path.join(HERE, 'logs', 'fable', f'rst_{tag}_N128.log')
    if not os.path.exists(log) or os.path.getsize(log) == 0:
        continue
    n_eval, f0, curve, trials, accepted, it = 0, None, [], [], [], 0
    last_trial = None
    for line in open(log):
        if line.startswith('EVAL'):
            n_eval += 1
            f = float(line.split()[3])
            if f0 is None:
                f0 = f
                curve.append((0, 0.0))
        elif line.startswith('TRIAL design'):
            last_trial = float(re.search(r'mean ([0-9.e+-]+)', line).group(1))
            trials.append((it + 1, last_trial))
        elif line.startswith('ITER'):
            it += 1
            f = float(line.split()[3])
            curve.append((n_eval - 1, 100 * (f0 - f) / f0))
            if last_trial is not None:
                accepted.append((it, last_trial))
    curve, trials, accepted = np.array(curve), np.array(trials), np.array(accepted)
    ax_f.plot(curve[:, 0], curve[:, 1], style, color=colour, marker=marker, markersize=6, linewidth=2.2,
              label=f'{label}: {it} it., {n_eval} eval.')
    if len(trials):
        ax_s.plot(trials[:, 0], trials[:, 1], marker, color=colour, alpha=0.35, markersize=5)
    if len(accepted):
        ax_s.plot(accepted[:, 0], accepted[:, 1], style, color=colour, marker=marker, markersize=7, linewidth=2.2)
ax_f.set(xlabel='objective evaluations since the restart', ylabel='decrease of the objective (%)', xlim=(0, None),
         ylim=(0, None), title='progress per evaluation')
ax_f.legend(fontsize=12, loc='upper right', frameon=False)
ax_s.set_yscale('log')
ax_s.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax_s.set(xlabel='L-BFGS iteration since the restart', ylabel='mean design change of the step',
         xlim=(0, None), title='trial steps (light dots) and accepted steps (lines)')
for ax in (ax_f, ax_s):
    ax.grid(alpha=0.3)
fig.suptitle('128³ restart from the converged design: first L-BFGS steps', fontsize=16)
fig.tight_layout()
out = os.path.join(HERE, 'figures', 'lbfgs_restart_128.png')
fig.savefig(out, dpi=120)
print('saved', out)
