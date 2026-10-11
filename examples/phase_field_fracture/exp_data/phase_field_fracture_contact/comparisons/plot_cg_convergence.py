"""CG convergence: Green vs Green_Jacobi.  python plot_cg_convergence.py out.png green_run/ gj_run/

Top: diagnostic PCG (to 1e-10, the Newton system at the start of the increment) at the same
load levels with both preconditioners. Bottom: every Steihaug-CG solve of each run
(inexact Newton), ||r_k|| / ||r_0|| over the CG iterations, coloured by the load lam.
"""
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

out, runs = sys.argv[1], sys.argv[2:]
logs = [pickle.load(open(os.path.join(r, 'cg_convergence.pkl'), 'rb')) for r in runs]

diag = logs[0]['diagnostic']               # both preconditioners are in every run's diagnostic
ncol = 2 * max(2, (len(diag) + 1) // 2 * 1)
fig = plt.figure(figsize=(4.2 * len(diag), 9.5))
gs = fig.add_gridspec(2, 2 * len(diag), height_ratios=[1, 1.15], hspace=0.38, wspace=0.6)
for k, entry in enumerate(diag):
    ax = fig.add_subplot(gs[0, 2 * k:2 * k + 2])
    for name, c in (('Green', 'C0'), ('Green_Jacobi', 'C1')):
        h = entry[name]
        ax.semilogy(h, c=c, label=f'{name} ({len(h) - 1} it.)')
    ax.set_title(f'lam = {entry["lam"]:.2f} (F11 = {1 - 0.3 * entry["lam"]:.3f})', fontsize=10)
    ax.set_xlabel('CG iteration')
    ax.set_ylabel(r'$\|r_k\| / \|r_0\|$')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig.text(0.5, 0.93, 'diagnostic PCG to 1e-10 on the Newton system at the start of the increment, '
         'same state, both preconditioners', ha='center', fontsize=10)

cmap = plt.get_cmap('viridis')
half = len(diag)
bottom = []
for k, log in enumerate(logs):
    ax = fig.add_subplot(gs[1, k * half:(k + 1) * half])
    bottom.append(ax)
    solves = log['steihaug']
    its = [len(s['res']) - 1 for s in solves]
    for s in solves:
        r = np.asarray(s['res'])
        ax.semilogy(r / r[0], c=cmap(s['lam']), lw=0.5, alpha=0.4)
    ax.set_title(f'{log["mechanics_preconditioner"]} run: all {len(solves)} Steihaug-CG solves (inexact Newton)\n'
                 f'iterations per solve: median {np.median(its):.0f}, mean {np.mean(its):.0f}, '
                 f'max {np.max(its)}', fontsize=10)
    ax.set_xlabel('CG iteration')
    ax.set_ylabel(r'$\|r_k\| / \|r_0\|$')
    ax.grid(alpha=0.3)
xmax = max(ax.get_xlim()[1] for ax in bottom)
for ax in bottom:
    ax.set_xlim(0, xmax)
    ax.set_ylim(1e-3, 3)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
fig.colorbar(sm, ax=bottom, label='load lam', fraction=0.02, pad=0.01)
fig.suptitle('sinusoidal_ligaments 32x32, compression - mechanics CG convergence', y=0.98)
fig.savefig(out, dpi=90, bbox_inches='tight')
print('saved', out)
