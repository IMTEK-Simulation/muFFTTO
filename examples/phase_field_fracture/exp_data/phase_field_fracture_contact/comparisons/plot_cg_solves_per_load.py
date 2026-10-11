"""In-run Steihaug-CG residuals at selected loads, Green vs Green_Jacobi.
python plot_cg_solves_per_load.py out.png green_run/ gj_run/ [lam ...]"""
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

out, runs = sys.argv[1], sys.argv[2:4]
lams = [float(a) for a in sys.argv[4:]] or [0.2, 0.46, 0.56, 0.8]
logs = [pickle.load(open(os.path.join(r, 'cg_convergence.pkl'), 'rb')) for r in runs]
fig, axes = plt.subplots(1, len(lams), figsize=(4.6 * len(lams), 4.4), sharey=True)
for ax, lam in zip(axes, lams):
    for log, c in zip(logs, ['C0', 'C1']):
        all_lams = np.array([s['lam'] for s in log['steihaug']])
        nearest = all_lams[np.argmin(np.abs(all_lams - lam))]      # the increment closest to lam
        solves = [s for s in log['steihaug'] if s['lam'] == nearest]
        its = [len(s['res']) - 1 for s in solves]
        for k, s in enumerate(solves):
            r = np.asarray(s['res'])
            ax.semilogy(r / r[0], c=c, lw=0.9, alpha=0.6, marker='x' if s['status'] == 'boundary' else None,
                        markevery=[len(r) - 1], ms=6,
                        label=(f'{log["mechanics_preconditioner"]} (lam {nearest:.3f}): {len(solves)} solves, '
                               f'{sum(its)} it. total' if k == 0 else None))
    ax.set_title(f'increment nearest to lam = {lam:.2f} (F11 = {1 - 0.3 * lam:.3f})', fontsize=10)
    ax.set_xlabel('CG iteration')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
axes[0].set_ylabel(r'$\|r_k\| / \|r_0\|$')
fig.suptitle('every Steihaug-CG solve of one load increment (all staggered and trust-region iterations); '
             'x = stopped on the trust-region boundary', fontsize=10)
fig.tight_layout()
fig.savefig(out, dpi=90)
print('saved', out)
