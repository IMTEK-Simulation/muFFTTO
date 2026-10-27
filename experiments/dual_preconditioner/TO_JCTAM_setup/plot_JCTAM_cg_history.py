"""PCG iterations per objective evaluation (sum over the 3 load cases) over the optimization, JCTAM setup,
kappa_void = 1e-5 (rows: mechanical / adjoint, columns: grid), Green / Green-Jacobi / dual. Thin: every evaluation;
thick: moving average over 25 evaluations.

    python plot_JCTAM_cg_history.py      -> figures/JCTAM_cg_history.png
"""
import os
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
GRIDS = [64, 128, 256]
PRECS = [('Green', 'Green', '#1a8a1a'), ('Green_Jacobi', 'Green-Jacobi', 'black'), ('dual', 'dual', '#2a78d6')]


def history(N, p, e=5):
    text = open(os.path.join(HERE, 'logs', f'JCTAM_N{N}_soft{e}_{p}.log')).read()
    m = np.array(re.findall(r'nb_ steps CG mech +=(\d+)', text), int)
    a = np.array(re.findall(r'nb_steps CG adjoint +=(\d+)', text), int)
    n = min(len(m), len(a)) // 3 * 3
    return m[:n].reshape(-1, 3).sum(1), a[:n].reshape(-1, 3).sum(1), 'Data saved' in text


def smooth(y, w=25):
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode='valid')


fig, axes = plt.subplots(2, len(GRIDS), figsize=(13, 7), sharey='row')
for j, N in enumerate(GRIDS):
    for p, label, c in PRECS:
        mech, adj, done = history(N, p)
        for i, y in enumerate((mech, adj)):
            ax = axes[i, j]
            x = np.arange(1, len(y) + 1)
            ax.plot(x, y, color=c, lw=0.5, alpha=0.3)
            ys = smooth(y)
            ax.plot(np.arange(len(ys)) + (len(y) - len(ys)) // 2 + 1, ys, color=c, lw=2,
                    label=label + ('' if done else ' (stopped at 790)'))
    for i, what in enumerate(('mechanical equilibrium', 'adjoint problem')):
        ax = axes[i, j]
        ax.set_yscale('log')
        ax.grid(alpha=0.3, which='both')
        ax.set_title(f'${N}^2$, {what}', fontsize=11)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        if i == 1:
            ax.set_xlabel('objective evaluation (~ L-BFGS iteration)')
axes[0, 0].set_ylabel('PCG iterations per evaluation\n(3 load cases)')
axes[1, 0].set_ylabel('PCG iterations per evaluation\n(3 load cases)')
axes[0, 0].legend(frameon=False, fontsize=9)
fig.suptitle(r'JCTAM setup, $\kappa^{\mathrm{void}}=10^{-5}$: PCG iterations over the optimization', fontsize=12)
fig.tight_layout()
os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
fig.savefig(os.path.join(HERE, 'figures', 'JCTAM_cg_history.png'), dpi=160)
