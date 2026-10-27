"""Grid scaling of the TO runs for the two interface widths: eta = 1/32 (solid) and eta = 0.014 (dashed, D = 1),
PCG iterations per objective evaluation for the mechanical equilibrium (sum over the 3 load cases; the adjoint problem
costs the same within 2 %), soft phase 1e-3 (a) and 1e-5 (b). Unfinished runs: hollow markers (mean so far, shown after
60 evaluations).

    python plot_TO_scaling_eta.py      -> figures/TO_scaling_eta.pdf, .png
"""
import os
import re
import time

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
GRIDS = [32, 64, 128, 256, 512]
PRECS = [('Green', 'Green', '#1a8a1a', 'o'), ('Green_Jacobi', 'Green--Jacobi', 'black', 's'),
         ('dual', 'dual', '#2a78d6', '^')]
ETAS = [('1-32', r'$\eta=1/32$', '-'), ('0.014', r'$\eta=0.014$', '--')]
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 9, 'axes.titlesize': 10})


def mean_mech(N, e, p, eta):
    if N == 1024:   # the 1024^2 runs stopped on the plateau of the objective (step test, 59 iterations): not comparable
        return None
    log = os.path.join(HERE, 'logs', f'TO_N{N}_soft1e-{e}_eta{eta}_warm_{p}.log')
    if not os.path.exists(log):
        return None
    text = open(log).read()
    m = [int(x) for x in re.findall(r'nb_ steps CG mech +=(\d+)', text)]
    n = len(m) // 3 * 3
    running = 'Data saved' not in text and time.time() - os.path.getmtime(log) < 600
    if n < 3 * (60 if running else 5):
        return None
    return np.reshape(m[:n], (-1, 3)).sum(1).mean(), 'Data saved' in text


fig, axes = plt.subplots(1, 2, figsize=(7.47, 3.5))
for ax, e, tag in zip(axes, (3, 5), ('(a)', '(b)')):
    for p, label, c, marker in PRECS:
        for eta, eta_label, ls in ETAS:
            rows = [(N, r) for N in GRIDS if (r := mean_mech(N, e, p, eta)) is not None]
            if not rows:
                continue
            x = np.array([N for N, _ in rows]); y = np.array([r[0] for _, r in rows]); done = np.array([r[1] for _, r in rows])
            ax.plot(x, y, color=c, ls=ls, lw=1.6)
            ax.plot(x[done], y[done], ls='none', marker=marker, color=c, ms=5)
            ax.plot(x[~done], y[~done], ls='none', marker=marker, mfc='white', mec=c, ms=5)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(GRIDS, [f'${N}^2$' for N in GRIDS])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(GRIDS[0] / 1.15, GRIDS[-1] * 1.15)
    ax.set_ylim(10, 1e4)
    ax.grid(axis='y', which='major', color='0.85', lw=0.6)
    ax.set_xlabel('Grid size')
    ax.set_ylabel(r'\# PCG iterations per L-BFGS step (mech.)')
    ax.set_title(rf'$\kappa^{{\mathrm{{void}}}}=10^{{-{e}}}$')
    ax.text(0.01, 1.02, rf'\textbf{{{tag}}}', transform=ax.transAxes, ha='left', va='bottom')
from matplotlib.lines import Line2D
handles = ([Line2D([], [], color=c, marker=m, ls='none', ms=5) for _, _, c, m in PRECS]
           + [Line2D([], [], color='0.3', ls=ls, lw=1.6) for _, _, ls in ETAS]
           + [Line2D([], [], color='0.3', marker='^', mfc='white', ls='none', ms=5)])
labels = [l for _, l, _, _ in PRECS] + [l for _, l, _ in ETAS] + ['running']
fig.legend(handles, labels, loc='lower center', ncol=6, frameon=False, fontsize=8.5)
fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.5)
fig.savefig(os.path.join(HERE, 'figures', 'TO_scaling_eta.pdf'))
fig.savefig(os.path.join(HERE, 'figures', 'TO_scaling_eta.png'), dpi=200)
