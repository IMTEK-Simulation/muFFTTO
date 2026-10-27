"""Grid scaling of the topology optimization runs (warm-start energy stop, eta = 1/32), in the style of fig6_scalling.pdf
of the JCTAM TO paper: PCG iterations per objective evaluation (sum over the 3 load cases), mechanical equilibrium (x)
and adjoint problem (triangle) separately, over the grid, for soft phase 1e-3 (a) and 1e-5 (b); one curve per
preconditioner with arrow labels. Runs that are not finished yet (512^2 / 1024^2 on the GPU) are drawn with hollow
markers (their mean so far).

    python plot_TO_scaling.py      -> figures/TO_scaling.pdf, .png and <paper>/figures/part2_TO_scaling.pdf
"""
import os
import re
import time

import matplotlib
import matplotlib.ticker

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
PAPER = os.path.expanduser('~/PycharmProjects/Paper-Double-Green-Preconditioner/figures')
import sys
ETA = sys.argv[1] if len(sys.argv) > 1 else '1-32'      # log tag eta<ETA>: '1-32' (eta = 1/32) or '0.014'
GRIDS = [32, 64, 128, 256, 512, 1024] if ETA == '1-32' else [128, 256, 512]
OUT = 'TO_scaling' + ('' if ETA == '1-32' else f'_eta{ETA}')
PRECS = [('Green', 'Green', '#1a8a1a', (0, (6, 2, 1, 2))),          # green, dash-dot (as FFT in the JCTAM figure)
         ('Green_Jacobi', 'Green--Jacobi', 'black', '-'),
         ('dual', 'dual', '#2a78d6', '-')]
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 9, 'axes.titlesize': 10})


def mean_per_evaluation(N, e, p):
    """mean PCG iterations per objective evaluation (3 load cases): mechanical, adjoint, finished?"""
    log = os.path.join(HERE, 'logs', f'TO_N{N}_soft1e-{e}_eta{ETA}_warm_{p}.log')
    if not os.path.exists(log):
        return None
    text = open(log).read()
    m = [int(x) for x in re.findall(r'nb_ steps CG mech +=(\d+)', text)]
    a = [int(x) for x in re.findall(r'nb_steps CG adjoint +=(\d+)', text)]
    n = min(len(m), len(a)) // 3 * 3
    # a run still going is drawn only after 60 evaluations: the first evaluations (big steps, good warm starts) are
    # cheaper than the rest, an early mean would be too low
    running = 'Data saved' not in text and time.time() - os.path.getmtime(log) < 600
    if n < 3 * (60 if running else 5):
        return None
    return (np.reshape(m[:n], (-1, 3)).sum(1).mean(), np.reshape(a[:n], (-1, 3)).sum(1).mean(),
            'Data saved' in text)


# one arrow label per preconditioner: (grid the arrow points to, text position in axes fraction)
LABELS = {3: {'Green': (64, (0.36, 0.88)), 'Green_Jacobi': (256, (0.62, 0.66)), 'dual': (128, (0.52, 0.22))},
          5: {'Green': (64, (0.36, 0.88)), 'Green_Jacobi': (64, (0.22, 0.30)), 'dual': (128, (0.45, 0.72))}}
if ETA != '1-32':
    LABELS = {3: {'Green_Jacobi': (256, (0.5, 0.88)), 'dual': (256, (0.5, 0.25))},
              5: {'Green_Jacobi': (256, (0.3, 0.30)), 'dual': (256, (0.5, 0.88))}}

fig, axes = plt.subplots(1, 2, figsize=(7.47, 3.32))
for ax, e, tag in zip(axes, (3, 5), ('(a)', '(b)')):
    for p, label, c, ls in PRECS:
        rows = [(N, r) for N in GRIDS if (r := mean_per_evaluation(N, e, p)) is not None]
        if not rows:
            continue
        x = np.array([N for N, _ in rows])
        done = np.array([r[2] for _, r in rows])
        for k, (problem, marker) in enumerate((('mech', 'x'), ('adj', '^'))):
            y = np.array([r[k] for _, r in rows])
            ax.plot(x, y, color=c, ls=ls, lw=1.8 if k == 0 else 1.4)
            ax.plot(x[done], y[done], ls='none', marker=marker, color=c, ms=6 if marker == 'x' else 5,
                    mew=1.5)
            ax.plot(x[~done], y[~done], ls='none', marker=marker, mfc='white', mec=c, ms=6 if marker == 'x' else 5,
                    mew=1.2)
            if k == 0 and p in LABELS[e] and LABELS[e][p][0] in x:
                N_arrow, xy_text = LABELS[e][p]
                ax.annotate(label, xy=(N_arrow, y[list(x).index(N_arrow)]),
                            xytext=xy_text, textcoords='axes fraction', color=c, fontsize=9.5, ha='center',
                            arrowprops=dict(arrowstyle='->', color=c, lw=0.8, shrinkA=2, shrinkB=3))
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(GRIDS, [f'${N}^2$' for N in GRIDS])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(GRIDS[0], GRIDS[-1])
    ax.set_ylim(10, 1e4)
    ax.grid(axis='y', which='major', color='0.85', lw=0.6)
    ax.set_xlabel('Grid size')
    ax.set_ylabel(r'\# PCG iterations per L-BFGS step')
    ax.set_title(rf'$\kappa^{{\mathrm{{void}}}}=10^{{-{e}}}$' + ('' if ETA == '1-32' else rf', $\eta={ETA}$'))
    ax.text(0.01, 1.02, rf'\textbf{{{tag}}}', transform=ax.transAxes, ha='left', va='bottom')
from matplotlib.lines import Line2D
axes[0].legend([Line2D([], [], ls='none', marker='x', color='0.3', mew=1.5),
                Line2D([], [], ls='none', marker='^', color='0.3'),
                Line2D([], [], ls='none', marker='^', mfc='white', mec='0.3')],
               ['mechanical equilibrium', 'adjoint problem', 'running / stopped early'], loc='upper right',
               frameon=False, fontsize=8, handletextpad=0.3)
fig.tight_layout(w_pad=1.5)
os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
fig.savefig(os.path.join(HERE, 'figures', OUT + '.pdf'))
fig.savefig(os.path.join(HERE, 'figures', OUT + '.png'), dpi=200)
fig.savefig(os.path.join(PAPER, 'part2_' + OUT.replace('.', 'p') + '.pdf'))
for e in (3, 5):
    for p, *_ in PRECS:
        print(e, p, {N: tuple(np.round(r[:2])) + (r[2],) for N in GRIDS if (r := mean_per_evaluation(N, e, p))})
