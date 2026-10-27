"""TO with the setup of Joedicke et al. (JCTAM): mean PCG iterations per solve (one load case) over the grid, mechanical
equilibrium (x) and adjoint problem (triangle), kappa_void = 0 (a) and 1e-5 (b), Green / Green-Jacobi / dual, in the
style of their fig6_scalling.pdf. In (b) their published values (FFT = Green, J-FFT = Green-Jacobi, kappa_void = 1e-5,
from the comments in Joedicke_et_al.tex) are drawn as faint lines for comparison. Runs still going: hollow markers.
In (c), Green / Green-Jacobi on 128^2 ... 512^2 are estimates (dashed, open squares/diamonds): 6 L-BFGS iterations from a
formed design, frame 500 of the dual run (run_JCTAM_soft3_from_dual.sh, logs/*_fromdual.log, first evaluation left out);
on 32^2 and 64^2 the same estimate is within ~10-20 % of the full runs.

    python plot_JCTAM_scaling.py      -> figures/JCTAM_scaling.pdf, .png and figures/JCTAM_scaling_soft3.pdf, .png
                                         (panel (c) alone)
"""
import os
import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
GRIDS = [16, 32, 64, 128, 256, 512]
PRECS = [('Green', 'Green', '#1a8a1a', (0, (6, 2, 1, 2))), ('Green_Jacobi', 'Green--Jacobi', 'black', '-'),
         ('dual', 'dual', '#2a78d6', '-')]
# Joedicke et al., kappa_void = 1e-5, grids 16^2 ... 1024^2 (comments next to their Fig. 6)
PAPER_GRIDS = [16, 32, 64, 128, 256, 512, 1024]
PAPER = {'Green': ([106.3, 241.5, 344.3, 301.2, 382.2, 423.3, 307.4], [243.7, 1287.3, 1895.8, 1856.0, 1943.9, 2000.7, 1899.0]),
         'Green_Jacobi': ([22.1, 30.7, 45.4, 47.7, 46.5, 41.4, 62.4], [53.0, 90.6, 156.1, 205.3, 265.0, 326.7, 587.5])}
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 9, 'axes.titlesize': 10})


def means(N, e, p):
    f = os.path.join(HERE, 'logs', f'JCTAM_N{N}_soft{e}_{p}.log')
    if not os.path.exists(f):
        return None
    text = open(f).read()
    m = [int(x) for x in re.findall(r'nb_ steps CG mech +=(\d+)', text)]
    a = [int(x) for x in re.findall(r'nb_steps CG adjoint +=(\d+)', text)]
    if len(m) < 30:
        return None
    # a run stopped by hand once its CG statistics had settled (marker file *.STOPPED) counts as finished
    return np.mean(m), np.mean(a), 'Data saved' in text or os.path.exists(f.replace('.log', '.STOPPED'))


def estimate(N, p):
    """Mean PCG iterations per solve of the short runs from a formed design (first evaluation left out)."""
    f = os.path.join(HERE, 'logs', f'JCTAM_N{N}_soft3_{p}_fromdual.log')
    if not os.path.exists(f):
        return None
    text = open(f).read()
    m = [int(x) for x in re.findall(r'nb_ steps CG mech +=(\d+)', text)][3:]
    a = [int(x) for x in re.findall(r'nb_steps CG adjoint +=(\d+)', text)][3:]
    return (np.mean(m), np.mean(a)) if 'Data saved' in text and m else None


def panel(ax, e, tag, title):


    for p, label, c, ls in PRECS:
        rows = [(N, r) for N in GRIDS if (r := means(N, e, p)) is not None]
        if not rows:
            continue
        x = np.array([N for N, _ in rows])
        done = np.array([r[2] for _, r in rows])
        for k, marker in enumerate(('x', '^')):
            y = np.array([r[k] for _, r in rows])
            ax.plot(x, y, color=c, ls=ls, lw=1.8 if k == 0 else 1.3)
            ax.plot(x[done], y[done], ls='none', marker=marker, color=c, ms=6 if marker == 'x' else 5, mew=1.5)
            ax.plot(x[~done], y[~done], ls='none', marker=marker, mfc='white', mec=c, ms=6 if marker == 'x' else 5)
    if e == 5:      # their published values
        for p, label, c, ls in PRECS[:2]:
            for k, marker in enumerate(('x', '^')):
                ax.plot(PAPER_GRIDS, PAPER[p][k], color=c, ls=':', lw=1.0, alpha=0.4, marker=marker, ms=4)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(PAPER_GRIDS, [f'${N}^2$' for N in PAPER_GRIDS])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(14, 1150)
    ax.set_ylim(5, 2e4)
    ax.grid(axis='y', which='major', color='0.85', lw=0.6)
    ax.set_xlabel('Grid size')
    ax.set_ylabel(r'mean \# PCG iterations per solve')
    ax.set_title(title)
    ax.text(0.01, 1.02, rf'\textbf{{{tag}}}', transform=ax.transAxes, ha='left', va='bottom')
    if e == 3:      # estimates where no full run exists, joined to the last full-run point
        for p, label, c, ls in PRECS:
            full = [N for N in GRIDS if means(N, e, p) is not None]
            est = [(N, r) for N in GRIDS if N not in full and (r := estimate(N, p)) is not None]
            if not est:
                continue
            x = [N for N, _ in est]
            for k, marker in enumerate(('s', 'D')):
                y = [r[k] for _, r in est]
                if full:
                    ax.plot([full[-1], x[0]], [means(full[-1], e, p)[k], y[0]], color=c, ls=(0, (2, 2)), lw=1.0)
                ax.plot(x, y, color=c, ls=(0, (2, 2)), lw=1.0, marker=marker, mfc='white', mec=c,
                        ms=5 if marker == 's' else 4.5, mew=1.2)


TITLES = {0: r'$\kappa^{\mathrm{void}}=0$', 5: r'$\kappa^{\mathrm{void}}=10^{-5}$', 3: r'$\kappa^{\mathrm{void}}=10^{-3}$'}
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
for ax, e, tag in zip(axes, (0, 5, 3), ('(a)', '(b)', '(c)')):
    panel(ax, e, tag, TITLES[e])
from matplotlib.lines import Line2D
handles = ([Line2D([], [], color=c, ls=ls, lw=1.8) for _, _, c, ls in PRECS]
           + [Line2D([], [], ls='none', marker='x', color='0.3', mew=1.5), Line2D([], [], ls='none', marker='^', color='0.3'),
              Line2D([], [], ls=(0, (2, 2)), color='0.3', marker='s', mfc='white', mec='0.3', lw=1.0),
              Line2D([], [], ls=(0, (2, 2)), color='0.3', marker='D', mfc='white', mec='0.3', lw=1.0, ms=4.5),
              Line2D([], [], color='0.5', ls=':', alpha=0.6)])
labels = [l for _, l, _, _ in PRECS] + ['mechanical', 'adjoint', 'mech., estimate', 'adj., estimate', 'Joedicke et al.']
fig.legend(handles, labels, loc='lower center', ncol=8, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.5)
os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
fig.savefig(os.path.join(HERE, 'figures', 'JCTAM_scaling.pdf'))
fig.savefig(os.path.join(HERE, 'figures', 'JCTAM_scaling.png'), dpi=200)

# panel (c) alone
fig, ax = plt.subplots(figsize=(4.6, 4.2))
panel(ax, 3, '', TITLES[3])
fig.legend(handles[:7], labels[:7], loc="lower center", ncol=3, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.16, 1, 1))
fig.savefig(os.path.join(HERE, 'figures', 'JCTAM_scaling_soft3.pdf'))
fig.savefig(os.path.join(HERE, 'figures', 'JCTAM_scaling_soft3.png'), dpi=200)
