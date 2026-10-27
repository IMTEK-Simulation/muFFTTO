"""
Talk figure: Green, Green-Jacobi and dual compared by COST, not by iterations (linear problems and TO, no fracture).

Cost of one PCG iteration in units of (one Green application + one K application): Green 1, Green-Jacobi 1, dual 2
(the dual applies G K(.) G: two FFT pairs and one more K). Measured wall time per iteration on the paper-1 runs
(median over P1-P3 and contrasts, same CPU): dual / Green = 1.94, 2.88, 1.93, 1.47 at 128^2, 256^2, 512^2, 1024^2;
Green-Jacobi / Green = 0.96-1.15.

    row 1  P1, P2, P3: cost per solve vs grid (kappa = 1e-3, production stop: energy estimate 1e-6, delay 10)
    row 2  P1, P2: cost per solve vs contrast (256^2);  TO (Joedicke et al. setup): cost per mechanical solve vs grid
           for kappa_void = 1e-5 (solid) and 1e-3 (dashed; Green / Green-Jacobi beyond 64^2 = short-run estimates)
Thin dotted line: the dual's iteration count before the cost factor.

usage:  python talk_cost_plot.py      -> figures/talk_cost.png, figures/talk_cost.pdf
"""
import os
import re

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = os.path.dirname(os.path.realpath(__file__))
P1DATA = os.path.join(HERE, 'paper1_problems', 'exp_data_elasticity')
TOLOGS = os.path.join(HERE, 'TO_JCTAM_setup', 'logs')
os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
plt.rcParams.update({'font.size': 13})
PRECS = [('green', 'Green', '#2a78d6', 'o', 1), ('green_jacobi', 'Green-Jacobi', '#eb6834', 's', 1),
         ('dual', 'dual', '#1baf7a', '^', 2)]
TO_NAME = {'green': 'Green', 'green_jacobi': 'Green_Jacobi', 'dual': 'dual'}
GRIDS = [32, 64, 128, 256, 512, 1024]
KAPPAS = [-3, -2, -1, 1, 2, 3]


def p1(problem, N, kappa, p):
    f = os.path.join(P1DATA, f'{problem}_N{N}_kappa1e{kappa:+d}_{p}.npz')
    return int(np.load(f, allow_pickle=True)['k_production']) if os.path.exists(f) else None


def to_mean(N, e, p, fromdual=False):
    f = os.path.join(TOLOGS, f'JCTAM_N{N}_soft{e}_{TO_NAME[p]}{"_fromdual" if fromdual else ""}.log')
    if not os.path.exists(f):
        return None
    m = [int(x) for x in re.findall(r'nb_ steps CG mech +=(\d+)', open(f).read())]
    m = m[3:] if fromdual else m
    return np.mean(m) if len(m) >= (3 if fromdual else 30) else None


def draw(ax, xs, ys, p, label, colour, marker, factor, ls='-', end=True, dy=0):
    pts = [(x, y * factor) for x, y in zip(xs, ys) if y is not None]
    if not pts:
        return
    ax.plot(*zip(*pts), ls=ls, marker=marker, color=colour, lw=2.2, ms=7, mfc=colour if ls == '-' else 'white')
    if factor > 1:
        raw = [(x, y) for x, y in zip(xs, ys) if y is not None]
        ax.plot(*zip(*raw), ls=':', color=colour, lw=1.2, alpha=0.7)
    if end:
        ax.annotate(f'{label}: {pts[-1][1]:.0f}', pts[-1], textcoords='offset points', xytext=(9, dy - 4),
                    fontsize=11.5, color=colour)


def style(ax, xlabel, ticks, ticklabels, xlim):
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(ticks, ticklabels)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.grid(True, color='0.92', which='both')
    ax.spines[['top', 'right']].set_visible(False)


fig, axes = plt.subplots(2, 3, figsize=(21, 11.5), layout='constrained')
for ax, problem in zip(axes[0], ('P1', 'P2', 'P3')):
    for p, label, c, m, f in PRECS:
        draw(ax, GRIDS, [p1(problem, N, -3, p) for N in GRIDS], p, label, c, m, f)
    style(ax, 'grid', GRIDS, [f'{N}$^2$' for N in GRIDS], (25, 4000))
    ax.set_title(f'{problem}, contrast $\\kappa$ = 1e-3: cost per solve vs grid')
    ax.set_ylabel('cost per solve  [Green + K applications]')
for ax, problem in zip(axes[1][:2], ('P1', 'P2')):
    for side in (KAPPAS[:3], KAPPAS[3:]):
        for p, label, c, m, f in PRECS:
            draw(ax, [10.0 ** k for k in side], [p1(problem, 256, k, p) for k in side], p, label, c, m, f,
                 end=side[0] > 0)
    style(ax, 'phase contrast $\\kappa$', [10.0 ** k for k in KAPPAS + [0]],
          ['1e-3', '1e-2', '1e-1', '1e1', '1e2', '1e3', '1'], (5e-4, 2e4))
    ax.set_title(f'{problem}, 256$^2$: cost per solve vs contrast')
    ax.set_ylabel('cost per solve  [Green + K applications]')
ax = axes[1][2]
TOGRIDS = [16, 32, 64, 128, 256, 512]
for p, label, c, m, f in PRECS:
    draw(ax, TOGRIDS, [to_mean(N, 5, p) for N in TOGRIDS], p, f'{label}, 1e-5', c, m, f)
    soft3 = [to_mean(N, 3, p) if N <= 64 or p == 'dual' else to_mean(N, 3, p, fromdual=True) for N in TOGRIDS]
    draw(ax, TOGRIDS, soft3, p, f'{label}, 1e-3', c, m, f, ls='--', dy=-12 if p == 'green_jacobi' else 0)
style(ax, 'grid', TOGRIDS, [f'{N}$^2$' for N in TOGRIDS], (13, 2500))
ax.set_title('topology optimization: cost per mechanical solve\n$\\kappa_{void}$ = 1e-5 (solid), 1e-3 (dashed)')
ax.set_ylabel('mean cost per solve  [Green + K applications]')
fig.suptitle('Linear elasticity: cost = PCG iterations x applications per iteration (Green, Green-Jacobi: 1; dual: 2, '
             'measured wall-time ratio 1.5-2.9).  Thin dotted: dual iterations before the factor 2', fontsize=14)
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(HERE, 'figures', f'talk_cost.{ext}'), dpi=110)
print('wrote figures/talk_cost.png')
