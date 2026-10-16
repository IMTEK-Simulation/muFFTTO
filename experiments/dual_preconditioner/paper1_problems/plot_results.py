"""
Figures and summary table of the paper-1 preconditioner comparison (run_all.sh first).

    python plot_results.py                 -> figures/*.png, figures/summary.md
    python plot_results.py elasticity      -> figures_elasticity/*.png, figures_elasticity/summary.md
                                              (run_all_elasticity.sh first)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ELASTICITY = len(sys.argv) > 1 and sys.argv[1] == 'elasticity'
DATA, FIG = (HERE / 'exp_data_elasticity', HERE / 'figures_elasticity') if ELASTICITY else (HERE / 'exp_data',
                                                                                            HERE / 'figures')
FIG.mkdir(exist_ok=True)

PRECS = ['green', 'green_jacobi', 'dual']
PREC_LABEL = {'green': 'Green', 'green_jacobi': 'Green-Jacobi', 'dual': r'dual $GK(A^{-1})G$'}
if ELASTICITY:
    PREC_LABEL = {'green': r'Green ($C_0$)', 'green_jacobi': r'Green-Jacobi ($2\mu I$)',
                  'dual': r'dual $GK(C_0A^{-1}C_0)G$'}
PREC_COLOR = {'green': '#2a78d6', 'green_jacobi': '#eb6834', 'dual': '#1baf7a'}
PREC_MARKER = {'green': 'o', 'green_jacobi': 's', 'dual': '^'}
KAPPAS = {'P1': [-3, -2, -1, 1, 2, 3], 'P2': [-3, -2, -1, 1, 2, 3], 'P3': [-3, -2, -1]}
GRIDS = [32, 64, 128, 256, 512, 1024] if ELASTICITY else [256, 512, 1024]
TOL = 1e-10

# line styles of the paper: v = upper bound or estimate, ^ = lower bound
STYLE = {
    'error':   dict(color='black',      ls='-',  marker=None, lw=1.2),
    'rr':      dict(color='tab:blue',   ls='-',  marker=None, lw=0.9),
    'triv_UB': dict(color='tab:purple', ls='-.', marker='v',  lw=0.9),
    'triv_LB': dict(color='tab:purple', ls='-.', marker='^',  lw=0.9),
    'GR':      dict(color='tab:green',  ls='--', marker='v',  lw=0.9),
    'PT_LB':   dict(color='tab:red',    ls='--', marker='^',  lw=0.9),
    'PT_UE':   dict(color='tab:red',    ls=':',  marker='v',  lw=0.9),
}


def load(problem, N, kexp, prec):
    f = DATA / f'{problem}_N{N}_kappa1e{kexp:+d}_{prec}.npz'
    if not f.exists():
        return None
    d = dict(np.load(f, allow_pickle=True))
    d['stop_iteration'] = d['stop_iteration'].item()
    d['k_star'] = int(d['k_star']) if int(d['k_star']) >= 0 else None
    return d


def last_iteration(y, floor):
    above = np.flatnonzero(np.abs(np.asarray(y)) >= floor)
    return int(above[-1]) + 1 if above.size else len(y) - 1


# ============================================================================
# 1. Iterative error, bounds and estimates, kappa = 1e-3 and 1e3
# ============================================================================
for problem in ['P1', 'P2', 'P3']:
    for kexp in ([-3, 3] if problem != 'P3' else [-3]):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for ax, prec in zip(axes, PRECS):
            h = load(problem, 256, kexp, prec)
            ax.set_title(f'{problem}, $\\kappa$ = 1e{kexp:+d}, {PREC_LABEL[prec]}')
            if h is None:
                continue
            k = np.arange(len(h['error']))
            curves = [('rr', k, h['rr'], r'residual $\|r_k\|^2$')]
            if np.isfinite(h['lam_min']):
                curves += [('triv_UB', k, h['rz'] / h['lam_min'], 'trivial upper bound'),
                           ('triv_LB', k, h['rz'] / h['lam_max'], 'trivial lower bound')]
            if h['GR'].size:
                curves += [('GR', np.arange(h['GR'].size), h['GR'], 'GR upper bound')]
            l = np.arange(h['PT_LB'].size)
            curves += [('PT_UE', l, h['PT_LB'] / (1 - h['tau']), 'PT upper estimate'),
                       ('PT_LB', l, h['PT_LB'], 'PT lower bound'),
                       ('error', k, h['error'], 'iterative error $e_k$')]
            for key, x, y, label in curves:
                ax.semilogy(x, np.abs(y), **STYLE[key], markevery=max(1, len(x) // 8), label=label)
            ax.set_xlim(0, max(last_iteration(h['error'], 1e-12), 3))
            ax.set_ylim(1e-12, 1e1)
            ax.set_xlabel('PCG iteration $k$')
            ax.grid(alpha=0.3)
            if not np.isfinite(h['lam_min']):
                ax.text(0.97, 0.97, 'no guaranteed $\\lambda_{\\min}$:\nno trivial / GR bound',
                        transform=ax.transAxes, ha='right', va='top', fontsize=8, color='0.35')
        axes[0].legend(fontsize=8, ncol=2, loc='lower left')
        fig.tight_layout()
        fig.savefig(FIG / f'errors_{problem}_kappa1e{kexp:+d}.png', dpi=150)
        plt.close(fig)

# ============================================================================
# 2. Iterations against the contrast
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, problem in zip(axes, ['P1', 'P2', 'P3']):
    for prec in PRECS:
        rows = [(kexp, load(problem, 256, kexp, prec)) for kexp in KAPPAS[problem]]
        rows = [(kexp, h) for kexp, h in rows if h is not None]
        for side in ([-1, 1] if problem != 'P3' else [-1]):     # low / high contrast drawn separately
            sel = [(10.0 ** kexp, h) for kexp, h in rows if np.sign(kexp) == side]
            if not sel:
                continue
            kap = [s[0] for s in sel]
            first = side == -1
            ax.semilogx(kap, [h['k_star'] for _, h in sel], c=PREC_COLOR[prec], marker=PREC_MARKER[prec],
                        lw=2, ms=8, label=f'{PREC_LABEL[prec]}: $e_k\\leq$1e-10' if first else None)
            ax.semilogx(kap, [h['k_production'] for _, h in sel], c=PREC_COLOR[prec], marker=PREC_MARKER[prec],
                        lw=1.2, ls='--', ms=8, mfc='white',
                        label=f'{PREC_LABEL[prec]}: production stop' if first else None)
    ax.set_title(problem)
    ax.set_xlabel(r'total phase contrast $\kappa$')
    ax.set_ylabel('PCG iterations')
    ax.set_yscale('log')
    ax.set_ylim(2, 10000 if ELASTICITY else 4000)
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / 'iterations_vs_contrast.png', dpi=150)
plt.close(fig)

# ============================================================================
# 3. Efficiency of the GR bound and the PT estimates, kappa = 1e-3
# ============================================================================
fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharey=True)
for row, problem in enumerate(['P1', 'P2', 'P3']):
    for ax, prec in zip(axes[row], PRECS):
        h = load(problem, 256, -3, prec)
        ax.set_title(f'{problem}, $\\kappa$ = 1e-3, {PREC_LABEL[prec]}', fontsize=10)
        if h is None:
            continue
        m = last_iteration(h['error'], 1e-12)
        err = h['error']
        for key, y, label in [('GR', h['GR'], 'GR upper bound'),
                              ('PT_UE', h['PT_LB'] / (1 - h['tau']), 'PT upper estimate'),
                              ('PT_LB', h['PT_LB'], 'PT lower bound')]:
            n = min(len(y), m)
            if n:
                ax.semilogy(np.arange(n), y[:n] / err[:n], **STYLE[key], markevery=max(1, n // 8), label=label)
        ax.axhline(1, c='0.5', lw=0.6)
        ax.set_ylim(1e-1, 1e3)
        ax.grid(alpha=0.3)
        if row == 2:
            ax.set_xlabel('PCG iteration $k$')
    axes[row, 0].set_ylabel('efficiency')
axes[0, 0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / 'efficiency_kappa1e-3.png', dpi=150)
plt.close(fig)

# ============================================================================
# 4. Grid study, kappa = 1e-3 (elasticity: also kappa = 1e3 for P1 and P2)
# ============================================================================
GRID_KAPPAS = {-3: ['P1', 'P2', 'P3'], 3: ['P1', 'P2']} if ELASTICITY else {-3: ['P1', 'P2', 'P3']}
for kexp_grid, problems in GRID_KAPPAS.items():
    fig, axes = plt.subplots(1, len(problems), figsize=(5 * len(problems), 4))
    for ax, problem in zip(axes, problems):
        for prec in PRECS:
            pts = [(N, load(problem, N, kexp_grid, prec)) for N in GRIDS]
            pts = [(N, h) for N, h in pts if h is not None]
            if pts:
                ax.semilogx([p[0] for p in pts], [p[1]['k_star'] for p in pts], base=2, c=PREC_COLOR[prec],
                          marker=PREC_MARKER[prec], lw=2, ms=8, label=PREC_LABEL[prec])
                for N, h in pts:                               # iteration count next to each point
                    others = [load(problem, N, kexp_grid, q) for q in PRECS if q != prec]
                    crowded_above = any(o is not None and 1 <= o['k_star'] / h['k_star'] < 2.2 for o in others)
                    if ELASTICITY:                             # linear axis: Green left, dual right, both above
                        offset = {'green': (-9, 5), 'dual': (9, 5), 'green_jacobi': (0, -14 if crowded_above else 7)}[prec]
                    else:
                        offset = (0, -14 if prec == 'dual' or crowded_above else 7)
                    ax.annotate(str(h['k_star']), (N, h['k_star']), textcoords='offset points', xytext=offset,
                                ha='center', fontsize=8, color='0.25')
        ax.set_xticks(GRIDS, [f'{N}$^2$' for N in GRIDS])
        if ELASTICITY:                                           # linear iteration axis
            ax.set_yscale('linear')
            ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
        else:
            ax.set_yscale('log', base=10)
            ax.set_ylim(2, 4000)
        if problem in ('P2', 'P3') and GRIDS[0] < 256:          # below 256^2 the stored geometry is coarsened
            ax.axvline(181, color='0.6', ls=':', lw=1)
            ax.text(175, ax.get_ylim()[1] * 0.97, 'coarsened\ngeometry', ha='right', va='top', fontsize=7, color='0.4')
        ax.set_xlim(GRIDS[0] / 1.3, GRIDS[-1] * 1.3)
        ax.set_title(f'{problem}, $\\kappa$ = 1e{kexp_grid:+d}')
        ax.set_xlabel('grid')
        ax.set_ylabel(r'PCG iterations to $e_k\leq$1e-10')
        ax.grid(alpha=0.3, which='both')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / f'grid_study_kappa1e{kexp_grid:+d}.png', dpi=150)
    plt.close(fig)

# ============================================================================
# 5. Summary table
# ============================================================================
lines = ['# Paper-1 problems: Green vs Green-Jacobi vs dual G K(A^-1) G', '',
         '`k*`: first iteration with true energy error e_k <= 1e-10.  '
         '`prod`: stop of the production test (PT lower estimate, relative 1e-6, d_min 10).  '
         '`rr / GR / PT_UE`: first iteration at which that criterion is below 1e-10 '
         '(monitored run, tau = 0.25).  `t`: wall time of the production solve.', '',
         '| problem | N | kappa | preconditioner | k* | prod | rr | GR | PT_UE | t [s] | A_h |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
for problem in ['P1', 'P2', 'P3']:
    cases = [(256, kexp) for kexp in KAPPAS[problem]] + [(N, kexp) for kexp in GRID_KAPPAS for N in GRIDS
                                                         if N != 256 and problem in GRID_KAPPAS[kexp]]
    for N, kexp in cases:
        for prec in PRECS:
            h = load(problem, N, kexp, prec)
            if h is None:
                continue
            s = h['stop_iteration']
            fmt = lambda v: '-' if v is None else str(v)
            lines.append(f'| {problem} | {N} | 1e{kexp:+d} | {prec} | {fmt(h["k_star"])} | {h["k_production"]} | '
                         f'{fmt(s.get("rr"))} | {fmt(s.get("energy_upper_bound"))} | '
                         f'{fmt(s.get("energy_upper_estim"))} | {float(h["t_production"]):.2f} | '
                         f'{float(h["A_h"]):.10f} |')
(FIG / 'summary.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
