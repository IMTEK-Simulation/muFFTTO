"""
3D Cahn-Hilliard AT2 case (ch3D: l = 1/16, k_res = 1e-4, 32^3 ... 256^3, NEMO): CG cost on the crack and the
staggered-scheme stall after it.

    (a) CG iterations per elasticity solve against the grid: before the crack (open, dotted) and in the first
        500 staggered iterations of the crack step (filled); dashed red = dual x 1.5 (cost of a dual iteration in 3D)
    (b) CG iterations of every staggered iteration of the crack step, 256^3
    (c) staggered convergence max|d - d_old|: crack step 31 (converges, identical for all preconditioners) against
        the later 128^3 steps that hit the 10000 cap (no trend)

Crack step = first step with >= 500 staggered iterations (128^3, 256^3: step 31); 32^3 / 64^3: the step with the most.
Input: exp_data/nemo_3D/AT2_N<N>_<P>_incl0.1_l0.0625_3D_k0.0001_ch_iterations.csv (copied from NEMO).

    python plot_AT2_ch3D_crack_staggered.py      ->  figures/AT2_ch3D_crack_staggered.png
"""
import collections
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
DATA = os.path.join(HERE, 'exp_data', 'nemo_3D', 'AT2_N{N}_{P}_incl0.1_l0.0625_3D_k0.0001_ch_iterations.csv')
GRIDS = [32, 64, 128, 256]
PRECONDS = [('Green', 'Green', 'tab:green'), ('Green_Jacobi', 'Green-Jacobi', 'tab:blue'), ('dual', 'dual', 'tab:red')]
DUAL_COST = 1.5
FIRST = 500
TOL = 1e-3
plt.rcParams.update({'font.size': 14})


def load(N, P):
    by = collections.defaultdict(list)
    with open(DATA.format(N=N, P=P)) as f:
        for r in csv.DictReader(f):
            by[int(r['step'])].append((int(r['cg_u']), float(r['damage_change'])))
    return by


def crack_step(by, N):
    n = {s: len(v) for s, v in by.items()}
    if N >= 128:
        return next(s for s in sorted(n) if n[s] >= FIRST)
    return max(n, key=n.get)


runs = {(N, P): load(N, P) for N in GRIDS for P, _, _ in PRECONDS}
fig, (a, b, c) = plt.subplots(1, 3, figsize=(19, 5.8), layout='constrained')

# (a) CG per solve against the grid
for P, lab, col in PRECONDS:
    pre, crack = [], []
    for N in GRIDS:
        by = runs[(N, P)]
        cs = crack_step(by, N)
        pre.append(np.mean([cg for s, v in by.items() if s < cs for cg, _ in v]))
        crack.append(np.mean([cg for cg, _ in by[cs][:FIRST]]))
    a.plot(GRIDS, crack, 'o-', color=col, lw=2.5, ms=9, label=f'{lab}: crack step')
    a.plot(GRIDS, pre, 'o:', color=col, lw=2, ms=8, mfc='white', label=f'{lab}: before crack')
    if P == 'dual':
        a.plot(GRIDS, DUAL_COST * np.array(crack), '--', color=col, lw=2, label=f'dual x {DUAL_COST:g} (cost), crack')
    print(lab, 'pre', np.round(pre, 1), 'crack', np.round(crack, 1))
a.set_xscale('log', base=2)
a.set_xticks(GRIDS), a.set_xticklabels([f'{n}$^3$' for n in GRIDS]), a.minorticks_off()
a.set_ylim(0, 560)
a.set(xlabel='grid (l = 1/16 fixed)', ylabel='CG iterations per elasticity solve')
a.set_title(f'(a) cost per solve (crack: first {FIRST} staggered its)')
a.legend(fontsize=11, ncol=2, loc='upper left', framealpha=0.95)
a.grid(alpha=0.3)

# (b) CG of every staggered iteration of the crack step, 256^3
N = 256
for P, lab, col in PRECONDS:
    by = runs[(N, P)]
    cs = crack_step(by, N)
    cg = [x for x, _ in by[cs]]
    b.plot(np.arange(1, len(cg) + 1), cg, '-', color=col, lw=1.2, label=f'{lab} (mean {np.mean(cg):.0f})')
b.set_ylim(0, None), b.set_xlim(0, None)
b.set(xlabel=f'staggered iteration in crack step {cs}', ylabel='CG iterations (elasticity)')
b.set_title(f'(b) through the crack, {N}$^3$')
b.legend(fontsize=12, loc='upper right', framealpha=0.95)
b.grid(alpha=0.3)

# (c) staggered convergence: converging crack step vs stalled later steps (128^3)
N = 128
by = runs[(N, 'dual')]
cs = crack_step(by, N)
dd = [d for _, d in by[cs]]
c.semilogy(np.arange(1, len(dd) + 1), dd, '-', color='0.15', lw=1.5,
           label=f'step {cs}: crack, converges in {len(dd)} (all preconditioners)')
for P, s, col in [('dual', 84, 'tab:red'), ('Green_Jacobi', 84, 'tab:blue'), ('Green', 74, 'tab:green')]:
    dd = [d for _, d in runs[(N, P)][s]]
    lab = dict((p, l) for p, l, _ in PRECONDS)[P]
    c.semilogy(np.arange(1, len(dd) + 1), dd, '-', color=col, lw=0.6, alpha=0.75,
               label=f'step {s}, {lab}: {len(dd)} its, not converged' + (' (same as dual)' if P == 'Green_Jacobi' else ''))
c.axhline(TOL, color='k', ls='--', lw=1.5, label=f'tolerance {TOL:g}')
yt = [1e-3, 1e-2, 1e-1, 1]
c.set_yticks(yt), c.set_yticklabels(['0.001', '0.01', '0.1', '1']), c.minorticks_off()
c.set_ylim(3e-4, 30)
c.set_xscale('log')
xt = [1, 10, 100, 1000, 10000]
c.set_xticks(xt), c.set_xticklabels([str(x) for x in xt])
c.set(xlabel='staggered iteration', ylabel='max |d - d$_{old}$|')
c.set_title(f'(c) staggered scheme stalls after the crack, {N}$^3$')
c.legend(fontsize=11, loc='upper left', framealpha=0.95)
c.grid(alpha=0.3)

fig.suptitle('AT2 3D Cahn-Hilliard microstructure (l = 1/16, k$_{res}$ = 10$^{-4}$): CG on the crack and the '
             'staggered stall', fontsize=16)
out = os.path.join(HERE, 'figures', 'AT2_ch3D_crack_staggered.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=130)
print('wrote', out)
