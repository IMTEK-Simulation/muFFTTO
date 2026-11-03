"""
Why is Green cheaper on some fibre realisations? Two diagnostics for the 8 seeds at N = 256 (Green vs dual):

figures/fibers_seeds_per_step.png      per load step: mean CG its per elastic solve (lines), staggered iterations of
                                       the step (grey bars), stress (thin grey), cumulative elastic work (inset row)
figures/fibers_seeds_per_stagger.png   per staggered iteration (all load steps concatenated): CG its of every elastic
                                       solve, Green and dual; bottom: ratio Green / dual with the break-even line 2
                                       (dual costs 2 per iteration), and the damage change of the iteration
usage:  python plot_fibers_seeds_diagnostics.py [N]
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 256
SEEDS = range(8)
COL = {'Green': 'tab:green', 'dual': 'tab:red'}
LS = {'Green': '--', 'dual': '-'}


def load(seed, p):
    f = os.path.join(HERE, 'exp_data', f'AT2_N{N}_{p}_fibers_s{seed}_l0.0117188_k0.0001_iterations.csv')
    return np.genfromtxt(f, delimiter=',', names=True, dtype=None, encoding=None)


def stress(seed):
    import netCDF4
    with netCDF4.Dataset(os.path.join(HERE, 'exp_data', f'AT2_N{N}_Green_fibers_s{seed}_l0.0117188_k0.0001.nc')) as ds:
        return 100 * np.asarray(ds['load'][:]), 1e3 * np.asarray(ds['macro_stress'][:, 0, 0])


plt.rcParams.update({'font.size': 12})
data = {s: {p: load(s, p) for p in COL} for s in SEEDS}
work = {s: {p: (2 if p == 'dual' else 1) * data[s][p]['cg_u'].sum() for p in COL} for s in SEEDS}

# --- per load step ------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(4, 4, figsize=(24, 19), layout='constrained', height_ratios=[2, 1, 2, 1])
for k, s in enumerate(SEEDS):
    ax, axw = axes[2 * (k // 4), k % 4], axes[2 * (k // 4) + 1, k % 4]
    ax2 = ax.twinx()
    for p in COL:
        d = data[s][p]
        steps = np.unique(d['step'])
        E = np.array([100 * d['load'][d['step'] == i][0] for i in steps])
        its = np.array([d['cg_u'][d['step'] == i].mean() for i in steps])
        stag = np.array([(d['step'] == i).sum() for i in steps])
        cum = np.cumsum([(2 if p == 'dual' else 1) * d['cg_u'][d['step'] == i].sum() for i in steps]) / 1e3
        ax.plot(E, its, color=COL[p], ls=LS[p], marker='o', ms=4, lw=2, label=p)
        axw.plot(E, cum, color=COL[p], ls=LS[p], marker='o', ms=4, lw=2, label=p)
        if p == 'Green':
            ax2.bar(E, stag, width=0.06, color='0.8', zorder=0)
            for e, n in zip(E, stag):
                if n >= 20:
                    ax2.text(e, n, f'{n}', ha='center', va='bottom', fontsize=10, color='0.35')
    Es, sig = stress(s)
    ax2.plot(Es, sig / sig.max() * ax2.get_ylim()[1] * 0.9, color='0.45', lw=1)
    ax2.set_ylabel('staggered its of the step (bars)')
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.set_ylim(0, None)
    axw.set_ylim(0, None)
    r = work[s]['Green'] / work[s]['dual']
    ax.set_title(f'#{s}: work Green {work[s]["Green"] / 1e3:.0f}k, dual {work[s]["dual"] / 1e3:.0f}k '
                 f'(Green/dual {r:.2f})', color='k' if r >= 1 else 'tab:green', fontweight='bold' if r < 1 else None)
    ax.set_ylabel('CG its per elastic solve\n(mean over the step)')
    axw.set_ylabel('cumulative work [k]')
    for a in (ax, axw):
        a.set_xlabel('E_xx [%]')
        a.grid(True, color='0.9')
    if k == 0:
        ax.legend(frameon=False, loc='upper left')
fig.suptitle(f'Fibres, N = {N}: per load step (grey bars: staggered iterations; thin grey: stress, scaled); '
             'titles in green: Green cheaper', fontsize=15)
out1 = os.path.join(HERE, 'figures', 'fibers_seeds_per_step.png')
fig.savefig(out1, dpi=120)

# --- per staggered iteration --------------------------------------------------------------------------------------
fig, axes = plt.subplots(4, 4, figsize=(24, 17), layout='constrained', height_ratios=[2, 1, 2, 1])
for k, s in enumerate(SEEDS):
    ax, axr = axes[2 * (k // 4), k % 4], axes[2 * (k // 4) + 1, k % 4]
    g, d = data[s]['Green'], data[s]['dual']
    n = min(len(g), len(d))
    it = np.arange(n)
    ax.plot(it, g['cg_u'][:n], color=COL['Green'], lw=1.2, label='Green')
    ax.plot(it, d['cg_u'][:n], color=COL['dual'], lw=1.2, label='dual')
    # load-step boundaries
    for b in np.nonzero(np.diff(g['step'][:n]))[0]:
        ax.axvline(b + 0.5, color='0.85', lw=0.6, zorder=0)
        axr.axvline(b + 0.5, color='0.85', lw=0.6, zorder=0)
    ax.set_ylim(0, None)
    ratio = g['cg_u'][:n] / np.maximum(d['cg_u'][:n], 1)
    axr.plot(it, ratio, color='k', lw=1)
    axr.axhline(2, color='tab:red', ls='--', lw=1.2)
    axr.set_ylim(0, max(4, np.percentile(ratio, 99)))
    axr2 = axr.twinx()
    axr2.semilogy(it, np.maximum(g['damage_change'][:n], 1e-8), color='tab:purple', lw=0.8, alpha=0.7)
    axr2.set_ylabel('max |d - d_old|', color='tab:purple')
    same = np.array_equal(g['step'][:n], d['step'][:n]) and len(g) == len(d)
    frac = (ratio > 2).mean()
    ax.set_title(f'#{s}: {len(g)} / {len(d)} staggered its (Green / dual){"" if same else " (paths differ)"}; '
                 f'ratio > 2 in {frac:.0%}', fontsize=12)
    ax.set_ylabel('CG its of the elastic solve')
    axr.set_ylabel('Green / dual its\n(red: break-even 2)')
    axr.set_xlabel('staggered iteration (whole run; grey: load steps)')
    for a in (ax, axr):
        a.grid(True, color='0.92')
    if k == 0:
        ax.legend(frameon=False, loc='upper left')
fig.suptitle(f'Fibres, N = {N}: every elastic solve of the run (Green vs dual); below: iteration ratio and damage change',
             fontsize=15)
out2 = os.path.join(HERE, 'figures', 'fibers_seeds_per_stagger.png')
fig.savefig(out2, dpi=120)
print(f'wrote {out1}\nwrote {out2}')

# --- table: where does the work go ------------------------------------------------------------------------------
print(f'{"seed":>4s} {"stag":>5s} | {"Green":>8s} {"dual":>8s} {"G/d":>5s} | break step: {"stag":>4s} {"G its":>7s} '
      f'{"d its":>7s} | median ratio in break step | first-solve ratio')
for s in SEEDS:
    g, d = data[s]['Green'], data[s]['dual']
    steps = np.unique(g['step'])
    b = max(steps, key=lambda i: g['cg_u'][g['step'] == i].sum())
    mg, md = g['step'] == b, d['step'] == b
    rat = g['cg_u'][mg] / np.maximum(d['cg_u'][md], 1) if mg.sum() == md.sum() else np.array([np.nan])
    print(f'{s:4d} {len(g):5d} | {work[s]["Green"]:8d} {work[s]["dual"]:8d} {work[s]["Green"] / work[s]["dual"]:5.2f} | '
          f'step {b:2d}: {mg.sum():4d} {g["cg_u"][mg].sum():7d} {d["cg_u"][md].sum():7d} | {np.median(rat):5.2f} '
          f'(q25 {np.percentile(rat, 25):.2f}, q75 {np.percentile(rat, 75):.2f}) | {rat[0]:.2f}')
