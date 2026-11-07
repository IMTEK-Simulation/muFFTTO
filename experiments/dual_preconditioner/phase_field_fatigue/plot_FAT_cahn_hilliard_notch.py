"""Figures of the phase-field fatigue study (run_FAT_cahn_hilliard_notch.sh), see README.md.

figures/FAT_crack_growth.png      crack length and peak stress vs cycle (all grids)
figures/FAT_cg_per_cycle.png      mean CG iterations per elastic solve vs cycle, Green / Green-Jacobi / dual, per grid
figures/FAT_cg_vs_grid.png        mean CG per solve before / during growth and total work vs grid
figures/FAT_damage_maps.png       final damage maps (exact pixels), all grids
Also prints the summary table (phases, CG per solve, work).
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
TAG = os.environ.get('TAG', '_incl0.1_gc1.7407_l0.0625_k0.0001_asymptotic_amp8e-05_cyc1500')
SUF = os.environ.get('FIG_SUFFIX', '')
L_SCALE = float(os.environ.get('L_SCALE', 0.0625))      # l in cell units (l/h = L_SCALE * N)
L_MM = float(os.environ.get('L_MM', 0))                  # cell size in mm (0: lengths in cell units)
NS = [int(n) for n in os.environ.get('NS', '32 64 128').split()]
PRECS = [('Green', 'Green', 'tab:green'), ('Green_Jacobi', 'Green-Jacobi', 'tab:blue'), ('dual', 'dual', 'tab:red')]
NOTCH_LENGTH = 0.25
plt.rcParams.update({'font.size': 13, 'axes.labelsize': 13, 'legend.fontsize': 12, 'axes.formatter.useoffset': False})
fig_dir = os.path.join(here, 'figures')
os.makedirs(fig_dir, exist_ok=True)


def load(N, P):
    f = os.path.join(here, 'exp_data', f'FAT_N{N}_{P}{TAG}_cycles.csv')
    if not os.path.exists(f):
        return None
    d = np.genfromtxt(f, delimiter=',', names=True)
    return d if d.size > 1 else None


def phases(d, N):
    """pre-crack: cycles before the crack leaves the notch; growth: from then on up to (excl.) the final break
    (the first cycle whose peak stress drops by more than 30 %)."""
    grown = d['crack_length'] > d['crack_length'][0] + 2.5 / N   # > 2 pixels longer than after cycle 1
    i_init = int(np.argmax(grown)) if grown.any() else len(d)
    broken = np.r_[False, d['sigma'][1:] < 0.7 * d['sigma'][:-1]]      # the final break: stress drop > 30 % in a cycle
    i_break = int(np.argmax(broken)) if broken.any() else len(d)
    return slice(0, i_init), slice(i_init, i_break), i_init, i_break


data = {(N, P): load(N, P) for N in NS for P, _, _ in PRECS}
summary = []
for (N, P), d in data.items():
    if d is None:
        continue
    pre, grow, i_init, i_break = phases(d, N)
    cg, its = d['cg_u_step'], d['staggered_its']
    w = 2.0 if P == 'dual' else 1.0
    per = lambda s: cg[s].sum() / max(its[s].sum(), 1)
    summary.append(dict(N=N, P=P, cycles=len(d), init=i_init + 1, brk=i_break + 1, pre=per(pre), grow=per(grow),
                        all=cg.sum() / its.sum(), solves=int(its.sum()), cg=int(cg.sum()), work=int(w * cg.sum()),
                        time=float(d['time_u_step'].sum()), notconv=int((d['converged'] == 0).sum()),
                        dadn=(d['crack_length'][max(i_break - 1, i_init)] - d['crack_length'][min(i_init, len(d) - 1)])
                        / max(i_break - 1 - i_init, 1)))
print(f'{"N":>4} {"P":>13} {"cycles":>6} {"init":>5} {"break":>5} {"CG/solve pre":>12} {"growth":>7} {"all":>6} '
      f'{"solves":>7} {"CG total":>9} {"work":>9} {"t_u [s]":>8} {"not conv":>8} {"da/dN [L]":>10}')
for s in summary:
    print(f'{s["N"]:4d} {s["P"]:>13} {s["cycles"]:6d} {s["init"]:5d} {s["brk"]:5d} {s["pre"]:12.1f} {s["grow"]:7.1f} '
          f'{s["all"]:6.1f} {s["solves"]:7d} {s["cg"]:9d} {s["work"]:9d} {s["time"]:8.0f} {s["notconv"]:8d} {s["dadn"]:10.2e}'
          + (f' = {s["dadn"] * L_MM * 1e3:.1f} um/cycle' if L_MM else ''))

# (a) crack length and peak stress vs cycle (physics: the dual runs; all preconditioners give the same curves)
styles = {32: ':', 64: '--', 128: '-'}
fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
for N in NS:
    d = next((data[(N, P)] for P in ('dual', 'Green', 'Green_Jacobi') if data[(N, P)] is not None), None)
    if d is None:
        continue
    cyc = d['step'] + 1
    axes[0].plot(cyc, d['crack_length'], styles.get(N, '-'), color='k', lw=2, label=f'{N}$^2$ (l/h = {L_SCALE * N:g})')
    axes[1].plot(cyc, d['sigma'] / d['sigma'][0], styles.get(N, '-'), color='k', lw=2, label=f'{N}$^2$')
    _, _, i_init, i_break = phases(d, N)
    win = 10                                  # da/dN over a window of 10 cycles, up to the final break
    a = d['crack_length'] * (L_MM * 1e3 if L_MM else 1.0)
    k = np.arange(win, max(i_break, win + 1))
    if len(k):
        axes[2].plot(cyc[k], (a[k] - a[k - win]) / win, styles.get(N, '-'), color='k', lw=2, label=f'{N}$^2$')
axes[0].set_ylabel('crack length a / L (d > 0.95)')
axes[1].set_ylabel(r'peak stress $\bar\sigma_{yy}$ / first cycle')
for ax in axes:
    ax.set_xlabel('cycle N')
    ax.set_xlim(left=0)
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3)
axes[0].set_ylim(0, 1.05)
axes[2].set_ylabel('da/dN [um / cycle]' if L_MM else 'da/dN [L / cycle]')
axes[2].set_title('crack growth rate (10-cycle window)', fontsize=13)
axes[0].legend(title='grid')
axes[0].set_title('crack growth (R = 0, constant strain amplitude)', fontsize=13)
axes[1].set_title('stiffness loss at the peak strain', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, f'FAT_crack_growth{SUF}.png'), dpi=150)

# (b) CG iterations per elastic solve vs cycle (mean over the staggered iterations of the cycle)
fig, axes = plt.subplots(1, len(NS), figsize=(5.2 * len(NS), 4.6), squeeze=False)
axes = axes[0]
for ax, N in zip(axes, NS):
    ymax = 0
    for P, label, c in PRECS:
        d = data[(N, P)]
        if d is None:
            continue
        y = d['cg_u_step'] / d['staggered_its']
        ax.plot(d['step'] + 1, y, color=c, lw=1.5, label=label)
        ymax = max(ymax, np.percentile(y, 99.5))
        _, _, i_init, i_break = phases(d, N)
    d = next((data[(N, P)] for P, _, _ in PRECS if data[(N, P)] is not None), None)
    if d is not None:
        _, _, i_init, i_break = phases(d, N)
        ax.axvline(i_init + 1, color='0.4', ls=':', lw=1.2)
        ax.axvline(i_break + 1, color='0.4', ls='--', lw=1.2)
    ax.set_title(f'{N}$^2$, l/h = {L_SCALE * N:g}', fontsize=13)
    ax.set_xlabel('cycle N')
    ax.set_ylabel('CG iterations per elastic solve')
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.1 * ymax)
    ax.grid(alpha=0.3)
handles, labels = axes[0].get_legend_handles_labels()
handles += [plt.Line2D([], [], color='0.4', ls=':'), plt.Line2D([], [], color='0.4', ls='--')]
labels += ['crack leaves the notch', 'final break']
fig.legend(handles, labels, loc='upper center', ncol=5, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(os.path.join(fig_dir, f'FAT_cg_per_cycle{SUF}.png'), dpi=150)

# (c) mean CG per solve vs grid, before / during growth, and total work (dual x 2)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for P, label, c in PRECS:
    rows = [s for s in summary if s['P'] == P]
    if not rows:
        continue
    n = [s['N'] for s in rows]
    axes[0].plot(n, [s['pre'] for s in rows], 'o-', color=c, lw=2, ms=8, label=label)
    axes[1].plot(n, [s['grow'] for s in rows], 'o-', color=c, lw=2, ms=8, label=label)
    axes[2].plot(n, [s['work'] / 1e3 for s in rows], 'o-', color=c, lw=2, ms=8, label=label + (' (x 2)' if P == 'dual' else ''))
axes[0].set_title('before growth (crack at the notch)', fontsize=13)
axes[1].set_title('during growth (Paris regime)', fontsize=13)
axes[2].set_title('total elastic work of the run', fontsize=13)
axes[0].set_ylabel('mean CG iterations per elastic solve')
axes[1].set_ylabel('mean CG iterations per elastic solve')
axes[2].set_ylabel('CG iterations x 1000 (dual counted x 2)')
for ax in axes:
    ax.set_xlabel('grid N (N x N pixels)')
    ax.set_xticks(NS)
    ax.set_xticklabels([str(n) for n in NS])
    ax.set_xlim(0, 1.1 * max(NS))
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3)
axes[2].legend()
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, f'FAT_cg_vs_grid{SUF}.png'), dpi=150)

# (d) damage maps, exact pixels (nodal values, one per pixel), dual runs: mid-growth (from the .nc, every 10th cycle)
# and final
import netCDF4
fig, axes = plt.subplots(2, len(NS), figsize=(5.0 * len(NS), 9.0), squeeze=False)
for k, N in enumerate(NS):
    f = os.path.join(here, 'exp_data', f'FAT_N{N}_dual{TAG}_damage.npz')
    d = data[(N, 'dual')]
    if not os.path.exists(f) or d is None:
        continue
    z = np.load(f)
    _, _, i_init, i_break = phases(d, N)
    nc = netCDF4.Dataset(f.replace('_damage.npz', '.nc'))
    steps = np.asarray(nc['step'][:]).astype(int)
    m = int(np.argmin(np.abs(steps - 0.5 * (i_init + i_break))))
    frames = [(np.asarray(nc['damage'][m, 0]), steps[m], 'during growth'),
              (z['damage'][-1], int(z['steps'][-1]), 'final')]
    for ax, (d_xy, st, what) in zip(axes[:, k], frames):
        im = ax.imshow(d_xy.T, origin='lower', extent=(0, 1, 0, 1), vmin=0, vmax=1, cmap='inferno',
                       interpolation='nearest')
        ax.contour(np.linspace(0.5 / N, 1 - 0.5 / N, N), np.linspace(0.5 / N, 1 - 0.5 / N, N), z['phase'].T,
                   levels=[0.5], colors='c', linewidths=0.7)
        ax.set_title(f'{N}$^2$, {what}: cycle {st + 1}', fontsize=13)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar(im, ax=ax, fraction=0.046, label='damage d')
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, f'FAT_damage_maps{SUF}.png'), dpi=150)
print('figures written to', fig_dir)
