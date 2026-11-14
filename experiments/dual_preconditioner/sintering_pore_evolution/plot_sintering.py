"""
Sintering mesh study: CG per elastic solve (Green / Green-Jacobi / dual) vs grid and time, physics vs time,
microstructure snapshots.  python plot_sintering.py [RUN_TAG (default W4_P50u_k0.0001)] [GRIDS (default 64,128,256)]
Writes figures/sintering_cg_summary.png, figures/sintering_physics.png, figures/sintering_snapshots_N<N>.png.
"""
import os
import sys
import glob
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

here = os.path.dirname(os.path.abspath(__file__))
tag = sys.argv[1] if len(sys.argv) > 1 else 'W4_P50u_k0.0001'
grids = [int(g) for g in (sys.argv[2] if len(sys.argv) > 2 else '64,128,256').split(',')]
PRECS = [('Green', 'Green', 'tab:green', 'o'), ('Green_Jacobi', 'Green-Jacobi', 'tab:blue', 's'),
         ('dual', 'dual', 'tab:red', '^')]
WORK = {'Green': 1.0, 'Green_Jacobi': 1.0, 'dual': 2.0}      # FFT pairs per CG iteration (2D)
plt.rcParams.update({'font.size': 13, 'axes.formatter.useoffset': False})
fig_dir = os.path.join(here, 'figures')
sfx = '' if tag == 'W4_P50u_k0.0001' else '_' + tag
os.makedirs(fig_dir, exist_ok=True)


def load(N, p):
    f = os.path.join(here, 'exp_data', f'SINT_N{N}_{p}_{tag}_steps.npz')
    return np.load(f, allow_pickle=True) if os.path.exists(f) else None


data = {(N, p): load(N, p) for N in grids for p, *_ in PRECS}
data = {k: v for k, v in data.items() if v is not None}
if not data:
    sys.exit('no data')
width = float(next(iter(data.values()))['width'])

# ------------------------------------------------------------ table
print(f'{"N":>5} {"W/h":>5} {"precond":>13} {"steps":>6} {"mean CG":>8} {"max CG":>7} {"work":>8} {"wall mech [s]":>13}')
rows = []
for (N, p), d in sorted(data.items()):
    cg = np.array(d['cg_mean'])
    rows.append((N, p, cg.mean(), WORK[p] * cg.mean(), float(np.sum(d['time_mech']))))
    print(f'{N:5d} {width / (128.0 / N):5.2g} {p:>13} {len(cg):6d} {cg.mean():8.1f} {np.max(d["cg"]):7d} '
          f'{WORK[p] * cg.mean():8.1f} {np.sum(d["time_mech"]):13.1f}')

# ------------------------------------------------------------ CG summary
fig, ax = plt.subplots(1, 4, figsize=(21, 5))
for p, lab, col, mk in PRECS:
    Ns = [N for N in grids if (N, p) in data]
    if not Ns:
        continue
    cg = [np.mean(data[(N, p)]['cg_mean']) for N in Ns]
    ax[0].plot(Ns, cg, color=col, marker=mk, ms=8, lw=2, label=lab)
    ax[1].plot(Ns, [WORK[p] * c for c in cg], color=col, marker=mk, ms=8, lw=2, label=lab)
    ax[2].plot(Ns, [np.sum(data[(N, p)]['time_mech']) / len(data[(N, p)]['cg_mean']) for N in Ns], color=col,
               marker=mk, ms=8, lw=2, label=lab)
N_t = 128 if any((128, p) in data for p, *_ in PRECS) else max(N for N, _ in data)
styles = {64: ':', 128: '-', 256: '--', 32: '-.'}
for p, lab, col, mk in PRECS:
    for N in grids:
        if (N, p) in data:
            d = data[(N, p)]
            ax[3].plot(np.array(d['t']) * 1e6, d['cg_mean'], color=col, ls=styles.get(N, '-'), lw=1.8,
                       label=f'{lab} {N}$^2$')
ax[0].set(xlabel='grid N (N$^2$ pixels)', ylabel='mean CG iterations per elastic solve',
          title=f'CG per solve (W = {width:g} nm fixed)')
ax[1].set(xlabel='grid N', ylabel='mean FFT work per solve\n(CG its, dual x 2)', title='work per solve')
ax[2].set(xlabel='grid N', ylabel='wall time of mechanics per step [s]', title='wall time (3 solves / step)')
ax[3].set(xlabel='time [$\\mu$s]', ylabel='mean CG iterations per elastic solve', title='CG per solve vs time')
for a in ax[:3]:
    a.set_xscale('log', base=2)
    a.set_xticks(grids)
    a.set_xticklabels([f'{N}\n(W/h {width * N / 128:g})' for N in grids])
    a.minorticks_off()
for a in ax:
    a.set_ylim(bottom=0)
    a.grid(alpha=0.3)
ax[0].legend()
ax[3].legend(fontsize=9, ncol=3)
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, f'sintering_cg_summary{sfx}.png'), dpi=130)

# ------------------------------------------------------------ physics vs time (same on all preconditioners)
fig, ax = plt.subplots(1, 4, figsize=(21, 5))
for N in grids:
    for p, lab, col, mk in PRECS:
        if (N, p) not in data:
            continue
        d = data[(N, p)]
        t = np.array(d['t']) * 1e6
        kw = dict(color={32: 'tab:gray', 64: 'tab:orange', 128: 'tab:purple', 256: 'black'}.get(N, 'k'),
                  ls='-' if p == 'dual' else ':', lw=2 if p == 'dual' else 1.2)
        lbl = f'{N}$^2$ (W/h {width * N / 128:g})' if p == 'dual' else None   # all preconditioners overlap
        ax[0].plot(t, d['gb_length'], label=lbl, **kw)
        ax[1].plot(t, d['surface_length'], label=lbl, **kw)
        ax[2].plot(t, np.array(d['C_eff'])[:, 1, 1], label=lbl, **kw)
        pm = np.array(d['pores']) >= 0
        ax[3].plot(t[pm], np.array(d['pores'])[pm], marker='o', ms=3, label=lbl, **kw)
ax[0].set(xlabel='time [$\\mu$s]', ylabel='grain-boundary (neck) length [nm]', title='neck growth')
ax[1].set(xlabel='time [$\\mu$s]', ylabel='free-surface length [nm]', title='surface (pore perimeter)')
ax[2].set(xlabel='time [$\\mu$s]', ylabel='C$^{eff}_{yyyy}$ / C$^{Cu}_{yyyy}$', title='effective stiffness')
ax[3].set(xlabel='time [$\\mu$s]', ylabel='number of pores (rho < 0.5)', title='pores')
for a in ax:
    a.set_ylim(bottom=0)
    a.grid(alpha=0.3)
ax[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(fig_dir, f'sintering_physics{sfx}.png'), dpi=130)

# ------------------------------------------------------------ snapshots (exact pixels)
p_snap = 'dual' if (N_t, 'dual') in data else next(p for N, p in data if N == N_t)
snaps = sorted(glob.glob(os.path.join(here, 'exp_data', f'SINT_N{N_t}_{p_snap}_{tag}_snaps', 'step_*.npz')))
if snaps:
    ts = np.array([float(np.load(f)['t']) for f in snaps])
    pick = [snaps[int(np.argmin(abs(ts - frac * ts[-1])))] for frac in (0.0, 0.02, 0.2, 1.0)]
    fig, ax = plt.subplots(2, 4, figsize=(18, 9.4), layout='constrained')
    rng = np.random.default_rng(3)
    cmap = ListedColormap(np.vstack([[1, 1, 1, 1], plt.cm.tab10(np.arange(10))]))
    for k, f in enumerate(pick):
        s = np.load(f)
        ext = [0, 128, 0, 128]
        im = ax[0, k].imshow(s['rho'].T, origin='lower', cmap='gray_r', vmin=0, vmax=1, interpolation='nearest',
                             extent=ext)
        g = np.where(s['rho'] > 0.5, s['grains'] + 1, 0)
        ax[1, k].imshow(g.T, origin='lower', cmap=cmap, vmin=-0.5, vmax=10.5, interpolation='nearest', extent=ext)
        ax[0, k].set_title(f't = {float(s["t"]) * 1e6:.0f} $\\mu$s')
        for a in ax[:, k]:
            a.set_xlabel('x [nm]')
            a.set_ylabel('y [nm]')
    fig.colorbar(im, ax=ax[:, -1], shrink=0.5, label='density rho')
    fig.suptitle(f'{N_t}$^2$ (h = {128 / N_t:g} nm, W/h = {width * N_t / 128:g}), {p_snap}: density (top) and '
                 f'grains where rho > 0.5 (bottom)')
    fig.savefig(os.path.join(fig_dir, f'sintering_snapshots_N{N_t}{sfx}.png'), dpi=110)
print('written', fig_dir)
